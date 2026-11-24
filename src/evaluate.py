"""评测：zero-shot 基线 vs LoRA 微调。

两个模式跑同一份测试集、同一个解析器、同一套指标——
这样才能说「微调带来了多少提升」，而不是「两次不同的实验各有各的数」。

    python -m src.evaluate --mode baseline
    python -m src.evaluate --mode lora --tag run1
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (Config, OUTPUT_DIR, build_messages, confusion_matrix,
                     load_split, parse_label)

TEST_BATCH = 8


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(cfg: Config, mode: str, tag: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 左侧 padding 更利于批量 generate，避免不同长度样本的生成位置错乱
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32)

    if mode == "lora":
        from peft import PeftModel
        adapter_dir = OUTPUT_DIR / f"lora_adapter_{tag}"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"找不到 adapter: {adapter_dir}，先跑 src.train")
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, texts, device: str, max_new_tokens: int = 8):
    """批量生成。

    一次喂多条能显著提速——单条推理时 GPU/MPS 的并行度根本用不满，
    108 条单独跑要几分钟，批量跑一分多钟。
    """
    preds = []
    for i in range(0, len(texts), TEST_BATCH):
        chunk = texts[i:i + TEST_BATCH]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(t), tokenize=False, add_generation_prompt=True
            )
            for t in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        for row in gen:
            preds.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "lora"], required=True)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，调试用")
    args = ap.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_name = args.model

    device = pick_device()
    rows = load_split("test")
    if args.limit:
        rows = rows[: args.limit]

    print(f"[env] mode={args.mode} device={device} model={cfg.model_name} "
          f"test={len(rows)} 条")

    model, tokenizer = load_model(cfg, args.mode, args.tag)
    model.to(device)

    started = time.time()
    raw_preds = predict(model, tokenizer, [r["text"] for r in rows], device)
    elapsed = time.time() - started

    y_true = [r["label"] for r in rows]
    y_pred = [parse_label(p) for p in raw_preds]
    metrics = confusion_matrix(y_true, y_pred)
    metrics["mode"] = args.mode
    metrics["model"] = cfg.model_name
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["per_item_ms"] = round(elapsed / len(rows) * 1000, 1)

    print(f"\n准确率 {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
          f"| 无效输出 {metrics['invalid_output']} ({metrics['invalid_rate']:.1%}) "
          f"| 平均 {metrics['per_item_ms']}ms/条")

    print("\n分类别：")
    print(f"  {'类别':10} {'P':>6} {'R':>6} {'F1':>6} {'样本':>5}")
    for lb, v in metrics["per_class"].items():
        print(f"  {lb:10} {v['precision']:6.3f} {v['recall']:6.3f} "
              f"{v['f1']:6.3f} {v['support']:5d}")

    print("\n混淆矩阵（行=真实，列=预测）：")
    labels = metrics["labels"]
    print("            " + " ".join(f"{lb[:6]:>6}" for lb in labels))
    for lb, row in zip(labels, metrics["matrix"]):
        print(f"  {lb:10} " + " ".join(f"{v:6d}" for v in row))

    # 打印错例，方便判断是模型问题还是数据问题
    print("\n错误样例（最多 8 条）：")
    shown = 0
    for r, p, raw in zip(rows, y_pred, raw_preds):
        if p != r["label"] and shown < 8:
            print(f"  [{r['label']} → {p or '未识别'}] {r['text'][:60]}")
            if p is None:
                print(f"       原始输出: {raw[:40]!r}")
            shown += 1
    if shown == 0:
        print("  无")

    out = OUTPUT_DIR / f"eval_{args.mode}_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
