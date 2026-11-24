"""LoRA 微调工单分类。

几个刻意的选择：

**1. 手写训练循环，不用 Trainer。**
    这个任务太小（216 条、4 个 epoch），Trainer 的封装反而掩盖了
    「loss 只在答案 token 上计算」这个关键细节。手写能看清每一步。

**2. loss 只算 assistant 回复部分。**
    prompt 部分用 -100 屏蔽掉。如果不屏蔽，模型会把大量算力花在
    「复述 prompt」上，学到的东西被稀释，而且 loss 曲线看着很低但
    实际分类能力很差。

**3. 用 float32 而不是 fp16。**
    MPS 后端上 fp16 的数值稳定性不如 CUDA，0.5B 模型用 fp32 只占 2G
    左右统一内存，16G 的 Mac 完全够。稳定性比省内存重要。

**4. 只训 LoRA 层，冻结底座。**
    可训练参数约 1.5M，占全量的 0.3%。这是消费级设备能微调的
    前提，也是 LoRA 存在的意义。
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (Config, OUTPUT_DIR, build_prompt_ids, load_split)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_example(tokenizer, row, max_length: int):
    prompt_ids = build_prompt_ids(tokenizer, row["text"])
    target_ids = tokenizer(
        row["label"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="run1", help="输出目录后缀")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.model:
        cfg.model_name = args.model

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"[env] device={device} model={cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float32：MPS 上更稳；统一内存下 0.5B 只占 ~2G
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.float32
    )
    model.to(device)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = load_split("train")
    examples = [make_example(tokenizer, r, cfg.max_length) for r in train_rows]
    print(f"[data] train={len(examples)} 条，"
          f"平均长度 {sum(len(e['input_ids']) for e in examples)//len(examples)} token")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(len(examples) / (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    history = []
    started = time.time()

    for epoch in range(cfg.epochs):
        random.shuffle(examples)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, len(examples), cfg.batch_size):
            chunk = examples[i:i + cfg.batch_size]
            input_ids, labels, attn = collate(chunk, tokenizer.pad_token_id)
            input_ids, labels, attn = (
                input_ids.to(device), labels.to(device), attn.to(device)
            )

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / cfg.grad_accum
            loss.backward()

            epoch_loss += out.loss.item()
            n_batches += 1

            if n_batches % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # 收尾残留梯度
        if n_batches % cfg.grad_accum != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch + 1, "loss": round(avg, 4)})
        print(f"[epoch {epoch+1}/{cfg.epochs}] loss={avg:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} "
              f"elapsed={time.time()-started:.0f}s")

        if device == "mps":
            torch.mps.empty_cache()

    out_dir = OUTPUT_DIR / f"lora_adapter_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "model": cfg.model_name,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "target_modules": cfg.target_modules,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "train_size": len(examples),
        "train_seconds": round(time.time() - started, 1),
        "device": device,
        "history": history,
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] adapter 已保存到 {out_dir}")
    print(f"[done] 训练耗时 {meta['train_seconds']}s，"
          f"最终 loss {history[-1]['loss']}")


if __name__ == "__main__":
    main()
