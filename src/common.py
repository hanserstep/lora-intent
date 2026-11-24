"""公共部分：配置、prompt、数据加载、评测指标。

prompt 设计的关键点：**输出空间必须被严格约束**。

    微调一个分类任务，最怕模型学会「说人话」而不是「输出标签」。
    所以 prompt 里明确「只输出类别名，不要解释」，训练数据里的答案
    也只放类别名。推理时再用严格的解析器兜底：只认第一行的合法标签。

    如果解析器宽松（比如用 `in` 判断），「不是 disk 也不是 network」
    这种输出会被误判成 disk，评测数字就假了。
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

LABELS = ["disk", "network", "cpu", "memory", "database", "other"]

LABEL_DESC = {
    "disk": "磁盘空间、inode、磁盘 IO、日志占满",
    "network": "网络连通性、丢包、超时、DNS、带宽、连接数",
    "cpu": "CPU 使用率、系统负载、进程占满单核、上下文切换",
    "memory": "内存使用率、OOM、内存泄漏、swap、GC",
    "database": "主从延迟、慢查询、死锁、连接池、索引",
    "other": "证书、定时任务、告警通知、配置变更、发布回滚",
}

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "tickets.jsonl"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"


def _default_model() -> str:
    """优先用本地目录，没有才回落到 HF 仓库名。

    国内直连 HF 下 1G 权重只有 500KB/s 左右，走 ModelScope 是 4MB/s。
    下载一次存本地，之后所有实验都读本地，不重复下载。
    """
    return str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "你是运维工单分类助手。把工单描述分到下列六类之一，"
    "只输出类别名，不要解释，不要标点。\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABEL_DESC.items())
)


@dataclass
class Config:
    model_name: str = field(default_factory=_default_model)
    max_length: int = 256

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 训练
    epochs: int = 4
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 4
    warmup_ratio: float = 0.1
    seed: int = 42

    @property
    def adapter_dir(self) -> Path:
        return OUTPUT_DIR / "lora_adapter"


def load_split(split: str) -> List[Dict]:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["split"] == split:
                rows.append(r)
    return rows


def build_messages(text: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"工单：{text}\n类别："},
    ]


def build_prompt_ids(tokenizer, text: str) -> List[int]:
    """取到 assistant 回复之前的 token，训练时用来做 label mask。"""
    msgs = build_messages(text)
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


_PARSE_RE = re.compile(r"^\s*([a-zA-Z]+)")


def parse_label(raw: str) -> Optional[str]:
    """严格解析：只看第一个词，必须是六个标签之一。

    不用 `label in raw` 是因为「不是 disk」这种输出会被误判。
    分类任务的评测必须建立在「模型的输出能被无歧义解读」的前提上，
    否则指标衡量的不是模型能力，而是解析器的宽容度。
    """
    m = _PARSE_RE.match(raw or "")
    if not m:
        return None
    word = m.group(1).lower().strip()
    return word if word in LABELS else None


def confusion_matrix(y_true: List[str], y_pred: List[Optional[str]]) -> Dict:
    """返回混淆矩阵和每类 P/R/F1。未解析出标签的一律记为 None。"""
    labels = LABELS
    idx = {lb: i for i, lb in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    invalid = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            invalid += 1
            continue
        mat[idx[t]][idx[p]] += 1

    per_class = {}
    for lb in labels:
        i = idx[lb]
        tp = mat[i][i]
        fp = sum(mat[r][i] for r in range(len(labels)) if r != i)
        fn = sum(mat[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[lb] = {"precision": round(precision, 4),
                         "recall": round(recall, 4),
                         "f1": round(f1, 4),
                         "support": tp + fn}

    total = len(y_true)
    correct = sum(mat[i][i] for i in range(len(labels)))
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(labels), 4),
        "invalid_output": invalid,
        "invalid_rate": round(invalid / total, 4) if total else 0.0,
        "per_class": per_class,
        "matrix": mat,
        "labels": labels,
        "total": total,
    }
