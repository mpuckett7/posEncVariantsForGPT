"""Small helpers shared by train / eval / scripts (AI-scaffolded)."""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

LN2 = math.log(2.0)  # divide a loss in nats by this to get bits


def set_seed(seed: int) -> None:
    """Seed every RNG we use so a (variant, seed) pair is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> str:
    """'auto' -> 'cuda' when a GPU is visible, otherwise 'cpu'. Anything else is returned as-is."""
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def nats_to_bits(x: float) -> float:
    """Convert a cross-entropy in nats (what PyTorch returns) to bits per character."""
    return x / LN2


def append_jsonl(path, record: dict) -> None:
    """Append one JSON record as a line (creates parent dirs). Used for metrics.jsonl."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def count_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters (tied weights are counted once)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
