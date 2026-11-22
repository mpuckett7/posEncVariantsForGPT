"""Data access: random training batches and fixed evaluation windows (AI-scaffolded).

Data lives in data/train.bin and data/val.bin as raw uint16 token ids
(written by data/prepare.py). We memory-map them so nothing is copied into RAM.
"""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def load_split(data_dir, split: str) -> np.ndarray:
    """Memory-map 'train' or 'val' token ids as a 1-D uint16 array."""
    return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint16, mode="r")


def load_meta(data_dir) -> dict:
    """Read data/meta.json (holds 'chars' and 'vocab_size')."""
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def get_batch(
    data: np.ndarray,
    batch_size: int,
    ctx_len: int,
    device: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows of `ctx_len` tokens.

    Returns (x, y), both LongTensors of shape [batch_size, ctx_len], where y is x
    shifted left by one token (y[t] is the token that follows x[t]).
    Pass a seeded torch.Generator to make the batch order reproducible.
    """
    # A window needs ctx_len inputs + 1 extra token for the final target,
    # so the last valid start index is len(data) - ctx_len - 1 (randint's high is exclusive).
    starts = torch.randint(0, len(data) - ctx_len, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[s : s + ctx_len].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1 : s + 1 + ctx_len].astype(np.int64)) for s in starts])
    return x.to(device), y.to(device)


def fixed_eval_windows(
    data: np.ndarray, window_len: int, max_windows: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, NON-overlapping windows of `window_len` tokens from `data`.

    Every model sees exactly the same text, so differences between variants are not
    sampling noise. Returns (x, y) of shape [n_windows, window_len] on CPU
    (n_windows may be 0 if the split is shorter than one window).
    """
    n = (len(data) - 1) // window_len          # how many full windows (+1 target token) fit
    if max_windows is not None:
        n = min(n, max_windows)
    xs, ys = [], []
    for i in range(n):
        s = i * window_len
        xs.append(torch.from_numpy(data[s : s + window_len].astype(np.int64)))
        ys.append(torch.from_numpy(data[s + 1 : s + 1 + window_len].astype(np.int64)))
    if not xs:
        empty = torch.empty((0, window_len), dtype=torch.long)
        return empty, empty.clone()
    return torch.stack(xs), torch.stack(ys)
