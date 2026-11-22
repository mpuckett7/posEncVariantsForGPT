"""Summary metrics computed from per-position loss curves (AI-scaffolded, pure Python).

Input: {"256": [loss at pos 0..255], "512": [loss at pos 0..511], ...} in nats,
each curve averaged over the same fixed validation windows.
"""
from typing import Dict, List


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def summarize_curves(per_position_loss: Dict[str, List[float]], ctx_len: int) -> dict:
    """Compute the two quantities the README predictions are about.

    in_dist_loss_nats : mean loss over positions [0, ctx_len) from the ctx_len-length eval.
    degradation_ratio : for each eval length N > ctx_len, the mean loss over the LAST
                        ctx_len positions of the N-length window divided by the mean loss
                        over a reference window just before the training limit,
                        positions [ctx_len - ref_len, ctx_len)   (192..255 for ctx_len=256).
                        A ratio near 1 means the model holds up; larger means it degrades.
    """
    summary = {}
    ref_len = min(64, max(1, ctx_len // 4))     # 64 positions for ctx_len=256 -> window 192..255
    summary["ref_window"] = [ctx_len - ref_len, ctx_len]

    key = str(ctx_len)
    if key in per_position_loss:
        summary["in_dist_loss_nats"] = _mean(per_position_loss[key][:ctx_len])

    ratios = {}
    for n_str, curve in per_position_loss.items():
        n = int(n_str)
        if n <= ctx_len:
            continue                             # only lengths beyond training count
        reference = _mean(curve[ctx_len - ref_len : ctx_len])   # same windows as the numerator
        tail = _mean(curve[n - ctx_len : n])
        ratios[n_str] = tail / reference
    summary["degradation_ratio"] = ratios
    return summary
