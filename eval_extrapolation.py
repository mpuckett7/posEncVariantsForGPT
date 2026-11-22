"""Measure loss-by-position for every trained run at longer context lengths (AI-scaffolded).

Usage (from the repo root, after the sweep):
    python eval_extrapolation.py                  # evaluates every runs/*/ckpt.pt
    python eval_extrapolation.py --force          # recompute even if eval.json exists

For each run it writes runs/<name>/eval.json containing
    per_position_loss_nats : {"256": [..256 floats..], "512": [..], "1024": [..]}
    skipped                : lengths the model could not run (e.g. learned embeddings past ctx_len)
    summary                : in_dist_loss_nats and degradation_ratio (see src/metrics.py)
Every length uses the SAME fixed validation windows for every model.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import Config
from src.data import fixed_eval_windows, load_split
from src.metrics import summarize_curves
from src.model import GPT
from src.utils import get_device


def per_position_loss(model, x, y, batch_size: int, device: str):
    """Mean next-token loss (nats) at each position, averaged over all windows.

    x, y: LongTensors [n_windows, L]. Returns a Python list of L floats.
    """
    n, length = x.shape
    totals = torch.zeros(length, dtype=torch.float64)            # running sum per position
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits, _ = model(xb)                                    # [b, L, V]
        # reduction="none" keeps one loss per token so we can average per POSITION ourselves.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none"
        ).view(xb.size(0), length)
        totals += loss.sum(dim=0).double().cpu()
    return (totals / n).tolist()


def evaluate_run(run_dir: Path, device: str, max_windows) -> dict:
    """Load one checkpoint and compute its per-position loss at every configured length."""
    # weights_only=False: the checkpoint is our own file and contains the config dict.
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    val = load_split(cfg.data_dir, "val")

    result = {
        "variant": cfg.variant,
        "seed": cfg.seed,
        "ctx_len": cfg.ctx_len,
        "per_position_loss_nats": {},
        "skipped": {},
    }
    with torch.no_grad():
        for length in cfg.eval_lengths:
            x, y = fixed_eval_windows(val, length, max_windows)
            if len(x) == 0:
                result["skipped"][str(length)] = "validation split shorter than one window"
                continue
            try:
                result["per_position_loss_nats"][str(length)] = per_position_loss(
                    model, x, y, cfg.eval_batch_size, device
                )
            except ValueError as err:
                # Expected for learned absolute embeddings when length > ctx_len.
                result["skipped"][str(length)] = str(err)
    result["summary"] = summarize_curves(result["per_position_loss_nats"], cfg.ctx_len)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-position loss at 256 / 512 / 1024 for every run")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_windows", type=int, default=None, help="cap windows per length (debug)")
    parser.add_argument("--force", action="store_true", help="recompute existing eval.json files")
    args = parser.parse_args()

    device = get_device(args.device)
    ckpts = sorted(Path(args.runs_dir).glob("*_s*/ckpt.pt"))
    if not ckpts:
        print(f"No checkpoints found under {args.runs_dir}/ -- run the sweep first.")
        return
    for ckpt_path in ckpts:
        run_dir = ckpt_path.parent
        out_path = run_dir / "eval.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {run_dir.name}: eval.json exists (use --force to recompute)")
            continue
        result = evaluate_run(run_dir, device, args.max_windows)
        out_path.write_text(json.dumps(result), encoding="utf-8")
        s = result["summary"]
        in_dist = s.get("in_dist_loss_nats")
        ratios = {k: round(v, 3) for k, v in s["degradation_ratio"].items()}
        in_dist_txt = f"{in_dist:.4f}" if in_dist is not None else "n/a"
        print(f"[done] {run_dir.name}: in-dist loss {in_dist_txt} nats | degradation ratios {ratios}"
              f" | skipped {list(result['skipped'])}")


if __name__ == "__main__":
    main()
