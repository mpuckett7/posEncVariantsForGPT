"""Turn the per-run eval.json files into a figure, summary tables and a check of the README predictions.

Usage (from the repo root, after eval_extrapolation.py):
    python scripts/plot_results.py [--runs_dir runs] [--out_dir figures] [--smooth 16]
Outputs: figures/loss_by_position.png, figures/summary.md (also printed to the console).
Only needs numpy + matplotlib (no torch).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                          # headless backend: just write PNG files
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import VARIANTS                # noqa: E402

LN2 = float(np.log(2.0))                        # nats -> bits
COLORS = {"learned": "tab:gray", "sinusoidal": "tab:orange", "rope": "tab:blue", "nope": "tab:green"}


def load_results(runs_dir: Path) -> list:
    """Read every runs/*/eval.json written by eval_extrapolation.py."""
    results = []
    for path in sorted(Path(runs_dir).glob("*/eval.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def group_by_variant(results: list) -> dict:
    groups = {}
    for r in results:
        groups.setdefault(r["variant"], []).append(r)
    return groups


def mean_std(values) -> tuple:
    """Mean and sample standard deviation (ddof=1) across seeds; std is 0 for a single seed."""
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average with edge padding (same length as the input)."""
    if k <= 1:
        return y
    pad_left = k // 2
    padded = np.pad(y, (pad_left, k - 1 - pad_left), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def plot_loss_by_position(groups: dict, ctx_len: int, out_path: Path, smooth_k: int) -> None:
    """Mean +/- std (over seeds) of per-position loss in bits, one line per variant."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = []
    for variant in VARIANTS:
        runs = groups.get(variant)
        if not runs:
            continue
        # Use the longest evaluation length that EVERY seed of this variant has.
        common = set.intersection(*[set(r["per_position_loss_nats"]) for r in runs])
        if not common:
            continue
        key = max(common, key=int)
        curves = np.stack([np.asarray(r["per_position_loss_nats"][key]) / LN2 for r in runs])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        m, s = smooth(mean, smooth_k), smooth(std, smooth_k)
        x = np.arange(len(m))
        ax.plot(x, m, color=COLORS[variant], label=f"{variant} (n={len(runs)}, up to {key})")
        ax.fill_between(x, m - s, m + s, color=COLORS[variant], alpha=0.2)
        plotted.append(m[16:])                  # ignore the first positions (little context) for y-limits
    ax.axvline(ctx_len, color="black", linestyle="--", linewidth=1, label=f"training length ({ctx_len})")
    if plotted:
        lo = min(float(p.min()) for p in plotted)
        hi = max(float(p.max()) for p in plotted)
        ax.set_ylim(lo * 0.95, hi * 1.05)
    ax.set_xlabel("position in context window")
    ax.set_ylabel("validation loss (bits per character)")
    ax.set_title("Loss by position (mean +/- std across seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_summary(groups: dict) -> dict:
    """Per variant: in-distribution loss (bits) and degradation ratios, each as (mean, std) over seeds."""
    rows = {}
    for variant, runs in groups.items():
        in_dist = [r["summary"]["in_dist_loss_nats"] / LN2
                   for r in runs if "in_dist_loss_nats" in r["summary"]]
        ratios = {}
        for r in runs:
            for n, value in r["summary"].get("degradation_ratio", {}).items():
                ratios.setdefault(n, []).append(value)
        rows[variant] = {
            "n_seeds": len(runs),
            "in_dist": mean_std(in_dist) if in_dist else None,
            "ratios": {n: mean_std(v) for n, v in ratios.items()},
        }
    return rows


def summary_markdown(rows: dict) -> str:
    """Markdown table of the headline numbers."""
    lengths = sorted({n for row in rows.values() for n in row["ratios"]}, key=int)
    header = ["variant", "seeds", "in-dist loss (bpc)"] + [f"ratio @ {n}" for n in lengths]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for variant in VARIANTS:
        if variant not in rows:
            continue
        row = rows[variant]
        in_dist = f"{row['in_dist'][0]:.4f} +/- {row['in_dist'][1]:.4f}" if row["in_dist"] else "n/a"
        cells = [variant, str(row["n_seeds"]), in_dist]
        for n in lengths:
            cells.append(f"{row['ratios'][n][0]:.3f} +/- {row['ratios'][n][1]:.3f}" if n in row["ratios"] else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_predictions(rows: dict) -> str:
    """Compare the results with the two README predictions (the verdict is for you to interpret).

    Prediction 1: NoPE has the lowest in-distribution loss, by MORE THAN ONE SEED STANDARD DEVIATION
                  versus every other variant. Interpretation used here: the gap between the means
                  must exceed the LARGER of the two variants' seed std. Edit if you prefer another rule.
    Prediction 2: RoPE has the smallest degradation ratio at every evaluated length beyond ctx_len
                  (among the variants that can be evaluated there).
    """
    out = []
    out.append("Prediction 1 -- NoPE lowest in-distribution loss, by > 1 seed std vs every other variant")
    nope = rows.get("nope")
    if not nope or not nope["in_dist"]:
        out.append("  (no NoPE in-distribution results yet)")
    else:
        all_ok = True
        for variant, row in rows.items():
            if variant == "nope" or not row["in_dist"]:
                continue
            gap = row["in_dist"][0] - nope["in_dist"][0]          # positive => NoPE is lower
            threshold = max(nope["in_dist"][1], row["in_dist"][1])
            ok = gap > threshold
            all_ok &= ok
            out.append(f"  vs {variant:<10} gap {gap:+.4f} bpc | larger std {threshold:.4f} | "
                       f"{'PASS' if ok else 'FAIL'}")
        out.append(f"  => {'SUPPORTED' if all_ok else 'NOT SUPPORTED'}")

    out.append("")
    out.append("Prediction 2 -- RoPE smallest degradation ratio at each length beyond the training length")
    lengths = sorted({n for row in rows.values() for n in row["ratios"]}, key=int)
    if not lengths or "rope" not in rows:
        out.append("  (no RoPE extrapolation results yet)")
    else:
        all_ok = True
        for n in lengths:
            candidates = {v: row["ratios"][n] for v, row in rows.items() if n in row["ratios"]}
            if "rope" not in candidates:
                out.append(f"  @ {n}: RoPE missing")
                all_ok = False
                continue
            best = min(candidates, key=lambda v: candidates[v][0])
            detail = ", ".join(f"{v} {m:.3f}" for v, (m, _) in sorted(candidates.items(), key=lambda kv: kv[1][0]))
            ok = best == "rope"
            all_ok &= ok
            out.append(f"  @ {n}: smallest = {best} | {detail} | {'PASS' if ok else 'FAIL'}")
        out.append(f"  => {'SUPPORTED' if all_ok else 'NOT SUPPORTED'}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results and check the README predictions")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument("--smooth", type=int, default=16, help="moving-average window for the curves")
    args = parser.parse_args()

    results = load_results(Path(args.runs_dir))
    if not results:
        print(f"No eval.json files under {args.runs_dir}/ -- run eval_extrapolation.py first.")
        return
    groups = group_by_variant(results)
    ctx_len = results[0]["ctx_len"]
    out_dir = Path(args.out_dir)
    plot_loss_by_position(groups, ctx_len, out_dir / "loss_by_position.png", args.smooth)

    rows = build_summary(groups)
    text = summary_markdown(rows) + "\n\n" + check_predictions(rows) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out_dir / 'loss_by_position.png'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
