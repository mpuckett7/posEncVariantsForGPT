"""Attention diagnostics: one head's heatmap per variant + attention entropy vs position (AI-scaffolded).

Needs the optional hooks from the contract in src/model.py:
    model.blocks[i].attn.save_attn (bool)  and  model.blocks[i].attn.last_attn ([B, H, T, T]).
Usage (from the repo root, on a machine that has the checkpoints):
    python scripts/attn_analysis.py --seed 0 --length 512 --layer 3 --head 0
Outputs: figures/attn_heatmaps.png, figures/attn_entropy.png

Why entropy? A softmax over t+1 keys has at most log(t+1) nats of entropy (uniform attention).
If a model's attention entropy climbs toward that ceiling past the training length, its attention is
"dispersing" -- one candidate explanation for degradation when the context gets longer.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402
import torch                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import VARIANTS, Config        # noqa: E402
from src.data import fixed_eval_windows, load_split  # noqa: E402
from src.model import GPT                      # noqa: E402
from src.utils import get_device               # noqa: E402


def load_model(run_dir: Path, device: str):
    """Rebuild a GPT from a checkpoint and switch it to eval mode."""
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def collect_attention(model, x: torch.Tensor) -> list:
    """Run one forward pass with attention saving on; return a list of [H, T, T] tensors (one per layer)."""
    for block in model.blocks:
        block.attn.save_attn = True            # ask each attention layer to keep its softmax weights
    model(x)
    maps = [block.attn.last_attn[0].float().cpu() for block in model.blocks]   # batch element 0
    for block in model.blocks:
        block.attn.save_attn = False
        block.attn.last_attn = None            # free memory
    return maps


def entropy_per_position(att: torch.Tensor) -> np.ndarray:
    """Shannon entropy (nats) of each query's attention distribution, averaged over heads. att: [H, T, T]."""
    ent = -(att * torch.log(att.clamp_min(1e-12))).sum(dim=-1)   # [H, T]
    return ent.mean(dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention heatmaps and entropy for each variant")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=512, help="window length to analyse (try > ctx_len)")
    parser.add_argument("--layer", type=int, default=3, help="layer shown in the heatmap")
    parser.add_argument("--head", type=int, default=0, help="head shown in the heatmap")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heat, entropy, ctx_len = {}, {}, None
    for variant in VARIANTS:
        run_dir = Path(args.runs_dir) / f"{variant}_s{args.seed}"
        if not (run_dir / "ckpt.pt").exists():
            print(f"[skip] {variant}: no checkpoint at {run_dir}")
            continue
        model, cfg = load_model(run_dir, device)
        ctx_len = cfg.ctx_len
        val = load_split(cfg.data_dir, "val")
        x, _ = fixed_eval_windows(val, args.length, 1)                  # one fixed window, same for all variants
        try:
            maps = collect_attention(model, x[:1].to(device))
        except ValueError as err:                                      # learned embeddings past ctx_len
            print(f"[skip] {variant}: {err}")
            continue
        layer = min(args.layer, len(maps) - 1)
        heat[variant] = maps[layer][min(args.head, maps[layer].shape[0] - 1)].numpy()
        # Entropy averaged over layers and heads -> one curve per variant.
        entropy[variant] = np.mean([entropy_per_position(m) for m in maps], axis=0)

    if not heat:
        print("Nothing to plot (no usable checkpoints).")
        return

    # ---- heatmaps: rows = query position, columns = key position; lines mark the training length ----
    fig, axes = plt.subplots(1, len(heat), figsize=(4.5 * len(heat), 4.5), squeeze=False)
    for ax, (variant, a) in zip(axes[0], heat.items()):
        ax.imshow(a, cmap="viridis", aspect="auto", vmin=0.0, vmax=float(np.quantile(a[a > 0], 0.99)))
        ax.axhline(ctx_len, color="white", linestyle="--", linewidth=0.8)
        ax.axvline(ctx_len, color="white", linestyle="--", linewidth=0.8)
        ax.set_title(f"{variant} (layer {args.layer}, head {args.head})")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
    fig.tight_layout()
    fig.savefig(out_dir / "attn_heatmaps.png", dpi=150)
    plt.close(fig)

    # ---- entropy vs position, with the uniform-attention ceiling log(t + 1) for reference ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant, ent in entropy.items():
        ax.plot(np.arange(len(ent)), ent, label=variant)
    t = np.arange(args.length)
    ax.plot(t, np.log(t + 1), color="black", linestyle=":", label="uniform attention: log(t+1)")
    ax.axvline(ctx_len, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("query position")
    ax.set_ylabel("attention entropy (nats)")
    ax.set_title(f"Attention entropy, window length {args.length}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "attn_entropy.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'attn_heatmaps.png'} and {out_dir / 'attn_entropy.png'}")


if __name__ == "__main__":
    main()
