"""Experiment configuration (AI-scaffolded; extend freely, but keep existing field names).

One dataclass holds every knob. It is saved as config.json in each run directory and
inside each checkpoint, so any run can be reproduced or re-evaluated later.
Deliberately free of torch imports so scripts that only read results stay light.
"""
import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Sequence, Tuple

# The four positional-encoding variants under test (also the order used in plots/tables).
VARIANTS = ("learned", "sinusoidal", "rope", "nope")


@dataclass
class Config:
    # --- experiment identity -------------------------------------------------
    variant: str = "rope"          # which positional encoding to use (one of VARIANTS)
    seed: int = 0                  # controls init, batch order; sweep uses 0, 1, 2

    # --- model shape (README: ~10M params) -----------------------------------
    n_layer: int = 6               # number of transformer blocks
    n_head: int = 6                # attention heads per block
    d_model: int = 384             # residual-stream width
    ctx_len: int = 256             # TRAINING context length (eval goes longer)
    dropout: float = 0.2           # regularisation; the data is small (~1M chars)
    vocab_size: int = 65           # TinyShakespeare character vocabulary

    # --- optimisation (small-GPT defaults for char-level Shakespeare) --------
    batch_size: int = 64
    max_iters: int = 5000
    lr: float = 1e-3               # peak learning rate (after warmup)
    min_lr: float = 1e-4           # floor of the cosine decay
    warmup_iters: int = 100        # linear warmup steps
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0         # max global gradient norm

    # --- logging / validation during training --------------------------------
    eval_interval: int = 250       # iterations between validation passes
    eval_iters: int = 100          # batches averaged per validation pass

    # --- paths / hardware ----------------------------------------------------
    data_dir: str = "data"
    out_dir: str = "runs"
    device: str = "auto"           # "auto" -> cuda if available else cpu
    dtype: str = "bfloat16"        # autocast dtype on CUDA; "float32" disables autocast

    # --- post-training extrapolation eval -----------------------------------
    eval_lengths: Tuple[int, ...] = (256, 512, 1024)
    eval_batch_size: int = 16      # windows per forward pass at eval time

    # ------------------------------------------------------------------------
    def __post_init__(self):
        # Fail fast on settings that would otherwise break deep inside the model.
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even (RoPE rotates channels in pairs)")

    # ---- derived values -----------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def name(self) -> str:
        """Unique run name, e.g. 'rope_s0'."""
        return f"{self.variant}_s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    @property
    def ckpt_path(self) -> Path:
        return self.run_dir / "ckpt.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def eval_path(self) -> Path:
        return self.run_dir / "eval.json"

    # ---- (de)serialisation --------------------------------------------------
    def to_json(self, path: Optional[Path] = None) -> Path:
        """Write the config as JSON (default: <run_dir>/config.json)."""
        path = Path(path) if path else self.run_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build a Config from a dict, ignoring unknown keys and fixing JSON lists -> tuples."""
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in d.items():
            if key not in known:
                continue  # tolerate configs saved by an older/newer version
            if isinstance(getattr(cls(), key), tuple) and isinstance(value, list):
                value = tuple(value)  # JSON has no tuples
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def tiny(cls, **overrides) -> "Config":
        """A very small CPU-friendly config used by the tests (runs in seconds)."""
        base = dict(
            n_layer=2, n_head=4, d_model=64,      # head_dim = 16 (even, as RoPE needs)
            ctx_len=32, dropout=0.0,              # no dropout => deterministic tests
            batch_size=8, max_iters=50, eval_interval=25, eval_iters=2,
            eval_lengths=(32, 64, 128), eval_batch_size=4,
            device="cpu", dtype="float32",
        )
        base.update(overrides)
        return cls(**base)


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    """Build a Config from command-line flags (one flag per scalar field).

    Example: python -m src.train --variant rope --seed 1 --max_iters 2000
    """
    parser = argparse.ArgumentParser(description="One run of the positional-encoding experiment")
    defaults = Config()
    for f in fields(Config):
        default = getattr(defaults, f.name)
        flag = f"--{f.name}"
        if f.name == "variant":
            parser.add_argument(flag, default=default, choices=VARIANTS)
        elif isinstance(default, tuple):
            # e.g. --eval_lengths 256 512 1024
            parser.add_argument(flag, nargs="+", type=int, default=list(default))
        else:
            parser.add_argument(flag, type=type(default), default=default)
    kwargs = vars(parser.parse_args(argv))
    kwargs["eval_lengths"] = tuple(kwargs["eval_lengths"])
    return Config(**kwargs)
