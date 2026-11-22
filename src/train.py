"""Training loop.  >>> HUMAN-OWNED: you write the functions. <<<

Run one experiment:   python -m src.train --variant rope --seed 0
(flags come from Config; see src/config.py::parse_args)

OUTPUT CONTRACT (the sweep, eval and plotting scripts depend on it)
    runs/<variant>_s<seed>/config.json   written with cfg.to_json()
    runs/<variant>_s<seed>/metrics.jsonl one JSON object per evaluation:
                                         {"iter": int, "train_loss": float, "val_loss": float, "lr": float}
                                         (losses in nats, as PyTorch returns them)
    runs/<variant>_s<seed>/ckpt.pt       torch.save({"model": state_dict, "config": asdict(cfg),
                                                     "iter": int, "val_loss": float}, path)
Helpers you can use: src.utils (set_seed, get_device, append_jsonl, count_params),
                     src.data (load_split, load_meta, get_batch).
"""
import math

import torch

from src.config import Config, parse_args
from src.data import get_batch, load_meta, load_split
from src.model import GPT
from src.utils import append_jsonl, count_params, get_device, set_seed


def get_lr(it: int, cfg: Config) -> float:
    """Learning rate at iteration `it`: linear warmup, then cosine decay to cfg.min_lr."""
    # TODO 1: if it < cfg.warmup_iters: return cfg.lr * (it + 1) / cfg.warmup_iters
    # TODO 2: if it >= cfg.max_iters: return cfg.min_lr
    # TODO 3: progress = (it - warmup_iters) / (max_iters - warmup_iters)   # in [0, 1]
    #         coeff = 0.5 * (1 + cos(pi * progress)); return min_lr + coeff * (lr - min_lr)
    raise NotImplementedError


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices/embeddings only."""
    # TODO 1: split parameters: dim >= 2 (weight matrices, embeddings) -> decay;
    #         dim < 2 (biases, LayerNorm gains) -> no decay.
    # TODO 2: torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model, data, cfg: Config, device: str) -> float:
    """Mean loss (nats) over cfg.eval_iters batches from `data`, with dropout off."""
    # TODO 1: model.eval()
    # TODO 2: use a torch.Generator seeded with a FIXED number (e.g. 1234) so every variant
    #         and seed is validated on the same batches -> comparable curves.
    # TODO 3: average the loss over cfg.eval_iters calls of get_batch(...)
    # TODO 4: model.train() before returning; return the float mean.
    raise NotImplementedError


def train(cfg: Config) -> None:
    """Train one (variant, seed) run and write config.json / metrics.jsonl / ckpt.pt."""
    # TODO 1: set_seed(cfg.seed); device = get_device(cfg.device); cfg.run_dir.mkdir(parents=True, exist_ok=True)
    #         cfg.to_json()   # saves runs/<name>/config.json
    # TODO 2: load data: train = load_split(cfg.data_dir, "train"); val = load_split(..., "val")
    #         meta = load_meta(cfg.data_dir); assert meta["vocab_size"] == cfg.vocab_size
    # TODO 3: model = GPT(cfg).to(device); print count_params(model) (README says ~10M)
    # TODO 4: optimizer = build_optimizer(model, cfg)
    # TODO 5: gen = torch.Generator().manual_seed(cfg.seed)   # reproducible batch order
    # TODO 6: mixed precision: on CUDA with cfg.dtype == "bfloat16" wrap the forward in
    #         torch.autocast(device_type="cuda", dtype=torch.bfloat16); otherwise use
    #         contextlib.nullcontext().
    # TODO 7: loop `for it in range(cfg.max_iters)`:
    #           lr = get_lr(it, cfg); set it on every optimizer param group
    #           x, y = get_batch(train, cfg.batch_size, cfg.ctx_len, device, gen)
    #           _, loss = model(x, y); loss.backward()
    #           torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    #           optimizer.step(); optimizer.zero_grad(set_to_none=True)
    #           every cfg.eval_interval (and at the last iter): estimate_loss on train and val,
    #           print, and append_jsonl(cfg.metrics_path, {...}) per the contract above.
    # TODO 8 (DECISION): pick ONE checkpoint rule (save the final model, or the best-val model)
    #         and use it for every variant. Note it in the README. Then torch.save the ckpt dict.
    raise NotImplementedError


def main(argv=None) -> None:
    cfg = parse_args(argv)  # command-line flags -> Config
    train(cfg)


if __name__ == "__main__":
    main()
