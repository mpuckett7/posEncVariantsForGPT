"""Positional encodings.  >>> HUMAN-OWNED: you write the variants. <<<

INTERFACE (do not change names/signatures: model.py, tests and eval depend on them)
    embed(x, positions)          -> x'       called ONCE, right after the token embedding.
                                             x: [B, T, D]
    rotate_qk(q, k, positions)   -> (q', k') called inside EVERY attention layer.
                                             q, k: [B, H, T, Dh]
    positions: LongTensor [T] holding the absolute indices 0..T-1. It is passed explicitly
               so that evaluating at length 512 / 1024 needs no special casing.

Each variant overrides only the hook(s) it needs. The base-class hooks are identities on
purpose, so a variant that does nothing in one hook (e.g. RoPE in `embed`) needs no code there.
"""
import torch
import torch.nn as nn


class PosEnc(nn.Module):
    """Base class: the identity for both hooks (this alone already IS the NoPE behaviour)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # Config dataclass (d_model, head_dim, ctx_len, ...)

    def embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x  # default: leave the token embeddings untouched

    def rotate_qk(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        return q, k  # default: leave queries/keys untouched


class LearnedAbsolute(PosEnc):
    """Trainable table of one vector per position, ADDED to the token embeddings."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # TODO 1: create the table: nn.Embedding(cfg.ctx_len, cfg.d_model)
        #         (only ctx_len rows exist -> positions >= ctx_len have no vector!)
        # TODO 2: initialise it with normal(mean=0, std=0.02)
        raise NotImplementedError

    def embed(self, x, positions):
        # TODO 3: if positions.max() >= cfg.ctx_len, raise ValueError with a clear message.
        #         (tests/test_model.py expects a ValueError; this is the "cannot be
        #         evaluated past the trained length" finding for the write-up.)
        # TODO 4: return x + table(positions)   # [T, D] broadcasts over the batch dim
        raise NotImplementedError


class SinusoidalAbsolute(PosEnc):
    """Fixed sine/cosine vectors, ADDED to the token embeddings (Vaswani et al. 2017, Sec. 3.5)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # No parameters needed. TODO: nothing here unless you want to store constants.

    def embed(self, x, positions):
        # TODO 1: build pe of shape [T, D] directly from `positions` (NOT a fixed-size buffer,
        #         so any length works):
        #            pe[pos, 2i]   = sin(pos / 10000^(2i/D))
        #            pe[pos, 2i+1] = cos(pos / 10000^(2i/D))
        #         Hint: div_term = exp(arange(0, D, 2) * -(log(10000) / D)); D must be even.
        # TODO 2: compute in float32, then cast to x.dtype.
        # TODO 3 (DECISION THAT AFFECTS RESULTS): scale. Vaswani multiplies the token
        #         embeddings by sqrt(D) before adding pe. With embeddings initialised at
        #         std 0.02 and unit-amplitude sinusoids (norm ~ sqrt(D/2)), position would
        #         swamp content. Choose ONE convention (e.g. scale pe down, or scale the token
        #         embeddings only for this variant in GPT.forward) and write it in the README.
        # TODO 4: return x + pe   # [T, D] broadcasts over the batch dim
        raise NotImplementedError


class RoPE(PosEnc):
    """Rotary embeddings (Su et al. 2021): rotate channel PAIRS of q and k by position-dependent angles."""

    def __init__(self, cfg, base: float = 10000.0):
        super().__init__(cfg)
        # TODO 1: head_dim = cfg.head_dim (must be even).
        # TODO 2: inv_freq[i] = base ** (-2i / head_dim) for i = 0 .. head_dim/2 - 1,
        #         stored with self.register_buffer("inv_freq", ..., persistent=False).
        raise NotImplementedError

    def rotate_qk(self, q, k, positions):
        # TODO 1: angles = positions[:, None].float() * inv_freq[None, :]   -> [T, Dh/2]
        #         cos, sin = angles.cos(), angles.sin(); reshape to [1, 1, T, Dh/2].
        # TODO 2: CHOOSE one pairing convention and use it identically for q and k:
        #           (a) interleaved pairs (x0,x1), (x2,x3), ...
        #           (b) half-split: first half of channels pairs with the second half
        #         rotation of a pair (a, b):  (a*cos - b*sin,  a*sin + b*cos)
        # TODO 3: do the math in float32 and cast back to q.dtype.
        # TODO 4: rotate q and k ONLY. Never v, never the residual stream.
        # TODO 5: return (q_rot, k_rot) with the same shapes as the inputs.
        # CHECK:  tests/test_model.py::test_rope_scores_depend_only_on_relative_distance
        raise NotImplementedError


class NoPE(PosEnc):
    """No positional encoding at all: position can only be inferred from the causal mask."""

    # TODO: no code is needed (the base-class hooks are identities). Instead write a short
    #       comment/docstring here answering: with NO position input, how can a causal
    #       transformer still tell token 3 from token 200?  (This is viva question 3.)
    pass


_REGISTRY = {
    "learned": LearnedAbsolute,
    "sinusoidal": SinusoidalAbsolute,
    "rope": RoPE,
    "nope": NoPE,
}


def build_posenc(cfg) -> PosEnc:
    """Factory: return the encoding named by cfg.variant."""
    return _REGISTRY[cfg.variant](cfg)
