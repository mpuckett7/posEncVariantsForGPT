"""Correctness tests for the from-scratch GPT (AI-written from the spec; YOU make them pass).

Run from the repo root:  pytest -q
Everything uses Config.tiny() (2 layers, d_model 64, ctx 32) so the whole file runs in
seconds on a laptop CPU. Until you implement src/model.py and src/posenc.py these fail
with NotImplementedError -- that is the expected starting point.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.config import VARIANTS, Config
from src.model import GPT, CausalSelfAttention
from src.posenc import build_posenc


@pytest.fixture(autouse=True)
def _seed():
    """Same random numbers every test, so failures are reproducible."""
    torch.manual_seed(0)


def tiny(variant: str) -> Config:
    return Config.tiny(variant=variant)


# --------------------------------------------------------------------------- 1. causality
@pytest.mark.parametrize("variant", VARIANTS)
def test_causality(variant):
    """Changing tokens AFTER position t must not change the logits at positions <= t."""
    cfg = tiny(variant)
    model = GPT(cfg).eval()                      # eval(): dropout off (also 0 in tiny cfg)
    T, t = cfg.ctx_len, cfg.ctx_len // 2
    idx = torch.randint(0, cfg.vocab_size, (2, T))
    idx_future_changed = idx.clone()
    # Guarantee a real change: shift every future token id by one (mod vocab).
    idx_future_changed[:, t + 1 :] = (idx[:, t + 1 :] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_a, _ = model(idx)
        logits_b, _ = model(idx_future_changed)
    # Past + present positions must be identical...
    assert torch.allclose(logits_a[:, : t + 1], logits_b[:, : t + 1], atol=1e-5)
    # ...and the test must be able to fail: positions that DO see the change must differ.
    assert not torch.allclose(logits_a[:, t + 1 :], logits_b[:, t + 1 :], atol=1e-5)


# --------------------------------------------------------------------------- 2. oracle match
def test_attention_matches_pytorch_sdpa_oracle():
    """Your attention equals F.scaled_dot_product_attention(is_causal=True) to ~1e-5.

    Uses the 'nope' variant so no q/k rotation interferes; RoPE has its own test below.
    Relies on the layout contract in src/model.py: attn.qkv -> [q|k|v], attn.proj output layer.
    """
    cfg = tiny("nope")
    attn = CausalSelfAttention(cfg).eval()
    posenc = build_posenc(cfg)
    B, T, D = 2, 16, cfg.d_model
    x = torch.randn(B, T, D)
    positions = torch.arange(T)
    with torch.no_grad():
        mine = attn(x, posenc, positions)

        # Reference path built from the SAME weights but PyTorch's fused attention.
        q, k, v = attn.qkv(x).split(D, dim=2)

        def to_heads(t):  # [B, T, D] -> [B, H, T, Dh]
            return t.view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = ref.transpose(1, 2).contiguous().view(B, T, D)   # merge heads
        ref = attn.proj(ref)
    assert torch.allclose(mine, ref, atol=1e-5)


# --------------------------------------------------------------------------- 3. RoPE property
def test_rope_scores_depend_only_on_relative_distance():
    """With identical q (and identical k) at every position, score[m, n] must depend only on m - n."""
    cfg = tiny("rope")
    posenc = build_posenc(cfg)
    T, Dh = 32, cfg.head_dim
    q0, k0 = torch.randn(1, 1, 1, Dh), torch.randn(1, 1, 1, Dh)
    q = q0.expand(1, 1, T, Dh).contiguous()      # the same vector at every position
    k = k0.expand(1, 1, T, Dh).contiguous()
    positions = torch.arange(T)
    q_rot, k_rot = posenc.rotate_qk(q, k, positions)

    # Rotations preserve vector length.
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5)

    scores = q_rot[0, 0] @ k_rot[0, 0].T         # [T, T]: scores[m, n] = q_m . k_n
    for shift in (1, 3, 7):
        # Sliding both positions by `shift` keeps m - n fixed, so the score must not change.
        assert torch.allclose(scores[: T - shift, : T - shift], scores[shift:, shift:], atol=1e-4)


# --------------------------------------------------------------------------- 4. initial loss
@pytest.mark.parametrize("variant", VARIANTS)
def test_initial_loss_is_about_log_vocab(variant):
    """An untrained model should guess ~uniformly: loss ~ ln(65) = 4.17 (within 0.3)."""
    cfg = tiny(variant)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.ctx_len))
    targets = torch.randint(0, cfg.vocab_size, (8, cfg.ctx_len))
    with torch.no_grad():
        _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3


# --------------------------------------------------------------------------- 5. overfit one batch
def test_can_overfit_one_batch():
    """Memorise a single fixed batch. If the loss cannot fall, something in the model is broken.

    The thresholds are heuristics: if you sit just above 0.5, try more steps before suspecting a bug.
    """
    cfg = tiny("rope")
    model = GPT(cfg).train()                     # dropout is 0 in the tiny config
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(0, cfg.vocab_size, (4, cfg.ctx_len + 1))
    idx, targets = tokens[:, :-1], tokens[:, 1:]
    first = None
    loss = None
    for _ in range(500):
        _, loss = model(idx, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        if loss.item() < 0.05:                   # clearly memorised; stop early
            break
    assert loss.item() < 0.5
    assert loss.item() < 0.2 * first


# --------------------------------------------------------------------------- 6. length handling
def test_learned_rejects_positions_beyond_training_length():
    """Learned embeddings only exist for positions < ctx_len; longer input must raise ValueError."""
    cfg = tiny("learned")
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.ctx_len + 1))
    with pytest.raises(ValueError):
        model(idx)


@pytest.mark.parametrize("variant", ["sinusoidal", "rope", "nope"])
def test_other_variants_run_beyond_training_length(variant):
    """These variants must accept 4x the training length (no hard-coded mask/buffer sizes)."""
    cfg = tiny(variant)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.ctx_len * 4))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.shape == (1, cfg.ctx_len * 4, cfg.vocab_size)
    assert torch.isfinite(logits).all()
