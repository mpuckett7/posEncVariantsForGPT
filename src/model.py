"""Decoder-only transformer (GPT).  >>> HUMAN-OWNED: you write the modules. <<<

Rules: nn.Linear / nn.Embedding / nn.LayerNorm / nn.Dropout / autograd are fine.
       NOT allowed: nn.MultiheadAttention, nn.Transformer, F.scaled_dot_product_attention
       (the last one is only used as an oracle inside tests/test_model.py).

CONTRACT (tests, eval and plotting scripts rely on these exact names/shapes)
    GPT(cfg).forward(idx, targets=None) -> (logits [B, T, V], loss or None)
        - logits for EVERY position (not just the last one)
        - must work for any T, including T > cfg.ctx_len (we evaluate at 512 / 1024)
    GPT.posenc                  the PosEnc instance from src.posenc.build_posenc(cfg)
    GPT.blocks                  nn.ModuleList of Block
    Block.attn                  a CausalSelfAttention
    CausalSelfAttention.qkv     nn.Linear(d_model, 3*d_model); output layout is [ q | k | v ],
                                and each of q, k, v is laid out as [heads x head_dim]
    CausalSelfAttention.proj    nn.Linear(d_model, d_model)  (output projection)
    CausalSelfAttention.save_attn  bool flag (default False)
    CausalSelfAttention.last_attn  [B, H, T, T] softmax weights, set only when save_attn is True
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.posenc import build_posenc


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only look at positions <= t."""

    def __init__(self, cfg):
        super().__init__()
        # TODO 1: store n_head, head_dim, d_model from cfg (assert d_model % n_head == 0).
        # TODO 2: self.qkv  = nn.Linear(d_model, 3 * d_model)      # REQUIRED name
        # TODO 3: self.proj = nn.Linear(d_model, d_model)          # REQUIRED name
        # TODO 4: attention dropout and residual dropout (nn.Dropout(cfg.dropout)).
        # TODO 5: self.save_attn = False; self.last_attn = None     # REQUIRED (heatmap script)
        raise NotImplementedError

    def forward(self, x, posenc, positions):
        """x: [B, T, D]; posenc: PosEnc; positions: LongTensor [T]. Returns [B, T, D]."""
        # TODO 1: q, k, v = self.qkv(x).split(D, dim=2); reshape each to [B, T, H, Dh]
        #         then transpose to [B, H, T, Dh].
        # TODO 2: q, k = posenc.rotate_qk(q, k, positions)      # identity except for RoPE
        # TODO 3: scores = q @ k.transpose(-2, -1) / sqrt(Dh)   # [B, H, T, T]
        #         (viva Q1: why divide by sqrt(Dh)?)
        # TODO 4: causal mask built from THIS call's T (do not hard-code ctx_len!):
        #         mask = torch.ones(T, T, device=x.device).tril(); fill scores where mask == 0
        #         with -inf BEFORE the softmax.
        # TODO 5: att = softmax(scores, dim=-1)  (float32 is safer under autocast); dropout.
        # TODO 6: if self.save_attn: self.last_attn = att.detach()
        # TODO 7: y = att @ v -> transpose back to [B, T, H*Dh] -> self.proj -> residual dropout.
        raise NotImplementedError


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, nonlinearity, project back."""

    def __init__(self, cfg):
        super().__init__()
        # TODO: fc = Linear(d_model, 4*d_model); GELU; proj = Linear(4*d_model, d_model); Dropout.
        raise NotImplementedError

    def forward(self, x):
        # TODO: fc -> GELU -> proj -> dropout
        raise NotImplementedError


class Block(nn.Module):
    """One pre-LN transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, cfg):
        super().__init__()
        # TODO: self.ln1, self.attn (REQUIRED name), self.ln2, self.mlp
        raise NotImplementedError

    def forward(self, x, posenc, positions):
        # TODO: x = x + self.attn(self.ln1(x), posenc, positions)
        #       x = x + self.mlp(self.ln2(x))
        # (viva Q2: why does pre-LN train more stably than post-LN at this depth?)
        raise NotImplementedError


class GPT(nn.Module):
    """Token embedding -> position handling -> N blocks -> final LayerNorm -> vocab logits."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # TODO 1: self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # TODO 2: self.posenc = build_posenc(cfg)                      # REQUIRED name
        # TODO 3: self.drop = nn.Dropout(cfg.dropout)
        # TODO 4: self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])  # REQUIRED
        # TODO 5: self.ln_f = nn.LayerNorm(cfg.d_model); self.lm_head = nn.Linear(d_model, vocab, bias=False)
        # TODO 6 (decision): tie lm_head.weight to tok_emb.weight? Record the choice in the README.
        # TODO 7: init weights: normal(0, 0.02) for Linear/Embedding, zeros for biases.
        #         Optional: scale residual projections (attn.proj, mlp.proj) by 1/sqrt(2*n_layer).
        raise NotImplementedError

    def forward(self, idx, targets=None):
        """idx: LongTensor [B, T]; targets: LongTensor [B, T] or None.

        Returns (logits [B, T, V], loss) where loss is a scalar cross-entropy in nats,
        or None when targets is None.
        """
        # TODO 1: B, T = idx.shape; positions = torch.arange(T, device=idx.device)
        #         (do NOT assert T <= ctx_len; eval runs at 512 / 1024)
        # TODO 2: x = self.tok_emb(idx); x = self.posenc.embed(x, positions); x = self.drop(x)
        # TODO 3: for block in self.blocks: x = block(x, self.posenc, positions)
        # TODO 4: x = self.ln_f(x); logits = self.lm_head(x)           # [B, T, V]
        # TODO 5: loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) if targets is not None
        # TODO 6: return logits, loss
        raise NotImplementedError
