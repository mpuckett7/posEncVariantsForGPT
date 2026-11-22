# GPT positional-encoding experiment (see README.md)

## Roles
- Human writes: src/posenc.py, src/model.py, src/train.py
- AI may write: data/prepare.py, src/data.py, src/config.py, src/utils.py,
  src/metrics.py, scripts/*.py, eval_extrapolation.py, tests/
- For human-owned files: review, find bugs, explain. Do not write or edit the
  implementation unless asked explicitly in that message.

## Rules
- Banned in model code: nn.MultiheadAttention, nn.Transformer,
  F.scaled_dot_product_attention (allowed only as a test oracle)
- Tests run on CPU with a tiny config (Config.tiny()) in seconds
- Never launch training sweeps locally; they run on the GPU desktop
- Predictions in README.md are fixed before any run; do not edit them

## Interfaces other files rely on (documented in the stub docstrings)
- posenc: PosEnc.embed(x, positions) and PosEnc.rotate_qk(q, k, positions);
  build_posenc(cfg) returns the variant named by cfg.variant
- model: GPT(cfg).forward(idx, targets=None) -> (logits [B,T,V], loss or None);
  GPT.posenc, GPT.blocks[i].attn; attention exposes .qkv, .proj, .save_attn, .last_attn
- train: `python -m src.train --variant V --seed S` writes runs/<V>_s<S>/
  {config.json, metrics.jsonl, ckpt.pt}; ckpt.pt holds {"model", "config", "iter", "val_loss"}
