# Positional Encodings in a From-Scratch GPT

## Overview
Build a small decoder-only transformer (GPT) from scratch in
PyTorch, then run a controlled experiment on how it represents token position.
Models are trained at context length 256 and evaluated at 256, 512, and 1024
to measure length extrapolation.

## Setup
- Data: character-level TinyShakespeare (vocab 65), 90/10 train/val split
- Model: ~10M parameters, 6 layers, d_model 384, 6 heads, pre-LN
- Runs: 4 variants x 3 seeds = 12 runs
- Rule: no `nn.MultiheadAttention`, `nn.Transformer`, or
  `F.scaled_dot_product_attention` (the last only as a test oracle)

## What I'm building directly
- `posenc.py`: the four positional encoding variants
- `model.py`: causal multi-head attention, transformer block, GPT
- `train.py`: training loop (AdamW, warmup + cosine decay, gradient clipping,
  validation loss)

## What I'm building with AI-assisted pair programming
- Data preparation and batching, run config, sweep runner
- Evaluation script (loss by position) and plotting
- Tests: causality, oracle match against PyTorch attention, RoPE relative-position
  property, initial loss ~ ln(65), overfit one batch
- Code review of my modules

## What I'm testing
Four positional encoding variants:
1. Learned absolute
2. Sinusoidal
3. RoPE (rotary)
4. NoPE (no positional encoding)

Metrics:
- **In-distribution:** mean validation loss over positions 0-255
- **Degradation ratio at length N (512, 1024):** mean loss over the last 256
  positions of an N-length window divided by mean loss over positions 192-255
- Learned absolute has no embeddings past position 255, so it is excluded from
  the extrapolation comparison.

## Predictions (recorded before any training runs)
1. **In-distribution:** NoPE will have the lowest mean loss up to position 256,
   by more than one seed standard deviation compared to every other variant.
2. **Extrapolation:** RoPE will have the smallest degradation ratio at both
   length 512 and length 1024.

## Results
_To be filled in after the sweep._
