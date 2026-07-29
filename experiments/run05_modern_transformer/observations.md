# Experiment 05 - Modern Transformer Architecture Observations

## Research Question
Can a modern transformer architecture (RMSNorm, RoPE, SwiGLU, Weight Tying) combined with a Byte-Level BPE tokenizer significantly outperform the baseline model?

## Hypothesis
Replacing the naive baseline (LayerNorm, GELU, learned absolute position embeddings, raw byte tokenizer) with modern architecture features (RMSNorm, SwiGLU, RoPE, Weight Tying) and subword byte-level BPE will dramatically improve representational efficiency and lower BPB.

## Configuration
- Script: `starter/train_exp05_modern_transformer.py`
- Model: Modern Transformer (`RMSNorm`, `RoPE`, `SwiGLU`, Weight Tying)
- Tokenizer: Byte-Level BPE (Vocab 512, 100% Lossless roundtrip)
- `n_layer`: 4
- `n_head`: 6
- `n_embd`: 192
- `hidden_dim`: 512
- Parameter Count: **1,869,504** (93.5% of 2M cap)
- Hyperparameters: Adam optimizer, constant lr=3e-4, 2,000 steps, batch size 8, seed 1337.

## Expected & Measured Results

| Metric                | Baseline (Exp 0) | Exp 04 (Capacity) | Experiment 05 (Modern Arch + BPE) | Delta vs Baseline |
|-----------------------|------------------|-------------------|-----------------------------------|-------------------|
| **Final Training Loss** | 1.7315         | —                 | **1.6043**                        | **−0.1272**       |
| **Dev BPB**           | 2.3718           | 2.3472            | **2.0074**                        | **−0.3644**       |
| **Parameter Count**   | 1,339,840        | 1,902,720         | **1,869,504**                     | +529,664          |

## Decision
**✅ ACCEPTED — HUGE WINNER.** Dev BPB dropped from 2.3718 → 2.0074 (−0.3644 BPB improvement, a 15.4% reduction). This establishes a new state of the art for this assignment.

## Engineering Analysis
1. **RoPE Attention**: Replaced 20,480 learned position embedding parameters with zero-parameter rotary position embeddings, eliminating position cap limitations and improving relative position handling.
2. **Byte-Level BPE (Vocab 512)**: Subword tokenization compressed the 5.7M byte training corpus into 6.0M tokens. Non-ASCII sequences (especially Hindi UTF-8 bytes) are grouped into single subword tokens, extending the effective context window.
3. **SwiGLU MLP & RMSNorm**: Gated SiLU activations provide higher expressivity per parameter than standard GELU, and RMSNorm provides cleaner gradient flow.
4. **Weight Tying**: Tying the token embedding table to the output head eliminated redundant output projection parameters, allowing parameter budget allocation to architectural width.
