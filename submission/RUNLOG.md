# RUNLOG — Plivo ML Assignment

Chronological log of all experiments. All measured values from `evaluate.py`. No fabricated results.

---

## Baseline — Experiment 0

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Establish reference BPB score using unmodified starter code. |
| **What changed**   | Nothing — pure baseline (LayerNorm, GELU, ByteTokenizer).|
| **Training Loss**  | 1.7315                                                   |
| **Dev BPB**        | 2.3718                                                   |
| **Runtime**        | 54 seconds                                               |
| **Params**         | 1,339,840                                                |
| **Decision**       | ✅ Reference benchmark.                                  |
| **Lessons Learned**| Baseline uses only 67% of parameter budget. Model is capacity-constrained. Constant Adam (lr=3e-4) is strong for 2,000 steps. |

---

## Experiment 01 — Optimizer Ablation

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | AdamW with weight_decay=0.1 will outperform Adam in BPB. |
| **What changed**   | `torch.optim.Adam` → `torch.optim.AdamW(weight_decay=0.1)` |
| **Training Loss**  | 1.7399                                                   |
| **Dev BPB**        | 2.3790                                                   |
| **Runtime**        | 58 seconds                                               |
| **Params**         | 1,339,840                                                |
| **Decision**       | ❌ Rejected. BPB worsened by +0.0072.                    |
| **Lessons Learned**| Weight decay is a regularizer. The small model at 2,000 steps is underfitting, not overfitting. Regularization worsens underfitting. |

---

## Experiment 02 — LR Scheduler Ablation

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | 100-step linear warmup + cosine decay will lower BPB.   |
| **What changed**   | Added linear warmup (100 steps) + cosine decay over 1900 steps. |
| **Training Loss**  | 1.9637                                                   |
| **Dev BPB**        | 2.6976                                                   |
| **Runtime**        | 53 seconds                                               |
| **Params**         | 1,339,840                                                |
| **Decision**       | ❌ Rejected. BPB significantly worsened by +0.3258.      |
| **Lessons Learned**| Cosine decay reduces step size to near-zero at step 2000, wasting compute steps when training budget is small. Constant LR is superior for 2000 steps. |

---

## Experiment 03 — Weight Tying

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Tying head to embedding weights will improve parameter efficiency without degrading BPB. |
| **What changed**   | `cfg.tie_weights = True`                                 |
| **Training Loss**  | 1.7651                                                   |
| **Dev BPB**        | 2.4122                                                   |
| **Runtime**        | 54 seconds                                               |
| **Params**         | 1,298,880 (−40,960 from baseline)                        |
| **Decision**       | ❌ Rejected for standalone use. BPB worsened by +0.0404. |
| **Lessons Learned**| Weight tying reduces parameters. Since parameter budget was underutilized (67%), reducing parameters alone hurts capacity. Must pair with architecture expansion. |

---

## Experiment 04 — Model Capacity Scaling

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Scaling n_embd from 160 to 192 will improve BPB by utilizing unused parameter budget. |
| **What changed**   | `cfg.n_embd = 192`                                       |
| **Training Loss**  | 1.6700                                                   |
| **Dev BPB**        | 2.3472                                                   |
| **Runtime**        | 55 seconds                                               |
| **Params**         | 1,902,720                                                |
| **Decision**       | ✅ Accepted — BPB improved by −0.0246.                   |
| **Lessons Learned**| Proved the active bottleneck was model capacity. Scaling width directly lowered BPB. |

---

## Experiment 05 — Modern Transformer + Byte-Level BPE ✅ FINAL WINNER

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Replacing baseline components with RMSNorm, RoPE, SwiGLU, Weight Tying, and Byte-Level BPE will dramatically improve representational efficiency and lower BPB. |
| **What changed**   | • **Tokenizer**: Byte-Level BPE (Vocab 512, 100% Lossless)<br>• **Norm**: RMSNorm<br>• **Pos Embeddings**: RoPE (Rotary Embeddings)<br>• **MLP**: SwiGLU ($F.silu(W_1 x) \times W_2 x \to W_3$)<br>• **Weight Tying**: Enabled (`head.weight = tok_emb.weight`) |
| **Training Loss**  | **1.6043**                                               |
| **Dev BPB**        | **2.0074**                                               |
| **Runtime**        | 88 seconds                                               |
| **Params**         | **1,869,504** (93.5% of 2M cap ✅)                      |
| **Decision**       | **✅ SELECTED AS FINAL MODEL** — Massive BPB reduction from 2.3718 → 2.0074 (−0.3644, 15.4% improvement). |
| **Lessons Learned**| Subword tokenization (BPE) effectively extends the context window by compressing UTF-8 multi-byte sequences into single tokens. RoPE eliminates position parameter cap limits. SwiGLU provides superior per-parameter expressivity. Weight tying enables allocating parameter budget to layer width ($n_{embd}=192$, $n_{head}=6$, $hidden_{dim}=512$). |

---

## Complete Experiment Summary

| Experiment | Architecture / Tokenizer Changes | Parameters | Dev BPB | Δ vs Base | Status |
|------------|----------------------------------|------------|---------|-----------|--------|
| Baseline   | LayerNorm, GELU, Raw Byte        | 1,339,840  | 2.3718  | —         | Reference |
| Exp 01     | AdamW (wd=0.1)                   | 1,339,840  | 2.3790  | +0.0072   | ❌ Rejected |
| Exp 02     | 100-step Warmup + Cosine LR      | 1,339,840  | 2.6976  | +0.3258   | ❌ Rejected |
| Exp 03     | Weight Tying (tie_weights=True)  | 1,298,880  | 2.4122  | +0.0404   | ❌ Rejected |
| Exp 04     | Capacity Scaling (n_embd=192)    | 1,902,720  | 2.3472  | −0.0246   | ✅ Accepted |
| **Exp 05** | **Modern Transformer + BPE**     | **1,869,504** | **2.0074** | **−0.3644** | **✅ FINAL WINNER** |
