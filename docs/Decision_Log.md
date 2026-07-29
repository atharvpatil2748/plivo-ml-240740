# Decision Log — Plivo ML Assignment

Chronological record of all architectural and experimental decisions.

---

## Entry 00

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Baseline (Experiment 0)                                             |
| **Objective**  | Establish reference benchmark using baseline starter code.           |
| **Outcome**    | ✅ Reference established                                            |
| **Result**     | Dev BPB: 2.3718 \| Params: 1,339,840 \| Steps: 2,000                |
| **Date**       | 2026-07-29                                                           |

---

## Entry 01

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 01                                                       |
| **Objective**  | Optimizer Ablation — Adam vs AdamW                                  |
| **Hypothesis** | AdamW with weight_decay=0.1 will outperform Adam in BPB.            |
| **Outcome**    | ❌ REJECTED                                                          |
| **Reason**     | BPB increased from 2.3718 → 2.3790 (+0.0072). Weight decay regularizes a model that is currently underfitting. |
| **Date**       | 2026-07-30                                                           |

---

## Entry 02

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 02                                                       |
| **Objective**  | Learning Rate Scheduler Ablation                                    |
| **Hypothesis** | Linear Warmup + Cosine Decay will improve convergence and lower BPB.|
| **Outcome**    | ❌ REJECTED                                                          |
| **Reason**     | Warmup + Cosine significantly increased BPB (2.3718 → 2.6976). Cosine decay reduces step size near end of training, wasting steps. |
| **Date**       | 2026-07-30                                                           |

---

## Entry 03

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 03                                                       |
| **Objective**  | Weight Tying Ablation                                               |
| **Hypothesis** | Weight tying will improve parameter efficiency without degrading BPB. |
| **Outcome**    | ❌ REJECTED FOR STANDALONE USE                                      |
| **Reason**     | Reduced parameter count by ~41k but produced slightly worse BPB (2.4122 vs 2.3718 baseline). |
| **Date**       | 2026-07-30                                                           |

---

## Entry 04

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 04                                                       |
| **Objective**  | Model Capacity Scaling                                              |
| **Hypothesis** | Scaling n_embd from 160 to 192 will improve BPB by using remaining parameter budget. |
| **Outcome**    | ✅ ACCEPTED (Superseded by Exp 05)                                  |
| **Result**     | BPB improved from 2.3718 → 2.3472 (−0.0246). Params: 1,902,720.     |
| **Date**       | 2026-07-30                                                           |

---

## Entry 05 ✅ FINAL SUBMISSION MODEL

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 05 — Modern Transformer + Byte-Level BPE                 |
| **Objective**  | Full architectural & tokenizer modernization                       |
| **Hypothesis** | RMSNorm + RoPE + SwiGLU + Weight Tying + Byte-Level BPE will dramatically lower BPB. |
| **Outcome**    | **🏆 FINAL WINNER / SELECTED FOR SUBMISSION**                       |
| **Result**     | **Dev BPB: 2.0074 (−0.3644, −15.4%) \| Params: 1,869,504 (93.5% cap)**|
| **Reason**     | BPE tokenization compresses multi-byte sequences into single subwords, extending the context window. RoPE eliminates position params. SwiGLU adds expressivity. Weight tying links head to tok_emb, allowing wider layers. |
| **Action**     | Set as primary repository model and copied to submission package.   |
| **Date**       | 2026-07-30                                                           |

---
