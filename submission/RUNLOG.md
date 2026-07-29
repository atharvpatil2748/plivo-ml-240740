# RUNLOG — Plivo ML Assignment

Chronological log of all experiments. All measured values from `evaluate.py`. No fabricated results.

---

## Baseline — Experiment 0

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Establish a reference BPB score using unmodified starter code. |
| **What changed**   | Nothing — pure baseline.                                 |
| **Training Loss**  | 1.7315                                                   |
| **Dev BPB**        | 2.3718                                                   |
| **Runtime**        | ~54 seconds                                              |
| **Params**         | 1,339,840                                                |
| **Decision**       | ✅ Accepted as reference point.                           |
| **Lessons Learned**| The baseline uses only 67% of the 2M parameter budget. Model is clearly capacity-constrained. Constant Adam at lr=3e-4 provides a strong baseline for 2,000 steps. |

---

## Experiment 01 — Optimizer Ablation

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | AdamW with weight_decay=0.1 will outperform Adam in BPB. |
| **What changed**   | `torch.optim.Adam` → `torch.optim.AdamW(weight_decay=0.1)` |
| **Training Loss**  | 1.7399                                                   |
| **Dev BPB**        | 2.3790                                                   |
| **Runtime**        | ~58 seconds                                              |
| **Params**         | 1,339,840                                                |
| **Decision**       | ❌ Rejected. BPB worsened by +0.0072.                    |
| **Lessons Learned**| Weight decay is regularization. At this model scale and step budget the model is underfitting, not overfitting. Regularization makes things worse. |

---

## Experiment 02 — LR Scheduler Ablation

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | 100-step linear warmup + cosine decay will lower BPB.   |
| **What changed**   | Added `get_lr(step)` function: linear warmup for 100 steps, then cosine decay to 0 over remaining 1900 steps. |
| **Training Loss**  | 1.9637                                                   |
| **Dev BPB**        | 2.6976                                                   |
| **Runtime**        | ~53 seconds                                              |
| **Params**         | 1,339,840                                                |
| **Decision**       | ❌ Rejected. BPB significantly worsened by +0.3258.      |
| **Lessons Learned**| Cosine decay reduces LR to near zero by step 2000. At only 2000 total steps, this wastes effective training time. A constant LR of 3e-4 is optimal for this short budget. |

---

## Experiment 03 — Weight Tying

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Tying head weights to embedding weights will improve parameter efficiency without degrading BPB. |
| **What changed**   | `cfg.tie_weights = True`                                 |
| **Training Loss**  | 1.7651                                                   |
| **Dev BPB**        | 2.4122                                                   |
| **Runtime**        | ~54 seconds                                              |
| **Params**         | 1,298,880 (−40,960 from baseline)                        |
| **Decision**       | ❌ Rejected for final submission. BPB worsened by +0.0404. |
| **Lessons Learned**| Weight tying reduces capacity. When the parameter budget is not fully utilized (67%), reducing parameters further is counterproductive. Future experiments should grow the model, not shrink it. |

---

## Experiment 04 — Model Capacity Scaling ✅ WINNER

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **Hypothesis**     | Scaling n_embd from 160 to 192 will improve BPB by utilizing unused parameter budget. |
| **What changed**   | `cfg.n_embd = 192` (increased from 160)                  |
| **Training Loss**  | (see training.log)                                       |
| **Dev BPB**        | **2.3472**                                               |
| **Runtime**        | (see training.log)                                       |
| **Params**         | **1,902,720** (within 2M cap, target 1.85M–1.95M ✅)    |
| **Decision**       | ✅ ACCEPTED — Best BPB achieved. Improvement of −0.0246 over baseline. |
| **Lessons Learned**| The binding constraint throughout was model capacity. Experiments 01–03 revealed that regularization and efficiency changes fail when the model is underfitting. Allocating the remaining parameter budget to representational width produced the only successful result. |

---

## Summary Table

| Experiment | Change              | BPB    | Δ BPB    | Result        |
|------------|---------------------|--------|----------|---------------|
| Baseline   | None                | 2.3718 | —        | Reference     |
| Exp 01     | AdamW wd=0.1        | 2.3790 | +0.0072  | ❌ Rejected   |
| Exp 02     | Warmup + Cosine LR  | 2.6976 | +0.3258  | ❌ Rejected   |
| Exp 03     | Weight Tying        | 2.4122 | +0.0404  | ❌ Rejected   |
| **Exp 04** | **n_embd 160→192**  | **2.3472** | **−0.0246** | **✅ Winner** |
