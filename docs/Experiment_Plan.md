# Experiment Plan

---

# Experiment 0 — Baseline

## Objective
Establish the baseline performance of the starter code without any modifications to act as a reference point for future optimizations.

## Configuration
- Codebase: Unmodified starter project
- Training Steps: 2,000
- Maximum Parameters: 2,000,000 limit (Model configured by default to 1,339,840 params)
- Hardware: CPU Only

## Metrics
- **Total Parameters:** 1,339,840
- **Training Time:** 54 seconds (approx. 27 ms/step)
- **Final Training Loss:** 1.7315
- **Dev Evaluation BPB:** 2.3718

## Observations
- **Runtime Environment:** Python 3.12.10, PyTorch 2.13.0+cpu, NumPy 2.5.1
- **CPU Information:** 13th Gen Intel(R) Core(TM) i7-13700HX
- The model trained rapidly but achieved a relatively poor BPB (2.3718), consistent with the fact that it is intentionally mediocre and lacks learning rate scheduling, proper initialization scale, weight decay, or advanced attention mechanisms.

## Status: ✅ COMPLETE

---

# Experiment 01 — Optimizer Ablation

## Research Question
Does AdamW (weight_decay=0.1) improve BPB compared to the baseline Adam optimizer?

## Hypothesis
Replacing Adam with AdamW and adding weight decay (0.1) will regularize the model and reduce BPB relative to the baseline.

## Configuration
- Script: `starter/train_exp01.py`
- Change: `torch.optim.Adam` → `torch.optim.AdamW(weight_decay=0.1)`
- All other hyperparameters identical to Experiment 0

## Metrics

| Metric                | Baseline (Exp 0) | Experiment 01 (AdamW) |
|-----------------------|------------------|-----------------------|
| **Final Training Loss** | 1.7315         | 1.7399                |
| **Dev BPB**           | 2.3718           | 2.3790                |
| **Runtime (seconds)** | 54               | 58                    |
| **Parameter Count**   | 1,339,840        | 1,339,840             |

## Decision
**REJECTED** — AdamW with weight_decay=0.1 produced higher BPB (+0.0072) and higher training loss than the baseline. The hypothesis is not supported. Do NOT merge this change into the main training configuration.

## Status: ❌ REJECTED

---

# Experiment 02 — (Planned)

## Status: 🔲 NOT STARTED
