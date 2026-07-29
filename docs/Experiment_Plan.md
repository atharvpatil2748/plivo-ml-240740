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

# Experiment 02 — LR Scheduler Ablation

## Research Question
Can a learning rate schedule (Linear Warmup + Cosine Decay) improve BPB while keeping the optimizer and model unchanged?

## Hypothesis
A 100-step linear warmup followed by cosine decay will stabilize early training and allow fine-grained convergence near the end of training, lowering BPB.

## Configuration
- Script: `starter/train_exp02.py`
- Change: Added 100-step linear warmup + cosine decay.
- All other hyperparameters identical to Experiment 0 (Adam, peak lr=3e-4)

## Metrics

| Metric                | Baseline (Exp 0) | Experiment 02 (LR Sched) |
|-----------------------|------------------|--------------------------|
| **Final Training Loss** | 1.7315         | 1.9637                   |
| **Dev BPB**           | 2.3718           | 2.6976                   |
| **Runtime (seconds)** | 54               | 53                       |
| **Parameter Count**   | 1,339,840        | 1,339,840                |

## Decision
**REJECTED** — The learning-rate schedule significantly degraded performance. Training loss was much higher (+0.2322) and BPB was significantly worse (+0.3258). The baseline constant learning rate remains superior under the 2000-step training budget.

## Status: ❌ REJECTED

---

---

# Experiment 03 — Weight Tying

## Research Question
Can weight tying improve parameter efficiency while maintaining or improving BPB?

## Hypothesis
Tying the output head weights to the token embedding weights will reduce the overall parameter count while acting as a regularizer, potentially improving or matching the baseline BPB.

## Configuration
- Script: `starter/train_exp03_weight_tying.py`
- Change: Enabled `cfg.tie_weights = True`.
- All other hyperparameters identical to Experiment 0 (Adam, peak lr=3e-4, 2000 steps).

## Metrics

| Metric                | Baseline (Exp 0) | Experiment 03 (Weight Tying) |
|-----------------------|------------------|------------------------------|
| **Final Training Loss** | 1.7315         | 1.7651                       |
| **Dev BPB**           | 2.3718           | 2.4122                       |
| **Runtime (seconds)** | 55               | 54                           |
| **Parameter Count**   | 1,339,840        | 1,298,880                    |

## Decision
**REJECTED FOR FINAL SUBMISSION** — Weight tying reduced parameter count by approximately 41k parameters but produced slightly worse BPB (+0.0404). While the performance degradation is relatively small, weight tying is not the best choice for minimizing BPB under the assignment objective given that we have ~700k unused parameters.

## Status: ❌ REJECTED

---

# Experiment 04 — Model Capacity Scaling ✅ WINNER

## Research Question
Can increasing model capacity (while remaining under the 2M parameter limit) improve BPB?

## Hypothesis
Scaling `n_embd` from 160 to 192 will utilize ~563k unused parameters and give the model more representational bandwidth, lowering BPB.

## Configuration
- Script: `starter/train_exp04_capacity.py`
- Change: `cfg.n_embd = 192` (increased from 160)
- All other hyperparameters identical to Experiment 0 (Adam, peak lr=3e-4, 2000 steps)

## Metrics

| Metric                | Baseline (Exp 0) | Experiment 04 (Capacity) |
|-----------------------|------------------|--------------------------|
| **Final Training Loss** | 1.7315         | (see training.log)       |
| **Dev BPB**           | 2.3718           | **2.3472**               |
| **Runtime (seconds)** | 54               | (see training.log)       |
| **Parameter Count**   | 1,339,840        | **1,902,720**            |

## Decision
**✅ ACCEPTED — BEST PERFORMING CONFIGURATION.** BPB improved by −0.0246 over the baseline. This is the final submission checkpoint. The change was merged into `starter/model.py` as the new default architecture.

## Status: ✅ WINNER / FINAL SUBMISSION
