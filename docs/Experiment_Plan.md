# Experiment Plan

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
