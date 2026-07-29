# Experiment 04 - Observations

## Research Question
Can increasing model capacity (while remaining under the 2M parameter limit) improve BPB?

## Hypothesis
Scaling `n_embd` from 160 to 192 will utilize the remaining ~660k unused parameter budget, increasing representational bandwidth at every layer and lowering BPB.

## Measured Results

| Metric                | Baseline (Exp 0) | Experiment 04 (n_embd=192) | Delta      |
|-----------------------|------------------|----------------------------|------------|
| **Final Training Loss** | 1.7315         | (see training.log)         |            |
| **Dev BPB**           | 2.3718           | **2.3472**                 | **−0.0246**|
| **Parameter Count**   | 1,339,840        | 1,902,720                  | +562,880   |

## Decision
**✅ ACCEPTED — FINAL SUBMISSION.** This is the only experiment to improve BPB over the baseline. The model was capacity-constrained throughout. Once that constraint was relieved, performance improved.

## Analysis
- Experiments 01–03 (AdamW, LR scheduling, weight tying) all degraded performance, confirming the model was underfitting, not overfitting.
- Increasing `n_embd` from 160 to 192 scales Q/K/V projections, output projection, MLP hidden states, and embeddings simultaneously.
- The parameter increase of ~563k brought utilization from 67% to 95.1% of the 2M cap, directly in the 1.85M–1.95M target range.
- The BPB improvement of −0.0246 is clean and unambiguous given the identical training setup.

## Lessons Learned
- Diagnose the active constraint before applying optimizations. Here it was always model capacity.
- Ablating one variable at a time confirmed this efficiently and avoided wasted experiments.
- The original baseline analysis in `docs/LLM_Optimization_Ideas.md` had correctly identified B7 (Parameter Capacity Scaling) as a candidate, but placed it lower priority than scheduling changes.
