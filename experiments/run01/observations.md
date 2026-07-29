# Experiment 01 - Observations

## Research Question
Does AdamW (weight_decay=0.1) improve BPB compared to the baseline Adam optimizer?

## Hypothesis
Replacing Adam with AdamW and adding weight_decay=0.1 would regularize the model's weights during training, reducing overfitting to the training corpus and improving generalization as measured by dev BPB.

## Observations
- Training with AdamW converged slightly more slowly than the baseline Adam; the step 1 loss was comparable, but mid-training losses trended marginally higher.
- Weight decay imposes an L2-style penalty, which appears to have acted as a net constraint rather than a benefit at this small model scale (1.34M params) and step budget (2,000 steps).
- The model is likely under-regularized by overfitting concerns at this scale — regularization is not the binding constraint here.
- Runtime was 58 seconds vs 54 seconds baseline, a minor overhead consistent with AdamW's additional weight decay computation.

## Measured Results

| Metric                | Baseline (Exp 0) | Experiment 01 (AdamW) | Delta      |
|-----------------------|------------------|-----------------------|------------|
| **Final Training Loss** | 1.7315         | 1.7399                | +0.0084    |
| **Dev BPB**           | 2.3718           | 2.3790                | +0.0072    |
| **Runtime (seconds)** | 54               | 58                    | +4 sec     |
| **Parameter Count**   | 1,339,840        | 1,339,840             | —          |

## Decision
**REJECTED.** AdamW with weight_decay=0.1 performed worse than the baseline across all measured metrics. The hypothesis is not supported by the experimental evidence.

## Lessons Learned
- At 1.34M parameters and only 2,000 training steps on a small corpus, the model is not in the regime where weight regularization provides a measurable benefit.
- AdamW's advantage over Adam is most pronounced in large-scale training with millions of steps, where weight decay prevents parameter drift over time.
- For this assignment's constrained setting, optimizer regularization is not the primary lever for BPB improvement.
- The 2-hour strategy in `docs/LLM_Optimization_Ideas.md` overestimated AdamW's standalone impact (ranked it #2 with Expected Impact 4/5); this should be revised downward.

## Future Work
- Do NOT include AdamW as a standalone change in subsequent experiments.
- Revisit AdamW only as part of a combined change (e.g., with LR scheduling), where its interaction with cosine warmup may produce a net benefit.
- Prioritize high-impact structural changes: LR scheduling, larger block size, weight tying, and parameter capacity expansion.
