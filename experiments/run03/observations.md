# Experiment 03 - Observations

## Research Question
Can weight tying improve parameter efficiency while maintaining or improving BPB?

## Hypothesis
Tying the output head weights to the token embedding weights will reduce the overall parameter count while acting as a regularizer, potentially improving or matching the baseline BPB.

## Measured Results

| Metric                | Baseline (Exp 0) | Experiment 03 (Weight Tying) | Delta      |
|-----------------------|------------------|------------------------------|------------|
| **Final Training Loss** | 1.7315         | 1.7651                       | +0.0336    |
| **Dev BPB**           | 2.3718           | 2.4122                       | +0.0404    |
| **Runtime (seconds)** | 55               | 54                           | -1 sec     |
| **Parameter Count**   | 1,339,840        | 1,298,880                    | -40,960    |

## Decision
**REJECTED FOR FINAL SUBMISSION.** Weight tying reduced parameter count by approximately 41k parameters but produced slightly worse BPB. The performance degradation is relatively small.

## Analysis & Lessons Learned
- Constant-LR Adam baseline remains the strongest performer.
- Weight tying sacrifices a small amount of accuracy in exchange for fewer parameters.
- The assignment still has approximately 700k unused parameters (cap is 2,000,000). Weight tying is unnecessary because parameter count is not the active bottleneck right now.
- Future experiments should focus on increasing representational capacity (e.g., more layers, larger embedding dim, bigger context window) rather than additional optimizer/parameter-efficiency tricks.

## Conclusion
Weight tying remains an interesting parameter-efficiency technique but is not the best choice for minimizing BPB under the assignment objective where the budget is not fully utilized.
