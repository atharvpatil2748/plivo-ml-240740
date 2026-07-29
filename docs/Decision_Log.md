# Decision Log

---

## Entry 01

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 01                                                       |
| **Objective**  | Optimizer Ablation — Adam vs AdamW                                  |
| **Hypothesis** | AdamW with weight_decay=0.1 will outperform Adam in BPB             |
| **Outcome**    | ❌ REJECTED                                                          |
| **Reason**     | BPB increased from 2.3718 → 2.3790 (+0.0072); training loss also higher (1.7315 → 1.7399). The optimizer change did not improve the objective. |
| **Action**     | Do NOT merge AdamW into main training config. Preserve experiment for reference. Proceed to Experiment 02. |
| **Date**       | 2026-07-30                                                           |

---

## Entry 02

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 02                                                       |
| **Objective**  | Learning Rate Scheduler Ablation                                    |
| **Hypothesis** | Linear Warmup + Cosine Decay will improve convergence and lower BPB.|
| **Outcome**    | ❌ REJECTED                                                          |
| **Reason**     | Warmup + Cosine significantly increased BPB (2.3718 → 2.6976).      |
| **Lessons Learned** | For this very small training budget (2000 steps), a constant learning rate appears better than this scheduler. |
| **Action**     | Move focus away from optimizer/scheduler tuning and investigate model representational improvements. Proceed to Experiment 03. |
| **Date**       | 2026-07-30                                                           |

---

## Entry 03

| Field          | Value                                                               |
|----------------|---------------------------------------------------------------------|
| **Experiment** | Experiment 03                                                       |
| **Objective**  | Weight Tying Ablation                                               |
| **Hypothesis** | Weight tying will improve parameter efficiency without degrading BPB. |
| **Outcome**    | ❌ REJECTED FOR FINAL SUBMISSION                                     |
| **Reason**     | Reduced parameter count by ~41k but produced slightly worse BPB (2.4122 vs 2.3718 baseline). |
| **Lessons Learned** | - Constant-LR Adam baseline remains the strongest performer. <br> - Weight tying sacrifices a small amount of accuracy in exchange for fewer parameters. <br> - The assignment still has approximately 700k unused parameters. <br> - Future experiments should focus on increasing representational capacity rather than additional optimizer tricks. |
| **Action**     | Do not include weight tying as a standalone optimization. Prepare for Experiment 04. |
| **Date**       | 2026-07-30                                                           |

---

(Further entries to be added as experiments complete.)
