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

(Further entries to be added as experiments complete.)
