# Experiment 02 - Observations

## Research Question
Can a learning rate schedule (Linear Warmup + Cosine Decay) improve BPB while keeping the optimizer and model unchanged?

## Hypothesis
A 100-step linear warmup followed by cosine decay will stabilize early training and allow fine-grained convergence near the end of training, lowering BPB.

## Measured Results

| Metric                | Baseline (Exp 0) | Experiment 02 (LR Sched) | Delta      |
|-----------------------|------------------|--------------------------|------------|
| **Final Training Loss** | 1.7315         | 1.9637                   | +0.2322    |
| **Dev BPB**           | 2.3718           | 2.6976                   | +0.3258    |
| **Runtime (seconds)** | 54               | 53                       | -1 sec     |
| **Parameter Count**   | 1,339,840        | 1,339,840                | —          |

## Decision
**REJECTED.** The learning-rate schedule significantly degraded performance. BPB and training loss were substantially higher than baseline. The baseline constant learning rate remains superior under this 2000-step training budget.

## Lessons Learned
- For this very small training budget (2000 steps), the model benefits from taking maximum-sized gradient steps (3e-4) constantly, rather than decaying the step size.
- A cosine decay reduces the learning rate to near-zero for the final 20% of training. In a fast 2000-step run, this amounts to wasting compute steps where the model could have been making meaningful parameter updates.
- The 2-hour strategy in `docs/LLM_Optimization_Ideas.md` significantly overestimated the value of scheduling at this specific scale.

## Future Work
- Reject optimizer/scheduler tuning as a priority. The baseline setup (Adam, constant LR) is extremely competitive for this exact step budget.
- Shift focus entirely to architectural improvements (weight tying, capacity scaling, context window) or tokenizer improvements.
