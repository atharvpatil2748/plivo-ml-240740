# Experiment 02 - Learning Rate Scheduler Ablation Config

## Experiment Name
Learning Rate Scheduler Ablation

## Research Question
Can a warmup + cosine learning rate schedule improve BPB?

## Changes from Baseline
| Setting       | Baseline          | Experiment 02              |
|---------------|-------------------|----------------------------|
| Optimizer     | Adam              | Adam (unchanged)           |
| Peak LR       | 3e-4              | 3e-4 (unchanged)           |
| Warmup        | None              | 100 steps                  |
| Scheduler     | None              | Cosine Decay               |
| Weight Decay  | None              | None (unchanged)           |
| Steps         | 2000              | 2000 (unchanged)           |
| Batch Size    | 8                 | 8 (unchanged)              |
| Seed          | 1337              | 1337 (unchanged)           |
| Architecture  | GPT (unchanged)   | GPT (unchanged)            |
| Tokenizer     | ByteTokenizer     | ByteTokenizer (unchanged)  |

Everything else is byte-for-byte identical to the baseline.

## Environment
- **Date**           : 2026-07-30 00:12:39
- **Python version** : 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **Torch version**  : 2.13.0+cpu
- **Git commit**     : 3410efeeded8f304497624247645c95f4be11139
- **Branch**         : main
- **Train command**  : `python starter\train_exp02.py --data data\train_corpus.txt --steps 2000 --batch 8 --lr 3e-4 --seed 1337 --out experiments\run02\ckpt.pt`
- **Eval command**   : `python starter\evaluate.py --checkpoint experiments\run02\ckpt.pt --text_file data\dev_eval.txt`
