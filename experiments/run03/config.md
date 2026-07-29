# Experiment 03 - Weight Tying Config

## Experiment Name
Weight Tying Ablation

## Research Question
Can weight tying improve parameter efficiency and reduce BPB without changing model capacity?

## Changes from Baseline
| Setting       | Baseline          | Experiment 03              |
|---------------|-------------------|----------------------------|
| tie_weights   | False             | True                       |
| Optimizer     | Adam              | Adam (unchanged)           |
| Learning Rate | 3e-4              | 3e-4 (unchanged)           |
| Scheduler     | None              | None (unchanged)           |
| Steps         | 2000              | 2000 (unchanged)           |
| Batch Size    | 8                 | 8 (unchanged)              |
| Seed          | 1337              | 1337 (unchanged)           |
| Architecture  | GPT               | GPT (unchanged)            |
| Tokenizer     | ByteTokenizer     | ByteTokenizer (unchanged)  |

Everything else is byte-for-byte identical to the baseline.

## Environment
- **Date**           : 2026-07-30 00:19:45
- **Python version** : 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **Torch version**  : 2.13.0+cpu
- **Git commit**     : 41380cd1138ee67c6f697c634bad5f865cd40c04
- **Branch**         : main
- **Train command**  : `python starter\train_exp03_weight_tying.py --data data\train_corpus.txt --steps 2000 --batch 8 --lr 3e-4 --seed 1337 --out experiments\run03\ckpt.pt`
- **Eval command**   : `python starter\evaluate.py --checkpoint experiments\run03\ckpt.pt --text_file data\dev_eval.txt`
