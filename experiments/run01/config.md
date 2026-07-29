# Experiment 01 - Optimizer Ablation Config

## Experiment Name
Optimizer Ablation

## Research Question
Does AdamW improve BPB compared to the baseline Adam optimizer?

## Changes from Baseline
| Setting       | Baseline          | Experiment 01              |
|---------------|-------------------|----------------------------|
| Optimizer     | Adam              | AdamW                      |
| Weight Decay  | (none)            | 0.1                        |
| Learning Rate | 3e-4              | 3e-4 (unchanged)           |
| Steps         | 2000              | 2000 (unchanged)           |
| Batch Size    | 8                 | 8 (unchanged)              |
| Seed          | 1337              | 1337 (unchanged)           |
| Architecture  | GPT (unchanged)   | GPT (unchanged)            |
| Tokenizer     | ByteTokenizer     | ByteTokenizer (unchanged)  |

Everything else is byte-for-byte identical to the baseline.

## Environment
- **Date**           : 2026-07-30 00:01:07
- **Python version** : 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **Torch version**  : 2.13.0+cpu
- **Git commit**     : 9ff8be731b5e361ac09438f9846df295ac7e0ec8
- **Branch**         : main
- **Train command**  : `python starter\train_exp01.py --data data\train_corpus.txt --steps 2000 --batch 8 --lr 3e-4 --seed 1337 --out experiments\run01\ckpt.pt`
- **Eval command**   : `python starter\evaluate.py --checkpoint experiments\run01\ckpt.pt --text_file data\dev_eval.txt`
