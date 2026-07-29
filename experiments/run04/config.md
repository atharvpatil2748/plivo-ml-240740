# Experiment 04 - Model Capacity Ablation Config

## Experiment Name
Model Capacity Ablation

## Research Question
Can increasing model capacity (while remaining under the 2M parameter limit) improve BPB?

## Changes from Baseline
| Setting       | Baseline          | Experiment 04              |
|---------------|-------------------|----------------------------|
| **n_embd**    | 160               | **192**                    |
| **Params**    | 1,339,840         | **1,902,720**              |
| n_layer       | 4                 | 4 (unchanged)              |
| n_head        | 4                 | 4 (unchanged)              |
| Optimizer     | Adam              | Adam (unchanged)           |
| Learning Rate | 3e-4              | 3e-4 (unchanged)           |
| Scheduler     | None              | None (unchanged)           |
| Weight Decay  | None              | None (unchanged)           |
| Steps         | 2000              | 2000 (unchanged)           |
| Batch Size    | 8                 | 8 (unchanged)              |

Everything else is byte-for-byte identical to the baseline.

## Expected Impact
By scaling 
_embd from 160 to 192, the model increases its representational bandwidth throughout all token embeddings, attention representations, and feed-forward hidden states. This directly uses the available budget (1.9M parameters, well within the 1.85M - 1.95M target) without altering the fundamental training dynamics. This should lower BPB.

## Environment
- **Date**           : 2026-07-30 00:26:50
- **Python version** : 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **Torch version**  : 2.13.0+cpu
- **Git commit**     : 4d517db22aff3a5a686712ef4b4a8391824c63c9
- **Branch**         : main
- **Train command**  : `python starter\train_exp04_capacity.py --data data\train_corpus.txt --steps 2000 --batch 8 --lr 3e-4 --seed 1337 --out experiments\run04\ckpt.pt`
- **Eval command**   : `python starter\evaluate.py --checkpoint experiments\run04\ckpt.pt --text_file data\dev_eval.txt`
