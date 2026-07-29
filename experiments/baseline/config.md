# Experiment 0 - Baseline Config

- **Date**           : 2026-07-29 23:51:16
- **Python version** : 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **Torch version**  : 2.13.0+cpu
- **Git commit**     : 345eb3aeb82f5a944a8cb86db47b7ef51fb1f650
- **Branch**         : main
- **Train command**  : `python starter\train.py --data data\train_corpus.txt --steps 2000 --batch 8 --lr 3e-4 --seed 1337 --out experiments\baseline\ckpt.pt`
- **Eval command**   : `python starter\evaluate.py --checkpoint experiments\baseline\ckpt.pt --text_file data\dev_eval.txt`
