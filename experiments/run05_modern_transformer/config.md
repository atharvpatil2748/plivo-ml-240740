# Experiment 05 - Modern Transformer Architecture Config

## Experiment Name
Modern Transformer Architecture + Byte-Level BPE

## Research Question
Can a modern transformer architecture (RMSNorm, RoPE, SwiGLU, Weight Tying) combined with a Byte-Level BPE tokenizer significantly outperform the baseline?

## Changes from Baseline
| Setting          | Baseline          | Experiment 05                          |
|------------------|-------------------|----------------------------------------|
| **Tokenizer**    | ByteTokenizer     | **ByteLevelBPETokenizer (Vocab 512)**  |
| **Norm**         | LayerNorm         | **RMSNorm**                            |
| **Pos Embedding**| Learned Absolute  | **RoPE (Rotary Position Embedding)**   |
| **MLP**          | GELU              | **SwiGLU (SiLU-gated linear unit)**    |
| **Weight Tying** | False             | **True** (head = tok_emb)              |
| **n_head**       | 4                 | **6**                                  |
| **n_embd**       | 160               | **192**                                |
| **hidden_dim**   | 640               | **512** (SwiGLU bottleneck)            |
| **Params**       | 1,339,840         | **1,869,504** (93.5% of 2M cap)        |
| Optimizer        | Adam              | Adam (unchanged)                       |
| Learning Rate    | 3e-4              | 3e-4 (unchanged)                       |
| Steps            | 2000              | 2000 (unchanged)                       |
| Batch Size       | 8                 | 8 (unchanged)                          |
| Seed             | 1337              | 1337 (unchanged)                       |

## Expected Advantages
1. **RoPE**: Zero-parameter relative positional encoding eliminates position cap limitations.
2. **Byte-Level BPE**: Subword compression increases effective context window on Hindi/multilingual text.
3. **SwiGLU & RMSNorm**: Higher representational expressivity and faster numerical scaling.
4. **Weight Tying**: Eliminates redundant output projection parameters.

## Environment
- **Date**           : 2026-07-30 00:51:30
- **Python version** : 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **Torch version**  : 2.13.0+cpu
- **Git commit**     : 207d40f1b6081a9c3aac06eca37af1ddc4b24696
- **Branch**         : main
- **Train command**  : `python starter\train_exp05_modern_transformer.py --data data\train_corpus.txt --steps 2000 --batch 8 --lr 3e-4 --seed 1337 --out experiments\run05_modern_transformer\ckpt.pt`
- **Eval command**   : `python starter\evaluate.py --checkpoint experiments\run05_modern_transformer\ckpt.pt --text_file data\dev_eval.txt`
