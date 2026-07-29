# LLM Optimization Ideas — Plivo ML Internship Assignment

> **Objective:** Minimize Bits Per Byte (BPB) on dev_eval.txt  
> **Constraints:** <=2,000 steps | <=2,000,000 params | provided corpus only | pure PyTorch/NumPy/stdlib | CPU-compatible | evaluate.py unchanged | lossless tokenizer

---

## 1. Baseline Weaknesses

### 1.1 Optimizer & Training Dynamics
* **W1. Plain Adam with No Weight Decay:** Baseline uses standard Adam (torch.optim.Adam) without weight decay. Decoupled weight decay (AdamW) is standard for transformers to regularize parameter weights.
* **W2. Fixed Learning Rate (3e-4) without Warmup or Decay:** A constant learning rate causes instability during early steps and prevents fine-grained convergence near step 2,000.
* **W3. Lack of Gradient Clipping:** Without clip_grad_norm_, exploding gradients from rare tokens destabilize training.
* **W4. Suboptimal Adam Betas:** Default beta2=0.999 adapts slowly over 2,000 steps. A lower beta2 (e.g., 0.95) adjusts variance estimates more dynamically.
* **W5. Small Batch Size (8):** A mini-batch size of 8 sequences (1,024 tokens/step at block size 128) results in high variance per gradient update.

### 1.2 Architecture
* **W6. Absolute Learned Position Embeddings:** Learned position embeddings do not generalize beyond block_size=128 and consume unnecessary parameter budget.
* **W7. Standard LayerNorm vs. RMSNorm:** LayerNorm computes mean and variance. RMSNorm only computes root mean square, reducing compute overhead.
* **W8. Standard GELU MLP vs. Gated Activation (SwiGLU/GeGLU):** Modern architectures use gated activations (e.g. SwiGLU) achieving lower perplexity per parameter.
* **W9. Untied Input/Output Embeddings (tie_weights=False):** Untied weight matrices for token embeddings and head waste parameter capacity.
* **W10. Naive Initialization:** Baseline uses uniform std=0.05 without scaling deep residual projections (e.g. GPT-2 style 1/sqrt(2*n_layer)), causing variance explosion.
* **W11. Small Context Window (block_size=128):** evaluate.py uses a sliding window with stride block_size // 2. Context window of 128 limits effective left-context during evaluation to 64 tokens.
* **W12. Underutilized Parameter Budget:** Baseline uses ~1.34M of 2.0M allowed parameters (~67% budget utilization), missing out on ~660K extra parameters for capacity.

### 1.3 Tokenizer
* **W13. Raw Byte Tokenizer (Vocab 256):** Raw UTF-8 bytes treat non-ASCII sequences inefficiently. For Hindi/Devanagari, 1 character = 3 byte tokens, expanding sequence length and inflating BPB.
* **W14. Lack of Subword Structure:** A 256-token vocabulary forces the model to learn subword and word construction strictly from raw bytes.

### 1.4 Data & Evaluation Alignment
* **W15. Pure Random Sampling:** Uniform random cropping across training text leads to uneven corpus coverage over 2,000 steps and breaks document-level continuity.
* **W16. Eval Set Multilingual Shift:** Dev eval set contains ~43.7% non-ASCII bytes compared to ~33.1% in training corpus. Tokenizer efficiency on non-ASCII text directly determines final BPB performance.

---
## 2. Optimization Ideas

### Category A: Optimizer & Training Dynamics
1. **A1. AdamW Optimizer:** Replace Adam with AdamW(weight_decay=0.1).
2. **A2. Cosine Annealing with Warmup:** 100-step linear warmup + cosine decay down to 10% peak LR.
3. **A3. Gradient Clipping:** Clip global gradient norm to 1.0 before optimizer step.
4. **A4. Adam Beta Tuning:** Set betas=(0.9, 0.95) for faster variance adaptation.
5. **A5. Increased Batch Size / Gradient Accumulation:** Increase effective batch size to 32 or 64.
6. **A6. Exponential Moving Average (EMA):** Maintain EMA of model weights (decay=0.995) for evaluation checkpoint.
7. **A7. Label Smoothing:** Apply cross-entropy label smoothing (eps=0.1) to calibrate probabilities.

### Category B: Architecture & Parameter Budget
8. **B1. Weight Tying:** Tie tok_emb.weight with head.weight.
9. **B2. SwiGLU MLP:** Replace Linear -> GELU -> Linear with SwiGLU (SiLU-gated linear unit).
10. **B3. RMSNorm:** Replace nn.LayerNorm with root-mean-square normalization.
11. **B4. Rotary Position Embeddings (RoPE) / ALiBi:** Remove learned position embeddings in favor of relative positional biases.
12. **B5. Scaled GPT-2 Initialization:** Initialize residual projections (attn.proj, mlp.fc2) with std = 0.02 / sqrt(2 * n_layer).
13. **B6. Increased Block Size:** Increase block_size from 128 to 256.
14. **B7. Full Parameter Utilization:** Expand model depth (n_layer=6) or width (n_embd=192/256) to reach ~1.95M parameters.
15. **B8. Output Head Bias:** Add learned or empirical unigram frequency bias to head.

### Category C: Tokenizer & Representation
16. **C1. Byte-Level BPE (Vocab 2,048):** Train a Byte-Level Byte-Pair Encoding (BPE) tokenizer on train_corpus.txt.
17. **C2. Byte-Level BPE (Vocab 4,096 / 8,192):** Scale BPE vocabulary size with weight tying to maximize character compression per sequence.
18. **C3. Lossless Byte-Fallback Verification:** Ensure all 256 base bytes are preserved in vocabulary so decode(encode(text)) == text is strictly lossless.

### Category D: Sampling & Data Processing
19. **C4. Sequential / Non-Overlapping Window Packing:** Pack training corpus sequentially to ensure uniform step coverage and preserve document structure.

---
## 3. Per-Idea Engineering Analysis

| ID | Name | Difficulty (1-5) | Expected Impact (1-5) | Est. Impl Time | Risk Level | Why It Helps | Drawbacks / Notes |
|---|---|---|---|---|---|---|---|
| A1 | AdamW Optimizer | 1 | 4 | 5 min | Low | Decouples weight decay from gradient scaling, improving transformer regularization | Requires tuning weight decay parameter |
| A2 | Cosine Warmup LR | 1 | 5 | 10 min | Low | Prevents early gradient shock and enables fine convergence near step 2000 | Needs tuning peak LR and warmup steps |
| A3 | Gradient Clipping | 1 | 3 | 5 min | Low | Prevents exploding gradient spikes during training | Minimal drawback; clip norm needs to be reasonable (1.0) |
| A4 | Adam Beta Tuning | 1 | 2 | 5 min | Low | Accelerates variance adaptation for short 2,000 step runs | Can slightly overshoot near convergence if beta2 is too low |
| A5 | Larger Batch Size | 1 | 3 | 5 min | Low | Reduces variance in gradient estimates per step | Increases per-step compute time |
| A6 | EMA Weights | 2 | 3 | 20 min | Low | Produces smoother final weights for validation evaluation | Requires extra memory for shadow weights |
| A7 | Label Smoothing | 1 | 2 | 5 min | Low | Prevents overconfidence, improving cross-entropy / BPB calibration | Over-smoothing can degrade top-1 probability prediction |
| B1 | Weight Tying | 1 | 3 | 5 min | Low | Reduces parameter count, allowing parameter budget reallocation to depth/width | Requires matching embedding dimension and head dimension |
| B2 | SwiGLU MLP | 2 | 4 | 30 min | Low | Gated activations provide higher expressivity per parameter than standard GELU | Requires 3 linear layers per MLP block instead of 2 |
| B3 | RMSNorm | 2 | 2 | 15 min | Low | Faster compute and better numerical scaling than standard LayerNorm | Minor code addition (custom PyTorch module) |
| B4 | RoPE / Relative Pos | 3 | 4 | 45 min | Medium | Improves relative position handling and eliminates learned position embedding params | More complex attention forward implementation |
| B5 | Scaled Init | 1 | 3 | 10 min | Low | Controls variance growth along deep residual streams | Must be applied selectively to projection weights |
| B6 | Increased Block Size | 1 | 4 | 10 min | Low | Direct BPB reduction in evaluate.py by providing larger left-context | Increases quadratic attention compute cost |
| B7 | Full Param Scaling | 1 | 3 | 15 min | Low | Utilizes the remaining ~660K parameter capacity up to 2M limit | Must carefully monitor exact param count under 2M cap |
| B8 | Head Bias Init | 2 | 3 | 15 min | Low | Gives baseline unigram frequency logit boost at step 0 | Minor impact past early steps |
| C1 | BPE Tokenizer (2K) | 3 | 5 | 45 min | Medium | Dramatically compresses Hindi/multilingual bytes into subword tokens | Must guarantee 100% lossless UTF-8 fallback |
| C2 | BPE Tokenizer (4K-8K)| 3 | 5 | 60 min | Medium | Higher compression ratio -> longer effective text context window | Larger vocab requires weight tying to stay under 2M params |
| C4 | Sequential Sampling| 2 | 3 | 20 min | Low | Guarantees complete corpus coverage over 2,000 steps | Reduces stochastic batch diversity |

---
## 4. Priority Ranking

*Ranked by Expected BPB Improvement per Hour of Engineering Effort:*

| Rank | ID | Name | Expected Impact (1-5) | Est Time (hrs) | ROI Score | Risk |
|---|---|---|---|---|---|---|
| 1 | A2 | Cosine Warmup LR | 5 | 0.17 | 29.4 | Low |
| 2 | A1 | AdamW Optimizer | 4 | 0.08 | 25.0 | Low |
| 3 | A3 | Gradient Clipping | 3 | 0.08 | 18.8 | Low |
| 4 | B5 | Scaled GPT-2 Init | 3 | 0.17 | 17.6 | Low |
| 5 | B6 | Increased Block Size (256) | 4 | 0.17 | 23.5 | Low |
| 6 | B1 | Weight Tying | 3 | 0.08 | 37.5 | Low |
| 7 | B7 | Parameter Capacity Scaling | 4 | 0.25 | 16.0 | Low |
| 8 | B2 | SwiGLU MLP | 4 | 0.50 | 8.0 | Low |
| 9 | C1 | BPE Tokenizer (Vocab 2048) | 5 | 0.75 | 6.7 | Medium |
| 10| C2 | BPE Tokenizer (Vocab 4096) | 5 | 1.00 | 5.0 | Medium |
| 11| A5 | Larger Batch Size (32) | 3 | 0.08 | 37.5 | Low |
| 12| B3 | RMSNorm | 2 | 0.25 | 8.0 | Low |
| 13| A6 | EMA Weights | 3 | 0.33 | 9.0 | Low |
| 14| B4 | RoPE Attention | 4 | 0.75 | 5.3 | Medium |
| 15| C4 | Sequential Sampling | 3 | 0.33 | 9.0 | Low |

---

## 5. Suggested Experiment Order

`
Run 0: Baseline Checkpoint
  └─ Establish benchmark score on dev_eval.txt using original setup.

Run 1: Optimizer & Scheduler Setup
  └─ Add AdamW (lr=3e-4, weight_decay=0.1) + Gradient Clipping (1.0) + Cosine Warmup.

Run 2: Weight Tying & Parameter Scaling
  └─ Enable tie_weights=True, expand n_embd and n_layer up to ~1.95M parameters.

Run 3: Initialization & Block Size Extension
  └─ Apply GPT-2 scaled init + expand block_size to 256.

Run 4: SwiGLU & RMSNorm Architecture Upgrade
  └─ Replace GELU MLP with SwiGLU + replace LayerNorm with RMSNorm.

Run 5: BPE Tokenizer (Vocab 2048)
  └─ Switch from ByteTokenizer to trained byte-level BPE (vocab 2048). Verify lossless check.

Run 6: Full Optimization Integration (Final Checkpoint)
  └─ Combine best hyperparams + BPE tokenizer + EMA weights.
`

---

## 6. Final Recommendation (2-Hour Strategy)

If limited to **2 hours**, execute the following high-ROI sequence:

1. **Minutes 0-15 (Optimizer + LR Schedule + Init + Clipping):**
   * Change Adam to AdamW(weight_decay=0.1), add gradient clipping (1.0), 100-step linear warmup with cosine decay, and GPT-2 scaled init.
   * *Why:* Instant reduction in BPB with zero risk of breaking evaluation contracts or parameter limits.

2. **Minutes 15-35 (Weight Tying + Architecture Expansion + Block Size):**
   * Enable tie_weights=True, increase block_size=256, scale n_layer=6 and n_embd=192/256 to fill parameter budget to ~1.95M.
   * *Why:* Leverages full parameter budget and expands evaluation left-context window for direct BPB gain.

3. **Minutes 35-90 (Byte-Level BPE Tokenizer - Vocab 2048):**
   * Implement a byte-level BPE tokenizer trained on train_corpus.txt. Ensure Base-256 byte fallback for lossless roundtrip.
   * *Why:* Single biggest BPB improvement factor on multilingual content (reduces Devanagari 3-byte expansion).

4. **Minutes 90-120 (Final Run & Evaluation):**
   * Run 2,000 steps with all combined improvements, generate ckpt.pt, run evaluate.py, and log results.
