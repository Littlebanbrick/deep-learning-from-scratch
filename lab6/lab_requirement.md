# Lab 6: Attention and the Transformer from Scratch

## 1. Objective

In Lab 5 you built RNNs and LSTMs, and you saw that even a gated cell pays a price for recurrence: training is sequential over time, and long-range memory requires careful engineering. In 2017, Vaswani et al. proposed to **drop recurrence entirely** and base sequence modeling on **attention** alone. The resulting architecture — the Transformer — is the foundation of every modern language model and increasingly of vision models (Lab 7).

This lab is the keystone of the advanced track: it is where "recurrence" gives way to "attention," and where the machinery you build will recur in every later topic. By completing this lab, you should be able to:

- Implement scaled dot-product attention from scratch and verify it against `F.scaled_dot_product_attention`.
- Derive why the dot-product must be **scaled** by `1/√d_k`, and what breaks if you do not.
- Build multi-head attention by hand and verify against `nn.MultiheadAttention`.
- Assemble a minimal Transformer (encoder and/or encoder-decoder) block by block: attention + residual + LayerNorm + feed-forward.
- Train a tiny character-level language model and feel, concretely, how attention solves the long-range problem that RNNs struggled with in Lab 5.
- Explain positional encoding: why attention alone is permutation-invariant, and how positions are injected.

The spirit is, again, Lab 2's: implement each piece by hand, then verify against the library implementation to prove you understood the contract correctly.

## 2. Prerequisites

You are expected to have completed:

- Lab 5: RNN/LSTM, BPTT, and the vanishing-gradient problem. This lab is the direct response to "recurrence is hard to train on long sequences."
- `minimal-rnn/sine_rnn.py`: comfort with a many-to-one / sequence setup.
- Andrew Ng's attention and Transformer lectures (attention motivation, the Q/K/V framing, multi-head attention, positional encoding).

You should be comfortable with: matrix multiplication batching, `nn.Module`, the idea of a learnable linear projection, and the Lab 2 pattern of "hand-implement, then check against autograd / library."

You do **not** need to have read "Attention is All You Need" beforehand, but you are encouraged to skim Sections 1–3 and the attention formulas after finishing Task 3, at which point the notation will be concrete.

## 3. Software Environment

- Python 3.8+
- PyTorch 2.x (for `F.scaled_dot_product_attention` and `nn.MultiheadAttention` — the verification targets)
- NumPy
- matplotlib

All experiments run on **CPU**. The Transformer here is deliberately tiny (2 layers, `d_model` = 64–128) and the corpus is a few hundred KB, so each epoch is seconds to a couple of minutes. Fixed seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).

Do not introduce additional frameworks. Write attention and the Transformer block by hand; use the library functions only as verification targets.

## 4. The Core Idea: Attention in One Page

Attention is a weighted average where the weights come from **content similarity**. Given queries `Q`, keys `K`, and values `V`:

```
Attention(Q, K, V) = softmax( Q K^T / √d_k ) · V
```

Three things to internalise:

- **Permutation invariance.** Attention with no positional information treats the sequence as a bag — swapping two tokens changes nothing about the per-token output *relative to its inputs*. This is why positional encoding is mandatory (Task 5).
- **Scaling.** As `d_k` grows, `Q·K` has larger variance, pushing softmax into saturated regions with tiny gradients. Dividing by `√d_k` keeps the pre-softmax logits at unit-ish scale. Task 2 makes you measure this.
- **Multi-head.** Instead of one big attention, run `h` smaller attentions in parallel on projected subspaces and concatenate. This lets the model attend to different relationships simultaneously (e.g. one head on syntax, another on coreference).

The full Transformer block is:

```
x → MultiHeadAttention → + (residual) → LayerNorm → FFN → + (residual) → LayerNorm → out
```

The **residual** connections and **LayerNorm** are not cosmetic — they are what make a stack of attention blocks trainable without the degradation you saw in deep plain RNNs. You will reuse the residual intuition from Lab 4's skip connections.

## 5. Tasks

Complete the tasks in order.

### Task 1: Scaled Dot-Product Attention by Hand

Implement `attention(Q, K, V)` from the formula above, with:

- An optional attention mask (to implement causality / padding).
- Manual `softmax` with numerical stability (subtract the row max — the same trick as Lab 2's numerically stable sigmoid).

Verify against `F.scaled_dot_product_attention` on a random input `(seq_len=8, d_k=16)`:
- Without mask: outputs must match to ~1e-6.
- With a causal mask: outputs must match.
- With a padding mask: outputs must match.

Record:
- Verification table: max abs difference vs `F.scaled_dot_product_attention`, for the three cases.
- The shape of each intermediate tensor (`QK^T`, after softmax, after × `V`).

### Task 2: Why Scale by 1/√d_k?

Demonstrate, empirically, what the scaling prevents:

- Generate `Q, K` with i.i.d. standard-normal entries for `d_k ∈ {4, 16, 64, 256}`.
- Compute `QK^T` and record the mean and std of its entries.
- Then push those logits through softmax and record the **max softmax probability per row** — a proxy for saturation.

Expect: as `d_k` grows, the unscaled logits grow like `√d_k`, the max softmax probability approaches 1 (saturation), and the gradient through softmax vanishes. Scaling restores std ≈ 1 and keeps softmax in a healthy regime.

Required figure: `results/scaling.png` — two panels: (a) std of `QK^T` vs `d_k`, scaled vs unscaled; (b) max softmax probability vs `d_k`, scaled vs unscaled.

Record a one-line takeaway: the scaling factor is exactly `1/√d_k` because the variance of a dot product of two length-`d_k` standard-normal vectors is `d_k`.

### Task 3: Multi-Head Attention by Hand

Implement multi-head attention:

1. Project `Q, K, V` into `h` heads with separate linear layers (or one big projection split into heads).
2. Run attention per head (reuse Task 1).
3. Concatenate the heads and apply a final output projection.

Verify against `nn.MultiheadAttention`:
- Disable the internal scaling/bias differences by matching the projection convention; use `batch_first=True`.
- Feed identical inputs (for self-attention, `Q = K = V = x`), copy weights into `nn.MultiheadAttention`, and assert outputs match to ~1e-5.
- Mind the gotcha: `nn.MultiheadAttention` packs `in_proj_weight` as a concatenation of `W_q, W_k, W_v`. Align your projections to this layout or the comparison will silently disagree.

Record:
- The verification table (output abs error per head, then concatenated).
- Parameter count of your implementation vs `nn.MultiheadAttention` for the same dims — must match.

### Task 4: Position-Wise Feed-Forward, Residual, and LayerNorm

Implement the remaining Transformer-block components:

- **Position-wise FFN**: `Linear(d_model, d_ff) → ReLU/GELU → Linear(d_ff, d_model)`. "Position-wise" just means it is applied identically to every position — i.e. a 1×1 conv, or a Linear over the last dim.
- **Residual connection**: `x = x + Sublayer(x)`. This is the Lab 4 skip-connection idea, applied inside a block instead of across layers.
- **LayerNorm**: implement it by hand (mean/var over the feature dim, learnable affine), then verify against `nn.LayerNorm`.

Assemble one **Transformer encoder block**:

```
x → LayerNorm → MultiHeadAttention → + residual → LayerNorm → FFN → + residual → out
```

(Use the pre-norm variant — it trains more stably in a tiny from-scratch setup. Note and justify this choice in your report.)

Verify the whole block forward pass runs and produces the expected shape; no library comparison needed for the assembled block, but each *component* (attention, LayerNorm) should already have been verified in isolation.

### Task 5: Positional Encoding

Attention alone is permutation-invariant — it cannot tell position 3 from position 30. Inject position information. Implement **sinusoidal** positional encoding (the original Vaswani formulation):

```
PE(pos, 2i)   = sin( pos / 10000^{2i/d_model} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d_model} )
```

Add `PE` to the token embeddings. Visualise `PE` as a heatmap (rows = position, cols = feature index) and save as `results/positional_encoding.png`.

Run a small ablation: train the character LM from Task 6 **without** positional encoding for a few steps and **with** it, and show that without it the model collapses to a position-agnostic bag-of-characters. (A short demonstration, not a full training run, is enough.)

### Task 6: Train a Tiny Character-Level Language Model

This is the payoff. Build a 2-layer Transformer encoder (or encoder-decoder with a causal mask if you want generation) with `d_model = 64` or `128`, `h = 4` heads, `d_ff = 256`, on a small text corpus (a few hundred KB — e.g. a slice of Shakespeare, or a textbook chapter). Character-level tokenisation keeps the vocabulary tiny (~100 tokens) and the model small.

Train to predict the next character. Generate a few samples with greedy and with temperature/top-k sampling.

Required outputs:
- Training loss curve `results/lm_loss.png`.
- Sample generated text (a few hundred characters) at the end of training.
- A short comparison to the Lab 5 RNN/LSTM on a comparable sequence task: which converges faster? Which handles long-range context better? You do not need a full apples-to-apples benchmark — a qualitative comparison grounded in your Lab 5 gradient plots is enough.

### Task 7 (Optional): Visualise Attention

Extract the attention weights from one head at inference time on a real input sentence and plot them as a heatmap (query position × key position). Save as `results/attention_heatmap.png`. Describe qualitatively what the head seems to attend to (is it looking at the previous token? At a delimiter? At nothing interpretable?). This is intentionally open-ended — the goal is to demystify "attention is interpretable."

## 6. Suggested Training Settings

- `d_model`: 64 or 128.
- Heads: 4.
- `d_ff`: 4 × `d_model`.
- Layers: 2 (this is a toy, not GPT).
- Sequence length: 64–128 characters.
- Batch size: 32–64.
- Optimiser: Adam, `lr = 1e-3`, with a warmup-then-decay schedule if you like (a linear warmup over the first few hundred steps is enough to demonstrate the idea; a flat lr also works at this scale).
- Epochs: enough to see the loss drop and produce coherent samples — a few epochs on a few-hundred-KB corpus.

If CPU is slow, reduce `d_model` or layers first. Do not change multiple factors at once. If a run threatens to exceed ~30 minutes, reduce `d_model` and report it.

## 7. Required Outputs

Your `lab6/` folder should contain:

- `task1_attention.py` — scaled dot-product attention + verification.
- `task2_scaling.py` — the `1/√d_k` demonstration + figure.
- `task3_multihead.py` — multi-head attention + verification.
- `task4_block.py` — FFN, residual, LayerNorm, assembled encoder block.
- `task5_positional.py` — sinusoidal PE + heatmap + ablation.
- `task6_charlm.py` — the character LM training + sampling.
- (optional) `task7_attention_viz.py`.
- `results/scaling.png`, `results/positional_encoding.png`, `results/lm_loss.png`, (optional) `results/attention_heatmap.png`.
- A comparison table: RNN vs LSTM vs Transformer (from Lab 5 + this lab) — convergence speed, long-range behaviour, parameter count.
- A report file `lab6.typ` (preferred) or `report.md`.

## 8. Report Questions

1. Write down the attention formula and derive the shape of every intermediate tensor. Why is attention `O(seq_len² · d_k)` in compute, and `O(seq_len²)` in memory? What does this imply for very long sequences?
2. From your Task 2 experiment: what is the variance of `QK^T` when `Q, K` are i.i.d. standard normal? Why does this make the `1/√d_k` scaling the *exact* choice rather than an arbitrary one?
3. Why is attention permutation-invariant without positional encoding? Give a concrete example: swap two tokens in the input and show that the per-token output (ignoring the positions you swapped) is unchanged. How does adding PE break this invariance?
4. Walk through multi-head attention. Why split into heads rather than use one big attention? What is the constraint on `d_model` and the number of heads?
5. You used the **pre-norm** variant (`x + Sublayer(LN(x))`). How does this differ from **post-norm** (`LN(x + Sublayer(x))`)? Relate the stability difference to the residual path: in pre-norm, the residual is always a clean identity-anchored path, like the skip connection in Lab 4.
6. Why LayerNorm and not BatchNorm here? (Hint: BatchNorm couples positions across the batch dimension; sequences have variable length and a per-position/per-feature statistic is more natural. Connect to Lab 4's BatchNorm-in-eval-mode subtlety.)
7. On your character-LM samples: does the model capture long-range dependencies that the Lab 5 RNN/LSTM struggled with on the copy task? Be honest — a tiny 2-layer Transformer on a small corpus will not be GPT; describe what it *does* get right and what it still gets wrong.
8. Connect this lab to Lab 5: Lab 5 showed recurrence pays a gradient cost for long-range memory. What is the *structural* reason attention avoids that cost? (Hint: every position attends to every other position in one step — no repeated multiplication.)

## 9. Evaluation Criteria

- Correct hand-implemented scaled dot-product attention, verified to ~1e-6 against `F.scaled_dot_product_attention`.
- A meaningful scaling experiment that actually shows saturation and the `1/√d_k` rationale, with the variance derivation stated.
- A hand-implemented multi-head attention matching `nn.MultiheadAttention` to ~1e-5, with the weight-packing alignment handled correctly.
- A correctly assembled, trainable Transformer block (pre-norm, residuals, LayerNorm).
- A trained character LM with a sensible loss curve and coherent-enough samples.
- Reproducibility through fixed seeds and documented hyperparameters.
- Thoughtful analysis connecting attention's structure to the vanishing-gradient problem from Lab 5.

## 10. Hints

- For the `nn.MultiheadAttention` comparison, the simplest path is: instantiate `nn.MultiheadAttention(d_model, nhead, bias=False, batch_first=True)`, then copy `in_proj_weight` (shape `(3*d_model, d_model)`) into your three projections and `out_proj.weight` into your output projection. Mind the transpose convention.
- Causal mask: a boolean tensor where `mask[i, j] = True` means "position `i` may not attend to position `j`" (for `j > i`). Pass it as `attn_mask`; `F.scaled_dot_product_attention` accepts a float mask with `-inf` at masked positions.
- Numerically stable softmax: subtract `max(logits, dim=-1, keepdim=True)` before `exp`. The Lab 2 stable-sigmoid idea, again.
- Pre-norm is the modern default for from-scratch training stability. If your post-norm run diverges, that itself is a teaching moment — record it.
- Positional encoding: you can either learn it (`nn.Embedding(max_len, d_model)`) or use the fixed sinusoidal form. This lab asks for the fixed form so you understand the original; you can compare to a learned one as a bonus.
- For the character LM, a clean training loop is `logits = model(x); loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))`. Nothing fancy.
- Sampling: greedy is `argmax`; temperature-scaled is `softmax(logits / T)`; top-k masks all but the top-k logits before sampling. Implement all three — they are short and clarifying.
- If `F.scaled_dot_product_attention` is not available in your PyTorch version, use `nn.functional.softmax` + the manual formula as the "library" reference; the verification idea is the same.
