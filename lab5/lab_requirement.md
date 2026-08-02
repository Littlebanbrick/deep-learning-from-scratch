# Lab 5: Sequence Modeling — From RNN to LSTM/GRU

## 1. Objective

In Lab 3 you built CNNs that exploit spatial locality, and in Lab 4 you reused a pretrained ResNet. But both labs work on *single fixed-size inputs* (one image → one label). Many real problems are **sequences**: text, time series, audio, sensor streams. The model's prediction at step `t` depends on information seen earlier in the sequence, so the network needs **memory**.

You already have a stub for this in `minimal-rnn/sine_rnn.py`, which uses `nn.RNN` to predict the next point of a sine wave. That stub shows the API but hides everything important: how the recurrence is unrolled, how gradients flow back through time (BPTT), why a vanilla RNN forgets long-range dependencies, and what a gated cell like LSTM actually does to fix it.

This lab grows `minimal-rnn` into a complete first-principles treatment. By completing this lab, you should be able to:

- Unroll an RNN over time by hand and derive one step of Backpropagation Through Time (BPTT).
- **Observe, not just read about, the vanishing/exploding gradient problem** on a task that genuinely requires long-range memory.
- Implement gradient clipping and show empirically that it tames explosion.
- Hand-write an LSTM cell and verify it against `nn.LSTM`.
- Explain, gate by gate, why gating mitigates the vanishing-gradient problem that vanilla RNNs suffer from.
- Decide when a gated cell is worth the extra parameters over a plain RNN.

The spirit of this lab is exactly Lab 2's: derive by hand, implement, then verify against PyTorch's autograd / library implementations.

## 2. Prerequisites

You are expected to have completed:

- Lab 2: manual forward/backward implementation and gradient-based learning. This lab reuses the "hand-derive the gradient, then verify against autograd" pattern from Lab 2 Task 4.
- `minimal-rnn/sine_rnn.py`: you should have run it and understood what `nn.RNN` does at the API level.
- Andrew Ng's RNN lectures (recurrence, BPTT concept, vanishing/exploding gradients, LSTM/GRU motivation).

You should be comfortable with: the chain rule, `nn.Module`, a manual training loop, and the idea that a recurrent network applies the *same* weights at every time step.

## 3. Software Environment

- Python 3.8+
- PyTorch 2.x
- NumPy
- matplotlib

All experiments should run on **CPU**. Use fixed random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`) for reproducibility. The sequence models here are intentionally tiny so that training stays in the seconds-to-minutes range per run.

Do not introduce additional deep learning frameworks. Write recurrence and gating by hand where the task asks for it; use `nn.LSTM` / `nn.GRU` only for the verification comparisons.

## 4. The Core Idea: Unrolling and BPTT

A vanilla RNN applies the same recurrence at every time step:

```
h_t = tanh( W_xh · x_t  +  W_hh · h_{t-1}  +  b_h )
y_t =       W_hy · h_t  +  b_y
```

Training means minimising a loss summed (or averaged) over time. Backpropagation through this *unrolled* computation graph is **BPTT** (Backpropagation Through Time). The thing to understand — and to *feel* in this lab — is that the gradient of the loss at step `t` with respect to the recurrent weights flows back through *every* intermediate `h_{t-1}, h_{t-2}, …`, and each step multiplies by `W_hh` (then through `tanh'`). Repeatedly multiplying by the same matrix is what produces the **vanishing** (singular values < 1) or **exploding** (singular values > 1) gradient behaviour.

This is the single most important takeaway of the lab, and Task 3 is designed to make you observe it directly rather than take it on faith.

## 5. Tasks

Complete the tasks in order. Each builds on the previous.

### Task 1: Hand-Written Vanilla RNN (Forward + BPTT)

Implement a vanilla RNN **from scratch** as a plain `nn.Module` (or pure NumPy) — do **not** use `nn.RNN`. Concretely:

- Implement the forward pass by explicitly looping over time steps, maintaining `h_t` from `h_{t-1}`.
- Pick the **many-to-one** regression setup from `minimal-rnn` as your warm-up: input a window of `T` sine samples, predict the next sample. This keeps the first implementation tiny.
- Implement the backward pass by hand for **one time step**: given `dL/dh_t`, derive `dL/dW_xh`, `dL/dW_hh`, `dL/dh_{t-1}`. Then chain them across the unrolled graph.
- **Verify** your manual gradients against `torch.autograd.grad` on a small random input (length `T=8`, hidden size `H=4`). This is the Lab 2 Task 4 pattern, applied to a recurrent graph.

Record:
- The verification table: for each parameter, your manual gradient norm vs autograd's, and the relative error (should be ~1e-6 or smaller).
- A short paragraph: why does `tanh'` (which is ≤ 1 everywhere) make vanishing gradients *worse* than if you used ReLU in the recurrence?

### Task 2: A Task That Genuinely Needs Memory

The sine wave is convenient but does not truly stress long-range memory. Switch to a small **copy / recall task** that is the classic diagnostic for recurrent memory:

- Input: a sequence of `T` symbols from a small alphabet (e.g. 10 symbols), followed by a delimiter, followed by blank/query positions. The model must reproduce the first `T` symbols at the end of the sequence (many-to-many, sequence-to-sequence).
- A plain feedforward network cannot solve this because the output depends on inputs seen many steps earlier.

Implement this task with your hand-written RNN (or with `nn.RNN` if you prefer speed here — the point of this task is the *behaviour*, not the implementation). Train on `T = 5` (short) and `T = 20` (long) memory lengths.

Record:
- Final accuracy for both lengths.
- The training loss curves.
- Your observation: how does performance degrade as the required memory span grows?

### Task 3: Observe the Vanishing/Exploding Gradient

This is the heart of the lab. Using your copy-task RNN from Task 2 (long span, `T = 20`), measure the gradient magnitude that reaches the **early** time steps versus the **late** time steps:

- Take a single training batch, do one forward + backward pass.
- For each time step `t`, record `‖∂L/∂h_t‖` (the norm of the gradient of the loss w.r.t. the hidden state at that step).
- Plot `‖∂L/∂h_t‖` against `t` (from `t=0` earliest to `t=T` latest).

Expect to see the gradient either **shrink toward zero** as you go back in time (vanishing) or **blow up** (exploding), depending on the initialisation of `W_hh`. Run two initialisations to see both regimes:

- Small `W_hh` (e.g. scaled by `0.5`) → vanishing.
- Large `W_hh` (e.g. scaled by `2.0`) → exploding.

Then implement **gradient clipping** (clip the global gradient norm to a threshold, e.g. 5.0) and show that it suppresses the explosion regime without changing the vanishing regime.

Required figure: `results/grad_vs_timestep.png` — gradient norm vs time step, for {small-W, large-W} × {no-clip, clip}.

### Task 4: Hand-Written LSTM Cell

Implement an LSTM cell **by hand** as a plain `nn.Module` (do not use `nn.LSTM`). The cell maintains a cell state `c_t` and hidden state `h_t`, with the four gates:

```
i_t = σ( W_xi·x_t + W_hi·h_{t-1} + b_i )   # input gate
f_t = σ( W_xf·x_t + W_hf·h_{t-1} + b_f )   # forget gate
g_t = tanh( W_xg·x_t + W_hg·h_{t-1} + b_g ) # cell candidate
o_t = σ( W_xo·x_t + W_ho·h_{t-1} + b_o )   # output gate
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

Where `σ` is the sigmoid. Loop over time steps explicitly, exactly as you did for the vanilla RNN.

**Verify** your hand-written LSTM against `nn.LSTM`:
- Feed the same input sequence and the same initial state.
- Assert that your `h_t` sequence matches `nn.LSTM`'s output to within ~1e-5.
- If they diverge, the usual culprit is the gate ordering / concatenation convention — check the PyTorch source for how `nn.LSTM` packs its weights, and align.

Record:
- The verification table (per-timestep `h_t`, your value vs `nn.LSTM`'s, absolute error).
- Param count of your cell vs `nn.LSTM` for the same `(input_size, hidden_size)` — they must match exactly.

### Task 5: Repeat the Gradient Experiment with LSTM

On the *same* long-span copy task (`T = 20`) from Task 2/3, train your hand-written LSTM and repeat the Task 3 gradient measurement:

- Plot `‖∂L/∂h_t‖` and `‖∂L/∂c_t‖` against time step.
- Compare to the vanilla RNN's curve from Task 3.

Expect: the cell-state gradient `‖∂L/∂c_t‖` stays roughly **flat across time** (because the forget gate provides an additive, near-identity path `c_t = f_t * c_{t-1} + …`), which is the whole point of gating.

Record:
- The side-by-side figure `results/grad_rnn_vs_lstm.png`.
- Final copy accuracy: vanilla RNN vs LSTM at `T = 20`.
- A short paragraph: relate the flat `c_t` gradient to the additive cell-state update, and explain why this is the structural fix for vanishing gradients (not just a patch like clipping).

### Task 6 (Optional): GRU as a Middle Ground

Implement a GRU cell by hand and verify against `nn.GRU`. Compare RNN / GRU / LSTM on the copy task across `T ∈ {5, 20, 40}` in a single accuracy-vs-span table. GRU has one fewer gate and no separate cell state; discuss whether the simpler structure is worth the lost expressiveness for your task.

## 6. Suggested Training Settings

- Hidden size: 16–64 (keep it tiny; the point is understanding, not capacity).
- Optimiser: Adam, `lr = 1e-3`, or SGD with momentum `lr = 1e-2`.
- Sequence lengths: `T = 5` (short) and `T = 20` (long).
- Gradient clipping: clip global norm to 5.0.
- Epochs: enough to converge on the copy task — these are toy tasks, so tens of epochs should suffice. If a single run exceeds ~30 minutes on CPU, reduce hidden size or sequence length first; **do not change multiple factors at once**.

If CPU training is slow, reduce hidden size first, then epochs. Each variant must differ from its baseline in exactly one axis.

## 7. Required Outputs

Your `lab5/` folder should contain:

- `task1_vanilla_rnn.py` — hand-written RNN + gradient verification.
- `task2_copy_task.py` — copy task with vanilla RNN, short and long spans.
- `task3_grad_analysis.py` — the gradient-vs-timestep experiment, with clipping.
- `task4_lstm_cell.py` — hand-written LSTM + verification against `nn.LSTM`.
- `task5_lstm_grad.py` — the RNN-vs-LSTM gradient comparison.
- (optional) `task6_gru.py`.
- `results/grad_vs_timestep.png` — Task 3 figure.
- `results/grad_rnn_vs_lstm.png` — Task 5 figure.
- A comparison table (markdown or in the report) of RNN vs LSTM: final accuracy at `T=20`, gradient behaviour, parameter count.
- A report file `lab5.typ` (preferred) or `report.md`.

## 8. Report Questions

1. Derive, for one time step, `dL/dW_hh` in terms of `dL/dh_t` and the forward quantities. Why does the repeated multiplication by `W_hh` cause vanishing/exploding gradients?
2. On your Task 3 plot, describe the shape of `‖∂L/∂h_t‖` for the small-`W_hh` and large-`W_hh` regimes. Which is vanishing and which is exploding? Why does gradient clipping fix only one of them?
3. Why does `tanh'` being ≤ 1 worsen vanishing gradients compared to ReLU? Would using ReLU in the recurrence introduce a *different* problem? (Hint: what happens to an exploding activation without a bounded nonlinearity?)
4. Walk through the four gates of your LSTM. Which gate is the "structural" fix for vanishing gradients — i.e. which one creates the additive path that lets `c_t` carry information across many steps?
5. On your Task 5 figure, why is `‖∂L/∂c_t‖` roughly flat across time while `‖∂L/∂h_t‖` is not? Relate this to the difference between the cell-state update (additive) and the hidden-state update (multiplicative).
6. Compare RNN vs LSTM on the `T = 20` copy task. How much does gating help, and is the gain larger at longer spans? Relate this to your gradient plots.
7. Connect this lab to `minimal-rnn`: the stub used `nn.RNN` as a black box on a sine wave. What did the stub hide that this lab made you confront directly?

## 9. Evaluation Criteria

- Correct hand-derived BPTT gradients, verified against autograd to ~1e-6.
- Honest, well-plotted gradient-vs-timestep measurements that actually show vanishing and exploding regimes — not just a correct-looking number.
- Correct gradient-clipping implementation that demonstrably tames explosion.
- A hand-written LSTM cell that matches `nn.LSTM` output to ~1e-5.
- A clear RNN-vs-LSTM gradient comparison that supports the "additive cell state fixes vanishing gradients" claim with evidence, not assertion.
- Reproducibility through fixed seeds and documented hyperparameters.
- Thoughtful analysis connecting the observed gradients to the structural design of LSTM.

## 10. Hints

- To get a per-timestep `dL/dh_t`, retain `h_t` as a leaf tensor with `requires_grad=True` (or use `torch.autograd.grad` with `retain_graph=True` looping backward from the loss). Alternatively, register forward hooks — but the manual approach is more instructive.
- For the exploding regime, initialise `W_hh` as `0.5 * torch.randn(H, H)`; for vanishing, `2.0 * torch.randn(H, H)`. Scale after the standard init.
- Gradient clipping by global norm:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
  ```
  But also implement it by hand once so you understand what "global norm" means.
- The copy task is sensitive to the delimiter design. Make sure there is a clear signal (a unique delimiter symbol) separating the "store" phase from the "recall" phase.
- When verifying your LSTM against `nn.LSTM`, pay attention to: (a) gate ordering in the packed weight tensor, (b) whether `nn.LSTM` expects batch-first or time-first, (c) initial state shape `(num_layers, batch, hidden)`.
- Keep `H` small (e.g. 16) for the gradient experiments — large enough to see the behaviour, small enough that the per-timestep autograd loop is fast.
