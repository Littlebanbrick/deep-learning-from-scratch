# Lab (opt): Optimizers from Scratch

## 1. Objective

Every lab so far has used `torch.optim.Adam` or SGD-with-momentum as a black box. This lab opens that box. The three workhorses of deep learning — **SGD with momentum**, **RMSProp**, and **Adam** — are each a small, precise update rule, and each one was invented to fix a specific failure of the previous. Implementing them by hand is the fastest way to understand *why* optimisation is hard (ill-conditioning, noisy gradients, saddle points) and *why* these particular rules help.

This lab is the purest expression of the repository's "Lab 2" philosophy: derive the update, implement it in NumPy, and verify against the framework. It depends on no data and trains in seconds.

By completing this lab, you should be able to:

- Implement SGD, SGD-with-momentum, RMSProp, and Adam from scratch in NumPy.
- Explain what problem each one solves that the previous did not, with a concrete pathological surface as evidence.
- Verify your hand-written optimiser against the PyTorch one by replaying identical trajectories.
- Diagnose, from a loss curve, which optimisation pathology is occurring (oscillation, stalling, divergence) and which optimiser would help.
- Explain the **bias-correction** in Adam and why it matters in the first few steps.

## 2. Prerequisites

You are expected to have completed:

- Lab 2: manual gradient descent (the SGD loop), and the idea that you compute a gradient and take a step `θ ← θ − lr · g`. This lab generalises that one step.
- Andrew Ng's lectures on optimisation (momentum, RMSProp, Adam), or equivalently the relevant sections of the original papers (Kingma & Ba 2014 for Adam; Hinton's lecture notes for RMSProp).

You should be comfortable with: a gradient-descent loop, NumPy array ops, and the idea that `θ` is just a vector of parameters.

## 3. Software Environment

- Python 3.8+
- NumPy
- PyTorch (only as the verification target — the optimisers themselves are written in NumPy)
- matplotlib

No datasets, no GPU, no long runs. Every experiment is a small synthetic surface that converges in seconds. Fixed seeds (`np.random.seed(42)`).

Do not introduce additional frameworks. Write the update rules by hand.

## 4. The Core Idea: One Update Step, Four Ways

All four optimisers share the skeleton: compute gradient `g`, maintain some state, update `θ`. They differ in **what state** they keep and **how** they combine it with the current gradient.

| Optimiser | State | Update intuition |
|---|---|---|
| **SGD** | none | step in the direction of `−g` |
| **SGD+momentum** | velocity `v` | accumulate a running mean of past gradients (inertia through flat/noisy regions) |
| **RMSProp** | squared-grad `s` | scale the step per-parameter by `1/√(running squared grad)` (small steps where grads are consistently large, big steps where they are small) |
| **Adam** | both `v` and `s`, bias-corrected | momentum *and* per-parameter scaling, with a correction so the first few steps are not biased toward zero |

The two pathologies that motivate everything:

- **Ill-conditioning** (a long narrow valley): SGD oscillates across the valley and crawls along it. Momentum damps the oscillation; per-parameter scaling (RMSProp/Adam) takes big steps along the long axis.
- **Noisy/sparse gradients**: the raw gradient is a bad estimator of the true descent direction. Running averages (momentum, and the squared-grad mean) are better estimators.

## 5. Tasks

Complete the tasks in order.

### Task 1: Plain SGD on a Quadratic — and Why It Is Slow

Implement plain SGD in NumPy:

```
g = ∇f(θ)
θ ← θ − lr · g
```

Test it on a **2-D quadratic** `f(x, y) = a·x² + b·y²` with `a = 1, b = 50` — the classic "long narrow valley" (the Rosenbrock-like ill-conditioning). Start from `(-3, 4)`.

Record the trajectory `(x_t, y_t)` over steps and plot it as a path over a contour plot of `f` (`results/sgd_valley.png`).

Expect: SGD zig-zags violently across the narrow `y` direction (steep gradient, large step) and crawls slowly along the flat `x` direction (small gradient, small step). This is the canonical ill-conditioning failure.

Also compute and record the **condition number** `b/a = 50`, and note how it predicts the zig-zag severity. Try `b/a = 5` and `b/a = 500` and observe how the trajectory degrades.

### Task 2: SGD with Momentum

Add momentum:

```
v ← β·v + g        # (or β·v + (1−β)·g; pick the convention and state it)
θ ← θ − lr·v
```

State which convention you use (the "heavy-ball" `v ← βv + g` is simpler; PyTorch uses `v ← βv + (1−β)g` with dampening — match PyTorch if you intend the Task 5 verification to be exact).

Run on the same valley from the same start, with `β = 0.9`. Plot the trajectory alongside plain SGD (`results/sgd_vs_momentum.png`).

Expect: the zig-zag is **damped** — momentum accumulates the consistent `x`-direction signal while averaging out the oscillating `y`-direction signal. The path straightens toward the minimum.

Record: number of steps to reach `f < 1e-3` for SGD vs SGD+momentum.

### Task 3: RMSProp — Per-Parameter Scaling

Implement RMSProp:

```
s ← β·s + (1−β)·g²
θ ← θ − lr · g / (√s + ε)
```

Run on the same valley. Plot alongside SGD and momentum (`results/three_on_valley.png`).

Expect: RMSProp takes **large steps in the flat `x` direction** (small `s`, so `1/√s` is large) and **small steps in the steep `y` direction** (large `s`). This directly attacks ill-conditioning without needing momentum.

Record: steps-to-converge for all three. Discuss: momentum and RMSProp attack the *same* problem (ill-conditioning) from two different angles — averaging the *direction* vs scaling the *magnitude*. Which converges faster on this particular valley?

### Task 4: Adam — and the Bias Correction

Implement Adam:

```
m ← β1·m + (1−β1)·g      # first moment (momentum)
v ← β2·v + (1−β2)·g²     # second moment (per-param scaling)
m̂ = m / (1 − β1ᵗ)        # bias-corrected first moment
v̂ = v / (1 − β2ᵗ)        # bias-corrected second moment
θ ← θ − lr · m̂ / (√v̂ + ε)
```

The **bias correction** is the subtle part. Because `m` and `v` are initialised to zero, the first few estimates are biased toward zero (they have not "warmed up"). The correction `1/(1−βᵗ)` scales them up to compensate, where `t` is the timestep.

Demonstrate the bias correction's effect:

- Run Adam **with** and **without** bias correction on the valley, plotting the loss over the first 20 steps (`results/bias_correction.png`).
- Expect: without correction, the first ~10 steps are far too small (the zero-initialised `m`, `v` have not accumulated), so the loss barely moves; with correction, the early steps are properly sized and the loss drops immediately.

Record a one-line explanation: the correction matters most in the **early** steps (when `t` is small and `1−βᵗ` is far from 1); as `t → ∞`, `1−βᵗ → 1` and the correction vanishes.

### Task 5: Verify Against PyTorch

This is the Lab 2 verification step, applied to optimisers. For each of momentum / RMSProp / Adam:

- Build the **same** small parameter vector in both NumPy and a `torch.Tensor`.
- Use your NumPy optimiser on one and the corresponding `torch.optim` optimiser on the other, with **identical** hyperparameters, identical initial values, and identical gradients fed in at each step.
- Step both for ~50 iterations, feeding the analytic gradient of the valley `∇f = (2ax, 2by)`.
- Assert that the two trajectories match to ~1e-7.

Gotchas that will break exact match if ignored:
- PyTorch's momentum convention (`v ← βv + (1−β)g` with `dampening`, vs heavy-ball). Match it exactly or your verification will fail by a small drift.
- PyTorch's RMSProp/Adam weight-decay and `eps` placement (Adam divides by `√v + eps`, not `√(v + eps)`). Check the source.
- Adam's `amsgrad` flag — leave it off.

Record a table: optimiser × {max abs trajectory difference vs PyTorch}. All should be ~1e-7.

### Task 6: A Realistic Test — A Tiny Neural Net

Train the Lab 2 logistic regression (or a 1-hidden-layer net) on the moons dataset with all four optimisers, same seed, same initialisation, same number of steps. Plot the four loss curves on one figure (`results/four_on_moons.png`).

Record: which converges fastest, which is most stable, whether the differences match your predictions from the synthetic-valley experiments. This is the "does it matter on a real problem" sanity check.

### Task 7 (Optional): Diagnose-then-Prescribe

Write a small function that takes a recorded loss curve and tries to classify the failure mode (oscillation → too-high lr or needs momentum; stalling on a plateau → needs momentum or per-param scaling; divergence → lr too high). Apply it to your Task 6 curves. This is open-ended and meant to build the diagnostic intuition that lets you look at a training run and know what to change.

## 6. Suggested Settings

- Valley: `f(x,y) = a x² + b y²`, start `(-3, 4)`, `a=1, b=50`.
- Steps: ~50–200 for the valley; convergence threshold `f < 1e-3`.
- `lr`: tune per-optimiser on the valley (Adam ~0.1, SGD ~0.01, momentum ~0.01, RMSProp ~0.05 — these are starting points, not prescriptions).
- `β = 0.9` (momentum, RMSProp), `β1 = 0.9, β2 = 0.999, ε = 1e-8` (Adam).
- Fixed seed `42`.

## 7. Required Outputs

Your `lab-opt/` folder should contain:

- `task1_sgd.py` — SGD on the valley + trajectory figure.
- `task2_momentum.py` — SGD+momentum.
- `task3_rmsprop.py` — RMSProp.
- `task4_adam.py` — Adam + bias-correction demonstration.
- `task5_verify.py` — trajectory verification against `torch.optim`.
- `task6_moons.py` — the four-optimiser comparison on a real problem.
- (optional) `task7_diagnose.py`.
- `results/sgd_valley.png`, `results/sgd_vs_momentum.png`, `results/three_on_valley.png`, `results/bias_correction.png`, `results/four_on_moons.png`.
- A comparison table: optimiser × {valley steps-to-converge, moons final loss, max traj diff vs PyTorch}.
- A report file `lab-opt.typ` (preferred) or `report.md`.

## 8. Report Questions

1. Write the four update rules. For each, state in one sentence what failure of the previous one it addresses.
2. On your valley trajectory: why does plain SGD zig-zag, and why does the zig-zag worsen as the condition number `b/a` grows? Relate this to the **ratio of the gradient's two components**.
3. Momentum and RMSProp both "fix" ill-conditioning, but differently — momentum averages the *direction*, RMSProp scales the *magnitude*. Give an intuitive argument for why combining the two (Adam) can be better than either alone. Is it always better? When might plain SGD-with-momentum win?
4. Derive the Adam bias correction. If `m` is initialised to zero and updated as `m ← β1·m + (1−β1)·g`, what is `E[m_t]` as a function of the true gradient mean, and why is `1/(1−β1ᵗ)` the exact correction? Why does the correction vanish as `t → ∞`?
5. From your Task 6 curves: which optimiser won on the moons problem, and was it the one you predicted? If a "fancier" optimiser (Adam) lost to plain momentum, what does that tell you about the **no-free-lunch** nature of optimisers?
6. You have now seen optimisation from both sides: Lab 2 derived the *gradient* by hand, this lab derives the *update step* by hand. Why is it that even with a perfect gradient, the *choice of step* still matters enormously? (Hint: the gradient is local; the minimum may be far, and the curvature may differ across directions.)
7. Connect to the rest of the track: in Lab 6 you trained a Transformer with Adam. Now that you know Adam's internals, can you explain why Transformers are notoriously sensitive to learning rate and benefit from warmup? (Hint: the early-step bias correction interacts with the scale of the initial gradients.)

## 9. Evaluation Criteria

- Correct hand-implementations of SGD, momentum, RMSProp, Adam in NumPy.
- A clear ill-conditioning demonstration on the quadratic valley that actually shows the zig-zag and its dependence on condition number.
- A working Adam bias-correction demonstration that shows the early-step difference.
- **Exact** trajectory match against `torch.optim` to ~1e-7, with the convention gotchas (momentum formulation, eps placement) handled.
- A honest four-optimiser comparison on a real problem, with the result discussed rather than buried.
- Reproducibility through fixed seeds and documented hyperparameters.
- Thoughtful connection from optimiser internals back to the training behaviour you observed in Labs 1–6.

## 10. Hints

- For exact PyTorch matching, read the source of `torch.optim.SGD`, `RMSprop`, and `Adam` — the conventions (especially momentum's `dampening` and Adam's `eps` placement: `/(sqrt(v)+eps)` not `sqrt(v+eps)`) are spelled out there. Matching them is the point of the exercise.
- The analytic gradient of `f(x,y) = a x² + b y²` is `(2ax, 2by)` — feed this directly rather than finite-differencing, so the verification is exact.
- For the bias-correction demo, log the loss at every one of the first ~20 steps; the effect is in the early steps only.
- `ε` in RMSProp/Adam is a numerical guard, not a tunable — keep it at `1e-8`.
- If your Adam trajectory drifts from PyTorch's after ~10 steps, the usual culprit is the bias-correction timestep `t` (increment it per step, starting at 1) or the `eps` placement.
- The "diagnose" task (Task 7) is deliberately heuristic — there is no canonical classifier of loss-curve pathologies. The point is to build the mental lookup table: oscillation ↔ lr/momentum, stalling ↔ needs scaling, divergence ↔ lr too high.
