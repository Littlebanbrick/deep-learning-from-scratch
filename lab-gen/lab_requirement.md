# Lab (gen): Autoencoders and Variational Autoencoders

## 1. Objective

Every lab so far has been **discriminative**: input in, label out. This lab turns to **generative** modeling: given a training distribution, learn to *produce new samples* from it. The simplest entry point is the **autoencoder (AE)** — compress an input into a low-dimensional code, reconstruct it from the code. The code ("latent") is a learned representation; if you constrain it enough, the model is forced to capture the structure that matters.

The **variational autoencoder (VAE)** goes further: instead of learning a single code per input, it learns a *distribution* over codes, which lets you **sample** new inputs that never existed in the training set. The trick — the **reparameterisation** of a sampled Gaussian as `mean + std · ε` — is the key idea that makes a stochastic sampling step differentiable. It is also one of the cleanest "derive the gradient, verify it" opportunities in the deep-learning curriculum, which is exactly why it belongs in this repository.

By completing this lab, you should be able to:

- Build an autoencoder and explain what the latent space captures.
- See, empirically, that a vanilla AE's latent space is **not** suitable for generation (holes, no structure) — motivating the VAE.
- Derive the VAE loss (reconstruction + KL divergence) and explain the **ELBO** intuition without merely quoting it.
- Implement the reparameterisation trick by hand and verify that gradients flow through the sampled node.
- Train a VAE on MNIST, sample new digits, and traverse the latent space smoothly between two digits.
- Explain the **reconstruction–regularisation trade-off** (the β-VAE idea) with your own measurements.

The spirit is Lab 2's: understand the gradient flow, implement it cleanly, then verify.

## 2. Prerequisites

You are expected to have completed:

- Lab 1: an MLP in PyTorch and the training-loop pattern.
- Lab 2: manual forward/backward and the idea that you can verify gradients against autograd. The reparameterisation trick is a small but exact gradient-derivation exercise in the same style.
- Andrew Ng's lectures on autoencoders, or the Kingma & Welling 2013 VAE paper (skim Sections 1–2 and the loss; the full variational inference derivation is optional and can come after Task 4).

You should be comfortable with: the Gaussian distribution (mean, std, the density), KL divergence at the level of "it measures the gap between two distributions," and `nn.Module`.

## 3. Software Environment

- Python 3.8+
- PyTorch 2.x
- torchvision (MNIST)
- NumPy, matplotlib

All experiments run on **CPU**. AEs/VAEs on MNIST with a 2-D latent space train in minutes. Fixed seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).

Do not introduce additional frameworks. Write the loss and the reparameterisation by hand.

## 4. The Core Idea: Compression, then Generation

**Autoencoder.** An encoder `q(z|x)` maps an input `x` to a code `z` (typically lower-dimensional); a decoder `p(x|z)` reconstructs `x` from `z`. Train with a reconstruction loss (BCE or MSE). The latent `z` is whatever the model finds useful for reconstruction — but it has no reason to be *structured*: large "holes" between training codes mean that sampling a random `z` there produces garbage.

**VAE.** Replace the deterministic code with a distribution: the encoder outputs `(μ, σ²)` for each input, and we sample `z ~ N(μ, σ²)`. Two forces shape the latent space:

- **Reconstruction loss** pulls each `z` toward whatever reconstructs its input — it wants `z` to *remember*.
- **KL divergence** `KL(N(μ,σ²) ‖ N(0,1))` pushes the encoded distributions toward a standard normal — it wants `z` to *forget* which input it came from, so that the whole latent space is densely populated and samplable.

The balance of these two is the whole game (Task 5's β controls it). The **reparameterisation trick** is what lets us backprop through the `z = μ + σ·ε` sample: gradients flow through `μ` and `σ` directly, while `ε ~ N(0,1)` is treated as external noise.

## 5. Tasks

Complete the tasks in order.

### Task 1: A Vanilla Autoencoder

Build a simple AE: encoder `784 → 128 → 32 → 2` (use a **2-D latent** so you can visualise it), decoder `2 → 32 → 128 → 784`, sigmoid on the output, BCE reconstruction loss. Train on MNIST for ~10–20 epochs.

Record:
- Reconstruction loss curve.
- A grid of reconstructions: original vs reconstructed for ~10 test images.
- A **2-D scatter of the latent codes** for the test set, coloured by digit label (`results/ae_latent.png`). You should see clusters by digit — but also **gaps** between clusters.

### Task 2: Why the AE Cannot Generate

Demonstrate the AE's failure as a generative model:

- Sample random points in the 2-D latent square (say `[-3, 3]²` on a grid) and decode each.
- Save the resulting image grid (`results/ae_samples.png`).

Expect: the well-encoded regions reconstruct recognisable digits, but the **gaps** between clusters decode to nonsense — because no training input ever mapped there, the decoder was never trained on those codes. This is the failure the VAE fixes.

Record a short observation: where do the decoded images look like digits, and where do they break down? Relate this to the gaps in your Task 1 scatter.

### Task 3: The Reparameterisation Trick (by Hand)

Implement the sampling step explicitly and verify gradients flow:

- `μ, logvar = encoder(x)` (output `logvar` for numerical stability, not `σ` directly).
- `std = exp(0.5 * logvar)`
- `ε ~ N(0, 1)` (detach `ε` from the graph — it is external noise).
- `z = μ + std * ε`

**Verify** that gradients flow to `μ` and `std` (equivalently `logvar`) but **not** to `ε`:
- Run `torch.autograd.grad(loss, [μ, std, ε])` and assert `dL/dε` is zero (or absent) while `dL/dμ` and `dL/dstd` are nonzero.
- Convince yourself with a 1-D toy: define `loss = (z - target)²` for a fixed target, sample `ε` once, and check `dL/dμ = 2(z - target)` and `dL/dstd = 2(z - target) · ε` match the hand-derived values.

This is the Lab 2 "hand-derive, verify against autograd" pattern applied to a stochastic graph.

### Task 4: The VAE Loss (ELBO)

Implement the VAE loss:

```
L = Reconstruction(x, x̂)  +  KL( N(μ, σ²) ‖ N(0, 1) )
```

For a Gaussian encoder and Bernoulli decoder, the KL has a closed form:

```
KL = -0.5 · Σ ( 1 + logvar - μ² - exp(logvar) )
```

State this formula in your report and *derive it in one line* from the general KL definition `KL = ∫ q(z) log(q/p) dz` — enough to show you know where it comes from, not a full page of algebra. (Ask if you want the full derivation; per the collaboration notes, derivations are provided on request.)

Train the VAE on MNIST, same architecture as Task 1 but with the encoder outputting `μ` and `logvar`.

Record:
- The two loss components over training (reconstruction and KL), plotted separately — watch how the balance shifts.
- Reconstructions of test images.
- A 2-D latent scatter (`results/vae_latent.png`) — expect a **denser, more continuous** distribution than the AE's, with clusters bleeding into each other.

### Task 5: Generation and Latent Traversal

Now the payoff — the VAE actually generates:

- Sample `z ~ N(0, 1)` directly (ignore the encoder) and decode. Save a grid of generated digits (`results/vae_samples.png`). Expect recognisable, novel digits — not perfect, but clearly digits.
- **Latent traversal**: pick two test images (e.g. a 1 and a 7), take their `μ` codes, and linearly interpolate between them in 2-D, decoding at each step. Save the traversal as a row of images (`results/vae_traversal.png`). Expect a *smooth* morph between the digits — no sudden nonsense, because the KL has populated the latent space.

### Task 6: The β-VAE — Reconstruction vs Regularisation Trade-off

Introduce a weight `β` on the KL term:

```
L = Reconstruction  +  β · KL
```

Run three values: `β = 0` (degenerates to an AE — confirm it matches Task 1), `β = 1` (standard VAE), `β = 4` (over-regularised). For each, record:

- Reconstruction quality (a few sample reconstructions).
- Latent scatter structure (how "squeezed" toward `N(0,1)` it is).
- Sample quality from random `z`.

Required figure: `results/beta_tradeoff.png` — reconstructions and random samples for the three β values side by side. Expect: low β → sharp reconstructions, poor samples (AE-like); high β → blurry reconstructions, smooth samples but they lose digit identity. The standard `β = 1` is the balance point.

## 6. Suggested Training Settings

- Latent dim: 2 (for visualisation). You may also run a 16-D version for better sample quality and visualise with t-SNE — optional.
- Hidden: 128 → 32 → latent.
- Adam `lr = 1e-3`, batch 128, ~20 epochs.
- Reconstruction loss: BCE per pixel (output sigmoid). Alternatively MSE on raw pixels — pick one, state it.
- Fixed seed `42`.

If CPU is slow, reduce epochs first. Each β variant must differ from `β = 1` in exactly that one axis.

## 7. Required Outputs

Your `lab-gen/` folder should contain:

- `task1_autoencoder.py` — AE + latent scatter.
- `task2_ae_failure.py` — the gap-decodes-to-nonsense demonstration.
- `task3_reparam.py` — reparameterisation + gradient verification.
- `task4_vae.py` — VAE loss + training + latent scatter.
- `task5_generation.py` — random sampling + latent traversal.
- `task6_beta.py` — the β sweep.
- `results/ae_latent.png`, `results/ae_samples.png`, `results/vae_latent.png`, `results/vae_samples.png`, `results/vae_traversal.png`, `results/beta_tradeoff.png`.
- A comparison table: AE vs VAE(β=1) — reconstruction error, sample quality (qualitative), latent-space structure.
- A report file `lab-gen.typ` (preferred) or `report.md`.

## 8. Report Questions

1. What does the latent code of a vanilla AE capture, and why is it *not* a generative model? Use your Task 2 figure: point to a region of the latent space where decoding fails, and explain why.
2. Derive the reparameterisation gradient: given `z = μ + σ·ε` and a loss `L(z)`, write `∂L/∂μ` and `∂L/∂σ` in terms of `∂L/∂z` and `ε`. Why does this let us train through a sampling step that is, fundamentally, non-differentiable?
3. State the KL formula `KL = -0.5 Σ(1 + logvar − μ² − exp(logvar))` and derive it in one or two lines from the Gaussian KL definition. Why is the KL term *necessary* for generation, not just a regulariser?
4. On your Task 5 traversal: why is the morph between two digits *smooth* for the VAE but would be discontinuous for the AE? Relate this to what the KL term does to the latent space's geometry.
5. From Task 6: describe the reconstruction-vs-generation trade-off as β varies. What does β → 0 recover, and what does β → ∞ collapse to? Why is β = 1 the principled choice, and when might you deliberately deviate?
6. The VAE is called a "variational" autoencoder. What is being approximated (the evidence lower bound, ELBO), and what is the "variational" part? (One-paragraph answer — do not derive the full ELBO unless you want to.)
7. Connect this lab to the rest of the track: the VAE's encoder-decoder structure is a precursor to the **encoder-decoder Transformer** and to **diffusion** (which you may meet later). What does the VAE share with a Transformer encoder-decoder, and what does it lack that diffusion later provides?

## 9. Evaluation Criteria

- A working AE with a clear latent-space scatter showing both clusters and gaps.
- A convincing demonstration that the AE fails to generate in its gaps.
- A correct reparameterisation, verified to pass gradients to `μ`/`σ` but not `ε`.
- A correct closed-form KL, with a one-line derivation showing its origin.
- A trained VAE that generates recognisable novel digits and traverses smoothly.
- An honest β sweep showing the reconstruction-regularisation trade-off.
- Reproducibility through fixed seeds and documented hyperparameters.

## 10. Hints

- Output `logvar`, not `σ`, to keep `σ` positive and avoid numerical issues: `std = exp(0.5 * logvar)`.
- `ε = torch.randn_like(std)`; do **not** forget to detach `ε` conceptually — though `torch.randn_like` produces a tensor without `requires_grad` already, so it naturally does not receive gradients.
- For the gradient verification, fix `ε` to a constant (e.g. `ε = 1.0`) so the algebra is checkable by hand.
- BCE for reconstruction works because MNIST pixels are in `[0, 1]`; the output sigmoid matches. If you use MSE, no sigmoid on the output.
- A 2-D latent is deliberately small — reconstructions will be blurry. That is fine and even instructive; it makes the regularisation-vs-reconstruction tension visible. Run a 16-D version only if you want sharper samples.
- For the β sweep, keep everything else fixed and only change β. Reset the seed before each run so the three models start from the same initialisation.
- If samples are all blurry blobs, the KL is dominating — lower β or train longer. If samples are sharp but the latent space has holes (AE-like), the KL is too weak — raise β.
