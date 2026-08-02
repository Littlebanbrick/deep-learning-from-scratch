# Lab 7: Vision Transformer — Where CV and Transformers Meet

## 1. Objective

In Lab 3 you built CNNs whose inductive bias — **locality** and **weight sharing** — was *baked into the architecture*. In Lab 4 you reused a pretrained ResNet and saw that transfer learning works because ImageNet pretraining injects a strong prior. In Lab 6 you built a Transformer that processes sequences with attention and carries **no spatial prior at all**.

This lab is the confluence. In 2020, Dosovitskiy et al. asked a provocative question: *do you even need convolutions?* Their Vision Transformer (ViT) cuts an image into patches, treats each patch as a token (like a word), feeds the sequence through a Transformer encoder, and — given enough data — matches or beats CNNs. The cost: it gives up CNN's locality prior, so it needs far more data to learn the same "nearby pixels are correlated" fact that a CNN gets for free.

By completing this lab, you should be able to:

- Implement a minimal Vision Transformer by hand, reusing the Transformer block you built in Lab 6.
- Explain the patch embedding: how a 2D image becomes a sequence of tokens.
- Articulate the central tension — **inductive bias vs data** — with your own measurements, not just the paper's claim.
- Run a **controlled comparison** between your ViT and your Lab 3 CNN on the *same* MNIST data, same seeds, same compute budget, and explain when each wins.
- (Optionally) reuse the Lab 4 transfer-learning playbook with a pretrained ViT, and compare it to the ResNet transfer-learning results.

The spirit is Lab 3/Lab 4's: a controlled comparison that changes one axis at a time, on data you already understand.

## 2. Prerequisites

You are expected to have completed:

- Lab 3: building and training CNNs in PyTorch, tensor-shape tracking, controlled comparisons on MNIST.
- Lab 6: the Transformer block — multi-head attention, residual + LayerNorm, positional encoding. **This lab reuses your Lab 6 Transformer block directly.** If your Lab 6 block is clean and modular, this lab is mostly a new input pipeline and a new head; the core is reused.
- Andrew Ng's lectures on the Transformer, or the "An Image is Worth 16x16 Words" paper (skim Sections 1–3 and Figure 1; the rest can come after Task 4).

You should be comfortable with: `nn.Module`, the MNIST `DataLoader` from Lab 3, and the Lab 6 Transformer block.

## 3. Software Environment

- Python 3.8+
- PyTorch 2.x
- torchvision
- NumPy, matplotlib

All experiments run on **CPU**. The ViT here is deliberately tiny (4 layers, `d_model` = 64–128, patch size 7 or 14) and MNIST is small, so each epoch is seconds to a couple of minutes. Fixed seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).

Do not introduce additional frameworks. Reuse *your own* Lab 6 Transformer block. If you did not modularise it, refactor it into a small `transformer.py` and import it here — that refactoring is itself a healthy exercise.

## 4. The Core Idea: An Image is a Sequence of Patches

A ViT does three things to turn an image into something a Transformer can eat:

1. **Patchify.** Split the `H×W` image into non-overlapping `P×P` patches. A `28×28` MNIST image with `P=7` yields `(28/7)² = 16` patches. Each patch is a `P·P·C = 7·7·1 = 49`-dim vector (flattened).
2. **Embed.** Linearly project each flattened patch to `d_model` (a learnable `Linear(P*P*C, d_model)`). This is the "patch embedding" — the analogue of a word embedding.
3. **Position.** Add a positional encoding (learned or sinusoidal — reuse Lab 6's) so the model knows *which* patch is where. Then prepend a learnable `[CLS]` token whose final hidden state is used for classification.

After that, the sequence goes through your Lab 6 Transformer encoder unchanged. The `[CLS]` token's output is fed to a small `Linear(d_model, num_classes)` head.

The whole point of the lab is the comparison in Task 3: a ViT has **no locality prior**, so on a small dataset like MNIST it has to *learn* that nearby pixels are correlated, whereas your Lab 3 CNN knew it from the first forward pass.

## 5. Tasks

Complete the tasks in order.

### Task 1: Patch Embedding and the [CLS] Token

Implement the image-to-sequence front end:

- A patch embedding. Two valid implementations to be aware of — implement the explicit one:
  - **Explicit** (do this): `nn.Unfold` or a manual reshape to cut the image into patches, then a `Linear(P*P*C, d_model)`.
  - **Conv trick** (mention in report): a `Conv2d(C, d_model, kernel_size=P, stride=P)` produces exactly the same result, because a strided conv over non-overlapping patches is a linear projection per patch. You do not need to use this, but explain the equivalence in your report.
- A learnable `[CLS]` token (`nn.Parameter` of shape `(1, 1, d_model)`) prepended to the patch sequence.
- A positional encoding added to the combined sequence (CLS + patches). Reuse your Lab 6 positional encoding.

Verify the shapes explicitly for one dummy batch — print the tensor shape after each stage:
- Input `(B, 1, 28, 28)`
- After patchify `(B, num_patches, P*P*C)`
- After projection `(B, num_patches, d_model)`
- After prepending CLS `(B, num_patches+1, d_model)`
- After adding PE `(B, num_patches+1, d_model)`

This task is mostly about getting the shapes right, exactly as in Lab 3 Task 2.

### Task 2: Assemble and Train a Minimal ViT

Stack your Lab 6 Transformer encoder blocks on top of the patch embedding. Take the final hidden state of the `[CLS]` token, pass it through a `Linear(d_model, 10)` head, and train on MNIST with `CrossEntropyLoss`.

Suggested config: `P=7` (so 16 patches), `d_model=64`, 4 heads, 4 layers, `d_ff=256`. Train ~10 epochs with Adam `lr=1e-3`.

Record:
- Final test accuracy.
- Training loss / accuracy curves.
- Parameter count.

### Task 3: Controlled Comparison — ViT vs Your Lab 3 CNN

This is the headline experiment, and it must be a **fair** comparison — the kind Lab 3 and Lab 4 demanded. Hold these fixed across both models:

- **Same data**: MNIST, same train/test split, same normalisation.
- **Same seed**: `42`, reset before each model.
- **Same compute budget**: same number of epochs, same batch size.
- **Comparable parameter count**: tune the ViT's `d_model`/layers so its parameter count is within ~2× of your Lab 3 LeNet-style CNN. A ViT with 10× the parameters winning is not interesting; what matters is the comparison at comparable capacity.

Then vary **one** axis at a time:

#### Experiment A: Full-data comparison
Train both on the full 60k MNIST training set. Record test accuracy and convergence speed (epochs to reach 98%). Expect: roughly comparable, CNN likely slightly ahead or on par.

#### Experiment B: Low-data comparison
Train both on **5% of the training set** (~3000 images), same seed for the subsample. Record test accuracy. Expect: the CNN's locality prior should help it degrade *less* than the ViT. This is the central "inductive bias vs data" demonstration.

#### Experiment C: Augmentation
With the low-data setting, add light augmentation (small random affine, as in Lab 3). Does augmentation close the ViT's gap to the CNN? Augmentation is a way to *inject* the locality prior the ViT lacks, indirectly.

Required figure: `results/vit_vs_cnn.png` — test accuracy for {CNN, ViT} × {full data, 5% data} × {no-aug, aug}. A grouped bar chart or a small table; either is fine.

### Task 4: When Does the Inductive Bias Matter?

Based on your Task 3 results, write a focused analysis:

- At what dataset size does the ViT catch up to or surpass the CNN? (You may need to add one more data point — e.g. 20% — to see the crossover.)
- Frame the answer in terms of the **inductive bias / data trade-off**: the CNN's locality prior is a *helpful constraint* when data is scarce (it rules out absurd hypotheses) and a *harmful constraint* when data is abundant (it prevents the model from finding better-than-convolutional solutions). Your measurements should make this concrete, not just quoted from the paper.

This connects directly to Lab 4 Task 5's data-size ablation — the same experimental design, applied to a different question.

### Task 5 (Optional): Pretrained ViT Transfer Learning

Reuse the Lab 4 transfer-learning playbook with a pretrained ViT (e.g. `vit_b_16` from torchvision, pretrained on ImageNet). Adapt it to the Oxford-IIIT Pet dataset from Lab 4 with the same three strategies (scratch / frozen / fine-tune). Compare to your Lab 4 ResNet18 transfer-learning results.

This task is optional and compute-heavier (ViT-B is bigger than ResNet18). If CPU runtime is a concern, you may restrict it to the frozen-backbone strategy and a reduced epoch count, and state that clearly.

## 6. Suggested Training Settings

- Patch size `P = 7` (MNIST `28×28` → 16 patches). `P = 14` (4 patches) also works and is faster but loses resolution; note the trade-off.
- `d_model = 64`, heads = 4, layers = 4, `d_ff = 256`.
- Batch size 64–128.
- Adam `lr = 1e-3`, ~10 epochs.
- Fixed seed `42`, reset before each variant.

If CPU is slow, reduce `d_model` or layers first. Do not change multiple factors at once. Each variant must differ from its baseline in exactly one axis.

## 7. Required Outputs

Your `lab7/` folder should contain:

- `task1_patchify.py` — patch embedding + CLS token + shape verification.
- `task2_vit.py` — assembled ViT + MNIST training.
- `task3_comparison.py` — the controlled ViT-vs-CNN comparison (full data, low data, augmentation).
- (optional) `task5_transfer.py`.
- `results/vit_vs_cnn.png` — the headline comparison figure.
- A comparison table: model × {full data, 5% data} × {no-aug, aug} → test accuracy, param count, epochs-to-98%.
- A report file `lab7.typ` (preferred) or `report.md`.

## 8. Report Questions

1. Explain the patch embedding. Why is a `Conv2d(C, d_model, kernel_size=P, stride=P)` mathematically equivalent to "flatten patches then linearly project"? Verify the equivalence numerically on a dummy input (run both, compare outputs).
2. Why does the ViT need a positional encoding when a CNN does not? Relate this to the permutation invariance you observed in Lab 6.
3. What is the `[CLS]` token for? Could you instead average-pool over patches and feed that to the head? What would change?
4. From your Task 3 results: describe how the CNN-vs-ViT gap changes with dataset size. At what size does the ViT become competitive? Frame this as the **inductive-bias-vs-data trade-off**: the CNN's locality is a free lunch when data is scarce and a straitjacket when data is abundant.
5. In Experiment C, did augmentation close the ViT's low-data gap? Why might affine augmentation be a *substitute* for the locality prior the ViT lacks? (Hint: augmentation teaches the model "a translated digit is the same digit," which is exactly the translation equivariance a CNN gets from weight sharing.)
6. Compare the parameter count of your ViT and your Lab 3 CNN at comparable accuracy. Is the ViT more parameter-efficient, less, or about the same on MNIST? Why might the answer flip on a larger, more diverse dataset?
7. Connect this lab to Lab 4: in Lab 4 the inductive bias came from *pretraining* (ImageNet weights), not from the architecture. In Lab 7, the ViT gives up architectural inductive bias. Are these two kinds of "where does the prior come from" — architecture vs pretraining — ever in tension? When would you prefer one over the other?

## 9. Evaluation Criteria

- Correct patch embedding with verified shapes and a correct `[CLS]` / PE setup.
- A correctly assembled ViT that trains and reaches competitive MNIST accuracy.
- A **fair** CNN-vs-ViT comparison: identical data, seeds, compute, comparable parameter count, only one axis varied at a time.
- An honest low-data result that actually shows the inductive-bias effect — if your ViT happens to beat the CNN at 5% data, investigate and report why rather than hiding it.
- A clear statement of the inductive-bias-vs-data trade-off grounded in your measurements.
- Reproducibility through fixed seeds and documented hyperparameters.
- Thoughtful connection to Lab 3 (inductive bias in architecture), Lab 4 (inductive bias from pretraining), and Lab 6 (the Transformer block reused).

## 10. Hints

- The cleanest patchify is `x.view(B, C, H//P, P, W//P, P).permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, -1)` — verify the permutation carefully; off-by-one in the permute is the most common bug.
- The `[CLS]` token: `self.cls = nn.Parameter(torch.zeros(1, 1, d_model)); cls = self.cls.expand(B, -1, -1)`.
- If your Lab 6 Transformer block expects `(seq_len, batch, d_model)` (time-first), either adapt it to batch-first or permute. Pick one convention and stick to it; document the choice.
- For the parameter-count-matching: a 4-layer `d_model=64` ViT is in the same ballpark as your Lab 3 LeNet-style CNN on MNIST. Verify and adjust.
- The 5% subsample must use the *same* indices for both models. Generate the indices once with a fixed seed and pass them to both training scripts.
- For the optional pretrained-ViT task: `torchvision.models.vit_b_16(weights=...)`; the head is `model.heads.head`. The same BatchNorm-in-eval subtlety from Lab 4 does not apply (ViT uses LayerNorm), but the "freeze → train head → unfreeze with differential lr" playbook transfers directly.
- If your ViT does not converge, the usual culprit is the learning rate (Transformers are sensitive to lr) or a missing/incorrect positional encoding. Check both before increasing capacity.
