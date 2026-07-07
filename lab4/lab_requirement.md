# Lab 4: Transfer Learning with Pretrained ResNet18 on Oxford-IIIT Pet

## 1. Objective

In Lab 3 you built CNNs from scratch for MNIST and saw that a 44k-parameter LeNet beats a 118k-parameter MLP. That worked because MNIST is small and simple. But in the real world, datasets are tiny, images are large colour photographs, and training a deep network from random initialisation requires far more data and compute than you have.

This lab introduces **transfer learning**: instead of initialising your network randomly, you start from weights that someone else has already trained on a large dataset, and you adapt them to your own task. You will use a ResNet18 pretrained on ImageNet (1.28M images, 1000 classes) and adapt it to the Oxford-IIIT Pet dataset (37 cat and dog breeds). By completing this lab, you should be able to:

- Explain what a residual connection is and why it helps train deep networks.
- Distinguish a network's **backbone** (feature extractor) from its **classification head**, and replace the head for a new task.
- Implement the three canonical transfer-learning strategies: **train from scratch**, **feature extraction** (frozen backbone), and **fine-tuning** (unfrozen backbone).
- Decide when to freeze, when to fine-tune, and why fine-tuning uses a smaller learning rate.
- Demonstrate, with a data-size ablation, that the benefit of pretraining grows as the dataset shrinks.
- Visualise pretrained features with t-SNE to see *why* transfer learning works.

The expected result is that a pretrained ResNet18, with only its head trained, clearly outperforms a ResNet18 trained from scratch on the same small dataset.

## 2. Prerequisites

You are expected to have completed:

- Lab 1: MNIST classification with MLP and hyperparameter exploration.
- Lab 3: building and training CNNs in PyTorch, including tensor shape tracking and controlled comparisons.
- Andrew Ng's lectures on transfer learning and on the general idea of residual networks.

You should be comfortable with the PyTorch training loop, `nn.Module`, `DataLoader`, and the difference between `requires_grad=True/False`. You do **not** need to have read the ResNet paper (He et al., 2016) beforehand — but you are encouraged to skim Sections 1–3 after finishing Task 4, by which point the motivation will be concrete.

## 3. Software Environment

- Python 3.8 or above
- PyTorch 2.x
- torchvision (with the `resnet18` weights and `OxfordIIITPet` dataset)
- NumPy
- matplotlib
- scikit-learn (for t-SNE)

All experiments should run on CPU. Use a fixed random seed (`torch.manual_seed(42)` and `np.random.seed(42)`) for reproducibility, and reset it before each variant so that all strategies start from the same data split and head initialisation.

Do not introduce additional deep learning frameworks. Load the pretrained weights through torchvision's official API only.

## 4. Background: ResNet18 in One Page

ResNet18 is an 18-layer convolutional network. Its core building block is the **BasicBlock**: two `3×3` convolutions, each followed by BatchNorm and ReLU, plus a **skip connection** that adds the block's input directly to its output:

```
        ┌─────────────────────────────┐
   x ───┤ 3×3 conv → BN → ReLU        │
   │    │ 3×3 conv → BN               ├──┐
   │    └─────────────────────────────┘  │ F(x)
   └────────────────── + ───────────────-┘──→ ReLU → out
            (identity, or 1×1 conv if shape changes)
```

The skip connection lets the block learn a **residual** `F(x) = H(x) - x` rather than the full mapping `H(x)` directly. When the optimal behaviour is "do nothing", the block only needs to drive `F(x)` towards zero, which is easier to optimise than learning an exact identity. The same connection also gives gradient a direct path backwards, which mitigates the vanishing-gradient problem in deep networks.

The full network has the shape:

```
Input 224×224×3
conv1    : 7×7 conv, 64, stride 2 → BN → ReLU → maxpool    → 56×56×64
conv2_x  : 2 × BasicBlock (64 channels,  stride 1)          → 56×56×64
conv3_x  : 2 × BasicBlock (128 channels, stride 2)         → 28×28×128
conv4_x  : 2 × BasicBlock (256 channels, stride 2)         → 14×14×256
conv5_x  : 2 × BasicBlock (512 channels, stride 2)         → 7×7×512
avgpool  : AdaptiveAvgPool2d(1)                              → 1×1×512
fc       : Linear(512, 1000)   ← the classification head     → 1000
```

Counting: `1 (conv1) + 2×2×4 (four stages, two blocks each, two convs per block) = 17 conv layers`, plus the final `fc`, giving 18. The pretrained weights were learned on ImageNet. The final `Linear(512, 1000)` head maps to ImageNet's 1000 classes, so for a 37-class pet task you must replace it with `Linear(512, 37)`.

Two design notes worth understanding before you write code:

- **Global average pooling** (`AdaptiveAvgPool2d(1)`) collapses the `7×7×512` feature map to `1×1×512` before the head. This is why ResNet has a single small `Linear` head instead of the large stacked FC layers you saw in LeNet — it avoids the parameter bloat you diagnosed in Lab 3's MinimalCNN.
- **BatchNorm** sits after every convolution. When you freeze the backbone, you should also keep BN in eval mode (see Task 3's hint) so its running statistics are not updated by your small pet batches.

## 5. Dataset

Use the **Oxford-IIIT Pet Dataset**, available through `torchvision.datasets.OxfordIIITPet`:

- 37 categories of pet breeds (cats and dogs), roughly 200 images per breed
- About 7,400 images total, RGB, variable resolution
- Standard split: roughly 3,680 train / 3,669 test images

Use **all 37 classes**. Resize every image to `224×224` to match ResNet's expected input, and normalise with the **ImageNet statistics** (this is not optional — the pretrained weights expect this distribution):

```python
from torchvision import transforms

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
```

For training and validation, take the standard `trainval` split and divide it into 90% train / 10% val using a fixed-seed `torch.Generator`, so that all three strategies in Task 3 use exactly the same indices. The standard `test` split is reserved for final evaluation only.

State clearly in your report that you used ImageNet normalisation and why.

## 6. Tasks

Complete the tasks in order. Each task should build on the previous one.

### Task 1: Data Preparation and Sanity Check

- Download the dataset via `torchvision.datasets.OxfordIIITPet`. Set `download=True` once; it is roughly 800 MB.
- Build the train / val / test `DataLoader`s as described in Section 5.
- Visualise a small grid of images with their breed names to confirm the labels are correct and the normalisation is sensible (you will need to *un-normalise* before plotting).
- Print the class index ↔ breed name mapping and the number of images per split.

This task does not train anything. It exists so that every later task starts from data you have already verified.

### Task 2: Train ResNet18 From Scratch (Baseline)

Instantiate a ResNet18 with **no** pretrained weights and replace the head:

```python
import torchvision
import torch.nn as nn

def build_resnet(num_classes=37, pretrained=False):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

Train for ~20 epochs with Adam, `lr=1e-3`, `CrossEntropyLoss`, batch size 64. Record:

- Training / validation loss and accuracy curves
- Final test accuracy
- Number of trainable parameters
- Time per epoch

This is your baseline. Expect it to overfit badly and to score low — a randomly-initialised 11M-parameter ResNet on only ~3,300 training images is exactly the regime where transfer learning is supposed to help.

### Task 3: Feature Extraction (Frozen Backbone)

Load the **pretrained** ResNet18, replace the head with a fresh `Linear(512, 37)`, and **freeze** the entire backbone:

```python
model = build_resnet(num_classes=37, pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 37)   # new head: trainable by default
```

Train only the head for ~20 epochs with the same optimiser settings as Task 2.

Important details:

- Put the model in `model.train()` for training so that dropout-style augmentation behaves, but call `model.eval()` *for the frozen backbone's BatchNorm layers*. The cleanest way is to call `model.eval()` after freezing, then `model.fc.train()` so only the head's (stateless) layers behave as training. If you skip this, BN running statistics will be polluted by your tiny pet batches and accuracy will drop.
- Because the backbone is frozen, the penultimate features are a deterministic function of the input. You may precompute them once and cache them to disk, then train the head on cached features — this turns a 20-epoch run into seconds. This precompute is optional but is itself a good teaching moment about why feature extraction is cheap.

Record the same metrics as Task 2 and compare.

### Task 4: Fine-Tuning

Start from the Task 3 checkpoint (pretrained backbone + trained head). **Unfreeze** the whole network and continue training for ~10 epochs, but with a **smaller** learning rate and **differential learning rates** — a smaller rate for the pretrained backbone, a larger rate for the freshly initialised head:

```python
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 1e-4},   # pretrained: small step
    {"params": model.fc.parameters(), "lr": 1e-3},  # fresh head: larger step
])
```

Keep the backbone in `train()` mode now (BN statistics should update, since the whole network is being adapted). Compare:

- frozen-only (Task 3) vs fine-tuned (Task 4)
- Does fine-tuning help? By how much?

### Task 5: Data-Size Ablation

This is the headline experiment. Repeat Task 2 (from scratch) and Task 3 (feature extraction) under **two training-set sizes**:

- 20% of the training data (≈ 660 images)
- 100% of the training data

Plot a single comparison figure: x-axis = fraction of training data (20%, 100%), y-axis = test accuracy, with two lines (scratch, frozen). The expected shape is that the gap between scratch and frozen is **large at 20% and smaller at 100%** — the smaller the dataset, the more pretraining helps. This is the single clearest demonstration of *why* transfer learning is the standard approach when data is scarce.

Use a fixed seed when subsampling so the 20% subset is identical across both strategies.

### Task 6: Feature Visualisation with t-SNE

Using the **frozen pretrained backbone** (no training), extract the 512-dimensional penultimate features (the input to `fc`) for the test set. Reduce them to 2D with `sklearn.manifold.TSNE` and scatter-plot, colouring points by breed.

Produce two plots:

1. **Pixels → t-SNE**: apply t-SNE directly to the flattened raw pixels. Expect a near-random blob.
2. **Pretrained features → t-SNE**: apply t-SNE to the backbone's 512-D features. Expect visible clustering by breed.

The contrast between these two plots is the visual proof that the pretrained backbone has already learned features useful for pet classification — long before you trained a single pet image.

## 7. Suggested Training Settings

- Batch size: 64
- Optimiser: Adam
- Loss: `torch.nn.CrossEntropyLoss` (input is raw logits)
- Learning rates:
  - From scratch and frozen-head: `lr=1e-3`
  - Fine-tuning: `1e-4` for backbone, `1e-3` for head
- Epochs:
  - Task 2 (scratch): ~20
  - Task 3 (frozen): ~20
  - Task 4 (fine-tune): ~10
- Fixed seed `42`, reset before each variant.

If CPU training is slow, reduce epochs first. Do not change multiple factors at once. Each variant must differ from its baseline in exactly one axis (initialisation, frozen/unfrozen, or data size).

Estimated total runtime on a modern multi-core CPU without GPU: roughly 30–60 minutes, dominated by Task 2's from-scratch run. If a single run threatens to exceed roughly one hour, stop and report it.

## 8. Required Outputs

Your final `lab4/` folder should contain:

- `lab4.py` (or one script per task, e.g. `task2_scratch.py`, `task3_frozen.py`, ...)
- Training / validation loss and accuracy curves for the scratch, frozen, and fine-tuned variants
- The data-size ablation comparison figure (Task 5)
- The two t-SNE plots (Task 6)
- A comparison table with columns:
  - Strategy (scratch / frozen / fine-tune)
  - Pretrained? (yes / no)
  - Trainable parameters
  - Final train accuracy
  - Final test accuracy
  - Time per epoch
- A report file, `lab4.typ` (preferred) or `report.md`

## 9. Report Questions

Answer the following in your report:

1. What is a residual connection, and what problem does it solve? Relate it to the vanishing-gradient issue you would face if you simply stacked many convolutions.
2. Distinguish a network's **backbone** from its **classification head**. Why must the head be replaced when adapting an ImageNet-pretrained model to a 37-class pet task?
3. Why does freezing the backbone help on a small dataset? What goes wrong if you fine-tune a pretrained 11M-parameter network on only ~660 images from scratch-style initialisation of the head?
4. Why does fine-tuning use a smaller learning rate for the backbone than for the head? What would likely happen if you used the same large `1e-3` rate for both?
5. From your Task 5 ablation, describe how the gap between from-scratch and pretrained training changes with dataset size. Why is this the central argument for transfer learning?
6. Compare your t-SNE plots. What does the clustering (or lack of it) tell you about what the pretrained backbone has learned?
7. Connect this lab to Lab 3: in Lab 3, the inductive bias (locality, weight sharing) was *built into the architecture*. In Lab 4, where does the inductive bias come from? Are the two kinds of inductive bias complementary or competing?

## 10. Evaluation Criteria

- Correct use of torchvision's pretrained ResNet18 API and correct replacement of the classification head.
- Correct freezing: backbone parameters truly not updated, BatchNorm handled properly.
- Fair comparison: identical data splits, identical seeds, only one axis changed per experiment.
- Clear data-size ablation showing the expected trend.
- Meaningful t-SNE visualisation with an honest comparison to raw-pixel t-SNE.
- Thoughtful analysis of when to freeze vs fine-tune, and of why fine-tuning uses differential learning rates.
- Reproducibility through fixed seeds and documented hyperparameters.

## 11. Hints

- Count trainable parameters with:
  ```python
  def count_parameters(model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)
  ```
- After freezing, the trainable parameter count of a frozen ResNet18 should drop from ~11.7M to roughly 19k (just the new `Linear(512, 37)` head: `512×37 + 37 = 18,981`). If it does not, your freeze did not take effect.
- Debug shapes with a dummy batch:
  ```python
  x = torch.randn(4, 3, 224, 224)
  print(model(x).shape)   # expect (4, 37)
  ```
- For t-SNE, subsample to ~1500 points if it is slow, and use `init="pca"` and `perplexity=30` as reasonable defaults.
- To extract penultimate features without running the head:
  ```python
  model.fc = nn.Identity()
  features = model(x)   # (batch, 512)
  ```
  Restore the head afterwards if you need logits.
- The pretrained weights download to `~/.cache/torch/hub/checkpoints/` by default. Once downloaded, subsequent runs are offline.
- If `OxfordIIITPet` download is slow or fails behind a proxy, you can download the archive manually and place it where torchvision expects it; check the torchvision source for the exact URL.
