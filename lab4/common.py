"""
Common utilities for Lab 4: transfer learning with pretrained ResNet18 on
Oxford-IIIT Pet.

This file is a thin layer of plumbing shared by every task script. It holds
NO interesting logic — only the deterministic scaffolding (seeds, dataloaders,
model construction, parameter counting, plotting) so each task script reads
as a self-contained recipe.

Design notes
------------
- All task scripts import from here. Changing a constant in one place
  propagates everywhere, which keeps the three transfer-learning strategies
  genuinely comparable.
- ImageNet normalisation is mandatory because the pretrained weights were
  learned under that input distribution. Using pet-specific statistics would
  silently break feature extraction.
- The train/val split uses a fixed seed so that scratch / frozen / fine-tune
  see the *same* images; otherwise comparisons are contaminated.
- `build_resnet` returns the model in a clean state: head replaced, backbone
  frozen or not depending on `freeze`. The caller decides train/eval mode.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision


# ============================================================
# Global constants — the experiment contract
# ============================================================

SEED = 42
NUM_CLASSES = 37
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4

# Where the dataset is cached and where outputs land. Resolved relative to the
# lab4/ directory regardless of which task script is run.
LAB_DIR = Path(__file__).resolve().parent
DATA_ROOT = LAB_DIR / "data"
RESULTS_DIR = LAB_DIR / "results"
CHECKPOINT_DIR = LAB_DIR / "checkpoints"

# ImageNet statistics — the pretrained ResNet18 expects inputs drawn from this
# distribution. Do NOT replace with pet-specific stats while using pretrained
# weights, or feature extraction will silently degrade.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU).

    Call this at the top of every task script, and again before each model
    variant, so that all strategies start from the same initial weights and
    the same data split. This is what makes a 'controlled comparison'.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # some torchvision ops warn otherwise


# ============================================================
# Transforms and data
# ============================================================

def get_transforms(train: bool = True) -> transforms.Compose:
    """Build the image transform pipeline.

    Eval transform: resize, to-tensor, normalise.
    Train transform: same plus a horizontal flip. We deliberately keep
    augmentation light — the point of the lab is transfer learning, not
    augmentation strength. Heavy augmentation would muddy the comparison.
    """
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.1,
    subset_fraction: float = 1.0,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, datasets.OxfordIIITPet]:
    """Build the train / val / test dataloaders.

    Parameters
    ----------
    val_split : float
        Fraction of the official `trainval` split held out for validation.
        The split uses a fixed-seed generator so every task sees the same
        indices.
    subset_fraction : float
        If < 1.0, keep only this fraction of the training set (used by the
        data-size ablation in Task 5). The subset is drawn deterministically
        so the 20% subset is identical across strategies.
    seed : int
        Seed for both the train/val split and the subset choice.

    Returns
    -------
    (train_loader, val_loader, test_loader, trainval_dataset)
        The dataset object is returned so callers can recover class names.
    """
    # Train/val come from the same trainval split, with a train-specific transform.
    # We build two dataset objects (train and eval transforms) over the same
    # underlying split, then take complementary subsets of the same index list.
    g_split = torch.Generator().manual_seed(seed)
    trainval_full = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        transform=get_transforms(train=True), download=True,
    )
    trainval_eval = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        transform=get_transforms(train=False), download=True,
    )

    n = len(trainval_full)
    n_val = int(n * val_split)
    perm = torch.randperm(n, generator=g_split).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Optionally subsample the training set (Task 5). Deterministic.
    if subset_fraction < 1.0:
        g_sub = torch.Generator().manual_seed(seed + 1)
        k = int(len(train_idx) * subset_fraction)
        sub_perm = torch.randperm(len(train_idx), generator=g_sub).tolist()[:k]
        train_idx = [train_idx[i] for i in sub_perm]

    train_set = Subset(trainval_full, train_idx)
    val_set = Subset(trainval_eval, val_idx)

    test_set = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="test", target_types="category",
        transform=get_transforms(train=False), download=True,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)
    return train_loader, val_loader, test_loader, trainval_full


def get_class_names() -> list[str]:
    """Return the 37 breed names in category-index order.

    OxfordIIITPet exposes the list as `dataset.classes` (e.g.
    'Abyssinian', 'American Bulldog', ..., 'Yorkshire Terrier'). The species
    (cat/dog) is stored separately in `bin_class_to_idx` ({'Cat': 0, 'Dog': 1}),
    which we read in get_species(); the breed names themselves carry no suffix.
    """
    ds = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        transform=None, download=False,
    )
    names = getattr(ds, "classes", None)
    if names is None:
        # Defensive fallback if a future torchvision version renames the attr.
        names = [str(i) for i in range(NUM_CLASSES)]
    return list(names)


def get_species() -> list[str]:
    """Return 'cat'/'dog' for each of the 37 class indices.

    Uses the dataset's `bin_class_to_idx` ({'Cat': 0, 'Dog': 1}) together with
    `_bin_labels` (a per-sample binary label) to map each *class index* to its
    species. We pick the species of the first sample belonging to each class.
    """
    ds = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        transform=None, download=False,
    )
    bin_map = getattr(ds, "bin_class_to_idx", {"Cat": 0, "Dog": 1})
    inv = {v: k.lower() for k, v in bin_map.items()}  # {0: 'cat', 1: 'dog'}
    labels = getattr(ds, "_labels", [])
    bin_labels = getattr(ds, "_bin_labels", [])
    species_by_class: dict[int, str] = {}
    for cls, binlab in zip(labels, bin_labels):
        if cls not in species_by_class:
            species_by_class[cls] = inv.get(int(binlab), "")
        if len(species_by_class) >= NUM_CLASSES:
            break
    return [species_by_class.get(i, "") for i in range(NUM_CLASSES)]


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalisation for plotting.

    Takes a (..., 3, H, W) tensor in normalised space and returns a (..., H, W, 3)
    tensor with values clamped to [0, 1] for imshow.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t = tensor.cpu() * std + mean
    t = t.clamp(0, 1)
    # (N, 3, H, W) -> (N, H, W, 3) for matplotlib
    if t.dim() == 4:
        t = t.permute(0, 2, 3, 1)
    elif t.dim() == 3:
        t = t.permute(1, 2, 0)
    return t


# ============================================================
# Model construction
# ============================================================

def build_resnet(num_classes: int = NUM_CLASSES, pretrained: bool = False,
                 freeze: bool = False) -> nn.Module:
    """Build a ResNet18 with a fresh classification head.

    Parameters
    ----------
    pretrained : bool
        If True, load ImageNet-pretrained backbone weights. If False, random
        init (Task 2: train from scratch).
    freeze : bool
        If True, freeze the entire backbone so only the new head trains
        (Task 3: feature extraction). Has no effect when pretrained=False.

    Returns
    -------
    The model, with `fc` replaced by Linear(512, num_classes). The caller
    is responsible for moving it to a device and setting train/eval mode
    appropriately (see Task 3 for the BatchNorm-in-eval subtlety when frozen).
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the head. When freeze=True the new head is the only trainable
    # part, and requires_grad defaults to True for fresh Linear layers.
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)
    return model


def split_params_by_lr(model: nn.Module) -> list[dict]:
    """Group parameters for differential learning rates (Task 4).

    Returns a param-group list suitable for torch.optim.Adam:
        - backbone (pretrained) -> lr 1e-4
        - head      (fresh)     -> lr 1e-3
    The two rates reflect that the pretrained backbone is already good and
    only needs gentle nudging, while the freshly-initialised head needs to
    learn quickly from scratch.
    """
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": 1e-4})
    if head_params:
        groups.append({"params": head_params, "lr": 1e-3})
    return groups


# ============================================================
# Small training utilities
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters — the standard sanity check that freezing
    actually took effect. A frozen ResNet18 should report ~19k, not ~11.7M."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Compute average loss and accuracy over a loader. No grads, model.eval()."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, total_correct / total


def get_device() -> torch.device:
    """CPU-only lab. Wrapped in a function so the message is in one place."""
    if torch.cuda.is_available():
        # The lab is CPU-targeted, but we won't refuse a GPU if one appears.
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs() -> None:
    """Create the output directories if missing. Called by each task script."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plotting
# ============================================================

def plot_curves(history: dict, save_path: Path, title: str) -> None:
    """Plot train/val loss and accuracy from a history dict.

    `history` keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc',
    each a list of length = number of epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "o-", label="train", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], "o-", label="val", color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="train", color="#1f77b4")
    axes[1].plot(epochs, history["val_acc"], "o-", label="val", color="#d62728")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(histories: list[dict], labels: list[str], save_path: Path,
                 title: str, metric: str = "val_acc") -> None:
    """Overlay one curve per experiment for cross-strategy comparison.

    Used by Task 5 (data-size ablation) and the summary plot. Only the
    validation curve is drawn — cross-strategy comparisons care about
    generalisation, not training fit.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for hist, lab in zip(histories, labels):
        epochs = range(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], "o-", label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy" if metric.endswith("acc") else "Validation Loss")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CLI smoke test: `python common.py` prints the contract
# ============================================================

if __name__ == "__main__":
    set_seed()
    ensure_dirs()
    print(f"Lab dir         : {LAB_DIR}")
    print(f"Data root       : {DATA_ROOT}")
    print(f"Results dir      : {RESULTS_DIR}")
    print(f"Num classes      : {NUM_CLASSES}")
    print(f"Image size       : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Device           : {get_device()}")
    print()

    # Sanity: build both variants, print trainable param counts. This is the
    # one-line proof that freezing works and that the head has the right size.
    scratch = build_resnet(pretrained=False, freeze=False)
    frozen = build_resnet(pretrained=True, freeze=True)
    ft = build_resnet(pretrained=True, freeze=False)
    print("Parameter counts (trainable):")
    print(f"  scratch (random init, all trainable) : {count_parameters(scratch):>10,}")
    print(f"  frozen  (pretrained, head only)      : {count_parameters(frozen):>10,}")
    print(f"  finetune (pretrained, all trainable) : {count_parameters(ft):>10,}")
    print()
    print("Expected: frozen ≈ 18,981 = 512*37 + 37.")
    print("If frozen reports ~11.7M, freezing did not take effect.")

    # Parameter-group check for fine-tuning (differential LR).
    groups = split_params_by_lr(ft)
    print()
    print("Fine-tuning parameter groups:")
    for i, g in enumerate(groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  group {i}: lr={g['lr']:.0e}  params={n:,}")
