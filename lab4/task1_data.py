"""
Task 1: Data Preparation and Sanity Check
=========================================

Goal
----
Download the Oxford-IIIT Pet dataset and verify, before any training, that:

1. The dataset loads correctly and has the expected number of images per split.
2. The class index <-> breed name mapping is what we think it is.
3. Images and labels line up (a picture of a Samoyed really is labelled
   "Samoyed") and the ImageNet normalisation looks reasonable visually.

This task trains nothing. It exists so every later task starts from data that
has already been eyeballed. Catching a label/normalisation bug here is cheap;
catching it after a 15-minute training run is not.

Run
---
    python task1_data.py

It will download ~800 MB into lab4/data/ on the first run (torchvision handles
the download and checksum). Subsequent runs are offline.

Outputs
-------
- results/task1_sample_grid.png   : a grid of images with breed titles
- results/task1_class_distribution.png : bar chart of per-class image counts
- Console: split sizes and the class mapping
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from collections import Counter

from common import (
    set_seed, ensure_dirs, get_dataloaders, get_class_names, get_species,
    unnormalize, RESULTS_DIR, NUM_CLASSES, SEED,
)


def show_sample_grid(train_loader, class_names, species, save_path) -> None:
    """Plot one batch of training images with breed titles.

    We un-normalise before plotting, otherwise the ImageNet-normalised tensors
    would look like nonsense (shifted/scaled RGB). Reading the titles against
    the pictures is the human sanity check: if a 'Samoyed' title sits on a
    white fluffy dog, labels and transforms are wired correctly.
    """
    images, labels = next(iter(train_loader))
    batch = images.shape[0]
    n_show = min(16, batch)
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    imgs = unnormalize(images[:n_show])
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            ax.imshow(imgs[i])
            cls = int(labels[i])
            breed = class_names[cls] if cls < len(class_names) else str(cls)
            sp = species[cls] if cls < len(species) else ""
            ax.set_title(f"{breed}\n[{sp}]", fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(trainval_dataset, test_loader, save_path) -> None:
    """Bar chart of per-class image counts in trainval vs test.

    Oxford-IIIT Pet is roughly balanced (~200 images per breed), so a healthy
    bar chart should show 37 bars of similar height. A wildly uneven chart
    would suggest a parsing problem rather than the real dataset.
    """
    trainval_counts = Counter(int(t) for _, t in trainval_dataset)
    test_counts = Counter(int(t) for t in test_loader.dataset.targets) \
        if hasattr(test_loader.dataset, "targets") else \
        Counter(int(t) for _, t in test_loader.dataset)

    classes = list(range(NUM_CLASSES))
    tv_vals = [trainval_counts[c] for c in classes]
    te_vals = [test_counts[c] for c in classes]

    import numpy as np
    x = np.arange(NUM_CLASSES)
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - width / 2, tv_vals, width, label="trainval")
    ax.bar(x + width / 2, te_vals, width, label="test")
    ax.set_xlabel("Class index")
    ax.set_ylabel("Number of images")
    ax.set_title("Oxford-IIIT Pet — per-class image counts")
    ax.set_xticks(x[::2])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()

    print("=" * 60)
    print("Task 1: data preparation and sanity check")
    print("=" * 60)

    # Build loaders. download=True is set inside get_dataloaders, so the first
    # run will fetch ~800 MB; later runs reuse the cache.
    train_loader, val_loader, test_loader, trainval_full = get_dataloaders()

    print(f"\nSplits:")
    print(f"  trainval (full) : {len(trainval_full)}")
    print(f"  train (this run) : {len(train_loader.dataset)}")
    print(f"  val   (this run) : {len(val_loader.dataset)}")
    print(f"  test            : {len(test_loader.dataset)}")

    class_names = get_class_names()
    species = get_species()
    print(f"\nNumber of classes: {len(class_names)}")
    print("\nClass index -> breed name (first 10):")
    for i in range(min(10, len(class_names))):
        breed = class_names[i]
        sp = species[i] if i < len(species) else "?"
        print(f"  {i:>2}: {breed:<28} [{sp}]")
    print(f"  ... ({len(class_names)} total)")

    # --- Visual sanity checks ---
    sample_grid_path = RESULTS_DIR / "task1_sample_grid.png"
    show_sample_grid(train_loader, class_names, species, sample_grid_path)
    print(f"\nSaved sample grid       : {sample_grid_path}")

    dist_path = RESULTS_DIR / "task1_class_distribution.png"
    plot_class_distribution(trainval_full, test_loader, dist_path)
    print(f"Saved class distribution: {dist_path}")

    print("\nTask 1 done. Inspect the two PNGs in results/ before training.")
    print("Things to confirm:")
    print("  - Each image's title matches what the picture actually shows.")
    print("  - trainval and test bars are roughly equal (~200 per class).")


if __name__ == "__main__":
    main()
