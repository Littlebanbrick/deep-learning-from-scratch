"""
Task 6: Feature Visualisation with t-SNE
========================================

Goal
----
Visualise why transfer learning works.

We compare two representations of the same Oxford-IIIT Pet test images:

1. Raw pixels: resize -> tensor -> unnormalize -> flatten.
2. ImageNet-pretrained ResNet18 features: image -> frozen backbone -> 512-D.

Both representations are projected to 2D using t-SNE and plotted with the same
class labels. If the pretrained backbone has learned useful visual structure,
the feature t-SNE should show clearer class/species clustering than the raw
pixel t-SNE.

This script does NOT train anything.

Run
---
    python -u task6_tsne.py

Outputs
-------
- results/task6_tsne_pixels.png
- results/task6_tsne_pretrained_features.png
- results/task6_tsne_side_by_side.png
- results/task6_tsne_summary.json
- results/task6_tsne_data.npz
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.manifold import TSNE

from common import (
    set_seed, ensure_dirs, get_dataloaders, get_class_names, get_species,
    unnormalize, get_device, RESULTS_DIR, SEED, NUM_CLASSES,
)


# t-SNE is quadratic in the number of samples. 1000 is enough to see structure
# while staying practical on CPU.
MAX_SAMPLES = 1000
PERPLEXITY = 30


def build_feature_extractor(device: torch.device) -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and return a 512-D feature extractor.

    Replacing `fc` with Identity makes the model output the penultimate
    512-dimensional vector after global average pooling instead of ImageNet
    logits. This is the exact feature representation used by the frozen-head
    transfer-learning strategy.
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def collect_representations(loader, feature_extractor, device, max_samples):
    """Collect raw-pixel vectors, pretrained features, labels, and species ids."""
    pixel_vectors = []
    feature_vectors = []
    labels = []

    seen = 0
    with torch.no_grad():
        for images, y in loader:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            if images.size(0) > remaining:
                images = images[:remaining]
                y = y[:remaining]

            images = images.to(device)

            # Raw-pixel baseline: undo ImageNet normalization for a natural
            # pixel-space representation in [0, 1], then flatten.
            raw = unnormalize(images.cpu())
            raw = raw.reshape(raw.shape[0], -1).numpy()

            # Pretrained feature representation: 512-D vector before fc.
            feats = feature_extractor(images).cpu().numpy()

            pixel_vectors.append(raw)
            feature_vectors.append(feats)
            labels.append(y.numpy())
            seen += images.size(0)

    return (
        np.concatenate(pixel_vectors, axis=0),
        np.concatenate(feature_vectors, axis=0),
        np.concatenate(labels, axis=0),
    )


def run_tsne(x: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Run t-SNE with parameters compatible across scikit-learn versions."""
    kwargs = dict(
        n_components=2,
        perplexity=min(PERPLEXITY, max(5, (len(x) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        tsne = TSNE(max_iter=1000, **kwargs)
    except TypeError:
        # Older scikit-learn used `n_iter` instead of `max_iter`.
        tsne = TSNE(n_iter=1000, **kwargs)
    return tsne.fit_transform(x)


def plot_tsne(points, labels, species_by_class, class_names, save_path, title):
    """Scatter t-SNE points using color for class and marker for cat/dog."""
    labels = labels.astype(int)
    species_labels = np.array([species_by_class[int(y)] for y in labels])

    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    markers = {"cat": "o", "dog": "^", "": "o"}

    fig, ax = plt.subplots(figsize=(9, 7))
    for sp in ["cat", "dog", ""]:
        idx = species_labels == sp
        if not np.any(idx):
            continue
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            c=labels[idx],
            cmap=cmap,
            vmin=0,
            vmax=NUM_CLASSES - 1,
            s=18,
            alpha=0.75,
            marker=markers[sp],
            linewidths=0,
            label=sp if sp else "unknown",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    ax.legend(title="species", loc="best")

    cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("breed class index")

    # Add a compact class-name note next to the figure instead of a huge legend
    # with 37 entries over the scatter plot.
    note = "\n".join(f"{i}: {name}" for i, name in enumerate(class_names[:12]))
    note += "\n..."
    ax.text(
        1.02, 0.02, note,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(pixel_points, feature_points, labels, species_by_class,
                      save_path):
    """Put raw pixels and pretrained features in one comparison figure."""
    labels = labels.astype(int)
    species_labels = np.array([species_by_class[int(y)] for y in labels])
    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    markers = {"cat": "o", "dog": "^", "": "o"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, points, title in [
        (axes[0], pixel_points, "Raw pixels -> t-SNE"),
        (axes[1], feature_points, "Pretrained ResNet18 features -> t-SNE"),
    ]:
        for sp in ["cat", "dog", ""]:
            idx = species_labels == sp
            if not np.any(idx):
                continue
            ax.scatter(
                points[idx, 0],
                points[idx, 1],
                c=labels[idx],
                cmap=cmap,
                vmin=0,
                vmax=NUM_CLASSES - 1,
                s=16,
                alpha=0.75,
                marker=markers[sp],
                linewidths=0,
                label=sp if sp else "unknown",
            )
        ax.set_title(title)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(alpha=0.2)

    axes[1].legend(title="species", loc="best")
    fig.suptitle("Task 6: Pixel space vs pretrained feature space", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    device = get_device()

    print("=" * 64)
    print("Task 6: Feature Visualisation with t-SNE")
    print("=" * 64)
    print(f"Device      : {device}")
    print(f"Max samples : {MAX_SAMPLES}")
    print(f"Perplexity  : {PERPLEXITY}")

    _, _, test_loader, _ = get_dataloaders()
    class_names = get_class_names()
    species_by_class = get_species()

    print("\nLoading ImageNet-pretrained ResNet18 feature extractor ...")
    feature_extractor = build_feature_extractor(device)

    print("Collecting raw pixels and pretrained features from the test set ...")
    pixels, features, labels = collect_representations(
        test_loader,
        feature_extractor,
        device,
        max_samples=MAX_SAMPLES,
    )
    print(f"Collected samples       : {len(labels)}")
    print(f"Raw pixel vector shape  : {pixels.shape}")
    print(f"Feature vector shape    : {features.shape}")

    print("\nRunning t-SNE on raw pixels ...")
    pixel_points = run_tsne(pixels)
    print("Running t-SNE on pretrained features ...")
    feature_points = run_tsne(features)

    pixel_path = RESULTS_DIR / "task6_tsne_pixels.png"
    feature_path = RESULTS_DIR / "task6_tsne_pretrained_features.png"
    side_path = RESULTS_DIR / "task6_tsne_side_by_side.png"
    data_path = RESULTS_DIR / "task6_tsne_data.npz"
    summary_path = RESULTS_DIR / "task6_tsne_summary.json"

    plot_tsne(
        pixel_points,
        labels,
        species_by_class,
        class_names,
        pixel_path,
        "Task 6A: Raw pixels -> t-SNE",
    )
    plot_tsne(
        feature_points,
        labels,
        species_by_class,
        class_names,
        feature_path,
        "Task 6B: Pretrained ResNet18 features -> t-SNE",
    )
    plot_side_by_side(pixel_points, feature_points, labels, species_by_class, side_path)

    np.savez_compressed(
        data_path,
        pixel_tsne=pixel_points,
        feature_tsne=feature_points,
        labels=labels,
        pixels=pixels,
        features=features,
    )

    summary = {
        "task": "task6_tsne",
        "max_samples": MAX_SAMPLES,
        "num_samples_used": int(len(labels)),
        "pixel_dim": int(pixels.shape[1]),
        "feature_dim": int(features.shape[1]),
        "perplexity": min(PERPLEXITY, max(5, (len(labels) - 1) // 3)),
        "outputs": {
            "pixels": str(pixel_path),
            "pretrained_features": str(feature_path),
            "side_by_side": str(side_path),
            "data": str(data_path),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"  {pixel_path}")
    print(f"  {feature_path}")
    print(f"  {side_path}")
    print(f"  {data_path}")
    print(f"  {summary_path}")
    print("\nTask 6 done. Compare whether pretrained features cluster more clearly.")


if __name__ == "__main__":
    main()
