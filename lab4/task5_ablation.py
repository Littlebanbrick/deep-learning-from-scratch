"""
Task 5: Data-Size Ablation
==========================

Goal
----
Show that transfer learning becomes more valuable when labelled data is scarce.

We compare two strategies:

1. ResNet18 trained from scratch.
2. ImageNet-pretrained ResNet18 with a frozen backbone and a trainable head.

at two training-set sizes:

- 20% of the training split
- 100% of the training split

Important runtime note
----------------------
The 100% runs have already been completed in Task 2 and Task 3. Repeating them
would waste a long CPU run and could introduce slightly different results. This
script therefore trains only the two new 20% variants, then reads:

- results/task2_summary.json  -> scratch, 100%
- results/task3_summary.json  -> frozen, 100%

and combines all four points into one ablation figure.

Run
---
    python -u task5_ablation.py

Outputs
-------
- results/task5_ablation_curves_scratch20.png
- results/task5_ablation_curves_frozen20.png
- results/task5_data_size_ablation.png
- results/task5_summary.json
- checkpoints/task5_scratch_20.pt
- checkpoints/task5_frozen_20.pt
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    set_seed, ensure_dirs, get_dataloaders, build_resnet, count_parameters,
    evaluate, get_device, plot_curves,
    RESULTS_DIR, CHECKPOINT_DIR, SEED, NUM_CLASSES, BATCH_SIZE,
)


SUBSET_FRACTION = 0.2
EPOCHS = 20
LR = 1e-3


def train_scratch(model, train_loader, val_loader, epochs, lr, device):
    """Train a randomly initialised model end to end."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "epoch_time": []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(dt)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | {dt:.1f}s")

    return history


def train_frozen(model, train_loader, val_loader, epochs, lr, device):
    """Train only the fresh head while keeping the pretrained backbone frozen.

    BatchNorm in the frozen backbone stays in eval mode; only the head is put
    into train mode. This matches Task 3 exactly.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "epoch_time": []}

    for epoch in range(epochs):
        t0 = time.time()
        model.eval()
        model.fc.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(dt)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | {dt:.1f}s")

    return history


def load_existing_summary(path):
    """Load a previous task summary and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required summary not found: {path}\n"
            "Run Task 2 and Task 3 first, because Task 5 reuses their 100% results."
        )
    with open(path) as f:
        return json.load(f)


def plot_ablation(summary, save_path):
    """Plot test accuracy against training-data fraction."""
    fractions = [20, 100]
    scratch = [
        summary["scratch_20"]["final_test_acc"],
        summary["scratch_100"]["final_test_acc"],
    ]
    frozen = [
        summary["frozen_20"]["final_test_acc"],
        summary["frozen_100"]["final_test_acc"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(fractions, scratch, "o-", label="Scratch ResNet18", color="#d62728")
    ax.plot(fractions, frozen, "o-", label="Frozen pretrained ResNet18", color="#2ca02c")
    ax.set_xticks(fractions)
    ax.set_xticklabels(["20%", "100%"])
    ax.set_xlabel("Fraction of training data")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Task 5: Data-size ablation")
    ax.grid(alpha=0.3)
    ax.legend()

    for xs, ys in [(fractions, scratch), (fractions, frozen)]:
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

    gap_20 = frozen[0] - scratch[0]
    gap_100 = frozen[1] - scratch[1]
    ax.text(0.02, 0.03,
            f"pretraining gap: 20% = {gap_20:.3f}, 100% = {gap_100:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9))

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    device = get_device()

    print("=" * 64)
    print("Task 5: Data-Size Ablation")
    print("=" * 64)
    print(f"Device          : {device}")
    print(f"Subset fraction : {SUBSET_FRACTION:.0%}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Optimizer       : Adam, lr={LR}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Num classes     : {NUM_CLASSES}")

    task2_100 = load_existing_summary(RESULTS_DIR / "task2_summary.json")
    task3_100 = load_existing_summary(RESULTS_DIR / "task3_summary.json")
    print("\nReusing existing 100% results:")
    print(f"  scratch 100% test acc: {task2_100['final_test_acc']:.4f}")
    print(f"  frozen  100% test acc: {task3_100['final_test_acc']:.4f}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        subset_fraction=SUBSET_FRACTION,
        seed=SEED,
    )
    print(f"\n20% Train / Val / Test : {len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / {len(test_loader.dataset)}")

    # ------------------------------------------------------------------
    # 20% scratch
    # ------------------------------------------------------------------
    print("\n" + "-" * 64)
    print("Ablation A: scratch ResNet18 on 20% training data")
    print("-" * 64)
    set_seed(SEED)
    scratch = build_resnet(num_classes=NUM_CLASSES, pretrained=False, freeze=False)
    scratch.to(device)
    print(f"Trainable params: {count_parameters(scratch):,}")
    scratch_hist = train_scratch(scratch, train_loader, val_loader, EPOCHS, LR, device)
    scratch_loss, scratch_acc = evaluate(scratch, test_loader, device)
    print(f"Scratch 20% TEST accuracy: {scratch_acc:.4f}")
    print(f"Scratch 20% TEST loss    : {scratch_loss:.4f}")
    plot_curves(scratch_hist, RESULTS_DIR / "task5_ablation_curves_scratch20.png",
                "Task 5: Scratch ResNet18, 20% data")
    torch.save(scratch.state_dict(), CHECKPOINT_DIR / "task5_scratch_20.pt")

    # ------------------------------------------------------------------
    # 20% frozen pretrained
    # ------------------------------------------------------------------
    print("\n" + "-" * 64)
    print("Ablation B: frozen pretrained ResNet18 on 20% training data")
    print("-" * 64)
    set_seed(SEED)
    frozen = build_resnet(num_classes=NUM_CLASSES, pretrained=True, freeze=True)
    frozen.to(device)
    print(f"Trainable params: {count_parameters(frozen):,}")
    frozen_hist = train_frozen(frozen, train_loader, val_loader, EPOCHS, LR, device)
    frozen_loss, frozen_acc = evaluate(frozen, test_loader, device)
    print(f"Frozen 20% TEST accuracy: {frozen_acc:.4f}")
    print(f"Frozen 20% TEST loss    : {frozen_loss:.4f}")
    plot_curves(frozen_hist, RESULTS_DIR / "task5_ablation_curves_frozen20.png",
                "Task 5: Frozen pretrained ResNet18, 20% data")
    torch.save(frozen.state_dict(), CHECKPOINT_DIR / "task5_frozen_20.pt")

    summary = {
        "task": "task5_data_size_ablation",
        "subset_fraction": SUBSET_FRACTION,
        "epochs_for_20_percent_runs": EPOCHS,
        "scratch_20": {
            "strategy": "from_scratch",
            "pretrained": False,
            "subset_fraction": SUBSET_FRACTION,
            "trainable_params": count_parameters(scratch),
            "final_train_acc": scratch_hist["train_acc"][-1],
            "final_val_acc": scratch_hist["val_acc"][-1],
            "final_test_acc": scratch_acc,
            "final_test_loss": scratch_loss,
            "avg_epoch_time_s": sum(scratch_hist["epoch_time"]) / len(scratch_hist["epoch_time"]),
            "history": scratch_hist,
        },
        "frozen_20": {
            "strategy": "feature_extraction",
            "pretrained": True,
            "frozen": True,
            "subset_fraction": SUBSET_FRACTION,
            "trainable_params": count_parameters(frozen),
            "final_train_acc": frozen_hist["train_acc"][-1],
            "final_val_acc": frozen_hist["val_acc"][-1],
            "final_test_acc": frozen_acc,
            "final_test_loss": frozen_loss,
            "avg_epoch_time_s": sum(frozen_hist["epoch_time"]) / len(frozen_hist["epoch_time"]),
            "history": frozen_hist,
        },
        "scratch_100": {
            "source": "task2_summary.json",
            "strategy": task2_100["strategy"],
            "pretrained": task2_100["pretrained"],
            "subset_fraction": 1.0,
            "trainable_params": task2_100["trainable_params"],
            "final_train_acc": task2_100["final_train_acc"],
            "final_val_acc": task2_100["final_val_acc"],
            "final_test_acc": task2_100["final_test_acc"],
            "final_test_loss": task2_100["final_test_loss"],
            "avg_epoch_time_s": task2_100["avg_epoch_time_s"],
        },
        "frozen_100": {
            "source": "task3_summary.json",
            "strategy": task3_100["strategy"],
            "pretrained": task3_100["pretrained"],
            "frozen": task3_100["frozen"],
            "subset_fraction": 1.0,
            "trainable_params": task3_100["trainable_params"],
            "final_train_acc": task3_100["final_train_acc"],
            "final_val_acc": task3_100["final_val_acc"],
            "final_test_acc": task3_100["final_test_acc"],
            "final_test_loss": task3_100["final_test_loss"],
            "avg_epoch_time_s": task3_100["avg_epoch_time_s"],
        },
    }

    summary_path = RESULTS_DIR / "task5_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    fig_path = RESULTS_DIR / "task5_data_size_ablation.png"
    plot_ablation(summary, fig_path)
    print(f"Saved ablation figure: {fig_path}")

    gap_20 = summary["frozen_20"]["final_test_acc"] - summary["scratch_20"]["final_test_acc"]
    gap_100 = summary["frozen_100"]["final_test_acc"] - summary["scratch_100"]["final_test_acc"]
    print("\nPretraining gap:")
    print(f"  20% data : {gap_20:.4f}")
    print(f"  100% data: {gap_100:.4f}")
    print("\nTask 5 done. The expected story is: the low-data gap should be largest.")


if __name__ == "__main__":
    main()
