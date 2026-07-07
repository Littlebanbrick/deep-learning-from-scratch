"""
Task 2: Train ResNet18 From Scratch (Baseline)
===============================================

Goal
----
Train a ResNet18 with **randomly initialised** weights on the Oxford-IIIT Pet
dataset. This is the control / baseline for the whole lab: the "no transfer
learning" case. It is supposed to look bad.

Why it should look bad
---------------------
The same 11.2M-parameter architecture that ImageNet trained on 1.28M images is
here asked to learn from ~3,300 pet images. The model has more than enough
capacity to *memorise* the training set, but almost nothing to *generalise*
from. Expected symptoms:
  - training loss drops, training accuracy climbs high (memorisation),
  - validation accuracy stays low (~10-30%, barely above the 1/37 ≈ 2.7% random
    baseline for the first few epochs),
  - a large train/val gap — the textbook overfitting signature.

This is the "effort for no reward" case that Task 3's pretrained run is meant
to dramatically beat, using the *same* architecture and the *same* data, with
only the weight initialisation changed.

Run
---
    python -u task2_scratch.py

Expect roughly 10-15 minutes on CPU (20 epochs × ~30-45 s/epoch). All
progress prints to stdout, so you can redirect with `> results/task2_log.txt`.

Outputs
-------
- results/task2_scratch_curves.png  : train/val loss & accuracy curves
- results/task2_summary.json        : final metrics (for the report summary table)
- checkpoints/task2_scratch.pt      : final model state (kept for completeness)
"""

from __future__ import annotations

import json
import os
import time

# Force a headless matplotlib backend before anything imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    set_seed, ensure_dirs, get_dataloaders, build_resnet, count_parameters,
    evaluate, get_device, plot_curves, RESULTS_DIR, CHECKPOINT_DIR, SEED,
    NUM_CLASSES, BATCH_SIZE,
)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_scratch(model, train_loader, val_loader, epochs, lr, device):
    """Plain training loop: Adam, CrossEntropy on raw logits, full-batch eval
    at the end of each epoch.

    Returns a history dict with keys matching common.plot_curves:
    'train_loss', 'val_loss', 'train_acc', 'val_acc' (each a list), plus
    'epoch_time' (per-epoch seconds) for the summary.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [],  "val_acc":  [], "epoch_time": []}

    for epoch in range(epochs):
        t0 = time.time()

        # --- train ---
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

        # --- eval ---
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

EPOCHS = 20
LR = 1e-3


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    device = get_device()

    print("=" * 64)
    print("Task 2: Train ResNet18 FROM SCRATCH (no pretrained weights)")
    print("=" * 64)
    print(f"Device        : {device}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Optimizer     : Adam, lr={LR}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Num classes   : {NUM_CLASSES}")

    train_loader, val_loader, test_loader, _ = get_dataloaders()
    print(f"Train / Val / Test : {len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / {len(test_loader.dataset)}")

    # Fresh ResNet18, random init, head replaced for 37 classes.
    model = build_resnet(num_classes=NUM_CLASSES, pretrained=False, freeze=False)
    model.to(device)
    print(f"Trainable params : {count_parameters(model):,}  "
          f"(expected ~11,195,493 = full ResNet18 + new head)")

    print("\nTraining ...")
    t_total = time.time()
    history = train_scratch(model, train_loader, val_loader, EPOCHS, LR, device)
    print(f"\nTotal training time: {(time.time()-t_total)/60:.1f} min")
    print(f"Avg epoch time     : {sum(history['epoch_time'])/len(history['epoch_time']):.1f}s")

    # --- Final test evaluation ---
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal TEST accuracy : {test_acc:.4f}")
    print(f"Final TEST loss     : {test_loss:.4f}")

    # --- Save artefacts ---
    curves_path = RESULTS_DIR / "task2_scratch_curves.png"
    plot_curves(history, curves_path,
                title="Task 2: ResNet18 from scratch")
    print(f"\nSaved curves   : {curves_path}")

    summary = {
        "task": "task2_scratch",
        "strategy": "from_scratch",
        "pretrained": False,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "trainable_params": count_parameters(model),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "avg_epoch_time_s": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "history": history,
    }
    summary_path = RESULTS_DIR / "task2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary  : {summary_path}")

    ckpt_path = CHECKPOINT_DIR / "task2_scratch.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint : {ckpt_path}")

    print("\nTask 2 done. Inspect the curves: expect a large train/val gap.")
    print("Compare with Task 3, where the same architecture is pretrained+frozen.")


if __name__ == "__main__":
    main()
