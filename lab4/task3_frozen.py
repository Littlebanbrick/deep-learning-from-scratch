"""
Task 3: Feature Extraction (Frozen Pretrained Backbone)
=========================================================

Goal
----
Load the ImageNet-pretrained ResNet18, **freeze** the entire backbone, replace
only the classification head with a fresh Linear(512, 37), and train *just the
head* on the Oxford-IIIT Pet dataset.

This is the "transfer learning, easy mode" case: the same architecture and the
same data as Task 2, but the backbone starts from weights that already encode
generic visual features (edges, textures, object parts) — and some quite
pet-specific ones, since ImageNet's 1000 classes include ~150 dog and cat
breeds. The head just has to learn a linear mapping from those 512-D features
to 37 breeds.

Why this should look good
-------------------------
- Only ~19k parameters are trainable (the new head), against ~3,300 training
  images — a healthy ratio, no overfitting capacity.
- The features are already good; the head is a convex-ish linear problem.
- Expect val accuracy in the 0.80-0.90 range, a dramatic jump from Task 2's
  ~0.42, achieved with the *same* architecture and the *same* data.

The BatchNorm subtlety (read this before tweaking the code)
------------------------------------------------------------
When the backbone is frozen, its BatchNorm layers must stay in eval mode.
Otherwise `model.train()` flips BN into training mode, and BN will update its
running mean/var from your tiny pet batches — polluting the ImageNet-learned
statistics and silently wrecking accuracy. The fix used here: call
`model.eval()` (which sets ALL layers, including BN, to eval), then
`model.fc.train()` to flip only the head back to training mode. The head has
no BN/Dropout, so this is a no-op on the head, but it documents intent.

Outputs
-------
- results/task3_frozen_curves.png
- results/task3_summary.json
- checkpoints/task3_frozen.pt   <- Task 4 resumes from this checkpoint
"""

from __future__ import annotations

import json
import os
import time

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
# Training loop — head only, backbone frozen
# ------------------------------------------------------------------

def train_frozen(model, train_loader, val_loader, epochs, lr, device):
    """Train only the head of a frozen-backbone ResNet18.

    Because the backbone is frozen and in eval mode, the penultimate features
    are a deterministic function of the input. A common optimisation is to
    precompute them once and train the head on cached features (turning 20
    epochs into seconds). We do NOT do that here — we keep the standard
    forward pass so the training loop mirrors Task 2 exactly, making the
    comparison clean. The runtime is still modest because no backbone gradients
    are computed.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [], "epoch_time": []}

    for epoch in range(epochs):
        t0 = time.time()

        # --- train ---
        # Backbone frozen + in eval (BN running stats frozen). Head in train.
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
    print("Task 3: Feature Extraction (pretrained backbone, FROZEN)")
    print("=" * 64)
    print(f"Device        : {device}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Optimizer     : Adam, lr={LR} (head only)")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Num classes   : {NUM_CLASSES}")

    train_loader, val_loader, test_loader, _ = get_dataloaders()
    print(f"Train / Val / Test : {len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / {len(test_loader.dataset)}")

    # Pretrained backbone, frozen. Head replaced and is the only trainable part.
    model = build_resnet(num_classes=NUM_CLASSES, pretrained=True, freeze=True)
    model.to(device)
    print(f"Trainable params : {count_parameters(model):,}  "
          f"(expected ~18,981 = head only: 512*37 + 37)")
    # Sanity: list which submodules are still trainable.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable tensors: {len(trainable_names)} (should be 2: fc.weight, fc.bias)")

    print("\nTraining head only (backbone frozen, BN in eval) ...")
    t_total = time.time()
    history = train_frozen(model, train_loader, val_loader, EPOCHS, LR, device)
    print(f"\nTotal training time: {(time.time()-t_total)/60:.1f} min")
    print(f"Avg epoch time     : {sum(history['epoch_time'])/len(history['epoch_time']):.1f}s")

    # --- Final test evaluation ---
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal TEST accuracy : {test_acc:.4f}")
    print(f"Final TEST loss     : {test_loss:.4f}")

    # --- Artefacts ---
    curves_path = RESULTS_DIR / "task3_frozen_curves.png"
    plot_curves(history, curves_path,
                title="Task 3: ResNet18 pretrained, backbone frozen")
    print(f"\nSaved curves   : {curves_path}")

    summary = {
        "task": "task3_frozen",
        "strategy": "feature_extraction",
        "pretrained": True,
        "frozen": True,
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
    summary_path = RESULTS_DIR / "task3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary  : {summary_path}")

    # Task 4 resumes from this checkpoint (pretrained backbone + trained head).
    ckpt_path = CHECKPOINT_DIR / "task3_frozen.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint : {ckpt_path}  (Task 4 resumes from here)")

    print("\nTask 3 done. Compare val/test accuracy with Task 2:")
    print("  Task 2 (scratch) ~0.36-0.42  vs  Task 3 (frozen) target ~0.80+")
    print("Same architecture, same data, only the initialisation changed.")


if __name__ == "__main__":
    main()
