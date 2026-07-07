"""
Task 4: Fine-Tuning (Unfrozen Pretrained Backbone)
===================================================

Goal
----
Resume from the Task 3 checkpoint (pretrained backbone + a head already trained
to ~0.91 val accuracy), **unfreeze** the entire backbone, and continue training
the whole network for ~10 more epochs with **differential learning rates**: a
small rate (1e-4) for the pretrained backbone, a larger rate (1e-3) for the head.

This is the third and most powerful of the three transfer-learning strategies.
The backbone stops being a fixed feature extractor and is *gently* nudged so that
its features become pet-specific, not just generically good. The head keeps
adapting to track the shifting representations underneath it.

Why differential learning rates (the core idea of this task)
-----------------------------------------------------------
The pretrained backbone already encodes useful visual features learned from 1.28M
ImageNet images. We do not want to overwrite them — a large learning rate would
cause *catastrophic forgetting*, smashing the carefully learned weights in a few
steps. So the backbone gets a small step (1e-4): enough to specialise toward
pets, not enough to forget ImageNet.

The head is different. Even though Task 3 already trained it, the backbone it
sits on is now *moving underneath it* — features that were frozen in Task 3 now
shift every step. The head needs to chase that moving target, so it keeps the
larger 1e-3 rate. Using 1e-4 for both would leave the head too sluggish to track
the backbone; using 1e-3 for both would wreck the backbone. Differential rates
are the compromise, and this is what report question 4 asks you to explain.

Two changes from Task 3 (read this before comparing the curves)
---------------------------------------------------------------
1. **The whole network is in train() mode now**, including BatchNorm. In Task 3
   we forced BN into eval so its running stats stayed frozen at ImageNet values;
   here, BN running stats *do* update, because the features are being adapted
   toward pet-relevant ones and BN must track that new distribution. Forgetting
   the Task 3 BN-eval trick here is *correct* — the two situations are genuinely
   different. (Task 3: frozen features must keep frozen stats. Task 4: moving
   features need moving stats.)
2. **We resume from the Task 3 checkpoint**, not from a fresh pretrained model.
   This makes "frozen vs fine-tuned" a fair, single-variable comparison: same
   starting head, same data, same seed — only the freeze flag differs for the
   extra epochs.

Run
---
    python -u task4_finetune.py

Expect ~7-10 minutes on CPU (10 epochs × ~40-50 s/epoch — slower than Task 3
because backbone gradients are now computed and back-propagated).

Prerequisite
------------
checkpoints/task3_frozen.pt must exist. Run `python task3_frozen.py` first if not.

Outputs
-------
- results/task4_finetune_curves.png
- results/task4_summary.json
- checkpoints/task4_finetune.pt
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
    split_params_by_lr, evaluate, get_device, plot_curves,
    RESULTS_DIR, CHECKPOINT_DIR, SEED, NUM_CLASSES, BATCH_SIZE,
)


# ------------------------------------------------------------------
# Training loop — whole network, differential learning rates
# ------------------------------------------------------------------

def train_finetune(model, train_loader, val_loader, epochs, device):
    """Fine-tune the whole ResNet18 with differential learning rates.

    The optimizer is built once from split_params_by_lr(): backbone params at
    lr=1e-4, head params at lr=1e-3. Everything is in train() mode, including
    BatchNorm, so the backbone's running statistics track the pet-adapted
    feature distribution.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(split_params_by_lr(model))

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [], "epoch_time": []}

    for epoch in range(epochs):
        t0 = time.time()

        # --- train ---
        # Full train() mode: backbone gradients flow, BN running stats update.
        # This is the deliberate opposite of Task 3's eval()+fc.train() trick.
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

EPOCHS = 10
CKPT_RESUME = CHECKPOINT_DIR / "task3_frozen.pt"


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    device = get_device()

    print("=" * 64)
    print("Task 4: Fine-Tuning (pretrained backbone, UNFROZEN)")
    print("=" * 64)
    print(f"Device        : {device}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Optimizer     : Adam, differential LR (backbone 1e-4, head 1e-3)")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Num classes   : {NUM_CLASSES}")

    train_loader, val_loader, test_loader, _ = get_dataloaders()
    print(f"Train / Val / Test : {len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / {len(test_loader.dataset)}")

    # Build the model in the *unfrozen* configuration, then load the Task 3
    # checkpoint (pretrained backbone + trained head). requires_grad is a
    # property of the live parameters, not of the state_dict, so building with
    # freeze=False is what makes everything trainable; load_state_dict only
    # copies the weight *values* in.
    if not CKPT_RESUME.exists():
        raise FileNotFoundError(
            f"Task 3 checkpoint not found at {CKPT_RESUME}.\n"
            f"Run `python task3_frozen.py` first — Task 4 resumes from it."
        )

    model = build_resnet(num_classes=NUM_CLASSES, pretrained=True, freeze=False)
    state = torch.load(CKPT_RESUME, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"\nResumed from     : {CKPT_RESUME}")
    print(f"Trainable params  : {count_parameters(model):,}  "
          f"(expected ~11,195,493 = full network, now all trainable)")

    # Show the differential-LR parameter groups — the core mechanic of this task.
    groups = split_params_by_lr(model)
    print("\nDifferential-LR parameter groups:")
    for i, g in enumerate(groups):
        n = sum(p.numel() for p in g["params"])
        tag = "backbone (pretrained, small LR)" if g["lr"] == 1e-4 \
              else "head (fresh, larger LR)"
        print(f"  group {i}: lr={g['lr']:.0e}  params={n:>11,}  [{tag}]")

    print("\nFine-tuning whole network (BN in train mode, full gradients) ...")
    t_total = time.time()
    history = train_finetune(model, train_loader, val_loader, EPOCHS, device)
    print(f"\nTotal training time: {(time.time()-t_total)/60:.1f} min")
    print(f"Avg epoch time     : {sum(history['epoch_time'])/len(history['epoch_time']):.1f}s")

    # --- Final test evaluation ---
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal TEST accuracy : {test_acc:.4f}")
    print(f"Final TEST loss     : {test_loss:.4f}")

    # --- Artefacts ---
    curves_path = RESULTS_DIR / "task4_finetune_curves.png"
    plot_curves(history, curves_path,
                title="Task 4: ResNet18 pretrained, fine-tuned (unfrozen)")
    print(f"\nSaved curves   : {curves_path}")

    summary = {
        "task": "task4_finetune",
        "strategy": "fine_tuning",
        "pretrained": True,
        "frozen": False,
        "resumed_from": str(CKPT_RESUME),
        "epochs": EPOCHS,
        "lr_backbone": 1e-4,
        "lr_head": 1e-3,
        "batch_size": BATCH_SIZE,
        "trainable_params": count_parameters(model),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "avg_epoch_time_s": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "history": history,
    }
    summary_path = RESULTS_DIR / "task4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary  : {summary_path}")

    ckpt_path = CHECKPOINT_DIR / "task4_finetune.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint : {ckpt_path}")

    print("\nTask 4 done. Compare with Task 3 (frozen-only, test ~0.885):")
    print("  Task 3 (frozen, 20 ep)     -> test 0.8850")
    print("  Task 4 (fine-tune, +10 ep) -> test {:.4f}  (target: small but real gain)"
          .format(test_acc))
    print("Same start point, same data, same seed — only the freeze flag changed.")


if __name__ == "__main__":
    main()
