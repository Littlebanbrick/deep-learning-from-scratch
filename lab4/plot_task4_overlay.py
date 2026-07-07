"""
Plot the Task 3 (frozen) vs Task 4 (fine-tune) overlay, offline.

Reads the two summary.json files (which embed the full training history) and
draws a single comparison figure. No training, no network — pure plotting, so
it runs in seconds and is safe to re-run after editing either task.

The story this figure exists to tell
------------------------------------
- Task 3's val accuracy plateaus around 0.91 over 20 epochs (frozen features).
- Task 4 RESUMES from that checkpoint and continues for 10 more epochs. On the
  plot Task 4 is shifted to epochs 21-30 so the two runs read as one continuous
  training trajectory.
- The visible *dip* at epoch 21 is the BN-mode-switch transient (Task 3 kept BN
  in eval; Task 4 flips it to train), not a bug. Recovery and a small net gain
  follow.

Run
---
    python plot_task4_overlay.py

Outputs
-------
- results/task4_vs_task3_overlay.png
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from common import RESULTS_DIR


def load_history(path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    t3 = load_history(RESULTS_DIR / "task3_summary.json")
    t4 = load_history(RESULTS_DIR / "task4_summary.json")

    h3, h4 = t3["history"], t4["history"]
    n3, n4 = len(h3["val_acc"]), len(h4["val_acc"])

    # Task 4 continues Task 3, so shift its epochs to 21..30.
    ep3 = list(range(1, n3 + 1))
    ep4 = list(range(n3 + 1, n3 + n4 + 1))
    boundary = n3 + 0.5  # vertical separator between the two runs

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Left: validation accuracy ---
    ax = axes[0]
    ax.plot(ep3, h3["val_acc"], "o-", color="#d62728",
            label=f"Task 3 frozen (20 ep) — final {h3['val_acc'][-1]:.4f}")
    ax.plot(ep4, h4["val_acc"], "s-", color="#2ca02c",
            label=f"Task 4 fine-tune (+10 ep) — final {h4['val_acc'][-1]:.4f}")
    ax.axvline(boundary, color="grey", linestyle="--", alpha=0.6)
    ax.text(boundary + 0.2, ax.get_ylim()[0] + 0.005, "unfreeze",
            color="grey", fontsize=9, rotation=90, va="bottom")
    # Mark the BN-switch dip at Task 4 epoch 1.
    ax.annotate("BN-switch\ndip",
                xy=(ep4[0], h4["val_acc"][0]),
                xytext=(ep4[0] + 1.5, h4["val_acc"][0] - 0.04),
                fontsize=8, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
    ax.set_xlabel("Epoch (Task 4 continues from Task 3 checkpoint)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Frozen vs fine-tuned — validation accuracy")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Right: validation loss ---
    ax = axes[1]
    ax.plot(ep3, h3["val_loss"], "o-", color="#d62728",
            label=f"Task 3 frozen — final {h3['val_loss'][-1]:.4f}")
    ax.plot(ep4, h4["val_loss"], "s-", color="#2ca02c",
            label=f"Task 4 fine-tune — final {h4['val_loss'][-1]:.4f}")
    ax.axvline(boundary, color="grey", linestyle="--", alpha=0.6)
    ax.set_xlabel("Epoch (Task 4 continues from Task 3 checkpoint)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Frozen vs fine-tuned — validation loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Headline test-accuracy deltas in the suptitle.
    fig.suptitle(
        f"Test accuracy: frozen {t3['final_test_acc']:.4f}  →  "
        f"fine-tuned {t4['final_test_acc']:.4f}  "
        f"(+{(t4['final_test_acc']-t3['final_test_acc'])*100:.2f} pp)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / "task4_vs_task3_overlay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
