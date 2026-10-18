"""
plot_training_loss.py
=====================
Plots train loss and eval loss curves from finetune log files.

Usage:
    python plot_training_loss.py \
        --ft_log  "C:\\...\\finetune_log_v3.txt" \
        --rag_log "C:\\...\\finetune_log_rag_v3.txt" \
        --output  "C:\\...\\Logs\\"
"""

import argparse
import ast
import os
import re
import matplotlib.pyplot as plt

# ── Parse log file ─────────────────────────────────────────────────────────────

def parse_log(path):
    train_loss = []
    eval_loss  = []

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            match = re.search(r"\{[^{}]+\}", line)
            if not match:
                continue
            try:
                d = ast.literal_eval(match.group())
            except Exception:
                continue

            epoch = d.get("epoch")
            if epoch is None:
                continue

            if "loss" in d and "eval_loss" not in d:
                train_loss.append((epoch, d["loss"]))
            elif "eval_loss" in d:
                eval_loss.append((epoch, d["eval_loss"]))

    return train_loss, eval_loss


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_separate(ft_train, ft_eval, rag_train, rag_eval, out_dir):
    """One subplot per model showing both train and eval loss."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharey=False)
    fig.suptitle("Training and Validation Loss", fontsize=13, y=1.01)
 
    for ax, train, evl, title in [
        (axes[0], ft_train,  ft_eval,  "Poro2 FT"),
        (axes[1], rag_train, rag_eval, "Poro2 FT RAG"),
    ]:
        if train:
            epochs_t, losses_t = zip(*train)
            ax.plot(epochs_t, losses_t, color="#4472C4", linewidth=1.2,
                    alpha=0.8, label="Train loss")
        if evl:
            epochs_e, losses_e = zip(*evl)
            ax.plot(epochs_e, losses_e, color="#ED7D31", linewidth=2,
                    marker="o", markersize=4, label="Validation loss")
 
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
 
        max_epoch = max(
            (max(e for e, _ in train) if train else 0),
            (max(e for e, _ in evl)   if evl   else 0),
        )
        ax.set_xticks(range(0, int(max_epoch) + 2))
 
    plt.tight_layout()
    out = os.path.join(out_dir, "training_loss_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
 
def plot_combined(ft_train, ft_eval, rag_train, rag_eval, out_dir):
    """Train loss and eval loss side by side comparing FT vs RAG."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle("Train Loss and Eval Loss — FT vs RAG", fontsize=13)
 
    ax = axes[0]
    if ft_train:
        e, l = zip(*ft_train)
        ax.plot(e, l, color="#4472C4", linewidth=1.2, alpha=0.8, label="FT train")
    if rag_train:
        e, l = zip(*rag_train)
        ax.plot(e, l, color="#ED7D31", linewidth=1.2, alpha=0.8, label="RAG train")
    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
 
    ax = axes[1]
    if ft_eval:
        e, l = zip(*ft_eval)
        ax.plot(e, l, color="#4472C4", linewidth=2,
                marker="o", markersize=5, label="FT eval")
    if rag_eval:
        e, l = zip(*rag_eval)
        ax.plot(e, l, color="#ED7D31", linewidth=2,
                marker="s", markersize=5, label="RAG eval")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
 
    plt.tight_layout()
    out = os.path.join(out_dir, "training_loss_combined.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_log",  required=True)
    parser.add_argument("--rag_log", required=True)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Parsing FT log...")
    ft_train, ft_eval = parse_log(args.ft_log)
    print(f"  Train: {len(ft_train)} points, Eval: {len(ft_eval)} points")

    print("Parsing RAG log...")
    rag_train, rag_eval = parse_log(args.rag_log)
    print(f"  Train: {len(rag_train)} points, Eval: {len(rag_eval)} points")

    plt.rcParams.update({"font.family": "Arial"})

    plot_separate(ft_train, ft_eval, rag_train, rag_eval, args.output)
    plot_combined(ft_train, ft_eval, rag_train, rag_eval, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
