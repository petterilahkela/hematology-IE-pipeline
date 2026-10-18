"""
plot_bert_training.py
=====================
Plots FinBERT training loss and validation metrics per epoch.

Usage:
    python plot_bert_training.py \
        --log    "C:\\...\\Logs\\finetune_log_bert_v1.txt" \
        --output "C:\\...\\Results\\Logs\\"
"""

import argparse
import os
import re
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial"})

    with open(args.log, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Parse train loss
    train_entries = re.findall(
        r"\{'loss': '([^']+)'.*?'epoch': '([^']+)'\}", text
    )
    train_epochs = [float(e) for _, e in train_entries]
    train_losses = [float(l) for l, _ in train_entries]

    # Parse eval metrics
    eval_entries = re.findall(
        r"eval_loss': '([^']+)', 'eval_macro_f1': '([^']+)', 'eval_micro_f1': '([^']+)', 'eval_accuracy': '([^']+)'.*?'epoch': '([^']+)'\}", text
    )
    eval_epochs   = [float(e)  for _, _, _, _, e  in eval_entries]
    eval_losses   = [float(l)  for l, _, _, _, _  in eval_entries]
    eval_macro_f1 = [float(f1) for _, f1, _, _, _ in eval_entries]
    eval_accuracy = [float(a)  for _, _, _, a, _  in eval_entries]

    # Plot 1: Train loss + Val loss
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle("FinBERT Training", fontsize=13, y=1.01)

    ax = axes[0]
    ax.plot(train_epochs, train_losses, color="#4472C4", linewidth=1.2, alpha=0.8, label="Train loss")
    ax.plot(eval_epochs,  eval_losses,  color="#ED7D31", linewidth=2, marker="o", markersize=4, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Plot 2: Val macro F1 + accuracy
    ax = axes[1]
    ax.plot(eval_epochs, eval_macro_f1, color="#5B7FA6", linewidth=2, marker="o", markersize=4, label="Val Macro F1")
    ax.plot(eval_epochs, eval_accuracy, color="#C0533A", linewidth=2, marker="s", markersize=4, label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Macro F1 and Accuracy")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = os.path.join(args.output, "bert_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
