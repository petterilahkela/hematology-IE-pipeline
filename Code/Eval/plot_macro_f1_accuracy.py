"""
plot_macro_f1_accuracy.py
=========================
Two bar charts:
1. Macro F1 for all models including FinBERT
2. Accuracy for all models including FinBERT

Usage:
    python plot_macro_f1_accuracy.py \
        --cur_dg_summary "C:\\...\\cur_dg_analysis_v5\\cur_dg_summary.csv" \
        --bert_summary   "C:\\...\\FinBERT\\analysis\\bert_summary.csv" \
        --output_dir     "C:\\...\\Results\\"
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_ORDER  = ["Llama 3.1", "Poro2", "Poro2 FT", "Poro2 FT RAG", "FinBERT"]
MODEL_COLORS = ["#9DB8C8", "#7B9DBF", "#5B7FA6", "#3D5A73", "#C0533A"]

# Map from cur_dg_summary model names to display names
LLM_NAME_MAP = {
    "Llama 3.1":    "Llama 3.1",
    "Poro2":        "Poro2",
    "Poro2 FT":     "Poro2 FT",
    "Poro2 FT RAG": "Poro2 FT RAG",
}


def plot_bar(values_dict, ylabel, title, filename, out_dir):
    models = [m for m in MODEL_ORDER if m in values_dict]
    vals   = [values_dict[m] for m in models]
    colors = [MODEL_COLORS[MODEL_ORDER.index(m)] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, vals, color=colors, width=0.6)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, filename)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cur_dg_summary", required=True)
    parser.add_argument("--bert_summary",   required=True)
    parser.add_argument("--output_dir",     required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial"})

    # Load LLM summary
    llm_df = pd.read_csv(args.cur_dg_summary)
    llm_df.columns = [c.strip() for c in llm_df.columns]

    # Load FinBERT summary
    bert_df = pd.read_csv(args.bert_summary)
    bert_df.columns = [c.strip() for c in bert_df.columns]

    print("Models in cur_dg_summary:", llm_df["model"].tolist())

    # Build dicts
    macro_f1  = {}
    accuracy  = {}

    for _, row in llm_df.iterrows():
        name = LLM_NAME_MAP.get(str(row["model"]).strip())
        if name:
            macro_f1[name] = float(row["macro_f1"])
            accuracy[name] = float(row["accuracy"])

    macro_f1["FinBERT"] = float(bert_df["macro_f1"].values[0])
    accuracy["FinBERT"] = float(bert_df["accuracy"].values[0])

    print("Macro F1:", macro_f1)
    print("Accuracy:", accuracy)

    plot_bar(macro_f1,
             ylabel="Macro F1",
             title="cur_dg — Macro F1 by model",
             filename="all_models_macro_f1.png",
             out_dir=args.output_dir)

    plot_bar(accuracy,
             ylabel="Accuracy",
             title="cur_dg — Accuracy by model",
             filename="all_models_accuracy.png",
             out_dir=args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()