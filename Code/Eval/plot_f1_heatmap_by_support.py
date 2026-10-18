"""
plot_f1_heatmap_by_support.py
=============================
Heatmap: per-category F1 for all models, categories sorted by training support.

Usage:
    python plot_f1_heatmap_by_support.py \
        --train      "C:\\...\\bert_data\\bert_train.csv" \
        --cur_dg_csv "C:\\...\\cur_dg_analysis_v4\\cur_dg_per_category.csv" \
        --bert_csv   "C:\\...\\FinBERT\\analysis\\bert_per_category.csv" \
        --output     "C:\\...\\Results\\f1_heatmap.png"
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


MODEL_COLS = {
    "Llama 3.1":    "llama 3.1_f1",
    "Poro2":        "poro2_f1",
    "Poro2 FT":     "poro2 ft_f1",
    "Poro2 FT RAG": "poro2 ft rag_f1",
    "FinBERT":      "finbert",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--cur_dg_csv", required=True)
    parser.add_argument("--bert_csv",   required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    # Training support from train_v3.xlsx
    train_df = pd.read_excel(args.train, engine="openpyxl")
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    support = train_df["cur_dg"].dropna().str.strip().str.lower().value_counts().to_dict()

    # LLM per-category
    llm_df = pd.read_csv(args.cur_dg_csv, encoding="latin-1")
    llm_df.columns = [c.strip().lower() for c in llm_df.columns]
    llm_df["category"] = llm_df["category"].str.strip().str.lower()
    print("LLM columns:", llm_df.columns.tolist())

    # FinBERT per-category
    bert_df = pd.read_csv(args.bert_csv, encoding="latin-1")
    bert_df.columns = [c.strip().lower() for c in bert_df.columns]
    bert_df["category"] = bert_df["category"].str.strip().str.lower()
    print("BERT columns:", bert_df.columns.tolist())

    # Build combined dataframe
    all_cats = sorted(
        [c for c in support.keys()
         if support.get(c, 0) > 0
         and (c in llm_df["category"].values or c in bert_df["category"].values)],
        key=lambda c: -support.get(c, 0)
    )

    model_labels = list(MODEL_COLS.keys())
    matrix = np.full((len(model_labels), len(all_cats)), np.nan)

    for j, cat in enumerate(all_cats):
        for i, (model, col) in enumerate(MODEL_COLS.items()):
            if model == "FinBERT":
                row = bert_df[bert_df["category"] == cat]
                if not row.empty:
                    matrix[i, j] = row["f1"].values[0]
            else:
                row = llm_df[llm_df["category"] == cat]
                if not row.empty and col in llm_df.columns:
                    matrix[i, j] = row[col].values[0]

    # Get test support from LLM per-category CSV
    test_support = {}
    if "n" in llm_df.columns:
        for _, row in llm_df.iterrows():
            test_support[str(row["category"]).strip().lower()] = int(row["n"]) if pd.notna(row["n"]) else 0

    # Remove categories where all LLM models are NaN or train/test support < 3
    llm_rows = [i for i, m in enumerate(model_labels) if m != "FinBERT"]
    keep_cols = [j for j in range(len(all_cats))
                 if not all(np.isnan(matrix[i, j]) for i in llm_rows)
                 and support.get(all_cats[j], 0) >= 2
                 and test_support.get(all_cats[j], 0) >= 2]
    matrix   = matrix[:, keep_cols]
    all_cats = [all_cats[j] for j in keep_cols]
    train_counts = [support.get(c, 0) for c in all_cats]

    # Plot
    fig, ax = plt.subplots(figsize=(max(14, len(all_cats) * 0.45), 4))
    plt.rcParams.update({"font.family": "Arial"})

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "soft_rg", ["#D9534F", "#F5E642", "#5CB85C"], N=256
    )
    cmap.set_bad("lightgrey")

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(all_cats)))
    train_counts = [support.get(c, 0) for c in all_cats]
    ax.set_xticklabels(
        [f"{c}\n(tr={support.get(c,0)}, te={test_support.get(c,0)})" for c in all_cats],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=9)

    # Cell values
    for i in range(len(model_labels)):
        for j in range(len(all_cats)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if 0.3 < val < 0.8 else "white")

    # Add horizontal lines between models
    for i in range(1, len(model_labels)):
        ax.axhline(i - 0.5, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, label="F1 score", shrink=0.8)
    ax.set_title("Per-category F1 by model — sorted by training set support (high → low)",
                 fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()