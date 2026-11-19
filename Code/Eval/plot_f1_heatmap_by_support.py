"""
plot_f1_heatmap_by_support.py
=============================
Heatmap: per-category F1 for all models (Llama 3.1, Poro2, Poro2 FT, Poro2 FT RAG, FinBERT).
Categories are sorted by their number of training examples (high -> low).
Only categories with at least 2 examples in both the training set and the test set are shown.

Inputs:
  --train       training set Excel file (train_v3.xlsx); the cur_dg column gives the training counts
  --cur_dg_csv  cur_dg_per_category.csv written by analyse_cur_dg_v5.py (LLM F1 values and test counts)
  --bert_csv    bert_per_category.csv written by evaluate_bert.py (FinBERT F1 values)
  --output      output image (.png)

Note: both CSV files are read as UTF-8, as written by the analysis scripts.
Do not re-save them in Excel, which changes the encoding and garbles letters such as ä and ö.

Usage:
    python plot_f1_heatmap_by_support.py \
        --train      "C:\\...\\FT_DATA\\train_v3.xlsx" \
        --cur_dg_csv "C:\\...\\cur_dg_analysis_v5\\cur_dg_per_category.csv" \
        --bert_csv   "C:\\...\\FinBERT\\analysis\\bert_per_category.csv" \
        --output     "C:\\...\\Results\\f1_heatmap.png"
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Model name shown in the figure -> F1 column in cur_dg_per_category.csv (lower-cased).
# FinBERT is read from bert_per_category.csv (column "f1"), so its value here is not used.
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
    llm_df = pd.read_csv(args.cur_dg_csv, encoding="utf-8-sig")
    llm_df.columns = [c.strip().lower() for c in llm_df.columns]
    llm_df["category"] = llm_df["category"].str.strip().str.lower()
    print("LLM columns:", llm_df.columns.tolist())

    # FinBERT per-category
    bert_df = pd.read_csv(args.bert_csv, encoding="utf-8-sig")
    bert_df.columns = [c.strip().lower() for c in bert_df.columns]
    bert_df["category"] = bert_df["category"].str.strip().str.lower()
    print("BERT columns:", bert_df.columns.tolist())

    # Categories found in the training set and in at least one per-category CSV,
    # sorted by training support (high -> low)
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

    # Keep only categories with an F1 value for at least one LLM and with at least
    # 2 examples in both the training set and the test set
    llm_rows = [i for i, m in enumerate(model_labels) if m != "FinBERT"]
    keep_cols = [j for j in range(len(all_cats))
                 if not all(np.isnan(matrix[i, j]) for i in llm_rows)
                 and support.get(all_cats[j], 0) >= 2
                 and test_support.get(all_cats[j], 0) >= 2]
    matrix   = matrix[:, keep_cols]
    all_cats = [all_cats[j] for j in keep_cols]
    train_counts = [support.get(c, 0) for c in all_cats]

    # Transpose: categories on y-axis, models on x-axis
    matrix_T = matrix.T
    display_labels = model_labels

    fig, ax = plt.subplots(figsize=(len(model_labels) * 1.6, max(10, len(all_cats) * 0.38)))
    plt.rcParams.update({"font.family": "Arial"})

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "soft_rg", ["#D9534F", "#F5E642", "#5CB85C"], N=256
    )
    cmap.set_bad("lightgrey")

    im = ax.imshow(matrix_T, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # X-axis: models
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=9, rotation=30, ha="right")

    # Y-axis: categories sorted by training support (tr = training examples, te = test examples)
    ax.set_yticks(range(len(all_cats)))
    ax.set_yticklabels(
        [f"{c}  (tr={support.get(c,0)}, te={test_support.get(c,0)})" for c in all_cats],
        fontsize=7
    )

    # Cell values (columns with an empty label, if any, are skipped)
    for j in range(len(display_labels)):
        if display_labels[j] == "":
            continue
        for i in range(len(all_cats)):
            val = matrix_T[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    # White vertical lines between the model columns (empty-label columns, if any, are skipped)
    for j in range(1, len(display_labels)):
        if display_labels[j] == "":
            continue
        ax.axvline(j - 0.5, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, label="F1 score", shrink=0.8)
    ax.set_title("Per-category F1 by model — sorted by training set support (high → low)",
                 fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()