"""
analyse_cur_dg_v3.py
====================
Reads the cur_dg sheet from eval_v3_norm_J.xlsx and computes
correct micro/macro/weighted F1 per model using the same
TP/FP/FN logic as the V2 classification_analysis.py.

Key logic (lenient scoring):
  score=1  → TP for gold class  (no FP for predicted, even if pred != gold)
  score=0  → FN for gold class, FP for normalised predicted class

Usage:
    python analyse_cur_dg_v3.py \
        --input   "C:\\...\\eval_v3_norm_J.xlsx" \
        --mapping "C:\\...\\label_mapping_extended_eval.json" \
        --output  "C:\\...\\cur_dg_analysis_v3\\"
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = {
    "llama": "Llama 3.1 8B",
    "poro2": "Poro2 Instruct",
    "ft":    "Poro2 + FT",
    "rag":   "Poro2 + FT RAG",
}

COL_SCORE = {
    "llama": "llama 3.1 8b score",
    "poro2": "poro2 instruct score",
    "ft":    "poro2 + ft score",
    "rag":   "poro2 + ft rag score",
}

COL_OUTPUT = {
    "llama": "llama 3.1 8b output",
    "poro2": "poro2 instruct output",
    "ft":    "poro2 + ft output",
    "rag":   "poro2 + ft rag output",
}

MODEL_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#2E75B6"]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_excel(path, sheet_name="cur_dg", engine="openpyxl", header=1)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_mapping(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cur_dg_mapping"]


def clean(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return None if s.lower() in {"", "nan", "none", "null"} else s


def normalise(val, mapping):
    v = clean(val)
    if v is None:
        return None
    if v in mapping:
        return mapping[v]
    for k, m in mapping.items():
        if k.lower() == v.lower():
            return m
    return v.lower()   # unknown — keep as lowercase


# ── Per-category metrics (same logic as V2) ───────────────────────────────────

def get_pairs(df, mk, mapping):
    """
    Returns list of (gold, pred_norm, score) for all rows.
    gold     = gold_norm (normalised annotation)
    pred_norm = normalised model output
    score    = 0 or 1 (null treated as 1 = null-null skip handled below)
    """
    pairs = []
    score_col  = COL_SCORE[mk]
    output_col = COL_OUTPUT[mk]

    for _, row in df.iterrows():
        gold  = clean(row.get("gold_norm"))
        pred  = normalise(row.get(output_col), mapping)
        sc_raw = row.get(score_col)

        # Parse score
        sc = None
        if sc_raw is not None and not (isinstance(sc_raw, float) and pd.isna(sc_raw)):
            try:
                sc = int(float(str(sc_raw).replace(",", ".")))
            except (ValueError, TypeError):
                sc = None

        pairs.append((gold, pred, sc))
    return pairs


def compute_per_category(pairs):
    """
    TP/FP/FN per category using lenient logic:
      score=1  → TP for gold
      score=0  → FN for gold, FP for predicted
      both None → skip
      gold None, pred not None, score=0 → FP for pred
      gold not None, pred None, score=0 → FN for gold
    """
    tp      = defaultdict(float)
    fp      = defaultdict(float)
    fn      = defaultdict(int)
    support = defaultdict(int)

    for gold, pred, score in pairs:
        if gold is None and pred is None:
            continue   # null-null: skip

        if gold is not None:
            support[gold] += 1

        if gold is not None and pred is not None:
            if score == 1:
                tp[gold] += 1
            else:   # score == 0 or None treated as wrong
                fn[gold] += 1
                if pred != gold:
                    fp[pred] += 1
        elif gold is not None and pred is None:
            fn[gold] += 1
        elif gold is None and pred is not None:
            fp[pred] += 1

    cats = sorted(support.keys())
    result = {}
    for cat in cats:
        t   = tp[cat]
        fpc = fp[cat]
        fnc = fn[cat]
        prec = t / (t + fpc) if (t + fpc) > 0 else 0.0
        rec  = t / (t + fnc) if (t + fnc) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        result[cat] = {
            "tp": t, "fp": fpc, "fn": fnc,
            "precision": prec, "recall": rec, "f1": f1,
            "support": support[cat],
        }
    return result


def aggregate_metrics(per_cat):
    """Compute macro, weighted, micro F1 from per-category metrics."""
    if not per_cat:
        return 0.0, 0.0, 0.0

    total_support = sum(v["support"] for v in per_cat.values())

    macro_f1    = sum(v["f1"] for v in per_cat.values()) / len(per_cat)
    weighted_f1 = (sum(v["f1"] * v["support"] for v in per_cat.values()) / total_support
                   if total_support > 0 else 0.0)

    # Micro: aggregate TP/FP/FN globally
    tp_total = sum(v["tp"] for v in per_cat.values())
    fp_total = sum(v["fp"] for v in per_cat.values())
    fn_total = sum(v["fn"] for v in per_cat.values())
    micro_p  = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2*micro_p*micro_r / (micro_p+micro_r)
                if (micro_p+micro_r) > 0 else 0.0)

    return round(macro_f1, 4), round(micro_f1, 4), round(weighted_f1, 4)


# ── Build summary table ───────────────────────────────────────────────────────

def build_summary(df, mapping):
    rows = []
    for mk, label in MODELS.items():
        if COL_SCORE[mk] not in df.columns:
            continue
        pairs   = get_pairs(df, mk, mapping)
        per_cat = compute_per_category(pairs)
        macro, micro, weighted = aggregate_metrics(per_cat)

        n_correct = sum(1 for _, _, sc in pairs if sc == 1)
        n_total   = sum(1 for g, _, _ in pairs if g is not None)

        rows.append({
            "model":        label,
            "macro_f1":     macro,
            "micro_f1":     micro,
            "weighted_f1":  weighted,
            "correct":      n_correct,
            "total":        n_total,
            "accuracy":     round(n_correct / n_total, 4) if n_total > 0 else 0,
        })
    return pd.DataFrame(rows)


def build_per_cat_table(df, mapping, min_count=3):
    """Per-category F1 per model."""
    all_cats = set()
    model_metrics = {}

    for mk, label in MODELS.items():
        if COL_SCORE[mk] not in df.columns:
            continue
        pairs = get_pairs(df, mk, mapping)
        pm = compute_per_category(pairs)
        model_metrics[label] = pm
        all_cats.update(pm.keys())

    # Filter by min_count (support from first model)
    first_label = list(MODELS.values())[0]
    first_pm    = model_metrics.get(first_label, {})
    cats = sorted(
        [c for c in all_cats if first_pm.get(c, {}).get("support", 0) >= min_count],
        key=lambda c: -first_pm.get(c, {}).get("support", 0)
    )

    rows = []
    for cat in cats:
        row = {"category": cat,
               "n": first_pm.get(cat, {}).get("support", 0)}
        for label in MODELS.values():
            pm  = model_metrics.get(label, {})
            row[f"{label}_f1"]      = pm.get(cat, {}).get("f1")
            row[f"{label}_prec"]    = pm.get(cat, {}).get("precision")
            row[f"{label}_rec"]     = pm.get(cat, {}).get("recall")
        rows.append(row)

    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_f1_grouped(summary, out_dir):
    """Grouped bar chart: micro / macro / weighted F1 per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    w = 0.25

    b1 = ax.bar(x - w, summary["macro_f1"],    w, label="Macro F1",    color="#4472C4")
    b2 = ax.bar(x,     summary["micro_f1"],    w, label="Micro F1",    color="#ED7D31")
    b3 = ax.bar(x + w, summary["weighted_f1"], w, label="Weighted F1", color="#70AD47")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("F1 Score")
    ax.set_title("cur_dg — Macro / Micro / Weighted F1 (lenient)", fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_f1_grouped.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_accuracy(summary, out_dir):
    """Simple accuracy bar per model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(summary["model"], summary["accuracy"],
                  color=MODEL_COLORS[:len(summary)], width=0.5, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Accuracy (lenient)")
    ax.set_title("cur_dg — Accuracy by Model", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_accuracy.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_category_heatmap(cat_df, out_dir):
    """Heatmap of F1 per category per model."""
    model_labels = list(MODELS.values())
    f1_cols = [f"{lbl}_f1" for lbl in model_labels if f"{lbl}_f1" in cat_df.columns]
    data = cat_df[f1_cols].values.T
    cats = cat_df["category"].tolist()
    ns   = cat_df["n"].tolist()

    cmap = LinearSegmentedColormap.from_list("rg", ["#FCE4D6", "#FFEB9C", "#C6EFCE"])
    fig, ax = plt.subplots(figsize=(max(10, len(cats)*0.55), 3.5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(cats, ns)],
        rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(f1_cols)))
    ax.set_yticklabels(model_labels[:len(f1_cols)], fontsize=9)

    for i in range(len(f1_cols)):
        for j in range(len(cats)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 format=mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("cur_dg — F1 per category (lenient, n≥3)", fontsize=11, pad=10)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_per_category_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_category_heatmap_rare(df, mapping, out_dir, max_count=2):
    """Heatmap for rare categories (n <= max_count)."""
    all_cats = set()
    model_metrics = {}

    for mk, label in MODELS.items():
        if COL_SCORE[mk] not in df.columns:
            continue
        pairs = get_pairs(df, mk, mapping)
        pm = compute_per_category(pairs)
        model_metrics[label] = pm
        all_cats.update(pm.keys())

    first_label = list(MODELS.values())[0]
    first_pm    = model_metrics.get(first_label, {})

    # Only rare categories
    cats = sorted(
        [c for c in all_cats
         if 1 <= first_pm.get(c, {}).get("support", 0) <= max_count],
        key=lambda c: -first_pm.get(c, {}).get("support", 0)
    )

    if not cats:
        print("No rare categories found.")
        return

    model_labels = list(MODELS.values())
    data = np.array([
        [model_metrics.get(lbl, {}).get(cat, {}).get("f1", np.nan)
         for cat in cats]
        for lbl in model_labels
    ])
    ns = [first_pm.get(c, {}).get("support", 0) for c in cats]

    cmap = LinearSegmentedColormap.from_list("rg", ["#FCE4D6", "#FFEB9C", "#C6EFCE"])
    fig, ax = plt.subplots(figsize=(max(8, len(cats)*0.7), 3.5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(cats, ns)],
        rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=9)

    for i in range(len(model_labels)):
        for j in range(len(cats)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 format=mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(f"cur_dg — F1 per rare category (n≤{max_count})", fontsize=11, pad=10)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_rare_categories_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

def plot_distribution(df, out_dir, top_n=20):
    counts = (df["gold_norm"].str.strip().str.lower()
              .value_counts().head(top_n))
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.sort_values().plot.barh(ax=ax, color="#2E75B6", edgecolor="white")
    for i, val in enumerate(counts.sort_values()):
        ax.text(val + 0.2, i, str(val), va="center", fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(f"cur_dg — Gold label distribution (top {top_n})", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True, help="eval_v3_norm_J.xlsx")
    parser.add_argument("--mapping",   required=True, help="label_mapping_extended_eval.json")
    parser.add_argument("--output",    required=True, help="Output directory")
    parser.add_argument("--min_count", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading data...")
    df      = load_data(args.input)
    mapping = load_mapping(args.mapping)
    print(f"  {len(df)} rows, mapping: {len(mapping)} entries")

    # Summary
    print("\nComputing metrics...")
    summary = build_summary(df, mapping)
    print(summary[["model", "macro_f1", "micro_f1", "weighted_f1", "accuracy",
                   "correct", "total"]].to_string(index=False))
    summary.to_csv(os.path.join(args.output, "cur_dg_summary.csv"), index=False)

    # Per-category
    cat_df = build_per_cat_table(df, mapping, min_count=args.min_count)
    cat_df.to_csv(os.path.join(args.output, "cur_dg_per_category.csv"), index=False)

    # Plots
    plt.rcParams.update({"font.family": "Arial", "axes.grid": False})
    plot_f1_grouped(summary, args.output)
    plot_accuracy(summary, args.output)
    plot_category_heatmap(cat_df, args.output)
    plot_category_heatmap_rare(df, mapping, args.output)
    plot_distribution(df, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
