"""
analyse_cur_dg_v3.py
====================
Reads the cur_dg sheet from eval_v3_norm_J.xlsx and computes
correct micro/macro/weighted F1, Precision, and Recall per model.

Key logic (lenient scoring):
  score=1  → TP for gold class  (no FP for predicted, even if pred != gold)
  score=0  → FN for gold class, FP for normalised predicted class

Changes:
  - Unknown model outputs are mapped to the special category "other".
  - "other" is included in the category list (support=0) so its False Positives
    affect Micro and Macro metrics, correctly penalising hallucinations.
  - Separate grouped bar charts for F1, Precision, and Recall (each with
    Macro / Micro / Weighted).
  - Per-category heatmaps for F1, Precision, and Recall.
  - Error analysis: three categories only (False normal, False positive,
    Wrong disease). "Missed" and "Unknown output" have been removed/merged.
  - Error analysis legend moved to the right side of the plot.

Usage:
    python analyse_cur_dg_v3.py \
        --input   "C:\\...\\eval_v3_norm_J.xlsx" \
        --mapping "C:\\...\\label_mapping_extended_eval.json" \
        --output  "C:\\...\\cur_dg_analysis_v4\\"
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
    "llama": "Llama 3.1",
    "poro2": "Poro2",
    "ft":    "Poro2 FT",
    "rag":   "Poro2 FT RAG",
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

MODEL_COLORS       = ["#9DB8C8", "#7B9DBF", "#5B7FA6", "#3D5A73"]
FINBERT_COLOR      = "#C0533A"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_excel(path, sheet_name="cur_dg", engine="openpyxl", skiprows=[0], header=0)
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
    """
    Normalise a model output using the mapping.
    Unknown outputs are mapped to the special category "other".
    """
    v = clean(val)
    if v is None:
        return None
    if v in mapping:
        return mapping[v]
    for k, m in mapping.items():
        if k.lower() == v.lower():
            return m
    return "other"


# ── Pair extraction ───────────────────────────────────────────────────────────

def get_pairs(df, mk, mapping):
    """
    Returns list of (gold, pred_norm, score) for all rows.
    """
    pairs = []
    score_col  = COL_SCORE[mk]
    output_col = COL_OUTPUT[mk]

    for _, row in df.iterrows():
        gold  = clean(row.get("gold_norm"))
        pred  = normalise(row.get(output_col), mapping)
        sc_raw = row.get(score_col)

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
    TP/FP/FN per category using lenient logic. See module docstring.
    """
    tp      = defaultdict(float)
    fp      = defaultdict(float)
    fn      = defaultdict(int)
    support = defaultdict(int)

    has_other_pred = False
    for gold, pred, score in pairs:
        if gold is not None:
            support[gold] += 1
        if pred == "other":
            has_other_pred = True

    if has_other_pred:
        support["other"] = 0

    for gold, pred, score in pairs:
        if gold is None and pred is None:
            continue

        if gold is not None and pred is not None:
            if score == 1:
                tp[gold] += 1
            else:
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
    """Compute macro, weighted, micro Precision/Recall/F1 from per-category metrics."""
    if not per_cat:
        return {k: 0.0 for k in [
            "macro_precision", "macro_recall", "macro_f1",
            "micro_precision", "micro_recall", "micro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
        ]}

    total_support = sum(v["support"] for v in per_cat.values())
    n_cats = len(per_cat)

    # Macro
    macro_f1        = sum(v["f1"]        for v in per_cat.values()) / n_cats
    macro_precision = sum(v["precision"] for v in per_cat.values()) / n_cats
    macro_recall    = sum(v["recall"]    for v in per_cat.values()) / n_cats

    # Weighted
    if total_support > 0:
        weighted_f1        = sum(v["f1"]        * v["support"] for v in per_cat.values()) / total_support
        weighted_precision = sum(v["precision"] * v["support"] for v in per_cat.values()) / total_support
        weighted_recall    = sum(v["recall"]    * v["support"] for v in per_cat.values()) / total_support
    else:
        weighted_f1 = weighted_precision = weighted_recall = 0.0

    # Micro
    tp_total = sum(v["tp"] for v in per_cat.values())
    fp_total = sum(v["fp"] for v in per_cat.values())
    fn_total = sum(v["fn"] for v in per_cat.values())
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    return {
        "macro_precision":    round(macro_precision,    3),
        "macro_recall":       round(macro_recall,       3),
        "macro_f1":           round(macro_f1,           3),
        "micro_precision":    round(micro_precision,    3),
        "micro_recall":       round(micro_recall,       3),
        "micro_f1":           round(micro_f1,           3),
        "weighted_precision": round(weighted_precision, 3),
        "weighted_recall":    round(weighted_recall,    3),
        "weighted_f1":        round(weighted_f1,        3),
    }


# ── Build tables ──────────────────────────────────────────────────────────────

def build_summary(df, mapping):
    rows = []
    for mk, label in MODELS.items():
        if COL_SCORE[mk] not in df.columns:
            continue
        pairs   = get_pairs(df, mk, mapping)
        per_cat = compute_per_category(pairs)
        metrics = aggregate_metrics(per_cat)

        n_correct = sum(1 for _, _, sc in pairs if sc == 1)
        n_total   = sum(1 for g, _, _ in pairs if g is not None)

        rows.append({
            "model":              label,
            "accuracy":           round(n_correct / n_total, 3) if n_total > 0 else 0,
            "macro_precision":    metrics["macro_precision"],
            "macro_recall":       metrics["macro_recall"],
            "macro_f1":           metrics["macro_f1"],
            "micro_precision":    metrics["micro_precision"],
            "micro_recall":       metrics["micro_recall"],
            "micro_f1":           metrics["micro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall":    metrics["weighted_recall"],
            "weighted_f1":        metrics["weighted_f1"],
            "correct":            n_correct,
            "total":              n_total,
        })
    return pd.DataFrame(rows)


def build_per_cat_table(df, mapping, min_count=0):
    """Per-category F1/Precision/Recall per model."""
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
    """Grouped bar chart: macro / micro / weighted F1 per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    w = 0.25

    b1 = ax.bar(x - w, summary["macro_f1"],    w, label="Macro F1",    color="#2D6A4F")
    b2 = ax.bar(x,     summary["micro_f1"],    w, label="Micro F1",    color="#6BAF92")
    b3 = ax.bar(x + w, summary["weighted_f1"], w, label="Weighted F1", color="#B8DDD0")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Current Diagnosis — Macro / Micro / Weighted F1", fontsize=11, pad=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_f1_grouped.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_precision_grouped(summary, out_dir):
    """Grouped bar chart: macro / micro / weighted Precision per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    w = 0.25

    b1 = ax.bar(x - w, summary["macro_precision"],    w, label="Macro Precision",    color="#2D6A4F")
    b2 = ax.bar(x,     summary["micro_precision"],    w, label="Micro Precision",    color="#6BAF92")
    b3 = ax.bar(x + w, summary["weighted_precision"], w, label="Weighted Precision", color="#B8DDD0")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Precision")
    ax.set_title("Current Diagnosis — Macro / Micro / Weighted Precision", fontsize=11, pad=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_precision_grouped.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_recall_grouped(summary, out_dir):
    """Grouped bar chart: macro / micro / weighted Recall per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    w = 0.25

    b1 = ax.bar(x - w, summary["macro_recall"],    w, label="Macro Recall",    color="#2D6A4F")
    b2 = ax.bar(x,     summary["micro_recall"],    w, label="Micro Recall",    color="#6BAF92")
    b3 = ax.bar(x + w, summary["weighted_recall"], w, label="Weighted Recall", color="#B8DDD0")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall")
    ax.set_title("Current Diagnosis — Macro / Micro / Weighted Recall", fontsize=11, pad=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_recall_grouped.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_accuracy(summary, out_dir, bert_accuracy=None):
    """Simple accuracy bar per model, optionally including FinBERT."""
    models = list(summary["model"])
    accs   = list(summary["accuracy"])
    colors = list(MODEL_COLORS[:len(summary)])

    if bert_accuracy is not None:
        models.append("FinBERT")
        accs.append(bert_accuracy)
        colors.append(FINBERT_COLOR)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, accs, color=colors, width=0.5, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Current Diagnosis — Accuracy by Model", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_accuracy.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def _plot_metric_heatmap(cat_df, metric_key, title, filename, out_dir):
    """
    Generic heatmap for a given per-category metric.
    metric_key is 'f1', 'prec', or 'rec'.
    """
    model_labels = list(MODELS.values())
    cols = [f"{lbl}_{metric_key}" for lbl in model_labels if f"{lbl}_{metric_key}" in cat_df.columns]
    if not cols:
        return
    data = cat_df[cols].values.T
    cats = cat_df["category"].tolist()
    ns   = cat_df["n"].tolist()

    cmap = LinearSegmentedColormap.from_list("rg", ["#FCE4D6", "#FFEB9C", "#C6EFCE"])
    fig, ax = plt.subplots(figsize=(max(10, len(cats)*0.55), 3.5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(cats, ns)],
        rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(model_labels[:len(cols)], fontsize=10)

    for i in range(len(cols)):
        for j in range(len(cats)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 format=mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    out = os.path.join(out_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_category_heatmap(cat_df, out_dir):
    _plot_metric_heatmap(cat_df, "f1",
                         "Current Diagnosis — F1 per category (n≥3)",
                         "cur_dg_per_category_heatmap.png", out_dir)


def plot_category_heatmap_precision(cat_df, out_dir):
    _plot_metric_heatmap(cat_df, "prec",
                         "Current Diagnosis — Precision per category (n≥3)",
                         "cur_dg_per_category_precision.png", out_dir)


def plot_category_heatmap_recall(cat_df, out_dir):
    _plot_metric_heatmap(cat_df, "rec",
                         "Current Diagnosis — Recall per category (n≥3)",
                         "cur_dg_per_category_recall.png", out_dir)


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
        rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)

    for i in range(len(model_labels)):
        for j in range(len(cats)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 format=mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(f"Current Diagnosis — F1 per rare category (n≤{max_count})", fontsize=11, pad=10)
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
        ax.text(val + 0.2, i, str(val), va="center", fontsize=10)
    ax.set_xlabel("Count")
    ax.set_title(f"Current Diagnosis — Gold label distribution (top {top_n})", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Error analysis ────────────────────────────────────────────────────────────

def build_error_analysis(df, mapping, bert_df=None):
    """
    For each model, classify score=0 rows into:
      - false_normal   : gold != normaali, pred == normaali
      - false_positive : gold == normaali, pred != normaali
      - wrong_disease  : gold != normaali, pred != normaali
    Optionally includes FinBERT from a separate predictions file.
    """
    def count_errors(pairs):
        false_normal = false_positive = wrong_disease = 0
        for gold, pred, sc in pairs:
            if sc != 0 or gold is None or pred is None:
                continue
            if gold == "normaali" and pred != "normaali":
                false_positive += 1
            elif gold != "normaali" and pred == "normaali":
                false_normal += 1
            elif gold != "normaali" and pred != "normaali":
                wrong_disease += 1
        return false_normal, false_positive, wrong_disease

    rows = []
    for mk, label in MODELS.items():
        if COL_SCORE[mk] not in df.columns:
            continue
        pairs = get_pairs(df, mk, mapping)
        fn, fp, wd = count_errors(pairs)
        total = fn + fp + wd
        rows.append({
            "model": label, "false_normal": fn, "false_positive": fp,
            "wrong_disease": wd, "total_errors": total,
            "n_total": sum(1 for g, _, _ in pairs if g is not None),
        })

    if bert_df is not None:
        bert_df_clean = bert_df.copy()
        bert_df_clean.columns = [c.strip().lower() for c in bert_df_clean.columns]
        pairs = []
        for _, row in bert_df_clean.iterrows():
            gold = None
            gv = row.get("gold_norm")
            if gv is not None and str(gv).strip().lower() not in {"", "nan", "none"}:
                gold = str(gv).strip().lower()
            pred = None
            pv = row.get("bert_pred")
            if pv is not None and str(pv).strip().lower() not in {"", "nan", "none"}:
                pred = str(pv).strip().lower()
            sc_raw = row.get("score")
            sc = None
            if sc_raw is not None and str(sc_raw).strip().lower() not in {"", "nan", "none"}:
                try:
                    sc = int(float(str(sc_raw).replace(",", ".")))
                except:
                    sc = None
            pairs.append((gold, pred, sc))
        fn, fp, wd = count_errors(pairs)
        total = fn + fp + wd
        rows.append({
            "model": "FinBERT", "false_normal": fn, "false_positive": fp,
            "wrong_disease": wd, "total_errors": total,
            "n_total": sum(1 for g, _, _ in pairs if g is not None),
        })

    return pd.DataFrame(rows)


def plot_error_analysis(error_df, out_dir):
    """
    Stacked bar chart where bar height = proportion of all responses that were errors (0-1).
    Segments show each error type as proportion of all responses.
    Labels inside segments show that segment's proportion of all responses.
    Segment labels sum to the total bar height.
    """
    if error_df.empty:
        return

    labels  = error_df["model"].tolist()
    n_total = error_df["n_total"].values
    total   = error_df["total_errors"].values

    cats       = ["false_normal", "false_positive", "wrong_disease"]
    cat_labels = ["False normal", "False positive", "Wrong disease"]
    colors     = ["#C0533A", "#D4785C", "#E8A98A"]

    # Each segment height = cat_count / n_total (proportion of all responses, 0-1)
    pcts_of_total = {}
    for cat in cats:
        pcts_of_total[cat] = [
            (row / nt) if nt > 0 else 0
            for row, nt in zip(error_df[cat].values, n_total)
        ]

    x   = np.arange(len(labels))
    w   = 0.5
    fig, ax = plt.subplots(figsize=(10, 5))

    bottoms = np.zeros(len(labels))
    for cat, cat_label, color in zip(cats, cat_labels, colors):
        vals = np.array(pcts_of_total[cat])
        bars = ax.bar(x, vals, w, bottom=bottoms, label=cat_label, color=color)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.015:
                ax.text(x[i], b + v/2, f"{v:.2f}",
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        bottoms += vals

    # Total error rate above each bar
    for i, (b, t, nt) in enumerate(zip(bottoms, total, n_total)):
        error_rate = t / nt if nt > 0 else 0
        ax.text(x[i], b + 0.005, f"{error_rate:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("Error rate")
    ax.set_title("Current Diagnosis — Error Distribution by Model", fontsize=11, pad=10)

    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "cur_dg_error_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True, help="eval_v3_norm_J.xlsx")
    parser.add_argument("--mapping",   required=True, help="label_mapping_extended_eval.json")
    parser.add_argument("--output",    required=True, help="Output directory")
    parser.add_argument("--min_count", type=int, default=0)
    parser.add_argument("--bert",      default=None,
                        help="Optional FinBERT predictions Excel for error analysis")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading data...")
    df      = load_data(args.input)
    mapping = load_mapping(args.mapping)
    print(f"  {len(df)} rows, mapping: {len(mapping)} entries")

    # Summary
    print("\nComputing metrics...")
    summary = build_summary(df, mapping)
    print(summary[["model", "accuracy",
                   "macro_precision", "macro_recall", "macro_f1",
                   "micro_precision", "micro_recall", "micro_f1",
                   "weighted_precision", "weighted_recall", "weighted_f1",
                   "correct", "total"]].to_string(index=False))
    summary.to_csv(os.path.join(args.output, "cur_dg_summary.csv"), index=False)

    # Per-category table
    cat_df = build_per_cat_table(df, mapping, min_count=args.min_count)
    cat_df.to_csv(os.path.join(args.output, "cur_dg_per_category.csv"), index=False)

    # Plots
    plt.rcParams.update({"font.family": "Arial", "axes.grid": False})
    plot_f1_grouped(summary, args.output)
    plot_precision_grouped(summary, args.output)
    plot_recall_grouped(summary, args.output)
    # Load FinBERT accuracy from bert_summary if available
    bert_acc = None
    if args.bert:
        import glob, os as _os
        bert_out_dir = _os.path.dirname(args.bert)
        bert_summary_candidates = glob.glob(_os.path.join(bert_out_dir, "**", "bert_summary.csv"), recursive=True)
        if bert_summary_candidates:
            bs = pd.read_csv(bert_summary_candidates[0])
            bs.columns = [c.strip() for c in bs.columns]
            if "accuracy" in bs.columns:
                bert_acc = float(bs["accuracy"].values[0])
    plot_accuracy(summary, args.output, bert_accuracy=bert_acc)
    plot_category_heatmap(cat_df, args.output)
    plot_category_heatmap_precision(cat_df, args.output)
    plot_category_heatmap_recall(cat_df, args.output)
    plot_category_heatmap_rare(df, mapping, args.output)
    plot_distribution(df, args.output)

    # Error analysis
    print("\nComputing error analysis...")
    bert_df = None
    if args.bert:
        print(f"Loading FinBERT predictions: {args.bert}")
        bert_df = pd.read_excel(args.bert, engine="openpyxl", skiprows=1, header=0)
    error_df = build_error_analysis(df, mapping, bert_df=bert_df)
    print(error_df[["model", "false_normal", "false_positive",
                     "wrong_disease", "total_errors"]].to_string(index=False))
    error_df.to_csv(os.path.join(args.output, "cur_dg_error_analysis.csv"), index=False)
    plot_error_analysis(error_df, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()