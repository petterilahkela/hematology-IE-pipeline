"""
evaluate_bert.py
================
Computes macro/micro/weighted precision, recall, F1 and accuracy
from manually evaluated bert_predictions.xlsx.

Uses the same lenient scoring logic as analyse_cur_dg_v4.py.

Usage:
    python evaluate_bert.py \
        --input  "C:\\...\\bert_predictions.xlsx" \
        --output "C:\\...\\bert_analysis\\"
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_LABEL = "FinBERT"
MODEL_COLOR = "#2E75B6"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_excel(path, engine="openpyxl", skiprows=1)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def has_value(val):
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str):
        return val.strip().lower() not in {"", "nan", "none", "null"}
    return True


def clean(val):
    if not has_value(val):
        return None
    return str(val).strip().lower()

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_per_category(pairs):
    tp      = defaultdict(float)
    fp      = defaultdict(float)
    fn      = defaultdict(int)
    support = defaultdict(int)

    for gold, pred, score in pairs:
        if gold is not None:
            support[gold] += 1

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

    result = {}
    for cat in sorted(support.keys()):
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
    if not per_cat:
        return {k: 0.0 for k in [
            "macro_precision", "macro_recall", "macro_f1",
            "micro_precision", "micro_recall", "micro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
        ]}

    total_support = sum(v["support"] for v in per_cat.values())
    n_cats = len(per_cat)

    macro_f1        = sum(v["f1"]        for v in per_cat.values()) / n_cats
    macro_precision = sum(v["precision"] for v in per_cat.values()) / n_cats
    macro_recall    = sum(v["recall"]    for v in per_cat.values()) / n_cats

    if total_support > 0:
        weighted_f1        = sum(v["f1"]        * v["support"] for v in per_cat.values()) / total_support
        weighted_precision = sum(v["precision"] * v["support"] for v in per_cat.values()) / total_support
        weighted_recall    = sum(v["recall"]    * v["support"] for v in per_cat.values()) / total_support
    else:
        weighted_f1 = weighted_precision = weighted_recall = 0.0

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

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_summary(metrics, accuracy, n_correct, n_total, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_names = [
        "macro\nprecision", "macro\nrecall", "macro\nF1",
        "micro\nprecision", "micro\nrecall", "micro\nF1",
        "weighted\nprecision", "weighted\nrecall", "weighted\nF1",
        "accuracy"
    ]
    values = [
        metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"],
        metrics["micro_precision"], metrics["micro_recall"], metrics["micro_f1"],
        metrics["weighted_precision"], metrics["weighted_recall"], metrics["weighted_f1"],
        accuracy
    ]

    x = np.arange(len(metric_names))
    bars = ax.bar(x, values, color=MODEL_COLOR, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"FinBERT — cur_dg Performance\n(n={n_correct}/{n_total} correct, accuracy={accuracy:.3f})",
                 fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "bert_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_per_category(per_cat, out_dir, top_n=20):
    cats = sorted(per_cat.keys(), key=lambda c: per_cat[c]["support"], reverse=True)[:top_n]
    f1s  = [per_cat[c]["f1"] for c in cats]
    precs = [per_cat[c]["precision"] for c in cats]
    recs  = [per_cat[c]["recall"] for c in cats]
    sups  = [per_cat[c]["support"] for c in cats]

    x = np.arange(len(cats))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, precs, w, label="Precision", color="#9DB8C8")
    ax.bar(x,     recs,  w, label="Recall",    color="#5B7FA6")
    ax.bar(x + w, f1s,   w, label="F1",        color="#3D5A73")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(cats, sups)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"FinBERT — Per-category Precision, Recall & F1 (top {top_n} by support)",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, "bert_per_category.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial"})

    print(f"Loading: {args.input}")
    df = load_data(args.input)
    print(f"  {len(df)} rows")

    # Build pairs
    pairs = []
    for _, row in df.iterrows():
        gold  = clean(row.get("gold_norm"))
        pred  = clean(row.get("bert_pred"))
        sc_raw = row.get("score")
        sc = None
        if has_value(sc_raw):
            try:
                sc = int(float(str(sc_raw).replace(",", ".").strip()))
            except:
                sc = None
        pairs.append((gold, pred, sc))

    # Metrics
    per_cat = compute_per_category(pairs)
    metrics = aggregate_metrics(per_cat)

    n_total   = sum(1 for g, _, _ in pairs if g is not None)
    n_correct = sum(1 for _, _, sc in pairs if sc == 1)
    n_null_null = sum(1 for g, p, _ in pairs if g is None and p is None)
    accuracy  = round((n_correct + n_null_null) / len(pairs), 3) if pairs else 0.0

    # Print
    print(f"\n{'='*50}")
    print(f"FinBERT cur_dg Results")
    print(f"{'='*50}")
    print(f"  Correct:          {n_correct}/{n_total}")
    print(f"  Accuracy:         {accuracy}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")

    # Plots
    plot_summary(metrics, accuracy, n_correct, n_total, args.output)
    plot_per_category(per_cat, args.output)

    # Save CSV
    summary = {"model": "FinBERT", "accuracy": accuracy, "correct": n_correct,
               "total": n_total, **metrics}
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output, "bert_summary.csv"), index=False)

    per_cat_rows = [{"category": c, **v} for c, v in per_cat.items()]
    pd.DataFrame(per_cat_rows).to_csv(
        os.path.join(args.output, "bert_per_category.csv"), index=False)

    print(f"\nSaved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
