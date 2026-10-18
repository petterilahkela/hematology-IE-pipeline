"""
analyse_all_fields_v3.py
========================
Computes for each free-text field (suspected_dg, prev_cur_dg, treatment):
- Overall accuracy (null-null = correct)
- Extraction Recall (only when gold is non-null)
- Extraction Precision (only when model output is non-null)
- Extraction F1 (binary F1 = 2PR/(P+R) on the extraction metrics)

Note: cur_dg is NOT included here — it is evaluated separately by
analyse_cur_dg_v3.py, which computes macro/micro/weighted F1 using
the categorical label mapping.

Usage:
    python analyse_all_fields_v3.py \
        --input  "C:\\...\\eval_v3_norm_J.xlsx" \
        --output "C:\\...\\all_fields_analysis_v3\\"
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────

FIELDS = [
    "suspected_dg",
    "prev_cur_dg",
    "treatment",
]

# cur_dg is included only for the accuracy summary plot
FIELDS_WITH_CUR_DG = FIELDS + ["cur_dg"]

FIELD_LABELS = {
    "cur_dg":       "Current diagnosis",
    "suspected_dg": "Suspected diagnosis",
    "prev_cur_dg":  "Previous diagnosis",
    "treatment":    "Treatment",
}

# Gold column name per field
GOLD_COL = {
    "cur_dg":       "gold_norm",   # cur_dg uses gold_norm column
    "suspected_dg": "gold",
    "prev_cur_dg":  "gold",
    "treatment":    "gold",
}

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

# One color per model for summary plots
MODEL_COLORS = ["#9DB8C8", "#7B9DBF", "#5B7FA6", "#3D5A73"]

# Slate shades for per-field metric bars
COLOR_ACCURACY  = "#3A7D5E"  # dark muted green
COLOR_RECALL    = "#6BAF92"  # medium muted green
COLOR_PRECISION = "#B8DDD0"  # light muted green
COLOR_F1        = "#2D5C4A"  # dark green for F1

# ── Data loading ──────────────────────────────────────────────────────────────

def load_sheet(path, sheet):
    """Load a sheet — row 0 is title, row 1 is header."""
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", skiprows=[0], header=0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def has_value(val):
    """Return True if val is a non-empty, non-null value."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str):
        cleaned = val.strip()
        return cleaned != "" and cleaned.lower() not in {"nan", "none", "null"}
    return True


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df, field):
    gold_col = GOLD_COL[field]

    if gold_col not in df.columns:
        print(f"  WARNING: gold column '{gold_col}' not found for {field}")
        return {}

    gold_raw = df[gold_col]
    n_total  = len(df)
    gold_has_value = [has_value(v) for v in gold_raw]

    results = {}

    for mk, label in MODELS.items():
        score_col  = COL_SCORE[mk]
        output_col = COL_OUTPUT[mk]

        if score_col not in df.columns:
            print(f"  WARNING: score column '{score_col}' not found for {label}")
            continue
        if output_col not in df.columns:
            print(f"  WARNING: output column '{output_col}' not found for {label}")
            continue

        scores_raw  = df[score_col]
        outputs_raw = df[output_col]

        n_null_null       = 0
        n_gold_non_null   = 0
        n_model_non_null  = 0
        correct_overall   = 0
        correct_active    = 0
        correct_extracted = 0

        for i in range(n_total):
            gold_has_val  = gold_has_value[i]
            model_has_val = has_value(outputs_raw.iloc[i])

            # Parse score
            sc_raw = scores_raw.iloc[i]
            sc = None
            if has_value(sc_raw):
                try:
                    sc = int(float(str(sc_raw).replace(",", ".").strip()))
                except (ValueError, TypeError):
                    sc = None

            # 1. Overall Accuracy — null-null counts as correct
            if sc == 1 or sc is None:
                correct_overall += 1

            # 2. Null-null count
            if (not gold_has_val) and (not model_has_val):
                n_null_null += 1

            # 3. Extraction Recall — only when gold has a value
            if gold_has_val:
                n_gold_non_null += 1
                if sc == 1:
                    correct_active += 1

            # 4. Extraction Precision — only when model output has a value
            if model_has_val:
                n_model_non_null += 1
                if sc == 1:
                    correct_extracted += 1

        accuracy             = correct_overall   / n_total          if n_total          > 0 else 0.0
        extraction_recall    = correct_active    / n_gold_non_null  if n_gold_non_null  > 0 else 0.0
        extraction_precision = correct_extracted / n_model_non_null if n_model_non_null > 0 else 0.0

        # Binary Extraction F1 = harmonic mean of precision and recall
        if (extraction_precision + extraction_recall) > 0:
            extraction_f1 = (2 * extraction_precision * extraction_recall /
                             (extraction_precision + extraction_recall))
        else:
            extraction_f1 = 0.0

        results[label] = {
            "accuracy":             round(accuracy,             4),
            "extraction_recall":    round(extraction_recall,    4),
            "extraction_precision": round(extraction_precision, 4),
            "extraction_f1":        round(extraction_f1,        4),
            "n_total":              n_total,
            "n_null_null":          n_null_null,
            "n_gold_non_null":      n_gold_non_null,
            "n_model_non_null":     n_model_non_null,
            "correct_overall":      correct_overall,
            "correct_active":       correct_active,
            "correct_extracted":    correct_extracted,
        }

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_field(field, results, out_dir):
    """Per-field grouped bar: Accuracy / Recall / Precision / F1."""
    if not results:
        return

    labels    = list(results.keys())
    accuracy  = [results[l]["accuracy"]             for l in labels]
    recall    = [results[l]["extraction_recall"]    for l in labels]
    precision = [results[l]["extraction_precision"] for l in labels]
    f1        = [results[l]["extraction_f1"]        for l in labels]

    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - 1.5*w, accuracy,  w, label="Overall Accuracy (incl. null-null)",   color=COLOR_ACCURACY)
    b2 = ax.bar(x - 0.5*w, recall,    w, label="Extraction Recall (gold non-null)",     color=COLOR_RECALL)
    b3 = ax.bar(x + 0.5*w, precision, w, label="Extraction Precision (model non-null)", color=COLOR_PRECISION)
    b4 = ax.bar(x + 1.5*w, f1,        w, label="Extraction F1",                          color=COLOR_F1)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"{FIELD_LABELS.get(field, field)} — Accuracy, Recall, Precision & F1", fontsize=11, pad=10)
    ax.legend(fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, f"{field}_metrics.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_summary(all_results, metric_key, ylabel, title, filename, out_dir, fields_override=None):
    """Generic summary bar chart across all fields and models."""
    fields       = fields_override if fields_override is not None else list(all_results.keys())
    model_labels = list(MODELS.values())

    x       = np.arange(len(fields))
    w       = 0.2
    offsets = np.linspace(-(len(model_labels)-1)/2 * w,
                           (len(model_labels)-1)/2 * w,
                           len(model_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, color) in enumerate(zip(model_labels, MODEL_COLORS)):
        vals = [all_results[f].get(label, {}).get(metric_key, 0) for f in fields]
        bars = ax.bar(x + offsets[i], vals, w, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_LABELS.get(f, f) for f in fields], fontsize=10, rotation=0, ha="center")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 1.06))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, filename)
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

    all_results  = {}
    summary_rows = []

    # Compute accuracy-only for cur_dg
    print(f"\nProcessing: cur_dg (accuracy only)")
    try:
        df_cur_dg = load_sheet(args.input, "cur_dg")
        cur_dg_results = {}
        for mk, label in MODELS.items():
            score_col  = COL_SCORE[mk]
            output_col = COL_OUTPUT[mk]
            if score_col not in df_cur_dg.columns or output_col not in df_cur_dg.columns:
                continue
            n_total = len(df_cur_dg)
            correct_overall = 0
            for i in range(n_total):
                sc_raw = df_cur_dg[score_col].iloc[i]
                sc = None
                if has_value(sc_raw):
                    try:
                        sc = int(float(str(sc_raw).replace(",", ".").strip()))
                    except (ValueError, TypeError):
                        sc = None
                gold_has_val  = has_value(df_cur_dg[GOLD_COL["cur_dg"]].iloc[i])
                model_has_val = has_value(df_cur_dg[output_col].iloc[i])
                if sc == 1 or (not gold_has_val and not model_has_val):
                    correct_overall += 1
            cur_dg_results[label] = {"accuracy": round(correct_overall / n_total, 4)}
        all_results["cur_dg"] = cur_dg_results
        print(f"  Done.")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Compute all metrics for other fields
    for field in FIELDS:
        print(f"\nProcessing: {field}")
        try:
            df = load_sheet(args.input, field)
        except Exception as e:
            print(f"  ERROR loading sheet: {e}")
            continue

        print(f"  {len(df)} rows")
        results = compute_metrics(df, field)
        all_results[field] = results

        print(f"  {'Model':<20} {'Overall Acc':>12} {'Extract Recall':>16} "
              f"{'Extract Prec':>14} {'Extract F1':>12} "
              f"{'N total':>8} {'N null-null':>12} {'N gold non-null':>16}")
        for label, m in results.items():
            print(f"  {label:<20} {m['accuracy']:>12.4f} {m['extraction_recall']:>16.4f} "
                  f"{m['extraction_precision']:>14.4f} {m['extraction_f1']:>12.4f} "
                  f"{m['n_total']:>8} {m['n_null_null']:>12} "
                  f"{m['n_gold_non_null']:>16}")
            summary_rows.append({"field": field, "model": label, **m})

        plot_field(field, results, args.output)

    # Accuracy summary — includes cur_dg
    plot_summary(all_results,
                 metric_key="accuracy",
                 ylabel="Overall Accuracy (incl. null-null)",
                 title="All fields — Overall Accuracy by Model",
                 filename="all_fields_accuracy.png",
                 out_dir=args.output,
                 fields_override=FIELDS_WITH_CUR_DG)

    plot_summary(all_results,
                 metric_key="extraction_recall",
                 ylabel="Extraction Recall (gold non-null)",
                 title="Extraction Recall by Model",
                 filename="all_fields_extraction_recall.png",
                 out_dir=args.output,
                 fields_override=FIELDS)

    plot_summary(all_results,
                 metric_key="extraction_precision",
                 ylabel="Extraction Precision (model non-null)",
                 title="Extraction Precision by Model",
                 filename="all_fields_extraction_precision.png",
                 out_dir=args.output,
                 fields_override=FIELDS)

    plot_summary(all_results,
                 metric_key="extraction_f1",
                 ylabel="Extraction F1 (binary)",
                 title="Extraction F1 by Model",
                 filename="all_fields_extraction_f1.png",
                 out_dir=args.output,
                 fields_override=FIELDS)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output, "all_fields_summary.csv"), index=False)
    print(f"\nSaved: {os.path.join(args.output, 'all_fields_summary.csv')}")
    print("\nDone!")


if __name__ == "__main__":
    main()