"""
prepare_bert_data.py
====================
Prepares training, validation and test data for BERT-based cur_dg classification.

Reads train/val/test Excel files, extracts report text and gold cur_dg label.
Train/val files already contain pre-normalised cur_dg labels from LLM fine-tuning
data preparation — no mapping needed.

Usage:
    python prepare_bert_data.py \
        --train   "C:\\...\\train_v3.xlsx" \
        --val     "C:\\...\\val_v3.xlsx" \
        --test    "C:\\...\\test_v3.xlsx" \
        --output  "C:\\...\\bert_data\\"
"""

import argparse
import json
import math
import os
import pandas as pd


def clean(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return None
    return s


def load_and_prepare(path, split_name, label_col="cur_dg"):
    df = pd.read_excel(path, engine="openpyxl")
    print(f"\n{split_name}: {len(df)} rows loaded")

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        report    = clean(row.get("report"))
        gold_norm = clean(row.get(label_col))

        if not report or gold_norm is None:
            skipped += 1
            continue

        # Use YHTEENVETO section if available, otherwise last 1500 chars
        if "YHTEENVETO" in report:
            report = report[report.find("YHTEENVETO"):]
        else:
            report = report[-1500:]

        rows.append({
            "id":    clean(row.get("id")) or "",
            "text":  report,
            "label": gold_norm,
        })

    df_out = pd.DataFrame(rows)
    print(f"  Kept:    {len(df_out)} rows")
    print(f"  Skipped: {skipped} rows (null label or report)")
    return df_out


def print_distribution(df, split_name, top_n=25):
    counts = df["label"].value_counts()
    print(f"\n{split_name} label distribution (top {top_n}):")
    print(f"  Total categories: {len(counts)}")
    print(f"  Total examples:   {len(df)}")
    print(f"  {'Category':<35} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*50}")
    for label, count in counts.head(top_n).items():
        pct = count / len(df) * 100
        print(f"  {label:<35} {count:>6}  {pct:>5.1f}%")
    rare = counts[counts < 3]
    if len(rare) > 0:
        print(f"\n  WARNING: {len(rare)} categories with < 3 examples:")
        for label, count in rare.items():
            print(f"    {label}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     required=True)
    parser.add_argument("--val",       required=True)
    parser.add_argument("--test",      required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--label_col", default="cur_dg",
                        help="Column name for gold label (default: cur_dg)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    train_df = load_and_prepare(args.train, "TRAIN", args.label_col)
    val_df   = load_and_prepare(args.val,   "VAL",   args.label_col)
    test_df  = load_and_prepare(args.test,  "TEST",  args.label_col)

    # Build label set from training data only
    train_labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(train_labels)}
    id2label = {str(i): label for label, i in label2id.items()}

    print(f"\nTotal unique categories in training: {len(label2id)}")

    # Check for unseen labels in val/test
    val_unseen  = set(val_df["label"].unique())  - set(train_labels)
    test_unseen = set(test_df["label"].unique()) - set(train_labels)
    if val_unseen:
        print(f"\nWARNING: {len(val_unseen)} val labels not in training:  {val_unseen}")
    if test_unseen:
        print(f"WARNING: {len(test_unseen)} test labels not in training: {test_unseen}")

    print_distribution(train_df, "TRAIN")
    print_distribution(val_df,   "VAL")
    print_distribution(test_df,  "TEST")

    train_df.to_csv(os.path.join(args.output, "bert_train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(  os.path.join(args.output, "bert_val.csv"),   index=False, encoding="utf-8")
    test_df.to_csv( os.path.join(args.output, "bert_test.csv"),  index=False, encoding="utf-8")

    with open(os.path.join(args.output, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output}")
    print("  bert_train.csv, bert_val.csv, bert_test.csv")
    print("  label2id.json, id2label.json")
    print("\nDone!")


if __name__ == "__main__":
    main()