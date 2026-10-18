"""
predict_bert.py
===============
Runs inference with the fine-tuned BERT cur_dg classifier on the test set
and produces an Excel file for manual lenient evaluation.

Auto-scores exact matches (score=1) and cases where both gold and pred
are null (score=None, grey). All mismatches are left blank for manual review.

Usage:
    python predict_bert.py \
        --model  "/mnt/data/petteril/models/bert-cur-dg-v1" \
        --test   "/mnt/data/petteril/V3/Data/bert_data/bert_test.csv" \
        --gold   "/mnt/data/petteril/V3/Data/test_v3.xlsx" \
        --output "/mnt/data/petteril/V3/Results/bert_predictions.xlsx"
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

MAX_LENGTH = 512
BATCH_SIZE = 32

# ── Styling (same as prepare_evaluation_v3) ───────────────────────────────────

thin   = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

FONT_HDR  = Font(name="Arial", bold=True, size=9, color="FFFFFF")
FONT_BODY = Font(name="Arial", size=9)
FONT_BOLD = Font(name="Arial", bold=True, size=9)

ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
ALIGN_LEFT   = Alignment(horizontal="left",   vertical="top",    wrap_text=True)

C_DARK   = "1F4E79"
C_GREEN  = "C6EFCE"
C_RED    = "FCE4D6"
C_WHITE  = "FFFFFF"
C_GREY   = "F2F2F2"

def _fill(c): return PatternFill("solid", fgColor=c)
def _hdr(ws, r, c, v, bg=C_DARK):
    cell = ws.cell(row=r, column=c, value=v)
    cell.fill = _fill(bg); cell.font = FONT_HDR
    cell.alignment = ALIGN_CENTER; cell.border = BORDER
def _body(ws, r, c, v, bg=C_WHITE, bold=False, align=None):
    cell = ws.cell(row=r, column=c, value=v)
    cell.fill = _fill(bg)
    cell.font = FONT_BOLD if bold else FONT_BODY
    cell.alignment = align or ALIGN_CENTER
    cell.border = BORDER
def _col(ws, col, w):
    ws.column_dimensions[get_column_letter(col)].width = w

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TextDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return None
    return s

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to fine-tuned BERT model")
    parser.add_argument("--test",   required=True, help="bert_test.csv from prepare_bert_data.py")
    parser.add_argument("--gold",   required=True, help="test_v3.xlsx for referral text")
    parser.add_argument("--output", required=True, help="Output Excel file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    with open(os.path.join(args.model, "id2label.json"), encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    print(f"  Categories: {len(id2label)}")

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"\nLoading test data from {args.test} ...")
    test_df = pd.read_csv(args.test, encoding="utf-8")
    print(f"  {len(test_df)} rows")

    # Load gold xlsx for referral text
    gold_df = pd.read_excel(args.gold, engine="openpyxl")
    gold_df.columns = [c.strip().lower() for c in gold_df.columns]

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\nRunning inference...")
    texts = test_df["text"].tolist()
    dataset = TextDataset(texts, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.max(dim=-1).values.cpu().numpy())

    pred_labels = [id2label[p] for p in all_preds]

    # ── Build results ─────────────────────────────────────────────────────────
    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        row_id    = clean(row.get("id"))
        gold_norm = clean(row.get("label"))
        bert_pred = pred_labels[i]
        confidence = round(float(all_probs[i]), 3)

        # Get referral text from original test xlsx
        gold_row = gold_df[gold_df["id"].astype(str).str.strip() == str(row_id).strip()]
        referral = clean(gold_row["referral"].values[0]) if len(gold_row) > 0 else None

        # Auto-score
        if gold_norm == bert_pred:
            score = 1
            bg    = C_GREEN
        else:
            score = None   # manual review needed
            bg    = C_WHITE

        rows.append({
            "id":         row_id,
            "referral":   referral,
            "report":     clean(row.get("text")),
            "gold_norm":  gold_norm,
            "bert_pred":  bert_pred,
            "confidence": confidence,
            "score":      score,
            "_bg":        bg,
        })

    # ── Write Excel ───────────────────────────────────────────────────────────
    print(f"\nWriting Excel to {args.output} ...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "cur_dg_bert"
    ws.freeze_panes = "A3"

    total_cols = 7

    # Row 1: title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    tc = ws.cell(1, 1, "BERT cur_dg Evaluation — Manual Lenient Scoring")
    tc.fill = _fill(C_DARK); tc.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    tc.alignment = ALIGN_CENTER
    ws.row_dimensions[1].height = 20

    # Row 2: headers
    for col, hdr in enumerate(
        ["id", "referral", "report", "gold_norm", "bert_pred", "confidence", "score"], 1
    ):
        _hdr(ws, 2, col, hdr)
    ws.row_dimensions[2].height = 20

    # Data rows
    n_correct = 0
    n_manual  = 0
    for r_idx, row in enumerate(rows, start=3):
        bg = C_GREY if r_idx % 2 == 0 else C_WHITE
        ws.row_dimensions[r_idx].height = 20

        _body(ws, r_idx, 1, row["id"],         bg=bg, bold=True)
        _body(ws, r_idx, 2, row["referral"],    bg=bg, align=ALIGN_LEFT)
        _body(ws, r_idx, 3, row["report"],      bg=bg, align=ALIGN_LEFT)
        _body(ws, r_idx, 4, row["gold_norm"],   bg="D6E4F0", bold=True, align=ALIGN_LEFT)
        _body(ws, r_idx, 5, row["bert_pred"],   bg=row["_bg"], align=ALIGN_LEFT)
        _body(ws, r_idx, 6, row["confidence"],  bg=bg)

        score    = row["score"]
        score_bg = C_GREEN if score == 1 else (C_RED if score == 0 else C_WHITE)
        if bg == C_GREY and score is None:
            score_bg = C_GREY
        _body(ws, r_idx, 7, score, bg=score_bg)

        if score == 1:
            n_correct += 1
        elif score is None:
            n_manual += 1

    # Column widths
    _col(ws, 1, 12)
    _col(ws, 2, 35)
    _col(ws, 3, 40)
    _col(ws, 4, 20)
    _col(ws, 5, 20)
    _col(ws, 6, 12)
    _col(ws, 7, 8)

    ws.auto_filter.ref = f"A2:G{len(rows)+2}"

    wb.save(args.output)

    print(f"\nDone → {args.output}")
    print(f"  Total:          {len(rows)}")
    print(f"  Auto-correct:   {n_correct} ({100*n_correct//len(rows)}%)")
    print(f"  Manual review:  {n_manual}")


if __name__ == "__main__":
    main()
