"""
prepare_for_evaluation_v3.py
========================
Creates an evaluation Excel file for manual scoring of V3 model results.
One sheet per field, all models side by side.
Auto-scores exact matches and null cases.
Adds an unknown cur_dg values sheet at the end.

Usage:
    python prepare_evaluation_v3.py \
        --test    "C:\\...\\test_v3.xlsx" \
        --llama   "C:\\...\\results_llama_v3.xlsx" \
        --poro2   "C:\\...\\results_poro2_v3.xlsx" \
        --ft      "C:\\...\\results_poro2_ft_v3.xlsx" \
        --rag     "C:\\...\\results_poro2_ft_rag_v3.xlsx" \
        --mapping "C:\\...\\label_mapping_extended_eval.json" \
        --output  "C:\\...\\eval_v3_norm.xlsx" \
"""

import argparse
import json
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────

FIELDS = [
    "cur_dg",
    "suspected_dg",
    "prev_cur_dg",
    "prev_cur_dg_status",
    "treatment",
    "cur_dg_status",
]

MODEL_KEYS   = ["llama", "poro2", "ft", "rag"]
MODEL_LABELS = {
    "llama": "Llama 3.1 8B",
    "poro2": "Poro2 Instruct",
    "ft":    "Poro2 + FT",
    "rag":   "Poro2 + FT RAG",
}

# ── Styling ───────────────────────────────────────────────────────────────────

thin   = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

FONT_HDR  = Font(name="Arial", bold=True, size=9, color="FFFFFF")
FONT_BODY = Font(name="Arial", size=9)
FONT_BOLD = Font(name="Arial", bold=True, size=9)
FONT_WARN = Font(name="Arial", size=9, color="C00000")

ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
ALIGN_LEFT   = Alignment(horizontal="left",   vertical="top",    wrap_text=True)
ALIGN_WRAP   = Alignment(vertical="top", wrap_text=True)

C_DARK    = "1F4E79"
C_MID     = "2E75B6"
C_GREEN   = "C6EFCE"
C_RED     = "FCE4D6"
C_WHITE   = "FFFFFF"
C_GREY    = "F2F2F2"
C_YELLOW  = "FFF2CC"
C_UNKNOWN = "FFE699"


def _fill(c): return PatternFill("solid", fgColor=c)


def _hdr(ws, r, c, v, bg=C_DARK):
    cell = ws.cell(row=r, column=c, value=v)
    cell.fill = _fill(bg); cell.font = FONT_HDR
    cell.alignment = ALIGN_CENTER; cell.border = BORDER
    return cell


def _body(ws, r, c, v, bg=C_WHITE, bold=False, align=None, warn=False):
    cell = ws.cell(row=r, column=c, value=v)
    cell.fill = _fill(bg)
    cell.font = FONT_WARN if warn else (FONT_BOLD if bold else FONT_BODY)
    cell.alignment = align or ALIGN_CENTER
    cell.border = BORDER
    return cell


def _col(ws, col, w):
    ws.column_dimensions[get_column_letter(col)].width = w


# ── Data loading ──────────────────────────────────────────────────────────────

def load_df(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return None if s.lower() in {"", "nan", "none", "null"} else s


def normalise_cur_dg(val, mapping):
    v = clean(val)
    if v is None:
        return None, False
    raw = str(val).strip()
    if raw in mapping:
        return mapping[raw], True
    for k, mapped in mapping.items():
        if k.lower() == raw.lower():
            return mapped, True
    return v.lower(), False   # unknown — not in mapping


def normalise_other(val):
    v = clean(val)
    if v is None:
        return None
    return v.strip().lower()


# ── Auto-scoring ──────────────────────────────────────────────────────────────

def score_cur_dg(pred_norm, gold_norm, accepted_list, pred_in_mapping):
    """Return (score, bg_color)."""
    if gold_norm is None and pred_norm is None:
        return None, C_GREY

    if gold_norm is not None and pred_norm is None:
        return 0, C_RED

    if gold_norm is None and pred_norm is not None:
        return 0, C_RED

    if not pred_in_mapping:
        return None, C_UNKNOWN   # unknown — manual review

    if accepted_list:
        accepted_norm = [a.strip().lower() for a in accepted_list]
        if pred_norm in accepted_norm:
            return 1, C_GREEN

    if pred_norm == gold_norm:
        return 1, C_GREEN

    return None, C_WHITE   # manual review


def score_other(pred_norm, gold_norm):
    """Return (score, bg_color)."""
    if gold_norm is None and pred_norm is None:
        return None, C_GREY

    if gold_norm is not None and pred_norm is None:
        return 0, C_RED

    if gold_norm is None and pred_norm is not None:
        return 0, C_RED

    if pred_norm == gold_norm:
        return 1, C_GREEN

    return None, C_WHITE   # manual review


# ── Build field sheet ─────────────────────────────────────────────────────────

def write_field_sheet(wb, field, rows, model_keys, model_labels, is_cur_dg):
    ws = wb.create_sheet(field)
    ws.freeze_panes = "A3"

    # Column count
    n_model_cols = len(model_keys) * 2
    extra_cols   = 2 if is_cur_dg else 1
    total_cols   = 3 + extra_cols + n_model_cols

    # Row 1: title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    tc = ws.cell(1, 1, f"Evaluation — {field}")
    tc.fill = _fill(C_DARK); tc.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    tc.alignment = ALIGN_CENTER
    ws.row_dimensions[1].height = 20

    # Row 2: headers
    col = 1
    _hdr(ws, 2, col, "id");       col += 1
    _hdr(ws, 2, col, "referral"); col += 1
    _hdr(ws, 2, col, "report");   col += 1

    if is_cur_dg:
        _hdr(ws, 2, col, "accepted_labels", bg=C_MID); col += 1
        _hdr(ws, 2, col, "gold_norm",        bg=C_MID); col += 1
    else:
        _hdr(ws, 2, col, "gold", bg=C_MID); col += 1

    for mk in model_keys:
        lbl = model_labels[mk]
        _hdr(ws, 2, col, f"{lbl} output", bg="2E75B6"); col += 1
        _hdr(ws, 2, col, f"{lbl} score",  bg="2E75B6"); col += 1

    ws.row_dimensions[2].height = 20

    # Data rows
    for r_idx, row in enumerate(rows, start=3):
        bg_row = C_GREY if r_idx % 2 == 0 else C_WHITE
        ws.row_dimensions[r_idx].height = 20

        col = 1
        _body(ws, r_idx, col, row["id"],       bg=bg_row, bold=True); col += 1
        _body(ws, r_idx, col, row["referral"], bg=bg_row, align=ALIGN_WRAP); col += 1
        _body(ws, r_idx, col, row["report"],   bg=bg_row, align=ALIGN_WRAP); col += 1

        if is_cur_dg:
            _body(ws, r_idx, col, row.get("accepted_labels_str", ""),
                  bg="EBF5FB", align=ALIGN_WRAP); col += 1
            _body(ws, r_idx, col, row.get("gold_norm", ""),
                  bg="D6E4F0", align=ALIGN_LEFT, bold=True); col += 1
        else:
            _body(ws, r_idx, col, row.get(f"gold_{field}", ""),
                  bg="EBF5FB", align=ALIGN_LEFT); col += 1

        for mk in model_keys:
            pred   = row.get(f"{mk}_output", "")
            score  = row.get(f"{mk}_score")
            bg_c   = row.get(f"{mk}_bg", bg_row)
            in_map = row.get(f"{mk}_in_mapping", True)

            _body(ws, r_idx, col, pred, bg=bg_c, align=ALIGN_LEFT,
                  warn=(not in_map and pred not in (None, ""))); col += 1

            score_bg = C_GREEN if score == 1 else (C_RED if score == 0 else C_WHITE)
            if bg_c == C_GREY:
                score_bg = C_GREY
            _body(ws, r_idx, col, score, bg=score_bg); col += 1

    # Auto filter
    ws.auto_filter.ref = f"A2:{get_column_letter(total_cols)}{len(rows)+2}"

    # Column widths
    col = 1
    _col(ws, col, 12); col += 1   # id
    _col(ws, col, 35); col += 1   # referral
    _col(ws, col, 35); col += 1   # report
    if is_cur_dg:
        _col(ws, col, 20); col += 1  # accepted_labels
        _col(ws, col, 18); col += 1  # gold_norm
    else:
        _col(ws, col, 20); col += 1  # gold
    for _ in model_keys:
        _col(ws, col, 18); col += 1  # output
        _col(ws, col, 8);  col += 1  # score


# ── Unknown values sheet ──────────────────────────────────────────────────────

def write_unknown_sheet(wb, unknowns_per_model, model_labels):
    ws = wb.create_sheet("Unknown cur_dg values")

    ws.merge_cells("A1:C1")
    tc = ws.cell(1, 1, "Unknown cur_dg values — not in label_mapping_extended_eval.json")
    tc.fill = _fill(C_DARK); tc.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    tc.alignment = ALIGN_CENTER
    ws.row_dimensions[1].height = 20

    _hdr(ws, 2, 1, "Model")
    _hdr(ws, 2, 2, "Unknown value")
    _hdr(ws, 2, 3, "Count")

    r = 3
    for mk in MODEL_KEYS:
        label    = model_labels[mk]
        unknowns = unknowns_per_model.get(mk, {})
        if not unknowns:
            continue
        for val, cnt in sorted(unknowns.items(), key=lambda x: -x[1]):
            bg = C_YELLOW if r % 2 == 0 else C_WHITE
            _body(ws, r, 1, label, bg=bg, bold=True)
            _body(ws, r, 2, val,   bg=bg, align=ALIGN_LEFT)
            _body(ws, r, 3, cnt,   bg=bg)
            r += 1

    _col(ws, 1, 20)
    _col(ws, 2, 40)
    _col(ws, 3, 8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    required=True)
    parser.add_argument("--llama",   required=True)
    parser.add_argument("--poro2",   required=True)
    parser.add_argument("--ft",      required=True)
    parser.add_argument("--rag",     required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()

    # Load mapping
    with open(args.mapping, encoding="utf-8") as f:
        mapping = json.load(f)["cur_dg_mapping"]

    # Load data
    print("Loading data...")
    test = load_df(args.test)
    dfs  = {
        "llama": load_df(args.llama),
        "poro2": load_df(args.poro2),
        "ft":    load_df(args.ft),
        "rag":   load_df(args.rag),
    }
    print(f"  Test rows: {len(test)}")
    for mk, df in dfs.items():
        print(f"  {MODEL_LABELS[mk]} rows: {len(df)}")

    # Build sheets
    unknowns_per_model = {mk: defaultdict(int) for mk in MODEL_KEYS}
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    for field in FIELDS:
        print(f"Building sheet: {field}")
        is_cur_dg = (field == "cur_dg")
        rows = []

        for i in range(len(test)):
            test_row = test.iloc[i]
            row_id   = clean(test_row.get("id"))

            accepted_list = None
            accepted_str  = ""
            if is_cur_dg:
                raw_acc = clean(test_row.get("accepted_cur_dg_labels"))
                if raw_acc:
                    try:
                        accepted_list = json.loads(raw_acc)
                        accepted_str  = ", ".join(accepted_list)
                    except Exception:
                        accepted_list = [raw_acc]
                        accepted_str  = raw_acc

                gold_val  = accepted_list[0] if accepted_list else test_row.get("cur_dg")
                gold_norm, _ = normalise_cur_dg(gold_val, mapping)
            else:
                gold_norm = normalise_other(test_row.get(field))

            row = {
                "id":                  row_id,
                "referral":            clean(test_row.get("referral")),
                "report":              clean(test_row.get("report")),
                f"gold_{field}":       normalise_other(test_row.get(field)),
                "accepted_labels_str": accepted_str,
                "gold_norm":           gold_norm,
            }

            for mk in MODEL_KEYS:
                pred_raw = clean(dfs[mk].iloc[i].get(field))

                if is_cur_dg:
                    pred_norm, in_map = normalise_cur_dg(pred_raw, mapping)
                    if pred_raw is not None and not in_map:
                        unknowns_per_model[mk][pred_raw] += 1
                    score, bg = score_cur_dg(pred_norm, gold_norm, accepted_list, in_map)
                else:
                    pred_norm = normalise_other(pred_raw)
                    in_map    = True
                    score, bg = score_other(pred_norm, gold_norm)

                row[f"{mk}_output"]     = pred_raw
                row[f"{mk}_score"]      = score
                row[f"{mk}_bg"]         = bg
                row[f"{mk}_in_mapping"] = in_map

            rows.append(row)

        write_field_sheet(wb, field, rows, MODEL_KEYS, MODEL_LABELS, is_cur_dg,
)

    write_unknown_sheet(wb, unknowns_per_model, MODEL_LABELS)

    wb.save(args.output)
    print(f"\nSaved → {args.output}")
    print(f"Sheets: {wb.sheetnames}")


if __name__ == "__main__":
    main()
