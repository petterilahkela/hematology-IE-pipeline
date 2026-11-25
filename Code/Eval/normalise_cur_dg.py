"""
normalize_labels.py
===================
Normalises cur_dg and cur_dg_status fields in annotated data
using label_mapping_extended.json.

Usage:
    python normalize_labels.py \
        --input  MGG_check1_140726_revised.xlsx \
        --mapping label_mapping_extended.json \
        --output  MGG_check1_140726_normalized.xlsx
"""

import pandas as pd
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Annotated Excel file")
    parser.add_argument("--mapping", required=True, help="label_mapping_extended.json")
    parser.add_argument("--output",  required=True, help="Output Excel file")
    args = parser.parse_args()

    # Load mapping
    with open(args.mapping, encoding='utf-8') as f:
        mapping = json.load(f)

    cur_dg_map        = mapping['cur_dg_mapping']
    cur_dg_status_map = mapping['cur_dg_status_mapping']

    # Load data
    print(f"Loading {args.input} ...")
    df = pd.read_excel(args.input, header=1)
    print(f"  {len(df)} rows")

    # ── cur_dg ────────────────────────────────────────────────────────────────
    not_found_dg = set()

    def map_cur_dg(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return None
        v = str(val).strip()
        if v in cur_dg_map:
            return cur_dg_map[v]
        not_found_dg.add(v)
        return v  # keep original if not in mapping

    df['cur_dg'] = df['cur_dg'].apply(map_cur_dg)

    # ── cur_dg_status ─────────────────────────────────────────────────────────
    not_found_status = set()

    def map_cur_dg_status(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return None
        v = str(val).strip()
        if v in cur_dg_status_map:
            return cur_dg_status_map[v]
        not_found_status.add(v)
        return v  # keep original if not in mapping

    df['cur_dg_status'] = df['cur_dg_status'].apply(map_cur_dg_status)

        # ── Save with formatting ──────────────────────────────────────────────────
    df.to_excel(args.output, index=False, engine="openpyxl")

    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = load_workbook(args.output)
    ws = wb.active

    headers = [cell.value for cell in ws[1]]

    FILL_HDR     = PatternFill("solid", fgColor="1F4E79")
    FILL_CUR_DG  = PatternFill("solid", fgColor="D6E4F0")
    FILL_WARN    = PatternFill("solid", fgColor="FCE4D6")
    FONT_HDR     = Font(bold=True, color="FFFFFF", name="Arial")
    FONT_BODY    = Font(name="Arial", size=10)

    # Header row
    for cell in ws[1]:
        cell.fill = FILL_HDR
        cell.font = FONT_HDR
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Highlight cur_dg column
    cur_dg_col = headers.index("cur_dg") + 1 if "cur_dg" in headers else None

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = FONT_BODY
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if cur_dg_col and cell.column == cur_dg_col:
                val = str(cell.value or "")
                # Red if value was not normalised (still in not_found_dg)
                if val in not_found_dg:
                    cell.fill = FILL_WARN
                else:
                    cell.fill = FILL_CUR_DG

    # Column widths
    for col_idx, header in enumerate(headers, 1):
        col_letter = get_column_letter(col_idx)
        if header in ("referral", "report"):
            ws.column_dimensions[col_letter].width = 50
        elif header == "cur_dg":
            ws.column_dimensions[col_letter].width = 25
        else:
            ws.column_dimensions[col_letter].width = 18

    ws.freeze_panes = "A2"
    wb.save(args.output)
    print(f"Saved → {args.output}")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\ncur_dg unique values after normalization: {df['cur_dg'].nunique()}")
    print(f"cur_dg_status unique values after normalization: {df['cur_dg_status'].nunique()}")

    if not_found_dg:
        print(f"\nWARN: {len(not_found_dg)} cur_dg values not in mapping (kept original):")
        for v in sorted(not_found_dg):
            print(f"  '{v}'")

    if not_found_status:
        print(f"\nWARN: {len(not_found_status)} cur_dg_status values not in mapping (kept original):")
        for v in sorted(not_found_status):
            print(f"  '{v}'")

if __name__ == "__main__":
    main()
