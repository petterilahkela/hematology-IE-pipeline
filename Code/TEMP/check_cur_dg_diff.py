"""
check_cur_dg_diff.py
====================
Compares cur_dg values between J's train/val/test splits and our
MGG_normalised_v3.xlsx, matched by id column.

Usage:
    python check_cur_dg_diff.py
"""

import pandas as pd

MY_FILE = r"C:\Users\pette\Desktop\Projekti\V3\Data\MGG_normalised_v3.xlsx"

J_FILES = {
    "train": r"C:\Users\pette\Desktop\Projekti\V3\Data\J_DATA\train_full_columns_120826.xlsx",
    "val":   r"C:\Users\pette\Desktop\Projekti\V3\Data\J_DATA\valid_full_columns_120826.xlsx",
    "test":  r"C:\Users\pette\Desktop\Projekti\V3\Data\J_DATA\test_full_columns_120826.xlsx",
}

# Load our normalised data
print("Loading our normalised data...")
my = pd.read_excel(MY_FILE, engine="openpyxl")
my_lookup = dict(zip(
    my["id"].astype(str).str.strip(),
    my["cur_dg"].astype(str).str.strip()
))
print(f"  {len(my_lookup)} rows")

# Compare
diffs = []
for split, fpath in J_FILES.items():
    print(f"Loading J {split}...")
    j = pd.read_excel(fpath, engine="openpyxl", header=1)
    j.columns = [c.strip().lower().replace(" ", "_") for c in j.columns]
    print(f"  {len(j)} rows")

    for _, row in j.iterrows():
        id_ = str(row.get("id", "")).strip()
        j_dg = str(row.get("cur_dg", "")).strip()
        my_dg = my_lookup.get(id_, "NOT FOUND")

        if j_dg != my_dg:
            diffs.append({
                "id":       id_,
                "split":    split,
                "j_cur_dg": j_dg,
                "my_cur_dg": my_dg,
            })

print(f"\nTotal differences: {len(diffs)}")
print()

# Show first 30
for d in diffs[:30]:
    print(f"  [{d['split']}] {d['id']}: J={d['j_cur_dg']} | Mine={d['my_cur_dg']}")

# Save full diff to Excel
if diffs:
    diff_df = pd.DataFrame(diffs)
    out = r"C:\Users\pette\Desktop\Projekti\V3\Data\cur_dg_diff.xlsx"
    diff_df.to_excel(out, index=False)
    print(f"\nFull diff saved to: {out}")
