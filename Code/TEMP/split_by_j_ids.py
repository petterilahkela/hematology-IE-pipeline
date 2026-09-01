"""
split_by_j_ids.py
=================
Splits MGG_normalised_v3.xlsx into train/val/test
based on J's id lists.
"""

import pandas as pd
import os

MY_FILE = r"C:\Users\pette\Desktop\Projekti\V3\Data\MGG_normalised_v3.xlsx"
OUT_DIR = r"C:\Users\pette\Desktop\Projekti\V3\Data\FT_DATA"

J_FILES = {
    "train": r"C:\Users\pette\Desktop\Projekti\V3\Data\J_DATA\train_full_columns_120826.xlsx",
    "val":   r"C:\Users\pette\Desktop\Projekti\V3\Data\J_DATA\valid_full_columns_120826.xlsx",
    "test":  r"C:\Users\pette\Desktop\Projekti\V3\Data\J_DATA\test_full_columns_120826.xlsx",
}

# Load our normalised data
my = pd.read_excel(MY_FILE, engine="openpyxl")
print(f"Loaded {len(my)} rows from MGG_normalised_v3.xlsx")

# Split by J's ids
for split, fpath in J_FILES.items():
    j = pd.read_excel(fpath, engine="openpyxl", header=1)
    ids = set(j["id"].astype(str).str.strip())
    subset = my[my["id"].astype(str).str.strip().isin(ids)]
    out = os.path.join(OUT_DIR, f"{split}_v3.xlsx")
    subset.to_excel(out, index=False)
    print(f"{split}: {len(subset)} rows → {out}")

print("Done!")