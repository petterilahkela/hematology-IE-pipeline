
"""
split_data_v3.py
================
Splits MGG_normalised_v3.xlsx into train/val/test sets (70/15/15).


Usage:
    python split_data_v3.py \
        --input      "C:\\...\\MGG_normalised_v3.xlsx" \
        --output_dir "C:\\...\\V3\\Data"
"""

import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    df = pd.read_excel(args.input, engine="openpyxl")
    print(f"Total rows: {len(df)}")

    from sklearn.model_selection import train_test_split

    # Stratified split based on cur_dg category
    train_val_df, test_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["cur_dg"])
    train_df, val_df = train_test_split(
    train_val_df, test_size=0.15/0.85, random_state=42, stratify=train_val_df["cur_dg"])

    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")

    train_df.to_excel(os.path.join(args.output_dir, "train_v3.xlsx"), index=False)
    val_df.to_excel(  os.path.join(args.output_dir, "val_v3.xlsx"),   index=False)
    test_df.to_excel( os.path.join(args.output_dir, "test_v3.xlsx"),  index=False)

    print("Done!")
    print(f"  Train → {os.path.join(args.output_dir, 'train_v3.xlsx')}")
    print(f"  Val   → {os.path.join(args.output_dir, 'val_v3.xlsx')}")
    print(f"  Test  → {os.path.join(args.output_dir, 'test_v3.xlsx')}")

if __name__ == "__main__":
    main()

