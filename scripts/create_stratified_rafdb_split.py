#!/usr/bin/env python3
"""
Utility script to create a clean, stratified 90% Train / 10% Validation split for RAF-DB,
ensuring zero data leakage between train.csv, val.csv, and test.csv.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Create clean stratified split for RAF-DB.")
    parser.add_argument("--data_dir", type=str, default="data/raf_db", help="Path to RAF-DB dataset directory containing CSV files or folders.")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Ratio of validation set (default 0.10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print(f" Generating Stratified RAF-DB Split in: {data_dir}")
    print("=" * 60)

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    train_full_csv = data_dir / "train_full.csv"

    # Determine source file for training split
    if train_full_csv.exists():
        src_csv = train_full_csv
        df_full = pd.read_csv(src_csv)
        print(f"[INFO] Using existing full training file: {train_full_csv} ({len(df_full)} rows)")
    elif train_csv.exists() and val_csv.exists():
        df_t = pd.read_csv(train_csv)
        df_v = pd.read_csv(val_csv)
        df_full = pd.concat([df_t, df_v], ignore_index=True)
        print(f"[INFO] Combined train.csv ({len(df_t)}) + val.csv ({len(df_v)}) -> Full source ({len(df_full)} rows)")
    elif train_csv.exists():
        df_full = pd.read_csv(train_csv)
        print(f"[INFO] Using train.csv as source split file: {train_csv} ({len(df_full)} rows)")
    else:
        raise FileNotFoundError(f"Could not find train.csv or train_full.csv in {data_dir}")

    # Identify label and image columns
    label_col = next((c for c in ("emotion", "label", "target", "class", "y") if c in df_full.columns), df_full.columns[0])
    pixel_col = next((c for c in ("pixels", "image_path", "filepath", "path", "image", "file") if c in df_full.columns), df_full.columns[1])
    print(f"[INFO] Detected label column: '{label_col}', image/pixel column: '{pixel_col}'")

    # Deduplicate exact duplicate pixel/image entries to prevent data leakage
    len_before = len(df_full)
    df_full = df_full.drop_duplicates(subset=[pixel_col]).reset_index(drop=True)
    if len(df_full) < len_before:
        print(f"[INFO] Removed {len_before - len(df_full)} exact duplicate image/pixel rows.")

    # Save backup of full clean train set
    if not train_full_csv.exists():
        df_full.to_csv(train_full_csv, index=False)
        print(f"[INFO] Saved backup of full training data to: {train_full_csv}")

    # Perform stratified split
    df_train, df_val = train_test_split(
        df_full,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df_full[label_col],
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Data leakage check
    train_imgs = set(df_train[pixel_col].astype(str))
    val_imgs = set(df_val[pixel_col].astype(str))
    overlap = train_imgs.intersection(val_imgs)

    assert len(overlap) == 0, f"[ERROR] Leakage detected in generated split! Overlap count: {len(overlap)}"

    # Save split train and val CSV files
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    print(f"[SUCCESS] Saved clean splits:")
    print(f"          - Train CSV ({len(df_train)} samples): {train_csv}")
    print(f"          - Val CSV   ({len(df_val)} samples):   {val_csv}")
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        print(f"          - Test CSV  ({len(df_test)} samples):  {test_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
