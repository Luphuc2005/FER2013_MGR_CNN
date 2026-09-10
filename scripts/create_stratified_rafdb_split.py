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

try:
    import pandas as pd
except ImportError:
    print("=" * 60)
    print("[ERROR] 'pandas' module not found in the current Python environment.")
    print("        Please run the script using the project's virtual environment:")
    print("        ./fer2013_env/bin/python scripts/create_stratified_rafdb_split.py --data_dir data/rafdb")
    print("=" * 60)
    sys.exit(1)

from sklearn.model_selection import train_test_split

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
except ImportError:
    print("=" * 60)
    print("[ERROR] Required modules ('pandas', 'scikit-learn') not found in environment.")
    print("        Please run with project virtual environment:")
    print("        ./fer2013_env/bin/python scripts/create_stratified_rafdb_split.py --data_dir data/rafdb")
    print("=" * 60)
    sys.exit(1)

from datasets.fer2013 import _collect_records_from_folder

RAFDB_RAW_MAP_INT = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}

def backup_file(filepath: Path):
    if filepath.exists():
        bak_path = filepath.with_suffix(filepath.suffix + ".bak")
        shutil.copy2(filepath, bak_path)
        print(f"[BACKUP] Saved backup of {filepath.name} -> {bak_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Create clean benchmark stratified split for RAF-DB.")
    parser.add_argument("--data_dir", type=str, default="data/rafdb", help="Path to RAF-DB dataset directory.")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Ratio of validation set (default 0.10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() and Path("data/raf_db").exists():
        data_dir = Path("data/raf_db")

    print("=" * 60)
    print(f" Generating Benchmark Stratified RAF-DB Split in: {data_dir}")
    print("=" * 60)

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    train_full_csv = data_dir / "train_full.csv"

    # Backup existing CSV files
    for csv_f in [train_csv, val_csv, test_csv, train_full_csv]:
        backup_file(csv_f)

    dfs_to_combine = []

    # Option A: Check if train_full.csv exists with full benchmark count (12,271)
    if train_full_csv.exists():
        df_tf = pd.read_csv(train_full_csv)
        if len(df_tf) == 12271:
            print(f"[INFO] Using train_full.csv with exact benchmark pool size: {len(df_tf)} rows.")
            dfs_to_combine.append(df_tf)

    # Option B: Combine train.csv + val.csv
    if not dfs_to_combine and train_csv.exists() and val_csv.exists():
        df_t = pd.read_csv(train_csv)
        df_v = pd.read_csv(val_csv)
        print(f"[INFO] Loaded train.csv ({len(df_t)}) + val.csv ({len(df_v)}) = {len(df_t) + len(df_v)} rows.")
        dfs_to_combine = [df_t, df_v]
    elif not dfs_to_combine and train_csv.exists():
        df_t = pd.read_csv(train_csv)
        dfs_to_combine = [df_t]

    # Option C: Rescan directory folders if available (train + val directories)
    if not dfs_to_combine or sum(len(d) for d in dfs_to_combine) != 12271:
        if (data_dir / "train").is_dir():
            print(f"[INFO] Scanning subdirectories in {data_dir / 'train'}...")
            px_t, lb_t, _ = _collect_records_from_folder(data_dir, "train")
            df_t = pd.DataFrame({"image_path": px_t, "label": lb_t})
            dfs_to_combine = [df_t]
            if (data_dir / "val").is_dir():
                print(f"[INFO] Scanning subdirectories in {data_dir / 'val'}...")
                px_v, lb_v, _ = _collect_records_from_folder(data_dir, "val")
                df_v = pd.DataFrame({"image_path": px_v, "label": lb_v})
                dfs_to_combine.append(df_v)

    if not dfs_to_combine:
        raise FileNotFoundError(f"Could not find valid RAF-DB data in CSVs or folders inside {data_dir}")

    df_full = pd.concat(dfs_to_combine, ignore_index=True)
    label_col = next((c for c in ("emotion", "label", "target", "class", "y") if c in df_full.columns), df_full.columns[0])
    pixel_col = next((c for c in ("pixels", "image_path", "filepath", "path", "image", "file") if c in df_full.columns), df_full.columns[1])
    print(f"[INFO] Detected label column: '{label_col}', image/pixel column: '{pixel_col}'")

    # Normalize labels: map raw 1-based [1..7] -> 0-based [0..6]
    lbl_vals = df_full[label_col].astype(int).to_numpy()
    if lbl_vals.min() == 1 and lbl_vals.max() == 7:
        print(f"[INFO] Normalizing raw RAF-DB 1-based labels [1..7] -> 0-based [0..6]...")
        df_full[label_col] = [RAFDB_RAW_MAP_INT.get(l, l) for l in lbl_vals]
    elif lbl_vals.min() == 1 and lbl_vals.max() == 6 and 0 not in lbl_vals and (data_dir / "train").is_dir() and (data_dir / "val").is_dir():
        print(f"[WARNING] Re-collecting clean 7-class records from folder structure...")
        px_t, lb_t, _ = _collect_records_from_folder(data_dir, "train")
        px_v, lb_v, _ = _collect_records_from_folder(data_dir, "val")
        df_full = pd.DataFrame({
            pixel_col: np.concatenate([px_t, px_v]),
            label_col: np.concatenate([lb_t, lb_v])
        })

    # Process test.csv (MUST BE EXACTLY 3,068 samples)
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        test_lbl_col = next((c for c in ("emotion", "label", "target", "class", "y") if c in df_test.columns), df_test.columns[0])
        test_pix_col = next((c for c in ("pixels", "image_path", "filepath", "path", "image", "file") if c in df_test.columns), df_test.columns[1])
        t_lbls = df_test[test_lbl_col].astype(int).to_numpy()
        if t_lbls.min() == 1 and t_lbls.max() == 7:
            print(f"[INFO] Normalizing test.csv raw labels [1..7] -> 0-based [0..6]...")
            df_test[test_lbl_col] = [RAFDB_RAW_MAP_INT.get(l, l) for l in t_lbls]
            df_test.to_csv(test_csv, index=False)
        elif (data_dir / "test").is_dir() and (t_lbls.min() == 1 and t_lbls.max() == 6 and 0 not in t_lbls):
            px_te, lb_te, _ = _collect_records_from_folder(data_dir, "test")
            df_test = pd.DataFrame({test_pix_col: px_te, test_lbl_col: lb_te})
            df_test.to_csv(test_csv, index=False)

        # Remove any test set images from full training pool to ensure zero data leakage
        test_imgs = set(df_test[test_pix_col].astype(str))
        train_pool_imgs = set(df_full[pixel_col].astype(str))
        overlap_test = train_pool_imgs.intersection(test_imgs)
        if len(overlap_test) > 0:
            print(f"[WARNING] Removing {len(overlap_test)} test set images from training pool.")
            df_full = df_full[~df_full[pixel_col].astype(str).isin(overlap_test)].reset_index(drop=True)

    # Save backup of full clean train set (12,271 samples)
    df_full.to_csv(train_full_csv, index=False)
    print(f"[INFO] Saved full benchmark training pool ({len(df_full)} samples) to: {train_full_csv}")

    assert len(df_full) == 12271, f"[ERROR] RAF-DB training pool has {len(df_full)} samples, expected exactly 12,271!"

    # Perform stratified split
    df_train, df_val = train_test_split(
        df_full,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df_full[label_col],
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Data leakage check between train and val
    train_imgs = set(df_train[pixel_col].astype(str))
    val_imgs = set(df_val[pixel_col].astype(str))
    overlap_tv = train_imgs.intersection(val_imgs)
    assert len(overlap_tv) == 0, f"[ERROR] Leakage detected between train and val splits! Overlap: {len(overlap_tv)}"

    # Save split train and val CSV files
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    print(f"[SUCCESS] Saved clean benchmark splits:")
    print(f"          - Train CSV ({len(df_train)} samples): {train_csv}")
    print(f"          - Val CSV   ({len(df_val)} samples):   {val_csv}")
    print(f"          - Total Train + Val: {len(df_train) + len(df_val)} samples (Expected: 12271)")
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        print(f"          - Test CSV  ({len(df_test)} samples):  {test_csv} (Expected: 3068)")
    print("=" * 60)

if __name__ == "__main__":
    main()
