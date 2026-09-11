#!/usr/bin/env python3
"""
Generate 5-Fold Stratified Cross Validation splits for RAF-DB.
Combines data/rafdb/train.csv + val.csv (12,271 samples) and creates 5 folds,
while preserving data/rafdb/test.csv (3,068 samples) untouched for testing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def create_kfold_splits(
    data_dir: Path,
    output_kfold_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing train.csv or test.csv in {data_dir}")

    df_train = pd.read_csv(train_csv)
    if val_csv.exists():
        df_val = pd.read_csv(val_csv)
        df_all_train = pd.concat([df_train, df_val], ignore_index=True)
        print(f"[INFO] Merged train.csv ({len(df_train)}) + val.csv ({len(df_val)}) = {len(df_all_train)} samples")
    else:
        df_all_train = df_train
        print(f"[INFO] Using train.csv = {len(df_all_train)} samples")

    df_test = pd.read_csv(test_csv)
    print(f"[INFO] Loaded test.csv = {len(df_test)} samples (kept untouched for evaluation)")

    # Normalize image_path with forward slashes for Linux compatibility
    image_col = next((c for c in ("image_path", "filepath", "path", "image", "pixels") if c in df_all_train.columns), df_all_train.columns[0])
    label_col = next((c for c in ("label", "emotion", "target", "class") if c in df_all_train.columns), df_all_train.columns[1])

    df_all_train[image_col] = df_all_train[image_col].astype(str).str.replace("\\", "/", regex=False)
    df_test[image_col] = df_test[image_col].astype(str).str.replace("\\", "/", regex=False)

    # Clean label types
    df_all_train[label_col] = df_all_train[label_col].astype(int)
    df_test[label_col] = df_test[label_col].astype(int)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df_all_train[label_col].values
    X = df_all_train[image_col].values

    output_kfold_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Generating {n_splits}-Fold Stratified splits in {output_kfold_dir}...")

    fold_summaries = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_dir = output_kfold_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train_df = df_all_train.iloc[train_idx].copy()
        fold_val_df = df_all_train.iloc[val_idx].copy()

        # Sanity check: zero overlap
        train_set = set(fold_train_df[image_col])
        val_set = set(fold_val_df[image_col])
        test_set = set(df_test[image_col])

        assert len(train_set.intersection(val_set)) == 0, f"Leakage detected between train and val in fold {fold}!"
        assert len(train_set.intersection(test_set)) == 0, f"Leakage detected between train and test in fold {fold}!"
        assert len(val_set.intersection(test_set)) == 0, f"Leakage detected between val and test in fold {fold}!"

        # Save CSVs
        fold_train_csv = fold_dir / "train.csv"
        fold_val_csv = fold_dir / "val.csv"
        fold_test_csv = fold_dir / "test.csv"

        fold_train_df.to_csv(fold_train_csv, index=False)
        fold_val_df.to_csv(fold_val_csv, index=False)
        df_test.to_csv(fold_test_csv, index=False)

        val_dist = fold_val_df[label_col].value_counts().sort_index().to_dict()
        fold_summaries.append({
            "fold": fold,
            "train_samples": len(fold_train_df),
            "val_samples": len(fold_val_df),
            "test_samples": len(df_test),
            "val_distribution": val_dist,
        })
        print(f"  -> Fold {fold}: Train={len(fold_train_df):5d}, Val={len(fold_val_df):5d}, Test={len(df_test):5d}")

    print(f"\n[OK] All {n_splits} folds successfully generated and verified with 0% leakage!")


def main():
    parser = argparse.ArgumentParser(description="Create 5-Fold Stratified RAF-DB splits")
    parser.add_argument("--data-dir", type=str, default="data/rafdb", help="Source RAF-DB data dir")
    parser.add_argument("--output-kfold-dir", type=str, default="data/rafdb/kfold_5", help="Output K-Fold dir")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_kfold_dir = ROOT / args.output_kfold_dir if not Path(args.output_kfold_dir).is_absolute() else Path(args.output_kfold_dir)

    create_kfold_splits(data_dir, output_kfold_dir, n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
