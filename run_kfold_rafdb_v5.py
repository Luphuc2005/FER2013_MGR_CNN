#!/usr/bin/env python3
"""
Master 1-Click Pipeline Runner for RAF-DB V5 5-Fold Cross Validation & Ensemble.
Coordinates:
1. Automatic 5-Fold Stratified Split generation (if not already done).
2. Sequential training of Fold 0 -> Fold 4 with automatic resume support.
3. Automatic 5-Fold Checkpoint Ensemble & Test Set Evaluation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser("RAF-DB V5 5-Fold 1-Click Automated Pipeline")
    parser.add_argument("--base-config", type=str, default="config_rafdb_v5_kfold_base.yaml")
    parser.add_argument("--kfold-dir", type=str, default="data/rafdb/kfold_5")
    parser.add_argument("--output-dir", type=str, default="outputs/papers/rafdb_v5_kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs per fold (default: 60)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience per fold (default: 20)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing training")
    return parser.parse_args()


def is_fold_finished(output_dir: Path, fold: int) -> bool:
    fold_dir = output_dir / f"fold_{fold}"
    best_dir = fold_dir / "checkpoints" / "best"
    rankings = best_dir / "top_k_rankings.json"
    if rankings.exists() and len(list(best_dir.glob("ckpt-*.index"))) > 0:
        return True
    return False


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    cmd_str = " ".join(cmd)
    print(f"\n[RUNNING] {cmd_str}")
    if dry_run:
        print("  [DRY-RUN] Skipping execution.")
        return
    start_t = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}: {cmd_str}")
        sys.exit(res.returncode)
    print(f"[FINISHED] Command finished in {elapsed / 60:.1f} minutes.")


def main():
    args = parse_args()
    kfold_dir = ROOT / args.kfold_dir if not Path(args.kfold_dir).is_absolute() else Path(args.kfold_dir)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    print("=" * 80)
    print("  RAF-DB V5 5-FOLD CROSS VALIDATION & ENSEMBLE AUTOMATED RUNNER")
    print(f"  Base Config : {args.base_config}")
    print(f"  K-Fold Dir  : {kfold_dir}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Folds       : {args.n_splits}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Patience    : {args.patience}")
    print("=" * 80)

    # STEP 0: Check / Generate splits
    splits_ready = True
    for f in range(args.n_splits):
        f_dir = kfold_dir / f"fold_{f}"
        if not (f_dir / "train.csv").exists() or not (f_dir / "val.csv").exists() or not (f_dir / "test.csv").exists():
            splits_ready = False
            break

    if not splits_ready:
        print("\n>>> STEP 0: Generating 5-Fold Stratified Splits for RAF-DB...")
        split_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "create_kfold_rafdb_splits.py"),
            "--data-dir", "data/rafdb",
            "--output-kfold-dir", str(kfold_dir),
            "--n-splits", str(args.n_splits),
        ]
        run_command(split_cmd, dry_run=args.dry_run)
    else:
        print("\n>>> STEP 0: 5-Fold Splits already verified at data/rafdb/kfold_5.")

    # STEP 1: Train Folds
    if not args.skip_train:
        print("\n>>> STEP 1: Training 5 Folds Sequentially...")
        for fold in range(args.n_splits):
            print("\n" + "#" * 80)
            print(f"  FOLD {fold} / {args.n_splits - 1}")
            print("#" * 80)

            if is_fold_finished(output_dir, fold):
                print(f"[INFO] Fold {fold} has already finished training! Skipping to next fold...")
                continue

            fold_cfg_path = ROOT / f"config_rafdb_v5_kfold_{fold}.yaml"
            if not fold_cfg_path.exists():
                with open(ROOT / args.base_config, "r", encoding="utf-8") as f_in:
                    base_text = f_in.read()
                fold_text = base_text.replace("fold_0", f"fold_{fold}")
                with open(fold_cfg_path, "w", encoding="utf-8") as f_out:
                    f_out.write(fold_text)

            train_cmd = [
                sys.executable,
                str(ROOT / "train.py"),
                "--config", str(fold_cfg_path),
                "--no-auto-increment",
            ]
            run_command(train_cmd, dry_run=args.dry_run)
    else:
        print("\n[INFO] Skipping training step as requested (--skip-train).")

    # STEP 2: Run 5-Fold Ensemble & Evaluation
    print("\n" + "=" * 80)
    print(">>> STEP 2: Running 5-Fold Ensemble & Complete Evaluation on Test Set...")
    print("=" * 80)

    eval_cmd = [
        sys.executable,
        str(ROOT / "evaluate_kfold_ensemble.py"),
        "--base-config", str(args.base_config),
        "--kfold-dir", str(kfold_dir),
        "--output-dir", str(output_dir),
        "--n-splits", str(args.n_splits),
        "--step", "0.05",
    ]
    run_command(eval_cmd, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("  ALL 5 FOLDS TRAINED AND SOTA ENSEMBLE EVALUATED SUCCESSFULLY!")
    print(f"  Summary Report: {output_dir / 'kfold_ensemble_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
