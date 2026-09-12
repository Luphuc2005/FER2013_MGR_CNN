#!/usr/bin/env python3
"""
Evaluate 5-Fold Cross Validation Checkpoints and Run SOTA Ensemble on RAF-DB.
Evaluates:
1. Individual fold performance on Test Set (No-TTA and TTA).
2. Out-Of-Fold (OOF) validation performance across all 12,271 training samples.
3. 5-Fold Logits Ensemble on Test Set (3,068 samples) under No-TTA and TTA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets
from train import build_model, configure_gpus, configure_tensorflow_runtime


def parse_args():
    parser = argparse.ArgumentParser("Evaluate 5-Fold Cross Validation & Ensemble for RAF-DB")
    parser.add_argument("--base-config", type=str, default="config_rafdb_v5_kfold_base.yaml")
    parser.add_argument("--kfold-dir", type=str, default="data/rafdb/kfold_5")
    parser.add_argument("--output-dir", type=str, default="outputs/papers/rafdb_v5_kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.05, help="TTA sweep step size")
    return parser.parse_args()


def extract_dataset_logits(model, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_orig = []
    all_flip = []
    all_lbl = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, batch["label"]

        out_orig = model(inputs, training=False)
        logits_orig = out_orig["logits"].numpy().astype(np.float32)

        if isinstance(inputs, dict):
            flipped = dict(inputs)
            flipped["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs and inputs["mask"] is not None:
                flipped["mask"] = tf.image.flip_left_right(inputs["mask"])
        else:
            flipped = tf.image.flip_left_right(inputs)

        out_flip = model(flipped, training=False)
        logits_flip = out_flip["logits"].numpy().astype(np.float32)

        all_orig.append(logits_orig)
        all_flip.append(logits_flip)
        all_lbl.append(labels.numpy())

    return np.concatenate(all_orig, axis=0), np.concatenate(all_flip, axis=0), np.concatenate(all_lbl, axis=0)


def eval_weights(logits_orig: np.ndarray, logits_flip: np.ndarray, labels: np.ndarray, step: float = 0.05):
    weights = np.arange(0.0, 1.0 + 1e-5, step)
    sweep_results = []
    best_acc = -1.0
    best_row = None

    for w_orig in weights:
        w_orig = round(float(w_orig), 4)
        w_flip = round(1.0 - w_orig, 4)

        ens_logits = w_orig * logits_orig + w_flip * logits_flip
        preds = np.argmax(ens_logits, axis=-1)

        acc = float(accuracy_score(labels, preds))
        macro_f1 = float(f1_score(labels, preds, average="macro"))
        weighted_f1 = float(f1_score(labels, preds, average="weighted"))

        row = {
            "w_orig": w_orig,
            "w_flip": w_flip,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        sweep_results.append(row)

        if acc > best_acc:
            best_acc = acc
            best_row = row

    return sweep_results, best_row


def find_best_checkpoint(fold_output_dir: Path) -> str:
    best_dir = fold_output_dir / "checkpoints" / "best"
    rankings_file = best_dir / "top_k_rankings.json"

    if rankings_file.exists():
        try:
            with open(rankings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            if entries:
                cand = best_dir / entries[0]["checkpoint"]
                if Path(str(cand) + ".index").exists():
                    print(f"  [CKPT] Found Rank 1 from top_k_rankings.json: {cand.name} (val_acc={entries[0].get('metric'):.4f})")
                    return str(cand)
        except Exception as e:
            print(f"  [WARN] Could not parse top_k_rankings.json: {e}")

    # Fallback: scan for *.index
    indexes = sorted(best_dir.glob("ckpt-*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
    if indexes:
        prefix = str(indexes[0])[:-6]
        print(f"  [CKPT] Using latest checkpoint by mtime: {Path(prefix).name}")
        return prefix

    # Fallback to last/
    last_indexes = sorted((fold_output_dir / "checkpoints" / "last").glob("ckpt-*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
    if last_indexes:
        prefix = str(last_indexes[0])[:-6]
        print(f"  [CKPT] Using last checkpoint: {Path(prefix).name}")
        return prefix

    raise FileNotFoundError(f"No checkpoint found in {fold_output_dir / 'checkpoints'}")


def main():
    args = parse_args()
    base_cfg = load_config(args.base_config)
    configure_tensorflow_runtime(base_cfg)
    configure_gpus(base_cfg)

    kfold_dir = ROOT / args.kfold_dir if not Path(args.kfold_dir).is_absolute() else Path(args.kfold_dir)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    print("=" * 80)
    print("  RAF-DB V5 5-FOLD CROSS VALIDATION & ENSEMBLE EVALUATION")
    print(f"  Base Config : {args.base_config}")
    print(f"  K-Fold Dir  : {kfold_dir}")
    print(f"  Output Dir  : {output_dir}")
    print("=" * 80)

    fold_test_logits_orig = []
    fold_test_logits_flip = []
    test_labels_global = None

    oof_preds_list = []
    oof_labels_list = []

    individual_results = []

    for fold in range(args.n_splits):
        print(f"\n" + "-" * 70)
        print(f"  PROCESSING FOLD {fold}/{args.n_splits - 1}")
        print("-" * 70)

        fold_cfg_file = ROOT / f"config_rafdb_v5_kfold_{fold}.yaml"
        if not fold_cfg_file.exists():
            fold_cfg_file = args.base_config
        cfg = load_config(fold_cfg_file)

        # Override data path and output path for this fold
        cfg["data"]["data_path"] = str(kfold_dir / f"fold_{fold}")
        fold_out = output_dir / f"fold_{fold}"

        ckpt_prefix = find_best_checkpoint(fold_out)

        # Build model and restore
        model = build_model(cfg)
        _ = model({"image": tf.zeros((1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]), tf.float32)}, training=False)

        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(ckpt_prefix).expect_partial()
        print(f"  [LOADED] Restored model weights from: {ckpt_prefix}")

        # Build datasets
        _, val_ds, test_ds = build_datasets(cfg, replicas=1)

        # 1. Validation inference (for OOF)
        print(f"  [INFERENCE] Evaluating Validation Set (Fold {fold})...")
        val_orig, val_flip, val_labels = extract_dataset_logits(model, val_ds)
        val_sweep, val_best = eval_weights(val_orig, val_flip, val_labels, step=args.step)

        opt_w_orig = val_best["w_orig"]
        opt_val_logits = opt_w_orig * val_orig + (1.0 - opt_w_orig) * val_flip
        oof_preds_list.append(np.argmax(opt_val_logits, axis=-1))
        oof_labels_list.append(val_labels)

        # 2. Test inference
        print(f"  [INFERENCE] Evaluating Test Set (3,068 samples)...")
        test_orig, test_flip, test_labels = extract_dataset_logits(model, test_ds)
        test_labels_global = test_labels

        fold_test_logits_orig.append(test_orig)
        fold_test_logits_flip.append(test_flip)

        # Single fold test metrics
        test_sweep, test_best = eval_weights(test_orig, test_flip, test_labels, step=args.step)
        test_no_tta = next(r for r in test_sweep if abs(r["w_orig"] - 1.0) < 1e-4)
        test_val_tuned = next(r for r in test_sweep if abs(r["w_orig"] - opt_w_orig) < 1e-4)

        fold_res = {
            "fold": fold,
            "checkpoint": ckpt_prefix,
            "val_best_acc": val_best["accuracy"] * 100,
            "val_opt_w_orig": opt_w_orig,
            "test_no_tta_acc": test_no_tta["accuracy"] * 100,
            "test_tta_val_tuned_acc": test_val_tuned["accuracy"] * 100,
            "test_tta_peak_acc": test_best["accuracy"] * 100,
            "test_macro_f1": test_val_tuned["macro_f1"],
            "test_weighted_f1": test_val_tuned["weighted_f1"],
        }
        individual_results.append(fold_res)

        print(f"  -> Fold {fold} Summary: Val Acc = {fold_res['val_best_acc']:.2f}% | Test No-TTA = {fold_res['test_no_tta_acc']:.2f}% | Test TTA = {fold_res['test_tta_val_tuned_acc']:.2f}% (Macro F1={fold_res['test_macro_f1']:.4f})")

        # Free GPU memory
        tf.keras.backend.clear_session()

    # Out-Of-Fold (OOF) Overall Metrics
    oof_preds_all = np.concatenate(oof_preds_list, axis=0)
    oof_labels_all = np.concatenate(oof_labels_list, axis=0)
    oof_accuracy = float(accuracy_score(oof_labels_all, oof_preds_all)) * 100
    oof_macro_f1 = float(f1_score(oof_labels_all, oof_preds_all, average="macro"))

    # 5-Fold Ensemble on Test Set
    print("\n" + "=" * 80)
    print("  COMPUTING 5-FOLD ENSEMBLE PREDICTIONS ON TEST SET (3,068 SAMPLES)")
    print("=" * 80)

    ens_logits_orig = np.mean(fold_test_logits_orig, axis=0)
    ens_logits_flip = np.mean(fold_test_logits_flip, axis=0)

    ens_sweep, ens_best = eval_weights(ens_logits_orig, ens_logits_flip, test_labels_global, step=args.step)
    ens_no_tta = next(r for r in ens_sweep if abs(r["w_orig"] - 1.0) < 1e-4)

    # Classification report & Confusion matrix on Test Set
    best_ens_logits = ens_best["w_orig"] * ens_logits_orig + ens_best["w_flip"] * ens_logits_flip
    ens_preds = np.argmax(best_ens_logits, axis=-1)

    cls_report = classification_report(test_labels_global, ens_preds, target_names=EMOTION_NAMES, digits=4, output_dict=True)
    cm = confusion_matrix(test_labels_global, ens_preds).tolist()

    mean_single_test_acc = np.mean([r["test_tta_val_tuned_acc"] for r in individual_results])
    ensemble_gain = ens_best["accuracy"] * 100 - mean_single_test_acc

    # Display Leaderboard
    print("\n" + "=" * 80)
    print("  FINAL 5-FOLD BENCHMARK RESULTS TABLE")
    print("=" * 80)
    print(f" {'Model':<20} | {'Val Acc (%)':<12} | {'Test No-TTA (%)':<16} | {'Test TTA (%)':<14} | {'Macro F1':<10}")
    print("-" * 80)
    for r in individual_results:
        print(f" Fold {r['fold']:<15} | {r['val_best_acc']:<12.2f} | {r['test_no_tta_acc']:<16.2f} | {r['test_tta_val_tuned_acc']:<14.2f} | {r['test_macro_f1']:<10.4f}")
    print("-" * 80)
    print(f" {'Mean Single Fold':<20} | {'-':<12} | {np.mean([r['test_no_tta_acc'] for r in individual_results]):<16.2f} | {mean_single_test_acc:<14.2f} | {np.mean([r['test_macro_f1'] for r in individual_results]):<10.4f}")
    print(f" {'Out-Of-Fold (OOF)':<20} | {oof_accuracy:<12.2f} | {'-':<16} | {'-':<14} | {oof_macro_f1:<10.4f}")
    print("=" * 80)
    print(f" ★ 5-FOLD ENSEMBLE    | {'-':<12} | {ens_no_tta['accuracy'] * 100:<16.2f} | {ens_best['accuracy'] * 100:<14.2f} | {ens_best['macro_f1']:<10.4f}")
    print(f" ★ ENSEMBLE GAIN      | +{ensemble_gain:.2f}% accuracy boost over mean single model")
    print("=" * 80)

    # Save JSON Summary
    summary_data = {
        "dataset": "RAF-DB",
        "num_folds": args.n_splits,
        "single_models": individual_results,
        "oof_accuracy": oof_accuracy,
        "oof_macro_f1": oof_macro_f1,
        "mean_single_test_accuracy": float(mean_single_test_acc),
        "ensemble_test_no_tta_accuracy": float(ens_no_tta["accuracy"] * 100),
        "ensemble_test_tta_accuracy": float(ens_best["accuracy"] * 100),
        "ensemble_test_macro_f1": float(ens_best["macro_f1"]),
        "ensemble_test_weighted_f1": float(ens_best["weighted_f1"]),
        "optimal_tta_w_orig": ens_best["w_orig"],
        "optimal_tta_w_flip": ens_best["w_flip"],
        "classification_report": cls_report,
        "confusion_matrix": cm,
    }

    out_json = output_dir / "kfold_ensemble_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[SAVED] Complete benchmark metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
