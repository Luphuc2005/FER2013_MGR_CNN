#!/usr/bin/env python3
"""
Comprehensive Validation-Tuned TTA Sweep & Checkpoint Ensemble for RAF-DB.

For each checkpoint in `--checkpoint-dir`:
  1. Extract unaugmented and H-flipped logits on Validation set.
  2. Sweep TTA weights w_orig in [0.0, 1.0] (step 0.05) to find optimal (w_orig*, w_flip*) on Validation.
  3. Apply (w_orig*, w_flip*) to Test set to obtain true generalization test score.
  4. Store predictions/probabilities.

Finally:
  5. Compute Softmax Probability Ensemble & Logit Ensemble across all checkpoints on Test set.
  6. Output detailed performance table and save full results to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import load_config
from datasets.fer2013 import build_datasets, EMOTION_NAMES
from train import build_model, configure_gpus, configure_tensorflow_runtime, get_class_names


def parse_args():
    parser = argparse.ArgumentParser("Validation-Tuned TTA Sweep and Multi-Checkpoint Ensemble")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rafdb_v1_clean_evolution.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (default: output_dir/checkpoints/best)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for TTA sweep in [0.0, 1.0] (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save summary JSON report",
    )
    return parser.parse_args()


def extract_dataset_logits(model, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_logits_orig = []
    all_logits_flip = []
    all_labels = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, batch["label"]

        outputs_orig = model(inputs, training=False)
        logits_orig = outputs_orig["logits"].numpy().astype(np.float32)

        if isinstance(inputs, dict):
            flipped_inputs = dict(inputs)
            flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs and inputs["mask"] is not None:
                flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
        else:
            flipped_inputs = tf.image.flip_left_right(inputs)

        outputs_flip = model(flipped_inputs, training=False)
        logits_flip = outputs_flip["logits"].numpy().astype(np.float32)

        all_logits_orig.append(logits_orig)
        all_logits_flip.append(logits_flip)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_logits_orig, axis=0),
        np.concatenate(all_logits_flip, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def sweep_weights(logits_orig: np.ndarray, logits_flip: np.ndarray, labels: np.ndarray, step: float = 0.05):
    weights = np.arange(0.0, 1.0 + 1e-5, step)
    sweep_results = []
    best_acc = -1.0
    best_row = None

    for w_orig in weights:
        w_orig = round(float(w_orig), 4)
        w_flip = round(1.0 - w_orig, 4)

        ensemble_logits = w_orig * logits_orig + w_flip * logits_flip
        preds = np.argmax(ensemble_logits, axis=-1)

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


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _extract_epoch_num(path: Path) -> int:
    name = path.stem if path.name.endswith(".index") else path.name
    digits = re.findall(r"\d+", name)
    return int(digits[-1]) if digits else 0


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints" / "best"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Discover checkpoints
    index_files = sorted(ckpt_dir.glob("ckpt-*.index"), key=_extract_epoch_num)
    if not index_files:
        # Fallback to periodic or last
        index_files = sorted((output_dir / "checkpoints" / "last").glob("ckpt-*.index"), key=_extract_epoch_num)

    if not index_files:
        raise FileNotFoundError(f"No ckpt-*.index files found in {ckpt_dir}")

    ckpt_prefixes = [Path(str(p)[:-6]) for p in index_files]
    num_ckpts = len(ckpt_prefixes)

    print("=" * 75)
    print(" VALIDATION-TUNED TTA SWEEP & CHECKPOINT ENSEMBLE")
    print("=" * 75)
    print(f" Config         : {args.config}")
    print(f" Checkpoint Dir : {ckpt_dir}")
    print(f" Total Models   : {num_ckpts} checkpoint(s)")
    for i, c in enumerate(ckpt_prefixes, 1):
        print(f"   [{i}/{num_ckpts}] {c.name}")
    print("=" * 75)

    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)

    # Build model once
    model = build_model(cfg)
    dummy = tf.zeros([1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]], tf.float32)
    model({"image": dummy}, training=False)
    ckpt_loader = tf.train.Checkpoint(model=model)

    # Build datasets (val and test)
    _, val_ds, test_ds = build_datasets(cfg, replicas=1)
    class_names = get_class_names(cfg)

    checkpoint_eval_results = []
    test_probs_list = []
    test_logits_list = []
    y_test_true = None

    for i, ckpt_prefix in enumerate(ckpt_prefixes, 1):
        ckpt_name = ckpt_prefix.name
        print(f"\n---------------------------------------------------------------------------")
        print(f" [{i}/{num_ckpts}] PROCESSING CHECKPOINT: {ckpt_name}")
        print(f"---------------------------------------------------------------------------")
        ckpt_loader.restore(str(ckpt_prefix)).expect_partial()

        # 1. Validation Sweep
        print(f"  -> Extracting Validation set logits (1,228 samples)...", flush=True)
        val_orig, val_flip, val_labels = extract_dataset_logits(model, val_ds)
        val_sweep, val_best = sweep_weights(val_orig, val_flip, val_labels, step=args.step)

        opt_w_orig = val_best["w_orig"]
        opt_w_flip = val_best["w_flip"]
        print(f"  -> Validation Optimal TTA: w_orig={opt_w_orig:.2f}, w_flip={opt_w_flip:.2f} | Val Acc: {val_best['accuracy']*100:.2f}%")

        # 2. Test Evaluation
        print(f"  -> Extracting Test set logits (3,068 samples)...", flush=True)
        test_orig, test_flip, test_labels = extract_dataset_logits(model, test_ds)
        if y_test_true is None:
            y_test_true = test_labels

        # No TTA (w_orig = 1.0)
        no_tta_test_logits = test_orig
        no_tta_preds = np.argmax(no_tta_test_logits, axis=-1)
        no_tta_acc = float(accuracy_score(y_test_true, no_tta_preds))
        no_tta_f1 = float(f1_score(y_test_true, no_tta_preds, average="macro"))

        # Val-Tuned TTA
        val_tuned_test_logits = opt_w_orig * test_orig + opt_w_flip * test_flip
        val_tuned_preds = np.argmax(val_tuned_test_logits, axis=-1)
        val_tuned_acc = float(accuracy_score(y_test_true, val_tuned_preds))
        val_tuned_macro_f1 = float(f1_score(y_test_true, val_tuned_preds, average="macro"))
        val_tuned_weighted_f1 = float(f1_score(y_test_true, val_tuned_preds, average="weighted"))

        gain = (val_tuned_acc - no_tta_acc) * 100.0

        print(f"  >> RESULTS for {ckpt_name}:")
        print(f"     No-TTA Test Accuracy       : {no_tta_acc*100:.2f}% (Macro F1: {no_tta_f1:.4f})")
        print(f"     Val-Tuned TTA Test Accuracy: {val_tuned_acc*100:.2f}% (Macro F1: {val_tuned_macro_f1:.4f}) [Gain: {gain:+.2f}%]")

        # Softmax probabilities for ensemble
        val_tuned_probs = softmax(val_tuned_test_logits, axis=-1)
        test_probs_list.append(val_tuned_probs)
        test_logits_list.append(val_tuned_test_logits)

        checkpoint_eval_results.append({
            "checkpoint": ckpt_name,
            "val_accuracy": val_best["accuracy"],
            "val_optimal_w_orig": opt_w_orig,
            "val_optimal_w_flip": opt_w_flip,
            "test_no_tta_accuracy": no_tta_acc,
            "test_no_tta_macro_f1": no_tta_f1,
            "test_val_tuned_accuracy": val_tuned_acc,
            "test_val_tuned_macro_f1": val_tuned_macro_f1,
            "test_val_tuned_weighted_f1": val_tuned_weighted_f1,
            "tta_gain_pct": gain,
        })

    # =========================================================================
    # MULTI-CHECKPOINT ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 75)
    print(f" COMPUTING ENSEMBLE ACROSS ALL {num_ckpts} CHECKPOINT(S)")
    print("=" * 75)

    # 1. Softmax Probability Average Ensemble
    avg_probs = np.mean(test_probs_list, axis=0)
    ensemble_preds = np.argmax(avg_probs, axis=-1)

    ens_acc = float(accuracy_score(y_test_true, ensemble_preds))
    ens_macro_f1 = float(f1_score(y_test_true, ensemble_preds, average="macro"))
    ens_weighted_f1 = float(f1_score(y_test_true, ensemble_preds, average="weighted"))
    ens_cm = confusion_matrix(y_test_true, ensemble_preds).tolist()
    ens_report = classification_report(y_test_true, ensemble_preds, target_names=class_names, output_dict=True)

    # 2. Logit Average Ensemble
    avg_logits = np.mean(test_logits_list, axis=0)
    logit_ens_preds = np.argmax(avg_logits, axis=-1)
    logit_ens_acc = float(accuracy_score(y_test_true, logit_ens_preds))
    logit_ens_macro_f1 = float(f1_score(y_test_true, logit_ens_preds, average="macro"))

    # Summary Table
    print(f"\n {'Checkpoint':<14} | {'Val Acc (TTA)':<14} | {'w_orig/flip':<12} | {'Test (No-TTA)':<14} | {'Test (TTA)':<12}")
    print("-" * 75)
    for r in checkpoint_eval_results:
        w_str = f"{r['val_optimal_w_orig']:.2f}/{r['val_optimal_w_flip']:.2f}"
        print(f" {r['checkpoint']:<14} | {r['val_accuracy']*100:<13.2f}% | {w_str:<12} | {r['test_no_tta_accuracy']*100:<13.2f}% | {r['test_val_tuned_accuracy']*100:<11.2f}%")
    print("=" * 75)

    print(f"\n===========================================================================")
    print(f" ★ FINAL ENSEMBLE RESULTS ON TEST SET (3,068 samples):")
    print(f"===========================================================================")
    print(f"  >> Softmax Prob Ensemble Accuracy : {ens_acc * 100:.2f}%")
    print(f"  >> Softmax Prob Ensemble Macro-F1 : {ens_macro_f1:.4f}")
    print(f"  >> Softmax Prob Ensemble W-F1     : {ens_weighted_f1:.4f}")
    print(f"  >> Logit Average Ensemble Accuracy: {logit_ens_acc * 100:.2f}% (Macro F1: {logit_ens_macro_f1:.4f})")
    print(f"---------------------------------------------------------------------------")

    print("\nPer-class Performance (Ensemble):")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print("-" * 60)
    for name in class_names:
        c_p = ens_report[name]["precision"] * 100
        c_r = ens_report[name]["recall"] * 100
        c_f1 = ens_report[name]["f1-score"] * 100
        c_supp = int(ens_report[name]["support"])
        print(f"{name:<12} {c_p:>9.2f}% {c_r:>9.2f}% {c_f1:>9.2f}% {c_supp:>10}")
    print("=" * 75)

    # Save output
    out_file = Path(args.output) if args.output else output_dir / "val_tuned_tta_ensemble_summary.json"
    summary_data = {
        "individual_checkpoints": checkpoint_eval_results,
        "ensemble_softmax_accuracy": ens_acc,
        "ensemble_softmax_macro_f1": ens_macro_f1,
        "ensemble_softmax_weighted_f1": ens_weighted_f1,
        "ensemble_logit_accuracy": logit_ens_acc,
        "ensemble_logit_macro_f1": logit_ens_macro_f1,
        "confusion_matrix": ens_cm,
        "classification_report": ens_report,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[SUCCESS] Full summary report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
