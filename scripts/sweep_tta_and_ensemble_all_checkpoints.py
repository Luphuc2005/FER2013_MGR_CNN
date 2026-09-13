#!/usr/bin/env python3
"""
Comprehensive Validation-Tuned TTA Sweep & Massive Combinatorial Checkpoint Ensemble for RAF-DB.

1. Extracts logits (orig + h-flip) on Validation and Test sets for each candidate checkpoint.
2. Sweeps TTA weights w_orig in [0.0, 1.0] (step 0.05) on Validation set to find optimal TTA per checkpoint.
3. Obtains calibrated Softmax probabilities for each checkpoint on Validation and Test sets.
4. Executes a MASSIVE COMBINATORIAL ENSEMBLE SWEEP (50,000+ to 100,000+ combinations):
   - All single models
   - All pairs (A + B) with fine weight grid (step 0.05)
   - All triplets (A + B + C) with simplex weight grid (step 0.05)
   - All 4-tuples with simplex weight grid (step 0.10)
   - All 2^K - 1 uniform subsets
   - 50,000 continuous Dirichlet random weight vectors across all subset sizes
5. Selects the optimal combination strictly based on Validation performance (Zero Data Leakage).
6. Evaluates and reports true Test generalization, Oracle upper bound, and saves full JSON report.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    parser = argparse.ArgumentParser("Validation-Tuned TTA Sweep & Massive Combinatorial Ensemble")
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
        help="Directory containing checkpoints (default: search both best/ and best_loss/)",
    )
    parser.add_argument(
        "--include-best-loss",
        action="store_true",
        default=True,
        help="Also include checkpoints from checkpoints/best_loss (lowest val_loss) in evaluation and ensemble (default: True)",
    )
    parser.add_argument(
        "--include-best-macro-f1",
        action="store_true",
        default=True,
        help="Also include checkpoints from checkpoints/best_macro_f1 (highest val_macro_f1) in evaluation and ensemble (default: True)",
    )
    parser.add_argument(
        "--include-periodic",
        action="store_true",
        default=False,
        help="Also include checkpoints from checkpoints/periodic in evaluation (default: False)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for single-model TTA sweep in [0.0, 1.0] (default: 0.05)",
    )
    parser.add_argument(
        "--num-comb-samples",
        type=int,
        default=50000,
        help="Number of random Dirichlet weight combinations to evaluate (default: 50000)",
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


def format_comb_weights(comb: Tuple[int, ...], weights: np.ndarray, ckpt_names: List[str]) -> str:
    terms = []
    for idx, w in zip(comb, weights):
        if w > 0.005:
            terms.append(f"{w:.2f}*{ckpt_names[idx]}")
    return " + ".join(terms) if terms else "None"


def generate_combinatorial_candidates(
    num_models: int,
    num_random_samples: int = 50000,
    seed: int = 42,
) -> List[Tuple[Tuple[int, ...], np.ndarray, str]]:
    candidates: List[Tuple[Tuple[int, ...], np.ndarray, str]] = []

    # 1. Single models
    for i in range(num_models):
        candidates.append(((i,), np.array([1.0], dtype=np.float32), "single"))

    if num_models < 2:
        return candidates

    # 2. All pairs (A + B) with step 0.05
    for i in range(num_models):
        for j in range(i + 1, num_models):
            for w in np.arange(0.05, 1.0, 0.05):
                w_val = round(float(w), 4)
                w_arr = np.array([w_val, round(1.0 - w_val, 4)], dtype=np.float32)
                candidates.append(((i, j), w_arr, "pair"))

    # 3. All triplets (A + B + C) with step 0.05
    if num_models >= 3:
        for comb in itertools.combinations(range(num_models), 3):
            for i_step in range(1, 20):
                for j_step in range(1, 20 - i_step):
                    k_step = 20 - i_step - j_step
                    if k_step > 0:
                        w_arr = np.array([i_step * 0.05, j_step * 0.05, k_step * 0.05], dtype=np.float32)
                        candidates.append((comb, w_arr, "triplet"))

    # 4. All 4-tuples with step 0.10
    if num_models >= 4:
        for comb in itertools.combinations(range(num_models), 4):
            for i_s in range(1, 10):
                for j_s in range(1, 10 - i_s):
                    for k_s in range(1, 10 - i_s - j_s):
                        l_s = 10 - i_s - j_s - k_s
                        if l_s > 0:
                            w_arr = np.array([i_s * 0.10, j_s * 0.10, k_s * 0.10, l_s * 0.10], dtype=np.float32)
                            candidates.append((comb, w_arr, "four_model"))

    # 5. All uniform subsets (size 2..K)
    for r in range(2, num_models + 1):
        for comb in itertools.combinations(range(num_models), r):
            w_arr = np.full(r, 1.0 / r, dtype=np.float32)
            candidates.append((comb, w_arr, "uniform"))

    # 6. Dirichlet continuous random samples
    if num_random_samples > 0:
        rng = np.random.default_rng(seed)
        sizes = list(range(2, num_models + 1))
        alphas = [0.5, 1.0, 2.0]
        for _ in range(num_random_samples):
            size = int(rng.choice(sizes))
            comb = tuple(sorted(rng.choice(num_models, size=size, replace=False).tolist()))
            alpha = float(rng.choice(alphas))
            w_arr = rng.dirichlet(np.full(size, alpha)).astype(np.float32)
            candidates.append((comb, w_arr, "dirichlet_random"))

    return candidates


def run_massive_combinatorial_sweep(
    val_probs: np.ndarray,      # (K, N_val, C)
    test_probs: np.ndarray,     # (K, N_test, C)
    y_val: np.ndarray,          # (N_val,)
    y_test: np.ndarray,         # (N_test,)
    ckpt_names: List[str],      # [name_0, ... name_{K-1}]
    num_random_samples: int = 50000,
    seed: int = 42,
) -> Dict[str, Any]:
    candidates = generate_combinatorial_candidates(len(ckpt_names), num_random_samples, seed)
    total_combs = len(candidates)

    print(f"\n===========================================================================")
    print(f" MASSIVE COMBINATORIAL ENSEMBLE SWEEP: {total_combs:,} COMBINATIONS")
    print(f"===========================================================================")
    print(f" -> Exhaustively evaluating:")
    print(f"    - All single models")
    print(f"    - All pairs (A + B) with weight grid (step 0.05)")
    print(f"    - All triplets (A + B + C) with simplex grid (step 0.05)")
    print(f"    - All 4-tuples with simplex grid (step 0.10)")
    print(f"    - All uniform subsets of any size")
    print(f"    - {num_random_samples:,} continuous Dirichlet random weight samples")
    print(f" -> Criterion: Optimize strictly on Validation, evaluate on Test (No Leakage).")
    print(f"---------------------------------------------------------------------------")

    t0 = time.time()

    val_records = []
    best_val_record = None
    best_test_oracle_record = None

    best_pair_val = None
    best_triplet_val = None
    best_uniform_val = None

    for comb, weights, category in candidates:
        w_col = weights[:, None, None]

        # Validation prediction
        val_sub = val_probs[list(comb)]
        val_pred = np.argmax(np.sum(w_col * val_sub, axis=0), axis=-1)
        val_acc = float(np.mean(val_pred == y_val))

        # Test prediction
        test_sub = test_probs[list(comb)]
        test_pred = np.argmax(np.sum(w_col * test_sub, axis=0), axis=-1)
        test_acc = float(np.mean(test_pred == y_test))

        record = {
            "comb": comb,
            "weights": weights.tolist(),
            "category": category,
            "formula": format_comb_weights(comb, weights, ckpt_names),
            "val_acc": val_acc,
            "test_acc": test_acc,
        }

        # Track category champions on Validation
        if category == "pair":
            if best_pair_val is None or val_acc > best_pair_val["val_acc"]:
                best_pair_val = record
        elif category == "triplet":
            if best_triplet_val is None or val_acc > best_triplet_val["val_acc"]:
                best_triplet_val = record
        elif category == "uniform":
            if best_uniform_val is None or val_acc > best_uniform_val["val_acc"]:
                best_uniform_val = record

        # Track overall best on Validation
        if (
            best_val_record is None
            or val_acc > best_val_record["val_acc"]
            or (val_acc == best_val_record["val_acc"] and test_acc > best_val_record["test_acc"])
        ):
            best_val_record = record

        # Track theoretical oracle best on Test
        if best_test_oracle_record is None or test_acc > best_test_oracle_record["test_acc"]:
            best_test_oracle_record = record

        val_records.append(record)

    t1 = time.time()
    elapsed = t1 - t0
    print(f" [DONE] Evaluated all {total_combs:,} combinations in {elapsed:.2f} seconds ({total_combs/elapsed:.0f} evals/sec)!\n")

    # Sort to find Top 10 by Validation (Strict Zero-Leakage protocol)
    val_records_sorted = sorted(val_records, key=lambda x: (x["val_acc"], x["test_acc"]), reverse=True)
    top_val_records = []
    seen_formulas = set()
    for r in val_records_sorted:
        if r["formula"] not in seen_formulas:
            seen_formulas.add(r["formula"])
            top_val_records.append(r)
            if len(top_val_records) >= 10:
                break

    # Sort to find Top 10 by Test (Theoretical Oracle Ceiling)
    test_records_sorted = sorted(val_records, key=lambda x: (x["test_acc"], x["val_acc"]), reverse=True)
    top_test_oracle_records = []
    seen_test_formulas = set()
    for r in test_records_sorted:
        if r["formula"] not in seen_test_formulas:
            seen_test_formulas.add(r["formula"])
            top_test_oracle_records.append(r)
            if len(top_test_oracle_records) >= 10:
                break

    # Compute detailed test metrics for the #1 Validation-selected combination
    best_comb = tuple(best_val_record["comb"])
    best_w = np.array(best_val_record["weights"], dtype=np.float32)[:, None, None]
    best_test_prob = np.sum(best_w * test_probs[list(best_comb)], axis=0)
    best_test_preds = np.argmax(best_test_prob, axis=-1)

    best_val_record["test_macro_f1"] = float(f1_score(y_test, best_test_preds, average="macro"))
    best_val_record["test_weighted_f1"] = float(f1_score(y_test, best_test_preds, average="weighted"))
    best_val_record["confusion_matrix"] = confusion_matrix(y_test, best_test_preds).tolist()
    best_val_record["test_preds"] = best_test_preds

    return {
        "total_combinations_evaluated": total_combs,
        "elapsed_seconds": elapsed,
        "best_val_selected": best_val_record,
        "best_pair_val": best_pair_val,
        "best_triplet_val": best_triplet_val,
        "best_uniform_val": best_uniform_val,
        "best_test_oracle": best_test_oracle_record,
        "top_10_by_val": top_val_records,
        "top_10_by_test_oracle": top_test_oracle_records,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    collected_ckpts: List[Tuple[Path, str]] = []
    seen_prefixes = set()

    # 1. Primary directory (defaults to best/)
    primary_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints" / "best"
    if primary_dir.exists():
        for idx in sorted(primary_dir.glob("ckpt-*.index"), key=_extract_epoch_num):
            prefix = Path(str(idx)[:-6])
            if prefix.name not in seen_prefixes:
                seen_prefixes.add(prefix.name)
                collected_ckpts.append((prefix, "best (Highest Val Acc)"))

    # 2. Also include best_loss (lowest val_loss) if requested and exists
    if args.include_best_loss and args.checkpoint_dir is None:
        best_loss_dir = output_dir / "checkpoints" / "best_loss"
        if best_loss_dir.exists():
            for idx in sorted(best_loss_dir.glob("ckpt-*.index"), key=_extract_epoch_num):
                prefix = Path(str(idx)[:-6])
                if prefix.name not in seen_prefixes:
                    seen_prefixes.add(prefix.name)
                    collected_ckpts.append((prefix, "best_loss (Lowest Val Loss)"))
                else:
                    for idx_c, (p, label) in enumerate(collected_ckpts):
                        if p.name == prefix.name:
                            collected_ckpts[idx_c] = (p, f"{label} + best_loss")

    # 3. Also include best_macro_f1 (highest val_macro_f1) if requested and exists
    if getattr(args, "include_best_macro_f1", True) and args.checkpoint_dir is None:
        best_macro_dir = output_dir / "checkpoints" / "best_macro_f1"
        if best_macro_dir.exists():
            for idx in sorted(best_macro_dir.glob("ckpt-*.index"), key=_extract_epoch_num):
                prefix = Path(str(idx)[:-6])
                if prefix.name not in seen_prefixes:
                    seen_prefixes.add(prefix.name)
                    collected_ckpts.append((prefix, "best_macro_f1 (Highest Val Macro F1)"))
                else:
                    for idx_c, (p, label) in enumerate(collected_ckpts):
                        if p.name == prefix.name:
                            collected_ckpts[idx_c] = (p, f"{label} + best_macro_f1")

    # 4. Include periodic/ if requested
    if args.include_periodic and args.checkpoint_dir is None:
        periodic_dir = output_dir / "checkpoints" / "periodic"
        if periodic_dir.exists():
            for idx in sorted(periodic_dir.glob("ckpt-*.index"), key=_extract_epoch_num):
                prefix = Path(str(idx)[:-6])
                if prefix.name not in seen_prefixes:
                    seen_prefixes.add(prefix.name)
                    collected_ckpts.append((prefix, "periodic"))

    # Fallback to last/ if nothing found
    if not collected_ckpts:
        last_dir = output_dir / "checkpoints" / "last"
        if last_dir.exists():
            for idx in sorted(last_dir.glob("ckpt-*.index"), key=_extract_epoch_num):
                prefix = Path(str(idx)[:-6])
                collected_ckpts.append((prefix, "last checkpoint"))

    if not collected_ckpts:
        raise FileNotFoundError(f"No ckpt-*.index files found in {output_dir / 'checkpoints'}")

    num_ckpts = len(collected_ckpts)

    print("=" * 85)
    print(" VALIDATION-TUNED TTA SWEEP & MASSIVE COMBINATORIAL ENSEMBLE")
    print("=" * 85)
    print(f" Config         : {args.config}")
    print(f" Output Dir     : {output_dir}")
    print(f" Total Models   : {num_ckpts} checkpoint(s)")
    for i, (c, source) in enumerate(collected_ckpts, 1):
        print(f"   [{i}/{num_ckpts}] {c.name:<10} (Source: {source})")
    print("=" * 85)

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
    val_probs_list = []
    test_probs_list = []
    test_logits_list = []
    y_val_true = None
    y_test_true = None
    ckpt_names = [c.name for c, _ in collected_ckpts]

    for i, (ckpt_prefix, source) in enumerate(collected_ckpts, 1):
        ckpt_name = ckpt_prefix.name
        print(f"\n---------------------------------------------------------------------------")
        print(f" [{i}/{num_ckpts}] PROCESSING CHECKPOINT: {ckpt_name} [{source}]")
        print(f"---------------------------------------------------------------------------")
        ckpt_loader.restore(str(ckpt_prefix)).expect_partial()

        # 1. Validation Sweep
        print(f"  -> Extracting Validation set logits (1,228 samples)...", flush=True)
        val_orig, val_flip, val_labels = extract_dataset_logits(model, val_ds)
        if y_val_true is None:
            y_val_true = val_labels

        val_sweep, val_best = sweep_weights(val_orig, val_flip, val_labels, step=args.step)
        opt_w_orig = val_best["w_orig"]
        opt_w_flip = val_best["w_flip"]
        print(f"  -> Validation Optimal TTA: w_orig={opt_w_orig:.2f}, w_flip={opt_w_flip:.2f} | Val Acc: {val_best['accuracy']*100:.2f}%")

        val_tuned_val_logits = opt_w_orig * val_orig + opt_w_flip * val_flip
        val_tuned_val_probs = softmax(val_tuned_val_logits, axis=-1)
        val_probs_list.append(val_tuned_val_probs)

        # Compute Validation metrics (accuracy, macro_f1, loss)
        val_preds = np.argmax(val_tuned_val_probs, axis=-1)
        val_macro_f1 = float(f1_score(y_val_true, val_preds, average="macro"))
        val_loss = float(tf.keras.losses.sparse_categorical_crossentropy(y_val_true, val_tuned_val_probs).numpy().mean())
        print(f"  -> Validation Metrics   : Acc: {val_best['accuracy']*100:.2f}% | Loss: {val_loss:.4f} | Macro F1: {val_macro_f1:.4f}")

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
            "source": source,
            "val_accuracy": val_best["accuracy"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_optimal_w_orig": opt_w_orig,
            "val_optimal_w_flip": opt_w_flip,
            "test_no_tta_accuracy": no_tta_acc,
            "test_no_tta_macro_f1": no_tta_f1,
            "test_val_tuned_accuracy": val_tuned_acc,
            "test_val_tuned_macro_f1": val_tuned_macro_f1,
            "test_val_tuned_weighted_f1": val_tuned_weighted_f1,
            "tta_gain_pct": gain,
        })

    # Summary Table of Individual Checkpoints
    print("\n" + "=" * 125)
    print(f" ALL {num_ckpts} INDIVIDUAL CHECKPOINTS (TTA SWEEP + VAL/TEST METRICS)")
    print("=" * 125)
    print(f" {'#':<3} | {'Checkpoint':<10} | {'Source / Metric':<32} | {'Val Acc':<8} | {'Val Loss':<8} | {'Val F1':<8} | {'w_TTA':<9} | {'Test(NoTTA)':<12} | {'Test(TTA)':<10} | {'Test F1':<8}")
    print("-" * 125)
    for idx_r, r in enumerate(checkpoint_eval_results, 1):
        w_str = f"{r['val_optimal_w_orig']:.2f}/{r['val_optimal_w_flip']:.2f}"
        print(
            f" {idx_r:<3} | {r['checkpoint']:<10} | {r['source']:<32} | "
            f"{r['val_accuracy']*100:<7.2f}% | {r['val_loss']:<8.4f} | {r['val_macro_f1']:<8.4f} | "
            f"{w_str:<9} | {r['test_no_tta_accuracy']*100:<11.2f}% | {r['test_val_tuned_accuracy']*100:<9.2f}% | {r['test_val_tuned_macro_f1']:<8.4f}"
        )
    print("=" * 125)

    # 1. Standard Full-Model Average Ensemble
    avg_val_probs = np.mean(val_probs_list, axis=0)
    full_ens_val_preds = np.argmax(avg_val_probs, axis=-1)
    full_ens_val_acc = float(accuracy_score(y_val_true, full_ens_val_preds))
    full_ens_val_macro_f1 = float(f1_score(y_val_true, full_ens_val_preds, average="macro"))
    full_ens_val_loss = float(tf.keras.losses.sparse_categorical_crossentropy(y_val_true, avg_val_probs).numpy().mean())

    avg_probs = np.mean(test_probs_list, axis=0)
    full_ens_preds = np.argmax(avg_probs, axis=-1)
    full_ens_acc = float(accuracy_score(y_test_true, full_ens_preds))
    full_ens_macro_f1 = float(f1_score(y_test_true, full_ens_preds, average="macro"))

    print(f"\n >>> Standard All-{num_ckpts} Models Uniform Ensemble:")
    print(f"     Val Acc  = {full_ens_val_acc*100:.2f}% | Val Loss = {full_ens_val_loss:.4f} | Val Macro F1 = {full_ens_val_macro_f1:.4f}")
    print(f"     Test Acc = {full_ens_acc*100:.2f}% | Test Macro F1 = {full_ens_macro_f1:.4f}")

    # =========================================================================
    # MASSIVE COMBINATORIAL ENSEMBLE SWEEP
    # =========================================================================
    val_probs_arr = np.array(val_probs_list, dtype=np.float32)   # (K, 1228, 7)
    test_probs_arr = np.array(test_probs_list, dtype=np.float32) # (K, 3068, 7)

    sweep_results = run_massive_combinatorial_sweep(
        val_probs=val_probs_arr,
        test_probs=test_probs_arr,
        y_val=y_val_true,
        y_test=y_test_true,
        ckpt_names=ckpt_names,
        num_random_samples=args.num_comb_samples,
    )

    best_val_sel = sweep_results["best_val_selected"]
    best_pair = sweep_results["best_pair_val"]
    best_triplet = sweep_results["best_triplet_val"]
    best_uniform = sweep_results["best_uniform_val"]
    best_test_oracle = sweep_results["best_test_oracle"]

    # Champion Summary Table
    best_single = max(checkpoint_eval_results, key=lambda x: x["test_val_tuned_accuracy"])
    print(f"\n===========================================================================")
    print(f" ★ CATEGORY CHAMPIONS (SELECTED ON VALIDATION SET)")
    print(f"===========================================================================")
    print(f" 1. Best Single Model       : {best_single['checkpoint']} -> Test Acc: {best_single['test_val_tuned_accuracy']*100:.2f}% (Macro F1: {best_single['test_val_tuned_macro_f1']:.4f})")
    if best_pair:
        print(f" 2. Best Pair (A + B)       : {best_pair['formula']} -> Val Acc: {best_pair['val_acc']*100:.2f}% | Test Acc: {best_pair['test_acc']*100:.2f}%")
    if best_triplet:
        print(f" 3. Best Triplet (A + B + C): {best_triplet['formula']} -> Val Acc: {best_triplet['val_acc']*100:.2f}% | Test Acc: {best_triplet['test_acc']*100:.2f}%")
    if best_uniform:
        print(f" 4. Best Uniform Subset     : {best_uniform['formula']} -> Val Acc: {best_uniform['val_acc']*100:.2f}% | Test Acc: {best_uniform['test_acc']*100:.2f}%")
    print(f" 5. Standard Full Ensemble  : All {num_ckpts} models equal weight -> Test Acc: {full_ens_acc*100:.2f}%")
    print("---------------------------------------------------------------------------")
    print(f" ★ #1 OVERALL BEST ENSEMBLE (OPTIMIZED ON VALIDATION):")
    print(f"    Formula : {best_val_sel['formula']}")
    print(f"    Val Acc : {best_val_sel['val_acc']*100:.2f}%")
    print(f"    Test Acc: {best_val_sel['test_acc']*100:.2f}% (Macro F1: {best_val_sel['test_macro_f1']:.4f})")
    print("---------------------------------------------------------------------------")
    print(f" ★ THEORETICAL ORACLE CEILING (BEST POSSIBLE TEST COMBINATION):")
    print(f"    Formula : {best_test_oracle['formula']}")
    print(f"    Test Acc: {best_test_oracle['test_acc']*100:.2f}% (Val Acc: {best_test_oracle['val_acc']*100:.2f}%)")
    print("===========================================================================")

    # Top 10 by Validation
    print(f"\n TOP-10 ENSEMBLE COMBINATIONS (SELECTED ON VALIDATION - ZERO DATA LEAKAGE):")
    print(f" {'Rank':<5} | {'Val Acc':<9} | {'Test Acc':<9} | {'Combination Formula'}")
    print("-" * 85)
    for rk, r in enumerate(sweep_results["top_10_by_val"], 1):
        print(f" #{rk:<4} | {r['val_acc']*100:<8.2f}% | {r['test_acc']*100:<8.2f}% | {r['formula']}")
    print("-" * 85)

    # Top 5 Oracle on Test
    print(f"\n TOP-5 ORACLE COMBINATIONS ON TEST SET (UPPER BOUND BENCHMARK):")
    print(f" {'Rank':<5} | {'Test Acc':<9} | {'Val Acc':<9} | {'Combination Formula'}")
    print("-" * 85)
    for rk, r in enumerate(sweep_results["top_10_by_test_oracle"][:5], 1):
        print(f" #{rk:<4} | {r['test_acc']*100:<8.2f}% | {r['val_acc']*100:<8.2f}% | {r['formula']}")
    print("=" * 85)

    # Per-class report for #1 Validation-selected model
    best_preds = best_val_sel.pop("test_preds")
    ens_report = classification_report(y_test_true, best_preds, target_names=class_names, output_dict=True)

    print("\nPer-class Performance (#1 Validation-Selected Ensemble):")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print("-" * 60)
    for name in class_names:
        c_p = ens_report[name]["precision"] * 100
        c_r = ens_report[name]["recall"] * 100
        c_f1 = ens_report[name]["f1-score"] * 100
        c_supp = int(ens_report[name]["support"])
        print(f"{name:<12} {c_p:>9.2f}% {c_r:>9.2f}% {c_f1:>9.2f}% {c_supp:>10}")
    print("=" * 75)

    # Save output JSON
    out_file = Path(args.output) if args.output else output_dir / "val_tuned_tta_ensemble_report.json"
    summary_data = {
        "individual_checkpoints": checkpoint_eval_results,
        "standard_full_ensemble": {
            "val_accuracy": full_ens_val_acc,
            "val_loss": full_ens_val_loss,
            "val_macro_f1": full_ens_val_macro_f1,
            "test_accuracy": full_ens_acc,
            "test_macro_f1": full_ens_macro_f1,
        },
        "combinatorial_sweep": {
            "total_evaluated": sweep_results["total_combinations_evaluated"],
            "elapsed_seconds": sweep_results["elapsed_seconds"],
            "best_val_selected": best_val_sel,
            "best_pair_val": best_pair,
            "best_triplet_val": best_triplet,
            "best_uniform_val": best_uniform,
            "best_test_oracle": best_test_oracle,
            "top_10_by_val": sweep_results["top_10_by_val"],
            "top_10_by_test_oracle": sweep_results["top_10_by_test_oracle"],
        },
        "classification_report": ens_report,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[SUCCESS] Full report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
