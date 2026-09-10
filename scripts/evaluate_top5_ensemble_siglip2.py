#!/usr/bin/env python3
"""
Top-5 Checkpoint Ensemble & Individual Evaluator for SigLIP 2 FER Models.

Evaluates each individual checkpoint in `checkpoints/best/` (e.g. ckpt-8, ckpt-9, ckpt-11,
ckpt-14, ckpt-24) with TTA (Horizontal Flip), then computes the Softmax probability ensemble.
Outputs Accuracy, Macro-F1, Weighted-F1, and Confusion Matrices for both individual models
and the final Ensemble.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# If CPU mode requested via env var before TF initializes
if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, resolve_auto_increment_output_dir
from datasets.fer2013 import build_datasets
from metrics.classification import save_metrics
from train import build_model, configure_gpus, configure_tensorflow_runtime, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Individual Checkpoints & Ensemble with TTA for SigLIP2 FER")
    parser.add_argument(
        "--config",
        type=str,
        default="config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Specific directory with checkpoints (default: scans output_dir/checkpoints/best)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Explicit list of checkpoint names or prefixes (e.g. ckpt-8 ckpt-9 ckpt-11 ckpt-14 ckpt-24)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "both"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation (CUDA_VISIBLE_DEVICES=-1)")
    parser.add_argument("--w-orig", type=float, default=None, help="TTA original image weight (default: from config)")
    parser.add_argument("--w-flip", type=float, default=None, help="TTA flipped image weight (default: from config)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON report")
    return parser.parse_args()


def _extract_epoch_num(path: Path) -> int:
    name = path.stem if path.name.endswith(".index") else path.name
    digits = re.findall(r"\d+", name)
    return int(digits[-1]) if digits else 0


def get_checkpoint_list(ckpt_dir: Path, explicit_ckpts: list[str] | None = None) -> list[Path]:
    if explicit_ckpts:
        prefixes = []
        for item in explicit_ckpts:
            clean = item[:-6] if item.endswith(".index") else item
            p = Path(clean)
            if not p.is_absolute():
                p = ckpt_dir / p.name
            prefixes.append(p)
        return prefixes

    index_files = list(ckpt_dir.glob("ckpt-*.index"))
    if not index_files:
        return []
    # Sort chronologically by epoch number
    index_files = sorted(index_files, key=_extract_epoch_num)
    prefixes = [Path(str(p)[:-6]) for p in index_files]
    return prefixes


def extract_probs(model, dataset, w_orig=0.50, w_flip=0.50):
    all_probs = []
    all_labels = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, batch["label"]

        outputs_orig = model(inputs, training=False)
        logits_orig = outputs_orig["logits"].numpy()

        flipped_inputs = dict(inputs)
        flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
        if "mask" in inputs:
            flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
        outputs_flip = model(flipped_inputs, training=False)
        logits_flip = outputs_flip["logits"].numpy()

        ensemble_logits = w_orig * logits_orig + w_flip * logits_flip

        exp_logits = np.exp(ensemble_logits - np.max(ensemble_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return probs_arr, labels_arr


def print_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str = "Confusion Matrix"):
    print(f"\n{title} (Rows: Ground Truth, Columns: Prediction):")
    header = f"{'':>10} " + " ".join(f"{name[:4]:>6}" for name in class_names)
    print(header)
    for idx, row in enumerate(cm):
        row_str = " ".join(f"{val:>6}" for val in row)
        print(f"{class_names[idx]:>10} {row_str}")


def print_classification_table(report: dict, class_names: list[str]):
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print("-" * 65)
    for name in class_names:
        if name in report:
            p = report[name].get("precision", 0.0)
            r = report[name].get("recall", 0.0)
            f = report[name].get("f1-score", 0.0)
            s = int(report[name].get("support", 0))
            print(f"{name:<12} {p * 100:>9.2f}% {r * 100:>9.2f}% {f * 100:>9.2f}% {s:>10}")
    print("-" * 65)


def main() -> int:
    args = parse_args()
    if args.cpu or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    cfg = load_config(args.config)
    class_names = get_class_names(cfg)
    label_ids = list(range(len(class_names)))
    resolve_auto_increment_output_dir(cfg, for_eval=True)

    is_cpu = bool(
        args.cpu
        or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        or len(tf.config.list_physical_devices("GPU")) == 0
    )
    if is_cpu:
        cfg["runtime"]["allow_cpu_fallback"] = True
        cfg["runtime"]["min_gpus"] = 0
        cfg["runtime"]["use_mixed_precision"] = False
        tf.keras.mixed_precision.set_global_policy("float32")

    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))

    if not is_cpu:
        configure_gpus(cfg)
        visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
        strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(max(visible_gpu_count, 1))])
    else:
        print("[INFO] Running in CPU mode (CUDA_VISIBLE_DEVICES=-1).", flush=True)
        strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])

    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = output_dir / "checkpoints" / "best"
        if not ckpt_dir.exists():
            ckpt_dir = output_dir / "checkpoints" / "best_loss"

    ckpt_prefixes = get_checkpoint_list(ckpt_dir, args.checkpoints)
    if not ckpt_prefixes:
        print(f"[ERROR] No checkpoint index files found in {ckpt_dir}")
        return 1

    # TTA weights from config or arguments
    w_orig = args.w_orig if args.w_orig is not None else float(cfg.get("tta", {}).get("original_weight", 0.50))
    w_flip = args.w_flip if args.w_flip is not None else float(cfg.get("tta", {}).get("flip_weight", 0.50))
    tot_w = w_orig + w_flip
    if tot_w > 0:
        w_orig /= tot_w
        w_flip /= tot_w

    data_cfg = cfg.get("data", {})
    expected_samples = data_cfg.get("expected_samples", {})

    print("\n" + "=" * 70)
    print("      CHECKPOINTS EVALUATION & ENSEMBLE (FERPlus Protocol)")
    print("=" * 70)
    print(f" Config            : {args.config}")
    print(f" Checkpoint Dir    : {ckpt_dir}")
    print(f" Checkpoints ({len(ckpt_prefixes)}): {[p.name for p in ckpt_prefixes]}")
    print(f" Target Split      : {args.split.upper()}")
    print(f" Number of Classes : {len(class_names)} ({', '.join(class_names)})")
    print(f" TTA Weights       : Orig={w_orig:.2f}, Flip={w_flip:.2f}")
    print(f" Compute Device    : {'CPU (/CPU:0)' if is_cpu else 'GPU'}")
    print(f" Weight Updatable  : FALSE (100% read-only inference)")
    print("=" * 70 + "\n", flush=True)

    _, val_ds, test_ds = build_datasets(cfg, replicas=strategy.num_replicas_in_sync)

    first_ds = test_ds if args.split == "test" else val_ds
    first_batch = next(iter(first_ds.take(1)))
    first_inputs = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch

    with strategy.scope():
        model = build_model(cfg)
        _ = model(first_inputs, training=False)
        ckpt = tf.train.Checkpoint(model=model)

    results_report = {
        "dataset_type": data_cfg.get("dataset_type", "ferplus_majority8"),
        "class_names": class_names,
        "split": args.split,
        "tta_weights": {"orig": w_orig, "flip": w_flip},
        "individual_checkpoints": {},
        "ensemble": {},
    }

    # Splits to evaluate
    splits_to_run = ["test"] if args.split == "test" else (["val"] if args.split == "val" else ["val", "test"])

    for cur_split in splits_to_run:
        cur_ds = test_ds if cur_split == "test" else val_ds
        cur_exp_count = expected_samples.get(cur_split, "N/A")
        print(f"\n>>> Starting evaluation on split: {cur_split.upper()} (expected: {cur_exp_count} samples) <<<", flush=True)

        probs_by_ckpt = []
        ground_truth = None

        for idx, prefix in enumerate(ckpt_prefixes, start=1):
            ckpt_name = prefix.name
            print(f"\n[EVAL {idx}/{len(ckpt_prefixes)}] Loading checkpoint: {ckpt_name} ...", flush=True)
            ckpt.restore(str(prefix)).expect_partial()

            probs, labs = extract_probs(model, cur_ds, w_orig=w_orig, w_flip=w_flip)
            preds = np.argmax(probs, axis=1)

            acc = float(accuracy_score(labs, preds))
            macro_f1 = float(f1_score(labs, preds, average="macro", labels=label_ids, zero_division=0))
            weighted_f1 = float(f1_score(labs, preds, average="weighted", labels=label_ids, zero_division=0))
            cm = confusion_matrix(labs, preds, labels=label_ids).tolist()
            report = classification_report(labs, preds, labels=label_ids, target_names=class_names, output_dict=True, zero_division=0)

            ckpt_metrics = {
                "checkpoint": str(prefix),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "confusion_matrix": cm,
                "classification_report": report,
            }

            if ckpt_name not in results_report["individual_checkpoints"]:
                results_report["individual_checkpoints"][ckpt_name] = {}
            results_report["individual_checkpoints"][ckpt_name][cur_split] = ckpt_metrics

            # Also save individual metrics JSON compatible with evaluate.py
            single_out_path = output_dir / f"{cur_split}_metrics_{ckpt_name}_tta_hflip.json"
            save_metrics(ckpt_metrics, single_out_path)

            tag = " ⭐ (Target CKPT)" if ckpt_name == "ckpt-24" else ""
            print(f"  --> {ckpt_name:8s}{tag} | Accuracy: {acc * 100:6.2f}% | Macro-F1: {macro_f1 * 100:6.2f}% ({macro_f1:.4f}) | W-F1: {weighted_f1 * 100:6.2f}%", flush=True)

            probs_by_ckpt.append(probs)
            ground_truth = labs

        # --- ENSEMBLE AVERAGE ---
        ens_probs = np.mean(probs_by_ckpt, axis=0)
        ens_preds = np.argmax(ens_probs, axis=1)

        ens_acc = float(accuracy_score(ground_truth, ens_preds))
        ens_macro_f1 = float(f1_score(ground_truth, ens_preds, average="macro", labels=label_ids, zero_division=0))
        ens_weighted_f1 = float(f1_score(ground_truth, ens_preds, average="weighted", labels=label_ids, zero_division=0))
        ens_cm = confusion_matrix(ground_truth, ens_preds, labels=label_ids).tolist()
        ens_report = classification_report(ground_truth, ens_preds, labels=label_ids, target_names=class_names, output_dict=True, zero_division=0)

        results_report["ensemble"][cur_split] = {
            "num_checkpoints": len(ckpt_prefixes),
            "checkpoints": [p.name for p in ckpt_prefixes],
            "accuracy": ens_acc,
            "macro_f1": ens_macro_f1,
            "weighted_f1": ens_weighted_f1,
            "confusion_matrix": ens_cm,
            "classification_report": ens_report,
        }

        # --- PRINT SUMMARY TABLE FOR THIS SPLIT ---
        print("\n" + "=" * 74)
        print(f"        SUMMARY: INDIVIDUAL CHECKPOINTS VS ENSEMBLE ({cur_split.upper()})")
        print("=" * 74)
        print(f"{'Checkpoint':<18} {'Accuracy (%)':>14} {'Macro-F1':>14} {'Weighted-F1':>14}")
        print("-" * 74)
        for p in ckpt_prefixes:
            m = results_report["individual_checkpoints"][p.name][cur_split]
            star = " (ckpt-24)" if p.name == "ckpt-24" else ""
            lbl = f"{p.name}{star}"
            print(f"{lbl:<18} {m['accuracy'] * 100:>13.2f}% {m['macro_f1']:>14.4f} {m['weighted_f1']:>14.4f}")
        print("-" * 74)
        print(f"{'TOP-5 ENSEMBLE 🔥':<18} {ens_acc * 100:>13.2f}% {ens_macro_f1:>14.4f} {ens_weighted_f1:>14.4f}")
        print("=" * 74)

        # Print Confusion Matrices for ckpt-24 and Ensemble
        if "ckpt-24" in results_report["individual_checkpoints"]:
            c24_cm = results_report["individual_checkpoints"]["ckpt-24"][cur_split]["confusion_matrix"]
            print_confusion_matrix(c24_cm, class_names, title=f"CONFUSION MATRIX: ckpt-24 ({cur_split.upper()})")

        print_confusion_matrix(ens_cm, class_names, title=f"CONFUSION MATRIX: TOP-{len(ckpt_prefixes)} ENSEMBLE ({cur_split.upper()})")

        print(f"\n--- Classification Report: TOP-{len(ckpt_prefixes)} ENSEMBLE ({cur_split.upper()}) ---")
        print_classification_table(ens_report, class_names)

    # Save comprehensive report
    out_file = Path(args.output) if args.output else output_dir / f"ferplus_{len(ckpt_prefixes)}ckpts_and_ensemble_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print(f" [SUCCESS] Evaluation completed for all {len(ckpt_prefixes)} checkpoints + Ensemble!")
    print(f" Report saved to: {out_file.resolve()}")
    print("=" * 74 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
