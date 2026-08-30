#!/usr/bin/env python3
"""
Top-5 Checkpoint Softmax Ensemble Evaluator for ConvNeXt-Base MS1M Baseline Model.

Loads the Top-K best validation accuracy checkpoints saved during baseline training,
runs TTA (Horizontal Flip) evaluation for each checkpoint, computes Softmax probability
predictions, and averages them to produce the final Ensemble accuracy and metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, collect_split_records, build_datasets
from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Top-K Checkpoint Ensemble for Baseline ConvNeXt FER")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle/config_convnext_base_ms1m_arcface_baseline_2gpu.yaml",
        help="Path to baseline YAML config",
    )
    parser.add_argument(
        "--top-k-json",
        type=str,
        default=None,
        help="Path to top_k_checkpoints.json. If None, auto-resolves from config output_dir.",
    )
    parser.add_argument("--disable-tta", action="store_true", help="Disable Test-Time Augmentation (Horizontal Flip)")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    resolved = p if p.is_absolute() else PROJECT_ROOT / p
    if not resolved.exists():
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            matches = list(kaggle_input.glob(f"**/{p.name}"))
            if matches:
                return matches[0]
    return resolved


def evaluate_checkpoint_probs(model, dataset: tf.data.Dataset, use_tta: bool = True):
    all_probs = []
    all_labels = []

    for item, y_batch in dataset:
        image = item["image"]
        outputs_orig = model({"image": image}, training=False)
        logits_orig = outputs_orig["logits"] if isinstance(outputs_orig, dict) else outputs_orig

        if use_tta:
            flipped_image = tf.image.flip_left_right(image)
            outputs_flip = model({"image": flipped_image}, training=False)
            logits_flip = outputs_flip["logits"] if isinstance(outputs_flip, dict) else outputs_flip
            logits = 0.5 * (logits_orig + logits_flip)
        else:
            logits = logits_orig

        # Softmax probability
        logits_np = logits.numpy()
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return probs_arr, labels_arr


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = resolve_path(cfg["paths"]["output_dir"])

    json_path = Path(args.top_k_json) if args.top_k_json else output_dir / "top_k_checkpoints.json"
    top_k_items = []

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            top_k_items = json.load(f)
    else:
        # Fallback: scan checkpoints directory for best acc checkpoints
        ckpt_dir = output_dir / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir = output_dir
        
        candidates = sorted(list(ckpt_dir.glob("*.index"))) + sorted(list(ckpt_dir.glob("*/*.index")))
        prefixes = [str(p.with_suffix("")) for p in candidates]
        # Keep unique prefixes up to 5
        seen = set()
        unique_prefixes = []
        for p in prefixes:
            if p not in seen:
                seen.add(p)
                unique_prefixes.append(p)
        
        top_k_items = [{"rank": i + 1, "epoch": i + 1, "val_acc": 0.0, "ckpt_prefix": p} for i, p in enumerate(unique_prefixes[:5])]

    if not top_k_items:
        print(f"[ERROR] No checkpoints found at: {output_dir}")
        return 1

    print("\n" + "=" * 76)
    print(f" BASELINE TOP-K CHECKPOINT ENSEMBLE EVALUATION")
    print(f" Config: {args.config}")
    print(f" Found {len(top_k_items)} Checkpoints for Ensemble")
    print(f" TTA Horizontal Flip: {'ENABLED' if not args.disable_tta else 'DISABLED'}")
    print("=" * 76)

    data_path = resolve_path(cfg["data"]["data_path"])
    val_records = collect_split_records(data_path, split="val", predecode_pixels=True)

    # Build validation dataset matching baseline pipeline
    _, val_ds, _ = build_datasets(cfg, custom_records={"val": val_records})

    # Initialize Baseline model
    model = ConvNeXtBaseFaceFERBaseline(cfg)

    # Dummy forward pass to build variables
    for item, _ in val_ds.take(1):
        _ = model({"image": item["image"]}, training=False)
    print(f"[INFO] Baseline Model initialized with {len(model.variables)} variables.")

    single_probs_list = []
    labels_arr = None

    for item in top_k_items:
        rank = item.get("rank", 0)
        epoch = item.get("epoch", 0)
        acc = item.get("val_acc", 0.0)
        prefix = item.get("ckpt_prefix")
        print(f"\n[EVAL] Loading Checkpoint #{rank} (Epoch {epoch:02d}, Single Acc: {acc*100:.2f}%) -> {prefix}")

        try:
            model.load_weights(prefix).expect_partial()
        except Exception as e:
            print(f"[WARNING] Could not load weights from {prefix}: {e}")
            continue

        probs, labels = evaluate_checkpoint_probs(model, val_ds, use_tta=not args.disable_tta)
        single_acc = float(np.mean(np.argmax(probs, axis=1) == labels))
        print(f"       Loaded & Evaluated successfully. Single TTA Acc: {single_acc*100:.2f}%")

        single_probs_list.append(probs)
        if labels_arr is None:
            labels_arr = labels

    if not single_probs_list:
        print("[ERROR] No checkpoints were loaded successfully!")
        return 1

    # Ensemble Average Probabilities across all Top-K baseline models
    ensemble_probs = np.mean(single_probs_list, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = float(np.mean(ensemble_preds == labels_arr))

    report = classification_report(
        labels_arr,
        ensemble_preds,
        labels=list(range(len(EMOTION_NAMES))),
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])
    cm = confusion_matrix(labels_arr, ensemble_preds, labels=list(range(7))).tolist()

    print("\n" + "*" * 76)
    print(f" 🔥 BASELINE TOP-{len(single_probs_list)} ENSEMBLE EVALUATION RESULTS 🔥")
    print(f"   Single Best Acc:     {max(item.get('val_acc', 0) for item in top_k_items)*100:.2f}%")
    print(f"   ENSEMBLE VAL ACC:    {ensemble_acc * 100:.2f}%")
    print(f"   ENSEMBLE MACRO F1:   {macro_f1:.4f}")
    print(f"   ENSEMBLE WEIGHTED F1:{weighted_f1:.4f}")
    print("*" * 76)

    # Print Per-class Accuracy
    print("\n[PER-CLASS ACCURACY]")
    for i, name in enumerate(EMOTION_NAMES):
        cls_acc = report[name]["recall"]
        print(f"  - {name:<10s}: {cls_acc*100:.2f}% (f1: {report[name]['f1-score']:.4f})")

    res = {
        "ensemble_accuracy": ensemble_acc,
        "ensemble_macro_f1": macro_f1,
        "ensemble_weighted_f1": weighted_f1,
        "num_models_ensembled": len(single_probs_list),
        "use_tta": not args.disable_tta,
        "top_k_checkpoints": top_k_items,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    out_file = output_dir / "top_k_baseline_ensemble_final_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[SAVED] Baseline Ensemble report saved to {out_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
