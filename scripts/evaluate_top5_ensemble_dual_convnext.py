#!/usr/bin/env python3
"""
Top-5 Checkpoint Softmax Ensemble Evaluator for Dual-ConvNeXt SMIRK 3D Guided Attention Model.

Loads the Top-K best validation accuracy checkpoints saved during training, runs TTA (Horizontal Flip)
evaluation for each checkpoint, computes Softmax probability predictions, and averages them to produce
the final Ensemble accuracy and classification metrics.
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
import yaml
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from models.dual_convnext_smirk_guided_attention_ms1m_fer_scratch import (
    DualConvNeXtSMIRKGuidedAttentionMS1MFERScratch,
)
from scripts.train_dual_convnext_smirk_guided_attention_ms1m_fer_scratch import (
    load_geometry_cache,
    create_dataset,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Top-K Checkpoint Ensemble for Dual-ConvNeXt FER")
    parser.add_argument(
        "--config",
        type=str,
        default="config_dual_convnext_smirk_guided_attention_ms1m_fer_scratch.yaml",
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--top-k-json",
        type=str,
        default=None,
        help="Path to top_k_checkpoints.json. If None, auto-resolves from config output_dir.",
    )
    parser.add_argument("--disable-tta", action="store_true", help="Disable Test-Time Augmentation (Horizontal Flip)")
    return parser.parse_args()


def evaluate_checkpoint_probs(model, dataset: tf.data.Dataset, use_tta: bool = True):
    all_probs = []
    all_labels = []

    for inputs, y_batch in dataset:
        outputs_orig = model(inputs, training=False)
        logits_orig = outputs_orig["final_logits"]

        if use_tta:
            flipped_inputs = {
                "image": tf.image.flip_left_right(inputs["image"]),
                "geometry_maps": tf.image.flip_left_right(inputs["geometry_maps"]),
            }
            outputs_flip = model(flipped_inputs, training=False)
            logits_flip = outputs_flip["final_logits"]
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


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = resolve_path(cfg["paths"]["output_dir"])

    def load_geometry_cache(cache_dir: Path, pattern: str, split: str) -> Dict[str, np.ndarray]:
        target_name = pattern.format(split=split)
        npz_path = cache_dir / target_name
        if not npz_path.exists():
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                for root, _, files in os.walk(kaggle_input):
                    if target_name in files:
                        npz_path = Path(root) / target_name
                        print(f"[INFO] Auto-resolved Kaggle geometry cache for {split} -> {npz_path}", flush=True)
                        break
        if not npz_path.exists():
            raise FileNotFoundError(f"Geometry cache map '{target_name}' not found under {cache_dir} or /kaggle/input")
        return np.load(npz_path)

    json_path = Path(args.top_k_json) if args.top_k_json else output_dir / "top_k_checkpoints.json"
    if not json_path.exists():
        print(f"[ERROR] Top-K JSON file not found at: {json_path}")
        return 1

    with json_path.open("r", encoding="utf-8") as f:
        top_k_items = json.load(f)

    print("\n" + "=" * 76)
    print(f" TOP-K CHECKPOINT ENSEMBLE EVALUATION")
    print(f" Config: {args.config}")
    print(f" Found {len(top_k_items)} Top Checkpoints in: {json_path}")
    print(f" TTA Horizontal Flip: {'ENABLED' if not args.disable_tta else 'DISABLED'}")
    print("=" * 76)

    data_path = resolve_path(cfg["data"]["data_path"])
    cache_dir = resolve_path(cfg["geometry_cache"]["feature_dir"])
    pattern = cfg["geometry_cache"]["map_file_pattern"]

    val_records = collect_split_records(data_path, split="val", predecode_pixels=True)
    geom_cache = load_geometry_cache(cache_dir, pattern, split="val")
    val_ds = create_dataset(val_records, geom_cache, cfg, batch_size=int(cfg["data"]["batch_size"]), is_training=False)

    # Initialize model architecture
    model = DualConvNeXtSMIRKGuidedAttentionMS1MFERScratch(
        num_classes=int(cfg["data"]["num_classes"]),
        rgb_pretrained_path=cfg["rgb_backbone"]["convnext_base_pretrained_path"],
        feature_dim=int(cfg["model"].get("feature_dim", 1024)),
        attention_hidden_dim=int(cfg["model"].get("attention_hidden_dim", 256)),
        alpha_max=float(cfg["model"].get("alpha_max", 0.2)),
        dropout1=float(cfg["rgb_backbone"].get("classifier_dropout1", 0.35)),
    )

    # Dummy forward pass to initialize variables
    for dummy_inputs, _ in val_ds.take(1):
        _ = model(dummy_inputs, training=False)
    print(f"[INFO] Model initialized with {len(model.variables)} variables.")

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

    # Ensemble Average Probabilities across all Top-K models
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
    print(f" 🔥 TOP-{len(single_probs_list)} ENSEMBLE EVALUATION RESULTS 🔥")
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

    out_file = output_dir / "top_k_ensemble_final_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[SAVED] Ensemble report saved to {out_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
