#!/usr/bin/env python3
"""
Create Model Soup / Stochastic Weight Averaging (SWA) from multiple checkpoints.
Averages weights across specified checkpoints and saves a unified soup checkpoint.
Compatible with standard TensorFlow Checkpoint restoration in evaluate.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from config import load_config
from train import build_model, configure_gpus, configure_tensorflow_runtime
from datasets.fer2013 import build_datasets, EMOTION_NAMES
from metrics.classification import classification_metrics


def parse_args():
    parser = argparse.ArgumentParser("Create and evaluate Model Soup from multiple checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint prefixes or paths (e.g. ckpt-50 ckpt-60)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Target output checkpoint prefix. Default: <exp_dir>/checkpoints/soup/ckpt-soup",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional relative weights for each checkpoint (default: uniform average)",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the soup checkpoint on test set")
    parser.add_argument("--tta-hflip", action="store_true", default=True, help="Enable TTA H-Flip during eval")
    return parser.parse_args()


def resolve_checkpoint_prefix(raw_path: str, exp_dir: Path) -> Path:
    p = Path(raw_path)
    if str(p).endswith(".index"):
        p = Path(str(p)[:-6])

    if p.exists() or Path(str(p) + ".index").exists():
        return p

    # Search in exp_dir subdirs
    for sub in ["checkpoints/periodic", "checkpoints/last", "checkpoints/best", "checkpoints"]:
        cand = exp_dir / sub / p.name
        if Path(str(cand) + ".index").exists():
            return cand

    raise FileNotFoundError(f"Cannot resolve checkpoint: {raw_path} (checked {exp_dir})")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp_dir = Path(cfg["paths"]["output_dir"])

    resolved_ckpts: List[Path] = [resolve_checkpoint_prefix(c, exp_dir) for c in args.checkpoints]
    k = len(resolved_ckpts)
    print("=" * 70)
    print(f" MODEL SOUP (WEIGHT AVERAGING) - {k} CHECKPOINTS")
    print("=" * 70)
    for i, c in enumerate(resolved_ckpts, 1):
        print(f"  [{i}/{k}] {c}")

    if args.weights is not None:
        assert len(args.weights) == k, f"Number of weights ({len(args.weights)}) != checkpoints ({k})"
        w_norm = np.array(args.weights, dtype=np.float32)
        w_norm = w_norm / np.sum(w_norm)
        print(f"  Normalized weights: {w_norm.tolist()}")
    else:
        w_norm = np.ones(k, dtype=np.float32) / float(k)
        print(f"  Uniform weights: 1/{k} = {1.0/k:.4f} each")

    # Output prefix
    if args.output_prefix is not None:
        out_prefix = Path(args.output_prefix)
    else:
        tag = "_".join([c.name.replace("ckpt-", "ep") for c in resolved_ckpts])
        out_prefix = exp_dir / "checkpoints" / f"soup_{tag}" / "ckpt-soup"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)

    # Build base model
    model = build_model(cfg)
    dummy = tf.zeros([1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]], tf.float32)
    model({"image": dummy}, training=False)

    ckpt_loader = tf.train.Checkpoint(model=model)

    accumulated_weights = None
    var_list = model.variables

    print(f"\n[INFO] Model total variables: {len(var_list)}")

    for i, (ckpt_path, weight) in enumerate(zip(resolved_ckpts, w_norm), 1):
        print(f"[{i}/{k}] Loading weights from {ckpt_path} (weight={weight:.4f})...", flush=True)
        ckpt_loader.restore(str(ckpt_path)).expect_partial()

        current_weights = [v.numpy() for v in var_list]

        if accumulated_weights is None:
            accumulated_weights = [w * weight for w in current_weights]
        else:
            for j in range(len(accumulated_weights)):
                accumulated_weights[j] += current_weights[j] * weight

    print("\n[INFO] Assigning averaged weights to model...", flush=True)
    for v, w in zip(var_list, accumulated_weights):
        v.assign(w)

    print(f"[INFO] Writing soup checkpoint to: {out_prefix}...", flush=True)
    out_saver = tf.train.Checkpoint(model=model)
    saved_path = out_saver.write(str(out_prefix))
    print(f"[SUCCESS] Soup checkpoint saved at: {saved_path}.index\n")

    if args.evaluate:
        print("=" * 70)
        print(" EVALUATING SOUP MODEL ON TEST SET (3,068 samples)")
        print("=" * 70)
        _, _, test_ds = build_datasets(cfg, replicas=1)

        # Standard eval (No TTA)
        all_preds = []
        all_labels = []
        all_preds_tta = []

        print("[EVAL] Running inference on Test set...", flush=True)
        for batch in test_ds:
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, batch["label"]

            out_orig = model(inputs, training=False)
            logits_orig = out_orig["logits"]

            pred_orig = tf.argmax(logits_orig, axis=-1)
            all_preds.append(pred_orig.numpy())
            all_labels.append(labels.numpy())

            if args.tta_hflip:
                if isinstance(inputs, dict):
                    flipped = dict(inputs)
                    flipped["image"] = tf.image.flip_left_right(inputs["image"])
                    if "mask" in inputs and inputs["mask"] is not None:
                        flipped["mask"] = tf.image.flip_left_right(inputs["mask"])
                else:
                    flipped = tf.image.flip_left_right(inputs)
                out_flip = model(flipped, training=False)
                logits_flip = out_flip["logits"]
                logits_tta = 0.5 * logits_orig + 0.5 * logits_flip
                pred_tta = tf.argmax(logits_tta, axis=-1)
                all_preds_tta.append(pred_tta.numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        no_tta_acc = np.mean(y_pred == y_true) * 100.0

        print(f"\n  >> SOUP RESULTS (No TTA)   : Accuracy = {no_tta_acc:.2f}%")

        if args.tta_hflip:
            y_pred_tta = np.concatenate(all_preds_tta, axis=0)
            tta_acc = np.mean(y_pred_tta == y_true) * 100.0
            print(f"  >> SOUP RESULTS (TTA H-Flip): Accuracy = {tta_acc:.2f}%")

        print("=" * 70)


if __name__ == "__main__":
    main()
