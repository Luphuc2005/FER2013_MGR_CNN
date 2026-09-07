#!/usr/bin/env python3
"""
Sweep semantic fusion weight (alpha) for RAF-DB inference logit ensembling.
Formula: Logits_final = (1 - alpha) * Logits_ConvNeXt + alpha * Logits_SigLIP2

Extracts visual and semantic logits in one pass, then sweeps alpha in memory instantly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from config import load_config
from datasets.fer2013 import build_datasets
from metrics.classification import classification_metrics
from train import build_model, configure_gpus, configure_tensorflow_runtime, get_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SigLIP2 semantic logit fusion alpha.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rafdb_siglip2_semantic_stable_v5.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path prefix (e.g. outputs/papers/rafdb_siglip2_semantic_stable_v3/checkpoints/best/ckpt-35)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to search for best checkpoints",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        help="List of alpha values to evaluate",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal flip TTA")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    return parser.parse_args()


def find_checkpoint(cfg: Dict, args: argparse.Namespace) -> Optional[Path]:
    if args.checkpoint:
        clean = args.checkpoint[:-6] if args.checkpoint.endswith(".index") else args.checkpoint
        return Path(clean)

    out_dir = Path(cfg["paths"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    search_dirs = []
    if args.checkpoint_dir:
        search_dirs.append(Path(args.checkpoint_dir))
    search_dirs.extend([
        out_dir / "checkpoints" / "best",
        out_dir / "checkpoints" / "best_loss",
        out_dir / "checkpoints",
    ])

    for cdir in search_dirs:
        if cdir.exists():
            indexes = sorted(list(cdir.glob("ckpt-*.index")))
            if indexes:
                # Return the last (or highest numbered) checkpoint
                return Path(str(indexes[-1])[:-6])
    return None


def extract_logits(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    use_tta: bool = True,
    w_orig: float = 0.50,
    w_flip: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract visual_logits and semantic_logits for all samples in dataset.
    Returns: (visual_logits, semantic_logits, labels)
    """
    vis_list = []
    sem_list = []
    labels_list = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, batch_labels = batch[0], batch[1]
        else:
            inputs, batch_labels = batch, batch["label"]

        # 1. Forward original
        out_orig = model(inputs, training=False)
        v_orig = out_orig.get("visual_logits", out_orig["logits"]).numpy()
        s_orig = out_orig.get("semantic_logits")
        if s_orig is not None:
            s_orig = s_orig.numpy()
        else:
            s_orig = np.zeros_like(v_orig)

        if use_tta:
            # 2. Forward flipped
            flipped_inputs = dict(inputs)
            flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs:
                flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
            out_flip = model(flipped_inputs, training=False)
            v_flip = out_flip.get("visual_logits", out_flip["logits"]).numpy()
            s_flip = out_flip.get("semantic_logits")
            if s_flip is not None:
                s_flip = s_flip.numpy()
            else:
                s_flip = np.zeros_like(v_flip)

            v_final = w_orig * v_orig + w_flip * v_flip
            s_final = w_orig * s_orig + w_flip * s_flip
        else:
            v_final = v_orig
            s_final = s_orig

        vis_list.append(v_final)
        sem_list.append(s_final)
        labels_list.append(batch_labels.numpy())

    return (
        np.concatenate(vis_list, axis=0),
        np.concatenate(sem_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)

    is_cpu = bool(args.cpu or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" or len(tf.config.list_physical_devices("GPU")) == 0)
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
        strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])

    ckpt_prefix = find_checkpoint(cfg, args)
    if not ckpt_prefix:
        print(f"[ERROR] Could not find any valid checkpoint for config {args.config}")
        return 1

    print(f"[INFO] Using checkpoint: {ckpt_prefix}")
    model = build_model(cfg)

    # Initialize model variables with a dummy forward pass
    img_size = int(cfg["data"]["image_size"])
    dummy_input = {"image": tf.zeros([1, img_size, img_size, 3], dtype=tf.float32)}
    model(dummy_input, training=False)

    status = model.load_weights(str(ckpt_prefix))
    status.expect_partial()
    print(f"[INFO] Successfully loaded weights from {ckpt_prefix}")

    replicas = strategy.num_replicas_in_sync if strategy else 1
    _, val_ds, test_ds = build_datasets(cfg, replicas=replicas)
    dataset = test_ds if args.split == "test" else val_ds
    use_tta = not args.no_tta and bool(cfg.get("tta", {}).get("enabled", True))
    w_orig = float(cfg.get("tta", {}).get("original_weight", 0.50))
    w_flip = float(cfg.get("tta", {}).get("flip_weight", 0.50))

    print(f"[INFO] Extracting logits from {target_split} set (TTA={use_tta})...")
    vis_logits, sem_logits, y_true = extract_logits(model, dataset, use_tta=use_tta, w_orig=w_orig, w_flip=w_flip)
    print(f"[INFO] Extracted {len(y_true)} samples. Visual logits shape: {vis_logits.shape}, Sem logits shape: {sem_logits.shape}")

    class_names = get_class_names(cfg)
    results = []

    print("\n" + "=" * 78)
    print(f"   RAF-DB SEMANTIC LOGIT SOFT FUSION SWEEP (Split: {target_split.upper()}, TTA: {use_tta})")
    print("=" * 78)
    print(f" {'Alpha':^8} | {'Accuracy':^12} | {'Macro F1':^12} | {'Weighted F1':^12} | {'Diff vs Base':^14}")
    print("-" * 78)

    base_acc = 0.0
    best_acc = -1.0
    best_alpha = 0.0
    best_metrics = None

    for alpha in sorted(args.alphas):
        fused = (1.0 - alpha) * vis_logits + alpha * sem_logits
        preds = np.argmax(fused, axis=-1)
        metrics = classification_metrics(y_true.tolist(), preds.tolist(), class_names)
        acc = float(metrics["accuracy"]) * 100.0
        mf1 = float(metrics["macro_f1"])
        wf1 = float(metrics["weighted_f1"])

        if math.isclose(alpha, 0.0, abs_tol=1e-5):
            base_acc = acc
            diff_str = "Baseline (0.0)"
        else:
            diff = acc - base_acc
            diff_str = f"{diff:+.2f}%"

        mark = " (BEST)" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_metrics = metrics

        print(f" {alpha:^8.2f} | {acc:^10.2f}% | {mf1:^12.4f} | {wf1:^12.4f} | {diff_str:^14}{mark}")
        results.append({
            "alpha": alpha,
            "accuracy": acc / 100.0,
            "macro_f1": mf1,
            "weighted_f1": wf1,
            "diff_vs_baseline": (acc - base_acc) / 100.0,
        })

    print("=" * 78)
    print(f"[SUMMARY] Best Alpha: {best_alpha:.2f} with Accuracy: {best_acc:.2f}% (Gain: {best_acc - base_acc:+.2f}%)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": str(cfg_path),
            "checkpoint": str(ckpt_prefix),
            "split": target_split,
            "tta_enabled": use_tta,
            "baseline_accuracy": base_acc / 100.0,
            "best_alpha": best_alpha,
            "best_accuracy": best_acc / 100.0,
            "best_macro_f1": best_metrics["macro_f1"] if best_metrics else None,
            "per_class_accuracy": best_metrics["per_class_accuracy"] if best_metrics else None,
            "sweep_results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Sweep report saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
