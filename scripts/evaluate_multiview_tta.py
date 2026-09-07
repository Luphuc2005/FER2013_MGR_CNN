#!/usr/bin/env python3
"""
Evaluate a trained checkpoint with Multi-View / Multi-Scale TTA on RAF-DB.
Supports:
1. No-TTA (Original image only)
2. 2-View Standard TTA (Original + Horizontal Flip)
3. 4-View Multi-Scale TTA (Original, Flip, Center-Crop 92% Zoom, Flipped Zoom)
4. 5-View Multi-Scale & Angle TTA (Original, Flip, Zoom 92%, Rotate -3 deg, Rotate +3 deg)
5. 6-View Comprehensive TTA (Original, Flip, Zoom, Flipped Zoom, Rotate -3 deg, Rotate +3 deg)

Can be executed on any existing checkpoint without retraining.
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
    parser = argparse.ArgumentParser(description="Multi-View / Multi-Scale TTA Evaluation on RAF-DB.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rafdb_siglip2_semantic_stable_v3.yaml",
        help="Path to YAML config file (e.g. config_rafdb_siglip2_semantic_stable_v3.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path prefix to checkpoint (e.g. outputs/papers/rafdb_siglip2_semantic_stable_v3/checkpoints/best/ckpt-35)",
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
    parser.add_argument("--crop-fraction", type=float, default=0.92, help="Central crop fraction for zoom (default: 0.92)")
    parser.add_argument("--rot-deg", type=float, default=3.0, help="Rotation angle in degrees (default: 3.0)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON report")
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
                return Path(str(indexes[-1])[:-6])
    return None


def restore_model_weights(model: tf.keras.Model, prefix: Union[str, Path]) -> None:
    prefix_str = str(prefix)
    errors = []

    # 1. Try tf.train.Checkpoint(model=model) - standard for train.py RankedCheckpointManager
    try:
        ckpt = tf.train.Checkpoint(model=model)
        status = ckpt.restore(prefix_str)
        status.expect_partial()
        status.assert_nontrivial_match()
        return
    except Exception as e:
        errors.append(f"tf.train.Checkpoint(model=model): {e}")

    # 2. Try direct model.load_weights
    try:
        status = model.load_weights(prefix_str)
        status.expect_partial()
        status.assert_nontrivial_match()
        return
    except Exception as e:
        errors.append(f"model.load_weights: {e}")

    # 3. Try tf.train.Checkpoint(root=model)
    try:
        ckpt = tf.train.Checkpoint(root=model)
        status = ckpt.restore(prefix_str)
        status.expect_partial()
        status.assert_nontrivial_match()
        return
    except Exception as e:
        errors.append(f"tf.train.Checkpoint(root=model): {e}")

    raise RuntimeError(
        f"Failed to restore weights from {prefix_str} using all strategies:\n" + "\n".join(errors)
    )



def rotate_batch(images: tf.Tensor, degrees: float) -> tf.Tensor:
    """Rotate a batch of images [B, H, W, C] by degrees with REFLECT padding."""
    radians = float(degrees) * np.pi / 180.0
    batch_size = tf.shape(images)[0]
    height = tf.cast(tf.shape(images)[1], tf.float32)
    width = tf.cast(tf.shape(images)[2], tf.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    cos_v = tf.cos(radians)
    sin_v = tf.sin(radians)
    transform = tf.stack([
        cos_v, sin_v, center_x - cos_v * center_x - sin_v * center_y,
        -sin_v, cos_v, center_y + sin_v * center_x - cos_v * center_y,
        0.0, 0.0,
    ])
    transforms = tf.tile(tf.expand_dims(transform, axis=0), [batch_size, 1])
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=tf.constant(0.0, dtype=tf.float32),
    )
    return rotated


def zoom_crop_batch(images: tf.Tensor, fraction: float = 0.92) -> tf.Tensor:
    """Central crop fraction of the batch and resize back to [112, 112]."""
    h, w = tf.shape(images)[1], tf.shape(images)[2]
    cropped = tf.image.central_crop(images, central_fraction=fraction)
    return tf.image.resize(cropped, [h, w], method="bilinear")


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def extract_view_logits(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    crop_fraction: float = 0.92,
    rot_deg: float = 3.0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Extract predictions for all 6 multi-scale/multi-angle views in a single pass.
    """
    view_names = ["orig", "hflip", "zoom", "hflip_zoom", "rot_neg", "rot_pos"]
    view_logits: Dict[str, List[np.ndarray]] = {k: [] for k in view_names}
    all_labels = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, batch_labels = batch[0], batch[1]
        else:
            inputs, batch_labels = batch, batch["label"]

        img_orig = inputs["image"]
        img_hflip = tf.image.flip_left_right(img_orig)
        img_zoom = zoom_crop_batch(img_orig, fraction=crop_fraction)
        img_hflip_zoom = tf.image.flip_left_right(img_zoom)
        img_rot_neg = rotate_batch(img_orig, degrees=-rot_deg)
        img_rot_pos = rotate_batch(img_orig, degrees=rot_deg)

        views_dict = {
            "orig": img_orig,
            "hflip": img_hflip,
            "zoom": img_zoom,
            "hflip_zoom": img_hflip_zoom,
            "rot_neg": img_rot_neg,
            "rot_pos": img_rot_pos,
        }

        for v_name, v_img in views_dict.items():
            v_input = dict(inputs)
            v_input["image"] = v_img
            if "mask" in v_input and ("hflip" in v_name):
                v_input["mask"] = tf.image.flip_left_right(v_input["mask"])
            out = model(v_input, training=False)
            logits = out["logits"].numpy()
            view_logits[v_name].append(logits)

        all_labels.append(batch_labels.numpy())

    concat_logits = {k: np.concatenate(v, axis=0) for k, v in view_logits.items()}
    y_true = np.concatenate(all_labels, axis=0)
    return concat_logits, y_true


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
        print(f"[ERROR] Could not find any checkpoint for config {args.config}")
        return 1

    print("\n" + "=" * 80)
    print("      MULTI-VIEW / MULTI-SCALE TTA EVALUATION ON RAF-DB")
    print("=" * 80)
    print(f" Config:         {cfg_path.name}")
    print(f" Checkpoint:     {ckpt_prefix}")
    print(f" Target Split:   {args.split.upper()}")
    print(f" Zoom Fraction:  {args.crop_fraction} (Central Crop + Resize)")
    print(f" Rotation Angle: +/- {args.rot_deg} deg")
    print("=" * 80 + "\n")

    model = build_model(cfg)
    img_size = int(cfg["data"]["image_size"])
    dummy_input = {"image": tf.zeros([1, img_size, img_size, 3], dtype=tf.float32)}
    model(dummy_input, training=False)

    restore_model_weights(model, ckpt_prefix)
    print(f"[INFO] Successfully loaded model weights from: {ckpt_prefix}")

    replicas = strategy.num_replicas_in_sync if strategy else 1
    _, val_ds, test_ds = build_datasets(cfg, replicas=replicas)
    dataset = test_ds if args.split == "test" else val_ds
    class_names = get_class_names(cfg)

    print(f"[INFO] Extracting predictions across all 6 augmented views...")
    view_logits, y_true = extract_view_logits(
        model, dataset, crop_fraction=args.crop_fraction, rot_deg=args.rot_deg
    )
    print(f"[INFO] Extraction finished on {len(y_true)} samples.\n")

    # Convert logits to probabilities
    probs_orig = softmax(view_logits["orig"])
    probs_hflip = softmax(view_logits["hflip"])
    probs_zoom = softmax(view_logits["zoom"])
    probs_hfzoom = softmax(view_logits["hflip_zoom"])
    probs_rot_neg = softmax(view_logits["rot_neg"])
    probs_rot_pos = softmax(view_logits["rot_pos"])

    # Strategy 1: No-TTA (Baseline original image)
    p_s1 = probs_orig
    m_s1 = classification_metrics(y_true.tolist(), np.argmax(p_s1, axis=-1).tolist(), class_names)

    # Strategy 2: 2-View Standard TTA (50% Orig + 50% Flip)
    p_s2 = 0.50 * probs_orig + 0.50 * probs_hflip
    m_s2 = classification_metrics(y_true.tolist(), np.argmax(p_s2, axis=-1).tolist(), class_names)

    # Strategy 3: 4-View Multi-Scale TTA (35% Orig + 35% Flip + 15% Zoom + 15% FlipZoom)
    p_s3 = 0.35 * probs_orig + 0.35 * probs_hflip + 0.15 * probs_zoom + 0.15 * probs_hfzoom
    m_s3 = classification_metrics(y_true.tolist(), np.argmax(p_s3, axis=-1).tolist(), class_names)

    # Strategy 4: 5-View Multi-Scale & Angle TTA (30% Orig + 30% Flip + 20% Zoom + 10% Rot- + 10% Rot+)
    p_s4 = 0.30 * probs_orig + 0.30 * probs_hflip + 0.20 * probs_zoom + 0.10 * probs_rot_neg + 0.10 * probs_rot_pos
    m_s4 = classification_metrics(y_true.tolist(), np.argmax(p_s4, axis=-1).tolist(), class_names)

    # Strategy 5: 6-View Comprehensive TTA (25% Orig + 25% Flip + 15% Zoom + 15% FlipZoom + 10% Rot- + 10% Rot+)
    p_s5 = (
        0.25 * probs_orig
        + 0.25 * probs_hflip
        + 0.15 * probs_zoom
        + 0.15 * probs_hfzoom
        + 0.10 * probs_rot_neg
        + 0.10 * probs_rot_pos
    )
    m_s5 = classification_metrics(y_true.tolist(), np.argmax(p_s5, axis=-1).tolist(), class_names)

    strategies = [
        ("1. No-TTA (Ảnh gốc thuần)", m_s1),
        ("2. 2-View TTA (Gốc + Lật ngang)", m_s2),
        ("3. 4-View Multi-Scale (Gốc + Lật + Zoom 92%)", m_s3),
        ("4. 5-View Multi-Scale & Góc (Gốc + Lật + Zoom + Xoay +/-3 deg)", m_s4),
        ("5. 6-View Toàn diện (Gốc + Lật + Zoom + LậtZoom + Xoay +/-3 deg)", m_s5),
    ]

    base_acc = float(m_s1["accuracy"]) * 100.0
    best_strategy = max(strategies, key=lambda x: float(x[1]["accuracy"]))

    print("=" * 86)
    print(f" {'Chiến lược TTA':<38} | {'Accuracy':^10} | {'Macro F1':^10} | {'Weighted F1':^12} | {'Tăng trưởng':^10}")
    print("-" * 86)
    for name, m in strategies:
        acc = float(m["accuracy"]) * 100.0
        mf1 = float(m["macro_f1"])
        wf1 = float(m["weighted_f1"])
        diff = acc - base_acc
        diff_str = "Gốc" if diff == 0.0 else f"{diff:+.2f}%"
        is_best = " ⭐ BEST" if name == best_strategy[0] else ""
        print(f" {name:<38} | {acc:^8.2f}% | {mf1:^10.4f} | {wf1:^12.4f} | {diff_str:^10}{is_best}")
    print("=" * 86)

    # Per-class Recall Comparison
    print(f"\n[BẢNG SO SÁNH RECALL TỪNG LỚP: GỐC vs TỐT NHẤT ({best_strategy[0]})]")
    print(f" {'Cảm xúc (Class)':<16} | {'Recall Gốc':^14} | {'Recall Best TTA':^18} | {'Chênh lệch':^12}")
    print("-" * 68)
    for c_idx, c_name in enumerate(class_names):
        r_base = float(m_s1["per_class_accuracy"][c_idx]) * 100.0
        r_best = float(best_strategy[1]["per_class_accuracy"][c_idx]) * 100.0
        d = r_best - r_base
        d_str = f"{d:+.2f}%" if d != 0.0 else "="
        print(f" {c_name:<16} | {r_base:^12.2f}% | {r_best:^16.2f}% | {d_str:^12}")
    print("-" * 68)

    print(f"\n[KẾT LUẬN]: Chiến lược '{best_strategy[0]}' giúp tăng {float(best_strategy[1]['accuracy'])*100.0 - base_acc:+.2f}% Accuracy!")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "checkpoint": str(ckpt_prefix),
            "split": args.split,
            "baseline_accuracy": base_acc / 100.0,
            "best_strategy": best_strategy[0],
            "best_accuracy": float(best_strategy[1]["accuracy"]),
            "best_macro_f1": float(best_strategy[1]["macro_f1"]),
            "per_class_recall_baseline": [float(v) for v in m_s1["per_class_accuracy"]],
            "per_class_recall_best_tta": [float(v) for v in best_strategy[1]["per_class_accuracy"]],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Đã lưu báo cáo chi tiết vào: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
