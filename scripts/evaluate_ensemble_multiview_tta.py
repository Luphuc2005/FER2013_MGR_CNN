#!/usr/bin/env python3
"""
Evaluate an Ensemble of Top Checkpoints with Multi-View / Multi-Scale TTA on RAF-DB.
Combines:
- Top-M Checkpoints (e.g. 5 best checkpoints from training)
- 6 Augmented Views:
    1. Original
    2. Horizontal Flip
    3. Center-Crop 92% Zoom
    4. Flipped Zoom
    5. Rotation -3 deg
    6. Rotation +3 deg
Evaluates all individual checkpoints and the grand Ensemble across 5 TTA strategies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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
    parser = argparse.ArgumentParser(description="Top-5 Ensemble + Multi-View TTA Evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rafdb_siglip2_semantic_stable_v3.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint-dirs",
        nargs="+",
        dest="checkpoint_dirs",
        default=None,
        help="One or more directories containing checkpoints (e.g. checkpoints/best checkpoints/best_loss)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Explicit list of checkpoints or prefixes",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split (default: test)",
    )
    parser.add_argument("--crop-fraction", type=float, default=0.92, help="Central crop fraction for zoom (default: 0.92)")
    parser.add_argument("--rot-deg", type=float, default=3.0, help="Rotation angle in degrees (default: 3.0)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report path")
    return parser.parse_args()


def _extract_epoch_num(path: Path) -> int:
    name = path.stem if path.name.endswith(".index") else path.name
    digits = re.findall(r"\d+", name)
    return int(digits[-1]) if digits else 0


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


def get_checkpoint_list(ckpt_dir: Path, explicit_ckpts: Optional[List[str]] = None) -> List[Path]:
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
    index_files = sorted(index_files, key=_extract_epoch_num)
    return [Path(str(p)[:-6]) for p in index_files]


def rotate_batch(images: tf.Tensor, degrees: float) -> tf.Tensor:
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
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=tf.constant(0.0, dtype=tf.float32),
    )


def zoom_crop_batch(images: tf.Tensor, fraction: float = 0.92) -> tf.Tensor:
    h, w = 112, 112
    cropped = tf.image.central_crop(images, central_fraction=fraction)
    return tf.image.resize(cropped, [h, w], method="bilinear")


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def extract_view_probs_single_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    crop_fraction: float = 0.92,
    rot_deg: float = 3.0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
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

    concat_probs = {k: softmax(np.concatenate(v, axis=0)) for k, v in view_logits.items()}
    y_true = np.concatenate(all_labels, axis=0)
    return concat_probs, y_true


def compute_tta_strategies(probs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    p_orig = probs_dict["orig"]
    p_hflip = probs_dict["hflip"]
    p_zoom = probs_dict["zoom"]
    p_hfzoom = probs_dict["hflip_zoom"]
    p_rneg = probs_dict["rot_neg"]
    p_rpos = probs_dict["rot_pos"]

    # 1. No-TTA
    s1 = p_orig

    # 2. 2-View TTA (50% Orig + 50% Flip)
    s2 = 0.50 * p_orig + 0.50 * p_hflip

    # 3. 4-View Multi-Scale TTA
    s3 = 0.35 * p_orig + 0.35 * p_hflip + 0.15 * p_zoom + 0.15 * p_hfzoom

    # 4. 5-View Multi-Scale & Angle TTA
    s4 = 0.30 * p_orig + 0.30 * p_hflip + 0.20 * p_zoom + 0.10 * p_rneg + 0.10 * p_rpos

    # 5. 6-View Comprehensive TTA
    s5 = 0.25 * p_orig + 0.25 * p_hflip + 0.15 * p_zoom + 0.15 * p_hfzoom + 0.10 * p_rneg + 0.10 * p_rpos

    return {
        "No-TTA": s1,
        "2-View TTA": s2,
        "4-View Multi-Scale": s3,
        "5-View Multi-Scale & Góc": s4,
        "6-View Toàn diện": s5,
    }


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

    out_dir = Path(cfg["paths"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    target_dirs = []
    if args.checkpoint_dirs:
        for d in args.checkpoint_dirs:
            p = Path(d)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                target_dirs.append(p)
            else:
                print(f"[WARNING] Checkpoint directory not found: {p}")
    else:
        best_dir = out_dir / "checkpoints" / "best"
        loss_dir = out_dir / "checkpoints" / "best_loss"
        if best_dir.exists():
            target_dirs.append(best_dir)
        if loss_dir.exists():
            target_dirs.append(loss_dir)
        if not target_dirs:
            fallback = out_dir / "checkpoints"
            if fallback.exists():
                target_dirs.append(fallback)

    if not target_dirs and not args.checkpoints:
        print(f"[ERROR] No valid checkpoint directories found for {cfg_path.name}")
        return 1

    replicas = strategy.num_replicas_in_sync if strategy else 1
    _, val_ds, test_ds = build_datasets(cfg, replicas=replicas)
    dataset = test_ds if args.split == "test" else val_ds
    class_names = get_class_names(cfg)

    first_batch = next(iter(dataset.take(1)))
    first_inputs = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch

    with strategy.scope():
        model = build_model(cfg)
        _ = model(first_inputs, training=False)

    strategy_names = [
        "No-TTA",
        "2-View TTA",
        "4-View Multi-Scale",
        "5-View Multi-Scale & Góc",
        "6-View Toàn diện",
    ]

    probs_cache: Dict[str, Dict[str, np.ndarray]] = {}
    y_true_holder: List[np.ndarray] = []
    all_group_results: List[Dict[str, Any]] = []
    all_collected_prefixes: List[Path] = []

    if args.checkpoints:
        ckpts = get_checkpoint_list(Path("."), args.checkpoints)
        res = evaluate_checkpoint_group(
            "Explicit Checkpoint List", ckpts, model, dataset, args, class_names, strategy_names, probs_cache, y_true_holder
        )
        all_group_results.append(res)
    else:
        for cdir in target_dirs:
            ckpts = get_checkpoint_list(cdir)
            if not ckpts:
                print(f"[WARNING] No checkpoint index files in: {cdir}")
                continue
            for p in ckpts:
                if p not in all_collected_prefixes:
                    all_collected_prefixes.append(p)
            g_title = f"Checkpoints Group: {cdir.name} ({cdir.parent.name}/{cdir.name})"
            res = evaluate_checkpoint_group(
                g_title, ckpts, model, dataset, args, class_names, strategy_names, probs_cache, y_true_holder
            )
            all_group_results.append(res)

        # If multiple directories were evaluated (e.g. best and best_loss), evaluate the GRAND JOINT ENSEMBLE!
        if len(target_dirs) > 1 and len(all_collected_prefixes) > len(all_group_results[0]["prefixes"]):
            grand_title = "⭐ GRAND JOINT ENSEMBLE (BEST VAL ACC + BEST LOSS COMBINED) ⭐"
            grand_res = evaluate_checkpoint_group(
                grand_title, all_collected_prefixes, model, dataset, args, class_names, strategy_names, probs_cache, y_true_holder
            )
            all_group_results.append(grand_res)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": str(cfg_path),
            "target_dirs": [str(d) for d in target_dirs],
            "split": args.split,
            "groups": all_group_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Báo cáo chi tiết đã lưu vào: {out_path}")

    return 0


def evaluate_checkpoint_group(
    group_title: str,
    prefixes: List[Path],
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    args: argparse.Namespace,
    class_names: List[str],
    strategy_names: List[str],
    probs_cache: Dict[str, Dict[str, np.ndarray]],
    y_true_holder: List[np.ndarray],
) -> Dict[str, Any]:
    print("\n" + "=" * 105)
    print(f"      {group_title.upper()}")
    print("=" * 105)
    print(f" Số lượng checkpoints: {len(prefixes)} -> {[p.name for p in prefixes]}")
    print("=" * 105 + "\n")

    group_probs = []
    for idx, prefix in enumerate(prefixes, 1):
        k = str(prefix)
        if k not in probs_cache:
            print(f"[{group_title} | Model {idx}/{len(prefixes)}] Loading weights: {prefix.name} ...")
            restore_model_weights(model, prefix)
            p_dict, labels = extract_view_probs_single_model(
                model, dataset, crop_fraction=args.crop_fraction, rot_deg=args.rot_deg
            )
            probs_cache[k] = p_dict
            if not y_true_holder:
                y_true_holder.append(labels)
        group_probs.append(probs_cache[k])

    y_true = y_true_holder[0]

    # Compute Ensemble View Probabilities for this group
    ensemble_view_probs: Dict[str, np.ndarray] = {}
    for v_name in ["orig", "hflip", "zoom", "hflip_zoom", "rot_neg", "rot_pos"]:
        stacked = np.stack([m[v_name] for m in group_probs], axis=0)
        ensemble_view_probs[v_name] = np.mean(stacked, axis=0)

    print("\n" + "=" * 105)
    print(f" {'Mô hình / Checkpoint':<25} | {'No-TTA':^13} | {'2-View TTA':^13} | {'4-View MS':^13} | {'5-View Góc':^13} | {'6-View All':^13}")
    print("-" * 105)

    all_table_data = []
    for prefix, m_probs in zip(prefixes, group_probs):
        strats = compute_tta_strategies(m_probs)
        row_accs = {}
        for s_name in strategy_names:
            preds = np.argmax(strats[s_name], axis=-1)
            acc = float(classification_metrics(y_true.tolist(), preds.tolist(), class_names)["accuracy"]) * 100.0
            row_accs[s_name] = acc
        all_table_data.append((prefix.name, row_accs))
        print(f" {prefix.name:<25} | {row_accs['No-TTA']:^11.2f}% | {row_accs['2-View TTA']:^11.2f}% | {row_accs['4-View Multi-Scale']:^11.2f}% | {row_accs['5-View Multi-Scale & Góc']:^11.2f}% | {row_accs['6-View Toàn diện']:^11.2f}%")

    ens_strats = compute_tta_strategies(ensemble_view_probs)
    ens_accs = {}
    ens_metrics = {}
    for s_name in strategy_names:
        preds = np.argmax(ens_strats[s_name], axis=-1)
        m = classification_metrics(y_true.tolist(), preds.tolist(), class_names)
        ens_accs[s_name] = float(m["accuracy"]) * 100.0
        ens_metrics[s_name] = m

    print("-" * 105)
    print(f" ⭐ {'ENSEMBLE TOP-' + str(len(prefixes)):<22} | {ens_accs['No-TTA']:^11.2f}% | {ens_accs['2-View TTA']:^11.2f}% | {ens_accs['4-View Multi-Scale']:^11.2f}% | {ens_accs['5-View Multi-Scale & Góc']:^11.2f}% | {ens_accs['6-View Toàn diện']:^11.2f}%")
    print("=" * 105)

    best_strat_name = max(ens_accs, key=ens_accs.get)
    best_acc = ens_accs[best_strat_name]
    baseline_acc = all_table_data[0][1]["No-TTA"] if all_table_data else ens_accs["No-TTA"]
    best_m = ens_metrics[best_strat_name]

    print(f"\n[{group_title} - KẾT QUẢ VƯỢT TRỘI NHẤT]:")
    print(f"  -> Cấu hình:  ENSEMBLE TOP-{len(prefixes)} + {best_strat_name}")
    print(f"  -> Accuracy:  {best_acc:.2f}% (Tăng {best_acc - baseline_acc:+.2f}% so với ảnh gốc đơn lẻ)")
    print(f"  -> Macro F1:  {float(best_m['macro_f1']):.4f}")
    print(f"  -> Weighted:  {float(best_m['weighted_f1']):.4f}")

    print(f"\n[BẢNG RECALL CHI TIẾT TỪNG LỚP: Đơn lẻ No-TTA vs ENSEMBLE + {best_strat_name}]")
    print(f" {'Cảm xúc':<14} | {'Recall Đơn lẻ':^16} | {'Recall Ensemble Best':^24} | {'Tăng trưởng':^14}")
    print("-" * 72)
    single_no_tta_m = classification_metrics(y_true.tolist(), np.argmax(group_probs[0]["orig"], axis=-1).tolist(), class_names)
    for c_idx, c_name in enumerate(class_names):
        r_base = float(single_no_tta_m["per_class_accuracy"][c_idx]) * 100.0
        r_best = float(best_m["per_class_accuracy"][c_idx]) * 100.0
        diff = r_best - r_base
        diff_str = f"{diff:+.2f}%" if diff != 0 else "="
        print(f" {c_name:<14} | {r_base:^14.2f}% | {r_best:^22.2f}% | {diff_str:^14}")
    print("-" * 72 + "\n")

    return {
        "group_title": group_title,
        "prefixes": [p.name for p in prefixes],
        "individual_models": [{"name": name, "accuracies": accs} for name, accs in all_table_data],
        "ensemble_accuracies": ens_accs,
        "best_combination": f"Ensemble Top-{len(prefixes)} + {best_strat_name}",
        "best_accuracy": best_acc / 100.0,
        "best_macro_f1": float(best_m["macro_f1"]),
        "best_per_class_recall": [float(v) for v in best_m["per_class_accuracy"]],
    }


if __name__ == "__main__":
    sys.exit(main())
