from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets, collect_split_records, _resolve_path
from train import build_model, configure_gpus, configure_tensorflow_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Select Train Source Samples for Targeted Diffusion")
    parser.add_argument(
        "--config",
        default="config_convnext_base_ms1m_arcface_baseline.yaml",
        help="Path to baseline config YAML file",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to baseline run output directory",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to specific baseline model checkpoint file",
    )
    parser.add_argument(
        "--output-json",
        default="data/synthetic_diffusion/pilot_sources.json",
        help="Output path for selected source metadata JSON",
    )
    return parser.parse_args()


def find_best_checkpoint(run_dir: Path, override_ckpt: str = None) -> Path:
    if override_ckpt:
        p = Path(override_ckpt)
        if p.exists() or Path(str(p) + ".index").exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {override_ckpt}")

    checkpoint_root = run_dir / "checkpoints"
    best_dir = checkpoint_root / "best"
    last_dir = checkpoint_root / "last"

    best_ckpt = tf.train.latest_checkpoint(str(best_dir))
    if best_ckpt:
        return Path(best_ckpt)

    last_ckpt = tf.train.latest_checkpoint(str(last_dir))
    if last_ckpt:
        return Path(last_ckpt)

    direct_ckpts = list(checkpoint_root.glob("*.ckpt.index"))
    if direct_ckpts:
        base_name = str(direct_ckpts[0]).replace(".index", "")
        return Path(base_name)

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_root}")


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)

    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg["paths"]["output_dir"])
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parents[1] / run_dir

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      TRAIN SOURCE SELECTION FOR TARGETED DIFFUSION (PILOT)")
    print("=" * 70)
    print(f"Run Directory: {run_dir}")
    print(f"Output JSON:   {output_path}")

    ckpt_path = find_best_checkpoint(run_dir, args.checkpoint)
    print(f"Restoring Checkpoint: {ckpt_path}")

    # Load Train split records
    print("\n[1/3] Loading Train Dataset (STRICT TRAIN SET ONLY)...")
    data_dir = _resolve_path(cfg["data"]["data_path"])
    train_records = collect_split_records(
        data_dir,
        split="train",
        mask_dir=_resolve_path(cfg["data"].get("mask_dir")),
        use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
        bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
        mask_ablation=cfg["data"].get("mask_ablation", "none"),
        allow_missing_masks=bool(cfg["data"].get("allow_missing_masks", False)),
    )

    train_ds, _, _ = build_datasets(cfg, replicas=1)

    # Build Model and Restore Checkpoint
    print("[2/3] Building Model & Running Inference on Train Set...")
    model = build_model(cfg)
    dummy_image = tf.zeros([1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]], tf.float32)
    dummy_mask = tf.zeros([1, cfg["model"].get("token_grid_size", 7), cfg["model"].get("token_grid_size", 7), cfg["model"].get("num_regions", 6)], tf.float32)
    model({"image": dummy_image, "mask": dummy_mask}, training=False)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(str(ckpt_path)).expect_partial()

    all_probs = []
    all_true_labels = []

    for batch in train_ds:
        inputs, labels = batch
        outputs = model(inputs, training=False)
        logits = outputs["logits"]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        all_probs.append(probs)
        all_true_labels.extend(labels.numpy().tolist())

    all_probs = np.vstack(all_probs)
    all_true_labels = np.array(all_true_labels, dtype=np.int64)

    # 3 Target Hard Pairs & 4 Target Classes
    # 1. fear <-> sad
    # 2. sad <-> neutral
    # 3. angry <-> sad
    target_classes = ["fear", "sad", "neutral", "angry"]
    target_indices = {name: EMOTION_NAMES.index(name) for name in target_classes}

    print("\n[3/3] Categorizing Train Source Samples (50% Medium, 30% Hard, 20% Clean)...")
    selected_sources = {}

    for c_name in target_classes:
        c_idx = target_indices[c_name]
        class_mask = (all_true_labels == c_idx)
        class_indices = np.where(class_mask)[0]
        class_probs = all_probs[class_indices, c_idx]

        # 50% Medium (0.45 <= p < 0.70)
        medium_mask = (class_probs >= 0.45) & (class_probs < 0.70)
        medium_sample_ids = train_records.sample_ids[class_indices[medium_mask]].tolist()

        # 30% Hard (p < 0.45)
        hard_mask = (class_probs < 0.45)
        hard_sample_ids = train_records.sample_ids[class_indices[hard_mask]].tolist()

        # 20% Clean (p >= 0.70)
        clean_mask = (class_probs >= 0.70)
        clean_sample_ids = train_records.sample_ids[class_indices[clean_mask]].tolist()

        selected_sources[c_name] = {
            "class_id": c_idx,
            "total_train_samples": int(np.sum(class_mask)),
            "medium_confidence_count": len(medium_sample_ids),
            "hard_misclassified_count": len(hard_sample_ids),
            "clean_high_confidence_count": len(clean_sample_ids),
            "medium_sample_ids": medium_sample_ids,
            "hard_sample_ids": hard_sample_ids,
            "clean_sample_ids": clean_sample_ids,
        }

        print(f"  Class [{c_name:<7}]: Total={np.sum(class_mask):<5} | Medium={len(medium_sample_ids):<4} | Hard={len(hard_sample_ids):<4} | Clean={len(clean_sample_ids):<4}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_sources, f, indent=2)

    print(f"\nSaved Train Source Selection JSON: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
