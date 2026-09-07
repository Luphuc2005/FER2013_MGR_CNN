from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import tensorflow as tf

from config import load_config
from datasets import FERPLUS_EMOTION_NAMES, build_datasets
from losses.classification import supervised_mgr_loss
from train import build_model, configure_gpus, configure_tensorflow_runtime, get_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test FERPlus official 8-class majority pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml",
        help="Path to FERPlus config file",
    )
    parser.add_argument("--num-batches", type=int, default=1, help="Number of train batches to inspect")
    parser.add_argument("--check-files", type=int, default=128, help="Number of manifest image paths to sample-check")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT_DIR / p


def check_manifest(csv_path: Path, split: str, class_names, expected_samples, image_root, check_files: int) -> int:
    print(f"\n--- Manifest: {split.upper()} ---", flush=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"{split} manifest not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["path", "label", "class_name", "image_name"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{split} manifest missing columns: {missing}")

    total = len(df)
    print(f"Samples: {total}", flush=True)
    if expected_samples is not None and total != int(expected_samples):
        raise ValueError(f"{split} expected {expected_samples} samples, got {total}.")

    labels = df["label"].to_numpy(dtype=np.int64)
    print(f"Label range: min={labels.min()}, max={labels.max()}", flush=True)
    if labels.min() < 0 or labels.max() >= len(class_names):
        raise ValueError(f"{split} labels out of range [0..{len(class_names)-1}].")

    counts = np.bincount(labels, minlength=len(class_names))[: len(class_names)]
    for idx, name in enumerate(class_names):
        observed_names = sorted(set(df.loc[df["label"] == idx, "class_name"].astype(str).str.strip()))
        if observed_names and any(n.lower() != name.lower() for n in observed_names):
            raise ValueError(
                f"{split} label {idx} expected class_name={name!r}, observed={observed_names[:5]!r}."
            )
        pct = 100.0 * counts[idx] / max(total, 1)
        print(f"  {idx}: {name:<8} {counts[idx]:>6} ({pct:5.2f}%)", flush=True)

    paths = df["path"].astype(str).to_numpy()
    rng = np.random.default_rng(42)
    sample_n = min(int(check_files), len(paths))
    missing_files = 0
    image_root_path = Path(image_root) if image_root else None
    for idx in rng.choice(len(paths), size=sample_n, replace=False):
        raw = paths[idx].replace("\\", "/").strip()
        p = Path(raw)
        candidates = [p] if p.is_absolute() else [ROOT_DIR / raw]
        if image_root_path and not p.is_absolute():
            candidates.insert(0, image_root_path / raw)
        if not any(candidate.exists() for candidate in candidates):
            missing_files += 1
    print(f"Sampled files checked: {sample_n}, missing: {missing_files}", flush=True)
    if missing_files:
        raise FileNotFoundError(f"{split} has {missing_files}/{sample_n} missing sampled image files.")
    return total


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)

    print("==================================================================", flush=True)
    print("      FERPlus Official 8-Class Majority Smoke Test", flush=True)
    print("==================================================================", flush=True)
    print(f"Config File: {config_path}", flush=True)

    cfg = load_config(str(config_path))
    class_names = get_class_names(cfg)
    if class_names != FERPLUS_EMOTION_NAMES:
        raise ValueError(f"FERPlus class names mismatch: {class_names} != {FERPLUS_EMOTION_NAMES}")
    if int(cfg["data"]["num_classes"]) != 8:
        raise ValueError(f"FERPlus config must set data.num_classes=8, got {cfg['data']['num_classes']}.")
    print(f"Class names: {class_names}", flush=True)

    data_cfg = cfg["data"]
    expected = data_cfg.get("expected_samples", {})
    image_root = data_cfg.get("image_root")
    split_counts = {
        "train": check_manifest(resolve_path(data_cfg["train_csv"]), "train", class_names, expected.get("train"), image_root, args.check_files),
        "val": check_manifest(resolve_path(data_cfg["val_csv"]), "val", class_names, expected.get("val"), image_root, args.check_files),
        "test": check_manifest(resolve_path(data_cfg["test_csv"]), "test", class_names, expected.get("test"), image_root, args.check_files),
    }
    print(f"\nSample summary: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}", flush=True)

    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)

    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
    print("Dataset objects built successfully.", flush=True)
    print(f"Cardinality: train={tf.data.experimental.cardinality(train_ds).numpy()} val={tf.data.experimental.cardinality(val_ds).numpy()} test={tf.data.experimental.cardinality(test_ds).numpy()}", flush=True)

    for batch_idx, (inputs, labels) in enumerate(train_ds.take(args.num_batches), start=1):
        images = inputs["image"]
        label_min = int(tf.reduce_min(labels).numpy())
        label_max = int(tf.reduce_max(labels).numpy())
        print(f"\nBatch {batch_idx}", flush=True)
        print(f"Input shape: {images.shape} dtype={images.dtype}", flush=True)
        print(f"Label shape: {labels.shape} dtype={labels.dtype} range=[{label_min}..{label_max}]", flush=True)
        if label_min < 0 or label_max > 7:
            raise ValueError(f"Batch labels out of FERPlus range [0..7]: [{label_min}..{label_max}].")

    first_inputs, first_labels = next(iter(train_ds.take(1)))
    model = build_model(cfg)
    outputs = model(first_inputs, training=False)
    logits = outputs["logits"]
    print("\nModel forward", flush=True)
    print(f"Logits shape: {logits.shape}", flush=True)
    if logits.shape[-1] != 8:
        raise ValueError(f"Expected logits last dim=8, got {logits.shape}.")

    if outputs.get("semantic_logits") is not None:
        semantic_logits = outputs["semantic_logits"]
        print(f"Semantic logits shape: {semantic_logits.shape}", flush=True)
        if semantic_logits.shape[-1] != 8:
            raise ValueError(f"Expected semantic logits last dim=8, got {semantic_logits.shape}.")
    if outputs.get("agg_sim") is not None:
        print(f"Aggregated semantic similarity shape: {outputs['agg_sim'].shape}", flush=True)
    if outputs.get("granularity_weights") is not None:
        print(f"Granularity weights shape: {outputs['granularity_weights'].shape}", flush=True)
    hard_pairs = outputs.get("hard_pairs_matrix")
    if hard_pairs is not None:
        print(f"Hard-pair matrix shape: {hard_pairs.shape}", flush=True)
        if tuple(hard_pairs.shape) != (8, 8):
            raise ValueError(f"Expected hard-pair matrix shape (8, 8), got {hard_pairs.shape}.")

    loss, parts = supervised_mgr_loss(
        first_labels,
        outputs,
        num_classes=8,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        ortho_weight=float(cfg["model"].get("ortho_loss_weight", 0.0)),
        cnn_aux_weight=float(cfg["model"].get("cnn_aux_loss_weight", 0.0)),
    )
    loss_value = float(loss.numpy())
    if not np.isfinite(loss_value):
        raise FloatingPointError("FERPlus smoke loss is not finite.")
    print(
        f"Loss: total={loss_value:.4f} ce={float(parts['ce'].numpy()):.4f} "
        f"semantic={float(parts['semantic'].numpy()):.4f} hard={float(parts['hard_semantic'].numpy()):.4f}",
        flush=True,
    )
    print("\nFERPLUS_SMOKE_1_BATCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
