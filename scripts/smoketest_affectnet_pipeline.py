from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pandas as pd
import tensorflow as tf

from config import load_config
from datasets import EMOTION_NAMES, build_datasets, collect_affectnet_split_records
from losses.classification import supervised_mgr_loss
from train import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test AffectNet-7 training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config_affectnet_convnext_base_ms1m_adaptive_siglip2_confusion_v2.yaml",
        help="Path to AffectNet config file",
    )
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to smoke test")
    parser.add_argument("--check-files", type=int, default=2000, help="Number of sample files to verify existence")
    return parser.parse_args()


def create_mock_affectnet_data(data_dir: Path):
    """Creates temporary mock CSV and dummy images for local smoke testing if real data is missing."""
    print(f"[SMOKE TEST] Real AffectNet-7 data not found at {data_dir}. Creating temporary mock dataset...")
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "Manually_Annotated_Images" / "100"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy images
    dummy_img_paths = []
    from PIL import Image

    for i in range(20):
        img_p = img_dir / f"mock_{i:04d}.jpg"
        if not img_p.exists():
            img_arr = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_p)
        rel_p = f"100/mock_{i:04d}.jpg"
        dummy_img_paths.append(rel_p)

    # Create mock train.csv (e.g. 140 samples across 7 classes)
    train_records = []
    for idx, p in enumerate(dummy_img_paths * 7):
        train_records.append({"subDirectory_filePath": p, "expression": idx % 7})
    df_train = pd.DataFrame(train_records)
    train_csv = data_dir / "affectnet7_train.csv"
    df_train.to_csv(train_csv, index=False)

    # Create mock val.csv (e.g. 35 samples)
    val_records = []
    for idx, p in enumerate(dummy_img_paths * 2):
        val_records.append({"subDirectory_filePath": p, "expression": idx % 7})
    df_val = pd.DataFrame(val_records)
    val_csv = data_dir / "affectnet7_val.csv"
    df_val.to_csv(val_csv, index=False)

    print(f"[SMOKE TEST] Created mock dataset at {data_dir}")
    return train_csv, val_csv


def check_csv_statistics(csv_path: Path, split_name: str, check_files_count: int, image_root: Optional[str] = None):
    print(f"\n--- [1/4] Inspecting {split_name.upper()} CSV: {csv_path} ---")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Total {split_name} samples: {total_rows:,}")

    # Column identification
    cols_lower = [str(c).strip().lower() for c in df.columns]
    lbl_col = next((df.columns[i] for i, c in enumerate(cols_lower) if c in ("emotion", "label", "target", "class", "y", "expression", "expr")), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    path_col = next((df.columns[i] for i, c in enumerate(cols_lower) if c in ("subdirectory_filepath", "image_path", "filepath", "path", "image", "file", "filename")), df.columns[0])

    print(f"Identified columns -> Image Path: {path_col!r}, Expression Label: {lbl_col!r}")

    # Label distribution
    labels = df[lbl_col].to_numpy()
    unique, counts = np.unique(labels, return_counts=True)
    dist_dict = dict(zip(unique, counts))

    print(f"Label range: min={labels.min()}, max={labels.max()}")
    print(f"Label Distribution ({split_name}):")
    for idx, name in enumerate(EMOTION_NAMES):
        # Support both 0-based and 1-based checks
        c_count = dist_dict.get(idx, dist_dict.get(idx + 1, dist_dict.get(str(idx), 0)))
        pct = (c_count / total_rows) * 100.0 if total_rows > 0 else 0
        print(f"  Class {idx} ({name:<8}): {c_count:>7,} samples ({pct:5.2f}%)")

    # Missing file check
    paths = df[path_col].astype(str).to_numpy()
    sample_indices = np.random.choice(len(paths), size=min(check_files_count, len(paths)), replace=False)
    missing_count = 0

    img_root_path = Path(image_root) if image_root else None
    for idx in sample_indices:
        p_str = paths[idx].replace("\\", "/").strip()
        p_obj = Path(p_str)
        if p_obj.is_absolute() and p_obj.exists():
            continue
        if (ROOT_DIR / p_str).exists():
            continue
        if img_root_path and (img_root_path / p_str).exists():
            continue
        if img_root_path and (ROOT_DIR / img_root_path / p_str).exists():
            continue
        missing_count += 1

    checked_n = len(sample_indices)
    missing_pct = (missing_count / checked_n) * 100.0
    print(f"Sampled File Verification: Checked {checked_n:,} files -> Missing: {missing_count:,} ({missing_pct:.2f}%)")
    if missing_count > 0:
        print(f"[WARNING] Some image files were not found on current filesystem. (Note: On cluster, verify image_root path: {image_root})")

    return total_rows, label_col, path_col


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    print("==================================================================")
    print("        AffectNet-7 Pipeline Smoke Test & Validation Script       ")
    print("==================================================================")
    print(f"Config File: {config_path}")

    cfg = load_config(str(config_path))
    data_cfg = cfg.get("data", {})

    train_csv_path = Path(data_cfg.get("train_csv", "data/affectnet/affectnet7_train.csv"))
    if not train_csv_path.is_absolute():
        train_csv_path = ROOT_DIR / train_csv_path

    val_csv_path = Path(data_cfg.get("val_csv", "data/affectnet/affectnet7_val.csv"))
    if not val_csv_path.is_absolute():
        val_csv_path = ROOT_DIR / val_csv_path

    image_root = data_cfg.get("image_root", "data/affectnet/Manually_Annotated_Images")

    # If local env without real AffectNet data, set up mock dataset to ensure full pipeline validation
    if not train_csv_path.exists():
        mock_dir = ROOT_DIR / "data" / "affectnet"
        t_csv, v_csv = create_mock_affectnet_data(mock_dir)
        cfg["data"]["train_csv"] = str(t_csv)
        cfg["data"]["val_csv"] = str(v_csv)
        cfg["data"]["image_root"] = str(mock_dir / "Manually_Annotated_Images")
        train_csv_path = t_csv
        val_csv_path = v_csv
        image_root = str(mock_dir / "Manually_Annotated_Images")

    # 1. Check CSV Stats & File Existence
    n_train, _, _ = check_csv_statistics(train_csv_path, "train", args.check_files, image_root)
    n_val, _, _ = check_csv_statistics(val_csv_path, "val", args.check_files, image_root)

    # 2. Test Dataset Loader
    print("\n--- [2/4] Testing AffectNet-7 tf.data Dataset Loader ---")
    start_t = time.time()
    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
    load_t = time.time() - start_t
    print(f"Datasets built successfully in {load_t:.2f}s!")

    # 3. Test Batch Extraction & Tensor Shapes
    print(f"\n--- [3/4] Testing Batch Extraction ({args.num_batches} batches) ---")
    for batch_idx, batch in enumerate(train_ds.take(args.num_batches), start=1):
        inputs, labels = batch
        img_tensor = inputs["image"]
        lbl_tensor = labels

        print(f"Batch {batch_idx}:")
        print(f"  Image Tensor Shape: {img_tensor.shape} | Dtype: {img_tensor.dtype}")
        print(f"  Label Tensor Shape: {lbl_tensor.shape} | Dtype: {lbl_tensor.dtype}")
        print(f"  Label values sample: {lbl_tensor.numpy()[:10]}")
        print(f"  Image Pixel Range: min={tf.reduce_min(img_tensor).numpy():.4f}, max={tf.reduce_max(img_tensor).numpy():.4f}, mean={tf.reduce_mean(img_tensor).numpy():.4f}")

        # Assertions
        expected_shape = (cfg["runtime"]["batch_size_per_gpu"], cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"])
        assert img_tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {img_tensor.shape}"
        assert lbl_tensor.shape[0] == cfg["runtime"]["batch_size_per_gpu"], f"Expected batch size {cfg['runtime']['batch_size_per_gpu']}"
        assert tf.reduce_min(lbl_tensor).numpy() >= 0 and tf.reduce_max(lbl_tensor).numpy() <= 6, "Labels out of range [0..6]!"

    print("Batch shape & normalization checks PASSED!")

    # 4. Test Model Forward Pass & Loss Computation
    print("\n--- [4/4] Testing Model Architecture & Forward Pass ---")
    first_batch = next(iter(train_ds.take(1)))
    inputs, labels = first_batch

    model = build_model(cfg)
    outputs = model(inputs, training=False)
    logits = outputs["logits"]

    print(f"Model Architecture: {cfg['model']['name']}")
    print(f"Output Logits Shape: {logits.shape} (Expected: ({cfg['runtime']['batch_size_per_gpu']}, 7))")

    assert logits.shape == (cfg["runtime"]["batch_size_per_gpu"], 7), f"Expected logits shape ({cfg['runtime']['batch_size_per_gpu']}, 7), got {logits.shape}"

    if outputs.get("semantic_logits") is not None:
        sem_logits = outputs["semantic_logits"]
        print(f"Semantic Logits Shape: {sem_logits.shape}")

    if outputs.get("granularity_weights") is not None:
        gw = outputs["granularity_weights"]
        print(f"Granularity Weights Shape: {gw.shape} | Sample: {gw.numpy()[0]}")

    # Calculate loss
    loss, loss_parts = supervised_mgr_loss(
        labels,
        outputs,
        num_classes=7,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.1)),
        ortho_weight=float(cfg["model"].get("ortho_loss_weight", 0.0)),
        cnn_aux_weight=float(cfg["model"].get("cnn_aux_loss_weight", 0.0)),
    )
    print(f"Total Loss: {loss.numpy():.4f} | CE Loss: {loss_parts['ce'].numpy():.4f} | Semantic Loss: {loss_parts['semantic'].numpy():.4f}")

    assert np.isfinite(loss.numpy()), "Calculated loss is not finite!"

    print("\n==================================================================")
    print("         SMOKE TEST COMPLETED SUCCESSFULLY! ALL CHECKS PASSED.   ")
    print("==================================================================")


if __name__ == "__main__":
    main()
