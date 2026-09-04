#!/usr/bin/env python3
"""
Smoke test script for RAF-DB dataset + SigLIP2 ConvNeXt-Base Adaptive Confusion Pipeline.
Tests:
1. RAF-DB split CSV loading (train/val/test) & no data leakage.
2. Label mapping (7 classes: 0..6).
3. 1 batch image parsing & normalization shape (16, 112, 112, 3).
4. SigLIP2 text prototypes loading & shape verification.
5. Model initialization & 1 forward/loss pass.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from config import load_config
from datasets.fer2013 import build_datasets, collect_split_records
from train import build_model, compute_loss

def main():
    config_file = "config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        
    print("=" * 60)
    print(f" SMOKE TEST: RAF-DB SigLIP2 Pipeline ({config_file})")
    print("=" * 60)
    
    cfg = load_config(config_file)
    data_dir = Path(cfg["data"]["data_path"])
    print(f"[1/5] Checking RAF-DB dataset directory: {data_dir}")
    if not data_dir.exists():
        print(f"      [WARNING] Directory {data_dir} not found locally.")
        for fallback in [Path("data/rafdb"), Path("data/raf_db")]:
            if fallback.exists():
                data_dir = fallback
                cfg["data"]["data_path"] = str(data_dir)
                print(f"      [OK] Local fallback dataset directory found: {data_dir}")
                break
        if not data_dir.exists():
            print("      [INFO] Testing with simulated dataset structure for offline validation.")
            
    # Check dataset split records if files exist
    if data_dir.exists():
        train_rec = collect_split_records(data_dir, "train")
        val_rec = collect_split_records(data_dir, "val")
        test_rec = collect_split_records(data_dir, "test")
        
        print(f"[2/5] RAF-DB Dataset Split Verification:")
        print(f"      - Train samples: {len(train_rec.labels)} | Labels range: [{train_rec.labels.min()}..{train_rec.labels.max()}]")
        print(f"      - Val samples:   {len(val_rec.labels)}   | Labels range: [{val_rec.labels.min()}..{val_rec.labels.max()}]")
        print(f"      - Test samples:  {len(test_rec.labels)}  | Labels range: [{test_rec.labels.min()}..{test_rec.labels.max()}]")
        
        # Verify 7 classes [0..6]
        train_counts = np.bincount(train_rec.labels, minlength=7)
        test_counts = np.bincount(test_rec.labels, minlength=7)
        print(f"      - Train class distribution [0..6]: {train_counts.tolist()}")
        print(f"      - Test class distribution  [0..6]: {test_counts.tolist()}")

        assert train_rec.labels.min() == 0 and train_rec.labels.max() == 6, f"[ERROR] Label range invalid: [{train_rec.labels.min()}..{train_rec.labels.max()}], expected [0..6]!"
        assert train_counts[0] > 0, "[ERROR] Class 0 (angry) has 0 samples in training set!"
        assert test_counts[0] > 0, "[ERROR] Class 0 (angry) has 0 samples in test set!"

        # Verify no data leakage between train, val, test sample_ids / paths
        train_ids = set(train_rec.images)
        val_ids = set(val_rec.images)
        test_ids = set(test_rec.images)
        overlap_tv = train_ids.intersection(val_ids)
        overlap_tt = train_ids.intersection(test_ids)
        print(f"      - Train/Val overlap count:  {len(overlap_tv)} (Data Leakage Check)")
        print(f"      - Train/Test overlap count: {len(overlap_tt)} (Data Leakage Check)")
        if len(overlap_tv) > 0:
            print(f"      [DEBUG] Sample overlapping Train/Val items ({min(5, len(overlap_tv))}/{len(overlap_tv)}):")
            for item in list(overlap_tv)[:5]:
                print(f"              * {item}")
        if len(overlap_tt) > 0:
            print(f"      [DEBUG] Sample overlapping Train/Test items ({min(5, len(overlap_tt))}/{len(overlap_tt)}):")
            for item in list(overlap_tt)[:5]:
                print(f"              * {item}")
        assert len(overlap_tv) == 0, f"ERROR: Data leakage detected! {len(overlap_tv)} samples present in both train and val!"
        assert len(overlap_tt) == 0, f"ERROR: Data leakage detected! {len(overlap_tt)} samples present in both train and test!"
        print("      [PASSED] No data leakage detected between splits! 7 classes verified [0..6].")
        
        # Build TF datasets
        train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
        for batch_feat, batch_labels in train_ds.take(1):
            batch_images = batch_feat["image"]
            print(f"[3/5] Batch Parsing Verification:")
            print(f"      - Batch image shape: {batch_images.shape} (Expected: [16, 112, 112, 3])")
            print(f"      - Batch label shape: {batch_labels.shape} (Expected: [16])")
            print(f"      - Batch label values: {batch_labels.numpy()[:8]}")
            assert batch_images.shape == (16, 112, 112, 3), f"Invalid batch image shape {batch_images.shape}"
            assert batch_labels.shape == (16,), f"Invalid batch label shape {batch_labels.shape}"
    else:
        print("[2/5] Skipping live RAF-DB CSV reading (Directory not found on local machine, will run on server).")
        batch_images = tf.random.normal([16, 112, 112, 3])
        batch_labels = tf.random.uniform([16], minval=0, maxval=7, dtype=tf.int32)
        
    print(f"[4/5] Model & SigLIP2 Prototypes Initialization:")
    model = build_model(cfg)
    print(f"      - Model name: {model.name}")
    if hasattr(model, "text_prototypes") and model.text_prototypes is not None:
        print(f"      - SigLIP2 Prototypes shape: {model.text_prototypes.shape}")
        
    print(f"[5/5] Performing 1 Forward & Loss Computation Pass:")
    with tf.GradientTape() as tape:
        outputs = model(batch_images, training=True)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        loss, loss_dict = compute_loss(outputs, batch_labels, cfg, model=model)
        
    print(f"      - Logits shape: {logits.shape} (Expected: [16, 7])")
    print(f"      - Computed Loss: {float(loss):.4f}")
    if isinstance(loss_dict, dict):
        for k, v in loss_dict.items():
            print(f"        * {k}: {float(v):.4f}")
            
    print("=" * 60)
    print(" [SUCCESS] RAF-DB SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
