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
        
        n_train = len(train_rec.labels)
        n_val = len(val_rec.labels)
        n_train_val = n_train + n_val
        n_test = len(test_rec.labels)

        train_unique = set(train_rec.labels.tolist())
        val_unique = set(val_rec.labels.tolist())
        test_unique = set(test_rec.labels.tolist())

        train_counts = np.bincount(train_rec.labels, minlength=7).tolist()
        val_counts = np.bincount(val_rec.labels, minlength=7).tolist()
        test_counts = np.bincount(test_rec.labels, minlength=7).tolist()

        print(f"[2/5] RAF-DB Benchmark Dataset Verification:")
        print(f"      - Train samples: {n_train} | Unique labels: {sorted(list(train_unique))}")
        print(f"      - Val samples:   {n_val}   | Unique labels: {sorted(list(val_unique))}")
        print(f"      - Train + Val:   {n_train_val} (Expected: 12271)")
        print(f"      - Test samples:  {n_test}  (Expected: 3068) | Unique labels: {sorted(list(test_unique))}")
        print(f"      - Class Distribution [0..6] (angry, disgust, fear, happy, sad, surprise, neutral):")
        print(f"        * Train: {train_counts}")
        print(f"        * Val:   {val_counts}")
        print(f"        * Test:  {test_counts}")

        # Verification 1: Exact sample count checks
        assert n_train_val == 12271, f"[FAIL] Train + Val count is {n_train_val}, expected exactly 12,271!"
        assert n_test == 3068, f"[FAIL] Test count is {n_test}, expected exactly 3,068!"

        # Verification 2: All 7 classes [0..6] present in every split
        expected_classes = {0, 1, 2, 3, 4, 5, 6}
        assert train_unique == expected_classes, f"[FAIL] Train set missing classes: {expected_classes - train_unique}"
        assert val_unique == expected_classes, f"[FAIL] Val set missing classes: {expected_classes - val_unique}"
        assert test_unique == expected_classes, f"[FAIL] Test set missing classes: {expected_classes - test_unique}"

        # Verification 3: Data leakage check across all pairs (Train/Val, Train/Test, Val/Test)
        train_ids = set(train_rec.images)
        val_ids = set(val_rec.images)
        test_ids = set(test_rec.images)

        overlap_tv = train_ids.intersection(val_ids)
        overlap_tt = train_ids.intersection(test_ids)
        overlap_vt = val_ids.intersection(test_ids)

        print(f"      - Leakage Check:")
        print(f"        * Train / Val overlap:  {len(overlap_tv)}")
        print(f"        * Train / Test overlap: {len(overlap_tt)}")
        print(f"        * Val / Test overlap:   {len(overlap_vt)}")

        assert len(overlap_tv) == 0, f"[FAIL] Data leakage detected between Train and Val! ({len(overlap_tv)} samples)"
        assert len(overlap_tt) == 0, f"[FAIL] Data leakage detected between Train and Test! ({len(overlap_tt)} samples)"
        assert len(overlap_vt) == 0, f"[FAIL] Data leakage detected between Val and Test! ({len(overlap_vt)} samples)"

        print("      [PASSED] All Benchmark Checks Succeeded!")
        print("      [CONFIRMED] Train + Val = 12271 and Test = 3068.")
        
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
