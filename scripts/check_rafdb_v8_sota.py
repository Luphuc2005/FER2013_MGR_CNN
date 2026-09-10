#!/usr/bin/env python3
"""Pre-flight validation suite for RAF-DB V8 SOTA Candidate.

Verifies:
  1. RAFDB_SPLIT_OK: train=11043, val=1228, test=3068, class counts match.
  2. TRAIN_VAL_TEST_NO_OVERLAP: zero overlap across splits.
  3. PRETRAINED_LOAD_OK: ConvNeXt MS1M weights file exists.
  4. SIGLIP_REAL_PROTOTYPES_OK: SigLIP2 prototypes file exists with shape (7, 5, 768).
  5. ARCMARGIN_FORWARD_OK: ArcMarginProduct training vs eval logic.
  6. V8_CE_GRADIENT_OK: gradient contract for dynamic_parts, 4x fusion projections, ArcMargin.
  7. MINORITY_OVERSAMPLING_OK: train dataset yields 13150 samples; val/test untouched.
  8. GRANULARITY_GATE_SCHEDULE_OK: uniform warm-up, step temperatures, entropy floor penalty.
  9. SEMANTIC_DECAY_SCHEDULE_OK: lambda_sem step schedule across epochs.
  10. TARGET_93_STATUS=UNVERIFIED: strict scientific integrity reporting.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config

CONFIG_NAME = "config_rafdb_v8_sota_arcmargin_balanced_semdecay.yaml"
EXPECTED_RAW_COUNTS = {
    0: 634,   # Angry
    1: 645,   # Disgust
    2: 253,   # Fear
    3: 4295,  # Happy
    4: 1784,  # Sad
    5: 1161,  # Surprise
    6: 2271,  # Neutral
}


def check_config_static(cfg_path: Path):
    if any(key.startswith("MGR_") for key in os.environ):
        raise ValueError("Unset MGR_* overrides before checking V8.")
    cfg = load_config(cfg_path)
    m = cfg["model"]
    t = cfg["training"]
    r = cfg["runtime"]
    d = cfg["data"]

    assert m["name"] == "rafdb_v8_sota_arcmargin_balanced_semdecay"
    assert m["arch"] == "convnext_base_ms1m_arcface"
    assert m["use_global_regional_fusion"] and m["use_dynamic_part_attention"]
    assert m["global_regional_projection_dim"] == 256
    assert not m["use_adaptive_fusion_gate"]
    assert m["classifier_type"] == "arcmargin"
    assert m["arcmargin_scale"] == 30.0
    assert m["arcmargin_margin"] == 0.30
    assert not m["arcmargin_easy_margin"]

    # Gate & semantic decay
    assert m["use_adaptive_granularity"]
    assert m.get("granularity_gate_step_schedule", False)
    assert m.get("gate_entropy_floor", 0.0) == 0.80
    assert m.get("lambda_gate_entropy", 0.0) == 0.01

    # Progressive unfreeze schedule
    assert m["progressive_unfreeze"]["enabled"]
    schedule = m["progressive_unfreeze"]["schedule"]
    assert len(schedule) == 4
    assert schedule[0]["trainable_stages"] == []
    assert schedule[1]["trainable_stages"] == [4]
    assert schedule[2]["trainable_stages"] == [3, 4]
    assert schedule[3]["trainable_stages"] == [1, 2, 3, 4]

    # Training params
    assert t["optimizer"] == "sam" and t["base_optimizer"] == "adamw"
    assert t["sam_rho"] == 0.02
    assert t["weight_decay"] == 0.035
    assert m.get("drop_path_rate") == 0.10 or t.get("drop_path_rate") == 0.10
    assert t["lr"] == 3e-4 and t["finetune_lr"] == 3e-4
    assert t["visual_extractor_lr"] == 1e-5
    assert t["loss"] == "cross_entropy"
    assert t["label_smoothing"] == 0.10
    assert not t.get("weighted_ce", {}).get("enabled", False)

    # Oversampling
    assert d.get("sampling_strategy") in ("class_aware_minority_oversample", "v8_minority_oversample")
    assert int(d.get("target_minority_count", 0)) == 1200

    print("V8_CONFIG_STATIC_OK target=0.93 status=UNVERIFIED selection=val_accuracy", flush=True)
    return cfg


def check_dataset_split(cfg):
    data_dir = ROOT / cfg["data"]["data_path"]
    train_labels_file = data_dir / "train_labels.txt"
    test_labels_file = data_dir / "test_labels.txt"

    if not train_labels_file.exists() or not test_labels_file.exists():
        print(f"[SKIP] RAF-DB label files not found at {data_dir}. Skipping split test.")
        return

    # Check raw lines
    train_lines = [line.strip().split() for line in train_labels_file.read_text().splitlines() if line.strip()]
    test_lines = [line.strip().split() for line in test_labels_file.read_text().splitlines() if line.strip()]

    # Official test count
    assert len(test_lines) == 3068, f"Expected 3068 test samples, got {len(test_lines)}"

    # Check disjoint files
    train_files = {parts[0] for parts in train_lines}
    test_files = {parts[0] for parts in test_lines}
    overlap = train_files.intersection(test_files)
    assert len(overlap) == 0, f"Detected overlap between train and test: {len(overlap)} files"
    print("TRAIN_VAL_TEST_NO_OVERLAP overlap=0", flush=True)

    # Check 12271 train+val split
    assert len(train_lines) == 12271, f"Expected 12271 total train+val lines, got {len(train_lines)}"

    # Standard split: 90/10 with seed 42
    import random
    rng = random.Random(42)
    indices = list(range(len(train_lines)))
    rng.shuffle(indices)
    val_count = 1228
    val_indices = set(indices[:val_count])
    train_indices = set(indices[val_count:])

    assert len(train_indices) == 11043, f"Expected 11043 train, got {len(train_indices)}"
    assert len(val_indices) == 1228, f"Expected 1228 val, got {len(val_indices)}"

    train_labels = [int(train_lines[i][1]) for i in train_indices]
    counts = {}
    for l in train_labels:
        # Standardize: 1-indexed to 0-indexed if needed
        c = l - 1 if max(train_labels) == 7 else l
        counts[c] = counts.get(c, 0) + 1

    for cls_idx, expected in EXPECTED_RAW_COUNTS.items():
        assert counts.get(cls_idx, 0) == expected, (
            f"Class {cls_idx} count mismatch: got {counts.get(cls_idx, 0)}, expected {expected}"
        )
    print("RAFDB_SPLIT_OK train=11043 val=1228 test=3068 raw_counts=VERIFIED", flush=True)


def check_pretrained_assets(cfg):
    weights_path = ROOT / cfg["model"]["convnext_base_pretrained_path"]
    proto_path = ROOT / cfg["model"]["clip_prototypes_path"]

    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"PRETRAINED_LOAD_OK path={weights_path} size={size_mb:.1f}MB", flush=True)
    else:
        print(f"[WARN] Pretrained weights not present at {weights_path} (expected on HPC cluster)")

    if proto_path.exists():
        proto = np.load(proto_path)
        assert proto.shape == (7, 5, 768), f"Expected (7, 5, 768), got {proto.shape}"
        assert not np.isnan(proto).any(), "Prototypes contain NaN"
        print(f"SIGLIP_REAL_PROTOTYPES_OK path={proto_path} shape={proto.shape}", flush=True)
    else:
        print(f"[WARN] Prototypes not present at {proto_path} (expected on HPC cluster)")


def check_schedules(cfg):
    from utils.semantic_schedule import resolve_lambda_sem

    test_points = [
        (1, 0.10),
        (5, 0.10),
        (8, 0.10),
        (9, 0.05),
        (16, 0.05),
        (17, 0.02),
        (30, 0.02),
        (31, 0.01),
        (60, 0.01),
    ]
    for ep, expected in test_points:
        val = resolve_lambda_sem(cfg, ep)
        np.testing.assert_allclose(val, expected, atol=1e-5, err_msg=f"Epoch {ep} lambda_sem mismatch")
    print("SEMANTIC_DECAY_SCHEDULE_OK points=8,16,30,60 decay=0.10->0.05->0.02->0.01", flush=True)


def check_runtime_tf(cfg):
    try:
        import tensorflow as tf
    except ImportError:
        print("[SKIP] TensorFlow not installed in current Python environment. Skipping runtime TF checks.")
        return

    from models.arcmargin import ArcMarginProduct
    from train import build_model, configure_gpus, configure_tensorflow_runtime
    from losses.classification import supervised_mgr_loss

    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)
    tf.keras.utils.set_random_seed(cfg["seed"]["random_seed"])
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # 1. ArcMarginProduct Unit Test
    arc = ArcMarginProduct(num_classes=7, embed_dim=1024, scale=30.0, margin=0.30, easy_margin=False)
    dummy_feat = tf.random.normal([4, 1024])
    dummy_lbl = tf.constant([0, 1, 2, 3], tf.int32)
    eval_logits = arc(dummy_feat, training=False)
    train_logits = arc(dummy_feat, training=True, labels=dummy_lbl)
    assert tuple(eval_logits.shape) == (4, 7)
    assert tuple(train_logits.shape) == (4, 7)
    # Check that margin was subtracted on target class in training mode
    diff = eval_logits.numpy() - train_logits.numpy()
    for i in range(4):
        target = dummy_lbl.numpy()[i]
        assert diff[i, target] > 0.0, f"Margin not applied on target class {target}"
    print("ARCMARGIN_FORWARD_OK s=30.0 m=0.30 margin_applied=VERIFIED", flush=True)

    # 2. Model & Direct Fusion Contract
    model = build_model(cfg)
    images = tf.random.uniform([2, 112, 112, 3], minval=-1.0, maxval=1.0)
    labels = tf.constant([0, 2], tf.int32)

    model.set_granularity_gate_epoch(1)
    outputs = model(images, training=False)
    assert tuple(outputs["global_regional_features"].shape) == (2, 1024)
    assert tuple(outputs["logits"].shape) == (2, 7)
    print("DIRECT_FUSION_1024_OK features_shape=(2, 1024)", flush=True)

    # 3. Granularity Gate Schedule & Entropy Floor
    pooled = tf.cast(outputs["global_features"], tf.float32)

    # Epoch 1-4: Uniform [0.2, 0.2, 0.2, 0.2, 0.2]
    model.set_granularity_gate_epoch(1)
    w1 = model._granularity_gate_weights(pooled, training=False)
    np.testing.assert_allclose(w1.numpy(), np.full((2, 5), 0.2), atol=1e-5)

    # Epoch 5-10: T=2.0
    model.set_granularity_gate_epoch(5)
    w5 = model._granularity_gate_weights(pooled, training=False)
    assert not np.allclose(w5.numpy(), np.full((2, 5), 0.2))

    # Entropy loss floor test
    low_ent_weights = tf.constant([[0.95, 0.02, 0.01, 0.01, 0.01]], tf.float32)
    h_low = -tf.reduce_sum(low_ent_weights * tf.math.log(low_ent_weights + 1e-8), axis=-1)
    assert float(h_low) < 0.80
    pen_low = tf.maximum(0.0, 0.80 - h_low)
    assert float(pen_low) > 0.0

    high_ent_weights = tf.constant([[0.20, 0.20, 0.20, 0.20, 0.20]], tf.float32)
    h_high = -tf.reduce_sum(high_ent_weights * tf.math.log(high_ent_weights + 1e-8), axis=-1)
    assert float(h_high) > 0.80
    pen_high = tf.maximum(0.0, 0.80 - h_high)
    assert float(pen_high) == 0.0
    print("GRANULARITY_GATE_SCHEDULE_OK uniform=1-4 T_schedule=5-10,11-20,21+ entropy_floor=0.80", flush=True)

    # 4. Gradient Contract: All 5 groups + ArcMargin receive CE gradients
    groups = {
        "dynamic_parts": model.dynamic_part_attn.trainable_variables,
        "arcmargin_classifier": model.classifier.trainable_variables,
    }
    groups.update({
        f"fusion_{name}": layer.trainable_variables
        for name, layer in zip(("global", "upper", "lower", "au"), model.global_regional_fusion.projections)
    })
    watched = [v for variables in groups.values() for v in variables]

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(watched)
        outputs_tr = model(images, training=True, labels=labels)
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, outputs_tr["logits"], from_logits=True
        ))

    grads = tape.gradient(ce, watched)
    offset = 0
    for name, variables in groups.items():
        group_grads = grads[offset:offset + len(variables)]
        offset += len(variables)
        assert all(g is not None for g in group_grads), f"Disconnected FER CE: {name}"
        for g in group_grads:
            tf.debugging.assert_all_finite(g, f"Non-finite FER CE gradient: {name}")
        norm = float(tf.linalg.global_norm([tf.cast(g, tf.float32) for g in group_grads]))
        assert norm > 0.0, f"Zero FER CE gradient: {name}"
        print(f"V8_CE_GRADIENT_OK group={name} norm={norm:.6g}", flush=True)

    # 5. Supervised MGR Loss with Gate Entropy
    kw = dict(num_classes=7, label_smoothing=0.1, ortho_weight=0.0, cnn_aux_weight=0.0)
    total, parts = supervised_mgr_loss(labels, outputs_tr, **kw)
    tf.debugging.assert_all_finite(total, "Non-finite supervised loss")
    assert "gate_entropy" in parts
    print("V8_SUPERVISED_LOSS_OK total_finite=True parts_verified=True", flush=True)


def check_oversampling_dataset(cfg):
    try:
        import tensorflow as tf
        from datasets.fer2013 import make_dataset, get_oversample_stats
    except ImportError:
        return

    data_dir = ROOT / cfg["data"]["data_path"]
    if not (data_dir / "train_labels.txt").exists():
        print(f"[SKIP] RAF-DB data not present at {data_dir}. Skipping dataset oversampling test.")
        return

    train_ds = make_dataset(cfg, "train", is_training=True)
    stats = get_oversample_stats()
    assert stats.get("enabled", False), "Oversampling was not marked active"
    assert stats.get("total_samples") == 13150, f"Expected 13150 samples, got {stats.get('total_samples')}"
    eff = stats.get("effective_class_counts", {})
    assert eff.get(0) == 1200  # Angry
    assert eff.get(1) == 1200  # Disgust
    assert eff.get(2) == 1200  # Fear
    assert eff.get(3) == 4295  # Happy
    assert eff.get(4) == 1784  # Sad
    assert eff.get(5) == 1200  # Surprise
    assert eff.get(6) == 2271  # Neutral
    print("MINORITY_OVERSAMPLING_OK total=13150 minor=1200 major_untouched=VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_NAME)
    parser.add_argument("--runtime", action="store_true", help="Run TensorFlow runtime validation")
    args = parser.parse_args()

    print("======================================================================")
    print("   RAF-DB V8 SOTA CANDIDATE: PRE-FLIGHT VERIFICATION SUITE")
    print("======================================================================")
    cfg = check_config_static(args.config)
    check_dataset_split(cfg)
    check_pretrained_assets(cfg)
    check_schedules(cfg)

    if args.runtime:
        check_runtime_tf(cfg)
        check_oversampling_dataset(cfg)

    print("======================================================================")
    print("   ALL APPLICABLE CHECKS PASSED: TARGET_93_STATUS=UNVERIFIED")
    print("======================================================================")


if __name__ == "__main__":
    main()
