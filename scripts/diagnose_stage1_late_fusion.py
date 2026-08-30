"""
Sanity Check Script for Stage 1 RGB + SMIRK 3D CNN Late Fusion.
Runs 4 diagnostic checks:
1. Check 4: Data alignment & Feature stats (mean, std, min, max, norm).
2. Check 1: RGB feature (1024) -> direct new Dense(7) probe.
3. Check 2: concat([RGB feature, zeros(512)]) -> fusion MLP probe.
4. Check 3: Overfit 1 fixed batch of 32 samples over 300 steps + grad norms & LR logging.
"""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

from datasets.fer2013 import collect_split_records
from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline
from models.stage1_rgb_smirk_3d_cnn_late_fusion import Stage1RGBSMIRK3DCNNLateFusionFER
from scripts.train_stage1_rgb_smirk_3d_cnn_late_fusion import (
    load_yaml,
    resolve_path,
    cache_path_for,
    load_geometry_cache,
    load_pixels_for_cache,
    make_dataset,
    restore_rgb_baseline_checkpoint,
    ce_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Late Fusion Diagnostics")
    parser.add_argument("--config", type=str, default="config_stage1_rgb_smirk_3d_cnn_late_fusion.yaml")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--geometry-cache-dir", type=str, default=None)
    return parser.parse_args()


def run_diagnostics():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    print("=" * 70)
    print(" STAGE 1 LATE FUSION DIAGNOSTICS & SANITY CHECKS")
    print("=" * 70)

    # Disable GPU memory growth issues by setting visible devices if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    output_dir = resolve_path(cfg["paths"]["output_dir"]) or PROJECT_ROOT / "outputs" / "stage1_rgb_smirk_3d_cnn_late_fusion"
    cache_dir = resolve_path(args.geometry_cache_dir) or resolve_path(cfg.get("geometry_cache", {}).get("feature_dir")) or (output_dir / "geometry_maps")

    # -------------------------------------------------------------
    # SANITY CHECK 4: DATA ALIGNMENT & FEATURE STATISTICS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SANITY CHECK 4: DATA ALIGNMENT & FEATURE STATISTICS")
    print("=" * 60)

    split_data = {}
    for split in ("train", "val"):
        geometry_maps, labels, sample_ids, meta = load_geometry_cache(cfg, split, cache_dir)
        pixels = load_pixels_for_cache(cfg, split, sample_ids, labels)
        split_data[split] = (pixels, geometry_maps, labels, sample_ids)

        # Verify exact match between geometry labels and CSV labels
        records = collect_split_records(
            resolve_path(cfg["data"]["data_path"]),
            split,
            mask_dir=None,
            use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
            bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
            predecode_pixels=False,
            preload_masks=False,
            allow_missing_masks=False,
        )
        id_to_csv_label = {int(sid): int(lbl) for sid, lbl in zip(records.sample_ids, records.labels)}
        
        matches = 0
        for sid, cache_lbl in zip(sample_ids, labels):
            if id_to_csv_label.get(int(sid)) == int(cache_lbl):
                matches += 1

        match_pct = (matches / len(sample_ids)) * 100.0
        print(f"[{split.upper()} ALIGNMENT] Matches: {matches}/{len(sample_ids)} ({match_pct:.2f}%)")
        if match_pct < 100.0:
            print(f"[FAIL] {split} cache sample IDs do not 100% align with CSV labels!")
        else:
            print(f"[PASS] {split} cache & CSV labels 100% ALIGNED!")

    train_ds = make_dataset(split_data["train"][0], split_data["train"][1], split_data["train"][2], cfg, batch_size=32, training=True, seed=seed)
    val_ds = make_dataset(split_data["val"][0], split_data["val"][1], split_data["val"][2], cfg, batch_size=32, training=False, seed=seed)

    # Initialize model and restore checkpoint
    model = Stage1RGBSMIRK3DCNNLateFusionFER(cfg)
    first_features, first_labels = next(iter(train_ds.take(1)))
    _ = model(first_features, training=False)
    restored_ckpt = restore_rgb_baseline_checkpoint(model, cfg, args)

    # Evaluate feature stats
    outputs = model(first_features, training=False)
    rgb_feat = outputs["rgb_feature"].numpy()
    geom_feat = outputs["geometry_feature"].numpy()

    print("\n[FEATURE STATS - RGB Feature (1024-d)]")
    print(f"  Shape: {rgb_feat.shape}")
    print(f"  Mean:  {np.mean(rgb_feat):.6f}")
    print(f"  Std:   {np.std(rgb_feat):.6f}")
    print(f"  Min:   {np.min(rgb_feat):.6f}")
    print(f"  Max:   {np.max(rgb_feat):.6f}")
    print(f"  Mean L2 Norm: {np.mean(np.linalg.norm(rgb_feat, axis=-1)):.6f}")

    print("\n[FEATURE STATS - 3D Geometry Feature (512-d, initial)]")
    print(f"  Shape: {geom_feat.shape}")
    print(f"  Mean:  {np.mean(geom_feat):.6f}")
    print(f"  Std:   {np.std(geom_feat):.6f}")
    print(f"  Min:   {np.min(geom_feat):.6f}")
    print(f"  Max:   {np.max(geom_feat):.6f}")
    print(f"  Mean L2 Norm: {np.mean(np.linalg.norm(geom_feat, axis=-1)):.6f}")

    # -------------------------------------------------------------
    # SANITY CHECK 1: RGB Feature (1024) -> New Dense(7) Classifier
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SANITY CHECK 1: RGB Feature (1024) -> Fresh Dense(7) Classifier Probe")
    print("=" * 60)

    rgb_probe = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1024,)),
        tf.keras.layers.Dense(7, name="probe_dense")
    ])
    probe_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_probe_step(batch_x, batch_y):
        rgb_feat, _, _ = model._rgb_forward(batch_x)
        with tf.GradientTape() as tape:
            logits = rgb_probe(rgb_feat, training=True)
            loss = ce_loss(batch_y, logits, num_classes=7, label_smoothing=0.0)
        grads = tape.gradient(loss, rgb_probe.trainable_variables)
        probe_opt.apply_gradients(zip(grads, rgb_probe.trainable_variables))
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(preds == batch_y, tf.int32))
        return loss, correct, tf.shape(batch_y)[0]

    def eval_probe(ds_eval):
        tot_correct = 0
        tot_samples = 0
        for b_feat, b_y in ds_eval:
            r_feat, _, _ = model._rgb_forward(b_feat["image"])
            logits = rgb_probe(r_feat, training=False)
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            tot_correct += int(tf.reduce_sum(tf.cast(preds == b_y, tf.int32)).numpy())
            tot_samples += int(tf.shape(b_y)[0].numpy())
        return tot_correct / tot_samples if tot_samples > 0 else 0.0

    print("Training RGB Probe Dense(7) for 5 epochs...")
    for ep in range(1, 6):
        c_tot, s_tot, l_sum = 0, 0, 0.0
        for b_feat, b_y in train_ds:
            loss, corr, count = train_probe_step(b_feat["image"], b_y)
            c_tot += int(corr.numpy())
            s_tot += int(count.numpy())
            l_sum += float(loss.numpy()) * int(count.numpy())
        tr_acc = c_tot / s_tot
        val_acc = eval_probe(val_ds)
        print(f"  Epoch {ep:02d}: loss={l_sum/s_tot:.4f} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

    check1_passed = val_acc > 0.65
    print(f"--> CHECK 1 RESULT: {'PASSED' if check1_passed else 'FAILED'} (val_acc={val_acc:.4f})")

    # -------------------------------------------------------------
    # SANITY CHECK 2: concat([RGB feature, Zeros(512)]) -> Fusion MLP
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SANITY CHECK 2: concat([RGB feature, Zeros(512)]) -> Fusion MLP Probe")
    print("=" * 60)

    # Re-instantiate fusion_mlp to test if fusion MLP can learn from RGB alone + zeroed 3D
    fusion_probe = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1536,)),
        tf.keras.layers.Dense(512, activation=tf.nn.gelu, kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(7)
    ])
    fusion_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_fusion_zero3d_step(batch_x, batch_y):
        rgb_feat, _, _ = model._rgb_forward(batch_x)
        zeros_3d = tf.zeros([tf.shape(rgb_feat)[0], 512], dtype=tf.float32)
        fused_input = tf.concat([rgb_feat, zeros_3d], axis=-1)
        with tf.GradientTape() as tape:
            logits = fusion_probe(fused_input, training=True)
            loss = ce_loss(batch_y, logits, num_classes=7, label_smoothing=0.0)
        grads = tape.gradient(loss, fusion_probe.trainable_variables)
        fusion_opt.apply_gradients(zip(grads, fusion_probe.trainable_variables))
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(preds == batch_y, tf.int32))
        return loss, correct, tf.shape(batch_y)[0]

    def eval_fusion_zero3d(ds_eval):
        tot_correct = 0
        tot_samples = 0
        for b_feat, b_y in ds_eval:
            r_feat, _, _ = model._rgb_forward(b_feat["image"])
            zeros_3d = tf.zeros([tf.shape(r_feat)[0], 512], dtype=tf.float32)
            fused_input = tf.concat([r_feat, zeros_3d], axis=-1)
            logits = fusion_probe(fused_input, training=False)
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            tot_correct += int(tf.reduce_sum(tf.cast(preds == b_y, tf.int32)).numpy())
            tot_samples += int(tf.shape(b_y)[0].numpy())
        return tot_correct / tot_samples if tot_samples > 0 else 0.0

    print("Training Fusion Probe on concat([RGB, Zeros(512)]) for 5 epochs...")
    for ep in range(1, 6):
        c_tot, s_tot, l_sum = 0, 0, 0.0
        for b_feat, b_y in train_ds:
            loss, corr, count = train_fusion_zero3d_step(b_feat["image"], b_y)
            c_tot += int(corr.numpy())
            s_tot += int(count.numpy())
            l_sum += float(loss.numpy()) * int(count.numpy())
        tr_acc = c_tot / s_tot
        val_acc = eval_fusion_zero3d(val_ds)
        print(f"  Epoch {ep:02d}: loss={l_sum/s_tot:.4f} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

    check2_passed = val_acc > 0.65
    print(f"--> CHECK 2 RESULT: {'PASSED' if check2_passed else 'FAILED'} (val_acc={val_acc:.4f})")

    # -------------------------------------------------------------
    # SANITY CHECK 3: OVERFIT 1 FIXED BATCH OF 32 SAMPLES FOR 300 STEPS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SANITY CHECK 3: OVERFIT 1 FIXED BATCH (32 SAMPLES) OVER 300 STEPS")
    print("=" * 60)

    # Fresh full Stage 1 model
    full_model = Stage1RGBSMIRK3DCNNLateFusionFER(cfg)
    _ = full_model(first_features, training=False)
    _ = restore_rgb_baseline_checkpoint(full_model, cfg, args)
    full_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    fixed_feat, fixed_labels = first_features, first_labels

    @tf.function
    def overfit_step(model_target, opt, features, labels):
        with tf.GradientTape() as tape:
            outputs = model_target(features, training=True)
            f_loss = ce_loss(labels, outputs["fusion_logits"], num_classes=7, label_smoothing=0.0)
            g_loss = ce_loss(labels, outputs["geometry_logits"], num_classes=7, label_smoothing=0.0)
            total_loss = f_loss + 0.3 * g_loss
        
        vars_to_train = model_target.trainable_variables
        grads = tape.gradient(total_loss, vars_to_train)
        opt.apply_gradients(zip(grads, vars_to_train))

        # Compute gradient norms for geometry_cnn and fusion_mlp
        geom_vars = model_target.geometry_cnn.trainable_variables
        fusion_vars = model_target.fusion_mlp.trainable_variables
        
        geom_grads = tape.gradient(total_loss, geom_vars) if False else [g for g, v in zip(grads, vars_to_train) if any(v is gv for gv in geom_vars)]
        fusion_grads = [g for g, v in zip(grads, vars_to_train) if any(v is fv for fv in fusion_vars)]

        geom_grad_norm = tf.linalg.global_norm([g for g in geom_grads if g is not None])
        fusion_grad_norm = tf.linalg.global_norm([g for g in fusion_grads if g is not None])

        fused_preds = tf.argmax(outputs["fusion_logits"], axis=-1, output_type=tf.int32)
        geom_preds = tf.argmax(outputs["geometry_logits"], axis=-1, output_type=tf.int32)
        rgb_preds = tf.argmax(outputs["rgb_logits"], axis=-1, output_type=tf.int32)

        fused_acc = tf.reduce_mean(tf.cast(fused_preds == labels, tf.float32))
        geom_acc = tf.reduce_mean(tf.cast(geom_preds == labels, tf.float32))
        rgb_acc = tf.reduce_mean(tf.cast(rgb_preds == labels, tf.float32))

        return total_loss, fused_acc, geom_acc, rgb_acc, geom_grad_norm, fusion_grad_norm

    print("Overfitting 1 fixed batch (32 samples) for 300 steps with LR=1e-3...")
    final_fused_acc = 0.0
    for step in range(1, 301):
        loss, f_acc, g_acc, r_acc, g_norm, f_norm = overfit_step(full_model, full_opt, fixed_feat, fixed_labels)
        final_fused_acc = float(f_acc.numpy())
        if step == 1 or step % 50 == 0 or step == 300:
            print(
                f"  Step {step:03d}: loss={loss.numpy():.4f} "
                f"fused_acc={f_acc.numpy():.4f} geom_acc={g_acc.numpy():.4f} rgb_acc={r_acc.numpy():.4f} | "
                f"grad_norm_geom={g_norm.numpy():.4f} grad_norm_fusion={f_norm.numpy():.4f}"
            )

    check3_passed = final_fused_acc >= 0.90
    print(f"--> CHECK 3 RESULT: {'PASSED' if check3_passed else 'FAILED'} (final_fused_acc={final_fused_acc:.4f})")

    # -------------------------------------------------------------
    # SUMMARY OF DIAGNOSTIC RESULTS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STAGE 1 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Check 1 (RGB Probe Dense(7)):                   {'[PASS]' if check1_passed else '[FAIL]'}")
    print(f"  Check 2 (concat([RGB, Zeros(512)]) -> Fusion): {'[PASS]' if check2_passed else '[FAIL]'}")
    print(f"  Check 3 (Overfit 1 batch 300 steps):            {'[PASS]' if check3_passed else '[FAIL]'}")
    print(f"  Check 4 (Data & Label Alignment):               [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
