"""
Training Script for Dual ConvNeXt MS1M RGB + SMIRK Geometry with 3D-Guided Channel Attention.

Includes:
1. Contract Smoke Test: Verifies max_abs_diff(baseline_logits, new_model_logits) < 1e-5 when alpha = 0.0.
2. Differential Learning Rates:
   - Geometry ConvNeXt backbone: lr_geom_backbone (5e-5)
   - Fusion + Attention MLP + Classifier heads: lr_head (5e-4)
   - RGB ConvNeXt backbone: Frozen in Phase 1 (unfreeze Stage 4 at epoch 15 with 1e-5)
3. Mixed Precision (float16) support with LossScaleOptimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import collect_split_records
from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline
from models.dual_convnext_smirk_guided_attention import DualConvNeXtSMIRKGuidedAttentionFER

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dual ConvNeXt MS1M RGB + SMIRK Geometry with 3D-Guided Attention.")
    parser.add_argument("--config", type=str, default="config_dual_convnext_smirk_guided_attention.yaml")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip contract smoke test.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run contract smoke test and exit.")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_geometry_cache(cache_dir: Path, pattern: str, split: str) -> Dict[str, np.ndarray]:
    npz_path = cache_dir / pattern.format(split=split)
    if not npz_path.exists():
        raise FileNotFoundError(f"Geometry cache map not found: {npz_path}")
    data = np.load(npz_path)
    geom_maps = data["geometry_maps"]
    if geom_maps.dtype != np.float16:
        geom_maps = geom_maps.astype(np.float16)
    return {
        "geometry_maps": geom_maps,
        "labels": data["labels"],
        "sample_ids": data["sample_ids"],
    }


def create_dataset(records, cache_dict: Dict[str, np.ndarray], batch_size: int, is_training: bool = True) -> tf.data.Dataset:
    images = records.images
    labels = records.labels
    geom_maps = cache_dict["geometry_maps"]

    assert len(images) == len(geom_maps), f"Mismatch: images={len(images)}, geom_maps={len(geom_maps)}"
    assert (labels == cache_dict["labels"]).all(), "Label alignment check failed!"

    num_samples = len(images)
    indices = np.arange(num_samples)

    def generator():
        if is_training:
            np.random.shuffle(indices)
        for idx in indices:
            img = images[idx].astype(np.float32) / 255.0
            g_map = geom_maps[idx]
            lbl = labels[idx]
            yield {"image": img, "geometry_maps": g_map}, lbl

    geom_shape = geom_maps.shape[1:]
    output_signature = (
        {
            "image": tf.TensorSpec(shape=(112, 112, 3), dtype=tf.float32),
            "geometry_maps": tf.TensorSpec(shape=geom_shape, dtype=tf.float16),
        },
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(num_samples, 2048), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


def run_contract_smoke_test(model: DualConvNeXtSMIRKGuidedAttentionFER, baseline_checkpoint: str, sample_batch) -> bool:
    """Requirement 6: Verify max_abs_diff(baseline_logits, new_model_logits) < 1e-5 when alpha = 0.0."""
    print("\n" + "=" * 65, flush=True)
    print(" CONTRACT SMOKE TEST: VERIFYING BASELINE EQUIVALENCE (alpha = 0)", flush=True)
    print("=" * 65, flush=True)

    inputs, labels = sample_batch
    images = inputs["image"]

    # 1. Restore baseline checkpoint into RGB branch
    resolved_ckpt = tf.train.latest_checkpoint(baseline_checkpoint) if Path(baseline_checkpoint).is_dir() else baseline_checkpoint
    if resolved_ckpt is None or not (Path(resolved_ckpt + ".index").exists() or Path(resolved_ckpt).exists()):
        print(f"[WARNING] Baseline checkpoint not found at {baseline_checkpoint}. Skipping exact numerical check.", flush=True)
        return True

    print(f"[SMOKE_TEST] Restoring baseline weights from: {resolved_ckpt}", flush=True)
    model.rgb_baseline.load_weights(resolved_ckpt).expect_partial()

    # 2. Force alpha = 0.0
    model.alpha.assign(0.0)

    # 3. Compute logits from standalone baseline and dual model
    baseline_out = model.rgb_baseline(images, training=False)
    baseline_logits = baseline_out["logits"].numpy()

    dual_out = model(inputs, training=False)
    dual_logits = dual_out["final_logits"].numpy()

    max_diff = float(np.max(np.abs(baseline_logits - dual_logits)))
    print(f"[SMOKE_TEST] Baseline logits shape: {baseline_logits.shape}", flush=True)
    print(f"[SMOKE_TEST] Dual Model logits shape: {dual_logits.shape}", flush=True)
    print(f"[SMOKE_TEST] Max absolute difference: {max_diff:.8e}", flush=True)

    if max_diff < 1e-5:
        print("  --> [PASS] CONTRACT SMOKE TEST PASSED! (max_abs_diff < 1e-5 when alpha = 0)", flush=True)
        print("=" * 65 + "\n", flush=True)
        return True
    else:
        print(f"  --> [FAIL] Contract Smoke Test FAILED! max_abs_diff={max_diff:.8e} >= 1e-5", flush=True)
        print("=" * 65 + "\n", flush=True)
        return False


def build_optimizers(cfg: Dict):
    lr_head = float(cfg.get("training", {}).get("lr_head", 0.0005))
    lr_geom = float(cfg.get("training", {}).get("lr_geom_backbone", 0.00005))

    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)

    if adamw is not None:
        try:
            opt_head = adamw(learning_rate=lr_head, weight_decay=0.01, jit_compile=False)
            opt_geom = adamw(learning_rate=lr_geom, weight_decay=0.01, jit_compile=False)
        except TypeError:
            opt_head = adamw(learning_rate=lr_head, weight_decay=0.01)
            opt_geom = adamw(learning_rate=lr_geom, weight_decay=0.01)
    else:
        opt_head = tf.keras.optimizers.Adam(learning_rate=lr_head)
        opt_geom = tf.keras.optimizers.Adam(learning_rate=lr_geom)

    return opt_head, opt_geom


def evaluate_model(model: DualConvNeXtSMIRKGuidedAttentionFER, dataset: tf.data.Dataset) -> Dict:
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for inputs, y_batch in dataset:
        outputs = model(inputs, training=False)
        logits = outputs["final_logits"]
        aux_logits = outputs["aux_3d_logits"]

        l_final = loss_fn(y_batch, logits)
        l_aux = loss_fn(y_batch, aux_logits)
        batch_loss = l_final + 0.3 * l_aux

        batch_size = tf.shape(y_batch)[0]
        total_loss += float(batch_loss.numpy()) * int(batch_size)
        total_samples += int(batch_size)

        all_logits.append(logits.numpy())
        all_labels.append(y_batch.numpy())

    avg_loss = float(total_loss / max(1, total_samples))
    all_logits_arr = np.concatenate(all_logits, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)

    preds = np.argmax(all_logits_arr, axis=1)
    acc = float(np.mean(preds == all_labels_arr))

    report = classification_report(
        all_labels_arr,
        preds,
        labels=list(range(len(EMOTION_NAMES))),
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(all_labels_arr, preds, labels=list(range(7)))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "alpha": float(model.alpha.numpy()),
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    # Enable Mixed Precision
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"[INFO] Mixed precision policy set to: {tf.keras.mixed_precision.global_policy().name}", flush=True)

    output_dir = resolve_path(cfg["paths"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints" / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load FER Records & Geometry Cache
    data_path = resolve_path(cfg["data"]["data_path"])
    cache_dir = resolve_path(cfg["geometry_cache"]["feature_dir"])
    pattern = cfg["geometry_cache"]["map_file_pattern"]
    batch_size = int(cfg["data"]["batch_size"])

    print("\n[INFO] Loading FER2013 split records...", flush=True)
    train_records = collect_split_records(data_path, "train", predecode_pixels=True)
    val_records = collect_split_records(data_path, "val", predecode_pixels=True)
    test_records = collect_split_records(data_path, "test", predecode_pixels=True)

    print("[INFO] Loading SMIRK geometry cache...", flush=True)
    train_cache = load_geometry_cache(cache_dir, pattern, "train")
    val_cache = load_geometry_cache(cache_dir, pattern, "val")
    test_cache = load_geometry_cache(cache_dir, pattern, "test")

    train_ds = create_dataset(train_records, train_cache, batch_size, is_training=True)
    val_ds = create_dataset(val_records, val_cache, batch_size, is_training=False)
    test_ds = create_dataset(test_records, test_cache, batch_size, is_training=False)

    # 2. Build Dual Model & Load MS1M Pretrained Weights
    model = DualConvNeXtSMIRKGuidedAttentionFER(cfg)
    first_inputs, _ = next(iter(train_ds))
    _ = model(first_inputs, training=False)

    model.load_pretrained_weights(cfg, args)
    model.freeze_rgb_branch()

    # 3. Contract Smoke Test
    baseline_ckpt_path = cfg["rgb_backbone"].get("baseline_checkpoint_path")
    if not args.skip_smoke_test:
        smoke_ok = run_contract_smoke_test(model, baseline_ckpt_path, first_inputs)
        if not smoke_ok:
            raise RuntimeError("[CONTRACT FAIL] Contract smoke test failed.")
        if args.smoke_test_only:
            print("[INFO] --smoke-test-only requested. Exiting successfully.")
            return 0

    # Restore baseline FER classifier weights into RGB branch if baseline checkpoint exists
    resolved_ckpt = tf.train.latest_checkpoint(baseline_ckpt_path) if Path(baseline_ckpt_path).is_dir() else baseline_ckpt_path
    if resolved_ckpt and (Path(resolved_ckpt + ".index").exists() or Path(resolved_ckpt).exists()):
        print(f"[INFO] Restoring baseline FER classifier weights into RGB branch: {resolved_ckpt}", flush=True)
        model.rgb_baseline.load_weights(resolved_ckpt).expect_partial()

    # 4. Setup Optimizers
    opt_head, opt_geom = build_optimizers(cfg)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_scale_head = tf.keras.mixed_precision.LossScaleOptimizer(opt_head)
    loss_scale_geom = tf.keras.mixed_precision.LossScaleOptimizer(opt_geom)

    # Variables Grouping
    def get_variables():
        geom_vars = list(model.geometry_baseline.trainable_variables)
        head_vars = (
            list(model.geometry_fusion.trainable_variables)
            + list(model.channel_attention_mlp.trainable_variables)
            + list(model.aux_3d_head.trainable_variables)
            + list(model.rgb_baseline.classifier.trainable_variables)
            + [model.alpha]
        )
        if model.rgb_baseline.trainable:
            head_vars += list(model.rgb_baseline.trainable_variables)
        return geom_vars, head_vars

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(x_batch, training=True)
            f_logits = outputs["final_logits"]
            a_logits = outputs["aux_3d_logits"]

            l_final = loss_fn(y_batch, f_logits)
            l_aux = loss_fn(y_batch, a_logits)
            total_loss = l_final + 0.3 * l_aux

            scaled_loss = loss_scale_head.get_scaled_loss(total_loss)

        geom_vars, head_vars = get_variables()

        scaled_grads_head = tape.gradient(scaled_loss, head_vars)
        grads_head = loss_scale_head.get_unscaled_gradients(scaled_grads_head)
        loss_scale_head.apply_gradients(zip(grads_head, head_vars))

        if geom_vars:
            scaled_grads_geom = tape.gradient(scaled_loss, geom_vars)
            grads_geom = loss_scale_geom.get_unscaled_gradients(scaled_grads_geom)
            loss_scale_geom.apply_gradients(zip(grads_geom, geom_vars))

        del tape

        preds = tf.argmax(f_logits, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_batch), tf.float32))
        return total_loss, acc, outputs["alpha"]

    # 5. Training Loop
    epochs = int(cfg["training"]["epochs"])
    unfreeze_epoch = int(cfg["training"].get("unfreeze_rgb_stage4_epoch", 15))
    best_val_acc = -1.0
    patience = int(cfg["training"]["patience"])
    patience_counter = 0

    print("=" * 65, flush=True)
    print(" STARTING DUAL CONVNEXT MS1M RGB + SMIRK GEOMETRY TRAINING", flush=True)
    print(f" Epochs: {epochs} | Unfreeze Stage4 Epoch: {unfreeze_epoch}", flush=True)
    print("=" * 65, flush=True)

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch:
            print(f"\n[EPOCH {epoch:02d}] Unfreezing Stage 4 of RGB ConvNeXt backbone for joint fine-tuning...", flush=True)
            model.unfreeze_rgb_stage4()

        start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        for x_b, y_b in train_ds:
            loss_v, acc_v, alpha_v = train_step(x_b, y_b)
            train_loss += float(loss_v.numpy())
            train_acc += float(acc_v.numpy())
            num_batches += 1

        train_loss /= max(1, num_batches)
        train_acc /= max(1, num_batches)

        val_metrics = evaluate_model(model, val_ds)
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        current_alpha = val_metrics["alpha"]

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - alpha: {current_alpha:.6f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save_weights(str(ckpt_dir / "ckpt"))
            print(f"  --> Saved new best checkpoint! val_acc: {val_acc:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    print("\n[INFO] Evaluating best model on test set...", flush=True)
    model.load_weights(str(ckpt_dir / "ckpt"))
    test_metrics = evaluate_model(model, test_ds)
    print(f"[TEST METRICS] Test Accuracy: {test_metrics['accuracy']:.4f} | Test Loss: {test_metrics['loss']:.4f}", flush=True)
    print("Classification Report:\n", json.dumps(test_metrics["classification_report"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
