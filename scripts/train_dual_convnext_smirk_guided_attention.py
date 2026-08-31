"""
Train frozen RGB ConvNeXt-B MS1M + FER ckpt-43 anchor with a small SMIRK
geometry encoder that only produces 3D-guided channel attention.

Stage 1 contract:
- RGB backbone, GAP/dropout, and classifier are frozen after ckpt-43 restore.
- Geometry ConvNeXt is not used; geometry is encoded by a small CNN.
- Train only geometry encoder + channel attention + alpha_raw.
- SAM is disabled for Stage 1.
- Mixed precision is enabled and gradients are clipped by global norm=1.0.
- Fail immediately on NaN/Inf.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import collect_split_records
from models.dual_convnext_smirk_guided_attention import DualConvNeXtSMIRKGuidedAttentionFER

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen RGB ckpt-43 + small SMIRK geometry guided attention.")
    parser.add_argument("--config", type=str, default="config_dual_convnext_smirk_guided_attention.yaml")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip baseline-equivalence smoke test.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run contract smoke test and exit.")
    parser.add_argument("--skip-baseline-checkpoint", action="store_true", help="Allow running without restoring FER baseline ckpt-43.")
    parser.add_argument("--use-sam", action="store_true", help="Ignored: SAM is disabled by the Stage 1 contract.")
    parser.add_argument("--sam-rho", type=float, default=0.0, help="Ignored: SAM is disabled by the Stage 1 contract.")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable MirroredStrategy scope when multiple GPUs are visible.")
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


def resolve_checkpoint_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    ckpt_path = resolve_path(path_value)
    if ckpt_path.is_dir():
        latest = tf.train.latest_checkpoint(str(ckpt_path))
        return latest
    return str(ckpt_path)


def checkpoint_exists(ckpt_prefix: Optional[str]) -> bool:
    if not ckpt_prefix:
        return False
    return Path(ckpt_prefix).exists() or Path(ckpt_prefix + ".index").exists()


def restore_rgb_baseline_checkpoint(model: DualConvNeXtSMIRKGuidedAttentionFER, checkpoint_path: Optional[str], required: bool = True) -> Optional[str]:
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_exists(resolved_ckpt):
        message = f"Baseline ckpt-43 checkpoint not found: {checkpoint_path}"
        if required:
            raise FileNotFoundError(message)
        print(f"[WARNING] {message}. RGB anchor restore skipped by request.", flush=True)
        return None

    print(f"[INFO] Restoring FER ckpt-43 into RGB anchor: {resolved_ckpt}", flush=True)
    status = tf.train.Checkpoint(model=model.rgb_baseline).restore(resolved_ckpt)
    status.expect_partial()
    model.freeze_rgb_branch()
    print(f"[INFO] RGB anchor restored and frozen from ckpt-43: {resolved_ckpt}", flush=True)
    return resolved_ckpt


def count_params(variables) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


def load_geometry_cache(cache_dir: Path, pattern: str, split: str) -> Dict[str, np.ndarray]:
    npz_path = cache_dir / pattern.format(split=split)
    if not npz_path.exists():
        raise FileNotFoundError(f"Geometry cache map not found: {npz_path}")
    data = np.load(npz_path)
    geom_maps = data["geometry_maps"]
    if geom_maps.dtype != np.float16:
        geom_maps = geom_maps.astype(np.float16)
    if not np.all(np.isfinite(geom_maps)):
        raise FloatingPointError(f"NaN/Inf in cached geometry maps: {npz_path}")
    return {
        "geometry_maps": geom_maps,
        "labels": data["labels"],
        "sample_ids": data["sample_ids"],
    }


def preprocess_batch_images(images: tf.Tensor, target_size: int = 112) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, [target_size, target_size], method="bilinear")
    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    return images / 255.0


def create_dataset(records, cache_dict: Dict[str, np.ndarray], batch_size: int, is_training: bool = True) -> tf.data.Dataset:
    images = records.images
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
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
            yield {"image": images[idx], "geometry_maps": geom_maps[idx]}, labels[idx]

    output_signature = (
        {
            "image": tf.TensorSpec(shape=images.shape[1:], dtype=tf.uint8),
            "geometry_maps": tf.TensorSpec(shape=geom_maps.shape[1:], dtype=tf.float16),
        },
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(num_samples, 2048), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    def batch_mapper(item, lbl):
        return {
            "image": preprocess_batch_images(item["image"], target_size=112),
            "geometry_maps": item["geometry_maps"],
        }, lbl

    dataset = dataset.map(batch_mapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def run_contract_smoke_test(model: DualConvNeXtSMIRKGuidedAttentionFER, sample_batch) -> bool:
    """Verify dual logits match the restored ckpt-43 RGB baseline when alpha is approximately zero."""
    print("\n" + "=" * 65, flush=True)
    print(" CONTRACT SMOKE TEST: BASELINE EQUIVALENCE (alpha ~= 0)", flush=True)
    print("=" * 65, flush=True)

    inputs = sample_batch[0] if isinstance(sample_batch, tuple) else sample_batch
    images = inputs["image"]

    original_alpha_raw = float(model.alpha_raw.numpy())
    # Force effective alpha to exactly 0.0 in float32 for equivalence only.
    # This does not use the training init and is restored before training.
    model.alpha_raw.assign(-1.0e9)

    baseline_out = model.rgb_baseline(images, training=False)
    baseline_logits = baseline_out["logits"].numpy()

    dual_out = model(inputs, training=False)
    dual_logits = dual_out["final_logits"].numpy()
    F_rgb = dual_out["F_rgb"].numpy()
    F_guided = dual_out["F_guided"].numpy()
    alpha = float(dual_out["alpha"].numpy())

    model.alpha_raw.assign(original_alpha_raw)

    feat_diff = float(np.max(np.abs(F_guided - F_rgb)))
    logits_diff = float(np.max(np.abs(baseline_logits - dual_logits)))

    print(f"[SMOKE_TEST] effective_alpha_for_check: {alpha:.12e}", flush=True)
    print(f"[SMOKE_TEST] Baseline logits shape: {baseline_logits.shape}", flush=True)
    print(f"[SMOKE_TEST] Dual logits shape: {dual_logits.shape}", flush=True)
    print(f"[SMOKE_TEST] Max abs diff (F_guided - F_rgb): {feat_diff:.8e}", flush=True)
    print(f"[SMOKE_TEST] Max abs diff (baseline_logits - dual_logits): {logits_diff:.8e}", flush=True)

    if feat_diff < 1e-5 and logits_diff < 1e-5:
        print("  --> [PASS] CONTRACT SMOKE TEST PASSED", flush=True)
        print("=" * 65 + "\n", flush=True)
        return True

    print(f"  --> [FAIL] Contract smoke test failed: feat_diff={feat_diff:.8e}, logits_diff={logits_diff:.8e}", flush=True)
    print("=" * 65 + "\n", flush=True)
    return False


def build_optimizer(cfg: Dict):
    training_cfg = cfg.get("training", {})
    lr = float(training_cfg.get("learning_rate", training_cfg.get("lr_head", 0.0003)))
    weight_decay = float(training_cfg.get("weight_decay", 0.035))

    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)

    if adamw is not None:
        try:
            return adamw(learning_rate=lr, weight_decay=weight_decay, jit_compile=False)
        except TypeError:
            return adamw(learning_rate=lr, weight_decay=weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def make_loss_fn(cfg: Dict):
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))
    if label_smoothing > 0.0:
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)

        def loss_fn(y_true, y_pred):
            return cce(tf.one_hot(y_true, 7), y_pred)

        return loss_fn
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def finite_or_raise(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"NaN/Inf detected in {name}: {value}")


def evaluate_model(model: DualConvNeXtSMIRKGuidedAttentionFER, dataset: tf.data.Dataset, loss_fn) -> Dict:
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    gate_means = []
    gate_stds = []
    gate_mins = []
    gate_maxs = []

    for inputs, y_batch in dataset:
        outputs = model(inputs, training=False)
        logits = outputs["final_logits"]
        batch_loss = loss_fn(y_batch, logits)
        tf.debugging.assert_all_finite(batch_loss, "NaN/Inf in validation loss")

        batch_size = int(tf.shape(y_batch)[0])
        total_loss += float(batch_loss.numpy()) * batch_size
        total_samples += batch_size

        all_logits.append(logits.numpy())
        all_labels.append(y_batch.numpy())
        gate_means.append(float(outputs["gate_mean"].numpy()))
        gate_stds.append(float(outputs["gate_std"].numpy()))
        gate_mins.append(float(outputs["gate_min"].numpy()))
        gate_maxs.append(float(outputs["gate_max"].numpy()))

    avg_loss = float(total_loss / max(1, total_samples))
    all_logits_arr = np.concatenate(all_logits, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)

    if not np.all(np.isfinite(all_logits_arr)):
        raise FloatingPointError("NaN/Inf detected in validation logits")

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
        "alpha_raw": float(model.alpha_raw.numpy()),
        "alpha": float(model.effective_alpha.numpy()),
        "gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
        "gate_std": float(np.mean(gate_stds)) if gate_stds else 0.0,
        "gate_min": float(np.min(gate_mins)) if gate_mins else 0.0,
        "gate_max": float(np.max(gate_maxs)) if gate_maxs else 0.0,
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"[WARNING] Could not set memory growth for {gpu}: {exc}", flush=True)

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"[INFO] Mixed precision policy set to: {tf.keras.mixed_precision.global_policy().name}", flush=True)

    if args.multi_gpu and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[INFO] MirroredStrategy initialized with {strategy.num_replicas_in_sync} GPUs", flush=True)
    else:
        strategy = tf.distribute.get_strategy()

    if args.use_sam or bool(cfg.get("training", {}).get("use_sam", False)):
        print("[WARNING] SAM requested but ignored: Stage 1 contract requires SAM disabled.", flush=True)

    output_dir = resolve_path(cfg["paths"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints" / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(cfg["data"]["data_path"])
    cache_dir = resolve_path(cfg["geometry_cache"]["feature_dir"])
    pattern = cfg["geometry_cache"]["map_file_pattern"]
    batch_size = int(cfg["data"]["batch_size"])

    print("\n[INFO] Loading FER2013 split records...", flush=True)
    train_records = collect_split_records(data_path, "train", predecode_pixels=True)
    val_records = collect_split_records(data_path, "val", predecode_pixels=True)
    test_records = collect_split_records(data_path, "test", predecode_pixels=True)

    print("[INFO] Loading SMIRK depth+normal geometry cache...", flush=True)
    train_cache = load_geometry_cache(cache_dir, pattern, "train")
    val_cache = load_geometry_cache(cache_dir, pattern, "val")
    test_cache = load_geometry_cache(cache_dir, pattern, "test")

    train_ds = create_dataset(train_records, train_cache, batch_size, is_training=True)
    val_ds = create_dataset(val_records, val_cache, batch_size, is_training=False)
    test_ds = create_dataset(test_records, test_cache, batch_size, is_training=False)

    with strategy.scope():
        model = DualConvNeXtSMIRKGuidedAttentionFER(cfg)
        first_batch = next(iter(train_ds))
        first_inputs, _ = first_batch

        model.load_pretrained_weights(cfg, args)
        restore_rgb_baseline_checkpoint(
            model,
            cfg.get("rgb_backbone", {}).get("baseline_checkpoint_path"),
            required=not args.skip_baseline_checkpoint,
        )
        model.freeze_rgb_branch()
        _ = model(first_inputs, training=False)

        if not args.skip_smoke_test:
            smoke_ok = run_contract_smoke_test(model, first_inputs)
            if not smoke_ok:
                raise RuntimeError("[CONTRACT FAIL] Contract smoke test failed.")
            if args.smoke_test_only:
                print("[INFO] --smoke-test-only requested. Exiting successfully.", flush=True)
                return 0

        optimizer = build_optimizer(cfg)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        loss_fn = make_loss_fn(cfg)

    stage1_vars = (
        list(model.geometry_encoder.trainable_variables)
        + list(model.channel_attention_mlp.trainable_variables)
        + [model.alpha_raw]
    )
    stage1_var_ids = {id(v) for v in stage1_vars}
    leaked_rgb_vars = [v.name for v in model.rgb_baseline.trainable_variables if id(v) in stage1_var_ids]
    if leaked_rgb_vars:
        raise RuntimeError(f"RGB variables leaked into Stage 1 trainables: {leaked_rgb_vars[:5]}")

    grad_clip_norm = float(cfg.get("training", {}).get("grad_clip_norm", 1.0))
    print("=" * 65, flush=True)
    print(" STARTING STAGE 1: FROZEN RGB CKPT-43 + SMALL SMIRK GEOMETRY ATTENTION", flush=True)
    print(f" Epochs: {int(cfg['training']['epochs'])} | SAM: disabled | global_grad_clip_norm: {grad_clip_norm:.3f}", flush=True)
    print(f" Stage 1 trainable params: {count_params(stage1_vars):,}", flush=True)
    print(f" RGB trainable params after freeze: {count_params(model.rgb_baseline.trainable_variables):,}", flush=True)
    print("=" * 65, flush=True)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            outputs = model(x_batch, training=True)
            logits = outputs["final_logits"]
            total_loss = loss_fn(y_batch, logits)
            tf.debugging.assert_all_finite(total_loss, "NaN/Inf in training loss")
            scaled_loss = optimizer.get_scaled_loss(total_loss)

        scaled_grads = tape.gradient(scaled_loss, stage1_vars)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        valid_pairs = [(g, v) for g, v in zip(grads, stage1_vars) if g is not None]
        if valid_pairs:
            valid_grads, valid_vars = zip(*valid_pairs)
            for grad in valid_grads:
                tf.debugging.assert_all_finite(grad, "NaN/Inf in gradients")
            clipped_grads, grad_norm = tf.clip_by_global_norm(list(valid_grads), grad_clip_norm)
            optimizer.apply_gradients(zip(clipped_grads, valid_vars))
        else:
            grad_norm = tf.constant(0.0, dtype=tf.float32)

        preds = tf.argmax(logits, axis=1, output_type=y_batch.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_batch), tf.float32))
        return {
            "loss": tf.cast(total_loss, tf.float32),
            "accuracy": acc,
            "alpha": tf.cast(outputs["alpha"], tf.float32),
            "gate_mean": tf.cast(outputs["gate_mean"], tf.float32),
            "gate_std": tf.cast(outputs["gate_std"], tf.float32),
            "gate_min": tf.cast(outputs["gate_min"], tf.float32),
            "gate_max": tf.cast(outputs["gate_max"], tf.float32),
            "grad_norm": tf.cast(grad_norm, tf.float32),
        }

    epochs = int(cfg["training"]["epochs"])
    best_val_acc = -1.0
    patience = int(cfg["training"]["patience"])
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_samples = 0
        gate_means = []
        gate_stds = []
        gate_mins = []
        gate_maxs = []
        grad_norms = []

        for x_b, y_b in train_ds:
            metrics = train_step(x_b, y_b)
            batch_size = int(tf.shape(y_b)[0])
            loss_v = float(metrics["loss"].numpy())
            acc_v = float(metrics["accuracy"].numpy())
            grad_norm_v = float(metrics["grad_norm"].numpy())
            finite_or_raise("train_loss", loss_v)
            finite_or_raise("train_accuracy", acc_v)
            finite_or_raise("gradient_norm", grad_norm_v)

            train_loss_sum += loss_v * batch_size
            train_acc_sum += acc_v * batch_size
            train_samples += batch_size
            gate_means.append(float(metrics["gate_mean"].numpy()))
            gate_stds.append(float(metrics["gate_std"].numpy()))
            gate_mins.append(float(metrics["gate_min"].numpy()))
            gate_maxs.append(float(metrics["gate_max"].numpy()))
            grad_norms.append(grad_norm_v)

        train_loss = train_loss_sum / max(1, train_samples)
        train_acc = train_acc_sum / max(1, train_samples)
        train_gate_mean = float(np.mean(gate_means)) if gate_means else 0.0
        train_gate_std = float(np.mean(gate_stds)) if gate_stds else 0.0
        train_gate_min = float(np.min(gate_mins)) if gate_mins else 0.0
        train_gate_max = float(np.max(gate_maxs)) if gate_maxs else 0.0
        train_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        val_metrics = evaluate_model(model, val_ds, loss_fn)
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        current_alpha = val_metrics["alpha"]

        for name, value in (
            ("val_loss", val_loss),
            ("val_accuracy", val_acc),
            ("alpha", current_alpha),
            ("val_gate_mean", val_metrics["gate_mean"]),
            ("val_gate_std", val_metrics["gate_std"]),
            ("val_gate_min", val_metrics["gate_min"]),
            ("val_gate_max", val_metrics["gate_max"]),
        ):
            finite_or_raise(name, value)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} - "
            f"alpha={current_alpha:.10f} alpha_raw={val_metrics['alpha_raw']:.6f} - "
            f"gate_train(mean/std/min/max)={train_gate_mean:.6f}/{train_gate_std:.6f}/{train_gate_min:.6f}/{train_gate_max:.6f} - "
            f"gate_val(mean/std/min/max)={val_metrics['gate_mean']:.6f}/{val_metrics['gate_std']:.6f}/{val_metrics['gate_min']:.6f}/{val_metrics['gate_max']:.6f} - "
            f"grad_norm={train_grad_norm:.6f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save_weights(str(ckpt_dir / "ckpt"))
            print(f"  --> Saved new best checkpoint. val_acc={val_acc:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    print("\n[INFO] Evaluating best Stage 1 checkpoint on test set...", flush=True)
    model.load_weights(str(ckpt_dir / "ckpt"))
    test_metrics = evaluate_model(model, test_ds, loss_fn)
    print(f"[TEST METRICS] Test Accuracy: {test_metrics['accuracy']:.4f} | Test Loss: {test_metrics['loss']:.4f}", flush=True)
    print("Classification Report:\n", json.dumps(test_metrics["classification_report"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

