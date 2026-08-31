"""Train ConvNeXt-B MS1M Cross-Stage Swin fusion for FER2013.

New experiment only. It never restores any FER checkpoint and writes only to
outputs/tf_runs/convnext_ms1m_cross_stage_swin by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import global_batch_size, load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets
from metrics.classification import classification_metrics, save_metrics
from models.convnext_ms1m_cross_stage_swin import ConvNeXtMS1MCrossStageSwinFER, count_params
from train import build_optimizer, configure_gpus, configure_tensorflow_runtime, resolve_phase_lrs, set_optimizer_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-B MS1M Cross-Stage Swin fusion on FER2013.")
    parser.add_argument("--config", type=str, default="config_convnext_ms1m_cross_stage_swin.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from this experiment's latest checkpoint only.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run shape/identity/gradient smoke tests and exit.")
    parser.add_argument("--skip-smoke-tests", action="store_true", help="Debug only: skip required smoke tests.")
    return parser.parse_args()


def ensure_finite_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"NaN/Inf detected in {name}: {value}")


def unwrap_optimizer(optimizer):
    for attr in ("inner_optimizer", "_optimizer"):
        inner = getattr(optimizer, attr, None)
        if inner is not None:
            return inner
    return optimizer


def set_lso_lr(optimizer, lr_value: float) -> None:
    set_optimizer_lr(unwrap_optimizer(optimizer), float(lr_value))


def cross_entropy_loss(labels: tf.Tensor, logits: tf.Tensor, cfg: Dict) -> tf.Tensor:
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.int32)
    num_classes = int(cfg["data"].get("num_classes", 7))
    smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    if smoothing > 0.0:
        targets = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        targets = targets * (1.0 - smoothing) + smoothing / float(num_classes)
        loss = tf.keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    tf.debugging.assert_all_finite(loss, "NaN/Inf in CE loss")
    return tf.cast(loss, tf.float32)


def tensor_shape_list(tensor: tf.Tensor) -> List[Optional[int]]:
    static = tensor.shape.as_list()
    dynamic = tf.shape(tensor).numpy().tolist()
    return [static[i] if static[i] is not None else int(dynamic[i]) for i in range(len(dynamic))]


def run_shape_smoke_test(model: ConvNeXtMS1MCrossStageSwinFER, sample_inputs: Dict[str, tf.Tensor]) -> Dict[str, object]:
    print("=" * 72, flush=True)
    print("SHAPE SMOKE TEST", flush=True)
    print("=" * 72, flush=True)
    outputs = model(sample_inputs, training=False)
    expected = {
        "S2": [28, 28, 256],
        "projected_S2": [14, 14, 512],
        "S3": [14, 14, 512],
        "G3": [14, 14, 512],
        "projected_G3": [7, 7, 1024],
        "S4": [7, 7, 1024],
        "G4": [7, 7, 1024],
        "logits": [7],
    }
    observed = {}
    for key, tail in expected.items():
        shape = tensor_shape_list(outputs[key])
        observed[key] = shape
        print(f"  {key}: {shape}", flush=True)
        if shape[1:] != tail:
            raise RuntimeError(f"Shape smoke failed for {key}: got {shape}, expected [B,{','.join(map(str, tail))}]")
    print("SHAPE_SMOKE_OK", flush=True)
    return observed


def run_identity_smoke_test(model: ConvNeXtMS1MCrossStageSwinFER, sample_inputs: Dict[str, tf.Tensor], cfg: Dict) -> Dict[str, float]:
    print("=" * 72, flush=True)
    print("IDENTITY / SAFE INITIALIZATION SMOKE TEST", flush=True)
    print("=" * 72, flush=True)
    outputs = model(sample_inputs, training=False)
    logits = outputs["logits"].numpy()
    plain_logits = outputs["plain_logits"].numpy()
    diff = np.abs(logits - plain_logits)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    alpha3_mean = float(outputs["alpha3_mean"].numpy())
    alpha4_mean = float(outputs["alpha4_mean"].numpy())
    tol = float(cfg.get("smoke_tests", {}).get("identity_max_abs_logit_diff", 1e-5))
    print(f"  max_abs_logit_diff: {max_diff:.8e}", flush=True)
    print(f"  mean_abs_logit_diff: {mean_diff:.8e}", flush=True)
    print(f"  alpha3_mean: {alpha3_mean:.8e}", flush=True)
    print(f"  alpha4_mean: {alpha4_mean:.8e}", flush=True)
    if not np.isfinite(max_diff) or not np.isfinite(mean_diff):
        raise FloatingPointError("NaN/Inf in identity smoke logit diff")
    if max_diff > tol:
        raise RuntimeError(f"Identity smoke failed: max_abs_logit_diff={max_diff:.8e} > {tol:.8e}")
    print("IDENTITY_SMOKE_OK", flush=True)
    return {
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "alpha3_mean": alpha3_mean,
        "alpha4_mean": alpha4_mean,
    }


def finite_grads_or_raise(label: str, grads: Sequence[Optional[tf.Tensor]]) -> None:
    for idx, grad in enumerate(grads):
        if grad is not None:
            tf.debugging.assert_all_finite(grad, f"NaN/Inf in {label} gradient #{idx}")


def non_none_grads_and_vars(grads: Sequence[Optional[tf.Tensor]], variables: Sequence[tf.Variable]):
    return [(g, v) for g, v in zip(grads, variables) if g is not None]


def run_gradient_smoke_test(model: ConvNeXtMS1MCrossStageSwinFER, sample_inputs: Dict[str, tf.Tensor], sample_labels: tf.Tensor, cfg: Dict) -> Dict[str, float]:
    print("=" * 72, flush=True)
    print("GRADIENT + SAM DRY-RUN SMOKE TEST", flush=True)
    print("=" * 72, flush=True)
    train_vars = model.head_variables() + model.backbone_variables()
    cross_ids = {id(v) for v in model.cross_stage_variables()}
    sam_rho = float(cfg["training"].get("sam_rho", 0.03))

    with tf.GradientTape() as tape:
        outputs = model(sample_inputs, training=True)
        loss = cross_entropy_loss(sample_labels, outputs["logits"], cfg)
    grads = tape.gradient(loss, train_vars)
    finite_grads_or_raise("first-step", grads)
    grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
    tf.debugging.assert_all_finite(grad_norm, "NaN/Inf in first-step global grad norm")

    cross_norms = [float(tf.norm(g).numpy()) for g, v in zip(grads, train_vars) if g is not None and id(v) in cross_ids]
    max_cross_norm = max(cross_norms) if cross_norms else 0.0
    min_cross_norm = float(cfg.get("smoke_tests", {}).get("min_cross_stage_grad_norm", 1e-12))
    if max_cross_norm <= min_cross_norm:
        raise RuntimeError(f"Cross-stage module gradient smoke failed: max_cross_stage_grad_norm={max_cross_norm:.8e}")

    eps_list = []
    scale = tf.cast(sam_rho, tf.float32) / (tf.cast(grad_norm, tf.float32) + 1e-12)
    for var, grad in zip(train_vars, grads):
        if grad is None:
            eps_list.append(None)
            continue
        eps = tf.cast(tf.cast(grad, tf.float32) * scale, var.dtype)
        var.assign_add(eps)
        eps_list.append(eps)

    try:
        with tf.GradientTape() as tape2:
            outputs2 = model(sample_inputs, training=True)
            loss2 = cross_entropy_loss(sample_labels, outputs2["logits"], cfg)
        grads2 = tape2.gradient(loss2, train_vars)
        finite_grads_or_raise("second-step", grads2)
        grad_norm2 = tf.linalg.global_norm([g for g in grads2 if g is not None])
        tf.debugging.assert_all_finite(grad_norm2, "NaN/Inf in second-step global grad norm")
    finally:
        for var, eps in zip(train_vars, eps_list):
            if eps is not None:
                var.assign_sub(eps)

    result = {
        "loss": float(loss.numpy()),
        "sam_perturbed_loss": float(loss2.numpy()),
        "grad_norm": float(grad_norm.numpy()),
        "sam_second_grad_norm": float(grad_norm2.numpy()),
        "max_cross_stage_grad_norm": float(max_cross_norm),
    }
    for key, value in result.items():
        ensure_finite_scalar(key, value)
        print(f"  {key}: {value:.8e}", flush=True)
    print("GRADIENT_SMOKE_OK", flush=True)
    return result


def make_train_step(model: ConvNeXtMS1MCrossStageSwinFER, cfg: Dict, optimizer_head, optimizer_backbone, train_backbone: bool):
    head_vars = model.head_variables()
    backbone_vars = model.backbone_variables() if train_backbone else []
    sam_rho = float(cfg["training"].get("sam_rho", 0.03))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))
    train_vars = head_vars + backbone_vars

    def _compute_unscaled_grads(tape, scaled_head_loss, scaled_backbone_loss):
        scaled_grads_head = tape.gradient(scaled_head_loss, head_vars)
        grads_head = optimizer_head.get_unscaled_gradients(scaled_grads_head)
        if backbone_vars:
            scaled_grads_backbone = tape.gradient(scaled_backbone_loss, backbone_vars)
            grads_backbone = optimizer_backbone.get_unscaled_gradients(scaled_grads_backbone)
        else:
            grads_backbone = []
        return grads_head, grads_backbone

    def _clip_all(grads_head, grads_backbone):
        valid = non_none_grads_and_vars(grads_head, head_vars) + non_none_grads_and_vars(grads_backbone, backbone_vars)
        if not valid:
            raise RuntimeError("No valid gradients in train_step; check LossScaleOptimizer/GradientTape path.")
        grads_all = [g for g, _ in valid]
        vars_all = [v for _, v in valid]
        finite_grads_or_raise("train", grads_all)
        clipped_all, grad_norm = tf.clip_by_global_norm(grads_all, grad_clip_norm)
        clipped_head = []
        vars_head = []
        clipped_backbone = []
        vars_backbone = []
        for grad, var in zip(clipped_all, vars_all):
            if id(var) in {id(h) for h in head_vars}:
                clipped_head.append(grad)
                vars_head.append(var)
            else:
                clipped_backbone.append(grad)
                vars_backbone.append(var)
        return (list(zip(clipped_head, vars_head)), list(zip(clipped_backbone, vars_backbone)), tf.cast(grad_norm, tf.float32))

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            loss = cross_entropy_loss(labels, outputs["logits"], cfg)
            scaled_head_loss = optimizer_head.get_scaled_loss(loss)
            scaled_backbone_loss = optimizer_backbone.get_scaled_loss(loss) if backbone_vars else None
        grads_head, grads_backbone = _compute_unscaled_grads(tape, scaled_head_loss, scaled_backbone_loss)
        del tape
        clipped_head_pairs, clipped_backbone_pairs, grad_norm = _clip_all(grads_head, grads_backbone)

        valid_pairs = clipped_head_pairs + clipped_backbone_pairs
        eps_list = []
        scale = tf.cast(sam_rho, tf.float32) / (grad_norm + 1e-12)
        for grad, var in valid_pairs:
            eps = tf.cast(tf.cast(grad, tf.float32) * scale, var.dtype)
            var.assign_add(eps)
            eps_list.append((var, eps))

        with tf.GradientTape(persistent=True) as tape2:
            outputs2 = model(inputs, training=True)
            loss2 = cross_entropy_loss(labels, outputs2["logits"], cfg)
            scaled_head_loss2 = optimizer_head.get_scaled_loss(loss2)
            scaled_backbone_loss2 = optimizer_backbone.get_scaled_loss(loss2) if backbone_vars else None
        grads_head2, grads_backbone2 = _compute_unscaled_grads(tape2, scaled_head_loss2, scaled_backbone_loss2)
        del tape2

        for var, eps in eps_list:
            var.assign_sub(eps)

        clipped_head_pairs2, clipped_backbone_pairs2, grad_norm2 = _clip_all(grads_head2, grads_backbone2)
        if clipped_head_pairs2:
            optimizer_head.apply_gradients(clipped_head_pairs2)
        if clipped_backbone_pairs2:
            optimizer_backbone.apply_gradients(clipped_backbone_pairs2)

        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        labels_i32 = tf.cast(labels, tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels_i32), tf.int32))
        count = tf.shape(labels_i32)[0]
        return {
            "loss": tf.cast(loss, tf.float32),
            "correct": correct,
            "count": count,
            "grad_norm": tf.cast(grad_norm, tf.float32),
            "sam_second_grad_norm": tf.cast(grad_norm2, tf.float32),
            "alpha3_mean": tf.cast(outputs["alpha3_mean"], tf.float32),
            "alpha3_min": tf.cast(outputs["alpha3_min"], tf.float32),
            "alpha3_max": tf.cast(outputs["alpha3_max"], tf.float32),
            "alpha3_std": tf.cast(outputs["alpha3_std"], tf.float32),
            "alpha4_mean": tf.cast(outputs["alpha4_mean"], tf.float32),
            "alpha4_min": tf.cast(outputs["alpha4_min"], tf.float32),
            "alpha4_max": tf.cast(outputs["alpha4_max"], tf.float32),
            "alpha4_std": tf.cast(outputs["alpha4_std"], tf.float32),
        }

    return train_step


def evaluate_dataset(model: ConvNeXtMS1MCrossStageSwinFER, dataset: tf.data.Dataset, cfg: Dict, use_tta_hflip: bool) -> Dict[str, object]:
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_count = 0
    alpha3_values: List[float] = []
    alpha4_values: List[float] = []
    tta_cfg = cfg.get("tta", {})
    w_orig = float(tta_cfg.get("original_weight", 0.5))
    w_flip = float(tta_cfg.get("flip_weight", 0.5))

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        logits = outputs["logits"]
        if use_tta_hflip:
            flipped = dict(inputs)
            flipped["image"] = tf.image.flip_left_right(inputs["image"])
            outputs_flip = model(flipped, training=False)
            logits = w_orig * logits + w_flip * outputs_flip["logits"]
        loss = cross_entropy_loss(labels, logits, cfg)
        count = int(tf.shape(labels)[0])
        total_loss += float(loss.numpy()) * count
        total_count += count
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(tf.cast(labels, tf.int32).numpy().tolist())
        alpha3_values.extend(tf.reshape(outputs["alpha3"], [-1]).numpy().astype(float).tolist())
        alpha4_values.extend(tf.reshape(outputs["alpha4"], [-1]).numpy().astype(float).tolist())

    metrics = classification_metrics(y_true, y_pred, EMOTION_NAMES)
    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["ce_loss"] = metrics["loss"]
    metrics["tta_hflip"] = bool(use_tta_hflip)
    for prefix, values in (("alpha3", alpha3_values), ("alpha4", alpha4_values)):
        arr = np.asarray(values, dtype=np.float64) if values else np.asarray([0.0])
        metrics[f"{prefix}_mean"] = float(np.mean(arr))
        metrics[f"{prefix}_min"] = float(np.min(arr))
        metrics[f"{prefix}_max"] = float(np.max(arr))
        metrics[f"{prefix}_std"] = float(np.std(arr))
    return metrics


def query_gpu_memory_mb() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return ",".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    except Exception:
        return "unavailable"


def write_history_csv(history: List[Dict[str, object]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def aggregate_train_values(values: List[Dict[str, float]], key: str, mode: str = "mean") -> float:
    vals = [float(item[key]) for item in values]
    if not vals:
        return 0.0
    if mode == "min":
        return float(np.min(vals))
    if mode == "max":
        return float(np.max(vals))
    return float(np.mean(vals))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]))
    run_dir = Path(cfg["paths"]["output_dir"])
    checkpoint_root = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"].get("random_seed", 42)))
    configure_gpus(cfg)
    print(f"TensorFlow {tf.__version__}", flush=True)
    print(f"Global batch size: {global_batch_size(cfg, replicas=1)}", flush=True)

    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
    first_inputs, first_labels = next(iter(train_ds.take(1)))

    with tf.device("/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"):
        model = ConvNeXtMS1MCrossStageSwinFER(cfg)
        _ = model(first_inputs, training=False)
        total_params = count_params(model.variables)
        trainable_params = count_params(model.trainable_variables)
        backbone_params = count_params(model.backbone_variables())
        head_params = count_params(model.head_variables())
        cross_params = count_params(model.cross_stage_variables())
        print("=" * 72, flush=True)
        print("PARAMETER SUMMARY", flush=True)
        print(f"  total_params: {total_params:,}", flush=True)
        print(f"  trainable_params: {trainable_params:,}", flush=True)
        print(f"  backbone_trainable_params: {backbone_params:,}", flush=True)
        print(f"  head_trainable_params: {head_params:,}", flush=True)
        print(f"  cross_stage_trainable_params: {cross_params:,}", flush=True)
        print("=" * 72, flush=True)

        if not args.skip_smoke_tests:
            shape_result = run_shape_smoke_test(model, first_inputs)
            identity_result = run_identity_smoke_test(model, first_inputs, cfg)
            grad_result = run_gradient_smoke_test(model, first_inputs, first_labels, cfg)
            smoke_payload = {
                "shape": shape_result,
                "identity": identity_result,
                "gradient": grad_result,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "backbone_trainable_params": backbone_params,
                "head_trainable_params": head_params,
                "cross_stage_trainable_params": cross_params,
            }
            save_metrics(smoke_payload, run_dir / "smoke_tests.json")
            if args.smoke_test_only:
                print("[INFO] --smoke-test-only requested. Stop before training.", flush=True)
                return 0

        optimizer_head = tf.keras.mixed_precision.LossScaleOptimizer(build_optimizer(cfg, float(cfg["training"]["lr"])))
        optimizer_backbone = tf.keras.mixed_precision.LossScaleOptimizer(
            build_optimizer(cfg, float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])))
        )
        ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt_best_acc = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        ckpt_best_macro = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        checkpoint = tf.train.Checkpoint(
            epoch=ckpt_epoch,
            best_accuracy=ckpt_best_acc,
            best_macro_f1=ckpt_best_macro,
            model=model,
            optimizer_head=optimizer_head,
            optimizer_backbone=optimizer_backbone,
        )
        last_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "last"), max_to_keep=1)
        best_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "best"), max_to_keep=1)
        if (args.resume or bool(cfg["training"].get("resume", False))) and last_manager.latest_checkpoint:
            checkpoint.restore(last_manager.latest_checkpoint).expect_partial()
            print(f"[INFO] Resumed this experiment from {last_manager.latest_checkpoint}", flush=True)

    epochs = int(cfg["training"]["epochs"])
    freeze_epochs = int(cfg["model"].get("freeze_backbone_epochs", 0) or 0)
    unfreeze_backbone = bool(cfg["model"].get("unfreeze_backbone", True))
    patience_limit = int(cfg["training"].get("patience", 15))
    monitor_name = str(cfg["training"].get("monitor", "val_accuracy"))
    start_epoch = int(ckpt_epoch.numpy())
    best_acc = float(ckpt_best_acc.numpy())
    best_macro = float(ckpt_best_macro.numpy())
    best_epoch = start_epoch if best_acc >= 0.0 else -1
    history: List[Dict[str, object]] = []
    train_step_head = make_train_step(model, cfg, optimizer_head, optimizer_backbone, train_backbone=False)
    train_step_full = make_train_step(model, cfg, optimizer_head, optimizer_backbone, train_backbone=True)

    print("=" * 72, flush=True)
    print("STARTING TRAINING: ConvNeXt-B MS1M + Cross-Stage Shifted-Window Fusion", flush=True)
    print(f"SAM rho={float(cfg['training'].get('sam_rho', 0.03)):.4f} | mixed_precision=mixed_float16 | grad_clip_norm={float(cfg['training'].get('grad_clip_norm', 1.0)):.3f}", flush=True)
    print(f"freeze_backbone_epochs={freeze_epochs} | output_dir={run_dir}", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_backbone = bool(unfreeze_backbone and epoch >= freeze_epochs)
        train_step = train_step_full if train_backbone else train_step_head
        lr_head, lr_backbone = resolve_phase_lrs(cfg, epoch, train_backbone)
        set_lso_lr(optimizer_head, lr_head)
        set_lso_lr(optimizer_backbone, lr_backbone)
        if train_backbone and epoch == freeze_epochs:
            print(f"[INFO] Unfreezing ConvNeXt backbone at epoch {epoch + 1}", flush=True)

        seen = 0
        correct = 0
        losses = []
        batch_metrics: List[Dict[str, float]] = []
        total_steps = int(tf.data.experimental.cardinality(train_ds).numpy())
        progress_interval = int(cfg["training"].get("progress_interval", 100) or 0)

        for step, (inputs, labels) in enumerate(train_ds, start=1):
            metrics = train_step(inputs, labels)
            loss_v = float(metrics["loss"].numpy())
            grad_norm_v = float(metrics["grad_norm"].numpy())
            sam_grad_norm_v = float(metrics["sam_second_grad_norm"].numpy())
            ensure_finite_scalar("train_loss", loss_v)
            ensure_finite_scalar("grad_norm", grad_norm_v)
            ensure_finite_scalar("sam_second_grad_norm", sam_grad_norm_v)
            losses.append(loss_v)
            correct += int(metrics["correct"].numpy())
            seen += int(metrics["count"].numpy())
            batch_record = {k: float(metrics[k].numpy()) for k in (
                "grad_norm", "sam_second_grad_norm",
                "alpha3_mean", "alpha3_min", "alpha3_max", "alpha3_std",
                "alpha4_mean", "alpha4_min", "alpha4_max", "alpha4_std",
            )}
            batch_metrics.append(batch_record)
            if progress_interval and step % progress_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                    f"loss={loss_v:.4f} acc={correct / max(seen, 1):.4f} "
                    f"grad_norm={grad_norm_v:.6f} lr_head={lr_head:.6f} lr_backbone={lr_backbone:.6f}",
                    flush=True,
                )

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = correct / max(seen, 1)
        ensure_finite_scalar("epoch_train_loss", train_loss)
        ensure_finite_scalar("epoch_train_accuracy", train_acc)

        val_metrics = evaluate_dataset(
            model,
            val_ds,
            cfg,
            use_tta_hflip=bool(cfg["runtime"].get("train_val_tta_hflip", False)),
        )
        val_acc = float(val_metrics["accuracy"])
        val_macro = float(val_metrics["macro_f1"])
        val_loss = float(val_metrics["loss"])
        for name, value in (("val_loss", val_loss), ("val_accuracy", val_acc), ("val_macro_f1", val_macro)):
            ensure_finite_scalar(name, value)

        improved = val_acc > best_acc + 1e-12 or (abs(val_acc - best_acc) <= 1e-12 and val_macro > best_macro)
        if improved:
            best_acc = val_acc
            best_macro = val_macro
            best_epoch = epoch + 1
            ckpt_best_acc.assign(best_acc)
            ckpt_best_macro.assign(best_macro)
            print(f"[INFO] Save best checkpoint at epoch {epoch + 1}: val_acc={val_acc:.4f}, val_macro_f1={val_macro:.4f}", flush=True)
            best_manager.save(checkpoint_number=epoch + 1)

        ckpt_epoch.assign(epoch + 1)
        last_manager.save(checkpoint_number=epoch + 1)
        patience_counter = 0 if best_epoch < 0 else (epoch + 1) - best_epoch
        epoch_time = time.time() - epoch_start
        gpu_mem = query_gpu_memory_mb()
        row = {
            "epoch": epoch + 1,
            "epoch_time_sec": round(epoch_time, 2),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro,
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "alpha3_mean": float(val_metrics["alpha3_mean"]),
            "alpha3_min": float(val_metrics["alpha3_min"]),
            "alpha3_max": float(val_metrics["alpha3_max"]),
            "alpha3_std": float(val_metrics["alpha3_std"]),
            "alpha4_mean": float(val_metrics["alpha4_mean"]),
            "alpha4_min": float(val_metrics["alpha4_min"]),
            "alpha4_max": float(val_metrics["alpha4_max"]),
            "alpha4_std": float(val_metrics["alpha4_std"]),
            "train_grad_norm": aggregate_train_values(batch_metrics, "grad_norm"),
            "sam_second_grad_norm": aggregate_train_values(batch_metrics, "sam_second_grad_norm"),
            "best_val_accuracy": best_acc,
            "best_val_macro_f1": best_macro,
            "best_epoch": best_epoch,
            "phase": "full" if train_backbone else "head",
            "gpu_memory_used_mb": gpu_mem,
            "improved": int(improved),
        }
        history.append(row)
        write_history_csv(history, run_dir / "training_history.csv")
        print(
            f"Epoch {epoch + 1}/{epochs} [{epoch_time:.1f}s] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro:.4f} "
            f"lr_head={lr_head:.6f} lr_backbone={lr_backbone:.6f} "
            f"alpha3(mean/min/max/std)={row['alpha3_mean']:.6f}/{row['alpha3_min']:.6f}/{row['alpha3_max']:.6f}/{row['alpha3_std']:.6f} "
            f"alpha4(mean/min/max/std)={row['alpha4_mean']:.6f}/{row['alpha4_min']:.6f}/{row['alpha4_max']:.6f}/{row['alpha4_std']:.6f} "
            f"grad_norm={row['train_grad_norm']:.6f} best_val_acc={best_acc:.4f} "
            f"gpu_mem_mb={gpu_mem} patience={patience_counter}/{patience_limit}",
            flush=True,
        )
        if patience_limit > 0 and patience_counter >= patience_limit:
            print(f"[INFO] Early stopping at epoch {epoch + 1}", flush=True)
            break

    best_ckpt = best_manager.latest_checkpoint or last_manager.latest_checkpoint
    if best_ckpt:
        checkpoint.restore(best_ckpt).expect_partial()
        print(f"[INFO] Restored best checkpoint for final test: {best_ckpt}", flush=True)

    print("[INFO] Final test evaluation: no TTA", flush=True)
    test_no_tta = evaluate_dataset(model, test_ds, cfg, use_tta_hflip=False)
    save_metrics(test_no_tta, run_dir / "test_metrics_no_tta.json")
    if bool(cfg.get("tta", {}).get("enabled", False)) and bool(cfg.get("tta", {}).get("hflip", False)):
        print("[INFO] Final test evaluation: hflip TTA", flush=True)
        test_tta = evaluate_dataset(model, test_ds, cfg, use_tta_hflip=True)
        save_metrics(test_tta, run_dir / "test_metrics_tta_hflip.json")
        save_metrics(test_tta, run_dir / "test_metrics.json")
    else:
        save_metrics(test_no_tta, run_dir / "test_metrics.json")

    print("=" * 72, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print(f"  Best epoch: {best_epoch}", flush=True)
    print(f"  Best val accuracy: {best_acc:.4f}", flush=True)
    print(f"  Best val macro-F1: {best_macro:.4f}", flush=True)
    print(f"  Output: {run_dir}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
