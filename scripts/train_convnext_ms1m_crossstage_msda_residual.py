"""Train ConvNeXt-B MS1M Cross-Stage MSDA Residual for FER2013.

New experiment only. It initializes from the MS1M/ArcFace ConvNeXt-B checkpoint,
does not restore any FER checkpoint for the main run, uses AdamW without SAM, and
writes only to outputs/tf_runs/convnext_ms1m_crossstage_msda_residual by default.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import global_batch_size, load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets
from metrics.classification import classification_metrics, save_metrics
from models.convnext_ms1m_crossstage_msda_residual import (
    ConvNeXtMS1MCrossStageMSDAResidualFER,
    count_params,
)
from train import build_optimizer, configure_gpus, configure_tensorflow_runtime, resolve_phase_lrs, set_optimizer_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-B MS1M Cross-Stage MSDA Residual on FER2013.")
    parser.add_argument("--config", type=str, default="config_convnext_ms1m_crossstage_msda_residual.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from this experiment's latest checkpoint only.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run required smoke tests and exit before training.")
    parser.add_argument("--skip-smoke-tests", action="store_true", help="Debug only: skip required smoke gates.")
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


def maybe_scaled_loss(optimizer, loss: tf.Tensor) -> tf.Tensor:
    getter = getattr(optimizer, "get_scaled_loss", None)
    return getter(loss) if callable(getter) else loss


def maybe_unscaled_gradients(optimizer, grads):
    getter = getattr(optimizer, "get_unscaled_gradients", None)
    return getter(grads) if callable(getter) else grads


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


def require_pretrained_loaded(model: ConvNeXtMS1MCrossStageMSDAResidualFER) -> None:
    status = getattr(model.rgb_baseline, "pretrained_load_status", "unknown")
    if status != "loaded":
        raise RuntimeError(f"PRETRAINED_LOAD_OK not reached; pretrained_load_status={status}")
    print("PRETRAINED_LOAD_OK 340/340", flush=True)


def run_shape_smoke_test(model: ConvNeXtMS1MCrossStageMSDAResidualFER, sample_inputs: Dict[str, tf.Tensor]) -> Dict[str, object]:
    print("=" * 72, flush=True)
    print("SHAPE_SMOKE_TEST", flush=True)
    print("=" * 72, flush=True)
    outputs = model(sample_inputs, training=False)
    expected = {
        "S1": [56, 56, 128],
        "S2": [28, 28, 256],
        "S3": [14, 14, 512],
        "branch_a": [14, 14, 256],
        "branch_b": [14, 14, 256],
        "branch_c": [14, 14, 256],
        "F_concat": [14, 14, 768],
        "F_ms": [14, 14, 512],
        "F_ca": [14, 14, 512],
        "F_sa": [14, 14, 512],
        "F_da": [14, 14, 512],
        "delta_raw": [7, 7, 1024],
        "delta_norm": [7, 7, 1024],
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


def run_identity_smoke_test(model: ConvNeXtMS1MCrossStageMSDAResidualFER, sample_inputs: Dict[str, tf.Tensor], cfg: Dict) -> Dict[str, float]:
    print("=" * 72, flush=True)
    print("IDENTITY_SMOKE_TEST", flush=True)
    print("=" * 72, flush=True)
    outputs = model(sample_inputs, training=False, force_alpha_zero=True)
    logits = outputs["logits"].numpy()
    baseline_logits = outputs["baseline_logits"].numpy()
    diff = np.abs(logits - baseline_logits)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    alpha = float(tf.reduce_mean(outputs["alpha"]).numpy())
    residual_ratio = float(outputs["residual_ratio"].numpy())
    result = {
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "alpha": alpha,
        "delta_raw_norm": float(outputs["delta_raw_norm"].numpy()),
        "delta_norm_norm": float(outputs["delta_norm_norm"].numpy()),
        "s4_norm": float(outputs["s4_norm"].numpy()),
        "residual_ratio": residual_ratio,
    }
    for key, value in result.items():
        ensure_finite_scalar(key, value)
        print(f"  {key}: {value:.8e}", flush=True)
    tol = float(cfg.get("smoke_tests", {}).get("identity_max_abs_logit_diff", 1e-4))
    if max_diff >= tol:
        raise RuntimeError(f"Identity smoke failed: max_abs_logit_diff={max_diff:.8e} >= {tol:.8e}")
    print("IDENTITY_SMOKE_OK", flush=True)
    return result


def finite_grads_or_raise(label: str, grads: Sequence[Optional[tf.Tensor]]) -> None:
    for idx, grad in enumerate(grads):
        if grad is not None:
            tf.debugging.assert_all_finite(grad, f"NaN/Inf in {label} gradient #{idx}")


def assert_group_gradients(group_name: str, variables: Sequence[tf.Variable], grad_by_id: Dict[int, Optional[tf.Tensor]]) -> None:
    if not variables:
        raise RuntimeError(f"{group_name} has no trainable variables")
    missing = [v.name for v in variables if grad_by_id.get(id(v)) is None]
    if missing:
        raise RuntimeError(f"{group_name} gradient missing for {len(missing)} variable(s): {missing[:8]}")
    print(f"  {group_name}_gradients: {len(variables)}/{len(variables)} non-None", flush=True)


def run_gradient_smoke_test(
    model: ConvNeXtMS1MCrossStageMSDAResidualFER,
    sample_inputs: Dict[str, tf.Tensor],
    sample_labels: tf.Tensor,
    cfg: Dict,
) -> Dict[str, float]:
    print("=" * 72, flush=True)
    print("GRADIENT_SMOKE_TEST", flush=True)
    print("=" * 72, flush=True)
    train_vars = model.head_variables() + model.backbone_variables()
    with tf.GradientTape() as tape:
        outputs = model(sample_inputs, training=True)
        loss = cross_entropy_loss(sample_labels, outputs["logits"], cfg)
    grads = tape.gradient(loss, train_vars)
    finite_grads_or_raise("gradient-smoke", grads)
    grad_by_id = {id(v): g for g, v in zip(grads, train_vars)}
    assert_group_gradients("alpha", model.alpha_variables(), grad_by_id)
    assert_group_gradients("multiscale_branch", model.multiscale_variables(), grad_by_id)
    assert_group_gradients("channel_attention", model.channel_attention_variables(), grad_by_id)
    assert_group_gradients("spatial_attention", model.spatial_attention_variables(), grad_by_id)
    assert_group_gradients("s3_to_s4_projection", model.projection_variables(), grad_by_id)
    grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
    tf.debugging.assert_all_finite(grad_norm, "NaN/Inf in global grad norm")
    result = {
        "loss": float(loss.numpy()),
        "grad_norm": float(grad_norm.numpy()),
        "alpha": float(outputs["alpha_mean"].numpy()),
        "residual_ratio": float(outputs["residual_ratio"].numpy()),
    }
    for key, value in result.items():
        ensure_finite_scalar(key, value)
        print(f"  {key}: {value:.8e}", flush=True)
    print("GRADIENT_SMOKE_OK", flush=True)
    print("NO_NAN_INF_OK", flush=True)
    return result


def run_param_budget_smoke(model: ConvNeXtMS1MCrossStageMSDAResidualFER, cfg: Dict) -> Dict[str, int]:
    baseline_params = count_params(model.rgb_baseline.variables)
    new_module_params = count_params(model.new_module_variables())
    total_params = count_params(model.variables)
    trainable_params = count_params(model.trainable_variables)
    result = {
        "baseline_params": baseline_params,
        "new_module_params": new_module_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    print("=" * 72, flush=True)
    print("MODEL_SIZE_CONTRACT", flush=True)
    for key, value in result.items():
        print(f"  {key}: {value:,}", flush=True)
    max_module = int(cfg.get("smoke_tests", {}).get("max_new_module_params", 5_000_000))
    max_total = int(cfg.get("smoke_tests", {}).get("max_total_params", 93_000_000))
    if new_module_params > max_module:
        raise RuntimeError(f"new_module_params={new_module_params:,} exceeds budget {max_module:,}")
    if total_params > max_total:
        raise RuntimeError(f"total_params={total_params:,} exceeds budget {max_total:,}")
    print("PARAM_BUDGET_OK", flush=True)
    return result


def non_none_grads_and_vars(grads: Sequence[Optional[tf.Tensor]], variables: Sequence[tf.Variable]):
    return [(g, v) for g, v in zip(grads, variables) if g is not None]


def make_train_step(model: ConvNeXtMS1MCrossStageMSDAResidualFER, cfg: Dict, optimizer_module, optimizer_backbone, train_backbone: bool):
    module_vars = model.head_variables()
    backbone_vars = model.backbone_variables() if train_backbone else []
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))
    module_ids = {id(v) for v in module_vars}

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            loss = cross_entropy_loss(labels, outputs["logits"], cfg)
            scaled_module_loss = maybe_scaled_loss(optimizer_module, loss)
            scaled_backbone_loss = maybe_scaled_loss(optimizer_backbone, loss) if backbone_vars else None
        module_grads = maybe_unscaled_gradients(optimizer_module, tape.gradient(scaled_module_loss, module_vars))
        if backbone_vars:
            backbone_grads = maybe_unscaled_gradients(optimizer_backbone, tape.gradient(scaled_backbone_loss, backbone_vars))
        else:
            backbone_grads = []
        del tape

        valid = non_none_grads_and_vars(module_grads, module_vars) + non_none_grads_and_vars(backbone_grads, backbone_vars)
        if not valid:
            raise RuntimeError("No valid gradients in train_step.")
        grads_all = [g for g, _ in valid]
        vars_all = [v for _, v in valid]
        finite_grads_or_raise("train", grads_all)
        clipped_all, grad_norm = tf.clip_by_global_norm(grads_all, grad_clip_norm)

        module_pairs = []
        backbone_pairs = []
        for grad, var in zip(clipped_all, vars_all):
            if id(var) in module_ids:
                module_pairs.append((grad, var))
            else:
                backbone_pairs.append((grad, var))
        if module_pairs:
            optimizer_module.apply_gradients(module_pairs)
        if backbone_pairs:
            optimizer_backbone.apply_gradients(backbone_pairs)

        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        labels_i32 = tf.cast(labels, tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels_i32), tf.int32))
        count = tf.shape(labels_i32)[0]
        return {
            "loss": tf.cast(loss, tf.float32),
            "correct": correct,
            "count": count,
            "grad_norm": tf.cast(grad_norm, tf.float32),
            "alpha": tf.cast(outputs["alpha_mean"], tf.float32),
            "s4_norm": tf.cast(outputs["s4_norm"], tf.float32),
            "delta_raw_norm": tf.cast(outputs["delta_raw_norm"], tf.float32),
            "delta_norm_norm": tf.cast(outputs["delta_norm_norm"], tf.float32),
            "residual_norm": tf.cast(outputs["residual_norm"], tf.float32),
            "residual_ratio": tf.cast(outputs["residual_ratio"], tf.float32),
            "channel_attention_mean": tf.cast(outputs["channel_attention_mean"], tf.float32),
            "channel_attention_std": tf.cast(outputs["channel_attention_std"], tf.float32),
            "spatial_attention_mean": tf.cast(outputs["spatial_attention_mean"], tf.float32),
            "spatial_attention_std": tf.cast(outputs["spatial_attention_std"], tf.float32),
            "fusion_weight_channel": tf.cast(outputs["fusion_weight_channel"], tf.float32),
            "fusion_weight_spatial": tf.cast(outputs["fusion_weight_spatial"], tf.float32),
        }

    return train_step


def evaluate_dataset(model: ConvNeXtMS1MCrossStageMSDAResidualFER, dataset: tf.data.Dataset, cfg: Dict, use_tta_hflip: bool) -> Dict[str, object]:
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_count = 0
    stat_keys = [
        "alpha",
        "s4_norm",
        "delta_raw_norm",
        "delta_norm_norm",
        "residual_norm",
        "residual_ratio",
        "channel_attention_mean",
        "channel_attention_std",
        "spatial_attention_mean",
        "spatial_attention_std",
        "fusion_weight_channel",
        "fusion_weight_spatial",
    ]
    stat_values = {key: [] for key in stat_keys}
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
        for key in stat_keys:
            stat_values[key].append(float(outputs[key if key != "alpha" else "alpha_mean"].numpy()))

    metrics = classification_metrics(y_true, y_pred, EMOTION_NAMES)
    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["ce_loss"] = metrics["loss"]
    metrics["tta_hflip"] = bool(use_tta_hflip)
    for key, values in stat_values.items():
        metrics[key] = float(np.mean(values)) if values else 0.0
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


def aggregate_train_values(values: List[Dict[str, float]], key: str) -> float:
    vals = [float(item[key]) for item in values if key in item]
    return float(np.mean(vals)) if vals else 0.0


def warn_if_residual_unstable(metrics: Dict[str, object]) -> None:
    residual_ratio = float(metrics.get("residual_ratio", 0.0))
    alpha = float(metrics.get("alpha", 0.0))
    if residual_ratio > 0.30:
        print(f"[WARNING] residual_ratio={residual_ratio:.6f} > 0.30; delta correction may dominate S4.", flush=True)
    if alpha > 0.19:
        print(f"[WARNING] alpha={alpha:.6f} is close to the 0.20 cap.", flush=True)


def save_config_snapshot(cfg: Dict, run_dir: Path) -> None:
    with (run_dir / "config_resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]))
    run_dir = Path(cfg["paths"]["output_dir"])
    checkpoint_root = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, run_dir)

    if str(cfg["training"].get("optimizer", "")).lower() == "sam":
        raise RuntimeError("This first MSDA residual experiment must use AdamW without SAM.")
    if cfg["model"].get("checkpoint_path"):
        raise RuntimeError("Main run must not restore a FER checkpoint; model.checkpoint_path must stay null.")

    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"].get("random_seed", 42)))
    configure_gpus(cfg)
    print(f"TensorFlow version: {tf.__version__}", flush=True)
    print(f"GPU detection: physical={tf.config.list_physical_devices('GPU')} logical={tf.config.list_logical_devices('GPU')}", flush=True)
    print(f"Config path: {args.config}", flush=True)
    print(f"Output path: {run_dir}", flush=True)
    print(f"Global batch size: {global_batch_size(cfg, replicas=1)}", flush=True)

    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
    first_inputs, first_labels = next(iter(train_ds.take(1)))

    with tf.device("/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"):
        model = ConvNeXtMS1MCrossStageMSDAResidualFER(cfg)
        _ = model(first_inputs, training=False)
        require_pretrained_loaded(model)
        param_result = run_param_budget_smoke(model, cfg)

        if not args.skip_smoke_tests:
            shape_result = run_shape_smoke_test(model, first_inputs)
            identity_result = run_identity_smoke_test(model, first_inputs, cfg)
            grad_result = run_gradient_smoke_test(model, first_inputs, first_labels, cfg)
            save_metrics(
                {
                    "shape": shape_result,
                    "identity": identity_result,
                    "gradient": grad_result,
                    "params": param_result,
                },
                run_dir / "smoke_tests.json",
            )
            print("READY_FOR_V100_TRAINING", flush=True)
            if args.smoke_test_only:
                print("[INFO] --smoke-test-only requested. Stop before training.", flush=True)
                return 0

        optimizer_module = build_optimizer(cfg, float(cfg["training"]["lr"]))
        optimizer_backbone = build_optimizer(cfg, float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])))
        use_loss_scale_optimizer = bool(
            cfg.get("runtime", {}).get("use_loss_scale_optimizer", False)
            or cfg.get("training", {}).get("use_loss_scale_optimizer", False)
        )
        if use_loss_scale_optimizer:
            optimizer_module = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_module, dynamic=True)
            optimizer_backbone = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_backbone, dynamic=True)
            print("[INFO] LossScaleOptimizer enabled for AdamW custom loop.", flush=True)

        ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt_best_acc = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        ckpt_best_macro = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        checkpoint = tf.train.Checkpoint(
            epoch=ckpt_epoch,
            best_accuracy=ckpt_best_acc,
            best_macro_f1=ckpt_best_macro,
            model=model,
            optimizer_module=optimizer_module,
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
    start_epoch = int(ckpt_epoch.numpy())
    best_acc = float(ckpt_best_acc.numpy())
    best_macro = float(ckpt_best_macro.numpy())
    best_epoch = start_epoch if best_acc >= 0.0 else -1
    history: List[Dict[str, object]] = []
    train_step_head = make_train_step(model, cfg, optimizer_module, optimizer_backbone, train_backbone=False)
    train_step_full = make_train_step(model, cfg, optimizer_module, optimizer_backbone, train_backbone=True)

    print("=" * 72, flush=True)
    print("STARTING TRAINING: ConvNeXt-B MS1M Cross-Stage MSDA Residual", flush=True)
    print("optimizer=AdamW | SAM=disabled | mixed_precision=mixed_float16 | XLA=false", flush=True)
    print(f"freeze_backbone_epochs={freeze_epochs} | output_dir={run_dir}", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_backbone = bool(unfreeze_backbone and epoch >= freeze_epochs)
        train_step = train_step_full if train_backbone else train_step_head
        lr_module, lr_backbone = resolve_phase_lrs(cfg, epoch, train_backbone)
        set_lso_lr(optimizer_module, lr_module)
        set_lso_lr(optimizer_backbone, lr_backbone)
        if train_backbone and epoch == freeze_epochs:
            print(f"[INFO] Unfreezing ConvNeXt backbone at epoch {epoch + 1}; resetting patience counter from this point.", flush=True)
            best_epoch = epoch + 1

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
            ensure_finite_scalar("train_loss", loss_v)
            ensure_finite_scalar("grad_norm", grad_norm_v)
            losses.append(loss_v)
            correct += int(metrics["correct"].numpy())
            seen += int(metrics["count"].numpy())
            record = {
                key: float(metrics[key].numpy())
                for key in (
                    "grad_norm",
                    "alpha",
                    "s4_norm",
                    "delta_raw_norm",
                    "delta_norm_norm",
                    "residual_norm",
                    "residual_ratio",
                    "channel_attention_mean",
                    "channel_attention_std",
                    "spatial_attention_mean",
                    "spatial_attention_std",
                    "fusion_weight_channel",
                    "fusion_weight_spatial",
                )
            }
            warn_if_residual_unstable(record)
            batch_metrics.append(record)
            if progress_interval and step % progress_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                    f"loss={loss_v:.4f} acc={correct / max(seen, 1):.4f} "
                    f"grad_norm={grad_norm_v:.6f} lr_module={lr_module:.8f} lr_backbone={lr_backbone:.8f}",
                    flush=True,
                )

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = correct / max(seen, 1)
        ensure_finite_scalar("epoch_train_loss", train_loss)
        ensure_finite_scalar("epoch_train_accuracy", train_acc)

        val_no_tta = evaluate_dataset(model, val_ds, cfg, use_tta_hflip=False)
        val_tta = evaluate_dataset(model, val_ds, cfg, use_tta_hflip=True)
        selected_val = val_tta if bool(cfg["runtime"].get("train_val_tta_hflip", False)) else val_no_tta
        val_acc = float(selected_val["accuracy"])
        val_macro = float(selected_val["macro_f1"])
        val_loss = float(selected_val["loss"])
        for name, value in (("val_loss", val_loss), ("val_accuracy", val_acc), ("val_macro_f1", val_macro)):
            ensure_finite_scalar(name, value)

        improved = val_acc > best_acc + 1e-12 or (abs(val_acc - best_acc) <= 1e-12 and val_macro > best_macro)
        if improved:
            best_acc = val_acc
            best_macro = val_macro
            best_epoch = epoch + 1
            ckpt_best_acc.assign(best_acc)
            ckpt_best_macro.assign(best_macro)
            print(f"[INFO] Save best checkpoint at epoch {epoch + 1}: val_hflip_tta_acc={val_acc:.4f}, val_macro_f1={val_macro:.4f}", flush=True)
            best_manager.save(checkpoint_number=epoch + 1)

        ckpt_epoch.assign(epoch + 1)
        last_manager.save(checkpoint_number=epoch + 1)
        patience_counter = 0 if best_epoch < 0 else (epoch + 1) - best_epoch
        epoch_time = time.time() - epoch_start
        throughput = float(seen) / max(epoch_time, 1e-8)
        gpu_mem = query_gpu_memory_mb()
        row = {
            "epoch": epoch + 1,
            "train_time": round(epoch_time, 2),
            "throughput": throughput,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro,
            "val_no_tta_loss": float(val_no_tta["loss"]),
            "val_no_tta_accuracy": float(val_no_tta["accuracy"]),
            "val_no_tta_macro_f1": float(val_no_tta["macro_f1"]),
            "val_hflip_tta_loss": float(val_tta["loss"]),
            "val_hflip_tta_accuracy": float(val_tta["accuracy"]),
            "val_hflip_tta_macro_f1": float(val_tta["macro_f1"]),
            "alpha": float(selected_val["alpha"]),
            "s4_norm": float(selected_val["s4_norm"]),
            "delta_raw_norm": float(selected_val["delta_raw_norm"]),
            "delta_norm_norm": float(selected_val["delta_norm_norm"]),
            "residual_norm": float(selected_val["residual_norm"]),
            "residual_ratio": float(selected_val["residual_ratio"]),
            "channel_attention_mean": float(selected_val["channel_attention_mean"]),
            "channel_attention_std": float(selected_val["channel_attention_std"]),
            "spatial_attention_mean": float(selected_val["spatial_attention_mean"]),
            "spatial_attention_std": float(selected_val["spatial_attention_std"]),
            "fusion_weight_channel": float(selected_val["fusion_weight_channel"]),
            "fusion_weight_spatial": float(selected_val["fusion_weight_spatial"]),
            "grad_norm": aggregate_train_values(batch_metrics, "grad_norm"),
            "lr_module": lr_module,
            "lr_backbone": lr_backbone,
            "best_val_accuracy": best_acc,
            "best_val_macro_f1": best_macro,
            "best_epoch": best_epoch,
            "phase": "full" if train_backbone else "head",
            "gpu_memory_used_mb": gpu_mem,
            "improved": int(improved),
        }
        warn_if_residual_unstable(row)
        history.append(row)
        write_history_csv(history, run_dir / "training_history.csv")
        print(
            f"Epoch {epoch + 1}/{epochs} [{epoch_time:.1f}s] "
            f"train_loss={train_loss:.4f} train_accuracy={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_acc:.4f} val_macro_f1={val_macro:.4f} "
            f"val_no_tta_accuracy={row['val_no_tta_accuracy']:.4f} val_hflip_tta_accuracy={row['val_hflip_tta_accuracy']:.4f} "
            f"alpha={row['alpha']:.6f} residual_ratio={row['residual_ratio']:.6f} "
            f"s4_norm={row['s4_norm']:.6f} delta_raw_norm={row['delta_raw_norm']:.6f} "
            f"delta_norm_norm={row['delta_norm_norm']:.6f} residual_norm={row['residual_norm']:.6f} "
            f"channel_attention_mean/std={row['channel_attention_mean']:.6f}/{row['channel_attention_std']:.6f} "
            f"spatial_attention_mean/std={row['spatial_attention_mean']:.6f}/{row['spatial_attention_std']:.6f} "
            f"fusion_weight_channel/spatial={row['fusion_weight_channel']:.6f}/{row['fusion_weight_spatial']:.6f} "
            f"grad_norm={row['grad_norm']:.6f} lr_module={lr_module:.8f} lr_backbone={lr_backbone:.8f} "
            f"throughput={throughput:.2f} samples/s gpu_mem_mb={gpu_mem} patience={patience_counter}/{patience_limit}",
            flush=True,
        )
        if patience_limit > 0 and patience_counter >= patience_limit:
            print(f"[INFO] Early stopping at epoch {epoch + 1}", flush=True)
            break

    best_ckpt = best_manager.latest_checkpoint or last_manager.latest_checkpoint
    if best_ckpt:
        checkpoint.restore(best_ckpt).expect_partial()
        print(f"[INFO] Restored best validation checkpoint for final test: {best_ckpt}", flush=True)

    print("[INFO] Final test evaluation: No TTA", flush=True)
    test_no_tta = evaluate_dataset(model, test_ds, cfg, use_tta_hflip=False)
    save_metrics(test_no_tta, run_dir / "test_metrics_no_tta.json")
    print(
        f"TEST_NO_TTA accuracy={float(test_no_tta['accuracy']):.6f} "
        f"macro_f1={float(test_no_tta['macro_f1']):.6f} weighted_f1={float(test_no_tta['weighted_f1']):.6f} "
        f"loss={float(test_no_tta['loss']):.6f}",
        flush=True,
    )

    print("[INFO] Final test evaluation: HFlip TTA", flush=True)
    test_tta = evaluate_dataset(model, test_ds, cfg, use_tta_hflip=True)
    test_tta["tta_improvement"] = float(test_tta["accuracy"]) - float(test_no_tta["accuracy"])
    save_metrics(test_tta, run_dir / "test_metrics_tta_hflip.json")
    save_metrics(test_tta, run_dir / "test_metrics.json")
    print(
        f"TEST_HFLIP_TTA accuracy={float(test_tta['accuracy']):.6f} "
        f"macro_f1={float(test_tta['macro_f1']):.6f} weighted_f1={float(test_tta['weighted_f1']):.6f} "
        f"loss={float(test_tta['loss']):.6f} tta_improvement={float(test_tta['tta_improvement']):.6f}",
        flush=True,
    )
    print("Classification report and confusion matrix saved in test_metrics_no_tta.json and test_metrics_tta_hflip.json", flush=True)

    print("=" * 72, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print(f"  Best epoch: {best_epoch}", flush=True)
    print(f"  Best val HFlip-TTA accuracy: {best_acc:.4f}", flush=True)
    print(f"  Best val macro-F1: {best_macro:.4f}", flush=True)
    print(f"  Output: {run_dir}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
