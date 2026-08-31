"""Train ConvNeXt-B MS1M Region Gated + Global Residual Fusion for FER2013.

This script executes the Region Gated + Global Residual Fusion experiment, using a shared region dictionary (256-dim),
masked cross-attention branches for Stage 3 (14x14) and Stage 4 (7x7), region-wise gated fusion, a single Transformer
encoder block, attention pooling, and a single classifier head.

Writes strictly to outputs/tf_runs/convnext_base_ms1m_region_gated_global_residual_fusion without modifying existing runs.
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
from models.convnext_ms1m_region_gated_global_residual_fusion import ConvNeXtMS1MRegionGatedGlobalResidualFusionFER, count_params
from train import build_optimizer, configure_gpus, configure_tensorflow_runtime, resolve_phase_lrs, set_optimizer_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-B MS1M Region Gated + Global Residual Fusion on FER2013.")
    parser.add_argument("--config", type=str, default="config_convnext_base_ms1m_region_gated_global_residual_fusion.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from this experiment's latest checkpoint.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run shape/gradient smoke tests and exit.")
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


def supervised_loss(labels: tf.Tensor, outputs: Dict[str, tf.Tensor], cfg: Dict) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    main_ce = cross_entropy_loss(labels, outputs["logits"], cfg)
    aux_weight = float(cfg.get("model", {}).get("global_aux_loss_weight", 0.2))
    if aux_weight > 0.0 and "global_logits" in outputs:
        global_ce = cross_entropy_loss(labels, outputs["global_logits"], cfg)
        total = main_ce + tf.cast(aux_weight, tf.float32) * global_ce
    else:
        global_ce = tf.constant(0.0, dtype=tf.float32)
        total = main_ce
    tf.debugging.assert_all_finite(total, "NaN/Inf in supervised loss")
    return tf.cast(total, tf.float32), {
        "main_ce": tf.cast(main_ce, tf.float32),
        "global_aux_ce": tf.cast(global_ce, tf.float32),
    }


def tensor_shape_list(tensor: tf.Tensor) -> List[Optional[int]]:
    static = tensor.shape.as_list()
    dynamic = tf.shape(tensor).numpy().tolist()
    return [static[i] if static[i] is not None else int(dynamic[i]) for i in range(len(dynamic))]


def run_shape_smoke_test(model: ConvNeXtMS1MRegionGatedGlobalResidualFusionFER, sample_inputs: Dict[str, tf.Tensor]) -> Dict[str, object]:
    print("=" * 72, flush=True)
    print("SMOKE TEST: SHAPES & ARCHITECTURE CONTRACT", flush=True)
    print("=" * 72, flush=True)
    outputs = model(sample_inputs, training=False)
    expected = {
        "S3": [14, 14, 512],
        "S4": [7, 7, 1024],
        "projected_S3": [14, 14, 256],
        "projected_S4": [7, 7, 256],
        "R3": [6, 256],
        "R4": [6, 256],
        "gate": [6, 256],
        "fused_regions": [6, 256],
        "pooled_feature": [256],
        "global_feature": [256],
        "global_region_gate": [256],
        "fused_feature": [256],
        "global_logits": [7],
        "logits": [7],
    }
    observed = {}
    for key, tail in expected.items():
        shape = tensor_shape_list(outputs[key])
        observed[key] = shape
        print(f"  {key}: {shape}", flush=True)
        if shape[1:] != tail:
            raise RuntimeError(f"Shape smoke failed for {key}: got {shape}, expected [B,{','.join(map(str, tail))}]")

    total_params = count_params(model.variables)
    trainable_params = count_params(model.trainable_variables)
    backbone_params = count_params(model.backbone_variables())
    head_params = count_params(model.head_variables())
    print(f"  total_params: {total_params:,}", flush=True)
    print(f"  trainable_params: {trainable_params:,}", flush=True)
    print(f"  backbone_trainable_params: {backbone_params:,}", flush=True)
    print(f"  head_trainable_params: {head_params:,}", flush=True)
    print("SHAPE_SMOKE_OK", flush=True)
    return observed


def finite_grads_or_raise(label: str, grads: Sequence[Optional[tf.Tensor]]) -> None:
    for idx, grad in enumerate(grads):
        if grad is not None:
            tf.debugging.assert_all_finite(grad, f"NaN/Inf in {label} gradient #{idx}")


def run_gradient_smoke_test(model: ConvNeXtMS1MRegionGatedGlobalResidualFusionFER, sample_inputs: Dict[str, tf.Tensor], sample_labels: tf.Tensor, cfg: Dict) -> Dict[str, float]:
    print("=" * 72, flush=True)
    print("GRADIENT DRY-RUN SMOKE TEST", flush=True)
    print("=" * 72, flush=True)
    train_vars = model.head_variables() + model.backbone_variables()

    with tf.GradientTape() as tape:
        outputs = model(sample_inputs, training=True)
        loss, loss_parts = supervised_loss(sample_labels, outputs, cfg)
    grads = tape.gradient(loss, train_vars)
    finite_grads_or_raise("first-step", grads)
    grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
    tf.debugging.assert_all_finite(grad_norm, "NaN/Inf in first-step global grad norm")

    gate_val = outputs["gate"].numpy()
    result = {
        "loss": float(loss.numpy()),
        "grad_norm": float(grad_norm.numpy()),
        "gate_mean": float(np.mean(gate_val)),
        "gate_std": float(np.std(gate_val)),
        "gate_min": float(np.min(gate_val)),
        "gate_max": float(np.max(gate_val)),
        "global_gate_mean": float(tf.reduce_mean(outputs["global_region_gate"]).numpy()),
        "global_aux_ce": float(loss_parts["global_aux_ce"].numpy()),
    }
    for key, value in result.items():
        ensure_finite_scalar(key, value)
        print(f"  {key}: {value:.8e}", flush=True)
    print("GRADIENT_SMOKE_OK", flush=True)
    return result


def make_train_step(model: ConvNeXtMS1MRegionGatedGlobalResidualFusionFER, cfg: Dict, optimizer_head, optimizer_backbone, train_backbone: bool):
    head_vars = model.head_variables()
    backbone_vars = model.backbone_variables() if train_backbone else []
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))

    def _compute_unscaled_grads(tape, scaled_head_loss, scaled_backbone_loss):
        scaled_grads_head = tape.gradient(scaled_head_loss, head_vars)
        grads_head = maybe_unscaled_gradients(optimizer_head, scaled_grads_head)
        if backbone_vars:
            scaled_grads_backbone = tape.gradient(scaled_backbone_loss, backbone_vars)
            grads_backbone = maybe_unscaled_gradients(optimizer_backbone, scaled_grads_backbone)
        else:
            grads_backbone = []
        return grads_head, grads_backbone

    def _clip_all(grads_head, grads_backbone):
        valid = [(g, v) for g, v in zip(grads_head, head_vars) if g is not None] + \
                [(g, v) for g, v in zip(grads_backbone, backbone_vars) if g is not None]
        if not valid:
            raise RuntimeError("No valid gradients in train_step.")
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

    use_sam = str(cfg.get("training", {}).get("optimizer", "")).lower() == "sam"
    sam_rho = float(cfg.get("training", {}).get("sam_rho", 0.03))

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            loss, loss_parts = supervised_loss(labels, outputs, cfg)
            scaled_head_loss = maybe_scaled_loss(optimizer_head, loss)
            scaled_backbone_loss = maybe_scaled_loss(optimizer_backbone, loss) if backbone_vars else None

        grads_head, grads_backbone = _compute_unscaled_grads(tape, scaled_head_loss, scaled_backbone_loss)
        del tape
        clipped_head_pairs, clipped_backbone_pairs, grad_norm = _clip_all(grads_head, grads_backbone)

        if use_sam:
            eps_list = []
            all_pairs = clipped_head_pairs + clipped_backbone_pairs
            for grad, var in all_pairs:
                scale = sam_rho / (grad_norm + 1e-12)
                eps = tf.cast(grad, var.dtype) * scale
                eps_list.append((eps, var))
                var.assign_add(eps)

            with tf.GradientTape(persistent=True) as tape2:
                outputs2 = model(inputs, training=True)
                loss2 = cross_entropy_loss(labels, outputs2["logits"], cfg)
                scaled_head_loss2 = maybe_scaled_loss(optimizer_head, loss2)
                scaled_backbone_loss2 = maybe_scaled_loss(optimizer_backbone, loss2) if backbone_vars else None

            grads_head2, grads_backbone2 = _compute_unscaled_grads(tape2, scaled_head_loss2, scaled_backbone_loss2)
            del tape2

            for eps, var in eps_list:
                var.assign_sub(eps)

            clipped_head_pairs, clipped_backbone_pairs, grad_norm = _clip_all(grads_head2, grads_backbone2)

        if clipped_head_pairs:
            optimizer_head.apply_gradients(clipped_head_pairs)
        if clipped_backbone_pairs:
            optimizer_backbone.apply_gradients(clipped_backbone_pairs)

        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        labels_i32 = tf.cast(labels, tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels_i32), tf.int32))
        count = tf.shape(labels_i32)[0]

        gate = outputs["gate"]
        return {
            "loss": tf.cast(loss, tf.float32),
            "correct": correct,
            "count": count,
            "grad_norm": tf.cast(grad_norm, tf.float32),
            "gate_mean": tf.cast(tf.reduce_mean(gate), tf.float32),
            "gate_min": tf.cast(tf.reduce_min(gate), tf.float32),
            "gate_max": tf.cast(tf.reduce_max(gate), tf.float32),
            "gate_std": tf.cast(tf.math.reduce_std(gate), tf.float32),
            "global_gate_mean": tf.cast(tf.reduce_mean(outputs["global_region_gate"]), tf.float32),
            "main_ce": tf.cast(loss_parts["main_ce"], tf.float32),
            "global_aux_ce": tf.cast(loss_parts["global_aux_ce"], tf.float32),
        }

    return train_step


def evaluate_dataset(model: ConvNeXtMS1MRegionGatedGlobalResidualFusionFER, dataset: tf.data.Dataset, cfg: Dict, use_tta_hflip: bool = False) -> Dict[str, object]:
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_count = 0
    gate_means: List[float] = []

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        logits = outputs["logits"]
        if use_tta_hflip:
            flipped = dict(inputs)
            flipped["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs:
                flipped["mask"] = tf.image.flip_left_right(inputs["mask"])
            outputs_flip = model(flipped, training=False)
            logits = 0.5 * logits + 0.5 * outputs_flip["logits"]

        loss = cross_entropy_loss(labels, logits, cfg)
        count = int(tf.shape(labels)[0])
        total_loss += float(loss.numpy()) * count
        total_count += count
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(tf.cast(labels, tf.int32).numpy().tolist())

        gate_means.append(float(tf.reduce_mean(outputs["gate"]).numpy()))

    metrics = classification_metrics(y_true, y_pred, EMOTION_NAMES)
    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["ce_loss"] = metrics["loss"]
    metrics["tta_hflip"] = bool(use_tta_hflip)
    metrics["gate_mean"] = float(np.mean(gate_means)) if gate_means else 0.5
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


def configure_gpus_safe(cfg: Dict) -> None:
    gpu_ids = cfg.get("runtime", {}).get("gpu_ids", [0])
    if os.environ.get("MGR_GPU_IDS") == "-1" or os.environ.get("CUDA_VISIBLE_DEVICES") == "" or gpu_ids == [-1]:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("[INFO] CPU mode forced by configuration (MGR_GPU_IDS=-1 or CUDA_VISIBLE_DEVICES='').", flush=True)
        return
    try:
        configure_gpus(cfg)
    except Exception as e:
        print(f"[WARNING] GPU configuration failed ({e}). Falling back to CPU mode.", flush=True)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]))
    run_dir = Path(cfg["paths"]["output_dir"])
    checkpoint_root = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    configure_gpus_safe(cfg)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"].get("random_seed", 42)))
    print(f"TensorFlow {tf.__version__}", flush=True)
    print(f"Global batch size: {global_batch_size(cfg, replicas=1)}", flush=True)

    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
    first_inputs, first_labels = next(iter(train_ds.take(1)))

    with tf.device("/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"):
        model = ConvNeXtMS1MRegionGatedGlobalResidualFusionFER(cfg)
        _ = model(first_inputs, training=False)

        total_params = count_params(model.variables)
        trainable_params = count_params(model.trainable_variables)
        backbone_params = count_params(model.backbone_variables())
        head_params = count_params(model.head_variables())

        print("=" * 72, flush=True)
        print("PARAMETER SUMMARY", flush=True)
        print(f"  total_params: {total_params:,}", flush=True)
        print(f"  trainable_params: {trainable_params:,}", flush=True)
        print(f"  backbone_trainable_params: {backbone_params:,}", flush=True)
        print(f"  head_trainable_params: {head_params:,}", flush=True)
        print("=" * 72, flush=True)

        if not args.skip_smoke_tests:
            shape_result = run_shape_smoke_test(model, first_inputs)
            grad_result = run_gradient_smoke_test(model, first_inputs, first_labels, cfg)
            smoke_payload = {
                "shape": shape_result,
                "gradient": grad_result,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "backbone_trainable_params": backbone_params,
                "head_trainable_params": head_params,
            }
            save_metrics(smoke_payload, run_dir / "smoke_tests.json")
            if args.smoke_test_only:
                print("[INFO] --smoke-test-only requested. Stop before training.", flush=True)
                return 0

        optimizer_head = build_optimizer(cfg, float(cfg["training"]["lr"]))
        optimizer_backbone = build_optimizer(cfg, float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])))

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
    freeze_epochs = int(cfg["model"].get("freeze_backbone_epochs", 4) or 4)
    unfreeze_backbone = bool(cfg["model"].get("unfreeze_backbone", True))
    patience_limit = int(cfg["training"].get("patience", 15))
    start_epoch = int(ckpt_epoch.numpy())
    best_acc = float(ckpt_best_acc.numpy())
    best_macro = float(ckpt_best_macro.numpy())
    best_epoch = start_epoch if best_acc >= 0.0 else -1
    history: List[Dict[str, object]] = []

    train_step_head = make_train_step(model, cfg, optimizer_head, optimizer_backbone, train_backbone=False)
    train_step_full = make_train_step(model, cfg, optimizer_head, optimizer_backbone, train_backbone=True)

    print("=" * 72, flush=True)
    print("STARTING TRAINING: ConvNeXt-B MS1M Region Gated + Global Residual Fusion", flush=True)
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
            ensure_finite_scalar("train_loss", loss_v)
            ensure_finite_scalar("grad_norm", grad_norm_v)

            losses.append(loss_v)
            correct += int(metrics["correct"].numpy())
            seen += int(metrics["count"].numpy())
            batch_record = {k: float(metrics[k].numpy()) for k in (
                "grad_norm", "gate_mean", "gate_std", "gate_min", "gate_max",
                "global_gate_mean", "main_ce", "global_aux_ce",
            )}
            batch_metrics.append(batch_record)

            if progress_interval and step % progress_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                    f"loss={loss_v:.4f} acc={correct / max(seen, 1):.4f} "
                    f"grad_norm={grad_norm_v:.6f} gate_mean={batch_record['gate_mean']:.4f}",
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
            "train_grad_norm": aggregate_train_values(batch_metrics, "grad_norm"),
            "train_main_ce": aggregate_train_values(batch_metrics, "main_ce"),
            "train_global_aux_ce": aggregate_train_values(batch_metrics, "global_aux_ce"),
            "gate_mean": aggregate_train_values(batch_metrics, "gate_mean"),
            "global_gate_mean": aggregate_train_values(batch_metrics, "global_gate_mean"),
            "gate_std": aggregate_train_values(batch_metrics, "gate_std"),
            "gate_min": aggregate_train_values(batch_metrics, "gate_min", mode="min"),
            "gate_max": aggregate_train_values(batch_metrics, "gate_max", mode="max"),
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
            f"gate_mean={row['gate_mean']:.4f} global_gate={row['global_gate_mean']:.4f} "
            f"best_val_acc={best_acc:.4f}",
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
    print(f"[INFO] Test (no TTA): accuracy={test_no_tta['accuracy']:.4f}, macro_f1={test_no_tta['macro_f1']:.4f}, loss={test_no_tta['loss']:.4f}", flush=True)
    save_metrics(test_no_tta, run_dir / "test_metrics_no_tta.json")
    save_metrics(test_no_tta, run_dir / "test_metrics.json")

    print("=" * 72, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print(f"  Best epoch: {best_epoch}", flush=True)
    print(f"  Best val accuracy: {best_acc:.4f}", flush=True)
    print(f"  Best val macro-F1: {best_macro:.4f}", flush=True)
    print(f"  Final test accuracy: {test_no_tta['accuracy']:.4f}", flush=True)
    print(f"  Final test macro-F1: {test_no_tta['macro_f1']:.4f}", flush=True)
    print(f"  Output: {run_dir}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
