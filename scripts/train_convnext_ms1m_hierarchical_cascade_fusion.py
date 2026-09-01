"""Training Script for ConvNeXt-B MS1M Hierarchical Cascade Region Gated Fusion.

Features:
- Standalone execution without modifying existing baseline scripts.
- SAM & AdamW optimization support with dual learning rate scheduling.
- Mixed precision support with float32 softmax stability.
- Per-epoch diagnostic logging (loss, accuracy, macro-F1, LRs, grad norm, gate stats, gamma_cascade value).
- Test evaluation with Horizontal Flip TTA at the end of training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, f1_score

def save_plots_and_history(history_logs: List[Dict], output_dir: Path):
    if not history_logs:
        return
    df = pd.DataFrame(history_logs)
    csv_path = output_dir / "history.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue", linewidth=2)
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange", linewidth=2)
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy
    axes[0, 1].plot(df["epoch"], df["train_acc"], label="Train Acc", color="blue", linewidth=2)
    axes[0, 1].plot(df["epoch"], df["val_acc"], label="Val Acc", color="green", linewidth=2)
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Gate Mean & Gamma Cascade
    axes[1, 0].plot(df["epoch"], df["gate_mean"], label="Gate Mean (S3/S4)", color="purple", linewidth=2)
    axes[1, 0].plot(df["epoch"], df["gamma_cascade"], label="Gamma Cascade (S3->S4)", color="red", linewidth=2)
    axes[1, 0].set_title("Fusion Parameters Evolution")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Gradient Norm
    axes[1, 1].plot(df["epoch"], df["grad_norm"], label="Grad Norm", color="brown", linewidth=2)
    axes[1, 1].set_title("Gradient Norm")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Norm")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_XLA"] = "1"
os.environ["TF_DISABLE_XLA_COMPILATION"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import build_datasets
from models.convnext_ms1m_hierarchical_cascade_fusion import (
    ConvNeXtMS1MHierarchicalCascadeFusionFER,
    count_params,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-B MS1M Hierarchical Cascade Region Gated Fusion")
    parser.add_argument("--config", default="config_convnext_base_ms1m_hierarchical_cascade_fusion.yaml", help="Path to config file")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run 1 train/val step then exit")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_gpus_safe(cfg: Dict) -> None:
    gpu_ids = cfg.get("runtime", {}).get("gpu_ids", [0])
    if os.environ.get("MGR_GPU_IDS") == "-1" or os.environ.get("CUDA_VISIBLE_DEVICES") == "" or gpu_ids == [-1]:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("[GPU CONFIG] Using CPU mode as requested.", flush=True)
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU CONFIG] No physical GPUs found, using CPU.", flush=True)
        return

    try:
        visible = [gpus[i] for i in gpu_ids if i < len(gpus)]
        if visible:
            tf.config.set_visible_devices(visible, "GPU")
            for gpu in visible:
                tf.config.set_memory_growth(gpu, True)
            print(f"[GPU CONFIG] Configured GPUs: {[g.name for g in visible]}", flush=True)
    except Exception as e:
        print(f"[GPU CONFIG WARNING] Failed setting GPU config: {e}", flush=True)


def configure_tensorflow_runtime(cfg: Dict) -> None:
    runtime_cfg = cfg.get("runtime", {})
    if bool(runtime_cfg.get("use_mixed_precision", True)):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[RUNTIME] Mixed precision policy set to mixed_float16", flush=True)
        except Exception as e:
            print(f"[RUNTIME WARNING] Failed setting mixed precision: {e}", flush=True)


def build_cosine_schedule(base_lr: float, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_fn


def cross_entropy_loss(labels: tf.Tensor, logits: tf.Tensor, cfg: Dict) -> tf.Tensor:
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.1))
    num_classes = int(cfg.get("data", {}).get("num_classes", 7))
    one_hot = tf.one_hot(labels, depth=num_classes)
    if label_smoothing > 0.0:
        smooth_pos = 1.0 - label_smoothing
        smooth_neg = label_smoothing / float(num_classes)
        one_hot = one_hot * smooth_pos + smooth_neg
    loss = tf.keras.losses.categorical_crossentropy(one_hot, logits, from_logits=True)
    return tf.reduce_mean(loss)


def maybe_scaled_loss(optimizer, loss: tf.Tensor) -> tf.Tensor:
    if hasattr(optimizer, "get_scaled_loss"):
        return optimizer.get_scaled_loss(loss)
    return loss


def all_grads_finite(grads: List[tf.Tensor]) -> tf.Tensor:
    checks = [tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None]
    if not checks:
        return tf.constant(True)
    return tf.reduce_all(tf.stack(checks))


def sanitize_grads(grads: List[tf.Tensor]) -> List[tf.Tensor]:
    return [
        None if g is None else tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        for g in grads
    ]


def maybe_wrap_loss_scale_optimizer(
    optimizer: tf.keras.optimizers.Optimizer,
    cfg: Dict,
    name: str,
) -> tf.keras.optimizers.Optimizer:
    training_cfg = cfg.get("training", {})
    use_loss_scale = bool(training_cfg.get("use_loss_scale_optimizer", False))
    if not use_loss_scale:
        print(f"[RUNTIME] LossScaleOptimizer disabled for {name}", flush=True)
        return optimizer

    initial_scale = float(training_cfg.get("loss_scale_initial_scale", 4096.0))
    dynamic_growth_steps = int(training_cfg.get("loss_scale_dynamic_growth_steps", 2000))
    print(
        f"[RUNTIME] LossScaleOptimizer enabled for {name} "
        f"(initial_scale={initial_scale:g}, dynamic_growth_steps={dynamic_growth_steps})",
        flush=True,
    )
    try:
        return tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer,
            dynamic=True,
            initial_scale=initial_scale,
            dynamic_growth_steps=dynamic_growth_steps,
        )
    except TypeError:
        return tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


def make_train_step(
    model: ConvNeXtMS1MHierarchicalCascadeFusionFER,
    optimizer_head: tf.keras.optimizers.Optimizer,
    optimizer_backbone: Optional[tf.keras.optimizers.Optimizer],
    cfg: Dict,
):
    grad_clip_norm = float(cfg.get("training", {}).get("grad_clip_norm", 1.0))
    skip_nonfinite_batches = bool(cfg.get("training", {}).get("skip_nonfinite_batches", True))
    head_vars = model.head_variables()
    backbone_vars = model.backbone_variables() if optimizer_backbone is not None else []
    head_var_ids = {id(h) for h in head_vars}

    def _compute_unscaled_grads(tape, scaled_head_loss, scaled_backbone_loss):
        if hasattr(optimizer_head, "get_unscaled_gradients"):
            scaled_grads_head = tape.gradient(scaled_head_loss, head_vars)
            grads_head = optimizer_head.get_unscaled_gradients(scaled_grads_head)
        else:
            grads_head = tape.gradient(scaled_head_loss, head_vars)

        if backbone_vars and optimizer_backbone is not None:
            if hasattr(optimizer_backbone, "get_unscaled_gradients"):
                scaled_grads_backbone = tape.gradient(scaled_backbone_loss, backbone_vars)
                grads_backbone = optimizer_backbone.get_unscaled_gradients(scaled_grads_backbone)
            else:
                grads_backbone = tape.gradient(scaled_backbone_loss, backbone_vars)
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
        grads_are_finite = all_grads_finite(grads_all)
        if not skip_nonfinite_batches:
            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    grads_are_finite,
                    True,
                    message="Non-finite gradient (NaN/Inf) detected in train_step",
                )
            ]):
                clipped_all, grad_norm = tf.clip_by_global_norm(grads_all, grad_clip_norm)
        else:
            safe_grads_all = sanitize_grads(grads_all)
            clipped_all, grad_norm = tf.clip_by_global_norm(safe_grads_all, grad_clip_norm)
            apply_scale = tf.cast(grads_are_finite, clipped_all[0].dtype)
            clipped_all = [g * apply_scale for g in clipped_all]

        clipped_head = []
        vars_head = []
        clipped_backbone = []
        vars_backbone = []
        for grad, var in zip(clipped_all, vars_all):
            if id(var) in head_var_ids:
                clipped_head.append(grad)
                vars_head.append(var)
            else:
                clipped_backbone.append(grad)
                vars_backbone.append(var)
        return (
            list(zip(clipped_head, vars_head)),
            list(zip(clipped_backbone, vars_backbone)),
            tf.cast(grad_norm, tf.float32),
            grads_are_finite,
        )

    use_sam = str(cfg.get("training", {}).get("optimizer", "")).lower() == "sam"
    sam_rho = float(cfg.get("training", {}).get("sam_rho", 0.03))

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            loss = cross_entropy_loss(labels, outputs["logits"], cfg)
            scaled_head_loss = maybe_scaled_loss(optimizer_head, loss)
            scaled_backbone_loss = maybe_scaled_loss(optimizer_backbone, loss) if backbone_vars else None

        grads_head, grads_backbone = _compute_unscaled_grads(tape, scaled_head_loss, scaled_backbone_loss)
        del tape
        clipped_head_pairs, clipped_backbone_pairs, grad_norm, grads_are_finite = _clip_all(grads_head, grads_backbone)

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

            clipped_head_pairs, clipped_backbone_pairs, grad_norm, grads_are_finite2 = _clip_all(grads_head2, grads_backbone2)
            grads_are_finite = tf.logical_and(grads_are_finite, grads_are_finite2)

        if clipped_head_pairs:
            optimizer_head.apply_gradients(clipped_head_pairs)
        if clipped_backbone_pairs:
            optimizer_backbone.apply_gradients(clipped_backbone_pairs)

        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        labels_i32 = tf.cast(labels, tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels_i32), tf.int32))
        count = tf.shape(labels_i32)[0]

        gate = outputs["gate"]
        gamma_cascade = outputs["gamma_cascade"]
        return {
            "loss": tf.cast(loss, tf.float32),
            "correct": correct,
            "count": count,
            "grad_norm": tf.cast(grad_norm, tf.float32),
            "gate_mean": tf.cast(tf.reduce_mean(gate), tf.float32),
            "gate_min": tf.cast(tf.reduce_min(gate), tf.float32),
            "gate_max": tf.cast(tf.reduce_max(gate), tf.float32),
            "gate_std": tf.cast(tf.math.reduce_std(gate), tf.float32),
            "gamma_cascade": tf.cast(gamma_cascade, tf.float32),
            "nonfinite_grads": tf.cast(tf.logical_not(grads_are_finite), tf.int32),
        }

    return train_step


def evaluate_dataset(model: ConvNeXtMS1MHierarchicalCascadeFusionFER, dataset: tf.data.Dataset, cfg: Dict, use_tta_hflip: bool = False) -> Dict[str, object]:
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_count = 0
    gate_means: List[float] = []

    for inputs, labels in dataset:
        if use_tta_hflip:
            flipped_inputs = dict(inputs)
            flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs:
                flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])

            out1 = model(inputs, training=False)
            out2 = model(flipped_inputs, training=False)
            logits = 0.5 * (out1["logits"] + out2["logits"])
            gate = 0.5 * (out1["gate"] + out2["gate"])
        else:
            out = model(inputs, training=False)
            logits = out["logits"]
            gate = out["gate"]

        batch_loss = cross_entropy_loss(labels, logits, cfg)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        batch_size = tf.shape(labels)[0]

        total_loss += float(batch_loss.numpy()) * int(batch_size.numpy())
        total_count += int(batch_size.numpy())

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.numpy().astype(int).tolist())
        gate_means.append(float(tf.reduce_mean(gate).numpy()))

    avg_loss = total_loss / max(1, total_count)
    y_true_arr = np.array(y_true, dtype=np.int32)
    y_pred_arr = np.array(y_pred, dtype=np.int32)
    acc = float(np.mean(y_true_arr == y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    report = classification_report(y_true_arr, y_pred_arr, digits=4, output_dict=True, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
        "classification_report": report,
    }


def build_adamw_optimizer(learning_rate: float, weight_decay: float = 0.035):
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)
    if adamw is not None:
        try:
            return adamw(learning_rate=learning_rate, weight_decay=weight_decay, jit_compile=False)
        except (TypeError, ValueError):
            return adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def set_opt_lr(opt, lr_val: float):
    if opt is None:
        return
    base = getattr(opt, "inner_optimizer", opt)
    if hasattr(base, "learning_rate"):
        base.learning_rate.assign(lr_val)
    elif hasattr(base, "lr"):
        base.lr.assign(lr_val)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_gpus_safe(cfg)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))

    print(f"\n==========================================", flush=True)
    print(f" EXPERIMENT: ConvNeXt-B MS1M Hierarchical Cascade Fusion", flush=True)
    print(f"==========================================\n", flush=True)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=1)
    steps_per_epoch = len(train_ds)
    print(f"[DATA] Train steps/epoch: {steps_per_epoch}, Val steps: {len(val_ds)}, Test steps: {len(test_ds)}", flush=True)

    model = ConvNeXtMS1MHierarchicalCascadeFusionFER(cfg)

    first_batch = next(iter(train_ds.take(1)))
    first_inputs, first_labels = first_batch
    _ = model(first_inputs, training=False)

    head_vars = model.head_variables()
    backbone_vars = model.backbone_variables()

    print(f"\n[MODEL SUMMARY]", flush=True)
    print(f"  Backbone Params:  {count_params(backbone_vars):,}", flush=True)
    print(f"  Head Params:      {count_params(head_vars):,}", flush=True)
    print(f"  Total Params:     {count_params(model.trainable_variables):,}", flush=True)
    print(f"  Pretrained Status: {model.pretrained_load_status}\n", flush=True)

    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.035))
    lr_head_base = float(cfg.get("training", {}).get("lr", 1e-4))
    lr_backbone_base = float(cfg.get("training", {}).get("visual_extractor_lr", 1e-5))
    total_epochs = int(cfg.get("training", {}).get("epochs", 60))
    warmup_epochs = int(cfg.get("training", {}).get("warmup_epochs", 5))
    freeze_epochs = int(cfg.get("model", {}).get("freeze_backbone_epochs", 4))

    optimizer_head = maybe_wrap_loss_scale_optimizer(
        build_adamw_optimizer(lr_head_base, weight_decay),
        cfg,
        "head",
    )
    optimizer_backbone = maybe_wrap_loss_scale_optimizer(
        build_adamw_optimizer(lr_backbone_base, weight_decay),
        cfg,
        "backbone",
    )

    lr_fn_head = build_cosine_schedule(lr_head_base, warmup_epochs, total_epochs, steps_per_epoch)
    lr_fn_backbone = build_cosine_schedule(lr_backbone_base, warmup_epochs, total_epochs, steps_per_epoch)

    train_step_fn = make_train_step(model, optimizer_head, optimizer_backbone, cfg)

    if args.smoke_test_only:
        print("[SMOKE TEST] Executing 1 train_step and 1 val_step...", flush=True)
        train_out = train_step_fn(first_inputs, first_labels)
        val_out = evaluate_dataset(model, val_ds.take(1), cfg)
        print(f"[SMOKE TEST SUCCESS] Train loss: {train_out['loss']:.4f}, Val loss: {val_out['loss']:.4f}, Gamma cascade: {train_out['gamma_cascade']:.4f}", flush=True)
        return 0

    best_val_acc = 0.0
    best_epoch = 0
    patience = int(cfg.get("training", {}).get("patience", 15))
    no_improve_count = 0
    history_logs = []

    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        is_backbone_frozen = (epoch <= freeze_epochs)

        train_loss_sum = 0.0
        train_correct_sum = 0
        train_count_sum = 0
        grad_norms = []
        gate_means = []
        gamma_cascades = []
        nonfinite_batches = 0

        active_opt_backbone = None if is_backbone_frozen else optimizer_backbone
        active_step_fn = make_train_step(model, optimizer_head, active_opt_backbone, cfg)

        step_idx = (epoch - 1) * steps_per_epoch
        for inputs, labels in train_ds:
            current_head_lr = lr_fn_head(step_idx)
            current_backbone_lr = 0.0 if is_backbone_frozen else lr_fn_backbone(step_idx)

            set_opt_lr(optimizer_head, current_head_lr)
            if not is_backbone_frozen:
                set_opt_lr(optimizer_backbone, current_backbone_lr)

            step_metrics = active_step_fn(inputs, labels)

            train_loss_sum += float(step_metrics["loss"].numpy()) * int(step_metrics["count"].numpy())
            train_correct_sum += int(step_metrics["correct"].numpy())
            train_count_sum += int(step_metrics["count"].numpy())

            grad_norms.append(float(step_metrics["grad_norm"].numpy()))
            gate_means.append(float(step_metrics["gate_mean"].numpy()))
            gamma_cascades.append(float(step_metrics["gamma_cascade"].numpy()))
            nonfinite_batches += int(step_metrics["nonfinite_grads"].numpy())

            step_in_epoch = ((step_idx) % steps_per_epoch) + 1
            if step_in_epoch % 100 == 0 or step_in_epoch == steps_per_epoch:
                cur_loss = train_loss_sum / max(1, train_count_sum)
                cur_acc = train_correct_sum / max(1, train_count_sum)
                print(
                    f"Epoch {epoch:02d}/{total_epochs:02d} step {step_in_epoch:03d}/{steps_per_epoch} "
                    f"loss={cur_loss:.4f} acc={cur_acc:.4f} "
                    f"grad_norm={float(step_metrics['grad_norm'].numpy()):.2f} "
                    f"gate_mean={float(step_metrics['gate_mean'].numpy()):.4f} "
                    f"gamma_cascade={float(step_metrics['gamma_cascade'].numpy()):.4f}",
                    flush=True,
                )
            step_idx += 1

        train_loss = train_loss_sum / max(1, train_count_sum)
        train_acc = train_correct_sum / max(1, train_count_sum)
        avg_grad_norm = float(np.mean(grad_norms))
        avg_gate_mean = float(np.mean(gate_means))
        avg_gamma_cascade = float(np.mean(gamma_cascades))

        val_metrics = evaluate_dataset(model, val_ds, cfg, use_tta_hflip=False)
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_macro_f1 = val_metrics["macro_f1"]

        elapsed = time.time() - epoch_start_time
        phase_str = "FREEZE" if is_backbone_frozen else "UNFREEZE"

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} [{phase_str}] ({elapsed:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f} | "
            f"GateMean: {avg_gate_mean:.3f}, GammaCascade: {avg_gamma_cascade:.4f}, "
            f"GradNorm: {avg_grad_norm:.2f}, NonFiniteBatches: {nonfinite_batches}",
            flush=True,
        )

        history_logs.append({
            "epoch": epoch,
            "phase": phase_str,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "gate_mean": avg_gate_mean,
            "gamma_cascade": avg_gamma_cascade,
            "grad_norm": avg_grad_norm,
        })
        save_plots_and_history(history_logs, output_dir)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0
            best_ckpt_path = output_dir / "best_model.ckpt"
            model.save_weights(str(best_ckpt_path))
            print(f"  --> Saved new best model checkpoint (Val Acc: {val_acc:.4f})", flush=True)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n[EARLY STOPPING] No improvement for {patience} consecutive epochs. Stopping at epoch {epoch}.", flush=True)
                break

    print(f"\n==========================================", flush=True)
    print(f" EVALUATION ON TEST SET (BEST CHECKPOINT)", flush=True)
    print(f"==========================================", flush=True)

    best_ckpt_path = output_dir / "best_model.ckpt"
    best_h5_path = output_dir / "best_model.h5"
    if (output_dir / "best_model.ckpt.index").exists():
        model.load_weights(str(best_ckpt_path))
        print(f"Loaded best weights from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})", flush=True)
    elif best_h5_path.exists():
        model.load_weights(str(best_h5_path))
        print(f"Loaded best weights from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})", flush=True)

    test_standard = evaluate_dataset(model, test_ds, cfg, use_tta_hflip=False)
    test_tta = evaluate_dataset(model, test_ds, cfg, use_tta_hflip=True)

    print(f"Standard Test Acc:  {test_standard['accuracy']:.4f} (Macro-F1: {test_standard['macro_f1']:.4f})", flush=True)
    print(f"TTA HFlip Test Acc: {test_tta['accuracy']:.4f} (Macro-F1: {test_tta['macro_f1']:.4f})", flush=True)

    test_results = {
        "experiment": "ConvNeXt-B MS1M Hierarchical Cascade Region Gated Fusion",
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_standard": test_standard,
        "test_tta_hflip": test_tta,
    }
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"Saved final test metrics to {output_dir / 'test_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
