from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_XLA"] = "1"
os.environ["TF_DISABLE_XLA_COMPILATION"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if sys.platform == "win32":
    env_dir = os.path.dirname(sys.executable)
    lib_bin = os.path.join(env_dir, "Library", "bin")
    if os.path.exists(lib_bin):
        os.environ["PATH"] = lib_bin + os.path.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_bin)
            except Exception:
                pass
else:
    try:
        import site
        for base in site.getsitepackages():
            for root, dirs, files in os.walk(os.path.join(base, "nvidia")):
                if "ptxas" in files or root.endswith("bin"):
                    os.environ["PATH"] = root + os.path.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*calling iterator did not fully read the dataset being cached.*")
tf.get_logger().setLevel('ERROR')

from config import load_config, global_batch_size
from datasets.fer2013 import EMOTION_NAMES, build_datasets
from losses.classification import supervised_mgr_loss
from metrics.classification import classification_metrics, save_metrics
from models import ConvNeXtBaseFaceFERBaseline, ConvNeXtBaseImageNetFERBaseline, IR50FERBaseline, MGRConvNeXtFER


class LegacyDecoupledAdamW(tf.keras.optimizers.Adam):
    """Small AdamW fallback for TF 2.10 environments without Keras AdamW."""

    def __init__(self, learning_rate: float, weight_decay: float = 0.0, name: str = "LegacyDecoupledAdamW", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.weight_decay = float(weight_decay)

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        if not grads_and_vars:
            return tf.no_op()
        train_op = super().apply_gradients(
            grads_and_vars,
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients,
        )
        if self.weight_decay <= 0.0:
            return train_op

        def _decay():
            lr = tf.cast(self.learning_rate, grads_and_vars[0][1].dtype)
            decay_ops = []
            for _, var in grads_and_vars:
                decay_ops.append(var.assign_sub(tf.cast(self.weight_decay, var.dtype) * lr * var))
            return tf.group(decay_ops)

        with tf.control_dependencies([train_op]):
            return _decay()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FER2013_SGU TensorFlow MGR-CNN training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def configure_tensorflow_runtime(cfg: Dict) -> None:
    runtime = cfg.get("runtime", {})
    intra_threads = runtime.get("intra_op_threads")
    inter_threads = runtime.get("inter_op_threads")
    if intra_threads:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
    if inter_threads:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
    tf.config.optimizer.set_jit(bool(runtime.get("xla", False)))
    if not bool(runtime.get("xla", False)):
        print("[INFO] XLA auto-JIT disabled (jit_compile=False; TF_XLA_FLAGS disables XLA devices)", flush=True)
    if bool(runtime.get("use_mixed_precision", True)):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled (mixed_float16) for accelerated GPU training", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not enable mixed precision: {e}", flush=True)


def configure_gpus(cfg: Dict) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[WARNING] No GPU devices visible to TensorFlow. Falling back to CPU mode.")
        return
    gpu_ids = cfg["runtime"].get("gpu_ids", [0])
    visible = [gpus[i] for i in gpu_ids if i < len(gpus)]
    if not visible:
        visible = gpus
    min_gpus = int(cfg["runtime"].get("min_gpus", 1))
    if len(visible) < min_gpus:
        if bool(cfg["runtime"].get("allow_cpu_fallback", True)) or min_gpus <= 1:
            visible = gpus
        else:
            raise RuntimeError(f"TensorFlow sees only {len(visible)} GPU(s), need {min_gpus}.")
    if visible:
        tf.config.set_visible_devices(visible, "GPU")
        if cfg["runtime"].get("memory_growth", True):
            for gpu in visible:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        print(f"[INFO] Configured {len(visible)} GPU device(s): {[g.name for g in visible]}")


def build_optimizer(cfg: Dict, learning_rate: float):
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)
    if adamw is not None:
        try:
            return adamw(learning_rate=learning_rate, weight_decay=weight_decay, jit_compile=False)
        except (TypeError, ValueError):
            return adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return LegacyDecoupledAdamW(learning_rate=learning_rate, weight_decay=weight_decay)


def get_param_count(model: tf.keras.Model) -> Tuple[int, int]:
    try:
        vars_list = model.variables
    except Exception:
        vars_list = []
    try:
        trainable_vars_list = model.trainable_variables
    except Exception:
        trainable_vars_list = []
    total = int(np.sum([np.prod(v.shape) for v in vars_list])) if vars_list else 0
    trainable = int(np.sum([np.prod(v.shape) for v in trainable_vars_list])) if trainable_vars_list else 0
    return total, trainable


def variable_key(variable) -> object:
    ref = getattr(variable, "ref", None)
    if callable(ref):
        return ref()
    experimental_ref = getattr(variable, "experimental_ref", None)
    if callable(experimental_ref):
        return experimental_ref()
    path = getattr(variable, "path", None)
    if path:
        return path
    name = getattr(variable, "name", None)
    if name:
        return name
    return id(variable)


def split_variables(model: MGRConvNeXtFER) -> Tuple[List[tf.Variable], List[tf.Variable]]:
    backbone_ids = {variable_key(v) for v in model.backbone.variables}
    backbone = [v for v in model.trainable_variables if variable_key(v) in backbone_ids]
    head = [v for v in model.trainable_variables if variable_key(v) not in backbone_ids]
    return backbone, head


def build_model(cfg: Dict) -> tf.keras.Model:
    arch = str(cfg.get("model", {}).get("arch", "convnext_tiny")).lower()
    name = str(cfg.get("model", {}).get("name", "")).lower()
    if arch in ("ir50", "iresnet50", "insightface_ir50") or name.startswith("ir50"):
        return IR50FERBaseline(cfg)
    if arch in ("convnext_base_imagenet1k", "convnext_base_imagenet", "convnext_base_imagenet_1k"):
        return ConvNeXtBaseImageNetFERBaseline(cfg)
    if "mgr" in name or "mgr" in arch or "dynamic_gate" in name:
        return MGRConvNeXtFER(cfg)
    if arch in ("convnext_base", "convnext_base_face", "convnext_base_ms1m_arcface") or name.startswith("convnext_base_ms1m"):
        return ConvNeXtBaseFaceFERBaseline(cfg)
    if name.startswith("convnext_base_imagenet"):
        return ConvNeXtBaseImageNetFERBaseline(cfg)
    if name.startswith("convnext_base"):
        return ConvNeXtBaseFaceFERBaseline(cfg)
    return MGRConvNeXtFER(cfg)


def ensure_optimizer_built(optimizer, variables: Sequence[tf.Variable], strategy: Optional[tf.distribute.Strategy] = None) -> None:
    variables = [v for v in variables if v is not None]
    if not variables:
        return
    build = getattr(optimizer, "build", None)
    if callable(build):
        try:
            build(variables)
            return
        except Exception:
            pass
    def _dummy():
        optimizer.apply_gradients([(tf.zeros_like(v), v) for v in variables])
    if strategy is not None:
        strategy.run(_dummy)
    else:
        _dummy()


def set_optimizer_lr(optimizer, lr_value: float):
    if optimizer is None:
        return
    try:
        if isinstance(getattr(optimizer, "learning_rate", None), tf.Variable):
            optimizer.learning_rate.assign(float(lr_value))
            return
    except Exception:
        pass
    try:
        tf.keras.backend.set_value(optimizer.learning_rate, float(lr_value))
        return
    except Exception:
        pass
    try:
        if hasattr(optimizer, "_set_hyper"):
            optimizer._set_hyper("learning_rate", float(lr_value))
            return
    except Exception:
        try:
            optimizer.learning_rate = float(lr_value)
        except Exception:
            pass


def cosine_lr(base_lr: float, epoch: int, epochs: int, min_lr: float = 1e-6, warmup_epochs: int = 5) -> float:
    if epochs <= 1:
        return base_lr
    if warmup_epochs > 0 and epoch < warmup_epochs:
        warmup_factor = float(epoch + 1) / float(warmup_epochs)
        return float(min_lr + (base_lr - min_lr) * warmup_factor)
    decay_epochs = max(1, epochs - warmup_epochs)
    decay_epoch = epoch - warmup_epochs
    ratio = min(1.0, max(0.0, float(decay_epoch) / float(max(decay_epochs - 1, 1))))
    cosine = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return float(min_lr + (base_lr - min_lr) * cosine)


def resolve_phase_lrs(cfg: Dict, epoch: int, train_backbone: bool) -> Tuple[float, float]:
    training_cfg = cfg["training"]
    total_epochs = int(training_cfg["epochs"])
    warmup_epochs = int(training_cfg.get("warmup_epochs", 5))
    stage_tr = cfg.get("stage_transition", {})
    if bool(stage_tr.get("enable_2stage_switching", False)) and epoch >= int(stage_tr.get("stage1_end_epoch", 60)):
        stage2_start = int(stage_tr.get("stage1_end_epoch", 60))
        phase2_epoch = epoch - stage2_start
        phase2_total = max(1, total_epochs - stage2_start)
        head_base_lr = float(stage_tr.get("stage2_finetune_lr", training_cfg.get("finetune_lr", 0.00004)))
        visual_base_lr = float(stage_tr.get("stage2_visual_extractor_lr", training_cfg.get("visual_extractor_lr", 0.000002)))
        return (
            cosine_lr(head_base_lr, phase2_epoch, phase2_total, warmup_epochs=min(warmup_epochs, phase2_total)),
            cosine_lr(visual_base_lr, phase2_epoch, phase2_total, warmup_epochs=min(warmup_epochs, phase2_total)),
        )
    if train_backbone:
        freeze_epochs = int(cfg["model"].get("freeze_backbone_epochs", 0) or 0)
        phase_epoch = max(0, int(epoch) - freeze_epochs)
        head_base_lr = float(training_cfg.get("finetune_lr", training_cfg["lr"]))
        visual_base_lr = float(training_cfg.get("visual_extractor_lr", head_base_lr))
        return (
            cosine_lr(head_base_lr, phase_epoch, total_epochs, warmup_epochs=warmup_epochs),
            cosine_lr(visual_base_lr, phase_epoch, total_epochs, warmup_epochs=warmup_epochs),
        )
    return cosine_lr(float(training_cfg["lr"]), int(epoch), total_epochs, warmup_epochs=warmup_epochs), 0.0


def resolve_monitor_value(metrics: Dict[str, object], monitor_name: str) -> float:
    aliases = {
        "val_macro_f1": "macro_f1",
        "val_weighted_f1": "weighted_f1",
        "val_accuracy": "accuracy",
        "val_acc": "accuracy",
        "val_loss": "loss",
    }
    key = aliases.get(monitor_name, monitor_name)
    if key not in metrics:
        raise KeyError(f"Monitor {monitor_name!r} resolved to {key!r}, but metric is not available.")
    value = float(metrics[key])
    return -value if key == "loss" else value


def _apply_escape_lr(
    cosine_head_lr: float,
    cosine_backbone_lr: float,
    state: Dict,
    esc_cfg: Dict,
) -> Tuple[float, float]:
    """Apply LR escape multiplier on top of cosine LR, capped by upper bound."""
    level = state["escape_level"]
    if level == 1:
        head_mult = float(esc_cfg.get("level1_head_lr_multiplier", 2.0))
        back_mult = float(esc_cfg.get("level1_backbone_lr_multiplier", 1.5))
    elif level == 2:
        head_mult = float(esc_cfg.get("level2_head_lr_multiplier", 3.0))
        back_mult = float(esc_cfg.get("level2_backbone_lr_multiplier", 1.5))
    else:
        return cosine_head_lr, cosine_backbone_lr
    cap_head = float(esc_cfg.get("cap_head_lr", 1e-4))
    cap_back = float(esc_cfg.get("cap_backbone_lr", 5e-6))
    return (
        min(cosine_head_lr * head_mult, cap_head),
        min(cosine_backbone_lr * back_mult, cap_back),
    )


def _update_lr_escape_state(
    state: Dict,
    esc_cfg: Dict,
    val_metrics: Dict,
    train_loss: float,
    epoch: int,
    best_score: float,
    best_epoch: int,
    patience_anchor_epoch: int,
    checkpoint,
    best_manager,
) -> Tuple[float, int, int]:
    """Update LR escape state machine after validation.

    Returns (best_score, best_epoch, patience_anchor_epoch),
    possibly modified by rollback.
    """
    val_loss = float(val_metrics["loss"])
    min_delta = float(esc_cfg.get("min_delta", 0.001))
    max_cycles = int(esc_cfg.get("max_cycles", 2))

    # Overfitting detection: train_loss down but val_loss up
    is_overfitting = (
        train_loss < state["prev_train_loss"]
        and val_loss > state["best_val_loss"]
    )
    state["prev_train_loss"] = train_loss

    # ==================== NORMAL MODE ====================
    if state["mode"] == "normal":
        # Cooldown: freeze plateau counter but still track best
        if state["cooldown_remaining"] > 0:
            state["cooldown_remaining"] -= 1
            if val_loss < state["best_val_loss"] - min_delta:
                state["best_val_loss"] = val_loss
                state["plateau_counter"] = 0
            return best_score, best_epoch, patience_anchor_epoch

        # Plateau detection: CHECK improvement FIRST, THEN update best
        improved = val_loss < state["best_val_loss"] - min_delta
        if improved:
            state["best_val_loss"] = val_loss
            state["plateau_counter"] = 0
        else:
            state["plateau_counter"] += 1

        patience = int(esc_cfg.get("plateau_patience", 18))
        if (
            state["plateau_counter"] >= patience
            and not is_overfitting
            and state["cycles_used"] < max_cycles
        ):
            # Save full pre-escape snapshot
            state["snapshot"] = {
                "best_val_loss": state["best_val_loss"],
                "best_score": best_score,
                "best_epoch": best_epoch,
                "patience_anchor_epoch": patience_anchor_epoch,
                "prev_train_loss": state["prev_train_loss"],
            }
            state["mode"] = "escaping"
            state["escape_level"] = 1
            state["escape_epoch_counter"] = 0
            print(
                f"[LR_ESCAPE] Cycle {state['cycles_used']+1}/{max_cycles}: "
                f"entering Level 1 at epoch {epoch+1} "
                f"(plateau {state['plateau_counter']} eps, "
                f"best_val_loss={state['best_val_loss']:.5f})",
                flush=True,
            )
        return best_score, best_epoch, patience_anchor_epoch

    # ==================== ESCAPING MODE ====================
    state["escape_epoch_counter"] += 1
    pre_best = state["snapshot"]["best_val_loss"]
    escaped_improved = val_loss < pre_best - min_delta

    if escaped_improved:
        # ---- Escape SUCCESS ----
        completed_level = state["escape_level"]
        cooldown = int(esc_cfg.get("cooldown_epochs", 12))
        state["best_val_loss"] = val_loss
        state["plateau_counter"] = 0
        state["cycles_used"] += 1
        state["mode"] = "normal"
        state["escape_level"] = 0
        state["escape_epoch_counter"] = 0
        state["cooldown_remaining"] = cooldown
        state["snapshot"] = None
        print(
            f"[LR_ESCAPE] Level {completed_level} succeeded at epoch {epoch+1}! "
            f"val_loss={val_loss:.5f} < pre_escape={pre_best:.5f}. "
            f"Entering cooldown ({cooldown} eps).",
            flush=True,
        )
        return best_score, best_epoch, patience_anchor_epoch

    # ---- Check level duration expiry ----
    if state["escape_level"] == 1:
        duration = int(esc_cfg.get("level1_duration", 5))
        if state["escape_epoch_counter"] >= duration:
            state["escape_level"] = 2
            state["escape_epoch_counter"] = 0
            print(
                f"[LR_ESCAPE] Level 1 failed after {duration} eps. "
                f"Escalating to Level 2 at epoch {epoch+1}.",
                flush=True,
            )

    elif state["escape_level"] == 2:
        duration = int(esc_cfg.get("level2_duration", 4))
        if state["escape_epoch_counter"] >= duration:
            # ---- Full cycle FAILED → ROLLBACK ----
            snap = state["snapshot"]
            best_ckpt = best_manager.latest_checkpoint
            if best_ckpt:
                checkpoint.restore(best_ckpt).expect_partial()
                print(
                    f"[LR_ESCAPE] Rollback: restored checkpoint {best_ckpt}",
                    flush=True,
                )
            # Restore full pre-escape tracking state
            restored_best_score = snap["best_score"]
            restored_best_epoch = snap["best_epoch"]
            restored_patience_anchor = snap["patience_anchor_epoch"]
            state["best_val_loss"] = snap["best_val_loss"]
            state["prev_train_loss"] = snap["prev_train_loss"]
            state["cycles_used"] += 1
            state["mode"] = "normal"
            state["escape_level"] = 0
            state["escape_epoch_counter"] = 0
            state["plateau_counter"] = 0
            state["cooldown_remaining"] = 0
            state["snapshot"] = None
            print(
                f"[LR_ESCAPE] Cycle failed at epoch {epoch+1}. "
                f"Rollback complete. "
                f"Cycles used: {state['cycles_used']}/{max_cycles}. "
                f"Training continues.",
                flush=True,
            )
            return restored_best_score, restored_best_epoch, restored_patience_anchor

    return best_score, best_epoch, patience_anchor_epoch


def classification_ce_loss(labels, logits, num_classes: int, label_smoothing: float):
    logits = tf.cast(logits, tf.float32)
    y_true = tf.one_hot(tf.cast(labels, tf.int32), int(num_classes), dtype=tf.float32)
    return tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true,
            logits,
            from_logits=True,
            label_smoothing=float(label_smoothing),
        )
    )


def make_step_function(
    cfg: Dict,
    model: MGRConvNeXtFER,
    optimizer_head,
    optimizer_backbone=None,
    loss_scale: float = 1.0,
):
    loss_cfg = cfg["training"]
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    ortho_weight = float(cfg["model"].get("ortho_loss_weight", 0.003))
    cnn_aux_weight = float(cfg["model"].get("cnn_aux_loss_weight", 0.4))
    sam_rho = float(loss_cfg.get("sam_rho", 0.03))
    sam_adaptive = bool(loss_cfg.get("sam_adaptive", False))
    use_sam = str(loss_cfg.get("optimizer", "sam")).lower() == "sam"
    skip_nonfinite = bool(loss_cfg.get("skip_nonfinite_batches", True))
    grad_clip_norm = float(loss_cfg.get("grad_clip_norm", 0.0)) if loss_cfg.get("grad_clip_norm") else None
    loss_scale_tensor = tf.constant(float(loss_scale), dtype=tf.float32)
    backbone_vars, head_vars = split_variables(model)
    all_vars = head_vars + backbone_vars

    def _all_finite(grads):
        finite_tensors = [tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None]
        return tf.constant(True) if not finite_tensors else tf.reduce_all(tf.stack(finite_tensors))

    def _batch_stats(outputs, labels):
        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
        count = tf.shape(labels)[0]
        sem_correct = tf.constant(0, dtype=tf.int32)
        if outputs.get("semantic_logits") is not None:
            sem_preds = tf.argmax(outputs["semantic_logits"], axis=-1, output_type=tf.int32)
            sem_correct = tf.reduce_sum(tf.cast(tf.equal(sem_preds, labels), tf.int32))

        gw_sum = tf.zeros([5], dtype=tf.float32)
        gw_sq_sum = tf.zeros([5], dtype=tf.float32)
        entropy_sum = tf.constant(0.0, dtype=tf.float32)
        gw = outputs.get("granularity_weights")
        if gw is not None:
            gw_f32 = tf.cast(gw, tf.float32)
            gw_sum = tf.reduce_sum(gw_f32, axis=0)
            gw_sq_sum = tf.reduce_sum(tf.square(gw_f32), axis=0)
            ent = -tf.reduce_sum(gw_f32 * tf.math.log(gw_f32 + 1e-9), axis=-1)
            entropy_sum = tf.reduce_sum(ent)

        return correct, sem_correct, count, gw_sum, gw_sq_sum, entropy_sum

    def _clip_gradients(grads):
        if grad_clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        return grads

    def _apply_gradients(grads, trainable_vars):
        grads = _clip_gradients(grads)
        backbone_ids = {variable_key(v) for v in backbone_vars}
        head_grads = [(g, v) for g, v in zip(grads, trainable_vars) if variable_key(v) not in backbone_ids and g is not None]
        backbone_grads = [(g, v) for g, v in zip(grads, trainable_vars) if variable_key(v) in backbone_ids and g is not None]
        if head_grads:
            optimizer_head.apply_gradients(head_grads)
        if optimizer_backbone is not None and backbone_grads:
            optimizer_backbone.apply_gradients(backbone_grads)

    def _step_impl(features, labels, trainable_vars):
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            raw_loss, parts = supervised_mgr_loss(
                labels,
                outputs,
                num_classes=cfg["data"]["num_classes"],
                label_smoothing=label_smoothing,
                ortho_weight=ortho_weight,
                cnn_aux_weight=cnn_aux_weight,
            )
            loss = raw_loss * loss_scale_tensor
        grads = tape.gradient(loss, trainable_vars)
        if skip_nonfinite:
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None for g in grads]

        hard_loss = parts.get("hard_semantic", tf.constant(0.0, dtype=tf.float32))

        if not use_sam:
            _apply_gradients(grads, trainable_vars)
            fer_correct, sem_correct, count, gw_sum, gw_sq_sum, entropy_sum = _batch_stats(outputs, labels)
            return raw_loss, parts["ce"], parts["semantic"], hard_loss, fer_correct, sem_correct, count, gw_sum, gw_sq_sum, entropy_sum, tf.constant(1, tf.int32)

        grads = _clip_gradients(grads)
        grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])

        eps_list = []
        for var, grad in zip(trainable_vars, grads):
            if grad is None:
                eps_list.append(None)
                continue
            scale = sam_rho / (grad_norm + 1e-12)
            if sam_adaptive:
                scale = scale * tf.square(tf.abs(var))
            eps_list.append(grad * scale)

        for var, eps in zip(trainable_vars, eps_list):
            if eps is not None:
                var.assign_add(eps)

        with tf.GradientTape() as tape2:
            outputs_2 = model(features, training=True)
            raw_loss_2, _ = supervised_mgr_loss(
                labels,
                outputs_2,
                num_classes=cfg["data"]["num_classes"],
                label_smoothing=label_smoothing,
                ortho_weight=ortho_weight,
                cnn_aux_weight=cnn_aux_weight,
            )
            loss_2 = raw_loss_2 * loss_scale_tensor
        grads_2 = tape2.gradient(loss_2, trainable_vars)

        for var, eps in zip(trainable_vars, eps_list):
            if eps is not None:
                var.assign_sub(eps)

        if skip_nonfinite:
            grads_2 = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None for g in grads_2]

        _apply_gradients(grads_2, trainable_vars)
        fer_correct, sem_correct, count, gw_sum, gw_sq_sum, entropy_sum = _batch_stats(outputs, labels)
        return raw_loss, parts["ce"], parts["semantic"], hard_loss, fer_correct, sem_correct, count, gw_sum, gw_sq_sum, entropy_sum, tf.constant(1, tf.int32)

    def train_step_head(features, labels):
        return _step_impl(features, labels, head_vars)

    def train_step_full(features, labels):
        return _step_impl(features, labels, all_vars)

    return train_step_head, train_step_full


def make_distributed_train_step(strategy: tf.distribute.Strategy, train_step):
    @tf.function(reduce_retracing=True, jit_compile=False)
    def distributed_step(batch):
        (
            per_loss, per_ce, per_sem, per_hard,
            per_correct, per_sem_correct, per_count,
            per_gw_sum, per_gw_sq_sum, per_ent_sum, per_ok
        ) = strategy.run(train_step, args=batch)
        ok = strategy.reduce(tf.distribute.ReduceOp.SUM, per_ok, axis=None)
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_loss, axis=None)
        ce = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_ce, axis=None)
        sem = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_sem, axis=None)
        hard = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_hard, axis=None)
        correct = strategy.reduce(tf.distribute.ReduceOp.SUM, per_correct, axis=None)
        sem_correct = strategy.reduce(tf.distribute.ReduceOp.SUM, per_sem_correct, axis=None)
        count = strategy.reduce(tf.distribute.ReduceOp.SUM, per_count, axis=None)
        gw_sum = strategy.reduce(tf.distribute.ReduceOp.SUM, per_gw_sum, axis=None)
        gw_sq_sum = strategy.reduce(tf.distribute.ReduceOp.SUM, per_gw_sq_sum, axis=None)
        ent_sum = strategy.reduce(tf.distribute.ReduceOp.SUM, per_ent_sum, axis=None)
        return loss, ce, sem, hard, correct, sem_correct, count, gw_sum, gw_sq_sum, ent_sum, ok

    return distributed_step


def evaluate_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    cfg: Dict,
    strategy: Optional[tf.distribute.Strategy] = None,
    use_tta_hflip: Optional[bool] = None,
    original_weight: Optional[float] = None,
    flip_weight: Optional[float] = None,
) -> Dict[str, object]:
    tta_cfg = cfg.get("tta", {})
    if use_tta_hflip is None:
        use_tta = bool(tta_cfg.get("enabled", False))
    else:
        use_tta = bool(use_tta_hflip)

    w_orig = float(original_weight if original_weight is not None else tta_cfg.get("original_weight", 0.5))
    w_flip = float(flip_weight if flip_weight is not None else tta_cfg.get("flip_weight", 0.5))

    if use_tta:
        total_w = w_orig + w_flip
        if total_w > 0 and abs(total_w - 1.0) > 1e-5:
            w_orig = w_orig / total_w
            w_flip = w_flip / total_w
        print(f"[TTA] Horizontal Flip: ENABLED", flush=True)
        print(f"[TTA] Original weight: {w_orig:.2f}", flush=True)
        print(f"[TTA] Flip weight:     {w_flip:.2f}", flush=True)
    else:
        print(f"[TTA] Horizontal Flip: DISABLED", flush=True)

    def _forward_outputs(inputs):
        outputs_orig = model(inputs, training=False)
        if not use_tta:
            return outputs_orig, outputs_orig

        flipped_inputs = dict(inputs)
        flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
        if "mask" in inputs:
            flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
        outputs_flip = model(flipped_inputs, training=False)

        outputs_tta = dict(outputs_orig)
        outputs_tta["logits"] = w_orig * outputs_orig["logits"] + w_flip * outputs_flip["logits"]
        if outputs_orig.get("semantic_logits") is not None and outputs_flip.get("semantic_logits") is not None:
            outputs_tta["semantic_logits"] = w_orig * outputs_orig["semantic_logits"] + w_flip * outputs_flip["semantic_logits"]

        return outputs_orig, outputs_tta

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _eval_step(inputs, labels):
        outputs_orig, outputs_tta = _forward_outputs(inputs)
        total_l_orig, parts_orig = supervised_mgr_loss(
            labels,
            outputs_orig,
            num_classes=cfg["data"]["num_classes"],
            label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
            ortho_weight=float(cfg["model"].get("ortho_loss_weight", 0.003)),
            cnn_aux_weight=float(cfg["model"].get("cnn_aux_loss_weight", 0.4)),
        )
        total_l_tta, parts_tta = supervised_mgr_loss(
            labels,
            outputs_tta,
            num_classes=cfg["data"]["num_classes"],
            label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
            ortho_weight=float(cfg["model"].get("ortho_loss_weight", 0.003)),
            cnn_aux_weight=float(cfg["model"].get("cnn_aux_loss_weight", 0.4)),
        )
        fer_preds_orig = tf.argmax(outputs_orig["logits"], axis=-1, output_type=tf.int32)
        fer_preds_tta = tf.argmax(outputs_tta["logits"], axis=-1, output_type=tf.int32)

        return (
            total_l_orig, parts_orig["ce"], parts_orig["semantic"], parts_orig.get("hard_semantic", tf.constant(0.0, tf.float32)), fer_preds_orig,
            total_l_tta, parts_tta["ce"], parts_tta["semantic"], parts_tta.get("hard_semantic", tf.constant(0.0, tf.float32)), fer_preds_tta,
            tf.shape(labels)[0], labels
        )

    y_true: List[int] = []
    y_pred_orig: List[int] = []
    y_pred_tta: List[int] = []

    total_loss_orig = 0.0
    total_ce_orig = 0.0
    total_sem_orig = 0.0
    total_hard_orig = 0.0

    total_loss_tta = 0.0
    total_ce_tta = 0.0
    total_sem_tta = 0.0
    total_hard_tta = 0.0
    total_count = 0

    if strategy is not None and strategy.num_replicas_in_sync > 1:
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        for batch in dist_dataset:
            (
                loc_tot_o, loc_ce_o, loc_sem_o, loc_hd_o, loc_p_o,
                loc_tot_t, loc_ce_t, loc_sem_t, loc_hd_t, loc_p_t,
                loc_cnts, loc_lbs
            ) = strategy.run(_eval_step, args=batch)

            for l_tot_o, l_ce_o, l_sem_o, l_hd_o, l_p_o, l_tot_t, l_ce_t, l_sem_t, l_hd_t, l_p_t, l_cnt, l_lb in zip(
                strategy.experimental_local_results(loc_tot_o),
                strategy.experimental_local_results(loc_ce_o),
                strategy.experimental_local_results(loc_sem_o),
                strategy.experimental_local_results(loc_hd_o),
                strategy.experimental_local_results(loc_p_o),
                strategy.experimental_local_results(loc_tot_t),
                strategy.experimental_local_results(loc_ce_t),
                strategy.experimental_local_results(loc_sem_t),
                strategy.experimental_local_results(loc_hd_t),
                strategy.experimental_local_results(loc_p_t),
                strategy.experimental_local_results(loc_cnts),
                strategy.experimental_local_results(loc_lbs),
            ):
                count = int(l_cnt.numpy())
                if count == 0:
                    continue
                total_loss_orig += float(l_tot_o.numpy()) * count
                total_ce_orig += float(l_ce_o.numpy()) * count
                total_sem_orig += float(l_sem_o.numpy()) * count
                total_hard_orig += float(l_hd_o.numpy()) * count

                total_loss_tta += float(l_tot_t.numpy()) * count
                total_ce_tta += float(l_ce_t.numpy()) * count
                total_sem_tta += float(l_sem_t.numpy()) * count
                total_hard_tta += float(l_hd_t.numpy()) * count

                total_count += count
                y_true.extend(l_lb.numpy().tolist())
                y_pred_orig.extend(l_p_o.numpy().tolist())
                y_pred_tta.extend(l_p_t.numpy().tolist())

    else:
        for batch in dataset:
            inputs, labels = batch
            tot_o, ce_o, sem_o, hd_o, p_o, tot_t, ce_t, sem_t, hd_t, p_t, count, _ = _eval_step(inputs, labels)
            c = int(count.numpy())
            y_true.extend(labels.numpy().tolist())
            y_pred_orig.extend(p_o.numpy().tolist())
            y_pred_tta.extend(p_t.numpy().tolist())

            total_loss_orig += float(tot_o.numpy()) * c
            total_ce_orig += float(ce_o.numpy()) * c
            total_sem_orig += float(sem_o.numpy()) * c
            total_hard_orig += float(hd_o.numpy()) * c

            total_loss_tta += float(tot_t.numpy()) * c
            total_ce_tta += float(ce_t.numpy()) * c
            total_sem_tta += float(sem_t.numpy()) * c
            total_hard_tta += float(hd_t.numpy()) * c

            total_count += c

    c_norm = max(total_count, 1)

    metrics_tta = classification_metrics(y_true, y_pred_tta, EMOTION_NAMES)
    metrics_tta["loss"] = total_loss_tta / c_norm
    metrics_tta["total_loss"] = total_loss_tta / c_norm
    metrics_tta["ce_loss"] = total_ce_tta / c_norm
    metrics_tta["semantic_loss"] = total_sem_tta / c_norm
    metrics_tta["hard_semantic_loss"] = total_hard_tta / c_norm
    metrics_tta["tta_hflip"] = bool(use_tta)
    metrics_tta["original_weight"] = w_orig
    metrics_tta["flip_weight"] = w_flip

    metrics_no_tta = classification_metrics(y_true, y_pred_orig, EMOTION_NAMES)
    metrics_tta["no_tta_accuracy"] = float(metrics_no_tta["accuracy"])
    metrics_tta["no_tta_macro_f1"] = float(metrics_no_tta["macro_f1"])
    metrics_tta["no_tta_weighted_f1"] = float(metrics_no_tta["weighted_f1"])
    metrics_tta["no_tta_loss"] = total_loss_orig / c_norm
    metrics_tta["no_tta_hard_semantic_loss"] = total_hard_orig / c_norm

    print(f"[EVALUATION SUMMARY]", flush=True)
    print(
        f"  NO_TTA    - Accuracy: {metrics_tta['no_tta_accuracy']*100:.2f}% | "
        f"Macro F1: {metrics_tta['no_tta_macro_f1']:.4f} | "
        f"Weighted F1: {metrics_tta['no_tta_weighted_f1']:.4f}",
        flush=True,
    )
    if use_tta:
        print(
            f"  HFLIP_TTA - Accuracy: {metrics_tta['accuracy']*100:.2f}% | "
            f"Macro F1: {metrics_tta['macro_f1']:.4f} | "
            f"Weighted F1: {metrics_tta['weighted_f1']:.4f}",
            flush=True,
        )

    gc.collect()
    return metrics_tta


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)
    visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
    strategy_devices = (
        [f"/GPU:{i}" for i in range(visible_gpu_count)]
        if visible_gpu_count > 0
        else ["/CPU:0"]
    )
    strategy = tf.distribute.MirroredStrategy(devices=strategy_devices)
    print(f"TensorFlow {tf.__version__}")
    print(f"Replicas in sync: {strategy.num_replicas_in_sync}")

    stage_tr = cfg.get("stage_transition", {})
    if bool(stage_tr.get("enable_2stage_switching", False)) and "stage1_batch_size_per_gpu" in stage_tr:
        cfg["runtime"]["batch_size_per_gpu"] = int(stage_tr["stage1_batch_size_per_gpu"])
        print(f"[STAGE 1] Initialized with stage1_batch_size_per_gpu={cfg['runtime']['batch_size_per_gpu']}", flush=True)

    train_ds, val_ds, test_ds = build_datasets(cfg, replicas=strategy.num_replicas_in_sync)
    train_loop_ds = strategy.experimental_distribute_dataset(train_ds)
    global_bs = global_batch_size(cfg, strategy.num_replicas_in_sync)
    if global_bs != int(cfg["runtime"]["batch_size_per_gpu"]) * strategy.num_replicas_in_sync:
        raise ValueError("Global batch size mismatch.")

    first_batch = next(iter(train_ds.take(1)))
    first_inputs, first_labels = first_batch

    run_dir = Path(cfg["paths"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_dir / "checkpoints"

    with strategy.scope():
        model = build_model(cfg)
        orig_ablation = getattr(model, "ablation", None)
        orig_disable = getattr(model, "disable_region_branch_when_cnn_only", None)
        try:
            if hasattr(model, "ablation"):
                model.ablation = "no_mask"
            if hasattr(model, "disable_region_branch_when_cnn_only"):
                model.disable_region_branch_when_cnn_only = False
            _ = model(first_inputs, training=False)
        finally:
            if orig_ablation is not None:
                model.ablation = orig_ablation
            if orig_disable is not None:
                model.disable_region_branch_when_cnn_only = orig_disable

        smoke = model(first_inputs, training=False)
        total_params, trainable_params = get_param_count(model)
        print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
        print(f"Smoke logits shape: {smoke['logits'].shape}")
        optimizer_head = build_optimizer(cfg, float(cfg["training"]["lr"]))
        optimizer_backbone = build_optimizer(cfg, float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])))
        backbone_vars_for_optimizer, head_vars_for_optimizer = split_variables(model)
        ensure_optimizer_built(optimizer_head, head_vars_for_optimizer, strategy)
        ensure_optimizer_built(optimizer_backbone, backbone_vars_for_optimizer, strategy)
        ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt_best_metric = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        checkpoint = tf.train.Checkpoint(
            epoch=ckpt_epoch,
            best_metric=ckpt_best_metric,
            model=model,
            optimizer_head=optimizer_head,
            optimizer_backbone=optimizer_backbone,
        )
        last_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(checkpoint_root / "last"),
            max_to_keep=1,
        )
        max_to_keep_acc = int(cfg["training"].get("max_to_keep_acc", 5))
        max_to_keep_loss = int(cfg["training"].get("max_to_keep_loss", 5))
        best_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(checkpoint_root / "best"),
            max_to_keep=max_to_keep_acc,
        )
        best_loss_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(checkpoint_root / "best_loss"),
            max_to_keep=max_to_keep_loss,
        )
        periodic_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(checkpoint_root / "periodic"),
            max_to_keep=5,
        )
        if (args.resume or cfg["training"].get("resume", True)) and last_manager.latest_checkpoint:
            checkpoint.restore(last_manager.latest_checkpoint).expect_partial()
            print(f"Resumed from {last_manager.latest_checkpoint}")

    first_outputs = model(first_inputs, training=False)
    first_loss, _ = supervised_mgr_loss(
        first_labels,
        first_outputs,
        num_classes=cfg["data"]["num_classes"],
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        ortho_weight=float(cfg["model"].get("ortho_loss_weight", 0.003)),
        cnn_aux_weight=float(cfg["model"].get("cnn_aux_loss_weight", 0.4)),
    )
    if not np.isfinite(float(first_loss.numpy())):
        raise FloatingPointError("Smoke-test loss is not finite.")

    backbone_vars, head_vars = split_variables(model)
    if backbone_vars:
        print(f"Backbone trainable vars: {len(backbone_vars)}")
    print(f"Head trainable vars: {len(head_vars)}")

    loss_scale = 1.0 / float(max(int(strategy.num_replicas_in_sync), 1))
    print(f"[INFO] Distributed gradient loss scale: {loss_scale:.6f}")
    train_step_head, train_step_full = make_step_function(
        cfg,
        model,
        optimizer_head,
        optimizer_backbone,
        loss_scale=loss_scale,
    )
    distributed_train_step_head = make_distributed_train_step(strategy, train_step_head)
    distributed_train_step_full = make_distributed_train_step(strategy, train_step_full)
    start_epoch = int(ckpt_epoch.numpy())
    monitor_name = str(cfg["training"].get("monitor", "val_macro_f1"))
    best_score = float(ckpt_best_metric.numpy())
    best_epoch = start_epoch if best_score >= 0.0 else -1
    best_checkpoint_start_epoch = int(cfg["training"].get("best_checkpoint_start_epoch", 1) or 1)
    patience_anchor_epoch = best_epoch if best_score >= 0.0 else max(start_epoch, best_checkpoint_start_epoch - 1)
    history = []
    csv_path = run_dir / "training_history.csv"
    progress_interval = int(cfg["training"].get("progress_interval", 0) or 0)
    periodic_interval = int(cfg["training"].get("periodic_checkpoint_interval", 10) or 0)
    eval_strategy = strategy if bool(cfg["runtime"].get("distributed_eval", False)) else None
    freeze_epochs = int(cfg["model"].get("freeze_backbone_epochs", 0) or 0)
    # --- LR Escape State Initialization ---
    lr_esc_cfg = cfg.get("lr_escape", {})
    lr_escape_enabled = bool(lr_esc_cfg.get("enabled", False))
    lr_esc_state = {
        "mode": "normal",
        "escape_level": 0,
        "escape_epoch_counter": 0,
        "cycles_used": 0,
        "plateau_counter": 0,
        "best_val_loss": float("inf"),
        "prev_train_loss": float("inf"),
        "cooldown_remaining": 0,
        "snapshot": None,
    }
    best_val_loss_tracked = float("inf")
    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        epoch_start_time = time.time()
        train_backbone = bool(cfg["model"].get("unfreeze_backbone", True)) and epoch >= freeze_epochs
        phase_transitioned = bool(train_backbone and epoch == freeze_epochs)
        if phase_transitioned:
            patience_anchor_epoch = epoch + 1
            print(
                f"[INFO] Unfreezing backbone at epoch {epoch+1}; resetting early-stopping patience counter",
                flush=True,
            )
        train_step = train_step_full if train_backbone else train_step_head
        distributed_train_step = distributed_train_step_full if train_backbone else distributed_train_step_head
        # Dynamic 2-Stage Transition check
        stage_tr = cfg.get("stage_transition", {})
        if bool(stage_tr.get("enable_2stage_switching", False)) and epoch == int(stage_tr.get("stage1_end_epoch", 60)):
            target_ablation = str(stage_tr.get("stage2_ablation", "region_only"))
            if hasattr(model, "set_ablation"):
                model.set_ablation(target_ablation)
            
            stage2_bs = stage_tr.get("stage2_batch_size_per_gpu")
            if stage2_bs and int(stage2_bs) != int(cfg["runtime"]["batch_size_per_gpu"]):
                cfg["runtime"]["batch_size_per_gpu"] = int(stage2_bs)
                train_ds, val_ds, _ = build_datasets(cfg, replicas=strategy.num_replicas_in_sync)
                train_loop_ds = strategy.experimental_distribute_dataset(train_ds)
                print(f"[STAGE SWITCH] Rebuilt train dataset for Stage 2 with batch_size_per_gpu={stage2_bs}", flush=True)

            patience_anchor_epoch = epoch + 1
            print(
                f"[STAGE SWITCH] Epoch {epoch+1}: Automated transition to Stage 2 (Ablation mode: {target_ablation!r})!",
                flush=True,
            )

        lr, backbone_lr = resolve_phase_lrs(cfg, epoch, train_backbone)
        set_optimizer_lr(optimizer_head, lr)
        set_optimizer_lr(optimizer_backbone, backbone_lr)
        # --- LR Escape Override ---
        if lr_escape_enabled and lr_esc_state["escape_level"] > 0:
            lr, backbone_lr = _apply_escape_lr(lr, backbone_lr, lr_esc_state, lr_esc_cfg)
            set_optimizer_lr(optimizer_head, lr)
            set_optimizer_lr(optimizer_backbone, backbone_lr)
            print(
                f"[LR_ESCAPE] Epoch {epoch+1}: Level {lr_esc_state['escape_level']} "
                f"override lr_head={lr:.6f} lr_backbone={backbone_lr:.6f}",
                flush=True,
            )
        total_losses = []
        ce_losses = []
        sem_losses = []
        hard_losses = []
        total_gw_sum = np.zeros(5, dtype=np.float64)
        total_gw_sq_sum = np.zeros(5, dtype=np.float64)
        total_entropy_sum = 0.0
        fer_correct = 0
        sem_correct = 0
        seen = 0
        total_steps = int(tf.data.experimental.cardinality(train_ds).numpy())
        for step_index, batch in enumerate(train_loop_ds, start=1):
            loss, ce_l, sem_l, hard_l, batch_correct, batch_sem_correct, batch_count, gw_sum, gw_sq_sum, ent_sum, ok = distributed_train_step(batch)
            if int(ok.numpy()) == 0:
                continue
            b_cnt = int(batch_count.numpy())
            fer_correct += int(batch_correct.numpy())
            sem_correct += int(batch_sem_correct.numpy())
            seen += b_cnt
            total_losses.append(float(loss.numpy()))
            ce_losses.append(float(ce_l.numpy()))
            sem_losses.append(float(sem_l.numpy()))
            hard_losses.append(float(hard_l.numpy()))
            total_gw_sum += gw_sum.numpy().astype(np.float64)
            total_gw_sq_sum += gw_sq_sum.numpy().astype(np.float64)
            total_entropy_sum += float(ent_sum.numpy())
            if progress_interval and step_index % progress_interval == 0:
                print(
                    f"Epoch {epoch+1}/{cfg['training']['epochs']} "
                    f"step {step_index}/{total_steps} "
                    f"total_loss={float(loss.numpy()):.4f} "
                    f"ce_loss={float(ce_l.numpy()):.4f} "
                    f"sem_loss={float(sem_l.numpy()):.4f} "
                    f"hard_loss={float(hard_l.numpy()):.4f} "
                    f"fer_acc={fer_correct / max(seen, 1):.4f} "
                    f"sem_acc={sem_correct / max(seen, 1):.4f} "
                    f"lr_head={lr:.6f} lr_backbone={backbone_lr:.6f}",
                    flush=True,
                )
        train_time_sec = time.time() - epoch_start_time
        train_steps = len(total_losses)
        train_samples_per_sec = float(seen) / max(train_time_sec, 1e-9)
        train_steps_per_sec = float(train_steps) / max(train_time_sec, 1e-9)
        train_loss = float(np.mean(total_losses)) if total_losses else float("nan")
        train_ce_loss = float(np.mean(ce_losses)) if ce_losses else float("nan")
        train_sem_loss = float(np.mean(sem_losses)) if sem_losses else float("nan")
        train_hard_loss = float(np.mean(hard_losses)) if hard_losses else float("nan")
        train_acc = fer_correct / max(seen, 1)
        train_sem_acc = sem_correct / max(seen, 1)

        # Compute granularity gate statistics
        n_samples = max(seen, 1)
        gw_means = total_gw_sum / n_samples
        gw_var = np.maximum(0.0, (total_gw_sq_sum / n_samples) - (gw_means ** 2))
        gw_stds = np.sqrt(gw_var)
        gate_entropy = total_entropy_sum / n_samples

        # Collapse check: warn if any weight > 0.90
        for k_idx, gw_m in enumerate(gw_means):
            if gw_m > 0.90:
                print(
                    f"[WARNING] Granularity weight {k_idx} mean ({gw_m:.4f}) > 0.90 at Epoch {epoch+1}! Gate may be collapsing.",
                    flush=True,
                )

        print(f"[INFO] Epoch {epoch+1}: starting validation", flush=True)
        val_metrics = evaluate_dataset(
            model,
            val_ds,
            cfg,
            strategy=eval_strategy,
            use_tta_hflip=bool(cfg["runtime"].get("train_val_tta_hflip", False)),
        )
        print(f"[INFO] Epoch {epoch+1}: validation finished", flush=True)
        # --- LR Escape State Update ---
        if lr_escape_enabled and (epoch + 1) >= int(lr_esc_cfg.get("start_epoch", 81)):
            best_score, best_epoch, patience_anchor_epoch = _update_lr_escape_state(
                lr_esc_state, lr_esc_cfg, val_metrics, train_loss,
                epoch, best_score, best_epoch, patience_anchor_epoch,
                checkpoint, best_manager,
            )
        monitor = resolve_monitor_value(val_metrics, monitor_name)
        checkpoint_eligible = (epoch + 1) >= best_checkpoint_start_epoch
        improved = bool(checkpoint_eligible and monitor > best_score)
        if improved:
            best_score = monitor
            best_epoch = epoch + 1
            patience_anchor_epoch = epoch + 1
            ckpt_best_metric.assign(best_score)
            val_loss_val = float(val_metrics['loss'])
            val_acc_val = float(val_metrics['accuracy'])
            print(
                f"[INFO] Save best at ep {epoch+1}, val_loss: {val_loss_val:.4f}, val_accuracy: {val_acc_val:.4f}, monitor: {monitor_name}",
                flush=True,
            )
            best_manager.save(checkpoint_number=epoch + 1)
        elif not checkpoint_eligible:
            print(
                f"[INFO] Epoch {epoch+1}: best checkpoint is not considered before epoch "
                f"{best_checkpoint_start_epoch}",
                flush=True,
            )
        val_loss_val = float(val_metrics['loss'])
        val_acc_val = float(val_metrics['accuracy'])
        if checkpoint_eligible and val_loss_val < best_val_loss_tracked:
            best_val_loss_tracked = val_loss_val
            print(
                f"[INFO] Save best_loss at ep {epoch+1}, val_loss: {val_loss_val:.4f}, val_accuracy: {val_acc_val:.4f}",
                flush=True,
            )
            best_loss_manager.save(checkpoint_number=epoch + 1)
        ckpt_epoch.assign(epoch + 1)
        print(f"[INFO] Epoch {epoch+1}: saving last checkpoint", flush=True)
        last_manager.save(checkpoint_number=epoch + 1)
        if periodic_interval and (epoch + 1) % periodic_interval == 0:
            print(f"[INFO] Epoch {epoch+1}: saving periodic checkpoint", flush=True)
            periodic_manager.save(checkpoint_number=epoch + 1)
        patience_limit = int(cfg["training"].get("patience", 75))
        patience_counter = 0 if not checkpoint_eligible else (epoch + 1) - patience_anchor_epoch
        epoch_time_sec = time.time() - epoch_start_time
        mins, secs = divmod(int(epoch_time_sec), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{epoch_time_sec:.1f}s"
        row = {
            "epoch": epoch + 1,
            "epoch_time_sec": round(epoch_time_sec, 2),
            "train_time_sec": round(train_time_sec, 2),
            "train_samples_per_sec": round(train_samples_per_sec, 2),
            "train_steps_per_sec": round(train_steps_per_sec, 4),
            "train_loss": train_loss,
            "train_ce_loss": train_ce_loss,
            "train_semantic_loss": train_sem_loss,
            "train_hard_semantic_loss": train_hard_loss,
            "train_accuracy": train_acc,
            "train_fer_accuracy": train_acc,
            "train_semantic_accuracy": train_sem_acc,
            "val_loss": float(val_metrics.get("loss", 0.0)),
            "val_ce_loss": float(val_metrics.get("ce_loss", 0.0)),
            "val_semantic_loss": float(val_metrics.get("semantic_loss", 0.0)),
            "val_hard_semantic_loss": float(val_metrics.get("hard_semantic_loss", 0.0)),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_fer_accuracy": float(val_metrics.get("fer_accuracy", val_metrics.get("accuracy", 0.0))),
            "val_semantic_accuracy": float(val_metrics.get("semantic_accuracy", 0.0)),
            "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
            "val_weighted_f1": float(val_metrics.get("weighted_f1", 0.0)),
            "gw_mean_0": float(gw_means[0]),
            "gw_mean_1": float(gw_means[1]),
            "gw_mean_2": float(gw_means[2]),
            "gw_mean_3": float(gw_means[3]),
            "gw_mean_4": float(gw_means[4]),
            "gw_std_0": float(gw_stds[0]),
            "gw_std_1": float(gw_stds[1]),
            "gw_std_2": float(gw_stds[2]),
            "gw_std_3": float(gw_stds[3]),
            "gw_std_4": float(gw_stds[4]),
            "gate_entropy": float(gate_entropy),
            "lr_head": lr,
            "lr_backbone": backbone_lr,
            "monitor_name": monitor_name,
            "monitor_value": monitor,
            "best_monitor_value": best_score,
            "best_epoch": best_epoch,
            "checkpoint_eligible": int(checkpoint_eligible),
            "best_checkpoint_start_epoch": best_checkpoint_start_epoch,
            "patience": f"{patience_counter}/{patience_limit}",
            "phase_transitioned": int(phase_transitioned),
            "improved": int(improved),
        }
        history.append(row)
        gw_means_str = ",".join([f"{m:.3f}" for m in gw_means])
        print(
            f"Epoch {epoch+1}/{cfg['training']['epochs']} [{time_str}] "
            f"loss={train_loss:.4f} sem_loss={train_sem_loss:.4f} hard_loss={train_hard_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} "
            f"gw_means=[{gw_means_str}] ent={gate_entropy:.3f} "
            f"throughput={train_samples_per_sec:.1f} samples/s "
            f"lr_head={lr:.6f} lr_backbone={backbone_lr:.6f} "
            f"patience={patience_counter}/{patience_limit} "
            f"{monitor_name}={monitor:.4f}",
            flush=True,
        )
        if patience_limit > 0 and patience_counter >= patience_limit:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1} (patience reached {patience_limit})", flush=True)
            break
        gc.collect()

    if history:
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(history[0].keys()) + "\n")
            for row in history:
                f.write(",".join(str(row[k]) for k in history[0].keys()) + "\n")
    else:
        print("[INFO] No new training epochs were run; skipping training_history.csv update.", flush=True)

    best_ckpt = best_manager.latest_checkpoint or last_manager.latest_checkpoint
    if best_ckpt:
        checkpoint.restore(best_ckpt).expect_partial()
        print(f"[INFO] Restored best checkpoint: {best_ckpt}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  FINAL TEST EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # --- No-TTA evaluation ---
    print("\n[INFO] Running final test evaluation (No TTA)...", flush=True)
    no_tta_metrics = evaluate_dataset(model, test_ds, cfg, strategy=eval_strategy, use_tta_hflip=False)
    save_metrics(no_tta_metrics, run_dir / "test_metrics_no_tta.json")
    no_tta_acc = float(no_tta_metrics['accuracy'])
    print(f"\n{'─' * 50}", flush=True)
    print(f"  TEST RESULTS (No TTA)", flush=True)
    print(f"{'─' * 50}", flush=True)
    print(f"  Test Accuracy:    {no_tta_acc * 100:.2f}%", flush=True)
    print(f"  Test Loss:        {float(no_tta_metrics['loss']):.4f}", flush=True)
    print(f"  Macro F1:         {float(no_tta_metrics['macro_f1']):.4f}", flush=True)
    print(f"  Weighted F1:      {float(no_tta_metrics['weighted_f1']):.4f}", flush=True)
    print(f"{'─' * 50}\n", flush=True)

    # --- TTA HFlip evaluation ---
    use_final_tta = bool(cfg["runtime"].get("eval_tta_hflip", False))
    if use_final_tta:
        print("[INFO] Running final test evaluation (TTA HFlip)...", flush=True)
        tta_metrics = evaluate_dataset(model, test_ds, cfg, strategy=eval_strategy, use_tta_hflip=True)
        save_metrics(tta_metrics, run_dir / "test_metrics_tta_hflip.json")
        save_metrics(tta_metrics, run_dir / "test_metrics.json")
        tta_acc = float(tta_metrics['accuracy'])
        print(f"\n{'─' * 50}", flush=True)
        print(f"  TEST RESULTS (TTA HFlip)", flush=True)
        print(f"{'─' * 50}", flush=True)
        print(f"  Test Accuracy:    {tta_acc * 100:.2f}%", flush=True)
        print(f"  Test Loss:        {float(tta_metrics['loss']):.4f}", flush=True)
        print(f"  Macro F1:         {float(tta_metrics['macro_f1']):.4f}", flush=True)
        print(f"  Weighted F1:      {float(tta_metrics['weighted_f1']):.4f}", flush=True)
        print(f"{'─' * 50}", flush=True)
        delta_acc = (tta_acc - no_tta_acc) * 100
        print(f"\n  TTA Improvement: {delta_acc:+.2f}%", flush=True)
    else:
        save_metrics(no_tta_metrics, run_dir / "test_metrics.json")
        tta_metrics = no_tta_metrics

    print("\n" + "=" * 70, flush=True)
    print(f"  Best Epoch: {best_epoch} | Best Val {monitor_name}: {best_score:.4f}", flush=True)
    print("=" * 70 + "\n", flush=True)
    print(json.dumps(tta_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





