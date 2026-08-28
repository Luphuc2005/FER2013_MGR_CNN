from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import train before TensorFlow so the project runtime environment guards run.
from train import (  # noqa: E402
    build_model,
    build_optimizer,
    classification_ce_loss,
    configure_gpus,
    configure_tensorflow_runtime,
    cosine_lr,
    ensure_optimizer_built,
    get_param_count,
    resolve_phase_lrs,
    set_optimizer_lr,
    split_variables,
)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import load_config, global_batch_size  # noqa: E402
from datasets.fer2013 import EMOTION_NAMES, build_datasets  # noqa: E402
from metrics.classification import classification_metrics, save_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train MGR with frozen CNN-anchor error-aware weighted CE.")
    parser.add_argument("--config", default="config_convnext_base_ms1m_arcface_mgr_error_aware.yaml")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve_project_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_checkpoint(path_like: str) -> str:
    path = _resolve_project_path(path_like)
    if path.is_dir():
        latest = tf.train.latest_checkpoint(str(path))
        if latest:
            return latest
    index_file = Path(str(path) + ".index")
    if not index_file.exists():
        raise FileNotFoundError(f"Checkpoint prefix not found: {path} (missing {index_file})")
    return str(path)


def _accuracy_metric_key(monitor_name: str) -> str:
    aliases = {
        "val_accuracy": "accuracy",
        "val_acc": "accuracy",
        "val_macro_f1": "macro_f1",
        "val_weighted_f1": "weighted_f1",
        "val_loss": "loss",
        "val_oracle_accuracy": "oracle_accuracy",
        "val_rescue_count": "rescue_count",
        "val_error_overlap": "error_overlap",
    }
    return aliases.get(monitor_name, monitor_name)


def _monitor_value(metrics: Dict[str, object], monitor_name: str) -> float:
    key = _accuracy_metric_key(monitor_name)
    if key not in metrics:
        raise KeyError(f"Monitor {monitor_name!r} resolved to {key!r}, but metric is absent.")
    value = float(metrics[key])
    return -value if key in {"loss", "error_overlap"} else value


def _weighted_ce_loss(
    labels: tf.Tensor,
    logits: tf.Tensor,
    sample_weights: tf.Tensor,
    *,
    num_classes: int,
    label_smoothing: float,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    logits = tf.cast(logits, tf.float32)
    targets = tf.one_hot(labels, depth=int(num_classes), dtype=tf.float32)
    if label_smoothing > 0.0:
        targets = targets * (1.0 - float(label_smoothing)) + float(label_smoothing) / float(num_classes)
    ce_per_sample = tf.keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
    return tf.reduce_mean(tf.cast(sample_weights, tf.float32) * tf.cast(ce_per_sample, tf.float32))


def _anchor_signals(
    anchor_model: tf.keras.Model,
    features: Dict[str, tf.Tensor],
    labels: tf.Tensor,
    ea_cfg: Dict,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    anchor_outputs = anchor_model(features, training=False)
    anchor_logits = tf.stop_gradient(tf.cast(anchor_outputs["logits"], tf.float32))
    anchor_probs = tf.nn.softmax(anchor_logits, axis=-1)
    anchor_pred = tf.argmax(anchor_probs, axis=-1, output_type=tf.int32)
    anchor_conf = tf.reduce_max(anchor_probs, axis=-1)
    labels = tf.cast(labels, tf.int32)

    correct = tf.equal(anchor_pred, labels)
    uncertain = tf.logical_and(correct, anchor_conf < float(ea_cfg.get("confidence_threshold", 0.70)))
    confident = tf.logical_and(correct, tf.logical_not(uncertain))
    wrong = tf.logical_not(correct)

    wrong_weight = tf.cast(float(ea_cfg.get("wrong_weight", 2.0)), tf.float32)
    uncertain_weight = tf.cast(float(ea_cfg.get("correct_uncertain_weight", 1.0)), tf.float32)
    confident_weight = tf.cast(float(ea_cfg.get("correct_confident_weight", 0.5)), tf.float32)
    weights = tf.where(
        wrong,
        tf.fill(tf.shape(labels), wrong_weight),
        tf.where(uncertain, tf.fill(tf.shape(labels), uncertain_weight), tf.fill(tf.shape(labels), confident_weight)),
    )

    stats = {
        "anchor_correct": tf.reduce_sum(tf.cast(correct, tf.int32)),
        "anchor_wrong": tf.reduce_sum(tf.cast(wrong, tf.int32)),
        "anchor_uncertain": tf.reduce_sum(tf.cast(uncertain, tf.int32)),
        "anchor_confident": tf.reduce_sum(tf.cast(confident, tf.int32)),
        "weight_sum": tf.reduce_sum(weights),
        "count": tf.shape(labels)[0],
    }
    return weights, stats


def _make_train_step(
    cfg: Dict,
    model: tf.keras.Model,
    anchor_model: tf.keras.Model,
    optimizer_head,
    optimizer_backbone,
    head_vars: List[tf.Variable],
    backbone_vars: List[tf.Variable],
    trainable_vars: List[tf.Variable],
    loss_scale: float,
):
    ea_cfg = cfg["error_aware_mgr"]
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    use_sam = str(cfg["training"].get("optimizer", "sam")).lower() == "sam"
    sam_rho = float(cfg["training"].get("sam_rho", 0.03))
    sam_adaptive = bool(cfg["training"].get("sam_adaptive", False))
    skip_nonfinite = bool(cfg["training"].get("skip_nonfinite_batches", True))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0)) if cfg["training"].get("grad_clip_norm") else None
    num_classes = int(cfg["data"]["num_classes"])
    loss_scale_tensor = tf.constant(float(loss_scale), dtype=tf.float32)
    head_count = len([v for v in trainable_vars if v in head_vars])

    def _loss(features, labels, weights):
        outputs = model(features, training=True)
        raw_loss = _weighted_ce_loss(
            labels,
            outputs["logits"],
            weights,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )
        return raw_loss, outputs

    def _clean_grads(grads):
        if skip_nonfinite:
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None for g in grads]
        if grad_clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        return grads

    def _apply(grads):
        grads = _clean_grads(grads)
        head_pairs = [(g, v) for g, v in zip(grads[:head_count], trainable_vars[:head_count]) if g is not None]
        backbone_pairs = [(g, v) for g, v in zip(grads[head_count:], trainable_vars[head_count:]) if g is not None]
        if head_pairs:
            optimizer_head.apply_gradients(head_pairs)
        if optimizer_backbone is not None and backbone_pairs:
            optimizer_backbone.apply_gradients(backbone_pairs)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(features, labels):
        weights, anchor_stats = _anchor_signals(anchor_model, features, labels, ea_cfg)
        with tf.GradientTape() as tape:
            raw_loss, outputs = _loss(features, labels, weights)
            loss = raw_loss * loss_scale_tensor
        grads = tape.gradient(loss, trainable_vars)

        if not use_sam:
            _apply(grads)
            preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.int32))
            return raw_loss, correct, anchor_stats

        grads = _clean_grads(grads)
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
            raw_loss_2, _ = _loss(features, labels, weights)
            loss_2 = raw_loss_2 * loss_scale_tensor
        grads_2 = tape2.gradient(loss_2, trainable_vars)

        for var, eps in zip(trainable_vars, eps_list):
            if eps is not None:
                var.assign_sub(eps)
        _apply(grads_2)

        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.int32))
        return raw_loss, correct, anchor_stats

    return train_step


def _forward_logits(model: tf.keras.Model, inputs: Dict[str, tf.Tensor], *, use_tta_hflip: bool) -> tf.Tensor:
    outputs = model(inputs, training=False)
    logits = tf.cast(outputs["logits"], tf.float32)
    if not use_tta_hflip:
        return logits
    flipped = dict(inputs)
    flipped["image"] = tf.image.flip_left_right(inputs["image"])
    if "mask" in inputs:
        flipped["mask"] = tf.image.flip_left_right(inputs["mask"])
    flipped_outputs = model(flipped, training=False)
    return (logits + tf.cast(flipped_outputs["logits"], tf.float32)) * 0.5


def evaluate_with_anchor(
    model: tf.keras.Model,
    anchor_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    cfg: Dict,
    *,
    use_tta_hflip: bool,
) -> Dict[str, object]:
    y_true: List[int] = []
    y_mgr: List[int] = []
    y_anchor: List[int] = []
    total_loss = 0.0
    total_count = 0
    for inputs, labels in dataset:
        mgr_logits = _forward_logits(model, inputs, use_tta_hflip=use_tta_hflip)
        anchor_logits = _forward_logits(anchor_model, inputs, use_tta_hflip=use_tta_hflip)
        loss = classification_ce_loss(
            labels,
            mgr_logits,
            num_classes=cfg["data"]["num_classes"],
            label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        )
        count = int(tf.shape(labels)[0].numpy())
        total_loss += float(loss.numpy()) * count
        total_count += count
        y_true.extend(labels.numpy().astype(np.int64).tolist())
        y_mgr.extend(tf.argmax(mgr_logits, axis=-1, output_type=tf.int32).numpy().astype(np.int64).tolist())
        y_anchor.extend(tf.argmax(anchor_logits, axis=-1, output_type=tf.int32).numpy().astype(np.int64).tolist())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_mgr_arr = np.asarray(y_mgr, dtype=np.int64)
    y_anchor_arr = np.asarray(y_anchor, dtype=np.int64)
    mgr_correct = y_mgr_arr == y_true_arr
    anchor_correct = y_anchor_arr == y_true_arr

    metrics = classification_metrics(y_true_arr, y_mgr_arr, EMOTION_NAMES)
    both_correct = int(np.sum(mgr_correct & anchor_correct))
    anchor_right_mgr_wrong = int(np.sum(anchor_correct & ~mgr_correct))
    anchor_wrong_mgr_right = int(np.sum(~anchor_correct & mgr_correct))
    both_wrong = int(np.sum(~anchor_correct & ~mgr_correct))
    error_union = int(np.sum(~anchor_correct | ~mgr_correct))
    metrics.update(
        {
            "loss": total_loss / max(total_count, 1),
            "tta_hflip": bool(use_tta_hflip),
            "anchor_accuracy": float(np.mean(anchor_correct)),
            "both_correct": both_correct,
            "cnn_correct_mgr_wrong": anchor_right_mgr_wrong,
            "rescue_count": anchor_wrong_mgr_right,
            "both_wrong": both_wrong,
            "oracle_accuracy": float(np.mean(anchor_correct | mgr_correct)),
            "error_overlap": float(both_wrong / error_union) if error_union else 0.0,
        }
    )
    return metrics


def _weights_checksum(model: tf.keras.Model) -> float:
    total = 0.0
    for variable in model.weights:
        total += float(np.sum(variable.numpy().astype(np.float64)))
    return total


def _restore_anchor(anchor_cfg: Dict, anchor_checkpoint: str, first_inputs: Dict[str, tf.Tensor]) -> tf.keras.Model:
    anchor_model = build_model(anchor_cfg)
    _ = anchor_model(first_inputs, training=False)
    checkpoint = tf.train.Checkpoint(model=anchor_model)
    resolved = _resolve_checkpoint(anchor_checkpoint)
    status = checkpoint.restore(resolved)
    status.assert_existing_objects_matched()
    anchor_model.trainable = False
    for variable in anchor_model.variables:
        variable._trainable = False
    trainable_count = int(sum(np.prod(v.shape) for v in anchor_model.trainable_variables))
    if trainable_count != 0:
        raise RuntimeError(f"CNN anchor still has trainable parameters: {trainable_count}")
    print(f"[CNN_ANCHOR_RESTORE_OK] {resolved}", flush=True)
    print("[CNN_ANCHOR_FROZEN] trainable_parameters=0", flush=True)
    return anchor_model


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ea_cfg = cfg.get("error_aware_mgr", {})
    if not bool(ea_cfg.get("enabled", False)):
        raise RuntimeError("error_aware_mgr.enabled must be true for this script.")
    if cfg["data"].get("mask_dir") in (None, ""):
        raise RuntimeError("Error-aware MGR training requires MediaPipe masks via data.mask_dir.")

    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)
    print(f"TensorFlow {tf.__version__}", flush=True)

    train_ds, val_ds, _ = build_datasets(cfg, replicas=1)
    if global_batch_size(cfg, 1) != int(cfg["runtime"]["batch_size_per_gpu"]):
        raise ValueError("Global batch size mismatch for one-replica error-aware training.")
    first_inputs, first_labels = next(iter(train_ds.take(1)))

    run_dir = Path(cfg["paths"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_dir / "checkpoints"

    anchor_config = ea_cfg.get("anchor_config", "config_convnext_base_ms1m_arcface_baseline.yaml")
    anchor_checkpoint = ea_cfg.get("anchor_checkpoint")
    if not anchor_checkpoint:
        raise KeyError("error_aware_mgr.anchor_checkpoint is required.")
    anchor_cfg = load_config(anchor_config)
    anchor_cfg["runtime"]["use_mixed_precision"] = cfg["runtime"].get("use_mixed_precision", True)

    model = build_model(cfg)
    _ = model(first_inputs, training=False)
    anchor_model = _restore_anchor(anchor_cfg, anchor_checkpoint, first_inputs)
    anchor_checksum_before = _weights_checksum(anchor_model)

    smoke_weights, smoke_anchor_stats = _anchor_signals(anchor_model, first_inputs, first_labels, ea_cfg)
    smoke_outputs = model(first_inputs, training=False)
    smoke_loss = _weighted_ce_loss(
        first_labels,
        smoke_outputs["logits"],
        smoke_weights,
        num_classes=cfg["data"]["num_classes"],
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
    )
    if not np.isfinite(float(smoke_loss.numpy())):
        raise FloatingPointError("Error-aware smoke weighted CE is not finite.")
    print(
        "[ERROR_AWARE_SMOKE] "
        f"weighted_ce={float(smoke_loss.numpy()):.6f} "
        f"anchor_wrong={int(smoke_anchor_stats['anchor_wrong'].numpy())} "
        f"avg_weight={float(tf.reduce_mean(smoke_weights).numpy()):.6f}",
        flush=True,
    )

    total_params, trainable_params = get_param_count(model)
    print(f"Model params: total={total_params:,}, trainable={trainable_params:,}", flush=True)

    optimizer_head = build_optimizer(cfg, float(cfg["training"]["lr"]))
    optimizer_backbone = build_optimizer(cfg, float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])))
    backbone_vars, head_vars = split_variables(model)
    ensure_optimizer_built(optimizer_head, head_vars)
    ensure_optimizer_built(optimizer_backbone, backbone_vars)

    ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt_best_metric = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
    checkpoint = tf.train.Checkpoint(
        epoch=ckpt_epoch,
        best_metric=ckpt_best_metric,
        model=model,
        optimizer_head=optimizer_head,
        optimizer_backbone=optimizer_backbone,
    )
    last_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "last"), max_to_keep=1)
    best_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "best"), max_to_keep=1)
    periodic_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "periodic"), max_to_keep=5)
    if (args.resume or cfg["training"].get("resume", True)) and last_manager.latest_checkpoint:
        checkpoint.restore(last_manager.latest_checkpoint).expect_partial()
        print(f"Resumed MGR from {last_manager.latest_checkpoint}", flush=True)

    loss_scale = 1.0
    train_step_head = _make_train_step(
        cfg, model, anchor_model, optimizer_head, optimizer_backbone, head_vars, backbone_vars, head_vars, loss_scale
    )
    train_step_full = _make_train_step(
        cfg,
        model,
        anchor_model,
        optimizer_head,
        optimizer_backbone,
        head_vars,
        backbone_vars,
        head_vars + backbone_vars,
        loss_scale,
    )

    monitor_name = str(cfg.get("checkpoint", {}).get("monitor", cfg["training"].get("monitor", "val_oracle_accuracy")))
    tie_break_name = str(cfg.get("checkpoint", {}).get("tie_break", "val_rescue_count"))
    best_score = float(ckpt_best_metric.numpy())
    best_tie = -float("inf")
    best_epoch = int(ckpt_epoch.numpy()) if best_score >= 0.0 else -1
    patience_anchor_epoch = best_epoch if best_score >= 0.0 else 0
    history = []
    csv_path = run_dir / "training_history.csv"
    periodic_interval = int(cfg["training"].get("periodic_checkpoint_interval", 10) or 0)
    progress_interval = int(cfg["training"].get("progress_interval", 0) or 0)
    freeze_epochs = int(cfg["model"].get("freeze_backbone_epochs", 0) or 0)
    start_epoch = int(ckpt_epoch.numpy())

    print(f"[CHECKPOINT_MONITOR] monitor={monitor_name}, tie_break={tie_break_name}", flush=True)
    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        epoch_start = time.time()
        train_backbone = bool(cfg["model"].get("unfreeze_backbone", True)) and epoch >= freeze_epochs
        train_step = train_step_full if train_backbone else train_step_head
        lr, backbone_lr = resolve_phase_lrs(cfg, epoch, train_backbone)
        set_optimizer_lr(optimizer_head, lr)
        set_optimizer_lr(optimizer_backbone, backbone_lr)

        losses = []
        correct = 0
        seen = 0
        anchor_correct = 0
        anchor_wrong = 0
        anchor_uncertain = 0
        anchor_confident = 0
        weight_sum = 0.0
        total_steps = int(tf.data.experimental.cardinality(train_ds).numpy())

        for step_index, (features, labels) in enumerate(train_ds, start=1):
            loss, batch_correct, stats = train_step(features, labels)
            batch_count = int(stats["count"].numpy())
            losses.append(float(loss.numpy()))
            correct += int(batch_correct.numpy())
            seen += batch_count
            anchor_correct += int(stats["anchor_correct"].numpy())
            anchor_wrong += int(stats["anchor_wrong"].numpy())
            anchor_uncertain += int(stats["anchor_uncertain"].numpy())
            anchor_confident += int(stats["anchor_confident"].numpy())
            weight_sum += float(stats["weight_sum"].numpy())
            if progress_interval and step_index % progress_interval == 0:
                print(
                    f"Epoch {epoch+1}/{cfg['training']['epochs']} step {step_index}/{total_steps} "
                    f"weighted_ce={float(loss.numpy()):.4f} mgr_acc={correct / max(seen, 1):.4f} "
                    f"avg_weight={weight_sum / max(seen, 1):.4f}",
                    flush=True,
                )

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = correct / max(seen, 1)
        anchor_train_acc = anchor_correct / max(seen, 1)
        wrong_rate = anchor_wrong / max(seen, 1)
        uncertain_rate = anchor_uncertain / max(seen, 1)
        confident_rate = anchor_confident / max(seen, 1)
        avg_weight = weight_sum / max(seen, 1)

        print(f"[INFO] Epoch {epoch+1}: starting validation oracle metrics", flush=True)
        val_metrics = evaluate_with_anchor(
            model,
            anchor_model,
            val_ds,
            cfg,
            use_tta_hflip=bool(cfg["runtime"].get("train_val_tta_hflip", False)),
        )
        print(f"[INFO] Epoch {epoch+1}: validation finished", flush=True)

        monitor = _monitor_value(val_metrics, monitor_name)
        tie_break = _monitor_value(val_metrics, tie_break_name)
        checkpoint_eligible = (epoch + 1) >= int(cfg["training"].get("best_checkpoint_start_epoch", 1) or 1)
        improved = bool(checkpoint_eligible and (monitor > best_score or (monitor == best_score and tie_break > best_tie)))
        if improved:
            best_score = monitor
            best_tie = tie_break
            best_epoch = epoch + 1
            patience_anchor_epoch = epoch + 1
            ckpt_best_metric.assign(best_score)
            best_manager.save(checkpoint_number=epoch + 1)
            print(
                f"[INFO] Save best at ep {epoch+1}, {monitor_name}={monitor:.6f}, "
                f"{tie_break_name}={tie_break:.6f}",
                flush=True,
            )

        ckpt_epoch.assign(epoch + 1)
        last_manager.save(checkpoint_number=epoch + 1)
        if periodic_interval and (epoch + 1) % periodic_interval == 0:
            periodic_manager.save(checkpoint_number=epoch + 1)

        patience_limit = int(cfg["training"].get("patience", 75))
        patience_counter = 0 if not checkpoint_eligible else (epoch + 1) - patience_anchor_epoch
        row = {
            "epoch": epoch + 1,
            "epoch_time_sec": round(time.time() - epoch_start, 2),
            "train_weighted_ce": train_loss,
            "train_accuracy": train_acc,
            "train_cnn_anchor_accuracy": anchor_train_acc,
            "train_cnn_wrong_rate": wrong_rate,
            "train_cnn_correct_uncertain_rate": uncertain_rate,
            "train_cnn_correct_confident_rate": confident_rate,
            "train_avg_sample_weight": avg_weight,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_anchor_accuracy": float(val_metrics["anchor_accuracy"]),
            "val_both_correct": int(val_metrics["both_correct"]),
            "val_cnn_correct_mgr_wrong": int(val_metrics["cnn_correct_mgr_wrong"]),
            "val_rescue_count": int(val_metrics["rescue_count"]),
            "val_both_wrong": int(val_metrics["both_wrong"]),
            "val_oracle_accuracy": float(val_metrics["oracle_accuracy"]),
            "val_error_overlap": float(val_metrics["error_overlap"]),
            "lr_head": lr,
            "lr_backbone": backbone_lr,
            "monitor_name": monitor_name,
            "monitor_value": monitor,
            "best_monitor_value": best_score,
            "best_epoch": best_epoch,
            "patience": f"{patience_counter}/{patience_limit}",
            "improved": int(improved),
        }
        history.append(row)
        print(
            f"Epoch {epoch+1}/{cfg['training']['epochs']} "
            f"loss={train_loss:.4f} mgr_train_acc={train_acc:.4f} "
            f"cnn_anchor_train_acc={anchor_train_acc:.4f} cnn_wrong={wrong_rate:.4f} "
            f"uncertain={uncertain_rate:.4f} confident={confident_rate:.4f} avg_w={avg_weight:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_rescue={row['val_rescue_count']} val_oracle={row['val_oracle_accuracy']:.4f} "
            f"val_overlap={row['val_error_overlap']:.4f} patience={row['patience']}",
            flush=True,
        )
        save_metrics(val_metrics, run_dir / "val_metrics_last.json")
        if patience_limit > 0 and patience_counter >= patience_limit:
            print(f"[INFO] Early stopping at epoch {epoch+1}", flush=True)
            break
        gc.collect()

    if history:
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(history[0].keys()) + "\n")
            for row in history:
                f.write(",".join(str(row[k]) for k in history[0].keys()) + "\n")

    anchor_checksum_after = _weights_checksum(anchor_model)
    checksum_delta = abs(anchor_checksum_after - anchor_checksum_before)
    print(
        f"[CNN_ANCHOR_UNCHANGED] checksum_before={anchor_checksum_before:.8f} "
        f"checksum_after={anchor_checksum_after:.8f} delta={checksum_delta:.12f}",
        flush=True,
    )
    if checksum_delta > float(ea_cfg.get("anchor_checksum_tolerance", 1e-8)):
        raise RuntimeError("CNN anchor weights changed during training.")

    summary = {
        "best_epoch": best_epoch,
        "best_monitor": monitor_name,
        "best_monitor_value": best_score,
        "best_checkpoint": best_manager.latest_checkpoint,
        "cnn_anchor_checksum_delta": checksum_delta,
        "test_leakage": False,
    }
    with (run_dir / "error_aware_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
