"""
Train RGB ConvNeXt-Base MS1M + SMIRK Geometry 3D-guided residual channel
attention from the FER task start, without restoring any FER checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from models.dual_convnext_smirk_guided_attention_ms1m_fer_scratch import (
    DualConvNeXtSMIRKGuidedAttentionMS1MFERScratch,
    count_params,
)

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual ConvNeXt MS1M + SMIRK guided attention without FER ckpt restore.")
    parser.add_argument("--config", type=str, default="config_dual_convnext_smirk_guided_attention_ms1m_fer_scratch.yaml")
    parser.add_argument("--multi-gpu", action="store_true", help="Use MirroredStrategy across visible GPUs.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run smoke/dry-run contract checks and exit before training.")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip dry-run checks. Intended only for emergency debugging.")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def assert_no_fer_checkpoint_restore_config(cfg: Dict) -> None:
    serialized = json.dumps(cfg, sort_keys=True)
    forbidden = ("ckpt-43", "baseline_checkpoint_path", "best_checkpoint_dir", "restore_checkpoint")
    hits = [token for token in forbidden if token in serialized]
    if hits:
        raise RuntimeError(f"FER checkpoint restore config is forbidden for this experiment. Found tokens={hits}")
    print("NO_FER_CHECKPOINT_RESTORE_CONFIG_OK", flush=True)


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

    if len(images) != len(geom_maps):
        raise RuntimeError(f"Mismatch: images={len(images)}, geom_maps={len(geom_maps)}")
    if not (labels == cache_dict["labels"]).all():
        raise RuntimeError("Label alignment check failed between FER split and geometry cache.")

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
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if is_training:
        ds = ds.shuffle(buffer_size=min(num_samples, 2048), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=is_training)

    def batch_mapper(item, label):
        return {
            "image": preprocess_batch_images(item["image"], target_size=112),
            "geometry_maps": item["geometry_maps"],
        }, label

    ds = ds.map(batch_mapper, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return ds.with_options(options)


def unique_vars(variables: Iterable[tf.Variable]) -> List[tf.Variable]:
    seen = set()
    out = []
    for var in variables:
        key = id(var)
        if key not in seen:
            seen.add(key)
            out.append(var)
    return out


def stage_vars(baseline, stage: int) -> List[tf.Variable]:
    backbone = baseline.backbone
    if stage == 1:
        return unique_vars([v for block in backbone.stages[0] for v in block.variables])
    if stage == 2:
        return unique_vars(list(backbone.downsample_layers[0].variables) + [v for block in backbone.stages[1] for v in block.variables])
    if stage == 3:
        return unique_vars(list(backbone.downsample_layers[1].variables) + [v for block in backbone.stages[2] for v in block.variables])
    if stage == 4:
        return unique_vars(list(backbone.downsample_layers[2].variables) + [v for block in backbone.stages[3] for v in block.variables])
    raise ValueError(f"Unsupported stage={stage}")


def stem_vars(baseline) -> List[tf.Variable]:
    b = baseline.backbone
    return unique_vars(list(b.stem_conv.variables) + list(b.stem_norm.variables))


def fer_head_vars(model) -> List[tf.Variable]:
    return unique_vars(list(model.rgb_baseline.classifier.variables))


def trainable_group_vars_for_phase(model, phase: int) -> Dict[str, List[tf.Variable]]:
    groups = {
        "fer_head": fer_head_vars(model),
        "fusion": list(model.geometry_fusion.trainable_variables),
        "attention": list(model.channel_attention_mlp.trainable_variables),
        "aux_head": list(model.aux_3d_head.trainable_variables),
        "alpha_raw": [model.alpha_raw],
        "rgb_stage3": [],
        "rgb_stage4": [],
        "geometry_stage3": [],
        "geometry_stage4": stage_vars(model.geometry_baseline, 4),
    }
    if phase >= 2:
        groups["rgb_stage4"] = stage_vars(model.rgb_baseline, 4)
    if phase >= 3:
        groups["rgb_stage3"] = stage_vars(model.rgb_baseline, 3)
        groups["geometry_stage3"] = stage_vars(model.geometry_baseline, 3)
    return {name: unique_vars(vars_) for name, vars_ in groups.items()}


def set_trainable_flags(model, phase: int) -> None:
    for layer in model.rgb_baseline.layers:
        layer.trainable = False
    for layer in model.geometry_baseline.layers:
        layer.trainable = False

    model.rgb_baseline.classifier.trainable = True
    model.rgb_baseline.head_dropout.trainable = True
    model.geometry_fusion.trainable = True
    model.channel_attention_mlp.trainable = True
    model.aux_3d_head.trainable = True

    for block in model.geometry_baseline.backbone.stages[3]:
        block.trainable = True
    model.geometry_baseline.backbone.downsample_layers[2].trainable = True

    if phase >= 2:
        for block in model.rgb_baseline.backbone.stages[3]:
            block.trainable = True
        model.rgb_baseline.backbone.downsample_layers[2].trainable = True

    if phase >= 3:
        for block in model.rgb_baseline.backbone.stages[2]:
            block.trainable = True
        model.rgb_baseline.backbone.downsample_layers[1].trainable = True
        for block in model.geometry_baseline.backbone.stages[2]:
            block.trainable = True
        model.geometry_baseline.backbone.downsample_layers[1].trainable = True


def phase_for_epoch(epoch: int, cfg: Dict) -> int:
    stages = cfg["training"].get("stages", {})
    if epoch <= int(stages.get("phase1_end_epoch", 5)):
        return 1
    if epoch <= int(stages.get("phase2_end_epoch", 20)):
        return 2
    return 3


def assert_group_contract(model, groups: Dict[str, List[tf.Variable]], phase: int) -> None:
    group_items = [(name, var) for name, vars_ in groups.items() for var in vars_]
    seen = {}
    for name, var in group_items:
        key = id(var)
        if key in seen:
            raise RuntimeError(f"Variable appears in multiple groups: {var.name} in {seen[key]} and {name}")
        seen[key] = name

    forbidden_rgb = []
    for name in ("fer_head", "fusion", "attention", "aux_head", "alpha_raw"):
        for var in groups[name]:
            if "convnext_base_fr_backbone" in var.name:
                forbidden_rgb.append((name, var.name))
    if forbidden_rgb:
        raise RuntimeError(f"Backbone variables leaked into head groups: {forbidden_rgb[:5]}")

    if phase == 1 and groups["rgb_stage3"] + groups["rgb_stage4"]:
        raise RuntimeError("Phase 1 must not include RGB backbone optimizer variables.")
    print(f"VARIABLE_GROUP_CONTRACT_OK phase={phase}", flush=True)


def print_phase_params(model, cfg: Dict) -> None:
    print("\n" + "=" * 72, flush=True)
    print("TRAINABLE PARAMS BY PHASE", flush=True)
    print("=" * 72, flush=True)
    for phase in (1, 2, 3):
        set_trainable_flags(model, phase)
        groups = trainable_group_vars_for_phase(model, phase)
        assert_group_contract(model, groups, phase)
        print(f"PHASE_{phase}_PARAMS", flush=True)
        print(f"  rgb_stem_trainable_params: 0", flush=True)
        print(f"  rgb_stage1_trainable_params: 0", flush=True)
        print(f"  rgb_stage2_trainable_params: 0", flush=True)
        print(f"  rgb_stage3_trainable_params: {count_params(groups['rgb_stage3']):,}", flush=True)
        print(f"  rgb_stage4_trainable_params: {count_params(groups['rgb_stage4']):,}", flush=True)
        print(f"  geometry_stem_trainable_params: 0", flush=True)
        print(f"  geometry_stage1_trainable_params: 0", flush=True)
        print(f"  geometry_stage2_trainable_params: 0", flush=True)
        print(f"  geometry_stage3_trainable_params: {count_params(groups['geometry_stage3']):,}", flush=True)
        print(f"  geometry_stage4_trainable_params: {count_params(groups['geometry_stage4']):,}", flush=True)
        print(f"  fer_head_trainable_params: {count_params(groups['fer_head']):,}", flush=True)
        print(f"  fusion_trainable_params: {count_params(groups['fusion']):,}", flush=True)
        print(f"  attention_trainable_params: {count_params(groups['attention']):,}", flush=True)
        print(f"  aux_head_trainable_params: {count_params(groups['aux_head']):,}", flush=True)
        print(f"  alpha_raw_trainable_params: {count_params(groups['alpha_raw']):,}", flush=True)
    set_trainable_flags(model, 1)
    print("=" * 72 + "\n", flush=True)


def make_adamw(lr: float, weight_decay: float):
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)
    if adamw is None:
        raise RuntimeError("AdamW optimizer is required for this experiment.")
    try:
        return adamw(learning_rate=lr, weight_decay=weight_decay, jit_compile=False)
    except TypeError:
        return adamw(learning_rate=lr, weight_decay=weight_decay)


def build_optimizers(cfg: Dict) -> Dict[str, tf.keras.mixed_precision.LossScaleOptimizer]:
    lr = cfg["training"]["learning_rates"]
    wd = float(cfg["training"].get("weight_decay", 0.01))
    group_lrs = {
        "fer_head": float(lr["head"]),
        "fusion": float(lr["head"]),
        "attention": float(lr["head"]),
        "aux_head": float(lr["head"]),
        "alpha_raw": float(lr["alpha_raw"]),
        "rgb_stage4": float(lr["rgb_stage4"]),
        "geometry_stage4": float(lr["geometry_stage4"]),
        "rgb_stage3": float(lr["rgb_stage3"]),
        "geometry_stage3": float(lr["geometry_stage3"]),
    }
    if group_lrs["alpha_raw"] > 1e-4:
        raise RuntimeError(f"alpha_raw LR must be <= 1e-4, got {group_lrs['alpha_raw']}")
    print("OPTIMIZER_GROUP_LR_CONTRACT", flush=True)
    for name, value in group_lrs.items():
        print(f"  {name}: AdamW lr={value:.8g} weight_decay={wd}", flush=True)
    return {
        name: tf.keras.mixed_precision.LossScaleOptimizer(make_adamw(value, wd))
        for name, value in group_lrs.items()
    }


def group_learning_rates(cfg: Dict) -> Dict[str, float]:
    lr = cfg["training"]["learning_rates"]
    return {
        "fer_head": float(lr["head"]),
        "fusion": float(lr["head"]),
        "attention": float(lr["head"]),
        "aux_head": float(lr["head"]),
        "alpha_raw": float(lr["alpha_raw"]),
        "rgb_stage4": float(lr["rgb_stage4"]),
        "geometry_stage4": float(lr["geometry_stage4"]),
        "rgb_stage3": float(lr["rgb_stage3"]),
        "geometry_stage3": float(lr["geometry_stage3"]),
    }


def compute_distributed_ce_loss(loss_fn, labels, logits, global_batch_size: int) -> tf.Tensor:
    per_example_loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

@tf.function(reduce_retracing=True)
def train_step(model, optimizers, groups, inputs, labels, loss_fn, loss_weight_3d: float, grad_clip_norm: float, global_batch_size: int):
    active_vars = unique_vars([var for vars_ in groups.values() for var in vars_])
    if not active_vars:
        raise RuntimeError("No active variables for train_step.")
    loss_scale_optimizer = next(opt for name, opt in optimizers.items() if groups.get(name))

    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        final_logits = outputs["final_logits"]
        aux_logits = outputs["aux_3d_logits"]
        l_final = compute_distributed_ce_loss(loss_fn, labels, final_logits, global_batch_size)
        l_aux = compute_distributed_ce_loss(loss_fn, labels, aux_logits, global_batch_size)
        total_loss = l_final + loss_weight_3d * l_aux
        scaled_loss = loss_scale_optimizer.get_scaled_loss(total_loss)

    scaled_grads = tape.gradient(scaled_loss, active_vars)
    grads = loss_scale_optimizer.get_unscaled_gradients(scaled_grads)
    valid = [(g, v) for g, v in zip(grads, active_vars) if g is not None]
    if not valid:
        raise RuntimeError("No valid gradients produced for active variables.")

    valid_grads, valid_vars = zip(*valid)
    grad_norm_before_clip = tf.linalg.global_norm(valid_grads)
    clipped_grads, _ = tf.clip_by_global_norm(valid_grads, grad_clip_norm)
    clipped_by_var = {id(var): grad for grad, var in zip(clipped_grads, valid_vars)}
    raw_by_var = {id(var): grad for grad, var in zip(valid_grads, valid_vars)}

    grad_norms = {}
    for name, variables in groups.items():
        group_pairs = [(clipped_by_var[id(var)], var) for var in variables if id(var) in clipped_by_var]
        if not group_pairs:
            grad_norms[name] = tf.constant(0.0, dtype=tf.float32)
            continue
        group_raw_grads = [raw_by_var[id(var)] for var in variables if id(var) in raw_by_var]
        grad_norms[name] = tf.cast(tf.linalg.global_norm(group_raw_grads), tf.float32)
        optimizers[name].apply_gradients(group_pairs)

    preds = tf.argmax(final_logits, axis=1, output_type=labels.dtype)
    aux_preds = tf.argmax(aux_logits, axis=1, output_type=labels.dtype)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
    aux_acc = tf.reduce_mean(tf.cast(tf.equal(aux_preds, labels), tf.float32))
    return {
        "loss": total_loss,
        "accuracy": acc,
        "aux_accuracy": aux_acc,
        "grad_norm": tf.cast(grad_norm_before_clip, tf.float32),
        "alpha_raw": tf.cast(outputs["alpha_raw"], tf.float32),
        "effective_alpha": tf.cast(outputs["effective_alpha"], tf.float32),
        "mean_abs_channel_gate": tf.cast(outputs["mean_abs_channel_gate"], tf.float32),
        "modulation_factor_min": tf.cast(outputs["modulation_factor_min"], tf.float32),
        "modulation_factor_max": tf.cast(outputs["modulation_factor_max"], tf.float32),
        "group_grad_norms": grad_norms,
    }


def distributed_train_step(strategy, model, optimizers, groups, dist_batch, loss_fn, loss_weight_3d: float, grad_clip_norm: float, global_batch_size: int):
    def replica_step(inputs, labels):
        return train_step(model, optimizers, groups, inputs, labels, loss_fn, loss_weight_3d, grad_clip_norm, global_batch_size)

    per_replica = strategy.run(replica_step, args=dist_batch)
    reduced = {}
    for key, value in per_replica.items():
        if key == "group_grad_norms":
            reduced[key] = {
                name: strategy.reduce(tf.distribute.ReduceOp.MEAN, group_value, axis=None)
                for name, group_value in value.items()
            }
        elif key == "loss":
            reduced[key] = strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None)
        else:
            reduced[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
    return reduced


def evaluate_model(model, dataset: tf.data.Dataset, loss_weight_3d: float) -> Dict:
    all_logits = []
    all_aux_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    gate_values = []
    mod_mins = []
    mod_maxs = []
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for inputs, y_batch in dataset:
        outputs = model(inputs, training=False)
        logits = outputs["final_logits"]
        aux_logits = outputs["aux_3d_logits"]
        l_final = loss_fn(y_batch, logits)
        l_aux = loss_fn(y_batch, aux_logits)
        batch_loss = l_final + loss_weight_3d * l_aux
        batch_size = int(tf.shape(y_batch)[0])
        total_loss += float(batch_loss.numpy()) * batch_size
        total_samples += batch_size
        all_logits.append(logits.numpy())
        all_aux_logits.append(aux_logits.numpy())
        all_labels.append(y_batch.numpy())
        gate_values.append(float(outputs["mean_abs_channel_gate"].numpy()))
        mod_mins.append(float(outputs["modulation_factor_min"].numpy()))
        mod_maxs.append(float(outputs["modulation_factor_max"].numpy()))

    labels_arr = np.concatenate(all_labels, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    aux_logits_arr = np.concatenate(all_aux_logits, axis=0)
    preds = np.argmax(logits_arr, axis=1)
    aux_preds = np.argmax(aux_logits_arr, axis=1)
    report = classification_report(
        labels_arr,
        preds,
        labels=list(range(len(EMOTION_NAMES))),
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": float(total_loss / max(1, total_samples)),
        "accuracy": float(np.mean(preds == labels_arr)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "aux_accuracy": float(np.mean(aux_preds == labels_arr)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels_arr, preds, labels=list(range(7))).tolist(),
        "alpha_raw": float(model.alpha_raw.numpy()),
        "effective_alpha": float(model.effective_alpha.numpy()),
        "mean_abs_channel_gate": float(np.mean(gate_values)) if gate_values else 0.0,
        "modulation_factor_min": float(np.min(mod_mins)) if mod_mins else 1.0,
        "modulation_factor_max": float(np.max(mod_maxs)) if mod_maxs else 1.0,
    }


def max_abs_diff(before: List[np.ndarray], variables: List[tf.Variable]) -> float:
    if not before or not variables:
        return 0.0
    return max(float(np.max(np.abs(a - v.numpy()))) for a, v in zip(before, variables))


def run_baseline_equivalence_smoke(model, inputs) -> None:
    model.alpha_raw.assign(0.0)
    baseline_logits = model.rgb_baseline(inputs["image"], training=False)["logits"].numpy()
    outputs = model(inputs, training=False)
    feat_diff = float(np.max(np.abs(outputs["F_guided"].numpy() - outputs["F_rgb"].numpy())))
    logits_diff = float(np.max(np.abs(baseline_logits - outputs["final_logits"].numpy())))
    print("BASELINE_EQUIVALENCE_SMOKE", flush=True)
    print(f"  max_abs(F_guided - F_rgb): {feat_diff:.8e}", flush=True)
    print(f"  max_abs(rgb_baseline_logits - dual_logits): {logits_diff:.8e}", flush=True)
    if feat_diff >= 1e-7 or logits_diff >= 1e-7:
        raise RuntimeError(f"Baseline equivalence failed: feat_diff={feat_diff:.8e}, logits_diff={logits_diff:.8e}")
    print("BASELINE_EQUIVALENCE_SMOKE_OK tolerance=1e-7", flush=True)


def run_dry_run(model, cfg: Dict, train_ds: tf.data.Dataset, strategy) -> None:
    print("\n" + "=" * 72, flush=True)
    print("DRY_RUN_PHASE1_CONTRACT", flush=True)
    print("=" * 72, flush=True)
    set_trainable_flags(model, phase=1)
    groups = trainable_group_vars_for_phase(model, phase=1)
    assert_group_contract(model, groups, phase=1)
    optimizers = build_optimizers(cfg)
    build_all_optimizer_slots(model, optimizers)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    loss_weight_3d = float(cfg["training"].get("loss_weight_3d", 0.1))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))
    if grad_clip_norm <= 0.0:
        raise RuntimeError("grad_clip_norm must be > 0 and is required for this experiment.")
    print(f"GRAD_CLIP_ENABLED global_norm={grad_clip_norm}", flush=True)

    rgb_backbone_vars = stem_vars(model.rgb_baseline) + stage_vars(model.rgb_baseline, 1) + stage_vars(model.rgb_baseline, 2) + stage_vars(model.rgb_baseline, 3) + stage_vars(model.rgb_baseline, 4)
    rgb_before = [v.numpy().copy() for v in rgb_backbone_vars]
    geom_frozen_vars = stem_vars(model.geometry_baseline) + stage_vars(model.geometry_baseline, 1) + stage_vars(model.geometry_baseline, 2) + stage_vars(model.geometry_baseline, 3)
    geom_frozen_before = [v.numpy().copy() for v in geom_frozen_vars]
    fer_head_before = [v.numpy().copy() for v in groups["fer_head"]]

    dry_batches = int(cfg["training"].get("dry_run_batches", 3))
    train_iter = strategy.experimental_distribute_dataset(train_ds)
    for batch_idx, batch in enumerate(train_iter, start=1):
        if batch_idx > dry_batches:
            break
        if strategy.num_replicas_in_sync > 1:
            metrics = distributed_train_step(strategy, model, optimizers, groups, batch, loss_fn, loss_weight_3d, grad_clip_norm, int(cfg["data"]["batch_size"]))
        else:
            inputs, labels = batch
            metrics = train_step(model, optimizers, groups, inputs, labels, loss_fn, loss_weight_3d, grad_clip_norm, int(cfg["data"]["batch_size"]))
        print(
            f"DRY_RUN_BATCH {batch_idx} "
            f"loss={float(metrics['loss'].numpy()):.6f} "
            f"acc={float(metrics['accuracy'].numpy()):.6f} "
            f"aux_acc={float(metrics['aux_accuracy'].numpy()):.6f} "
            f"grad_norm_before_clip={float(metrics['grad_norm'].numpy()):.6f} "
            f"alpha_raw={float(metrics['alpha_raw'].numpy()):.8f} "
            f"effective_alpha={float(metrics['effective_alpha'].numpy()):.8f} "
            f"mean_abs_channel_gate={float(metrics['mean_abs_channel_gate'].numpy()):.8f} "
            f"modulation_min={float(metrics['modulation_factor_min'].numpy()):.8f} "
            f"modulation_max={float(metrics['modulation_factor_max'].numpy()):.8f}",
            flush=True,
        )

    rgb_diff = max_abs_diff(rgb_before, rgb_backbone_vars)
    geom_frozen_diff = max_abs_diff(geom_frozen_before, geom_frozen_vars)
    fer_head_diff = max_abs_diff(fer_head_before, groups["fer_head"])
    print(f"DRY_RUN_RGB_BACKBONE_MAX_DIFF phase1_expected_0={rgb_diff:.8e}", flush=True)
    print(f"DRY_RUN_GEOMETRY_STEM_STAGE1_2_3_MAX_DIFF phase1_expected_0={geom_frozen_diff:.8e}", flush=True)
    print(f"DRY_RUN_FER_HEAD_MAX_DIFF phase1_expected_trainable={fer_head_diff:.8e}", flush=True)
    if rgb_diff != 0.0:
        raise RuntimeError(f"Phase 1 RGB backbone changed during dry-run: max_diff={rgb_diff:.8e}")
    if geom_frozen_diff != 0.0:
        raise RuntimeError(f"Phase 1 frozen geometry variables changed during dry-run: max_diff={geom_frozen_diff:.8e}")
    print("DRY_RUN_PHASE1_CONTRACT_OK", flush=True)


def build_all_optimizer_slots(model, optimizers) -> None:
    groups = trainable_group_vars_for_phase(model, phase=3)
    for name, variables in groups.items():
        if not variables:
            continue
        inner = getattr(optimizers[name], "inner_optimizer", getattr(optimizers[name], "_optimizer", optimizers[name]))
        if hasattr(inner, "build"):
            inner.build(variables)
        elif hasattr(inner, "_create_all_weights"):
            inner._create_all_weights(variables)
    print("OPTIMIZER_SLOT_BUILD_OK all_phase3_groups", flush=True)

def print_epoch_lr_contract(cfg: Dict, phase: int) -> None:
    lrs = group_learning_rates(cfg)
    active = {
        1: ["fer_head", "fusion", "attention", "aux_head", "alpha_raw", "geometry_stage4"],
        2: ["fer_head", "fusion", "attention", "aux_head", "alpha_raw", "rgb_stage4", "geometry_stage4"],
        3: ["fer_head", "fusion", "attention", "aux_head", "alpha_raw", "rgb_stage3", "rgb_stage4", "geometry_stage3", "geometry_stage4"],
    }[phase]
    print("ACTIVE_LR_GROUPS " + " ".join(f"{name}={lrs[name]:.8g}" for name in active), flush=True)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    assert_no_fer_checkpoint_restore_config(cfg)
    set_seed(int(cfg.get("seed", 42)))

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"[WARNING] Could not set memory growth for {gpu}: {exc}", flush=True)

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"MIXED_PRECISION_POLICY={tf.keras.mixed_precision.global_policy().name}", flush=True)
    print(f"TENSORFLOW_VERSION={tf.__version__}", flush=True)
    print(f"VISIBLE_GPU_COUNT={len(gpus)}", flush=True)

    if args.multi_gpu:
        if len(gpus) < 2:
            raise RuntimeError(f"--multi-gpu requested but TensorFlow sees only {len(gpus)} GPU(s).")
        strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{idx}" for idx in range(len(gpus))])
        print(f"MIRRORED_STRATEGY_OK replicas={strategy.num_replicas_in_sync}", flush=True)
    else:
        strategy = tf.distribute.get_strategy()
        print(f"DEFAULT_STRATEGY replicas={strategy.num_replicas_in_sync}", flush=True)

    output_dir = resolve_path(cfg["paths"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints" / "best"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(cfg["data"]["data_path"])
    cache_dir = resolve_path(cfg["geometry_cache"]["feature_dir"])
    pattern = cfg["geometry_cache"]["map_file_pattern"]
    batch_size = int(cfg["data"]["batch_size"])

    print("[INFO] Loading FER2013 train/val records and SMIRK geometry maps...", flush=True)
    train_records = collect_split_records(data_path, "train", predecode_pixels=True)
    val_records = collect_split_records(data_path, "val", predecode_pixels=True)
    train_cache = load_geometry_cache(cache_dir, pattern, "train")
    val_cache = load_geometry_cache(cache_dir, pattern, "val")
    train_ds = create_dataset(train_records, train_cache, batch_size, is_training=True)
    val_ds = create_dataset(val_records, val_cache, batch_size, is_training=False)

    with strategy.scope():
        model = DualConvNeXtSMIRKGuidedAttentionMS1MFERScratch(cfg)
        # Build the random FER classifier first, then load only ConvNeXt backbones.
        model.rgb_baseline._build_variables()
        model.geometry_baseline._build_variables()
        model.load_ms1m_pretrained_weights(cfg)

        first_inputs, _ = next(iter(train_ds))
        _ = model(first_inputs, training=False)
        run_baseline_equivalence_smoke(model, first_inputs)
        print_phase_params(model, cfg)

        initial_weights = model.get_weights()
        if not args.skip_dry_run:
            run_dry_run(model, cfg, train_ds, strategy)
            model.set_weights(initial_weights)
            print("DRY_RUN_WEIGHTS_RESTORED_BEFORE_REAL_TRAINING_OK", flush=True)

        if args.smoke_test_only:
            print("[INFO] --smoke-test-only requested. Exiting before real training.", flush=True)
            return 0

        epochs = int(cfg["training"]["epochs"])
        patience = int(cfg["training"]["patience"])
        loss_weight_3d = float(cfg["training"].get("loss_weight_3d", 0.1))
        grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizers = build_optimizers(cfg)

        build_all_optimizer_slots(model, optimizers)
    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)

    best_monitor = -1.0
    patience_counter = 0
    current_phase = None

    print("\n" + "=" * 72, flush=True)
    print("STARTING REAL TRAINING: DUAL CONVNEXT MS1M RGB + SMIRK GEOMETRY", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(1, epochs + 1):
        phase = phase_for_epoch(epoch, cfg)
        if phase != current_phase:
            current_phase = phase
            set_trainable_flags(model, phase)
            groups = trainable_group_vars_for_phase(model, phase)
            assert_group_contract(model, groups, phase)
            print(f"EPOCH_{epoch:02d}_ENTER_PHASE_{phase}", flush=True)
            print_epoch_lr_contract(cfg, phase)
        else:
            groups = trainable_group_vars_for_phase(model, phase)

        start = time.time()
        progress_interval = int(cfg["training"].get("progress_interval", 20))
        losses, accs, aux_accs, grad_norms = [], [], [], []
        gate_means, mod_mins, mod_maxs = [], [], []
        for batch_idx, batch in enumerate(dist_train_ds, start=1):
            if strategy.num_replicas_in_sync > 1:
                metrics = distributed_train_step(strategy, model, optimizers, groups, batch, loss_fn, loss_weight_3d, grad_clip_norm, int(cfg["data"]["batch_size"]))
            else:
                inputs, labels = batch
                metrics = train_step(model, optimizers, groups, inputs, labels, loss_fn, loss_weight_3d, grad_clip_norm, int(cfg["data"]["batch_size"]))
            losses.append(float(metrics["loss"].numpy()))
            accs.append(float(metrics["accuracy"].numpy()))
            aux_accs.append(float(metrics["aux_accuracy"].numpy()))
            grad_norms.append(float(metrics["grad_norm"].numpy()))
            gate_means.append(float(metrics["mean_abs_channel_gate"].numpy()))
            mod_mins.append(float(metrics["modulation_factor_min"].numpy()))
            mod_maxs.append(float(metrics["modulation_factor_max"].numpy()))
            if batch_idx == 1 or (progress_interval > 0 and batch_idx % progress_interval == 0):
                batch_elapsed = time.time() - start
                print(
                    f"EPOCH_PROGRESS epoch={epoch:02d} phase={phase} batch={batch_idx} "
                    f"elapsed={batch_elapsed:.1f}s loss={float(metrics['loss'].numpy()):.6f} "
                    f"acc={float(metrics['accuracy'].numpy()):.6f} aux_acc={float(metrics['aux_accuracy'].numpy()):.6f} "
                    f"grad_norm_before_clip={float(metrics['grad_norm'].numpy()):.6f} "
                    f"alpha_raw={float(metrics['alpha_raw'].numpy()):.8f} effective_alpha={float(metrics['effective_alpha'].numpy()):.8f}",
                    flush=True,
                )

        val_metrics = evaluate_model(model, val_ds, loss_weight_3d)
        elapsed = time.time() - start
        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = float(np.mean(accs)) if accs else 0.0
        train_aux_acc = float(np.mean(aux_accs)) if aux_accs else 0.0
        train_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        train_gate = float(np.mean(gate_means)) if gate_means else 0.0
        train_mod_min = float(np.min(mod_mins)) if mod_mins else 1.0
        train_mod_max = float(np.max(mod_maxs)) if mod_maxs else 1.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} phase={phase} [{elapsed:.1f}s] "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} train_aux_3d_acc={train_aux_acc:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} val_aux_3d_acc={val_metrics['aux_accuracy']:.6f} "
            f"grad_norm_before_clip_mean={train_grad_norm:.6f} "
            f"alpha_raw={val_metrics['alpha_raw']:.8f} effective_alpha={val_metrics['effective_alpha']:.8f} "
            f"mean_abs_channel_gate_train={train_gate:.8f} mean_abs_channel_gate_val={val_metrics['mean_abs_channel_gate']:.8f} "
            f"modulation_min_train={train_mod_min:.8f} modulation_max_train={train_mod_max:.8f} "
            f"modulation_min_val={val_metrics['modulation_factor_min']:.8f} modulation_max_val={val_metrics['modulation_factor_max']:.8f}",
            flush=True,
        )
        print_epoch_lr_contract(cfg, phase)

        monitor_value = val_metrics["macro_f1"] if cfg["training"].get("monitor") == "val_macro_f1" else val_metrics["accuracy"]
        if monitor_value > best_monitor:
            best_monitor = monitor_value
            patience_counter = 0
            model.save_weights(str(ckpt_dir / "ckpt"))
            with (output_dir / "best_val_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)
            print(f"SAVED_BEST_CHECKPOINT monitor={cfg['training'].get('monitor')} value={monitor_value:.6f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"EARLY_STOPPING epoch={epoch} patience={patience}", flush=True)
                break

    print("TRAINING_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
