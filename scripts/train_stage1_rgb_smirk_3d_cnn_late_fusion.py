from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_XLA"] = "1"
os.environ["TF_DISABLE_XLA_COMPILATION"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline
from models.stage1_rgb_smirk_3d_cnn_late_fusion import (
    Stage1RGBSMIRK3DCNNLateFusionFER,
    count_params,
    resolve_latest_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 RGB + SMIRK depth/normal 3D CNN late fusion for FER2013.")
    parser.add_argument("--config", type=str, default="config_stage1_rgb_smirk_3d_cnn_late_fusion.yaml")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--geometry-cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-reference-compare", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_runtime(cfg: Dict) -> None:
    runtime = cfg.get("runtime", {})
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu_ids = list(runtime.get("gpu_ids", [0]))
        visible = [gpus[i] for i in gpu_ids if i < len(gpus)] or [gpus[0]]
        try:
            tf.config.set_visible_devices(visible, "GPU")
            if bool(runtime.get("memory_growth", True)):
                for gpu in visible:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] TensorFlow visible GPU(s): {[gpu.name for gpu in visible]} with memory_growth=True", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not set GPU visible devices/memory growth: {e}", flush=True)
    else:
        print("[WARNING] TensorFlow sees no GPU. Running on CPU.", flush=True)

    intra_threads = runtime.get("intra_op_threads")
    inter_threads = runtime.get("inter_op_threads")
    if intra_threads:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
    if inter_threads:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
    tf.config.optimizer.set_jit(False)
    if bool(runtime.get("use_mixed_precision", True)):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] TensorFlow mixed_float16 enabled for Stage 1 late fusion", flush=True)


def cache_path_for(cache_dir: Path, pattern: str, split: str) -> Path:
    return cache_dir / pattern.format(split=split)


def load_geometry_cache(cfg: Dict, split: str, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    cache_cfg = cfg.get("geometry_cache", {})
    path = cache_path_for(cache_dir, str(cache_cfg.get("map_file_pattern", "{split}_smirk_depth_normal_maps.npz")), split)
    if not path.exists():
        raise FileNotFoundError(f"Missing SMIRK depth+normal cache for {split}: {path}")
    cache = np.load(path, allow_pickle=False)
    maps = cache["geometry_maps"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    sample_ids = cache["sample_ids"].astype(np.int64)
    if maps.ndim != 4:
        raise ValueError(f"Expected geometry_maps [N,H,W,4] for {split}, got {maps.shape}")
    if int(maps.shape[-1]) != int(cache_cfg.get("expected_channels", 4)):
        raise ValueError(f"Expected {cache_cfg.get('expected_channels', 4)} geometry channels for {split}, got {maps.shape[-1]}")
    if len(maps) != len(labels) or len(labels) != len(sample_ids):
        raise ValueError(f"Length mismatch in geometry cache: {path}")
    if not np.isfinite(maps).all():
        raise FloatingPointError(f"NaN/Inf in geometry cache: {path}")
    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    smirk_frozen = bool(cache["smirk_frozen"]) if "smirk_frozen" in cache.files else bool(meta.get("smirk_frozen", False))
    smirk_trainable = int(cache["smirk_trainable_params"]) if "smirk_trainable_params" in cache.files else int(meta.get("smirk_trainable_params", -1))
    if not smirk_frozen or smirk_trainable != 0:
        raise RuntimeError(f"SMIRK cache freeze contract failed for {split}: smirk_frozen={smirk_frozen} trainable={smirk_trainable}")
    print(
        f"SMIRK_DEPTH_NORMAL_USED[{split}]={maps.shape} labels={labels.shape} "
        f"smirk_frozen={smirk_frozen} smirk_trainable_params={smirk_trainable}",
        flush=True,
    )
    return maps, labels, sample_ids, meta


def load_pixels_for_cache(cfg: Dict, split: str, sample_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    records = collect_split_records(
        resolve_path(cfg["data"]["data_path"]),
        split,
        mask_dir=None,
        use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
        bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
        predecode_pixels=True,
        preload_masks=False,
        allow_missing_masks=False,
    )
    id_to_pos = {int(sample_id): pos for pos, sample_id in enumerate(records.sample_ids)}
    pixels = []
    for sample_id, label in zip(sample_ids, labels):
        pos = id_to_pos.get(int(sample_id))
        if pos is None:
            raise KeyError(f"sample_id={int(sample_id)} from geometry cache is missing in FER2013 {split} split.")
        if int(records.labels[pos]) != int(label):
            raise ValueError(f"Label mismatch for {split} sample_id={int(sample_id)}: cache={int(label)} csv={int(records.labels[pos])}")
        pixels.append(records.images[pos])
    arr = np.stack(pixels, axis=0).astype(np.uint8)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)
    print(f"FER_PIXELS_USED[{split}]={arr.shape}", flush=True)
    return arr


def preprocess_images(images: tf.Tensor, cfg: Dict) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    target_size = int(cfg["data"]["image_size"])
    images = tf.image.resize(images, [target_size, target_size], method="bilinear")
    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    images = images / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    return (images - mean) / std


def preprocess_geometry(geometry_maps: tf.Tensor, cfg: Dict) -> tf.Tensor:
    geometry_maps = tf.cast(geometry_maps, tf.float32)
    target_size = int(cfg.get("geometry_cache", {}).get("geometry_input_size", cfg["data"]["image_size"]))
    geometry_maps = tf.image.resize(geometry_maps, [target_size, target_size], method="bilinear")
    return tf.clip_by_value(geometry_maps, 0.0, 1.0)


def make_dataset(
    pixels: np.ndarray,
    geometry_maps: np.ndarray,
    labels: np.ndarray,
    cfg: Dict,
    batch_size: int,
    *,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        {
            "pixels": pixels,
            "geometry_maps": geometry_maps.astype(np.float32),
            "labels": labels.astype(np.int32),
        }
    )
    if training:
        ds = ds.shuffle(min(len(labels), int(cfg["data"].get("shuffle_buffer", 10000))), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    def batch_mapper(item):
        image = preprocess_images(item["pixels"], cfg)
        geometry_maps = preprocess_geometry(item["geometry_maps"], cfg)
        if training and bool(cfg.get("augmentation", {}).get("horizontal_flip", False)):
            flips = tf.random.uniform([tf.shape(image)[0], 1, 1, 1]) < 0.5
            image = tf.where(flips, tf.image.flip_left_right(image), image)
            geometry_maps = tf.where(flips, tf.image.flip_left_right(geometry_maps), geometry_maps)
        return {"image": image, "geometry_maps": geometry_maps}, item["labels"]

    runtime = cfg.get("runtime", {})
    parallel_calls = runtime.get("tf_data_num_parallel_calls") or tf.data.AUTOTUNE
    ds = ds.map(batch_mapper, num_parallel_calls=parallel_calls)
    return ds.prefetch(runtime.get("prefetch_buffer") or tf.data.AUTOTUNE)


def maybe_take(ds: tf.data.Dataset, max_batches: Optional[int]) -> tf.data.Dataset:
    return ds.take(int(max_batches)) if max_batches else ds


def build_optimizer(cfg: Dict):
    lr = float(cfg["training"].get("lr", 1e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    clipnorm = cfg["training"].get("grad_clip_norm")
    kwargs = {"learning_rate": lr}
    if clipnorm:
        kwargs["clipnorm"] = float(clipnorm)
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)
    if adamw is not None:
        try:
            return adamw(weight_decay=weight_decay, jit_compile=False, **kwargs)
        except TypeError:
            return adamw(weight_decay=weight_decay, **kwargs)
    return tf.keras.optimizers.Adam(**kwargs)


def restore_baseline_into(model, checkpoint_path: str, *, strict: bool) -> str:
    latest = resolve_latest_checkpoint(checkpoint_path)
    if latest is None:
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    status = tf.train.Checkpoint(model=model).restore(latest)
    if strict:
        status.assert_existing_objects_matched()
    else:
        status.expect_partial()
    return latest


def restore_rgb_baseline_checkpoint(model: Stage1RGBSMIRK3DCNNLateFusionFER, cfg: Dict, args: argparse.Namespace) -> str:
    ckpt_cfg = cfg.get("baseline_checkpoint", {})
    ckpt_path = args.baseline_checkpoint or ckpt_cfg.get("best_checkpoint_dir")
    if not ckpt_path:
        raise ValueError("baseline_checkpoint.best_checkpoint_dir or --baseline-checkpoint is required.")
    restored = restore_baseline_into(
        model.rgb_baseline,
        ckpt_path,
        strict=bool(ckpt_cfg.get("strict_existing_objects_matched", True)),
    )
    model.freeze_rgb_branch()
    print(f"RGB_BASELINE_CKPT43_RESTORED path={restored}", flush=True)
    print(f"RGB_BRANCH_FROZEN_OK trainable_variables={len(model.rgb_baseline.trainable_variables)}", flush=True)
    return restored


def ce_loss(labels: tf.Tensor, logits: tf.Tensor, num_classes: int, label_smoothing: float) -> tf.Tensor:
    logits = tf.cast(logits, tf.float32)
    if label_smoothing > 0.0:
        targets = tf.one_hot(tf.cast(labels, tf.int32), num_classes, dtype=tf.float32)
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                targets,
                logits,
                from_logits=True,
                label_smoothing=float(label_smoothing),
            )
        )
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))


def get_strategy(cfg: Dict) -> tf.distribute.Strategy:
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(len(gpus))])
        print(f"[INFO] MirroredStrategy initialized across {strategy.num_replicas_in_sync} GPUs", flush=True)
        return strategy
    print("[INFO] Single device execution strategy", flush=True)
    return tf.distribute.get_strategy()


@tf.function
def train_step(model, optimizer, features, labels, loss_weight_3d: float, label_smoothing: float, loss_scale: float = 1.0):
    with tf.GradientTape() as tape:
        outputs = model(features, training=True)
        fusion_loss = ce_loss(labels, outputs["fusion_logits"], model.num_classes, label_smoothing)
        geometry_loss = ce_loss(labels, outputs["geometry_logits"], model.num_classes, label_smoothing)
        raw_loss = fusion_loss + tf.cast(loss_weight_3d, tf.float32) * geometry_loss
        if model.losses:
            raw_loss = raw_loss + tf.add_n([tf.cast(item, tf.float32) for item in model.losses])
        scaled_loss = raw_loss * tf.cast(loss_scale, tf.float32)
    variables = model.trainable_variables
    grads = tape.gradient(scaled_loss, variables)
    optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
    fused_pred = tf.argmax(outputs["fusion_logits"], axis=-1, output_type=tf.int32)
    geom_pred = tf.argmax(outputs["geometry_logits"], axis=-1, output_type=tf.int32)
    rgb_pred = tf.argmax(outputs["rgb_logits"], axis=-1, output_type=tf.int32)
    batch_total = tf.shape(labels)[0]
    return (
        raw_loss,
        fusion_loss,
        geometry_loss,
        tf.reduce_sum(tf.cast(fused_pred == labels, tf.int32)),
        tf.reduce_sum(tf.cast(geom_pred == labels, tf.int32)),
        tf.reduce_sum(tf.cast(rgb_pred == labels, tf.int32)),
        batch_total,
    )


@tf.function
def eval_step(model, features, labels, loss_weight_3d: float, label_smoothing: float):
    outputs = model(features, training=False)
    fusion_loss = ce_loss(labels, outputs["fusion_logits"], model.num_classes, label_smoothing)
    geometry_loss = ce_loss(labels, outputs["geometry_logits"], model.num_classes, label_smoothing)
    total_loss = fusion_loss + tf.cast(loss_weight_3d, tf.float32) * geometry_loss
    return (
        total_loss,
        fusion_loss,
        geometry_loss,
        tf.nn.softmax(tf.cast(outputs["rgb_logits"], tf.float32), axis=-1),
        tf.nn.softmax(tf.cast(outputs["geometry_logits"], tf.float32), axis=-1),
        tf.nn.softmax(tf.cast(outputs["fusion_logits"], tf.float32), axis=-1),
    )


def train_one_epoch(
    model,
    optimizer,
    ds: tf.data.Dataset,
    loss_weight_3d: float,
    label_smoothing: float,
    strategy: Optional[tf.distribute.Strategy] = None,
) -> Dict[str, float]:
    losses = []
    fusion_losses = []
    geometry_losses = []
    fused_correct = 0
    geom_correct = 0
    rgb_correct = 0
    total = 0

    use_dist = strategy is not None and strategy.num_replicas_in_sync > 1
    loss_scale = 1.0 / float(strategy.num_replicas_in_sync) if use_dist else 1.0

    if use_dist:
        dist_ds = strategy.experimental_distribute_dataset(ds)
        for batch_features, batch_labels in dist_ds:
            per_loss, per_f_loss, per_g_loss, per_f_ok, per_g_ok, per_r_ok, per_total = strategy.run(
                train_step,
                args=(model, optimizer, batch_features, batch_labels, loss_weight_3d, label_smoothing, loss_scale),
            )
            loss_v = float(strategy.reduce(tf.distribute.ReduceOp.MEAN, per_loss, axis=None).numpy())
            f_loss_v = float(strategy.reduce(tf.distribute.ReduceOp.MEAN, per_f_loss, axis=None).numpy())
            g_loss_v = float(strategy.reduce(tf.distribute.ReduceOp.MEAN, per_g_loss, axis=None).numpy())
            f_ok_v = int(strategy.reduce(tf.distribute.ReduceOp.SUM, per_f_ok, axis=None).numpy())
            g_ok_v = int(strategy.reduce(tf.distribute.ReduceOp.SUM, per_g_ok, axis=None).numpy())
            r_ok_v = int(strategy.reduce(tf.distribute.ReduceOp.SUM, per_r_ok, axis=None).numpy())
            tot_v = int(strategy.reduce(tf.distribute.ReduceOp.SUM, per_total, axis=None).numpy())

            losses.append(loss_v)
            fusion_losses.append(f_loss_v)
            geometry_losses.append(g_loss_v)
            fused_correct += f_ok_v
            geom_correct += g_ok_v
            rgb_correct += r_ok_v
            total += tot_v
    else:
        for features, labels in ds:
            loss, f_loss, g_loss, f_ok, g_ok, r_ok, batch_total = train_step(
                model,
                optimizer,
                features,
                labels,
                loss_weight_3d=loss_weight_3d,
                label_smoothing=label_smoothing,
                loss_scale=1.0,
            )
            losses.append(float(loss.numpy()))
            fusion_losses.append(float(f_loss.numpy()))
            geometry_losses.append(float(g_loss.numpy()))
            fused_correct += int(f_ok.numpy())
            geom_correct += int(g_ok.numpy())
            rgb_correct += int(r_ok.numpy())
            total += int(batch_total.numpy())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "fusion_loss": float(np.mean(fusion_losses)) if fusion_losses else float("nan"),
        "geometry_loss": float(np.mean(geometry_losses)) if geometry_losses else float("nan"),
        "rgb_accuracy": rgb_correct / max(total, 1),
        "geometry_accuracy": geom_correct / max(total, 1),
        "fused_accuracy": fused_correct / max(total, 1),
    }


def evaluate(
    model,
    ds: tf.data.Dataset,
    loss_weight_3d: float,
    label_smoothing: float,
    strategy: Optional[tf.distribute.Strategy] = None,
) -> Dict:
    losses = []
    fusion_losses = []
    geometry_losses = []
    y_true = []
    y_rgb = []
    y_geometry = []
    y_fused = []
    confidence = []

    use_dist = strategy is not None and strategy.num_replicas_in_sync > 1
    if use_dist:
        dist_ds = strategy.experimental_distribute_dataset(ds)
        for batch_features, batch_labels in dist_ds:
            per_loss, per_f_loss, per_g_loss, per_rgb_probs, per_geom_probs, per_fused_probs = strategy.run(
                eval_step,
                args=(model, batch_features, batch_labels, loss_weight_3d, label_smoothing),
            )
            loc_losses = strategy.experimental_local_results(per_loss)
            loc_f_losses = strategy.experimental_local_results(per_f_loss)
            loc_g_losses = strategy.experimental_local_results(per_g_loss)
            loc_rgb_p = strategy.experimental_local_results(per_rgb_probs)
            loc_geom_p = strategy.experimental_local_results(per_geom_probs)
            loc_fused_p = strategy.experimental_local_results(per_fused_probs)
            loc_labels = strategy.experimental_local_results(batch_labels)

            for l_tot, l_f, l_g, r_p, g_p, f_p, lbl in zip(
                loc_losses, loc_f_losses, loc_g_losses, loc_rgb_p, loc_geom_p, loc_fused_p, loc_labels
            ):
                if tf.shape(lbl)[0] == 0:
                    continue
                losses.append(float(l_tot.numpy()))
                fusion_losses.append(float(l_f.numpy()))
                geometry_losses.append(float(l_g.numpy()))
                rgb_np = r_p.numpy()
                geom_np = g_p.numpy()
                fused_np = f_p.numpy()
                lbl_np = lbl.numpy().astype(int)
                y_true.extend(lbl_np.tolist())
                y_rgb.extend(rgb_np.argmax(axis=1).astype(int).tolist())
                y_geometry.extend(geom_np.argmax(axis=1).astype(int).tolist())
                y_fused.extend(fused_np.argmax(axis=1).astype(int).tolist())
                confidence.extend(fused_np.max(axis=1).astype(float).tolist())
    else:
        for features, labels in ds:
            loss, f_loss, g_loss, rgb_probs, geometry_probs, fused_probs = eval_step(
                model,
                features,
                labels,
                loss_weight_3d=loss_weight_3d,
                label_smoothing=label_smoothing,
            )
            rgb_np = rgb_probs.numpy()
            geom_np = geometry_probs.numpy()
            fused_np = fused_probs.numpy()
            losses.append(float(loss.numpy()))
            fusion_losses.append(float(f_loss.numpy()))
            geometry_losses.append(float(g_loss.numpy()))
            y_true.extend(labels.numpy().astype(int).tolist())
            y_rgb.extend(rgb_np.argmax(axis=1).astype(int).tolist())
            y_geometry.extend(geom_np.argmax(axis=1).astype(int).tolist())
            y_fused.extend(fused_np.argmax(axis=1).astype(int).tolist())
            confidence.extend(fused_np.max(axis=1).astype(float).tolist())

    y_true_arr = np.asarray(y_true, dtype=int)
    y_rgb_arr = np.asarray(y_rgb, dtype=int)
    y_geometry_arr = np.asarray(y_geometry, dtype=int)
    y_fused_arr = np.asarray(y_fused, dtype=int)
    rgb_correct = y_rgb_arr == y_true_arr
    fused_correct = y_fused_arr == y_true_arr
    rescue_count = int((~rgb_correct & fused_correct).sum())
    harmed_count = int((rgb_correct & ~fused_correct).sum())
    ids = list(range(len(EMOTION_NAMES)))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "fusion_loss": float(np.mean(fusion_losses)) if fusion_losses else float("nan"),
        "geometry_loss": float(np.mean(geometry_losses)) if geometry_losses else float("nan"),
        "rgb_accuracy": float(rgb_correct.mean()) if len(y_true_arr) else 0.0,
        "geometry_accuracy": float((y_geometry_arr == y_true_arr).mean()) if len(y_true_arr) else 0.0,
        "fused_accuracy": float(fused_correct.mean()) if len(y_true_arr) else 0.0,
        "macro_f1": float(f1_score(y_true, y_fused, labels=ids, average="macro", zero_division=0)),
        "rescue_count": rescue_count,
        "harmed_count": harmed_count,
        "net_gain": rescue_count - harmed_count,
        "confusion_matrix": confusion_matrix(y_true, y_fused, labels=ids).tolist(),
        "y_true": y_true,
        "y_rgb": y_rgb,
        "y_geometry": y_geometry,
        "y_fused": y_fused,
        "confidence": confidence,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_predictions(path: Path, sample_ids: np.ndarray, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "pred_rgb_baseline", "pred_3d_only", "pred_fused", "fused_confidence"])
        for sample_id, y_t, p_r, p_g, p_f, conf in zip(
            sample_ids,
            metrics["y_true"],
            metrics["y_rgb"],
            metrics["y_geometry"],
            metrics["y_fused"],
            metrics["confidence"],
        ):
            writer.writerow([int(sample_id), int(y_t), int(p_r), int(p_g), int(p_f), float(conf)])


def build_reference_baseline(cfg: Dict, restored_checkpoint: str, image_batch: tf.Tensor) -> ConvNeXtBaseFaceFERBaseline:
    ref = ConvNeXtBaseFaceFERBaseline(Stage1RGBSMIRK3DCNNLateFusionFER._make_rgb_baseline_cfg(cfg))
    _ = ref({"image": image_batch}, training=False)
    restore_baseline_into(
        ref,
        restored_checkpoint,
        strict=bool(cfg.get("baseline_checkpoint", {}).get("strict_existing_objects_matched", True)),
    )
    ref.trainable = False
    return ref


def run_contract_smoke_test(
    model: Stage1RGBSMIRK3DCNNLateFusionFER,
    optimizer,
    ds: tf.data.Dataset,
    cfg: Dict,
    restored_checkpoint: str,
    *,
    skip_reference_compare: bool,
) -> None:
    features, labels = next(iter(ds.take(1)))
    outputs = model(features, training=False)
    model.print_contract_summary()

    rgb_trainable = len(model.rgb_baseline.trainable_variables)
    if rgb_trainable != 0:
        raise RuntimeError(f"ConvNeXt/RGB branch is not frozen: trainable_variables={rgb_trainable}")

    expected_keys = model.expected_trainable_variable_keys()
    actual_keys = {getattr(v, "ref", lambda: getattr(v, "name", id(v)))() for v in model.trainable_variables}
    if actual_keys != expected_keys:
        raise RuntimeError("Trainable variable set is not exactly geometry_cnn + geometry_head + fusion_mlp.")

    if not skip_reference_compare:
        reference = build_reference_baseline(cfg, restored_checkpoint, features["image"])
        ref_logits = tf.cast(reference({"image": features["image"]}, training=False)["logits"], tf.float32)
        max_diff = float(tf.reduce_max(tf.abs(ref_logits - outputs["rgb_logits"])).numpy())
        if max_diff > float(cfg.get("baseline_checkpoint", {}).get("rgb_logit_tolerance", 1e-5)):
            raise RuntimeError(f"RGB output mismatch versus ckpt-43 reference: max_abs_diff={max_diff:.8f}")
        print(f"RGB_OUTPUT_MATCHES_CKPT43 max_abs_diff={max_diff:.8f}", flush=True)
    else:
        print("[WARNING] Reference RGB compare skipped by user request.", flush=True)

    loss_weight_3d = float(cfg.get("training", {}).get("loss_weight_3d", 0.3))
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))
    with tf.GradientTape() as tape:
        train_outputs = model(features, training=True)
        fusion_loss = ce_loss(labels, train_outputs["fusion_logits"], model.num_classes, label_smoothing)
        geometry_loss = ce_loss(labels, train_outputs["geometry_logits"], model.num_classes, label_smoothing)
        total_loss = fusion_loss + loss_weight_3d * geometry_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    non_none = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    if not non_none:
        raise RuntimeError("No gradients found for Stage 1 trainable branches.")

    geom_vars, geom_head_vars, fusion_vars = model.trainable_branch_variables()
    branch_sets = {
        "geometry_cnn": {getattr(v, "ref", lambda: getattr(v, "name", id(v)))() for v in geom_vars},
        "geometry_head": {getattr(v, "ref", lambda: getattr(v, "name", id(v)))() for v in geom_head_vars},
        "fusion_mlp_head": {getattr(v, "ref", lambda: getattr(v, "name", id(v)))() for v in fusion_vars},
    }
    grad_keys = {getattr(v, "ref", lambda: getattr(v, "name", id(v)))() for _, v in non_none}
    grad_counts = {name: len(keys & grad_keys) for name, keys in branch_sets.items()}
    if any(count == 0 for count in grad_counts.values()):
        raise RuntimeError(f"Gradient coverage missing a Stage 1 branch: {grad_counts}")

    optimizer.build(model.trainable_variables) if hasattr(optimizer, "build") else None
    print("STAGE1_GRADIENT_SCOPE_OK", flush=True)
    print(f"  gradient_vars={len(non_none)}/{len(model.trainable_variables)}", flush=True)
    print(f"  gradient_branch_counts={grad_counts}", flush=True)
    print("SMIRK_FROZEN_CONFIRMED_FROM_CACHE_AND_NO_TRAIN_GRAPH", flush=True)
    print("STAGE1_SMOKE_TEST_PASSED", flush=True)


def strip_prediction_arrays(metrics: Dict) -> Dict:
    return {k: v for k, v in metrics.items() if k not in ("y_true", "y_rgb", "y_geometry", "y_fused", "confidence")}


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    configure_runtime(cfg)
    tf.keras.utils.set_random_seed(seed)

    output_dir = resolve_path(cfg["paths"]["output_dir"]) or PROJECT_ROOT / "outputs" / "stage1_rgb_smirk_3d_cnn_late_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_path(args.geometry_cache_dir) or resolve_path(cfg.get("geometry_cache", {}).get("feature_dir")) or (output_dir / "geometry_maps")
    strategy = get_strategy(cfg)
    bs_per_gpu = int(args.batch_size or cfg["runtime"].get("batch_size_per_gpu", 16))
    batch_size = bs_per_gpu * strategy.num_replicas_in_sync
    print(f"[INFO] Strategy replicas: {strategy.num_replicas_in_sync}, per-GPU BS: {bs_per_gpu}, Global BS: {batch_size}", flush=True)

    split_data = {}
    cache_meta = {}
    for split in ("train", "val", "test"):
        geometry_maps, labels, sample_ids, meta = load_geometry_cache(cfg, split, cache_dir)
        pixels = load_pixels_for_cache(cfg, split, sample_ids, labels)
        split_data[split] = (pixels, geometry_maps, labels, sample_ids)
        cache_meta[split] = meta

    train_ds = make_dataset(split_data["train"][0], split_data["train"][1], split_data["train"][2], cfg, batch_size, training=True, seed=seed)
    val_ds = make_dataset(split_data["val"][0], split_data["val"][1], split_data["val"][2], cfg, batch_size, training=False, seed=seed)
    test_ds = make_dataset(split_data["test"][0], split_data["test"][1], split_data["test"][2], cfg, batch_size, training=False, seed=seed)
    train_loop_ds = maybe_take(train_ds, args.max_train_batches)
    val_loop_ds = maybe_take(val_ds, args.max_eval_batches)

    with strategy.scope():
        model = Stage1RGBSMIRK3DCNNLateFusionFER(cfg)
        first_features, _ = next(iter(train_ds.take(1)))
        _ = model(first_features, training=False)
        restored_checkpoint = restore_rgb_baseline_checkpoint(model, cfg, args)
        optimizer = build_optimizer(cfg)

    run_contract_smoke_test(
        model,
        optimizer,
        train_loop_ds,
        cfg,
        restored_checkpoint,
        skip_reference_compare=bool(args.skip_reference_compare),
    )
    if args.smoke_only:
        save_json(output_dir / "stage1_smoke_contract.json", {
            "status": "passed",
            "baseline_checkpoint": restored_checkpoint,
            "rgb_trainable_params": count_params(model.rgb_baseline.trainable_variables),
            "stage1_trainable_params": count_params(model.trainable_variables),
            "cache_meta": cache_meta,
        })
        return 0

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    best_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "best_fused_val_accuracy"), max_to_keep=1)
    last_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "last"), max_to_keep=1)
    best_fused_acc = -1.0
    best_net_gain = -999999
    best_epoch = -1
    patience = int(cfg["training"].get("patience", 15))
    history = []
    loss_weight_3d = float(cfg.get("training", {}).get("loss_weight_3d", 0.3))
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))

    print("============================================================", flush=True)
    print(" STARTING STAGE 1 RGB + SMIRK 3D CNN LATE FUSION", flush=True)
    print(f" Baseline checkpoint: {restored_checkpoint}", flush=True)
    print(f" Loss: CE_fusion + {loss_weight_3d} * CE_3D", flush=True)
    print(" Trainable: geometry_cnn + geometry_head + fusion_mlp_head", flush=True)
    print(" Frozen: ConvNeXt RGB backbone + RGB Dense(7); SMIRK cache only", flush=True)
    print("============================================================", flush=True)

    for epoch in range(int(args.epochs or cfg["training"].get("epochs", 60))):
        start = time.time()
        train_metrics = train_one_epoch(model, optimizer, train_loop_ds, loss_weight_3d, label_smoothing, strategy=strategy)
        val_metrics = evaluate(model, val_loop_ds, loss_weight_3d, label_smoothing, strategy=strategy)
        fused_acc = float(val_metrics["fused_accuracy"])
        net_gain = int(val_metrics["net_gain"])
        improved = (fused_acc > best_fused_acc) or (abs(fused_acc - best_fused_acc) < 1e-12 and net_gain > best_net_gain)
        if improved:
            best_fused_acc = fused_acc
            best_net_gain = net_gain
            best_epoch = epoch + 1
            best_manager.save(checkpoint_number=epoch + 1)
            print(f"[INFO] Save best fused val accuracy at epoch {epoch+1}: fused_acc={best_fused_acc:.6f} net_gain={best_net_gain}", flush=True)
        last_manager.save(checkpoint_number=epoch + 1)

        row = {
            "epoch": epoch + 1,
            "time_sec": round(time.time() - start, 2),
            "train_loss": train_metrics["loss"],
            "train_fusion_loss": train_metrics["fusion_loss"],
            "train_3d_loss": train_metrics["geometry_loss"],
            "train_rgb_accuracy": train_metrics["rgb_accuracy"],
            "train_3d_accuracy": train_metrics["geometry_accuracy"],
            "train_fused_accuracy": train_metrics["fused_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_fusion_loss": val_metrics["fusion_loss"],
            "val_3d_loss": val_metrics["geometry_loss"],
            "val_rgb_accuracy": val_metrics["rgb_accuracy"],
            "val_3d_accuracy": val_metrics["geometry_accuracy"],
            "val_fused_accuracy": val_metrics["fused_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_rescue_count": val_metrics["rescue_count"],
            "val_harmed_count": val_metrics["harmed_count"],
            "val_net_gain": val_metrics["net_gain"],
            "best_epoch": best_epoch,
            "best_fused_val_accuracy": best_fused_acc,
            "best_net_gain": best_net_gain,
        }
        history.append(row)
        print(
            f"Epoch {epoch+1:02d}: loss={row['train_loss']:.4f} "
            f"train_rgb={row['train_rgb_accuracy']:.4f} train_3d={row['train_3d_accuracy']:.4f} train_fused={row['train_fused_accuracy']:.4f} "
            f"val_rgb={row['val_rgb_accuracy']:.4f} val_3d={row['val_3d_accuracy']:.4f} val_fused={row['val_fused_accuracy']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} rescue={row['val_rescue_count']} harmed={row['val_harmed_count']} net_gain={row['val_net_gain']}",
            flush=True,
        )
        if best_epoch > 0 and (epoch + 1 - best_epoch) >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}; best_epoch={best_epoch}", flush=True)
            break

    if history:
        with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    if best_manager.latest_checkpoint:
        checkpoint.restore(best_manager.latest_checkpoint).expect_partial()
        print(f"[INFO] Restored best fused checkpoint for final test: {best_manager.latest_checkpoint}", flush=True)

    full_val_metrics = evaluate(model, val_ds, loss_weight_3d, label_smoothing, strategy=strategy)
    full_test_metrics = evaluate(model, test_ds, loss_weight_3d, label_smoothing, strategy=strategy)
    save_json(output_dir / "val_metrics_best_fused.json", strip_prediction_arrays(full_val_metrics))
    save_json(output_dir / "test_metrics_no_tta.json", strip_prediction_arrays(full_test_metrics))
    save_json(output_dir / "test_metrics.json", strip_prediction_arrays(full_test_metrics))
    save_predictions(output_dir / "predictions_stage1_test_no_tta.csv", split_data["test"][3], full_test_metrics)

    print("STAGE1_RGB_SMIRK_3D_CNN_LATE_FUSION_FINAL_TEST_NO_TTA", flush=True)
    print(f"  rgb_baseline_accuracy={full_test_metrics['rgb_accuracy']:.6f}", flush=True)
    print(f"  geometry_3d_only_accuracy={full_test_metrics['geometry_accuracy']:.6f}", flush=True)
    print(f"  fused_accuracy={full_test_metrics['fused_accuracy']:.6f}", flush=True)
    print(f"  macro_f1={full_test_metrics['macro_f1']:.6f}", flush=True)
    print(f"  rescue_count={full_test_metrics['rescue_count']} harmed_count={full_test_metrics['harmed_count']} net_gain={full_test_metrics['net_gain']}", flush=True)
    print(f"  predictions={output_dir / 'predictions_stage1_test_no_tta.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

