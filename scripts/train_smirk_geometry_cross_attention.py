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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_DISABLE_XLA", "1")
os.environ.setdefault("TF_DISABLE_XLA_COMPILATION", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from models.smirk_geometry_cross_attention import SMIRKGeometryCrossAttentionFER, resolve_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-MS1M RGB Q x SMIRK/VLM geometry KV cross-attention on FER2013.")
    parser.add_argument("--config", type=str, default="config_smirk_geometry_cross_attention.yaml")
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--skip-baseline-checkpoint", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_runtime(cfg: Dict) -> None:
    runtime = cfg.get("runtime", {})
    intra_threads = runtime.get("intra_op_threads")
    inter_threads = runtime.get("inter_op_threads")
    if intra_threads:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
    if inter_threads:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
    tf.config.optimizer.set_jit(False)
    if bool(runtime.get("use_mixed_precision", True)):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] TensorFlow mixed_float16 enabled for cross-attention training", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[WARNING] TensorFlow sees no GPU. Running on CPU.", flush=True)
        return
    gpu_ids = list(runtime.get("gpu_ids", [0]))
    visible = [gpus[i] for i in gpu_ids if i < len(gpus)] or [gpus[0]]
    tf.config.set_visible_devices(visible, "GPU")
    if bool(runtime.get("memory_growth", True)):
        for gpu in visible:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    print(f"[INFO] TensorFlow visible GPU(s): {[gpu.name for gpu in visible]}", flush=True)


def cache_path_for(feature_dir: Path, pattern: str, split: str) -> Path:
    return feature_dir / pattern.format(split=split)


def load_geometry_cache(cfg: Dict, split: str, feature_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_cfg = cfg.get("geometry_cache", {})
    pattern = str(cache_cfg.get("token_file_pattern", "{split}_smirk_vlm_geometry_tokens.npz"))
    path = cache_path_for(feature_dir, pattern, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached geometry tokens: {path}")
    cache = np.load(path, allow_pickle=False)
    tokens = cache["geometry_tokens"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    sample_ids = cache["sample_ids"].astype(np.int64)
    if tokens.ndim != 3:
        raise ValueError(f"Expected geometry token cache [N,T,D] for {split}, got {tokens.shape}")
    if len(tokens) != len(labels) or len(labels) != len(sample_ids):
        raise ValueError(f"Length mismatch in {path}")
    if not np.isfinite(tokens).all():
        raise FloatingPointError(f"NaN/Inf in geometry token cache: {path}")
    expected_dim = int(cache_cfg.get("expected_token_dim", tokens.shape[2]))
    if expected_dim != int(tokens.shape[2]):
        raise ValueError(f"Config expected_token_dim={expected_dim}, cache token dim={tokens.shape[2]}")
    expected_tokens = int(cache_cfg.get("expected_num_tokens", tokens.shape[1]))
    if expected_tokens != int(tokens.shape[1]):
        raise ValueError(f"Config expected_num_tokens={expected_tokens}, cache num tokens={tokens.shape[1]}")
    print(f"GEOMETRY_TOKENS_USED[{split}]={tokens.shape} labels={labels.shape} nan_count={int(np.isnan(tokens).sum())}", flush=True)
    return tokens, labels, sample_ids


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
    print(f"FER_PIXELS_USED[{split}]={arr.shape}", flush=True)
    return arr


def preprocess_batch_images(images: tf.Tensor, cfg: Dict, training: bool) -> tf.Tensor:
    images = tf.expand_dims(tf.cast(images, tf.float32), axis=-1)
    target_size = int(cfg["data"]["image_size"])
    images = tf.image.resize(images, [target_size, target_size], method="bilinear")
    images = tf.image.grayscale_to_rgb(images)
    aug = cfg.get("augmentation", {})
    if training and bool(aug.get("horizontal_flip", False)):
        flips = tf.random.uniform([tf.shape(images)[0], 1, 1, 1]) < 0.5
        images = tf.where(flips, tf.image.flip_left_right(images), images)
    images = tf.cast(images, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    return (images - mean) / std


def make_dataset(
    pixels: np.ndarray,
    geometry_tokens: np.ndarray,
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
            "geometry_tokens": geometry_tokens.astype(np.float32),
            "labels": labels.astype(np.int32),
        }
    )
    if training:
        ds = ds.shuffle(min(len(labels), int(cfg["data"].get("shuffle_buffer", 10000))), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    def batch_mapper(item):
        features = {
            "image": preprocess_batch_images(item["pixels"], cfg, training),
            "geometry_tokens": tf.cast(item["geometry_tokens"], tf.float32),
        }
        return features, item["labels"]

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


def restore_rgb_baseline_checkpoint(model: SMIRKGeometryCrossAttentionFER, cfg: Dict, args: argparse.Namespace) -> None:
    if args.skip_baseline_checkpoint:
        print("[WARNING] Skipping trained RGB baseline checkpoint restore by user request.", flush=True)
        return
    ckpt_cfg = cfg.get("baseline_checkpoint", {})
    ckpt_path = args.baseline_checkpoint or ckpt_cfg.get("best_checkpoint_dir")
    latest = resolve_latest_checkpoint(ckpt_path)
    if latest is None:
        message = f"Best ConvNeXt-MS1M baseline checkpoint not found: {ckpt_path}"
        if bool(ckpt_cfg.get("require", True)):
            raise FileNotFoundError(message)
        print(f"[WARNING] {message}", flush=True)
        return
    status = tf.train.Checkpoint(model=model.rgb_baseline).restore(latest)
    status.expect_partial()
    print(f"RGB_BASELINE_BEST_CHECKPOINT_LOADED path={latest}", flush=True)


def ce_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, tf.cast(logits, tf.float32), from_logits=True))


def train_one_epoch(model, optimizer, ds: tf.data.Dataset) -> Dict[str, float]:
    losses = []
    correct = 0
    total = 0
    for features, labels in ds:
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            loss = ce_loss(labels, outputs["logits"])
            if model.losses:
                loss = loss + tf.add_n([tf.cast(item, tf.float32) for item in model.losses])
        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
        preds = tf.argmax(outputs["logits"], axis=-1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy())
        total += int(labels.shape[0])
        losses.append(float(loss.numpy()))
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "accuracy": correct / max(total, 1)}


def evaluate(model, ds: tf.data.Dataset) -> Dict:
    losses = []
    y_true = []
    y_pred = []
    confidences = []
    for features, labels in ds:
        outputs = model(features, training=False)
        logits = tf.cast(outputs["logits"], tf.float32)
        losses.append(float(ce_loss(labels, logits).numpy()))
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        preds = probs.argmax(axis=1)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.astype(int).tolist())
        confidences.extend(probs.max(axis=1).astype(float).tolist())
    ids = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    y_arr = np.asarray(y_true)
    per_class = {}
    for class_id, class_name in enumerate(EMOTION_NAMES):
        denom = int((y_arr == class_id).sum())
        per_class[class_name] = float(cm[class_id, class_id] / denom) if denom else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidences,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_predictions(path: Path, sample_ids: np.ndarray, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "pred_smirk_geometry", "confidence"])
        for sample_id, y_true, pred, conf in zip(sample_ids, metrics["y_true"], metrics["y_pred"], metrics["confidence"]):
            writer.writerow([int(sample_id), int(y_true), int(pred), float(conf)])


def smoke_test(model, ds: tf.data.Dataset) -> None:
    features, labels = next(iter(ds.take(1)))
    outputs = model(features, training=False, return_attention=True)
    loss = ce_loss(labels, outputs["logits"])
    if not np.isfinite(float(loss.numpy())):
        raise FloatingPointError("Smoke-test loss is not finite.")
    for key in ("image", "geometry_tokens"):
        if not np.isfinite(features[key].numpy()).all():
            raise FloatingPointError(f"Smoke-test input has NaN/Inf: {key}")
    print("SMOKE_1_BATCH_OK", flush=True)
    print(f"  image_batch={features['image'].shape}", flush=True)
    print(f"  geometry_tokens_batch={features['geometry_tokens'].shape}", flush=True)
    print(f"  logits={outputs['logits'].shape}", flush=True)
    print(f"  attention_scores={outputs['attention_scores'].shape}", flush=True)
    print(f"  loss={float(loss.numpy()):.6f}", flush=True)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    configure_runtime(cfg)

    output_dir = resolve_path(cfg["paths"]["output_dir"]) or PROJECT_ROOT / "outputs" / "smirk_geometry_cross_attention"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = resolve_path(args.feature_dir) or resolve_path(cfg.get("geometry_cache", {}).get("feature_dir")) or (output_dir / "geometry_tokens")
    batch_size = int(args.batch_size or cfg["runtime"].get("batch_size_per_gpu", 16))

    split_data = {}
    for split in ("train", "val", "test"):
        geometry_tokens, labels, sample_ids = load_geometry_cache(cfg, split, feature_dir)
        pixels = load_pixels_for_cache(cfg, split, sample_ids, labels)
        split_data[split] = (pixels, geometry_tokens, labels, sample_ids)

    train_ds = make_dataset(split_data["train"][0], split_data["train"][1], split_data["train"][2], cfg, batch_size, training=True, seed=seed)
    val_ds = make_dataset(split_data["val"][0], split_data["val"][1], split_data["val"][2], cfg, batch_size, training=False, seed=seed)
    test_ds = make_dataset(split_data["test"][0], split_data["test"][1], split_data["test"][2], cfg, batch_size, training=False, seed=seed)
    train_loop_ds = maybe_take(train_ds, args.max_train_batches)
    val_loop_ds = maybe_take(val_ds, args.max_eval_batches)

    model = SMIRKGeometryCrossAttentionFER(cfg)
    first_batch = next(iter(train_ds.take(1)))
    _ = model(first_batch[0], training=False)
    restore_rgb_baseline_checkpoint(model, cfg, args)
    optimizer = build_optimizer(cfg)
    smoke_test(model, train_loop_ds)
    if args.smoke_only:
        return 0

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    best_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "best_val_accuracy"), max_to_keep=1)
    last_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "last"), max_to_keep=1)
    best_score = -1.0
    best_epoch = -1
    patience = int(cfg["training"].get("patience", 15))
    history = []

    for epoch in range(int(args.epochs or cfg["training"].get("epochs", 80))):
        start = time.time()
        train_metrics = train_one_epoch(model, optimizer, train_loop_ds)
        val_metrics = evaluate(model, val_loop_ds)
        improved = float(val_metrics["accuracy"]) > best_score
        if improved:
            best_score = float(val_metrics["accuracy"])
            best_epoch = epoch + 1
            best_manager.save(checkpoint_number=epoch + 1)
            print(f"[INFO] Save best_val_accuracy at epoch {epoch+1}: {best_score:.6f}", flush=True)
        last_manager.save(checkpoint_number=epoch + 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_score,
            "alpha": float(model.alpha.numpy()),
            "time_sec": round(time.time() - start, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch+1}: loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} alpha={row['alpha']:.6f}",
            flush=True,
        )
        if best_epoch > 0 and (epoch + 1 - best_epoch) >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}; best_epoch={best_epoch}", flush=True)
            break

    if history:
        keys = list(history[0].keys())
        with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)

    if best_manager.latest_checkpoint:
        checkpoint.restore(best_manager.latest_checkpoint).expect_partial()
        print(f"[INFO] Restored best checkpoint for final test: {best_manager.latest_checkpoint}", flush=True)

    full_val_metrics = evaluate(model, val_ds)
    full_test_metrics = evaluate(model, test_ds)
    save_json(output_dir / "val_metrics_best_val_accuracy.json", {k: v for k, v in full_val_metrics.items() if k not in ("y_true", "y_pred", "confidence")})
    save_json(output_dir / "test_metrics.json", {k: v for k, v in full_test_metrics.items() if k not in ("y_true", "y_pred", "confidence")})
    save_predictions(output_dir / "predictions_smirk_geometry_test.csv", split_data["test"][3], full_test_metrics)
    print("SMIRK_GEOMETRY_CROSS_ATTENTION_FINAL_TEST", flush=True)
    print(f"  accuracy={full_test_metrics['accuracy']:.6f}", flush=True)
    print(f"  macro_f1={full_test_metrics['macro_f1']:.6f}", flush=True)
    print(f"  predictions={output_dir / 'predictions_smirk_geometry_test.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

