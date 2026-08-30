from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from models.smirk_geometry_cross_attention import SMIRKGeometryCrossAttentionFER, resolve_latest_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMIRK Geometry FER with Confidence-Aware Dynamic Gating")
    parser.add_argument("--config", type=str, default="config_smirk_geometry_cross_attention.yaml")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-baseline-checkpoint", action="store_true")
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
    tokens = cache["geometry_tokens"].astype(np.float16)
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
    print(f"GEOMETRY_TOKENS_USED[{split}]={tokens.shape} (float16) labels={labels.shape} nan_count={int(np.isnan(tokens).sum())}", flush=True)
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
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)
    print(f"FER_PIXELS_USED[{split}]={arr.shape}", flush=True)
    return arr


def preprocess_batch_images(images: tf.Tensor, cfg: Dict, training: bool) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    target_size = int(cfg["data"]["image_size"])
    images = tf.image.resize(images, [target_size, target_size], method="bilinear")
    if images.shape[-1] == 1:
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
            "geometry_tokens": geometry_tokens.astype(np.float16),
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
    # Freeze entire baseline backbone & classifier head
    model.rgb_baseline.trainable = False
    for layer in model.rgb_baseline.layers:
        layer.trainable = False
    print(f"RGB_BASELINE_BEST_CHECKPOINT_LOADED path={latest} (FROZEN=True)", flush=True)


def ce_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, tf.cast(logits, tf.float32), from_logits=True))


@tf.function
def train_step(model, optimizer, features, labels):
    with tf.GradientTape() as tape:
        outputs = model(features, training=True)
        final_logits = tf.cast(outputs["logits"], tf.float32)
        baseline_logits = tf.cast(outputs["baseline_logits"], tf.float32)
        gate = tf.cast(outputs["gate"], tf.float32) # [B, 1]
        baseline_confidence = outputs["baseline_confidence"] # [B, 1]

        # 1. Primary Classification Loss
        l_ce = ce_loss(labels, final_logits)

        # 2. Preserve Loss: Penalize changing logits if baseline is correct & confident (>0.6)
        baseline_preds = tf.argmax(baseline_logits, axis=-1, output_type=tf.int32)
        baseline_correct = tf.cast(baseline_preds == labels, tf.float32)
        high_conf = tf.cast(baseline_confidence[:, 0] > 0.6, tf.float32)
        preserve_mask = baseline_correct * high_conf # [B]
        logit_diff_sq = tf.reduce_sum(tf.square(final_logits - baseline_logits), axis=-1) # [B]
        preserve_loss = 0.05 * tf.reduce_mean(preserve_mask * logit_diff_sq)

        # 3. Gate Regularization Loss: Prevent gate from opening too wide everywhere
        gate_reg_loss = 0.01 * tf.reduce_mean(gate)

        total_loss = l_ce + preserve_loss + gate_reg_loss
        if model.losses:
            total_loss = total_loss + tf.add_n([tf.cast(item, tf.float32) for item in model.losses])

    variables = model.trainable_variables
    grads = tape.gradient(total_loss, variables)
    optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])

    preds = tf.argmax(final_logits, axis=-1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(preds == labels, tf.int32))
    batch_size = tf.shape(labels)[0]
    return total_loss, correct, batch_size


@tf.function
def eval_step(model, features, labels):
    outputs = model(features, training=False)
    final_logits = tf.cast(outputs["logits"], tf.float32)
    baseline_logits = tf.cast(outputs["baseline_logits"], tf.float32)
    gate = tf.cast(outputs["gate"], tf.float32)
    loss = ce_loss(labels, final_logits)
    final_probs = tf.nn.softmax(final_logits, axis=-1)
    baseline_probs = tf.nn.softmax(baseline_logits, axis=-1)
    return loss, final_probs, baseline_probs, gate


def train_one_epoch(model, optimizer, ds: tf.data.Dataset) -> Dict[str, float]:
    losses = []
    correct = 0
    total = 0
    for features, labels in ds:
        loss, step_correct, step_total = train_step(model, optimizer, features, labels)
        losses.append(float(loss.numpy()))
        correct += int(step_correct.numpy())
        total += int(step_total.numpy())
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "accuracy": correct / max(total, 1)}


def evaluate(model, ds: tf.data.Dataset) -> Dict:
    losses = []
    y_true = []
    y_baseline = []
    y_fused = []
    all_gates = []
    confidences = []
    for features, labels in ds:
        loss, fused_probs_tensor, baseline_probs_tensor, gate_tensor = eval_step(model, features, labels)
        fused_probs = fused_probs_tensor.numpy()
        baseline_probs = baseline_probs_tensor.numpy()
        gates = gate_tensor.numpy()[:, 0]

        losses.append(float(loss.numpy()))
        y_true.extend(labels.numpy().astype(int).tolist())
        y_baseline.extend(baseline_probs.argmax(axis=1).astype(int).tolist())
        y_fused.extend(fused_probs.argmax(axis=1).astype(int).tolist())
        all_gates.extend(gates.astype(float).tolist())
        confidences.extend(fused_probs.max(axis=1).astype(float).tolist())

    y_true_arr = np.asarray(y_true, dtype=int)
    y_base_arr = np.asarray(y_baseline, dtype=int)
    y_fused_arr = np.asarray(y_fused, dtype=int)
    gates_arr = np.asarray(all_gates, dtype=float)

    base_correct = (y_base_arr == y_true_arr)
    fused_correct = (y_fused_arr == y_true_arr)

    rescue_count = int((~base_correct & fused_correct).sum())
    harmed_count = int((base_correct & ~fused_correct).sum())
    net_gain = rescue_count - harmed_count

    baseline_acc = float(base_correct.mean()) if len(base_correct) > 0 else 0.0
    fused_acc = float(fused_correct.mean()) if len(fused_correct) > 0 else 0.0
    mean_gate = float(gates_arr.mean()) if len(gates_arr) > 0 else 0.0
    gate_correct_baseline = float(gates_arr[base_correct].mean()) if base_correct.any() else 0.0
    gate_wrong_baseline = float(gates_arr[~base_correct].mean()) if (~base_correct).any() else 0.0

    ids = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(y_true, y_fused, labels=ids)
    per_class = {}
    for class_id, class_name in enumerate(EMOTION_NAMES):
        denom = int((y_true_arr == class_id).sum())
        per_class[class_name] = float(cm[class_id, class_id] / denom) if denom else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": fused_acc,
        "baseline_accuracy": baseline_acc,
        "rescue_count": rescue_count,
        "harmed_count": harmed_count,
        "net_gain": net_gain,
        "mean_gate": mean_gate,
        "gate_correct_baseline": gate_correct_baseline,
        "gate_wrong_baseline": gate_wrong_baseline,
        "macro_f1": float(f1_score(y_true, y_fused, labels=ids, average="macro", zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_baseline": y_baseline,
        "y_pred": y_fused,
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
        writer.writerow(["index", "y_true", "pred_baseline", "pred_smirk_geometry_fused", "confidence"])
        for sample_id, y_t, p_b, p_f, conf in zip(sample_ids, metrics["y_true"], metrics["y_baseline"], metrics["y_pred"], metrics["confidence"]):
            writer.writerow([int(sample_id), int(y_t), int(p_b), int(p_f), float(conf)])


def run_baseline_preservation_smoke_test(model: SMIRKGeometryCrossAttentionFER, ds: tf.data.Dataset) -> None:
    features, labels = next(iter(ds.take(1)))

    # 1. Forward pass with dynamic gate
    outputs = model(features, training=False, return_attention=True)
    loss = ce_loss(labels, outputs["logits"])
    if not np.isfinite(float(loss.numpy())):
        raise FloatingPointError("Smoke-test loss is not finite.")

    # 2. Strict baseline preservation test: force_zero_gate = True
    model.force_zero_gate = True
    zero_outputs = model(features, training=False)
    final_logits = zero_outputs["logits"].numpy()
    baseline_logits = zero_outputs["baseline_logits"].numpy()
    max_diff = float(np.max(np.abs(final_logits - baseline_logits)))
    model.force_zero_gate = False

    if max_diff > 1e-4:
        raise ValueError(f"Baseline preservation smoke test failed: gate=0 max logit difference = {max_diff:.6f} > 1e-4")

    print("============================================================", flush=True)
    print("[SMOKE TEST PASSED] Confidence-Aware Sample-Wise Dynamic Gate Verified", flush=True)
    print(f"  gate=0 max logit diff: {max_diff:.8f} (< 0.0001)", flush=True)
    print(f"  sample_gate_shape: {outputs['gate'].shape}", flush=True)
    print(f"  mean_sample_gate: {float(tf.reduce_mean(outputs['gate']).numpy()):.5f}", flush=True)
    print(f"  image_batch: {features['image'].shape}", flush=True)
    print(f"  geometry_tokens_batch: {features['geometry_tokens'].shape}", flush=True)
    print(f"  baseline_logits: {outputs['baseline_logits'].shape}", flush=True)
    print(f"  geometry_delta: {outputs['geometry_delta'].shape}", flush=True)
    print(f"  final_logits: {outputs['logits'].shape}", flush=True)
    print("============================================================", flush=True)


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
    batch_size = int(args.batch_size or cfg["runtime"].get("batch_size_per_gpu", 32))

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

    # Mandatory Smoke Test
    run_baseline_preservation_smoke_test(model, train_loop_ds)
    if args.smoke_only:
        return 0

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    best_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "best_val_accuracy"), max_to_keep=1)
    last_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "last"), max_to_keep=1)
    best_score = -1.0
    best_net_gain = -999999
    best_epoch = -1
    patience = int(cfg["training"].get("patience", 15))
    history = []

    for epoch in range(int(args.epochs or cfg["training"].get("epochs", 80))):
        start = time.time()
        train_metrics = train_one_epoch(model, optimizer, train_loop_ds)
        val_metrics = evaluate(model, val_loop_ds)

        fused_acc = float(val_metrics["accuracy"])
        net_gain = int(val_metrics["net_gain"])
        improved = (fused_acc > best_score) or (abs(fused_acc - best_score) < 1e-6 and net_gain > best_net_gain)

        if improved:
            best_score = fused_acc
            best_net_gain = net_gain
            best_epoch = epoch + 1
            best_manager.save(checkpoint_number=epoch + 1)
            print(f"[INFO] Save best_val_accuracy at epoch {epoch+1}: fused_acc={best_score:.6f} net_gain={best_net_gain}", flush=True)
        last_manager.save(checkpoint_number=epoch + 1)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "baseline_acc": val_metrics["baseline_accuracy"],
            "fused_acc": val_metrics["accuracy"],
            "rescue_count": val_metrics["rescue_count"],
            "harmed_count": val_metrics["harmed_count"],
            "net_gain": val_metrics["net_gain"],
            "mean_gate": val_metrics["mean_gate"],
            "gate_correct_baseline": val_metrics["gate_correct_baseline"],
            "gate_wrong_baseline": val_metrics["gate_wrong_baseline"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_score,
            "time_sec": round(time.time() - start, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch+1:02d}: loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} baseline_acc={row['baseline_acc']:.4f} fused_acc={row['fused_acc']:.4f} "
            f"rescue={row['rescue_count']} harmed={row['harmed_count']} net_gain={row['net_gain']} "
            f"mean_gate={row['mean_gate']:.5f} gate_correct={row['gate_correct_baseline']:.5f} "
            f"gate_wrong={row['gate_wrong_baseline']:.5f} val_f1={row['val_macro_f1']:.4f}",
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
    save_json(output_dir / "val_metrics_best_val_accuracy.json", {k: v for k, v in full_val_metrics.items() if k not in ("y_true", "y_baseline", "y_pred", "confidence")})
    save_json(output_dir / "test_metrics.json", {k: v for k, v in full_test_metrics.items() if k not in ("y_true", "y_baseline", "y_pred", "confidence")})
    save_predictions(output_dir / "predictions_smirk_geometry_test.csv", split_data["test"][3], full_test_metrics)
    print("SMIRK_GEOMETRY_CROSS_ATTENTION_FINAL_TEST", flush=True)
    print(f"  baseline_accuracy={full_test_metrics['baseline_accuracy']:.6f}", flush=True)
    print(f"  fused_accuracy={full_test_metrics['accuracy']:.6f}", flush=True)
    print(f"  rescue_count={full_test_metrics['rescue_count']} harmed_count={full_test_metrics['harmed_count']} net_gain={full_test_metrics['net_gain']}", flush=True)
    print(f"  mean_gate={full_test_metrics['mean_gate']:.5f} gate_correct={full_test_metrics['gate_correct_baseline']:.5f} gate_wrong={full_test_metrics['gate_wrong_baseline']:.5f}", flush=True)
    print(f"  macro_f1={full_test_metrics['macro_f1']:.6f}", flush=True)
    print(f"  predictions={output_dir / 'predictions_smirk_geometry_test.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
