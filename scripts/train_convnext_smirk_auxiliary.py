from __future__ import annotations

import argparse
import csv
import json
import random
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from models.convnext_smirk_auxiliary import ConvNeXtSMIRKAuxiliaryFER, resolve_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt FER with SMIRK 3D Expression Auxiliary Supervision")
    parser.add_argument("--config", type=str, default="config_convnext_smirk_auxiliary.yaml")
    parser.add_argument("--ablation", type=str, default=None, choices=("baseline", "exp", "exp_jaw", "exp_jaw_head"))
    parser.add_argument("--lambda-geo", type=float, default=None)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--auxiliary-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-baseline-checkpoint", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
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
        print("[INFO] TensorFlow mixed_float16 enabled", flush=True)
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


def load_3d_targets_for_split(aux_dir: Path, split: str, ablation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = aux_dir / f"smirk_3d_targets_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing 3D auxiliary target cache for {split}: {path}. Run scripts/extract_smirk_auxiliary_targets.py first.")
    data = np.load(path, allow_pickle=False)
    exp = data["expression_params"].astype(np.float32) # [N, 50]
    jaw = data["jaw_params"].astype(np.float32)       # [N, 3]
    head = data["head_pose_params"].astype(np.float32) # [N, 3]
    labels = data["labels"].astype(np.int64)
    sample_ids = data["sample_ids"].astype(np.int64)

    if ablation == "baseline":
        targets = np.zeros((len(labels), 0), dtype=np.float32)
    elif ablation == "exp":
        targets = exp
    elif ablation == "exp_jaw":
        targets = np.concatenate([exp, jaw], axis=-1)
    elif ablation in ("exp_jaw_head", "all"):
        targets = np.concatenate([exp, jaw, head], axis=-1)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    print(f"[TARGETS] {split} split ablation={ablation} shape={targets.shape}", flush=True)
    return targets, labels, sample_ids


def load_pixels_for_targets(cfg: Dict, split: str, sample_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
            raise KeyError(f"sample_id={int(sample_id)} is missing in FER2013 {split} split.")
        if int(records.labels[pos]) != int(label):
            raise ValueError(f"Label mismatch for {split} sample_id={int(sample_id)}")
        pixels.append(records.images[pos])
    arr = np.stack(pixels, axis=0).astype(np.uint8)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)
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
    norm_targets: np.ndarray,
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
            "norm_targets": norm_targets.astype(np.float32),
            "labels": labels.astype(np.int32),
        }
    )
    if training:
        ds = ds.shuffle(min(len(labels), int(cfg["data"].get("shuffle_buffer", 10000))), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    def batch_mapper(item):
        features = {
            "image": preprocess_batch_images(item["pixels"], cfg, training),
            "norm_targets": tf.cast(item["norm_targets"], tf.float32),
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
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    clipnorm = cfg["training"].get("grad_clip_norm")
    kwargs = {"learning_rate": lr}
    if clipnorm:
        kwargs["clipnorm"] = float(clipnorm)
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is not None:
        try:
            return adamw(weight_decay=weight_decay, jit_compile=False, **kwargs)
        except TypeError:
            return adamw(weight_decay=weight_decay, **kwargs)
    return tf.keras.optimizers.Adam(**kwargs)


def restore_rgb_baseline_checkpoint(model: ConvNeXtSMIRKAuxiliaryFER, cfg: Dict, args: argparse.Namespace) -> None:
    if args.skip_baseline_checkpoint:
        print("[WARNING] Skipping baseline restore.", flush=True)
        return
    ckpt_cfg = cfg.get("baseline_checkpoint", {})
    ckpt_path = args.baseline_checkpoint or ckpt_cfg.get("best_checkpoint_dir")
    latest = resolve_latest_checkpoint(ckpt_path)
    if latest is None:
        message = f"Best ConvNeXt baseline checkpoint not found: {ckpt_path}"
        if bool(ckpt_cfg.get("require", True)):
            raise FileNotFoundError(message)
        print(f"[WARNING] {message}", flush=True)
        return
    status = tf.train.Checkpoint(model=model.rgb_baseline).restore(latest)
    status.expect_partial()
    # Re-apply Stage 1-2 freezing after restoring checkpoint weights
    model._freeze_stage1_2()
    print(f"RGB_BASELINE_CHECKPOINT_RESTORED path={latest}", flush=True)


def ce_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, tf.cast(logits, tf.float32), from_logits=True))


def geometry_regression_loss(targets: tf.Tensor, predictions: tf.Tensor, loss_type: str = "smooth_l1") -> tf.Tensor:
    if targets.shape[-1] == 0 or predictions is None:
        return tf.constant(0.0, dtype=tf.float32)
    targets_f32 = tf.cast(targets, tf.float32)
    preds_f32 = tf.cast(predictions, tf.float32)
    if loss_type == "smooth_l1":
        huber = tf.keras.losses.Huber(delta=1.0)
        return huber(targets_f32, preds_f32)
    return tf.reduce_mean(tf.square(targets_f32 - preds_f32))


@tf.function
def train_step(model, optimizer, features, labels, lambda_geo: float, loss_type: str, backbone_lr_mult: float):
    with tf.GradientTape() as tape:
        outputs = model(features, training=True)
        logits = tf.cast(outputs["logits"], tf.float32)
        l_fer = ce_loss(labels, logits)

        if outputs.get("geometry_pred") is not None and features["norm_targets"].shape[-1] > 0:
            l_geo = geometry_regression_loss(features["norm_targets"], outputs["geometry_pred"], loss_type=loss_type)
        else:
            l_geo = tf.constant(0.0, dtype=tf.float32)

        total_loss = l_fer + lambda_geo * l_geo
        if model.losses:
            total_loss = total_loss + tf.add_n([tf.cast(item, tf.float32) for item in model.losses])

    variables = model.trainable_variables
    grads = tape.gradient(total_loss, variables)

    # Apply differential learning rate (scale backbone gradients by multiplier)
    scaled_grads_vars = []
    for g, v in zip(grads, variables):
        if g is not None:
            if "convnext" in v.name.lower() or "backbone" in v.name.lower() or "stage" in v.name.lower():
                scaled_grads_vars.append((g * backbone_lr_mult, v))
            else:
                scaled_grads_vars.append((g, v))

    optimizer.apply_gradients(scaled_grads_vars)

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(preds == labels, tf.int32))
    batch_size = tf.shape(labels)[0]
    return total_loss, l_fer, l_geo, correct, batch_size


@tf.function
def eval_step(model, features, labels, loss_type: str):
    outputs = model(features, training=False)
    logits = tf.cast(outputs["logits"], tf.float32)
    l_fer = ce_loss(labels, logits)

    if outputs.get("geometry_pred") is not None and features["norm_targets"].shape[-1] > 0:
        l_geo = geometry_regression_loss(features["norm_targets"], outputs["geometry_pred"], loss_type=loss_type)
    else:
        l_geo = tf.constant(0.0, dtype=tf.float32)

    probs = tf.nn.softmax(logits, axis=-1)
    return l_fer, l_geo, probs


def train_one_epoch(model, optimizer, ds: tf.data.Dataset, lambda_geo: float, loss_type: str, backbone_lr_mult: float) -> Dict[str, float]:
    tot_losses = []
    fer_losses = []
    geo_losses = []
    correct = 0
    total = 0
    for features, labels in ds:
        t_loss, f_loss, g_loss, step_correct, step_total = train_step(
            model, optimizer, features, labels, lambda_geo=lambda_geo, loss_type=loss_type, backbone_lr_mult=backbone_lr_mult
        )
        tot_losses.append(float(t_loss.numpy()))
        fer_losses.append(float(f_loss.numpy()))
        geo_losses.append(float(g_loss.numpy()))
        correct += int(step_correct.numpy())
        total += int(step_total.numpy())

    return {
        "train_loss": float(np.mean(tot_losses)) if tot_losses else float("nan"),
        "fer_loss": float(np.mean(fer_losses)) if fer_losses else float("nan"),
        "geometry_loss": float(np.mean(geo_losses)) if geo_losses else float("nan"),
        "accuracy": correct / max(total, 1),
    }


def evaluate(model, ds: tf.data.Dataset, loss_type: str) -> Dict:
    fer_losses = []
    geo_losses = []
    y_true = []
    y_pred = []
    confidences = []
    for features, labels in ds:
        f_loss, g_loss, probs_tensor = eval_step(model, features, labels, loss_type=loss_type)
        probs = probs_tensor.numpy()
        fer_losses.append(float(f_loss.numpy()))
        geo_losses.append(float(g_loss.numpy()))
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(probs.argmax(axis=1).astype(int).tolist())
        confidences.extend(probs.max(axis=1).astype(float).tolist())

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    accuracy = float((y_pred_arr == y_true_arr).mean()) if len(y_true_arr) > 0 else 0.0

    ids = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    per_class = {}
    for class_id, class_name in enumerate(EMOTION_NAMES):
        denom = int((y_true_arr == class_id).sum())
        per_class[class_name] = float(cm[class_id, class_id] / denom) if denom else 0.0

    return {
        "fer_loss": float(np.mean(fer_losses)) if fer_losses else float("nan"),
        "geometry_loss": float(np.mean(geo_losses)) if geo_losses else float("nan"),
        "accuracy": accuracy,
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
        writer.writerow(["index", "y_true", "pred_fer", "confidence"])
        for sample_id, y_t, p_f, conf in zip(sample_ids, metrics["y_true"], metrics["y_pred"], metrics["confidence"]):
            writer.writerow([int(sample_id), int(y_t), int(p_f), float(conf)])


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    configure_runtime(cfg)

    ablation = args.ablation or str(cfg.get("model", {}).get("ablation", "exp_jaw_head"))
    cfg.setdefault("model", {})["ablation"] = ablation
    lambda_geo = float(args.lambda_geo if args.lambda_geo is not None else cfg.get("auxiliary", {}).get("lambda_geo", 0.5))
    loss_type = str(cfg.get("auxiliary", {}).get("loss_type", "smooth_l1"))
    backbone_lr_mult = float(cfg.get("training", {}).get("backbone_lr_multiplier", 0.1))

    output_dir = resolve_path(cfg["paths"]["output_dir"]) / f"ablation_{ablation}"
    output_dir.mkdir(parents=True, exist_ok=True)
    aux_dir = resolve_path(args.auxiliary_dir) or resolve_path(cfg.get("paths", {}).get("auxiliary_target_dir")) or (PROJECT_ROOT / "outputs" / "smirk_auxiliary" / "3d_targets")
    batch_size = int(args.batch_size or cfg["runtime"].get("batch_size_per_gpu", 32))

    # Load targets & pixels
    raw_targets = {}
    split_data = {}
    for split in ("train", "val", "test"):
        targets, labels, sample_ids = load_3d_targets_for_split(aux_dir, split, ablation)
        pixels = load_pixels_for_targets(cfg, split, sample_ids, labels)
        raw_targets[split] = targets
        split_data[split] = (pixels, labels, sample_ids)

    # Normalize targets per group on TRAIN split
    if raw_targets["train"].shape[-1] > 0:
        mean_train = raw_targets["train"].mean(axis=0, keepdims=True)
        std_train = raw_targets["train"].std(axis=0, keepdims=True) + 1e-7
        print(f"[NORM] Train target mean={mean_train.mean():.4f} std={std_train.mean():.4f}", flush=True)
    else:
        mean_train = np.zeros((1, 0), dtype=np.float32)
        std_train = np.ones((1, 0), dtype=np.float32)

    norm_targets = {split: (raw_targets[split] - mean_train) / std_train for split in ("train", "val", "test")}

    train_ds = make_dataset(split_data["train"][0], norm_targets["train"], split_data["train"][1], cfg, batch_size, training=True, seed=seed)
    val_ds = make_dataset(split_data["val"][0], norm_targets["val"], split_data["val"][1], cfg, batch_size, training=False, seed=seed)
    test_ds = make_dataset(split_data["test"][0], norm_targets["test"], split_data["test"][1], cfg, batch_size, training=False, seed=seed)

    train_loop_ds = maybe_take(train_ds, args.max_train_batches)
    val_loop_ds = maybe_take(val_ds, args.max_eval_batches)

    model = ConvNeXtSMIRKAuxiliaryFER(cfg)
    first_batch = next(iter(train_ds.take(1)))
    _ = model(first_batch[0], training=False)
    restore_rgb_baseline_checkpoint(model, cfg, args)
    optimizer = build_optimizer(cfg)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    best_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "best_val_accuracy"), max_to_keep=1)
    last_manager = tf.train.CheckpointManager(checkpoint, str(output_dir / "checkpoints" / "last"), max_to_keep=1)
    best_score = -1.0
    best_epoch = -1
    patience = int(cfg["training"].get("patience", 15))
    history = []

    print(f"============================================================", flush=True)
    print(f" STARTING AUXILIARY 3D TRAINING: Ablation={ablation} LambdaGeo={lambda_geo}", flush=True)
    print(f"============================================================", flush=True)

    for epoch in range(int(args.epochs or cfg["training"].get("epochs", 60))):
        start = time.time()
        train_metrics = train_one_epoch(model, optimizer, train_loop_ds, lambda_geo=lambda_geo, loss_type=loss_type, backbone_lr_mult=backbone_lr_mult)
        val_metrics = evaluate(model, val_loop_ds, loss_type=loss_type)

        val_acc = float(val_metrics["accuracy"])
        improved = val_acc > best_score

        if improved:
            best_score = val_acc
            best_epoch = epoch + 1
            best_manager.save(checkpoint_number=epoch + 1)
            print(f"[INFO] Save best_val_accuracy at epoch {epoch+1}: {best_score:.6f}", flush=True)
        last_manager.save(checkpoint_number=epoch + 1)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "fer_loss": train_metrics["fer_loss"],
            "geometry_loss": train_metrics["geometry_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_fer_loss": val_metrics["fer_loss"],
            "val_geometry_loss": val_metrics["geometry_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_score,
            "time_sec": round(time.time() - start, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch+1:02d}: train_loss={row['train_loss']:.4f} fer_loss={row['fer_loss']:.4f} "
            f"geo_loss={row['geometry_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}",
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

    full_val_metrics = evaluate(model, val_ds, loss_type=loss_type)
    full_test_metrics = evaluate(model, test_ds, loss_type=loss_type)
    save_json(output_dir / "val_metrics_best.json", {k: v for k, v in full_val_metrics.items() if k not in ("y_true", "y_pred", "confidence")})
    save_json(output_dir / "test_metrics.json", {k: v for k, v in full_test_metrics.items() if k not in ("y_true", "y_pred", "confidence")})
    save_predictions(output_dir / "predictions_test.csv", split_data["test"][2], full_test_metrics)

    print("CONVNEXT_SMIRK_AUXILIARY_FINAL_TEST", flush=True)
    print(f"  ablation={ablation}", flush=True)
    print(f"  test_accuracy={full_test_metrics['accuracy']:.6f}", flush=True)
    print(f"  test_macro_f1={full_test_metrics['macro_f1']:.6f}", flush=True)
    print(f"  predictions={output_dir / 'predictions_test.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
