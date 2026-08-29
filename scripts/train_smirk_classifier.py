from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_DISABLE_XLA", "1")
os.environ.setdefault("TF_DISABLE_XLA_COMPILATION", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TensorFlow MLP classifier on cached SMIRK FER2013 features.")
    parser.add_argument("--config", type=str, default="config_smirk_only.yaml")
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--run-subdir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
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
    intra_threads = runtime.get("intra_op_threads")
    inter_threads = runtime.get("inter_op_threads")
    if intra_threads:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
    if inter_threads:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
    tf.config.optimizer.set_jit(False)
    if bool(runtime.get("use_mixed_precision", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[WARNING] TensorFlow sees no GPU. Training SMIRK MLP on CPU.", flush=True)
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


def load_split(feature_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = feature_dir / f"{split}_smirk_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing SMIRK feature cache: {path}")
    cache = np.load(path, allow_pickle=False)
    features = cache["features"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    sample_ids = cache["sample_ids"].astype(np.int64)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features for {split}, got {features.shape}")
    if len(features) != len(labels) or len(labels) != len(sample_ids):
        raise ValueError(f"Length mismatch in {path}: features={features.shape}, labels={labels.shape}, ids={sample_ids.shape}")
    if not np.isfinite(features).all():
        raise FloatingPointError(f"NaN/Inf in cached SMIRK features: {path}")
    print(
        f"SMIRK_FEATURE_SHAPE_USED[{split}]={features.shape} "
        f"labels_shape={labels.shape} nan_count={int(np.isnan(features).sum())}",
        flush=True,
    )
    return features, labels, sample_ids


def make_dataset(features: np.ndarray, labels: np.ndarray, batch_size: int, *, training: bool, seed: int):
    ds = tf.data.Dataset.from_tensor_slices((features, labels.astype(np.int32)))
    if training:
        ds = ds.shuffle(min(len(labels), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(feature_dim: int, cfg: Dict) -> tf.keras.Model:
    model_cfg = cfg.get("model", {})
    inputs = tf.keras.Input(shape=(feature_dim,), name="smirk_expression_feature")
    norm = str(model_cfg.get("normalization", "layernorm")).lower()
    if norm in ("batchnorm", "batch_norm", "bn"):
        x = tf.keras.layers.BatchNormalization(name="feature_batchnorm")(inputs)
    elif norm in ("none", "identity", ""):
        x = inputs
    else:
        x = tf.keras.layers.LayerNormalization(name="feature_layernorm")(inputs)

    activation = str(model_cfg.get("activation", "relu"))
    dropout = float(model_cfg.get("dropout", 0.35))
    weight_decay = float(model_cfg.get("l2_weight_decay", cfg.get("training", {}).get("weight_decay", 0.0)))
    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    for i, hidden_dim in enumerate(model_cfg.get("hidden_dims", [128, 64]), start=1):
        x = tf.keras.layers.Dense(
            int(hidden_dim),
            activation=activation,
            kernel_regularizer=regularizer,
            name=f"mlp_dense_{i}",
        )(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    logits = tf.keras.layers.Dense(int(model_cfg.get("num_classes", 7)), name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="SMIRKOnlyFERClassifier")


def build_optimizer(cfg: Dict):
    training = cfg.get("training", {})
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    clipnorm = training.get("grad_clip_norm")
    kwargs = {"learning_rate": lr}
    if clipnorm:
        kwargs["clipnorm"] = float(clipnorm)
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)
    if str(training.get("optimizer", "adamw")).lower() == "adamw" and adamw is not None:
        try:
            return adamw(weight_decay=weight_decay, jit_compile=False, **kwargs)
        except TypeError:
            return adamw(weight_decay=weight_decay, **kwargs)
    return tf.keras.optimizers.Adam(**kwargs)


def evaluate_arrays(model: tf.keras.Model, features: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray, batch_size: int) -> Dict:
    logits = model.predict(features, batch_size=batch_size, verbose=0)
    probs = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1).numpy()
    preds = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1).astype(np.float32)
    labels = labels.astype(np.int64)
    ids = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(labels, preds, labels=ids)
    per_class = {}
    for class_id, class_name in enumerate(EMOTION_NAMES):
        denom = int((labels == class_id).sum())
        per_class[class_name] = float(cm[class_id, class_id] / denom) if denom else 0.0
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=ids, average="macro", zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "preds": preds,
        "confidence": confidence,
        "sample_ids": sample_ids,
        "labels": labels,
    }


def save_prediction_csv(path: Path, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "pred_smirk", "confidence"])
        for sample_id, y_true, pred, conf in zip(
            metrics["sample_ids"],
            metrics["labels"],
            metrics["preds"],
            metrics["confidence"],
        ):
            writer.writerow([int(sample_id), int(y_true), int(pred), float(conf)])


def save_confusion_csv(path: Path, cm) -> None:
    df = pd.DataFrame(cm, index=EMOTION_NAMES, columns=EMOTION_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="y_true\\pred")


def save_metrics(path: Path, metrics: Dict) -> None:
    serializable = {
        key: value
        for key, value in metrics.items()
        if key not in {"preds", "confidence", "sample_ids", "labels"}
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def limit_batches(ds, max_batches: Optional[int]):
    return ds.take(int(max_batches)) if max_batches else ds


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    configure_runtime(cfg)

    output_dir = resolve_path(cfg["paths"]["output_dir"]) or PROJECT_ROOT / "outputs" / "smirk_only"
    if args.run_subdir:
        output_dir = output_dir / args.run_subdir
    feature_dir = resolve_path(args.feature_dir) or resolve_path(cfg["paths"].get("feature_dir")) or (output_dir / "features")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    x_train, y_train, ids_train = load_split(feature_dir, "train")
    x_val, y_val, ids_val = load_split(feature_dir, "val")
    x_test, y_test, ids_test = load_split(feature_dir, "test")
    feature_dim = int(x_train.shape[1])
    configured_dim = int(cfg.get("model", {}).get("feature_dim", feature_dim))
    if configured_dim != feature_dim:
        raise ValueError(f"Config model.feature_dim={configured_dim}, but cached SMIRK feature_dim={feature_dim}")
    print(f"SMIRK_FEATURE_DIM_FOR_CLASSIFIER={feature_dim}", flush=True)

    batch_size = int(args.batch_size or cfg.get("runtime", {}).get("batch_size_per_gpu", 256))
    train_ds = make_dataset(x_train, y_train, batch_size, training=True, seed=seed)
    val_ds = make_dataset(x_val, y_val, batch_size, training=False, seed=seed)
    train_ds_fit = limit_batches(train_ds, args.max_train_batches)
    val_ds_fit = limit_batches(val_ds, args.max_eval_batches)

    model = build_model(feature_dim, cfg)
    model.compile(
        optimizer=build_optimizer(cfg),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary(print_fn=lambda line: print(line, flush=True))

    checkpoint_path = output_dir / "checkpoints" / "best_val_accuracy"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=int(cfg.get("training", {}).get("patience", 20)),
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]
    if str(cfg.get("training", {}).get("scheduler", "")).lower() == "reduce_on_plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                mode="max",
                factor=0.5,
                patience=max(3, int(cfg.get("training", {}).get("patience", 20)) // 4),
                min_lr=1e-6,
                verbose=1,
            )
        )

    if not args.evaluate_only:
        history = model.fit(
            train_ds_fit,
            validation_data=val_ds_fit,
            epochs=int(args.epochs or cfg.get("training", {}).get("epochs", 120)),
            callbacks=callbacks,
            verbose=2,
        )
        hist = pd.DataFrame(history.history)
        hist.to_csv(output_dir / "training_history_full.csv", index=False)
    if checkpoint_path.with_suffix(".index").exists() or checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))
        print(f"[INFO] Restored best_val_accuracy checkpoint: {checkpoint_path}", flush=True)
    else:
        print("[WARNING] No best checkpoint found; evaluating current weights.", flush=True)

    val_metrics = evaluate_arrays(model, x_val, y_val, ids_val, batch_size)
    save_metrics(output_dir / "val_metrics_best_val_accuracy.json", val_metrics)
    test_metrics = evaluate_arrays(model, x_test, y_test, ids_test, batch_size)
    save_metrics(output_dir / "test_metrics.json", test_metrics)
    save_prediction_csv(output_dir / "predictions_smirk_test.csv", test_metrics)
    save_confusion_csv(output_dir / "confusion_matrix_smirk_test.csv", test_metrics["confusion_matrix"])
    np.save(output_dir / "confusion_matrix_smirk_test.npy", np.asarray(test_metrics["confusion_matrix"], dtype=np.int64))

    print("SMIRK_ONLY_FINAL_TEST", flush=True)
    print(f"  accuracy={test_metrics['accuracy']:.6f}", flush=True)
    print(f"  macro_f1={test_metrics['macro_f1']:.6f}", flush=True)
    print(f"  per_class_accuracy={json.dumps(test_metrics['per_class_accuracy'], ensure_ascii=False)}", flush=True)
    print(f"  confusion_matrix={output_dir / 'confusion_matrix_smirk_test.csv'}", flush=True)
    print(f"  predictions_csv={output_dir / 'predictions_smirk_test.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

