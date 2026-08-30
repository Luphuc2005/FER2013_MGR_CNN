from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import EMOTION_NAMES
from models.stage2a_smirk_delta_mesh_gnn import Stage2ASMIRKDeltaMeshGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2A SMIRK Delta Mesh 3D-Only FER Probe.")
    parser.add_argument("--config", type=str, default="config_stage2a_smirk_delta_mesh_probe.yaml")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--smoke-only", action="store_true", help="Run 1-batch 2-epoch smoke test verification.")
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


def load_cache_split(cache_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"stage2a_delta_mesh_cache_{split}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing Stage 2A cache for split {split}: {cache_path}. Run extract_stage2a_smirk_delta_mesh.py first.")

    data = np.load(cache_path)
    region_features = data["region_features"].astype(np.float32)  # [N, 12, 10]
    labels = data["labels"].astype(np.int64)  # [N]
    sample_ids = data["sample_ids"].astype(np.int64)  # [N]

    if not np.isfinite(region_features).all():
        raise FloatingPointError(f"NaN or Inf detected in cache for split {split}!")
    if region_features.shape[0] != labels.shape[0]:
        raise ValueError(f"Length mismatch in {split} cache: features={region_features.shape[0]} vs labels={labels.shape[0]}")

    print(f"[INFO] Loaded Stage 2A cache for {split}: region_features={region_features.shape}, labels={labels.shape}", flush=True)
    return region_features, labels, sample_ids


def create_tf_dataset(region_features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((region_features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_model(model: Stage2ASMIRKDeltaMeshGNN, dataset: tf.data.Dataset) -> Dict[str, object]:
    all_logits = []
    all_labels = []

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataset:
        out = model(x_batch, training=False)
        logits = out["logits"]
        loss = loss_fn(y_batch, logits)

        batch_size = tf.shape(x_batch)[0].numpy()
        total_loss += loss.numpy() * batch_size
        total_samples += batch_size

        all_logits.append(logits.numpy())
        all_labels.append(y_batch.numpy())

    avg_loss = float(total_loss / max(1, total_samples))
    all_logits_arr = np.concatenate(all_logits, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)

    preds = np.argmax(all_logits_arr, axis=1)
    acc = float(np.mean(preds == all_labels_arr))
    macro_f1 = float(f1_score(all_labels_arr, preds, average="macro", zero_division=0))

    report = classification_report(
        all_labels_arr,
        preds,
        labels=list(range(len(EMOTION_NAMES))),
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(all_labels_arr, preds, labels=list(range(7)))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "predictions": preds,
        "labels": all_labels_arr,
        "report": report,
        "confusion_matrix": conf_mat.tolist(),
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    cache_dir = resolve_path(args.cache_dir or cfg.get("paths", {}).get("cache_dir", "outputs/stage2a_smirk_delta_mesh_probe/cache"))
    output_dir = resolve_path(args.output_dir or cfg.get("paths", {}).get("output_dir", "outputs/stage2a_smirk_delta_mesh_probe"))
    ckpt_dir = resolve_path(cfg.get("paths", {}).get("checkpoints_dir", output_dir / "checkpoints"))

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(args.batch_size or cfg.get("runtime", {}).get("batch_size_per_gpu", 128))
    epochs = int(args.epochs or cfg.get("training", {}).get("epochs", 100))
    learning_rate = float(args.lr or cfg.get("training", {}).get("lr", 0.001))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.0001))

    # 1. Load Caches
    train_x, train_y, _ = load_cache_split(cache_dir, "train")
    val_x, val_y, _ = load_cache_split(cache_dir, "val")
    test_x, test_y, _ = load_cache_split(cache_dir, "test")

    if args.smoke_only:
        train_x, train_y = train_x[:32], train_y[:32]
        val_x, val_y = val_x[:32], val_y[:32]
        test_x, test_y = test_x[:32], test_y[:32]
        epochs = 2
        batch_size = 16
        print("[INFO] Running in --smoke-only mode with 32 samples & 2 epochs.", flush=True)

    train_ds = create_tf_dataset(train_x, train_y, batch_size=batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_x, val_y, batch_size=batch_size, shuffle=False)
    test_ds = create_tf_dataset(test_x, test_y, batch_size=batch_size, shuffle=False)

    # 2. Build Stage 2A Model
    model = Stage2ASMIRKDeltaMeshGNN(cfg)

    # Warmup call for shape trace and parameter count
    dummy_input = tf.zeros((2, 12, 10), dtype=tf.float32)
    dummy_out = model(dummy_input, training=False)

    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_variables)

    print("\n" + "=" * 65, flush=True)
    print(" STAGE 2A MODEL COMPONENT & GRADIENT FLOW VERIFICATION", flush=True)
    print("=" * 65, flush=True)
    print(f"Model Name: {model.name}", flush=True)
    print(f"Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"Non-Trainable Parameters: {non_trainable_params:,}", flush=True)
    print(f"Total Parameters: {trainable_params + non_trainable_params:,}", flush=True)

    assert trainable_params > 0, "Model has 0 trainable parameters!"

    # 3. Test Gradient Flow (Fail-fast check 7)
    with tf.GradientTape() as tape:
        logits = model(dummy_input, training=True)["logits"]
        loss_val = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.constant([0, 1]), logits, from_logits=True))
    grads = tape.gradient(loss_val, model.trainable_variables)

    print(f"Gradient Check across {len(model.trainable_variables)} trainable tensors:", flush=True)
    for var, grad in zip(model.trainable_variables, grads):
        assert grad is not None, f"Gradient is None for variable: {var.name}"
        assert not np.isnan(grad.numpy()).any(), f"NaN in gradient for variable: {var.name}"
    print("  --> PASS: Gradient flows exclusively into Graph Encoder + Classifier Head.", flush=True)
    print("=" * 65 + "\n", flush=True)

    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is None:
        adamw = getattr(getattr(tf.keras.optimizers, "experimental", object()), "AdamW", None)
    if adamw is not None:
        try:
            optimizer = adamw(learning_rate=learning_rate, weight_decay=weight_decay, jit_compile=False)
        except TypeError:
            optimizer = adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience = int(cfg.get("training", {}).get("patience", 20))
    patience_counter = 0

    best_ckpt_path = ckpt_dir / "best_stage2a_delta_mesh_gnn.h5"

    print(f"[INFO] Starting Stage 2A training for {epochs} epochs (lr={learning_rate}, batch_size={batch_size})...", flush=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                logits = out["logits"]
                loss = loss_fn(y_batch, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            b_size = tf.shape(x_batch)[0].numpy()
            total_loss += loss.numpy() * b_size
            preds = np.argmax(logits.numpy(), axis=1)
            correct_preds += np.sum(preds == y_batch.numpy())
            total_samples += b_size

        train_loss = total_loss / max(1, total_samples)
        train_acc = correct_preds / max(1, total_samples)

        val_metrics = evaluate_model(model, val_ds)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Val Macro F1: {val_macro_f1:.4f}",
            flush=True,
        )

        # Save Best Checkpoint (Primary: val_accuracy, Tie-break: val_macro_f1)
        if (val_acc > best_val_acc) or (abs(val_acc - best_val_acc) < 1e-6 and val_macro_f1 > best_val_macro_f1):
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            model.save_weights(str(best_ckpt_path))
            print(f"  [SAVED] New best Stage 2A checkpoint -> Val Acc: {val_acc * 100:.2f}%, F1: {val_macro_f1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Triggered after {patience} epochs without improvement.", flush=True)
                break

    # 5. Final Evaluation on Test Set using Best Checkpoint
    if best_ckpt_path.exists():
        model.load_weights(str(best_ckpt_path))
        print(f"\n[INFO] Loaded best model checkpoint from epoch {best_epoch} for final test evaluation.", flush=True)

    test_metrics = evaluate_model(model, test_ds)

    print("\n" + "=" * 65, flush=True)
    print(" FINAL STAGE 2A TEST EVALUATION RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Best Epoch: {best_epoch}", flush=True)
    print(f"Test Loss:     {test_metrics['loss']:.4f}", flush=True)
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%", flush=True)
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}", flush=True)
    print("\nPer-Class Performance:", flush=True)
    for emotion, metrics in test_metrics["report"].items():
        if emotion in EMOTION_NAMES:
            print(f"  {emotion:10s} -> Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}, Support: {metrics['support']}", flush=True)

    print("\nConfusion Matrix:", flush=True)
    print(np.array(test_metrics["confusion_matrix"]), flush=True)
    print("=" * 65 + "\n", flush=True)

    summary_meta = {
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "per_class_report": test_metrics["report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "checkpoint_path": str(best_ckpt_path),
    }

    with (output_dir / "stage2a_training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
