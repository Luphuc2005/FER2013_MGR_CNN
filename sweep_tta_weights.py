from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from config import load_config
from datasets.fer2013 import build_datasets
from train import configure_gpus, configure_tensorflow_runtime, build_model, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser("Sweep TTA Weights for FER2013 Model")
    parser.add_argument("--config", default="config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=float, default=0.05, help="Step size for original_weight sweep (default: 0.05)")
    return parser.parse_args()


def extract_dataset_logits(model, dataset):
    all_logits_orig = []
    all_logits_flip = []
    all_labels = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, batch["label"]

        outputs_orig = model(inputs, training=False)
        logits_orig = outputs_orig["logits"].numpy()

        flipped_inputs = dict(inputs)
        flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
        if "mask" in inputs:
            flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
        outputs_flip = model(flipped_inputs, training=False)
        logits_flip = outputs_flip["logits"].numpy()

        all_logits_orig.append(logits_orig)
        all_logits_flip.append(logits_flip)
        all_labels.append(labels.numpy())

    logits_orig = np.concatenate(all_logits_orig, axis=0)
    logits_flip = np.concatenate(all_logits_flip, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits_orig, logits_flip, labels


def eval_weights(logits_orig, logits_flip, labels, step=0.05):
    weights = np.arange(0.0, 1.0 + 1e-5, step)
    sweep_results = []
    best_acc = -1.0
    best_row = None

    for w_orig in weights:
        w_orig = round(float(w_orig), 4)
        w_flip = round(1.0 - w_orig, 4)

        ensemble_logits = w_orig * logits_orig + w_flip * logits_flip
        preds = np.argmax(ensemble_logits, axis=-1)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")

        row = {
            "w_orig": w_orig,
            "w_flip": w_flip,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        }
        sweep_results.append(row)

        if acc > best_acc:
            best_acc = acc
            best_row = row

    return sweep_results, best_row


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)
    
    visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
    strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(max(visible_gpu_count, 1))])
    
    _, val_ds, test_ds = build_datasets(cfg, replicas=strategy.num_replicas_in_sync)
    
    first_batch = next(iter(val_ds.take(1)))
    first_inputs = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch

    with strategy.scope():
        model = build_model(cfg)
        _ = model(first_inputs, training=False)
        ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt_best_metric = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
        optimizer_head = build_optimizer(cfg, float(cfg["training"]["lr"]))
        optimizer_backbone = build_optimizer(cfg, float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])))
        checkpoint = tf.train.Checkpoint(
            epoch=ckpt_epoch,
            best_metric=ckpt_best_metric,
            model=model,
            optimizer_head=optimizer_head,
            optimizer_backbone=optimizer_backbone,
        )
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_root = Path(cfg["paths"]["output_dir"]) / "checkpoints"
            best_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "best"), max_to_keep=1)
            best_loss_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "best_loss"), max_to_keep=1)
            last_manager = tf.train.CheckpointManager(checkpoint, directory=str(checkpoint_root / "last"), max_to_keep=1)
            checkpoint_path = best_manager.latest_checkpoint or best_loss_manager.latest_checkpoint or last_manager.latest_checkpoint
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoint found in {cfg['paths']['output_dir']}")
        checkpoint.restore(checkpoint_path).expect_partial()
        print(f"[INFO] Restored checkpoint: {checkpoint_path}")

    print("\n[INFO] Extracting raw logits for VALIDATION set...", flush=True)
    val_orig, val_flip, val_labels = extract_dataset_logits(model, val_ds)
    val_sweep, val_best = eval_weights(val_orig, val_flip, val_labels, step=args.step)

    print("\n" + "=" * 65)
    print("  1. VALIDATION SET SWEEP RESULTS")
    print("=" * 65)
    print(f" {'w_orig':<8} | {'w_flip':<8} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 65)
    for r in val_sweep:
        mark = " ★ BEST" if r['w_orig'] == val_best['w_orig'] else ""
        print(f" {r['w_orig']:<8.2f} | {r['w_flip']:<8.2f} | {r['accuracy'] * 100:<9.2f}% | {r['macro_f1']:<10.4f} | {r['weighted_f1']:<12.4f}{mark}")
    print("=" * 65)

    print("\n[INFO] Extracting raw logits for TEST set...", flush=True)
    test_orig, test_flip, test_labels = extract_dataset_logits(model, test_ds)
    test_sweep, test_direct_best = eval_weights(test_orig, test_flip, test_labels, step=args.step)

    # Validation-tuned Test evaluation
    opt_w_orig = val_best["w_orig"]
    test_val_tuned = next(r for r in test_sweep if abs(r["w_orig"] - opt_w_orig) < 1e-4)
    no_tta_test = next(r for r in test_sweep if abs(r["w_orig"] - 1.0) < 1e-4)
    gain_test = (test_val_tuned["accuracy"] - no_tta_test["accuracy"]) * 100

    print("\n" + "=" * 65)
    print("  2. TEST SET SWEEP RESULTS")
    print("=" * 65)
    print(f" {'w_orig':<8} | {'w_flip':<8} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 65)
    for r in test_sweep:
        mark = " ★ VAL-TUNED" if r['w_orig'] == opt_w_orig else (" ★ TEST-BEST" if r['w_orig'] == test_direct_best['w_orig'] else "")
        print(f" {r['w_orig']:<8.2f} | {r['w_flip']:<8.2f} | {r['accuracy'] * 100:<9.2f}% | {r['macro_f1']:<10.4f} | {r['weighted_f1']:<12.4f}{mark}")
    print("=" * 65)

    print(f"\n[FINAL OPTIMAL RESULTS - VAL TUNED ON TEST SET]")
    print(f"  Validation Optimal w_orig: {val_best['w_orig']:.2f}")
    print(f"  Validation Optimal w_flip: {val_best['w_flip']:.2f}")
    print(f"  Val Accuracy (TTA):         {val_best['accuracy'] * 100:.2f}%")
    print(f"  Test Accuracy (No TTA):    {no_tta_test['accuracy'] * 100:.2f}%")
    print(f"  Test Accuracy (TTA):       {test_val_tuned['accuracy'] * 100:.2f}%")
    print(f"  Test Macro F1 (TTA):       {test_val_tuned['macro_f1']:.4f}")
    print(f"  Test Weighted F1 (TTA):    {test_val_tuned['weighted_f1']:.4f}")
    print(f"  TTA Improvement Gain:      {gain_test:+.2f}%")

    output_file = Path(cfg["paths"]["output_dir"]) / "tta_sweep_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "val_optimal": val_best,
            "test_val_tuned": test_val_tuned,
            "test_direct_best": test_direct_best,
            "test_no_tta": no_tta_test,
            "test_tta_gain_pct": gain_test,
            "val_sweep": val_sweep,
            "test_sweep": test_sweep,
        }, f, indent=2)
    print(f"\n[INFO] Complete sweep results saved to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

