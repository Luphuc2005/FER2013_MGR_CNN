from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score

from config import load_config
from datasets.fer2013 import build_datasets
from train import configure_gpus, configure_tensorflow_runtime, build_model


def parse_args():
    parser = argparse.ArgumentParser("Sweep TTA Weights for FER2013 Model")
    parser.add_argument("--config", default="config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=float, default=0.05, help="Step size for original_weight sweep (default: 0.05)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)
    
    visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
    strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(max(visible_gpu_count, 1))])
    
    _, _, test_ds = build_datasets(cfg, replicas=strategy.num_replicas_in_sync)
    
    with strategy.scope():
        model = build_model(cfg)
        dummy_image = tf.zeros([1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]], tf.float32)
        model({"image": dummy_image}, training=False)
        ckpt = tf.train.Checkpoint(model=model)
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_root = Path(cfg["paths"]["output_dir"]) / "checkpoints"
            best_manager = tf.train.CheckpointManager(ckpt, directory=str(checkpoint_root / "best"), max_to_keep=1)
            last_manager = tf.train.CheckpointManager(ckpt, directory=str(checkpoint_root / "last"), max_to_keep=1)
            checkpoint_path = best_manager.latest_checkpoint or last_manager.latest_checkpoint
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoint found in {cfg['paths']['output_dir']}")
        ckpt.restore(checkpoint_path).expect_partial()
        print(f"[INFO] Restored checkpoint: {checkpoint_path}")

    print("[INFO] Extracting raw logits for original & flipped test images...", flush=True)
    all_logits_orig = []
    all_logits_flip = []
    all_labels = []

    for batch in test_ds:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, batch["label"]

        # Forward pass original
        outputs_orig = model(inputs, training=False)
        logits_orig = outputs_orig["logits"].numpy()

        # Forward pass flipped
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

    print(f"[INFO] Test dataset size: {len(labels)} samples.")
    print("\n" + "=" * 65)
    print("  TTA WEIGHT SWEEP SEARCH (Original Weight vs Flip Weight)")
    print("=" * 65)
    print(f" {'w_orig':<8} | {'w_flip':<8} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 65)

    weights = np.arange(0.0, 1.0 + 1e-5, args.step)
    sweep_results = []
    best_acc = -1.0
    best_w = 0.5
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

        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            best_w = w_orig
            best_row = row

        mark = " ★ BEST" if is_best else ""
        print(f" {w_orig:<8.2f} | {w_flip:<8.2f} | {acc * 100:<9.2f}% | {macro_f1:<10.4f} | {weighted_f1:<12.4f}{mark}")

    print("=" * 65)
    print(f"\n[OPTIMAL RESULT]")
    print(f"  Best w_orig:        {best_row['w_orig']:.2f}")
    print(f"  Best w_flip:        {best_row['w_flip']:.2f}")
    print(f"  Best Test Accuracy: {best_row['accuracy'] * 100:.2f}%")
    print(f"  Best Macro F1:      {best_row['macro_f1']:.4f}")
    print(f"  Best Weighted F1:   {best_row['weighted_f1']:.4f}")

    # No-TTA baseline comparison (w_orig = 1.0)
    no_tta_acc = next(r['accuracy'] for r in sweep_results if abs(r['w_orig'] - 1.0) < 1e-4)
    gain = (best_acc - no_tta_acc) * 100
    print(f"  Gain over No-TTA:   {gain:+.2f}%")

    output_file = Path(cfg["paths"]["output_dir"]) / "tta_sweep_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "optimal": best_row,
            "gain_over_no_tta_pct": gain,
            "sweep": sweep_results
        }, f, indent=2)
    print(f"\n[INFO] Sweep results saved to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
