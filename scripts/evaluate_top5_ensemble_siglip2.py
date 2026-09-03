#!/usr/bin/env python3
"""
Top-5 Checkpoint Ensemble & TTA Evaluator for SigLIP 2 FER Models.

Finds top checkpoints in `checkpoints/best/` and `checkpoints/best_loss/`,
runs TTA (Horizontal Flip) evaluation for each checkpoint, computes Softmax probability
predictions, and averages them to produce the final Ensemble accuracy and metrics on
both Validation and Test sets.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets
from train import build_model, build_optimizer, configure_gpus, configure_tensorflow_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Top-5 Checkpoint Ensemble with TTA for SigLIP2 FER")
    parser.add_argument(
        "--config",
        type=str,
        default="config_convnext_base_ms1m_adaptive_siglip2_confusion_bs16_acc.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Specific directory with checkpoints (default: scans output_dir/checkpoints/best)",
    )
    parser.add_argument("--w-orig", type=float, default=0.40, help="TTA original image weight")
    parser.add_argument("--w-flip", type=float, default=0.60, help="TTA flipped image weight")
    return parser.parse_args()


def get_checkpoint_list(ckpt_dir: Path) -> list[Path]:
    index_files = sorted(ckpt_dir.glob("ckpt-*.index"), key=os.path.getmtime, reverse=True)
    prefixes = [Path(str(p)[:-6]) for p in index_files]
    return prefixes[:5]


def extract_probs(model, dataset, w_orig=0.40, w_flip=0.60):
    all_probs = []
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

        ensemble_logits = w_orig * logits_orig + w_flip * logits_flip

        exp_logits = np.exp(ensemble_logits - np.max(ensemble_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return probs_arr, labels_arr


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)

    visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
    strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(max(visible_gpu_count, 1))])

    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = output_dir / "checkpoints" / "best"

    ckpt_prefixes = get_checkpoint_list(ckpt_dir)
    if not ckpt_prefixes:
        # Fallback to output_dir / checkpoints / best_loss
        ckpt_dir = output_dir / "checkpoints" / "best_loss"
        ckpt_prefixes = get_checkpoint_list(ckpt_dir)

    if not ckpt_prefixes:
        print(f"[ERROR] No checkpoint index files found in {output_dir}")
        return 1

    print("\n" + "=" * 70)
    print(f" TOP-5 CHECKPOINT ENSEMBLE + TTA EVALUATION")
    print(f" Config:         {args.config}")
    print(f" Checkpoint Dir: {ckpt_dir}")
    print(f" Checkpoints:    {len(ckpt_prefixes)} found")
    print(f" TTA Weights:    Orig={args.w_orig:.2f}, Flip={args.w_flip:.2f}")
    print("=" * 70)

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
        ckpt = tf.train.Checkpoint(
            epoch=ckpt_epoch,
            best_metric=ckpt_best_metric,
            model=model,
            optimizer_head=optimizer_head,
            optimizer_backbone=optimizer_backbone,
        )

    val_probs_list = []
    test_probs_list = []
    val_labels = None
    test_labels = None

    for idx, prefix in enumerate(ckpt_prefixes, start=1):
        print(f"\n[EVAL] [{idx}/{len(ckpt_prefixes)}] Loading checkpoint: {prefix.name} ...")
        ckpt.restore(str(prefix)).expect_partial()

        v_probs, v_labs = extract_probs(model, val_ds, w_orig=args.w_orig, w_flip=args.w_flip)
        t_probs, t_labs = extract_probs(model, test_ds, w_orig=args.w_orig, w_flip=args.w_flip)

        v_acc = accuracy_score(v_labs, np.argmax(v_probs, axis=1))
        t_acc = accuracy_score(t_labs, np.argmax(t_probs, axis=1))
        print(f"       -> Single Model TTA Acc | Val: {v_acc*100:.2f}% | Test: {t_acc*100:.2f}%")

        val_probs_list.append(v_probs)
        test_probs_list.append(t_probs)
        val_labels = v_labs
        test_labels = t_labs

    # --- TOP-K ENSEMBLE AVERAGE ---
    val_ens_probs = np.mean(val_probs_list, axis=0)
    test_ens_probs = np.mean(test_probs_list, axis=0)

    val_ens_preds = np.argmax(val_ens_probs, axis=1)
    test_ens_preds = np.argmax(test_ens_probs, axis=1)

    val_ens_acc = accuracy_score(val_labels, val_ens_preds)
    test_ens_acc = accuracy_score(test_labels, test_ens_preds)

    test_macro_f1 = f1_score(test_labels, test_ens_preds, average="macro")
    test_weighted_f1 = f1_score(test_labels, test_ens_preds, average="weighted")
    test_report = classification_report(test_labels, test_ens_preds, target_names=EMOTION_NAMES, output_dict=True, zero_division=0)
    test_cm = confusion_matrix(test_labels, test_ens_preds).tolist()

    print("\n" + "*" * 70)
    print(f" 🔥 TOP-{len(ckpt_prefixes)} ENSEMBLE + TTA FINAL RESULTS 🔥")
    print(f"   VAL ENSEMBLE ACCURACY:  {val_ens_acc * 100:.2f}%")
    print(f"   TEST ENSEMBLE ACCURACY: {test_ens_acc * 100:.2f}%")
    print(f"   TEST MACRO F1:          {test_macro_f1:.4f}")
    print(f"   TEST WEIGHTED F1:       {test_weighted_f1:.4f}")
    print("*" * 70)

    report_data = {
        "num_checkpoints": len(ckpt_prefixes),
        "val_ensemble_acc": float(val_ens_acc),
        "test_ensemble_acc": float(test_ens_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm,
        "checkpoints": [str(p) for p in ckpt_prefixes],
    }

    out_file = output_dir / "top5_ensemble_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[INFO] Full ensemble report saved to {out_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
