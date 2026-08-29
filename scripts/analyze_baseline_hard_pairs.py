from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure root directory is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets, collect_split_records, _resolve_path
from train import build_model, configure_gpus, configure_tensorflow_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Hard Pairs Error Analysis for FER2013")
    parser.add_argument(
        "--config",
        default="config_convnext_base_ms1m_arcface_baseline.yaml",
        help="Path to baseline config YAML file",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to run output directory (defaults to paths.output_dir in config)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to specific model checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save analysis results (defaults to <run_dir>/analysis)",
    )
    return parser.parse_args()


def find_best_checkpoint(run_dir: Path, override_ckpt: str = None) -> Path:
    if override_ckpt:
        p = Path(override_ckpt)
        if p.exists() or Path(str(p) + ".index").exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {override_ckpt}")

    # Check training_history.csv for context if present
    hist_file = run_dir / "training_history.csv"
    if hist_file.exists():
        try:
            df_hist = pd.read_csv(hist_file)
            acc_col = "val_accuracy" if "val_accuracy" in df_hist.columns else "val_fer_accuracy"
            if acc_col in df_hist.columns:
                best_row = df_hist.loc[df_hist[acc_col].idxmax()]
                best_epoch = int(best_row["epoch"])
                best_val_acc = float(best_row[acc_col])
                print(f"[Training History] Best Recorded Epoch: {best_epoch} ({acc_col}: {best_val_acc * 100:.2f}%)")
        except Exception as e:
            print(f"[Training History] Could not parse training_history.csv: {e}")

    checkpoint_root = run_dir / "checkpoints"
    best_dir = checkpoint_root / "best"
    last_dir = checkpoint_root / "last"

    best_ckpt = tf.train.latest_checkpoint(str(best_dir))
    if best_ckpt:
        return Path(best_ckpt)

    last_ckpt = tf.train.latest_checkpoint(str(last_dir))
    if last_ckpt:
        return Path(last_ckpt)

    # Check for direct .ckpt files in run_dir/checkpoints
    direct_ckpts = list(checkpoint_root.glob("*.ckpt.index"))
    if direct_ckpts:
        base_name = str(direct_ckpts[0]).replace(".index", "")
        return Path(base_name)

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_root}")


def plot_confusion_matrices(cm_raw: np.ndarray, cm_norm: np.ndarray, class_names: List[str], save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw counts
    im0 = axes[0].imshow(cm_raw, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=14, pad=12)
    fig.colorbar(im0, ax=axes[0])
    tick_marks = np.arange(len(class_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)

    thresh0 = cm_raw.max() / 2.0
    for i in range(cm_raw.shape[0]):
        for j in range(cm_raw.shape[1]):
            axes[0].text(
                j, i, f"{int(cm_raw[i, j])}",
                ha="center", va="center",
                color="white" if cm_raw[i, j] > thresh0 else "black",
                fontsize=10,
            )

    # Normalized
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Oranges)
    axes[1].set_title("Confusion Matrix (Normalized Recall)", fontsize=14, pad=12)
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)

    thresh1 = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh1 else "black",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    cfg = load_config(str(cfg_path))
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)

    # Determine run directory and output directory
    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg["paths"]["output_dir"])
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parents[1] / run_dir

    analysis_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      FER2013 BASELINE HARD PAIRS ERROR ANALYSIS")
    print("=" * 70)
    print(f"Run Directory:      {run_dir}")
    print(f"Analysis Output:    {analysis_dir}")

    # Resolve Checkpoint
    ckpt_path = find_best_checkpoint(run_dir, args.checkpoint)
    print(f"Restoring Checkpoint: {ckpt_path}")

    # Build dataset (Strict Validation set, NO-TTA)
    print("\n[1/4] Loading Validation Dataset (NO-TTA)...")
    data_dir = _resolve_path(cfg["data"]["data_path"])
    val_records = collect_split_records(
        data_dir,
        split="val",
        mask_dir=_resolve_path(cfg["data"].get("mask_dir")),
        use_clean_filter=False,
        bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
        mask_ablation=cfg["data"].get("mask_ablation", "none"),
        allow_missing_masks=bool(cfg["data"].get("allow_missing_masks", False)),
    )
    val_sample_ids = val_records.sample_ids

    _, val_ds, _ = build_datasets(cfg, replicas=1)

    # Build Model and Restore Checkpoint
    print("[2/4] Building Model & Restoring Weights...")
    model = build_model(cfg)
    dummy_image = tf.zeros([1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]], tf.float32)
    dummy_mask = tf.zeros([1, cfg["model"].get("token_grid_size", 7), cfg["model"].get("token_grid_size", 7), cfg["model"].get("num_regions", 6)], tf.float32)
    model({"image": dummy_image, "mask": dummy_mask}, training=False)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(str(ckpt_path)).expect_partial()
    print("  -> Weights restored successfully!")

    # Run Inference on Validation Set (NO-TTA)
    print("\n[3/4] Running NO-TTA Inference on Validation Set...")
    all_true_labels = []
    all_pred_labels = []
    all_probs = []

    for batch in val_ds:
        inputs, labels = batch
        outputs = model(inputs, training=False)
        logits = outputs["logits"]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        preds = np.argmax(probs, axis=-1)

        all_true_labels.extend(labels.numpy().tolist())
        all_pred_labels.extend(preds.tolist())
        all_probs.append(probs)

    all_true_labels = np.array(all_true_labels, dtype=np.int64)
    all_pred_labels = np.array(all_pred_labels, dtype=np.int64)
    all_probs = np.vstack(all_probs)
    all_confidences = np.max(all_probs, axis=-1)

    num_samples = len(all_true_labels)
    num_correct = int(np.sum(all_true_labels == all_pred_labels))
    num_wrong = num_samples - num_correct
    val_acc = num_correct / num_samples

    print(f"  -> Total Validation Samples: {num_samples}")
    print(f"  -> Correct Predictions:    {num_correct}")
    print(f"  -> Wrong Predictions:      {num_wrong}")
    print(f"  -> Validation Accuracy:    {val_acc * 100:.2f}%")

    # Export val_predictions.csv
    pred_data = {
        "sample_id": val_sample_ids[:num_samples],
        "true_label_id": all_true_labels,
        "true_label_name": [EMOTION_NAMES[i] for i in all_true_labels],
        "pred_label_id": all_pred_labels,
        "pred_label_name": [EMOTION_NAMES[i] for i in all_pred_labels],
        "confidence": np.round(all_confidences, 5),
        "correct": (all_true_labels == all_pred_labels).astype(int),
    }
    for idx, name in enumerate(EMOTION_NAMES):
        pred_data[f"prob_{name}"] = np.round(all_probs[:, idx], 5)

    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(analysis_dir / "val_predictions.csv", index=False)
    print(f"\nSaved: {analysis_dir / 'val_predictions.csv'}")

    # Compute Confusion Matrices (Raw and Normalized)
    num_classes = len(EMOTION_NAMES)
    cm_raw = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_lbl, pred_lbl in zip(all_true_labels, all_pred_labels):
        cm_raw[true_lbl, pred_lbl] += 1

    cm_norm = cm_raw.astype(np.float64) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1e-9)

    df_cm_raw = pd.DataFrame(cm_raw, index=EMOTION_NAMES, columns=EMOTION_NAMES)
    df_cm_norm = pd.DataFrame(cm_norm, index=EMOTION_NAMES, columns=EMOTION_NAMES)

    df_cm_raw.to_csv(analysis_dir / "confusion_matrix_raw.csv")
    df_cm_norm.to_csv(analysis_dir / "confusion_matrix_normalized.csv")
    plot_confusion_matrices(cm_raw, cm_norm, EMOTION_NAMES, analysis_dir / "confusion_matrix.png")
    print(f"Saved: {analysis_dir / 'confusion_matrix_raw.csv'}")
    print(f"Saved: {analysis_dir / 'confusion_matrix_normalized.csv'}")
    print(f"Saved: {analysis_dir / 'confusion_matrix.png'}")

    # Compute Bidirectional Hard Pairs
    print("\n[4/4] Analyzing Hard Class Pairs...")
    hard_pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            name_a, name_b = EMOTION_NAMES[i], EMOTION_NAMES[j]
            count_a_to_b = int(cm_raw[i, j])  # True i predicted as j
            count_b_to_a = int(cm_raw[j, i])  # True j predicted as i
            total_errors = count_a_to_b + count_b_to_a

            total_pair_samples = int(np.sum(all_true_labels == i) + np.sum(all_true_labels == j))
            error_rate = total_errors / max(total_pair_samples, 1)

            hard_pairs.append({
                "class_A_id": i,
                "class_A_name": name_a,
                "class_B_id": j,
                "class_B_name": name_b,
                "pair_name": f"{name_a} <-> {name_b}",
                "count_A_to_B": count_a_to_b,
                "count_B_to_A": count_b_to_a,
                "total_errors": total_errors,
                "total_pair_samples": total_pair_samples,
                "error_rate": np.round(error_rate, 4),
            })

    df_hard = pd.DataFrame(hard_pairs).sort_values(by=["total_errors", "error_rate"], ascending=False).reset_index(drop=True)
    df_hard.to_csv(analysis_dir / "hard_pairs.csv", index=False)
    print(f"Saved: {analysis_dir / 'hard_pairs.csv'}")

    # Export Top-3 Hard Pairs Sample Lists
    top3_pairs = df_hard.head(3)
    for rank, row in top3_pairs.iterrows():
        a_id, a_name = row["class_A_id"], row["class_A_name"]
        b_id, b_name = row["class_B_id"], row["class_B_name"]

        # Filter misclassified samples between a and b (a->b and b->a)
        mask_misclassified = (
            ((all_true_labels == a_id) & (all_pred_labels == b_id)) |
            ((all_true_labels == b_id) & (all_pred_labels == a_id))
        )
        df_pair_samples = df_preds[mask_misclassified][[
            "sample_id", "true_label_name", "pred_label_name", "confidence", f"prob_{a_name}", f"prob_{b_name}"
        ]].sort_values(by="confidence", ascending=False)

        pair_filename = f"hard_pair_{rank+1}_{a_name}_{b_name}.csv"
        df_pair_samples.to_csv(analysis_dir / pair_filename, index=False)
        print(f"Saved Top-{rank+1} Pair Samples: {analysis_dir / pair_filename}")

    # Summary Output
    print("\n" + "=" * 70)
    print("                 FINAL HARD PAIRS ANALYSIS REPORT")
    print("=" * 70)
    print(f"Validation Accuracy (NO-TTA): {val_acc * 100:.2f}%")
    print(f"Total Correct: {num_correct} / {num_samples}")
    print(f"Total Wrong:   {num_wrong} / {num_samples}")

    print("\nTOP-5 HARD CLASS PAIRS (Ranked by Total Bidirectional Errors):")
    print("-" * 70)
    print(f"{'Rank':<5} | {'Pair (Class A <-> Class B)':<22} | {'A -> B':<7} | {'B -> A':<7} | {'Total Error':<11} | {'Pair Error Rate':<15}")
    print("-" * 70)
    top5_pairs = df_hard.head(5)
    for rank, row in top5_pairs.iterrows():
        print(
            f"{rank+1:<5} | {row['pair_name']:<22} | {row['count_A_to_B']:<7} | {row['count_B_to_A']:<7} | "
            f"{row['total_errors']:<11} | {row['error_rate']*100:.2f}% ({row['total_errors']}/{row['total_pair_samples']})"
        )
    print("-" * 70)

    print("\nTOP-3 RECOMMENDED HARD PAIRS FOR DIFFUSION AUGMENTATION:")
    for rank, row in top3_pairs.iterrows():
        print(
            f"  {rank+1}. [{row['pair_name']}] - Total Misclassifications: {row['total_errors']} "
            f"({row['class_A_name']} -> {row['class_B_name']}: {row['count_A_to_B']}, {row['class_B_name']} -> {row['class_A_name']}: {row['count_B_to_A']})"
        )

    print("=" * 70)
    print("Analysis complete!\n")


if __name__ == "__main__":
    main()
