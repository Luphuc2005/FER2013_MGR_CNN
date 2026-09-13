#!/usr/bin/env python3
"""
Generate publication-quality Confusion Matrix reports and Heatmap figures
for Best Single Checkpoint (ckpt-42) and Ensemble on FER2013.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def parse_args():
    parser = argparse.ArgumentParser("Visualize and Report Confusion Matrix")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="outputs/papers/siglip2-confusion_v4",
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt-42",
        help="Target checkpoint to report (default: ckpt-42)",
    )
    return parser.parse_args()


def print_ascii_cm(cm: np.ndarray, names: list[str], title: str = "CONFUSION MATRIX"):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)
    header = "          " + " ".join(f"{n[:5]:>7}" for n in names) + " | Total  Recall(%)"
    print(header)
    print("-" * 75)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{int(val):>7}" for val in row)
        tot = np.sum(row)
        rec = (row[i] / tot * 100.0) if tot > 0 else 0.0
        print(f"{names[i]:>9} {row_str} | {int(tot):>5}  {rec:>6.2f}%")
    print("-" * 75)


def print_normalized_cm(cm: np.ndarray, names: list[str], title: str = "NORMALIZED CONFUSION MATRIX (%)"):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)
    header = "          " + " ".join(f"{n[:5]:>7}" for n in names)
    print(header)
    print("-" * 75)
    norm_cm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    for i, row in enumerate(norm_cm):
        row_str = " ".join(f"{val:>6.1f}%" for val in row)
        print(f"{names[i]:>9} {row_str}")
    print("-" * 75)


def print_classification_table(report: dict, names: list[str]):
    print("\n" + "=" * 68)
    print("  CLASSIFICATION REPORT")
    print("=" * 68)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print("-" * 68)
    for n in names:
        key = n.lower()
        if key in report:
            p = report[key].get("precision", 0.0) * 100.0
            r = report[key].get("recall", 0.0) * 100.0
            f = report[key].get("f1-score", 0.0) * 100.0
            s = int(report[key].get("support", 0))
            print(f"{n:<12} {p:>9.2f}% {r:>9.2f}% {f:>9.2f}% {s:>10}")
    print("-" * 68)


def plot_and_save_cm(cm: np.ndarray, names: list[str], out_path: Path, title: str, normalize: bool = False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"[WARNING] matplotlib or seaborn not installed, skipping image export for {out_path.name}")
        return

    plt.figure(figsize=(8.5, 7.0), dpi=300)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    if normalize:
        plot_matrix = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
        cbar_label = "Percentage (%)"
        annot_matrix = np.array([[f"{val:.1f}%" for val in row] for row in plot_matrix])
        cmap = sns.light_palette("#2ca02c", as_cmap=True)
    else:
        plot_matrix = cm
        cbar_label = "Sample Count"
        annot_matrix = plot_matrix
        cmap = sns.light_palette("#1f77b4", as_cmap=True)

    sns.heatmap(
        plot_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        xticklabels=names,
        yticklabels=names,
        linewidths=0.5,
        linecolor="#e0e0e0",
    )

    plt.title(title, fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("True Emotion Label", fontsize=12, labelpad=10, fontweight="semibold")
    plt.xlabel("Predicted Emotion Label", fontsize=12, labelpad=10, fontweight="semibold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] High-resolution heatmap saved to: {out_path}")


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    target_ckpt = args.checkpoint

    print("=" * 75)
    print(" FER2013 CONFUSION MATRIX & EVALUATION REPORT GENERATOR")
    print(f" Exp Directory : {exp_dir}")
    print(f" Target Model  : {target_ckpt}")
    print("=" * 75)

    # 1. Look for target checkpoint JSON
    candidate_paths = [
        exp_dir / f"test_metrics_{target_ckpt}_tta_hflip.json",
        exp_dir / "eval_individual_ckpts" / f"{target_ckpt}_eval.json",
        exp_dir / f"test_metrics_{target_ckpt}.json",
    ]

    target_json = None
    for p in candidate_paths:
        if p.exists():
            target_json = p
            break

    if target_json is None:
        print(f"[ERROR] Could not find JSON report for {target_ckpt} in {exp_dir}")
        print("Searched paths:")
        for p in candidate_paths:
            print(f"  - {p}")
        return 1

    with open(target_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n[LOADED] Successfully read metrics from: {target_json.name}")

    acc = data.get("accuracy", data.get("test_val_tuned_accuracy", data.get("test_no_tta_accuracy", 0.0)))
    macro_f1 = data.get("macro_f1", data.get("test_val_tuned_macro_f1", data.get("test_no_tta_macro_f1", 0.0)))
    weighted_f1 = data.get("weighted_f1", data.get("test_val_tuned_weighted_f1", 0.0))
    cm_list = data.get("confusion_matrix", None)
    report_dict = data.get("classification_report", None)

    print("\n" + "=" * 75)
    print(f"         OFFICIAL TEST SET RESULTS: {target_ckpt.upper()}")
    print("=" * 75)
    print(f"  -> Test Accuracy : {acc * 100:.2f}%")
    print(f"  -> Macro F1      : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"  -> Weighted F1   : {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")

    if cm_list is not None:
        cm = np.array(cm_list)
        # ASCII Table
        print_ascii_cm(cm, EMOTION_NAMES, title=f"CONFUSION MATRIX: {target_ckpt.upper()} (COUNTS)")
        print_normalized_cm(cm, EMOTION_NAMES, title=f"NORMALIZED CONFUSION MATRIX: {target_ckpt.upper()} (%)")

        # PNG figures
        out_counts = exp_dir / f"confusion_matrix_{target_ckpt}_counts.png"
        plot_and_save_cm(cm, EMOTION_NAMES, out_counts, f"FER2013 Confusion Matrix: {target_ckpt.upper()} (Acc: {acc*100:.2f}%)", normalize=False)

        out_norm = exp_dir / f"confusion_matrix_{target_ckpt}_normalized.png"
        plot_and_save_cm(cm, EMOTION_NAMES, out_norm, f"FER2013 Normalized Confusion Matrix: {target_ckpt.upper()} (Acc: {acc*100:.2f}%)", normalize=True)

    if report_dict is not None:
        print_classification_table(report_dict, EMOTION_NAMES)

    # 2. Look for Ensemble report if available
    ensemble_json = exp_dir / "ferplus_5ckpts_and_ensemble_report.json"
    if not ensemble_json.exists():
        for p in exp_dir.glob("*ensemble_report.json"):
            ensemble_json = p
            break

    if ensemble_json.exists():
        with open(ensemble_json, "r", encoding="utf-8") as f:
            ens_data = json.load(f)
        ens_metrics = ens_data.get("ensemble", {}).get("test", None)
        if ens_metrics:
            print("\n" + "=" * 75)
            print("         OFFICIAL TEST SET RESULTS: TOP-5 ENSEMBLE")
            print("=" * 75)
            ens_acc = ens_metrics.get("accuracy", 0.0)
            ens_f1 = ens_metrics.get("macro_f1", 0.0)
            ens_wf1 = ens_metrics.get("weighted_f1", 0.0)
            print(f"  -> Ensemble Test Acc : {ens_acc * 100:.2f}%")
            print(f"  -> Ensemble Macro F1 : {ens_f1:.4f} ({ens_f1 * 100:.2f}%)")
            print(f"  -> Ensemble Weighted : {ens_wf1:.4f} ({ens_wf1 * 100:.2f}%)")

            ens_cm = np.array(ens_metrics.get("confusion_matrix", []))
            if len(ens_cm) > 0:
                print_ascii_cm(ens_cm, EMOTION_NAMES, title="CONFUSION MATRIX: TOP-5 ENSEMBLE (COUNTS)")
                print_normalized_cm(ens_cm, EMOTION_NAMES, title="NORMALIZED CONFUSION MATRIX: TOP-5 ENSEMBLE (%)")
                out_ens = exp_dir / "confusion_matrix_top5_ensemble_normalized.png"
                plot_and_save_cm(ens_cm, EMOTION_NAMES, out_ens, f"FER2013 Normalized Confusion Matrix: Top-5 Ensemble (Acc: {ens_acc*100:.2f}%)", normalize=True)

    print("\n" + "=" * 75)
    print(" [COMPLETED] Confusion Matrix reporting and visualization finished!")
    print("=" * 75 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
