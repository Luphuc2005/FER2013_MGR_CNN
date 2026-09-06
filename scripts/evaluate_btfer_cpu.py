#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, SplitRecords, make_dataset
from train import (
    build_model,
    build_optimizer,
    ensure_optimizer_built,
    split_variables,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "btfer" / "Batch_Ready 7"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "papers" / "siglip2-confusion"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "papers"
    / "btfer_crossdomain"
    / "fer2013_siglip2_confusion"
)
DEFAULT_CONFIG = PROJECT_ROOT / "config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU zero-shot cross-domain BTFER evaluation for FER SigLIP2 checkpoints."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ensemble-top-k",
        type=int,
        default=1,
        help="Average probabilities from the newest K ckpt-* prefixes in --checkpoint-dir. Use 1 for single-checkpoint evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--intra-op-threads", type=int, default=16)
    parser.add_argument("--inter-op-threads", type=int, default=4)
    parser.add_argument("--expected-total", type=int, default=2100)
    parser.add_argument("--expected-per-class", type=int, default=300)
    parser.add_argument("--tta-hflip", action="store_true", help="Force horizontal flip TTA.")
    parser.add_argument("--no-tta-hflip", action="store_true", help="Disable horizontal flip TTA.")
    parser.add_argument("--orig-weight", type=float, default=None)
    parser.add_argument("--flip-weight", type=float, default=None)
    parser.add_argument("--save-confusion-png", action="store_true")
    return parser.parse_args()


def configure_cpu_runtime(intra_threads: int, inter_threads: int) -> None:
    tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
    tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy("float32")
    gpu_devices = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"GPU devices: {gpu_devices}", flush=True)
    print(f"CPU threads: intra={intra_threads} inter={inter_threads}", flush=True)
    if gpu_devices:
        raise RuntimeError("CPU-only evaluation requires GPU devices to be hidden.")


def find_config_path(experiment_dir: Path, explicit_config: Optional[Path]) -> Path:
    if explicit_config is not None:
        return explicit_config
    candidates = [
        experiment_dir / "config.yaml",
        experiment_dir / "config.yml",
        experiment_dir / "run_config.yaml",
        experiment_dir / "used_config.yaml",
        experiment_dir / "config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml",
        experiment_dir / "config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion_v2.yaml",
        DEFAULT_CONFIG,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG


def normalize_checkpoint_prefix(path: Path) -> Path:
    raw = str(path)
    if raw.endswith(".index"):
        raw = raw[:-6]
    raw = re.sub(r"\.data-\d+-of-\d+$", "", raw)
    return Path(raw)


def infer_source_domain(experiment_dir: Path, config_path: Path) -> str:
    text = f"{experiment_dir} {config_path}".lower()
    if "expw" in text:
        return "ExpW"
    if "rafdb" in text or "raf-db" in text:
        return "RAF-DB"
    if "fer2013" in text or "fer13" in text or "siglip2-confusion" in text:
        return "FER2013"
    return "unknown"

def resolve_checkpoint(checkpoint_dir: Path, explicit_checkpoint: Optional[Path]) -> Path:
    if explicit_checkpoint is not None:
        return normalize_checkpoint_prefix(explicit_checkpoint)
    checkpoint_state = checkpoint_dir / "checkpoint"
    if not checkpoint_state.exists():
        raise FileNotFoundError(f"Checkpoint state file not found: {checkpoint_state}")
    state = tf.train.get_checkpoint_state(str(checkpoint_dir))
    if state is None or not state.model_checkpoint_path:
        raise FileNotFoundError(f"No model_checkpoint_path found in {checkpoint_state}")
    ckpt_path = Path(state.model_checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = checkpoint_dir / ckpt_path
    return normalize_checkpoint_prefix(ckpt_path)


def resolve_checkpoints(
    checkpoint_dir: Path,
    explicit_checkpoint: Optional[Path],
    ensemble_top_k: int,
) -> List[Path]:
    if explicit_checkpoint is not None:
        return [normalize_checkpoint_prefix(explicit_checkpoint)]
    if ensemble_top_k <= 1:
        return [resolve_checkpoint(checkpoint_dir, explicit_checkpoint)]

    index_files = sorted(
        checkpoint_dir.glob("ckpt-*.index"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    prefixes = [normalize_checkpoint_prefix(path) for path in index_files]
    if not prefixes:
        raise FileNotFoundError(f"No ckpt-*.index files found in checkpoint dir: {checkpoint_dir}")
    return prefixes[:ensemble_top_k]


def btfer_folder_to_model_label(folder_name: str, class_to_idx: Dict[str, int]) -> int:
    key = folder_name.strip().lower()
    aliases = {
        "angry": "angry",
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "happiness": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
    }
    if key not in aliases:
        raise ValueError(f"Unknown BTFER class folder {folder_name!r}")
    return class_to_idx[aliases[key]]


def collect_btfer_records(
    data_root: Path,
    expected_total: int,
    expected_per_class: int,
) -> Tuple[SplitRecords, List[str], Dict[str, int]]:
    if not data_root.exists():
        raise FileNotFoundError(f"BTFER data root not found: {data_root}")

    class_to_idx = {name.lower(): idx for idx, name in enumerate(EMOTION_NAMES)}
    paths: List[str] = []
    labels: List[int] = []
    folder_counts: Dict[str, int] = {}

    for folder in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        label = btfer_folder_to_model_label(folder.name, class_to_idx)
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)
        folder_counts[folder.name] = len(files)
        for file_path in files:
            paths.append(str(file_path))
            labels.append(label)

    total = len(paths)
    if total != expected_total:
        raise ValueError(f"BTFER sample count mismatch: expected {expected_total}, got {total}.")
    bad_counts = {name: count for name, count in folder_counts.items() if count != expected_per_class}
    if bad_counts:
        raise ValueError(
            f"BTFER per-class count mismatch: expected {expected_per_class} each, got {folder_counts}."
        )

    records = SplitRecords(
        images=np.asarray(paths, dtype=object),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.arange(total, dtype=np.int64),
        mask_paths=None,
        masks=None,
    )
    return records, paths, folder_counts


def build_btfer_dataset(records: SplitRecords, cfg: Dict, batch_size: int) -> tf.data.Dataset:
    cfg["runtime"]["batch_size_per_gpu"] = int(batch_size)
    cfg["runtime"]["tf_data_num_parallel_calls"] = cfg["runtime"].get("tf_data_num_parallel_calls", 4)
    cfg["runtime"]["prefetch_buffer"] = 0
    return make_dataset(records, cfg, split="test", training=False, replicas=1)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def predict_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    use_tta_hflip: bool,
    original_weight: float,
    flip_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    conf: List[np.ndarray] = []

    for inputs, labels in dataset:
        outputs_orig = model(inputs, training=False)
        logits = outputs_orig["logits"]
        if use_tta_hflip:
            flipped_inputs = dict(inputs)
            flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs:
                flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
            outputs_flip = model(flipped_inputs, training=False)
            logits = original_weight * logits + flip_weight * outputs_flip["logits"]

        probs = softmax_np(logits.numpy())
        preds = np.argmax(probs, axis=1).astype(np.int64)
        y_true.append(labels.numpy().astype(np.int64))
        y_pred.append(preds)
        conf.append(np.max(probs, axis=1))

    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(conf)


def predict_dataset_probs(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    use_tta_hflip: bool,
    original_weight: float,
    flip_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []

    for inputs, labels in dataset:
        outputs_orig = model(inputs, training=False)
        logits = outputs_orig["logits"]
        if use_tta_hflip:
            flipped_inputs = dict(inputs)
            flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
            if "mask" in inputs:
                flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
            outputs_flip = model(flipped_inputs, training=False)
            logits = original_weight * logits + flip_weight * outputs_flip["logits"]

        probs_all.append(softmax_np(logits.numpy()))
        y_true.append(labels.numpy().astype(np.int64))

    return np.concatenate(y_true), np.concatenate(probs_all)


def write_predictions_csv(
    output_path: Path,
    filepaths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "true_class", "pred_label", "pred_class", "confidence"])
        for path, true_label, pred_label, conf in zip(filepaths, y_true, y_pred, confidence):
            writer.writerow(
                [
                    path,
                    int(true_label),
                    EMOTION_NAMES[int(true_label)],
                    int(pred_label),
                    EMOTION_NAMES[int(pred_label)],
                    float(conf),
                ]
            )


def write_confusion_csv(output_path: Path, cm: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *EMOTION_NAMES])
        for class_name, row in zip(EMOTION_NAMES, cm.tolist()):
            writer.writerow([class_name, *row])


def maybe_write_confusion_png(output_path: Path, cm: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(EMOTION_NAMES)), labels=EMOTION_NAMES, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(EMOTION_NAMES)), labels=EMOTION_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("BTFER zero-shot confusion matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARNING] Could not save confusion_matrix.png: {exc}", flush=True)


def prepare_config(cfg: Dict, args: argparse.Namespace, config_path: Path) -> Dict:
    cfg["paths"]["output_dir"] = str(args.experiment_dir)
    cfg["runtime"]["gpu_ids"] = []
    cfg["runtime"]["min_gpus"] = 0
    cfg["runtime"]["require_two_gpus"] = False
    cfg["runtime"]["allow_cpu_fallback"] = True
    cfg["runtime"]["distributed_eval"] = False
    cfg["runtime"]["use_mixed_precision"] = False
    cfg["runtime"]["intra_op_threads"] = int(args.intra_op_threads)
    cfg["runtime"]["inter_op_threads"] = int(args.inter_op_threads)
    cfg["runtime"]["batch_size_per_gpu"] = int(args.batch_size)
    print(f"Config: {config_path}", flush=True)
    print(f"Experiment dir: {args.experiment_dir}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Model name: {cfg['model'].get('name')}", flush=True)
    print(f"Model arch: {cfg['model'].get('arch')}", flush=True)
    return cfg


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or (args.experiment_dir / "checkpoints" / "best")
    config_path = find_config_path(args.experiment_dir, args.config)
    cfg = prepare_config(load_config(config_path), args, config_path)
    source_domain = infer_source_domain(args.experiment_dir, config_path)

    configure_cpu_runtime(args.intra_op_threads, args.inter_op_threads)

    print(f"Class order verified from datasets.fer2013.EMOTION_NAMES: {list(EMOTION_NAMES)}", flush=True)
    btfer_records, filepaths, folder_counts = collect_btfer_records(
        args.data_root,
        args.expected_total,
        args.expected_per_class,
    )
    label_counts = Counter(btfer_records.labels.tolist())
    print(f"BTFER data root: {args.data_root}", flush=True)
    print(f"BTFER folder counts: {folder_counts}", flush=True)
    print(f"BTFER mapped label counts: {dict(sorted(label_counts.items()))}", flush=True)
    print("BTFER folder label mapping:", flush=True)
    for folder_name in sorted(folder_counts):
        label = btfer_folder_to_model_label(folder_name, {name: i for i, name in enumerate(EMOTION_NAMES)})
        print(f"  {folder_name} -> {label} ({EMOTION_NAMES[label]})", flush=True)

    dataset = build_btfer_dataset(btfer_records, cfg, args.batch_size)
    first_batch = next(iter(dataset.take(1)))
    first_inputs, _ = first_batch

    checkpoint_paths = resolve_checkpoints(checkpoint_dir, args.checkpoint, int(args.ensemble_top_k))
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"Checkpoint state file: {checkpoint_dir / 'checkpoint'}", flush=True)
    print(f"Resolved checkpoints to restore: {len(checkpoint_paths)}", flush=True)
    for idx, ckpt_path in enumerate(checkpoint_paths, start=1):
        print(f"  [{idx}/{len(checkpoint_paths)}] {ckpt_path}", flush=True)

    model = build_model(cfg)
    _ = model(first_inputs, training=False)
    optimizer_head = build_optimizer(cfg, float(cfg["training"]["lr"]))
    optimizer_backbone = build_optimizer(
        cfg,
        float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])),
    )
    backbone_vars, head_vars = split_variables(model)
    ensure_optimizer_built(optimizer_head, head_vars, None)
    ensure_optimizer_built(optimizer_backbone, backbone_vars, None)
    ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt_best_metric = tf.Variable(-1.0, dtype=tf.float32, trainable=False)
    checkpoint = tf.train.Checkpoint(
        epoch=ckpt_epoch,
        best_metric=ckpt_best_metric,
        model=model,
        optimizer_head=optimizer_head,
        optimizer_backbone=optimizer_backbone,
    )

    use_tta_hflip = bool(cfg.get("runtime", {}).get("eval_tta_hflip", False) or cfg.get("tta", {}).get("enabled", False))
    if args.tta_hflip:
        use_tta_hflip = True
    if args.no_tta_hflip:
        use_tta_hflip = False
    orig_weight = float(args.orig_weight if args.orig_weight is not None else cfg.get("tta", {}).get("original_weight", 0.5))
    flip_weight = float(args.flip_weight if args.flip_weight is not None else cfg.get("tta", {}).get("flip_weight", 0.5))
    if use_tta_hflip:
        total_weight = orig_weight + flip_weight
        if total_weight <= 0:
            raise ValueError("TTA weights must sum to a positive value.")
        orig_weight /= total_weight
        flip_weight /= total_weight
    print(f"TTA hflip: {use_tta_hflip} | original_weight={orig_weight:.4f} flip_weight={flip_weight:.4f}", flush=True)

    member_probs: List[np.ndarray] = []
    member_metrics: List[Dict[str, object]] = []
    y_true: Optional[np.ndarray] = None

    for idx, ckpt_path in enumerate(checkpoint_paths, start=1):
        checkpoint.restore(str(ckpt_path)).expect_partial()
        print(f"Restored checkpoint [{idx}/{len(checkpoint_paths)}]: {ckpt_path}", flush=True)
        print(f"Restored epoch variable: {int(ckpt_epoch.numpy())}", flush=True)
        print(f"Restored best_metric variable: {float(ckpt_best_metric.numpy()):.6f}", flush=True)

        member_y_true, probs = predict_dataset_probs(
            model,
            dataset,
            use_tta_hflip=use_tta_hflip,
            original_weight=orig_weight,
            flip_weight=flip_weight,
        )
        if y_true is None:
            y_true = member_y_true
        elif not np.array_equal(y_true, member_y_true):
            raise RuntimeError("Dataset label order changed between checkpoint evaluations.")

        member_pred = np.argmax(probs, axis=1).astype(np.int64)
        member_acc = float(accuracy_score(member_y_true, member_pred))
        member_macro_f1 = float(f1_score(member_y_true, member_pred, average="macro", labels=list(range(len(EMOTION_NAMES))), zero_division=0))
        print(
            f"Checkpoint [{idx}/{len(checkpoint_paths)}] Accuracy (%): {member_acc * 100.0:.4f} | Macro-F1: {member_macro_f1:.6f}",
            flush=True,
        )
        member_probs.append(probs)
        member_metrics.append(
            {
                "checkpoint": str(ckpt_path),
                "epoch_variable": int(ckpt_epoch.numpy()),
                "best_metric_variable": float(ckpt_best_metric.numpy()),
                "accuracy": member_acc,
                "accuracy_percent": member_acc * 100.0,
                "macro_f1": member_macro_f1,
            }
        )

    if y_true is None or not member_probs:
        raise RuntimeError("No checkpoint probabilities were produced.")

    ensemble_probs = np.mean(member_probs, axis=0)
    y_pred = np.argmax(ensemble_probs, axis=1).astype(np.int64)
    confidence = np.max(ensemble_probs, axis=1)

    labels = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=EMOTION_NAMES,
        digits=6,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class_accuracy = {}
    for idx, class_name in enumerate(EMOTION_NAMES):
        mask = y_true == idx
        per_class_accuracy[class_name] = float(np.mean(y_pred[mask] == idx)) if np.any(mask) else 0.0

    metrics = {
        "dataset": "BTFER",
        "source_checkpoint_domain": source_domain,
        "zero_shot_cross_domain": True,
        "config": str(config_path),
        "experiment_dir": str(args.experiment_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "restored_checkpoint": str(checkpoint_paths[0]),
        "restored_checkpoints": [str(path) for path in checkpoint_paths],
        "num_checkpoints_ensembled": int(len(checkpoint_paths)),
        "checkpoint_member_metrics": member_metrics,
        "ensemble_method": "softmax_probability_average",
        "class_order": list(EMOTION_NAMES),
        "btfer_folder_counts": folder_counts,
        "total_samples": int(y_true.size),
        "correct_samples": int(np.sum(y_true == y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_percent": float(accuracy_score(y_true, y_pred) * 100.0),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "input_shape": [None, int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"]), int(cfg["data"]["channels"])],
        "preprocessing": "datasets.fer2013.make_dataset split=test: decode image, bilinear resize, ImageNet mean/std normalize, no augmentation",
        "tta_hflip": bool(use_tta_hflip),
        "original_weight": float(orig_weight),
        "flip_weight": float(flip_weight),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    (args.output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    write_predictions_csv(args.output_dir / "predictions.csv", filepaths, y_true, y_pred, confidence)
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", cm)
    if args.save_confusion_png:
        maybe_write_confusion_png(args.output_dir / "confusion_matrix.png", cm)

    print(f"\nBTFER zero-shot {source_domain} checkpoint evaluation", flush=True)
    print(f"Checkpoints ensembled: {len(checkpoint_paths)}", flush=True)
    print(f"Total samples: {metrics['total_samples']}", flush=True)
    print(f"Correct samples: {metrics['correct_samples']}", flush=True)
    print(f"Accuracy (%): {metrics['accuracy_percent']:.4f}", flush=True)
    print(f"Macro-F1: {metrics['macro_f1']:.6f}", flush=True)
    print(f"Weighted-F1: {metrics['weighted_f1']:.6f}", flush=True)
    print(f"Precision macro: {metrics['precision_macro']:.6f}", flush=True)
    print(f"Recall macro: {metrics['recall_macro']:.6f}", flush=True)
    print("Per-class accuracy:", flush=True)
    for class_name, value in per_class_accuracy.items():
        print(f"  {class_name}: {value:.6f}", flush=True)
    print("Confusion matrix:", flush=True)
    print(cm, flush=True)
    print(report_text, flush=True)
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}", flush=True)
    print(f"Saved predictions: {args.output_dir / 'predictions.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
