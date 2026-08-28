from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import train before tensorflow so the project runtime env guards are applied.
from train import (  # noqa: E402
    build_model,
    build_optimizer,
    configure_gpus,
    configure_tensorflow_runtime,
    ensure_optimizer_built,
    split_variables,
)

import tensorflow as tf  # noqa: E402

from config import load_config  # noqa: E402
from datasets.fer2013 import collect_split_records, make_dataset  # noqa: E402


DEFAULT_CNN_CONFIG = "config_convnext_base_ms1m_arcface_baseline.yaml"
DEFAULT_MGR_CONFIG = "config_convnext_base_ms1m_arcface_mgr.yaml"
DEFAULT_CNN_CHECKPOINT = (
    "/home/ptbao/projects/FER2013_MGR_CNN/"
    "outputs/tf_runs/convnext_base_ms1m_arcface_baseline/checkpoints/best/ckpt-43"
)
DEFAULT_MGR_CHECKPOINT = (
    "/home/ptbao/projects/FER2013_MGR_CNN/"
    "outputs/tf_runs/convnext_base_ms1m_arcface_mgr/checkpoints/best/ckpt-33"
)
DEFAULT_OUTPUT = "outputs/oracle_cnn_vs_mgr.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute oracle accuracy between the ConvNeXt-Base CNN baseline "
            "and standard MGR checkpoint on the same FER2013 test samples."
        )
    )
    parser.add_argument("--cnn-config", default=DEFAULT_CNN_CONFIG)
    parser.add_argument("--mgr-config", default=DEFAULT_MGR_CONFIG)
    parser.add_argument("--cnn-checkpoint", default=DEFAULT_CNN_CHECKPOINT)
    parser.add_argument("--mgr-checkpoint", default=DEFAULT_MGR_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-samples", type=int, default=3589)
    parser.add_argument(
        "--expected-cnn-accuracy",
        type=float,
        default=None,
        help="Optional reference accuracy as fraction (0.742) or percent (74.2).",
    )
    parser.add_argument(
        "--expected-mgr-accuracy",
        type=float,
        default=None,
        help="Optional reference accuracy as fraction (0.752) or percent (75.2).",
    )
    parser.add_argument(
        "--sanity-tolerance",
        type=float,
        default=0.005,
        help="Absolute accuracy tolerance as a fraction; default is 0.005 = 0.5 percentage points.",
    )
    parser.add_argument(
        "--allow-missing-sanity-reference",
        action="store_true",
        help="Compute oracle even when no existing test metric or expected accuracy is available.",
    )
    parser.add_argument(
        "--tta-hflip",
        action="store_true",
        help="Force horizontal-flip TTA for both models.",
    )
    parser.add_argument(
        "--no-tta-hflip",
        action="store_true",
        help="Disable horizontal-flip TTA for both models.",
    )
    return parser.parse_args()


def _as_project_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clean_eval_cfg(cfg: Dict) -> Dict:
    cfg["data"]["max_train_samples"] = None
    cfg["data"]["max_val_samples"] = None
    cfg["data"]["max_test_samples"] = None
    cfg["runtime"]["tf_data_deterministic"] = True
    return cfg


def _resolve_checkpoint(path_like: str) -> str:
    path = Path(path_like)
    if path.is_dir():
        latest = tf.train.latest_checkpoint(str(path))
        if latest:
            return latest
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    index_file = Path(str(path) + ".index")
    if not index_file.exists():
        raise FileNotFoundError(f"Checkpoint prefix not found: {path} (missing {index_file})")
    return str(path)


def _normalize_accuracy(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _load_reference_accuracy(cfg: Dict, *, use_tta_hflip: bool) -> Tuple[Optional[float], Optional[Path]]:
    output_dir = Path(cfg["paths"]["output_dir"])
    candidates = (
        ["test_metrics_tta_hflip.json", "test_metrics.json"]
        if use_tta_hflip
        else ["test_metrics_no_tta.json", "test_metrics.json"]
    )
    for name in candidates:
        path = output_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "accuracy" not in payload:
            continue
        file_tta = payload.get("tta_hflip")
        if file_tta is not None and bool(file_tta) != bool(use_tta_hflip):
            continue
        return float(payload["accuracy"]), path
    return None, None


def _check_sanity(
    name: str,
    observed: float,
    expected: Optional[float],
    reference_path: Optional[Path],
    tolerance: float,
    *,
    allow_missing_reference: bool,
) -> None:
    if expected is None:
        if allow_missing_reference:
            print(f"[WARNING] {name}: no sanity reference found; observed accuracy={observed:.6f}", flush=True)
            return
        raise RuntimeError(
            f"{name}: missing sanity reference. Provide --expected-{name.lower()}-accuracy "
            "or keep the run's test_metrics*.json under its output_dir. "
            "Use --allow-missing-sanity-reference only for a diagnostic run."
        )
    delta = abs(observed - expected)
    print(
        f"[SANITY] {name}: observed={observed:.6f}, reference={expected:.6f}, "
        f"delta={delta:.6f}, source={reference_path or 'CLI'}",
        flush=True,
    )
    if delta > tolerance:
        raise RuntimeError(
            f"{name}: accuracy drift {delta:.6f} exceeds tolerance {tolerance:.6f}. "
            "Stop: checkpoint/data/restore/TTA pipeline likely does not match the original test run."
        )


def _build_test_dataset(mgr_cfg: Dict, replicas: int) -> Tuple[tf.data.Dataset, np.ndarray]:
    records = collect_split_records(
        mgr_cfg["data"]["data_path"],
        "test",
        mask_dir=mgr_cfg["data"].get("mask_dir"),
        use_clean_filter=bool(mgr_cfg["data"].get("use_clean_filter", False)),
        bad_row_indices_path=mgr_cfg["data"].get("bad_row_indices_path"),
        mask_ablation=mgr_cfg["data"].get("mask_ablation", "none"),
        mask_region_permutation=mgr_cfg["data"].get("mask_region_permutation"),
        predecode_pixels=bool(mgr_cfg["data"].get("predecode_pixels", False)),
        preload_masks=bool(mgr_cfg["data"].get("preload_masks", False)),
        allow_missing_masks=bool(mgr_cfg["data"].get("allow_missing_masks", False)),
    )
    dataset = make_dataset(records, mgr_cfg, split="test", training=False, replicas=replicas)
    return dataset, records.labels.astype(np.int64)


def _assert_same_test_labels(cnn_cfg: Dict, mgr_labels: np.ndarray, expected_samples: int) -> None:
    cnn_records = collect_split_records(
        cnn_cfg["data"]["data_path"],
        "test",
        mask_dir=None,
        use_clean_filter=bool(cnn_cfg["data"].get("use_clean_filter", False)),
        bad_row_indices_path=cnn_cfg["data"].get("bad_row_indices_path"),
        mask_ablation=cnn_cfg["data"].get("mask_ablation", "none"),
        mask_region_permutation=cnn_cfg["data"].get("mask_region_permutation"),
        predecode_pixels=False,
        preload_masks=False,
        allow_missing_masks=bool(cnn_cfg["data"].get("allow_missing_masks", False)),
    )
    if len(mgr_labels) != int(expected_samples):
        raise RuntimeError(f"Expected {expected_samples} test samples, got {len(mgr_labels)}")
    if len(cnn_records.labels) != len(mgr_labels):
        raise RuntimeError(
            f"CNN/MGR test sample count mismatch: {len(cnn_records.labels)} vs {len(mgr_labels)}"
        )
    if not np.array_equal(cnn_records.labels.astype(np.int64), mgr_labels):
        raise RuntimeError("CNN and MGR test labels are not identical in order.")
    print(f"[INFO] Same FER2013 test labels/order verified: {len(mgr_labels)} samples", flush=True)


def _forward_logits(model: tf.keras.Model, inputs: Dict[str, tf.Tensor], *, use_tta_hflip: bool) -> tf.Tensor:
    outputs = model(inputs, training=False)
    logits = tf.cast(outputs["logits"], tf.float32)
    if not use_tta_hflip:
        return logits
    flipped_inputs = dict(inputs)
    flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
    if "mask" in inputs:
        flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
    flipped_outputs = model(flipped_inputs, training=False)
    return (logits + tf.cast(flipped_outputs["logits"], tf.float32)) * 0.5


def _restore_model(
    cfg: Dict,
    checkpoint_path: str,
    dataset: tf.data.Dataset,
    strategy: tf.distribute.Strategy,
    label: str,
) -> tf.keras.Model:
    first_inputs, _ = next(iter(dataset.take(1)))
    with strategy.scope():
        model = build_model(cfg)
        _ = model(first_inputs, training=False)
        optimizer_head = build_optimizer(cfg, float(cfg["training"]["lr"]))
        optimizer_backbone = build_optimizer(
            cfg,
            float(cfg["training"].get("visual_extractor_lr", cfg["training"]["lr"])),
        )
        backbone_vars, head_vars = split_variables(model)
        ensure_optimizer_built(optimizer_head, head_vars, strategy)
        ensure_optimizer_built(optimizer_backbone, backbone_vars, strategy)
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
            best_metric=tf.Variable(-1.0, dtype=tf.float32, trainable=False),
            model=model,
            optimizer_head=optimizer_head,
            optimizer_backbone=optimizer_backbone,
        )
        resolved = _resolve_checkpoint(checkpoint_path)
        status = checkpoint.restore(resolved)
        status.assert_existing_objects_matched()
    print(f"[RESTORE_MODEL_OK] {label}: {resolved}", flush=True)
    return model


def _predict(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    use_tta_hflip: bool,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for inputs, labels in dataset:
        logits = _forward_logits(model, inputs, use_tta_hflip=use_tta_hflip)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        y_true.extend(labels.numpy().astype(np.int64).tolist())
        y_pred.extend(preds.numpy().astype(np.int64).tolist())
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    print(f"[INFO] {label}: predicted {len(y_pred_arr)} samples", flush=True)
    return y_true_arr, y_pred_arr


def _write_predictions(path: Path, y_true: np.ndarray, pred_cnn: np.ndarray, pred_mgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "pred_cnn", "pred_mgr"])
        for idx, (label, cnn, mgr) in enumerate(zip(y_true, pred_cnn, pred_mgr)):
            writer.writerow([idx, int(label), int(cnn), int(mgr)])


def main() -> int:
    args = parse_args()
    if args.tta_hflip and args.no_tta_hflip:
        raise ValueError("Choose only one of --tta-hflip or --no-tta-hflip.")

    cnn_cfg = _clean_eval_cfg(load_config(args.cnn_config))
    mgr_cfg = _clean_eval_cfg(load_config(args.mgr_config))
    configure_tensorflow_runtime(mgr_cfg)
    tf.keras.utils.set_random_seed(int(mgr_cfg["seed"]["random_seed"]))
    configure_gpus(mgr_cfg)

    visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
    strategy_devices = [f"/GPU:{i}" for i in range(visible_gpu_count)] if visible_gpu_count else ["/CPU:0"]
    strategy = tf.distribute.MirroredStrategy(devices=strategy_devices)
    print(f"[INFO] TensorFlow {tf.__version__}; replicas={strategy.num_replicas_in_sync}", flush=True)

    use_tta_hflip = bool(mgr_cfg["runtime"].get("eval_tta_hflip", False))
    if bool(cnn_cfg["runtime"].get("eval_tta_hflip", False)) != use_tta_hflip:
        raise RuntimeError("CNN and MGR configs disagree on runtime.eval_tta_hflip; pass --tta-hflip/--no-tta-hflip.")
    if args.tta_hflip:
        use_tta_hflip = True
    if args.no_tta_hflip:
        use_tta_hflip = False
    print(f"[INFO] Evaluation TTA hflip: {use_tta_hflip}", flush=True)

    test_ds, mgr_labels = _build_test_dataset(mgr_cfg, replicas=strategy.num_replicas_in_sync)
    _assert_same_test_labels(cnn_cfg, mgr_labels, int(args.expected_samples))

    cnn_model = _restore_model(cnn_cfg, args.cnn_checkpoint, test_ds, strategy, "CNN")
    y_true_cnn, pred_cnn = _predict(cnn_model, test_ds, use_tta_hflip=use_tta_hflip, label="CNN")
    del cnn_model
    gc.collect()

    mgr_model = _restore_model(mgr_cfg, args.mgr_checkpoint, test_ds, strategy, "MGR")
    y_true_mgr, pred_mgr = _predict(mgr_model, test_ds, use_tta_hflip=use_tta_hflip, label="MGR")
    del mgr_model
    gc.collect()

    if not np.array_equal(y_true_cnn, y_true_mgr):
        raise RuntimeError("CNN and MGR inference labels differ; dataset order is not identical.")
    y_true = y_true_cnn
    total = len(y_true)
    if total != int(args.expected_samples):
        raise RuntimeError(f"Expected {args.expected_samples} predictions, got {total}")

    cnn_correct = pred_cnn == y_true
    mgr_correct = pred_mgr == y_true
    both_correct = int(np.sum(cnn_correct & mgr_correct))
    cnn_right_mgr_wrong = int(np.sum(cnn_correct & ~mgr_correct))
    cnn_wrong_mgr_right = int(np.sum(~cnn_correct & mgr_correct))
    both_wrong = int(np.sum(~cnn_correct & ~mgr_correct))
    cnn_acc = float(np.mean(cnn_correct))
    mgr_acc = float(np.mean(mgr_correct))
    oracle_acc = float(np.mean(cnn_correct | mgr_correct))
    error_union = int(np.sum(~cnn_correct | ~mgr_correct))
    error_overlap = float(both_wrong / error_union) if error_union else 0.0

    ref_cnn, ref_cnn_path = _load_reference_accuracy(cnn_cfg, use_tta_hflip=use_tta_hflip)
    ref_mgr, ref_mgr_path = _load_reference_accuracy(mgr_cfg, use_tta_hflip=use_tta_hflip)
    expected_cnn = _normalize_accuracy(args.expected_cnn_accuracy)
    expected_mgr = _normalize_accuracy(args.expected_mgr_accuracy)
    _check_sanity(
        "CNN",
        cnn_acc,
        expected_cnn if expected_cnn is not None else ref_cnn,
        None if expected_cnn is not None else ref_cnn_path,
        float(args.sanity_tolerance),
        allow_missing_reference=bool(args.allow_missing_sanity_reference),
    )
    _check_sanity(
        "MGR",
        mgr_acc,
        expected_mgr if expected_mgr is not None else ref_mgr,
        None if expected_mgr is not None else ref_mgr_path,
        float(args.sanity_tolerance),
        allow_missing_reference=bool(args.allow_missing_sanity_reference),
    )

    output_path = _as_project_path(args.output)
    _write_predictions(output_path, y_true, pred_cnn, pred_mgr)

    print("\nOracle CNN Baseline vs MGR Standard", flush=True)
    print(f"Samples: {total}", flush=True)
    print(f"1. CNN Accuracy: {cnn_acc:.6f} ({cnn_acc * 100:.2f}%)", flush=True)
    print(f"2. MGR Accuracy: {mgr_acc:.6f} ({mgr_acc * 100:.2f}%)", flush=True)
    print(f"3. Ca hai dung: {both_correct}", flush=True)
    print(f"4. CNN dung / MGR sai: {cnn_right_mgr_wrong}", flush=True)
    print(f"5. CNN sai / MGR dung: {cnn_wrong_mgr_right}", flush=True)
    print(f"6. Ca hai sai: {both_wrong}", flush=True)
    print(f"7. So anh MGR cuu duoc khi CNN sai: {cnn_wrong_mgr_right}", flush=True)
    print(f"8. Oracle Accuracy: {oracle_acc:.6f} ({oracle_acc * 100:.2f}%)", flush=True)
    print(f"9. Error overlap (Jaccard of error sets): {error_overlap:.6f} ({error_overlap * 100:.2f}%)", flush=True)
    print(f"Saved per-sample predictions: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
