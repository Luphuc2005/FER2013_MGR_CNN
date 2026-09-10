from __future__ import annotations

import argparse
import os
from pathlib import Path

# If CPU mode requested via env var before TF initializes
if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from config import load_config
from datasets.fer2013 import build_datasets
from metrics.classification import save_metrics
from train import build_model, configure_gpus, configure_tensorflow_runtime, evaluate_dataset, get_class_names


def parse_args():
    parser = argparse.ArgumentParser("Evaluate TensorFlow MGR-CNN")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (e.g. .../ckpt-24).")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--cpu", action="store_true", help="Force evaluation on CPU (CUDA_VISIBLE_DEVICES=-1).")
    parser.add_argument("--tta-hflip", action="store_true", help="Average logits from original and horizontal-flip inputs.")
    parser.add_argument("--no-tta-hflip", action="store_true", help="Disable hflip TTA even when enabled in config.")
    parser.add_argument("--orig-weight", type=float, default=None, help="Weight for original image prediction in TTA.")
    parser.add_argument("--flip-weight", type=float, default=None, help="Weight for flipped image prediction in TTA.")
    parser.add_argument("--output", default=None, help="Custom path to save output evaluation metrics JSON.")
    return parser.parse_args()


def print_evaluation_summary(metrics: dict, class_names: list[str], output_paths: list[Path]):
    acc = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    weighted_f1 = float(metrics.get("weighted_f1", 0.0))
    no_tta_acc = float(metrics.get("no_tta_accuracy", acc))
    no_tta_macro_f1 = float(metrics.get("no_tta_macro_f1", macro_f1))

    print("\n" + "=" * 65)
    print("                    EVALUATION RESULTS")
    print("=" * 65)
    print(f"  TTA (H-Flip) Accuracy : {acc * 100:6.2f}%  (score: {acc:.4f})")
    print(f"  TTA (H-Flip) Macro-F1 : {macro_f1 * 100:6.2f}%  (score: {macro_f1:.4f})")
    print(f"  TTA (H-Flip) W-F1     : {weighted_f1 * 100:6.2f}%  (score: {weighted_f1:.4f})")
    print(f"  No-TTA Accuracy       : {no_tta_acc * 100:6.2f}%  (score: {no_tta_acc:.4f})")
    print(f"  No-TTA Macro-F1       : {no_tta_macro_f1 * 100:6.2f}%  (score: {no_tta_macro_f1:.4f})")
    print("-" * 65)

    cm = metrics.get("confusion_matrix", [])
    if cm and class_names:
        print("Confusion Matrix (Rows: Ground Truth, Columns: Prediction):")
        header = f"{'':>10} " + " ".join(f"{name[:4]:>6}" for name in class_names)
        print(header)
        for idx, row in enumerate(cm):
            row_str = " ".join(f"{val:>6}" for val in row)
            print(f"{class_names[idx]:>10} {row_str}")
        print("-" * 65)

    report = metrics.get("classification_report", {})
    if report and isinstance(report, dict):
        print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
        print("-" * 65)
        for name in class_names:
            if name in report:
                p = report[name].get("precision", 0.0)
                r = report[name].get("recall", 0.0)
                f = report[name].get("f1-score", 0.0)
                s = int(report[name].get("support", 0))
                print(f"{name:<12} {p * 100:>9.2f}% {r * 100:>9.2f}% {f * 100:>9.2f}% {s:>10}")
        print("-" * 65)

    print("Output Metrics Files:")
    for p in output_paths:
        print(f"  -> {p.resolve()}")
    print("=" * 65 + "\n", flush=True)


def main() -> int:
    args = parse_args()
    if args.cpu or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    cfg = load_config(args.config)
    is_cpu = bool(
        args.cpu
        or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        or len(tf.config.list_physical_devices("GPU")) == 0
    )
    if is_cpu:
        cfg["runtime"]["allow_cpu_fallback"] = True
        cfg["runtime"]["min_gpus"] = 0
        cfg["runtime"]["use_mixed_precision"] = False
        tf.keras.mixed_precision.set_global_policy("float32")

    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))

    if not is_cpu:
        configure_gpus(cfg)
        visible_gpu_count = len(tf.config.list_logical_devices("GPU"))
        strategy = tf.distribute.MirroredStrategy(devices=[f"/GPU:{i}" for i in range(max(visible_gpu_count, 1))])
    else:
        print("[INFO] Running evaluation in pure CPU mode (CUDA_VISIBLE_DEVICES=-1).", flush=True)
        strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])

    class_names = get_class_names(cfg)
    data_cfg = cfg.get("data", {})
    expected_samples = data_cfg.get("expected_samples", {}).get(args.split)

    _, val_ds, test_ds = build_datasets(cfg, replicas=strategy.num_replicas_in_sync)
    dataset = val_ds if args.split == "val" else test_ds

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
            raise FileNotFoundError("No checkpoint supplied and no best/last checkpoint found.")

        checkpoint_path = str(checkpoint_path)
        if checkpoint_path.endswith(".index"):
            checkpoint_path = checkpoint_path[:-6]

        ckpt.restore(checkpoint_path).expect_partial()
        print(f"[INFO] Restored weights: {checkpoint_path}", flush=True)

    use_tta_hflip = bool(cfg.get("tta", {}).get("enabled", False))
    if args.tta_hflip:
        use_tta_hflip = True
    if args.no_tta_hflip:
        use_tta_hflip = False

    dataset_name = data_cfg.get("dataset_type")
    if not dataset_name:
        data_path_str = str(data_cfg.get("data_path", "")).lower()
        if "raf" in data_path_str:
            dataset_name = "RAF-DB"
        elif "affectnet" in data_path_str:
            dataset_name = "AffectNet"
        elif "expw" in data_path_str:
            dataset_name = "ExpW"
        else:
            dataset_name = "FER2013"

    print("\n" + "=" * 65)
    print("           EVALUATION PRE-CHECK VERIFICATION")
    print("=" * 65)
    print(f"  Checkpoint Path    : {checkpoint_path}")
    print(f"  Dataset Name       : {dataset_name}")
    print(f"  Split              : {args.split.upper()}")
    print(f"  Total Test Samples : {expected_samples if expected_samples is not None else 'N/A'}")
    print(f"  Number of Classes  : {len(class_names)} ({', '.join(class_names)})")
    print(f"  Compute Device     : {'CPU (/CPU:0)' if is_cpu else 'GPU'}")
    print(f"  TTA Horizontal Flip: {use_tta_hflip}")
    print(f"  Weights Updatable  : FALSE (Read-only evaluation, zero gradient updates)")
    print("=" * 65 + "\n", flush=True)

    eval_strategy = strategy if bool(cfg["runtime"].get("distributed_eval", False)) else None
    metrics = evaluate_dataset(
        model,
        dataset,
        cfg,
        strategy=eval_strategy,
        use_tta_hflip=use_tta_hflip,
        original_weight=args.orig_weight,
        flip_weight=args.flip_weight,
    )

    suffix = "_tta_hflip" if use_tta_hflip else "_no_tta"
    ckpt_name = Path(checkpoint_path).name

    output_paths = []
    if args.output:
        out_custom = Path(args.output)
        save_metrics(metrics, out_custom)
        output_paths.append(out_custom)
    else:
        out_ckpt = Path(cfg["paths"]["output_dir"]) / f"{args.split}_metrics_{ckpt_name}{suffix}.json"
        save_metrics(metrics, out_ckpt)
        output_paths.append(out_ckpt)

        out_std = Path(cfg["paths"]["output_dir"]) / f"{args.split}_metrics{suffix}.json"
        save_metrics(metrics, out_std)
        output_paths.append(out_std)

    print_evaluation_summary(metrics, class_names, output_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
