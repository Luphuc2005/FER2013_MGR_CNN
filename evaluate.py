from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from config import load_config
from datasets.fer2013 import build_datasets
from metrics.classification import save_metrics
from train import build_model, configure_gpus, configure_tensorflow_runtime, evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser("Evaluate TensorFlow MGR-CNN")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--tta-hflip", action="store_true", help="Average logits from original and horizontal-flip inputs.")
    parser.add_argument("--no-tta-hflip", action="store_true", help="Disable hflip TTA even when enabled in config.")
    parser.add_argument("--orig-weight", type=float, default=None, help="Weight for original image prediction in TTA.")
    parser.add_argument("--flip-weight", type=float, default=None, help="Weight for flipped image prediction in TTA.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_gpus(cfg)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    visible_gpus = tf.config.get_visible_devices("GPU")
    if visible_gpus:
        strategy_devices = [f"/GPU:{i}" for i in range(len(visible_gpus))]
        strategy = tf.distribute.MirroredStrategy(devices=strategy_devices)
    else:
        strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
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
        ckpt.restore(checkpoint_path).expect_partial()
        print(f"Restored: {checkpoint_path}")
    eval_strategy = strategy if bool(cfg["runtime"].get("distributed_eval", False)) else None
    use_tta_hflip = bool(cfg["tta"].get("enabled", False))
    if args.tta_hflip:
        use_tta_hflip = True
    if args.no_tta_hflip:
        use_tta_hflip = False
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
    out = Path(cfg["paths"]["output_dir"]) / f"{args.split}_metrics{suffix}.json"
    save_metrics(metrics, out)
    save_metrics(metrics, Path(cfg["paths"]["output_dir"]) / f"{args.split}_metrics.json")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
