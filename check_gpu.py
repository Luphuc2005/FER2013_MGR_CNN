from __future__ import annotations

import argparse

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gpus", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow:", tf.__version__)
    print("GPUs:", [gpu.name for gpu in gpus])
    if len(gpus) < args.min_gpus:
        raise RuntimeError(f"Need at least {args.min_gpus} GPUs, found {len(gpus)}.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
