#!/usr/bin/env python3
"""V6 batch16 ablation: only batch size differs from the batch32 experiment."""
from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from check_rafdb_v6_multistage import (
    NAME, check_contract as check_batch32_contract,
    smoke_components, smoke_evaluation_metrics, smoke_model,
)

CONFIG = ROOT / f"config_{NAME}_batch16.yaml"


def check_contract(path):
    # Also enforces the existing no-MGR-overrides and v5 recipe contract.
    base = check_batch32_contract(ROOT / f"config_{NAME}.yaml")
    cfg = load_config(path)
    expected = copy.deepcopy(base)
    expected["runtime"]["batch_size_per_gpu"] = 16
    expected["paths"]["output_dir"] = str(ROOT / "outputs/papers" / (NAME + "_batch16"))
    expected["source"] = cfg["source"]  # Descriptive metadata only.
    assert cfg == expected, "Batch16 must differ only in batch size and output/source metadata."
    assert cfg["runtime"]["tf_data_num_parallel_calls"] == 8
    assert cfg["runtime"]["tf_data_private_threadpool_size"] == 8
    assert cfg["runtime"]["prefetch_buffer"] == 4
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    print("V6_BATCH16_ONLY_DIFF_OK: batch32 -> batch16; workers8 pool8 prefetch4; "
          "model, SAM, LR/freeze schedules, losses, augmentation, dropout, semantics, "
          "TTA, seed and 60 epochs unchanged.", flush=True)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--metrics-smoke", action="store_true")
    args = parser.parse_args()
    cfg = check_contract(args.config)
    if args.smoke or args.model_smoke or args.metrics_smoke:
        import tensorflow as tf
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
        if args.smoke:
            smoke_components(tf)
        if args.smoke or args.metrics_smoke:
            smoke_evaluation_metrics(tf, cfg)
        if args.model_smoke:
            smoke_model(tf, cfg)


if __name__ == "__main__":
    main()
