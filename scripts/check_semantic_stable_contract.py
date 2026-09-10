#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from utils.semantic_schedule import resolve_lambda_sem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the RAF-DB semantic-stable v2 config contract.")
    parser.add_argument(
        "config",
        nargs="?",
        default="config_rafdb_siglip2_semantic_stable_v2.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    expected_schedule = {
        1: 0.10,
        2: 0.10,
        3: 0.10,
        4: 0.10,
        5: 0.10,
        6: 0.12,
        7: 0.14,
        8: 0.16,
        9: 0.18,
        10: 0.20,
        11: 0.20,
    }
    actual_schedule = {epoch: resolve_lambda_sem(cfg, epoch) for epoch in expected_schedule}
    for epoch, expected in expected_schedule.items():
        actual = actual_schedule[epoch]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"epoch {epoch}: lambda_sem={actual}, expected {expected}")

    if not math.isclose(float(training_cfg["visual_extractor_lr"]), 1e-5, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("visual_extractor_lr must be 1e-5")
    if not bool(model_cfg.get("use_adaptive_granularity", False)):
        raise AssertionError("adaptive weighted-mean aggregation must remain enabled")
    if not math.isclose(float(model_cfg["semantic_logit_scale"]), 20.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("semantic_logit_scale must remain 20.0")
    if str(training_cfg.get("optimizer", "")).lower() != "sam":
        raise AssertionError("SAM must remain enabled")
    if str(training_cfg.get("base_optimizer", "")).lower() != "adamw":
        raise AssertionError("AdamW must remain the SAM base optimizer")
    if str(model_cfg.get("name", "")) != "rafdb_siglip2_semantic_stable_v2":
        raise AssertionError("model.name must preserve the semantic-stable v2 lineage")
    if Path(cfg["paths"]["output_dir"]).name != "rafdb_siglip2_semantic_stable_v2":
        raise AssertionError("output_dir must not overwrite an existing RAF-DB experiment")

    schedule_text = ", ".join(f"ep{epoch}={value:.2f}" for epoch, value in actual_schedule.items())
    print("SEMANTIC_STABLE_CONFIG_OK")
    print(f"config={args.config}")
    print(f"output_dir={cfg['paths']['output_dir']}")
    print(f"lambda_sem_schedule: {schedule_text}")
    print(f"visual_extractor_lr={float(training_cfg['visual_extractor_lr']):.8f}")
    print("aggregation=adaptive_weighted_mean")
    print(f"semantic_logit_scale={float(model_cfg['semantic_logit_scale']):.1f}")
    print(f"optimizer={training_cfg['optimizer']} base_optimizer={training_cfg['base_optimizer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
