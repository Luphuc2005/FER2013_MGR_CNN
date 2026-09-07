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
    parser = argparse.ArgumentParser(description="Validate the RAF-DB semantic-stable v5 config contract.")
    parser.add_argument(
        "config",
        nargs="?",
        default="config_rafdb_siglip2_semantic_stable_v5.yaml",
    )
    return parser.parse_args()


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"{name}={actual}, expected {expected}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    clip_cfg = model_cfg.get("clip_semantic", {})
    training_cfg = cfg["training"]

    expected_schedule = {
        1: 0.10,
        2: 0.10,
        3: 0.10,
        4: 0.10,
        5: 0.10,
        6: 0.11,
        7: 0.12,
        8: 0.13,
        9: 0.14,
        10: 0.15,
        11: 0.15,
    }
    actual_schedule = {epoch: resolve_lambda_sem(cfg, epoch) for epoch in expected_schedule}
    for epoch, expected in expected_schedule.items():
        assert_close(actual_schedule[epoch], expected, f"epoch {epoch} lambda_sem")

    assert_close(training_cfg["visual_extractor_lr"], 1e-5, "visual_extractor_lr")
    assert_close(model_cfg["lambda_sem"], 0.10, "model.lambda_sem")
    assert_close(model_cfg["lambda_hard"], 0.10, "model.lambda_hard")
    assert_close(model_cfg["hard_margin"], 0.30, "model.hard_margin")
    assert_close(clip_cfg["lambda_sem"], 0.10, "model.clip_semantic.lambda_sem")
    assert_close(clip_cfg["lambda_hard"], 0.10, "model.clip_semantic.lambda_hard")
    assert_close(clip_cfg["hard_margin"], 0.30, "model.clip_semantic.hard_margin")
    assert_close(model_cfg["semantic_logit_scale"], 20.0, "semantic_logit_scale")

    # Verify logit soft fusion parameters
    assert_close(model_cfg.get("semantic_fusion_alpha", 0.0), 0.15, "model.semantic_fusion_alpha")
    assert_close(clip_cfg.get("semantic_fusion_alpha", 0.0), 0.15, "model.clip_semantic.semantic_fusion_alpha")
    if bool(model_cfg.get("semantic_fusion_training", False)):
        raise AssertionError("model.semantic_fusion_training must be False for inference-only soft fusion")

    if not bool(model_cfg.get("use_adaptive_granularity", False)):
        raise AssertionError("adaptive weighted-mean aggregation must remain enabled")
    if not bool(model_cfg.get("use_hard_semantic_loss", False)):
        raise AssertionError("hard semantic loss must remain enabled")

    expected_pairs = {
        0: [1, 4],
        1: [4, 0, 6, 3],
        2: [5, 4, 0],
        4: [6, 1],
        5: [2],
        6: [4, 1],
    }
    hard_pairs = {int(key): [int(value) for value in values] for key, values in model_cfg["hard_pairs"].items()}
    if hard_pairs != expected_pairs:
        raise AssertionError(f"model.hard_pairs={hard_pairs}, expected {expected_pairs}")
    if str(training_cfg.get("optimizer", "")).lower() != "sam":
        raise AssertionError("SAM must remain enabled")
    if str(training_cfg.get("base_optimizer", "")).lower() != "adamw":
        raise AssertionError("AdamW must remain the SAM base optimizer")
    if str(model_cfg.get("name", "")) != "rafdb_siglip2_semantic_stable_v5":
        raise AssertionError("model.name must use the semantic-stable v5 lineage")
    if Path(cfg["paths"]["output_dir"]).name != "rafdb_siglip2_semantic_stable_v5":
        raise AssertionError("output_dir must use the independent semantic-stable v5 lineage")

    schedule_text = ", ".join(f"ep{epoch}={value:.2f}" for epoch, value in actual_schedule.items())
    print("SEMANTIC_STABLE_V5_CONFIG_OK")
    print(f"config={args.config}")
    print(f"output_dir={cfg['paths']['output_dir']}")
    print(f"lambda_sem_schedule: {schedule_text}")
    print(f"visual_extractor_lr={float(training_cfg['visual_extractor_lr']):.8f}")
    print("aggregation=adaptive_weighted_mean")
    print(f"lambda_hard={float(model_cfg['lambda_hard']):.2f} hard_margin={float(model_cfg['hard_margin']):.2f}")
    print(f"semantic_logit_scale={float(model_cfg['semantic_logit_scale']):.1f}")
    print(f"semantic_fusion_alpha={float(model_cfg['semantic_fusion_alpha']):.2f} (inference_only=True)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
