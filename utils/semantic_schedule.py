from __future__ import annotations

from typing import Any, Dict


def configured_lambda_sem(cfg: Dict[str, Any]) -> float:
    """Return the static semantic-loss weight using the model's precedence rules."""
    model_cfg = cfg.get("model", {})
    clip_semantic_cfg = model_cfg.get("clip_semantic", {})
    if not isinstance(clip_semantic_cfg, dict):
        clip_semantic_cfg = {}
    return float(clip_semantic_cfg.get("lambda_sem", model_cfg.get("lambda_sem", 0.1)))


def resolve_lambda_sem(cfg: Dict[str, Any], epoch_number: int) -> float:
    """Resolve lambda_sem for a one-based epoch number."""
    base_value = configured_lambda_sem(cfg)
    schedule_cfg = cfg.get("training", {}).get("lambda_sem_schedule", {})
    if not isinstance(schedule_cfg, dict) or not bool(schedule_cfg.get("enabled", False)):
        return base_value

    epoch_number = int(epoch_number)
    if epoch_number < 1:
        raise ValueError(f"epoch_number must be >= 1, got {epoch_number}.")

    start_epoch = int(schedule_cfg.get("start_epoch", 5))
    end_epoch = int(schedule_cfg.get("end_epoch", 10))
    start_value = float(schedule_cfg.get("start_value", base_value))
    end_value = float(schedule_cfg.get("end_value", start_value))
    before_value = float(schedule_cfg.get("before_value", base_value))
    after_value = float(schedule_cfg.get("after_value", end_value))

    if start_epoch < 1 or end_epoch < start_epoch:
        raise ValueError(
            "lambda_sem_schedule requires 1 <= start_epoch <= end_epoch, "
            f"got start_epoch={start_epoch}, end_epoch={end_epoch}."
        )
    if min(before_value, start_value, end_value, after_value) < 0.0:
        raise ValueError("lambda_sem_schedule values must be non-negative.")

    if epoch_number < start_epoch:
        return before_value
    if epoch_number > end_epoch:
        return after_value
    if end_epoch == start_epoch:
        return end_value

    progress = float(epoch_number - start_epoch) / float(end_epoch - start_epoch)
    return start_value + progress * (end_value - start_value)
