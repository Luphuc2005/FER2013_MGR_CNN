from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

REQUIRED_SECTIONS = ("seed", "runtime", "data", "augmentation", "model", "training", "paths")


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    config_path = _resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_config(cfg, config_path)
    resolve_paths(cfg)
    apply_env_overrides(cfg)
    return cfg


def _resolve_config_path(path: Optional[Union[str, Path]]) -> Path:
    if path in (None, ""):
        return DEFAULT_CONFIG_PATH
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path


def validate_config(cfg: Dict[str, Any], config_path: Path) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing required section(s) in {config_path}: {joined}")


def resolve_paths(cfg: Dict[str, Any]) -> None:
    for section, keys in {
        "data": ["data_path", "mask_dir", "bad_row_indices_path"],
        "paths": ["output_dir", "logs_dir"],
    }.items():
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value in (None, ""):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            cfg[section][key] = str(path)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_list(value: str) -> List[int]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(parsed, int):
        return [int(parsed)]
    if isinstance(parsed, (list, tuple)):
        return [int(item) for item in parsed]
    raise ValueError(f"Cannot parse integer list from {value!r}")


def _env_int(name: str, current: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    return current if raw in (None, "") else int(raw)


def _env_bool(name: str, current: bool) -> bool:
    raw = os.environ.get(name)
    return current if raw in (None, "") else _parse_bool(raw)


def _env_list(name: str, current: Iterable[int]) -> List[int]:
    raw = os.environ.get(name)
    return list(current) if raw in (None, "") else _parse_int_list(raw)


def apply_env_overrides(cfg: Dict[str, Any]) -> None:
    runtime = cfg.setdefault("runtime", {})
    data = cfg.setdefault("data", {})
    training = cfg.setdefault("training", {})
    runtime["batch_size_per_gpu"] = _env_int("MGR_BATCH_SIZE_PER_GPU", runtime.get("batch_size_per_gpu"))
    runtime["fallback_batch_size_per_gpu"] = _env_int(
        "MGR_FALLBACK_BATCH_SIZE_PER_GPU",
        runtime.get("fallback_batch_size_per_gpu"),
    )
    runtime["gpu_ids"] = _env_list("MGR_GPU_IDS", runtime.get("gpu_ids", []))
    runtime["require_two_gpus"] = _env_bool("MGR_REQUIRE_TWO_GPUS", bool(runtime.get("require_two_gpus", True)))
    runtime["min_gpus"] = _env_int("MGR_MIN_GPUS", runtime.get("min_gpus"))
    runtime["memory_growth"] = _env_bool("MGR_MEMORY_GROWTH", bool(runtime.get("memory_growth", True)))
    runtime["intra_op_threads"] = _env_int("MGR_TF_INTRA_OP_THREADS", runtime.get("intra_op_threads"))
    runtime["inter_op_threads"] = _env_int("MGR_TF_INTER_OP_THREADS", runtime.get("inter_op_threads"))
    runtime["tf_data_num_parallel_calls"] = _env_int(
        "MGR_TF_DATA_NUM_PARALLEL_CALLS",
        runtime.get("tf_data_num_parallel_calls"),
    )
    runtime["tf_data_private_threadpool_size"] = _env_int(
        "MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE",
        runtime.get("tf_data_private_threadpool_size"),
    )
    runtime["tf_data_deterministic"] = _env_bool(
        "MGR_TF_DATA_DETERMINISTIC",
        bool(runtime.get("tf_data_deterministic", True)),
    )
    runtime["prefetch_buffer"] = _env_int("MGR_PREFETCH_BUFFER", runtime.get("prefetch_buffer"))
    runtime["distributed_eval"] = _env_bool("MGR_DISTRIBUTED_EVAL", bool(runtime.get("distributed_eval", False)))
    data["shuffle_buffer"] = _env_int("MGR_SHUFFLE_BUFFER", data.get("shuffle_buffer"))
    data["max_train_samples"] = _env_int("MGR_MAX_TRAIN_SAMPLES", data.get("max_train_samples"))
    data["max_val_samples"] = _env_int("MGR_MAX_VAL_SAMPLES", data.get("max_val_samples"))
    data["max_test_samples"] = _env_int("MGR_MAX_TEST_SAMPLES", data.get("max_test_samples"))
    training["epochs"] = _env_int("MGR_EPOCHS", training.get("epochs"))
    training["patience"] = _env_int("MGR_PATIENCE", training.get("patience"))


def global_batch_size(cfg: Dict[str, Any], replicas: Optional[int] = None) -> int:
    runtime = cfg["runtime"]
    if replicas is None:
        replicas = len(runtime["gpu_ids"])
    return int(runtime["batch_size_per_gpu"]) * int(replicas)
