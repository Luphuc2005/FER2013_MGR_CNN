from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

REQUIRED_SECTIONS = ("seed", "runtime", "data", "augmentation", "model", "training", "paths")


def stringify_dict_keys(d: Any) -> Any:
    if isinstance(d, dict):
        return {str(k): stringify_dict_keys(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [stringify_dict_keys(x) for x in d]
    return d


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    config_path = _resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = stringify_dict_keys(cfg)
    validate_config(cfg, config_path)
    resolve_paths(cfg)
    apply_env_overrides(cfg)
    resolve_tta_config(cfg)
    return cfg


def resolve_tta_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = cfg.get("runtime", {})
    tta_cfg = cfg.get("tta", {})

    # Backward compatibility with old flags
    eval_hflip_old = bool(runtime.get("eval_tta_hflip", False))
    train_val_hflip_old = bool(runtime.get("train_val_tta_hflip", False))

    enabled = bool(tta_cfg.get("enabled", eval_hflip_old or train_val_hflip_old))
    hflip = bool(tta_cfg.get("hflip", True if enabled else False))
    orig_w = float(tta_cfg.get("original_weight", 0.5))
    flip_w = float(tta_cfg.get("flip_weight", 0.5))

    if enabled and hflip:
        total_w = orig_w + flip_w
        if total_w > 0 and abs(total_w - 1.0) > 1e-5:
            orig_w = orig_w / total_w
            flip_w = flip_w / total_w

    resolved = {
        "enabled": enabled,
        "hflip": hflip,
        "original_weight": orig_w,
        "flip_weight": flip_w,
    }
    cfg["tta"] = resolved
    return resolved


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


def _env_str(name: str, current: Optional[str]) -> Optional[str]:
    raw = os.environ.get(name)
    return current if raw in (None, "") else raw


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
    runtime["eval_tta_hflip"] = _env_bool("MGR_EVAL_TTA_HFLIP", bool(runtime.get("eval_tta_hflip", False)))
    runtime["train_val_tta_hflip"] = _env_bool(
        "MGR_TRAIN_VAL_TTA_HFLIP",
        bool(runtime.get("train_val_tta_hflip", False)),
    )
    data["data_path"] = _env_str("MGR_DATA_PATH", data.get("data_path"))
    data["mask_dir"] = _env_str("MGR_MASK_DIR", data.get("mask_dir"))
    data["predecode_pixels"] = _env_bool("MGR_PREDECODE_PIXELS", bool(data.get("predecode_pixels", False)))
    data["preload_masks"] = _env_bool("MGR_PRELOAD_MASKS", bool(data.get("preload_masks", False)))
    data["allow_missing_masks"] = _env_bool(
        "MGR_ALLOW_MISSING_MASKS",
        bool(data.get("allow_missing_masks", False)),
    )
    data["cache"] = _env_bool("MGR_CACHE_DATA", bool(data.get("cache", False)))
    data["shuffle_buffer"] = _env_int("MGR_SHUFFLE_BUFFER", data.get("shuffle_buffer"))
    data["max_train_samples"] = _env_int("MGR_MAX_TRAIN_SAMPLES", data.get("max_train_samples"))
    data["max_val_samples"] = _env_int("MGR_MAX_VAL_SAMPLES", data.get("max_val_samples"))
    data["max_test_samples"] = _env_int("MGR_MAX_TEST_SAMPLES", data.get("max_test_samples"))
    training["epochs"] = _env_int("MGR_EPOCHS", training.get("epochs"))
    training["patience"] = _env_int("MGR_PATIENCE", training.get("patience"))
    training["best_checkpoint_start_epoch"] = _env_int(
        "MGR_BEST_CHECKPOINT_START_EPOCH",
        training.get("best_checkpoint_start_epoch"),
    )
    cfg["paths"]["output_dir"] = _env_str("MGR_OUTPUT_DIR", cfg["paths"].get("output_dir"))
    cfg["paths"]["logs_dir"] = _env_str("MGR_LOGS_DIR", cfg["paths"].get("logs_dir"))
    cfg["paths"]["auto_increment"] = _env_bool("MGR_AUTO_INCREMENT_DIR", bool(cfg["paths"].get("auto_increment", True)))
    resolve_paths(cfg)


def get_versioned_directories(base_dir: Path) -> List[Tuple[int, Path]]:
    """Returns sorted list of (version_number, directory_path) matching base_dir name pattern."""
    import re
    base_dir = Path(base_dir)
    parent = base_dir.parent
    stem = base_dir.name

    if not parent.exists():
        return []

    pattern = re.compile(rf"^{re.escape(stem)}(?:_v?(\d+))?$")
    versioned = []

    for p in parent.iterdir():
        if p.is_dir():
            match = pattern.match(p.name)
            if match:
                ver_str = match.group(1)
                ver = int(ver_str) if ver_str else 1
                versioned.append((ver, p))

    versioned.sort(key=lambda x: x[0])
    return versioned


def get_next_versioned_dir(base_dir: Path) -> Path:
    """Returns next available versioned directory path if base_dir or latest version already has checkpoints."""
    base_dir = Path(base_dir)
    versioned = get_versioned_directories(base_dir)
    if not versioned:
        return base_dir

    highest_ver, highest_path = versioned[-1]
    ckpt_dir = highest_path / "checkpoints"
    has_checkpoints = ckpt_dir.exists() and any(ckpt_dir.rglob("ckpt-*"))
    has_history = (highest_path / "training_history.csv").exists()

    if not has_checkpoints and not has_history:
        return highest_path

    next_ver = highest_ver + 1
    return base_dir.parent / f"{base_dir.name}_v{next_ver}"


def get_latest_versioned_dir(base_dir: Path) -> Path:
    """Returns the latest existing versioned directory that contains checkpoints."""
    base_dir = Path(base_dir)
    versioned = get_versioned_directories(base_dir)
    if not versioned:
        return base_dir

    for ver, p in reversed(versioned):
        ckpt_dir = p / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.rglob("ckpt-*")):
            return p

    return versioned[-1][1]


def resolve_auto_increment_output_dir(
    cfg: Dict[str, Any],
    is_resume: bool = False,
    for_eval: bool = False,
    auto_increment: Optional[bool] = None,
) -> Path:
    """Resolves output_dir in cfg. If auto_increment is enabled and resume is False,
    creates next version directory (_v2, _v3, etc.) when checkpoints already exist.
    """
    paths = cfg.setdefault("paths", {})
    raw_output = paths.get("output_dir", "outputs/default")
    base_path = Path(raw_output)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path

    if auto_increment is None:
        auto_increment = bool(paths.get("auto_increment", True))

    if not auto_increment:
        paths["output_dir"] = str(base_path)
        return base_path

    if for_eval or is_resume:
        target_path = get_latest_versioned_dir(base_path)
    else:
        target_path = get_next_versioned_dir(base_path)

    paths["output_dir"] = str(target_path)
    return target_path


def global_batch_size(cfg: Dict[str, Any], replicas: Optional[int] = None) -> int:
    runtime = cfg["runtime"]
    if replicas is None:
        replicas = len(runtime["gpu_ids"])
    return int(runtime["batch_size_per_gpu"]) * int(replicas)
