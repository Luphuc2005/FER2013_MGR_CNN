from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import pandas as pd
except ImportError:
    pd = None

from .fer2013 import SplitRecords, _limit_records, _resolve_path, make_dataset


FERPLUS_EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
    "Contempt",
]


def _required_column(df, name: str, csv_path: Path) -> str:
    cols = {str(col).strip().lower(): col for col in df.columns}
    if name not in cols:
        raise ValueError(f"FERPlus manifest {csv_path} is missing required column {name!r}.")
    return cols[name]


def _resolve_image_path(path_value: str, image_root: Optional[str]) -> str:
    raw = str(path_value).replace("\\", "/").strip()
    p = Path(raw)
    if p.is_absolute():
        return str(p)

    candidates = []
    if image_root:
        root = _resolve_path(image_root)
        if root is not None:
            candidates.append(root / raw)
    project_root = Path(__file__).resolve().parents[1]
    candidates.append(project_root / raw)
    data_root = _resolve_path("data")
    if data_root is not None:
        candidates.append(data_root / raw)

    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(project_root))
            except ValueError:
                return str(candidate)
    return str(candidates[0] if candidates else p)


def collect_ferplus_split_records(
    csv_path: str,
    split: str,
    *,
    image_root: Optional[str] = None,
    expected_samples: Optional[int] = None,
    class_names=None,
) -> SplitRecords:
    if pd is None:
        raise ImportError("pandas is required to load FERPlus official manifest CSV files.")

    resolved_csv = _resolve_path(csv_path)
    if resolved_csv is None or not resolved_csv.exists():
        raise FileNotFoundError(f"FERPlus {split} manifest not found: {csv_path} (resolved: {resolved_csv})")

    df = pd.read_csv(resolved_csv)
    path_col = _required_column(df, "path", resolved_csv)
    label_col = _required_column(df, "label", resolved_csv)
    class_col = _required_column(df, "class_name", resolved_csv)
    image_name_col = _required_column(df, "image_name", resolved_csv)

    if expected_samples is not None and len(df) != int(expected_samples):
        raise ValueError(
            f"FERPlus {split} sample count mismatch: got {len(df)}, expected {int(expected_samples)}."
        )

    labels = df[label_col].to_numpy(dtype=np.int64)
    if labels.size == 0:
        raise ValueError(f"FERPlus {split} manifest is empty: {resolved_csv}")
    if labels.min() < 0 or labels.max() >= len(FERPLUS_EMOTION_NAMES):
        raise ValueError(
            f"FERPlus {split} labels out of range [0..7]: min={labels.min()}, max={labels.max()}."
        )

    expected_names = list(class_names or FERPLUS_EMOTION_NAMES)
    if len(expected_names) != len(FERPLUS_EMOTION_NAMES):
        raise ValueError(f"FERPlus class_names must have 8 entries, got {len(expected_names)}.")

    bad_names = []
    for class_id, class_name in zip(labels.tolist(), df[class_col].astype(str).tolist()):
        expected = expected_names[int(class_id)].lower()
        observed = class_name.strip().lower()
        if observed != expected:
            bad_names.append((int(class_id), class_name, expected_names[int(class_id)]))
            if len(bad_names) >= 5:
                break
    if bad_names:
        raise ValueError(
            f"FERPlus {split} class_name does not match required label mapping. "
            f"First mismatches: {bad_names}"
        )

    paths = np.array(
        [_resolve_image_path(p, image_root) for p in df[path_col].astype(str).tolist()],
        dtype=object,
    )
    image_names = df[image_name_col].astype(str).tolist()
    sample_ids = np.arange(len(labels), dtype=np.int64)

    counts = np.bincount(labels, minlength=len(FERPLUS_EMOTION_NAMES))[: len(FERPLUS_EMOTION_NAMES)]
    print(
        f"[INFO] Loaded FERPlus {split}: {len(labels)} samples from {resolved_csv} | "
        f"label_range=[{labels.min()}..{labels.max()}] | counts={counts.tolist()}",
        flush=True,
    )
    if image_names:
        print(f"[INFO] FERPlus {split} first image_name={image_names[0]!r}", flush=True)

    return SplitRecords(
        images=paths,
        labels=labels,
        sample_ids=sample_ids,
        mask_paths=None,
    )


def build_ferplus_datasets(cfg: Dict, replicas: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    data_cfg = cfg.get("data", {})
    expected = data_cfg.get("expected_samples", {})
    class_names = data_cfg.get("class_names", FERPLUS_EMOTION_NAMES)

    records = {
        "train": collect_ferplus_split_records(
            data_cfg["train_csv"],
            "train",
            image_root=data_cfg.get("image_root"),
            expected_samples=expected.get("train"),
            class_names=class_names,
        ),
        "val": collect_ferplus_split_records(
            data_cfg["val_csv"],
            "val",
            image_root=data_cfg.get("image_root"),
            expected_samples=expected.get("val"),
            class_names=class_names,
        ),
        "test": collect_ferplus_split_records(
            data_cfg["test_csv"],
            "test",
            image_root=data_cfg.get("image_root"),
            expected_samples=expected.get("test"),
            class_names=class_names,
        ),
    }

    records["train"] = _limit_records(records["train"], data_cfg.get("max_train_samples"))
    records["val"] = _limit_records(records["val"], data_cfg.get("max_val_samples"))
    records["test"] = _limit_records(records["test"], data_cfg.get("max_test_samples"))

    return (
        make_dataset(records["train"], cfg, split="train", training=True, replicas=replicas),
        make_dataset(records["val"], cfg, split="val", training=False, replicas=replicas),
        make_dataset(records["test"], cfg, split="test", training=False, replicas=replicas),
    )
