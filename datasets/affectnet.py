from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import pandas as pd
except ImportError:
    pd = None

from .fer2013 import EMOTION_NAMES, SplitRecords, _limit_records, _resolve_path, make_dataset


def collect_affectnet_split_records(
    data_dir,
    split: str,
    *,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    image_root: Optional[str] = None,
) -> SplitRecords:
    """Collects dataset records for AffectNet-7 from CSV files.

    Supports custom CSV paths, image root paths, and label remapping.
    Expected label order (7 classes):
        0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    """
    data_dir_path = Path(data_dir) if data_dir else Path("data/affectnet")

    # 1. Resolve CSV Path for the split
    if split == "train":
        target_csv = train_csv or (data_dir_path / "affectnet7_train.csv")
    elif split == "val":
        target_csv = val_csv or (data_dir_path / "affectnet7_val.csv")
    else:  # test
        target_csv = test_csv or (data_dir_path / "affectnet7_test.csv")
        resolved_test = _resolve_path(target_csv)
        if resolved_test is None or not resolved_test.exists():
            # Fallback to official validation set if no separate test CSV is specified
            target_csv = val_csv or (data_dir_path / "affectnet7_val.csv")

    resolved_csv = _resolve_path(target_csv)
    if resolved_csv is None or not resolved_csv.exists():
        # Fallback names
        fallback = _resolve_path(data_dir_path / f"{split}.csv")
        if fallback is not None and fallback.exists():
            resolved_csv = fallback
        else:
            raise FileNotFoundError(
                f"AffectNet-7 CSV file not found for split '{split}': {target_csv} (resolved: {resolved_csv})"
            )

    print(f"[INFO] Loading AffectNet-7 {split} split from CSV: {resolved_csv}")

    # 2. Read CSV file
    if pd is not None:
        df = pd.read_csv(resolved_csv)
        cols_lower = [str(c).strip().lower() for c in df.columns]

        # Identify label column
        label_col_names = ["emotion", "label", "target", "class", "y", "expression", "expr", "label_idx"]
        lbl_col = None
        for cand in label_col_names:
            if cand in cols_lower:
                lbl_col = df.columns[cols_lower.index(cand)]
                break
        if lbl_col is None:
            lbl_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Identify image path column
        path_col_names = ["subdirectory_filepath", "image_path", "filepath", "path", "image", "file", "filename", "pixels"]
        path_col = None
        for cand in path_col_names:
            if cand in cols_lower:
                path_col = df.columns[cols_lower.index(cand)]
                break
        if path_col is None:
            path_col = df.columns[0]

        raw_labels = df[lbl_col].to_numpy()
        raw_paths = df[path_col].astype(str).to_numpy()
    else:
        import csv
        raw_labels_list, raw_paths_list = [], []
        with resolved_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            fieldnames_lower = [f.strip().lower() for f in fieldnames]
            lbl_key = next(
                (fieldnames[i] for i, f in enumerate(fieldnames_lower) if f in ("emotion", "label", "target", "class", "y", "expression", "expr")),
                fieldnames[1] if len(fieldnames) > 1 else fieldnames[0],
            )
            path_key = next(
                (fieldnames[i] for i, f in enumerate(fieldnames_lower) if f in ("subdirectory_filepath", "image_path", "filepath", "path", "image", "file", "filename")),
                fieldnames[0],
            )
            for row in reader:
                raw_labels_list.append(row[lbl_key])
                raw_paths_list.append(row[path_key])
        raw_labels = np.array(raw_labels_list)
        raw_paths = np.array(raw_paths_list, dtype=str)

    # 3. Label conversion & mapping
    labels = []
    emotion_name_map = {name.lower(): idx for idx, name in enumerate(EMOTION_NAMES)}
    emotion_name_map.update({"anger": 0, "happiness": 3, "sadness": 4})

    for val in raw_labels:
        val_str = str(val).strip().lower()
        if val_str in emotion_name_map:
            labels.append(emotion_name_map[val_str])
        else:
            try:
                l_int = int(float(val))
                labels.append(l_int)
            except ValueError:
                raise ValueError(f"Unrecognized label value {val!r} in CSV {resolved_csv}")

    labels_arr = np.array(labels, dtype=np.int64)

    # Check for 1-based labels [1..7] and remap to 0-based [0..6]
    if labels_arr.min() == 1 and labels_arr.max() == 7:
        print(f"[INFO] Remapping 1-based labels [1..7] -> 0-based [0..6] for {resolved_csv.name}")
        labels_arr = labels_arr - 1

    # Verify label range is within [0..6]
    if labels_arr.min() < 0 or labels_arr.max() > 6:
        print(f"[WARNING] Labels range in {resolved_csv.name} is [{labels_arr.min()}..{labels_arr.max()}], expected [0..6].")

    # 4. Resolve image paths
    img_root_path = _resolve_path(image_root) if image_root else None
    resolved_paths = []

    for p_str in raw_paths:
        p_str = p_str.replace("\\", "/").strip()
        p_obj = Path(p_str)

        if p_obj.is_absolute() and p_obj.exists():
            resolved_paths.append(str(p_obj))
        elif (_resolve_path(p_obj) is not None and _resolve_path(p_obj).exists()):
            resolved_paths.append(str(p_str))
        elif img_root_path is not None:
            combined = img_root_path / p_str
            if combined.exists():
                try:
                    rel_p = str(combined.relative_to(Path(__file__).resolve().parents[1]))
                except ValueError:
                    rel_p = str(combined)
                resolved_paths.append(rel_p)
            else:
                # If relative to image_root
                resolved_paths.append(str(img_root_path / p_str))
        else:
            resolved_paths.append(p_str)

    sample_ids = np.arange(len(labels_arr), dtype=np.int64)

    print(
        f"[INFO] Loaded {len(labels_arr)} samples from {resolved_csv.name} | "
        f"Label range: [{labels_arr.min()}..{labels_arr.max()}]"
    )

    return SplitRecords(
        images=np.array(resolved_paths, dtype=object),
        labels=labels_arr,
        sample_ids=sample_ids,
        mask_paths=None,
        masks=None,
    )


def build_affectnet_datasets(cfg: Dict, replicas: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Builds train, val, and test tf.data.Datasets for AffectNet-7."""
    data_cfg = cfg.get("data", {})
    data_dir = data_cfg.get("data_path", "data/affectnet")
    train_csv = data_cfg.get("train_csv")
    val_csv = data_cfg.get("val_csv")
    test_csv = data_cfg.get("test_csv")
    image_root = data_cfg.get("image_root")

    records = {
        split: collect_affectnet_split_records(
            data_dir,
            split,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            image_root=image_root,
        )
        for split in ("train", "val", "test")
    }

    records["train"] = _limit_records(records["train"], data_cfg.get("max_train_samples"))
    records["val"] = _limit_records(records["val"], data_cfg.get("max_val_samples"))
    records["test"] = _limit_records(records["test"], data_cfg.get("max_test_samples"))

    return (
        make_dataset(records["train"], cfg, split="train", training=True, replicas=replicas),
        make_dataset(records["val"], cfg, split="val", training=False, replicas=replicas),
        make_dataset(records["test"], cfg, split="test", training=False, replicas=replicas),
    )
