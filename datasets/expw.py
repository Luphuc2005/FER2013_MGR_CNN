from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import pandas as pd
except ImportError:
    pd = None

from .fer2013 import (
    EMOTION_NAMES,
    SplitRecords,
    _augment_pair,
    _limit_records,
    _normalize_image,
    _random_erasing,
    _resolve_path,
)


def _find_column(df, candidates, default_idx=None):
    cols_lower = [str(c).strip().lower() for c in df.columns]
    for cand in candidates:
        if cand in cols_lower:
            return df.columns[cols_lower.index(cand)]
    if default_idx is not None and default_idx < len(df.columns):
        return df.columns[default_idx]
    return None


def collect_expw_split_records(
    data_dir,
    split: str,
    *,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    image_root: Optional[str] = None,
) -> SplitRecords:
    """Collects dataset records for ExpW from CSV files with bounding box cropping.

    ExpW 7 classes:
        0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    Expected CSV columns: image path/filename, label/emotion, left, top, right, bottom bbox coordinates.
    """
    data_dir_path = Path(data_dir) if data_dir else Path("data/expw")

    # 1. Resolve CSV Path for the split
    if split == "train":
        target_csv = train_csv or (data_dir_path / "expw_train.csv")
    elif split == "val":
        target_csv = val_csv or (data_dir_path / "expw_val.csv")
    else:  # test
        target_csv = test_csv or (data_dir_path / "expw_test.csv")

    resolved_csv = _resolve_path(target_csv)
    if resolved_csv is None or not resolved_csv.exists():
        # Fallback names
        fallback = _resolve_path(data_dir_path / f"{split}.csv")
        if fallback is not None and fallback.exists():
            resolved_csv = fallback
        else:
            raise FileNotFoundError(
                f"ExpW CSV file not found for split '{split}': {target_csv} (resolved: {resolved_csv})"
            )

    print(f"[INFO] Loading ExpW {split} split from CSV: {resolved_csv}")

    # 2. Read CSV file
    if pd is not None:
        df = pd.read_csv(resolved_csv)
        cols_lower = [str(c).strip().lower() for c in df.columns]

        # Identify label column
        lbl_candidates = ["label", "emotion", "expression", "expr", "target", "class", "y"]
        lbl_col = _find_column(df, lbl_candidates, default_idx=1 if len(df.columns) > 1 else 0)

        # Identify image path column
        path_candidates = ["image_name", "image_path", "filename", "filepath", "path", "image", "file", "name", "orig_filename"]
        path_col = _find_column(df, path_candidates, default_idx=0)

        # Identify bbox columns
        left_candidates = ["left", "bbox_left", "x1", "xmin", "l"]
        top_candidates = ["top", "bbox_top", "y1", "ymin", "t"]
        right_candidates = ["right", "bbox_right", "x2", "xmax", "r"]
        bottom_candidates = ["bottom", "bbox_bottom", "y2", "ymax", "b"]

        left_col = _find_column(df, left_candidates)
        top_col = _find_column(df, top_candidates)
        right_col = _find_column(df, right_candidates)
        bottom_col = _find_column(df, bottom_candidates)

        bbox_candidates = ["bbox", "box", "bounding_box", "face_box", "crop_box"]
        bbox_col = _find_column(df, bbox_candidates)

        if left_col and top_col and right_col and bottom_col:
            bboxes = df[[left_col, top_col, right_col, bottom_col]].to_numpy(dtype=np.float32)
        elif bbox_col is not None:
            parsed_bboxes = []
            for val in df[bbox_col]:
                v_str = str(val).replace("[", "").replace("]", "").replace(",", " ").strip()
                parts = [float(x) for x in v_str.split()]
                if len(parts) >= 4:
                    parsed_bboxes.append(parts[:4])
                else:
                    parsed_bboxes.append([0.0, 0.0, 0.0, 0.0])
            bboxes = np.array(parsed_bboxes, dtype=np.float32)
        else:
            bboxes = np.zeros((len(df), 4), dtype=np.float32)

        raw_labels = df[lbl_col].to_numpy()
        raw_paths = df[path_col].astype(str).to_numpy()
    else:
        import csv
        raw_labels_list, raw_paths_list, bboxes_list = [], [], []
        with resolved_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            fn_lower = [f.strip().lower() for f in fieldnames]

            lbl_key = next((fieldnames[i] for i, f in enumerate(fn_lower) if f in ("label", "emotion", "expression", "expr", "target", "class")), fieldnames[1] if len(fieldnames) > 1 else fieldnames[0])
            path_key = next((fieldnames[i] for i, f in enumerate(fn_lower) if f in ("image_name", "image_path", "filename", "filepath", "path", "image", "file")), fieldnames[0])

            l_key = next((fieldnames[i] for i, f in enumerate(fn_lower) if f in ("left", "bbox_left", "x1", "xmin")), None)
            t_key = next((fieldnames[i] for i, f in enumerate(fn_lower) if f in ("top", "bbox_top", "y1", "ymin")), None)
            r_key = next((fieldnames[i] for i, f in enumerate(fn_lower) if f in ("right", "bbox_right", "x2", "xmax")), None)
            b_key = next((fieldnames[i] for i, f in enumerate(fn_lower) if f in ("bottom", "bbox_bottom", "y2", "ymax")), None)

            for row in reader:
                raw_labels_list.append(row[lbl_key])
                raw_paths_list.append(row[path_key])
                if l_key and t_key and r_key and b_key:
                    bboxes_list.append([float(row[l_key]), float(row[t_key]), float(row[r_key]), float(row[b_key])])
                else:
                    bboxes_list.append([0.0, 0.0, 0.0, 0.0])

        raw_labels = np.array(raw_labels_list)
        raw_paths = np.array(raw_paths_list, dtype=str)
        bboxes = np.array(bboxes_list, dtype=np.float32)

    # 3. Label conversion & mapping
    expw_emotion_map = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6,
        "anger": 0, "happiness": 3, "sadness": 4, "surprised": 5
    }

    labels = []
    for val in raw_labels:
        val_str = str(val).strip().lower()
        if val_str in expw_emotion_map:
            labels.append(expw_emotion_map[val_str])
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

    # Verify label range
    if labels_arr.min() < 0 or labels_arr.max() > 6:
        print(f"[WARNING] Labels range in {resolved_csv.name} is [{labels_arr.min()}..{labels_arr.max()}], expected [0..6].")

    # 4. Resolve image paths
    default_img_root = "/home/ptbao/projects/FER2013_MGR_CNN/data/expw_gdrive/data/image/extracted_full/origin"
    img_root_path = _resolve_path(image_root) if image_root else _resolve_path(default_img_root)

    resolved_paths = []
    for p_str in raw_paths:
        p_str = p_str.replace("\\", "/").strip()
        p_obj = Path(p_str)

        if p_obj.is_absolute() and p_obj.exists():
            resolved_paths.append(str(p_obj))
        elif img_root_path is not None and (img_root_path / p_str).exists():
            resolved_paths.append(str(img_root_path / p_str))
        elif img_root_path is not None and (img_root_path / p_obj.name).exists():
            resolved_paths.append(str(img_root_path / p_obj.name))
        elif img_root_path is not None:
            resolved_paths.append(str(img_root_path / p_str))
        else:
            resolved_paths.append(p_str)

    sample_ids = np.arange(len(labels_arr), dtype=np.int64)

    print(
        f"[INFO] Loaded {len(labels_arr)} samples from {resolved_csv.name} | "
        f"Label range: [{labels_arr.min()}..{labels_arr.max()}] | "
        f"BBox sample[0]: {bboxes[0].tolist()}"
    )

    return SplitRecords(
        images=np.array(resolved_paths, dtype=object),
        labels=labels_arr,
        sample_ids=sample_ids,
        mask_paths=None,
        masks=None,
        bboxes=bboxes,
    )


def _decode_expw_image(image_path_tensor: tf.Tensor, bbox_tensor: tf.Tensor, image_size: int, channels: int) -> tf.Tensor:
    """Decodes image file, crops face bounding box (left, top, right, bottom), and resizes to target resolution."""
    target_h, target_w = int(image_size), int(image_size)

    def _read_crop_resize(p_tensor, box_tensor):
        p_str = p_tensor.numpy().decode("utf-8") if hasattr(p_tensor, "numpy") else str(p_tensor)
        bbox = box_tensor.numpy() if hasattr(box_tensor, "numpy") else box_tensor
        left, top, right, bottom = [float(x) for x in bbox]

        p_path = Path(p_str)
        if not p_path.is_absolute():
            p_path = Path(__file__).resolve().parents[1] / p_str

        if p_path.exists() and p_path.is_file():
            try:
                from PIL import Image
                with Image.open(p_path) as pil_img:
                    if channels == 3 and pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
                    elif channels == 1 and pil_img.mode != "L":
                        pil_img = pil_img.convert("L")

                    w, h = pil_img.size
                    if right <= 1.0 and bottom <= 1.0 and max(right, bottom) > 0:
                        l_px = max(0, int(round(left * w)))
                        t_px = max(0, int(round(top * h)))
                        r_px = min(w, int(round(right * w)))
                        b_px = min(h, int(round(bottom * h)))
                    else:
                        l_px = max(0, int(round(left)))
                        t_px = max(0, int(round(top)))
                        r_px = min(w, int(round(right)))
                        b_px = min(h, int(round(bottom)))

                    if r_px > l_px and b_px > t_px:
                        cropped = pil_img.crop((l_px, t_px, r_px, b_px))
                    else:
                        cropped = pil_img

                    resized = cropped.resize((target_w, target_h), Image.BILINEAR)
                    arr = np.array(resized, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=-1)
                    return arr
            except Exception:
                pass

            try:
                img_raw = tf.io.read_file(str(p_path))
                img = tf.io.decode_image(img_raw, channels=channels, expand_animations=False)
                img = tf.cast(img, tf.float32)
                h_img, w_img = tf.shape(img)[0], tf.shape(img)[1]

                l_px = tf.clip_by_value(tf.cast(left, tf.int32), 0, w_img)
                t_px = tf.clip_by_value(tf.cast(top, tf.int32), 0, h_img)
                r_px = tf.clip_by_value(tf.cast(right, tf.int32), l_px + 1, w_img)
                b_px = tf.clip_by_value(tf.cast(bottom, tf.int32), t_px + 1, h_img)

                cropped = img[t_px:b_px, l_px:r_px, :]
                if tf.shape(cropped)[0] == 0 or tf.shape(cropped)[1] == 0:
                    cropped = img
                if cropped.shape[-1] == 1 and channels == 3:
                    cropped = tf.image.grayscale_to_rgb(cropped)
                return tf.image.resize(cropped, [target_h, target_w], method="bilinear")
            except Exception:
                pass

        return np.zeros((target_h, target_w, channels), dtype=np.float32)

    image = tf.py_function(func=_read_crop_resize, inp=[image_path_tensor, bbox_tensor], Tout=tf.float32)
    image.set_shape([target_h, target_w, channels])
    return image


def _parse_expw_example(pixels, bboxes, label, sample_id, *, cfg: Dict, split: str):
    image = _decode_expw_image(pixels, bboxes, int(cfg["data"]["image_size"]), int(cfg["data"]["channels"]))
    mask = None

    image, mask = _augment_pair(image, mask, sample_id, cfg["augmentation"], split)
    image = _normalize_image(image, int(cfg["data"]["channels"]))

    if split == "train":
        image = _random_erasing(image, cfg["augmentation"])

    features = {"image": image}
    return features, tf.cast(label, tf.int32)


def make_expw_dataset(records: SplitRecords, cfg: Dict, *, split: str, training: bool, replicas: int) -> tf.data.Dataset:
    with tf.device("/CPU:0"):
        pixel_tensor = (
            tf.convert_to_tensor(records.images)
            if isinstance(records.images, np.ndarray) and records.images.dtype != object
            else tf.convert_to_tensor(records.images.astype(str))
        )
        bbox_tensor = (
            tf.convert_to_tensor(records.bboxes, dtype=tf.float32)
            if records.bboxes is not None
            else tf.zeros((len(records.images), 4), dtype=tf.float32)
        )
        tensors = {
            "pixels": pixel_tensor,
            "bboxes": bbox_tensor,
            "labels": tf.convert_to_tensor(records.labels),
            "sample_ids": tf.convert_to_tensor(records.sample_ids),
        }

    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(int(cfg["data"].get("shuffle_buffer", 10000)), seed=int(cfg["seed"]["random_seed"]), reshuffle_each_iteration=True)

    options = tf.data.Options()
    runtime_cfg = cfg.get("runtime", {})
    options.experimental_deterministic = bool(runtime_cfg.get("tf_data_deterministic", False))
    private_threads = runtime_cfg.get("tf_data_private_threadpool_size")
    if private_threads:
        options.threading.private_threadpool_size = int(private_threads)
        options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)

    def mapper(item):
        return _parse_expw_example(
            item["pixels"],
            item["bboxes"],
            item["labels"],
            item["sample_ids"],
            cfg=cfg,
            split=split,
        )

    parallel_calls = runtime_cfg.get("tf_data_num_parallel_calls")
    if parallel_calls in (None, "", 0):
        parallel_calls = tf.data.AUTOTUNE
    else:
        parallel_calls = int(parallel_calls)

    ds = ds.map(mapper, num_parallel_calls=parallel_calls, deterministic=bool(runtime_cfg.get("tf_data_deterministic", False)))
    if cfg["data"].get("cache", False) and not training:
        ds = ds.cache()

    ds = ds.batch(int(cfg["runtime"]["batch_size_per_gpu"]) * int(replicas), drop_remainder=training)

    prefetch_buffer = runtime_cfg.get("prefetch_buffer")
    if prefetch_buffer in (None, "", 0):
        prefetch_buffer = tf.data.AUTOTUNE
    else:
        prefetch_buffer = int(prefetch_buffer)

    return ds.prefetch(prefetch_buffer)


def build_expw_datasets(cfg: Dict, replicas: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Builds train, val, and test tf.data.Datasets for ExpW with face cropping."""
    data_cfg = cfg.get("data", {})
    data_dir = data_cfg.get("data_path", "data/expw")
    train_csv = data_cfg.get("train_csv")
    val_csv = data_cfg.get("val_csv")
    test_csv = data_cfg.get("test_csv")
    image_root = data_cfg.get("image_root")

    records = {
        split: collect_expw_split_records(
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
        make_expw_dataset(records["train"], cfg, split="train", training=True, replicas=replicas),
        make_expw_dataset(records["val"], cfg, split="val", training=False, replicas=replicas),
        make_expw_dataset(records["test"], cfg, split="test", training=False, replicas=replicas),
    )
