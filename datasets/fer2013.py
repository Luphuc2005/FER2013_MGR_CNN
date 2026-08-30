from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import pandas as pd
except ImportError:
    pd = None


EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


@dataclass
class SplitRecords:
    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    mask_paths: Optional[np.ndarray]
    masks: Optional[np.ndarray] = None


def _resolve_path(path):
    if path in (None, ""):
        return None
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[1] / p


def _limit_records(records: SplitRecords, limit: Optional[int]) -> SplitRecords:
    if limit is None:
        return records
    sl = slice(0, min(int(limit), len(records.labels)))
    return SplitRecords(
        images=records.images[sl],
        labels=records.labels[sl],
        sample_ids=records.sample_ids[sl],
        mask_paths=None if records.mask_paths is None else records.mask_paths[sl],
        masks=None if records.masks is None else records.masks[sl],
    )


def _load_bad_indices(path: Optional[Path]) -> set:
    target_path = path if (path is not None and path.exists()) else None
    if target_path is None:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for p in kaggle_input.rglob("bad_row_indices_drop345_mediapipe_failed.txt"):
                target_path = p
                print(f"[INFO] Auto-resolved bad_row_indices file: {target_path}")
                break
    if target_path is None or not target_path.exists():
        return set()
    with target_path.open("r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def _safe_load_npy(path_str: str, *, allow_missing: bool = False) -> np.ndarray:
    p = Path(path_str)
    if p.exists():
        return np.load(p).astype(np.float32)
    if allow_missing:
        return np.ones((6, 7, 7), dtype=np.float32)
    raise FileNotFoundError(f"Missing mask file: {p}")


def _verify_mask_paths(mask_paths: np.ndarray, split: str, *, allow_missing: bool) -> None:
    missing = [path for path in mask_paths if not Path(path).exists()]
    if not missing:
        print(f"[INFO] Verified {len(mask_paths)} mask files for {split}")
        return
    preview = "\n".join(str(path) for path in missing[:10])
    message = (
        f"Missing {len(missing)}/{len(mask_paths)} mask file(s) for split {split}. "
        f"First missing paths:\n{preview}"
    )
    if allow_missing:
        print(f"[WARNING] {message}\n[WARNING] Falling back to all-one masks because allow_missing_masks=true.")
        return
    raise FileNotFoundError(message)


def _resolve_split_csv_dir(data_dir: Path) -> Path:
    if data_dir.exists() and all((data_dir / f"{split}.csv").exists() for split in ("train", "val", "test")):
        return data_dir
    if data_dir.exists():
        candidates = sorted({p.parent for p in data_dir.rglob("train.csv")})
        for candidate in candidates:
            if all((candidate / f"{split}.csv").exists() for split in ("train", "val", "test")):
                print(f"[INFO] Resolved FER split CSV directory: {candidate}")
                return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = sorted({p.parent for p in kaggle_input.rglob("train.csv")})
        for candidate in candidates:
            if all((candidate / f"{split}.csv").exists() for split in ("train", "val", "test")):
                print(f"[INFO] Auto-resolved Kaggle FER split CSV directory: {candidate}")
                return candidate
    return data_dir


def _mask_coverage(split_mask_dir: Path, sample_ids: np.ndarray) -> int:
    if not split_mask_dir.exists():
        return -1
    return sum((split_mask_dir / f"{int(i):06d}.npy").exists() for i in sample_ids)


def _resolve_mask_split_dir(mask_root: Path, split: str, sample_ids: np.ndarray) -> Path:
    direct = mask_root / split
    candidates = [direct]
    if mask_root.exists():
        candidates.extend(sorted(p for p in mask_root.rglob(split) if p.is_dir() and p != direct))

    scored = [(candidate, _mask_coverage(candidate, sample_ids)) for candidate in candidates]
    best_dir, best_count = max(scored, key=lambda item: item[1])
    direct_count = _mask_coverage(direct, sample_ids)
    if best_dir != direct:
        print(
            f"[INFO] Resolved mask directory for {split}: {best_dir} "
            f"({best_count}/{len(sample_ids)} masks; direct had {direct_count}/{len(sample_ids)})"
        )
    return best_dir


def collect_split_records(
    data_dir,
    split: str,
    *,
    mask_dir=None,
    use_clean_filter: bool = False,
    bad_row_indices_path=None,
    mask_ablation: str = "none",
    mask_region_permutation: Optional[Iterable[int]] = None,
    predecode_pixels: bool = False,
    preload_masks: bool = False,
    allow_missing_masks: bool = False,
) -> SplitRecords:
    data_dir = _resolve_split_csv_dir(Path(data_dir))
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")
    if pd is not None:
        df = pd.read_csv(csv_path)
        label_col = "emotion" if "emotion" in df.columns else df.columns[0]
        pixel_col = "pixels" if "pixels" in df.columns else df.columns[1]
        labels = df[label_col].astype("int64").to_numpy()
        pixels = df[pixel_col].astype(str).to_numpy()
        sample_ids = np.arange(len(df), dtype=np.int64)
    else:
        import csv
        labels_list = []
        pixels_list = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_val = row.get("emotion") if "emotion" in row else list(row.values())[0]
                pixel_val = row.get("pixels") if "pixels" in row else list(row.values())[1]
                labels_list.append(int(label_val))
                pixels_list.append(str(pixel_val))
        labels = np.array(labels_list, dtype=np.int64)
        pixels = np.array(pixels_list, dtype=object)
        sample_ids = np.arange(len(labels), dtype=np.int64)
    if split == "train" and use_clean_filter:
        bad = _load_bad_indices(_resolve_path(bad_row_indices_path))
        if bad:
            keep = np.array([idx not in bad for idx in sample_ids], dtype=bool)
            labels, pixels, sample_ids = labels[keep], pixels[keep], sample_ids[keep]
    images = pixels.astype(object)
    if predecode_pixels:
        images = np.stack(
            [np.fromstring(pixel, sep=" ", dtype=np.float32).reshape(48, 48, 1) for pixel in pixels],
            axis=0,
        ).astype(np.uint8)
    mask_paths = None
    masks = None
    if mask_dir is not None:
        mask_root = Path(mask_dir)
        if not mask_root.is_absolute():
            mask_root = Path(__file__).resolve().parents[1] / mask_root
        split_mask_dir = _resolve_mask_split_dir(mask_root, split, sample_ids)
        if not split_mask_dir.exists():
            raise FileNotFoundError(f"Missing mask split directory: {split_mask_dir}")
        mask_paths = np.asarray([str(split_mask_dir / f"{int(i):06d}.npy") for i in sample_ids], dtype=str)
        _verify_mask_paths(mask_paths, split, allow_missing=allow_missing_masks)
        if preload_masks:
            masks = np.stack([_safe_load_npy(path, allow_missing=allow_missing_masks) for path in mask_paths], axis=0)
            mask_paths = None
    return SplitRecords(images, labels.astype(np.int64), sample_ids, mask_paths, masks)


def _decode_pixels(pixels: tf.Tensor, image_size: int, channels: int) -> tf.Tensor:
    if pixels.dtype == tf.string:
        values = tf.strings.to_number(tf.strings.split(tf.expand_dims(pixels, axis=0)).values, out_type=tf.float32)
        image = tf.reshape(values, [48, 48, 1])
    else:
        image = tf.reshape(tf.cast(pixels, tf.float32), [48, 48, 1])
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    if channels == 3:
        image = tf.image.grayscale_to_rgb(image)
    return image


def _normalize_image(image: tf.Tensor, channels: int) -> tf.Tensor:
    image = tf.cast(image, tf.float32) / 255.0
    if channels == 3:
        mean = tf.constant([0.485, 0.456, 0.406], tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], tf.float32)
    else:
        mean = tf.constant([0.5], tf.float32)
        std = tf.constant([0.5], tf.float32)
    return (image - mean) / std


def _load_mask_npy(mask_path: tf.Tensor, *, allow_missing: bool = False) -> tf.Tensor:
    def _reader(path_bytes):
        if hasattr(path_bytes, "numpy"):
            path_bytes = path_bytes.numpy()
        if isinstance(path_bytes, np.ndarray):
            path_bytes = path_bytes.item()
        path_str = path_bytes.decode("utf-8")
        return _safe_load_npy(path_str, allow_missing=allow_missing)
    mask = tf.py_function(_reader, [mask_path], Tout=tf.float32)
    mask.set_shape([6, 7, 7])
    return mask


def _resize_mask(mask: tf.Tensor, grid_size: int, method: str = "area") -> tf.Tensor:
    mask = tf.transpose(mask, [1, 2, 0])
    if mask.shape[0] != grid_size or mask.shape[1] != grid_size:
        mask = tf.image.resize(mask, [grid_size, grid_size], method=method)
    return tf.clip_by_value(mask, 0.0, 1.0)



def _target_mask_grid_size(cfg: Dict) -> int:
    model_cfg = cfg["model"]
    if bool(model_cfg.get("multi_scale_mgr", False)):
        return max(
            int(model_cfg.get("stage3_token_grid_size", 14)),
            int(model_cfg.get("stage4_token_grid_size", model_cfg.get("token_grid_size", 7))),
        )
    return int(model_cfg.get("token_grid_size", model_cfg.get("stage4_token_grid_size", 7)))


def _apply_mask_ablation(mask: tf.Tensor, ablation: str, mask_floor: float, permutation) -> tf.Tensor:
    if ablation == "uniform":
        mask = tf.ones_like(mask)
    elif ablation in ("shuffle_regions", "shuffled_mask"):
        perm = tf.constant(list(permutation or [4, 2, 0, 5, 1, 3]), dtype=tf.int32)
        mask = tf.gather(mask, perm, axis=-1)
    return tf.clip_by_value(mask, mask_floor, 1.0)



def _random_erasing(image: tf.Tensor, cfg: Dict) -> tf.Tensor:
    prob = float(cfg.get("random_erasing_prob", 0.0))
    if prob <= 0.0:
        return image
    draw = tf.random.uniform([])
    def erase():
        h, w, c = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
        area = tf.cast(h * w, tf.float32)
        erase_area = tf.random.uniform([], minval=float(cfg.get("random_erasing_area_min", 0.02)), maxval=float(cfg.get("random_erasing_area_max", 0.15))) * area
        side = tf.clip_by_value(tf.cast(tf.sqrt(erase_area), tf.int32), 1, tf.minimum(h, w))
        y = tf.random.uniform([], minval=0, maxval=tf.maximum(h - side + 1, 1), dtype=tf.int32)
        x = tf.random.uniform([], minval=0, maxval=tf.maximum(w - side + 1, 1), dtype=tf.int32)
        erase_shape = [side, side, c]
        value = str(cfg.get("random_erasing_value", "random")).lower()
        patch = (
            tf.random.normal(erase_shape, dtype=image.dtype)
            if value == "random"
            else tf.zeros(erase_shape, image.dtype)
        )
        keep = tf.ones_like(image)
        erase_mask = tf.pad(tf.zeros(erase_shape, image.dtype), [[y, h - y - side], [x, w - x - side], [0, 0]], constant_values=1.0)
        patch = tf.pad(patch, [[y, h - y - side], [x, w - x - side], [0, 0]], constant_values=0.0)
        return image * erase_mask + patch * (keep - erase_mask)
    return tf.cond(draw < prob, erase, lambda: image)


def _rotate_tensor(tensor: tf.Tensor, radians: tf.Tensor, interpolation: str = "BILINEAR") -> tf.Tensor:
    shape = tf.shape(tensor)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    cos_v = tf.cos(radians)
    sin_v = tf.sin(radians)
    transform = tf.stack([
        cos_v,
        sin_v,
        center_x - cos_v * center_x - sin_v * center_y,
        -sin_v,
        cos_v,
        center_y + sin_v * center_x - cos_v * center_y,
        0.0,
        0.0,
    ])
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tensor, axis=0),
        transforms=tf.expand_dims(transform, axis=0),
        output_shape=shape[:2],
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=tf.constant(0.0, dtype=tensor.dtype),
    )
    return tf.squeeze(rotated, axis=0)


def _augment_pair(image, mask, sample_id, aug_cfg, split: str):
    if split != "train":
        return image, mask
    if aug_cfg.get("horizontal_flip", True):
        flip = tf.random.uniform([]) < 0.5
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
        if mask is not None:
            mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
    degrees = float(aug_cfg.get("rotation_degrees", 0.0))
    if degrees > 0.0:
        angle = tf.random.uniform([], minval=-degrees, maxval=degrees)
        radians = angle * np.pi / 180.0
        image = _rotate_tensor(image, radians, interpolation="NEAREST")
        if mask is not None:
            mask = tf.clip_by_value(_rotate_tensor(mask, radians, interpolation="BILINEAR"), 0.0, 1.0)
    brightness_delta = float(aug_cfg.get("brightness_delta", 0.0))
    if brightness_delta > 0.0:
        brightness = tf.random.uniform(
            [],
            minval=max(0.0, 1.0 - brightness_delta),
            maxval=1.0 + brightness_delta,
        )
        image = image * brightness
    image = tf.image.random_contrast(image, lower=float(aug_cfg.get("contrast_lower", 1.0)), upper=float(aug_cfg.get("contrast_upper", 1.0)))
    image = tf.clip_by_value(image, 0.0, 255.0)
    gamma_prob = float(aug_cfg.get("gamma_prob", 0.0))
    if gamma_prob > 0.0:
        use_gamma = tf.random.uniform([]) < gamma_prob
        def gamma_aug():
            gamma = tf.random.uniform([], minval=float(aug_cfg.get("gamma_min", 0.5)), maxval=float(aug_cfg.get("gamma_max", 2.0)))
            return tf.image.adjust_gamma(tf.clip_by_value(image, 0.0, 255.0) / 255.0, gamma=gamma) * 255.0
        image = tf.cond(use_gamma, gamma_aug, lambda: image)
    return tf.clip_by_value(image, 0.0, 255.0), mask


def _parse_example(pixels, label, sample_id, mask_path, mask_tensor, *, cfg: Dict, split: str):
    image = _decode_pixels(pixels, int(cfg["data"]["image_size"]), int(cfg["data"]["channels"]))
    mask = None
    resize_method = str(cfg["model"].get("mgr_mask_resize_method", "area"))
    if mask_tensor is not None:
        mask_grid_size = _target_mask_grid_size(cfg)
        mask = _resize_mask(tf.cast(mask_tensor, tf.float32), mask_grid_size, method=resize_method)
    elif mask_path is not None:
        mask_grid_size = _target_mask_grid_size(cfg)
        mask = _resize_mask(
            _load_mask_npy(mask_path, allow_missing=bool(cfg["data"].get("allow_missing_masks", False))),
            mask_grid_size,
            method=resize_method,
        )
    if mask is not None:
        mask = _apply_mask_ablation(
            mask,
            cfg["data"].get("mask_ablation", "none"),
            float(cfg["model"].get("mask_floor", 0.05)),
            cfg["data"].get("mask_region_permutation"),
        )
    image, mask = _augment_pair(image, mask, sample_id, cfg["augmentation"], split)
    image = _normalize_image(image, int(cfg["data"]["channels"]))
    if split == "train":
        image = _random_erasing(image, cfg["augmentation"])
    features = {"image": image}
    if mask is not None:
        features["mask"] = mask
    return features, tf.cast(label, tf.int32)


def make_dataset(records: SplitRecords, cfg: Dict, *, split: str, training: bool, replicas: int) -> tf.data.Dataset:
    with tf.device("/CPU:0"):
        pixel_tensor = (
            tf.convert_to_tensor(records.images)
            if isinstance(records.images, np.ndarray) and records.images.dtype != object
            else tf.convert_to_tensor(records.images.astype(str))
        )
        tensors = {
            "pixels": pixel_tensor,
            "labels": tf.convert_to_tensor(records.labels),
            "sample_ids": tf.convert_to_tensor(records.sample_ids),
        }
        if records.mask_paths is not None:
            tensors["mask_paths"] = tf.convert_to_tensor(records.mask_paths.astype(str))
        if records.masks is not None:
            tensors["masks"] = tf.convert_to_tensor(records.masks)
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(int(cfg["data"].get("shuffle_buffer", 4096)), seed=int(cfg["seed"]["random_seed"]), reshuffle_each_iteration=True)
    options = tf.data.Options()
    runtime_cfg = cfg.get("runtime", {})
    options.experimental_deterministic = bool(runtime_cfg.get("tf_data_deterministic", True))
    private_threads = runtime_cfg.get("tf_data_private_threadpool_size")
    if private_threads:
        options.threading.private_threadpool_size = int(private_threads)
        options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)
    def mapper(item):
        return _parse_example(
            item["pixels"],
            item["labels"],
            item["sample_ids"],
            item["mask_paths"] if "mask_paths" in item else None,
            item["masks"] if "masks" in item else None,
            cfg=cfg,
            split=split,
        )
    parallel_calls = runtime_cfg.get("tf_data_num_parallel_calls")
    if parallel_calls in (None, "", 0):
        parallel_calls = tf.data.AUTOTUNE
    else:
        parallel_calls = int(parallel_calls)
    ds = ds.map(mapper, num_parallel_calls=parallel_calls, deterministic=bool(runtime_cfg.get("tf_data_deterministic", True)))
    if cfg["data"].get("cache", False) and not training:
        ds = ds.cache()
    ds = ds.batch(int(cfg["runtime"]["batch_size_per_gpu"]) * int(replicas), drop_remainder=training)
    prefetch_buffer = runtime_cfg.get("prefetch_buffer")
    if prefetch_buffer in (None, "", 0):
        prefetch_buffer = tf.data.AUTOTUNE
    else:
        prefetch_buffer = int(prefetch_buffer)
    return ds.prefetch(prefetch_buffer)


def build_datasets(cfg: Dict, replicas: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    data_dir = _resolve_path(cfg["data"]["data_path"])
    mask_dir = _resolve_path(cfg["data"].get("mask_dir"))
    records = {
        split: collect_split_records(
            data_dir,
            split,
            mask_dir=mask_dir,
            use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
            bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
            mask_ablation=cfg["data"].get("mask_ablation", "none"),
            mask_region_permutation=cfg["data"].get("mask_region_permutation"),
            predecode_pixels=bool(cfg["data"].get("predecode_pixels", False)),
            preload_masks=bool(cfg["data"].get("preload_masks", False)),
            allow_missing_masks=bool(cfg["data"].get("allow_missing_masks", False)),
        )
        for split in ("train", "val", "test")
    }
    records["train"] = _limit_records(records["train"], cfg["data"].get("max_train_samples"))
    records["val"] = _limit_records(records["val"], cfg["data"].get("max_val_samples"))
    records["test"] = _limit_records(records["test"], cfg["data"].get("max_test_samples"))

    if bool(cfg["data"].get("use_synthetic_diffusion", False)):
        syn_meta_path = _resolve_path(cfg["data"].get("synthetic_metadata_json"))
        if syn_meta_path and syn_meta_path.exists():
            import json
            with open(syn_meta_path, "r", encoding="utf-8") as f:
                syn_meta = json.load(f)
            syn_imgs, syn_lbls, syn_sids = [], [], []
            base_id = len(records["train"].sample_ids) + 900000
            for idx, entry in enumerate(syn_meta):
                img_p = _resolve_path(entry["image_path"])
                if img_p and img_p.exists():
                    img_pil = tf.keras.utils.load_img(img_p, color_mode="grayscale", target_size=(48, 48))
                    img_arr = np.array(img_pil, dtype=np.uint8).reshape(48, 48, 1)
                    syn_imgs.append(img_arr)
                    syn_lbls.append(EMOTION_NAMES.index(entry["target_class"]))
                    syn_sids.append(base_id + idx)

            if syn_imgs:
                syn_imgs_arr = np.stack(syn_imgs, axis=0)
                syn_lbls_arr = np.array(syn_lbls, dtype=np.int64)
                syn_sids_arr = np.array(syn_sids, dtype=np.int64)

                records["train"] = SplitRecords(
                    images=np.concatenate([records["train"].images, syn_imgs_arr], axis=0),
                    labels=np.concatenate([records["train"].labels, syn_lbls_arr], axis=0),
                    sample_ids=np.concatenate([records["train"].sample_ids, syn_sids_arr], axis=0),
                    mask_paths=None if records["train"].mask_paths is None else np.concatenate([records["train"].mask_paths, np.array([""] * len(syn_imgs_arr))], axis=0),
                    masks=records["train"].masks,
                )
                print(f"[Synthetic Diffusion] Successfully injected {len(syn_imgs)} accepted synthetic samples into TRAIN dataset.")

    return (
        make_dataset(records["train"], cfg, split="train", training=True, replicas=replicas),
        make_dataset(records["val"], cfg, split="val", training=False, replicas=replicas),
        make_dataset(records["test"], cfg, split="test", training=False, replicas=replicas),
    )

