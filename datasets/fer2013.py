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
    bboxes: Optional[np.ndarray] = None


def _resolve_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path in (None, ""):
        return None
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    shm_p = Path("/dev/shm") / p
    if shm_p.exists():
        return shm_p
    rel_p = Path(__file__).resolve().parents[1] / p
    if rel_p.exists():
        return rel_p
    return shm_p if shm_p.parent.exists() else rel_p


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
        bboxes=None if records.bboxes is None else records.bboxes[sl],
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


def _collect_records_from_folder(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_dir = data_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Neither {split}.csv nor directory {split_dir} exists in {data_dir}")

    emotion_map = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6,
        "happiness": 3, "sadness": 4, "anger": 0,
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6
    }
    rafdb_raw_map = {"1": 5, "2": 2, "3": 1, "4": 3, "5": 4, "6": 0, "7": 6}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".npy"}

    img_paths, labels = [], []
    subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [split_dir]

    subdir_prefixes = [sdir.name.lower().strip().split("_")[0] for sdir in subdirs]
    is_rafdb_1based_folders = (len(subdirs) == 7 and "7" in subdir_prefixes)

    for sdir in sorted(subdirs):
        folder_name = sdir.name.lower().strip()
        prefix = folder_name.split("_")[0]
        label_idx = None

        if is_rafdb_1based_folders and prefix in rafdb_raw_map:
            label_idx = rafdb_raw_map[prefix]
        elif prefix in emotion_map:
            label_idx = emotion_map[prefix]
        elif folder_name in emotion_map:
            label_idx = emotion_map[folder_name]

        for img_p in sorted(sdir.rglob("*")):
            if img_p.is_file() and img_p.suffix.lower() in valid_exts:
                curr_label = label_idx
                if curr_label is None:
                    p_name = img_p.parent.name.lower().split("_")[0]
                    if is_rafdb_1based_folders and p_name in rafdb_raw_map:
                        curr_label = rafdb_raw_map[p_name]
                    else:
                        curr_label = emotion_map.get(p_name, 0)
                try:
                    rel_p = str(img_p.relative_to(Path(__file__).resolve().parents[1]))
                except ValueError:
                    rel_p = str(img_p)
                img_paths.append(rel_p)
                labels.append(curr_label)

    if not img_paths:
        raise FileNotFoundError(f"No image files found in directory {split_dir}")

    csv_path = data_dir / f"{split}.csv"
    try:
        if pd is not None:
            df_gen = pd.DataFrame({"image_path": img_paths, "label": labels})
            df_gen.to_csv(csv_path, index=False)
            print(f"[INFO] Auto-generated and saved {csv_path} ({len(df_gen)} samples; label range [{min(labels)}..{max(labels)}]).")
    except Exception as e:
        print(f"[WARNING] Could not save auto-generated {csv_path}: {e}")

    return np.array(img_paths, dtype=object), np.array(labels, dtype=np.int64), np.arange(len(img_paths), dtype=np.int64)


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
    rafdb_raw_map_int = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}

    if csv_path.exists():
        if pd is not None:
            df = pd.read_csv(csv_path)
            label_col = next((c for c in ("emotion", "label", "target", "class", "y") if c in df.columns), df.columns[0])
            pixel_col = next((c for c in ("pixels", "image_path", "filepath", "path", "image", "file") if c in df.columns), df.columns[1])
            labels = df[label_col].astype("int64").to_numpy()
            pixels = df[pixel_col].astype(str).to_numpy()
            sample_ids = np.arange(len(df), dtype=np.int64)

            # Check if CSV has raw 1-based RAF-DB labels [1..7]
            if labels.min() == 1 and labels.max() == 7:
                print(f"[INFO] Auto-remapping RAF-DB 1-based CSV labels [1..7] -> 0-based [0..6] for {csv_path}")
                labels = np.array([rafdb_raw_map_int.get(int(l), int(l)) for l in labels], dtype=np.int64)
                df[label_col] = labels
                df.to_csv(csv_path, index=False)
            elif labels.min() == 1 and labels.max() == 6 and 0 not in labels and (data_dir / split).is_dir():
                print(f"[WARNING] Detected legacy mis-mapped CSV with label range [1..6] for {csv_path}. Regenerating clean split from directory {data_dir / split}...")
                pixels, labels, sample_ids = _collect_records_from_folder(data_dir, split)
        else:
            import csv
            labels_list = []
            pixels_list = []
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                lbl_key = next((c for c in ("emotion", "label", "target", "class", "y") if c in fieldnames), fieldnames[0])
                pix_key = next((c for c in ("pixels", "image_path", "filepath", "path", "image", "file") if c in fieldnames), fieldnames[1])
                for row in reader:
                    labels_list.append(int(row[lbl_key]))
                    pixels_list.append(str(row[pix_key]))
            labels = np.array(labels_list, dtype=np.int64)
            if labels.min() == 1 and labels.max() == 7:
                labels = np.array([rafdb_raw_map_int.get(int(l), int(l)) for l in labels], dtype=np.int64)
            pixels = np.array(pixels_list, dtype=object)
            sample_ids = np.arange(len(labels), dtype=np.int64)
    elif (data_dir / split).is_dir():
        pixels, labels, sample_ids = _collect_records_from_folder(data_dir, split)
    else:
        raise FileNotFoundError(f"Missing split CSV or directory for split '{split}' in {data_dir}")
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
    target_h, target_w = int(image_size), int(image_size)

    def _read_image_or_pixels(p_tensor):
        p_str = p_tensor.numpy().decode("utf-8") if hasattr(p_tensor, "numpy") else str(p_tensor)
        # Avoid OSError: [Errno 36] File name too long when p_str is a pixel string
        if len(p_str) <= 255 and not (" " in p_str.strip() and p_str.strip().count(" ") > 3):
            p_path = Path(p_str)
            if not p_path.is_absolute():
                p_path = Path(__file__).resolve().parents[1] / p_str
            try:
                if p_path.exists() and p_path.is_file():
                    try:
                        from PIL import Image
                        with Image.open(p_path) as pil_img:
                            if channels == 3 and pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                            elif channels == 1 and pil_img.mode != "L":
                                pil_img = pil_img.convert("L")
                            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                            arr = np.array(pil_img, dtype=np.float32)
                            if arr.ndim == 2:
                                arr = np.expand_dims(arr, axis=-1)
                            return arr
                    except Exception:
                        pass
                    img_raw = tf.io.read_file(str(p_path))
                    img = tf.io.decode_image(img_raw, channels=channels, expand_animations=False)
                    img = tf.cast(img, tf.float32)
                    if img.shape[-1] == 1 and channels == 3:
                        img = tf.image.grayscale_to_rgb(img)
                    return tf.image.resize(img, [target_h, target_w], method="bilinear")
            except OSError:
                pass
        
        # Fallback to space-separated pixel string
        vals = np.fromstring(p_str, sep=" ", dtype=np.float32)
        if len(vals) > 0:
            side = int(np.round(np.sqrt(len(vals))))
            img = vals.reshape(side, side, 1)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, [target_h, target_w], method="bilinear")
            if channels == 3:
                img = tf.image.grayscale_to_rgb(img)
            return img
        raise ValueError(f"Unable to parse image path or pixel string: {p_str[:50]}")

    if pixels.dtype == tf.string:
        image = tf.py_function(func=_read_image_or_pixels, inp=[pixels], Tout=tf.float32)
        image.set_shape([target_h, target_w, channels])
        return image
    else:
        image = tf.reshape(tf.cast(pixels, tf.float32), [48, 48, 1])
        image = tf.image.resize(image, [target_h, target_w], method="bilinear")
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


def _affine_transform_tensor(
    tensor: tf.Tensor,
    radians: tf.Tensor,
    shift_h: tf.Tensor,
    shift_w: tf.Tensor,
    zoom: tf.Tensor,
    interpolation: str = "BILINEAR",
    fill_mode: str = "REFLECT",
) -> tf.Tensor:
    shape = tf.shape(tensor)
    orig_dtype = tensor.dtype
    if orig_dtype != tf.float32:
        tensor = tf.cast(tensor, tf.float32)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    t_x = shift_w * width
    t_y = shift_h * height
    k = tf.math.divide_no_nan(1.0, zoom)
    cos_v = tf.cos(radians)
    sin_v = tf.sin(radians)
    a0 = k * cos_v
    a1 = k * sin_v
    a2 = center_x - k * (cos_v * (center_x + t_x) + sin_v * (center_y + t_y))
    a3 = -k * sin_v
    a4 = k * cos_v
    a5 = center_y - k * (-sin_v * (center_x + t_x) + cos_v * (center_y + t_y))
    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tensor, axis=0),
        transforms=tf.expand_dims(transform, axis=0),
        output_shape=shape[:2],
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=tf.constant(0.0, dtype=tf.float32),
    )
    transformed = tf.squeeze(transformed, axis=0)
    if transformed.dtype != orig_dtype:
        transformed = tf.cast(transformed, orig_dtype)
    return transformed


def _rotate_tensor(tensor: tf.Tensor, radians: tf.Tensor, interpolation: str = "BILINEAR", fill_mode: str = "CONSTANT") -> tf.Tensor:
    zero = tf.constant(0.0, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    return _affine_transform_tensor(tensor, radians, zero, zero, one, interpolation=interpolation, fill_mode=fill_mode)


def _augment_minority_img(image: tf.Tensor) -> tf.Tensor:
    flip = tf.random.uniform([]) < 0.50
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)

    radians = tf.random.uniform([], minval=-10.0, maxval=10.0) * (np.pi / 180.0)
    shift_h = tf.random.uniform([], minval=-0.05, maxval=0.05)
    shift_w = tf.random.uniform([], minval=-0.05, maxval=0.05)
    zoom = tf.random.uniform([], minval=0.90, maxval=1.10)
    image = _affine_transform_tensor(image, radians, shift_h, shift_w, zoom, interpolation="BILINEAR", fill_mode="REFLECT")

    brightness = tf.random.uniform([], minval=0.80, maxval=1.20)
    image = image * brightness
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    return tf.clip_by_value(image, 0.0, 255.0)


def _augment_standard_img(image: tf.Tensor, aug_cfg: Dict) -> tf.Tensor:
    do_hflip = bool(aug_cfg.get("horizontal_flip", True))
    if do_hflip:
        flip = tf.random.uniform([]) < 0.50
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)

    degrees = float(aug_cfg.get("rotation_degrees", 0.0))
    trans_h = float(aug_cfg.get("translation_height", 0.0))
    trans_w = float(aug_cfg.get("translation_width", 0.0))
    zoom_min = float(aug_cfg.get("zoom_min", 1.0))
    zoom_max = float(aug_cfg.get("zoom_max", 1.0))
    has_affine = (degrees > 0.0 or trans_h > 0.0 or trans_w > 0.0 or zoom_min != 1.0 or zoom_max != 1.0)
    if has_affine:
        fill_mode = str(aug_cfg.get("fill_mode", "REFLECT" if (trans_h > 0 or trans_w > 0 or zoom_min != 1.0 or zoom_max != 1.0) else "CONSTANT")).upper()
        radians = (
            tf.random.uniform([], minval=-degrees, maxval=degrees) * (np.pi / 180.0)
            if degrees > 0.0 else tf.constant(0.0, dtype=tf.float32)
        )
        shift_h = (
            tf.random.uniform([], minval=-trans_h, maxval=trans_h)
            if trans_h > 0.0 else tf.constant(0.0, dtype=tf.float32)
        )
        shift_w = (
            tf.random.uniform([], minval=-trans_w, maxval=trans_w)
            if trans_w > 0.0 else tf.constant(0.0, dtype=tf.float32)
        )
        zoom = (
            tf.random.uniform([], minval=zoom_min, maxval=zoom_max)
            if zoom_max > zoom_min else tf.constant(1.0, dtype=tf.float32)
        )
        image = _affine_transform_tensor(image, radians, shift_h, shift_w, zoom, interpolation="BILINEAR", fill_mode=fill_mode)

    brightness_delta = float(aug_cfg.get("brightness_delta", 0.0))
    if brightness_delta > 0.0:
        brightness = tf.random.uniform(
            [],
            minval=max(0.0, 1.0 - brightness_delta),
            maxval=1.0 + brightness_delta,
        )
        image = image * brightness

    contrast_lower = float(aug_cfg.get("contrast_lower", 1.0))
    contrast_upper = float(aug_cfg.get("contrast_upper", 1.0))
    if contrast_upper > contrast_lower:
        image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
    image = tf.clip_by_value(image, 0.0, 255.0)

    gamma_prob = float(aug_cfg.get("gamma_prob", 0.0))
    if gamma_prob > 0.0:
        use_gamma = tf.random.uniform([]) < gamma_prob
        def gamma_aug():
            gamma = tf.random.uniform([], minval=float(aug_cfg.get("gamma_min", 0.5)), maxval=float(aug_cfg.get("gamma_max", 2.0)))
            return tf.image.adjust_gamma(tf.clip_by_value(image, 0.0, 255.0) / 255.0, gamma=gamma) * 255.0
        image = tf.cond(use_gamma, gamma_aug, lambda: image)

    return tf.clip_by_value(image, 0.0, 255.0)


def _augment_pair(image, mask, sample_id, aug_cfg, split: str, is_minority=False):
    if split != "train":
        return image, mask

    if tf.is_tensor(is_minority):
        image = tf.cond(
            is_minority,
            lambda: _augment_minority_img(image),
            lambda: _augment_standard_img(image, aug_cfg),
        )
    else:
        if bool(is_minority):
            image = _augment_minority_img(image)
        else:
            image = _augment_standard_img(image, aug_cfg)

    if mask is not None:
        do_hflip = bool(aug_cfg.get("horizontal_flip", True))
        if do_hflip:
            flip = tf.random.uniform([]) < 0.50
            mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    return image, mask


def _parse_example(pixels, label, sample_id, mask_path, mask_tensor, *, cfg: Dict, split: str, is_minority=False):
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
    image, mask = _augment_pair(image, mask, sample_id, cfg["augmentation"], split, is_minority=is_minority)
    image = _normalize_image(image, int(cfg["data"]["channels"]))
    if split == "train":
        if tf.is_tensor(is_minority):
            image = tf.cond(
                is_minority,
                lambda: tf.cond(
                    tf.random.uniform([]) < 0.25,
                    lambda: _random_erasing(image, cfg["augmentation"]),
                    lambda: image,
                ),
                lambda: _random_erasing(image, cfg["augmentation"]),
            )
        else:
            image = _random_erasing(image, cfg["augmentation"])
    features = {"image": image}
    if mask is not None:
        features["mask"] = mask
    return features, tf.cast(label, tf.int32)


def _make_tensor_dataset(tensors: Dict[str, tf.Tensor]) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(tensors)


def _make_class_balanced_tensor_dataset(
    tensors: Dict[str, tf.Tensor],
    labels: np.ndarray,
    cfg: Dict,
    split: str,
) -> tf.data.Dataset:
    num_classes = int(cfg["data"].get("num_classes", len(EMOTION_NAMES)))
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=num_classes)[:num_classes]
    active_classes = [idx for idx, count in enumerate(counts) if count > 0]
    if not active_classes:
        raise ValueError(f"Cannot build class-balanced sampler for empty {split} split.")

    configured_weights = cfg["data"].get("class_sampling_weights")
    sampling_strategy = str(cfg["data"].get("sampling_strategy", "")).lower()
    if configured_weights is not None:
        raw_weights = np.asarray(configured_weights, dtype=np.float64)
        if raw_weights.shape[0] != num_classes:
            raise ValueError(
                f"class_sampling_weights must have {num_classes} entries, got {raw_weights.shape[0]}."
            )
        weights = raw_weights[active_classes]
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("class_sampling_weights must sum to a positive value for present classes.")
        weights = weights / weight_sum
    elif sampling_strategy in {"square_root", "sqrt", "power"}:
        gamma = float(cfg["data"].get("sampling_gamma", 0.5))
        active_counts = counts[active_classes].astype(np.float64)
        raw_weights = np.power(active_counts, gamma)
        weights = raw_weights / float(raw_weights.sum())
    else:
        weights = np.ones(len(active_classes), dtype=np.float64) / float(len(active_classes))

    class_datasets = []
    for class_id in active_classes:
        indices = np.flatnonzero(labels_arr == class_id).astype(np.int64)
        class_tensors = {key: tf.gather(value, indices) for key, value in tensors.items()}
        class_datasets.append(_make_tensor_dataset(class_tensors).repeat())

    print(
        f"[INFO] Class-aware ({sampling_strategy or 'uniform_balanced'}) sampling enabled for "
        f"{split}: counts={counts.tolist()} weights={weights.round(6).tolist()} "
        f"epoch_samples={len(labels_arr)}",
        flush=True,
    )
    return tf.data.experimental.sample_from_datasets(
        class_datasets,
        weights=weights.tolist(),
        seed=int(cfg["seed"]["random_seed"]),
    ).take(len(labels_arr))

_OVERSAMPLE_STATS: Optional[Dict[str, Any]] = None


def get_oversample_stats() -> Optional[Dict[str, Any]]:
    global _OVERSAMPLE_STATS
    return _OVERSAMPLE_STATS


def _make_minority_oversampled_dataset(
    tensors: Dict[str, tf.Tensor],
    records: SplitRecords,
    cfg: Dict,
    split: str,
) -> tf.data.Dataset:
    global _OVERSAMPLE_STATS
    num_classes = int(cfg["data"].get("num_classes", len(EMOTION_NAMES)))
    labels_arr = np.asarray(records.labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=num_classes)[:num_classes]
    target_minority_count = int(cfg["data"].get("target_minority_count", 1200))
    seed = int(cfg["seed"].get("random_seed", 42))

    oversampled_counts = np.zeros(num_classes, dtype=np.int64)
    extra_indices_list = []

    for c in range(num_classes):
        if counts[c] < target_minority_count:
            shortfall = target_minority_count - counts[c]
            oversampled_counts[c] = shortfall
            c_indices = np.flatnonzero(labels_arr == c)
            if len(c_indices) > 0 and shortfall > 0:
                rng = np.random.default_rng(seed + c * 37)
                chosen = rng.choice(c_indices, size=shortfall, replace=True)
                extra_indices_list.append(chosen)

    effective_counts = counts + oversampled_counts
    total_effective = int(effective_counts.sum())
    distribution = (effective_counts / total_effective).round(4).tolist()

    _OVERSAMPLE_STATS = {
        "original_class_counts": counts.tolist(),
        "effective_class_counts": effective_counts.tolist(),
        "oversampled_counts": oversampled_counts.tolist(),
        "effective_class_distribution": distribution,
        "total_samples": total_effective,
    }

    print(
        f"[INFO] Class-Aware Minority Oversampling enabled for {split}:\n"
        f"  Original counts:    {counts.tolist()} (total={len(labels_arr)})\n"
        f"  Oversampled counts: {oversampled_counts.tolist()} (extra={int(oversampled_counts.sum())})\n"
        f"  Effective counts:   {effective_counts.tolist()} (total={total_effective})\n"
        f"  Class distribution: {distribution}",
        flush=True,
    )

    if extra_indices_list:
        all_extra = np.concatenate(extra_indices_list, axis=0)
        orig_minority_flag = np.array([counts[l] < target_minority_count for l in labels_arr], dtype=bool)
        extra_minority_flag = np.ones(len(all_extra), dtype=bool)
        full_is_minority = np.concatenate([orig_minority_flag, extra_minority_flag], axis=0)

        combined_tensors = {}
        for key, tensor in tensors.items():
            orig_t = tensor
            extra_t = tf.gather(tensor, all_extra)
            combined_tensors[key] = tf.concat([orig_t, extra_t], axis=0)
        combined_tensors["is_minority_aug"] = tf.convert_to_tensor(full_is_minority, dtype=tf.bool)
    else:
        combined_tensors = dict(tensors)
        combined_tensors["is_minority_aug"] = tf.convert_to_tensor(
            np.array([counts[l] < target_minority_count for l in labels_arr], dtype=bool), dtype=tf.bool
        )

    ds = tf.data.Dataset.from_tensor_slices(combined_tensors)
    shuffle_buffer = int(cfg["data"].get("shuffle_buffer", max(4096, total_effective)))
    return ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)


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
        if training and float(cfg.get("training", {}).get("lambda_vlm_kd", 0.0)) > 0:
            from utils.vlm_teacher_cache import load_teacher_logits
            tensors["teacher_logits"] = tf.convert_to_tensor(
                load_teacher_logits(cfg, records, split), dtype=tf.float32)
    sampling_strategy = str(cfg["data"].get("sampling_strategy", "")).lower()
    is_minority_oversample = training and sampling_strategy in {
        "minority_oversample",
        "class_aware_minority_oversample",
        "v8_minority_oversample",
        "class_aware_oversample",
    }
    use_class_balanced = training and sampling_strategy in {
        "class_balanced",
        "class-balanced",
        "balanced",
        "balanced_classes",
        "square_root",
        "sqrt",
        "power",
    }
    if is_minority_oversample:
        ds = _make_minority_oversampled_dataset(tensors, records, cfg, split)
    elif use_class_balanced:
        ds = _make_class_balanced_tensor_dataset(tensors, records.labels, cfg, split)
    else:
        ds = _make_tensor_dataset(tensors)
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
        features, label = _parse_example(
            item["pixels"],
            item["labels"],
            item["sample_ids"],
            item["mask_paths"] if "mask_paths" in item else None,
            item["masks"] if "masks" in item else None,
            cfg=cfg,
            split=split,
            is_minority=item.get("is_minority_aug", False),
        )
        if "teacher_logits" in item:
            features["teacher_logits"] = item["teacher_logits"]
        return features, label
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
    dataset_type = str(cfg.get("data", {}).get("dataset_type", "")).lower()
    if dataset_type in ("expw", "expw_gdrive"):
        from .expw import build_expw_datasets
        return build_expw_datasets(cfg, replicas)
    if dataset_type in ("ferplus", "ferplus_official", "ferplus_majority8"):
        from .ferplus import build_ferplus_datasets
        return build_ferplus_datasets(cfg, replicas)
    if dataset_type in ("affectnet", "affectnet7") or ("train_csv" in cfg.get("data", {}) and "expw" not in str(cfg.get("data", {}).get("train_csv", "")).lower()):
        from .affectnet import build_affectnet_datasets
        return build_affectnet_datasets(cfg, replicas)
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

    if bool(cfg.get("data", {}).get("full_train", False)):
        train_rec = records["train"]
        val_rec = records.get("val")
        if val_rec is not None:
            comb_images = np.concatenate([train_rec.images, val_rec.images], axis=0)
            comb_labels = np.concatenate([train_rec.labels, val_rec.labels], axis=0)
            comb_sids = np.arange(len(comb_images), dtype=np.int64)
            comb_mpaths = None
            if train_rec.mask_paths is not None and val_rec.mask_paths is not None:
                comb_mpaths = np.concatenate([train_rec.mask_paths, val_rec.mask_paths], axis=0)
            comb_masks = None
            if train_rec.masks is not None and val_rec.masks is not None:
                comb_masks = np.concatenate([train_rec.masks, val_rec.masks], axis=0)
            records["train"] = SplitRecords(comb_images, comb_labels, comb_sids, comb_mpaths, comb_masks)
            records["val"] = None
            print(
                f"[FULL_TRAIN] Merged train ({len(train_rec.images)}) + val ({len(val_rec.images)}) "
                f"-> Total Full Train: {len(records['train'].images)} samples (Expected: 12271). "
                f"Validation split disabled (val_ds=None). Test set untouched ({len(records['test'].images)} samples)."
            )

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

    if (cfg.get("training", {}).get("weighted_ce", {}).get("enabled", False)
            or cfg.get("training", {}).get("loss") == "weighted_cross_entropy"):
        from utils.ce_class_weights import configure_training_ce_weights
        configure_training_ce_weights(cfg, records["train"].labels)

    if cfg.get("training", {}).get("logit_adjustment", {}).get("enabled", False):
        from utils.logit_adjustment import configure_training_logit_adjustment
        configure_training_logit_adjustment(cfg, records["train"].labels)

    return (
        make_dataset(records["train"], cfg, split="train", training=True, replicas=replicas),
        make_dataset(records["val"], cfg, split="val", training=False, replicas=replicas) if records.get("val") is not None else None,
        make_dataset(records["test"], cfg, split="test", training=False, replicas=replicas),
    )
