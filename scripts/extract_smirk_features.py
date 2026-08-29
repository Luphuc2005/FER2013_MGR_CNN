from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import EMOTION_NAMES, collect_split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen official SMIRK features for FER2013 splits.")
    parser.add_argument("--config", type=str, default="config_smirk_only.yaml")
    parser.add_argument("--smirk-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-subdir", type=str, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--on-crop-failure", choices=("resize", "skip", "error"), default=None)
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def crop_face(frame: np.ndarray, landmarks: np.ndarray, scale: float = 1.0, image_size: int = 224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, dst_pts)


def import_smirk(smirk_root: Path):
    if not smirk_root.exists():
        raise FileNotFoundError(
            f"SMIRK root not found: {smirk_root}. Clone https://github.com/georgeretsi/smirk and run quick_install.sh first."
        )
    sys.path.insert(0, str(smirk_root))
    with pushd(smirk_root):
        import src.smirk_encoder as smirk_encoder_module
        from utils.mediapipe_utils import run_mediapipe
    return smirk_encoder_module, run_mediapipe


def load_frozen_encoder(smirk_encoder_module, checkpoint_path: Path, device: torch.device, strict: bool, init_timm_pretrained: bool = False):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SMIRK checkpoint not found: {checkpoint_path}. Expected the official quick_install.sh checkpoint SMIRK_em1.pt."
        )
    if not init_timm_pretrained:
        original_create_backbone = smirk_encoder_module.create_backbone

        def create_backbone_without_pretrained(backbone_name, pretrained=True):
            return original_create_backbone(backbone_name, pretrained=False)

        smirk_encoder_module.create_backbone = create_backbone_without_pretrained
    encoder = smirk_encoder_module.SmirkEncoder().to(device)
    raw = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")
    state = {}
    for key, value in raw.items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("smirk_encoder."):
            state[clean_key.replace("smirk_encoder.", "", 1)] = value
    if not state:
        state = {key.replace("module.", ""): value for key, value in raw.items()}
    missing, unexpected = encoder.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Strict SMIRK load failed: missing={missing}, unexpected={unexpected}")
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()
    print(
        f"SMIRK_LOAD_OK checkpoint={checkpoint_path} strict={strict} "
        f"trainable_params={sum(p.requires_grad for p in encoder.parameters())}",
        flush=True,
    )
    return encoder


def pixels_to_gray(pixel_entry) -> np.ndarray:
    if isinstance(pixel_entry, str):
        arr = np.fromstring(pixel_entry, sep=" ", dtype=np.uint8)
        return arr.reshape(48, 48)
    arr = np.asarray(pixel_entry)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8).reshape(48, 48)


def prepare_smirk_image(
    pixel_entry,
    *,
    run_mediapipe,
    use_crop: bool,
    crop_scale: float,
    image_size: int,
    mediapipe_input_size: int,
    on_crop_failure: str,
) -> Tuple[Optional[torch.Tensor], bool]:
    gray = pixels_to_gray(pixel_entry)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mediapipe_input_size and mediapipe_input_size != 48:
        interp = cv2.INTER_CUBIC if mediapipe_input_size > 48 else cv2.INTER_AREA
        bgr_for_smirk = cv2.resize(bgr, (mediapipe_input_size, mediapipe_input_size), interpolation=interp)
    else:
        bgr_for_smirk = bgr

    crop_success = False
    if use_crop:
        landmarks = run_mediapipe(bgr_for_smirk)
        if landmarks is not None:
            tform = crop_face(bgr_for_smirk, landmarks[..., :2], scale=crop_scale, image_size=image_size)
            bgr_for_smirk = warp(
                bgr_for_smirk,
                tform.inverse,
                output_shape=(image_size, image_size),
                preserve_range=True,
            ).astype(np.uint8)
            crop_success = True
        elif on_crop_failure == "skip":
            return None, False
        elif on_crop_failure == "error":
            raise RuntimeError("SMIRK crop requested but Mediapipe did not detect a face.")

    rgb = cv2.cvtColor(bgr_for_smirk, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (image_size, image_size):
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return tensor, crop_success


def join_feature_blocks(outputs: Dict[str, torch.Tensor], feature_keys: Sequence[str]) -> np.ndarray:
    blocks = []
    for key in feature_keys:
        if key not in outputs:
            raise KeyError(f"SMIRK output does not contain requested feature key: {key}")
        blocks.append(outputs[key].detach().float().cpu().numpy())
    return np.concatenate(blocks, axis=1).astype(np.float32)


def batched(iterable: Sequence[torch.Tensor], batch_size: int) -> Iterable[Sequence[torch.Tensor]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def output_path_for(feature_dir: Path, split: str) -> Path:
    return feature_dir / f"{split}_smirk_features.npz"


def extract_split(
    split: str,
    records,
    *,
    encoder,
    run_mediapipe,
    device: torch.device,
    batch_size: int,
    out_path: Path,
    cfg: Dict,
    args: argparse.Namespace,
) -> None:
    if out_path.exists() and not args.force:
        cached = np.load(out_path)
        print(
            f"SMIRK_FEATURE_CACHE_EXISTS split={split} path={out_path} "
            f"features_shape={cached['features'].shape}",
            flush=True,
        )
        return

    smirk_cfg = cfg.get("smirk", {})
    use_crop = bool(smirk_cfg.get("crop", True)) and not args.no_crop
    on_crop_failure = args.on_crop_failure or str(smirk_cfg.get("on_crop_failure", "resize"))
    crop_scale = float(smirk_cfg.get("crop_scale", 1.4))
    image_size = int(smirk_cfg.get("image_size", 224))
    mediapipe_input_size = int(smirk_cfg.get("mediapipe_input_size", image_size))
    feature_keys = list(smirk_cfg.get("feature_keys", ["expression_params", "eyelid_params", "jaw_params"]))

    tensors: List[torch.Tensor] = []
    labels: List[int] = []
    sample_ids: List[int] = []
    crop_success_flags: List[bool] = []
    skipped = 0
    total = len(records.labels)
    limit = args.max_samples_per_split
    if limit is not None:
        total = min(total, int(limit))

    for i in tqdm(range(total), desc=f"preprocess {split}", dynamic_ncols=True):
        tensor, crop_success = prepare_smirk_image(
            records.images[i],
            run_mediapipe=run_mediapipe,
            use_crop=use_crop,
            crop_scale=crop_scale,
            image_size=image_size,
            mediapipe_input_size=mediapipe_input_size,
            on_crop_failure=on_crop_failure,
        )
        if tensor is None:
            skipped += 1
            continue
        tensors.append(tensor)
        labels.append(int(records.labels[i]))
        sample_ids.append(int(records.sample_ids[i]))
        crop_success_flags.append(bool(crop_success))

    if not tensors:
        raise RuntimeError(f"No samples prepared for split={split}; skipped={skipped}.")

    features_blocks = []
    expression_blocks = []
    eyelid_blocks = []
    jaw_blocks = []
    with torch.no_grad():
        for tensor_batch in tqdm(list(batched(tensors, batch_size)), desc=f"SMIRK {split}", dynamic_ncols=True):
            images = torch.stack(list(tensor_batch), dim=0).to(device, non_blocking=True)
            outputs = encoder(images)
            features_blocks.append(join_feature_blocks(outputs, feature_keys))
            expression_blocks.append(outputs["expression_params"].detach().float().cpu().numpy())
            eyelid_blocks.append(outputs["eyelid_params"].detach().float().cpu().numpy())
            jaw_blocks.append(outputs["jaw_params"].detach().float().cpu().numpy())

    features = np.concatenate(features_blocks, axis=0).astype(np.float32)
    expression = np.concatenate(expression_blocks, axis=0).astype(np.float32)
    eyelid = np.concatenate(eyelid_blocks, axis=0).astype(np.float32)
    jaw = np.concatenate(jaw_blocks, axis=0).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64)
    crop_success_arr = np.asarray(crop_success_flags, dtype=bool)

    if not np.isfinite(features).all():
        raise FloatingPointError(f"NaN/Inf detected in SMIRK features for split={split}")
    if features.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"Feature/label length mismatch for split={split}: {features.shape} vs {labels_arr.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features,
        expression_params=expression,
        eyelid_params=eyelid,
        jaw_params=jaw,
        labels=labels_arr,
        sample_ids=sample_ids_arr,
        crop_success=crop_success_arr,
        feature_keys=np.asarray(feature_keys),
        emotion_names=np.asarray(EMOTION_NAMES),
    )
    metadata = {
        "split": split,
        "path": str(out_path),
        "num_samples": int(features.shape[0]),
        "skipped": int(skipped),
        "feature_shape": list(features.shape),
        "feature_keys": feature_keys,
        "expression_params_shape": list(expression.shape),
        "eyelid_params_shape": list(eyelid.shape),
        "jaw_params_shape": list(jaw.shape),
        "crop": use_crop,
        "crop_success_count": int(crop_success_arr.sum()),
        "crop_failure_count": int((~crop_success_arr).sum()),
        "on_crop_failure": on_crop_failure,
        "finite": bool(np.isfinite(features).all()),
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"SMIRK_FEATURE_SHAPE[{split}]={features.shape} feature_dim={features.shape[1]} "
        f"expression_params={expression.shape} eyelid_params={eyelid.shape} jaw_params={jaw.shape} "
        f"nan_count={int(np.isnan(features).sum())} saved={out_path}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    smirk_cfg = cfg.get("smirk", {})
    paths_cfg = cfg.get("paths", {})

    smirk_root = resolve_path(args.smirk_root or os.environ.get("SMIRK_ROOT") or smirk_cfg.get("smirk_root"))
    checkpoint = resolve_path(args.checkpoint or os.environ.get("SMIRK_CHECKPOINT") or smirk_cfg.get("checkpoint"))
    if smirk_root is None or checkpoint is None:
        raise ValueError("Both smirk_root and checkpoint must be configured or passed as arguments.")
    device_name = args.device or os.environ.get("SMIRK_DEVICE") or smirk_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SMIRK extraction, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    batch_size = int(args.batch_size or smirk_cfg.get("extract_batch_size", 128))
    splits = args.splits or ["train", "val", "test"]
    feature_dir = resolve_path(paths_cfg.get("feature_dir")) or (resolve_path(paths_cfg["output_dir"]) / "features")
    if args.output_subdir:
        feature_dir = (resolve_path(paths_cfg["output_dir"]) or PROJECT_ROOT / "outputs" / "smirk_only") / args.output_subdir

    torch.backends.cudnn.benchmark = True
    smirk_encoder_module, run_mediapipe = import_smirk(smirk_root)
    encoder = load_frozen_encoder(
        smirk_encoder_module,
        checkpoint,
        device,
        strict=bool(smirk_cfg.get("strict_load", True)),
        init_timm_pretrained=bool(smirk_cfg.get("init_timm_pretrained", False)),
    )

    data_dir = resolve_path(cfg["data"]["data_path"])
    for split in splits:
        records = collect_split_records(
            data_dir,
            split,
            mask_dir=None,
            use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
            bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
            predecode_pixels=bool(cfg["data"].get("predecode_pixels", True)),
            preload_masks=False,
            allow_missing_masks=False,
        )
        extract_split(
            split,
            records,
            encoder=encoder,
            run_mediapipe=run_mediapipe,
            device=device,
            batch_size=batch_size,
            out_path=output_path_for(feature_dir, split),
            cfg=cfg,
            args=args,
        )
    print(f"SMIRK_FEATURE_DIR={feature_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

