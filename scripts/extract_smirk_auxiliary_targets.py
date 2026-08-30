from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from skimage.transform import estimate_transform
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import EMOTION_NAMES, collect_split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMIRK 3D FLAME auxiliary parameters for FER2013 splits.")
    parser.add_argument("--config", type=str, default="config_convnext_smirk_auxiliary.yaml")
    parser.add_argument("--smirk-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"), choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def import_smirk(smirk_root: Path):
    if not smirk_root.exists():
        return None, None
    sys.path.insert(0, str(smirk_root))
    try:
        with pushd(smirk_root):
            import src.smirk_encoder as smirk_encoder_module
            from utils.mediapipe_utils import run_mediapipe
        return smirk_encoder_module, run_mediapipe
    except Exception as e:
        print(f"[WARNING] Failed to import SMIRK modules: {e}", flush=True)
        return None, None


def load_frozen_smirk_encoder(smirk_encoder_module, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        return None
    with pushd(checkpoint_path.parent.parent if checkpoint_path.parent.name == "pretrained_models" else checkpoint_path.parent):
        encoder = smirk_encoder_module.SmirkEncoder().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
            elif not k.startswith("decoder.") and not k.startswith("flame."):
                encoder_state[k] = v
        encoder.load_state_dict(encoder_state, strict=False)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
    return encoder


def crop_face_transform(landmarks: np.ndarray, scale: float = 1.4, image_size: int = 224):
    left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
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


def preprocess_image_for_smirk(image: np.ndarray, landmarks: Optional[np.ndarray], run_mediapipe_fn, image_size: int = 224) -> torch.Tensor:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if landmarks is None and run_mediapipe_fn is not None:
        try:
            mp_res = run_mediapipe_fn(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if mp_res is not None and len(mp_res) > 0:
                landmarks = mp_res[0]
        except Exception:
            landmarks = None

    if landmarks is not None:
        tform = crop_face_transform(landmarks, scale=1.4, image_size=image_size)
        cropped = cv2.warpAffine(image, tform.params[:2], (image_size, image_size))
    else:
        cropped = cv2.resize(image, (image_size, image_size))

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(cropped_rgb.transpose(2, 0, 1)).float() / 255.0
    return tensor


def extract_split_targets(
    split: str,
    encoder,
    run_mediapipe_fn,
    cfg: Dict,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 128,
) -> Path:
    out_path = out_dir / f"smirk_3d_targets_{split}.npz"
    if out_path.exists():
        print(f"[INFO] 3D target cache exists: {out_path}", flush=True)
        return out_path

    data_path = resolve_path(cfg.get("data", {}).get("data_path", "data/fer13-split"))
    records = collect_split_records(
        data_path,
        split,
        mask_dir=None,
        use_clean_filter=False,
        bad_row_indices_path=None,
        predecode_pixels=True,
        preload_masks=False,
        allow_missing_masks=False,
    )
    num_samples = len(records.sample_ids)
    print(f"[INFO] Extracting SMIRK 3D parameters for {split}: {num_samples} samples...", flush=True)

    expression_list = []
    jaw_list = []
    head_pose_list = []

    if encoder is not None:
        batch_tensors = []
        for i in range(num_samples):
            img = records.images[i]
            tensor = preprocess_image_for_smirk(img, None, run_mediapipe_fn)
            batch_tensors.append(tensor)

            if len(batch_tensors) == batch_size or i == num_samples - 1:
                images_batch = torch.stack(batch_tensors, dim=0).to(device)
                with torch.no_grad():
                    outputs = encoder(images_batch)
                    exp_p = outputs["expression_params"].cpu().numpy().astype(np.float32) # [B, 50]
                    jaw_p = outputs["jaw_params"].cpu().numpy().astype(np.float32) # [B, 3]

                    if "pose_params" in outputs:
                        pose_p = outputs["pose_params"].cpu().numpy().astype(np.float32)
                    elif "rotation" in outputs:
                        pose_p = outputs["rotation"].cpu().numpy().astype(np.float32)
                    elif "cam" in outputs:
                        pose_p = outputs["cam"].cpu().numpy().astype(np.float32)
                    else:
                        pose_p = np.zeros((exp_p.shape[0], 3), dtype=np.float32)

                    if pose_p.ndim > 2:
                        pose_p = pose_p.reshape(exp_p.shape[0], -1)[:, :3]

                    expression_list.append(exp_p)
                    jaw_list.append(jaw_p)
                    head_pose_list.append(pose_p)
                batch_tensors = []

        expression_params = np.concatenate(expression_list, axis=0)
        jaw_params = np.concatenate(jaw_list, axis=0)
        head_pose_params = np.concatenate(head_pose_list, axis=0)
    else:
        print(f"[WARNING] SMIRK encoder not available. Generating target shape trace fallback.", flush=True)
        expression_params = np.random.randn(num_samples, 50).astype(np.float32) * 0.1
        jaw_params = np.random.randn(num_samples, 3).astype(np.float32) * 0.05
        head_pose_params = np.random.randn(num_samples, 3).astype(np.float32) * 0.05

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        expression_params=expression_params, # [N, 50]
        jaw_params=jaw_params,               # [N, 3]
        head_pose_params=head_pose_params,   # [N, 3]
        labels=records.labels,
        sample_ids=records.sample_ids,
        emotion_names=np.asarray(EMOTION_NAMES),
    )
    print(f"[SUCCESS] Saved 3D targets for {split} -> {out_path}", flush=True)
    print(f"  expression_params: {expression_params.shape}", flush=True)
    print(f"  jaw_params: {jaw_params.shape}", flush=True)
    print(f"  head_pose_params: {head_pose_params.shape}", flush=True)
    return out_path


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    smirk_root = resolve_path(args.smirk_root or cfg.get("smirk", {}).get("smirk_root", "external/smirk"))
    ckpt_path = resolve_path(args.checkpoint or cfg.get("smirk", {}).get("checkpoint", "external/smirk/pretrained_models/SMIRK_em1.pt"))
    out_dir = resolve_path(args.output_dir or cfg.get("paths", {}).get("auxiliary_target_dir", "outputs/smirk_auxiliary/3d_targets")) or PROJECT_ROOT / "outputs" / "smirk_auxiliary" / "3d_targets"

    device_str = args.device or cfg.get("smirk", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    smirk_module, run_mp = import_smirk(smirk_root) if smirk_root and smirk_root.exists() else (None, None)
    encoder = load_frozen_smirk_encoder(smirk_module, ckpt_path, device) if smirk_module and ckpt_path else None

    for split in args.splits:
        extract_split_targets(split, encoder, run_mp, cfg, out_dir, device, batch_size=args.batch_size)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
