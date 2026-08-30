from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import collect_split_records
from scripts.extract_smirk_features import import_smirk, load_frozen_encoder, prepare_smirk_image, pushd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen SMIRK/FLAME depth+normal maps for FER2013 Stage 1 late fusion.")
    parser.add_argument("--config", type=str, default="config_stage1_rgb_smirk_3d_cnn_late_fusion.yaml")
    parser.add_argument("--smirk-root", type=str, default=None)
    parser.add_argument("--smirk-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_path_for(cache_dir: Path, pattern: str, split: str) -> Path:
    return cache_dir / pattern.format(split=split)


def import_geometry_stack(smirk_root: Path):
    smirk_encoder_module, run_mediapipe = import_smirk(smirk_root)
    with pushd(smirk_root):
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer
        from src.renderer.util import batch_orth_proj, face_vertices, vertex_normals
    return smirk_encoder_module, run_mediapipe, FLAME, Renderer, batch_orth_proj, face_vertices, vertex_normals


def make_fixed_cam(batch_size: int, scale: float, device: torch.device) -> torch.Tensor:
    cam = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    cam[:, 0] = float(scale)
    return cam


def render_depth_normal_maps(
    renderer,
    vertices: torch.Tensor,
    cam: torch.Tensor,
    *,
    batch_orth_proj,
    face_vertices,
    vertex_normals,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(vertices.shape[0])
    transformed_vertices = batch_orth_proj(vertices, cam)
    transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]

    render_vertices = vertices
    render_transformed = transformed_vertices
    if not renderer.render_full_head:
        keep = torch.as_tensor(renderer.final_mask, dtype=torch.long, device=vertices.device)
        render_vertices = vertices[:, keep, :]
        render_transformed = transformed_vertices[:, keep, :]

    render_transformed = render_transformed.clone()
    faces = renderer.faces.expand(batch_size, -1, -1)
    normals = vertex_normals(render_vertices, faces)
    face_normals = face_vertices(normals, faces)
    face_vertices_xyz = face_vertices(render_transformed, faces)
    face_depth = face_vertices_xyz[..., 2:3]

    render_transformed[:, :, 2] = render_transformed[:, :, 2] + 10.0
    attributes = torch.cat([face_normals, face_depth], dim=-1)
    raster = renderer.rasterize(render_transformed, faces, attributes)
    normal = raster[:, :3, :, :]
    depth = raster[:, 3:4, :, :]
    mask = raster[:, 4:5, :, :]

    valid = mask > 0.0
    inf = torch.full_like(depth, float("inf"))
    neg_inf = torch.full_like(depth, float("-inf"))
    d_min = torch.amin(torch.where(valid, depth, inf), dim=(1, 2, 3), keepdim=True)
    d_max = torch.amax(torch.where(valid, depth, neg_inf), dim=(1, 2, 3), keepdim=True)
    d_min = torch.where(torch.isfinite(d_min), d_min, torch.zeros_like(d_min))
    d_max = torch.where(torch.isfinite(d_max), d_max, torch.ones_like(d_max))
    depth_1ch = ((depth - d_min) / torch.clamp(d_max - d_min, min=1e-6)).clamp(0.0, 1.0) * mask
    normal_3ch = (normal * 0.5 + 0.5).clamp(0.0, 1.0) * mask
    geometry_4ch = torch.cat([depth_1ch, normal_3ch], dim=1)
    return depth_1ch, normal_3ch, geometry_4ch


def tensor_to_uint8_hwc(images: torch.Tensor) -> List[np.ndarray]:
    arr = images.detach().float().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    return [(img * 255.0).round().astype(np.uint8) for img in arr]


def save_preview(preview_dir: Path, split: str, sample_ids: Sequence[int], depth: torch.Tensor, normal: torch.Tensor) -> None:
    split_dir = preview_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    depth_rgb = depth.repeat(1, 3, 1, 1)
    for sample_id, depth_img, normal_img in zip(sample_ids[:8], tensor_to_uint8_hwc(depth_rgb[:8]), tensor_to_uint8_hwc(normal[:8])):
        combined = np.concatenate([depth_img, normal_img], axis=1)
        cv2.imwrite(str(split_dir / f"{int(sample_id):06d}_depth_normal.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def trainable_param_count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def extract_split(
    split: str,
    cfg: Dict,
    *,
    encoder,
    flame,
    renderer,
    run_mediapipe,
    device: torch.device,
    batch_size: int,
    output_path: Path,
    args: argparse.Namespace,
    batch_orth_proj,
    face_vertices,
    vertex_normals,
) -> None:
    if output_path.exists() and not args.force:
        cached = np.load(output_path)
        print(f"SMIRK_DEPTH_NORMAL_CACHE_EXISTS split={split} path={output_path} geometry_maps_shape={cached['geometry_maps'].shape}", flush=True)
        return

    records = collect_split_records(
        resolve_path(cfg["data"]["data_path"]),
        split,
        mask_dir=None,
        use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
        bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
        predecode_pixels=bool(cfg["data"].get("predecode_pixels", True)),
        preload_masks=False,
        allow_missing_masks=False,
    )
    total = len(records.labels)
    if args.max_samples_per_split is not None:
        total = min(total, int(args.max_samples_per_split))

    smirk_cfg = cfg.get("smirk", {})
    render_cfg = smirk_cfg.get("render", {})
    cache_dtype = np.float16 if str(cfg.get("geometry_cache", {}).get("dtype", "float16")).lower() == "float16" else np.float32
    geometry_chunks: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[int] = []
    crop_success: List[bool] = []
    skipped = 0
    first_shape_logged = False
    preview_saved = False

    for start in tqdm(range(0, total, batch_size), desc=f"SMIRK depth+normal {split}", dynamic_ncols=True):
        batch_tensors: List[torch.Tensor] = []
        batch_labels: List[int] = []
        batch_ids: List[int] = []
        batch_crop: List[bool] = []
        for i in range(start, min(start + batch_size, total)):
            tensor, ok = prepare_smirk_image(
                records.images[i],
                run_mediapipe=run_mediapipe,
                use_crop=bool(smirk_cfg.get("crop", True)) and not args.no_crop,
                crop_scale=float(smirk_cfg.get("crop_scale", 1.4)),
                image_size=int(smirk_cfg.get("image_size", 224)),
                mediapipe_input_size=int(smirk_cfg.get("mediapipe_input_size", 224)),
                on_crop_failure=str(smirk_cfg.get("on_crop_failure", "resize")),
            )
            if tensor is None:
                skipped += 1
                continue
            batch_tensors.append(tensor)
            batch_labels.append(int(records.labels[i]))
            batch_ids.append(int(records.sample_ids[i]))
            batch_crop.append(bool(ok))
        if not batch_tensors:
            continue

        images = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            outputs = encoder(images)
            flame_output = flame.forward(outputs)
            cam = outputs["cam"] if bool(render_cfg.get("use_smirk_cam", False)) else make_fixed_cam(images.shape[0], float(render_cfg.get("fixed_cam_scale", 7.0)), device)
            depth, normal, geometry_4ch = render_depth_normal_maps(
                renderer,
                flame_output["vertices"],
                cam,
                batch_orth_proj=batch_orth_proj,
                face_vertices=face_vertices,
                vertex_normals=vertex_normals,
            )
            geometry_nhwc = geometry_4ch.detach().float().cpu().permute(0, 2, 3, 1).numpy().astype(cache_dtype)

        if not np.isfinite(geometry_nhwc).all():
            raise FloatingPointError(f"NaN/Inf in depth+normal maps for split={split}, batch_start={start}")
        if not first_shape_logged:
            first_shape_logged = True
            print(f"SMIRK_DEPTH_NORMAL_SHAPE_TRACE[{split}]", flush=True)
            print(f"  smirk_input: {tuple(images.shape)}", flush=True)
            print(f"  flame_vertices: {tuple(flame_output['vertices'].shape)}", flush=True)
            print(f"  depth_map_1ch: {tuple(depth.shape)}", flush=True)
            print(f"  normal_map_3ch: {tuple(normal.shape)}", flush=True)
            print(f"  geometry_maps_cached_nhwc: {geometry_nhwc.shape}", flush=True)
        if args.save_preview and not preview_saved:
            preview_dir = (resolve_path(cfg["paths"]["output_dir"]) or PROJECT_ROOT / "outputs" / "stage1_rgb_smirk_3d_cnn_late_fusion") / "geometry_previews"
            save_preview(preview_dir, split, batch_ids, depth, normal)
            preview_saved = True

        geometry_chunks.append(geometry_nhwc)
        labels.extend(batch_labels)
        sample_ids.extend(batch_ids)
        crop_success.extend(batch_crop)

    if not geometry_chunks:
        raise RuntimeError(f"No SMIRK depth+normal maps extracted for split={split}. skipped={skipped}")
    geometry = np.concatenate(geometry_chunks, axis=0)
    labels_arr = np.asarray(labels, dtype=np.int64)
    ids_arr = np.asarray(sample_ids, dtype=np.int64)
    crop_arr = np.asarray(crop_success, dtype=bool)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        geometry_maps=geometry,
        labels=labels_arr,
        sample_ids=ids_arr,
        crop_success=crop_arr,
        channel_order=np.asarray(["depth", "normal_x", "normal_y", "normal_z"]),
        smirk_frozen=np.asarray(True),
        smirk_trainable_params=np.asarray(0, dtype=np.int64),
        flame_trainable_params=np.asarray(0, dtype=np.int64),
        renderer_trainable_params=np.asarray(0, dtype=np.int64),
    )
    meta = {
        "cache_kind": "smirk_depth_normal_4ch",
        "split": split,
        "path": str(output_path),
        "shape": list(geometry.shape),
        "dtype": str(geometry.dtype),
        "num_samples": int(geometry.shape[0]),
        "height": int(geometry.shape[1]),
        "width": int(geometry.shape[2]),
        "channels": int(geometry.shape[3]),
        "channel_order": ["depth", "normal_x", "normal_y", "normal_z"],
        "smirk_frozen": True,
        "smirk_trainable_params": 0,
        "flame_trainable_params": 0,
        "renderer_trainable_params": 0,
        "skipped": int(skipped),
        "crop_success": int(crop_arr.sum()),
        "crop_failure": int((~crop_arr).sum()),
    }
    with output_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"SMIRK_DEPTH_NORMAL_CACHE_SHAPE[{split}]={geometry.shape} saved={output_path}", flush=True)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    smirk_cfg = cfg.get("smirk", {})
    cache_cfg = cfg.get("geometry_cache", {})

    smirk_root = resolve_path(args.smirk_root or os.environ.get("SMIRK_ROOT") or smirk_cfg.get("smirk_root"))
    checkpoint = resolve_path(args.smirk_checkpoint or os.environ.get("SMIRK_CHECKPOINT") or smirk_cfg.get("checkpoint"))
    if smirk_root is None or checkpoint is None:
        raise ValueError("Both SMIRK root and official checkpoint are required.")
    device_name = args.device or os.environ.get("SMIRK_DEVICE") or smirk_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    batch_size = int(args.batch_size or smirk_cfg.get("render_batch_size", 32))
    splits = args.splits or ["train", "val", "test"]
    cache_dir = resolve_path(cache_cfg.get("feature_dir")) or (PROJECT_ROOT / "outputs" / "stage1_rgb_smirk_3d_cnn_late_fusion" / "geometry_maps")
    pattern = str(cache_cfg.get("map_file_pattern", "{split}_smirk_depth_normal_maps.npz"))

    torch.backends.cudnn.benchmark = True
    smirk_encoder_module, run_mediapipe, flame_cls, renderer_cls, batch_orth_proj, face_vertices, vertex_normals = import_geometry_stack(smirk_root)
    encoder = load_frozen_encoder(
        smirk_encoder_module,
        checkpoint,
        device,
        strict=bool(smirk_cfg.get("strict_load", True)),
        init_timm_pretrained=bool(smirk_cfg.get("init_timm_pretrained", False)),
    )
    with pushd(smirk_root):
        flame = flame_cls().to(device).eval()
        renderer = renderer_cls().to(device).eval()
    for module in (flame, renderer):
        for param in module.parameters():
            param.requires_grad_(False)
    if trainable_param_count(encoder) != 0 or trainable_param_count(flame) != 0 or trainable_param_count(renderer) != 0:
        raise RuntimeError("SMIRK/FLAME/renderer freeze check failed before caching.")
    print("SMIRK_FROZEN_OK trainable_params=0", flush=True)
    print("FLAME_RENDERER_FROZEN_OK trainable_params=0", flush=True)

    for split in splits:
        extract_split(
            split,
            cfg,
            encoder=encoder,
            flame=flame,
            renderer=renderer,
            run_mediapipe=run_mediapipe,
            device=device,
            batch_size=batch_size,
            output_path=cache_path_for(cache_dir, pattern, split),
            args=args,
            batch_orth_proj=batch_orth_proj,
            face_vertices=face_vertices,
            vertex_normals=vertex_normals,
        )
    print(f"SMIRK_DEPTH_NORMAL_CACHE_DIR={cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

