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
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import collect_split_records
from scripts.extract_smirk_features import import_smirk, load_frozen_encoder, prepare_smirk_image, pushd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen SMIRK/FLAME depth+normal VLM geometry tokens for FER2013.")
    parser.add_argument("--config", type=str, default="config_smirk_geometry_cross_attention.yaml")
    parser.add_argument("--smirk-root", type=str, default=None)
    parser.add_argument("--smirk-checkpoint", type=str, default=None)
    parser.add_argument("--vlm-model", type=str, default=None)
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
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_path_for(feature_dir: Path, pattern: str, split: str) -> Path:
    return feature_dir / pattern.format(split=split)


def import_geometry_stack(smirk_root: Path):
    smirk_encoder_module, run_mediapipe = import_smirk(smirk_root)
    with pushd(smirk_root):
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer
        from src.renderer.util import batch_orth_proj, face_vertices, vertex_normals
    return smirk_encoder_module, run_mediapipe, FLAME, Renderer, batch_orth_proj, face_vertices, vertex_normals


def load_frozen_vlm(model_name: str, device: torch.device):
    from transformers import CLIPImageProcessor, CLIPVisionModel

    processor = CLIPImageProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad_(False)
    print(f"VLM_LOAD_OK model={model_name} trainable_params=0", flush=True)
    return processor, vision_model


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = vertices.shape[0]
    transformed_vertices = batch_orth_proj(vertices, cam)
    transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]
    render_vertices = vertices
    render_transformed = transformed_vertices
    if not renderer.render_full_head:
        keep = torch.as_tensor(renderer.final_mask, dtype=torch.long, device=vertices.device)
        render_vertices = vertices[:, keep, :]
        render_transformed = transformed_vertices[:, keep, :]
    render_transformed = render_transformed.clone()
    raw_depth = render_transformed[:, :, 2:3]
    render_transformed[:, :, 2] = render_transformed[:, :, 2] + 10.0
    faces = renderer.faces.expand(batch_size, -1, -1)
    normals = vertex_normals(render_vertices, faces)
    face_normals = face_vertices(normals, faces)
    face_depth = face_vertices(raw_depth, faces)
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
    depth_norm = (depth - d_min) / torch.clamp(d_max - d_min, min=1e-6)
    depth_rgb = depth_norm.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * mask
    normal_rgb = (normal * 0.5 + 0.5).clamp(0.0, 1.0) * mask
    return depth_rgb, normal_rgb


def tensor_images_to_uint8_hwc(images: torch.Tensor) -> List[np.ndarray]:
    arr = images.detach().float().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    return [(img * 255.0).round().astype(np.uint8) for img in arr]


def encode_vlm_tokens(
    processor,
    vision_model,
    images: torch.Tensor,
    device: torch.device,
    *,
    drop_cls_token: bool,
) -> torch.Tensor:
    image_list = tensor_images_to_uint8_hwc(images)
    encoded = processor(images=image_list, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)
    outputs = vision_model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state
    if drop_cls_token:
        tokens = tokens[:, 1:, :]
    return tokens.detach().float().cpu()


def save_preview_images(preview_dir: Path, split: str, sample_ids: Sequence[int], depth: torch.Tensor, normal: torch.Tensor) -> None:
    split_dir = preview_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    depth_imgs = tensor_images_to_uint8_hwc(depth)
    normal_imgs = tensor_images_to_uint8_hwc(normal)
    for sample_id, depth_img, normal_img in zip(sample_ids[:8], depth_imgs[:8], normal_imgs[:8]):
        combined = np.concatenate([depth_img, normal_img], axis=1)
        cv2.imwrite(str(split_dir / f"{int(sample_id):06d}_depth_normal.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def extract_split(
    split: str,
    cfg: Dict,
    *,
    smirk_root: Path,
    encoder,
    flame,
    renderer,
    run_mediapipe,
    processor,
    vision_model,
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
        print(f"GEOMETRY_TOKEN_CACHE_EXISTS split={split} path={output_path} shape={cached['geometry_tokens'].shape}", flush=True)
        return

    data_dir = resolve_path(cfg["data"]["data_path"])
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
    total = len(records.labels)
    if args.max_samples_per_split is not None:
        total = min(total, int(args.max_samples_per_split))

    smirk_cfg = cfg.get("smirk", {})
    vlm_cfg = cfg.get("vlm", {})
    render_cfg = smirk_cfg.get("render", {})
    token_sources = list(vlm_cfg.get("token_sources", ["depth", "normal"]))
    drop_cls = bool(vlm_cfg.get("drop_cls_token", True))
    cache_dtype = np.float16 if str(vlm_cfg.get("cache_dtype", "float16")).lower() == "float16" else np.float32

    geometry_chunks: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[int] = []
    crop_success: List[bool] = []
    skipped = 0
    first_shape_logged = False
    preview_saved = False

    for start in tqdm(range(0, total, batch_size), desc=f"SMIRK+VLM {split}", dynamic_ncols=True):
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
            smirk_outputs = encoder(images)
            flame_output = flame.forward(smirk_outputs)
            cam = smirk_outputs["cam"] if bool(render_cfg.get("use_smirk_cam", False)) else make_fixed_cam(images.shape[0], float(render_cfg.get("fixed_cam_scale", 7.0)), device)
            depth_rgb, normal_rgb = render_depth_normal_maps(
                renderer,
                flame_output["vertices"],
                cam,
                batch_orth_proj=batch_orth_proj,
                face_vertices=face_vertices,
                vertex_normals=vertex_normals,
            )
            token_blocks = []
            depth_tokens = normal_tokens = None
            if "depth" in token_sources:
                depth_tokens = encode_vlm_tokens(processor, vision_model, depth_rgb, device, drop_cls_token=drop_cls)
                token_blocks.append(depth_tokens)
            if "normal" in token_sources:
                normal_tokens = encode_vlm_tokens(processor, vision_model, normal_rgb, device, drop_cls_token=drop_cls)
                token_blocks.append(normal_tokens)
            geometry_tokens = torch.cat(token_blocks, dim=1).numpy().astype(cache_dtype)

        if not np.isfinite(geometry_tokens).all():
            raise FloatingPointError(f"NaN/Inf in geometry tokens for split={split}, batch_start={start}")
        if not first_shape_logged:
            first_shape_logged = True
            print(f"GEOMETRY_SHAPE_TRACE[{split}]", flush=True)
            print(f"  smirk_input: {tuple(images.shape)}", flush=True)
            print(f"  flame_vertices: {tuple(flame_output['vertices'].shape)}", flush=True)
            print(f"  depth_rgb: {tuple(depth_rgb.shape)}", flush=True)
            print(f"  normal_rgb: {tuple(normal_rgb.shape)}", flush=True)
            if depth_tokens is not None:
                print(f"  depth_vlm_tokens: {tuple(depth_tokens.shape)}", flush=True)
            if normal_tokens is not None:
                print(f"  normal_vlm_tokens: {tuple(normal_tokens.shape)}", flush=True)
            print(f"  geometry_tokens_cached: {geometry_tokens.shape}", flush=True)
        if args.save_preview and not preview_saved:
            preview_dir = (resolve_path(cfg["paths"]["output_dir"]) or PROJECT_ROOT / "outputs" / "smirk_geometry_cross_attention") / "geometry_previews"
            save_preview_images(preview_dir, split, batch_ids, depth_rgb, normal_rgb)
            preview_saved = True

        geometry_chunks.append(geometry_tokens)
        labels.extend(batch_labels)
        sample_ids.extend(batch_ids)
        crop_success.extend(batch_crop)

    if not geometry_chunks:
        raise RuntimeError(f"No geometry tokens extracted for split={split}. skipped={skipped}")
    geometry = np.concatenate(geometry_chunks, axis=0)
    labels_arr = np.asarray(labels, dtype=np.int64)
    ids_arr = np.asarray(sample_ids, dtype=np.int64)
    crop_arr = np.asarray(crop_success, dtype=bool)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        geometry_tokens=geometry,
        labels=labels_arr,
        sample_ids=ids_arr,
        crop_success=crop_arr,
        token_sources=np.asarray(token_sources),
        vlm_model=np.asarray(str(vlm_cfg.get("model_name", ""))),
    )
    meta = {
        "split": split,
        "path": str(output_path),
        "shape": list(geometry.shape),
        "dtype": str(geometry.dtype),
        "num_samples": int(geometry.shape[0]),
        "num_tokens": int(geometry.shape[1]),
        "token_dim": int(geometry.shape[2]),
        "token_sources": token_sources,
        "skipped": int(skipped),
        "crop_success": int(crop_arr.sum()),
        "crop_failure": int((~crop_arr).sum()),
    }
    with output_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"GEOMETRY_TOKEN_SHAPE[{split}]={geometry.shape} saved={output_path}", flush=True)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    smirk_cfg = cfg.get("smirk", {})
    vlm_cfg = cfg.get("vlm", {})
    cache_cfg = cfg.get("geometry_cache", {})

    smirk_root = resolve_path(args.smirk_root or os.environ.get("SMIRK_ROOT") or smirk_cfg.get("smirk_root"))
    checkpoint = resolve_path(args.smirk_checkpoint or os.environ.get("SMIRK_CHECKPOINT") or smirk_cfg.get("checkpoint"))
    if smirk_root is None or checkpoint is None:
        raise ValueError("Both SMIRK root and official checkpoint are required.")
    device_name = args.device or os.environ.get("GEOMETRY_DEVICE") or smirk_cfg.get("device", vlm_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    batch_size = int(args.batch_size or vlm_cfg.get("batch_size", 32))
    splits = args.splits or ["train", "val", "test"]
    feature_dir = resolve_path(cache_cfg.get("feature_dir")) or (PROJECT_ROOT / "outputs" / "smirk_geometry_cross_attention" / "geometry_tokens")
    pattern = str(cache_cfg.get("token_file_pattern", "{split}_smirk_vlm_geometry_tokens.npz"))

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
    for param in flame.parameters():
        param.requires_grad_(False)
    for param in renderer.parameters():
        param.requires_grad_(False)
    print("SMIRK_FLAME_RENDERER_FROZEN_OK", flush=True)

    processor, vision_model = load_frozen_vlm(args.vlm_model or str(vlm_cfg.get("model_name", "openai/clip-vit-base-patch32")), device)
    for split in splits:
        extract_split(
            split,
            cfg,
            smirk_root=smirk_root,
            encoder=encoder,
            flame=flame,
            renderer=renderer,
            run_mediapipe=run_mediapipe,
            processor=processor,
            vision_model=vision_model,
            device=device,
            batch_size=batch_size,
            output_path=cache_path_for(feature_dir, pattern, split),
            args=args,
            batch_orth_proj=batch_orth_proj,
            face_vertices=face_vertices,
            vertex_normals=vertex_normals,
        )
    print(f"GEOMETRY_TOKEN_CACHE_DIR={feature_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
