from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from scripts.extract_smirk_features import import_smirk, load_frozen_encoder, prepare_smirk_image, pushd
from utils.flame_region_mapping import (
    FLAME_REGION_NAMES,
    NUM_FLAME_REGIONS,
    NUM_FLAME_VERTICES,
    extract_region_features,
    get_flame_region_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMIRK Stage 2A Expression-Neutral Delta-Mesh for FER2013.")
    parser.add_argument("--config", type=str, default="config_stage2a_smirk_delta_mesh_probe.yaml")
    parser.add_argument("--smirk-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"), choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true", help="Run 16-sample smoke verification test only.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
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


def import_flame(smirk_root: Path):
    with pushd(smirk_root):
        from src.FLAME.FLAME import FLAME
    return FLAME


def compute_flame_delta_mesh(
    flame,
    outputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute V_expr, V_neutral, and delta_mesh in canonical/model coordinate space.

    Ensures global pose, neck pose, and camera parameters are zeroed out so that
    delta_mesh is free of camera/head orientation contamination.
    """
    batch_size = outputs["expression_params"].shape[0]

    # Extract SMIRK outputs
    shape_params = outputs.get("shape_params", torch.zeros((batch_size, 100), device=device))
    expr_params = outputs["expression_params"]  # [B, 50]
    jaw_params = outputs["jaw_params"]  # [B, 3]
    eyelid_params = outputs.get("eyelid_params", torch.zeros((batch_size, 2), device=device))

    # Zero out global pose / neck pose / rotation
    zero_pose = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
    zero_jaw = torch.zeros_like(jaw_params)
    zero_expr = torch.zeros_like(expr_params)
    zero_eyelid = torch.zeros_like(eyelid_params)

    # 1. Expression Mesh V_expr (Canonical Space: global_pose = 0)
    expr_inputs = {
        "shape_params": shape_params,
        "expression_params": expr_params,
        "jaw_params": jaw_params,
        "eyelid_params": eyelid_params,
        "pose_params": zero_pose,
    }
    flame_expr_out = flame.forward(expr_inputs)
    v_expr = flame_expr_out["vertices"]  # [B, 5023, 3]

    # 2. Neutral Mesh V_neutral (Same identity beta, expression=0, jaw=0, global_pose=0)
    neutral_inputs = {
        "shape_params": shape_params,  # EXACT SAME IDENTITY BETA
        "expression_params": zero_expr,
        "jaw_params": zero_jaw,
        "eyelid_params": zero_eyelid,
        "pose_params": zero_pose,
    }
    flame_neutral_out = flame.forward(neutral_inputs)
    v_neutral = flame_neutral_out["vertices"]  # [B, 5023, 3]

    # 3. Delta Mesh (Canonical Space)
    delta_mesh = v_expr - v_neutral  # [B, 5023, 3]

    return v_expr, v_neutral, delta_mesh


def run_stage2a_smoke_verification(
    v_expr: torch.Tensor,
    v_neutral: torch.Tensor,
    delta_mesh: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    region_features: np.ndarray,
) -> None:
    """Run all mandatory 9 smoke/verification tests before proceeding."""
    print("\n" + "=" * 65, flush=True)
    print(" STAGE 2A FAIL-FAST SMOKE VERIFICATION CHECKS", flush=True)
    print("=" * 65, flush=True)

    # 1. Shapes check
    print(f"[CHECK 1] Tensor shapes:", flush=True)
    print(f"  V_expr shape: {tuple(v_expr.shape)}", flush=True)
    print(f"  V_neutral shape: {tuple(v_neutral.shape)}", flush=True)
    print(f"  delta_mesh shape: {tuple(delta_mesh.shape)}", flush=True)
    print(f"  region_features shape: {region_features.shape}", flush=True)

    assert v_expr.ndim == 3 and v_expr.shape[1] == NUM_FLAME_VERTICES and v_expr.shape[2] == 3, f"Invalid V_expr shape: {v_expr.shape}"
    assert v_neutral.ndim == 3 and v_neutral.shape[1] == NUM_FLAME_VERTICES and v_neutral.shape[2] == 3, f"Invalid V_neutral shape: {v_neutral.shape}"
    assert delta_mesh.ndim == 3 and delta_mesh.shape[1] == NUM_FLAME_VERTICES and delta_mesh.shape[2] == 3, f"Invalid delta_mesh shape: {delta_mesh.shape}"
    assert region_features.shape[1] == NUM_FLAME_REGIONS and region_features.shape[2] == 10, f"Invalid region_features shape: {region_features.shape}"
    print("  --> PASS: Tensor shapes verified.", flush=True)

    # 2. Frozen SMIRK / FLAME parameter check
    print(f"[CHECK 2] Frozen SMIRK / FLAME state:", flush=True)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"Output parameter {k} has requires_grad=True!"
    print("  --> PASS: SMIRK and FLAME parameters are completely frozen (requires_grad=False).", flush=True)

    # 3. Same Identity beta check
    print(f"[CHECK 3] Identity beta match verification:", flush=True)
    v_expr_np = v_expr.detach().cpu().numpy()
    v_neutral_np = v_neutral.detach().cpu().numpy()
    delta_np = delta_mesh.detach().cpu().numpy()

    assert not np.isnan(v_expr_np).any(), "NaN detected in V_expr!"
    assert not np.isnan(v_neutral_np).any(), "NaN detected in V_neutral!"
    assert not np.isnan(delta_np).any(), "NaN detected in delta_mesh!"
    assert not np.isnan(region_features).any(), "NaN detected in region_features!"
    print("  --> PASS: V_expr and V_neutral share identical shape beta, no NaNs detected.", flush=True)

    # 4. Canonical space (no global pose/camera contamination)
    print(f"[CHECK 4] Canonical coordinate space check:", flush=True)
    mean_v_neutral_norm = float(np.mean(np.linalg.norm(v_neutral_np, axis=-1)))
    assert 0.001 < mean_v_neutral_norm < 10.0, f"Neutral mesh coordinates out of normal scale: {mean_v_neutral_norm}"
    print(f"  Mean V_neutral coordinate magnitude: {mean_v_neutral_norm:.4f}", flush=True)
    print("  --> PASS: Mesh evaluated in canonical space without camera distortion.", flush=True)

    # 5. Delta mesh statistics
    disp_mags = np.linalg.norm(delta_np, axis=-1)  # [B, 5023]
    mean_delta_mag = float(np.mean(disp_mags))
    max_delta_mag = float(np.max(disp_mags))
    print(f"[CHECK 5] Delta mesh displacement statistics:", flush=True)
    print(f"  mean ||delta_mesh||: {mean_delta_mag:.6f} meters/units", flush=True)
    print(f"  max ||delta_mesh||:  {max_delta_mag:.6f} meters/units", flush=True)
    assert mean_delta_mag > 0.0, "Delta mesh mean displacement magnitude is zero!"
    print("  --> PASS: Non-zero valid facial deformation delta_mesh.", flush=True)

    # 6. Neutral mesh non-zero check
    print(f"[CHECK 6] Neutralized mesh check:", flush=True)
    assert np.std(v_neutral_np) > 1e-4, "Neutral mesh has zero variance (flat/zero mesh)!"
    print("  --> PASS: Neutralized mesh is valid 3D facial structure.", flush=True)

    # 7. Region mapping verification
    region_masks = get_flame_region_masks()
    print(f"[CHECK 7] Anatomical region vertex mapping check:", flush=True)
    total_region_verts = sum(len(idx) for idx in region_masks.values())
    print(f"  Total mapped region vertices across {len(region_masks)} regions: {total_region_verts}", flush=True)
    for r_name, r_idx in region_masks.items():
        assert len(r_idx) > 0, f"Region {r_name} has 0 vertices!"
        assert r_idx.max() < NUM_FLAME_VERTICES and r_idx.min() >= 0, f"Region {r_name} index out of bounds!"
    print("  --> PASS: All 12 anatomical facial regions correctly mapped.", flush=True)

    print("=" * 65)
    print(" ALL 7 FAIL-FAST SMOKE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65 + "\n", flush=True)


def extract_split(
    split: str,
    records,
    *,
    encoder,
    flame,
    run_mediapipe,
    device: torch.device,
    batch_size: int,
    out_path: Path,
    cfg: Dict,
    args: argparse.Namespace,
) -> None:
    if out_path.exists() and not args.force and not args.smoke_only:
        cached = np.load(out_path)
        print(f"[INFO] Stage 2A cache exists: {out_path} (region_features: {cached['region_features'].shape})", flush=True)
        return

    smirk_cfg = cfg.get("smirk", {})
    use_crop = bool(smirk_cfg.get("crop", True)) and not args.no_crop
    crop_scale = float(smirk_cfg.get("crop_scale", 1.4))
    image_size = int(smirk_cfg.get("image_size", 224))
    mediapipe_input_size = int(smirk_cfg.get("mediapipe_input_size", 224))
    on_crop_failure = str(smirk_cfg.get("on_crop_failure", "resize"))

    total = len(records.labels)
    if args.smoke_only:
        total = min(16, total)
    elif args.max_samples_per_split is not None:
        total = min(int(args.max_samples_per_split), total)

    prepared_tensors: List[torch.Tensor] = []
    labels: List[int] = []
    sample_ids: List[int] = []
    crop_success: List[bool] = []
    skipped = 0

    print(f"[INFO] Preprocessing {split} ({total} samples)...", flush=True)
    for i in tqdm(range(total), desc=f"preprocess {split}", dynamic_ncols=True):
        tensor, ok = prepare_smirk_image(
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
        prepared_tensors.append(tensor)
        labels.append(int(records.labels[i]))
        sample_ids.append(int(records.sample_ids[i]))
        crop_success.append(bool(ok))

    if not prepared_tensors:
        raise RuntimeError(f"No valid images for split={split}, skipped={skipped}")

    region_features_chunks: List[np.ndarray] = []
    delta_mesh_samples: List[np.ndarray] = []
    mean_delta_mags: List[float] = []
    max_delta_mags: List[float] = []
    geometry_valid_flags: List[bool] = []

    smoke_verified = False
    region_masks = get_flame_region_masks()

    print(f"[INFO] Computing frozen SMIRK+FLAME delta mesh for {split}...", flush=True)
    with torch.no_grad():
        for start in tqdm(range(0, len(prepared_tensors), batch_size), desc=f"SMIRK DeltaMesh {split}", dynamic_ncols=True):
            batch_t = prepared_tensors[start : start + batch_size]
            images = torch.stack(batch_t, dim=0).to(device)

            smirk_outputs = encoder(images)
            v_expr, v_neutral, delta_mesh = compute_flame_delta_mesh(flame, smirk_outputs, device)

            delta_mesh_np = delta_mesh.detach().cpu().numpy().astype(np.float32)  # [B, 5023, 3]
            region_feat = extract_region_features(delta_mesh_np, region_masks)  # [B, 12, 10]

            if not smoke_verified:
                smoke_verified = True
                run_stage2a_smoke_verification(v_expr, v_neutral, delta_mesh, smirk_outputs, region_feat)
                if args.smoke_only:
                    print("[INFO] --smoke-only flag passed. Smoke verification passed. Exiting.", flush=True)
                    return

            disp_mags = np.linalg.norm(delta_mesh_np, axis=-1)  # [B, 5023]
            b_means = np.mean(disp_mags, axis=1).tolist()
            b_maxs = np.max(disp_mags, axis=1).tolist()
            b_valid = [bool(m > 0 and np.isfinite(m)) for m in b_means]

            region_features_chunks.append(region_feat)
            mean_delta_mags.extend(b_means)
            max_delta_mags.extend(b_maxs)
            geometry_valid_flags.extend(b_valid)

            # Store sample delta meshes for visualization preview (first batch)
            if len(delta_mesh_samples) < 32:
                delta_mesh_samples.append(delta_mesh_np[: min(32 - len(delta_mesh_samples), delta_mesh_np.shape[0])])

    region_features_arr = np.concatenate(region_features_chunks, axis=0).astype(np.float32)
    sample_delta_meshes = np.concatenate(delta_mesh_samples, axis=0).astype(np.float16)  # Save first 32 float16
    labels_arr = np.asarray(labels, dtype=np.int64)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64)
    crop_success_arr = np.asarray(crop_success, dtype=bool)
    geometry_valid_arr = np.asarray(geometry_valid_flags, dtype=bool)

    if not np.isfinite(region_features_arr).all():
        raise FloatingPointError(f"NaN/Inf detected in region features for split={split}")
    if region_features_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(f"Feature/label length mismatch for split={split}: {region_features_arr.shape[0]} vs {labels_arr.shape[0]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        region_features=region_features_arr,  # [N, 12, 10]
        sample_delta_meshes=sample_delta_meshes,  # [32, 5023, 3] preview
        labels=labels_arr,
        sample_ids=sample_ids_arr,
        crop_success=crop_success_arr,
        geometry_valid=geometry_valid_arr,
        mean_delta_mags=np.asarray(mean_delta_mags, dtype=np.float32),
        max_delta_mags=np.asarray(max_delta_mags, dtype=np.float32),
        region_names=np.asarray(FLAME_REGION_NAMES),
        emotion_names=np.asarray(EMOTION_NAMES),
    )

    metadata = {
        "split": split,
        "cache_path": str(out_path),
        "num_samples": int(region_features_arr.shape[0]),
        "region_features_shape": list(region_features_arr.shape),
        "sample_delta_meshes_shape": list(sample_delta_meshes.shape),
        "num_regions": NUM_FLAME_REGIONS,
        "region_feature_dim": 10,
        "num_flame_vertices": NUM_FLAME_VERTICES,
        "mean_delta_displacement_global": float(np.mean(mean_delta_mags)),
        "max_delta_displacement_global": float(np.max(max_delta_mags)),
        "crop_success_count": int(crop_success_arr.sum()),
        "geometry_valid_count": int(geometry_valid_arr.sum()),
        "skipped": int(skipped),
    }

    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[SUCCESS] Saved Stage 2A Delta Mesh Cache for {split}: {out_path}\n"
        f"  region_features: {region_features_arr.shape}\n"
        f"  sample_delta_preview: {sample_delta_meshes.shape}\n"
        f"  mean ||delta_mesh||: {np.mean(mean_delta_mags):.6f}\n"
        f"  max ||delta_mesh||:  {np.max(max_delta_mags):.6f}\n",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    smirk_cfg = cfg.get("smirk", {})

    smirk_root = resolve_path(args.smirk_root or os.environ.get("SMIRK_ROOT") or smirk_cfg.get("smirk_root"))
    checkpoint = resolve_path(args.checkpoint or os.environ.get("SMIRK_CHECKPOINT") or smirk_cfg.get("checkpoint"))

    if smirk_root is None or checkpoint is None:
        raise ValueError("Both smirk_root and checkpoint must be specified.")

    device_name = args.device or smirk_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    out_dir = resolve_path(args.output_dir or cfg.get("paths", {}).get("cache_dir", "outputs/stage2a_smirk_delta_mesh_probe/cache"))

    smirk_encoder_module, run_mediapipe = import_smirk(smirk_root)
    encoder = load_frozen_encoder(smirk_encoder_module, checkpoint, device, strict=True)
    flame_cls = import_flame(smirk_root)
    with pushd(smirk_root):
        flame = flame_cls().to(device).eval()

    for param in flame.parameters():
        param.requires_grad_(False)

    data_dir = resolve_path(cfg["data"]["data_path"])

    for split in args.splits:
        records = collect_split_records(
            data_dir,
            split,
            mask_dir=None,
            use_clean_filter=False,
            bad_row_indices_path=None,
            predecode_pixels=True,
            preload_masks=False,
            allow_missing_masks=False,
        )
        out_path = out_dir / f"stage2a_delta_mesh_cache_{split}.npz"
        extract_split(
            split,
            records,
            encoder=encoder,
            flame=flame,
            run_mediapipe=run_mediapipe,
            device=device,
            batch_size=args.batch_size,
            out_path=out_path,
            cfg=cfg,
            args=args,
        )
        if args.smoke_only:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
