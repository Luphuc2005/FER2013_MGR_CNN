from __future__ import annotations

import argparse
import csv
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

from datasets.fer2013 import EMOTION_NAMES, collect_split_records
from scripts.extract_smirk_features import (

    load_frozen_encoder,
    prepare_smirk_image,
    pushd,
    resolve_path as resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FER2013 SMIRK 3D reconstruction grids for paper figures.")
    parser.add_argument("--config", type=str, default="config_smirk_only.yaml")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smirk-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--no-classifier", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixed-cam-scale", type=float, default=7.0)
    parser.add_argument("--use-smirk-cam", action="store_true", help="Use each sample's estimated SMIRK camera instead of one fixed camera.")
    parser.add_argument("--yaw-degrees", nargs=2, type=float, default=[-30.0, 30.0])
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--on-crop-failure", choices=("resize", "skip", "error"), default=None)
    parser.add_argument("--dpi-scale", type=int, default=2, help="Nearest-neighbor upscaling factor for the final paper grid.")
    return parser.parse_args()



def import_smirk_visual_stack(smirk_root: Path):
    if not smirk_root.exists():
        raise FileNotFoundError(
            f"SMIRK root not found: {smirk_root}. Clone https://github.com/georgeretsi/smirk and run quick_install.sh first."
        )
    sys.path.insert(0, str(smirk_root))
    with pushd(smirk_root):
        import src.smirk_encoder as smirk_encoder_module
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer
        from utils.mediapipe_utils import run_mediapipe
    return smirk_encoder_module, run_mediapipe, FLAME, Renderer


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


def select_balanced_indices(labels: np.ndarray, samples_per_class: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    labels = np.asarray(labels, dtype=np.int64)
    for class_id in range(len(EMOTION_NAMES)):
        indices = np.flatnonzero(labels == class_id)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        selected.extend(indices[: min(samples_per_class, indices.size)].tolist())
    return selected


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def resize_for_cell(img: np.ndarray, cell_size: int) -> np.ndarray:
    if img.shape[0] == cell_size and img.shape[1] == cell_size:
        return img
    return cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA if img.shape[0] > cell_size else cv2.INTER_CUBIC)


def add_caption(img: np.ndarray, caption: str, *, font_scale: float = 0.42) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = 30
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.78, out, 0.22, 0)
    cv2.putText(out, caption[:64], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_fixed_cam(batch_size: int, scale: float, device: torch.device) -> torch.Tensor:
    cam = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    cam[:, 0] = float(scale)
    return cam


def rotate_vertices_y(vertices: torch.Tensor, yaw_degrees: float) -> torch.Tensor:
    radians = torch.tensor(float(yaw_degrees) * np.pi / 180.0, dtype=vertices.dtype, device=vertices.device)
    cos_v = torch.cos(radians)
    sin_v = torch.sin(radians)
    rot = torch.stack(
        [
            torch.stack([cos_v, torch.zeros_like(cos_v), sin_v]),
            torch.stack([torch.zeros_like(cos_v), torch.ones_like(cos_v), torch.zeros_like(cos_v)]),
            torch.stack([-sin_v, torch.zeros_like(cos_v), cos_v]),
        ]
    )
    center = vertices.mean(dim=1, keepdim=True)
    return torch.matmul(vertices - center, rot.T) + center


def make_expression_only_params(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    expr_outputs = {k: v for k, v in outputs.items()}
    expr_outputs["shape_params"] = torch.zeros_like(outputs["shape_params"])
    if "pose_params" in outputs:
        expr_outputs["pose_params"] = torch.zeros_like(outputs["pose_params"])
    # Keep expression_params, jaw_params, and eyelid_params unchanged: this is the expression-only geometry.
    return expr_outputs


def render_mesh(renderer, vertices: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    return renderer.forward(vertices, cam)["rendered_img"]


def overlay_render(input_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    render_mask = (render_rgb.max(axis=2, keepdims=True) > 8).astype(np.float32)
    blended = input_rgb.astype(np.float32) * (1.0 - render_mask * 0.48) + render_rgb.astype(np.float32) * (render_mask * 0.85)
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_classifier_if_available(cfg: Dict, checkpoint_path: Optional[Path], feature_dim: int, disabled: bool):
    if disabled:
        return None, None
    output_dir = resolve_path(cfg.get("paths", {}).get("output_dir")) or PROJECT_ROOT / "outputs" / "smirk_only"
    checkpoint = checkpoint_path or (output_dir / "checkpoints" / "best_val_accuracy")
    if not checkpoint.with_suffix(".index").exists() and not checkpoint.exists():
        print(f"[WARNING] SMIRK classifier checkpoint not found: {checkpoint}. Prediction/confidence will be blank.", flush=True)
        return None, None

    import tensorflow as tf
    from scripts.train_smirk_classifier import build_model

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    model = build_model(feature_dim, cfg)
    model.load_weights(str(checkpoint))
    print(f"[INFO] Loaded SMIRK classifier checkpoint: {checkpoint}", flush=True)
    return model, tf


def predict_classifier(model, tf_module, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if model is None or tf_module is None:
        return np.full((features.shape[0],), -1, dtype=np.int64), np.full((features.shape[0],), np.nan, dtype=np.float32)
    logits = model(features.astype(np.float32), training=False)
    probs = tf_module.nn.softmax(tf_module.cast(logits, tf_module.float32), axis=-1).numpy()
    return probs.argmax(axis=1).astype(np.int64), probs.max(axis=1).astype(np.float32)


def stack_feature_blocks(outputs: Dict[str, torch.Tensor], keys: Sequence[str]) -> np.ndarray:
    blocks = [outputs[key].detach().float().cpu().numpy() for key in keys]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", {}).get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smirk_cfg = cfg.get("smirk", {})
    smirk_root = resolve_project_path(args.smirk_root or os.environ.get("SMIRK_ROOT") or smirk_cfg.get("smirk_root"))
    checkpoint = resolve_project_path(args.checkpoint or os.environ.get("SMIRK_CHECKPOINT") or smirk_cfg.get("checkpoint"))
    if smirk_root is None or checkpoint is None:
        raise ValueError("Both smirk_root and checkpoint must be configured or passed as arguments.")
    device_name = args.device or os.environ.get("SMIRK_DEVICE") or smirk_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SMIRK visualization, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)

    output_dir = resolve_path(args.output_dir) or ((resolve_path(cfg.get("paths", {}).get("output_dir")) or PROJECT_ROOT / "outputs" / "smirk_only") / "visualizations")
    per_render_dir = output_dir / "renders" / args.split
    grid_dir = output_dir / "grids"
    output_dir.mkdir(parents=True, exist_ok=True)

    smirk_encoder_module, run_mediapipe, flame_cls, renderer_cls = import_smirk_visual_stack(smirk_root)
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

    data_dir = resolve_project_path(cfg["data"]["data_path"])
    records = collect_split_records(
        data_dir,
        args.split,
        mask_dir=None,
        use_clean_filter=bool(cfg["data"].get("use_clean_filter", False)),
        bad_row_indices_path=cfg["data"].get("bad_row_indices_path"),
        predecode_pixels=bool(cfg["data"].get("predecode_pixels", True)),
        preload_masks=False,
        allow_missing_masks=False,
    )
    selected_indices = select_balanced_indices(records.labels, int(args.samples_per_class), seed)
    if not selected_indices:
        raise RuntimeError(f"No samples selected for split={args.split}.")

    prepared_tensors: List[torch.Tensor] = []
    labels: List[int] = []
    sample_ids: List[int] = []
    crop_success_flags: List[bool] = []
    skipped = 0
    for idx in tqdm(selected_indices, desc="preprocess selected FER", dynamic_ncols=True):
        tensor, crop_success = prepare_smirk_image(
            records.images[idx],
            run_mediapipe=run_mediapipe,
            use_crop=bool(smirk_cfg.get("crop", True)) and not args.no_crop,
            crop_scale=float(smirk_cfg.get("crop_scale", 1.4)),
            image_size=int(smirk_cfg.get("image_size", 224)),
            mediapipe_input_size=int(smirk_cfg.get("mediapipe_input_size", 224)),
            on_crop_failure=args.on_crop_failure or str(smirk_cfg.get("on_crop_failure", "resize")),
        )
        if tensor is None:
            skipped += 1
            continue
        prepared_tensors.append(tensor)
        labels.append(int(records.labels[idx]))
        sample_ids.append(int(records.sample_ids[idx]))
        crop_success_flags.append(bool(crop_success))

    if not prepared_tensors:
        raise RuntimeError(f"All selected samples were skipped. skipped={skipped}")

    feature_keys = list(smirk_cfg.get("feature_keys", ["expression_params", "eyelid_params", "jaw_params"]))
    all_rows: List[np.ndarray] = []
    index_rows: List[Dict[str, object]] = []
    first_feature_dim: Optional[int] = None
    classifier_model = None
    tf_module = None

    for batch_start in tqdm(range(0, len(prepared_tensors), int(args.batch_size)), desc="render SMIRK grids", dynamic_ncols=True):
        batch_tensors = prepared_tensors[batch_start : batch_start + int(args.batch_size)]
        batch_labels = labels[batch_start : batch_start + int(args.batch_size)]
        batch_ids = sample_ids[batch_start : batch_start + int(args.batch_size)]
        images = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            outputs = encoder(images)
            features = stack_feature_blocks(outputs, feature_keys)
            if first_feature_dim is None:
                first_feature_dim = int(features.shape[1])
                classifier_model, tf_module = load_classifier_if_available(
                    cfg,
                    resolve_path(args.classifier_checkpoint),
                    first_feature_dim,
                    args.no_classifier,
                )
            preds, confidences = predict_classifier(classifier_model, tf_module, features)

            flame_output = flame.forward(outputs)
            cam = outputs["cam"] if args.use_smirk_cam else make_fixed_cam(images.shape[0], args.fixed_cam_scale, device)
            frontal = render_mesh(renderer, flame_output["vertices"], cam)
            yaw_left = render_mesh(renderer, rotate_vertices_y(flame_output["vertices"], float(args.yaw_degrees[0])), cam)
            yaw_right = render_mesh(renderer, rotate_vertices_y(flame_output["vertices"], float(args.yaw_degrees[1])), cam)
            expr_only_output = flame.forward(make_expression_only_params(outputs))
            expr_only = render_mesh(renderer, expr_only_output["vertices"], cam)

        for i, (label, sample_id) in enumerate(zip(batch_labels, batch_ids)):
            true_name = EMOTION_NAMES[int(label)]
            pred = int(preds[i])
            pred_name = "NA" if pred < 0 else EMOTION_NAMES[pred]
            confidence = float(confidences[i]) if np.isfinite(confidences[i]) else float("nan")
            conf_text = "NA" if not np.isfinite(confidence) else f"{confidence:.3f}"
            title = f"idx {int(sample_id)} | true {true_name} | pred {pred_name} | conf {conf_text}"

            input_rgb = tensor_to_rgb_uint8(batch_tensors[i])
            frontal_rgb = tensor_to_rgb_uint8(frontal[i])
            yaw_left_rgb = tensor_to_rgb_uint8(yaw_left[i])
            yaw_right_rgb = tensor_to_rgb_uint8(yaw_right[i])
            expr_only_rgb = tensor_to_rgb_uint8(expr_only[i])
            overlay_rgb = overlay_render(input_rgb, frontal_rgb)

            columns = [
                ("Input", input_rgb),
                ("SMIRK frontal", frontal_rgb),
                (f"yaw {float(args.yaw_degrees[0]):+.0f} deg", yaw_left_rgb),
                (f"yaw {float(args.yaw_degrees[1]):+.0f} deg", yaw_right_rgb),
                ("expression-only", expr_only_rgb),
                ("overlay", overlay_rgb),
            ]
            captioned = [add_caption(resize_for_cell(img, 224), name) for name, img in columns]
            row = np.concatenate(captioned, axis=1)
            row = add_caption(row, title, font_scale=0.46)
            all_rows.append(row)

            sample_dir = per_render_dir / f"{int(sample_id):06d}_true_{true_name}_pred_{pred_name}"
            for name, img in columns:
                safe_name = name.replace(" ", "_").replace("+", "plus").replace("-", "minus").replace("deg", "deg")
                save_rgb(sample_dir / f"{safe_name}.png", img)
            np.savez_compressed(
                sample_dir / "smirk_params_and_feature.npz",
                feature=features[i],
                expression_params=outputs["expression_params"][i].detach().float().cpu().numpy(),
                eyelid_params=outputs["eyelid_params"][i].detach().float().cpu().numpy(),
                jaw_params=outputs["jaw_params"][i].detach().float().cpu().numpy(),
                label=np.asarray(label, dtype=np.int64),
                pred_smirk=np.asarray(pred, dtype=np.int64),
                confidence=np.asarray(confidence, dtype=np.float32),
                sample_id=np.asarray(sample_id, dtype=np.int64),
            )
            index_rows.append(
                {
                    "index": int(sample_id),
                    "true_label": int(label),
                    "true_name": true_name,
                    "pred_smirk": pred,
                    "pred_name": pred_name,
                    "confidence": confidence if np.isfinite(confidence) else "",
                    "crop_success": bool(crop_success_flags[batch_start + i]),
                    "render_dir": str(sample_dir),
                }
            )
            print(f"VIS_SAMPLE index={int(sample_id)} true={true_name} pred_smirk={pred_name} confidence={conf_text}", flush=True)

    grid = np.concatenate(all_rows, axis=0)
    if int(args.dpi_scale) > 1:
        grid = cv2.resize(grid, None, fx=int(args.dpi_scale), fy=int(args.dpi_scale), interpolation=cv2.INTER_NEAREST)
    grid_path = grid_dir / f"smirk_3d_grid_{args.split}_{int(args.samples_per_class)}perclass.png"
    save_rgb(grid_path, grid)

    csv_path = output_dir / f"visualization_index_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    json_path = output_dir / f"visualization_index_{args.split}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(index_rows, f, indent=2, ensure_ascii=False)

    print("SMIRK_3D_VISUALIZATION_DONE", flush=True)
    print(f"  split={args.split}", flush=True)
    print(f"  selected_samples={len(prepared_tensors)} skipped={skipped}", flush=True)
    print(f"  feature_shape=({len(prepared_tensors)}, {first_feature_dim})", flush=True)
    print(f"  grid={grid_path}", flush=True)
    print(f"  renders={per_render_dir}", flush=True)
    print(f"  index_csv={csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

