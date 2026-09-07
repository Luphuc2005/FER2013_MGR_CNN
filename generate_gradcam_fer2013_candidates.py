#!/usr/bin/env python3
"""
Grad-CAM Candidate Generator for FER2013 Paper Visualization.
Compares Baseline ConvNeXt-Base MS1M vs Ours (Adaptive SigLIP2 Confusion-Aware).

Features:
1. Restores both model checkpoints and strictly verifies restoration.
2. Evaluates FER2013 test set with identical preprocessing as evaluation.
3. Automatically identifies samples where BOTH models predict correctly.
4. Selects up to 20 high-confidence, sharp, diverse candidates per emotion.
5. Computes Grad-CAM at the last spatial feature map before GAP/classifier.
6. Automatically inspects and validates the target layer (aborts if spatial dim <= 1).
7. Targets ground-truth class, normalizes heatmap to [0, 1], overlays with identical colormap/alpha.
8. Generates individual images, per-emotion 20x3 contact sheets, metadata_all.csv, and summary.txt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional Matplotlib setup for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Force pure CPU execution by default before TensorFlow initializes
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Ensure no GPU devices are visible to TensorFlow
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, SplitRecords, collect_split_records, make_dataset
from train import build_model


# Standard 7 FER emotions in Title case
EMOTION_NAMES_TITLE = [name.title() for name in EMOTION_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Candidates on FER2013 for Paper")
    parser.add_argument(
        "--baseline-ckpt",
        default="/home/ptbao/projects/FER2013_MGR_CNN/outputs/tf_runs/convnext_base_ms1m_raw_paper_aug/checkpoints/best/ckpt-25",
        help="Path/prefix to Baseline checkpoint",
    )
    parser.add_argument(
        "--ours-ckpt",
        default="/home/ptbao/projects/FER2013_MGR_CNN/outputs/papers/siglip2-confusion/checkpoints/best/ckpt-29",
        help="Path/prefix to Ours checkpoint",
    )
    parser.add_argument(
        "--baseline-config",
        default="config_convnext_base_ms1m_raw_paper_aug.yaml",
        help="Path to Baseline config YAML",
    )
    parser.add_argument(
        "--ours-config",
        default="config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml",
        help="Path to Ours config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ptbao/projects/FER2013_MGR_CNN/outputs/gradcam_fer2013_candidates",
        help="Output directory for Grad-CAM images and metadata",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=20,
        help="Target number of candidate samples per emotion class",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for test set evaluation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.42,
        help="Heatmap overlay alpha blending factor",
    )
    parser.add_argument(
        "--colormap",
        default="jet",
        help="Colormap for Grad-CAM heatmap visualization",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=True,
        help="Force pure CPU execution (default: True)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU execution if explicitly requested",
    )
    return parser.parse_args()


# =========================================================================
# Checkpoint & Model Utilities
# =========================================================================

def resolve_checkpoint_prefix(path_str: str) -> str:
    """Normalize checkpoint path, resolving .index suffix and verifying shard existence."""
    p = Path(path_str)
    if str(p).endswith(".index"):
        p = Path(str(p)[:-6])
    
    # Check if exact path exists
    index_file = Path(f"{p}.index")
    if not index_file.exists():
        # Try relative to PROJECT_ROOT
        alt_p = PROJECT_ROOT / p
        if Path(f"{alt_p}.index").exists():
            p = alt_p
            index_file = Path(f"{p}.index")
        else:
            raise FileNotFoundError(
                f"[ERROR] Checkpoint index not found: {index_file} (or {PROJECT_ROOT / index_file.name})"
            )
    
    data_files = list(p.parent.glob(f"{p.name}.data-*"))
    if not data_files:
        raise FileNotFoundError(f"[ERROR] No checkpoint data shard(s) found for prefix: {p}")
    
    return str(p)


def resolve_config_file(config_path_str: str) -> Path:
    """Resolve configuration file path."""
    p = Path(config_path_str)
    if p.is_file():
        return p
    alt = PROJECT_ROOT / p
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"[ERROR] Config file not found: {config_path_str} (or {alt})")


def restore_and_verify_model(
    cfg: Dict[str, Any],
    ckpt_prefix: str,
    model_name_tag: str,
) -> tf.keras.Model:
    """Build model, run dummy forward pass, restore checkpoint, and verify match."""
    print(f"\n[{model_name_tag}] Initializing architecture from config...")
    
    # Disable strict pretrained requirement if checkpoint is being restored
    if "model" in cfg:
        cfg["model"]["convnext_base_require_pretrained"] = False
    
    model = build_model(cfg)
    
    # Forward pass dummy tensor to instantiate weights
    img_size = int(cfg.get("data", {}).get("image_size", 112))
    channels = int(cfg.get("data", {}).get("channels", 3))
    dummy_input = {"image": tf.zeros([1, img_size, img_size, channels], dtype=tf.float32)}
    _ = model(dummy_input, training=False)
    
    num_weights_before = len(model.weights)
    if num_weights_before == 0:
        raise RuntimeError(f"[{model_name_tag}] Model instantiated with 0 weights!")
    
    print(f"[{model_name_tag}] Model built. Trainable variables: {len(model.trainable_variables)}, Total weights: {num_weights_before}")
    print(f"[{model_name_tag}] Restoring checkpoint: {ckpt_prefix}")
    
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_prefix)
    
    # expect_partial allows ignoring optimizer / step / epoch variables stored in checkpoint
    status.expect_partial()
    
    # Verify that existing model variables were matched from checkpoint
    try:
        status.assert_existing_objects_matched()
        print(f"[{model_name_tag}] RESTORE SUCCESS: All model weights matched checkpoint!")
    except Exception as exc:
        raise RuntimeError(
            f"[{model_name_tag}] CHECKPOINT RESTORE FAILED: Model weights could not be matched completely!\nDetails: {exc}"
        ) from exc
    
    return model


# =========================================================================
# Grad-CAM Layer Inspection
# =========================================================================

@contextmanager
def capture_layer_output(layer: tf.keras.layers.Layer, tape: Optional[tf.GradientTape] = None):
    """Context manager to intercept and watch the output tensor of a layer."""
    original_call = layer.call
    box = {"value": None}

    def wrapped_call(*args, **kwargs):
        out = original_call(*args, **kwargs)
        if tape is not None and tf.is_tensor(out):
            tape.watch(out)
        box["value"] = out
        return out

    layer.call = wrapped_call
    try:
        yield box
    finally:
        layer.call = original_call


def is_valid_4d_feature_map(tensor: Any) -> bool:
    """Verify that tensor is a 4D feature map with valid spatial dimensions H > 1, W > 1."""
    if not tf.is_tensor(tensor):
        return False
    if tensor.shape.rank != 4:
        return False
    h = int(tensor.shape[1] or 0)
    w = int(tensor.shape[2] or 0)
    return h > 1 and w > 1


def inspect_and_find_gradcam_layer(
    model: tf.keras.Model,
    sample_inputs: Dict[str, tf.Tensor],
    model_name_tag: str,
) -> Tuple[tf.keras.layers.Layer, Tuple[int, ...]]:
    """
    Inspect model to find the last spatial feature map of ConvNeXt before Global Average Pooling.
    Verifies that layer produces a 4D tensor with H > 1, W > 1 and receives non-zero gradients.
    """
    print(f"\n[{model_name_tag}] Inspecting model to locate final spatial feature map before GAP...")
    
    # 1. Primary architecture candidates based on ConvNeXtBaseFaceFERBaseline design:
    # If ECA is enabled and present: stage4_eca feeds into GAP
    # Otherwise: the last block of stage4 of backbone feeds into GAP
    candidate_layers: List[Tuple[str, tf.keras.layers.Layer]] = []
    
    if getattr(model, "use_eca", False) and getattr(model, "stage4_eca", None) is not None:
        candidate_layers.append(("model.stage4_eca", model.stage4_eca))
    
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages") and len(model.backbone.stages) >= 4:
        stage4_blocks = model.backbone.stages[3]
        if stage4_blocks:
            last_block = stage4_blocks[-1]
            candidate_layers.append((f"backbone.stage4_blocks[-1] ({last_block.name})", last_block))
    
    # 2. General recursive search across all layers as fallback / verification
    def collect_layers(obj, seen=None):
        if seen is None:
            seen = set()
        res = []
        if id(obj) in seen:
            return res
        seen.add(id(obj))
        if isinstance(obj, tf.keras.layers.Layer):
            res.append(obj)
        for v in getattr(obj, "__dict__", {}).values():
            if isinstance(v, tf.keras.layers.Layer):
                res.extend(collect_layers(v, seen))
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, tf.keras.layers.Layer):
                        res.extend(collect_layers(item, seen))
        return res

    all_layers = collect_layers(model)
    exclude_types = (
        tf.keras.layers.Dense,
        tf.keras.layers.GlobalAveragePooling2D,
        tf.keras.layers.GlobalMaxPooling2D,
        tf.keras.layers.Dropout,
        tf.keras.layers.Flatten,
        tf.keras.layers.Softmax,
    )
    
    for l in all_layers:
        if not isinstance(l, exclude_types):
            lname = str(getattr(l, "name", ""))
            if "stage4" in lname or "block" in lname or "eca" in lname:
                if (lname, l) not in candidate_layers:
                    candidate_layers.append((lname, l))
    
    # Test candidate layers
    verified_candidates = []
    for desc, layer in candidate_layers:
        try:
            with capture_layer_output(layer) as cap:
                _ = model(sample_inputs, training=False)
            feat = cap["value"]
            if is_valid_4d_feature_map(feat):
                shape = tuple(int(dim) if dim is not None else 1 for dim in feat.shape)
                verified_candidates.append((layer, shape, desc))
        except Exception:
            continue
    
    if not verified_candidates:
        raise RuntimeError(
            f"[{model_name_tag}] FATAL: Could not find any 4D feature map layer before GAP! "
            f"Model inspection found 0 valid spatial layers."
        )
    
    # The first candidate in verified_candidates is the primary architectural candidate before GAP
    selected_layer, selected_shape, selected_desc = verified_candidates[0]
    
    # Verify spatial dimension > 1
    h, w = selected_shape[1], selected_shape[2]
    if h <= 1 or w <= 1:
        raise RuntimeError(
            f"[{model_name_tag}] FATAL: Selected Grad-CAM layer '{selected_layer.name}' "
            f"has invalid spatial dimensions ({h}x{w}). Must be > 1 to avoid fake heatmaps!"
        )
    
    # Verify GradientTape backpropagation on sample input
    with tf.GradientTape() as tape:
        with capture_layer_output(selected_layer, tape=tape) as cap:
            out = model(sample_inputs, training=False)
            logits = out["logits"] if isinstance(out, dict) else out[0]
            feature = cap["value"]
            loss = tf.reduce_mean(logits[:, 0])
        grads = tape.gradient(loss, feature)
    
    if grads is None:
        raise RuntimeError(
            f"[{model_name_tag}] FATAL: GradientTape returned None gradients for layer '{selected_layer.name}'. "
            f"Gradients cannot backpropagate from classifier to this layer!"
        )
    
    print(f"[{model_name_tag}] Selected Grad-CAM Layer: '{selected_layer.name}'")
    print(f"[{model_name_tag}]   Type: {type(selected_layer).__name__}")
    print(f"[{model_name_tag}]   Feature Map Tensor Shape: {selected_shape} (Spatial: {h}x{w}, Channels: {selected_shape[3]})")
    print(f"[{model_name_tag}]   Gradient Verification: PASSED (grad shape: {tuple(grads.shape)})")
    
    return selected_layer, selected_shape


# =========================================================================
# Grad-CAM Computation
# =========================================================================

def denormalize_image(image_tensor: tf.Tensor, channels: int = 3) -> np.ndarray:
    """Denormalize preprocessed image tensor back to [0, 1] RGB."""
    image = tf.cast(image_tensor, tf.float32)
    if channels == 3:
        mean = tf.constant([0.485, 0.456, 0.406], tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], tf.float32)
    else:
        mean = tf.constant([0.5], tf.float32)
        std = tf.constant([0.5], tf.float32)
    image = image * std + mean
    image = tf.clip_by_value(image, 0.0, 1.0)
    arr = image.numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def compute_gradcam(
    model: tf.keras.Model,
    inputs: Dict[str, tf.Tensor],
    target_layer: tf.keras.layers.Layer,
    target_class_idx: int,
) -> Tuple[np.ndarray, float, int]:
    """
    Compute Grad-CAM for a given sample and target class.
    
    Returns:
        heatmap_norm: 2D numpy array [H, W] normalized to [0, 1]
        confidence: predicted probability for predicted class
        pred_idx: predicted class index
    """
    with tf.GradientTape() as tape:
        with capture_layer_output(target_layer, tape=tape) as cap:
            outputs = model(inputs, training=False)
            logits = tf.cast(outputs["logits"] if isinstance(outputs, dict) else outputs[0], tf.float32)
            probs = tf.nn.softmax(logits, axis=-1)
            pred_idx = int(tf.argmax(probs[0]).numpy())
            conf = float(probs[0, pred_idx].numpy())
            
            feature = cap["value"]
            if not is_valid_4d_feature_map(feature):
                raise RuntimeError(
                    f"Captured target feature map is invalid: {type(feature)} "
                    f"shape={getattr(feature, 'shape', None)}"
                )
            feature = tf.cast(feature, tf.float32)
            # Target score: ground-truth class logit
            score = logits[:, target_class_idx]
        
        grads = tape.gradient(score, feature)
    
    if grads is None:
        raise RuntimeError(f"GradientTape returned None for target layer {target_layer.name}!")
    
    grads = tf.cast(grads, tf.float32)
    # Global average pooling of gradients over spatial dimensions (H, W) -> [1, 1, 1, C]
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    # Channel-weighted sum of feature map -> [1, H, W]
    cam = tf.reduce_sum(weights * feature, axis=-1)
    # Apply ReLU: only features that have a positive influence
    cam = tf.nn.relu(cam)[0]
    
    # Normalize heatmap strictly to [0, 1]
    cam_min = tf.reduce_min(cam)
    cam_max = tf.reduce_max(cam)
    denom = cam_max - cam_min
    if float(denom.numpy()) > 1e-8:
        heatmap_norm = ((cam - cam_min) / denom).numpy()
    else:
        heatmap_norm = tf.zeros_like(cam).numpy()
        
    return heatmap_norm, conf, pred_idx


def resize_heatmap_to_image(heatmap: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize 2D heatmap [H, W] to target image dimensions using bilinear interpolation."""
    h, w = target_hw
    hm = tf.convert_to_tensor(heatmap[..., None], dtype=tf.float32)
    hm = tf.image.resize(hm, [h, w], method="bilinear")
    hm = tf.clip_by_value(hm, 0.0, 1.0)
    return np.squeeze(hm.numpy(), axis=-1)


def generate_heatmap_overlay(
    image01: np.ndarray,
    heatmap01: np.ndarray,
    alpha: float = 0.42,
    colormap_name: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate colormapped heatmap and overlay on original image.
    image01: [H, W, 3] in [0, 1]
    heatmap01: [H, W] in [0, 1]
    Returns: (heat_rgb, overlay) both [H, W, 3] in [0, 1]
    """
    cmap = plt.get_cmap(colormap_name)
    heat_rgba = cmap(np.clip(heatmap01, 0.0, 1.0))
    heat_rgb = heat_rgba[..., :3]
    overlay = np.clip((1.0 - alpha) * image01 + alpha * heat_rgb, 0.0, 1.0)
    return heat_rgb, overlay


# =========================================================================
# Candidate Selection (Confidence, Clarity, Diversity)
# =========================================================================

def compute_image_clarity(img01: np.ndarray) -> float:
    """Compute image sharpness/contrast score using intensity variance and gradient energy."""
    gray = np.dot(img01[..., :3], [0.2989, 0.5870, 0.1140])
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    grad_energy = float(np.mean(dx ** 2) + np.mean(dy ** 2))
    var_energy = float(np.var(gray))
    return grad_energy * 10.0 + var_energy


def select_best_candidates_for_class(
    candidate_indices: List[int],
    base_confs: np.ndarray,
    ours_confs: np.ndarray,
    preprocessed_images: List[np.ndarray],
    target_count: int = 20,
) -> List[int]:
    """
    Select up to target_count candidate samples prioritizing:
    1. Both models correct (already filtered).
    2. High average confidence between Baseline and Ours.
    3. Image clarity / sharpness.
    4. Visual diversity across selected samples.
    """
    n_cand = len(candidate_indices)
    if n_cand <= target_count:
        return candidate_indices

    cand_indices_arr = np.array(candidate_indices, dtype=np.int64)
    
    # 1. Confidence score: geometric mean / average confidence
    avg_conf = 0.5 * (base_confs[cand_indices_arr] + ours_confs[cand_indices_arr])
    
    # 2. Image clarity score
    clarity_scores = np.array([
        compute_image_clarity(preprocessed_images[i]) for i in candidate_indices
    ], dtype=np.float32)
    
    # Normalize scores to [0, 1]
    conf_min, conf_max = avg_conf.min(), avg_conf.max()
    conf_norm = (avg_conf - conf_min) / (conf_max - conf_min + 1e-8)
    
    clar_min, clar_max = clarity_scores.min(), clarity_scores.max()
    clar_norm = (clarity_scores - clar_min) / (clar_max - clar_min + 1e-8)
    
    quality_score = 0.65 * conf_norm + 0.35 * clar_norm
    
    # 3. Downsampled feature representation for diversity (16x16 grayscale)
    feat_vectors = []
    for idx in candidate_indices:
        img = preprocessed_images[idx]
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        # Simple block-averaging to 16x16
        h, w = gray.shape
        gh, gw = h // 16, w // 16
        small = gray[:gh*16, :gw*16].reshape(16, gh, 16, gw).mean(axis=(1, 3))
        vec = small.flatten()
        norm = np.linalg.norm(vec) + 1e-8
        feat_vectors.append(vec / norm)
    feat_matrix = np.stack(feat_vectors, axis=0)  # [N_cand, 256]
    
    # Greedy diversity selection:
    # Pick highest quality candidate first
    selected_local = [int(np.argmax(quality_score))]
    
    while len(selected_local) < target_count:
        # For each unselected candidate, compute min cosine distance to selected set
        unselected = [i for i in range(n_cand) if i not in selected_local]
        best_cand = None
        best_combined = -1e9
        
        for cand in unselected:
            sims = np.dot(feat_matrix[selected_local], feat_matrix[cand])
            min_dist = float(1.0 - np.max(sims))  # Cosine distance to nearest selected
            combined_rank = quality_score[cand] + 0.40 * min_dist
            if combined_rank > best_combined:
                best_combined = combined_rank
                best_cand = cand
                
        if best_cand is not None:
            selected_local.append(best_cand)
        else:
            break
            
    # Sort selected samples by quality score descending
    selected_local.sort(key=lambda idx: quality_score[idx], reverse=True)
    return [candidate_indices[idx] for idx in selected_local]


# =========================================================================
# Contact Sheet Generator
# =========================================================================

def build_contact_sheet(
    sample_records: List[Dict[str, Any]],
    emotion_name: str,
    output_path: Path,
):
    """
    Generate a contact sheet PNG:
    - Rows: up to 20 samples
    - 3 Columns: Column 1 = Original, Column 2 = Baseline Grad-CAM, Column 3 = Ours Grad-CAM
    - Row metadata: Sample ID, Index, Baseline Conf, Ours Conf
    """
    num_rows = len(sample_records)
    if num_rows == 0:
        return
        
    fig, axes = plt.subplots(num_rows, 3, figsize=(10.0, max(2.5 * num_rows, 4.0)), dpi=180)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for r_idx, rec in enumerate(sample_records):
        ax_orig = axes[r_idx, 0]
        ax_base = axes[r_idx, 1]
        ax_ours = axes[r_idx, 2]
        
        ax_orig.imshow(rec["original_img"])
        ax_base.imshow(rec["baseline_overlay"])
        ax_ours.imshow(rec["ours_overlay"])
        
        for ax in (ax_orig, ax_base, ax_ours):
            ax.set_xticks([])
            ax.set_yticks([])
            
        row_label = (
            f"#{rec['sample_id']}\n"
            f"idx:{rec['dataset_index']}\n"
            f"B:{rec['baseline_confidence']:.2f}\n"
            f"O:{rec['ours_confidence']:.2f}"
        )
        ax_orig.set_ylabel(row_label, rotation=0, labelpad=34, va="center", fontsize=8, fontweight="bold")
        
        if r_idx == 0:
            ax_orig.set_title("Original Image", fontsize=11, fontweight="bold", pad=8)
            ax_base.set_title("Baseline Grad-CAM", fontsize=11, fontweight="bold", pad=8)
            ax_ours.set_title("Ours Grad-CAM", fontsize=11, fontweight="bold", pad=8)
            
    fig.suptitle(f"Grad-CAM Candidates - Emotion: {emotion_name} ({num_rows} Samples)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{emotion_name}] Saved contact sheet: {output_path}")


# =========================================================================
# Main Execution Pipeline
# =========================================================================

def main() -> int:
    args = parse_args()
    
    print("=" * 75)
    print(" FER2013 Grad-CAM Candidate Generator (Paper Visualization)")
    print(" Baseline vs Ours (Adaptive SigLIP2 Confusion-Aware)")
    print("=" * 75)
    
    # 0. Runtime Environment Configuration (Default: 100% Pure CPU)
    use_cpu = not args.gpu or args.cpu or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        cpu_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, cpu_threads // 2))
        except Exception:
            pass
        print(f"[INFO] Running in 100% pure CPU mode (CUDA_VISIBLE_DEVICES=-1, threads={cpu_threads}).")
        print("[INFO] GPU allocations: NONE (No VRAM consumed, zero conflict with running GPU jobs).")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[INFO] Detected {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
                
    # Explicit float32 precision for accurate gradient computations
    tf.keras.mixed_precision.set_global_policy("float32")
    
    # 1. Resolve Checkpoints & Configs
    base_ckpt_prefix = resolve_checkpoint_prefix(args.baseline_ckpt)
    ours_ckpt_prefix = resolve_checkpoint_prefix(args.ours_ckpt)
    base_cfg_path = resolve_config_file(args.baseline_config)
    ours_cfg_path = resolve_config_file(args.ours_config)
    
    print(f"\n[CONFIG] Baseline Config : {base_cfg_path}")
    print(f"[CONFIG] Baseline CKPT   : {base_ckpt_prefix}")
    print(f"[CONFIG] Ours Config     : {ours_cfg_path}")
    print(f"[CONFIG] Ours CKPT       : {ours_ckpt_prefix}")
    
    base_cfg = load_config(base_cfg_path)
    ours_cfg = load_config(ours_cfg_path)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] Output Directory: {output_dir}")
    
    # 2. Build & Restore Models
    base_model = restore_and_verify_model(base_cfg, base_ckpt_prefix, "Baseline")
    ours_model = restore_and_verify_model(ours_cfg, ours_ckpt_prefix, "Ours")
    
    # 3. Inspect Grad-CAM Target Layers
    img_size = int(base_cfg.get("data", {}).get("image_size", 112))
    channels = int(base_cfg.get("data", {}).get("channels", 3))
    sample_inputs = {"image": tf.zeros([1, img_size, img_size, channels], dtype=tf.float32)}
    
    base_target_layer, base_shape = inspect_and_find_gradcam_layer(base_model, sample_inputs, "Baseline")
    ours_target_layer, ours_shape = inspect_and_find_gradcam_layer(ours_model, sample_inputs, "Ours")
    
    # 4. Load FER2013 Test Set
    data_path = base_cfg["data"]["data_path"]
    print(f"\n[DATASET] Loading FER2013 test set from: {data_path}")
    test_records = collect_split_records(
        data_path,
        "test",
        mask_dir=base_cfg["data"].get("mask_dir"),
        use_clean_filter=False,
        bad_row_indices_path=None,
        mask_ablation="none",
        predecode_pixels=True,
        preload_masks=False,
        allow_missing_masks=True,
    )
    total_test_samples = len(test_records.labels)
    print(f"[DATASET] Total FER2013 test samples: {total_test_samples}")
    
    test_ds = make_dataset(
        test_records,
        base_cfg,
        split="test",
        training=False,
        replicas=1,
    )
    
    # 5. Full Evaluation Forward Pass
    print("\n[EVALUATION] Running test set predictions for both models...")
    all_y_true: List[int] = []
    base_preds: List[int] = []
    base_confs: List[float] = []
    ours_preds: List[int] = []
    ours_confs: List[float] = []
    denorm_images: List[np.ndarray] = []
    raw_input_batches: List[Dict[str, tf.Tensor]] = []
    
    step = 0
    for batch_inputs, batch_labels in test_ds:
        b_img = batch_inputs["image"]
        # Baseline forward
        b_out = base_model(batch_inputs, training=False)
        b_logits = tf.cast(b_out["logits"] if isinstance(b_out, dict) else b_out[0], tf.float32)
        b_prob = tf.nn.softmax(b_logits, axis=-1).numpy()
        b_p = np.argmax(b_prob, axis=-1)
        b_c = np.max(b_prob, axis=-1)
        
        # Ours forward
        o_out = ours_model(batch_inputs, training=False)
        o_logits = tf.cast(o_out["logits"] if isinstance(o_out, dict) else o_out[0], tf.float32)
        o_prob = tf.nn.softmax(o_logits, axis=-1).numpy()
        o_p = np.argmax(o_prob, axis=-1)
        o_c = np.max(o_prob, axis=-1)
        
        y_t = batch_labels.numpy().astype(int)
        
        all_y_true.extend(y_t.tolist())
        base_preds.extend(b_p.tolist())
        base_confs.extend(b_c.tolist())
        ours_preds.extend(o_p.tolist())
        ours_confs.extend(o_c.tolist())
        
        # Denormalize images in batch
        for i in range(b_img.shape[0]):
            denorm_images.append(denormalize_image(b_img[i], channels=channels))
            
        raw_input_batches.append(batch_inputs)
        step += 1
        if step % 25 == 0 or len(all_y_true) == total_test_samples:
            print(f"  Processed {len(all_y_true)}/{total_test_samples} samples...", flush=True)
            
    y_true_arr = np.array(all_y_true, dtype=np.int64)
    base_preds_arr = np.array(base_preds, dtype=np.int64)
    base_confs_arr = np.array(base_confs, dtype=np.float32)
    ours_preds_arr = np.array(ours_preds, dtype=np.int64)
    ours_confs_arr = np.array(ours_confs, dtype=np.float32)
    
    base_acc = float(np.mean(y_true_arr == base_preds_arr) * 100)
    ours_acc = float(np.mean(y_true_arr == ours_preds_arr) * 100)
    both_correct_mask = (y_true_arr == base_preds_arr) & (y_true_arr == ours_preds_arr)
    both_correct_count = int(np.sum(both_correct_mask))
    
    print("\n" + "=" * 75)
    print("                     EVALUATION SUMMARY")
    print("=" * 75)
    print(f" Baseline Accuracy : {base_acc:6.2f}%")
    print(f" Ours Accuracy     : {ours_acc:6.2f}%")
    print(f" Both Correct      : {both_correct_count}/{total_test_samples} ({both_correct_count / total_test_samples * 100:.2f}%)")
    print("-" * 75)
    
    # 6. Candidate Selection per Emotion
    print("[SELECTION] Selecting top candidate samples per emotion where BOTH models are correct...")
    selected_by_class: Dict[int, List[int]] = {}
    found_counts: Dict[str, int] = {}
    
    for cls_idx, cls_name in enumerate(EMOTION_NAMES):
        cls_title = cls_name.title()
        cand_indices = np.where((y_true_arr == cls_idx) & both_correct_mask)[0].tolist()
        n_cand = len(cand_indices)
        found_counts[cls_title] = n_cand
        
        chosen = select_best_candidates_for_class(
            candidate_indices=cand_indices,
            base_confs=base_confs_arr,
            ours_confs=ours_confs_arr,
            preprocessed_images=denorm_images,
            target_count=args.samples_per_class,
        )
        selected_by_class[cls_idx] = chosen
        print(f"  {cls_title:<9}: {len(chosen)} selected / {n_cand} available matching samples")
    
    # 7. Compute Grad-CAM for Selected Samples and Save Artifacts
    print("\n[GRAD-CAM] Computing Grad-CAM heatmaps and generating visualization candidates...")
    metadata_rows: List[Dict[str, Any]] = []
    
    for cls_idx, cls_name in enumerate(EMOTION_NAMES):
        cls_title = cls_name.title()
        cls_dir = output_dir / cls_title
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        chosen_indices = selected_by_class[cls_idx]
        sample_records_for_sheet: List[Dict[str, Any]] = []
        
        for rank_num, sample_idx in enumerate(chosen_indices, start=1):
            sample_id_str = f"{rank_num:03d}"
            
            # Extract sample inputs
            one_sample_records = SplitRecords(
                images=test_records.images[sample_idx : sample_idx + 1],
                labels=test_records.labels[sample_idx : sample_idx + 1],
                sample_ids=test_records.sample_ids[sample_idx : sample_idx + 1],
                mask_paths=None if test_records.mask_paths is None else test_records.mask_paths[sample_idx : sample_idx + 1],
                masks=None if test_records.masks is None else test_records.masks[sample_idx : sample_idx + 1],
                bboxes=None if test_records.bboxes is None else test_records.bboxes[sample_idx : sample_idx + 1],
            )
            cfg_single = copy.deepcopy(base_cfg)
            cfg_single["runtime"]["batch_size_per_gpu"] = 1
            cfg_single["runtime"]["prefetch_buffer"] = 1
            one_ds = make_dataset(one_sample_records, cfg_single, split="test", training=False, replicas=1)
            single_input, _ = next(iter(one_ds))
            
            image01 = denorm_images[sample_idx]
            h_img, w_img = image01.shape[:2]
            
            # Baseline Grad-CAM (Target = Ground Truth Class)
            base_cam_small, b_conf, b_pred = compute_gradcam(
                base_model, single_input, base_target_layer, target_class_idx=cls_idx
            )
            base_cam_full = resize_heatmap_to_image(base_cam_small, (h_img, w_img))
            _, base_overlay = generate_heatmap_overlay(
                image01, base_cam_full, alpha=args.alpha, colormap_name=args.colormap
            )
            
            # Ours Grad-CAM (Target = Ground Truth Class)
            ours_cam_small, o_conf, o_pred = compute_gradcam(
                ours_model, single_input, ours_target_layer, target_class_idx=cls_idx
            )
            ours_cam_full = resize_heatmap_to_image(ours_cam_small, (h_img, w_img))
            _, ours_overlay = generate_heatmap_overlay(
                image01, ours_cam_full, alpha=args.alpha, colormap_name=args.colormap
            )
            
            # File paths
            orig_img_path = cls_dir / f"{sample_id_str}_original.png"
            base_cam_path = cls_dir / f"{sample_id_str}_baseline_cam.png"
            ours_cam_path = cls_dir / f"{sample_id_str}_ours_cam.png"
            
            # Save individual PNGs pixel-for-pixel (lossless PNG)
            Image.fromarray((np.clip(image01, 0.0, 1.0) * 255).astype(np.uint8)).save(orig_img_path)
            Image.fromarray((np.clip(base_overlay, 0.0, 1.0) * 255).astype(np.uint8)).save(base_cam_path)
            Image.fromarray((np.clip(ours_overlay, 0.0, 1.0) * 255).astype(np.uint8)).save(ours_cam_path)
            
            # Get original image path or identifier if available
            orig_path_str = ""
            if isinstance(test_records.images[sample_idx], str) and not test_records.images[sample_idx].startswith(("", " ")):
                orig_path_str = str(test_records.images[sample_idx])
            
            # Append record
            record_item = {
                "emotion": cls_title,
                "sample_id": sample_id_str,
                "dataset_index": int(sample_idx),
                "image_path": orig_path_str,
                "ground_truth": cls_title,
                "baseline_pred": EMOTION_NAMES_TITLE[b_pred],
                "baseline_confidence": round(float(b_conf), 4),
                "ours_pred": EMOTION_NAMES_TITLE[o_pred],
                "ours_confidence": round(float(o_conf), 4),
                "baseline_correct": bool(b_pred == cls_idx),
                "ours_correct": bool(o_pred == cls_idx),
                "gradcam_layer_baseline": base_target_layer.name,
                "gradcam_layer_ours": ours_target_layer.name,
                "original_img": image01,
                "baseline_overlay": base_overlay,
                "ours_overlay": ours_overlay,
            }
            sample_records_for_sheet.append(record_item)
            
            meta_row = dict(record_item)
            meta_row.pop("original_img")
            meta_row.pop("baseline_overlay")
            meta_row.pop("ours_overlay")
            metadata_rows.append(meta_row)
            
        # Generate contact sheet for this emotion
        sheet_path = cls_dir / f"contact_sheet_{cls_name.lower()}.png"
        build_contact_sheet(sample_records_for_sheet, cls_title, sheet_path)
        
    # 8. Save metadata_all.csv
    csv_path = output_dir / "metadata_all.csv"
    csv_columns = [
        "emotion",
        "sample_id",
        "dataset_index",
        "image_path",
        "ground_truth",
        "baseline_pred",
        "baseline_confidence",
        "ours_pred",
        "ours_confidence",
        "baseline_correct",
        "ours_correct",
        "gradcam_layer_baseline",
        "gradcam_layer_ours",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)
    print(f"\n[OUTPUT] Saved metadata CSV: {csv_path} ({len(metadata_rows)} entries)")
    
    # 9. Save summary.txt
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write(" GRAD-CAM CANDIDATES SELECTION SUMMARY FOR FER2013 PAPER\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Timestamp               : {tf.timestamp().numpy()}\n")
        f.write(f"Total Test Samples      : {total_test_samples}\n")
        f.write(f"Baseline Checkpoint     : {base_ckpt_prefix}\n")
        f.write(f"Baseline Accuracy       : {base_acc:.2f}%\n")
        f.write(f"Baseline Grad-CAM Layer : {base_target_layer.name} (Tensor Shape: {base_shape})\n\n")
        f.write(f"Ours Checkpoint         : {ours_ckpt_prefix}\n")
        f.write(f"Ours Accuracy           : {ours_acc:.2f}%\n")
        f.write(f"Ours Grad-CAM Layer     : {ours_target_layer.name} (Tensor Shape: {ours_shape})\n\n")
        f.write(f"Both Correct Samples    : {both_correct_count}/{total_test_samples} ({both_correct_count / total_test_samples * 100:.2f}%)\n")
        f.write(f"Colormap & Alpha        : {args.colormap} (alpha={args.alpha})\n")
        f.write(f"Target Samples per Class: {args.samples_per_class}\n\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Emotion':<12} {'Available Both Correct':<25} {'Selected Samples':<20} {'Status':<15}\n")
        f.write("-" * 75 + "\n")
        for cls_name in EMOTION_NAMES:
            c_title = cls_name.title()
            c_idx = EMOTION_NAMES.index(cls_name)
            chosen_cnt = len(selected_by_class[c_idx])
            avail_cnt = found_counts[c_title]
            status = "FULL (20/20)" if chosen_cnt >= args.samples_per_class else f"MAX_FOUND ({chosen_cnt})"
            f.write(f"{c_title:<12} {avail_cnt:<25} {chosen_cnt:<20} {status:<15}\n")
        f.write("-" * 75 + "\n\n")
        f.write("Output Structure:\n")
        f.write(f"  {output_dir}/\n")
        f.write("    metadata_all.csv\n")
        f.write("    summary.txt\n")
        for cls_name in EMOTION_NAMES:
            c_title = cls_name.title()
            f.write(f"    {c_title}/\n")
            f.write(f"      001_original.png ... 020_original.png\n")
            f.write(f"      001_baseline_cam.png ... 020_baseline_cam.png\n")
            f.write(f"      001_ours_cam.png ... 020_ours_cam.png\n")
            f.write(f"      contact_sheet_{cls_name.lower()}.png\n")
    print(f"[OUTPUT] Saved summary report: {summary_path}")

    # 10. Required Final Terminal Output
    print("\n" + "=" * 75)
    print("                      FINAL EXECUTION REPORT")
    print("=" * 75)
    print("1. RESTORED CHECKPOINTS:")
    print(f"   - Baseline : {base_ckpt_prefix} (STATUS: OK)")
    print(f"   - Ours     : {ours_ckpt_prefix} (STATUS: OK)")
    print()
    print("2. GRAD-CAM LAYERS & TENSOR SHAPES:")
    print(f"   - Baseline Layer : '{base_target_layer.name}' (Shape: {base_shape}, Spatial: {base_shape[1]}x{base_shape[2]})")
    print(f"   - Ours Layer     : '{ours_target_layer.name}' (Shape: {ours_shape}, Spatial: {ours_shape[1]}x{ours_shape[2]})")
    print()
    print("3. SAMPLES FOUND & SELECTED PER EMOTION:")
    for cls_name in EMOTION_NAMES:
        c_title = cls_name.title()
        c_idx = EMOTION_NAMES.index(cls_name)
        cnt = len(selected_by_class[c_idx])
        avail = found_counts[c_title]
        note = "" if cnt >= args.samples_per_class else f" (Max found: {cnt} due to both-correct condition)"
        print(f"   - {c_title:<9}: {cnt}/{args.samples_per_class} selected (available: {avail}){note}")
    print()
    print("4. FINAL OUTPUT DIRECTORY:")
    print(f"   {output_dir.resolve()}")
    print("=" * 75 + "\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
