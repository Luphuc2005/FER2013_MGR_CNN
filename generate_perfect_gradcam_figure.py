#!/usr/bin/env python3
"""
Generate publication-quality Grad-CAM visualizations without vertical line artifacts.
Uses Stage 3 (14x14x512) feature maps instead of coarse Stage 4 (7x7).
Renders the exact 2x7 figure matching the user's paper format.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

tf.keras.mixed_precision.set_global_policy("float32")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from train import build_model

EMOTION_NAMES_TITLE = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def render_smooth_heatmap(cam_map: np.ndarray, target_size: int = 224, gamma: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    v_min = float(np.min(cam_map))
    v_max = float(np.percentile(cam_map, 99.5))
    denom = v_max - v_min
    if denom > 1e-8:
        cam_norm = np.clip((cam_map - v_min) / denom, 0.0, 1.0)
    else:
        cam_norm = np.zeros_like(cam_map)
        
    if gamma != 1.0 and np.max(cam_norm) > 0:
        cam_norm = np.power(cam_norm, gamma)
        
    cam_resized = cv2.resize(cam_norm.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    
    # Smooth Gaussian blur
    cam_smoothed = cv2.GaussianBlur(cam_resized, (15, 15), sigmaX=3.0)
    s_min = float(np.min(cam_smoothed))
    s_max = float(np.max(cam_smoothed))
    if (s_max - s_min) > 1e-8:
        cam_smoothed = (cam_smoothed - s_min) / (s_max - s_min)
        
    cam_u8 = (cam_smoothed * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb, cam_smoothed


def compute_stage3_gradcam(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    target_class_idx: int,
) -> np.ndarray:
    stage3_layer = model.backbone.stages[2][-1]
    box = {"value": None}
    
    with tf.GradientTape() as tape:
        original_call = stage3_layer.call
        def wrapped_call(*args, **kwargs):
            out = original_call(*args, **kwargs)
            tape.watch(out)
            box["value"] = out
            return out
        stage3_layer.call = wrapped_call
        try:
            outputs = model({"image": input_tensor}, training=False)
            logits = tf.cast(outputs["logits"] if isinstance(outputs, dict) else outputs[0], tf.float32)
            score = logits[:, target_class_idx]
            feat = box["value"]
        finally:
            stage3_layer.call = original_call
            
        grads = tape.gradient(score, feat)

    if grads is None:
        raise RuntimeError("Gradient backpropagation returned None!")

    grads = tf.cast(grads, tf.float32)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1))[0].numpy()
    return cam


def main():
    print("=" * 70)
    print(" GENERATING PERFECT STAGE 3 GRAD-CAM (NO VERTICAL STRIPES)")
    print("=" * 70)

    # 1. Load model
    cfg_path = PROJECT_ROOT / "config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
    cfg = load_config(str(cfg_path))
    cfg["runtime"]["allow_cpu_fallback"] = True
    cfg["runtime"]["min_gpus"] = 0
    cfg["runtime"]["use_mixed_precision"] = False
    
    print("[1/4] Building ConvNeXt AMGSA-FER model...")
    model = build_model(cfg)
    dummy_input = {"image": tf.zeros([1, 112, 112, 3], dtype=tf.float32)}
    _ = model(dummy_input, training=False)
    
    ckpt_prefix = str(PROJECT_ROOT / "checkpoints-down/ckpt-29")
    print(f"[2/4] Restoring weights from {ckpt_prefix}...")
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_prefix)
    status.expect_partial()
    print("      Model weights restored successfully!")

    # 2. Extract the 7 faces from the user's uploaded figure
    user_img_path = Path(r"C:/Users/ADMIN/.gemini/antigravity/brain/c60f1991-b5f6-4a82-b7ab-186ff1b3ac2a/.user_uploaded/media_1788792620274.png")
    user_img = cv2.imread(str(user_img_path))
    
    emo_order = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    emo_cls_map = {
        "Angry": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happy": 3,
        "Sad": 4,
        "Surprise": 5,
        "Neutral": 6,
    }
    
    x_starts = [124, 212, 300, 388, 476, 564, 652]
    
    output_dir = PROJECT_ROOT / "outputs/perfect_gradcam_stage3"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/4] Computing Stage 3 Grad-CAM for all 7 emotions...")
    results = {}
    
    for emo_name, x0 in zip(emo_order, x_starts):
        cls_idx = emo_cls_map[emo_name]
        # Crop 70x70 face tile from row 1 (Origin)
        face_crop = user_img[55:125, x0:x0+70]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_224 = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Prepare 112x112 input tensor
        face_112 = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_norm = (face_112.astype(np.float32) / 255.0 - MEAN) / STD
        inp_tensor = tf.convert_to_tensor(face_norm[None, ...], dtype=tf.float32)
        
        # Prediction info
        out = model({"image": inp_tensor}, training=False)
        logits = tf.cast(out["logits"] if isinstance(out, dict) else out[0], tf.float32)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        conf = float(probs[cls_idx])
        pred_idx = int(np.argmax(probs))
        
        # Compute Stage 3 CAM (14x14)
        cam_14 = compute_stage3_gradcam(model, inp_tensor, target_class_idx=cls_idx)
        heatmap_rgb, cam_smooth = render_smooth_heatmap(cam_14, target_size=224, gamma=0.85)
        overlay_rgb = cv2.addWeighted(face_224, 0.55, heatmap_rgb, 0.45, 0)
        
        # Save individual images
        emo_dir = output_dir / emo_name
        emo_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(emo_dir / "original.png"), cv2.cvtColor(face_224, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(emo_dir / "heatmap_stage3.png"), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(emo_dir / "overlay_stage3.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        
        results[emo_name] = {
            "face_224": face_224,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "conf": conf,
        }
        print(f"  - {emo_name:<9}: Stage 3 CAM generated successfully (Confidence: {conf*100:5.1f}%)")

    # 4. Generate Figures
    print("\n[4/4] Building High-Resolution Publication Deliverables...")
    
    # Figure A: Clean Publication 2x7
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.2), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    
    for c_idx, emo_name in enumerate(emo_order):
        d = results[emo_name]
        axes[0, c_idx].imshow(d["face_224"])
        axes[0, c_idx].set_title(emo_name, fontsize=12, fontweight="bold", pad=8)
        axes[0, c_idx].axis("off")
        
        axes[1, c_idx].imshow(d["overlay_rgb"])
        axes[1, c_idx].axis("off")
        
        if c_idx == 0:
            axes[0, 0].text(-0.15, 0.5, "Origin", transform=axes[0, 0].transAxes,
                           va="center", ha="right", fontsize=13, fontweight="bold")
            axes[1, 0].text(-0.15, 0.5, "Our model", transform=axes[1, 0].transAxes,
                           va="center", ha="right", fontsize=13, fontweight="bold")
                           
    pub_clean_png = output_dir / "figure_2x7_clean_publication.png"
    pub_clean_pdf = output_dir / "figure_2x7_clean_publication.pdf"
    fig.savefig(pub_clean_png, bbox_inches="tight", dpi=300)
    fig.savefig(pub_clean_pdf, bbox_inches="tight")
    plt.close(fig)

    # Figure B: Styled Card with Golden Rounded Frame (Matching user's report style)
    fig_card = plt.figure(figsize=(15, 5.2), dpi=300)
    ax_card = fig_card.add_subplot(111)
    ax_card.set_xlim(0, 15)
    ax_card.set_ylim(0, 5.2)
    ax_card.axis("off")
    
    # Draw golden rounded rectangle
    rect = FancyBboxPatch(
        (0.2, 0.3), 14.6, 4.6,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        edgecolor="#d49a37",
        facecolor="#fffaf0",
        linewidth=2.0,
    )
    ax_card.add_patch(rect)
    
    # Row titles
    ax_card.text(1.2, 3.6, "Origin", fontsize=15, fontweight="bold", ha="center", va="center", color="#222222")
    ax_card.text(1.2, 1.8, "Our model", fontsize=15, fontweight="bold", ha="center", va="center", color="#222222")
    
    tile_w = 1.6
    tile_h = 1.6
    x_base = 2.4
    x_gap = 1.75
    
    for c_idx, emo_name in enumerate(emo_order):
        d = results[emo_name]
        xc = x_base + c_idx * x_gap
        
        # Column title
        ax_card.text(xc + tile_w / 2, 4.7, emo_name, fontsize=14, fontweight="bold", ha="center", va="center", color="#111111")
        
        # Origin image
        ax_orig = fig_card.add_axes([ (xc) / 15.0, (2.8) / 5.2, tile_w / 15.0, tile_h / 5.2 ])
        ax_orig.imshow(d["face_224"])
        ax_orig.axis("off")
        
        # Overlay image
        ax_over = fig_card.add_axes([ (xc) / 15.0, (1.0) / 5.2, tile_w / 15.0, tile_h / 5.2 ])
        ax_over.imshow(d["overlay_rgb"])
        ax_over.axis("off")
        
    styled_png = output_dir / "figure_8_styled_report.png"
    styled_pdf = output_dir / "figure_8_styled_report.pdf"
    fig_card.savefig(styled_png, bbox_inches="tight", dpi=300)
    fig_card.savefig(styled_pdf, bbox_inches="tight")
    plt.close(fig_card)

    print("\n" + "=" * 70)
    print(" ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f" 1. Styled Report Figure (Hình 8): {styled_png.resolve()}")
    print(f" 2. Clean Publication Figure     : {pub_clean_png.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
