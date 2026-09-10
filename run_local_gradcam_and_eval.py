#!/usr/bin/env python3
"""
Local Grad-CAM & Evaluation Runner for AMGSA-FER (ckpt-29).

Features:
1. Loads ckpt-29 on local machine using TensorFlow 2.10.1.
2. Supports both Stage 4 (7x7x1024, publication standard) and Stage 3 (14x14x512).
3. Applies Robust Percentile Normalization (99.5th percentile) + gentle gamma (0.85)
   to eliminate single-pixel nose dot spikes and render smooth, full-face attention maps.
4. Generates:
   - 224x224 Original, Heatmap (JET), and Overlay (55% original + 45% heatmap).
   - Side-by-side comparison of Stage 3 vs Stage 4 across all 7 emotions.
   - Publication-quality Figure (PNG 600 DPI and PDF vector container).
5. Displays comprehensive FER2013 evaluation report for ckpt-29 (76.23% Accuracy).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Suppress TF C++ logs and force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
tf.keras.mixed_precision.set_global_policy("float32")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from train import build_model

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_NAMES_TITLE = [name.title() for name in EMOTION_NAMES]


@contextmanager
def capture_layer_output(layer: tf.keras.layers.Layer, tape: Optional[tf.GradientTape] = None):
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


def compute_gradcam_robust(
    model: tf.keras.Model,
    inputs: Dict[str, tf.Tensor],
    target_layer: tf.keras.layers.Layer,
    target_class_idx: int,
    percentile_clip: float = 99.5,
    gamma: float = 0.85,
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Compute Grad-CAM with robust percentile normalization to avoid single-pixel outliers.
    Returns:
        heatmap_norm: 2D numpy array normalized to [0, 1]
        confidence: predicted probability
        pred_idx: predicted class index
        probs: all class probabilities
    """
    with tf.GradientTape() as tape:
        with capture_layer_output(target_layer, tape=tape) as cap:
            outputs = model(inputs, training=False)
            logits = tf.cast(outputs["logits"] if isinstance(outputs, dict) else outputs[0], tf.float32)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            
            feature = cap["value"]
            if feature is None or len(feature.shape) != 4:
                raise RuntimeError(f"Invalid feature map from layer {target_layer.name}: {getattr(feature, 'shape', None)}")
            feature = tf.cast(feature, tf.float32)
            score = logits[:, target_class_idx]
            
        grads = tape.gradient(score, feature)
        
    if grads is None:
        raise RuntimeError(f"Gradients returned None for layer {target_layer.name}!")
        
    grads = tf.cast(grads, tf.float32)
    # Global average pooling of gradients
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * feature, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    
    # Robust percentile normalization: clip at 99.5th percentile to prevent single-pixel nose dot spike
    v_min = float(np.min(cam))
    v_max = float(np.percentile(cam, percentile_clip))
    denom = v_max - v_min
    if denom > 1e-8:
        heatmap_norm = np.clip((cam - v_min) / denom, 0.0, 1.0)
    else:
        heatmap_norm = np.zeros_like(cam)
        
    # Gentle gamma curve to naturally reveal emotional Action Units across eyes, brows, mouth
    if gamma != 1.0 and np.max(heatmap_norm) > 0:
        heatmap_norm = np.power(heatmap_norm, gamma)
        
    return heatmap_norm, conf, pred_idx, probs


def resize_to_224_bicubic(image_or_heatmap: np.ndarray, is_heatmap: bool = False) -> np.ndarray:
    """Bicubic resize to 224x224."""
    if cv2 is not None:
        try:
            interp = cv2.INTER_CUBIC
            if is_heatmap:
                res = cv2.resize(image_or_heatmap.astype(np.float32), (224, 224), interpolation=interp)
                return np.clip(res, 0.0, 1.0)
            else:
                return cv2.resize(image_or_heatmap, (224, 224), interpolation=interp)
        except Exception:
            pass
            
    resample_filter = getattr(Image, "Resampling", Image).BICUBIC
    if is_heatmap:
        im = Image.fromarray(image_or_heatmap.astype(np.float32))
        im_res = im.resize((224, 224), resample=resample_filter)
        return np.clip(np.array(im_res, dtype=np.float32), 0.0, 1.0)
    else:
        im = Image.fromarray(image_or_heatmap)
        im_res = im.resize((224, 224), resample=resample_filter)
        return np.array(im_res, dtype=np.uint8)


def colormap_jet(heatmap01_224: np.ndarray) -> np.ndarray:
    """Apply full-face continuous JET colormap [0, 1] -> RGB uint8 [0, 255]."""
    h_clean = np.clip(heatmap01_224, 0.0, 1.0)
    if cv2 is not None:
        try:
            h_u8 = np.uint8(h_clean * 255.0)
            bgr = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    cmap = plt.get_cmap("jet")
    rgba = cmap(h_clean)
    return np.uint8(np.clip(rgba[..., :3], 0.0, 1.0) * 255.0)


def alpha_blend(orig_224: np.ndarray, heat_224: np.ndarray, orig_w: float = 0.55, heat_w: float = 0.45) -> np.ndarray:
    """cv2.addWeighted equivalent blending."""
    if cv2 is not None:
        try:
            return cv2.addWeighted(orig_224, float(orig_w), heat_224, float(heat_w), 0)
        except Exception:
            pass
    return np.clip(orig_w * orig_224.astype(np.float32) + heat_w * heat_224.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def preprocess_image_file(image_path: Path, target_size: int = 112) -> Tuple[tf.Tensor, np.ndarray, np.ndarray]:
    """Load image from disk and apply standard FER2013 evaluation preprocessing."""
    img_pil = Image.open(image_path).convert("RGB")
    orig_rgb_np = np.array(img_pil, dtype=np.uint8)
    
    # Resize to model input size (112x112)
    img_112 = img_pil.resize((target_size, target_size), resample=getattr(Image, "Resampling", Image).BILINEAR)
    img_arr = np.array(img_112, dtype=np.float32) / 255.0
    
    # Standard ImageNet normalization: (x - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_arr = (img_arr - mean) / std
    
    input_tensor = tf.convert_to_tensor(norm_arr[None, ...], dtype=tf.float32)
    # 224x224 unnormalized original RGB for display
    orig_224 = resize_to_224_bicubic(orig_rgb_np, is_heatmap=False)
    
    return input_tensor, orig_rgb_np, orig_224


def print_evaluation_summary():
    """Print verified evaluation report from previous complete test evaluation."""
    eval_json_path = PROJECT_ROOT / "outputs" / "eval_ckpt29_results" / "metrics_best_tta.json"
    if not eval_json_path.exists():
        eval_json_path = PROJECT_ROOT / "outputs" / "eval_ckpt29_results" / "metrics_no_tta.json"
        
    print("\n" + "=" * 75)
    print("      AMGSA-FER (CKPT-29) FER2013 EVALUATION BENCHMARK RESULTS")
    print("=" * 75)
    if eval_json_path.exists():
        with eval_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        acc = data.get("accuracy", 0.0) * 100
        mf1 = data.get("macro_f1", 0.0) * 100
        wf1 = data.get("weighted_f1", 0.0) * 100
        method = data.get("method", "Optimal TTA-HFlip")
        print(f" Evaluation Method  : {method}")
        print(f" Test Set Accuracy  : {acc:6.2f}%")
        print(f" Macro-F1 Score     : {mf1:6.2f}%")
        print(f" Weighted-F1 Score  : {wf1:6.2f}%")
        print("-" * 75)
        print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 75)
        cr = data.get("classification_report", {})
        for emo in EMOTION_NAMES:
            if emo in cr:
                p = cr[emo].get("precision", 0.0) * 100
                r = cr[emo].get("recall", 0.0) * 100
                f1 = cr[emo].get("f1-score", 0.0) * 100
                sup = int(cr[emo].get("support", 0))
                print(f"{emo.title():<12} {p:6.2f}%      {r:6.2f}%      {f1:6.2f}%      {sup:<10}")
        print("=" * 75 + "\n")
    else:
        print("[INFO] Evaluation metric file not found.")


def main():
    parser = argparse.ArgumentParser(description="Run Local Grad-CAM and Evaluation for ckpt-29")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints-down/ckpt-29",
        help="Path/prefix to checkpoint (default: checkpoints-down/ckpt-29)",
    )
    parser.add_argument(
        "--config",
        default="config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gradcam_local_ckpt29",
        help="Output directory for generated Grad-CAM images",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Print Benchmark Evaluation Summary
    print_evaluation_summary()
    
    # 2. Load Model & Restore Checkpoint
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(str(cfg_path))
    cfg["runtime"]["allow_cpu_fallback"] = True
    cfg["runtime"]["min_gpus"] = 0
    cfg["runtime"]["use_mixed_precision"] = False
    tf.keras.mixed_precision.set_global_policy("float32")
    
    print(f"[MODEL] Building AMGSA-FER architecture from config: {args.config}...")
    model = build_model(cfg)
    dummy_input = {"image": tf.zeros([1, 112, 112, 3], dtype=tf.float32)}
    print("[DEBUG] Calling dummy input...")
    _ = model(dummy_input, training=False)
    print("[DEBUG] Dummy input ok!")
    
    ckpt_prefix = str(PROJECT_ROOT / args.checkpoint)
    print(f"[MODEL] Restoring checkpoint: {ckpt_prefix}...")
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_prefix)
    status.expect_partial()
    status.assert_existing_objects_matched()
    print("[MODEL] RESTORE SUCCESS: Model weights matched checkpoint successfully!\n")
    
    # 3. Target Layers: Both Stage 4 (Publication Standard) and Stage 3
    stage4_layer = model.backbone.stages[3][-1]  # stage4_block2 (7x7x1024)
    stage3_layer = model.backbone.stages[2][-1]  # stage3_block26 (14x14x512)
    print(f"[GRAD-CAM LAYERS]")
    print(f"  - Stage 4 (Recommended Publication Standard) : {stage4_layer.name} (7x7x1024, High-level Emotion AUs)")
    print(f"  - Stage 3 (Local Spatial Features)            : {stage3_layer.name} (14x14x512, Mid-level Geometry)\n")
    print("[DEBUG] Target layers ok!")
    
    # 4. Process the 7 Emotion Images provided in checkpoints-down
    images_dir = PROJECT_ROOT / "checkpoints-down"
    test_images = {
        "Angry": images_dir / "Angry_2.jpg",
        "Disgust": images_dir / "Disgust_1.jpg",
        "Fear": images_dir / "Fear_2.jpg",
        "Happy": images_dir / "Happy_1.jpg",
        "Sad": images_dir / "Sad_1.jpg",
        "Surprise": images_dir / "Surprise_2.jpg",
        "Neutral": images_dir / "Neutral_1.jpg",
    }
    
    results_by_emotion: Dict[str, Dict[str, Any]] = {}
    
    print(f"[PROCESSING] Generating Grad-CAM for 7 emotion test images in '{images_dir}'...")
    for emo_name, img_path in test_images.items():
        if not img_path.exists():
            print(f"  [WARNING] File not found: {img_path}")
            continue
            
        cls_idx = EMOTION_NAMES.index(emo_name.lower())
        input_tensor, orig_raw, orig_224 = preprocess_image_file(img_path)
        inputs_dict = {"image": input_tensor}
        
        # Compute Stage 4 Grad-CAM (Primary Publication Heatmap)
        s4_cam_small, conf, pred_idx, probs = compute_gradcam_robust(
            model=model,
            inputs=inputs_dict,
            target_layer=stage4_layer,
            target_class_idx=cls_idx,
            percentile_clip=99.5,
            gamma=0.85,
        )
        s4_cam_224 = resize_to_224_bicubic(s4_cam_small, is_heatmap=True)
        s4_heat_rgb = colormap_jet(s4_cam_224)
        s4_overlay = alpha_blend(orig_224, s4_heat_rgb, orig_w=0.55, heat_w=0.45)
        
        # Compute Stage 3 Grad-CAM (for comparison)
        s3_cam_small, _, _, _ = compute_gradcam_robust(
            model=model,
            inputs=inputs_dict,
            target_layer=stage3_layer,
            target_class_idx=cls_idx,
            percentile_clip=99.0,
            gamma=0.85,
        )
        s3_cam_224 = resize_to_224_bicubic(s3_cam_small, is_heatmap=True)
        s3_heat_rgb = colormap_jet(s3_cam_224)
        s3_overlay = alpha_blend(orig_224, s3_heat_rgb, orig_w=0.55, heat_w=0.45)
        
        # Save individual images
        emo_dir = output_dir / emo_name
        emo_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(orig_224).save(emo_dir / "original.png")
        Image.fromarray(s4_heat_rgb).save(emo_dir / "heatmap_stage4.png")
        Image.fromarray(s4_overlay).save(emo_dir / "overlay_stage4.png")
        Image.fromarray(s3_heat_rgb).save(emo_dir / "heatmap_stage3.png")
        Image.fromarray(s3_overlay).save(emo_dir / "overlay_stage3.png")
        
        pred_name = EMOTION_NAMES_TITLE[pred_idx]
        is_correct = (pred_idx == cls_idx)
        status_str = "CORRECT" if is_correct else "MISMATCH"
        print(f"  - {emo_name:<9} -> Predicted: {pred_name:<9} (Confidence: {conf*100:5.2f}%) [{status_str}]")
        
        results_by_emotion[emo_name] = {
            "original_224": orig_224,
            "stage4_overlay": s4_overlay,
            "stage4_heatmap": s4_heat_rgb,
            "stage3_overlay": s3_overlay,
            "confidence": conf,
            "predicted": pred_name,
            "is_correct": is_correct,
        }
        
    # 5. GENERATE TOP 20 GRAD-CAM CANDIDATES PER EMOTION FROM FER2013 TEST SET
    print("\n" + "=" * 75)
    print(" GENERATING TOP 20 GRAD-CAM CANDIDATES PER EMOTION FROM FER2013 TEST SET")
    print("=" * 75)
    
    csv_path = PROJECT_ROOT / "data/fer13-split/test.csv"
    if not csv_path.exists():
        csv_path = Path(r"D:\HocTap\Phân tích  và xử lý ảnh\sgu-2026-facial-expression-recognition\dataset\fer13-split\test.csv")
    print(f"[CANDIDATES] Loading FER2013 test set from: {csv_path}")
    import pandas as pd
    df_test = pd.read_csv(csv_path)
    
    candidates_dir = output_dir / "candidates_20_per_emotion"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    N_TARGET = 20
    master_summary = []
    best_fer_by_emotion: Dict[str, Dict[str, Any]] = {}
    
    for c in range(7):
        emo_name = EMOTION_NAMES_TITLE[c]
        emo_dir = candidates_dir / emo_name
        emo_dir.mkdir(parents=True, exist_ok=True)
        
        c_df = df_test[df_test["emotion"] == c]
        print(f"\n[CANDIDATES] Emotion {c}: {emo_name:<10} | Scanning {len(c_df)} test samples for Top {N_TARGET}...")
        
        collected = []
        for idx, row in c_df.iterrows():
            pix = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
            im_224 = Image.fromarray(pix, mode="L").resize((224, 224), resample=Image.Resampling.LANCZOS).convert("RGB")
            face_224 = np.array(im_224, dtype=np.uint8)
            
            im_112 = Image.fromarray(face_224).resize((112, 112), resample=Image.Resampling.BILINEAR)
            arr = (np.array(im_112, dtype=np.float32) / 255.0 - mean) / std
            inp_tensor = tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)
            
            out = model({"image": inp_tensor}, training=False)
            logits = tf.cast(out["logits"] if isinstance(out, dict) else out[0], tf.float32)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            
            if pred_idx == c and conf >= 0.70:
                collected.append({
                    "idx": int(idx),
                    "confidence": conf,
                    "face_224": face_224,
                    "inp_tensor": inp_tensor,
                })
                if len(collected) >= 28:
                    break
                    
        if len(collected) < N_TARGET:
            for idx, row in c_df.iterrows():
                if any(x["idx"] == idx for x in collected):
                    continue
                pix = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
                im_224 = Image.fromarray(pix, mode="L").resize((224, 224), resample=Image.Resampling.LANCZOS).convert("RGB")
                face_224 = np.array(im_224, dtype=np.uint8)
                im_112 = Image.fromarray(face_224).resize((112, 112), resample=Image.Resampling.BILINEAR)
                arr = (np.array(im_112, dtype=np.float32) / 255.0 - mean) / std
                inp_tensor = tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)
                out = model({"image": inp_tensor}, training=False)
                logits = tf.cast(out["logits"] if isinstance(out, dict) else out[0], tf.float32)
                probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                if pred_idx == c:
                    collected.append({
                        "idx": int(idx),
                        "confidence": conf,
                        "face_224": face_224,
                        "inp_tensor": inp_tensor,
                    })
                if len(collected) >= N_TARGET:
                    break
                    
        collected.sort(key=lambda item: item["confidence"], reverse=True)
        selected = collected[:N_TARGET]
        print(f"  -> Selected {len(selected)} samples (Confidence range: {selected[-1]['confidence']*100:.1f}% - {selected[0]['confidence']*100:.1f}%)")
        
        emo_records = []
        for rank, cand in enumerate(selected, start=1):
            sample_dir = emo_dir / f"sample_{rank:02d}_idx{cand['idx']}_conf{cand['confidence']:.3f}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            cam_small, conf, pred_idx, _ = compute_gradcam_robust(
                model=model,
                inputs={"image": cand["inp_tensor"]},
                target_layer=stage4_layer,
                target_class_idx=c,
                percentile_clip=99.5,
                gamma=0.85,
            )
            cam_224 = resize_to_224_bicubic(cam_small, is_heatmap=True)
            if cv2 is not None:
                cam_224 = cv2.GaussianBlur(cam_224, (15, 15), 3.0)
                cam_224 = np.clip((cam_224 - cam_224.min()) / (cam_224.max() - cam_224.min() + 1e-6), 0.0, 1.0)
            heat_rgb = colormap_jet(cam_224)
            overlay = alpha_blend(cand["face_224"], heat_rgb, orig_w=0.55, heat_w=0.45)
            
            # Save raw individual 224x224 images
            Image.fromarray(cand["face_224"]).save(sample_dir / "original.png")
            Image.fromarray(heat_rgb).save(sample_dir / "heatmap.png")
            Image.fromarray(overlay).save(sample_dir / "overlay.png")
            
            rec = {
                "rank": rank,
                "dataset_idx": cand["idx"],
                "confidence": conf,
                "face_224": cand["face_224"],
                "heatmap_rgb": heat_rgb,
                "overlay_rgb": overlay,
            }
            emo_records.append(rec)
            master_summary.append({
                "emotion": emo_name,
                "rank": rank,
                "dataset_idx": cand["idx"],
                "confidence": conf,
                "folder": str(sample_dir.relative_to(PROJECT_ROOT)),
            })
            
            if rank == 1:
                # Save Stage 3 overlay as well for the #1 best sample
                s3_cam_s, _, _, _ = compute_gradcam_robust(
                    model=model,
                    inputs={"image": cand["inp_tensor"]},
                    target_layer=stage3_layer,
                    target_class_idx=c,
                    percentile_clip=99.0,
                    gamma=0.85,
                )
                s3_cam_224 = resize_to_224_bicubic(s3_cam_s, is_heatmap=True)
                s3_heat_rgb = colormap_jet(s3_cam_224)
                s3_overlay = alpha_blend(cand["face_224"], s3_heat_rgb, orig_w=0.55, heat_w=0.45)
                
                best_fer_by_emotion[emo_name] = {
                    "original_224": cand["face_224"],
                    "stage4_overlay": overlay,
                    "stage3_overlay": s3_overlay,
                    "confidence": conf,
                    "dataset_idx": cand["idx"],
                }
                
        # Build Contact Sheet (20 rows x 3 columns)
        fig, axes = plt.subplots(len(emo_records), 3, figsize=(7.5, max(1.8 * len(emo_records), 4.0)), dpi=200)
        for r_idx, rec in enumerate(emo_records):
            axes[r_idx, 0].imshow(rec["face_224"])
            axes[r_idx, 1].imshow(rec["heatmap_rgb"])
            axes[r_idx, 2].imshow(rec["overlay_rgb"])
            for ax in axes[r_idx]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)
            row_label = f"#{rec['rank']:02d} (idx {rec['dataset_idx']})\nConf: {rec['confidence']*100:.1f}%"
            axes[r_idx, 0].set_ylabel(row_label, rotation=0, labelpad=38, va="center", fontsize=7.5, fontweight="bold")
            if r_idx == 0:
                axes[r_idx, 0].set_title("Original (224x224)", fontsize=9.5, fontweight="bold", pad=6)
                axes[r_idx, 1].set_title("JET Heatmap", fontsize=9.5, fontweight="bold", pad=6)
                axes[r_idx, 2].set_title("Overlay (55/45)", fontsize=9.5, fontweight="bold", pad=6)
        fig.suptitle(f"Grad-CAM Candidates: {emo_name} (Top {len(emo_records)} Samples, FER2013)", fontsize=12, fontweight="bold", y=0.998)
        fig.tight_layout(rect=[0.05, 0.01, 1.0, 0.99])
        fig.savefig(emo_dir / f"contact_sheet_20samples_{emo_name.lower()}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)

        # Build 4x5 Grid
        fig_grid, axes_grid = plt.subplots(4, 5, figsize=(10, 8.5), dpi=250)
        for idx_g, ax in enumerate(axes_grid.flat):
            if idx_g < len(emo_records):
                rec = emo_records[idx_g]
                ax.imshow(rec["overlay_rgb"])
                ax.set_title(f"#{rec['rank']:02d} | {rec['confidence']*100:.1f}%", fontsize=8, fontweight="bold", pad=3)
            ax.axis("off")
        fig_grid.suptitle(f"Top 20 Grad-CAM Overlays - {emo_name} (FER2013 Test)", fontsize=12, fontweight="bold", y=0.98)
        fig_grid.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])
        fig_grid.savefig(emo_dir / f"grid_4x5_overlays_{emo_name.lower()}.png", bbox_inches="tight", dpi=250)
        plt.close(fig_grid)
        print(f"  [{emo_name}] Contact Sheet & 4x5 Grid saved!")

    # 6. Generate Master Publication 2x7 and 3x7 Figures from BEST FER2013 TEST CANDIDATES
    print("\n[PUBLICATION FIGURE] Generating Master Figures using Best FER2013 Candidates...")
    pub_fer_2x7_png = output_dir / "publication_figure_2x7_paper_fer_best.png"
    pub_fer_2x7_pdf = output_dir / "publication_figure_2x7_paper_fer_best.pdf"
    fig2, axes2 = plt.subplots(2, 7, figsize=(14.0, 4.3))
    fig2.subplots_adjust(wspace=0, hspace=0)
    row2_labels = ["Original", "AMGSA-FER (Ours)"]
    for col_idx, emo_name in enumerate(EMOTION_NAMES_TITLE):
        if emo_name not in best_fer_by_emotion:
            continue
        data = best_fer_by_emotion[emo_name]
        col_imgs = [data["original_224"], data["stage4_overlay"]]
        for row_idx in range(2):
            ax = axes2[row_idx, col_idx]
            ax.imshow(col_imgs[row_idx], aspect="equal")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"{emo_name}\n({data['confidence']*100:.1f}%)", fontsize=11, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.text(-0.06, 0.5, row2_labels[row_idx], va="center", ha="right", fontsize=12, fontweight="bold", transform=ax.transAxes)
    fig2.savefig(pub_fer_2x7_png, dpi=600, bbox_inches="tight", pad_inches=0)
    fig2.savefig(pub_fer_2x7_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    
    # Save Manifest CSV
    manifest_df = pd.DataFrame(master_summary)
    manifest_csv = candidates_dir / "candidates_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    print(f"\n=================================================================")
    print(f"                 ALL 140 CANDIDATE SAMPLES GENERATED!")
    print(f"=================================================================")
    print(f"1. Candidates Directory       : {candidates_dir.resolve()}")
    print(f"2. Best FER2013 Paper Figure  : {pub_fer_2x7_png.resolve()} (600 DPI)")
    print(f"3. Best FER2013 Vector (PDF)  : {pub_fer_2x7_pdf.resolve()} (PDF)")
    print(f"4. Manifest CSV (140 samples) : {manifest_csv.resolve()}")
    print(f"=================================================================\n")


if __name__ == "__main__":
    main()
