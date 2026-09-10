"""
Fast and Robust Grad-CAM 20-Candidates Generator per Emotion on FER2013 Test Set.
Model: ConvNeXt-Base MS1M AMGSA-FER (ckpt-29).
Target Layer: Stage 4 ConvNeXt-Base (stage4_block2, 7x7x1024).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Pure CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES
from train import build_model


EMOTION_NAMES_TITLE = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def capture_layer_output(layer: tf.keras.layers.Layer, tape: tf.GradientTape):
    class Hook:
        def __init__(self):
            self.value = None
            self.orig_call = layer.call

        def __enter__(self):
            def hooked_call(*args, **kwargs):
                val = self.orig_call(*args, **kwargs)
                tape.watch(val)
                self.value = val
                return val

            layer.call = hooked_call
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            layer.call = self.orig_call

    return Hook()


def compute_gradcam(
    model: tf.keras.Model,
    target_layer: tf.keras.layers.Layer,
    inputs: Dict[str, tf.Tensor],
    target_class_idx: int,
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """Compute Grad-CAM heatmap at target layer."""
    with tf.GradientTape() as tape:
        with capture_layer_output(target_layer, tape=tape) as cap:
            outputs = model(inputs, training=False)
            logits = tf.cast(outputs["logits"] if isinstance(outputs, dict) else outputs[0], tf.float32)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            feature = cap["value"]
            if feature is None or len(feature.shape) != 4:
                raise RuntimeError(f"Invalid feature map from layer {target_layer.name}")
            feature = tf.cast(feature, tf.float32)
            score = logits[:, target_class_idx]

        grads = tape.gradient(score, feature)

    grads = tf.cast(grads, tf.float32)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * feature, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()

    # Robust percentile normalization (clip 99.5th percentile to suppress single-pixel nose dot spike)
    v_min = float(np.min(cam))
    v_max = float(np.percentile(cam, 99.5))
    denom = v_max - v_min
    if denom > 1e-8:
        heatmap_norm = np.clip((cam - v_min) / denom, 0.0, 1.0)
    else:
        heatmap_norm = np.zeros_like(cam)

    return heatmap_norm, conf, pred_idx, probs


def render_smooth_heatmap(cam_map: np.ndarray, target_size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsample CAM with bicubic interpolation and gentle Gaussian blur for silky-smooth paper quality.
    Returns:
        jet_rgb: 224x224 RGB uint8 JET heatmap
        cam_smoothed: 224x224 float32 [0, 1]
    """
    cam_resized = cv2.resize(cam_map.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    
    # Gentle Gaussian blur for smooth publication grade gradient
    cam_smoothed = cv2.GaussianBlur(cam_resized, (15, 15), sigmaX=3.0)
    cam_smoothed = np.clip(cam_smoothed, 0.0, 1.0)
    
    # Normalize to full 0..1 range
    s_min = float(cam_smoothed.min())
    s_max = float(cam_smoothed.max())
    if s_max - s_min > 1e-6:
        cam_smoothed = (cam_smoothed - s_min) / (s_max - s_min)
        
    cam_u8 = np.uint8(np.clip(cam_smoothed * 255.0, 0, 255))
    jet_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    
    return jet_rgb, cam_smoothed


def row_to_face224(pixels_str: str) -> np.ndarray:
    """Convert FER2013 48x48 pixel string into a high quality 224x224 RGB face image."""
    pixels = np.fromstring(pixels_str, sep=" ", dtype=np.uint8).reshape(48, 48)
    im = Image.fromarray(pixels, mode="L").resize((224, 224), resample=Image.Resampling.LANCZOS).convert("RGB")
    return np.array(im, dtype=np.uint8)


def main():
    print("=" * 75)
    print(" FER2013: Fast Stream Generation of 20 Grad-CAM Candidates Per Emotion")
    print(" Checkpoint: ckpt-29 | Target: Stage 4 ConvNeXt-Base (stage4_block2)")
    print("=" * 75)

    # 1. Config & Checkpoint
    cfg_path = Path("config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml")
    cfg = load_config(cfg_path)
    ckpt_prefix = r"checkpoints-down\ckpt-29"

    output_base_dir = Path("outputs/gradcam_20_candidates")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build model & restore weights
    print("\n[1/4] Building AMGSA-FER model and restoring ckpt-29...")
    model = build_model(cfg)
    
    dummy_input = {"image": tf.zeros([1, 112, 112, 3], dtype=tf.float32)}
    _ = model(dummy_input, training=False)
    
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_prefix)
    status.expect_partial()
    print(f"  -> Checkpoint successfully restored from {ckpt_prefix}")

    target_layer = model.backbone.stages[3][-1]
    print(f"  -> Grad-CAM Target Layer: {target_layer.name} (Stage 4, 7x7x1024, High-level Emotion AUs)")

    # 3. Load FER2013 test set
    csv_path = Path("data/fer13-split/test.csv")
    print(f"\n[2/4] Loading FER2013 test set from {csv_path}...")
    df_test = pd.read_csv(csv_path)
    print(f"  -> Total test samples: {len(df_test)}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    N_TARGET = 20
    master_summary = []

    print("\n[3/4] Streaming candidate selection and Grad-CAM generation across 7 emotions...")

    for c in range(7):
        emo_name = EMOTION_NAMES_TITLE[c]
        emo_dir = output_base_dir / emo_name
        emo_dir.mkdir(parents=True, exist_ok=True)

        c_df = df_test[df_test["emotion"] == c]
        print(f"\n--- Emotion {c}: {emo_name} (Searching {len(c_df)} available samples for Top {N_TARGET}) ---")

        candidates_pool = []
        
        # Scan samples of this emotion until we find enough high-confidence correct ones
        for idx, row in c_df.iterrows():
            face_224 = row_to_face224(row["pixels"])
            face_112 = Image.fromarray(face_224).resize((112, 112), resample=Image.Resampling.BILINEAR)
            face_arr = (np.array(face_112, dtype=np.float32) / 255.0 - mean) / std
            inp_tensor = tf.convert_to_tensor(face_arr[None, ...], dtype=tf.float32)

            out = model({"image": inp_tensor}, training=False)
            logits = tf.cast(out["logits"] if isinstance(out, dict) else out[0], tf.float32)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            if pred_idx == c and conf >= 0.65:
                candidates_pool.append({
                    "dataset_idx": int(idx),
                    "confidence": conf,
                    "face_224": face_224,
                    "inp_tensor": inp_tensor,
                })
                # Gather slightly more (e.g. 25) so we can pick the top 20 by highest confidence
                if len(candidates_pool) >= 28:
                    break

        # Fallback if fewer than 28 found with conf >= 0.65
        if len(candidates_pool) < N_TARGET:
            for idx, row in c_df.iterrows():
                if any(cp["dataset_idx"] == idx for cp in candidates_pool):
                    continue
                face_224 = row_to_face224(row["pixels"])
                face_112 = Image.fromarray(face_224).resize((112, 112), resample=Image.Resampling.BILINEAR)
                face_arr = (np.array(face_112, dtype=np.float32) / 255.0 - mean) / std
                inp_tensor = tf.convert_to_tensor(face_arr[None, ...], dtype=tf.float32)
                out = model({"image": inp_tensor}, training=False)
                logits = tf.cast(out["logits"] if isinstance(out, dict) else out[0], tf.float32)
                probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                if pred_idx == c:
                    candidates_pool.append({
                        "dataset_idx": int(idx),
                        "confidence": conf,
                        "face_224": face_224,
                        "inp_tensor": inp_tensor,
                    })
                if len(candidates_pool) >= N_TARGET:
                    break

        # Sort by confidence descending and take top 20
        candidates_pool.sort(key=lambda item: item["confidence"], reverse=True)
        selected_candidates = candidates_pool[:N_TARGET]
        print(f"  -> Selected {len(selected_candidates)} best candidates (Conf: {selected_candidates[-1]['confidence']:.3f} to {selected_candidates[0]['confidence']:.3f})")

        emo_records = []
        for rank, cand in enumerate(selected_candidates, start=1):
            dataset_idx = cand["dataset_idx"]
            conf = cand["confidence"]
            face_224 = cand["face_224"]
            inp_tensor = cand["inp_tensor"]

            sample_slug = f"sample_{rank:02d}_idx{dataset_idx}_conf{conf:.3f}"
            sample_dir = emo_dir / sample_slug
            sample_dir.mkdir(parents=True, exist_ok=True)

            cam_7x7, _, _, _ = compute_gradcam(
                model=model,
                target_layer=target_layer,
                inputs={"image": inp_tensor},
                target_class_idx=c,
            )

            heatmap_rgb, _ = render_smooth_heatmap(cam_7x7, target_size=224)
            overlay_rgb = cv2.addWeighted(face_224, 0.55, heatmap_rgb, 0.45, 0)

            # Save raw individual 224x224 PNGs
            cv2.imwrite(str(sample_dir / "original.png"), cv2.cvtColor(face_224, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sample_dir / "heatmap.png"), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sample_dir / "overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            rec = {
                "rank": rank,
                "dataset_idx": dataset_idx,
                "confidence": conf,
                "face_224": face_224,
                "heatmap_rgb": heatmap_rgb,
                "overlay_rgb": overlay_rgb,
            }
            emo_records.append(rec)
            master_summary.append({
                "emotion": emo_name,
                "rank": rank,
                "dataset_idx": dataset_idx,
                "confidence": conf,
                "folder": str(sample_dir.relative_to(PROJECT_ROOT)),
            })

        # Save Contact Sheet (20 rows x 3 columns)
        fig, axes = plt.subplots(len(emo_records), 3, figsize=(7.5, max(1.8 * len(emo_records), 4.0)), dpi=200)
        for r_idx, rec in enumerate(emo_records):
            ax_orig = axes[r_idx, 0]
            ax_heat = axes[r_idx, 1]
            ax_over = axes[r_idx, 2]

            ax_orig.imshow(rec["face_224"])
            ax_heat.imshow(rec["heatmap_rgb"])
            ax_over.imshow(rec["overlay_rgb"])

            for ax in (ax_orig, ax_heat, ax_over):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)

            row_label = f"#{rec['rank']:02d} (idx {rec['dataset_idx']})\nConf: {rec['confidence']:.3f}"
            ax_orig.set_ylabel(row_label, rotation=0, labelpad=38, va="center", fontsize=7.5, fontweight="bold")

            if r_idx == 0:
                ax_orig.set_title("Original (224x224)", fontsize=9.5, fontweight="bold", pad=6)
                ax_heat.set_title("JET Heatmap", fontsize=9.5, fontweight="bold", pad=6)
                ax_over.set_title("Overlay (55/45)", fontsize=9.5, fontweight="bold", pad=6)

        fig.suptitle(f"Grad-CAM Candidate Gallery: {emo_name} (Top {len(emo_records)} Samples)", fontsize=12, fontweight="bold", y=0.998)
        fig.tight_layout(rect=[0.05, 0.01, 1.0, 0.99])
        contact_path = emo_dir / f"contact_sheet_20samples_{emo_name.lower()}.png"
        fig.savefig(contact_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        # Save 4x5 Grid of Overlays
        fig_grid, axes_grid = plt.subplots(4, 5, figsize=(10, 8.5), dpi=250)
        for idx_g, ax in enumerate(axes_grid.flat):
            if idx_g < len(emo_records):
                rec = emo_records[idx_g]
                ax.imshow(rec["overlay_rgb"])
                ax.set_title(f"#{rec['rank']:02d} | C:{rec['confidence']:.2f}", fontsize=8, fontweight="bold", pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#dddddd")
        fig_grid.suptitle(f"Top 20 Grad-CAM Overlays - {emo_name} (FER2013 Test)", fontsize=12, fontweight="bold", y=0.98)
        fig_grid.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])
        grid_path = emo_dir / f"grid_4x5_overlays_{emo_name.lower()}.png"
        fig_grid.savefig(grid_path, bbox_inches="tight", dpi=250)
        plt.close(fig_grid)
        print(f"  [{emo_name}] Contact Sheet & 4x5 Grid saved successfully!")

    # 4. Save HTML interactive gallery & Manifest
    print("\n[4/4] Building interactive HTML gallery and manifest CSV...")
    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Grad-CAM Candidates Gallery (Top 20 Per Emotion)</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
h1 { text-align: center; color: #38bdf8; margin-bottom: 5px; }
p.subtitle { text-align: center; color: #94a3b8; margin-top: 0; margin-bottom: 25px; }
.tabs { display: flex; justify-content: center; gap: 8px; margin-bottom: 25px; flex-wrap: wrap; }
.tab-btn { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; }
.tab-btn.active { background: #2563eb; color: #ffffff; border-color: #3b82f6; }
.emotion-section { display: none; }
.emotion-section.active { display: block; }
.overview-card { background: #1e293b; border-radius: 8px; padding: 15px; margin-bottom: 25px; text-align: center; }
.overview-card img { max-width: 100%; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
.grid-20 { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }
.sample-card { background: #1e293b; border-radius: 8px; padding: 10px; border: 1px solid #334155; text-align: center; }
.sample-card img { width: 100%; aspect-ratio: 1; border-radius: 4px; object-fit: cover; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #059669; color: #ffffff; margin-top: 6px; }
.details { font-size: 12px; color: #94a3b8; margin-top: 4px; }
</style>
<script>
function showTab(emo) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.emotion-section').forEach(s => s.classList.remove('active'));
  document.getElementById('btn-' + emo).classList.add('active');
  document.getElementById('sec-' + emo).classList.add('active');
}
</script>
</head>
<body>
<h1>FER2013 Grad-CAM Candidates Gallery</h1>
<p class="subtitle">ConvNeXt-Base MS1M AMGSA-FER (ckpt-29) | Top 20 Candidates Per Emotion (Total: 140 Samples)</p>
<div class="tabs">
"""
    for idx_c, emo in enumerate(EMOTION_NAMES_TITLE):
        active_cls = " active" if idx_c == 0 else ""
        html_content += f'<button id="btn-{emo}" class="tab-btn{active_cls}" onclick="showTab(\'{emo}\')">{emo} (20)</button>\n'
    html_content += "</div>\n"

    for idx_c, emo in enumerate(EMOTION_NAMES_TITLE):
        active_cls = " active" if idx_c == 0 else ""
        html_content += f'<div id="sec-{emo}" class="emotion-section{active_cls}">\n'
        html_content += f'<div class="overview-card"><h3>{emo} - 4x5 Gallery Grid</h3><img src="{emo}/grid_4x5_overlays_{emo.lower()}.png"></div>\n'
        html_content += f'<div class="overview-card"><h3>{emo} - Contact Sheet (Original | Heatmap | Overlay)</h3><img src="{emo}/contact_sheet_20samples_{emo.lower()}.png"></div>\n'
        html_content += '<div class="grid-20">\n'
        emo_samples = [m for m in master_summary if m["emotion"] == emo]
        for s in emo_samples:
            rel_f = s["folder"].replace("\\", "/").replace("outputs/gradcam_20_candidates/", "")
            html_content += f'''<div class="sample-card">
  <img src="{rel_f}/overlay.png">
  <div class="badge">#{s['rank']:02d} | Conf: {s['confidence']:.3f}</div>
  <div class="details">Dataset Index: {s['dataset_idx']}</div>
</div>\n'''
        html_content += "</div>\n</div>\n"

    html_content += "</body>\n</html>"

    html_path = output_base_dir / "gallery_candidates.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    summary_df = pd.DataFrame(master_summary)
    summary_csv = output_base_dir / "candidates_manifest.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[DONE] Successfully generated:")
    print(f"  - 140 individual candidate folders with original.png, heatmap.png, overlay.png (420 images)")
    print(f"  - 7 Contact Sheets (20 rows x 3 cols)")
    print(f"  - 7 High-res 4x5 Gallery Grids")
    print(f"  - 1 Interactive HTML Gallery: {html_path.resolve()}")
    print(f"  - 1 Manifest CSV: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
