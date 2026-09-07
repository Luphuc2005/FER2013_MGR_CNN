"""
Generate 20 Publication-Quality Grad-CAM Candidates Per Emotion for FER2013 Test Set.
Model: ConvNeXt-Base MS1M AMGSA-FER (ckpt-29).
Target Layer: Stage 4 ConvNeXt-Base (7x7x1024, High-Level Emotional Action Units).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Pure CPU mode & clean environment
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
    "Sad",
    "Surprise",
    "Neutral",
]


def compute_stage4_gradcam_clean(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    target_class_idx: int,
    percentile_clip: float = 99.5,
    gamma: float = 0.85,
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Pure native TensorFlow Grad-CAM computation using model.backbone(return_endpoints=True).
    Avoids monkey-patching layer.call, ensuring zero memory corruption or C++ crashes.
    """
    with tf.GradientTape() as tape:
        endpoints = model.backbone(image_tensor, training=False, return_endpoints=True)
        feat = endpoints["stage4"]
        tape.watch(feat)
        
        feat_head = model.stage4_eca(feat, training=False) if (model.use_eca and model.stage4_eca is not None) else feat
        pooled = model.gap(feat_head)
        dropped = model.head_dropout(pooled, training=False)
        logits = model.classifier(dropped)
        
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        score = logits[:, target_class_idx]
        
    grads = tape.gradient(score, feat)
    grads = tf.cast(grads, tf.float32)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * feat, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    
    # Robust percentile normalization
    v_min = float(np.min(cam))
    v_max = float(np.percentile(cam, percentile_clip))
    denom = v_max - v_min
    if denom > 1e-8:
        cam_norm = np.clip((cam - v_min) / denom, 0.0, 1.0)
    else:
        cam_norm = np.zeros_like(cam)
        
    if gamma != 1.0 and np.max(cam_norm) > 0:
        cam_norm = np.power(cam_norm, gamma)
        
    return cam_norm, conf, pred_idx, probs


def render_smooth_heatmap(cam_map: np.ndarray, target_size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """Upsample CAM with bicubic interpolation and Gaussian blur for silky-smooth paper quality."""
    cam_resized = cv2.resize(cam_map.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    
    cam_smoothed = cv2.GaussianBlur(cam_resized, (15, 15), sigmaX=3.0)
    cam_smoothed = np.clip(cam_smoothed, 0.0, 1.0)
    
    s_min = float(cam_smoothed.min())
    s_max = float(cam_smoothed.max())
    if s_max - s_min > 1e-6:
        cam_smoothed = (cam_smoothed - s_min) / (s_max - s_min)
        
    cam_u8 = np.uint8(np.clip(cam_smoothed * 255.0, 0, 255))
    jet_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    
    return jet_rgb, cam_smoothed


def row_to_face224(pixels_str: str) -> np.ndarray:
    """Convert FER2013 48x48 pixel string into a high-quality 224x224 RGB face image."""
    pixels = np.fromstring(pixels_str, sep=" ", dtype=np.uint8).reshape(48, 48)
    im = Image.fromarray(pixels, mode="L").resize((224, 224), resample=Image.Resampling.LANCZOS).convert("RGB")
    return np.array(im, dtype=np.uint8)


def main():
    print("=" * 75)
    print(" FER2013: Generating Top 20 Grad-CAM Candidates Per Emotion")
    print(" Model: ConvNeXt-Base MS1M AMGSA-FER (ckpt-29)")
    print(" Target: Stage 4 (7x7x1024, High-Level Action Units)")
    print("=" * 75)

    # 1. Config & Checkpoint
    cfg_path = PROJECT_ROOT / "config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
    cfg = load_config(str(cfg_path))
    cfg["runtime"]["allow_cpu_fallback"] = True
    cfg["runtime"]["min_gpus"] = 0
    cfg["runtime"]["use_mixed_precision"] = False
    
    ckpt_prefix = str(PROJECT_ROOT / "checkpoints-down/ckpt-29")
    output_dir = PROJECT_ROOT / "outputs/gradcam_20_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build Model & Restore Checkpoint
    print("\n[1/4] Building AMGSA-FER architecture and restoring checkpoint...")
    model = build_model(cfg)
    dummy_input = {"image": tf.zeros([1, 112, 112, 3], dtype=tf.float32)}
    _ = model(dummy_input, training=False)
    
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_prefix)
    status.expect_partial()
    print(f"  -> Successfully restored checkpoint: {ckpt_prefix}")

    # 3. Load FER2013 Test Set
    csv_path = PROJECT_ROOT / "data/fer13-split/test.csv"
    if not csv_path.exists():
        csv_path = Path(r"D:\HocTap\Phân tích  và xử lý ảnh\sgu-2026-facial-expression-recognition\dataset\fer13-split\test.csv")
    print(f"\n[2/4] Loading FER2013 test set from: {csv_path}")
    df_test = pd.read_csv(csv_path)
    print(f"  -> Total test samples: {len(df_test)}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    N_TARGET = 20
    master_summary = []
    best_candidate_per_emotion: Dict[str, Dict[str, Any]] = {}

    print(f"\n[3/4] Searching test set and generating Top {N_TARGET} Grad-CAM candidates per emotion...")

    for c in range(7):
        emo_name = EMOTION_NAMES_TITLE[c]
        emo_dir = output_dir / emo_name
        emo_dir.mkdir(parents=True, exist_ok=True)
        
        c_df = df_test[df_test["emotion"] == c]
        print(f"\n--- Emotion {c}: {emo_name:<10} | Available samples: {len(c_df)} ---")
        
        # Collect candidate samples
        candidates_pool = []
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
            
            if pred_idx == c and conf >= 0.70:
                candidates_pool.append({
                    "idx": int(idx),
                    "confidence": conf,
                    "face_224": face_224,
                    "inp_tensor": inp_tensor,
                })
                if len(candidates_pool) >= 28:
                    break
                    
        # Fallback if needed
        if len(candidates_pool) < N_TARGET:
            for idx, row in c_df.iterrows():
                if any(x["idx"] == idx for x in candidates_pool):
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
                        "idx": int(idx),
                        "confidence": conf,
                        "face_224": face_224,
                        "inp_tensor": inp_tensor,
                    })
                if len(candidates_pool) >= N_TARGET:
                    break
                    
        candidates_pool.sort(key=lambda item: item["confidence"], reverse=True)
        selected = candidates_pool[:N_TARGET]
        print(f"  -> Found {len(selected)} high-confidence samples: {selected[0]['confidence']*100:.1f}% to {selected[-1]['confidence']*100:.1f}%")
        
        emo_records = []
        for rank, cand in enumerate(selected, start=1):
            sample_dir = emo_dir / f"sample_{rank:02d}_idx{cand['idx']}_conf{cand['confidence']:.3f}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            cam_7x7, conf, pred_idx, _ = compute_stage4_gradcam_clean(
                model=model,
                image_tensor=cand["inp_tensor"],
                target_class_idx=c,
                percentile_clip=99.5,
                gamma=0.85,
            )
            
            heatmap_rgb, _ = render_smooth_heatmap(cam_7x7, target_size=224)
            overlay_rgb = cv2.addWeighted(cand["face_224"], 0.55, heatmap_rgb, 0.45, 0)
            
            # Save raw individual 224x224 images (no text, no axis, no padding)
            Image.fromarray(cand["face_224"]).save(sample_dir / "original.png")
            Image.fromarray(heatmap_rgb).save(sample_dir / "heatmap.png")
            Image.fromarray(overlay_rgb).save(sample_dir / "overlay.png")
            
            rec = {
                "rank": rank,
                "dataset_idx": cand["idx"],
                "confidence": conf,
                "face_224": cand["face_224"],
                "heatmap_rgb": heatmap_rgb,
                "overlay_rgb": overlay_rgb,
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
                best_candidate_per_emotion[emo_name] = {
                    "face_224": cand["face_224"],
                    "overlay_rgb": overlay_rgb,
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
        contact_path = emo_dir / f"contact_sheet_20samples_{emo_name.lower()}.png"
        fig.savefig(contact_path, bbox_inches="tight", dpi=200)
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
        grid_path = emo_dir / f"grid_4x5_overlays_{emo_name.lower()}.png"
        fig_grid.savefig(grid_path, bbox_inches="tight", dpi=250)
        plt.close(fig_grid)
        print(f"  [{emo_name}] Contact Sheet & 4x5 Grid saved!")

    # 4. Generate Master Publication 2x7 Figure (Best Candidates on FER2013)
    print("\n[4/4] Generating Master Publication 2x7 Figure from Best FER2013 Samples...")
    pub_2x7_png = output_dir / "publication_figure_2x7_fer_best.png"
    pub_2x7_pdf = output_dir / "publication_figure_2x7_fer_best.pdf"
    fig2, axes2 = plt.subplots(2, 7, figsize=(14.0, 4.3))
    fig2.subplots_adjust(wspace=0, hspace=0)
    row2_labels = ["Original", "AMGSA-FER (Ours)"]
    for col_idx, emo_name in enumerate(EMOTION_NAMES_TITLE):
        if emo_name not in best_candidate_per_emotion:
            continue
        data = best_candidate_per_emotion[emo_name]
        col_imgs = [data["face_224"], data["overlay_rgb"]]
        for row_idx in range(2):
            ax = axes2[row_idx, col_idx]
            ax.imshow(col_imgs[row_idx], aspect="equal")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"{emo_name}\n({data['confidence']*100:.1f}%)", fontsize=11, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.text(-0.06, 0.5, row2_labels[row_idx], va="center", ha="right", fontsize=12, fontweight="bold", transform=ax.transAxes)
    fig2.savefig(pub_2x7_png, dpi=600, bbox_inches="tight", pad_inches=0)
    fig2.savefig(pub_2x7_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)

    # Save Manifest CSV
    manifest_df = pd.DataFrame(master_summary)
    manifest_csv = output_dir / "candidates_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    # Build Interactive HTML Gallery
    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FER2013 Grad-CAM Candidates Gallery (Top 20 Per Emotion)</title>
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
  <div class="badge">#{s['rank']:02d} | Conf: {s['confidence']*100:.1f}%</div>
  <div class="details">Dataset Index: {s['dataset_idx']}</div>
</div>\n'''
        html_content += "</div>\n</div>\n"

    html_content += "</body>\n</html>"

    html_path = output_dir / "gallery_candidates.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n=================================================================")
    print(f"                 ALL 140 CANDIDATE SAMPLES GENERATED!")
    print(f"=================================================================")
    print(f"1. Output Directory           : {output_dir.resolve()}")
    print(f"2. Best FER2013 2x7 Figure    : {pub_2x7_png.resolve()} (600 DPI)")
    print(f"3. Best FER2013 Vector (PDF)  : {pub_2x7_pdf.resolve()} (PDF)")
    print(f"4. Interactive HTML Gallery   : {html_path.resolve()}")
    print(f"5. Manifest CSV (140 samples) : {manifest_csv.resolve()}")
    print(f"=================================================================\n")


if __name__ == "__main__":
    main()
