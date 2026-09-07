import os
import sys
import gc
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy('float32')

PROJECT_ROOT = Path(r'd:\HocTap\Phân tích  và xử lý ảnh\sgu-2026-facial-expression-recognition\FER2013_SGU')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from train import build_model

EMOTION_NAMES_TITLE = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral',
]

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
    
    cam_smoothed = cv2.GaussianBlur(cam_resized, (15, 15), sigmaX=3.0)
    s_min = float(np.min(cam_smoothed))
    s_max = float(np.max(cam_smoothed))
    if (s_max - s_min) > 1e-8:
        cam_smoothed = (cam_smoothed - s_min) / (s_max - s_min)
        
    cam_u8 = (cam_smoothed * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb, cam_smoothed

def main():
    import traceback
    try:
        _main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

def _main():
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", type=int, default=-1, help="Emotion index (0-6) or -1 for master coordinator")
    parser.add_argument("--render-only", action="store_true", help="Only build figures/gallery from saved images")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / 'config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml'
    ckpt_path = PROJECT_ROOT / 'checkpoints-down/ckpt-29'
    pred_path = Path(r'D:\HocTap\Phân tích  và xử lý ảnh\sgu-2026-facial-expression-recognition\outputs\xai_mgr_cnn_7512_paper\predictions.csv')
    test_path = PROJECT_ROOT / 'data/fer13-split/test.csv'
    output_dir = PROJECT_ROOT / 'outputs/gradcam_20_candidates'
    output_dir.mkdir(parents=True, exist_ok=True)

    # MASTER COORDINATOR: Launch separate processes for each emotion to guarantee clean memory
    if args.emotion == -1 and not args.render_only:
        print('=' * 70)
        print('MASTER GRAD-CAM COORDINATOR: 20 CANDIDATES PER EMOTION')
        print(f'Model Checkpoint : {ckpt_path}')
        print(f'Output Directory : {output_dir}')
        print('=' * 70)

        for emo_idx in range(7):
            emo_name = EMOTION_NAMES_TITLE[emo_idx]
            emo_dir = output_dir / emo_name
            emo_dir.mkdir(parents=True, exist_ok=True)
            existing = [d for d in emo_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")]
            if len(existing) >= 20:
                print(f"[{emo_idx+1}/7] {emo_name}: All 20 samples already generated! Skipping.")
                continue

            print(f"\n>>> Running Emotion {emo_idx} ({emo_name}) in isolated process...")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--emotion", str(emo_idx)]
            subprocess.run(cmd, check=True)
            print(f">>> Emotion {emo_idx} ({emo_name}) finished successfully!")

    # SINGLE EMOTION WORKER PROCESS
    elif args.emotion >= 0:
        c = args.emotion
        emo_name = EMOTION_NAMES_TITLE[c]
        emo_dir = output_dir / emo_name
        emo_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Worker] Processing Emotion {c}: {emo_name}...")
        cfg = load_config(str(config_path))
        model = build_model(cfg)
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore('checkpoints-down/ckpt-29').expect_partial()

        df_pred = pd.read_csv(pred_path)
        df_test = pd.read_csv(test_path)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        sub_pred = df_pred[(df_pred['true_label'] == c) & (df_pred['correct'] == True)].sort_values('confidence', ascending=False).head(20)
        indices = sub_pred['dataset_index'].tolist()
        confs = sub_pred['confidence'].tolist()

        batch_faces_224 = []
        batch_inputs = []
        for idx in indices:
            raw_pixels = df_test.loc[idx, 'pixels']
            arr_48 = np.fromstring(raw_pixels, sep=' ', dtype=np.uint8).reshape((48, 48))
            face_224 = cv2.resize(arr_48, (224, 224), interpolation=cv2.INTER_CUBIC)
            face_rgb_224 = cv2.cvtColor(face_224, cv2.COLOR_GRAY2RGB)
            batch_faces_224.append(face_rgb_224)

            face_112 = Image.fromarray(face_rgb_224).resize((112, 112), resample=Image.Resampling.BILINEAR)
            face_norm = (np.array(face_112, dtype=np.float32) / 255.0 - mean) / std
            batch_inputs.append(face_norm)

        # Ultra-efficient Grad-CAM: backbone forward OUTSIDE tape, tape only on lightweight classification head
        batch_tensor = tf.convert_to_tensor(np.stack(batch_inputs, axis=0), dtype=tf.float32)
        endpoints = model.backbone(batch_tensor, training=False, return_endpoints=True)
        feat = endpoints['stage4']

        with tf.GradientTape() as tape:
            tape.watch(feat)
            feat_head = model.stage4_eca(feat, training=False) if (model.use_eca and model.stage4_eca is not None) else feat
            pooled = model.gap(feat_head)
            dropped = model.head_dropout(pooled, training=False)
            logits = model.classifier(dropped)
            score = tf.reduce_sum(logits[:, c])

        grads = tape.gradient(score, feat)
        grads = tf.cast(grads, tf.float32)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cams = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1)).numpy()

        for rank in range(1, len(indices) + 1):
            idx_in_batch = rank - 1
            cand_idx = indices[idx_in_batch]
            cand_conf = confs[idx_in_batch]
            face_224 = batch_faces_224[idx_in_batch]
            cam_7x7 = cams[idx_in_batch]

            sample_dir = emo_dir / f'sample_{rank:02d}_idx{cand_idx}_conf{cand_conf:.3f}'
            sample_dir.mkdir(parents=True, exist_ok=True)

            heatmap_rgb, _ = render_smooth_heatmap(cam_7x7, target_size=224, gamma=0.85)
            overlay_rgb = cv2.addWeighted(face_224, 0.55, heatmap_rgb, 0.45, 0)

            Image.fromarray(face_224).save(sample_dir / 'original.png')
            Image.fromarray(heatmap_rgb).save(sample_dir / 'heatmap.png')
            Image.fromarray(overlay_rgb).save(sample_dir / 'overlay.png')

        print(f"[Worker] All 20 samples for {emo_name} generated and saved!")
        sys.exit(0)

    # PHASE 2: Load saved images to build contact sheets, grids, and master figures
    print("\n[PHASE 2] Building contact sheets, 4x5 grids, and master deliverables...")
    master_summary = []
    best_candidate_per_emotion: Dict[str, Dict[str, Any]] = {}

    for c in range(7):
        emo_name = EMOTION_NAMES_TITLE[c]
        emo_dir = output_dir / emo_name
        sample_dirs = sorted([d for d in emo_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

        emo_records = []
        for r_idx, s_dir in enumerate(sample_dirs, start=1):
            face_224 = np.array(Image.open(s_dir / "original.png"))
            heatmap_rgb = np.array(Image.open(s_dir / "heatmap.png"))
            overlay_rgb = np.array(Image.open(s_dir / "overlay.png"))

            # Parse index and confidence from dir name: sample_01_idx354_conf0.987
            parts = s_dir.name.split("_")
            cand_idx = int(parts[2].replace("idx", "")) if len(parts) >= 3 else 0
            cand_conf = float(parts[3].replace("conf", "")) if len(parts) >= 4 else 0.0

            rec = {
                'rank': r_idx,
                'dataset_idx': cand_idx,
                'confidence': cand_conf,
                'face_224': face_224,
                'heatmap_rgb': heatmap_rgb,
                'overlay_rgb': overlay_rgb,
                'rel_path': str(s_dir.relative_to(PROJECT_ROOT)).replace('\\\\', '/'),
            }
            emo_records.append(rec)
            master_summary.append({
                'emotion': emo_name,
                'rank': r_idx,
                'dataset_idx': cand_idx,
                'confidence': cand_conf,
                'folder': str(s_dir.relative_to(PROJECT_ROOT)).replace('\\\\', '/'),
            })

            if r_idx == 1:
                best_candidate_per_emotion[emo_name] = {
                    'face_224': face_224,
                    'heatmap_rgb': heatmap_rgb,
                    'overlay_rgb': overlay_rgb,
                    'confidence': cand_conf,
                    'dataset_idx': cand_idx,
                }

        # Contact Sheet
        fig, axes = plt.subplots(len(emo_records), 3, figsize=(7.5, max(1.8 * len(emo_records), 4.0)), dpi=200)
        for r_idx, rec in enumerate(emo_records):
            axes[r_idx, 0].imshow(rec['face_224'])
            axes[r_idx, 1].imshow(rec['heatmap_rgb'])
            axes[r_idx, 2].imshow(rec['overlay_rgb'])
            for ax in axes[r_idx]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                    spine.set_linewidth(0.5)
            rank_val = rec['rank']
            idx_val = rec['dataset_idx']
            conf_pct = rec['confidence'] * 100.0
            row_label = f'#{rank_val:02d} (idx {idx_val})\nConf: {conf_pct:.1f}%'
            axes[r_idx, 0].set_ylabel(row_label, rotation=0, labelpad=38, va='center', fontsize=7.5, fontweight='bold')
            if r_idx == 0:
                axes[r_idx, 0].set_title('Original (224x224)', fontsize=9.5, fontweight='bold', pad=6)
                axes[r_idx, 1].set_title('JET Heatmap', fontsize=9.5, fontweight='bold', pad=6)
                axes[r_idx, 2].set_title('Overlay (55/45)', fontsize=9.5, fontweight='bold', pad=6)
        fig.suptitle(f'Grad-CAM Candidates: {emo_name} (Top {len(emo_records)} Samples, FER2013)', fontsize=12, fontweight='bold', y=0.998)
        fig.tight_layout(rect=[0.05, 0.01, 1.0, 0.99])
        contact_path = emo_dir / f'contact_sheet_20samples_{emo_name.lower()}.png'
        fig.savefig(contact_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

        # 4x5 Grid
        fig_grid, axes_grid = plt.subplots(4, 5, figsize=(10, 8.5), dpi=250)
        for idx_g, ax in enumerate(axes_grid.flat):
            if idx_g < len(emo_records):
                rec = emo_records[idx_g]
                r_rank = rec['rank']
                r_conf = rec['confidence'] * 100.0
                ax.imshow(rec['overlay_rgb'])
                ax.set_title(f"#{r_rank:02d} | {r_conf:.1f}%", fontsize=8, fontweight='bold', pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#dddddd')
                spine.set_linewidth(0.5)
        fig_grid.suptitle(f'Grad-CAM Overlays (4x5 Gallery): {emo_name} (Top 20 Samples)', fontsize=13, fontweight='bold', y=0.995)
        fig_grid.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])
        grid_path = emo_dir / f'grid_4x5_overlays_{emo_name.lower()}.png'
        fig_grid.savefig(grid_path, bbox_inches='tight', dpi=250)
        plt.close(fig_grid)
        print(f'  -> Built contact sheet and 4x5 grid for {emo_name}')

    # Manifest
    manifest_df = pd.DataFrame(master_summary)
    manifest_path = output_dir / 'candidates_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)

    # 2x7 Master Figure
    fig_2x7, axes_2x7 = plt.subplots(2, 7, figsize=(14, 4.4), dpi=300)
    for c, emo_name in enumerate(EMOTION_NAMES_TITLE):
        best = best_candidate_per_emotion[emo_name]
        ax_orig = axes_2x7[0, c]
        ax_over = axes_2x7[1, c]

        b_conf = best['confidence'] * 100.0
        ax_orig.imshow(best['face_224'])
        ax_orig.set_title(f"{emo_name}\n({b_conf:.1f}%)", fontsize=10, fontweight='bold', pad=4)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        ax_over.imshow(best['overlay_rgb'])
        ax_over.set_xticks([])
        ax_over.set_yticks([])

        for ax in [ax_orig, ax_over]:
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)

    axes_2x7[0, 0].set_ylabel('Original Face', fontsize=10, fontweight='bold', labelpad=6)
    axes_2x7[1, 0].set_ylabel('Stage 4 Grad-CAM', fontsize=10, fontweight='bold', labelpad=6)
    fig_2x7.suptitle('ConvNeXt-Base MS1M (ckpt-29): High-Confidence Grad-CAM Attention Across 7 Facial Expressions (FER2013)', fontsize=11, fontweight='bold', y=0.98)
    fig_2x7.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])

    p2x7_png = output_dir / 'publication_figure_2x7_fer_best.png'
    p2x7_pdf = output_dir / 'publication_figure_2x7_fer_best.pdf'
    fig_2x7.savefig(p2x7_png, bbox_inches='tight', dpi=600)
    fig_2x7.savefig(p2x7_pdf, bbox_inches='tight')
    plt.close(fig_2x7)

    # 3x7 Master Figure
    fig_3x7, axes_3x7 = plt.subplots(3, 7, figsize=(14, 6.5), dpi=300)
    for c, emo_name in enumerate(EMOTION_NAMES_TITLE):
        best = best_candidate_per_emotion[emo_name]
        ax_orig = axes_3x7[0, c]
        ax_heat = axes_3x7[1, c]
        ax_over = axes_3x7[2, c]

        b_conf = best['confidence'] * 100.0
        ax_orig.imshow(best['face_224'])
        ax_orig.set_title(f"{emo_name}\n({b_conf:.1f}%)", fontsize=10, fontweight='bold', pad=4)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        ax_heat.imshow(best['heatmap_rgb'])
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

        ax_over.imshow(best['overlay_rgb'])
        ax_over.set_xticks([])
        ax_over.set_yticks([])

        for ax in [ax_orig, ax_heat, ax_over]:
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)

    axes_3x7[0, 0].set_ylabel('Original Face', fontsize=10, fontweight='bold', labelpad=6)
    axes_3x7[1, 0].set_ylabel('JET Heatmap', fontsize=10, fontweight='bold', labelpad=6)
    axes_3x7[2, 0].set_ylabel('Overlay (55/45)', fontsize=10, fontweight='bold', labelpad=6)
    fig_3x7.suptitle('ConvNeXt-Base MS1M (ckpt-29): Interpretability Triplet Comparison Across 7 Expressions', fontsize=11, fontweight='bold', y=0.98)
    fig_3x7.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])

    p3x7_png = output_dir / 'publication_figure_3x7_fer_best.png'
    p3x7_pdf = output_dir / 'publication_figure_3x7_fer_best.pdf'
    fig_3x7.savefig(p3x7_png, bbox_inches='tight', dpi=600)
    fig_3x7.savefig(p3x7_pdf, bbox_inches='tight')
    plt.close(fig_3x7)

    # HTML Gallery
    html_content = generate_html_gallery(master_summary)
    html_path = output_dir / 'gallery_candidates.html'
    html_path.write_text(html_content, encoding='utf-8')
    print(f'  -> Interactive gallery generated: {html_path}')

    # Copy to artifacts directory
    art_dir = Path(r'C:\Users\ADMIN\.gemini\antigravity\brain\a8aa799f-8fd8-442e-8d7c-d7f3130ee5c2')
    if art_dir.exists():
        shutil.copy(str(p2x7_png), str(art_dir / 'publication_figure_2x7_paper.png'))
        shutil.copy(str(p3x7_png), str(art_dir / 'publication_figure_3x7.png'))
        for emo_name in EMOTION_NAMES_TITLE:
            g_path = output_dir / emo_name / f'grid_4x5_overlays_{emo_name.lower()}.png'
            if g_path.exists():
                shutil.copy(str(g_path), str(art_dir / f'grid_4x5_{emo_name.lower()}.png'))

    print('\nALL 140 CANDIDATES AND MASTER FIGURES GENERATED SUCCESSFULLY!')

def generate_html_gallery(summary_list: List[Dict[str, Any]]) -> str:
    cards_per_emo = {emo: [] for emo in EMOTION_NAMES_TITLE}
    for item in summary_list:
        folder = item['folder']
        emo = item['emotion']
        rank = item['rank']
        conf = item['confidence']
        idx = item['dataset_idx']
        card_html = f'''
        <div class="card">
            <div class="card-header">
                <span class="rank">#{rank:02d}</span>
                <span class="idx">Index: {idx}</span>
                <span class="conf">{conf*100:.1f}%</span>
            </div>
            <div class="img-row">
                <div class="img-box">
                    <img src="../../{folder}/original.png" alt="Original" loading="lazy">
                    <span>Original</span>
                </div>
                <div class="img-box">
                    <img src="../../{folder}/heatmap.png" alt="Heatmap" loading="lazy">
                    <span>Heatmap</span>
                </div>
                <div class="img-box">
                    <img src="../../{folder}/overlay.png" alt="Overlay" loading="lazy">
                    <span>Overlay</span>
                </div>
            </div>
        </div>
        '''
        cards_per_emo[emo].append(card_html)

    tabs_html = ""
    sections_html = ""
    for idx_e, emo in enumerate(EMOTION_NAMES_TITLE):
        active_cls = "active" if idx_e == 0 else ""
        tabs_html += f'<button class="tab-btn {active_cls}" onclick="switchTab(\'{emo}\')">{emo} (20)</button>\n'
        content = "\n".join(cards_per_emo[emo])
        sections_html += f'''
        <div id="section-{emo}" class="tab-section {active_cls}">
            <div class="section-overview">
                <h3>{emo} Overview</h3>
                <p>20 highest confidence test samples from FER2013 test set correctly classified by ConvNeXt-Base (ckpt-29).</p>
                <div class="overview-images">
                    <a href="{emo}/grid_4x5_overlays_{emo.lower()}.png" target="_blank">
                        <img src="{emo}/grid_4x5_overlays_{emo.lower()}.png" alt="4x5 Grid" style="max-width: 48%;">
                    </a>
                    <a href="{emo}/contact_sheet_20samples_{emo.lower()}.png" target="_blank">
                        <img src="{emo}/contact_sheet_20samples_{emo.lower()}.png" alt="Contact Sheet" style="max-width: 48%;">
                    </a>
                </div>
            </div>
            <div class="grid-container">
                {content}
            </div>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FER2013 Grad-CAM Candidates Gallery (140 Samples)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid #334155;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            color: #38bdf8;
        }}
        .header p {{
            margin: 0;
            color: #94a3b8;
            font-size: 15px;
        }}
        .master-banner {{
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-bottom: 24px;
        }}
        .banner-btn {{
            background: #1e293b;
            color: #38bdf8;
            border: 1px solid #38bdf8;
            padding: 10px 18px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .banner-btn:hover {{
            background: #38bdf8;
            color: #0f172a;
        }}
        .tabs {{
            display: flex;
            gap: 8px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }}
        .tab-btn {{
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            padding: 8px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: #0284c7;
            color: #ffffff;
            border-color: #38bdf8;
        }}
        .tab-section {{
            display: none;
        }}
        .tab-section.active {{
            display: block;
        }}
        .section-overview {{
            background: #1e293b;
            padding: 16px;
            border-radius: 10px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .overview-images {{
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-top: 12px;
        }}
        .overview-images img {{
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        .rank {{
            background: #0284c7;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .idx {{
            color: #94a3b8;
        }}
        .conf {{
            color: #4ade80;
            font-weight: bold;
        }}
        .img-row {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 6px;
        }}
        .img-box {{
            text-align: center;
        }}
        .img-box img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
            border: 1px solid #475569;
        }}
        .img-box span {{
            display: block;
            margin-top: 4px;
            font-size: 10px;
            color: #94a3b8;
        }}
    </style>
    <script>
        function switchTab(emo) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('section-' + emo).classList.add('active');
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>FER2013 Grad-CAM Candidates Gallery</h1>
        <p>140 Publication Candidates (20 per emotion) | ConvNeXt-Base MS1M (ckpt-29) | Stage 4 High-Level AUs</p>
    </div>
    <div class="master-banner">
        <a class="banner-btn" href="publication_figure_2x7_fer_best.png" target="_blank">View Master 2x7 Figure (600 DPI)</a>
        <a class="banner-btn" href="publication_figure_3x7_fer_best.png" target="_blank">View Master 3x7 Figure (600 DPI)</a>
        <a class="banner-btn" href="candidates_manifest.csv" download>Download Manifest CSV</a>
    </div>
    <div class="tabs">
        {tabs_html}
    </div>
    {sections_html}
</body>
</html>
'''
    return html

if __name__ == '__main__':
    main()
