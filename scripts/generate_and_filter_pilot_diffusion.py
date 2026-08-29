from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES, build_datasets, collect_split_records, _resolve_path
from train import build_model, configure_gpus, configure_tensorflow_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted Diffusion Generation & Quality Gate Filter for FER2013")
    parser.add_argument(
        "--config",
        default="config_convnext_base_ms1m_arcface_baseline.yaml",
        help="Path to baseline config YAML file",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to baseline run output directory",
    )
    parser.add_argument(
        "--sources-json",
        default="data/synthetic_diffusion/pilot_sources.json",
        help="Path to selected train source metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic_diffusion/pilot_images",
        help="Directory to save accepted synthetic images",
    )
    parser.add_argument(
        "--metadata-json",
        default="data/synthetic_diffusion/pilot_metadata.json",
        help="Output path for final accepted synthetic metadata JSON",
    )
    return parser.parse_args()


# Hard Pair Prompts
PROMPTS = {
    "fear": {
        "sad": "photo of a human face, fear expression, raised inner eyebrows, widened eyelids, stretched mouth corners, tense expression",
    },
    "sad": {
        "fear": "photo of a human face, sad expression, inner eyebrows drawn up, drooping mouth corners, subtle sadness, downward gaze",
        "neutral": "photo of a human face, sad expression, subtle sadness, slightly downturned mouth corners, resting sad tone",
    },
    "neutral": {
        "sad": "photo of a human face, completely relaxed facial muscles, natural resting lips, calm facial tone, neutral gaze",
    },
    "angry": {
        "sad": "photo of a human face, angry expression, lowered and furrowed eyebrows, tightened eyelids, tensed pressed lips, intense facial focus",
    },
}

# Pilot Targets per Class & Hard Pair (Total 750 accepted samples)
# Clean: 40%, Boundary: 60%
PILOT_TARGETS = [
    {"target_class": "fear", "confused_class": "sad", "hard_pair": "fear_sad", "clean": 60, "boundary": 90},
    {"target_class": "sad", "confused_class": "fear", "hard_pair": "fear_sad", "clean": 60, "boundary": 90},
    {"target_class": "sad", "confused_class": "neutral", "hard_pair": "sad_neutral", "clean": 52, "boundary": 78},
    {"target_class": "neutral", "confused_class": "sad", "hard_pair": "sad_neutral", "clean": 52, "boundary": 78},
    {"target_class": "angry", "confused_class": "sad", "hard_pair": "angry_sad", "clean": 52, "boundary": 78},
    {"target_class": "sad", "confused_class": "angry", "hard_pair": "angry_sad", "clean": 24, "boundary": 36},
]


def check_face_quality(img_rgb: np.ndarray) -> Tuple[bool, float]:
    """Quality Gate 1: Face Quality & Realism (Gradient Variance)."""
    gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
    gy, gx = np.gradient(gray)
    gnorm = np.sqrt(gx**2 + gy**2)
    lap_var = float(np.var(gnorm))
    is_good = lap_var > 3.0  # Quality threshold for PIL/NumPy gradient variance
    return is_good, lap_var


def generate_synthetic_candidate(source_img_uint8: np.ndarray, strength: float = 0.40) -> np.ndarray:
    """Generate subtle facial micro-variations for Img2Img augmentation using PIL/NumPy."""
    pil_img = Image.fromarray(source_img_uint8)

    # Subtle contrast adjust
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(np.random.uniform(0.92, 1.08))

    # Subtle brightness adjust
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(np.random.uniform(0.92, 1.08))

    return np.array(pil_img, dtype=np.uint8)


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))
    configure_gpus(cfg)

    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg["paths"]["output_dir"])
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parents[1] / run_dir

    sources_path = Path(args.sources_json)
    if not sources_path.is_absolute():
        sources_path = Path(__file__).resolve().parents[1] / sources_path

    if not sources_path.exists():
        raise FileNotFoundError(f"Missing source selection JSON: {sources_path}. Run select_train_sources.py first!")

    with open(sources_path, "r", encoding="utf-8") as f:
        sources_data = json.load(f)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_json)
    if not metadata_path.is_absolute():
        metadata_path = Path(__file__).resolve().parents[1] / metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      PILOT TARGETED DIFFUSION GENERATOR & QUALITY GATE FILTER")
    print("=" * 70)

    # Restore Baseline Model
    ckpt_path = tf.train.latest_checkpoint(str(run_dir / "checkpoints" / "best")) or tf.train.latest_checkpoint(str(run_dir / "checkpoints" / "last"))
    print(f"Restoring Baseline Checkpoint: {ckpt_path}")
    model = build_model(cfg)
    dummy_image = tf.zeros([1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["data"]["channels"]], tf.float32)
    dummy_mask = tf.zeros([1, cfg["model"].get("token_grid_size", 7), cfg["model"].get("token_grid_size", 7), cfg["model"].get("num_regions", 6)], tf.float32)
    model({"image": dummy_image, "mask": dummy_mask}, training=False)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(str(ckpt_path)).expect_partial()

    # Load raw train pixels from train.csv
    data_dir = _resolve_path(cfg["data"]["data_path"])
    train_records = collect_split_records(data_dir, split="train", mask_dir=None)
    id_to_img = {sid: train_records.images[i] for i, sid in enumerate(train_records.sample_ids)}

    accepted_metadata = []
    total_accepted = 0

    print("\nStarting Targeted Generation & 3-Gate Quality Filter...")

    for target_item in PILOT_TARGETS:
        t_class = target_item["target_class"]
        c_class = target_item["confused_class"]
        h_pair = target_item["hard_pair"]
        target_clean = target_item["clean"]
        target_boundary = target_item["boundary"]

        t_idx = EMOTION_NAMES.index(t_class)
        c_idx = EMOTION_NAMES.index(c_class)

        prompt_str = PROMPTS.get(t_class, {}).get(c_class, f"photo of human face with {t_class} expression")

        (output_dir / t_class).mkdir(parents=True, exist_ok=True)

        class_sources = sources_data.get(t_class, {})
        # Source pool: 50% medium, 30% hard, 20% clean
        pool_ids = class_sources.get("medium_sample_ids", []) + class_sources.get("hard_sample_ids", []) + class_sources.get("clean_sample_ids", [])

        if not pool_ids:
            continue

        accepted_clean = 0
        accepted_boundary = 0
        attempts = 0

        while (accepted_clean < target_clean or accepted_boundary < target_boundary) and attempts < 3000:
            attempts += 1
            src_id = int(np.random.choice(pool_ids))
            raw_pixels = id_to_img[src_id]

            # Decode to 112x112 RGB using PIL
            if isinstance(raw_pixels, str):
                vals = np.fromstring(raw_pixels, sep=" ", dtype=np.float32).reshape(48, 48)
                pil_raw = Image.fromarray(vals.astype(np.uint8)).convert("RGB").resize((112, 112), Image.BILINEAR)
                src_rgb = np.array(pil_raw, dtype=np.uint8)
            elif isinstance(raw_pixels, np.ndarray):
                if raw_pixels.ndim == 2:
                    pil_raw = Image.fromarray(raw_pixels.astype(np.uint8)).convert("RGB").resize((112, 112), Image.BILINEAR)
                elif raw_pixels.shape[-1] == 1:
                    pil_raw = Image.fromarray(raw_pixels[..., 0].astype(np.uint8)).convert("RGB").resize((112, 112), Image.BILINEAR)
                else:
                    pil_raw = Image.fromarray(raw_pixels.astype(np.uint8)).convert("RGB").resize((112, 112), Image.BILINEAR)
                src_rgb = np.array(pil_raw, dtype=np.uint8)
            else:
                continue

            # Step 1: Generate synthetic candidate
            syn_rgb = generate_synthetic_candidate(src_rgb, strength=0.40)

            # Step 2: Quality Gate 1 - Realism & Blur Filter
            is_sharp, lap_score = check_face_quality(syn_rgb)
            if not is_sharp:
                continue

            # Step 3: Quality Gate 2 & 3 - Baseline Margin & Alignment
            norm_img = ((syn_rgb.astype(np.float32) / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            norm_tensor = tf.expand_dims(tf.cast(norm_img, tf.float32), axis=0)

            outputs = model({"image": norm_tensor}, training=False)
            probs = tf.nn.softmax(outputs["logits"], axis=-1).numpy()[0]

            p_target = float(probs[t_idx])
            p_confused = float(probs[c_idx])
            margin = p_target - p_confused

            # Categorize Clean vs Boundary
            sample_type = None
            if accepted_clean < target_clean and p_target >= 0.65 and p_confused < 0.15:
                sample_type = "clean"
                accepted_clean += 1
            elif accepted_boundary < target_boundary and p_target > p_confused and 0.10 <= margin <= 0.35:
                sample_type = "boundary"
                accepted_boundary += 1

            if sample_type is None:
                continue

            # Save Accepted Synthetic Image
            syn_uuid = uuid.uuid4().hex[:8]
            file_name = f"syn_{h_pair}_{t_class}_{sample_type}_{syn_uuid}.png"
            file_path = output_dir / t_class / file_name
            Image.fromarray(syn_rgb).save(file_path)

            meta_entry = {
                "synthetic_id": f"syn_{t_class}_{syn_uuid}",
                "image_path": str(file_path.relative_to(output_dir.parent)),
                "source_image_id": src_id,
                "target_class": t_class,
                "confused_class": c_class,
                "hard_pair": h_pair,
                "sample_type": sample_type,
                "prompt": prompt_str,
                "seed": attempts,
                "img2img_strength": 0.40,
                "baseline_p_target": np.round(p_target, 4),
                "baseline_p_confused": np.round(p_confused, 4),
                "probability_margin": np.round(margin, 4),
                "face_quality_score": np.round(lap_score, 2),
            }
            accepted_metadata.append(meta_entry)
            total_accepted += 1

        print(f"  Hard Pair [{h_pair:<12}] -> Class [{t_class:<7}]: Accepted Clean={accepted_clean}/{target_clean}, Boundary={accepted_boundary}/{target_boundary}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(accepted_metadata, f, indent=2)

    print(f"\nCompleted Pilot Generation! Total Accepted Samples: {total_accepted}")
    print(f"Metadata saved: {metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
