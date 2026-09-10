#!/usr/bin/env python3
"""Frozen SigLIP2 teacher extraction in its own GPU process; no student training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# TensorFlow is imported explicitly; transformers is used for config/processor and CPU reference only.
os.environ["USE_TF"] = "0"
from config import load_config
from utils.vlm_teacher_cache import (
    CLASSES, RECIPE, read_records, resolve, source_hashes, text_module,
    prototype_contract, load_teacher_logits, sha256,
)


def diagnostics(logits, labels, temperature):
    scores = logits.astype(np.float64) / temperature
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    return dict(accuracy=float(np.mean(logits.argmax(1) == labels)),
                mean_entropy=float(np.mean(-np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1))),
                mean_max_probability=float(p.max(1).mean()),
                predicted_class_counts=np.bincount(logits.argmax(1), minlength=7).tolist(),
                class_counts=np.bincount(labels, minlength=7).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from check_rafdb_v5_lgsa_vlm_kd import check_contract
    cfg = check_contract(args.config)
    if float(cfg["training"]["lambda_vlm_kd"]) == 0:
        print("VLM_KD_DISABLED: no teacher load/cache required")
        return
    records = read_records(cfg, "train")
    path = resolve(cfg["vlm_teacher"]["cache_path"])
    if path.exists():
        load_teacher_logits(cfg, records)
        print("Reusing verified teacher cache: " + str(path))
        return

    import tensorflow as tf
    # Use the same TensorFlow installation/GPU allocation as the FER student.
    devices = tf.config.list_physical_devices("GPU")
    print(f"[TEACHER_TF_ENV] python={sys.executable} tf={tf.__version__} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
          f"gpus={devices}", flush=True)
    if not devices:
        raise RuntimeError("TensorFlow teacher cannot see the allocated GPU; check Slurm/TF CUDA setup")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    import transformers
    from transformers import AutoImageProcessor, SiglipVisionConfig
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from PIL import Image
    from utils.siglip_vision_tf import FrozenSiglipVisionTF, verify_cpu_reference

    name = cfg["vlm_teacher"]["model_name"]
    config_path = hf_hub_download(name, "config.json")
    commit = Path(config_path).parent.name
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise RuntimeError("Could not resolve an immutable Hugging Face model commit")
    with open(config_path, encoding="utf-8") as handle:
        hf_config = json.load(handle)
    if hf_config.get("model_type") != "siglip":
        raise ValueError("Unsupported teacher architecture")
    vision_config = SiglipVisionConfig(**hf_config["vision_config"]).to_dict()
    weights_path = hf_hub_download(name, "model.safetensors", revision=commit)
    with safe_open(weights_path, framework="np") as handle:
        weights = {k: handle.get_tensor(k).astype(np.float32)
                   for k in handle.keys() if k.startswith("vision_model.")}
        scale = float(np.exp(handle.get_tensor("logit_scale").astype(np.float32)).item())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid pretrained SigLIP2 logit scale")

    # Reuse verified text vectors. Generating a missing text cache may use the
    # existing CPU text encoder, but no PyTorch CUDA is required.
    mod = text_module()
    bank = mod.get_or_compute_clip_text_prototypes(
        model_name=name, cache_path=str(resolve(cfg["model"]["clip_prototypes_path"])),
        embedding_dim=768, multi_prototype=True, num_classes=7, class_names=CLASSES)
    bank_hash = prototype_contract(cfg)
    prototypes = bank.mean(axis=1).astype(np.float32)
    norms = np.linalg.norm(prototypes, axis=-1, keepdims=True)
    if not np.isfinite(prototypes).all() or not np.all(norms > 1e-8):
        raise ValueError("Invalid mean emotion prototype")
    prototypes /= norms
    processor = AutoImageProcessor.from_pretrained(name, revision=commit, use_fast=False)

    def prepare_pixels(paths):
        images = []
        for value in paths:
            with Image.open(resolve(str(value))) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="np")["pixel_values"]
        if tuple(pixels.shape[1:]) != (3, 224, 224):
            raise ValueError(f"Expected teacher pixels [B,3,224,224], got {pixels.shape}")
        return pixels.transpose(0, 2, 3, 1).astype(np.float32)

    with tf.device("/GPU:0"):
        teacher = FrozenSiglipVisionTF(weights, vision_config)
        text_vectors = tf.constant(prototypes)
    print(f"SIGLIP_TF_WEIGHTS_OK matched={teacher.matched_tensors}/{len(weights)} "
          f"commit={commit} scale={scale}; frozen constants", flush=True)
    # Fail before cache creation if the port differs from the reference.
    probe = prepare_pixels(records.images[:2])
    parity_pixels = np.concatenate([probe, probe[:, :, ::-1, :]], axis=0)
    parity = verify_cpu_reference(teacher, weights, vision_config,
                                  parity_pixels, prototypes, scale)
    del weights
    assert not teacher.trainable_variables

    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)],
                 jit_compile=False)
    def teacher_logits(pixels):
        vectors = tf.math.l2_normalize(teacher(pixels), axis=-1)
        return scale * tf.matmul(vectors, text_vectors, transpose_b=True)

    batch_size = int(cfg["vlm_teacher"]["batch_size"])
    def predict(split_records):
        results = []
        for start in range(0, len(split_records.labels), batch_size):
            pixels = prepare_pixels(split_records.images[start:start + batch_size])
            logits = teacher_logits(tf.constant(pixels))
            tf.debugging.assert_all_finite(logits, "Nonfinite TensorFlow teacher logits")
            results.append(logits.numpy())
            if start % (batch_size * 20) == 0:
                print(f"TEACHER_TF_BATCH {start}/{len(split_records.labels)} "
                      f"device={logits.device}", flush=True)
        return np.concatenate(results).astype(np.float32)

    logits = predict(records)
    val_records = read_records(cfg, "val")
    val_logits = predict(val_records)
    assert not teacher.trainable_variables
    meta = dict(recipe=RECIPE, class_names=CLASSES, split="train", model_name=name,
                model_commit=commit, processor=processor.to_dict(),
                prototype_sha256=bank_hash, logit_scale=scale,
                logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                backend="tensorflow", tensorflow_version=tf.__version__,
                implementation_sha256=sha256(ROOT / "utils/siglip_vision_tf.py"),
                reference_parity=parity, transformers_version=transformers.__version__,
                teacher_view="original clean image -> official 224 processor",
                student_view="unchanged V5 112 augmentation",
                train_diagnostics=diagnostics(logits, records.labels, cfg["training"]["kd_temperature"]),
                val_diagnostics=diagnostics(val_logits, val_records.labels, cfg["training"]["kd_temperature"]))
    print("TEACHER_DIAGNOSTICS " + json.dumps({k: meta[k] for k in
          ("train_diagnostics", "val_diagnostics")}), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: do not replace another job's cache.
    with path.open("xb") as handle:
        np.savez_compressed(handle, logits=logits, labels=records.labels,
                            sample_ids=records.sample_ids, images=records.images.astype(str),
                            source_sha256=source_hashes(records.images), metadata=json.dumps(meta))
    load_teacher_logits(cfg, records)
    print("TEACHER_CACHE_READY " + str(path), flush=True)


if __name__ == "__main__":
    main()
