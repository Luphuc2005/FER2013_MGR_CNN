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
# Keep transformers from initializing TensorFlow in the teacher GPU process.
os.environ["USE_TF"] = "0"
from config import load_config
from utils.vlm_teacher_cache import (
    CLASSES, RECIPE, read_records, resolve, source_hashes, text_module,
    prototype_contract, load_teacher_logits,
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

    import torch
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher extraction requires the allocated GPU")
    name = cfg["vlm_teacher"]["model_name"]
    mod = text_module()
    bank = mod.get_or_compute_clip_text_prototypes(
        model_name=name, cache_path=str(resolve(cfg["model"]["clip_prototypes_path"])),
        embedding_dim=768, multi_prototype=True, num_classes=7, class_names=CLASSES)
    bank_hash = prototype_contract(cfg)
    # Reuse all five RAW text vectors, without student semantic centering.
    prototypes = torch.tensor(bank.mean(axis=1), dtype=torch.float32, device="cuda")
    if not torch.isfinite(prototypes).all() or not torch.all(prototypes.norm(dim=-1) > 1e-8):
        raise ValueError("Invalid mean emotion prototype")
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    assert tuple(prototypes.shape) == (7, 768)
    teacher = AutoModel.from_pretrained(name).float().to("cuda").eval()
    teacher.requires_grad_(False)
    commit = getattr(teacher.config, "_commit_hash", None)
    if not commit:
        raise RuntimeError("Cannot record resolved teacher model commit")
    processor = AutoImageProcessor.from_pretrained(name, revision=commit)
    scale = float(teacher.logit_scale.exp().item())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid pretrained SigLIP2 logit scale")
    print(f"TEACHER_FROZEN commit={commit} scale={scale} prototypes=(7,768)", flush=True)
    batch_size = int(cfg["vlm_teacher"]["batch_size"])

    def predict(split_records):
        results = []
        with torch.inference_mode():
            for start in range(0, len(split_records.labels), batch_size):
                images = []
                for value in split_records.images[start:start + batch_size]:
                    with Image.open(resolve(str(value))) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt")
                pixels = inputs["pixel_values"]
                if tuple(pixels.shape[1:]) != (3, 224, 224):
                    raise ValueError(f"Expected teacher pixels [B,3,224,224], got {pixels.shape}")
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                vectors = teacher.get_image_features(**inputs)
                if not isinstance(vectors, torch.Tensor):
                    vectors = getattr(vectors, "pooler_output", None)
                if vectors is None or tuple(vectors.shape) != (len(images), 768):
                    raise ValueError("Unexpected SigLIP2 pooled image features")
                vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
                # Shared scalar logit_bias cancels exactly under softmax.
                logits = scale * vectors @ prototypes.T
                results.append(logits.cpu().numpy())
                if start % (batch_size * 20) == 0:
                    print(f"TEACHER_BATCH {start}/{len(split_records.labels)}", flush=True)
        return np.concatenate(results).astype(np.float32)

    logits = predict(records)
    val_records = read_records(cfg, "val")
    val_logits = predict(val_records)
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    meta = dict(recipe=RECIPE, class_names=CLASSES, split="train", model_name=name,
                model_commit=commit, processor=processor.to_dict(),
                prototype_sha256=bank_hash, logit_scale=scale,
                logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                torch_version=torch.__version__, transformers_version=transformers.__version__,
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
