"""Verified, train-only teacher targets. No TensorFlow/PyTorch import here."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
RECIPE = "clean_rgb224_processor__raw_multi5_mean_l2__learned_scale__v1"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_module():
    # Load just the prompt/cache utility; models/__init__ imports TensorFlow.
    spec = importlib.util.spec_from_file_location("kd_clip_text", ROOT / "models/clip_text_encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_records(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("Teacher preparation never reads test samples")
    root = resolve(cfg["data"]["data_path"])
    candidates = [root] + sorted({p.parent for p in root.rglob("train.csv")})
    directory = next(p for p in candidates if all((p / (s + ".csv")).is_file()
                                                 for s in ("train", "val", "test")))
    with (directory / (split + ".csv")).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        lk = next(k for k in ("emotion", "label", "target", "class", "y") if k in reader.fieldnames)
        ik = next(k for k in ("pixels", "image_path", "filepath", "path", "image", "file")
                  if k in reader.fieldnames)
        rows = list(reader)
    labels = np.array([int(r[lk]) for r in rows], dtype=np.int64)
    if not len(labels) or labels.min() != 0 or labels.max() != 6:
        raise ValueError("Expected existing normalized RAF-DB CSV labels [0,6]; no CSV rewriting")
    return SimpleNamespace(images=np.array([r[ik] for r in rows]),
                           labels=labels, sample_ids=np.arange(len(rows), dtype=np.int64))


def source_hashes(images):
    # This lineage requires original image files, not upsampled student tensors.
    return np.array([sha256(str(path)) for path in images])


def prototype_contract(cfg):
    path = resolve(cfg["model"]["clip_prototypes_path"])
    mod = text_module()
    prompt_hash, prompt_count = mod.compute_prompt_hash(mod.default_prompt_bank(7))
    valid = mod.validate_cache_provenance(str(path), str(path) + ".meta.json",
                                         cfg["vlm_teacher"]["model_name"], (7, 5, 768),
                                         prompt_hash, prompt_count)
    if not valid:
        raise ValueError("Missing/invalid REAL SigLIP2 multi5 text cache: " + str(path))
    return sha256(path)


def load_teacher_logits(cfg, records, split="train"):
    if split != "train":
        raise ValueError("KD targets are training-only")
    path = resolve(cfg["vlm_teacher"]["cache_path"])
    with np.load(path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"].item()))
        expected = dict(recipe=RECIPE, class_names=CLASSES, split="train",
                        model_name=cfg["vlm_teacher"]["model_name"],
                        prototype_sha256=prototype_contract(cfg))
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError("Teacher cache provenance mismatch: " + key)
        if not meta.get("model_commit") or not meta.get("processor"):
            raise ValueError("Teacher cache missing model/processor provenance")
        for key, current in (("sample_ids", records.sample_ids),
                             ("labels", records.labels),
                             ("images", np.asarray(records.images).astype(str)),
                             ("source_sha256", source_hashes(records.images))):
            if not np.array_equal(cache[key], current):
                raise ValueError("Teacher cache sample alignment/content mismatch: " + key)
        logits = np.asarray(cache["logits"], dtype=np.float32)
        if logits.shape != (len(records.labels), 7) or not np.isfinite(logits).all():
            raise ValueError("Invalid teacher logits shape/values")
        if hashlib.sha256(logits.tobytes()).hexdigest() != meta.get("logits_sha256"):
            raise ValueError("Teacher logits checksum mismatch")
    print(f"[VLM_KD_CACHE_OK] n={len(logits)} logits={logits.shape} "
          f"commit={meta['model_commit']} classes={CLASSES}", flush=True)
    return logits
