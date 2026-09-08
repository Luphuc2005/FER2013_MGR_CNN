"""Train-label-only class weighting for the isolated RAF V5 experiment."""
from __future__ import annotations

import hashlib
import numpy as np

RAF_CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def weights_from_labels(labels, target_classes, power=0.5, max_ratio=2.0):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu":
        raise ValueError("Expected a nonempty integer training label vector.")
    if np.any(labels < 0) or np.any(labels >= len(RAF_CLASSES)):
        raise ValueError("RAF labels must be in [0,6].")
    if not np.isfinite(power) or not 0 < power <= 1:
        raise ValueError("power must be in (0,1].")
    if not np.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("max_ratio must be finite and >= 1.")
    if not target_classes or len(set(target_classes)) != len(target_classes):
        raise ValueError("Expected distinct target classes.")
    if any(name not in RAF_CLASSES for name in target_classes):
        raise ValueError("Unknown target class.")
    counts = np.bincount(labels.astype(np.int64), minlength=len(RAF_CLASSES))
    if np.any(counts == 0):
        raise ValueError("All seven classes must occur in train.")
    ratios = np.ones(len(RAF_CLASSES), dtype=np.float64)
    for name in target_classes:
        i = RAF_CLASSES.index(name)
        ratios[i] = np.clip((len(labels) / (len(RAF_CLASSES) * counts[i])) ** power,
                            1.0, max_ratio)
    weights = ratios / np.average(ratios, weights=counts)
    return dict(class_names=list(RAF_CLASSES), counts=counts.tolist(),
                raw_ratios=ratios.tolist(), weights=weights.tolist(),
                normalization="train_frequency_mean_one", reduction="batch_mean",
                train_samples=int(len(labels)),
                train_labels_sha256=hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest())


def configure_training_ce_weights(cfg, train_labels):
    settings = cfg.get("training", {}).get("weighted_ce", {})
    enabled = bool(settings.get("enabled", False))
    requested = cfg.get("training", {}).get("loss") == "weighted_cross_entropy"
    if enabled != requested:
        raise ValueError("weighted_ce.enabled and training.loss must agree.")
    if not enabled:
        return
    if (cfg["model"]["name"] != "rafdb_siglip2_semantic_stable_v5_combined_ultimate"
            or cfg["data"]["num_classes"] != 7):
        raise ValueError("This weighting recipe is restricted to RAF V5 Combined.")
    names = cfg["data"].get("class_names") or cfg["data"].get("emotion_names") or RAF_CLASSES
    if tuple(names) != RAF_CLASSES:
        raise ValueError("Unexpected RAF class order.")
    report = weights_from_labels(train_labels, settings["target_classes"],
                                float(settings["power"]), float(settings["max_ratio"]))
    report.update(source_split="train", target_classes=settings["target_classes"],
                  power=float(settings["power"]), max_ratio=float(settings["max_ratio"]))
    cfg["training"]["resolved_ce_weight_report"] = report
    print("[WEIGHTED_CE] source=train reduction=batch_mean; CE only; both SAM passes", flush=True)
    for name, count, weight in zip(report["class_names"], report["counts"], report["weights"]):
        print(f"[WEIGHTED_CE] {name}: train_count={count} weight={weight:.8f}", flush=True)


def training_ce_kwargs(cfg):
    settings = cfg.get("training", {}).get("weighted_ce", {})
    if not settings.get("enabled", False):
        if cfg.get("training", {}).get("loss") == "weighted_cross_entropy":
            raise ValueError("weighted_cross_entropy requires weighted_ce.enabled.")
        return {}
    if cfg["training"].get("loss") != "weighted_cross_entropy":
        raise ValueError("weighted_ce.enabled requires weighted_cross_entropy.")
    report = cfg["training"].get("resolved_ce_weight_report")
    if report is None:
        raise ValueError("Compute CE weights from training records before building train steps.")
    return dict(class_weights=report["weights"], class_weight_reduction="batch_mean")
