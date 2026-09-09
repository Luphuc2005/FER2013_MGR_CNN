"""Train-label-only Logit Adjustment (Menon et al., ICLR 2021) for RAF-DB class imbalance.

Reference:
    "Long-Tail Learning via Logit Adjustment" (ICLR 2021)
    https://arxiv.org/abs/2007.07314

Formula:
    pi_c = N_c / N
    offsets_c = tau * log(pi_c)
    L_LA = - log( exp(z_y + offsets_y) / sum_j exp(z_j + offsets_j) )

During training: CE is computed on (logits + offsets).
During inference/eval: model uses raw logits z (unaltered feature geometry).
"""
from __future__ import annotations

import hashlib
import numpy as np

RAF_CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def compute_logit_adjustment(
    labels: np.ndarray,
    tau: float = 0.5,
    num_classes: int = 7,
) -> dict:
    """Compute logit adjustment offsets from training labels only.

    Args:
        labels: 1D array of integer labels [0, num_classes-1].
        tau: Temperature / scaling factor (typically 0.5 to 1.0).
        num_classes: Number of target classes (default 7).

    Returns:
        dict with class names, counts, frequencies (priors), offsets, and metadata.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu":
        raise ValueError("Expected a nonempty integer training label vector.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}].")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite float.")

    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("All classes must occur at least once in training data.")

    total = float(len(labels))
    priors = counts / total
    # Natural log of prior probabilities
    log_priors = np.log(priors)
    offsets = tau * log_priors

    class_names = list(RAF_CLASSES[:num_classes])
    return dict(
        class_names=class_names,
        counts=counts.tolist(),
        priors=priors.tolist(),
        offsets=offsets.astype(np.float32).tolist(),
        tau=float(tau),
        train_samples=int(len(labels)),
        train_labels_sha256=hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
    )


def configure_training_logit_adjustment(cfg: dict, train_labels: np.ndarray) -> None:
    """Resolve and attach logit adjustment offsets to training configuration."""
    settings = cfg.get("training", {}).get("logit_adjustment", {})
    if not settings.get("enabled", False):
        return

    tau = float(settings.get("tau", 0.5))
    num_classes = int(cfg.get("data", {}).get("num_classes", 7))
    report = compute_logit_adjustment(train_labels, tau=tau, num_classes=num_classes)

    cfg["training"]["resolved_logit_adjustment_report"] = report
    print(f"[LOGIT_ADJUSTMENT] Enabled (ICLR 2021) with tau={tau:.2f}; train_samples={report['train_samples']}", flush=True)
    for name, count, prior, offset in zip(report["class_names"], report["counts"], report["priors"], report["offsets"]):
        print(f"  [{name:8s}] count={count:5d}  prior={prior:6.4f}  offset={offset:+.4f}", flush=True)


def logit_adjustment_kwargs(cfg: dict) -> dict:
    """Return kwargs for supervised_mgr_loss if logit adjustment is enabled."""
    settings = cfg.get("training", {}).get("logit_adjustment", {})
    if not settings.get("enabled", False):
        return {}

    report = cfg.get("training", {}).get("resolved_logit_adjustment_report")
    if report is None:
        raise ValueError("Compute logit adjustment from training records before building train steps.")

    return dict(logit_adj_offsets=report["offsets"])
