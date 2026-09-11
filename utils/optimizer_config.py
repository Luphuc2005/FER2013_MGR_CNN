"""Resolve the base optimizer without importing TensorFlow."""
import math


def resolve_base_optimizer(training):
    # Preserve the historical AdamW default for configs without this field.
    name = str(training.get("base_optimizer", "adamw")).strip().lower()
    decay = float(training.get("weight_decay", 0.0))
    if name not in ("adam", "adamw", "sgd"):
        raise ValueError("base_optimizer must be adam, adamw, or sgd")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if name == "adam" and decay != 0.0:
        raise ValueError("Plain Adam requires weight_decay=0.0; coupled L2 is not implemented")
    return name, decay
