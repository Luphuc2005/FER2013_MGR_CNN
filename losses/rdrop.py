"""Training-only R-Drop over V5's existing fused logits; no model parameters."""
from __future__ import annotations

import math
import tensorflow as tf

from .classification import supervised_mgr_loss


def symmetric_kl(logits_a, logits_b):
    """Mean over samples of 0.5 * (KL(p||q) + KL(q||p)); sum over classes.

    Log-softmax and the KL reduction run in float32 even with mixed precision.
    Both predictions receive gradients. Temperature is fixed at one.
    """
    a = tf.cast(logits_a, tf.float32)
    b = tf.cast(logits_b, tf.float32)
    log_p = tf.nn.log_softmax(a, axis=-1)
    log_q = tf.nn.log_softmax(b, axis=-1)
    per_sample = 0.5 * tf.reduce_sum(
        (tf.exp(log_p) - tf.exp(log_q)) * (log_p - log_q), axis=-1
    )
    return tf.reduce_mean(per_sample)


def forward_training_loss(
    model, features, labels, *, lambda_rdrop=0.0,
    lambda_sem_runtime=None, lambda_vlm_kd=0.0, kd_temperature=2.0, **loss_kwargs,
):
    """One stochastic forward when disabled; two at the SAME weights when enabled.

    The caller owns GradientTape and SAM perturbation. Call this helper once at
    theta and again at theta+epsilon, never compare predictions across SAM passes.
    Each forward includes the complete existing supervised objective (also LGSA).
    Metrics use the first prediction, never a training-only prediction ensemble.
    """
    weight = float(lambda_rdrop)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("lambda_rdrop must be finite and >= 0.")

    from .vlm_kd import validate_kd_settings
    kd_weight, temperature = validate_kd_settings(dict(
        lambda_vlm_kd=lambda_vlm_kd, kd_temperature=kd_temperature, lambda_rdrop=weight))
    student_features = features
    if kd_weight > 0:
        if not isinstance(features, dict) or "teacher_logits" not in features:
            raise ValueError("VLM-KD requires verified teacher_logits in the training batch")
        student_features = {k: v for k, v in features.items() if k != "teacher_logits"}

    def forward_once():
        outputs = model(student_features, training=True, labels=labels)
        loss_outputs = dict(outputs)
        if lambda_sem_runtime is not None and outputs.get("semantic_logits") is not None:
            loss_outputs["lambda_sem"] = lambda_sem_runtime.read_value()
        loss, parts = supervised_mgr_loss(labels, loss_outputs, **loss_kwargs)
        if kd_weight > 0:
            from .vlm_kd import classification_kd
            kd = classification_kd(outputs["logits"], features["teacher_logits"], temperature)
            parts = dict(parts, vlm_kd=kd, weighted_vlm_kd=kd_weight * kd)
            loss = loss + parts["weighted_vlm_kd"]
        return outputs, loss, parts

    first, loss_a, parts_a = forward_once()
    if weight == 0.0:
        # Do not draw extra dropout masks or average a second objective.
        return first, loss_a, parts_a

    second, loss_b, parts_b = forward_once()
    kl = symmetric_kl(first["logits"], second["logits"])
    parts = {key: 0.5 * (parts_a[key] + parts_b[key]) for key in parts_a}
    parts["rdrop"] = kl
    parts["weighted_rdrop"] = tf.cast(weight, tf.float32) * kl
    parts["weighted_local_semantic"] = 0.5 * (
        tf.cast(first.get("lambda_local_sem", 0.0), tf.float32) * parts_a["local_semantic"]
        + tf.cast(second.get("lambda_local_sem", 0.0), tf.float32) * parts_b["local_semantic"]
    )
    return first, 0.5 * (loss_a + loss_b) + parts["weighted_rdrop"], parts
