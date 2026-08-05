from __future__ import annotations

from typing import Dict, Optional, Tuple

import tensorflow as tf


def supervised_mgr_loss(
    labels: tf.Tensor,
    outputs: Dict[str, tf.Tensor],
    *,
    num_classes: int,
    label_smoothing: float = 0.0,
    class_weights: Optional[tf.Tensor] = None,
    ortho_weight: float = 0.003,
    cnn_aux_weight: float = 0.4,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    logits = outputs["logits"]
    if label_smoothing > 0.0:
        targets = tf.one_hot(labels, depth=num_classes, dtype=logits.dtype)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / float(num_classes)
        ce = tf.keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
    else:
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    if class_weights is not None:
        weights = tf.gather(class_weights, labels)
        ce = tf.reduce_sum(ce * weights) / tf.reduce_sum(weights)
    else:
        ce = tf.reduce_mean(ce)
    ortho = outputs.get("ortho_loss", tf.constant(0.0, dtype=tf.float32))
    total = ce + ortho_weight * ortho
    cnn_aux_logits = outputs.get("cnn_aux_logits")
    if cnn_aux_logits is not None:
        if label_smoothing > 0.0:
            targets = tf.one_hot(labels, depth=num_classes, dtype=cnn_aux_logits.dtype)
            targets = targets * (1.0 - label_smoothing) + label_smoothing / float(num_classes)
            aux = tf.keras.losses.categorical_crossentropy(targets, cnn_aux_logits, from_logits=True)
        else:
            aux = tf.keras.losses.sparse_categorical_crossentropy(labels, cnn_aux_logits, from_logits=True)
        aux = tf.reduce_mean(aux)
        total = total + cnn_aux_weight * aux
    else:
        aux = tf.constant(0.0, dtype=tf.float32)
    return total, {"ce": ce, "ortho": ortho, "cnn_aux": aux}
