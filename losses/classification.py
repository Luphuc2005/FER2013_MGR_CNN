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
    logits = tf.cast(outputs["logits"], tf.float32)
    if label_smoothing > 0.0:
        targets = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / float(num_classes)
        ce = tf.keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
    else:
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    if class_weights is not None:
        weights = tf.cast(tf.gather(class_weights, labels), tf.float32)
        ce = tf.reduce_sum(ce * weights) / tf.reduce_sum(weights)
    else:
        ce = tf.reduce_mean(ce)
    ce = tf.cast(ce, tf.float32)
    ortho_raw = outputs.get("ortho_loss", tf.constant(0.0, dtype=tf.float32))
    ortho = tf.cast(ortho_raw, tf.float32) if ortho_raw is not None else tf.constant(0.0, dtype=tf.float32)
    total = ce + tf.cast(ortho_weight, tf.float32) * ortho
    cnn_aux_logits = outputs.get("cnn_aux_logits")
    if cnn_aux_logits is not None:
        cnn_aux_logits = tf.cast(cnn_aux_logits, tf.float32)
        if label_smoothing > 0.0:
            targets = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
            targets = targets * (1.0 - label_smoothing) + label_smoothing / float(num_classes)
            aux = tf.keras.losses.categorical_crossentropy(targets, cnn_aux_logits, from_logits=True)
        else:
            aux = tf.keras.losses.sparse_categorical_crossentropy(labels, cnn_aux_logits, from_logits=True)
        aux = tf.reduce_mean(aux)
        aux = tf.cast(aux, tf.float32)
        total = total + tf.cast(cnn_aux_weight, tf.float32) * aux
    else:
        aux = tf.constant(0.0, dtype=tf.float32)
    return total, {"ce": ce, "ortho": ortho, "cnn_aux": aux}
