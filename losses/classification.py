from __future__ import annotations

from typing import Dict, Optional, Tuple

import tensorflow as tf


def confusion_aware_hard_semantic_loss(
    labels: tf.Tensor,
    agg_sim: tf.Tensor,
    hard_pairs_matrix: tf.Tensor,
    margin: float = 0.15,
) -> tf.Tensor:
    """
    Confusion-Aware Hard Semantic Separation Loss.

    L_hard = mean(ReLU(margin - s_pos + s_neg))
    where s_pos is cosine similarity of true class, and s_neg is cosine similarity of hard negative pairs.
    Uses agg_sim in range [-1.0, 1.0] before scaling by semantic_logit_scale.
    """
    labels_int = tf.cast(labels, tf.int32)
    agg_sim = tf.cast(agg_sim, tf.float32)
    margin_t = tf.cast(margin, tf.float32)

    # s_pos: [B] -> positive cosine similarity for true class
    s_pos = tf.gather(agg_sim, labels_int, batch_dims=1)
    s_pos_exp = tf.expand_dims(s_pos, axis=1)  # [B, 1]

    # hard_mask: [B, num_classes] -> 1.0 for hard negative pairs of true label
    hard_mask = tf.gather(hard_pairs_matrix, labels_int)  # [B, num_classes]

    # Pairwise margin loss: ReLU(margin - s_pos + s_neg)
    pair_loss = tf.nn.relu(margin_t - s_pos_exp + agg_sim)  # [B, num_classes]
    valid_pair_loss = pair_loss * hard_mask  # [B, num_classes]

    total_pairs = tf.reduce_sum(hard_mask)
    hard_loss = tf.cond(
        total_pairs > 0.0,
        lambda: tf.reduce_sum(valid_pair_loss) / total_pairs,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    return hard_loss


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

    semantic_logits = outputs.get("semantic_logits")
    if semantic_logits is not None:
        semantic_logits = tf.cast(semantic_logits, tf.float32)
        if label_smoothing > 0.0:
            targets = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
            targets = targets * (1.0 - label_smoothing) + label_smoothing / float(num_classes)
            sem = tf.keras.losses.categorical_crossentropy(targets, semantic_logits, from_logits=True)
        else:
            sem = tf.keras.losses.sparse_categorical_crossentropy(labels, semantic_logits, from_logits=True)
        sem_loss = tf.reduce_mean(sem)
        sem_loss = tf.cast(sem_loss, tf.float32)
        lambda_sem = float(outputs.get("lambda_sem", 0.1))
        total = total + tf.cast(lambda_sem, tf.float32) * sem_loss
    else:
        sem_loss = tf.constant(0.0, dtype=tf.float32)

    hard_loss = tf.constant(0.0, dtype=tf.float32)
    agg_sim = outputs.get("agg_sim")
    hard_pairs_matrix = outputs.get("hard_pairs_matrix")
    lambda_hard = float(outputs.get("lambda_hard", 0.0))
    if agg_sim is not None and hard_pairs_matrix is not None and lambda_hard > 0.0:
        hard_margin = float(outputs.get("hard_margin", 0.15))
        hard_loss = confusion_aware_hard_semantic_loss(
            labels=labels,
            agg_sim=agg_sim,
            hard_pairs_matrix=hard_pairs_matrix,
            margin=hard_margin,
        )
        total = total + tf.cast(lambda_hard, tf.float32) * hard_loss

    return total, {
        "ce": ce,
        "ortho": ortho,
        "cnn_aux": aux,
        "semantic": sem_loss,
        "hard_semantic": hard_loss,
    }
