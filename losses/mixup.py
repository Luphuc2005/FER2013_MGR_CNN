"""Mixup data augmentation and loss for FER2013.

Implements sample-level Beta(alpha, alpha) interpolation of input images
and one-hot emotion targets.
Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
"""
from __future__ import annotations

import tensorflow as tf


def sample_beta(alpha: float, batch_size: tf.Tensor) -> tf.Tensor:
    """Sample batch_size independent values from Beta(alpha, alpha).

    Uses the property: if X ~ Gamma(alpha, 1) and Y ~ Gamma(alpha, 1) independently,
    then X / (X + Y) ~ Beta(alpha, alpha).

    Clamps lambda to [0.5, 1.0] so that sample i always retains its original label
    as the dominant class.
    """
    alpha_t = tf.cast(alpha, tf.float32)
    # tf.random.gamma produces shape [batch_size]
    g1 = tf.random.gamma(shape=[batch_size], alpha=alpha_t, beta=1.0, dtype=tf.float32)
    g2 = tf.random.gamma(shape=[batch_size], alpha=alpha_t, beta=1.0, dtype=tf.float32)
    g1 = tf.reshape(g1, [batch_size])
    g2 = tf.reshape(g2, [batch_size])

    lam = g1 / (g1 + g2 + 1e-7)
    lam = tf.clip_by_value(lam, 0.0, 1.0)
    # Ensure dominant label is the original sample (lam >= 0.5)
    lam = tf.maximum(lam, 1.0 - lam)
    return lam


def mixup_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    num_classes: int,
    alpha: float = 0.3,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply mixup interpolation to a batch of images and integer labels.

    Args:
        images: [B, H, W, C] images tensor.
        labels: [B] integer class labels.
        num_classes: total number of emotion classes.
        alpha: Beta distribution parameter (e.g. 0.2 to 0.4).

    Returns:
        mixed_images: [B, H, W, C] mixed images with same dtype as input.
        mixed_labels: [B, num_classes] float32 one-hot soft label distribution.
    """
    batch_size = tf.shape(images)[0]
    if alpha <= 0.0:
        labels_oh = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes, dtype=tf.float32)
        return images, labels_oh

    indices = tf.random.shuffle(tf.range(batch_size))
    lam = sample_beta(alpha, batch_size)  # [B] float32 in [0.5, 1.0]

    # Image interpolation (broadcasting over H, W, C)
    lam_img = tf.cast(tf.reshape(lam, [-1, 1, 1, 1]), images.dtype)
    shuffled_images = tf.gather(images, indices)
    mixed_images = lam_img * images + (1.0 - lam_img) * shuffled_images

    # Label interpolation (broadcasting over num_classes)
    lam_lbl = tf.reshape(lam, [-1, 1])  # [B, 1] float32
    labels_oh = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes, dtype=tf.float32)
    shuffled_labels_oh = tf.gather(labels_oh, indices)
    mixed_labels = lam_lbl * labels_oh + (1.0 - lam_lbl) * shuffled_labels_oh

    return mixed_images, mixed_labels


def mixup_loss(
    mixed_labels: tf.Tensor,
    logits: tf.Tensor,
    label_smoothing: float = 0.0,
) -> tf.Tensor:
    """Categorical cross-entropy loss with optional label smoothing for mixed labels.

    Args:
        mixed_labels: [B, num_classes] float32 soft label targets.
        logits: [B, num_classes] raw model output logits.
        label_smoothing: label smoothing factor applied to targets.

    Returns:
        Scalar mean cross-entropy loss.
    """
    logits = tf.cast(logits, tf.float32)
    targets = tf.cast(mixed_labels, tf.float32)
    if label_smoothing > 0.0:
        num_classes = tf.cast(tf.shape(targets)[-1], tf.float32)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / num_classes
    loss = tf.keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
    return tf.reduce_mean(loss)
