"""ArcMarginProduct (ArcFace) classification head for FER.

Reference:
    Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019).
"""
from __future__ import annotations

import math
from typing import Optional

import tensorflow as tf


class ArcMarginProduct(tf.keras.layers.Layer):
    """Additive Angular Margin (ArcFace) classification head.

    Args:
        num_classes: Number of expression classes (e.g. 7 for RAF-DB).
        scale: Feature scale factor s (default: 30.0).
        margin: Angular margin m in radians (default: 0.30 rad).
        easy_margin: Whether to use easy margin fallback (default: False).
        name: Layer name.
    """

    def __init__(
        self,
        num_classes: int = 7,
        scale: float = 30.0,
        margin: float = 0.30,
        easy_margin: bool = False,
        embed_dim: Optional[int] = None,
        s: Optional[float] = None,
        m: Optional[float] = None,
        name: Optional[str] = "arcmargin_classifier",
        **kwargs,
    ):
        super().__init__(name=name, dtype="float32")
        self.num_classes = int(num_classes)
        self.scale = float(s if s is not None else scale)
        self.margin = float(m if m is not None else margin)
        self.easy_margin = bool(easy_margin)
        self.embed_dim = int(embed_dim) if embed_dim is not None else None

        # Precompute trigonometric constants
        self.cos_m = float(math.cos(self.margin))
        self.sin_m = float(math.sin(self.margin))
        self.th = float(math.cos(math.pi - self.margin))
        self.mm = float(math.sin(math.pi - self.margin) * self.margin)

        if self.embed_dim is not None:
            self.build((None, self.embed_dim))

    def build(self, input_shape):
        feat_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="weight",
            shape=(feat_dim, self.num_classes),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, features, labels=None, training=False):
        """Forward pass.

        Args:
            features: [B, D] input feature representation (float32 or float16).
            labels: [B] integer class labels (ground-truth). Used during training.
            training: Boolean training flag.

        Returns:
            [B, num_classes] classification logits scaled by s.
        """
        features_f32 = tf.cast(features, tf.float32)
        # 1. L2 normalize features and classifier weights
        features_norm = tf.math.l2_normalize(features_f32, axis=-1)
        w_norm = tf.math.l2_normalize(self.W, axis=0)

        # 2. Cosine similarity: cos(theta) = x_norm * w_norm
        cosine = tf.matmul(features_norm, w_norm)
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # In inference/validation, output scaled cosine logits directly without label margin
        if not training or labels is None:
            return cosine * self.scale

        # 3. Training: compute cos(theta + m)
        sine = tf.sqrt(tf.clip_by_value(1.0 - tf.square(cosine), 1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = tf.where(cosine > 0.0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Inject margin only for ground-truth label
        labels_int = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels_int, depth=self.num_classes, dtype=tf.float32)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "scale": self.scale,
            "margin": self.margin,
            "easy_margin": self.easy_margin,
        })
        return config