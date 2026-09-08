"""Direct global/part feature fusion for the opt-in RAF-DB v6 FER head."""
from __future__ import annotations

import tensorflow as tf


class GlobalRegionalFusion(tf.keras.layers.Layer):
    """Normalize/project global, upper, lower and AU vectors, then concatenate.

    Inputs: [B,1024], [B,512], [B,512], [B,512].
    Output: [B,4*projection_dim]. No residual bypass or feature fusion gate.
    Small projections stay in float32 under mixed precision.
    """

    def __init__(self, projection_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = int(projection_dim)
        if self.projection_dim <= 0:
            raise ValueError("global_regional_projection_dim must be positive.")
        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=2, axes={-1: channels})
            for channels in (1024, 512, 512, 512)
        ]
        self.projections = [
            tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32", name="norm"),
                tf.keras.layers.Dense(
                    self.projection_dim, kernel_initializer="he_normal",
                    dtype="float32", name="projection",
                ),
            ], name=f"{region}_projection")
            for region in ("global", "upper", "lower", "au")
        ]

    def call(self, inputs, training=False):
        projected = [
            layer(tf.cast(vector, tf.float32), training=training)
            for layer, vector in zip(self.projections, inputs)
        ]
        return tf.concat(projected, axis=-1)

    def get_config(self):
        return {**super().get_config(), "projection_dim": self.projection_dim}
