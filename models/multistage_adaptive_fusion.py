"""Stage-2 local grid + Stage-3 parts + Stage-4 global adaptive fusion."""
from __future__ import annotations

import tensorflow as tf


class MultiStageAdaptiveFusion(tf.keras.layers.Layer):
    """Inputs: S2 [B,H,W,256], upper/lower/AU [B,512], global [B,1024].

    Returns fused [B,output_dim] and sample-wise S2/S3/S4 weights [B,3].
    All projections, softmax and fusion arithmetic use float32.
    """
    def __init__(self, projection_dim=256, output_dim=512, gate_hidden_dim=128, **kwargs):
        kwargs["dtype"] = "float32"
        super().__init__(**kwargs)
        self.projection_dim = int(projection_dim)
        self.output_dim = int(output_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        if min(self.projection_dim, self.output_dim, self.gate_hidden_dim) <= 0:
            raise ValueError("Fusion dimensions must be positive.")
        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=4, axes={-1: 256}),
            *[tf.keras.layers.InputSpec(ndim=2, axes={-1: 512}) for _ in range(3)],
            tf.keras.layers.InputSpec(ndim=2, axes={-1: 1024}),
        ]
        self.projections = [
            tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32"),
                tf.keras.layers.Dense(self.projection_dim, dtype="float32"),
            ], name=f"stage{stage}_projection")
            for stage in (2, 3, 4)
        ]
        self.gate = tf.keras.Sequential([
            tf.keras.layers.Dense(self.gate_hidden_dim, activation="gelu", dtype="float32"),
            tf.keras.layers.Dense(
                3, kernel_initializer="zeros", bias_initializer="zeros", dtype="float32",
            ),
        ], name="stage_weight_gate")
        self.fusion_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.output_dim, activation="gelu", dtype="float32"),
            tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32"),
        ], name="fusion_mlp")

    @staticmethod
    def local_grid_pool(stage2):
        """Average each quadrant independently, ordered TL, TR, BL, BR."""
        stage2 = tf.cast(stage2, tf.float32)
        shape = tf.shape(stage2)
        tf.debugging.assert_equal(shape[1:3] % 2, [0, 0],
                                  message="Stage-2 spatial dimensions must be even.")
        cells = tf.reshape(stage2, [shape[0], 2, shape[1] // 2, 2, shape[2] // 2, 256])
        cells = tf.reduce_mean(cells, axis=[2, 4])
        return tf.reshape(cells, [shape[0], 4 * 256])

    def call(self, inputs, training=False):
        stage2, upper, lower, au, global_vector = inputs
        vectors = (
            self.local_grid_pool(stage2),
            tf.concat([tf.cast(z, tf.float32) for z in (upper, lower, au)], axis=-1),
            tf.cast(global_vector, tf.float32),
        )
        projected = [layer(z, training=training)
                     for layer, z in zip(self.projections, vectors)]
        joined = tf.concat(projected, axis=-1)
        weights = tf.nn.softmax(self.gate(joined, training=training), axis=-1)
        weighted = tf.concat([z * weights[:, i:i + 1]
                              for i, z in enumerate(projected)], axis=-1)
        return self.fusion_mlp(weighted, training=training), weights

    def get_config(self):
        return {**super().get_config(), "projection_dim": self.projection_dim,
                "output_dim": self.output_dim, "gate_hidden_dim": self.gate_hidden_dim}
