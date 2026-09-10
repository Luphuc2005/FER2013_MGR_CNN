"""Sample-Adaptive Semantic Logit Fusion Gate.
Predicts an instance-specific fusion coefficient alpha(x) in [0, max_alpha]
from pooled visual features to dynamically weigh visual and semantic logits.
"""
from __future__ import annotations
import tensorflow as tf


class SampleAdaptiveFusionGate(tf.keras.layers.Layer):
    """
    Predicts a sample-adaptive fusion coefficient alpha(x) in [0, max_alpha]
    from pooled visual features, balancing visual and semantic logits dynamically.
    """
    def __init__(self, max_alpha: float = 0.20, hidden_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.max_alpha = float(max_alpha)
        self.hidden_dim = int(hidden_dim)
        self.dense1 = tf.keras.layers.Dense(self.hidden_dim, use_bias=True, name="gate_dense1")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="gate_norm")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="gate_gelu")
        self.dense2 = tf.keras.layers.Dense(1, use_bias=True, dtype="float32", name="gate_dense2")

    def call(self, pooled_features: tf.Tensor, training: bool = False) -> tf.Tensor:
        feat_f32 = tf.cast(pooled_features, tf.float32)
        h = self.dense1(feat_f32)
        h = self.norm(h)
        h = self.act(h)
        raw_gate = self.dense2(h)  # [B, 1]
        alpha = self.max_alpha * tf.nn.sigmoid(raw_gate)  # [B, 1] in [0, max_alpha]
        return alpha
