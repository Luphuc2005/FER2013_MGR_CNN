"""Dynamic Part-Aware Spatial Attention Layer.
Learns adaptive spatial attention distributions for Upper face, Lower face, and AU regions
from Stage 3 feature maps [B, H, W, C], eliminating rigid manual row slicing.
"""
from __future__ import annotations
import tensorflow as tf


class DynamicPartAttention(tf.keras.layers.Layer):
    """
    Learns 3 soft spatial attention masks (upper face, lower face, AU region)
    from Stage 3 spatial feature maps [B, H, W, C], replacing fixed row-slicing.
    """
    def __init__(self, channels: int = 512, bottleneck: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.bottleneck = int(bottleneck)
        self.conv_reduce = tf.keras.layers.Conv2D(
            self.bottleneck, kernel_size=3, padding="same", use_bias=True, name="part_conv1"
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="part_norm")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="part_gelu")
        # 3 attention maps: index 0: Upper, index 1: Lower, index 2: AU
        self.conv_out = tf.keras.layers.Conv2D(
            3, kernel_size=1, padding="same", use_bias=True, dtype="float32", name="part_conv2"
        )

    def call(self, features: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        features_f32 = tf.cast(features, tf.float32)
        shape = tf.shape(features_f32)
        B, H, W = shape[0], shape[1], shape[2]

        h = self.conv_reduce(features_f32)
        h = self.norm(h)
        h = self.act(h)
        logits = self.conv_out(h)  # [B, H, W, 3]

        # Spatial softmax across H*W independently for each of the 3 part channels
        logits_flat = tf.reshape(logits, [B, H * W, 3])  # [B, HW, 3]
        attn_flat = tf.nn.softmax(logits_flat, axis=1)    # [B, HW, 3]
        attn_maps = tf.reshape(attn_flat, [B, H, W, 3])   # [B, H, W, 3]

        # Weighted spatial sum over (H, W) -> [B, C]
        z_upper = tf.reduce_sum(tf.expand_dims(attn_maps[:, :, :, 0], axis=-1) * features_f32, axis=[1, 2])
        z_lower = tf.reduce_sum(tf.expand_dims(attn_maps[:, :, :, 1], axis=-1) * features_f32, axis=[1, 2])
        z_au = tf.reduce_sum(tf.expand_dims(attn_maps[:, :, :, 2], axis=-1) * features_f32, axis=[1, 2])

        return (
            tf.cast(z_upper, features.dtype),
            tf.cast(z_lower, features.dtype),
            tf.cast(z_au, features.dtype),
            attn_maps,
        )
