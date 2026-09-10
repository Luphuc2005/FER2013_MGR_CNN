"""Opt-in spatial pooling, initialized to the existing regional mean pooling."""
import tensorflow as tf


class SoftRegionalPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Zero scores -> uniform attention at initialization. Float32 softmax is
        # deliberate even under mixed precision; no temperature or extra loss.
        self.score = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer="zeros", dtype="float32", name="spatial_score"
        )

    def call(self, features):
        features_f32 = tf.cast(features, tf.float32)
        shape = tf.shape(features_f32)
        tokens = tf.reshape(features_f32, [shape[0], shape[1] * shape[2], shape[3]])
        weights = tf.nn.softmax(self.score(tokens), axis=1)
        pooled = tf.reduce_sum(weights * tokens, axis=1)
        return tf.cast(pooled, features.dtype)
