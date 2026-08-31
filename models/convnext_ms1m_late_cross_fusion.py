"""ConvNeXt-B MS1M FER with Late Cross-Stage (S3->S4) Attention & Direct Feature Fusion.

Architecture details:
- Removes S2->S3 fusion; focuses solely on S3->S4 cross-attention.
- Projects S3 [B, 14, 14, 512] -> P3 [B, 7, 7, 1024].
- Cross Attention: Q = S4, K = P3, V = P3 -> delta4.
- Residual gate: G4 = S4 + alpha4 * delta4, where alpha4 = 0.2 * sigmoid(raw).
- Direct Feature Fusion in Classifier Head:
  f_main = GAP(G4) [B, 1024]
  f_cross = GAP(delta4) [B, 1024]
  proj_main = LayerNorm -> Dense(proj_dim) -> GELU -> Dropout(0.3)
  proj_cross = LayerNorm -> Dense(proj_dim) -> GELU -> Dropout(0.3)
  f_fused = Concat([proj_main, proj_cross]) [B, 2 * proj_dim]
  logits = MLP(f_fused) -> Dense(7)
- Logs alpha4, norm(delta4), norm(G4), and cross feature contribution ratio.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf

from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


class ProjectionDownsample(tf.keras.layers.Layer):
    """2x spatial downsample plus channel projection."""

    def __init__(self, out_dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            int(out_dim),
            kernel_size=2,
            strides=2,
            padding="valid",
            kernel_initializer="he_normal",
            name="conv2x2_stride2",
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")

    def call(self, x, training=False):
        x = self.conv(x)
        return self.norm(x)


class CrossAttention(tf.keras.layers.Layer):
    """Multi-head cross-attention with float32 logits/softmax."""

    def __init__(self, dim: int, num_heads: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q = tf.keras.layers.Dense(self.dim, use_bias=True, name="q")
        self.k = tf.keras.layers.Dense(self.dim, use_bias=True, name="k")
        self.v = tf.keras.layers.Dense(self.dim, use_bias=True, name="v")
        self.proj = tf.keras.layers.Dense(
            self.dim,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
            bias_initializer="zeros",
            name="proj_small_init",
        )

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        x = tf.reshape(x, [b, n, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        b = tf.shape(x)[0]
        n = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, n, self.dim])

    def call(self, q_tokens, kv_tokens, training=False):
        q = self._split_heads(self.q(q_tokens))
        k = self._split_heads(self.k(kv_tokens))
        v = self._split_heads(self.v(kv_tokens))

        logits = tf.matmul(tf.cast(q, tf.float32), tf.cast(k, tf.float32), transpose_b=True)
        logits = logits * tf.cast(self.scale, tf.float32)
        attn = tf.nn.softmax(logits, axis=-1)
        out = tf.matmul(attn, tf.cast(v, tf.float32))
        out = self._merge_heads(out)
        out = self.proj(tf.cast(out, q_tokens.dtype))
        return out


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, name: Optional[str] = None):
        super().__init__(name=name)
        hidden_dim = int(dim * float(mlp_ratio))
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu, name="fc1")
        self.fc2 = tf.keras.layers.Dense(
            dim,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
            bias_initializer="zeros",
            name="fc2_small_init",
        )

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


class GlobalCrossAttentionBlock(tf.keras.layers.Layer):
    """Global cross-attention block over the 7x7 S4 token grid."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = int(dim)
        self.norm_q = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm_q")
        self.norm_kv = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm_kv")
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, name="cross_attn")
        self.norm_mlp = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm_mlp")
        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, name="mlp")

    def call(self, q_map, kv_map, training=False):
        b = tf.shape(q_map)[0]
        h = tf.shape(q_map)[1]
        w = tf.shape(q_map)[2]
        q_tokens = tf.reshape(self.norm_q(q_map), [b, h * w, self.dim])
        kv_tokens = tf.reshape(self.norm_kv(kv_map), [b, h * w, self.dim])
        attn_tokens = self.attn(q_tokens, kv_tokens, training=training)
        attn_map = tf.reshape(attn_tokens, [b, h, w, self.dim])
        return attn_map


class AdaptiveResidualGate(tf.keras.layers.Layer):
    """Per-sample safe residual gate: alpha in [0, alpha_max]."""

    def __init__(self, hidden_dim: int = 256, alpha_max: float = 0.2, bias_init: float = -6.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.alpha_max = float(alpha_max)
        self.fc1 = tf.keras.layers.Dense(int(hidden_dim), activation=tf.nn.gelu, name="fc1")
        self.fc2 = tf.keras.layers.Dense(
            1,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(float(bias_init)),
            name="fc2_alpha_bias_init",
        )

    def call(self, current_stage, previous_projected, training=False):
        batch = tf.shape(current_stage)[0]
        cur_gap = tf.reduce_mean(tf.cast(current_stage, tf.float32), axis=[1, 2])
        prev_gap = tf.reduce_mean(tf.cast(previous_projected, tf.float32), axis=[1, 2])
        gate_input = tf.concat([cur_gap, prev_gap], axis=-1)
        raw = self.fc2(self.fc1(gate_input, training=training), training=training)
        alpha = tf.cast(self.alpha_max, tf.float32) * tf.sigmoid(tf.cast(raw, tf.float32))
        return tf.reshape(alpha, [batch, 1, 1, 1])


class FeatureProjectionHead(tf.keras.layers.Layer):
    """Project 1024-dim feature vector via LN -> Dense -> GELU -> Dropout."""

    def __init__(self, out_dim: int = 256, dropout_rate: float = 0.3, name: Optional[str] = None):
        super().__init__(name=name)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")
        self.dense = tf.keras.layers.Dense(
            int(out_dim),
            activation=tf.nn.gelu,
            kernel_initializer="he_normal",
            name="dense_proj",
        )
        self.dropout = tf.keras.layers.Dropout(float(dropout_rate), name="dropout")

    def call(self, x, training=False):
        x = self.norm(x)
        x = self.dense(x)
        return self.dropout(x, training=training)


class ConvNeXtMS1MLateCrossFeatureFusionFER(tf.keras.Model):
    """ConvNeXt-B MS1M with S3->S4 late cross-attention and direct feature-level classifier fusion."""

    def __init__(self, cfg: Dict):
        model_cfg = cfg.get("model", {})
        super().__init__(name=model_cfg.get("name", "convnext_ms1m_late_cross_fusion"))
        self.cfg = cfg
        self.cross_cfg = cfg.get("cross_stage", {})
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self._shape_logged = False

        baseline_cfg = dict(cfg)
        baseline_cfg["model"] = dict(model_cfg)
        baseline_cfg["model"]["name"] = "convnext_base_ms1m_late_cross_backbone"
        baseline_cfg["model"]["arch"] = "convnext_base_ms1m_arcface"
        baseline_cfg["model"].setdefault("ablation", "cnn_only")
        baseline_cfg["model"].setdefault("classifier_dropout1", 0.35)
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(baseline_cfg)

        alpha_max = float(self.cross_cfg.get("alpha_max", 0.2))
        gate_hidden = int(self.cross_cfg.get("gate_hidden_dim", 256))
        gate_bias = float(self.cross_cfg.get("gate_bias_init", -6.0))
        mlp_ratio = float(self.cross_cfg.get("mlp_ratio", 4.0))
        num_heads_s4 = int(self.cross_cfg.get("num_heads_s4", 8))
        proj_dim = int(self.cross_cfg.get("proj_dim", 256))
        dropout_rate = float(self.cross_cfg.get("dropout_rate", 0.3))
        mlp_hidden_dim = int(self.cross_cfg.get("mlp_hidden_dim", 128))

        # 1. S3 -> S4 Projection & Cross-Attention
        self.project_s3_to_s4 = ProjectionDownsample(1024, name="project_s3_to_s4")
        self.s3_s4_global = GlobalCrossAttentionBlock(
            dim=1024,
            num_heads=num_heads_s4,
            mlp_ratio=mlp_ratio,
            name="s3_s4_global_cross_attn",
        )
        self.alpha4_gate = AdaptiveResidualGate(
            hidden_dim=gate_hidden,
            alpha_max=alpha_max,
            bias_init=gate_bias,
            name="alpha4_adaptive_gate",
        )

        # 2. Dual-Branch Feature Projection Heads
        self.proj_main = FeatureProjectionHead(out_dim=proj_dim, dropout_rate=dropout_rate, name="proj_main")
        self.proj_cross = FeatureProjectionHead(out_dim=proj_dim, dropout_rate=dropout_rate, name="proj_cross")

        # 3. Concatenated Feature MLP Classifier
        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_norm")
        self.mlp_dense = tf.keras.layers.Dense(
            mlp_hidden_dim,
            activation=tf.nn.gelu,
            kernel_initializer="he_normal",
            name="mlp_dense",
        )
        self.mlp_dropout = tf.keras.layers.Dropout(dropout_rate, name="mlp_dropout")
        self.classifier = tf.keras.layers.Dense(
            self.num_classes,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            name="classifier_dense",
        )

    @property
    def backbone(self):
        return self.rgb_baseline.backbone

    def backbone_variables(self):
        return list(self.rgb_baseline.backbone.trainable_variables)

    def head_variables(self):
        backbone_ids = {id(v) for v in self.backbone_variables()}
        return [v for v in self.trainable_variables if id(v) not in backbone_ids]

    def cross_stage_variables(self):
        modules = [
            self.project_s3_to_s4,
            self.s3_s4_global,
            self.alpha4_gate,
            self.proj_main,
            self.proj_cross,
            self.mlp_norm,
            self.mlp_dense,
            self.mlp_dropout,
            self.classifier,
        ]
        variables = []
        for module in modules:
            variables.extend(module.trainable_variables)
        return variables

    @staticmethod
    def _stats(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        t = tf.cast(tensor, tf.float32)
        return tf.reduce_mean(t), tf.reduce_min(t), tf.reduce_max(t), tf.math.reduce_std(t)

    def _log_shapes_once(self, image, s3, p3, s4, delta4, g4, f_main, f_cross, p_main, p_cross, f_fused, logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[ConvNeXtMS1MLateCrossFeatureFusionFER] Shape Trace:", flush=True)
        print(f"  input: {image.shape}", flush=True)
        print(f"  S3: {s3.shape}", flush=True)
        print(f"  projected_S3_to_S4 (P3): {p3.shape}", flush=True)
        print(f"  S4 (Q): {s4.shape}", flush=True)
        print(f"  delta4: {delta4.shape}", flush=True)
        print(f"  G4: {g4.shape}", flush=True)
        print(f"  GAP(G4) f_main: {f_main.shape}", flush=True)
        print(f"  GAP(delta4) f_cross: {f_cross.shape}", flush=True)
        print(f"  proj(f_main): {p_main.shape}", flush=True)
        print(f"  proj(f_cross): {p_cross.shape}", flush=True)
        print(f"  f_fused (concat): {f_fused.shape}", flush=True)
        print(f"  logits: {logits.shape}", flush=True)
        total = count_params(self.variables)
        trainable = count_params(self.trainable_variables)
        backbone = count_params(self.backbone_variables())
        head = count_params(self.head_variables())
        print("[ConvNeXtMS1MLateCrossFeatureFusionFER] Parameter Trace:", flush=True)
        print(f"  total_params: {total:,}", flush=True)
        print(f"  trainable_params: {trainable:,}", flush=True)
        print(f"  backbone_trainable_params: {backbone:,}", flush=True)
        print(f"  head_trainable_params: {head:,}", flush=True)

    def call(self, inputs, training=False, return_endpoints: bool = False):
        image = inputs["image"] if isinstance(inputs, dict) else inputs
        endpoints = self.rgb_baseline.backbone(image, training=training, return_endpoints=True)
        s3 = endpoints["stage3"]  # [B, 14, 14, 512]
        s4 = endpoints["stage4"]  # [B, 7, 7, 1024]

        # 1. Project S3 to P3 [B, 7, 7, 1024]
        p3 = self.project_s3_to_s4(s3, training=training)

        # 2. Compute Cross-Stage Attention delta4 [B, 7, 7, 1024]
        delta4_raw = self.s3_s4_global(s4, p3, training=training)
        delta4 = tf.cast(delta4_raw, tf.float32)

        # 3. Compute Adaptive Residual Gate alpha4 and G4
        alpha4 = self.alpha4_gate(s4, p3, training=training)  # [B, 1, 1, 1]
        g4 = tf.cast(s4, tf.float32) + alpha4 * delta4       # [B, 7, 7, 1024]

        # 4. Feature-Level Fusion
        f_main = tf.reduce_mean(g4, axis=[1, 2])              # [B, 1024]
        f_cross = tf.reduce_mean(delta4, axis=[1, 2])          # [B, 1024]

        # 5. Project each feature branch
        p_main = self.proj_main(f_main, training=training)     # [B, proj_dim]
        p_cross = self.proj_cross(f_cross, training=training)  # [B, proj_dim]

        # 6. Concatenate & MLP Classifier
        f_fused = tf.concat([p_main, p_cross], axis=-1)       # [B, 2 * proj_dim]
        mlp_feat = self.mlp_dropout(self.mlp_dense(self.mlp_norm(f_fused)), training=training)
        logits = tf.cast(self.classifier(mlp_feat), tf.float32)

        # Compute baseline plain logits (pure ConvNeXt-B Stage 4 GAP without cross-stage)
        plain_pooled = self.rgb_baseline.gap(s4)
        plain_dropped = self.rgb_baseline.head_dropout(plain_pooled, training=False)
        plain_logits = tf.cast(self.rgb_baseline.classifier(plain_dropped), tf.float32)

        self._log_shapes_once(image, s3, p3, s4, delta4, g4, f_main, f_cross, p_main, p_cross, f_fused, logits)

        tf.debugging.assert_all_finite(logits, "NaN/Inf in cross-stage logits")
        tf.debugging.assert_all_finite(g4, "NaN/Inf in G4")
        tf.debugging.assert_all_finite(delta4, "NaN/Inf in delta4")

        # Diagnostics: Norms and Contribution Ratios
        delta4_norm = tf.reduce_mean(tf.norm(delta4, axis=-1))
        g4_norm = tf.reduce_mean(tf.norm(g4, axis=-1))
        p_main_norm = tf.reduce_mean(tf.norm(tf.cast(p_main, tf.float32), axis=-1))
        p_cross_norm = tf.reduce_mean(tf.norm(tf.cast(p_cross, tf.float32), axis=-1))
        cross_ratio = p_cross_norm / (p_main_norm + p_cross_norm + 1e-8)

        alpha4_mean, alpha4_min, alpha4_max, alpha4_std = self._stats(alpha4)

        outputs = {
            "logits": logits,
            "plain_logits": plain_logits,
            "cnn_aux_logits": None,
            "semantic_logits": None,
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "S3": s3,
            "projected_S3": p3,
            "S4": s4,
            "G4": g4,
            "delta4": delta4,
            "alpha4": alpha4,
            "alpha4_mean": alpha4_mean,
            "alpha4_min": alpha4_min,
            "alpha4_max": alpha4_max,
            "alpha4_std": alpha4_std,
            "delta4_norm": delta4_norm,
            "g4_norm": g4_norm,
            "p_main_norm": p_main_norm,
            "p_cross_norm": p_cross_norm,
            "cross_ratio": cross_ratio,
        }
        if return_endpoints:
            outputs["backbone_endpoints"] = endpoints
        return outputs
