"""ConvNeXt-B MS1M FER with cross-stage shifted-window residual fusion.

This experiment trains FER from the MS1M/ArcFace ConvNeXt-B pretrained
backbone. It does not restore any FER checkpoint and does not use masks, SMIRK,
3D features, MGR, CLIP, or auxiliary heads.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf

from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


def _window_partition(x: tf.Tensor, window_size: int) -> tf.Tensor:
    b = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    c = tf.shape(x)[3]
    ws = int(window_size)
    x = tf.reshape(x, [b, h // ws, ws, w // ws, ws, c])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, ws * ws, c])


def _window_reverse(windows: tf.Tensor, height: tf.Tensor, width: tf.Tensor, window_size: int) -> tf.Tensor:
    ws = int(window_size)
    num_h = height // ws
    num_w = width // ws
    b = tf.shape(windows)[0] // (num_h * num_w)
    c = tf.shape(windows)[2]
    x = tf.reshape(windows, [b, num_h, num_w, ws, ws, c])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [b, height, width, c])


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
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="proj_zero_init",
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
        logits = tf.clip_by_value(logits, -50.0, 50.0)
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
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="fc2_zero_init",
        )

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


class WindowCrossSwinBlock(tf.keras.layers.Layer):
    """Window or shifted-window cross-attention block with Swin-style residuals."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm_q = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm_q")
        self.norm_kv = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm_kv")
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, name="cross_attn")
        self.norm_mlp = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm_mlp")
        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, name="mlp")

    def call(self, q_map, kv_map, training=False):
        h = tf.shape(q_map)[1]
        w = tf.shape(q_map)[2]
        q_norm = self.norm_q(q_map)
        kv_norm = self.norm_kv(kv_map)
        if self.shift_size > 0:
            shift = int(self.shift_size)
            q_norm = tf.roll(q_norm, shift=[-shift, -shift], axis=[1, 2])
            kv_norm = tf.roll(kv_norm, shift=[-shift, -shift], axis=[1, 2])

        q_windows = _window_partition(q_norm, self.window_size)
        kv_windows = _window_partition(kv_norm, self.window_size)
        attn_windows = self.attn(q_windows, kv_windows, training=training)
        attn_map = _window_reverse(attn_windows, h, w, self.window_size)

        if self.shift_size > 0:
            shift = int(self.shift_size)
            attn_map = tf.roll(attn_map, shift=[shift, shift], axis=[1, 2])

        x = q_map + tf.cast(attn_map, q_map.dtype)
        mlp_out = self.mlp(self.norm_mlp(x), training=training)
        return x + tf.cast(mlp_out, x.dtype)


class GlobalCrossSwinBlock(tf.keras.layers.Layer):
    """Global cross-attention over the 7x7 S4 token grid."""

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
        x = q_map + tf.cast(attn_map, q_map.dtype)
        mlp_out = self.mlp(self.norm_mlp(x), training=training)
        return x + tf.cast(mlp_out, x.dtype)


class AdaptiveResidualGate(tf.keras.layers.Layer):
    """Per-sample safe residual gate: alpha in [0, alpha_max]."""

    def __init__(self, hidden_dim: int = 256, alpha_max: float = 0.2, bias_init: float = -6.0, adaptive: bool = True, fixed_alpha: float = 0.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.alpha_max = float(alpha_max)
        self.adaptive = bool(adaptive)
        self.fixed_alpha = float(fixed_alpha)
        if self.adaptive:
            self.fc1 = tf.keras.layers.Dense(int(hidden_dim), activation=tf.nn.gelu, name="fc1")
            self.fc2 = tf.keras.layers.Dense(
                1,
                kernel_initializer="zeros",
                bias_initializer=tf.keras.initializers.Constant(float(bias_init)),
                name="fc2_alpha_bias_init",
            )
        else:
            self.fc1 = None
            self.fc2 = None

    def call(self, current_stage, previous_projected, training=False):
        batch = tf.shape(current_stage)[0]
        if not self.adaptive:
            value = min(max(self.fixed_alpha, 0.0), self.alpha_max)
            return tf.ones([batch, 1, 1, 1], dtype=tf.float32) * tf.cast(value, tf.float32)
        cur_gap = tf.reduce_mean(tf.cast(current_stage, tf.float32), axis=[1, 2])
        prev_gap = tf.reduce_mean(tf.cast(previous_projected, tf.float32), axis=[1, 2])
        gate_input = tf.concat([cur_gap, prev_gap], axis=-1)
        raw = self.fc2(self.fc1(gate_input, training=training), training=training)
        alpha = tf.cast(self.alpha_max, tf.float32) * tf.sigmoid(tf.cast(raw, tf.float32))
        return tf.reshape(alpha, [batch, 1, 1, 1])


class ConvNeXtMS1MCrossStageSwinFER(tf.keras.Model):
    """ConvNeXt-B MS1M baseline plus configurable S2->S3->S4 residual fusion."""

    def __init__(self, cfg: Dict):
        model_cfg = cfg.get("model", {})
        super().__init__(name=model_cfg.get("name", "convnext_ms1m_cross_stage_swin"))
        self.cfg = cfg
        self.cross_cfg = cfg.get("cross_stage", {})
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self._shape_logged = False

        baseline_cfg = dict(cfg)
        baseline_cfg["model"] = dict(model_cfg)
        baseline_cfg["model"]["name"] = "convnext_base_ms1m_cross_stage_backbone"
        baseline_cfg["model"]["arch"] = "convnext_base_ms1m_arcface"
        baseline_cfg["model"].setdefault("ablation", "cnn_only")
        baseline_cfg["model"].setdefault("classifier_dropout1", 0.35)
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(baseline_cfg)

        self.use_s2_s3 = bool(self.cross_cfg.get("cross_stage_s2_s3", True))
        self.use_s3_s4 = bool(self.cross_cfg.get("cross_stage_s3_s4", True))
        self.use_shifted = bool(self.cross_cfg.get("shifted_window", True))
        adaptive_gate = bool(self.cross_cfg.get("adaptive_gate", True))
        alpha_max = float(self.cross_cfg.get("alpha_max", 0.2))
        gate_hidden = int(self.cross_cfg.get("gate_hidden_dim", 256))
        gate_bias = float(self.cross_cfg.get("gate_bias_init", -6.0))
        fixed_alpha = float(self.cross_cfg.get("fixed_alpha", 0.0))
        mlp_ratio = float(self.cross_cfg.get("mlp_ratio", 4.0))
        window_size = int(self.cross_cfg.get("window_size", 7))
        shift_size = int(self.cross_cfg.get("shift_size", 3))
        num_heads_s3 = int(self.cross_cfg.get("num_heads_s3", 8))
        num_heads_s4 = int(self.cross_cfg.get("num_heads_s4", 8))

        self.project_s2_to_s3 = ProjectionDownsample(512, name="project_s2_to_s3")
        self.project_g3_to_s4 = ProjectionDownsample(1024, name="project_g3_to_s4")
        self.s2_s3_window = WindowCrossSwinBlock(
            dim=512,
            num_heads=num_heads_s3,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            name="s2_s3_window_cross_attn",
        )
        self.s2_s3_shifted = WindowCrossSwinBlock(
            dim=512,
            num_heads=num_heads_s3,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            name="s2_s3_shifted_window_cross_attn",
        )
        self.s3_s4_global = GlobalCrossSwinBlock(
            dim=1024,
            num_heads=num_heads_s4,
            mlp_ratio=mlp_ratio,
            name="s3_s4_global_cross_attn",
        )
        self.alpha3_gate = AdaptiveResidualGate(
            hidden_dim=gate_hidden,
            alpha_max=alpha_max,
            bias_init=gate_bias,
            adaptive=adaptive_gate,
            fixed_alpha=fixed_alpha,
            name="alpha3_adaptive_gate",
        )
        self.alpha4_gate = AdaptiveResidualGate(
            hidden_dim=gate_hidden,
            alpha_max=alpha_max,
            bias_init=gate_bias,
            adaptive=adaptive_gate,
            fixed_alpha=fixed_alpha,
            name="alpha4_adaptive_gate",
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
            self.project_s2_to_s3,
            self.project_g3_to_s4,
            self.s2_s3_window,
            self.s2_s3_shifted,
            self.s3_s4_global,
            self.alpha3_gate,
            self.alpha4_gate,
        ]
        variables = []
        for module in modules:
            variables.extend(module.trainable_variables)
        return variables

    @staticmethod
    def _stats(alpha: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        alpha = tf.cast(alpha, tf.float32)
        return tf.reduce_mean(alpha), tf.reduce_min(alpha), tf.reduce_max(alpha), tf.math.reduce_std(alpha)

    def _zero_alpha(self, ref: tf.Tensor) -> tf.Tensor:
        return tf.zeros([tf.shape(ref)[0], 1, 1, 1], dtype=tf.float32)

    def _log_shapes_once(self, image, s2, p2, s3, g3, p3, s4, g4, logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[ConvNeXtMS1MCrossStageSwinFER] Shape Trace:", flush=True)
        print(f"  input: {image.shape}", flush=True)
        print(f"  S2: {s2.shape}", flush=True)
        print(f"  projected_S2_to_S3: {p2.shape}", flush=True)
        print(f"  S3: {s3.shape}", flush=True)
        print(f"  G3: {g3.shape}", flush=True)
        print(f"  projected_G3_to_S4: {p3.shape}", flush=True)
        print(f"  S4: {s4.shape}", flush=True)
        print(f"  G4: {g4.shape}", flush=True)
        print(f"  logits: {logits.shape}", flush=True)
        total = count_params(self.variables)
        trainable = count_params(self.trainable_variables)
        backbone = count_params(self.backbone_variables())
        head = count_params(self.head_variables())
        cross = count_params(self.cross_stage_variables())
        print("[ConvNeXtMS1MCrossStageSwinFER] Parameter Trace:", flush=True)
        print(f"  total_params: {total:,}", flush=True)
        print(f"  trainable_params: {trainable:,}", flush=True)
        print(f"  backbone_trainable_params: {backbone:,}", flush=True)
        print(f"  head_trainable_params: {head:,}", flush=True)
        print(f"  cross_stage_trainable_params: {cross:,}", flush=True)

    def call(self, inputs, training=False, return_endpoints: bool = False):
        image = inputs["image"] if isinstance(inputs, dict) else inputs
        endpoints = self.rgb_baseline.backbone(image, training=training, return_endpoints=True)
        s2 = endpoints["stage2"]
        s3 = endpoints["stage3"]
        s4 = endpoints["stage4"]

        p2 = self.project_s2_to_s3(s2, training=training)
        if self.use_s2_s3:
            x3 = self.s2_s3_window(s3, p2, training=training)
            if self.use_shifted:
                x3 = self.s2_s3_shifted(x3, p2, training=training)
            delta3 = tf.cast(x3 - s3, tf.float32)
            alpha3 = self.alpha3_gate(s3, p2, training=training)
            g3 = tf.cast(s3, tf.float32) + alpha3 * delta3
        else:
            delta3 = tf.zeros_like(tf.cast(s3, tf.float32))
            alpha3 = self._zero_alpha(s3)
            g3 = tf.cast(s3, tf.float32)

        p3 = self.project_g3_to_s4(g3, training=training)
        if self.use_s3_s4:
            x4 = self.s3_s4_global(s4, p3, training=training)
            delta4 = tf.cast(x4 - s4, tf.float32)
            alpha4 = self.alpha4_gate(s4, p3, training=training)
            g4 = tf.cast(s4, tf.float32) + alpha4 * delta4
        else:
            delta4 = tf.zeros_like(tf.cast(s4, tf.float32))
            alpha4 = self._zero_alpha(s4)
            g4 = tf.cast(s4, tf.float32)

        pooled = self.rgb_baseline.gap(g4)
        dropped = self.rgb_baseline.head_dropout(pooled, training=training)
        logits = tf.cast(self.rgb_baseline.classifier(dropped), tf.float32)

        plain_pooled = self.rgb_baseline.gap(s4)
        plain_dropped = self.rgb_baseline.head_dropout(plain_pooled, training=False)
        plain_logits = tf.cast(self.rgb_baseline.classifier(plain_dropped), tf.float32)

        semantic_logits = None
        granularity_weights = None
        if self.rgb_baseline.use_semantic_branch and self.rgb_baseline.visual_projector is not None:
            text_protos = tf.cast(self.rgb_baseline.text_prototypes, dtype=tf.float32)
            t_norm = tf.math.l2_normalize(text_protos, axis=-1)

            if self.rgb_baseline.use_au_region_routed and self.rgb_baseline.visual_projector_upper is not None:
                stage3_feat = g3
                z_upper = tf.reduce_mean(stage3_feat[:, 0:8, :, :], axis=[1, 2])
                z_lower = tf.reduce_mean(stage3_feat[:, 5:14, :, :], axis=[1, 2])
                z_au = tf.reduce_mean(stage3_feat[:, 3:11, :, :], axis=[1, 2])

                v_global_proj = self.rgb_baseline.visual_projector(pooled, training=training)
                v_upper_proj = self.rgb_baseline.visual_projector_upper(z_upper, training=training)
                v_lower_proj = self.rgb_baseline.visual_projector_lower(z_lower, training=training)
                v_au_proj = self.rgb_baseline.visual_projector_au(z_au, training=training)

                v_global_norm = tf.math.l2_normalize(v_global_proj, axis=-1, epsilon=1e-5)
                v_upper_norm = tf.math.l2_normalize(v_upper_proj, axis=-1, epsilon=1e-5)
                v_lower_norm = tf.math.l2_normalize(v_lower_proj, axis=-1, epsilon=1e-5)
                v_au_norm = tf.math.l2_normalize(v_au_proj, axis=-1, epsilon=1e-5)

                t_norm = tf.cast(t_norm, dtype=v_global_norm.dtype)

                s0 = tf.einsum("bd,cd->bc", v_global_norm, t_norm[:, 0, :])
                s1 = tf.einsum("bd,cd->bc", v_au_norm, t_norm[:, 1, :])
                s2 = tf.einsum("bd,cd->bc", v_upper_norm, t_norm[:, 2, :])
                s3 = tf.einsum("bd,cd->bc", v_lower_norm, t_norm[:, 3, :])
                s4 = tf.einsum("bd,cd->bc", v_global_norm, t_norm[:, 4, :])

                raw_sim = tf.stack([s0, s1, s2, s3, s4], axis=-1)
            else:
                v_proj = self.rgb_baseline.visual_projector(pooled, training=training)
                v_norm = tf.math.l2_normalize(v_proj, axis=-1, epsilon=1e-5)
                t_norm = tf.cast(t_norm, dtype=v_norm.dtype)
                if len(t_norm.shape) == 3 or (hasattr(t_norm.shape, "rank") and t_norm.shape.rank == 3):
                    raw_sim = tf.einsum("bd,ckd->bck", v_norm, t_norm)
                else:
                    raw_sim = tf.einsum("bd,cd->bc", v_norm, t_norm)

            if self.rgb_baseline.multi_prototype or self.rgb_baseline.use_adaptive_granularity:
                raw_sim_f32 = tf.cast(raw_sim, tf.float32)

                if self.rgb_baseline.use_adaptive_granularity and self.rgb_baseline.granularity_gate is not None:
                    granularity_weights = self.rgb_baseline.granularity_gate(pooled, training=training)
                    granularity_weights_f32 = tf.cast(granularity_weights, tf.float32)
                    gw_exp = tf.expand_dims(granularity_weights_f32, axis=1)
                    agg_sim = tf.reduce_sum(gw_exp * raw_sim_f32, axis=-1)
                else:
                    if self.rgb_baseline.prototype_aggregation == "logsumexp":
                        tau = tf.constant(self.rgb_baseline.prototype_temperature, dtype=tf.float32)
                        K = tf.constant(float(raw_sim.shape[-1]), dtype=tf.float32)
                        lse = tf.reduce_logsumexp(raw_sim_f32 / tau, axis=-1)
                        agg_sim = tau * (lse - tf.math.log(K))
                    elif self.rgb_baseline.prototype_aggregation == "mean":
                        agg_sim = tf.reduce_mean(raw_sim_f32, axis=-1)
                    elif self.rgb_baseline.prototype_aggregation == "max":
                        agg_sim = tf.reduce_max(raw_sim_f32, axis=-1)
                    else:
                        raise ValueError(f"Unsupported prototype_aggregation: {self.rgb_baseline.prototype_aggregation}")
                semantic_logits = agg_sim * tf.cast(self.rgb_baseline.semantic_logit_scale, tf.float32)
            else:
                agg_sim = tf.cast(raw_sim, tf.float32)
                semantic_logits = agg_sim * tf.cast(self.rgb_baseline.semantic_logit_scale, tf.float32)

            agg_sim = tf.where(tf.math.is_finite(agg_sim), agg_sim, tf.zeros_like(agg_sim))
            semantic_logits = tf.where(tf.math.is_finite(semantic_logits), semantic_logits, tf.zeros_like(semantic_logits))

        self._log_shapes_once(image, s2, p2, s3, g3, p3, s4, g4, logits)

        logits = tf.where(tf.math.is_finite(logits), logits, tf.zeros_like(logits))
        g3 = tf.where(tf.math.is_finite(g3), g3, tf.zeros_like(g3))
        g4 = tf.where(tf.math.is_finite(g4), g4, tf.zeros_like(g4))

        alpha3_mean, alpha3_min, alpha3_max, alpha3_std = self._stats(alpha3)
        alpha4_mean, alpha4_min, alpha4_max, alpha4_std = self._stats(alpha4)
        outputs = {
            "logits": logits,
            "plain_logits": plain_logits,
            "cnn_aux_logits": None,
            "semantic_logits": semantic_logits,
            "granularity_weights": granularity_weights,
            "agg_sim": agg_sim if 'agg_sim' in locals() else None,
            "hard_pairs_matrix": getattr(self.rgb_baseline, "hard_pairs_matrix", None),
            "lambda_sem": getattr(self.rgb_baseline, "lambda_sem", 0.1),
            "lambda_hard": getattr(self.rgb_baseline, "lambda_hard", 0.05),
            "hard_margin": getattr(self.rgb_baseline, "hard_margin", 0.15),
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "S2": s2,
            "projected_S2": p2,
            "S3": s3,
            "G3": g3,
            "projected_G3": p3,
            "S4": s4,
            "G4": g4,
            "delta3": delta3,
            "delta4": delta4,
            "alpha3": alpha3,
            "alpha4": alpha4,
            "alpha3_mean": alpha3_mean,
            "alpha3_min": alpha3_min,
            "alpha3_max": alpha3_max,
            "alpha3_std": alpha3_std,
            "alpha4_mean": alpha4_mean,
            "alpha4_min": alpha4_min,
            "alpha4_max": alpha4_max,
            "alpha4_std": alpha4_std,
        }
        if return_endpoints:
            outputs["backbone_endpoints"] = endpoints
        return outputs
