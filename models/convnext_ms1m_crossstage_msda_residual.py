"""ConvNeXt-B MS1M cross-stage MSDA residual FER experiment.

This model keeps the existing ConvNeXt-Base MS1M/ArcFace backbone contract and
adds one bounded S3 -> S4 correction path. It does not add a late direct feature
branch, auxiliary classifier, Transformer, non-local attention, SMIRK, masks, or
FER checkpoint restoration.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


class ConvNormGELU(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 1, strides: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            int(filters),
            int(kernel_size),
            strides=int(strides),
            padding="same",
            kernel_initializer="he_normal",
            name="conv",
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="gelu")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DepthwiseSeparableBranch(tf.keras.layers.Layer):
    def __init__(self, branch_dim: int = 256, dilation_rate: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.reduce = tf.keras.layers.Conv2D(
            int(branch_dim),
            1,
            padding="same",
            kernel_initializer="he_normal",
            name="reduce_1x1",
        )
        self.reduce_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="reduce_ln")
        self.dw = tf.keras.layers.DepthwiseConv2D(
            3,
            dilation_rate=int(dilation_rate),
            padding="same",
            depthwise_initializer="he_normal",
            name=f"dwconv_3x3_d{int(dilation_rate)}",
        )
        self.pw = tf.keras.layers.Conv2D(
            int(branch_dim),
            1,
            padding="same",
            kernel_initializer="he_normal",
            name="pointwise_1x1",
        )
        self.out_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="out_ln")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="gelu")

    def call(self, x, training=False):
        x = self.reduce(x)
        x = self.reduce_norm(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.out_norm(x)
        return self.act(x)


class Stage2ToStage3Bridge(tf.keras.layers.Layer):
    """Downsamples Stage 2 micro-textures (28x28x256) and fuses with Stage 3 (14x14x512)."""
    def __init__(self, out_dim: int = 512, name: Optional[str] = None):
        super().__init__(name=name)
        self.s2_conv = tf.keras.layers.Conv2D(
            256,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            name="s2_downsample_conv"
        )
        self.s2_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="s2_ln")
        self.fuse_conv = tf.keras.layers.Conv2D(
            out_dim,
            kernel_size=1,
            padding="same",
            kernel_initializer="he_normal",
            name="s2_s3_fusion_proj"
        )
        self.fuse_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="fuse_ln")

    def call(self, s2, s3, training=False):
        s2_proj = tf.nn.gelu(self.s2_norm(self.s2_conv(s2, training=training)))
        f_concat = tf.concat([s3, s2_proj], axis=-1)
        f_fused = tf.nn.gelu(self.fuse_norm(self.fuse_conv(f_concat, training=training)))
        return f_fused


class S3MultiscaleRefinement(tf.keras.layers.Layer):
    """Four lightweight parallel S3 branches (1x1, dw_d1, dw_d2, dw_d4) producing F_ms [B,14,14,512]."""

    def __init__(self, branch_dim: int = 256, out_dim: int = 512, name: Optional[str] = None):
        super().__init__(name=name)
        self.branch_dim = int(branch_dim)
        self.out_dim = int(out_dim)
        self.branch_a = ConvNormGELU(self.branch_dim, kernel_size=1, name="branch_a_1x1")
        self.branch_b = DepthwiseSeparableBranch(
            branch_dim=self.branch_dim,
            dilation_rate=1,
            name="branch_b_dwsep_d1",
        )
        self.branch_c = DepthwiseSeparableBranch(
            branch_dim=self.branch_dim,
            dilation_rate=2,
            name="branch_c_dwsep_d2",
        )
        self.branch_d = DepthwiseSeparableBranch(
            branch_dim=self.branch_dim,
            dilation_rate=4,
            name="branch_d_dwsep_d4",
        )
        self.project = tf.keras.layers.Conv2D(
            self.out_dim,
            1,
            padding="same",
            kernel_initializer="he_normal",
            name="concat_pointwise_1024_to_512",
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="f_ms_ln")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="f_ms_gelu")

    def call(self, s3, training=False):
        a = self.branch_a(s3, training=training)
        b = self.branch_b(s3, training=training)
        c = self.branch_c(s3, training=training)
        d = self.branch_d(s3, training=training)
        merged = tf.concat([a, b, c, d], axis=-1)
        f_ms = self.project(merged)
        f_ms = self.norm(f_ms)
        f_ms = self.act(f_ms)
        return f_ms, a, b, c, d, merged


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels: int = 512, reduction: int = 16, name: Optional[str] = None):
        super().__init__(name=name)
        hidden = max(1, int(channels) // int(reduction))
        self.fc1 = tf.keras.layers.Dense(hidden, activation=tf.nn.gelu, name="shared_fc1")
        self.fc2 = tf.keras.layers.Dense(int(channels), name="shared_fc2")

    def call(self, f_ms, training=False):
        x = tf.cast(f_ms, tf.float32)
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x, axis=[1, 2])
        avg_logits = self.fc2(self.fc1(avg_pool, training=training), training=training)
        max_logits = self.fc2(self.fc1(max_pool, training=training), training=training)
        w_c = tf.sigmoid(avg_logits + max_logits)
        w_c = tf.reshape(w_c, [tf.shape(f_ms)[0], 1, 1, tf.shape(f_ms)[-1]])
        w_c = tf.cast(w_c, f_ms.dtype)
        return f_ms * w_c, w_c


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int = 7, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            1,
            int(kernel_size),
            padding="same",
            kernel_initializer="he_normal",
            name="spatial_7x7",
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="spatial_ln")

    def call(self, f_ms, training=False):
        x = tf.cast(f_ms, tf.float32)
        avg_map = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_map = tf.reduce_max(x, axis=-1, keepdims=True)
        pooled = tf.concat([avg_map, max_map], axis=-1)
        spatial_logits = self.norm(self.conv(pooled, training=training))
        w_s = tf.sigmoid(spatial_logits)
        w_s = tf.cast(w_s, f_ms.dtype)
        return f_ms * w_s, w_s


class SoftmaxDualAttentionFusion(tf.keras.layers.Layer):
    """Feature-wise safe fusion with non-negative weights that sum to one."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.fusion_logits = self.add_weight(
            name="fusion_logits",
            shape=(2,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, f_ca, f_sa, training=False):
        weights = tf.nn.softmax(tf.cast(self.fusion_logits, tf.float32))
        f_da = tf.cast(weights[0], f_ca.dtype) * f_ca + tf.cast(weights[1], f_sa.dtype) * f_sa
        return f_da, weights


class S3ToS4Projection(tf.keras.layers.Layer):
    def __init__(self, out_dim: int = 1024, name: Optional[str] = None):
        super().__init__(name=name)
        self.dw = tf.keras.layers.DepthwiseConv2D(
            3,
            strides=2,
            padding="same",
            depthwise_initializer="he_normal",
            name="dwconv_3x3_stride2",
        )
        self.pw = tf.keras.layers.Conv2D(
            int(out_dim),
            1,
            padding="same",
            kernel_initializer="he_normal",
            name="pointwise_512_to_1024",
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="delta_norm")

    def call(self, f_da, training=False):
        x = self.dw(f_da)
        delta_raw = self.pw(x)
        delta_norm = self.norm(delta_raw)
        return delta_raw, delta_norm


class BoundedResidualAlpha(tf.keras.layers.Layer):
    """Scalar alpha constrained to [0, alpha_max], initialized near 0.005."""

    def __init__(self, alpha_max: float = 0.20, initial_alpha: float = 0.005, name: Optional[str] = None):
        super().__init__(name=name)
        self.alpha_max = float(alpha_max)
        init_ratio = min(max(float(initial_alpha) / max(self.alpha_max, 1e-8), 1e-6), 1.0 - 1e-6)
        init_raw = math.log(init_ratio / (1.0 - init_ratio))
        self.alpha_raw = self.add_weight(
            name="alpha_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(init_raw),
            trainable=True,
        )

    def call(self, reference, force_zero: bool = False):
        if force_zero:
            return tf.zeros([tf.shape(reference)[0], 1, 1, 1], dtype=tf.float32)
        alpha = tf.cast(self.alpha_max, tf.float32) * tf.sigmoid(tf.cast(self.alpha_raw, tf.float32))
        return tf.ones([tf.shape(reference)[0], 1, 1, 1], dtype=tf.float32) * alpha


class ArcFaceCosineClassifier(tf.keras.layers.Layer):
    """Normalized Cosine Margin Classifier Head for Face & Expression Prototypes."""
    def __init__(self, num_classes: int = 7, scale: float = 30.0, margin: float = 0.20, name: Optional[str] = None):
        super().__init__(name=name)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)

    def build(self, input_shape):
        feat_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="weight",
            shape=(feat_dim, self.num_classes),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )

    def call(self, features, labels=None, training=False):
        features_f32 = tf.cast(features, tf.float32)
        features_norm = tf.math.l2_normalize(features_f32, axis=-1)
        w_norm = tf.math.l2_normalize(tf.cast(self.W, tf.float32), axis=0)
        
        cos_theta = tf.matmul(features_norm, w_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        if training and labels is not None:
            labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
            target_cosine = cos_theta - self.margin
            cos_theta = tf.where(labels_one_hot > 0.5, target_cosine, cos_theta)
            
        return cos_theta * self.scale


class ConvNeXtMS1MCrossStageMSDAResidualFER(tf.keras.Model):
    """ConvNeXt-B MS1M backbone plus bounded MSDA S3 correction into S4."""

    def __init__(self, cfg: Dict):
        model_cfg = cfg.get("model", {})
        msda_cfg = cfg.get("cross_stage_msda", {})
        super().__init__(name=model_cfg.get("name", "convnext_ms1m_crossstage_msda_residual"))
        self.cfg = cfg
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self._shape_logged = False

        baseline_cfg = dict(cfg)
        baseline_cfg["model"] = dict(model_cfg)
        baseline_cfg["model"]["name"] = "convnext_base_ms1m_msda_residual_backbone"
        baseline_cfg["model"]["arch"] = "convnext_base_ms1m_arcface"
        baseline_cfg["model"].setdefault("ablation", "cnn_only")
        baseline_cfg["model"].setdefault("classifier_dropout1", 0.35)
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(baseline_cfg)

        self.s2_bridge = Stage2ToStage3Bridge(out_dim=512, name="s2_microtexture_bridge")
        self.ms_refine = S3MultiscaleRefinement(
            branch_dim=int(msda_cfg.get("branch_dim", 256)),
            out_dim=512,
            name="s3_multiscale_refinement",
        )
        self.channel_attention = ChannelAttention(
            channels=512,
            reduction=int(msda_cfg.get("channel_reduction", 16)),
            name="dual_channel_attention",
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=int(msda_cfg.get("spatial_kernel_size", 7)),
            name="dual_spatial_attention",
        )
        self.dual_fusion = SoftmaxDualAttentionFusion(name="dual_attention_softmax_fusion")
        self.s3_to_s4 = S3ToS4Projection(out_dim=1024, name="s3_to_s4_projection")
        self.alpha_gate = BoundedResidualAlpha(
            alpha_max=float(msda_cfg.get("alpha_max", 0.20)),
            initial_alpha=float(msda_cfg.get("initial_alpha", 0.005)),
            name="bounded_cross_stage_alpha",
        )
        self.use_aux_loss = bool(model_cfg.get("use_aux_loss", False))
        self.aux_classifier = tf.keras.layers.Dense(
            self.num_classes,
            kernel_initializer="he_normal",
            name="aux_classifier",
        )
        self.use_arcface = bool(model_cfg.get("use_arcface", True))
        if self.use_arcface:
            self.arcface_head = ArcFaceCosineClassifier(
                num_classes=self.num_classes,
                scale=float(model_cfg.get("arcface_scale", 30.0)),
                margin=float(model_cfg.get("arcface_margin", 0.20)),
                name="arcface_cosine_head",
            )

    @property
    def backbone(self):
        return self.rgb_baseline.backbone

    def backbone_variables(self) -> List[tf.Variable]:
        vars_list = list(self.rgb_baseline.backbone.trainable_variables)
        if bool(self.cfg.get("model", {}).get("freeze_stage12", False)):
            filtered = [
                v for v in vars_list
                if not any(k in v.name.lower() for k in ["stem", "stage1", "stage2", "downsample_stage2"])
            ]
            return filtered
        return vars_list

    def classifier_variables(self) -> List[tf.Variable]:
        return list(self.rgb_baseline.classifier.trainable_variables) + list(self.rgb_baseline.head_dropout.trainable_variables)

    def multiscale_variables(self) -> List[tf.Variable]:
        return list(self.ms_refine.trainable_variables)

    def channel_attention_variables(self) -> List[tf.Variable]:
        return list(self.channel_attention.trainable_variables)

    def spatial_attention_variables(self) -> List[tf.Variable]:
        return list(self.spatial_attention.trainable_variables)

    def fusion_variables(self) -> List[tf.Variable]:
        return list(self.dual_fusion.trainable_variables)

    def projection_variables(self) -> List[tf.Variable]:
        return list(self.s3_to_s4.trainable_variables)

    def alpha_variables(self) -> List[tf.Variable]:
        return list(self.alpha_gate.trainable_variables)

    def bridge_variables(self) -> List[tf.Variable]:
        return list(self.s2_bridge.trainable_variables)

    def arcface_variables(self) -> List[tf.Variable]:
        if hasattr(self, "arcface_head") and self.use_arcface:
            return list(self.arcface_head.trainable_variables)
        return []

    def new_module_variables(self) -> List[tf.Variable]:
        variables: List[tf.Variable] = []
        for module_vars in (
            self.bridge_variables(),
            self.multiscale_variables(),
            self.channel_attention_variables(),
            self.spatial_attention_variables(),
            self.fusion_variables(),
            self.projection_variables(),
            self.alpha_variables(),
            self.arcface_variables(),
        ):
            variables.extend(module_vars)
        return variables

    def head_variables(self) -> List[tf.Variable]:
        backbone_ids = {id(v) for v in self.backbone_variables()}
        return [v for v in self.trainable_variables if id(v) not in backbone_ids]

    @staticmethod
    def _norm(x: tf.Tensor) -> tf.Tensor:
        return tf.linalg.global_norm([tf.cast(x, tf.float32)])

    @staticmethod
    def _mean_std(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(x, tf.float32)
        return tf.reduce_mean(x), tf.math.reduce_std(x)

    def _log_shapes_once(self, image, outputs) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[ConvNeXtMS1MCrossStageMSDAResidualFER] Shape trace:", flush=True)
        for key in ("input", "S1", "S2", "S3", "F_ms", "F_ca", "F_sa", "F_da", "delta_raw", "delta_norm", "S4", "G4", "logits"):
            tensor = image if key == "input" else outputs[key]
            print(f"  {key}: {tensor.shape}", flush=True)

        baseline_params = count_params(self.rgb_baseline.variables)
        new_module_params = count_params(self.new_module_variables())
        total_params = count_params(self.variables)
        trainable_params = count_params(self.trainable_variables)
        print("[ConvNeXtMS1MCrossStageMSDAResidualFER] Parameter trace:", flush=True)
        print(f"  baseline_params: {baseline_params:,}", flush=True)
        print(f"  new_module_params: {new_module_params:,}", flush=True)
        print(f"  total_params: {total_params:,}", flush=True)
        print(f"  trainable_params: {trainable_params:,}", flush=True)

    def call(self, inputs, training=False, return_endpoints: bool = False, force_alpha_zero: bool = False, labels=None):
        if isinstance(inputs, dict):
            image = inputs.get("image", inputs)
            if labels is None:
                labels = inputs.get("label", None)
        else:
            image = inputs

        endpoints = self.rgb_baseline.backbone(image, training=training, return_endpoints=True)
        s1 = endpoints["stage1"]
        s2 = endpoints["stage2"]
        s3 = endpoints["stage3"]
        s4 = endpoints["stage4"]

        s3_fused = self.s2_bridge(s2, s3, training=training)
        f_ms, branch_a, branch_b, branch_c, branch_d, f_concat = self.ms_refine(s3_fused, training=training)
        f_ca, w_c = self.channel_attention(f_ms, training=training)
        f_sa, w_s = self.spatial_attention(f_ms, training=training)
        f_da, fusion_weights = self.dual_fusion(f_ca, f_sa, training=training)
        delta_raw, delta_norm = self.s3_to_s4(f_da, training=training)

        alpha = self.alpha_gate(s4, force_zero=force_alpha_zero)
        residual = tf.cast(alpha, tf.float32) * tf.cast(delta_norm, tf.float32)
        g4 = tf.cast(s4, tf.float32) + residual

        pooled = self.rgb_baseline.gap(g4)
        dropped = self.rgb_baseline.head_dropout(pooled, training=training)
        if hasattr(self, "arcface_head") and self.use_arcface:
            logits = tf.cast(self.arcface_head(dropped, labels=labels, training=training), tf.float32)
        else:
            logits = tf.cast(self.rgb_baseline.classifier(dropped), tf.float32)

        baseline_pooled = self.rgb_baseline.gap(tf.cast(s4, tf.float32))
        baseline_dropped = self.rgb_baseline.head_dropout(baseline_pooled, training=False)
        if hasattr(self, "arcface_head") and self.use_arcface:
            baseline_logits = tf.cast(self.arcface_head(baseline_dropped, labels=labels, training=False), tf.float32)
        else:
            baseline_logits = tf.cast(self.rgb_baseline.classifier(baseline_dropped), tf.float32)

        pooled_feat = tf.reduce_mean(f_da, axis=[1, 2])  # [B, 512] for SupCon
        aux_logits = None
        if self.use_aux_loss:
            aux_logits = tf.cast(self.aux_classifier(baseline_pooled), tf.float32)
            aux_logits = tf.where(tf.math.is_finite(aux_logits), aux_logits, tf.zeros_like(aux_logits))

        s4_norm = self._norm(s4)
        delta_raw_norm = self._norm(delta_raw)
        delta_norm_norm = self._norm(delta_norm)
        residual_norm = self._norm(residual)
        residual_ratio = residual_norm / (s4_norm + 1e-8)
        channel_mean, channel_std = self._mean_std(w_c)
        spatial_mean, spatial_std = self._mean_std(w_s)
        alpha_mean, alpha_std = self._mean_std(alpha)

        outputs = {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "cnn_aux_logits": aux_logits,
            "aux_logits": aux_logits,
            "pooled_feature": pooled_feat,
            "semantic_logits": None,
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "S4": s4,
            "branch_a": branch_a,
            "branch_b": branch_b,
            "branch_c": branch_c,
            "branch_d": branch_d,
            "F_concat": f_concat,
            "F_ms": f_ms,
            "F_ca": f_ca,
            "F_sa": f_sa,
            "F_da": f_da,
            "w_c": w_c,
            "w_s": w_s,
            "delta_raw": delta_raw,
            "delta_norm": delta_norm,
            "alpha": alpha,
            "G4": g4,
            "residual": residual,
            "s4_norm": tf.cast(s4_norm, tf.float32),
            "delta_raw_norm": tf.cast(delta_raw_norm, tf.float32),
            "delta_norm_norm": tf.cast(delta_norm_norm, tf.float32),
            "residual_norm": tf.cast(residual_norm, tf.float32),
            "residual_ratio": tf.cast(residual_ratio, tf.float32),
            "alpha_mean": tf.cast(alpha_mean, tf.float32),
            "alpha_std": tf.cast(alpha_std, tf.float32),
            "channel_attention_mean": tf.cast(channel_mean, tf.float32),
            "channel_attention_std": tf.cast(channel_std, tf.float32),
            "spatial_attention_mean": tf.cast(spatial_mean, tf.float32),
            "spatial_attention_std": tf.cast(spatial_std, tf.float32),
            "fusion_weight_channel": tf.cast(fusion_weights[0], tf.float32),
            "fusion_weight_spatial": tf.cast(fusion_weights[1], tf.float32),
        }
        logits_f32 = tf.cast(logits, tf.float32)
        logits_f32 = tf.where(tf.math.is_finite(logits_f32), logits_f32, tf.zeros_like(logits_f32))
        outputs["logits"] = logits_f32

        if return_endpoints:
            outputs["backbone_endpoints"] = endpoints

        self._log_shapes_once(image, outputs)
        return outputs
