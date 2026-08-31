"""
Dual ConvNeXt MS1M RGB anchor + small SMIRK geometry encoder with
3D-guided channel attention for FER2013.

Contract:
1. RGB branch is the ConvNeXt-Base MS1M + FER ckpt-43 baseline anchor.
   Stage 1 keeps the full RGB backbone, GAP/dropout, and classifier frozen.
2. Geometry branch is a small CNN over cached SMIRK depth+normal maps
   [B,H,W,4], producing F_3d [B,7,7,1024]. It does not own a classifier.
3. Geometry only guides RGB channel attention over F_rgb [B,7,7,1024].
   gate = 2 * sigmoid(g) - 1, so gate is bounded to [-1, 1].
4. F_guided = F_rgb * (1 + alpha * gate), where
   alpha = 0.2 * sigmoid(alpha_raw). alpha_raw is initialized negative so
   effective alpha starts close to zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf

from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


class GeometryConvBlock(tf.keras.layers.Layer):
    """Conv -> LayerNorm -> GELU block for NHWC geometry maps."""

    def __init__(self, filters: int, stride: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            int(filters),
            kernel_size=3,
            strides=int(stride),
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


class SmallSMIRKGeometryEncoder(tf.keras.Model):
    """~8M parameter CNN encoder for depth + normal maps."""

    def __init__(self, out_dim: int = 1024, width: int = 64, name: str = "small_smirk_geometry_encoder"):
        super().__init__(name=name)
        w = int(width)
        self.blocks = [
            GeometryConvBlock(w, stride=2, name="stem_56"),
            GeometryConvBlock(w, stride=1, name="block_56"),
            GeometryConvBlock(w * 2, stride=2, name="down_28"),
            GeometryConvBlock(w * 2, stride=1, name="block_28_a"),
            GeometryConvBlock(w * 2, stride=1, name="block_28_b"),
            GeometryConvBlock(w * 4, stride=2, name="down_14"),
            GeometryConvBlock(w * 4, stride=1, name="block_14_a"),
            GeometryConvBlock(w * 4, stride=1, name="block_14_b"),
            GeometryConvBlock(w * 8, stride=2, name="down_7"),
            GeometryConvBlock(w * 8, stride=1, name="block_7_a"),
            GeometryConvBlock(w * 8, stride=1, name="block_7_b"),
        ]
        self.proj = tf.keras.layers.Conv2D(
            int(out_dim),
            kernel_size=1,
            padding="same",
            kernel_initializer="he_normal",
            name="project_to_rgb_dim",
        )
        self.proj_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="project_ln")
        self.proj_act = tf.keras.layers.Activation(tf.nn.gelu, name="project_gelu")

    def call(self, x, training=False):
        x = tf.cast(x, tf.float32)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.proj(x)
        x = self.proj_norm(x)
        return self.proj_act(x)


class DualConvNeXtSMIRKGuidedAttentionFER(tf.keras.Model):
    """Frozen RGB ConvNeXt-B anchor guided by a small SMIRK geometry encoder."""

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "dual_convnext_smirk_guided_attention"))
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self.feature_dim = int(model_cfg.get("feature_dim", 1024))
        self.attention_hidden_dim = int(model_cfg.get("attention_hidden_dim", 256))
        self.alpha_max = float(model_cfg.get("alpha_max", 0.2))
        self.alpha_raw_init = float(model_cfg.get("alpha_raw_init", -4.6))
        self._shape_logged = False

        rgb_cfg = self._make_backbone_cfg(cfg, name="convnext_rgb_anchor")
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(rgb_cfg)

        geometry_width = int(model_cfg.get("geometry_width", 64))
        self.geometry_encoder = SmallSMIRKGeometryEncoder(
            out_dim=self.feature_dim,
            width=geometry_width,
            name="small_smirk_geometry_encoder",
        )

        self.channel_attention_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.attention_hidden_dim, activation=tf.nn.gelu, name="dense_hidden"),
                tf.keras.layers.Dense(self.feature_dim, name="dense_gate_raw"),
            ],
            name="channel_attention_mlp",
        )

        self.alpha_raw = self.add_weight(
            name="alpha_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.alpha_raw_init),
            trainable=True,
            dtype=tf.float32,
        )

    @staticmethod
    def _make_backbone_cfg(cfg: Dict, name: str) -> Dict:
        base_cfg = {
            "seed": cfg.get("seed", {}),
            "runtime": cfg.get("runtime", {}),
            "data": dict(cfg.get("data", {})),
            "augmentation": cfg.get("augmentation", {}),
            "model": dict(cfg.get("rgb_backbone", cfg.get("model", {}))),
            "training": cfg.get("training", {}),
            "paths": cfg.get("paths", {}),
        }
        base_cfg["model"]["name"] = name
        base_cfg["model"].setdefault("arch", "convnext_base_ms1m_arcface")
        base_cfg["model"].setdefault("classifier_dropout1", 0.35)
        base_cfg["model"].setdefault("ablation", "cnn_only")
        return base_cfg

    @property
    def effective_alpha(self):
        return tf.cast(self.alpha_max, tf.float32) * tf.sigmoid(tf.cast(self.alpha_raw, tf.float32))

    def freeze_rgb_branch(self) -> None:
        self.rgb_baseline.trainable = False
        for layer in self.rgb_baseline.layers:
            layer.trainable = False

    def unfreeze_rgb_stage4(self) -> None:
        raise RuntimeError("Stage 1 contract keeps the entire RGB ckpt-43 anchor frozen.")

    def load_pretrained_weights(self, cfg: Dict, args=None) -> Tuple[str, str]:
        """Load MS1M-ArcFace weights for the RGB anchor only."""
        existing_status = getattr(self.rgb_baseline, "pretrained_load_status", "not_requested")
        if existing_status == "loaded":
            print("[DualConvNeXt] RGB MS1M-ArcFace weights already loaded by baseline constructor.", flush=True)
            return existing_status, "small_cnn_random_init"

        req = bool(cfg.get("rgb_backbone", {}).get("convnext_base_require_pretrained", False))
        ckpt_path = cfg.get("rgb_backbone", {}).get("convnext_base_pretrained_path") or "pretrained/convnext_base_ms1m_arcface.pth"

        print("[DualConvNeXt] Loading MS1M-ArcFace pretrained weights for frozen RGB ConvNeXt anchor...", flush=True)
        loader_rgb = getattr(self.rgb_baseline, "load_pytorch_pretrained", getattr(self.rgb_baseline, "_load_pytorch_pretrained", None))
        status_rgb = loader_rgb(ckpt_path, require=req) if loader_rgb else "skipped"
        return status_rgb, "small_cnn_random_init"

    def print_contract_summary(self) -> None:
        rgb_total = count_params(self.rgb_baseline.variables)
        rgb_trainable = count_params(self.rgb_baseline.trainable_variables)
        geom_total = count_params(self.geometry_encoder.variables)
        geom_trainable = count_params(self.geometry_encoder.trainable_variables)
        attn_params = count_params(self.channel_attention_mlp.variables)

        try:
            alpha_raw_val = float(self.alpha_raw.numpy())
            alpha_val = float(self.effective_alpha.numpy())
            alpha_str = f"alpha_raw={alpha_raw_val:.6f} effective_alpha={alpha_val:.10f}"
        except Exception:
            alpha_str = "alpha_raw/effective_alpha symbolic"

        print("=" * 65, flush=True)
        print("[DUAL_CONVNEXT_STAGE1_CONTRACT] Parameter Breakdown:", flush=True)
        print(f"  RGB ConvNeXt ckpt-43 anchor (Total: {rgb_total:,} | Trainable: {rgb_trainable:,})", flush=True)
        print(f"  Small SMIRK geometry CNN (Total: {geom_total:,} | Trainable: {geom_trainable:,})", flush=True)
        print(f"  3D-guided Channel Attention MLP: {attn_params:,}", flush=True)
        print(f"  Alpha constraint: alpha = {self.alpha_max:.3f} * sigmoid(alpha_raw) ({alpha_str})", flush=True)
        print(f"  Total Model Trainable Params: {count_params(self.trainable_variables):,}", flush=True)
        print("=" * 65, flush=True)

    def _log_shapes_once(self, image, geometry_maps, F_rgb, F_3d, gate_4d, F_guided, final_logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[DualConvNeXtSMIRKGuidedAttention] Shape Trace:", flush=True)
        print(f"  image_rgb: {image.shape}", flush=True)
        print(f"  geometry_maps_depth_normal_4ch: {geometry_maps.shape}", flush=True)
        print(f"  F_rgb_stage4_anchor: {F_rgb.shape}", flush=True)
        print(f"  F_3d_small_cnn: {F_3d.shape}", flush=True)
        print(f"  channel_gate_4d: {gate_4d.shape}", flush=True)
        print(f"  F_guided: {F_guided.shape}", flush=True)
        print(f"  final_logits: {final_logits.shape}", flush=True)
        self.print_contract_summary()

    def call(self, inputs, training=False):
        image = inputs["image"]
        geometry_maps = inputs["geometry_maps"]

        geom_f32 = tf.cast(geometry_maps, tf.float32)
        target_h = tf.shape(image)[1]
        target_w = tf.shape(image)[2]
        geom_f32 = tf.image.resize(geom_f32, [target_h, target_w], method="bilinear")

        # RGB is the frozen ckpt-43 anchor in Stage 1. Keep stochastic layers off.
        rgb_endpoints = self.rgb_baseline.backbone(image, training=False, return_endpoints=True)
        F_rgb = tf.stop_gradient(tf.cast(rgb_endpoints["stage4"], tf.float32))

        F_3d = tf.cast(self.geometry_encoder(geom_f32, training=training), tf.float32)

        gap_rgb = tf.reduce_mean(F_rgb, axis=[1, 2])
        gap_3d = tf.reduce_mean(F_3d, axis=[1, 2])
        concat_gap = tf.concat([gap_rgb, gap_3d], axis=-1)

        gate_raw = tf.cast(self.channel_attention_mlp(concat_gap, training=training), tf.float32)
        gate = 2.0 * tf.sigmoid(gate_raw) - 1.0
        gate_4d = tf.reshape(gate, [-1, 1, 1, self.feature_dim])

        alpha = self.effective_alpha
        F_guided = F_rgb * (1.0 + alpha * gate_4d)

        pooled_guided = self.rgb_baseline.gap(F_guided)
        dropped_guided = self.rgb_baseline.head_dropout(pooled_guided, training=False)
        final_logits = tf.cast(self.rgb_baseline.classifier(dropped_guided), tf.float32)

        self._log_shapes_once(image, geom_f32, F_rgb, F_3d, gate_4d, F_guided, final_logits)

        tf.debugging.assert_all_finite(F_3d, "NaN/Inf in F_3d geometry features")
        tf.debugging.assert_all_finite(gate, "NaN/Inf in 3D-guided channel gate")
        tf.debugging.assert_all_finite(final_logits, "NaN/Inf in final logits")

        mod_factor = 1.0 + alpha * gate_4d

        return {
            "logits": final_logits,
            "final_logits": final_logits,
            "channel_gate": gate,
            "gate_mean": tf.reduce_mean(gate),
            "gate_std": tf.math.reduce_std(gate),
            "gate_min": tf.reduce_min(gate),
            "gate_max": tf.reduce_max(gate),
            "alpha_raw": tf.cast(self.alpha_raw, tf.float32),
            "alpha": alpha,
            "effective_alpha": alpha,
            "modulation_factor_min": tf.reduce_min(mod_factor),
            "modulation_factor_max": tf.reduce_max(mod_factor),
            "F_rgb": F_rgb,
            "F_3d": F_3d,
            "F_guided": F_guided,
        }

