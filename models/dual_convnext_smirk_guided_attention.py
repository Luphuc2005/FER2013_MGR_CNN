"""
Dual ConvNeXt MS1M RGB + SMIRK Geometry with 3D-Guided Channel Attention for FER2013.

Architecture:
1. RGB Branch:
   Input FER RGB [B, 112, 112, 3] -> ConvNeXt-Base MS1M-ArcFace -> Stage4 feature F_rgb [B, 7, 7, 1024].
2. 3D Branch:
   depth_rgb [B, 112, 112, 3] and normal_rgb [B, 112, 112, 3] -> Shared-weight Geometry ConvNeXt MS1M
   -> F_depth [B, 7, 7, 1024] and F_normal [B, 7, 7, 1024]
   -> Concat -> Conv1x1 Fusion MLP -> F_3d [B, 7, 7, 1024].
3. 3D-Guided Channel Attention:
   GAP(F_rgb) [1024] + GAP(F_3d) [1024] -> Concat [2048] -> Dense(256, GELU) -> Dense(1024, Tanh)
   -> channel_gate [B, 1, 1, 1024]
   Modulation: F_guided = F_rgb * (1 + alpha * channel_gate), where alpha is a trainable scalar initialized to 0.0.
4. Classification:
   F_guided -> GAP -> Dropout -> Dense(7) (final_logits)
   F_3d -> GAP -> Dense(7) (aux_3d_logits)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


class DualConvNeXtSMIRKGuidedAttentionFER(tf.keras.Model):
    """Dual ConvNeXt MS1M RGB + SMIRK Geometry with 3D-Guided Channel Attention."""

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "dual_convnext_smirk_guided_attention"))
        self.cfg = cfg
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self.feature_dim = int(cfg.get("model", {}).get("feature_dim", 1024))
        self.attention_hidden_dim = int(cfg.get("model", {}).get("attention_hidden_dim", 256))
        self._shape_logged = False

        # 1. RGB ConvNeXt Baseline (MS1M Pretrained)
        rgb_cfg = self._make_backbone_cfg(cfg, name="convnext_rgb")
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(rgb_cfg)

        # 2. Shared-Weight Geometry ConvNeXt Baseline (MS1M Pretrained)
        geom_cfg = self._make_backbone_cfg(cfg, name="convnext_geometry")
        self.geometry_baseline = ConvNeXtBaseFaceFERBaseline(geom_cfg)

        # 3. Geometry Feature Fusion (1x1 Conv MLP)
        self.geometry_fusion = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(self.feature_dim, 1, padding="same", activation=tf.nn.gelu, name="conv1"),
                tf.keras.layers.Conv2D(self.feature_dim, 1, padding="same", name="conv2"),
            ],
            name="geometry_fusion",
        )

        # 4. 3D-Guided Channel Attention MLP
        self.channel_attention_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.attention_hidden_dim, activation=tf.nn.gelu, name="dense_hidden"),
                tf.keras.layers.Dense(self.feature_dim, activation=tf.nn.tanh, name="dense_gate"),
            ],
            name="channel_attention_mlp",
        )

        # 5. Trainable modulation scalar alpha (initialized to 0.0)
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32,
        )

        # 6. Auxiliary 3D Classifier Head
        self.aux_3d_head = tf.keras.layers.Dense(self.num_classes, name="aux_3d_head")

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

    def freeze_rgb_branch(self) -> None:
        self.rgb_baseline.trainable = False
        for layer in self.rgb_baseline.layers:
            layer.trainable = False

    def unfreeze_rgb_stage4(self) -> None:
        """Unfreeze Stage 4 of RGB ConvNeXt for fine-tuning."""
        self.rgb_baseline.trainable = True
        # Keep stem and stages 1..3 frozen, unfreeze stage 4 and head
        for layer in self.rgb_baseline.backbone.stem_layers:
            layer.trainable = False
        for stage_idx, stage in enumerate(self.rgb_baseline.backbone.stages):
            if stage_idx < 3:
                for block in stage:
                    block.trainable = False
            else:
                for block in stage:
                    block.trainable = True

    def load_pretrained_weights(self, cfg: Dict, args=None) -> Tuple[str, str]:
        """Load PyTorch MS1M-ArcFace pretrained weights into BOTH RGB and Geometry backbones."""
        req = bool(cfg.get("rgb_backbone", {}).get("convnext_base_require_pretrained", False))
        ckpt_path = cfg.get("rgb_backbone", {}).get("convnext_base_pretrained_path") or "pretrained/convnext_base_ms1m_arcface.pth"
        
        print("[DualConvNeXt] Loading MS1M-ArcFace pretrained weights for RGB ConvNeXt...", flush=True)
        status_rgb = self.rgb_baseline.load_pytorch_pretrained(ckpt_path, require=req)
        
        print("[DualConvNeXt] Loading MS1M-ArcFace pretrained weights for Shared Geometry ConvNeXt...", flush=True)
        status_geom = self.geometry_baseline.load_pytorch_pretrained(ckpt_path, require=req)
        
        return status_rgb, status_geom

    def print_contract_summary(self) -> None:
        rgb_total = count_params(self.rgb_baseline.variables)
        rgb_trainable = count_params(self.rgb_baseline.trainable_variables)
        geom_total = count_params(self.geometry_baseline.variables)
        geom_trainable = count_params(self.geometry_baseline.trainable_variables)
        fusion_params = count_params(self.geometry_fusion.variables)
        attn_params = count_params(self.channel_attention_mlp.variables)
        aux_params = count_params(self.aux_3d_head.variables)

        print("=" * 65, flush=True)
        print("[DUAL_CONVNEXT_CONTRACT] Parameter Breakdown:", flush=True)
        print(f"  RGB ConvNeXt (Total: {rgb_total:,} | Trainable: {rgb_trainable:,})", flush=True)
        print(f"  Geometry ConvNeXt (Total: {geom_total:,} | Trainable: {geom_trainable:,})", flush=True)
        print(f"  Geometry Fusion Conv1x1: {fusion_params:,}", flush=True)
        print(f"  3D-Guided Attention MLP: {attn_params:,}", flush=True)
        print(f"  Auxiliary 3D Head: {aux_params:,}", flush=True)
        print(f"  Modulation Alpha (scalar): 1 (value={float(self.alpha.numpy()):.6f})", flush=True)
        print(f"  Total Model Trainable Params: {count_params(self.trainable_variables):,}", flush=True)
        print("=" * 65, flush=True)

    def _log_shapes_once(self, image, depth_rgb, normal_rgb, F_rgb, F_depth, F_normal, F_3d, gate_4d, F_guided, final_logits, aux_3d_logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[DualConvNeXtSMIRKGuidedAttention] Shape Trace:", flush=True)
        print(f"  image_rgb: {image.shape}", flush=True)
        print(f"  depth_rgb_3ch: {depth_rgb.shape}", flush=True)
        print(f"  normal_rgb_3ch: {normal_rgb.shape}", flush=True)
        print(f"  F_rgb_stage4: {F_rgb.shape}", flush=True)
        print(f"  F_depth_stage4: {F_depth.shape}", flush=True)
        print(f"  F_normal_stage4: {F_normal.shape}", flush=True)
        print(f"  F_3d_fused: {F_3d.shape}", flush=True)
        print(f"  channel_gate_4d: {gate_4d.shape}", flush=True)
        print(f"  F_guided: {F_guided.shape}", flush=True)
        print(f"  final_logits: {final_logits.shape}", flush=True)
        print(f"  aux_3d_logits: {aux_3d_logits.shape}", flush=True)
        self.print_contract_summary()

    def call(self, inputs, training=False):
        image = inputs["image"]
        geometry_maps = inputs["geometry_maps"]

        # Extract 3-channel depth and normal images from 4-channel geometry cache
        depth_1ch = geometry_maps[..., 0:1]
        depth_rgb = tf.repeat(depth_1ch, 3, axis=-1)
        normal_rgb = geometry_maps[..., 1:4]

        # 1. RGB Branch
        rgb_endpoints = self.rgb_baseline.backbone(image, training=training, return_endpoints=True)
        F_rgb = tf.cast(rgb_endpoints["stage4"], tf.float32)  # [B, 7, 7, 1024]

        # 2. 3D Branch via Shared Geometry ConvNeXt
        depth_endpoints = self.geometry_baseline.backbone(depth_rgb, training=training, return_endpoints=True)
        F_depth = tf.cast(depth_endpoints["stage4"], tf.float32)  # [B, 7, 7, 1024]

        normal_endpoints = self.geometry_baseline.backbone(normal_rgb, training=training, return_endpoints=True)
        F_normal = tf.cast(normal_endpoints["stage4"], tf.float32)  # [B, 7, 7, 1024]

        concat_geom = tf.concat([F_depth, F_normal], axis=-1)  # [B, 7, 7, 2048]
        F_3d = tf.cast(self.geometry_fusion(concat_geom, training=training), tf.float32)  # [B, 7, 7, 1024]

        # 3. 3D-Guided Channel Attention
        gap_rgb = tf.reduce_mean(F_rgb, axis=[1, 2])  # [B, 1024]
        gap_3d = tf.reduce_mean(F_3d, axis=[1, 2])  # [B, 1024]
        concat_gap = tf.concat([gap_rgb, gap_3d], axis=-1)  # [B, 2048]

        gate = tf.cast(self.channel_attention_mlp(concat_gap, training=training), tf.float32)  # [B, 1024]
        gate_4d = tf.reshape(gate, [-1, 1, 1, self.feature_dim])  # [B, 1, 1, 1024]

        # 4. Channel Modulation: F_guided = F_rgb * (1 + alpha * channel_gate)
        alpha_f32 = tf.cast(self.alpha, tf.float32)
        F_guided = F_rgb * (1.0 + alpha_f32 * gate_4d)  # [B, 7, 7, 1024]

        # 5. Final Classification Head
        pooled_guided = self.rgb_baseline.gap(F_guided)
        dropped_guided = self.rgb_baseline.head_dropout(pooled_guided, training=training)
        final_logits = tf.cast(self.rgb_baseline.classifier(dropped_guided), tf.float32)

        # 6. Auxiliary 3D Head
        pooled_3d = tf.reduce_mean(F_3d, axis=[1, 2])
        aux_3d_logits = tf.cast(self.aux_3d_head(pooled_3d), tf.float32)

        self._log_shapes_once(
            image,
            depth_rgb,
            normal_rgb,
            F_rgb,
            F_depth,
            F_normal,
            F_3d,
            gate_4d,
            F_guided,
            final_logits,
            aux_3d_logits,
        )

        return {
            "logits": final_logits,
            "final_logits": final_logits,
            "aux_3d_logits": aux_3d_logits,
            "channel_gate": gate,
            "alpha": self.alpha,
            "F_rgb": F_rgb,
            "F_3d": F_3d,
            "F_guided": F_guided,
        }
