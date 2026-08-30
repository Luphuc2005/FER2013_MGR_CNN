"""
Dual ConvNeXt-Base MS1M RGB + SMIRK Geometry with bounded 3D-guided
residual channel attention.

This experiment intentionally does not restore any FER checkpoint. Both
ConvNeXt backbones are initialized only from the MS1M-ArcFace pretrained
checkpoint; FER classifier, fusion, attention, and auxiliary head stay random.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf

from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))


class DualConvNeXtSMIRKGuidedAttentionMS1MFERScratch(tf.keras.Model):
    """MS1M-initialized dual ConvNeXt with random FER heads and bounded alpha."""

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "dual_convnext_smirk_guided_attention_ms1m_fer_scratch"))
        self.cfg = cfg
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self.feature_dim = int(cfg.get("model", {}).get("feature_dim", 1024))
        self.attention_hidden_dim = int(cfg.get("model", {}).get("attention_hidden_dim", 256))
        self.alpha_max = float(cfg.get("model", {}).get("alpha_max", 0.2))
        self._shape_logged = False

        rgb_cfg = self._make_backbone_cfg(cfg, name="convnext_rgb_ms1m_fer_scratch")
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(rgb_cfg)

        geom_cfg = self._make_backbone_cfg(cfg, name="convnext_geometry_ms1m_fer_scratch")
        self.geometry_baseline = ConvNeXtBaseFaceFERBaseline(geom_cfg)

        self.geometry_fusion = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(self.feature_dim, 1, padding="same", activation=tf.nn.gelu, name="conv1"),
                tf.keras.layers.Conv2D(self.feature_dim, 1, padding="same", name="conv2"),
            ],
            name="geometry_fusion",
        )

        self.channel_attention_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.attention_hidden_dim, activation=tf.nn.gelu, name="dense_hidden"),
                tf.keras.layers.Dense(self.feature_dim, activation=tf.nn.tanh, name="dense_gate"),
            ],
            name="channel_attention_mlp",
        )

        self.alpha_raw = self.add_weight(
            name="alpha_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32,
        )

        self.aux_3d_head = tf.keras.layers.Dense(self.num_classes, name="aux_3d_head")

    @staticmethod
    def _make_backbone_cfg(cfg: Dict, name: str) -> Dict:
        model_cfg = dict(cfg.get("rgb_backbone", cfg.get("model", {})))
        # Pretrained loading is done explicitly by load_ms1m_pretrained_weights()
        # so the smoke test can verify the FER classifier is untouched.
        model_cfg.pop("convnext_base_pretrained_path", None)
        model_cfg.pop("pretrained_path", None)
        model_cfg["name"] = name
        model_cfg.setdefault("arch", "convnext_base_ms1m_arcface")
        model_cfg.setdefault("classifier_dropout1", 0.35)
        model_cfg.setdefault("ablation", "cnn_only")

        return {
            "seed": cfg.get("seed", {}),
            "runtime": cfg.get("runtime", {}),
            "data": dict(cfg.get("data", {})),
            "augmentation": cfg.get("augmentation", {}),
            "model": model_cfg,
            "training": cfg.get("training", {}),
            "paths": cfg.get("paths", {}),
        }

    @property
    def effective_alpha(self) -> tf.Tensor:
        return tf.cast(self.alpha_max, tf.float32) * tf.tanh(tf.cast(self.alpha_raw, tf.float32))

    def load_ms1m_pretrained_weights(self, cfg: Dict) -> Tuple[str, str]:
        ckpt_path = cfg.get("rgb_backbone", {}).get("convnext_base_pretrained_path") or "pretrained/convnext_base_ms1m_arcface.pth"
        require = bool(cfg.get("rgb_backbone", {}).get("convnext_base_require_pretrained", True))

        rgb_classifier_before = [v.numpy().copy() for v in self.rgb_baseline.classifier.weights]

        print("[DualConvNeXtMS1MScratch] Loading MS1M-ArcFace weights for RGB ConvNeXt backbone...", flush=True)
        status_rgb = self.rgb_baseline.load_pytorch_pretrained(ckpt_path, require=require)
        rgb_targets = len(self.rgb_baseline.backbone.weights)
        if status_rgb != "loaded" or rgb_targets != 340:
            raise RuntimeError(f"RGB MS1M load contract failed: status={status_rgb}, target_tensors={rgb_targets}")
        print(f"RGB_MS1M_LOAD_OK matched=340/340 path={ckpt_path}", flush=True)

        print("[DualConvNeXtMS1MScratch] Loading MS1M-ArcFace weights for Geometry ConvNeXt backbone...", flush=True)
        status_geom = self.geometry_baseline.load_pytorch_pretrained(ckpt_path, require=require)
        geom_targets = len(self.geometry_baseline.backbone.weights)
        if status_geom != "loaded" or geom_targets != 340:
            raise RuntimeError(f"Geometry MS1M load contract failed: status={status_geom}, target_tensors={geom_targets}")
        print(f"GEOMETRY_MS1M_LOAD_OK matched=340/340 path={ckpt_path}", flush=True)

        max_classifier_diff = 0.0
        for before, after in zip(rgb_classifier_before, self.rgb_baseline.classifier.weights):
            max_classifier_diff = max(max_classifier_diff, float(np.max(np.abs(before - after.numpy()))))
        if max_classifier_diff != 0.0:
            raise RuntimeError(f"FER classifier changed during MS1M backbone load: max_diff={max_classifier_diff:.8e}")
        print(f"FER_CLASSIFIER_RANDOM_INIT_OK not_restored=True max_diff_after_ms1m_load={max_classifier_diff:.8e}", flush=True)
        print("FER_CHECKPOINT_RESTORE_SKIPPED_OK no_ckpt_43_restore_path_used", flush=True)
        return status_rgb, status_geom

    def _log_shapes_once(
        self,
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
    ) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[DualConvNeXtSMIRKGuidedAttentionMS1MFERScratch] Shape Trace:", flush=True)
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
        print(f"  alpha_raw: {float(self.alpha_raw.numpy()):.8f}", flush=True)
        print(f"  effective_alpha: {float(self.effective_alpha.numpy()):.8f}", flush=True)
        print(f"  alpha_range_theoretical: [{-self.alpha_max:.3f}, {self.alpha_max:.3f}]", flush=True)

    def call(self, inputs, training=False):
        image = inputs["image"]
        geometry_maps = inputs["geometry_maps"]

        geom_f32 = tf.cast(geometry_maps, tf.float32)
        target_h = tf.shape(image)[1]
        target_w = tf.shape(image)[2]
        geom_f32 = tf.image.resize(geom_f32, [target_h, target_w], method="bilinear")

        depth_1ch = geom_f32[..., 0:1]
        depth_rgb = tf.repeat(depth_1ch, 3, axis=-1)
        normal_rgb = geom_f32[..., 1:4]

        rgb_endpoints = self.rgb_baseline.backbone(image, training=training, return_endpoints=True)
        F_rgb = tf.cast(rgb_endpoints["stage4"], tf.float32)

        depth_endpoints = self.geometry_baseline.backbone(depth_rgb, training=training, return_endpoints=True)
        F_depth = tf.cast(depth_endpoints["stage4"], tf.float32)

        normal_endpoints = self.geometry_baseline.backbone(normal_rgb, training=training, return_endpoints=True)
        F_normal = tf.cast(normal_endpoints["stage4"], tf.float32)

        concat_geom = tf.concat([F_depth, F_normal], axis=-1)
        F_3d = tf.cast(self.geometry_fusion(concat_geom, training=training), tf.float32)

        gap_rgb = tf.reduce_mean(F_rgb, axis=[1, 2])
        gap_3d = tf.reduce_mean(F_3d, axis=[1, 2])
        concat_gap = tf.concat([gap_rgb, gap_3d], axis=-1)

        gate = tf.cast(self.channel_attention_mlp(concat_gap, training=training), tf.float32)
        gate_4d = tf.reshape(gate, [-1, 1, 1, self.feature_dim])

        effective_alpha = self.effective_alpha
        modulation_factor = 1.0 + effective_alpha * gate_4d
        F_guided = F_rgb * modulation_factor

        pooled_guided = self.rgb_baseline.gap(F_guided)
        dropped_guided = self.rgb_baseline.head_dropout(pooled_guided, training=training)
        final_logits = tf.cast(self.rgb_baseline.classifier(dropped_guided), tf.float32)

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
            "alpha_raw": self.alpha_raw,
            "effective_alpha": effective_alpha,
            "alpha": effective_alpha,
            "modulation_factor": modulation_factor,
            "modulation_factor_min": tf.reduce_min(modulation_factor),
            "modulation_factor_max": tf.reduce_max(modulation_factor),
            "mean_abs_channel_gate": tf.reduce_mean(tf.abs(gate)),
            "F_rgb": F_rgb,
            "F_3d": F_3d,
            "F_guided": F_guided,
        }
