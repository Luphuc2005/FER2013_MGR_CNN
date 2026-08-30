from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from .convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


class SMIRKGeometryCrossAttentionFER(tf.keras.Model):
    """ConvNeXt-Base FER with Confidence-Aware Sample-Wise Dynamic Gating.

    The ConvNeXt-Base MS1M-ArcFace baseline backbone & classifier head remain
    completely frozen. Cross-attention on SMIRK geometry tokens produces a 7-dim
    residual correction (geometry_delta).

    A sample-wise dynamic gate g_i in (0, 1) is computed based on:
        [baseline_confidence_i, baseline_entropy_i, RGB_feature_i, geometry_feature_i]

    Final Logits:
        final_logits_i = baseline_logits_i + g_i * geometry_delta_i
    """

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "smirk_geometry_cross_attention"))
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        self.num_classes = int(data_cfg.get("num_classes", 7))
        self.rgb_dim = int(model_cfg.get("rgb_token_dim", 1024))
        self.geometry_dim = int(model_cfg.get("geometry_token_dim", 768))
        self.fusion_dim = int(model_cfg.get("fusion_dim", 512))
        self.num_heads = int(model_cfg.get("num_heads", 8))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 1024))
        self.dropout_rate = float(model_cfg.get("dropout", 0.20))
        self.force_zero_gate = False
        self._shape_logged = False

        # Build baseline ConvNeXt and freeze completely (backbone + classifier)
        baseline_cfg = self._make_rgb_baseline_cfg(cfg)
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(baseline_cfg)
        self.rgb_baseline.trainable = False

        # Trainable Geometry Cross-Attention components
        self.rgb_project = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                tf.keras.layers.Dense(self.fusion_dim, name="proj"),
            ],
            name="rgb_token_projection",
        )
        self.geometry_project = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                tf.keras.layers.Dense(self.fusion_dim, name="proj"),
            ],
            name="geometry_token_projection",
        )
        key_dim = max(1, self.fusion_dim // self.num_heads)
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name="rgb_query_geometry_kv_cross_attention",
        )
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate, name="cross_attention_dropout")

        # Geometry Residual Correction Head (MLP: GAP(cross_tokens) -> 7 logits)
        self.correction_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="correction_ln"),
                tf.keras.layers.Dense(self.ffn_dim, activation=tf.nn.gelu, name="correction_dense_1"),
                tf.keras.layers.Dropout(self.dropout_rate, name="correction_drop"),
                tf.keras.layers.Dense(self.num_classes, name="correction_dense_2"),
            ],
            name="geometry_correction_mlp",
        )

        # Confidence-Aware Sample-Wise Dynamic Gate MLP
        # Inputs: [baseline_confidence (1), baseline_entropy (1), rgb_feat (512), geometry_feat (512)] -> 1026
        init_bias = float(np.log(float(model_cfg.get("beta_init", 0.05)) / (1.0 - float(model_cfg.get("beta_init", 0.05)))))
        self.gate_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="gate_ln"),
                tf.keras.layers.Dense(128, activation=tf.nn.gelu, name="gate_dense_1"),
                tf.keras.layers.Dropout(self.dropout_rate, name="gate_drop"),
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    name="gate_dense_2",
                    bias_initializer=tf.keras.initializers.Constant(init_bias),
                ),
            ],
            name="confidence_aware_dynamic_gate_mlp",
        )

    @staticmethod
    def _make_rgb_baseline_cfg(cfg: Dict) -> Dict:
        baseline_cfg = {
            "seed": cfg.get("seed", {}),
            "runtime": cfg.get("runtime", {}),
            "data": dict(cfg.get("data", {})),
            "augmentation": cfg.get("augmentation", {}),
            "model": dict(cfg.get("rgb_backbone", cfg.get("model", {}))),
            "training": cfg.get("training", {}),
            "paths": cfg.get("paths", {}),
        }
        baseline_cfg["model"].setdefault("name", "convnext_base_ms1m_arcface_baseline")
        baseline_cfg["model"].setdefault("arch", "convnext_base_ms1m_arcface")
        baseline_cfg["model"].setdefault("pretrained", True)
        baseline_cfg["model"].setdefault("convnext_base_require_pretrained", True)
        baseline_cfg["model"].setdefault("classifier_dropout1", 0.35)
        baseline_cfg["model"].setdefault("ablation", "cnn_only")
        return baseline_cfg

    def call(self, inputs, training=False, return_attention: bool = False):
        image = inputs["image"]
        geometry_tokens = inputs["geometry_tokens"]

        # 1. Baseline forwarding (always training=False for frozen baseline)
        baseline_outputs = self.rgb_baseline(image, training=False)
        baseline_logits = tf.cast(baseline_outputs["logits"], tf.float32)

        # Baseline confidence & normalized entropy
        baseline_probs = tf.nn.softmax(baseline_logits, axis=-1)
        baseline_confidence = tf.reduce_max(baseline_probs, axis=-1, keepdims=True)
        entropy_raw = -tf.reduce_sum(baseline_probs * tf.math.log(baseline_probs + 1e-7), axis=-1, keepdims=True)
        baseline_entropy = entropy_raw / 1.9459101490553132 # log(7)

        # 2. Extract stage4 RGB tokens for Cross-Attention Query
        endpoints = self.rgb_baseline.backbone(
            image,
            training=False,
            return_endpoints=True,
            stage3_adapter=getattr(self.rgb_baseline, "stage3_adapter", None),
        )
        stage4 = endpoints["stage4"]
        if getattr(self.rgb_baseline, "use_eca", False) and self.rgb_baseline.stage4_eca is not None:
            stage4 = self.rgb_baseline.stage4_eca(stage4, training=False)

        batch_size = tf.shape(stage4)[0]
        rgb_tokens = tf.reshape(stage4, [batch_size, -1, self.rgb_dim])

        # 3. Geometry Cross-Attention
        rgb_proj = self.rgb_project(rgb_tokens, training=training)
        geometry_proj = self.geometry_project(geometry_tokens, training=training)
        cross_tokens, attention_scores = self.cross_attention(
            query=rgb_proj,
            key=geometry_proj,
            value=geometry_proj,
            training=training,
            return_attention_scores=True,
        )
        cross_tokens = self.attn_dropout(cross_tokens, training=training)

        # 4. Geometry Delta Correction via MLP(GAP(cross_tokens))
        gap = tf.reduce_mean(cross_tokens, axis=1)
        rgb_feat = tf.reduce_mean(rgb_proj, axis=1)
        geometry_delta = tf.cast(self.correction_mlp(gap, training=training), tf.float32)

        # 5. Confidence-Aware Sample-Wise Dynamic Gate
        baseline_conf_f32 = tf.cast(baseline_confidence, tf.float32)
        baseline_ent_f32 = tf.cast(baseline_entropy, tf.float32)
        rgb_feat_f32 = tf.cast(rgb_feat, tf.float32)
        gap_f32 = tf.cast(gap, tf.float32)
        gate_inputs = tf.concat([baseline_conf_f32, baseline_ent_f32, rgb_feat_f32, gap_f32], axis=-1)
        gate = tf.cast(self.gate_mlp(gate_inputs, training=training), tf.float32)

        if getattr(self, "force_zero_gate", False):
            gate = tf.zeros_like(gate)

        final_logits = baseline_logits + gate * geometry_delta

        if not self._shape_logged:
            self._shape_logged = True
            print("[SMIRKGeometryCrossAttention (Sample-Wise Dynamic Gate)] Shape trace:", flush=True)
            print(f"  image: {image.shape}", flush=True)
            print(f"  convnext_stage4: {stage4.shape}", flush=True)
            print(f"  rgb_tokens_Q_raw: {rgb_tokens.shape}", flush=True)
            print(f"  geometry_tokens_KV_raw: {geometry_tokens.shape}", flush=True)
            print(f"  rgb_tokens_Q_projected: {rgb_proj.shape}", flush=True)
            print(f"  geometry_tokens_KV_projected: {geometry_proj.shape}", flush=True)
            print(f"  cross_attention: {cross_tokens.shape}", flush=True)
            print(f"  gap: {gap.shape}", flush=True)
            print(f"  geometry_delta: {geometry_delta.shape}", flush=True)
            print(f"  baseline_logits: {baseline_logits.shape}", flush=True)
            print(f"  gate_sample_wise: {gate.shape}", flush=True)
            print(f"  final_logits: {final_logits.shape}", flush=True)

        result = {
            "logits": tf.cast(final_logits, tf.float32),
            "baseline_logits": tf.cast(baseline_logits, tf.float32),
            "geometry_delta": tf.cast(geometry_delta, tf.float32),
            "gate": tf.cast(gate, tf.float32),
            "baseline_confidence": baseline_confidence,
            "baseline_entropy": baseline_entropy,
            "rgb_tokens": rgb_tokens,
            "geometry_tokens": geometry_tokens,
            "cross_tokens": cross_tokens,
        }
        if return_attention:
            result["attention_scores"] = attention_scores
        return result


def resolve_latest_checkpoint(path_value: Optional[str]) -> Optional[str]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if path.is_dir():
        latest = tf.train.latest_checkpoint(str(path))
        return latest
    index_path = Path(str(path) + ".index")
    if index_path.exists() or path.exists():
        return str(path)
    return None
