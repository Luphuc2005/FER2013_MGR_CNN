from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


class ConvNeXtSMIRKAuxiliaryFER(tf.keras.Model):
    """ConvNeXt-Base FER with SMIRK/FLAME 3D Expression Auxiliary Supervision.

    Architecture:
        RGB Image -> ConvNeXt Backbone (Stage 1-2 Frozen, Stage 3-4 Fine-tuned) -> GAP (1024-dim)
            |--> Branch 1 (FER Classifier Head): Dense(1024) -> GELU -> Dropout -> Dense(7) -> 7 FER Logits
            |--> Branch 2 (Auxiliary Geometry Head): Dense(512) -> GELU -> Dense(geo_dim) -> 3D Parameters

    Inference:
        Only Branch 1 is required. Zero SMIRK runtime overhead at inference time!
    """

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "convnext_smirk_auxiliary"))
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        self.num_classes = int(data_cfg.get("num_classes", 7))
        self.ablation = str(model_cfg.get("ablation", "exp_jaw_head")).lower()

        # Geometry dimensions based on ablation mode
        # exp: 50 | exp_jaw: 53 | exp_jaw_head: 56 | baseline: 0
        if self.ablation == "baseline":
            self.geo_dim = 0
        elif self.ablation == "exp":
            self.geo_dim = 50
        elif self.ablation == "exp_jaw":
            self.geo_dim = 53
        elif self.ablation in ("exp_jaw_head", "all"):
            self.geo_dim = 56
        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation}. Choice: ('baseline', 'exp', 'exp_jaw', 'exp_jaw_head')")

        self.dropout_rate = float(model_cfg.get("dropout", 0.35))
        self._shape_logged = False

        # Build baseline ConvNeXt backbone
        baseline_cfg = self._make_rgb_baseline_cfg(cfg)
        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(baseline_cfg)

        # Apply Stage 1-2 Freezing
        self._freeze_stage1_2()

        # Branch 1: FER Classification Head
        self.fer_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1024, activation=tf.nn.gelu, name="fer_dense_1"),
                tf.keras.layers.Dropout(self.dropout_rate, name="fer_dropout"),
                tf.keras.layers.Dense(self.num_classes, name="fer_logits"),
            ],
            name="fer_classification_head",
        )

        # Branch 2: Auxiliary Geometry Regression Head (Only built if geo_dim > 0)
        if self.geo_dim > 0:
            self.geometry_head = tf.keras.Sequential(
                [
                    tf.keras.layers.LayerNormalization(epsilon=1e-6, name="geo_ln"),
                    tf.keras.layers.Dense(512, activation=tf.nn.gelu, name="geo_dense_1"),
                    tf.keras.layers.Dropout(0.20, name="geo_dropout"),
                    tf.keras.layers.Dense(self.geo_dim, name="geo_predictions"),
                ],
                name="auxiliary_geometry_head",
            )
        else:
            self.geometry_head = None

    def _freeze_stage1_2(self) -> None:
        """Freeze stem, stage1, and stage2 of ConvNeXt backbone to preserve pretrained MS1M features."""
        backbone = self.rgb_baseline.backbone
        # Stem
        if hasattr(backbone, "stem"):
            backbone.stem.trainable = False
            for l in getattr(backbone.stem, "layers", []):
                l.trainable = False
        # Stage 1 & 2
        for stage_idx in (0, 1):
            if hasattr(backbone, "stages") and len(backbone.stages) > stage_idx:
                backbone.stages[stage_idx].trainable = False
                for l in getattr(backbone.stages[stage_idx], "layers", []):
                    l.trainable = False
        print(f"[ConvNeXtSMIRKAuxiliaryFER] Stem, Stage 1, Stage 2 FROZEN=True. Stage 3, Stage 4 TRAINABLE=True.", flush=True)

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

    def call(self, inputs, training=False):
        image = inputs["image"] if isinstance(inputs, dict) else inputs

        endpoints = self.rgb_baseline.backbone(
            image,
            training=training,
            return_endpoints=True,
            stage3_adapter=getattr(self.rgb_baseline, "stage3_adapter", None),
        )
        stage4 = endpoints["stage4"]
        if getattr(self.rgb_baseline, "use_eca", False) and self.rgb_baseline.stage4_eca is not None:
            stage4 = self.rgb_baseline.stage4_eca(stage4, training=training)

        gap_feature = tf.reduce_mean(stage4, axis=[1, 2])

        fer_logits = tf.cast(self.fer_head(gap_feature, training=training), tf.float32)

        if self.geometry_head is not None:
            geometry_pred = tf.cast(self.geometry_head(gap_feature, training=training), tf.float32)
        else:
            geometry_pred = None

        if not self._shape_logged:
            self._shape_logged = True
            print("[ConvNeXtSMIRKAuxiliaryFER] Shape trace:", flush=True)
            print(f"  image: {image.shape}", flush=True)
            print(f"  stage4: {stage4.shape}", flush=True)
            print(f"  gap_feature: {gap_feature.shape}", flush=True)
            print(f"  fer_logits: {fer_logits.shape}", flush=True)
            if geometry_pred is not None:
                print(f"  geometry_pred: {geometry_pred.shape} (ablation={self.ablation}, geo_dim={self.geo_dim})", flush=True)

        result = {"logits": fer_logits, "feature": gap_feature}
        if geometry_pred is not None:
            result["geometry_pred"] = geometry_pred
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
