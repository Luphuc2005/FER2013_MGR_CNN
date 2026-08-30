from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf

from .convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline


def _variable_key(variable) -> object:
    ref = getattr(variable, "ref", None)
    if callable(ref):
        return ref()
    experimental_ref = getattr(variable, "experimental_ref", None)
    if callable(experimental_ref):
        return experimental_ref()
    return getattr(variable, "path", None) or getattr(variable, "name", None) or id(variable)


def count_params(variables: Iterable[tf.Variable]) -> int:
    return int(np.sum([np.prod(v.shape) for v in variables])) if variables else 0


class GeometryResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, stride: int = 1, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.stride = int(stride)
        self.conv1 = tf.keras.layers.Conv2D(self.filters, 3, strides=self.stride, padding="same", use_bias=False, name="conv1")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.act1 = tf.keras.layers.Activation(tf.nn.gelu, name="gelu1")
        self.drop = tf.keras.layers.Dropout(float(dropout), name="drop")
        self.conv2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", use_bias=False, name="conv2")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.proj = None
        self.out_act = tf.keras.layers.Activation(tf.nn.gelu, name="out_gelu")

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        if in_channels != self.filters or self.stride != 1:
            self.proj = tf.keras.layers.Conv2D(
                self.filters,
                1,
                strides=self.stride,
                padding="same",
                use_bias=False,
                name="skip_proj",
            )
        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = x if self.proj is None else self.proj(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop(y, training=training)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.out_act(shortcut + y)


class SmallDepthNormalCNN(tf.keras.layers.Layer):
    """Small 4-channel depth+normal CNN that emits a 512-d geometry feature."""

    def __init__(self, cfg: Dict, **kwargs):
        super().__init__(name=kwargs.pop("name", "geometry_cnn"), **kwargs)
        model_cfg = cfg.get("model", {})
        width = int(model_cfg.get("geometry_base_width", 48))
        dropout = float(model_cfg.get("geometry_dropout", 0.15))
        self.stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(width, 3, strides=2, padding="same", use_bias=False, name="conv"),
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                tf.keras.layers.Activation(tf.nn.gelu, name="gelu"),
            ],
            name="stem",
        )
        self.block1 = GeometryResidualBlock(width, stride=1, dropout=dropout, name="block1")
        self.block2 = GeometryResidualBlock(width * 2, stride=2, dropout=dropout, name="block2")
        self.block3 = GeometryResidualBlock(width * 4, stride=2, dropout=dropout, name="block3")
        self.block4 = GeometryResidualBlock(width * 8, stride=2, dropout=dropout, name="block4")
        self.block5 = GeometryResidualBlock(width * 8, stride=1, dropout=dropout, name="block5")
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.feature = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                tf.keras.layers.Dense(512, activation=tf.nn.gelu, kernel_initializer="he_normal", name="dense"),
                tf.keras.layers.Dropout(dropout, name="dropout"),
            ],
            name="geometry_feature_512",
        )

    def call(self, geometry_maps, training=False, return_endpoints: bool = False):
        endpoints = {}
        x = self.stem(geometry_maps, training=training)
        endpoints["geometry_stem"] = x
        for name in ("block1", "block2", "block3", "block4", "block5"):
            x = getattr(self, name)(x, training=training)
            endpoints[f"geometry_{name}"] = x
        pooled = self.gap(x)
        geom_feat = self.feature(pooled, training=training)
        endpoints["geometry_gap"] = pooled
        endpoints["geometry_feature"] = geom_feat
        if return_endpoints:
            return geom_feat, endpoints
        return geom_feat


class Stage1RGBSMIRK3DCNNLateFusionFER(tf.keras.Model):
    """Stage 1 RGB + frozen-SMIRK depth/normal 3D CNN late fusion for FER2013."""

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "stage1_rgb_smirk_3d_cnn_late_fusion"))
        self.cfg = cfg
        self.num_classes = int(cfg.get("data", {}).get("num_classes", 7))
        self.rgb_dim = int(cfg.get("model", {}).get("rgb_feature_dim", 1024))
        self.geometry_dim = int(cfg.get("model", {}).get("geometry_feature_dim", 512))
        self._shape_logged = False

        self.rgb_baseline = ConvNeXtBaseFaceFERBaseline(self._make_rgb_baseline_cfg(cfg))
        self.rgb_baseline.trainable = False

        self.geometry_cnn = SmallDepthNormalCNN(cfg, name="geometry_cnn")
        self.geometry_head = tf.keras.layers.Dense(self.num_classes, name="geometry_head")
        self.fusion_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation=tf.nn.gelu, kernel_initializer="he_normal", name="dense_512"),
                tf.keras.layers.Dropout(float(cfg.get("model", {}).get("fusion_dropout", 0.30)), name="dropout"),
                tf.keras.layers.Dense(self.num_classes, name="fusion_logits"),
            ],
            name="fusion_mlp_head",
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
        baseline_cfg["model"].setdefault("classifier_dropout1", 0.35)
        baseline_cfg["model"].setdefault("ablation", "cnn_only")
        return baseline_cfg

    def freeze_rgb_branch(self) -> None:
        self.rgb_baseline.trainable = False
        for layer in self.rgb_baseline.layers:
            layer.trainable = False
        for variable in self.rgb_baseline.variables:
            try:
                variable._trainable = False
            except Exception:
                pass

    def _rgb_forward(self, image):
        @tf.custom_gradient
        def _custom_rgb_pass(x):
            endpoints = self.rgb_baseline.backbone(
                x,
                training=False,
                return_endpoints=True,
                stage3_adapter=getattr(self.rgb_baseline, "stage3_adapter", None),
            )
            stage4 = endpoints["stage4"]
            if getattr(self.rgb_baseline, "use_eca", False) and self.rgb_baseline.stage4_eca is not None:
                stage4 = self.rgb_baseline.stage4_eca(stage4, training=False)
                endpoints["stage4_eca"] = stage4
            feat = self.rgb_baseline.gap(stage4)
            dropped = self.rgb_baseline.head_dropout(feat, training=False)
            logits = self.rgb_baseline.classifier(dropped)
            
            feat = tf.cast(feat, tf.float32)
            logits = tf.cast(logits, tf.float32)

            def grad(d_feat, d_logits):
                return tf.zeros_like(x)

            return (feat, logits), grad

        rgb_feature, rgb_logits = _custom_rgb_pass(image)
        return rgb_feature, rgb_logits, {}

    def trainable_branch_variables(self) -> Tuple[list, list, list]:
        return (
            list(self.geometry_cnn.trainable_variables),
            list(self.geometry_head.trainable_variables),
            list(self.fusion_mlp.trainable_variables),
        )

    def expected_trainable_variable_keys(self) -> set:
        keys = set()
        for group in self.trainable_branch_variables():
            keys.update(_variable_key(v) for v in group)
        return keys

    def print_contract_summary(self) -> None:
        rgb_total = count_params(self.rgb_baseline.variables)
        rgb_trainable = count_params(self.rgb_baseline.trainable_variables)
        geom_vars, geom_head_vars, fusion_vars = self.trainable_branch_variables()
        print("[STAGE1_CONTRACT] ConvNeXt/RGB branch frozen:", flush=True)
        print(f"  rgb_total_params={rgb_total:,}", flush=True)
        print(f"  rgb_trainable_params={rgb_trainable:,}", flush=True)
        print("[STAGE1_CONTRACT] Trainable Stage 1 branches:", flush=True)
        print(f"  geometry_cnn_trainable_params={count_params(geom_vars):,}", flush=True)
        print(f"  geometry_head_trainable_params={count_params(geom_head_vars):,}", flush=True)
        print(f"  fusion_mlp_head_trainable_params={count_params(fusion_vars):,}", flush=True)
        print(f"  total_stage1_trainable_params={count_params(self.trainable_variables):,}", flush=True)

    def _log_shapes_once(self, image, geometry_maps, endpoints, rgb_feature, geometry_feature, rgb_logits, geometry_logits, fusion_logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[Stage1RGBSMIRK3DCNNLateFusion] Shape trace:", flush=True)
        print(f"  rgb_image: {image.shape}", flush=True)
        for key in ("stem", "stage1", "stage2", "stage3", "stage4"):
            if key in endpoints:
                print(f"  convnext_{key}: {endpoints[key].shape}", flush=True)
        print(f"  rgb_feature_1024: {rgb_feature.shape}", flush=True)
        print(f"  rgb_logits_frozen: {rgb_logits.shape}", flush=True)
        print(f"  depth_normal_4ch: {geometry_maps.shape}", flush=True)
        print(f"  geometry_feature_512: {geometry_feature.shape}", flush=True)
        print(f"  geometry_logits: {geometry_logits.shape}", flush=True)
        print(f"  fusion_concat_1536: {tf.concat([rgb_feature, geometry_feature], axis=-1).shape}", flush=True)
        print(f"  fusion_logits: {fusion_logits.shape}", flush=True)
        self.print_contract_summary()

    def call(self, inputs, training=False):
        image = inputs["image"]
        geometry_maps = inputs["geometry_maps"]
        rgb_feature, rgb_logits, endpoints = self._rgb_forward(image)
        geometry_feature = tf.cast(self.geometry_cnn(geometry_maps, training=training), tf.float32)
        geometry_logits = tf.cast(self.geometry_head(geometry_feature), tf.float32)
        fusion_input = tf.concat([rgb_feature, geometry_feature], axis=-1)
        fusion_logits = tf.cast(self.fusion_mlp(fusion_input, training=training), tf.float32)
        self._log_shapes_once(
            image,
            geometry_maps,
            endpoints,
            rgb_feature,
            geometry_feature,
            rgb_logits,
            geometry_logits,
            fusion_logits,
        )
        return {
            "logits": fusion_logits,
            "fusion_logits": fusion_logits,
            "rgb_logits": rgb_logits,
            "geometry_logits": geometry_logits,
            "rgb_feature": rgb_feature,
            "geometry_feature": geometry_feature,
        }


def resolve_latest_checkpoint(path_value: Optional[str]) -> Optional[str]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if path.is_dir():
        return tf.train.latest_checkpoint(str(path))
    if Path(str(path) + ".index").exists() or path.exists():
        return str(path)
    return None

