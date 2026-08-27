"""
IR50 (InsightFace-style ResNet-50) Backbone for FER2013 Baseline.

Architecture based on ArcFace paper (Deng et al., 2019):
  - BN → Conv3x3(stride=1) → BN → PReLU  (stem, no max-pool)
  - 4 stages of IR Bottleneck blocks: [3, 4, 14, 3]
  - Each bottleneck: BN → Conv3x3 → BN → PReLU → Conv3x3 → BN + SE + residual
  - Output: 512-d embedding after BN → Dropout → Dense → BN

Pretrained weights:
  - Keras .h5 files from https://github.com/leondgarse/Keras_insightface
  - Recommended: glint360k_cosface_r50_fp16_0.1.h5 (Glint360k, CosFace)
  - Alternative: r50_magface_MS1MV2.h5 (MS1MV2, MagFace)
"""
from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

def _bn(name: Optional[str] = None):
    return tf.keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-5, name=name
    )


def _conv(filters: int, kernel_size: int, strides: int = 1, use_bias: bool = False, name: Optional[str] = None):
    return tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides,
        padding="same", use_bias=use_bias, name=name,
        kernel_initializer="he_normal",
    )


class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation="relu", use_bias=False)
        self.fc2 = tf.keras.layers.Dense(channels, activation="sigmoid", use_bias=False)

    def call(self, x, training=False):
        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        return x * s


class IRBottleneck(tf.keras.layers.Layer):
    """IR Bottleneck block (InsightFace-style)."""
    def __init__(self, filters: int, strides: int = 1, use_se: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.bn1 = _bn()
        self.conv1 = _conv(filters, 3, strides=1)
        self.bn2 = _bn()
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.conv2 = _conv(filters, 3, strides=strides)
        self.bn3 = _bn()
        self.se = SEBlock(filters) if use_se else None

        self.shortcut = None
        self.strides = strides
        self.filters = filters

    def build(self, input_shape):
        in_filters = int(input_shape[-1])
        if self.strides != 1 or in_filters != self.filters:
            self.shortcut = tf.keras.Sequential([
                _conv(self.filters, 1, strides=self.strides),
                _bn(),
            ])
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x if self.shortcut is None else self.shortcut(x, training=training)
        out = self.bn1(x, training=training)
        out = self.conv1(out)
        out = self.bn2(out, training=training)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out, training=training)
        if self.se is not None:
            out = self.se(out, training=training)
        return out + residual


def _make_stage(filters: int, num_blocks: int, strides: int = 2, use_se: bool = True, name_prefix: str = ""):
    blocks = []
    blocks.append(IRBottleneck(filters, strides=strides, use_se=use_se, name=f"{name_prefix}_0"))
    for i in range(1, num_blocks):
        blocks.append(IRBottleneck(filters, strides=1, use_se=use_se, name=f"{name_prefix}_{i}"))
    return blocks


# ---------------------------------------------------------------------------
#  IR50 Backbone
# ---------------------------------------------------------------------------

class IR50Backbone(tf.keras.Model):
    """
    InsightFace-style IR-50 backbone.
    Stage layout: [3, 4, 14, 3] with SE blocks.
    Input: 112x112x3 (face-recognition standard) or 224x224x3 (FER2013).
    Output: feature map from last stage (before head).
    """
    def __init__(self, input_size: int = 112, use_se: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size

        # Stem
        self.stem_bn = _bn(name="stem_bn")
        self.stem_conv = _conv(64, 3, strides=1, name="stem_conv")
        self.stem_bn2 = _bn(name="stem_bn2")
        self.stem_prelu = tf.keras.layers.PReLU(shared_axes=[1, 2], name="stem_prelu")

        # Stages: [3, 4, 14, 3]
        self.stage1 = _make_stage(64, 3, strides=2, use_se=use_se, name_prefix="stage1")
        self.stage2 = _make_stage(128, 4, strides=2, use_se=use_se, name_prefix="stage2")
        self.stage3 = _make_stage(256, 14, strides=2, use_se=use_se, name_prefix="stage3")
        self.stage4 = _make_stage(512, 3, strides=2, use_se=use_se, name_prefix="stage4")

    def call(self, x, training=False):
        # Stem
        x = self.stem_bn(x, training=training)
        x = self.stem_conv(x)
        x = self.stem_bn2(x, training=training)
        x = self.stem_prelu(x)

        # Stages
        for block in self.stage1:
            x = block(x, training=training)
        for block in self.stage2:
            x = block(x, training=training)
        for block in self.stage3:
            x = block(x, training=training)
        for block in self.stage4:
            x = block(x, training=training)
        return x


# ---------------------------------------------------------------------------
#  IR50 FER Baseline Model (compatible with train.py pipeline)
# ---------------------------------------------------------------------------

class IR50FERBaseline(tf.keras.Model):
    """
    Simple IR50 baseline for FER2013.

    Architecture:
        224x224x3 → IR50 Backbone → GlobalAveragePooling2D → Dropout → Dense(7)

    Output dict interface (compatible with supervised_mgr_loss):
        {
            "logits": [B, num_classes],
            "cnn_aux_logits": None,
            "ortho_loss": 0.0,
            "attn_scores": zeros,
            "attention_logits": None,
        }
    """
    def __init__(self, cfg: Dict):
        model_cfg = cfg["model"]
        super().__init__(name=model_cfg.get("name", "ir50_fer_baseline"))
        self.num_classes = int(cfg["data"]["num_classes"])
        self.ablation = model_cfg.get("ablation", "full")
        input_size = int(cfg["data"].get("image_size", 224))

        # IR50 Backbone
        self.backbone = IR50Backbone(
            input_size=input_size,
            use_se=bool(model_cfg.get("ir50_use_se", True)),
            name="ir50_backbone",
        )

        # Classification head
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.head_bn = _bn(name="head_bn")
        self.head_dropout = tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout1", 0.4)))
        self.classifier = tf.keras.layers.Dense(
            self.num_classes,
            kernel_initializer="he_normal",
            name="fer_classifier",
        )

        self.pretrained_load_status = "not_requested"

        # Load pretrained weights if specified
        pretrained_path = model_cfg.get("ir50_pretrained_path")
        if pretrained_path:
            self.pretrained_load_status = self._load_pretrained(
                pretrained_path,
                require=bool(model_cfg.get("ir50_require_pretrained", False)),
            )

    def _load_pretrained(self, weight_path: str, require: bool = False) -> str:
        """Load pretrained face-recognition weights into backbone."""
        from pathlib import Path

        resolved = Path(weight_path)
        if not resolved.is_absolute():
            resolved = Path(__file__).resolve().parents[1] / weight_path

        if not resolved.exists():
            message = f"[IR50] Pretrained weight file not found: {resolved}"
            if require:
                raise FileNotFoundError(message)
            print(f"[IR50] WARNING: {message}")
            print(f"[IR50] Training from scratch (random initialization).")
            return "missing"

        print(f"[IR50] Loading pretrained weights from: {resolved}")
        try:
            # Build model with dummy input to initialize all layers
            dummy = tf.zeros([1, 224, 224, 3])
            _ = self({"image": dummy}, training=False)

            # Load the pretrained h5 model to extract backbone weights
            pretrained = tf.keras.models.load_model(str(resolved), compile=False)
            pretrained_weights = {w.name: w for w in pretrained.weights}

            matched, skipped, missing = 0, 0, 0
            for var in self.backbone.weights:
                # Try exact name match first
                if var.name in pretrained_weights:
                    try:
                        var.assign(pretrained_weights[var.name])
                        matched += 1
                    except Exception as e:
                        print(f"[IR50] Shape mismatch for {var.name}: {var.shape} vs {pretrained_weights[var.name].shape}")
                        skipped += 1
                else:
                    missing += 1

            # If exact name matching failed, try deterministic by-order shape matching.
            if matched == 0 and missing > 0:
                print(f"[IR50] Exact name matching failed. Trying positional shape-matched transfer...")
                pretrained_vars = list(pretrained.weights)
                used_pretrained = set()
                matched = 0
                skipped = 0
                missing = 0
                for our_var in self.backbone.weights:
                    our_shape = tuple(our_var.shape)
                    found = False
                    for idx, pt_var in enumerate(pretrained_vars):
                        if idx in used_pretrained:
                            continue
                        if tuple(pt_var.shape) != our_shape:
                            continue
                        try:
                            our_var.assign(pt_var)
                            used_pretrained.add(idx)
                            matched += 1
                            found = True
                            break
                        except Exception:
                            skipped += 1
                    if not found:
                        missing += 1

            total = len(self.backbone.weights)
            print(f"[IR50] Pretrained weight loading complete:")
            print(f"[IR50]   Matched: {matched}/{total}")
            print(f"[IR50]   Skipped (shape mismatch): {skipped}")
            print(f"[IR50]   Missing in pretrained: {missing}")
            if matched <= 0:
                message = f"[IR50] No compatible pretrained weights were loaded from: {resolved}"
                if require:
                    raise RuntimeError(message)
                print(f"[IR50] WARNING: {message}")
                print(f"[IR50] Falling back to random initialization.")
                return "no_match"
            print(f"[IR50] PRETRAINED_LOAD_OK path={resolved} matched={matched}/{total}")

            del pretrained
            import gc
            gc.collect()
            return "loaded"

        except Exception as e:
            if require:
                raise
            print(f"[IR50] ERROR loading pretrained weights: {e}")
            print(f"[IR50] Falling back to random initialization.")
            return "error"

    def call(self, inputs, training=False, **kwargs):
        image = inputs["image"] if isinstance(inputs, dict) else inputs
        feat = self.backbone(image, training=training)
        pooled = self.gap(feat)
        pooled = self.head_bn(pooled, training=training)
        pooled = self.head_dropout(pooled, training=training)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "cnn_aux_logits": None,
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "attn_scores": tf.zeros([tf.shape(image)[0], 1, 1, 1], dtype=logits.dtype),
            "attention_logits": None,
        }
