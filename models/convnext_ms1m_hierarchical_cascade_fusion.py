"""ConvNeXt-B MS1M Hierarchical Cascade Region Gated Fusion FER Model.

Architecture Contract:
  Input: Image [B, 112, 112, 3], MediaPipe soft region masks [B, 6, 112, 112]
  Backbone: ConvNeXt-Base MS1M returning Stage 3 [B, 14, 14, 512] and Stage 4 [B, 7, 7, 1024]
  Hierarchical Cascade Bridge (Stage 3 -> Stage 4 Guidance):
    S3 [B, 14, 14, 512] -> DepthwiseConv2D 3x3 stride 2 -> [B, 7, 7, 512]
                        -> Conv2D 1x1 512->1024 -> LayerNorm -> GELU -> [B, 7, 7, 1024]
    S4_refined = S4 + gamma_cascade * CascadeBridge(S3)
  Projections:
    S3: Conv2D 1x1 512->256 -> [B, 14, 14, 256] -> Flatten [B, 196, 256]
    S4_refined: Conv2D 1x1 1024->256 -> [B, 7, 7, 256] -> Flatten [B, 49, 256]
  Soft Region Masks:
    S3 soft masks [B, 6, 14, 14] -> Flatten [B, 6, 196]
    S4 soft masks [B, 6, 7, 7] -> Flatten [B, 6, 49]
  Shared Region Dictionary: 6 queries, embed_dim=256
  Masked Cross-Attention (float32 softmax):
    S3: Q [B, 6, 256], V [B, 196, 256], mask [B, 6, 196] -> R3 [B, 6, 256]
    S4_refined: Q [B, 6, 256], V [B, 49, 256], mask [B, 6, 49] -> R4 [B, 6, 256]
  Region-wise Gated Fusion:
    concat [R3, R4] -> [B, 6, 512]
    Gate MLP: Dense(256) -> GELU -> Dense(256) -> Sigmoid -> G [B, 6, 256]
    R = G * R3 + (1 - G) * R4 -> [B, 6, 256]
  Encoder: Exactly 1 TransformerEncoderBlock (embed_dim=256, num_heads=4, ffn=512, dropout=0.1) -> [B, 6, 256]
  Attention Pooling: Dense(1) -> Softmax(axis=1) -> Weighted Sum -> [B, 256]
  Classifier: LayerNorm -> Dropout(0.2) -> Dense(7) (Single Head)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
import tensorflow as tf

from .convnext_base_face_baseline import ConvNeXtBaseFRBackbone, ConvNeXtBaseFaceFERBaseline


def _norm(epsilon: float = 1e-5):
    return tf.keras.layers.LayerNormalization(epsilon=epsilon)


class MaskedCrossAttention(tf.keras.layers.Layer):
    """
    Masked Cross-Attention layer (embed_dim=256, num_heads=4).
    Calculates attention logits and softmax in float32 for numerical stability.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        mask_alpha: float = 0.3,
        mask_floor: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = float(self.head_dim ** -0.5)
        self.mask_alpha = float(mask_alpha)
        self.mask_floor = float(mask_floor)

        self.q_proj = tf.keras.layers.Dense(self.embed_dim)
        self.k_proj = tf.keras.layers.Dense(self.embed_dim)
        self.v_proj = tf.keras.layers.Dense(self.embed_dim)
        self.out_proj = tf.keras.layers.Dense(self.embed_dim)
        self.attn_drop = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])  # [B, H, N, D]

    def _merge_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])  # [B, N, H, D]
        return tf.reshape(x, [batch_size, seq_len, self.embed_dim])

    def call(self, queries, keys_values, mask=None, training=False):
        # queries: [B, 6, 256], keys_values: [B, N, 256] (N=196 or 49)
        # mask: [B, 6, N]
        q = self._split_heads(self.q_proj(queries))      # [B, H, 6, D]
        k = self._split_heads(self.k_proj(keys_values))  # [B, H, N, D]
        v = self._split_heads(self.v_proj(keys_values))  # [B, H, N, D]

        # Compute raw scores
        scores = tf.einsum("bhqd,bhkd->bhqk", q, k) * self.scale  # [B, H, 6, N]

        # Mixed precision safety: convert to float32 for log & softmax
        scores_f32 = tf.cast(scores, tf.float32)

        if mask is not None and self.mask_alpha > 0.0:
            mask_f32 = tf.cast(mask, tf.float32)
            mask_clipped = tf.clip_by_value(mask_f32, self.mask_floor, 1.0)
            mask_penalty = tf.math.log(mask_clipped + 1e-6) * self.mask_alpha  # [B, 6, N]
            scores_f32 = scores_f32 + tf.expand_dims(mask_penalty, axis=1)    # [B, H, 6, N]

        scores_clipped = tf.clip_by_value(scores_f32, -50.0, 50.0)
        attn_weights_f32 = tf.nn.softmax(scores_clipped, axis=-1)
        attn_weights_f32 = tf.where(tf.math.is_finite(attn_weights_f32), attn_weights_f32, tf.ones_like(attn_weights_f32) / tf.cast(tf.shape(attn_weights_f32)[-1], tf.float32))
        attn_weights = self.attn_drop(tf.cast(attn_weights_f32, scores.dtype), training=training)

        context = tf.einsum("bhqk,bhkd->bhqd", attn_weights, v)  # [B, H, 6, D]
        context = self._merge_heads(context)                    # [B, 6, 256]
        output = self.out_proj(context)                         # [B, 6, 256]
        return output, attn_weights_f32


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Single Transformer Encoder block (embed_dim=256, num_heads=4, FFN=512, dropout=0.1)."""
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = _norm(1e-5)

        self.ffn = tf.keras.Sequential([
            tf.keras.Input(shape=(None, embed_dim)),
            tf.keras.layers.Dense(ffn_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = _norm(1e-5)

    def call(self, x, training=False):
        # x: [B, 6, 256]
        attn_out = self.mha(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


class DynamicRegionUncertaintyRouter(tf.keras.layers.Layer):
    """
    Dynamic Region Uncertainty Weighting:
    Predicts dynamic per-region importance/uncertainty weights in range [0, 2]
    via a 2-layer MLP (Dense(128) -> GELU -> Dense(1) -> Sigmoid * 2.0).
    """
    def __init__(self, embed_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.router_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.gelu),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

    def call(self, x):
        # x: [B, 6, 256]
        weights = self.router_mlp(x) * 2.0  # [B, 6, 1]
        return weights


class RegionAttentionPooling(tf.keras.layers.Layer):
    """
    Attention pooling across 6 region tokens:
    Dense(1) per region -> Softmax over 6 regions (in float32) -> Weighted sum -> [B, 256].
    """
    def __init__(self, embed_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.score_proj = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, x):
        # x: [B, 6, 256]
        x_f32 = tf.cast(x, tf.float32)
        scores = self.score_proj(x_f32)  # [B, 6, 1] in float32
        scores_sq = tf.squeeze(scores, axis=-1)  # [B, 6]
        scores_clipped = tf.clip_by_value(scores_sq, -50.0, 50.0)
        weights_f32 = tf.nn.softmax(scores_clipped, axis=-1)  # [B, 6]
        weights_f32 = tf.where(tf.math.is_finite(weights_f32), weights_f32, tf.ones_like(weights_f32) / 6.0)
        weights = tf.cast(tf.expand_dims(weights_f32, axis=-1), x.dtype)  # [B, 6, 1]
        pooled = tf.reduce_sum(weights * x, axis=1)  # [B, 256]
        return pooled, weights_f32


class HierarchicalCascadeBridge(tf.keras.layers.Layer):
    """
    Stage 3 -> Stage 4 Hierarchical Feature Flow Cascade Bridge.
    Downsamples S3 [B, 14, 14, 512] to [B, 7, 7, 512] via DepthwiseConv 3x3 stride 2,
    then projects to 1024 channels + LayerNorm + GELU activation.
    """
    def __init__(self, in_channels: int = 512, out_channels: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same")
        self.pw_conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1, strides=1, padding="valid")
        self.norm = _norm(1e-5)

    def call(self, feat_s3):
        # feat_s3: [B, 14, 14, 512]
        x = self.dw_conv(feat_s3)  # [B, 7, 7, 512]
        x = self.pw_conv(x)        # [B, 7, 7, 1024]
        x = self.norm(x)
        x = tf.nn.gelu(x)
        return x


class ConvNeXtMS1MHierarchicalCascadeFusionFER(tf.keras.Model):
    """
    ConvNeXt-B MS1M Hierarchical Cascade Region Gated Fusion Model for FER2013.
    Injects Stage 3 fine-grained features directly into Stage 4 before Cross-Attention.
    """
    def __init__(self, cfg: Dict):
        model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg
        data_cfg = cfg.get("data", cfg) if isinstance(cfg, dict) else cfg
        super().__init__(name=model_cfg.get("name", "convnext_base_ms1m_hierarchical_cascade_fusion"))
        self.cfg = cfg
        self.num_classes = int(data_cfg.get("num_classes", 7))
        self.embed_dim = 256
        self.num_regions = 6
        self.num_heads = 4

        # 1. ConvNeXt-Base MS1M Backbone
        self.backbone = ConvNeXtBaseFRBackbone(
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
            layer_scale_init_value=float(model_cfg.get("layer_scale_init_value", 1e-6)),
            name="convnext_base_fr_backbone",
        )

        # 2. Hierarchical Cascade Bridge (S3 512 -> S4 1024)
        self.cascade_bridge = HierarchicalCascadeBridge(in_channels=512, out_channels=1024, name="cascade_bridge")
        self.gamma_cascade = self.add_weight(
            name="gamma_cascade",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )

        # 3. Multi-scale 1x1 Projections (512->256 for S3, 1024->256 for S4)
        self.proj_s3 = tf.keras.layers.Conv2D(self.embed_dim, 1, strides=1, padding="valid", name="proj_s3")
        self.proj_s4 = tf.keras.layers.Conv2D(self.embed_dim, 1, strides=1, padding="valid", name="proj_s4")

        # 4. Learnable Region Dictionary Queries [1, 6, 256]
        self.region_queries = self.add_weight(
            name="region_queries",
            shape=(1, self.num_regions, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        # 5. Masked Cross-Attention Branches (S3 & S4)
        self.cross_attn_s3 = MaskedCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=float(model_cfg.get("transformer_dropout", 0.1)),
            mask_alpha=float(model_cfg.get("mask_attention_alpha", 0.3)),
            mask_floor=float(model_cfg.get("mask_floor", 0.05)),
            name="cross_attn_s3",
        )
        self.cross_attn_s4 = MaskedCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=float(model_cfg.get("transformer_dropout", 0.1)),
            mask_alpha=float(model_cfg.get("mask_attention_alpha", 0.3)),
            mask_floor=float(model_cfg.get("mask_floor", 0.05)),
            name="cross_attn_s4",
        )

        # 6. Region-wise Gated Fusion MLP: Dense(256) -> GELU -> Dense(256) -> Sigmoid
        self.gate_mlp = tf.keras.Sequential([
            tf.keras.Input(shape=(None, self.embed_dim * 2)),
            tf.keras.layers.Dense(self.embed_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(self.embed_dim, activation="sigmoid"),
        ], name="gate_mlp")

        # 7. Post-fusion Transformer Encoder Blocks (Configurable depth, e.g. 1 or 2 layers)
        num_layers = int(model_cfg.get("num_encoder_layers", 1))
        self.encoder_blocks = [
            TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_dim=512,
                dropout=float(model_cfg.get("transformer_dropout", 0.1)),
                name=f"encoder_block_{i}",
            )
            for i in range(num_layers)
        ]

        # 8. Dynamic Region Uncertainty Router (Optional)
        self.use_dynamic_region_uncertainty = bool(model_cfg.get("use_dynamic_region_uncertainty", False))
        if self.use_dynamic_region_uncertainty:
            self.region_router = DynamicRegionUncertaintyRouter(embed_dim=self.embed_dim, name="region_router")

        # 9. Attention Pooling across 6 regions
        self.attn_pooling = RegionAttentionPooling(embed_dim=self.embed_dim, name="attn_pooling")

        # 9. Single Classifier Head: LayerNorm -> Dropout(0.35) -> Dense(7)
        self.norm = _norm(1e-5)
        self.head_dropout = tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout", 0.35)), name="head_dropout")
        self.classifier = tf.keras.layers.Dense(self.num_classes, kernel_initializer="he_normal", name="classifier")

        # 10. Optional Auxiliary Classifier Head on Stage 4 Backbone Features
        self.use_cnn_aux_loss = bool(model_cfg.get("use_cnn_aux_loss", False))
        if self.use_cnn_aux_loss:
            self.aux_classifier = tf.keras.layers.Dense(self.num_classes, kernel_initializer="he_normal", name="aux_classifier")

        # Load Pretrained Weights into Backbone
        self.pretrained_load_status = "not_requested"
        pretrained_path = model_cfg.get("convnext_base_pretrained_path") or model_cfg.get("pretrained_path")
        if pretrained_path:
            self.pretrained_load_status = self._load_backbone_pretrained(
                pretrained_path,
                require=bool(model_cfg.get("convnext_base_require_pretrained", False)),
            )

    def _load_backbone_pretrained(self, weight_path: str, require: bool = False) -> str:
        helper = ConvNeXtBaseFaceFERBaseline(self.cfg)
        helper.backbone = self.backbone
        status = helper._load_pytorch_pretrained(weight_path, require=require)
        return status

    def backbone_variables(self) -> List[tf.Variable]:
        vars_list = list(self.backbone.trainable_variables)
        if bool(self.cfg.get("model", {}).get("freeze_stage12", False)):
            filtered = [
                v for v in vars_list
                if not any(k in v.name.lower() for k in ["stem", "stage1", "stage2", "downsample_stage2"])
            ]
            return filtered
        return vars_list

    def head_variables(self) -> List[tf.Variable]:
        backbone_ids = {id(v) for v in self.backbone.trainable_variables}
        return [v for v in self.trainable_variables if id(v) not in backbone_ids]

    def _process_masks(self, mask: tf.Tensor, grid_size: int) -> tf.Tensor:
        mask_f32 = tf.cast(mask, tf.float32)
        batch_size = tf.shape(mask_f32)[0]

        if mask_f32.shape[-1] == self.num_regions:
            mask_hwc = mask_f32
        elif mask_f32.shape[1] == self.num_regions:
            mask_hwc = tf.transpose(mask_f32, [0, 2, 3, 1])
        else:
            mask_hwc = tf.cond(
                tf.equal(tf.shape(mask_f32)[-1], self.num_regions),
                lambda: mask_f32,
                lambda: tf.transpose(mask_f32, [0, 2, 3, 1]),
            )

        mask_resized = tf.image.resize(mask_hwc, [grid_size, grid_size], method="area")
        mask_chw = tf.transpose(mask_resized, [0, 3, 1, 2])
        mask_flat = tf.reshape(mask_chw, [batch_size, self.num_regions, grid_size * grid_size])
        return mask_flat

    def call(self, inputs, training=False) -> Dict[str, tf.Tensor]:
        if isinstance(inputs, dict):
            image = inputs["image"]
            mask = inputs.get("mask")
        else:
            image = inputs
            mask = None

        if mask is None:
            raise ValueError("Mask tensor is required for Hierarchical Cascade Fusion.")

        batch_size = tf.shape(image)[0]

        # 1. Extract Backbone Feature Maps
        endpoints = self.backbone(image, training=training, return_endpoints=True)
        feat_s3 = endpoints["stage3"]  # [B, 14, 14, 512]
        feat_s4 = endpoints["stage4"]  # [B, 7, 7, 1024]

        # 2. Hierarchical Cascade Guidance: Refine S4 using S3 features
        cascade_guidance = self.cascade_bridge(feat_s3)  # [B, 7, 7, 1024]
        feat_s4_refined = feat_s4 + self.gamma_cascade * cascade_guidance  # [B, 7, 7, 1024]

        # 3. Multi-scale 1x1 Projections to 256 channels
        proj_s3_map = self.proj_s3(feat_s3)          # [B, 14, 14, 256]
        proj_s4_map = self.proj_s4(feat_s4_refined)  # [B, 7, 7, 256]

        # Flatten Spatial Dimensions
        tokens_s3 = tf.reshape(proj_s3_map, [batch_size, 196, self.embed_dim])  # [B, 196, 256]
        tokens_s4 = tf.reshape(proj_s4_map, [batch_size, 49, self.embed_dim])   # [B, 49, 256]

        # 4. Soft Region Masks Processing
        mask_s3 = self._process_masks(mask, grid_size=14)  # [B, 6, 196]
        mask_s4 = self._process_masks(mask, grid_size=7)   # [B, 6, 49]

        # 5. Region Queries Tile
        queries = tf.tile(self.region_queries, [batch_size, 1, 1])  # [B, 6, 256]

        # 6. Masked Cross-Attention Branches
        r3, attn_weights_s3 = self.cross_attn_s3(queries, tokens_s3, mask=mask_s3, training=training)  # R3: [B, 6, 256]
        r4, attn_weights_s4 = self.cross_attn_s4(queries, tokens_s4, mask=mask_s4, training=training)  # R4: [B, 6, 256]

        # 7. Region-wise Gated Fusion
        concat_r = tf.concat([r3, r4], axis=-1)  # [B, 6, 512]
        gate = self.gate_mlp(concat_r, training=training)  # G: [B, 6, 256]
        fused_regions = gate * r3 + (1.0 - gate) * r4       # R: [B, 6, 256]

        # 8. Post-fusion Transformer Encoder Blocks
        encoded_regions = fused_regions
        for block in self.encoder_blocks:
            encoded_regions = block(encoded_regions, training=training)  # [B, 6, 256]

        region_uncertainty_weights = None
        if self.use_dynamic_region_uncertainty:
            region_uncertainty_weights = self.region_router(encoded_regions)  # [B, 6, 1]
            encoded_regions = encoded_regions * region_uncertainty_weights

        # 9. Region Attention Pooling
        pooled_feat, region_weights = self.attn_pooling(encoded_regions)  # [B, 256]

        # 10. Single Classifier Head
        x = self.norm(pooled_feat)
        x = self.head_dropout(x, training=training)
        logits = self.classifier(x)  # [B, 7]

        aux_logits = None
        if self.use_cnn_aux_loss:
            pooled_s4 = tf.reduce_mean(feat_s4_refined, axis=[1, 2])  # [B, 1024]
            aux_logits = tf.cast(self.aux_classifier(pooled_s4), tf.float32)

        tf.debugging.assert_all_finite(logits, "NaN/Inf detected in output logits")

        return {
            "logits": tf.cast(logits, tf.float32),
            "aux_logits": aux_logits,
            "S3": feat_s3,
            "S4": feat_s4,
            "S4_refined": feat_s4_refined,
            "gamma_cascade": self.gamma_cascade,
            "projected_S3": proj_s3_map,
            "projected_S4": proj_s4_map,
            "R3": r3,
            "R4": r4,
            "gate": gate,
            "fused_regions": fused_regions,
            "encoded_regions": encoded_regions,
            "region_uncertainty_weights": region_uncertainty_weights,
            "pooled_feature": pooled_feat,
            "region_weights": region_weights,
        }


def count_params(variables: Sequence[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in variables))
