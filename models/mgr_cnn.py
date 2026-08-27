from __future__ import annotations

from typing import Dict, Optional, Sequence

import tensorflow as tf


def _norm(epsilon: float = 1e-6):
    return tf.keras.layers.LayerNormalization(epsilon=epsilon)


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def call(self, x, training=False):
        if not training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (x.shape.rank - 1)
        mask = tf.floor(keep_prob + tf.random.uniform(shape, dtype=x.dtype))
        return tf.math.divide_no_nan(x, keep_prob) * mask


class ConvNeXtBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(7, padding="same")
        self.norm = _norm()
        self.pw1 = tf.keras.layers.Dense(4 * dim)
        self.act = tf.keras.layers.Activation(tf.nn.gelu)
        self.pw2 = tf.keras.layers.Dense(dim)
        self.drop_path = DropPath(drop_path)
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(1e-6),
            trainable=True,
        )

    def call(self, x, training=False):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x * tf.cast(self.gamma, x.dtype)
        return residual + self.drop_path(x, training=training)


class GroupNormalization(tf.keras.layers.Layer):
    """Native GroupNormalization for TensorFlow 2.x compatibility."""
    def __init__(self, groups: int = 32, epsilon: float = 1e-5):
        super().__init__()
        self.groups = int(groups)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=False):
        shape = tf.shape(x)
        rank = x.shape.rank
        if rank == 3:
            B, N, C = shape[0], shape[1], shape[2]
            G = min(self.groups, self.gamma.shape[0])
            x_reshaped = tf.reshape(x, [B, N, G, C // G])
            mean, variance = tf.nn.moments(x_reshaped, axes=[1, 3], keepdims=True)
            x_norm = (x_reshaped - mean) / tf.sqrt(variance + self.epsilon)
            x_norm = tf.reshape(x_norm, [B, N, C])
        else:
            B, H, W, C = shape[0], shape[1], shape[2], shape[3]
            G = min(self.groups, self.gamma.shape[0])
            x_reshaped = tf.reshape(x, [B, H, W, G, C // G])
            mean, variance = tf.nn.moments(x_reshaped, axes=[1, 2, 4], keepdims=True)
            x_norm = (x_reshaped - mean) / tf.sqrt(variance + self.epsilon)
            x_norm = tf.reshape(x_norm, [B, H, W, C])
        return tf.cast(self.gamma, x_norm.dtype) * x_norm + tf.cast(self.beta, x_norm.dtype)


class ELABlock(tf.keras.layers.Layer):
    """
    Efficient Local Attention (ELA) Block.
    Input tensor shape: [Batch, Height, Width, Channels]
    H-branch: AvgPool(W) -> Conv1D(kernel_size) -> GroupNorm -> Sigmoid -> Ah [B, H, 1, C]
    W-branch: AvgPool(H) -> Conv1D(kernel_size) -> GroupNorm -> Sigmoid -> Aw [B, 1, W, C]
    Output = X + X * Ah * Aw
    """
    def __init__(self, channels: int, kernel_size: int = 7, num_groups: int = 32):
        super().__init__()
        self.channels = int(channels)
        self.conv_h = tf.keras.layers.Conv1D(self.channels, kernel_size, padding="same", use_bias=False)
        self.conv_w = tf.keras.layers.Conv1D(self.channels, kernel_size, padding="same", use_bias=False)
        groups = min(num_groups, self.channels)
        self.gn_h = GroupNormalization(groups=groups)
        self.gn_w = GroupNormalization(groups=groups)
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, x, training=False):
        # x: [B, H, W, C]
        # H branch: pool over W (axis 2) -> [B, H, C]
        x_h = tf.reduce_mean(x, axis=2)
        x_h = self.conv_h(x_h)
        x_h = self.gn_h(x_h, training=training)
        a_h = tf.nn.sigmoid(x_h)
        a_h = tf.expand_dims(a_h, axis=2)  # [B, H, 1, C]

        # W branch: pool over H (axis 1) -> [B, W, C]
        x_w = tf.reduce_mean(x, axis=1)
        x_w = self.conv_w(x_w)
        x_w = self.gn_w(x_w, training=training)
        a_w = tf.nn.sigmoid(x_w)
        a_w = tf.expand_dims(a_w, axis=1)  # [B, 1, W, C]

        y = x * a_h * a_w
        return x + tf.cast(self.gamma, x.dtype) * y


class PixelUnshuffle(tf.keras.layers.Layer):
    """
    PixelUnshuffle (Space-to-Depth) layer.
    Transforms tensor of shape [B, H, W, C] to [B, H//r, W//r, C * (r**2)]
    """
    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        self.downscale_factor = int(downscale_factor)

    def call(self, x):
        return tf.nn.space_to_depth(x, block_size=self.downscale_factor)


class LocalExpert(tf.keras.layers.Layer):
    """
    Local Expert for capturing fine-grained facial expression details (eyes, mouth corners, wrinkles).
    Pipeline: DWConv 3x3 (groups=channels) -> GELU -> ELA Attention
    Input shape: [B, H, W, C]
    Output shape: [B, H, W, C]
    """
    def __init__(self, channels: int, ela_kernel_size: int = 7):
        super().__init__()
        self.channels = int(channels)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.ela = ELABlock(channels=self.channels, kernel_size=ela_kernel_size)

    def call(self, x, training=False):
        l = self.dwconv(x)
        l = tf.nn.gelu(l)
        f_l = self.ela(l, training=training)
        return f_l


class GlobalExpert(tf.keras.layers.Layer):
    """
    Global Expert for capturing multi-scale spatial receptive fields across facial regions.
    Pipeline: 3 Parallel Dilated DWConvs (dilation=1, 2, 3) -> Element-wise mean
    Input shape: [B, H, W, C]
    Output shape: [B, H, W, C]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.dwconv_d1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, dilation_rate=1, padding="same", use_bias=False
        )
        self.dwconv_d2 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, dilation_rate=2, padding="same", use_bias=False
        )
        self.dwconv_d3 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, dilation_rate=3, padding="same", use_bias=False
        )

    def call(self, x, training=False):
        g1 = self.dwconv_d1(x)
        g2 = self.dwconv_d2(x)
        g3 = self.dwconv_d3(x)
        f_g = (g1 + g2 + g3) / 3.0
        return f_g, g1, g2, g3


class LocalExpertBlock(tf.keras.layers.Layer):
    """
    Local Expert Block inserted after Stage 3 (14x14x384) for Ablation Study.
    Pipeline:
      DWConv 3x3 (stride=1, padding='same', use_bias=False)
      -> LayerNorm
      -> GELU
      -> ELA (ELABlock)
      -> Conv1x1 (384 -> 384, use_bias=False)
      -> Residual Add (Input + Output)
    Output shape: [B, 14, 14, 384]
    """
    def __init__(self, dim: int = 384, ela_kernel_size: int = 7):
        super().__init__()
        self.dim = int(dim)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ela = ELABlock(channels=self.dim, kernel_size=ela_kernel_size)
        self.pwconv = tf.keras.layers.Conv2D(
            filters=self.dim, kernel_size=1, strides=1, padding="same", use_bias=False
        )

    def call(self, x, training=False):
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = tf.nn.gelu(out)
        out = self.ela(out, training=training)
        out = self.pwconv(out)
        return residual + out


class LocalGlobalExpertBlock(tf.keras.layers.Layer):
    """
    Local-Global Expert Block (LGEB) inserted after Stage 3 (14x14x384).
    Local Expert captures micro-details; Global Expert captures macro spatial relations.
    Run 1 Fusion: F = F_L + F_G (optional residual: F = X + F_L + F_G if use_residual=True).
    Input shape: [B, H, W, C] (e.g. [B, 14, 14, 384])
    Output shape: [B, H, W, C]
    """
    def __init__(self, dim: int, use_residual: bool = False, debug: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.use_residual = bool(use_residual)
        self.debug = bool(debug)
        self.local_expert = LocalExpert(channels=self.dim)
        self.global_expert = GlobalExpert(channels=self.dim)
        self._printed_debug = False

    def call(self, x, training=False):
        f_l = self.local_expert(x, training=training)
        f_g, g1, g2, g3 = self.global_expert(x, training=training)

        if self.use_residual:
            f = x + f_l + f_g
        else:
            f = f_l + f_g

        if self.debug and not self._printed_debug:
            self._printed_debug = True
            print(f"[DEBUG LGEB] Stage3 X      : {x.shape}")
            print(f"[DEBUG LGEB] Local F_L     : {f_l.shape}")
            print(f"[DEBUG LGEB] Global G1     : {g1.shape}")
            print(f"[DEBUG LGEB] Global G2     : {g2.shape}")
            print(f"[DEBUG LGEB] Global G3     : {g3.shape}")
            print(f"[DEBUG LGEB] Global F_G    : {f_g.shape}")
            print(f"[DEBUG LGEB] LGEB output   : {f.shape}")

        return f


class ConvNeXtTinyBackbone(tf.keras.layers.Layer):
    def __init__(
        self,
        pretrained: bool = False,
        weights: Optional[str] = None,
        use_builtin_convnext: bool = False,
        builtin_include_preprocessing: bool = False,
        use_ela: bool = False,
        ela_kernel_size: int = 7,
        use_pixel_unshuffle: bool = False,
        use_lgeb: bool = False,
        lgeb_use_residual: bool = False,
        use_local_expert: bool = False,
    ):
        super().__init__()
        self.pretrained = bool(pretrained)
        self.backbone_weights = weights
        self.use_builtin_convnext = bool(use_builtin_convnext)
        self.builtin_include_preprocessing = bool(builtin_include_preprocessing)
        self.use_ela = bool(use_ela)
        self.use_pixel_unshuffle = bool(use_pixel_unshuffle)
        self.use_lgeb = bool(use_lgeb)
        self.lgeb_use_residual = bool(lgeb_use_residual)
        self.use_local_expert = bool(use_local_expert)
        if self.use_builtin_convnext:
            self.app = self._build_builtin_convnext()
            self.downsample_layers = []
            self.stages = []
            self.stage_blocks = []
            self.ela_stage2 = None
            self.ela_stage3 = None
            self.lgeb_stage3 = None
            self.local_expert_stage3 = None
            return

        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        self.downsample_layers = [
            tf.keras.Sequential([tf.keras.layers.Conv2D(dims[0], 4, strides=4, padding="same"), _norm()])
        ]
        for i in range(3):
            if self.use_pixel_unshuffle:
                self.downsample_layers.append(
                    tf.keras.Sequential([
                        PixelUnshuffle(2),
                        tf.keras.layers.Conv2D(dims[i + 1], 1, strides=1, padding="same", use_bias=False),
                        _norm(),
                    ])
                )
            else:
                self.downsample_layers.append(
                    tf.keras.Sequential([_norm(), tf.keras.layers.Conv2D(dims[i + 1], 2, strides=2, padding="same")])
                )
        rates = tf.linspace(0.0, 0.1, sum(depths))
        self.stages = []
        cursor = 0
        for i, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(dims[i], float(rates[cursor])))
                cursor += 1
            self.stages.append(blocks)
        self.stage_blocks = [blk for stage in self.stages for blk in stage]

        self.ela_stage2 = None
        self.ela_stage3 = None
        if self.use_ela:
            self.ela_stage2 = ELABlock(dims[2], kernel_size=ela_kernel_size)
            self.ela_stage3 = ELABlock(dims[3], kernel_size=ela_kernel_size)

        self.lgeb_stage3 = None
        if self.use_lgeb:
            self.lgeb_stage3 = LocalGlobalExpertBlock(dim=dims[2], use_residual=self.lgeb_use_residual)

        self.local_expert_stage3 = None
        if self.use_local_expert:
            self.local_expert_stage3 = LocalExpertBlock(dim=dims[2], ela_kernel_size=ela_kernel_size)

        if self.pretrained:
            self._load_imagenet_pretrained()

    def _resolve_builtin_weights(self):
        if self.backbone_weights not in (None, "", "none", "None", False):
            return self.backbone_weights
        return "imagenet" if self.pretrained else None

    def _build_builtin_convnext(self):
        convnext_tiny = getattr(tf.keras.applications, "ConvNeXtTiny", None)
        if convnext_tiny is None:
            raise RuntimeError("tf.keras.applications.ConvNeXtTiny is not available in this TensorFlow build.")
        weights = self._resolve_builtin_weights()
        kwargs = {
            "include_top": False,
            "weights": weights,
        }
        try:
            app = convnext_tiny(
                include_preprocessing=self.builtin_include_preprocessing,
                **kwargs,
            )
        except TypeError:
            app = convnext_tiny(**kwargs)
        print(
            "[INFO] Using built-in tf.keras.applications.ConvNeXtTiny "
            "(include_top=False, "
            f"include_preprocessing={self.builtin_include_preprocessing}, "
            f"weights={weights})"
        )
        return app

    def _load_imagenet_pretrained(self):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        convnext_tiny = getattr(tf.keras.applications, "ConvNeXtTiny", None)
        if convnext_tiny is None:
            return
        try:
            old_policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy("float32")
            try:
                with tf.init_scope():
                    app = convnext_tiny(include_top=False, include_preprocessing=False, weights="imagenet")
                    dummy = tf.cast(tf.zeros([1, 224, 224, 3]), tf.float32)
                    _ = self(dummy, training=False)
            finally:
                tf.keras.mixed_precision.set_global_policy(old_policy)

            stem_app = app.get_layer("convnext_tiny_stem")
            self.downsample_layers[0].layers[0].set_weights(stem_app.layers[0].get_weights())
            self.downsample_layers[0].layers[1].set_weights(stem_app.layers[1].get_weights())

            if not self.use_pixel_unshuffle:
                for idx in range(3):
                    ds_app = app.get_layer(f"convnext_tiny_downsampling_block_{idx}")
                    self.downsample_layers[idx + 1].layers[0].set_weights(ds_app.layers[0].get_weights())
                    self.downsample_layers[idx + 1].layers[1].set_weights(ds_app.layers[1].get_weights())
            else:
                print("[INFO] PixelUnshuffle enabled for downsampling: loaded 100% ImageNet weights for Stem and 18 Stage blocks, downsample Conv1x1 initialized freshly.")

            depths = [3, 3, 9, 3]
            for i, d in enumerate(depths):
                for j in range(d):
                    blk = self.stages[i][j]
                    dw_w, dw_b = app.get_layer(f"convnext_tiny_stage_{i}_block_{j}_depthwise_conv").get_weights()
                    blk.dwconv.set_weights([tf.reshape(dw_w, blk.dwconv.weights[0].shape), dw_b])
                    blk.norm.set_weights(app.get_layer(f"convnext_tiny_stage_{i}_block_{j}_layernorm").get_weights())
                    blk.pw1.set_weights(app.get_layer(f"convnext_tiny_stage_{i}_block_{j}_pointwise_conv_1").get_weights())
                    blk.pw2.set_weights(app.get_layer(f"convnext_tiny_stage_{i}_block_{j}_pointwise_conv_2").get_weights())
                    blk.gamma.assign(app.get_layer(f"convnext_tiny_stage_{i}_block_{j}_layer_scale").weights[0])

            print("[INFO] Successfully loaded and mapped 100% ImageNet pretrained weights into custom ConvNeXtTinyBackbone!")
        except Exception as exc:
            print(f"[WARNING] Failed to load ImageNet pretrained weights: {exc}")

    def call(self, x, training=False):
        if self.use_builtin_convnext:
            return self.app(x, training=training)
        for i, down in enumerate(self.downsample_layers):
            x = down(x, training=training)
            for block in self.stages[i]:
                x = block(x, training=training)
            if i == 2 and self.use_local_expert and self.local_expert_stage3 is not None:
                x = self.local_expert_stage3(x, training=training)
            if i == 2 and self.use_lgeb and self.lgeb_stage3 is not None:
                x = self.lgeb_stage3(x, training=training)
            if self.use_ela:
                if i == 2 and self.ela_stage2 is not None:
                    x = self.ela_stage2(x, training=training)
                elif i == 3 and self.ela_stage3 is not None:
                    x = self.ela_stage3(x, training=training)
        return x


class ChannelSE(tf.keras.layers.Layer):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 1)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(hidden, activation=tf.nn.gelu)
        self.fc2 = tf.keras.layers.Dense(int(channels), activation="sigmoid")

    def call(self, x, training=False):
        scale = self.pool(x)
        scale = self.fc1(scale, training=training)
        scale = self.fc2(scale, training=training)
        scale = tf.reshape(scale, [tf.shape(x)[0], 1, 1, tf.shape(x)[-1]])
        return x * scale


class RegionDictionary(tf.keras.layers.Layer):
    def __init__(self, num_regions: int, embed_dim: int):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            num_regions,
            embed_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        )

    def call(self, batch_size):
        ids = tf.range(self.embedding.input_dim, dtype=tf.int32)
        tokens = tf.expand_dims(self.embedding(ids), axis=0)
        return tf.tile(tokens, [batch_size, 1, 1])


class CrossAttentionWithMask(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, visual_dim: int, num_heads: int, dropout: float, mask_attention_alpha: float, mask_floor: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(embed_dim) // int(num_heads)
        self.scale = self.head_dim ** -0.5
        self.mask_attention_alpha = float(mask_attention_alpha)
        self.mask_floor = float(mask_floor)
        self.q_proj = tf.keras.layers.Dense(embed_dim)
        self.k_proj = tf.keras.layers.Dense(embed_dim)
        self.v_proj = tf.keras.layers.Dense(embed_dim)
        self.out_proj = tf.keras.layers.Dense(embed_dim)
        self.attn_drop = tf.keras.layers.Dropout(dropout)
        self.drop_path = DropPath(dropout if dropout > 0.0 else 0.0)
        self.norm1 = _norm(1e-5)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm2 = _norm(1e-5)

    def _split(self, x):
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        x = tf.reshape(x, [b, n, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.embed_dim])

    def call(self, region_tokens, visual_tokens, region_masks=None, training=False):
        q = self._split(self.q_proj(region_tokens))
        k = self._split(self.k_proj(visual_tokens))
        v = self._split(self.v_proj(visual_tokens))
        scores = tf.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if region_masks is not None and self.mask_attention_alpha > 0.0:
            mask = tf.clip_by_value(region_masks, self.mask_floor, 1.0)
            scores = scores + tf.cast(tf.expand_dims(tf.math.log(mask + 1e-6) * self.mask_attention_alpha, axis=1), scores.dtype)
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.attn_drop(attn, training=training)
        context = tf.einsum("bhqk,bhkd->bhqd", attn, v)
        context = self._merge(context)
        x = self.norm1(region_tokens + self.drop_path(self.out_proj(context), training=training))
        y = self.ffn(x, training=training)
        return self.norm2(x + self.drop_path(y, training=training)), attn


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.norm1 = _norm(1e-5)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm2 = _norm(1e-5)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.drop(attn, training=training))
        y = self.ffn(x, training=training)
        return self.norm2(x + self.drop(y, training=training))


class RelationTokenBuilder(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, relation_pairs: Sequence[Dict], dropout: float):
        super().__init__()
        self.relation_pairs = list(relation_pairs)
        self.fusions = [
            tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(epsilon=1e-5),
                tf.keras.layers.Dense(embed_dim),
                tf.keras.layers.Activation(tf.nn.gelu),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(embed_dim),
            ])
            for _ in self.relation_pairs
        ]

    def call(self, region_features, training=False):
        tokens = []
        for pair, fusion in zip(self.relation_pairs, self.fusions):
            left = tf.reduce_mean(tf.gather(region_features, pair["left"], axis=1), axis=1)
            right = tf.reduce_mean(tf.gather(region_features, pair["right"], axis=1), axis=1)
            tokens.append(tf.expand_dims(fusion(tf.concat([left, right], axis=-1), training=training), axis=1))
        return region_features if not tokens else tf.concat([region_features] + tokens, axis=1)


class MGRConvNeXtFER(tf.keras.Model):
    def __init__(self, cfg: Dict):
        super().__init__(name=cfg["model"]["name"])
        model_cfg = cfg["model"]
        self.cfg = cfg
        self.ablation = model_cfg.get("ablation", "full")
        self.num_classes = int(cfg["data"]["num_classes"])
        self.embed_dim = int(model_cfg["embed_dim"])
        self.visual_dim = int(model_cfg.get("visual_dim", 768))
        self.num_regions = int(model_cfg["num_regions"])
        self.region_pooling = model_cfg.get("region_pooling", "concat")
        self.mask_guided_attention = bool(model_cfg.get("mask_guided_attention", True))
        self.disable_region_branch_when_cnn_only = bool(model_cfg.get("disable_region_branch_when_cnn_only", False))
        self.use_global_visual_bias = bool(model_cfg.get("use_global_visual_bias", True))
        self.use_region_relation_tokens = bool(model_cfg.get("use_region_relation_tokens", True))
        self.use_cnn_aux_logits = bool(model_cfg.get("use_cnn_aux_logits", True))
        self.cnn_aux_pooling = model_cfg.get("cnn_aux_pooling", "avg").lower()
        if self.cnn_aux_pooling not in ("avg", "avgmax"):
            raise ValueError("model.cnn_aux_pooling must be one of: avg, avgmax")
        self.cnn_aux_logit_weight = float(model_cfg.get("cnn_aux_logit_weight", 0.8))
        self.attention_logit_weight = float(model_cfg.get("attention_logit_weight", 0.2))
        self.ortho_loss_type = model_cfg.get("ortho_loss_type", "squared_offdiag")
        use_lgeb_flag = bool(model_cfg.get("use_lgeb", False)) or (model_cfg.get("ablation") == "lgeb_stage3")
        use_local_expert_flag = bool(model_cfg.get("use_local_expert", False)) or (model_cfg.get("ablation") == "local_expert_stage3")
        self.backbone = ConvNeXtTinyBackbone(
            pretrained=bool(model_cfg.get("pretrained", False)),
            weights=model_cfg.get("weights"),
            use_builtin_convnext=bool(model_cfg.get("use_builtin_convnext", False)),
            builtin_include_preprocessing=bool(model_cfg.get("builtin_include_preprocessing", False)),
            use_ela=bool(model_cfg.get("use_ela", False)),
            ela_kernel_size=int(model_cfg.get("ela_kernel_size", 7)),
            use_pixel_unshuffle=bool(model_cfg.get("use_pixel_unshuffle", False)),
            use_lgeb=use_lgeb_flag,
            lgeb_use_residual=bool(model_cfg.get("lgeb_use_residual", False)),
            use_local_expert=use_local_expert_flag,
        )
        self.cnn_se = None
        if bool(model_cfg.get("use_cnn_se", False)):
            self.cnn_se = ChannelSE(
                int(model_cfg.get("cnn_se_channels", self.visual_dim)),
                int(model_cfg.get("cnn_se_reduction", 16)),
            )
        self.use_visual_pos_embed = bool(model_cfg.get("use_visual_pos_embed", True))
        if self.use_visual_pos_embed:
            _token_grid = int(model_cfg.get("token_grid_size", 7))
            self.visual_pos_embed = self.add_weight(
                name="visual_pos_embed",
                shape=(1, _token_grid * _token_grid, self.visual_dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                trainable=True,
            )
        self.region_dict = RegionDictionary(self.num_regions, self.embed_dim)
        self.cross_attention = CrossAttentionWithMask(
            self.embed_dim,
            self.visual_dim,
            int(model_cfg["num_heads"]),
            float(model_cfg.get("transformer_dropout", 0.25)),
            float(model_cfg.get("mask_attention_alpha", 0.3)),
            float(model_cfg.get("mask_floor", 0.05)),
        )
        relation_pairs = model_cfg.get("region_relation_pairs", [])
        self.relation_builder = RelationTokenBuilder(
            self.embed_dim,
            relation_pairs,
            float(model_cfg.get("region_relation_dropout", 0.1)),
        ) if self.use_region_relation_tokens else None
        relation_count = len(relation_pairs) if self.use_region_relation_tokens else 0
        self.region_token_count = self.num_regions + relation_count
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.region_token_count, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.global_proj = tf.keras.Sequential([_norm(1e-5), tf.keras.layers.Dense(self.embed_dim), tf.keras.layers.Dropout(float(model_cfg.get("transformer_dropout", 0.25)))])
        self.encoder = [
            TransformerEncoderBlock(self.embed_dim, int(model_cfg["num_heads"]), float(model_cfg.get("transformer_dropout", 0.25)))
            for _ in range(int(model_cfg["num_encoder_layers"]))
        ]
        self.classifier = tf.keras.Sequential([
            _norm(1e-5),
            tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout1", 0.45))),
            tf.keras.layers.Dense(int(model_cfg.get("classifier_hidden_dim", 512))),
            tf.keras.layers.Activation(tf.nn.gelu),
            tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout2", 0.35))),
            tf.keras.layers.Dense(self.num_classes),
        ])
        self.cnn_aux_classifier = None
        if self.use_cnn_aux_logits:
            self.cnn_aux_classifier = tf.keras.Sequential([
                _norm(1e-5),
                tf.keras.layers.Dropout(float(model_cfg.get("cnn_aux_dropout", 0.2))),
                tf.keras.layers.Dense(int(model_cfg.get("cnn_aux_hidden_dim", 768))),
                tf.keras.layers.Activation(tf.nn.gelu),
                tf.keras.layers.Dropout(float(model_cfg.get("cnn_aux_dropout", 0.2))),
                tf.keras.layers.Dense(self.num_classes),
            ])

    def _pool_regions(self, encoded):
        if self.region_pooling == "concat":
            return tf.reshape(encoded, [tf.shape(encoded)[0], self.region_token_count * self.embed_dim])
        return tf.reduce_mean(encoded, axis=1)

    def set_ablation(self, mode: str):
        print(f"[MODEL] Switched ablation mode from {self.ablation!r} to {mode!r}", flush=True)
        self.ablation = str(mode)

    def _apply_ablation(self, logits, cnn_aux_logits):
        if self.ablation == "cnn_only":
            return cnn_aux_logits
        if self.ablation == "region_only":
            return logits
        if self.ablation in ("full", "no_mask", "shuffled_mask") and cnn_aux_logits is not None:
            return self.attention_logit_weight * logits + self.cnn_aux_logit_weight * cnn_aux_logits
        return logits

    def _cnn_aux_features(self, global_avg, global_max):
        if self.cnn_aux_pooling == "avg":
            return global_avg
        return tf.concat([global_avg, global_max], axis=-1)

    def call(self, inputs, training=False, return_attn=False, return_region_weights=False):
        image = inputs["image"]
        mask = inputs.get("mask")
        feat_map = self.backbone(image, training=training)
        if self.cnn_se is not None:
            feat_map = self.cnn_se(feat_map, training=training)
        feat_map = tf.image.resize(feat_map, [7, 7])
        visual_tokens = tf.reshape(feat_map, [tf.shape(feat_map)[0], -1, self.visual_dim])
        if self.use_visual_pos_embed:
            visual_tokens = visual_tokens + tf.cast(self.visual_pos_embed, visual_tokens.dtype)
        global_avg = tf.reduce_mean(visual_tokens, axis=1)
        global_max = tf.reduce_max(visual_tokens, axis=1)
        skip_region_branch = True  # Permanently skip MGR region branch
        if skip_region_branch:
            if self.cnn_aux_classifier is None:
                raise ValueError("CNN-only without MGR requires model.use_cnn_aux_logits=true.")
            cnn_aux_logits = self.cnn_aux_classifier(
                self._cnn_aux_features(global_avg, global_max),
                training=training,
            )
            return {
                "logits": cnn_aux_logits,
                "attention_logits": None,
                "cnn_aux_logits": None,
                "ortho_loss": tf.constant(0.0, dtype=tf.float32),
                "attn_scores": tf.zeros([tf.shape(image)[0], 1, self.num_regions, 49], dtype=cnn_aux_logits.dtype),
            }
        region_tokens = self.region_dict(tf.shape(image)[0])
        attn_mask = None
        if self.mask_guided_attention and self.ablation not in ("no_mask", "cnn_only"):
            if mask is None:
                raise ValueError("Mask guidance is enabled but no mask tensor was provided.")
            attn_mask = tf.transpose(tf.reshape(mask, [tf.shape(mask)[0], -1, self.num_regions]), [0, 2, 1])
        region_features, attn_scores = self.cross_attention(region_tokens, visual_tokens, region_masks=attn_mask, training=training)
        if self.relation_builder is not None:
            region_features = self.relation_builder(region_features, training=training)
        region_features = region_features + tf.cast(self.pos_embed[:, : region_features.shape[1], :], region_features.dtype)
        if self.use_global_visual_bias:
            region_features = region_features + tf.cast(tf.expand_dims(self.global_proj(global_avg, training=training), axis=1), region_features.dtype)
        encoded = region_features
        for block in self.encoder:
            encoded = block(encoded, training=training)
        logits = self.classifier(self._pool_regions(encoded), training=training)
        cnn_aux_logits = None
        if self.cnn_aux_classifier is not None:
            cnn_aux_logits = self.cnn_aux_classifier(
                self._cnn_aux_features(global_avg, global_max),
                training=training,
            )
        fused_logits = self._apply_ablation(logits, cnn_aux_logits)
        attn_region = tf.reduce_mean(attn_scores, axis=1)
        attn_norm = tf.math.divide_no_nan(attn_region, tf.norm(attn_region, ord=2, axis=-1, keepdims=True))
        sim = tf.matmul(attn_norm, attn_norm, transpose_b=True)
        eye = tf.eye(tf.shape(sim)[-1], batch_shape=[tf.shape(sim)[0]], dtype=sim.dtype)
        off_diag_mask = 1.0 - eye
        off_diag_vals = sim * off_diag_mask
        off_diag_count = tf.maximum(tf.reduce_sum(off_diag_mask), 1.0)
        if self.ortho_loss_type == "squared_offdiag":
            ortho_loss = tf.reduce_sum(tf.square(off_diag_vals)) / off_diag_count
        else:
            ortho_loss = tf.reduce_sum(off_diag_vals) / off_diag_count
        if self.ablation == "cnn_only":
            ortho_loss = tf.constant(0.0, dtype=tf.float32)

        effective_cnn_aux = cnn_aux_logits if self.ablation != "region_only" else None

        fused_logits_f32 = tf.cast(fused_logits, tf.float32) if fused_logits is not None else None
        logits_f32 = tf.cast(logits, tf.float32) if logits is not None else None
        effective_cnn_aux_f32 = tf.cast(effective_cnn_aux, tf.float32) if effective_cnn_aux is not None else None

        outputs = {
            "logits": fused_logits_f32,
            "attention_logits": logits_f32,
            "cnn_aux_logits": effective_cnn_aux_f32,
            "ortho_loss": tf.cast(ortho_loss, tf.float32),
            "attn_scores": attn_scores,
        }
        if return_region_weights:
            outputs["region_weights"] = None
        return outputs
