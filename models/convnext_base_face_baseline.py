"""
ConvNeXt-Base MS1M/ArcFace baseline for FER2013.

FRBench-style shape contract used here:
    input 112x112x3
    stem Conv2D 2x2 stride 2 -> 56x56x128
    stages depths [3, 3, 27, 3], dims [128, 256, 512, 1024]
    downsample 2x2 stride 2 between stages -> final 7x7x1024
    FER head: GlobalAveragePooling2D -> Dropout -> Dense(7)

No Keras Applications backbone is used. PyTorch checkpoint tensors are converted
and assigned into the TensorFlow backbone layer by layer. The original face
recognition embedding/classification head is ignored because only backbone
variables are assignment targets.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .clip_text_encoder import get_or_compute_clip_text_prototypes


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=False):
        if not training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (x.shape.rank - 1)
        mask = tf.floor(keep_prob + tf.random.uniform(shape, dtype=x.dtype))
        return tf.math.divide_no_nan(x, keep_prob) * mask


class ConvNeXtBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(7, padding="same", name="dwconv")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.pwconv1 = tf.keras.layers.Dense(4 * self.dim, name="pwconv1")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="gelu")
        self.pwconv2 = tf.keras.layers.Dense(self.dim, name="pwconv2")
        self.drop_path = DropPath(drop_path, name="drop_path")
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.dim,),
            initializer=tf.keras.initializers.Constant(float(layer_scale_init_value)),
            trainable=True,
        )

    def call(self, x, training=False):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * tf.cast(self.gamma, x.dtype)
        return residual + self.drop_path(x, training=training)


class ConvNeXtDownsample(tf.keras.layers.Layer):
    def __init__(self, out_dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.conv = tf.keras.layers.Conv2D(out_dim, 2, strides=2, padding="valid", name="conv")

    def call(self, x, training=False):
        x = self.norm(x)
        return self.conv(x)


class LocalMultiScaleConvAdapter(tf.keras.layers.Layer):
    """
    Local Multi-Scale Conv Adapter after Stage 3 (B, 14, 14, 512).
    Branches: DepthwiseConv 3x3, 5x5, 7x7.
    Fusion: Learnable softmax weights -> Conv 1x1 (512 out) -> LayerNorm -> GELU.
    Residual connection: X_out = X + alpha * F, with alpha trainable and initialized to 0.0.
    """

    def __init__(self, channels: int = 512, name: str = "stage3_multiscale_adapter", **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = int(channels)
        self.dw3 = tf.keras.layers.DepthwiseConv2D(3, padding="same", name="dw_3x3")
        self.dw5 = tf.keras.layers.DepthwiseConv2D(5, padding="same", name="dw_5x5")
        self.dw7 = tf.keras.layers.DepthwiseConv2D(7, padding="same", name="dw_7x7")
        self.pwconv = tf.keras.layers.Conv2D(self.channels, 1, padding="valid", name="pwconv_1x1")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.act = tf.keras.layers.Activation(tf.nn.gelu, name="gelu")

        self.w_branches = self.add_weight(
            name="w_branches",
            shape=(3,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, x, training=False):
        b3 = self.dw3(x)
        b5 = self.dw5(x)
        b7 = self.dw7(x)

        weights = tf.nn.softmax(self.w_branches)
        fused = weights[0] * b3 + weights[1] * b5 + weights[2] * b7

        f = self.pwconv(fused)
        f = self.norm(f)
        f = self.act(f)

        return x + tf.cast(self.alpha, x.dtype) * f


class ECALayer(tf.keras.layers.Layer):
    """
    Efficient Channel Attention (ECA) Layer.
    Ref: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2020)

    Performs 1D convolution over the channel dimension after Global Average Pooling.
    Adaptive kernel size k = |(log2(C) + b) / gamma|_odd.
    """

    def __init__(
        self,
        channels: int,
        gamma: float = 2.0,
        b: float = 1.0,
        k_size: Optional[int] = None,
        name: str = "stage4_eca",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.channels = int(channels)
        self.gamma = float(gamma)
        self.b = float(b)
        if k_size is None:
            t = int(abs((np.log2(self.channels) + self.b) / self.gamma))
            k_size = t if t % 2 != 0 else t + 1
        self.k_size = int(k_size)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="gap")
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.k_size,
            padding="same",
            use_bias=False,
            name="conv1d",
        )
        self.sigmoid = tf.keras.layers.Activation("sigmoid", name="sigmoid")

    def call(self, x, training=False):
        # Input x shape: [B, H, W, C]
        y = self.gap(x)  # [B, 1, 1, C]
        batch_size = tf.shape(y)[0]
        y_1d = tf.reshape(y, [batch_size, self.channels, 1])  # [B, C, 1]
        y_conv = self.conv1d(y_1d)  # [B, C, 1]
        y_att = self.sigmoid(y_conv)  # [B, C, 1]
        y_att = tf.reshape(y_att, [batch_size, 1, 1, self.channels])  # [B, 1, 1, C]
        return x * tf.cast(y_att, x.dtype)


class ConvNeXtBaseFRBackbone(tf.keras.layers.Layer):
    """ConvNeXt-B variant used by the requested FRBench checkpoint shape."""

    def __init__(self, drop_path_rate: float = 0.1, layer_scale_init_value: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dims = [128, 256, 512, 1024]
        self.depths = [3, 3, 27, 3]
        self.stem_conv = tf.keras.layers.Conv2D(
            self.dims[0], 2, strides=2, padding="valid", name="stem_conv"
        )
        self.stem_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="stem_norm")
        self.downsample_layers = [
            ConvNeXtDownsample(self.dims[1], name="downsample_stage2"),
            ConvNeXtDownsample(self.dims[2], name="downsample_stage3"),
            ConvNeXtDownsample(self.dims[3], name="downsample_stage4"),
        ]
        rates = np.linspace(0.0, float(drop_path_rate), sum(self.depths)).tolist()
        cursor = 0
        self.stages = []
        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths), start=1):
            blocks = []
            for block_idx in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dim,
                        drop_path=rates[cursor],
                        layer_scale_init_value=layer_scale_init_value,
                        name=f"stage{stage_idx}_block{block_idx}",
                    )
                )
                cursor += 1
            self.stages.append(blocks)

    def call(self, x, training=False, return_endpoints: bool = False, stage3_adapter=None):
        endpoints = {}
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        endpoints["stem"] = x
        for block in self.stages[0]:
            x = block(x, training=training)
        endpoints["stage1"] = x

        for stage_offset in range(1, 4):
            x = self.downsample_layers[stage_offset - 1](x, training=training)
            for block in self.stages[stage_offset]:
                x = block(x, training=training)
            endpoints[f"stage{stage_offset + 1}"] = x
            if stage_offset == 2 and stage3_adapter is not None:
                x = stage3_adapter(x, training=training)
                endpoints["stage3_adapter"] = x
        if return_endpoints:
            return endpoints
        return endpoints["stage4"]


class ConvNeXtBaseFaceFERBaseline(tf.keras.Model):
    """Single-head ConvNeXt-Base face-pretrained FER baseline."""

    def __init__(self, cfg: Dict):
        model_cfg = cfg.get("model", cfg)
        data_cfg = cfg.get("data", cfg)
        super().__init__(name=model_cfg.get("name", "convnext_base_ms1m_arcface_baseline"))
        self.num_classes = int(data_cfg.get("num_classes", 7))
        self.ablation = model_cfg.get("ablation", "cnn_only")
        self.input_size = int(data_cfg.get("image_size", 112))
        self.channels = int(data_cfg.get("channels", 3))
        self._shape_logged = False

        self.backbone = ConvNeXtBaseFRBackbone(
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
            layer_scale_init_value=float(model_cfg.get("layer_scale_init_value", 1e-6)),
            name="convnext_base_fr_backbone",
        )
        self.use_multiscale_adapter = bool(
            model_cfg.get("use_multiscale_adapter", False)
            or model_cfg.get("ablation") == "multiscale_adapter"
        )
        if self.use_multiscale_adapter:
            self.stage3_adapter = LocalMultiScaleConvAdapter(channels=512, name="stage3_multiscale_adapter")
        else:
            self.stage3_adapter = None
        self.use_eca = bool(
            model_cfg.get("use_eca", False)
            or model_cfg.get("ablation") in ("eca", "eca_stage4")
        )
        if self.use_eca:
            self.stage4_eca = ECALayer(channels=self.backbone.dims[3], name="stage4_eca")
        else:
            self.stage4_eca = None
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="fer_gap")
        self.head_dropout = tf.keras.layers.Dropout(
            float(model_cfg.get("classifier_dropout1", 0.35)),
            name="fer_dropout",
        )
        self.classifier = tf.keras.layers.Dense(
            self.num_classes,
            kernel_initializer="he_normal",
            name="fer_classifier",
        )

        clip_sem_cfg = model_cfg.get("clip_semantic", {})
        if not isinstance(clip_sem_cfg, dict):
            clip_sem_cfg = {}

        self.use_semantic_branch = bool(
            clip_sem_cfg.get("enabled", False)
            or model_cfg.get("use_semantic_branch", False)
            or model_cfg.get("use_clip_semantic", False)
            or model_cfg.get("ablation") in ("semantic_clip", "clip_semantic", "semantic", "adaptive_clip", "adaptive_clip_confusion")
        )
        self.multi_prototype = bool(
            clip_sem_cfg.get("multi_prototype", model_cfg.get("multi_prototype", False))
            or model_cfg.get("ablation") in ("adaptive_clip", "adaptive_clip_confusion")
        )
        self.use_adaptive_granularity = bool(
            clip_sem_cfg.get("use_adaptive_granularity", False)
            or model_cfg.get("use_adaptive_granularity", False)
            or model_cfg.get("ablation") in ("adaptive_clip", "adaptive_clip_confusion")
        )
        self.use_hard_semantic_loss = bool(
            clip_sem_cfg.get("use_hard_semantic_loss", False)
            or model_cfg.get("use_hard_semantic_loss", False)
            or model_cfg.get("ablation") == "adaptive_clip_confusion"
        )
        self.prototype_aggregation = str(
            clip_sem_cfg.get("prototype_aggregation", model_cfg.get("prototype_aggregation", "logsumexp"))
        )
        self.prototype_temperature = float(
            clip_sem_cfg.get("prototype_temperature", model_cfg.get("prototype_temperature", 0.1))
        )
        self.lambda_sem = float(
            clip_sem_cfg.get("lambda_sem", model_cfg.get("lambda_sem", 0.1))
        )
        self.lambda_hard = float(
            clip_sem_cfg.get("lambda_hard", model_cfg.get("lambda_hard", 0.05))
        )
        self.hard_margin = float(
            clip_sem_cfg.get("hard_margin", model_cfg.get("hard_margin", 0.15))
        )
        self.semantic_logit_scale = float(
            clip_sem_cfg.get("semantic_logit_scale", model_cfg.get("semantic_logit_scale", 20.0))
        )

        default_hard_pairs = {
            0: [4, 2],    # angry -> sad, fear
            2: [4, 0],    # fear -> sad, angry
            4: [0, 2, 6], # sad -> angry, fear, neutral
            6: [4],       # neutral -> sad
        }
        hard_pairs_config = model_cfg.get("hard_pairs", clip_sem_cfg.get("hard_pairs", default_hard_pairs))
        hard_matrix_np = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)
        if hard_pairs_config:
            for c, hard_list in hard_pairs_config.items():
                c_int = int(c)
                if isinstance(hard_list, (list, tuple)):
                    for j in hard_list:
                        hard_matrix_np[c_int, int(j)] = 1.0
        self.hard_pairs_matrix = tf.constant(hard_matrix_np, dtype=tf.float32, name="hard_pairs_matrix")

        self.use_au_region_routed = bool(
            clip_sem_cfg.get("use_au_region_routed", False)
            or model_cfg.get("use_au_region_routed", False)
            or model_cfg.get("ablation") in ("au_region_routed", "adaptive_clip_confusion", "au_routed_clip")
        )

        if self.use_semantic_branch:
            embed_dim = int(clip_sem_cfg.get("clip_embedding_dim", model_cfg.get("clip_embedding_dim", 512)))
            self.visual_projector = tf.keras.Sequential([
                tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc1"),
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                tf.keras.layers.Activation("gelu", name="gelu"),
                tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout1", 0.35)), name="drop"),
                tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc2"),
            ], name="visual_semantic_projector")

            if self.use_au_region_routed:
                self.visual_projector_upper = tf.keras.Sequential([
                    tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc1"),
                    tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                    tf.keras.layers.Activation("gelu", name="gelu"),
                    tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout1", 0.35)), name="drop"),
                    tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc2"),
                ], name="visual_projector_upper")

                self.visual_projector_lower = tf.keras.Sequential([
                    tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc1"),
                    tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                    tf.keras.layers.Activation("gelu", name="gelu"),
                    tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout1", 0.35)), name="drop"),
                    tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc2"),
                ], name="visual_projector_lower")

                self.visual_projector_au = tf.keras.Sequential([
                    tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc1"),
                    tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln"),
                    tf.keras.layers.Activation("gelu", name="gelu"),
                    tf.keras.layers.Dropout(float(model_cfg.get("classifier_dropout1", 0.35)), name="drop"),
                    tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal", name="fc2"),
                ], name="visual_projector_au")
            else:
                self.visual_projector_upper = None
                self.visual_projector_lower = None
                self.visual_projector_au = None

            if self.use_adaptive_granularity:
                self.granularity_gate = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, kernel_initializer="he_normal", name="fc1"),
                    tf.keras.layers.Activation("gelu", name="gelu"),
                    tf.keras.layers.Dropout(0.1, name="drop"),
                    tf.keras.layers.Dense(5, kernel_initializer="he_normal", name="fc2"),
                    tf.keras.layers.Activation("softmax", name="softmax"),
                ], name="granularity_gate")
            else:
                self.granularity_gate = None

            clip_model_name = clip_sem_cfg.get("clip_model_name", model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32"))
            cache_path = clip_sem_cfg.get("clip_prototypes_path", model_cfg.get("clip_prototypes_path", None))
            text_proto_array = get_or_compute_clip_text_prototypes(
                model_name=clip_model_name,
                cache_path=cache_path,
                embedding_dim=embed_dim,
                multi_prototype=self.multi_prototype or self.use_adaptive_granularity,
            )
            self.text_prototypes = tf.constant(text_proto_array, dtype=tf.float32, name="frozen_clip_text_prototypes")
        else:
            self.visual_projector = None
            self.granularity_gate = None
            self.text_prototypes = None

        self.pretrained_load_status = "not_requested"
        pretrained_path = model_cfg.get("convnext_base_pretrained_path") or model_cfg.get("pretrained_path")
        if pretrained_path:
            self.pretrained_load_status = self._load_pytorch_pretrained(
                pretrained_path,
                require=bool(model_cfg.get("convnext_base_require_pretrained", False)),
            )

    def _resolve_weight_path(self, weight_path: str) -> Path:
        resolved = Path(weight_path)
        if resolved.exists():
            return resolved
        if not resolved.is_absolute():
            resolved_rel = Path(__file__).resolve().parents[1] / weight_path
            if resolved_rel.exists():
                return resolved_rel

        # Kaggle & environment auto-resolution search
        filename = Path(weight_path).name
        search_dirs = [
            Path("/kaggle/input/models/lhngphc/ms1m-pretrained/tensorflow2/default/1"),
            Path("/kaggle/input"),
            Path("/kaggle/working"),
            Path(__file__).resolve().parents[1] / "pretrained",
            Path(__file__).resolve().parents[1] / "data",
        ]
        for search_dir in search_dirs:
            if search_dir.exists():
                try:
                    for found_file in search_dir.rglob(filename):
                        if found_file.is_file():
                            print(f"[ConvNeXtBaseFace] Auto-resolved pretrained weight path on Kaggle: {found_file}", flush=True)
                            return found_file
                except Exception:
                    pass

        return Path(__file__).resolve().parents[1] / weight_path

    def _build_variables(self) -> None:
        dummy = tf.zeros([1, self.input_size, self.input_size, self.channels], dtype=tf.float32)
        _ = self({"image": dummy}, training=False)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint
        for key in ("state_dict", "model", "model_state_dict", "net", "backbone", "encoder"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    @staticmethod
    def _strip_prefixes(key: str) -> str:
        prefixes = (
            "module.",
            "model.",
            "backbone.",
            "visual_extractor.",
            "encoder.",
            "convnext.",
            "convnext_base.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    @classmethod
    def _normalize_state_dict(cls, state_dict) -> Dict[str, np.ndarray]:
        normalized = {}
        for raw_key, value in state_dict.items():
            key = cls._strip_prefixes(str(raw_key))
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)
            normalized[key] = value
        return normalized

    @staticmethod
    def _conv_kernel_pt_to_tf(value: np.ndarray) -> np.ndarray:
        return np.transpose(value, (2, 3, 1, 0))

    @staticmethod
    def _depthwise_kernel_pt_to_tf(value: np.ndarray) -> np.ndarray:
        if value.shape[1] != 1:
            raise ValueError(f"Expected PyTorch depthwise kernel [C,1,K,K], got {value.shape}")
        return np.transpose(value, (2, 3, 0, 1))

    @staticmethod
    def _linear_kernel_pt_to_tf(value: np.ndarray) -> np.ndarray:
        return np.transpose(value, (1, 0))

    @staticmethod
    def _first_available(state: Dict[str, np.ndarray], candidates: Iterable[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        for key in candidates:
            if key in state:
                return key, state[key]
        return None, None

    def _assign_from_candidates(
        self,
        state: Dict[str, np.ndarray],
        target: tf.Variable,
        candidates: Iterable[str],
        label: str,
        transform=None,
        used_keys: Optional[set] = None,
    ) -> Tuple[int, str]:
        key, value = self._first_available(state, candidates)
        if value is None:
            return 0, f"missing: {label} candidates={list(candidates)[:3]}"
        try:
            if transform is not None:
                value = transform(value)
            if tuple(value.shape) != tuple(target.shape):
                return 0, f"shape_mismatch: {label} key={key} pt={value.shape} tf={tuple(target.shape)}"
            target.assign(value)
            if used_keys is not None:
                used_keys.add(key)
            return 1, ""
        except Exception as exc:
            return 0, f"assign_error: {label} key={key} error={exc}"

    def load_pytorch_pretrained(self, weight_path: str, require: bool = False) -> str:
        return self._load_pytorch_pretrained(weight_path, require=require)

    def _load_pytorch_pretrained(self, weight_path: str, require: bool = False) -> str:
        resolved = self._resolve_weight_path(weight_path)
        npz_resolved = resolved.with_suffix(".npz")

        raw_state = None
        if npz_resolved.exists():
            print(f"[ConvNeXtBaseFace] Loading NumPy (.npz) pretrained checkpoint: {npz_resolved}", flush=True)
            loaded_npz = np.load(str(npz_resolved))
            raw_state = {k: loaded_npz[k] for k in loaded_npz.files}
        elif resolved.exists():
            try:
                import torch
                print(f"[ConvNeXtBaseFace] Loading PyTorch pretrained checkpoint: {resolved}", flush=True)
                checkpoint = torch.load(str(resolved), map_location="cpu")
                raw_state = self._extract_state_dict(checkpoint)
            except Exception as exc:
                message = f"PyTorch import/load failed ({exc}). Please install CPU PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) or provide a .npz checkpoint."
                if require:
                    raise RuntimeError(message) from exc
                print(f"[ConvNeXtBaseFace] WARNING: {message}", flush=True)
                return "torch_missing"
        else:
            message = f"[ConvNeXtBaseFace] Pretrained checkpoint not found: {resolved} (or {npz_resolved})"
            if require:
                raise FileNotFoundError(message)
            print(f"[ConvNeXtBaseFace] WARNING: {message}", flush=True)
            return "missing"

        self._build_variables()
        state = self._normalize_state_dict(raw_state)
        used_keys = set()
        matched = 0
        unmatched: List[str] = []

        unmatched_targets = []

        def assign(target, candidates, label, transform=None):
            nonlocal matched
            ok, msg = self._assign_from_candidates(
                state,
                target,
                candidates,
                label,
                transform=transform,
                used_keys=used_keys,
            )
            matched += ok
            if not ok:
                unmatched.append(msg)
                unmatched_targets.append((target, label, transform))

        # Stem: official ConvNeXt uses downsample_layers.0.{0,1} for stem.
        assign(
            self.backbone.stem_conv.kernel,
            ["downsample_layers.0.0.weight", "stem.0.weight", "stem_conv.weight", "patch_embed.proj.weight"],
            "stem_conv.kernel",
            self._conv_kernel_pt_to_tf,
        )
        assign(
            self.backbone.stem_conv.bias,
            ["downsample_layers.0.0.bias", "stem.0.bias", "stem_conv.bias", "patch_embed.proj.bias"],
            "stem_conv.bias",
        )
        assign(
            self.backbone.stem_norm.gamma,
            ["downsample_layers.0.1.weight", "stem.1.weight", "stem_norm.weight", "patch_embed.norm.weight"],
            "stem_norm.gamma",
        )
        assign(
            self.backbone.stem_norm.beta,
            ["downsample_layers.0.1.bias", "stem.1.bias", "stem_norm.bias", "patch_embed.norm.bias"],
            "stem_norm.beta",
        )

        for ds_idx, downsample in enumerate(self.backbone.downsample_layers, start=1):
            stage_num = ds_idx + 1
            assign(
                downsample.norm.gamma,
                [
                    f"downsample_layers.{ds_idx}.0.weight",
                    f"downsample_layers.{ds_idx}.norm.weight",
                    f"downsample_stage{stage_num}.norm.weight",
                    f"stages.{ds_idx - 1}.downsample.0.weight",
                    f"stages.{ds_idx - 1}.downsample.norm.weight",
                    f"stages.{ds_idx}.downsample.0.weight",
                    f"stages.{ds_idx}.downsample.norm.weight",
                    f"features.{stage_num * 2 - 1}.0.weight",
                ],
                f"downsample_stage{stage_num}.norm.gamma",
            )
            assign(
                downsample.norm.beta,
                [
                    f"downsample_layers.{ds_idx}.0.bias",
                    f"downsample_layers.{ds_idx}.norm.bias",
                    f"downsample_stage{stage_num}.norm.bias",
                    f"stages.{ds_idx - 1}.downsample.0.bias",
                    f"stages.{ds_idx - 1}.downsample.norm.bias",
                    f"stages.{ds_idx}.downsample.0.bias",
                    f"stages.{ds_idx}.downsample.norm.bias",
                    f"features.{stage_num * 2 - 1}.0.bias",
                ],
                f"downsample_stage{stage_num}.norm.beta",
            )
            assign(
                downsample.conv.kernel,
                [
                    f"downsample_layers.{ds_idx}.1.weight",
                    f"downsample_layers.{ds_idx}.conv.weight",
                    f"downsample_stage{stage_num}.conv.weight",
                    f"stages.{ds_idx - 1}.downsample.1.weight",
                    f"stages.{ds_idx - 1}.downsample.conv.weight",
                    f"stages.{ds_idx - 1}.downsample.reduction.weight",
                    f"stages.{ds_idx}.downsample.1.weight",
                    f"stages.{ds_idx}.downsample.conv.weight",
                    f"stages.{ds_idx}.downsample.reduction.weight",
                    f"features.{stage_num * 2 - 1}.1.weight",
                ],
                f"downsample_stage{stage_num}.conv.kernel",
                self._conv_kernel_pt_to_tf,
            )
            assign(
                downsample.conv.bias,
                [
                    f"downsample_layers.{ds_idx}.1.bias",
                    f"downsample_layers.{ds_idx}.conv.bias",
                    f"downsample_stage{stage_num}.conv.bias",
                    f"stages.{ds_idx - 1}.downsample.1.bias",
                    f"stages.{ds_idx - 1}.downsample.conv.bias",
                    f"stages.{ds_idx - 1}.downsample.reduction.bias",
                    f"stages.{ds_idx}.downsample.1.bias",
                    f"stages.{ds_idx}.downsample.conv.bias",
                    f"stages.{ds_idx}.downsample.reduction.bias",
                    f"features.{stage_num * 2 - 1}.1.bias",
                ],
                f"downsample_stage{stage_num}.conv.bias",
            )

        for stage_idx, blocks in enumerate(self.backbone.stages):
            for block_idx, block in enumerate(blocks):
                legacy_prefix = f"stages.{stage_idx}.{block_idx}"
                frbench_prefix = f"stages.{stage_idx}.blocks.{block_idx}"
                tv_prefix = f"features.{stage_idx * 2 + 1}.{block_idx}.block"
                label_prefix = f"stage{stage_idx + 1}.block{block_idx}"
                dw_kernel = getattr(block.dwconv, "depthwise_kernel", None)
                if dw_kernel is None:
                    dw_kernel = getattr(block.dwconv, "kernel", block.dwconv.weights[0])
                assign(
                    dw_kernel,
                    [
                        f"{frbench_prefix}.dwconv.weight",
                        f"{legacy_prefix}.dwconv.weight",
                        f"{frbench_prefix}.conv_dw.weight",
                        f"{tv_prefix}.0.weight",
                    ],
                    f"{label_prefix}.dwconv.depthwise_kernel",
                    self._depthwise_kernel_pt_to_tf,
                )
                assign(
                    block.dwconv.bias,
                    [
                        f"{frbench_prefix}.dwconv.bias",
                        f"{legacy_prefix}.dwconv.bias",
                        f"{frbench_prefix}.conv_dw.bias",
                        f"{tv_prefix}.0.bias",
                    ],
                    f"{label_prefix}.dwconv.bias",
                )
                assign(
                    block.norm.gamma,
                    [
                        f"{frbench_prefix}.norm.weight",
                        f"{legacy_prefix}.norm.weight",
                        f"{tv_prefix}.2.weight",
                    ],
                    f"{label_prefix}.norm.gamma",
                )
                assign(
                    block.norm.beta,
                    [
                        f"{frbench_prefix}.norm.bias",
                        f"{legacy_prefix}.norm.bias",
                        f"{tv_prefix}.2.bias",
                    ],
                    f"{label_prefix}.norm.beta",
                )
                assign(
                    block.pwconv1.kernel,
                    [
                        f"{frbench_prefix}.pwconv1.weight",
                        f"{legacy_prefix}.pwconv1.weight",
                        f"{frbench_prefix}.mlp.fc1.weight",
                        f"{tv_prefix}.3.weight",
                    ],
                    f"{label_prefix}.pwconv1.kernel",
                    self._linear_kernel_pt_to_tf,
                )
                assign(
                    block.pwconv1.bias,
                    [
                        f"{frbench_prefix}.pwconv1.bias",
                        f"{legacy_prefix}.pwconv1.bias",
                        f"{frbench_prefix}.mlp.fc1.bias",
                        f"{tv_prefix}.3.bias",
                    ],
                    f"{label_prefix}.pwconv1.bias",
                )
                assign(
                    block.pwconv2.kernel,
                    [
                        f"{frbench_prefix}.pwconv2.weight",
                        f"{legacy_prefix}.pwconv2.weight",
                        f"{frbench_prefix}.mlp.fc2.weight",
                        f"{tv_prefix}.5.weight",
                    ],
                    f"{label_prefix}.pwconv2.kernel",
                    self._linear_kernel_pt_to_tf,
                )
                assign(
                    block.pwconv2.bias,
                    [
                        f"{frbench_prefix}.pwconv2.bias",
                        f"{legacy_prefix}.pwconv2.bias",
                        f"{frbench_prefix}.mlp.fc2.bias",
                        f"{tv_prefix}.5.bias",
                    ],
                    f"{label_prefix}.pwconv2.bias",
                )
                assign(
                    block.gamma,
                    [
                        f"{frbench_prefix}.gamma",
                        f"{legacy_prefix}.gamma",
                        f"{legacy_prefix}.layer_scale",
                        f"{frbench_prefix}.ls.gamma",
                        f"{tv_prefix}.layer_scale",
                    ],
                    f"{label_prefix}.gamma",
                )

        # Fallback shape matching for any unassigned backbone variable
        if len(unmatched_targets) > 0:
            print(f"[ConvNeXtBaseFace] Attempting shape-based fallback matching for {len(unmatched_targets)} unmatched targets...", flush=True)
            unused_pt_keys = [k for k in state.keys() if k not in used_keys and not k.startswith("output_layer.")]
            
            still_unmatched = []
            for target_var, label, transform in unmatched_targets:
                target_shape = tuple(target_var.shape)
                matched_key = None
                matched_val = None

                for pt_k in list(unused_pt_keys):
                    pt_v = state[pt_k]
                    # Try direct shape match
                    if pt_v.shape == target_shape:
                        matched_key, matched_val = pt_k, pt_v
                        break
                    # Try 4D conv transpose
                    elif pt_v.ndim == 4 and self._conv_kernel_pt_to_tf(pt_v).shape == target_shape:
                        matched_key, matched_val = pt_k, self._conv_kernel_pt_to_tf(pt_v)
                        break
                    # Try depthwise conv transpose
                    elif pt_v.ndim == 4 and pt_v.shape[1] == 1 and self._depthwise_kernel_pt_to_tf(pt_v).shape == target_shape:
                        matched_key, matched_val = pt_k, self._depthwise_kernel_pt_to_tf(pt_v)
                        break
                    # Try 2D linear transpose
                    elif pt_v.ndim == 2 and self._linear_kernel_pt_to_tf(pt_v).shape == target_shape:
                        matched_key, matched_val = pt_k, self._linear_kernel_pt_to_tf(pt_v)
                        break

                if matched_key is not None:
                    target_var.assign(matched_val)
                    used_keys.add(matched_key)
                    unused_pt_keys.remove(matched_key)
                    matched += 1
                    print(f"[ConvNeXtBaseFace] Fallback matched {label} -> {matched_key}", flush=True)
                else:
                    still_unmatched.append(f"missing: {label}")

            unmatched = still_unmatched



        total_targets = len(self.backbone.weights)
        unused_keys = sorted(set(state.keys()) - used_keys)
        allowed_unused_prefixes = ("output_layer.",)
        allowed_unused = [key for key in unused_keys if key.startswith(allowed_unused_prefixes)]
        unexpected_unused = [key for key in unused_keys if not key.startswith(allowed_unused_prefixes)]
        fully_matched = matched == total_targets and not unmatched
        print("[ConvNeXtBaseFace] PyTorch -> TensorFlow weight loading complete:", flush=True)
        print(f"[ConvNeXtBaseFace]   matched target tensors: {matched}/{total_targets}", flush=True)
        print(f"[ConvNeXtBaseFace]   unmatched target tensors: {len(unmatched)}", flush=True)
        print(f"[ConvNeXtBaseFace]   unused checkpoint tensors: {len(unused_keys)}", flush=True)
        print(f"[ConvNeXtBaseFace]   allowed unused face-head tensors: {len(allowed_unused)}", flush=True)
        print(f"[ConvNeXtBaseFace]   unexpected unused checkpoint tensors: {len(unexpected_unused)}", flush=True)
        if unmatched:
            print("[ConvNeXtBaseFace]   first unmatched targets:", flush=True)
            for item in unmatched[:80]:
                print(f"[ConvNeXtBaseFace]     {item}", flush=True)
        if allowed_unused:
            print("[ConvNeXtBaseFace]   first allowed unused face-head keys:", flush=True)
            for key in allowed_unused[:40]:
                print(f"[ConvNeXtBaseFace]     {key}", flush=True)
        if unexpected_unused:
            print("[ConvNeXtBaseFace]   first unexpected unused checkpoint keys:", flush=True)
            for key in unexpected_unused[:80]:
                print(f"[ConvNeXtBaseFace]     {key}", flush=True)

        if not fully_matched:
            message = (
                f"[ConvNeXtBaseFace] Backbone pretrained load incomplete: "
                f"matched={matched}/{total_targets}, unmatched={len(unmatched)}. "
                "PRETRAINED_LOAD_OK is only emitted for a full backbone match."
            )
            if require:
                raise RuntimeError(message)
            print(f"[ConvNeXtBaseFace] WARNING: {message}", flush=True)
            return "partial" if matched > 0 else "no_match"
        print(f"[ConvNeXtBaseFace] PRETRAINED_LOAD_OK path={resolved} matched={matched}/{total_targets}", flush=True)
        return "loaded"

    def _log_shapes_once(self, image, endpoints, pooled, dropped, logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[ConvNeXtBaseFace] Shape trace:", flush=True)
        print(f"[ConvNeXtBaseFace]   input: {image.shape}", flush=True)
        for key in ("stem", "stage1", "stage2", "stage3"):
            if key in endpoints:
                print(f"[ConvNeXtBaseFace]   {key}: {endpoints[key].shape}", flush=True)
        if "stage3_adapter" in endpoints:
            print(f"[ConvNeXtBaseFace]   stage3_adapter: {endpoints['stage3_adapter'].shape}", flush=True)
        if "stage4" in endpoints:
            print(f"[ConvNeXtBaseFace]   stage4 (before ECA): {endpoints['stage4'].shape}", flush=True)
        if "stage4_eca" in endpoints:
            print(f"[ConvNeXtBaseFace]   stage4_eca (after ECA): {endpoints['stage4_eca'].shape}", flush=True)
        if "visual_projector" in endpoints:
            print(f"[ConvNeXtBaseFace]   visual_projector: {endpoints['visual_projector'].shape}", flush=True)
            print(f"[ConvNeXtBaseFace]   text_prototypes (frozen): {self.text_prototypes.shape}", flush=True)
            print(f"[ConvNeXtBaseFace]   semantic_logits: {endpoints['semantic_logits'].shape}", flush=True)

        print(f"[ConvNeXtBaseFace]   gap: {pooled.shape}", flush=True)
        print(f"[ConvNeXtBaseFace]   dropout: {dropped.shape}", flush=True)
        print(f"[ConvNeXtBaseFace]   logits: {logits.shape}", flush=True)

        if self.use_multiscale_adapter and self.stage3_adapter is not None:
            adapter_params = int(np.sum([np.prod(v.shape) for v in self.stage3_adapter.trainable_variables]))
            backbone_params = int(np.sum([np.prod(v.shape) for v in self.backbone.trainable_variables]))
            total_params = int(np.sum([np.prod(v.shape) for v in self.trainable_variables]))
            print(f"[ConvNeXtBaseFace] Local Multi-Scale Adapter Enabled:", flush=True)
            print(f"[ConvNeXtBaseFace]   Adapter trainable params: {adapter_params:,}", flush=True)
            print(f"[ConvNeXtBaseFace]   Backbone trainable params: {backbone_params:,}", flush=True)
            print(f"[ConvNeXtBaseFace]   Total trainable params: {total_params:,}", flush=True)

        if self.use_eca and self.stage4_eca is not None:
            eca_params = int(np.sum([np.prod(v.shape) for v in self.stage4_eca.trainable_variables]))
            backbone_params = int(np.sum([np.prod(v.shape) for v in self.backbone.trainable_variables]))
            total_params = int(np.sum([np.prod(v.shape) for v in self.trainable_variables]))
            print(f"[ConvNeXtBaseFace] ECA Stage4 Attention Enabled:", flush=True)
            print(f"[ConvNeXtBaseFace]   ECA kernel size: {self.stage4_eca.k_size}", flush=True)
            print(f"[ConvNeXtBaseFace]   ECA trainable params: {eca_params:,}", flush=True)

        if self.use_semantic_branch and self.visual_projector is not None:
            proj_params = int(np.sum([np.prod(v.shape) for v in self.visual_projector.trainable_variables]))
            total_params = int(np.sum([np.prod(v.shape) for v in self.trainable_variables]))
            if self.multi_prototype:
                v_shape = endpoints.get("visual_projector", pooled).shape
                r_shape = endpoints.get("raw_semantic_similarity", tf.zeros([1, 7, 5])).shape
                s_shape = endpoints.get("semantic_logits", tf.zeros([1, 7])).shape
                print("MULTI_PROTOTYPE_CLIP_ENABLED", flush=True)
                print(f"Text prototypes shape: {tuple(self.text_prototypes.shape)}", flush=True)
                print(f"Visual embedding shape: ({v_shape[0]}, {v_shape[1]})", flush=True)
                print(f"Raw semantic similarity: ({r_shape[0]}, {r_shape[1]}, {r_shape[2]})", flush=True)
                print(f"Aggregated semantic logits: ({s_shape[0]}, {s_shape[1]})", flush=True)
                print(f"Aggregation: {self.prototype_aggregation}", flush=True)
                print(f"Prototype temperature: {self.prototype_temperature}", flush=True)
                print(f"Semantic logit scale: {self.semantic_logit_scale}", flush=True)
                print(f"lambda_sem: {self.lambda_sem}", flush=True)
            else:
                print(f"[ConvNeXtBaseFace] CLIP Text Semantic Alignment Branch Enabled:", flush=True)
                print(f"[ConvNeXtBaseFace]   Visual Projector trainable params: {proj_params:,}", flush=True)
                print(f"[ConvNeXtBaseFace]   Frozen CLIP Text Prototypes params: 0 (Non-trainable {self.text_prototypes.shape})", flush=True)
                print(f"[ConvNeXtBaseFace]   lambda_sem: {self.lambda_sem}", flush=True)
                print(f"[ConvNeXtBaseFace]   semantic_logit_scale: {self.semantic_logit_scale}", flush=True)
            print(f"[ConvNeXtBaseFace]   Total trainable params: {total_params:,}", flush=True)

    def call(self, inputs, training=False, **kwargs):
        image = inputs["image"] if isinstance(inputs, dict) else inputs
        endpoints = self.backbone(
            image, training=training, return_endpoints=True, stage3_adapter=self.stage3_adapter
        )
        feat = endpoints["stage4"]
        if self.use_eca and self.stage4_eca is not None:
            feat = self.stage4_eca(feat, training=training)
            endpoints["stage4_eca"] = feat
        pooled = self.gap(feat)
        dropped = self.head_dropout(pooled, training=training)
        logits = self.classifier(dropped)

        semantic_logits = None
        agg_sim = None
        granularity_weights = None

        if self.use_semantic_branch and self.visual_projector is not None:
            text_protos = tf.cast(self.text_prototypes, dtype=tf.float32)
            t_norm = tf.math.l2_normalize(text_protos, axis=-1) # [7, 5, dim]

            if self.use_au_region_routed and self.visual_projector_upper is not None:
                # Extract Stage 3 spatial feature maps [B, 14, 14, 512]
                stage3_feat = endpoints.get("stage3_adapter", endpoints.get("stage3"))
                z_upper = tf.reduce_mean(stage3_feat[:, 0:8, :, :], axis=[1, 2])
                z_lower = tf.reduce_mean(stage3_feat[:, 5:14, :, :], axis=[1, 2])
                z_au = tf.reduce_mean(stage3_feat[:, 3:11, :, :], axis=[1, 2])

                v_global_proj = self.visual_projector(pooled, training=training)
                v_upper_proj = self.visual_projector_upper(z_upper, training=training)
                v_lower_proj = self.visual_projector_lower(z_lower, training=training)
                v_au_proj = self.visual_projector_au(z_au, training=training)

                v_global_norm = tf.math.l2_normalize(v_global_proj, axis=-1, epsilon=1e-5)
                v_upper_norm = tf.math.l2_normalize(v_upper_proj, axis=-1, epsilon=1e-5)
                v_lower_norm = tf.math.l2_normalize(v_lower_proj, axis=-1, epsilon=1e-5)
                v_au_norm = tf.math.l2_normalize(v_au_proj, axis=-1, epsilon=1e-5)

                endpoints["visual_projector"] = v_global_proj
                endpoints["visual_projector_upper"] = v_upper_proj
                endpoints["visual_projector_lower"] = v_lower_proj
                endpoints["visual_projector_au"] = v_au_proj

                # Match dtype for mixed precision (float16 vs float32) compatibility
                t_norm = tf.cast(t_norm, dtype=v_global_norm.dtype)

                # Spatial-Semantic Routing to 5 Prototypes:
                # P0 (Emotion): Global -> t_norm[:,0,:]
                # P1 (AU-level): AU-region -> t_norm[:,1,:]
                # P2 (Upper-face): Upper-region -> t_norm[:,2,:]
                # P3 (Lower-face): Lower-region -> t_norm[:,3,:]
                # P4 (Combined): Global -> t_norm[:,4,:]
                s0 = tf.einsum("bd,cd->bc", v_global_norm, t_norm[:, 0, :])
                s1 = tf.einsum("bd,cd->bc", v_au_norm, t_norm[:, 1, :])
                s2 = tf.einsum("bd,cd->bc", v_upper_norm, t_norm[:, 2, :])
                s3 = tf.einsum("bd,cd->bc", v_lower_norm, t_norm[:, 3, :])
                s4 = tf.einsum("bd,cd->bc", v_global_norm, t_norm[:, 4, :])

                raw_sim = tf.stack([s0, s1, s2, s3, s4], axis=-1)  # [B, 7, 5]
            else:
                v_proj = self.visual_projector(pooled, training=training)
                v_norm = tf.math.l2_normalize(v_proj, axis=-1, epsilon=1e-5)
                t_norm = tf.cast(t_norm, dtype=v_norm.dtype)
                endpoints["visual_projector"] = v_proj
                if len(t_norm.shape) == 3 or (hasattr(t_norm.shape, "rank") and t_norm.shape.rank == 3):
                    raw_sim = tf.einsum("bd,ckd->bck", v_norm, t_norm)
                else:
                    raw_sim = tf.einsum("bd,cd->bc", v_norm, t_norm)

            if self.multi_prototype or self.use_adaptive_granularity:
                endpoints["raw_semantic_similarity"] = raw_sim
                raw_sim_f32 = tf.cast(raw_sim, tf.float32)

                if self.use_adaptive_granularity and self.granularity_gate is not None:
                    # Adaptive Multi-Granularity Weighting: Sample-adaptive gate [B, 5]
                    granularity_weights = self.granularity_gate(pooled, training=training)  # [B, 5]
                    granularity_weights_f32 = tf.cast(granularity_weights, tf.float32)
                    gw_exp = tf.expand_dims(granularity_weights_f32, axis=1)  # [B, 1, 5]
                    agg_sim = tf.reduce_sum(gw_exp * raw_sim_f32, axis=-1)  # [B, 7]
                    endpoints["granularity_weights"] = granularity_weights
                else:
                    if self.prototype_aggregation == "logsumexp":
                        tau = tf.constant(self.prototype_temperature, dtype=tf.float32)
                        K = tf.constant(float(raw_sim.shape[-1]), dtype=tf.float32)
                        lse = tf.reduce_logsumexp(raw_sim_f32 / tau, axis=-1)
                        agg_sim = tau * (lse - tf.math.log(K))
                    elif self.prototype_aggregation == "mean":
                        agg_sim = tf.reduce_mean(raw_sim_f32, axis=-1)
                    elif self.prototype_aggregation == "max":
                        agg_sim = tf.reduce_max(raw_sim_f32, axis=-1)
                    else:
                        raise ValueError(f"Unsupported prototype_aggregation: {self.prototype_aggregation}")
                semantic_logits = agg_sim * tf.cast(self.semantic_logit_scale, tf.float32)
            else:
                agg_sim = tf.cast(raw_sim, tf.float32)
                semantic_logits = agg_sim * tf.cast(self.semantic_logit_scale, tf.float32)
            
            agg_sim = tf.where(tf.math.is_finite(agg_sim), agg_sim, tf.zeros_like(agg_sim))
            semantic_logits = tf.where(tf.math.is_finite(semantic_logits), semantic_logits, tf.zeros_like(semantic_logits))
            endpoints["semantic_logits"] = semantic_logits

        self._log_shapes_once(image, endpoints, pooled, dropped, logits)
        return {
            "logits": tf.cast(logits, tf.float32),
            "semantic_logits": semantic_logits,
            "agg_sim": agg_sim,
            "granularity_weights": granularity_weights,
            "lambda_sem": self.lambda_sem,
            "lambda_hard": self.lambda_hard if self.use_hard_semantic_loss else 0.0,
            "hard_margin": self.hard_margin,
            "hard_pairs_matrix": self.hard_pairs_matrix,
            "cnn_aux_logits": None,
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "attn_scores": tf.zeros([tf.shape(image)[0], 1, 1, 1], dtype=logits.dtype),
            "attention_logits": None,
        }



class ConvNeXtBaseImageNetFERBaseline(tf.keras.Model):
    """Single-head ConvNeXt-Base ImageNet-1K pretrained FER baseline."""

    def __init__(self, cfg: Dict):
        model_cfg = cfg["model"]
        super().__init__(name=model_cfg.get("name", "convnext_base_imagenet1k_baseline"))
        self.num_classes = int(cfg["data"]["num_classes"])
        self.ablation = model_cfg.get("ablation", "cnn_only")
        self.input_size = int(cfg["data"].get("image_size", 112))
        self.channels = int(cfg["data"].get("channels", 3))
        self._shape_logged = False

        if self.channels != 3:
            raise ValueError("ConvNeXt-Base ImageNet-1K baseline requires RGB input channels=3.")
        if not bool(model_cfg.get("pretrained", True)):
            raise ValueError(
                "ConvNeXt-Base ImageNet-1K baseline requires model.pretrained=true; "
                "use config_convnext_base_scratch_baseline.yaml for random initialization."
            )

        source = str(model_cfg.get("convnext_base_pretrained_source", "imagenet")).lower()
        if source not in ("imagenet", "imagenet1k", "imagenet-1k"):
            raise ValueError(
                "ConvNeXt-Base ImageNet-1K baseline only supports "
                "convnext_base_pretrained_source='imagenet'."
            )

        print("ConvNeXt-Base + ImageNet-1K Pretrained Baseline", flush=True)
        print(
            "[ConvNeXtBaseImageNet] Loading tf.keras.applications.ConvNeXtBase "
            f"with weights='imagenet', include_top=False, input_size={self.input_size}",
            flush=True,
        )
        self.backbone = self._build_imagenet_backbone(model_cfg)
        self.use_eca = bool(
            model_cfg.get("use_eca", False)
            or model_cfg.get("ablation") in ("eca", "eca_stage4")
        )
        if self.use_eca:
            self.stage4_eca = ECALayer(channels=1024, name="stage4_eca")
        else:
            self.stage4_eca = None
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="fer_gap")
        self.head_dropout = tf.keras.layers.Dropout(
            float(model_cfg.get("classifier_dropout1", 0.35)),
            name="fer_dropout",
        )
        self.classifier = tf.keras.layers.Dense(
            self.num_classes,
            kernel_initializer="he_normal",
            name="fer_classifier",
        )
        self.pretrained_load_status = "loaded"
        print(
            "[ConvNeXtBaseImageNet] PRETRAINED_LOAD_OK "
            "source=tf.keras.applications.ConvNeXtBase(weights='imagenet')",
            flush=True,
        )
        print(
            f"[ConvNeXtBaseImageNet] FER classifier head: fer_classifier units={self.num_classes}",
            flush=True,
        )

    def _build_imagenet_backbone(self, model_cfg: Dict) -> tf.keras.Model:
        convnext_base = getattr(tf.keras.applications, "ConvNeXtBase", None)
        if convnext_base is None:
            raise RuntimeError("tf.keras.applications.ConvNeXtBase is not available in this TensorFlow build.")

        kwargs = {
            "include_top": False,
            "weights": "imagenet",
            "input_shape": (self.input_size, self.input_size, self.channels),
        }
        try:
            if "include_preprocessing" in inspect.signature(convnext_base).parameters:
                kwargs["include_preprocessing"] = bool(model_cfg.get("builtin_include_preprocessing", False))
        except (TypeError, ValueError):
            pass

        class MixedPrecisionSafeLayerScale(tf.keras.layers.Layer):
            def __init__(self, init_values, projection_dim, **layer_kwargs):
                super().__init__(**layer_kwargs)
                self.init_values = init_values
                self.projection_dim = int(projection_dim)

            def build(self, input_shape):
                self.gamma = self.add_weight(
                    name="gamma",
                    shape=(self.projection_dim,),
                    initializer=tf.keras.initializers.Constant(self.init_values),
                    trainable=True,
                )
                super().build(input_shape)

            def call(self, x):
                return x * tf.cast(self.gamma, x.dtype)

        convnext_module = __import__(convnext_base.__module__, fromlist=["LayerScale"])
        original_layerscale = getattr(convnext_module, "LayerScale", None)
        patch_layerscale = bool(model_cfg.get("convnext_patch_layerscale_dtype", True))
        if patch_layerscale and original_layerscale is not None:
            setattr(convnext_module, "LayerScale", MixedPrecisionSafeLayerScale)
            print("[ConvNeXtBaseImageNet] Mixed precision LayerScale dtype patch enabled", flush=True)

        try:
            return convnext_base(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ConvNeXt-Base ImageNet-1K pretrained weights; "
                "random initialization fallback is disabled."
            ) from exc
        finally:
            if patch_layerscale and original_layerscale is not None:
                setattr(convnext_module, "LayerScale", original_layerscale)

    def _log_shapes_once(self, image, feat_before, feat_after, pooled, dropped, logits) -> None:
        if self._shape_logged:
            return
        self._shape_logged = True
        print("[ConvNeXtBaseImageNet] Shape trace:", flush=True)
        print(f"[ConvNeXtBaseImageNet]   input: {image.shape}", flush=True)
        print(f"[ConvNeXtBaseImageNet]   backbone (stage4): {feat_before.shape}", flush=True)
        if feat_after is not None:
            print(f"[ConvNeXtBaseImageNet]   stage4_eca (after ECA): {feat_after.shape}", flush=True)
        print(f"[ConvNeXtBaseImageNet]   gap: {pooled.shape}", flush=True)
        print(f"[ConvNeXtBaseImageNet]   dropout: {dropped.shape}", flush=True)
        print(f"[ConvNeXtBaseImageNet]   logits: {logits.shape}", flush=True)

        if self.use_eca and self.stage4_eca is not None:
            eca_params = int(np.sum([np.prod(v.shape) for v in self.stage4_eca.trainable_variables]))
            print(f"[ConvNeXtBaseImageNet] ECA Stage4 Attention Enabled:", flush=True)
            print(f"[ConvNeXtBaseImageNet]   ECA kernel size: {self.stage4_eca.k_size}", flush=True)
            print(f"[ConvNeXtBaseImageNet]   ECA trainable params: {eca_params:,}", flush=True)

    def call(self, inputs, training=False, **kwargs):
        image = inputs["image"] if isinstance(inputs, dict) else inputs
        feat = self.backbone(image, training=training)
        feat_before = feat
        if self.use_eca and self.stage4_eca is not None:
            feat = self.stage4_eca(feat, training=training)
        pooled = self.gap(feat)
        dropped = self.head_dropout(pooled, training=training)
        logits = self.classifier(dropped)
        self._log_shapes_once(image, feat_before, feat if self.use_eca else None, pooled, dropped, logits)
        return {
            "logits": logits,
            "cnn_aux_logits": None,
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "attn_scores": tf.zeros([tf.shape(image)[0], 1, 1, 1], dtype=logits.dtype),
            "attention_logits": None,
        }



