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

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf


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

    def call(self, x, training=False, return_endpoints: bool = False):
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
        if return_endpoints:
            return endpoints
        return endpoints["stage4"]


class ConvNeXtBaseFaceFERBaseline(tf.keras.Model):
    """Single-head ConvNeXt-Base face-pretrained FER baseline."""

    def __init__(self, cfg: Dict):
        model_cfg = cfg["model"]
        super().__init__(name=model_cfg.get("name", "convnext_base_ms1m_arcface_baseline"))
        self.num_classes = int(cfg["data"]["num_classes"])
        self.ablation = model_cfg.get("ablation", "cnn_only")
        self.input_size = int(cfg["data"].get("image_size", 112))
        self.channels = int(cfg["data"].get("channels", 3))
        self._shape_logged = False

        self.backbone = ConvNeXtBaseFRBackbone(
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
            layer_scale_init_value=float(model_cfg.get("layer_scale_init_value", 1e-6)),
            name="convnext_base_fr_backbone",
        )
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

        self.pretrained_load_status = "not_requested"
        pretrained_path = model_cfg.get("convnext_base_pretrained_path") or model_cfg.get("pretrained_path")
        if pretrained_path:
            self.pretrained_load_status = self._load_pytorch_pretrained(
                pretrained_path,
                require=bool(model_cfg.get("convnext_base_require_pretrained", False)),
            )

    def _resolve_weight_path(self, weight_path: str) -> Path:
        resolved = Path(weight_path)
        if not resolved.is_absolute():
            resolved = Path(__file__).resolve().parents[1] / weight_path
        return resolved

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

    def _load_pytorch_pretrained(self, weight_path: str, require: bool = False) -> str:
        resolved = self._resolve_weight_path(weight_path)
        if not resolved.exists():
            message = f"[ConvNeXtBaseFace] PyTorch pretrained checkpoint not found: {resolved}"
            if require:
                raise FileNotFoundError(message)
            print(f"[ConvNeXtBaseFace] WARNING: {message}", flush=True)
            return "missing"

        try:
            import torch
        except Exception as exc:
            if require:
                raise RuntimeError("PyTorch is required to load ConvNeXt-B MS1M/ArcFace checkpoint.") from exc
            print(f"[ConvNeXtBaseFace] WARNING: PyTorch import failed: {exc}", flush=True)
            return "torch_missing"

        self._build_variables()
        print(f"[ConvNeXtBaseFace] Loading PyTorch pretrained checkpoint: {resolved}", flush=True)
        checkpoint = torch.load(str(resolved), map_location="cpu")
        state = self._normalize_state_dict(self._extract_state_dict(checkpoint))
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
                assign(
                    block.dwconv.depthwise_kernel,
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
        for key in ("stem", "stage1", "stage2", "stage3", "stage4"):
            print(f"[ConvNeXtBaseFace]   {key}: {endpoints[key].shape}", flush=True)
        print(f"[ConvNeXtBaseFace]   gap: {pooled.shape}", flush=True)
        print(f"[ConvNeXtBaseFace]   dropout: {dropped.shape}", flush=True)
        print(f"[ConvNeXtBaseFace]   logits: {logits.shape}", flush=True)

    def call(self, inputs, training=False, **kwargs):
        image = inputs["image"] if isinstance(inputs, dict) else inputs
        endpoints = self.backbone(image, training=training, return_endpoints=True)
        feat = endpoints["stage4"]
        pooled = self.gap(feat)
        dropped = self.head_dropout(pooled, training=training)
        logits = self.classifier(dropped)
        self._log_shapes_once(image, endpoints, pooled, dropped, logits)
        return {
            "logits": logits,
            "cnn_aux_logits": None,
            "ortho_loss": tf.constant(0.0, dtype=tf.float32),
            "attn_scores": tf.zeros([tf.shape(image)[0], 1, 1, 1], dtype=logits.dtype),
            "attention_logits": None,
        }


