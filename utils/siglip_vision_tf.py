"""Frozen TensorFlow inference for HF SigLIP fixed-resolution vision encoders.

Tensor layout follows the Apache-2.0 Hugging Face SigLIP implementation:
https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py
No trainable variables; all weights come from a strict safetensors mapping.
"""
import numpy as np
import tensorflow as tf


def expected_shapes(c):
    d, m = c["hidden_size"], c["intermediate_size"]
    p, side = c["patch_size"], c["image_size"]
    shapes = {"embeddings.patch_embedding.weight": (d, 3, p, p),
              "embeddings.patch_embedding.bias": (d,),
              "embeddings.position_embedding.weight": ((side // p)**2, d)}
    def norm(prefix):
        shapes[prefix + ".weight"] = (d,)
        shapes[prefix + ".bias"] = (d,)
    def linear(prefix, out, inp):
        shapes[prefix + ".weight"] = (out, inp)
        shapes[prefix + ".bias"] = (out,)
    def mlp(prefix):
        linear(prefix + ".fc1", m, d)
        linear(prefix + ".fc2", d, m)
    for i in range(c["num_hidden_layers"]):
        prefix = f"encoder.layers.{i}"
        norm(prefix + ".layer_norm1")
        norm(prefix + ".layer_norm2")
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(prefix + ".self_attn." + proj, d, d)
        mlp(prefix + ".mlp")
    norm("post_layernorm")
    shapes["head.probe"] = (1, 1, d)
    shapes["head.attention.in_proj_weight"] = (3*d, d)
    shapes["head.attention.in_proj_bias"] = (3*d,)
    linear("head.attention.out_proj", d, d)
    norm("head.layernorm")
    mlp("head.mlp")
    return {"vision_model." + k: v for k, v in shapes.items()}


class FrozenSiglipVisionTF(tf.Module):
    def __init__(self, weights, config):
        super().__init__(name="frozen_siglip_vision")
        self.c = config
        if (config["image_size"], config["patch_size"], config["hidden_size"]) != (224, 16, 768):
            raise ValueError("Only the configured SigLIP2 base-patch16-224 is supported")
        if (config.get("num_channels", 3) != 3 or not config.get("vision_use_head", True)
                or config["hidden_size"] % config["num_attention_heads"] != 0):
            raise ValueError("Unsupported SigLIP channels/pooling/attention configuration")
        if config["hidden_act"] not in ("gelu_pytorch_tanh", "gelu"):
            raise ValueError("Unsupported activation: " + config["hidden_act"])
        expected = expected_shapes(config)
        if set(weights) != set(expected):
            raise ValueError(f"Vision tensor mismatch: missing={set(expected)-set(weights)} "
                             f"unexpected={set(weights)-set(expected)}")
        self.w = {}
        for key, shape in expected.items():
            value = weights[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid vision weight: " + key)
            self.w[key.removeprefix("vision_model.")] = tf.constant(value, tf.float32)
        self.patch_kernel = tf.transpose(self.w["embeddings.patch_embedding.weight"], [2, 3, 1, 0])
        self.matched_tensors = len(expected)

    def linear(self, x, prefix):
        return tf.matmul(x, self.w[prefix + ".weight"], transpose_b=True) + self.w[prefix + ".bias"]

    def norm(self, x, prefix):
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(variance + self.c["layer_norm_eps"]) * self.w[prefix + ".weight"] + self.w[prefix + ".bias"]

    def mlp(self, x, prefix):
        x = self.linear(x, prefix + ".fc1")
        x = tf.nn.gelu(x, approximate=self.c["hidden_act"] == "gelu_pytorch_tanh")
        return self.linear(x, prefix + ".fc2")

    def attend(self, q, k, v):
        heads = self.c["num_attention_heads"]
        width = self.c["hidden_size"]
        depth = width // heads
        def split(x):
            return tf.transpose(tf.reshape(x, [tf.shape(x)[0], -1, heads, depth]), [0, 2, 1, 3])
        q, k, v = split(q), split(k), split(v)
        probabilities = tf.nn.softmax(tf.matmul(q * (depth ** -0.5), k, transpose_b=True), axis=-1)
        out = tf.transpose(tf.matmul(probabilities, v), [0, 2, 1, 3])
        return tf.reshape(out, [tf.shape(out)[0], -1, width])

    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)], jit_compile=False)
    def __call__(self, images):
        x = tf.nn.conv2d(images, self.patch_kernel, strides=[1, 16, 16, 1], padding="VALID")
        x = x + self.w["embeddings.patch_embedding.bias"]
        x = tf.reshape(x, [tf.shape(x)[0], 196, 768]) + self.w["embeddings.position_embedding.weight"]
        for i in range(self.c["num_hidden_layers"]):
            prefix = f"encoder.layers.{i}"
            z = self.norm(x, prefix + ".layer_norm1")
            attn = prefix + ".self_attn"
            z = self.attend(self.linear(z, attn + ".q_proj"),
                            self.linear(z, attn + ".k_proj"),
                            self.linear(z, attn + ".v_proj"))
            x = x + self.linear(z, attn + ".out_proj")
            x = x + self.mlp(self.norm(x, prefix + ".layer_norm2"), prefix + ".mlp")
        x = self.norm(x, "post_layernorm")
        probe = tf.tile(self.w["head.probe"], [tf.shape(x)[0], 1, 1])
        kernels = tf.split(self.w["head.attention.in_proj_weight"], 3, axis=0)
        biases = tf.split(self.w["head.attention.in_proj_bias"], 3, axis=0)
        q, k, v = [tf.matmul(t, w, transpose_b=True) + b
                   for t, w, b in zip((probe, x, x), kernels, biases)]
        x = self.linear(self.attend(q, k, v), "head.attention.out_proj")
        x = x + self.mlp(self.norm(x, "head.layernorm"), "head.mlp")
        return tf.stop_gradient(x[:, 0])


def verify_cpu_reference(teacher, weights, vision_config, pixels_nhwc, prototypes, scale):
    """One small mandatory parity check. Torch is a CPU reference, never a GPU teacher."""
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(**vision_config)
    cfg._attn_implementation = "eager"
    reference = SiglipVisionModel(cfg).cpu().float().eval()
    reference.requires_grad_(False)
    reference.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
    with torch.inference_mode():
        expected = reference(pixel_values=torch.from_numpy(pixels_nhwc.transpose(0, 3, 1, 2).copy())).pooler_output.numpy()
    actual = teacher(tf.constant(pixels_nhwc)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4,
                               err_msg="TF SigLIP pooled embedding parity failed")
    def logits(v):
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        return scale * v @ prototypes.T
    np.testing.assert_allclose(logits(actual), logits(expected), rtol=3e-4, atol=3e-4,
                               err_msg="TF SigLIP teacher logits parity failed")
    report = dict(reference="HuggingFace SiglipVisionModel on CPU",
                  samples=len(actual), embedding_max_abs=float(np.max(np.abs(actual-expected))),
                  logits_max_abs=float(np.max(np.abs(logits(actual)-logits(expected)))),
                  rtol=3e-4, atol=3e-4, torch_version=torch.__version__)
    print("SIGLIP_TF_CPU_REFERENCE_PARITY_OK " + str(report), flush=True)
    return report
