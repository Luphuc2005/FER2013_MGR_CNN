#!/usr/bin/env python3
"""Check V5 LGSA wiring with real pretrained/prototypes and a synthetic batch.

No optimizer update or training run. Requires TensorFlow and the configured
pretrained files. Checks LGSA-only gradients, loss contribution, zero-weight
equivalence, and unchanged inference logits.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config


def run_smoke_test(config_path):
    import numpy as np
    import tensorflow as tf
    from losses.classification import supervised_mgr_loss
    from train import build_model, split_variables

    if any(key.startswith("MGR_") for key in os.environ):
        raise ValueError("Unset MGR_* overrides before checking V5 LGSA.")
    cfg = load_config(config_path)
    assert cfg["model"]["lambda_local_sem"] == 0.02
    assert cfg["model"]["use_dynamic_part_attention"]
    assert cfg["model"]["use_adaptive_fusion_gate"]
    assert not cfg["model"].get("use_multistage_adaptive_fusion", False)
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_lgsa"

    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(42)
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if cfg["runtime"].get("use_mixed_precision") else "float32"
    )
    # Keep the actual pretrained contract; never silently fall back to scratch.
    model = build_model(cfg)
    images = tf.random.normal([2, cfg["data"]["image_size"], cfg["data"]["image_size"], 3])
    labels = tf.constant([1, 4], dtype=tf.int32)
    model(images, training=False)
    assert model.pretrained_load_status == "loaded", model.pretrained_load_status
    _, head_vars = split_variables(model)

    loss_kwargs = dict(
        num_classes=cfg["data"]["num_classes"],
        label_smoothing=cfg["training"]["label_smoothing"],
        ortho_weight=cfg["model"]["ortho_loss_weight"],
        cnn_aux_weight=cfg["model"]["cnn_aux_loss_weight"],
    )
    local_keys = ("s_upper", "s_lower", "s_au", "local_semantic_logits", "lambda_local_sem")
    # Watch heads only to keep this wiring regression small. All actual forward
    # layers, including the backbone and Dynamic Part Attention, still execute.
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(head_vars)
        out = model(images, training=True)
        for key in ("logits", "s_upper", "s_lower", "s_au"):
            assert tuple(out[key].shape) == (2, 7), (key, out[key].shape)
        for name in ("upper", "lower", "au"):
            tf.debugging.assert_equal(out["local_semantic_logits"][name], out["s_" + name])
        tf.debugging.assert_near(out["lambda_local_sem"], tf.constant(0.02))
        total, parts = supervised_mgr_loss(labels, out, **loss_kwargs)
        zero_out = dict(out, lambda_local_sem=0.0)
        zero_total, _ = supervised_mgr_loss(labels, zero_out, **loss_kwargs)
        baseline_out = {key: value for key, value in out.items() if key not in local_keys}
        baseline_total, _ = supervised_mgr_loss(labels, baseline_out, **loss_kwargs)
        local_loss = parts["local_semantic"]
        # Independent CE reference for all three regions, including smoothing.
        smoothing = loss_kwargs["label_smoothing"]
        targets = tf.one_hot(labels, 7) * (1.0 - smoothing) + smoothing / 7
        reference_local = tf.add_n([
            tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                targets, out[key], from_logits=True
            )) for key in ("s_upper", "s_lower", "s_au")
        ]) / 3.0
        weighted_local = out["lambda_local_sem"] * local_loss

    tf.debugging.assert_all_finite(total, "Non-finite total loss")
    assert float(local_loss) > 0.0
    np.testing.assert_allclose(local_loss.numpy(), reference_local.numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        (total - zero_total).numpy(), weighted_local.numpy(), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_array_equal(zero_total.numpy(), baseline_total.numpy())

    total_grads = tape.gradient(total, head_vars)
    zero_grads = tape.gradient(zero_total, head_vars)
    baseline_grads = tape.gradient(baseline_total, head_vars)
    local_grads = tape.gradient(weighted_local, head_vars)
    for variable, grad, zero_grad, baseline_grad, local_grad in zip(
        head_vars, total_grads, zero_grads, baseline_grads, local_grads
    ):
        assert grad is not None, "Disconnected total gradient: " + variable.name
        tf.debugging.assert_all_finite(grad, variable.name)
        assert zero_grad is not None and baseline_grad is not None, variable.name
        np.testing.assert_allclose(zero_grad.numpy(), baseline_grad.numpy(), rtol=1e-5, atol=1e-6)
        # Mixed precision can round the combined backward path; allow FP16 error.
        expected = tf.cast(zero_grad, tf.float32)
        if local_grad is not None:
            expected += tf.cast(local_grad, tf.float32)
        np.testing.assert_allclose(
            tf.cast(grad, tf.float32).numpy(), expected.numpy(), rtol=1e-2, atol=2e-4
        )

    for layer in (
        model.visual_projector_upper, model.visual_projector_lower,
        model.visual_projector_au, model.dynamic_part_attn,
    ):
        grads = tape.gradient(local_loss, layer.trainable_variables)
        assert grads and all(grad is not None for grad in grads), layer.name
        for grad in grads:
            tf.debugging.assert_all_finite(grad, layer.name)
        assert float(tf.linalg.global_norm([tf.cast(g, tf.float32) for g in grads])) > 0, layer.name
        print("LGSA_GRADIENT_OK " + layer.name, flush=True)
    del tape

    # Lambda affects the loss only. Compare deterministic inference at the same
    # weights with LGSA enabled and disabled, keeping adaptive fusion intact.
    enabled_logits = model(images, training=False)["logits"]
    original_lambda = model.lambda_local_sem
    try:
        model.lambda_local_sem = 0.0
        disabled_logits = model(images, training=False)["logits"]
    finally:
        model.lambda_local_sem = original_lambda
    np.testing.assert_array_equal(enabled_logits.numpy(), disabled_logits.numpy())
    print(
        f"LGSA_MODEL_SMOKE_OK local={float(local_loss):.6f} "
        f"weighted={float(weighted_local):.6f} total={float(total):.6f}; "
        "zero-weight loss/gradients match baseline; inference unchanged. "
        "Full SAM step and dataset training were not tested.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config_rafdb_siglip2_semantic_stable_v5_lgsa.yaml"))
    run_smoke_test(parser.parse_args().config)
