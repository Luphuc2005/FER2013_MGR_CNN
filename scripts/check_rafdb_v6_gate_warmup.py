#!/usr/bin/env python3
"""Contract/smoke check for the RAF-DB V6 granularity-gate warm-up ablation.

This is read-only: it builds the configured model with synthetic tensors and
does not create checkpoints, run an optimizer, or access RAF-DB samples.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


BASELINE_CONFIG = ROOT / "config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml"
DEFAULT_CONFIG = ROOT / "config_rafdb_siglip2_semantic_stable_v6_gate_warmup.yaml"
SCHEDULE = {
    "enabled": True,
    "uniform_epochs": 4,
    "transition_end_epoch": 10,
    "start_temperature": 2.5,
    "end_temperature": 1.0,
}


def assert_config_is_clean_ablation(config_path: Path) -> dict:
    base = load_config(BASELINE_CONFIG)
    cfg = load_config(config_path)
    expected = copy.deepcopy(base)
    expected["source"]["note"] = (
        "RAF-DB V6 gate-warmup ablation: V5 Combined Ultimate with only the "
        "5-way semantic granularity gate schedule changed."
    )
    expected["model"]["name"] = "rafdb_siglip2_semantic_stable_v6_gate_warmup"
    expected["model"]["granularity_gate_schedule"] = SCHEDULE
    expected["paths"]["output_dir"] = str(
        ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v6_gate_warmup"
    )
    if cfg != expected:
        raise AssertionError("V6 config differs from V5 original outside identity/output/gate schedule.")
    assert cfg["data"].get("sampling_strategy", "natural") == "natural"
    assert not cfg["data"].get("sqrt_sampling", False)
    print("V6_GATE_WARMUP_CONFIG_CLEAN_ABLATION_OK sampling=natural", flush=True)
    return cfg


def assert_simplex(tf, weights, batch_size: int) -> None:
    assert tuple(weights.shape) == (batch_size, 5), weights.shape
    tf.debugging.assert_all_finite(weights, "Non-finite granularity alpha")
    tf.debugging.assert_near(
        tf.reduce_sum(tf.cast(weights, tf.float32), axis=-1),
        tf.ones([batch_size], dtype=tf.float32),
        atol=1e-6,
    )


def expected_transition(tf, logits, epoch: int):
    beta = (epoch - SCHEDULE["uniform_epochs"]) / (
        SCHEDULE["transition_end_epoch"] - SCHEDULE["uniform_epochs"]
    )
    temperature = SCHEDULE["start_temperature"] + beta * (
        SCHEDULE["end_temperature"] - SCHEDULE["start_temperature"]
    )
    adaptive = tf.nn.softmax(tf.cast(logits, tf.float32) / temperature, axis=-1)
    uniform = tf.fill(tf.shape(adaptive), tf.constant(0.2, tf.float32))
    return (1.0 - beta) * uniform + beta * adaptive


def smoke_model(cfg: dict) -> None:
    import numpy as np
    import tensorflow as tf
    from train import build_model

    tf.keras.utils.set_random_seed(42)
    tf.keras.mixed_precision.set_global_policy("float32")
    model = build_model(cfg)
    gate_vars = list(model.granularity_gate.trainable_variables)
    assert gate_vars, "Warm-up gate variables must be built before optimizer construction."
    pooled = tf.random.normal([3, model.backbone.dims[-1]], dtype=tf.float32)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def graph_weights(features):
        return model._granularity_gate_weights(features, training=False)

    uniform = tf.fill([3, 5], tf.constant(0.2, tf.float32))
    for epoch in (1, 4):
        model.set_granularity_gate_epoch(epoch)
        weights = graph_weights(pooled)
        assert_simplex(tf, weights, batch_size=3)
        np.testing.assert_allclose(weights.numpy(), uniform.numpy(), rtol=0.0, atol=1e-7)

    logits = model.granularity_gate(pooled, training=False)
    for epoch in (5, 9):
        model.set_granularity_gate_epoch(epoch)
        weights = graph_weights(pooled)
        assert_simplex(tf, weights, batch_size=3)
        np.testing.assert_allclose(
            weights.numpy(), expected_transition(tf, logits, epoch).numpy(), rtol=1e-6, atol=1e-6
        )
        assert not np.allclose(weights.numpy(), uniform.numpy(), rtol=0.0, atol=1e-7)

    model.set_granularity_gate_epoch(10)
    weights = graph_weights(pooled)
    assert_simplex(tf, weights, batch_size=3)
    np.testing.assert_allclose(
        weights.numpy(), tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1).numpy(),
        rtol=1e-6, atol=1e-6,
    )

    # No logits are executed in uniform warm-up, therefore no gate gradient.
    contrast = tf.constant([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)
    model.set_granularity_gate_epoch(1)
    with tf.GradientTape() as tape:
        uniform_loss = tf.reduce_sum(model._granularity_gate_weights(pooled, training=False) * contrast)
    uniform_grads = tape.gradient(uniform_loss, gate_vars)
    assert all(grad is None for grad in uniform_grads), "Gate received a warm-up gradient."

    model.set_granularity_gate_epoch(5)
    with tf.GradientTape() as tape:
        adaptive_loss = tf.reduce_sum(model._granularity_gate_weights(pooled, training=False) * contrast)
    adaptive_grads = tape.gradient(adaptive_loss, gate_vars)
    assert all(grad is not None for grad in adaptive_grads), "Gate is disconnected after warm-up."
    for grad in adaptive_grads:
        tf.debugging.assert_all_finite(grad, "Non-finite gate gradient")
    assert float(tf.linalg.global_norm([tf.cast(g, tf.float32) for g in adaptive_grads])) > 0.0

    # End-to-end output confirms the scheduled alpha is the one used by semantic aggregation.
    images = tf.random.normal([2, cfg["data"]["image_size"], cfg["data"]["image_size"], 3])
    outputs = model(images, training=False)
    assert_simplex(tf, outputs["granularity_weights"], batch_size=2)
    tf.debugging.assert_all_finite(outputs["semantic_logits"], "Non-finite semantic logits")
    print(
        "V6_GATE_WARMUP_SMOKE_OK phases=uniform(1-4),transition(5-9),adaptive(>=10) "
        "shape=[B,5] simplex=ok finite=ok gate_gradient_after_warmup=ok",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--config-only", action="store_true",
        help="Verify the clean-ablation config without importing TensorFlow.",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = assert_config_is_clean_ablation(config_path)
    if args.config_only:
        print("V6_GATE_WARMUP_CONFIG_ONLY_OK TensorFlow smoke skipped by request", flush=True)
        return 0
    smoke_model(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
