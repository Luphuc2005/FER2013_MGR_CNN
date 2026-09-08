#!/usr/bin/env python3
"""Check v5->v6 recipe parity; optionally test ECA/parts on synthetic features.

--smoke needs TensorFlow. It verifies the regional component only, not the
full backbone, pretrained checkpoint, dataset, or training accuracy.
"""
from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config

BASE = "rafdb_siglip2_semantic_stable_v5_combined_ultimate"
NAME = "rafdb_siglip2_semantic_stable_v6_eca_dynamic_part_attention"


def check_contract(config_path):
    if any(key.startswith("MGR_") for key in os.environ):
        raise ValueError("Unset MGR_* overrides before checking the experiment.")
    base = load_config(ROOT / f"config_{BASE}.yaml")
    cfg = load_config(config_path)
    model = cfg["model"]
    assert model["use_stage3_eca"] is True
    assert model["use_dynamic_part_attention"] is True
    assert not model.get("use_global_regional_fusion", False)
    assert not model.get("use_eca", False), "Stage-4 ECA must stay disabled."
    assert model["checkpoint_path"] is None and not cfg["training"]["resume"]
    assert model["convnext_base_require_pretrained"] is True
    assert model["name"] == NAME
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers" / (NAME + "_batch32")
    comparable = copy.deepcopy(cfg)
    comparable["source"] = copy.deepcopy(base["source"])
    comparable["model"]["name"] = base["model"]["name"]
    comparable["paths"]["output_dir"] = base["paths"]["output_dir"]
    del comparable["model"]["use_stage3_eca"]
    # Explicitly authorized HPC throughput changes; all other v5 settings match.
    runtime_changes = {
        "batch_size_per_gpu": 32,
        "tf_data_num_parallel_calls": 8,
        "tf_data_private_threadpool_size": 8,
        "prefetch_buffer": 4,
    }
    for key, expected in runtime_changes.items():
        assert cfg["runtime"][key] == expected, f"{key} must be {expected}"
        comparable["runtime"][key] = base["runtime"][key]
    assert comparable == base, "Unexpected change beyond ECA, identity and HPC runtime settings."
    for relative in (
        "models/convnext_base_face_baseline.py",
        "models/dynamic_part_attention.py",
        "scripts/check_rafdb_v6_eca.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    print("V6_ECA_CONFIG_CONTRACT_OK: v5 + regional ECA; SAM retained; batch=32 workers=8 prefetch=4", flush=True)


def smoke_components():
    import numpy as np
    import tensorflow as tf
    from models.convnext_base_face_baseline import ECALayer
    from models.dynamic_part_attention import DynamicPartAttention

    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    original_policy = tf.keras.mixed_precision.global_policy()
    try:
        for policy in ("float32", "mixed_float16"):
            tf.keras.mixed_precision.set_global_policy(policy)
            tf.random.set_seed(42)
            dtype = tf.float16 if policy == "mixed_float16" else tf.float32
            features = tf.cast(tf.random.normal([2, 14, 14, 512]), dtype)
            eca = ECALayer(channels=512, name="stage3_eca")
            parts = DynamicPartAttention(channels=512, bottleneck=128)
            eca(features)
            assert eca.k_size == 5 and eca.count_params() == 5
            # Zero channel scores imply sigmoid(0)=0.5, testing channel axis/broadcast.
            eca.conv1d.kernel.assign(tf.zeros_like(eca.conv1d.kernel))
            channel_half = eca(features)
            assert channel_half.dtype == dtype
            # Round the reference to the layer's compute dtype as well. FP16
            # halving near zero can differ from an FP32 reference by 2**-25.
            expected_half = tf.cast(
                0.5 * tf.cast(features, tf.float32), channel_half.dtype,
            )
            # Allow one FP16 subnormal step; retain exact comparison for FP32.
            half_atol = (
                float(np.nextafter(np.float16(0), np.float16(1)))
                if dtype == tf.float16 else 0.0
            )
            np.testing.assert_allclose(
                tf.cast(channel_half, tf.float32).numpy(),
                tf.cast(expected_half, tf.float32).numpy(), rtol=0, atol=half_atol,
            )
            parts(eca(features))

            @tf.function
            def forward_and_grad(x):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    channel = eca(x, training=True)
                    upper, lower, au, maps = parts(channel, training=True)
                    loss = sum(tf.reduce_mean(tf.square(tf.cast(z, tf.float32)))
                               for z in (upper, lower, au))
                variables = [x] + eca.trainable_variables + parts.trainable_variables
                grads = tape.gradient(loss, variables)
                return channel, (upper, lower, au), maps, grads

            channel, regions, maps, grads = forward_and_grad(features)
            assert tuple(maps.shape) == (2, 14, 14, 3)
            np.testing.assert_allclose(
                tf.reduce_sum(maps, axis=[1, 2]).numpy(), np.ones((2, 3)), atol=1e-5,
            )
            # All three pools must use the recalibrated features, with upper/lower/AU order.
            for idx, region in enumerate(regions):
                assert tuple(region.shape) == (2, 512)
                expected = tf.reduce_sum(
                    maps[..., idx, None] * tf.cast(channel, tf.float32), axis=[1, 2],
                )
                np.testing.assert_allclose(
                    tf.cast(region, tf.float32).numpy(), expected.numpy(),
                    rtol=2e-3, atol=2e-4,
                )
            for grad in grads:
                assert grad is not None, "Disconnected gradient in ECA/part branch."
                tf.debugging.assert_all_finite(grad, "Non-finite ECA/part gradient")
            assert float(tf.linalg.global_norm(grads[1:1 + len(eca.trainable_variables)])) > 0
            print(f"V6_ECA_COMPONENT_SMOKE_OK policy={policy} F3={features.shape} "
                  f"maps={maps.shape} regions={[z.shape for z in regions]} "
                  f"eca_params={eca.count_params()}", flush=True)
    finally:
        tf.keras.mixed_precision.set_global_policy(original_policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / f"config_{NAME}.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    check_contract(args.config)
    if args.smoke:
        smoke_components()


if __name__ == "__main__":
    main()
