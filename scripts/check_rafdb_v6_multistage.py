#!/usr/bin/env python3
"""V6 config contract and opt-in TensorFlow smoke checks (no training)."""
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
NAME = "rafdb_siglip2_semantic_stable_v6_multistage_adaptive_fusion"


def check_contract(path):
    if any(k.startswith("MGR_") for k in os.environ):
        raise ValueError("Unset MGR_* overrides.")
    base = load_config(ROOT / f"config_{BASE}.yaml")
    cfg = load_config(path)
    expected = copy.deepcopy(base)
    expected["source"] = cfg["source"]
    expected["model"].update(
        name=NAME, use_multistage_adaptive_fusion=True,
        multistage_projection_dim=256, multistage_output_dim=512,
        multistage_gate_hidden_dim=128,
    )
    expected["runtime"].update(
        batch_size_per_gpu=32, tf_data_num_parallel_calls=8,
        tf_data_private_threadpool_size=8, prefetch_buffer=4,
    )
    expected["paths"]["output_dir"] = str(ROOT / "outputs/papers" / (NAME + "_batch32"))
    assert cfg == expected, "Unexpected change beyond multi-stage fusion and authorized HPC settings."
    assert cfg["model"]["checkpoint_path"] is None
    assert cfg["model"]["convnext_base_require_pretrained"]
    for file in ("models/multistage_adaptive_fusion.py",
                 "models/convnext_base_face_baseline.py", "utils/stage_fusion_metrics.py",
                 "train.py", "scripts/check_rafdb_v6_multistage.py"):
        ast.parse((ROOT / file).read_text(encoding="utf-8"), filename=file)
    print("V6_MULTISTAGE_CONFIG_CONTRACT_OK: v5 SAM/loss/TTA retained; batch32 workers8 prefetch4",
          flush=True)
    return cfg


def smoke_components(tf):
    import numpy as np
    from models.multistage_adaptive_fusion import MultiStageAdaptiveFusion
    from models.dynamic_part_attention import DynamicPartAttention
    from utils.stage_fusion_metrics import StageFusionMetrics

    original_policy = tf.keras.mixed_precision.global_policy()
    try:
        for policy in ("float32", "mixed_float16"):
            tf.keras.mixed_precision.set_global_policy(policy)
            tf.random.set_seed(42)
            dtype = tf.float16 if policy == "mixed_float16" else tf.float32
            s2 = tf.cast(tf.random.normal([2, 28, 28, 256]), dtype)
            s3 = tf.cast(tf.random.normal([2, 14, 14, 512]), dtype)
            s4 = tf.cast(tf.random.normal([2, 1024]), dtype)
            parts = DynamicPartAttention()
            fusion = MultiStageAdaptiveFusion()
            upper, lower, au, _ = parts(s3)
            initial, weights = fusion((s2, upper, lower, au, s4))
            assert initial.shape == (2, 512)
            assert initial.dtype == tf.float32
            np.testing.assert_allclose(weights.numpy(), np.full((2, 3), 1 / 3), atol=1e-6)

            grid = tf.reshape(tf.range(16 * 256, dtype=tf.float32), [1, 4, 4, 256])
            pooled = fusion.local_grid_pool(grid).numpy().reshape(4, 256)
            manual = np.stack([
                grid.numpy()[:, y:y+2, x:x+2, :].mean(axis=(1, 2))[0]
                for y, x in ((0, 0), (0, 2), (2, 0), (2, 2))
            ])
            np.testing.assert_allclose(pooled, manual, rtol=1e-6, atol=1e-6)

            @tf.function
            def gradients(a, b, c):
                with tf.GradientTape() as tape:
                    tape.watch((a, b, c))
                    u, l, au, maps = parts(b, training=True)
                    fused, weights = fusion((a, u, l, au, c), training=True)
                    # Asymmetric probe avoids a constant squared-LayerNorm objective.
                    probe = tf.linspace(-1.0, 1.0, 512)
                    loss = tf.reduce_mean(fused * probe)
                variables = [a, b, c] + parts.trainable_variables + fusion.trainable_variables
                return weights, tape.gradient(loss, variables)

            _, grads = gradients(s2, s3, s4)
            for grad in grads:
                assert grad is not None, "Disconnected fusion gradient."
                tf.debugging.assert_all_finite(grad, "Non-finite fusion gradient")
            for grad in grads[:3]:
                assert float(tf.linalg.global_norm([tf.cast(grad, tf.float32)])) > 0
            # Gate must respond per sample once the zero-initialized output kernel learns.
            fusion.gate.layers[-1].kernel.assign(
                tf.random.normal(fusion.gate.layers[-1].kernel.shape, stddev=0.1))
            _, adaptive_weights = fusion((s2, upper, lower, au, s4))
            assert not np.allclose(adaptive_weights.numpy()[0], adaptive_weights.numpy()[1])
            np.testing.assert_allclose(adaptive_weights.numpy().sum(-1), np.ones(2), atol=1e-6)
            print(f"V6_MULTISTAGE_COMPONENT_SMOKE_OK policy={policy} "
                  "S2=[2,28,28,256] S3=[2,14,14,512] fused=[2,512] weights=[2,3]",
                  flush=True)

        tracker = StageFusionMetrics("test_fusion")
        known = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]], np.float32)
        @tf.function
        def update(w):
            tracker.update_state({"stage_fusion_weights": w,
                                  "adaptive_fusion_alpha": tf.fill([tf.shape(w)[0], 1], 0.1)})
        update(tf.constant(known[:2]))
        update(tf.constant(known[2:]))
        snapshot = tracker.snapshot()
        assert snapshot["stage_fusion_samples"] == 3
        for index, stage in enumerate((2, 3, 4)):
            np.testing.assert_allclose(snapshot[f"stage_fusion_s{stage}_mean"],
                                       known[:, index].mean(), atol=1e-6)
            np.testing.assert_allclose(snapshot[f"stage_fusion_s{stage}_std"],
                                       known[:, index].std(), atol=1e-6)
        entropy = -(known * np.log(known)).sum(-1).mean() / np.log(3)
        np.testing.assert_allclose(snapshot["stage_fusion_entropy"], entropy, atol=1e-6)
        tracker.reset_state()
        assert tracker.snapshot()["stage_fusion_samples"] == 0
        print("V6_FUSION_METRICS_SMOKE_OK: sample-weighted unequal batches and reset", flush=True)
    finally:
        tf.keras.mixed_precision.set_global_policy(original_policy)


def smoke_model(tf, cfg):
    """Uses real pretrained/prototype loading, synthetic images, and no optimizer step."""
    from train import build_model, compute_loss, split_variables
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(cfg)
    images = tf.random.normal([2, cfg["data"]["image_size"], cfg["data"]["image_size"], 3])
    model(images, training=False)
    assert model.pretrained_load_status == "loaded", model.pretrained_load_status
    _, head_vars = split_variables(model)
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        loss, _ = compute_loss(outputs, tf.constant([0, 1]), cfg, model=model)
    gradients = tape.gradient(loss, head_vars)
    tf.debugging.assert_all_finite(loss, "Non-finite model smoke loss")
    for variable, gradient in zip(head_vars, gradients):
        assert gradient is not None, f"Disconnected head variable: {variable.name}"
        tf.debugging.assert_all_finite(gradient, variable.name)
    assert tuple(outputs["logits"].shape) == (2, 7)
    assert tuple(outputs["semantic_logits"].shape) == (2, 7)
    assert tuple(outputs["stage_fusion_weights"].shape) == (2, 3)
    tf.debugging.assert_near(tf.reduce_sum(outputs["stage_fusion_weights"], -1), tf.ones([2]))
    alpha = outputs["adaptive_fusion_alpha"]
    tf.debugging.assert_near(
        outputs["logits"],
        (1 - alpha) * outputs["visual_logits"] + alpha * outputs["semantic_logits"],
    )
    # Forward alone must not count samples (SAM has two forward passes).
    assert model.stage_fusion_train_metrics.snapshot()["stage_fusion_samples"] == 0
    print(f"V6_MULTISTAGE_MODEL_SMOKE_OK synthetic_batch=2 loss={float(loss):.6f}; "
          "real pretrained/prototypes loaded; head gradients finite. "
          "Full batch32 SAM VRAM and dataset remain untested.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / f"config_{NAME}.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()
    cfg = check_contract(args.config)
    if args.smoke or args.model_smoke:
        import tensorflow as tf
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
        if args.smoke:
            smoke_components(tf)
        if args.model_smoke:
            smoke_model(tf, cfg)


if __name__ == "__main__":
    main()
