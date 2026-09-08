#!/usr/bin/env python3
"""Isolated direct global/part fusion + LGSA + R-Drop candidate checks.

Default: exact recipe parity, without TensorFlow.
--smoke: R-Drop/SAM regression and fusion component gradients.
--model-smoke: real pretrained model, direct local-to-visual-head gradients,
then the existing real batch-16 head/full SAM smoke. No checkpoint writes.
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

BASE = "rafdb_siglip2_semantic_stable_v5_lgsa_rdrop"
NAME = "rafdb_siglip2_semantic_stable_v6_global_regional_lgsa_rdrop"


def check_contract(path):
    if any(key.startswith("MGR_") for key in os.environ):
        raise ValueError("Unset MGR_* overrides before checking this experiment.")
    base = load_config(ROOT / ("config_" + BASE + ".yaml"))
    cfg = load_config(path)
    expected = copy.deepcopy(base)
    expected["source"] = cfg["source"]
    expected["model"].update(
        name=NAME, use_global_regional_fusion=True,
        global_regional_projection_dim=256,
        use_multistage_adaptive_fusion=False,
    )
    expected["paths"]["output_dir"] = str(ROOT / "outputs/papers" / NAME)
    assert cfg == expected, "Unexpected changes beyond direct fusion and experiment identity."
    assert cfg["training"]["lambda_rdrop"] == 0.5
    assert cfg["model"]["lambda_local_sem"] == 0.02
    assert cfg["model"]["clip_semantic"]["lambda_local_sem"] == 0.02
    assert cfg["model"]["convnext_base_require_pretrained"]
    assert cfg["model"]["checkpoint_path"] is None
    assert not cfg["training"]["resume"]
    assert cfg["runtime"]["batch_size_per_gpu"] == 16
    for file in ("train.py", "models/global_regional_fusion.py",
                 "models/convnext_base_face_baseline.py", "losses/rdrop.py",
                 "scripts/check_rafdb_v6_global_regional_lgsa_rdrop.py"):
        ast.parse((ROOT / file).read_text(encoding="utf-8"), filename=file)
    print("V6_GLOBAL_REGIONAL_LGSA_RDROP_CONFIG_OK: direct fusion; "
          "LGSA=0.02 RDrop=0.5; V5 training/TTA recipe retained", flush=True)
    return cfg


def check_gradients(tf, gradients, message):
    assert gradients and all(g is not None for g in gradients), message
    for grad in gradients:
        tf.debugging.assert_all_finite(grad, message)
    assert float(tf.linalg.global_norm([tf.cast(g, tf.float32) for g in gradients])) > 0, message


def smoke_components():
    from check_rafdb_v5_lgsa_rdrop import smoke_components as rdrop_smoke
    rdrop_smoke()
    import tensorflow as tf
    from models.global_regional_fusion import GlobalRegionalFusion

    original_policy = tf.keras.mixed_precision.global_policy()
    try:
        for policy in ("float32", "mixed_float16"):
            tf.keras.mixed_precision.set_global_policy(policy)
            fusion = GlobalRegionalFusion(projection_dim=256, dtype="float32")
            inputs = [tf.random.normal([2, d]) for d in (1024, 512, 512, 512)]
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                fused = fusion(inputs)
                probe = tf.reduce_sum(fused * tf.linspace(-1., 1., 1024))
            gradients = tape.gradient(probe, inputs + fusion.trainable_variables)
            assert tuple(fused.shape) == (2, 1024)
            assert fused.dtype == tf.float32
            assert fusion.count_params() == 661504
            check_gradients(tf, gradients, "Fusion gradient")
            for gradient in gradients[:4]:
                check_gradients(tf, [gradient], "Disconnected input branch")
            print(f"DIRECT_FUSION_COMPONENT_OK policy={policy} params=661504", flush=True)
    finally:
        tf.keras.mixed_precision.set_global_policy(original_policy)


def smoke_model(cfg):
    from unittest.mock import patch
    import tensorflow as tf
    import train
    from check_rafdb_v5_lgsa_rdrop import smoke_model as rdrop_model_smoke

    build_model = train.build_model

    def checked_build(model_cfg):
        model = build_model(model_cfg)
        images = tf.random.normal([2, 112, 112, 3])
        out = model(images, training=False)
        assert model.use_global_regional_fusion
        assert not model.use_multistage_adaptive_fusion
        assert tuple(out["global_regional_features"].shape) == (2, 1024)
        assert tuple(out["global_features"].shape) == (2, 1024)
        assert all(tuple(v.shape) == (2, 512) for v in out["regional_features"])
        assert model.global_regional_fusion.count_params() == 661504
        alpha = tf.cast(out["adaptive_fusion_alpha"], tf.float32)
        tf.debugging.assert_near(out["logits"],
                                 (1 - alpha) * out["visual_logits"] + alpha * out["semantic_logits"])
        # Visual CE alone must reach DPA and the fusion projections. This checks
        # the new path without any gradient supplied by semantic/LGSA losses.
        variables = (model.dynamic_part_attn.trainable_variables
                     + model.global_regional_fusion.trainable_variables)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            outputs = model(images, training=False)
            visual_ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                tf.constant([1, 4]), outputs["visual_logits"], from_logits=True,
            ))
        gradients = tape.gradient(visual_ce, variables)
        count = len(model.dynamic_part_attn.trainable_variables)
        check_gradients(tf, gradients[:count], "Visual CE -> DPA")
        check_gradients(tf, gradients[count:], "Visual CE -> fusion")
        print("DIRECT_LOCAL_TO_VISUAL_HEAD_OK: shapes, fusion params, "
              "visual-CE-only gradients to DPA/fusion, adaptive semantic fusion", flush=True)
        return model

    with patch.object(train, "build_model", side_effect=checked_build):
        rdrop_model_smoke(cfg)
    print("V6_GLOBAL_REGIONAL_LGSA_RDROP_MODEL_SMOKE_OK; full training untested", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / ("config_" + NAME + ".yaml")))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()
    cfg = check_contract(args.config)
    if args.smoke:
        smoke_components()
    if args.model_smoke:
        smoke_model(cfg)


if __name__ == "__main__":
    main()
