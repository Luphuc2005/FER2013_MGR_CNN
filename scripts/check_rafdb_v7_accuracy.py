#!/usr/bin/env python3
"""Validate the isolated V7 recipe; --runtime checks its real TensorFlow paths.

No optimizer steps, checkpoint writes, or training are performed here.
The launcher separately runs the standard RAF-DB dataset/pretrained smoke.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config

NAME = "rafdb_v7_accuracy_direct_lgsa"


def check_config(path):
    if any(key.startswith("MGR_") for key in os.environ):
        raise ValueError("Unset MGR_* overrides before checking V7.")
    cfg = load_config(path)
    m, t, r, d = (cfg[key] for key in ("model", "training", "runtime", "data"))
    assert m["name"] == NAME
    assert Path(cfg["paths"]["output_dir"]).resolve() == ROOT / "outputs" / "papers" / NAME
    assert not cfg["paths"]["auto_increment"] and not t["resume"]
    assert m["checkpoint_path"] is None and m["convnext_base_require_pretrained"]
    assert m["arch"] == "convnext_base_ms1m_arcface"
    assert m["use_global_regional_fusion"] and m["use_dynamic_part_attention"]
    assert m["global_regional_projection_dim"] == 256
    assert not m["use_soft_regional_pooling"] and not m["use_multistage_adaptive_fusion"]
    assert not m["use_adaptive_fusion_gate"] and m["semantic_fusion_alpha"] == 0.0
    assert m["use_adaptive_granularity"]
    assert m["granularity_gate_schedule"] == dict(
        enabled=True, uniform_epochs=4, transition_end_epoch=12,
        start_temperature=2.5, end_temperature=1.0,
    )
    for key, value in dict(lambda_local_sem=0.02, lambda_sem=0.10,
                           lambda_hard=0.05, semantic_fusion_alpha=0.0).items():
        assert m[key] == m["clip_semantic"][key] == value, key
    assert not m["progressive_unfreeze"]["enabled"]
    assert m["freeze_backbone_epochs"] == 4 and m["unfreeze_backbone"]
    assert t["optimizer"] == "sam" and t["base_optimizer"] == "adamw"
    assert t["visual_extractor_lr"] == 1e-5 and t["lr"] == 3e-4
    assert t["sam_rho"] == 0.02 and t["weight_decay"] == 0.05
    assert t["loss"] == "cross_entropy" and not t["logit_adjustment"]["enabled"]
    assert not t["lambda_sem_schedule"]["enabled"] and t["label_smoothing"] == 0.1
    assert t["monitor"] == "val_accuracy" and t["save_best_macro_f1"]
    assert r["train_val_tta_hflip"] and r["eval_tta_hflip"]
    assert cfg["tta"] == dict(enabled=True, hflip=True, original_weight=0.5, flip_weight=0.5)
    assert r["batch_size_per_gpu"] == 16 and r["gpu_ids"] == [0]
    assert d["num_classes"] == 7 and d["image_size"] == 112 and d["channels"] == 3
    assert d["mask_dir"] is None and d["sampling_strategy"] == "natural"
    print("V7_CONFIG_OK target=0.93 status=UNVERIFIED selection=val_accuracy_HFlip50", flush=True)
    return cfg


def check_runtime(cfg):
    import numpy as np
    import tensorflow as tf
    from train import build_model, configure_gpus, configure_tensorflow_runtime
    from losses.classification import supervised_mgr_loss

    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)
    tf.keras.utils.set_random_seed(cfg["seed"]["random_seed"])
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(cfg)
    images = tf.random.uniform([2, 112, 112, 3], minval=-1.0, maxval=1.0)
    labels = tf.constant([0, 2], tf.int32)
    model.set_granularity_gate_epoch(1)
    outputs = model(images, training=False)
    assert tuple(outputs["global_features"].shape) == (2, 1024)
    assert tuple(outputs["global_regional_features"].shape) == (2, 1024)
    assert all(tuple(z.shape) == (2, 512) for z in outputs["regional_features"])
    assert tuple(outputs["logits"].shape) == (2, 7)
    np.testing.assert_allclose(outputs["logits"], outputs["visual_logits"], atol=0, rtol=0)
    assert "adaptive_fusion_alpha" not in outputs

    # Same graph sees epoch changes; uniform warm-up must not be baked into tracing.
    pooled = tf.cast(outputs["global_features"], tf.float32)
    @tf.function(jit_compile=False)
    def weights():
        return model._granularity_gate_weights(pooled, training=False)

    for epoch in (1, 4, 5, 11, 12):
        model.set_granularity_gate_epoch(epoch)
        w = weights()
        tf.debugging.assert_all_finite(w, "Non-finite gate weights")
        np.testing.assert_allclose(tf.reduce_sum(w, axis=-1), [1.0, 1.0], atol=1e-6)
        if epoch <= 4:
            np.testing.assert_allclose(w, np.full((2, 5), 0.2), atol=1e-7)
        elif epoch < 12:
            beta = (epoch - 4) / 8.0
            assert float(tf.reduce_min(w)) >= (1.0 - beta) * 0.2 - 1e-6
        else:
            expected = tf.nn.softmax(tf.cast(model.granularity_gate(pooled, training=False), tf.float32))
            np.testing.assert_allclose(w, expected, atol=1e-6)

    model.set_granularity_gate_epoch(5)
    # Watch only head variables to keep this path check inexpensive on the server.
    groups = {"dynamic_parts": model.dynamic_part_attn.trainable_variables}
    groups.update({f"fusion_{name}": layer.trainable_variables for name, layer in
                   zip(("global", "upper", "lower", "au"), model.global_regional_fusion.projections)})
    watched = [v for variables in groups.values() for v in variables]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(watched)
        outputs = model(images, training=False)
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, outputs["logits"], from_logits=True))
    grads = tape.gradient(ce, watched)
    offset = 0
    for name, variables in groups.items():
        group_grads = grads[offset:offset + len(variables)]
        offset += len(variables)
        assert all(g is not None for g in group_grads), f"Disconnected FER CE: {name}"
        for g in group_grads:
            tf.debugging.assert_all_finite(g, f"Non-finite FER CE gradient: {name}")
        norm = float(tf.linalg.global_norm([tf.cast(g, tf.float32) for g in group_grads]))
        assert norm > 0.0, f"Zero FER CE gradient: {name}"
        print(f"V7_CE_GRADIENT_OK group={name} norm={norm:.6g}", flush=True)

    kw = dict(num_classes=7, label_smoothing=0.1, ortho_weight=0.0, cnn_aux_weight=0.0)
    total, parts = supervised_mgr_loss(labels, outputs, **kw)
    without_local, _ = supervised_mgr_loss(labels, dict(outputs, lambda_local_sem=0.0), **kw)
    tf.debugging.assert_all_finite(total, "Non-finite supervised loss")
    assert float(parts["local_semantic"]) > 0
    np.testing.assert_allclose(float(total - without_local),
                               0.02 * float(parts["local_semantic"]), atol=1e-6, rtol=1e-4)
    print("V7_RUNTIME_PATHS_OK dtype=mixed_float16 direct_CE=connected LGSA=weighted gate=scheduled", flush=True)
    print("No optimizer step or training was performed by this checker.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"config_{NAME}.yaml")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    cfg = check_config(args.config)
    if args.runtime:
        check_runtime(cfg)


if __name__ == "__main__":
    main()
