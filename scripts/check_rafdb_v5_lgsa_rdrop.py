#!/usr/bin/env python3
"""V5 + LGSA + R-Drop contract, component regression, and real one-batch smoke.

Default: static config parity only (no TensorFlow).
--smoke: synthetic component/SAM regression, no real model or dataset.
--model-smoke: configured pretrained model, real training batch, head/full SAM
steps on a throwaway model. No checkpoint writes or validation/test predictions.
Run the two smoke modes in separate processes.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config

BASE = "rafdb_siglip2_semantic_stable_v5_lgsa"
NAME = BASE + "_rdrop"


def check_contract(path):
    if any(key.startswith("MGR_") for key in os.environ):
        raise ValueError("Unset MGR_* overrides before checking this experiment.")
    base = load_config(ROOT / ("config_" + BASE + ".yaml"))
    cfg = load_config(path)
    weight = float(cfg["training"]["lambda_rdrop"])
    assert math.isfinite(weight) and weight >= 0
    assert cfg["model"]["name"] == NAME
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers" / NAME
    comparable = copy.deepcopy(cfg)
    comparable["source"] = copy.deepcopy(base["source"])
    comparable["model"]["name"] = base["model"]["name"]
    comparable["paths"]["output_dir"] = base["paths"]["output_dir"]
    del comparable["training"]["lambda_rdrop"]
    assert comparable == base, "Unexpected change beyond identity and lambda_rdrop."
    assert cfg["model"]["lambda_local_sem"] == 0.02
    assert cfg["model"]["clip_semantic"]["lambda_local_sem"] == 0.02
    assert not cfg["model"].get("use_multistage_adaptive_fusion", False)
    assert not cfg["model"].get("use_global_regional_fusion", False)
    assert cfg["training"]["optimizer"] == "sam"
    assert cfg["training"]["sam_rho"] == 0.02
    assert cfg["model"]["checkpoint_path"] is None
    assert not cfg["training"]["resume"]
    for name in (
        "train.py", "models/convnext_base_face_baseline.py",
        "losses/rdrop.py", "utils/rdrop_metrics.py",
        "scripts/check_rafdb_v5_lgsa_rdrop.py",
    ):
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    print(f"V5_LGSA_RDROP_CONFIG_OK lambda_rdrop={weight}; baseline recipe retained", flush=True)
    return cfg


def smoke_components():
    import numpy as np
    import tensorflow as tf
    from losses.classification import supervised_mgr_loss
    from losses.rdrop import forward_training_loss, symmetric_kl
    from train import make_step_function
    from utils.rdrop_metrics import RDropMetrics

    # CPU component fixture, independent from pretrained weights and the GPU job.
    tf.config.set_visible_devices([], "GPU")
    tf.keras.mixed_precision.set_global_policy("float32")
    a_np = np.array([[1, -2, 3], [-3, 2, 0]], np.float32)
    b_np = np.array([[-1, 1, 2], [0, -1, 3]], np.float32)
    def softmax(x):
        x = np.exp(x - x.max(-1, keepdims=True))
        return x / x.sum(-1, keepdims=True)
    p, q = softmax(a_np), softmax(b_np)
    expected = 0.5 * ((p * np.log(p / q)).sum(-1) + (q * np.log(q / p)).sum(-1)).mean()
    for dtype in (tf.float32, tf.float16):
        a, b = tf.Variable(a_np, dtype=dtype), tf.Variable(b_np, dtype=dtype)
        with tf.GradientTape() as tape:
            value = symmetric_kl(a, b)
        assert value.dtype == tf.float32
        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-5)
        for grad in tape.gradient(value, [a, b]):
            assert grad is not None
            tf.debugging.assert_all_finite(grad, "KL gradient")
            assert float(tf.linalg.global_norm([tf.cast(grad, tf.float32)])) > 0
        np.testing.assert_allclose(symmetric_kl(b, a).numpy(), value.numpy(), rtol=1e-6)
        np.testing.assert_array_equal(symmetric_kl(a, a).numpy(), 0.0)
        large = tf.constant([[10000., -10000., 0.]], dtype=dtype)
        tf.debugging.assert_all_finite(symmetric_kl(large, -large), "Large logits KL")

    class Probe(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.backbone = tf.keras.Sequential([tf.keras.layers.Dense(4, activation="tanh")])
            self.head = tf.keras.layers.Dense(7)
            self.regions = [tf.keras.layers.Dense(7) for _ in range(3)]
            self.calls = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.weight_trace = tf.Variable(tf.zeros([4, 4, 7]), trainable=False)
            self.input_trace = tf.Variable(tf.zeros([4, 2, 4]), trainable=False)

        def call(self, features, training=False):
            x = features["image"]
            h = self.backbone(x)
            if training:
                slot = self.calls.read_value()
                self.weight_trace.scatter_nd_update(tf.reshape(slot, [1, 1]), self.head.kernel[None])
                self.input_trace.scatter_nd_update(tf.reshape(slot, [1, 1]), x[None])
                # Reproducible stochastic masks make zero-weight comparisons exact.
                keep = tf.random.stateless_uniform(tf.shape(h), seed=tf.stack([42, slot])) > 0.4
                h = h * tf.cast(keep, h.dtype) / 0.6
                self.calls.assign_add(1)
            local = [layer(h) for layer in self.regions]
            semantic = tf.add_n(local) / 3.0
            return {
                "logits": 0.9 * self.head(h) + 0.1 * semantic,
                "semantic_logits": semantic,
                "lambda_sem": 0.10,
                "lambda_hard": 0.0,
                "s_upper": local[0], "s_lower": local[1], "s_au": local[2],
                "lambda_local_sem": 0.02,
            }

    tf.keras.utils.set_random_seed(42)
    features = {"image": tf.constant([[1., 2., -1., 0.], [0., 1., 2., -2.]])}
    labels = tf.constant([1, 4])
    probe = Probe()
    probe(features, training=False)
    kwargs = dict(num_classes=7, label_smoothing=0.1, ortho_weight=0., cnn_aux_weight=0.)
    schedule = tf.Variable(0.15, trainable=False)

    with tf.GradientTape() as tape:
        first, zero_loss, zero_parts = forward_training_loss(
            probe, features, labels, lambda_rdrop=0.,
            lambda_sem_runtime=schedule, **kwargs,
        )
    zero_grads = tape.gradient(zero_loss, probe.trainable_variables)
    assert int(probe.calls) == 1, "Zero coefficient must not draw a second mask."
    probe.calls.assign(0)
    with tf.GradientTape() as tape:
        baseline_out = probe(features, training=True)
        baseline_out["lambda_sem"] = schedule.read_value()
        baseline_loss, _ = supervised_mgr_loss(labels, baseline_out, **kwargs)
    baseline_grads = tape.gradient(baseline_loss, probe.trainable_variables)
    np.testing.assert_array_equal(zero_loss.numpy(), baseline_loss.numpy())
    for actual, expected_grad in zip(zero_grads, baseline_grads):
        np.testing.assert_array_equal(actual.numpy(), expected_grad.numpy())

    probe.calls.assign(0)
    _, paired_loss, parts = forward_training_loss(
        probe, features, labels, lambda_rdrop=0.5,
        lambda_sem_runtime=schedule, **kwargs,
    )
    assert int(probe.calls) == 2
    probe.calls.assign(0)
    a = probe(features, training=True)
    b = probe(features, training=True)
    a["lambda_sem"] = b["lambda_sem"] = schedule.read_value()
    la, pa = supervised_mgr_loss(labels, a, **kwargs)
    lb, pb = supervised_mgr_loss(labels, b, **kwargs)
    kl = symmetric_kl(a["logits"], b["logits"])
    assert float(kl) > 0
    np.testing.assert_allclose(paired_loss.numpy(), (0.5 * (la + lb) + 0.5 * kl).numpy(), rtol=1e-6)
    np.testing.assert_allclose(parts["local_semantic"], 0.5 * (pa["local_semantic"] + pb["local_semantic"]))
    np.testing.assert_allclose(parts["weighted_local_semantic"], 0.02 * parts["local_semantic"])
    assert float(parts["local_semantic"]) > 0

    # Exercise the actual trainer under tf.function: two equal-weight forwards
    # before perturbation, then two equal-weight forwards at perturbed weights.
    for weight, expected_calls in ((0.0, 2), (0.5, 4)):
        probe.calls.assign(0)
        cfg = {
            "data": {"num_classes": 7},
            "model": {"ortho_loss_weight": 0., "cnn_aux_loss_weight": 0.},
            "training": {"optimizer": "sam", "sam_rho": 0.02, "lambda_rdrop": weight,
                         "label_smoothing": 0.1, "grad_clip_norm": 1.,
                         "skip_nonfinite_batches": False},
        }
        tracker = RDropMetrics()
        # LR=0 means the only temporary weight changes are the SAM perturbations.
        head_opt = tf.keras.optimizers.SGD(0.)
        back_opt = tf.keras.optimizers.SGD(0.)
        _, full = make_step_function(cfg, probe, head_opt, back_opt,
                                     lambda_sem_runtime=schedule, rdrop_metrics=tracker)
        before = [v.numpy().copy() for v in probe.trainable_variables]
        result = tf.function(full)(features, labels)
        assert int(probe.calls) == expected_calls
        tf.debugging.assert_all_finite(result[0], "SAM loss")
        trace = probe.weight_trace.numpy()[:expected_calls]
        for recorded in probe.input_trace.numpy()[:expected_calls]:
            np.testing.assert_array_equal(recorded, features["image"])
        if weight:
            np.testing.assert_array_equal(trace[0], trace[1])
            np.testing.assert_array_equal(trace[2], trace[3])
            assert not np.array_equal(trace[0], trace[2]), "SAM perturbation missing."
            assert tracker.snapshot()["rdrop_train_samples"] == 2
            assert tracker.snapshot()["train_weighted_rdrop_loss"] > 0
        else:
            assert tracker.snapshot()["rdrop_train_samples"] == 0
        for original, var in zip(before, probe.trainable_variables):
            np.testing.assert_allclose(var.numpy(), original, rtol=1e-6, atol=1e-7)
    print("RDROP_COMPONENT_SMOKE_OK: KL, both gradients, complete LGSA objective, "
          "zero parity, same inputs/weights, SAM 2/4 forwards, metrics counted once", flush=True)


def smoke_model(cfg):
    import numpy as np
    import tensorflow as tf
    from datasets.fer2013 import _resolve_split_csv_dir, build_datasets
    from losses.rdrop import forward_training_loss
    from train import (
        build_model, build_optimizer, configure_gpus, configure_tensorflow_runtime,
        ensure_optimizer_built, get_class_names, make_distributed_train_step,
        make_step_function, resolve_phase_lrs, split_variables,
    )
    from utils.rdrop_metrics import RDropMetrics, format_rdrop
    from utils.semantic_schedule import resolve_lambda_sem

    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(cfg["seed"]["random_seed"])
    configure_gpus(cfg)
    devices = tf.config.list_logical_devices("GPU")
    if not devices:
        raise RuntimeError("--model-smoke needs the configured GPU; no CPU fallback.")
    strategy = tf.distribute.MirroredStrategy(devices=[devices[0].name])
    classes = get_class_names(cfg)
    print("SMOKE_CLASSES " + repr(classes), flush=True)
    # Inspect normalized split metadata first; avoid the loader's legacy CSV
    # rewriting path. Validation/test labels are counted only, never optimized.
    data_dir = _resolve_split_csv_dir(Path(cfg["data"]["data_path"]))
    for split in ("train", "val", "test"):
        path = data_dir / (split + ".csv")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            key = next(k for k in ("emotion", "label", "target", "class", "y") if k in reader.fieldnames)
            labels_np = np.array([int(row[key]) for row in reader], dtype=np.int32)
        assert labels_np.size and labels_np.min() == 0 and labels_np.max() == 6, (
            "Smoke requires existing normalized RAF labels [0..6]; inspect " + str(path)
        )
        print(f"SMOKE_SPLIT {split} n={len(labels_np)} label_range=[0,6] "
              f"class_counts={np.bincount(labels_np, minlength=7).tolist()}", flush=True)
    train_ds, _, _ = build_datasets(cfg, replicas=1)
    features, labels = next(iter(train_ds.take(1)))
    batch_size = int(cfg["runtime"]["batch_size_per_gpu"])
    assert tuple(features["image"].shape) == (batch_size, 112, 112, 3)
    assert tuple(labels.shape) == (batch_size,)
    print(f"SMOKE_BATCH image={features['image'].shape} labels={labels.shape} "
          f"range=[{int(tf.reduce_min(labels))},{int(tf.reduce_max(labels))}]", flush=True)
    with strategy.scope():
        model = build_model(cfg)
        out = model(features, training=False)
        assert model.pretrained_load_status == "loaded"
        for key in ("logits", "s_upper", "s_lower", "s_au"):
            assert tuple(out[key].shape) == (batch_size, 7), key
        assert float(out["lambda_local_sem"]) > 0
        logits_before = out["logits"].numpy()
        old_lambda = model.lambda_local_sem
        try:
            model.lambda_local_sem = 0.
            np.testing.assert_array_equal(model(features, training=False)["logits"], logits_before)
        finally:
            model.lambda_local_sem = old_lambda
        parameter_count = model.count_params()
        backbone, heads = split_variables(model)
        opt_h = build_optimizer(cfg, cfg["training"]["lr"])
        opt_b = build_optimizer(cfg, cfg["training"]["visual_extractor_lr"])
        ensure_optimizer_built(opt_h, heads, strategy)
        ensure_optimizer_built(opt_b, backbone, strategy)
        schedule = tf.Variable(0.1, trainable=False)
        tracker = RDropMetrics()

    # Explicit local-gradient check before optimizers: catch disconnected LGSA
    # rather than merely observing a positive total loss.
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(heads)
        _, loss, parts = forward_training_loss(
            model, features, labels, lambda_rdrop=cfg["training"]["lambda_rdrop"],
            lambda_sem_runtime=schedule, num_classes=7,
            label_smoothing=cfg["training"]["label_smoothing"],
            ortho_weight=0., cnn_aux_weight=0.,
        )
        local_loss = parts["local_semantic"]
    local_vars = (model.visual_projector_upper.trainable_variables
                  + model.visual_projector_lower.trainable_variables
                  + model.visual_projector_au.trainable_variables
                  + model.dynamic_part_attn.trainable_variables)
    grads = tape.gradient(local_loss, local_vars)
    assert float(local_loss) > 0
    assert all(g is not None for g in grads), "LGSA disconnected from local branches."
    for grad in grads:
        tf.debugging.assert_all_finite(grad, "LGSA gradient")
    assert float(tf.linalg.global_norm(grads)) > 0
    tf.debugging.assert_all_finite(loss, "R-Drop objective")

    # Fail on nonfinite gradients in the throwaway smoke rather than masking
    # them. The saved experiment config and production skip setting stay intact.
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["training"]["skip_nonfinite_batches"] = False
    head_step, full_step = make_step_function(
        smoke_cfg, model, opt_h, opt_b,
        lambda_sem_runtime=schedule, rdrop_metrics=tracker,
    )
    one_batch = tf.data.Dataset.from_tensors((features, labels))
    distributed_batch = next(iter(strategy.experimental_distribute_dataset(one_batch)))
    for phase, epoch, step in (
        ("head", 0, head_step),
        ("full", cfg["model"]["freeze_backbone_epochs"], full_step),
    ):
        lr_h, lr_b = resolve_phase_lrs(cfg, epoch, phase == "full")
        opt_h.learning_rate.assign(lr_h)
        opt_b.learning_rate.assign(lr_b)
        schedule.assign(resolve_lambda_sem(cfg, epoch + 1))
        tracker.reset_state()
        old_kernel = model.classifier.kernel.numpy().copy()
        result = make_distributed_train_step(strategy, step)(distributed_batch)
        tf.debugging.assert_all_finite(result[0], phase + " SAM loss")
        for var in model.trainable_variables:
            tf.debugging.assert_all_finite(var, var.name)
        assert not np.array_equal(old_kernel, model.classifier.kernel.numpy()), "No optimizer update."
        expected_samples = batch_size if cfg["training"]["lambda_rdrop"] > 0 else 0
        assert tracker.snapshot()["rdrop_train_samples"] == expected_samples
        assert model.count_params() == parameter_count
        print(f"RDROP_SAM_{phase.upper()}_SMOKE_OK " + format_rdrop(tracker.snapshot()), flush=True)
    print(f"V5_LGSA_RDROP_MODEL_SMOKE_OK batch={batch_size} params={parameter_count}; "
          "real pretrained/prototypes, LGSA gradients, head/full SAM, inference contract. "
          "No checkpoints saved; full training/accuracy not measured.", flush=True)


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
