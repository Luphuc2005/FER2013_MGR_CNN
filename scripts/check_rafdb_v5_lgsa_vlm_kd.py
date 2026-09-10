#!/usr/bin/env python3
"""V5 + LGSA + VLM-KD: static parity, CPU regression, or real GPU SAM smoke."""
import argparse
import copy
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config

BASE = "rafdb_siglip2_semantic_stable_v5_lgsa"
NAME = BASE + "_vlm_kd"


def check_contract(path):
    if any(k.startswith("MGR_") for k in os.environ):
        raise ValueError("Unset MGR_* overrides")
    cfg = load_config(path)
    base = load_config(ROOT / ("config_" + BASE + ".yaml"))
    variants = {
        NAME: ("adamw", 0.05),
        NAME + "_adamw_sam": ("adamw", 0.05),
        NAME + "_adam_sam": ("adam", 0.0),
    }
    name = cfg["model"]["name"]
    assert name in variants, "Unknown KD lineage"
    expected_optimizer, expected_decay = variants[name]
    from utils.optimizer_config import resolve_base_optimizer
    assert resolve_base_optimizer(cfg["training"]) == (expected_optimizer, expected_decay)
    comparable = copy.deepcopy(cfg)
    comparable["training"]["base_optimizer"] = base["training"]["base_optimizer"]
    comparable["training"]["weight_decay"] = base["training"]["weight_decay"]
    comparable["source"] = base["source"]
    comparable["model"]["name"] = base["model"]["name"]
    comparable["paths"]["output_dir"] = base["paths"]["output_dir"]
    for key in ("lambda_rdrop", "lambda_vlm_kd", "kd_temperature"):
        del comparable["training"][key]
    teacher = comparable.pop("vlm_teacher")
    assert comparable == base, "Unexpected change to V5+LGSA recipe"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers" / name
    weight = float(cfg["training"]["lambda_vlm_kd"])
    assert math.isfinite(weight) and weight >= 0
    assert cfg["training"]["kd_temperature"] == 2.0
    assert cfg["training"]["lambda_rdrop"] == 0.0
    assert teacher == dict(model_name="google/siglip2-base-patch16-224",
                           cache_path="pretrained/rafdb_v5_lgsa_vlm_kd_train.npz",
                           batch_size=32)
    print(f"V5_LGSA_VLM_KD_CONFIG_OK lambda={weight} T=2 "
          f"base={expected_optimizer} wd={expected_decay}; remaining baseline recipe retained", flush=True)
    return cfg



def smoke_optimizers():
    """Real TensorFlow factory check, including observable weight decay."""
    import numpy as np
    import tensorflow as tf
    from train import build_optimizer
    for name, decay in (("adam", 0.0), ("adamw", 0.05)):
        optimizer = build_optimizer(
            {"training": {"base_optimizer": name, "weight_decay": decay}}, .01)
        if name == "adam":
            assert type(optimizer) is tf.keras.optimizers.Adam
        else:
            assert "adamw" in type(optimizer).__name__.lower()
        variable = tf.Variable([1., -2.], dtype=tf.float32)
        before = variable.numpy().copy()
        # With zero data gradient, plain Adam stays fixed; AdamW decays weights.
        optimizer.apply_gradients([(tf.zeros_like(variable), variable)])
        np.testing.assert_allclose(variable.numpy(), before * (1. - .01 * decay),
                                   rtol=1e-6, atol=1e-7)
        tf.debugging.assert_all_finite(variable, "Optimizer update")
    print("BASE_OPTIMIZER_SMOKE_OK: native Adam versus AdamW, zero-gradient decay", flush=True)


def smoke_components():
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    smoke_optimizers()
    from losses.vlm_kd import classification_kd, VLMKDMetrics
    from losses.rdrop import forward_training_loss
    from losses.classification import supervised_mgr_loss
    from train import make_step_function
    from unittest.mock import patch

    for dtype in (tf.float32, tf.float16):
        s = tf.Variable([[2., -1., 3.], [0., 2., -2.]], dtype=dtype)
        t = tf.Variable([[0., 3., -1.], [3., 0., 1.]], dtype=dtype)
        with tf.GradientTape(persistent=True) as tape:
            kd = classification_kd(s, t, 2.)
        gs, gt = tape.gradient(kd, s), tape.gradient(kd, t)
        assert gs is not None and gt is None
        assert float(tf.linalg.global_norm([gs])) > 0
        a, b = t.numpy().astype(np.float64)/2, s.numpy().astype(np.float64)/2
        a -= np.log(np.exp(a).sum(1, keepdims=True))
        b -= np.log(np.exp(b).sum(1, keepdims=True))
        np.testing.assert_allclose(kd, 4 * np.mean(np.sum(np.exp(a)*(a-b), axis=1)), rtol=1e-5)
        np.testing.assert_array_equal(classification_kd(s, s, 2.), 0.)
        tf.debugging.assert_all_finite(classification_kd(s*1000, -s*1000), "KL overflow")

    class Probe(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.backbone = tf.keras.layers.Dense(4, activation="tanh")
            self.head = tf.keras.layers.Dense(7)
            self.calls = 0

        def call(self, features, training=False):
            assert "teacher_logits" not in features, "Target leaked into student"
            self.calls += int(training)
            logits = self.head(self.backbone(features["image"]))
            return dict(logits=logits, semantic_logits=logits, lambda_sem=.1,
                        s_upper=logits, s_lower=logits, s_au=logits, lambda_local_sem=.02)

    tf.keras.utils.set_random_seed(42)
    model = Probe()
    clean = {"image": tf.constant([[1., 2., -1., 0.], [0., 1., 2., -2.]])}
    labels = tf.constant([1, 4])
    model(clean)
    features = dict(clean, teacher_logits=tf.constant([[5.,0.,0.,0.,0.,0.,0.]]*2))
    kw = dict(num_classes=7, label_smoothing=.1, ortho_weight=0., cnn_aux_weight=0.)
    with tf.GradientTape() as tape:
        _, zero, _ = forward_training_loss(model, clean, labels, lambda_vlm_kd=0., **kw)
    zero_g = tape.gradient(zero, model.trainable_variables)
    with tf.GradientTape() as tape:
        baseline, _ = supervised_mgr_loss(labels, model(clean, training=True), **kw)
    base_g = tape.gradient(baseline, model.trainable_variables)
    np.testing.assert_array_equal(zero, baseline)
    for a, b in zip(zero_g, base_g):
        np.testing.assert_array_equal(a, b)
    _, total, parts = forward_training_loss(model, features, labels, lambda_vlm_kd=.05, **kw)
    np.testing.assert_allclose(total - baseline, parts["weighted_vlm_kd"], rtol=1e-5, atol=1e-6)
    with tf.GradientTape() as tape:
        _, total, _ = forward_training_loss(model, features, labels, lambda_vlm_kd=.05, **kw)
    for g in tape.gradient(total, model.trainable_variables):
        assert g is not None
        tf.debugging.assert_all_finite(g, "KD total gradient")
    for key, bad in (("lambda_vlm_kd", -1.), ("kd_temperature", 0.), ("lambda_rdrop", .5)):
        kwargs = dict(lambda_vlm_kd=.05, kd_temperature=2., lambda_rdrop=0.)
        kwargs[key] = bad
        try:
            forward_training_loss(model, features, labels, **kwargs, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid settings accepted")

    # Exercise the real SAM implementation; check both objective calls and
    # frozen targets, perturbation between calls, and once-per-batch metrics.
    cfg = dict(training=dict(optimizer="sam", sam_rho=.02, lambda_vlm_kd=.05,
                             kd_temperature=2., lambda_rdrop=0., skip_nonfinite_batches=False,
                             label_smoothing=.1),
               model=dict(ortho_loss_weight=0., cnn_aux_loss_weight=0.), data=dict(num_classes=7))
    seen = []
    def spy(*args, **kwargs):
        out = forward_training_loss(*args, **kwargs)
        seen.append((model.head.kernel.numpy().copy(), kwargs["lambda_vlm_kd"],
                     out[2]["weighted_vlm_kd"].numpy()))
        return out
    tracker = VLMKDMetrics()
    head, full = make_step_function(cfg, model, tf.keras.optimizers.SGD(.01),
                                   tf.keras.optimizers.SGD(.01), vlm_kd_metrics=tracker)
    before = model.head.kernel.numpy().copy()
    target_before = features["teacher_logits"].numpy().copy()
    model.calls = 0
    with patch("train.forward_training_loss", side_effect=spy):
        full(features, labels)
    assert model.calls == 2 and len(seen) == 2
    assert all(w == .05 and loss > 0 for _, w, loss in seen)
    assert not np.array_equal(seen[0][0], seen[1][0]), "Missing SAM perturbation"
    assert not np.array_equal(before, model.head.kernel.numpy()), "No optimizer update"
    np.testing.assert_array_equal(target_before, features["teacher_logits"])
    assert tracker.snapshot()["vlm_kd_samples"] == 2
    print("VLM_KD_COMPONENT_OK: direction/T^2, frozen teacher, zero loss/gradient parity, "
          "two SAM forwards with KD, metrics counted once", flush=True)


def smoke_model(cfg):
    import numpy as np
    import tensorflow as tf
    from datasets.fer2013 import build_datasets
    from utils.vlm_teacher_cache import read_records, CLASSES
    from losses.rdrop import forward_training_loss
    from losses.vlm_kd import VLMKDMetrics
    from train import (configure_tensorflow_runtime, configure_gpus, build_model,
                       build_optimizer, split_variables, ensure_optimizer_built,
                       make_step_function, make_distributed_train_step)
    configure_tensorflow_runtime(cfg)
    tf.keras.utils.set_random_seed(cfg["seed"]["random_seed"])
    configure_gpus(cfg)
    devices = tf.config.list_logical_devices("GPU")
    if not devices:
        raise RuntimeError("Real model smoke requires allocated GPU")
    strategy = tf.distribute.MirroredStrategy(devices=[devices[0].name])
    for split in ("train", "val"):
        records = read_records(cfg, split)
        print(f"SMOKE_SPLIT {split} n={len(records.labels)} classes={CLASSES} "
              f"label_range=[0,6] counts={np.bincount(records.labels, minlength=7).tolist()}", flush=True)
    train_ds, _, _ = build_datasets(cfg, replicas=1)
    features, labels = next(iter(train_ds.take(1)))
    batch = cfg["runtime"]["batch_size_per_gpu"]
    assert tuple(features["image"].shape) == (batch, 112, 112, 3)
    enabled = cfg["training"]["lambda_vlm_kd"] > 0
    if enabled:
        assert tuple(features["teacher_logits"].shape) == (batch, 7)
    clean = {k: v for k, v in features.items() if k != "teacher_logits"}
    with strategy.scope():
        model = build_model(cfg)
        out = model(clean, training=False)
        assert model.pretrained_load_status == "loaded"
        assert not model.use_global_regional_fusion and not model.use_multistage_adaptive_fusion
        for key in ("logits", "s_upper", "s_lower", "s_au"):
            assert tuple(out[key].shape) == (batch, 7)
        count = model.count_params()
        # Cache has no student weights and no effect on deterministic inference.
        np.testing.assert_array_equal(out["logits"], model(features, training=False)["logits"])
        backbone, heads = split_variables(model)
        opt_h = build_optimizer(cfg, cfg["training"]["lr"])
        opt_b = build_optimizer(cfg, cfg["training"]["visual_extractor_lr"])
        for optimizer in (opt_h, opt_b):
            if cfg["training"].get("base_optimizer", "adamw") == "adam":
                assert type(optimizer) is tf.keras.optimizers.Adam
            else:
                assert "adamw" in type(optimizer).__name__.lower()
        ensure_optimizer_built(opt_h, heads, strategy)
        ensure_optimizer_built(opt_b, backbone, strategy)
        tracker = VLMKDMetrics()

    def check_gradients():
        # Entire GradientTape and variable access stay INSIDE replica context.
        variables = heads
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(variables)
            _, total, parts = forward_training_loss(
                model, features, labels, lambda_vlm_kd=cfg["training"]["lambda_vlm_kd"],
                kd_temperature=2., num_classes=7, label_smoothing=.1,
                ortho_weight=0., cnn_aux_weight=0.)
            local = parts["local_semantic"]
            kd = parts.get("vlm_kd", tf.constant(0.))
        local_vars = (model.visual_projector_upper.trainable_variables
                      + model.visual_projector_lower.trainable_variables
                      + model.visual_projector_au.trainable_variables
                      + model.dynamic_part_attn.trainable_variables)
        gradients = tape.gradient(local, local_vars)
        assert all(g is not None for g in gradients)
        for g in gradients:
            tf.debugging.assert_all_finite(g, "LGSA gradient")
        tf.debugging.assert_positive(tf.linalg.global_norm(gradients))
        if enabled:
            gradients = tape.gradient(kd, model.classifier.trainable_variables)
            assert all(g is not None for g in gradients)
            for g in gradients:
                tf.debugging.assert_all_finite(g, "KD classifier gradient")
            tf.debugging.assert_positive(tf.linalg.global_norm(gradients))
        tf.debugging.assert_all_finite(total, "Training objective")
        return total
    strategy.run(check_gradients)
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["training"]["skip_nonfinite_batches"] = False
    head, full = make_step_function(smoke_cfg, model, opt_h, opt_b, vlm_kd_metrics=tracker)
    ds = tf.data.Dataset.from_tensors((features, labels))
    distributed_batch = next(iter(strategy.experimental_distribute_dataset(ds)))
    for phase, step in (("head", head), ("full", full)):
        tracker.reset_state()
        old = model.classifier.kernel.numpy().copy()
        result = make_distributed_train_step(strategy, step)(distributed_batch)
        tf.debugging.assert_all_finite(result[0], "SAM objective")
        for variable in model.trainable_variables:
            tf.debugging.assert_all_finite(variable, variable.name)
        assert not np.array_equal(old, model.classifier.kernel.numpy())
        assert tracker.snapshot()["vlm_kd_samples"] == (batch if enabled else 0)
        assert model.count_params() == count
        print(f"VLM_KD_SAM_{phase.upper()}_OK " + str(tracker.snapshot()), flush=True)
    print(f"V5_LGSA_VLM_KD_MODEL_SMOKE_OK params={count} batch={batch}; "
          "no checkpoint saved; accuracy unmeasured", flush=True)


if __name__ == "__main__":
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
