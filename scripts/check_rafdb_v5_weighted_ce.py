#!/usr/bin/env python3
"""Offline contracts plus optional TensorFlow loss/gradient checks (no training)."""
from __future__ import annotations

import argparse
import ast
import copy
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ce_class_weights import weights_from_labels, configure_training_ce_weights, training_ce_kwargs

BASE = ROOT / "config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml"
VARIANT = BASE.with_name(BASE.stem + "_weighted_ce.yaml")


class Contracts(unittest.TestCase):
    def test_config_diff(self):
        base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
        variant = yaml.safe_load(VARIANT.read_text(encoding="utf-8"))
        expected = copy.deepcopy(base)
        expected["source"] = variant["source"]
        expected["paths"]["output_dir"] += "_weighted_ce"
        expected["training"].update(loss="weighted_cross_entropy", monitor="val_macro_f1",
                                    weighted_ce=dict(enabled=True, target_classes=["fear", "disgust"],
                                                     power=0.5, max_ratio=2.0))
        self.assertEqual(variant, expected)

    def test_train_frequency_normalization_and_targets(self):
        counts = [100, 10, 5, 400, 100, 100, 100]
        labels = np.repeat(np.arange(7), counts)
        report = weights_from_labels(labels, ["fear", "disgust"])
        w = np.array(report["weights"])
        self.assertAlmostEqual(float(np.average(w, weights=counts)), 1.0)
        np.testing.assert_array_equal(w[[0, 3, 4, 5, 6]], np.repeat(w[0], 5))
        self.assertGreater(w[1], w[0])
        self.assertGreater(w[2], w[0])
        self.assertLessEqual(max(w / w[0]), 2.0)
        shuffled = weights_from_labels(labels[::-1], ["fear", "disgust"])
        self.assertEqual(report["weights"], shuffled["weights"])

    def test_balanced_weights_are_one(self):
        report = weights_from_labels(np.tile(np.arange(7), 3), ["fear", "disgust"])
        np.testing.assert_array_equal(report["weights"], np.ones(7))

    def test_invalid_inputs_fail(self):
        for labels in ([], [0, 1], list(range(8)), [float(i) for i in range(7)]):
            with self.assertRaises(ValueError):
                weights_from_labels(labels, ["fear"])
        for kwargs in (dict(power=0), dict(power=float("nan")), dict(max_ratio=0)):
            with self.assertRaises(ValueError):
                weights_from_labels(np.arange(7), ["fear"], **kwargs)

    def test_runtime_resolution_and_disabled_baseline(self):
        base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
        self.assertEqual(training_ce_kwargs(base), {})
        configure_training_ce_weights(base, np.arange(7))
        self.assertNotIn("resolved_ce_weight_report", base["training"])
        variant = yaml.safe_load(VARIANT.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            training_ce_kwargs(variant)
        configure_training_ce_weights(variant, np.arange(7))
        self.assertEqual(training_ce_kwargs(variant)["class_weight_reduction"], "batch_mean")
        self.assertEqual(variant["training"]["resolved_ce_weight_report"]["source_split"], "train")

    def test_sam_and_evaluation_wiring(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        step = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_step_function")
        calls = [n for n in ast.walk(step) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "supervised_mgr_loss"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(any(k.arg is None and ast.unparse(k.value) == "ce_kwargs" for k in call.keywords))
        evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_dataset")
        self.assertNotIn("ce_kwargs", ast.unparse(evaluate))
        self.assertNotIn("class_weights", ast.unparse(evaluate))
        data = (ROOT / "datasets/fer2013.py").read_text(encoding="utf-8")
        self.assertIn('configure_training_ce_weights(cfg, records["train"].labels)', data)


    def test_macro_retention_and_final_restore_source(self):
        from utils.ranked_checkpoint_manager import RankedCheckpointManager

        class FakeCheckpoint:
            def write(self, prefix):
                Path(prefix + ".index").write_bytes(b"fixture")

        with tempfile.TemporaryDirectory() as folder:
            manager = RankedCheckpointManager(FakeCheckpoint(), Path(folder), 1, "val_macro_f1", "max")
            manager.consider(1, 0.80, {"val_accuracy": 0.95})
            manager.consider(2, 0.82, {"val_accuracy": 0.90})
            manager.consider(3, 0.79, {"val_accuracy": 0.99})
            manager.consider(4, 0.82, {"val_accuracy": 0.99})
            self.assertEqual(Path(manager.latest_checkpoint).name, "ckpt-2")
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn('selection_manager = macro_manager if macro_manager is not None else best_manager', source)
        self.assertIn('best_ckpt = selection_manager.latest_checkpoint or last_manager.latest_checkpoint', source)


def tensorflow_checks():
    import tensorflow as tf
    from losses.classification import supervised_mgr_loss

    labels = tf.constant([1, 2, 3], tf.int32)
    logits = tf.Variable(np.arange(21, dtype=np.float32).reshape(3, 7) / 10)
    sem_logits = tf.constant(np.arange(21, dtype=np.float32).reshape(3, 7) / 20)
    weights = tf.constant([0.8, 1.4, 1.6, 0.8, 0.8, 0.8, 0.8])
    outputs = dict(logits=logits, semantic_logits=sem_logits, lambda_sem=0.15,
                   agg_sim=tf.zeros([3, 7]), hard_pairs_matrix=tf.ones([7, 7]) - tf.eye(7),
                   lambda_hard=0.1, hard_margin=0.15)
    kwargs = dict(num_classes=7, label_smoothing=0.1, ortho_weight=0.0, cnn_aux_weight=0.0)
    plain, plain_parts = supervised_mgr_loss(labels, outputs, **kwargs)
    with tf.GradientTape() as tape:
        weighted, parts = supervised_mgr_loss(labels, outputs, class_weights=weights,
                                               class_weight_reduction="batch_mean", **kwargs)
    grad = tape.gradient(weighted, logits)
    targets = tf.one_hot(labels, 7) * 0.9 + 0.1 / 7
    ce_vector = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
    expected_ce = tf.reduce_mean(ce_vector * tf.gather(weights, labels))
    np.testing.assert_allclose(parts["ce"], expected_ce, rtol=1e-6)
    for key in ("semantic", "hard_semantic", "ortho", "cnn_aux"):
        np.testing.assert_array_equal(parts[key], plain_parts[key])
    np.testing.assert_allclose(weighted - plain, parts["ce"] - plain_parts["ce"], atol=1e-6)
    expected_grad = (tf.nn.softmax(logits) - targets) * tf.gather(weights, labels)[:, None] / 3
    np.testing.assert_allclose(grad, expected_grad, atol=1e-6)
    ones, _ = supervised_mgr_loss(labels, outputs, class_weights=tf.ones(7),
                                  class_weight_reduction="batch_mean", **kwargs)
    np.testing.assert_allclose(ones, plain, atol=1e-6)
    # Batch size 1 must still scale the gradient/loss; no normalization by batch weight sum.
    one_label = tf.constant([2])
    one_outputs = {"logits": logits[:1]}
    one_plain, _ = supervised_mgr_loss(one_label, one_outputs, **kwargs)
    one_weighted, _ = supervised_mgr_loss(one_label, one_outputs, class_weights=weights,
                                         class_weight_reduction="batch_mean", **kwargs)
    np.testing.assert_allclose(one_weighted, one_plain * weights[2], atol=1e-6)
    # Pre-existing callers retain sum-of-weights normalization.
    legacy, _ = supervised_mgr_loss(labels, {"logits": logits}, class_weights=weights, **kwargs)
    np.testing.assert_allclose(legacy, tf.reduce_sum(ce_vector * tf.gather(weights, labels)) /
                              tf.reduce_sum(tf.gather(weights, labels)), atol=1e-6)
    print("TF_WEIGHTED_CE_LOSS_GRADIENT_OK (synthetic tensors; no model training)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tensorflow", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.require_tensorflow:
        tensorflow_checks()
    else:
        print("OFFLINE_CONTRACTS_OK; TensorFlow loss/gradient checks not run. Use --require-tensorflow on HPC.")
