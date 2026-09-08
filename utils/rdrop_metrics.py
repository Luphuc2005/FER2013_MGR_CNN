"""Training-only diagnostics, counted once per batch (not four times for SAM)."""
from __future__ import annotations

import tensorflow as tf


class RDropMetrics(tf.Module):
    def __init__(self, name="rdrop_train"):
        super().__init__(name=name)
        self.means = {
            key: tf.keras.metrics.Mean(name=name + "_" + key, dtype="float32")
            for key in ("rdrop", "weighted_rdrop", "local_semantic", "weighted_local_semantic")
        }
        self.samples = tf.keras.metrics.Sum(name=name + "_samples", dtype="float32")

    def update_state(self, parts, batch_count):
        count = tf.cast(batch_count, tf.float32)
        for key, metric in self.means.items():
            metric.update_state(tf.stop_gradient(parts[key]), sample_weight=count)
        self.samples.update_state(count)

    def reset_state(self):
        for metric in list(self.means.values()) + [self.samples]:
            metric.reset_state()

    def snapshot(self):
        result = {
            "train_" + key + "_loss": float(metric.result().numpy())
            for key, metric in self.means.items()
        }
        result["rdrop_train_samples"] = int(self.samples.result().numpy())
        return result


def format_rdrop(values):
    return (
        f"local_sem={values['train_local_semantic_loss']:.6f} "
        f"weighted_local_sem={values['train_weighted_local_semantic_loss']:.6f} "
        f"rdrop_kl={values['train_rdrop_loss']:.6f} "
        f"weighted_rdrop={values['train_weighted_rdrop_loss']:.6f} "
        f"n={values['rdrop_train_samples']}"
    )
