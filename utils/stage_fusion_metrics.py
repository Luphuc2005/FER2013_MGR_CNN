"""Sample-weighted fusion diagnostics; updates occur in the trainer, never forward."""
from __future__ import annotations

import csv
from pathlib import Path
import tensorflow as tf


class StageFusionMetrics(tf.Module):
    """Distributed-safe Keras accumulators for original, unperturbed model outputs."""
    def __init__(self, name):
        super().__init__(name=name)
        self.means = [tf.keras.metrics.Mean(name=f"{name}_s{s}", dtype="float32")
                      for s in (2, 3, 4)]
        self.squares = [tf.keras.metrics.Mean(name=f"{name}_s{s}_square", dtype="float32")
                        for s in (2, 3, 4)]
        self.entropy = tf.keras.metrics.Mean(name=f"{name}_entropy", dtype="float32")
        self.dominant = tf.keras.metrics.Mean(name=f"{name}_dominant", dtype="float32")
        self.alpha = tf.keras.metrics.Mean(name=f"{name}_alpha", dtype="float32")
        self.samples = tf.keras.metrics.Sum(name=f"{name}_samples", dtype="float32")

    def reset_state(self):
        for metric in self.means + self.squares + [
            self.entropy, self.dominant, self.alpha, self.samples,
        ]:
            metric.reset_state()

    def update_state(self, outputs):
        weights = tf.stop_gradient(tf.cast(outputs["stage_fusion_weights"], tf.float32))
        for i, (mean, square) in enumerate(zip(self.means, self.squares)):
            mean.update_state(weights[:, i])
            square.update_state(tf.square(weights[:, i]))
        entropy = -tf.reduce_sum(weights * tf.math.log(tf.maximum(weights, 1e-9)), axis=-1)
        self.entropy.update_state(entropy / tf.math.log(tf.constant(3.0)))
        self.dominant.update_state(tf.cast(tf.reduce_max(weights, axis=-1) > 0.9, tf.float32))
        self.samples.update_state(tf.cast(tf.shape(weights)[0], tf.float32))
        if outputs.get("adaptive_fusion_alpha") is not None:
            self.alpha.update_state(tf.cast(outputs["adaptive_fusion_alpha"], tf.float32))

    def snapshot(self):
        count = float(self.samples.result().numpy())
        result = {"stage_fusion_samples": int(count)}
        for stage, mean, square in zip((2, 3, 4), self.means, self.squares):
            m = float(mean.result().numpy()) if count else float("nan")
            variance = max(0.0, float(square.result().numpy()) - m * m) if count else float("nan")
            result[f"stage_fusion_s{stage}_mean"] = m
            result[f"stage_fusion_s{stage}_std"] = variance ** 0.5
        result["stage_fusion_entropy"] = float(self.entropy.result().numpy()) if count else float("nan")
        result["stage_fusion_dominant_fraction"] = float(self.dominant.result().numpy()) if count else float("nan")
        result["semantic_fusion_alpha_mean"] = float(self.alpha.result().numpy()) if count else float("nan")
        return result


def format_stage_fusion(values):
    stages = " ".join(
        f"S{s}={values[f'stage_fusion_s{s}_mean']:.4f}"
        f"+/-{values[f'stage_fusion_s{s}_std']:.4f}" for s in (2, 3, 4)
    )
    return (f"{stages} H_norm={values['stage_fusion_entropy']:.4f} "
            f"max_gt_0.9={values['stage_fusion_dominant_fraction']:.4f} "
            f"alpha_sem={values['semantic_fusion_alpha_mean']:.4f} "
            f"n={values['stage_fusion_samples']}")


def append_fusion_epoch(path, epoch, train_values, val_values):
    """Flush one scalar CSV row per epoch so logs survive an interrupted job."""
    path = Path(path)
    row = {"epoch": int(epoch)}
    for prefix, values in (("train", train_values), ("val_orig", val_values)):
        row.update({f"{prefix}_{key}": value for key, value in values.items()})
    has_header = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not has_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
