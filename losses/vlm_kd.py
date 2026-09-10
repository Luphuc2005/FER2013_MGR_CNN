"""Frozen-teacher classification distillation, with FP32 KL and no new weights."""
import math
import tensorflow as tf


def validate_kd_settings(training):
    weight = float(training.get("lambda_vlm_kd", 0.0))
    temperature = float(training.get("kd_temperature", 2.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("lambda_vlm_kd must be finite and nonnegative")
    if weight > 0:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("kd_temperature must be finite and positive")
        if float(training.get("lambda_rdrop", 0.0)) != 0:
            raise ValueError("This VLM-KD experiment excludes R-Drop")
    return weight, temperature


def classification_kd(student_logits, teacher_logits, temperature=2.0):
    student = tf.cast(student_logits, tf.float32)
    teacher = tf.stop_gradient(tf.cast(teacher_logits, tf.float32))
    tf.debugging.assert_equal(tf.shape(student), tf.shape(teacher))
    tf.debugging.assert_all_finite(teacher, "Invalid teacher logits")
    log_p = tf.nn.log_softmax(teacher / temperature, axis=-1)
    log_q = tf.nn.log_softmax(student / temperature, axis=-1)
    return temperature ** 2 * tf.reduce_mean(
        tf.reduce_sum(tf.exp(log_p) * (log_p - log_q), axis=-1))


class VLMKDMetrics(tf.Module):
    def __init__(self):
        super().__init__(name="vlm_kd_train")
        self.means = {key: tf.keras.metrics.Mean(name=key, dtype="float32")
                      for key in ("vlm_kd", "weighted_vlm_kd")}
        self.samples = tf.keras.metrics.Sum(dtype="float32", name="vlm_kd_samples")

    def update_state(self, parts, count):
        for key, metric in self.means.items():
            metric.update_state(tf.stop_gradient(parts[key]), sample_weight=tf.cast(count, tf.float32))
        self.samples.update_state(tf.cast(count, tf.float32))

    def reset_state(self):
        for metric in list(self.means.values()) + [self.samples]:
            metric.reset_state()

    def snapshot(self):
        result = {"train_" + k: float(v.result().numpy()) for k, v in self.means.items()}
        result["vlm_kd_samples"] = int(self.samples.result().numpy())
        return result
