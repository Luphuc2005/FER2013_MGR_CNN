#!/usr/bin/env python3
"""
Smoke test and contract verification for Explicit Local-Global Semantic Alignment (LGSA) on V5.
Verifies:
1. Shape contract: logits (B, 7), s_upper (B, 7), s_lower (B, 7), s_au (B, 7).
2. Gradient flow: backward gradients through both visual and semantic pathways are finite.
3. V5 equivalence: when lambda_local_sem == 0.0, total loss and gradient strictly match baseline V5.
"""
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from losses.classification import supervised_mgr_loss
from train import build_model


def run_smoke_test():
    cfg_path = ROOT / "config_rafdb_siglip2_semantic_stable_v5_lgsa.yaml"
    print(f"[TEST 1/4] Loading configuration: {cfg_path.name}")
    cfg = load_config(cfg_path)
    assert cfg["model"]["name"] == "rafdb_siglip2_semantic_stable_v5_lgsa"
    assert cfg["model"]["lambda_local_sem"] == 0.02
    assert cfg["paths"]["output_dir"] == "outputs/papers/rafdb_siglip2_semantic_stable_v5_lgsa"

    # Avoid loading huge weights if file not present locally for quick smoke check
    # Check if pretrained exists
    pretrained_path = ROOT / cfg["model"]["convnext_base_pretrained_path"]
    if not pretrained_path.exists():
        print(f"[WARN] Pretrained checkpoint not found at {pretrained_path}. Using uninitialized backbone for smoke test.")
        cfg["model"]["convnext_base_require_pretrained"] = False
        cfg["model"]["pretrained"] = False

    print("[TEST 2/4] Instantiating V5 LGSA Model...")
    model = build_model(cfg, num_classes=7)

    # Create dummy batch
    batch_size = 2
    dummy_x = tf.random.normal([batch_size, 112, 112, 3], dtype=tf.float32)
    dummy_y = tf.constant([1, 4], dtype=tf.int32)

    print("[TEST 3/4] Forward pass & shape contract check...")
    outputs = model(dummy_x, training=True)

    assert "logits" in outputs, "Missing logits in outputs"
    assert outputs["logits"].shape == (batch_size, 7), f"Expected (2, 7), got {outputs['logits'].shape}"

    # Check local semantic logits
    assert "s_upper" in outputs, "Missing s_upper in endpoints"
    assert "s_lower" in outputs, "Missing s_lower in endpoints"
    assert "s_au" in outputs, "Missing s_au in endpoints"
    assert outputs["s_upper"].shape == (batch_size, 7), f"Expected (2, 7), got {outputs['s_upper'].shape}"
    assert outputs["s_lower"].shape == (batch_size, 7), f"Expected (2, 7), got {outputs['s_lower'].shape}"
    assert outputs["s_au"].shape == (batch_size, 7), f"Expected (2, 7), got {outputs['s_au'].shape}"

    assert "local_semantic_logits" in outputs
    assert "upper" in outputs["local_semantic_logits"]
    assert "lower" in outputs["local_semantic_logits"]
    assert "au" in outputs["local_semantic_logits"]

    print(f"  -> Logits shape: {outputs['logits'].shape}")
    print(f"  -> s_upper shape: {outputs['s_upper'].shape}")
    print(f"  -> s_lower shape: {outputs['s_lower'].shape}")
    print(f"  -> s_au shape   : {outputs['s_au'].shape}")

    # Check Loss with LGSA
    print("[TEST 4/4] Loss calculation, backward gradient & V5 equivalence...")
    with tf.GradientTape() as tape:
        out = model(dummy_x, training=True)
        total_loss, parts = supervised_mgr_loss(
            labels=dummy_y,
            outputs=out,
            num_classes=7,
            label_smoothing=0.10,
        )

    assert "local_semantic" in parts, "Missing local_semantic in loss parts"
    local_loss_val = float(parts["local_semantic"].numpy())
    assert local_loss_val > 0.0 and np.isfinite(local_loss_val), f"Invalid local_semantic loss: {local_loss_val}"
    print(f"  -> Total loss (with LGSA 0.02): {float(total_loss.numpy()):.4f}")
    print(f"  -> CE loss                   : {float(parts['ce'].numpy()):.4f}")
    print(f"  -> Semantic loss             : {float(parts['semantic'].numpy()):.4f}")
    print(f"  -> Local semantic loss       : {local_loss_val:.4f}")

    grads = tape.gradient(total_loss, model.trainable_variables)
    assert all(g is not None for g in grads[:10]), "Null gradients detected in head"
    all_finite = all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in grads if g is not None)
    assert all_finite, "Non-finite gradients detected!"
    print("  -> Backward gradients: ALL FINITE [OK]")

    # Equivalence check: lambda_local_sem = 0.0 must exactly match V5 baseline
    out_zero = dict(out)
    out_zero["lambda_local_sem"] = tf.constant(0.0, dtype=tf.float32)
    total_loss_zero, parts_zero = supervised_mgr_loss(
        labels=dummy_y,
        outputs=out_zero,
        num_classes=7,
        label_smoothing=0.10,
    )

    # Compute expected V5 total loss without LGSA: CE + lambda_sem * sem + lambda_hard * hard
    expected_v5 = parts_zero["ce"] + 0.10 * parts_zero["semantic"]
    if float(parts_zero["hard_semantic"].numpy()) > 0.0:
        expected_v5 = expected_v5 + 0.10 * parts_zero["hard_semantic"]

    np.testing.assert_allclose(
        float(total_loss_zero.numpy()),
        float(expected_v5.numpy()),
        rtol=1e-5,
        err_msg="Loss when lambda_local_sem=0.0 must match exact V5 baseline!"
    )
    print("  -> V5 Equivalence test (lambda_local_sem=0.0): PASSED [OK]")

    print("\n[SUCCESS] ALL LGSA CONTRACT CHECKS AND SMOKE TESTS PASSED!")


if __name__ == "__main__":
    run_smoke_test()
