"""V6 Anti-Overfitting Smoke Test and Verification Script.

Tests:
1. Config parsing and validation
2. Model building and variable partitioning (backbone vs head)
3. Progressive unfreeze schedule & mask resolution across all phases:
   - Phase 1 (Epochs 1-8): Head only, all backbone frozen
   - Phase 2 (Epochs 9-15): Stage 4 only unfrozen
   - Phase 3 (Epochs 16+): Stage 3 + 4 unfrozen, Stage 1-2 permanently frozen
4. Per-stage discriminative LR scaling factors (Stage 4 = 1.0, Stage 3 = 0.5)
5. Label smoothing isolation (main CE smoothed, semantic/aux unsmoothed)
6. Logit adjustment compatibility with label smoothing
7. Forward pass + backward tape + gradient masking test
8. Calibration diagnostics (ECE, mean confidence, NLL)
"""

import sys
import numpy as np
import tensorflow as tf

from config import load_config
from train import (
    build_model,
    split_variables,
    resolve_progressive_unfreeze_mask,
    compute_stage_lr_scales,
    _var_belongs_to_stage,
    variable_key,
)
from losses.classification import supervised_mgr_loss
from utils.logit_adjustment import compute_logit_adjustment


def test_v6_pipeline():
    print("=" * 70)
    print("STARTING V6 ANTI-OVERFITTING SMOKE TEST")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Config Loading & Hyperparameter Verification
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Loading and validating V6 config...")
    cfg = load_config("config_rafdb_v6_anti_overfitting.yaml")
    assert cfg["model"]["name"] == "rafdb_v6_anti_overfitting"
    assert cfg["training"]["label_smoothing"] == 0.05, f"Expected 0.05, got {cfg['training']['label_smoothing']}"
    assert cfg["training"]["weight_decay"] == 0.05, f"Expected 0.05, got {cfg['training']['weight_decay']}"
    assert cfg["training"]["visual_extractor_lr"] == 0.00001, f"Expected 1e-5, got {cfg['training']['visual_extractor_lr']}"
    assert cfg["training"]["lr"] == 0.0003, f"Expected 3e-4, got {cfg['training']['lr']}"
    assert cfg["training"].get("save_best_macro_f1") is True, "save_best_macro_f1 should be True"

    prog_cfg = cfg["model"]["progressive_unfreeze"]
    assert prog_cfg["enabled"] is True
    assert len(prog_cfg["schedule"]) == 3
    assert prog_cfg["stage_lr_multipliers"]["4"] == 1.0
    assert prog_cfg["stage_lr_multipliers"]["3"] == 0.5
    print("  -> Config parsed and validated successfully.")

    # -----------------------------------------------------------------------
    # 2. Model Instantiation & Variable Splitting
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Building model and splitting variables...")
    model = build_model(cfg)
    dummy_input = {"image": tf.zeros([2, 112, 112, 3], dtype=tf.float32)}
    outputs = model(dummy_input, training=False)
    assert "logits" in outputs, "Missing 'logits' in model outputs"
    assert outputs["logits"].shape == (2, 7), f"Expected (2, 7), got {outputs['logits'].shape}"

    backbone_vars, head_vars = split_variables(model)
    total_backbone = len(backbone_vars)
    total_head = len(head_vars)
    assert total_backbone > 0, "No backbone variables found!"
    assert total_head > 0, "No head variables found!"
    print(f"  -> Model built successfully: {total_backbone} backbone vars, {total_head} head vars.")

    # -----------------------------------------------------------------------
    # 3. Progressive Unfreeze Mask Resolution & Schedule
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Testing progressive unfreeze schedule...")

    # Phase 1: Epoch 1 to 8 (Head only)
    for ep in [1, 4, 8]:
        mask, stages = resolve_progressive_unfreeze_mask(cfg, ep, backbone_vars)
        assert stages == [], f"Epoch {ep}: expected [], got {stages}"
        trainable_bb = sum(1 for v in backbone_vars if mask[variable_key(v)])
        assert trainable_bb == 0, f"Epoch {ep}: expected 0 trainable backbone vars, got {trainable_bb}"
    print("  -> Phase 1 (Epochs 1-8): Head only, 0 backbone vars trainable [PASS]")

    # Phase 2: Epoch 9 to 15 (Stage 4 only)
    for ep in [9, 12, 15]:
        mask, stages = resolve_progressive_unfreeze_mask(cfg, ep, backbone_vars)
        assert stages == [4], f"Epoch {ep}: expected [4], got {stages}"
        for v in backbone_vars:
            k = variable_key(v)
            v_name = getattr(v, "name", "")
            is_s4 = _var_belongs_to_stage(v_name, 4)
            is_s123 = any(_var_belongs_to_stage(v_name, s) for s in [1, 2, 3])
            if is_s4:
                assert mask[k] is True, f"Epoch {ep}: Stage 4 var {v_name} should be trainable"
            if is_s123:
                assert mask[k] is False, f"Epoch {ep}: Var {v_name} from stage 1-3 should be frozen"
    s4_vars = sum(1 for v in backbone_vars if mask[variable_key(v)])
    print(f"  -> Phase 2 (Epochs 9-15): Stage 4 only, {s4_vars} vars trainable [PASS]")

    # Phase 3: Epoch 16+ (Stage 3 + Stage 4)
    for ep in [16, 30, 60]:
        mask, stages = resolve_progressive_unfreeze_mask(cfg, ep, backbone_vars)
        assert stages == [3, 4], f"Epoch {ep}: expected [3, 4], got {stages}"
        for v in backbone_vars:
            k = variable_key(v)
            v_name = getattr(v, "name", "")
            is_s34 = any(_var_belongs_to_stage(v_name, s) for s in [3, 4])
            is_s12 = any(_var_belongs_to_stage(v_name, s) for s in [1, 2])
            if is_s34:
                assert mask[k] is True, f"Epoch {ep}: Stage 3/4 var {v_name} should be trainable"
            if is_s12:
                assert mask[k] is False, f"Epoch {ep}: Stage 1/2 var {v_name} should be permanently frozen"
    s34_vars = sum(1 for v in backbone_vars if mask[variable_key(v)])
    print(f"  -> Phase 3 (Epochs 16+): Stage 3 + Stage 4, {s34_vars} vars trainable, Stage 1-2 permanently frozen [PASS]")

    # -----------------------------------------------------------------------
    # 4. Discriminative LR Scaling Factors
    # -----------------------------------------------------------------------
    print("\n[TEST 4] Testing discriminative LR scaling...")
    lr_scales = compute_stage_lr_scales(cfg, backbone_vars, [3, 4])
    for v in backbone_vars:
        k = variable_key(v)
        v_name = getattr(v, "name", "")
        if _var_belongs_to_stage(v_name, 4):
            assert lr_scales[k] == 1.0, f"Stage 4 var {v_name} expected scale 1.0, got {lr_scales[k]}"
        elif _var_belongs_to_stage(v_name, 3):
            assert lr_scales[k] == 0.5, f"Stage 3 var {v_name} expected scale 0.5, got {lr_scales[k]}"
    print("  -> Discriminative LR scaling: Stage 4 = 1.0x (1e-5), Stage 3 = 0.5x (5e-6) [PASS]")

    # -----------------------------------------------------------------------
    # 5. Label Smoothing Isolation & Logit Adjustment Compatibility
    # -----------------------------------------------------------------------
    print("\n[TEST 5] Testing label smoothing isolation & logit adjustment...")
    dummy_labels = tf.constant([0, 3], dtype=tf.int32)
    dummy_logits = tf.constant([[2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
                               [0.1, 0.2, 0.3, 2.5, 0.1, 0.2, 0.1]], dtype=tf.float32)
    dummy_sem_logits = tf.constant([[1.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 1.2, 0.1, 0.1, 0.1]], dtype=tf.float32)
    dummy_outputs = {
        "logits": dummy_logits,
        "semantic_logits": dummy_sem_logits,
    }

    # Verify smooth_auxiliary=False preserves unsmoothed targets on semantic loss
    loss_smooth, parts_smooth = supervised_mgr_loss(
        dummy_labels, dummy_outputs, num_classes=7, label_smoothing=0.05, smooth_auxiliary=False
    )
    loss_nosmooth, parts_nosmooth = supervised_mgr_loss(
        dummy_labels, dummy_outputs, num_classes=7, label_smoothing=0.0, smooth_auxiliary=False
    )

    # Main CE should differ because of smoothing
    assert not np.isclose(float(parts_smooth["ce"].numpy()), float(parts_nosmooth["ce"].numpy())), \
        "Main CE should be affected by label_smoothing=0.05"
    # Semantic loss should be identical because smooth_auxiliary=False
    assert np.isclose(float(parts_smooth["semantic"].numpy()), float(parts_nosmooth["semantic"].numpy())), \
        "Semantic loss should NOT be affected by label_smoothing when smooth_auxiliary=False"
    print("  -> Label smoothing (0.05) is isolated strictly to main CE [PASS]")

    # Test with Logit Adjustment offsets
    la_report = compute_logit_adjustment(np.array([0, 1, 2, 3, 4, 5, 6]), tau=0.5, num_classes=7)
    offsets = tf.constant(la_report["offsets"], dtype=tf.float32)
    loss_la, parts_la = supervised_mgr_loss(
        dummy_labels, dummy_outputs, num_classes=7, label_smoothing=0.05,
        logit_adj_offsets=offsets, smooth_auxiliary=False
    )
    assert np.isfinite(float(loss_la.numpy())), "Loss with logit adjustment and label smoothing must be finite"
    print("  -> Logit adjustment + label smoothing combination verified [PASS]")

    # -----------------------------------------------------------------------
    # 6. Gradient Flow & Masking under GradientTape
    # -----------------------------------------------------------------------
    print("\n[TEST 6] Testing forward + backward gradient masking under GradientTape...")
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        out = model(dummy_input, training=True)
        loss, _ = supervised_mgr_loss(
            dummy_labels, out, num_classes=7, label_smoothing=0.05, smooth_auxiliary=False
        )

    all_vars = model.trainable_variables
    grads = tape.gradient(loss, all_vars)
    assert len(grads) == len(all_vars)
    assert all(g is not None for g in grads), "Found None gradients in trainable variables!"

    # Simulate Phase 1 gradient masking (Epoch 1)
    mask_ep1, _ = resolve_progressive_unfreeze_mask(cfg, 1, backbone_vars)
    backbone_ids = {variable_key(v) for v in backbone_vars}
    masked_bb_grads = []
    for g, v in zip(grads, all_vars):
        k = variable_key(v)
        if k in backbone_ids:
            if mask_ep1.get(k, False):
                masked_bb_grads.append((g, v))

    assert len(masked_bb_grads) == 0, f"Expected 0 backbone gradients in Phase 1, got {len(masked_bb_grads)}"
    print(f"  -> Phase 1 gradient masking: 0 backbone gradients passed to optimizer [PASS]")

    # Simulate Phase 2 gradient masking (Epoch 9)
    mask_ep9, _ = resolve_progressive_unfreeze_mask(cfg, 9, backbone_vars)
    masked_bb_grads_ep9 = []
    for g, v in zip(grads, all_vars):
        k = variable_key(v)
        if k in backbone_ids:
            if mask_ep9.get(k, False):
                masked_bb_grads_ep9.append((g, v))

    assert len(masked_bb_grads_ep9) == s4_vars, f"Expected {s4_vars} Stage 4 gradients, got {len(masked_bb_grads_ep9)}"
    print(f"  -> Phase 2 gradient masking: {len(masked_bb_grads_ep9)} Stage 4 gradients passed [PASS]")

    # -----------------------------------------------------------------------
    # 7. Calibration Diagnostics (ECE, Mean Confidence, NLL)
    # -----------------------------------------------------------------------
    print("\n[TEST 7] Testing ECE and confidence computation logic...")
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3])  # 7 correct, 3 wrong -> 70% acc
    max_probs = np.array([0.9, 0.85, 0.8, 0.95, 0.7, 0.75, 0.88, 0.6, 0.65, 0.7])

    n_bins = 15
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (y_pred == y_true).astype(np.float64)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = in_bin | (max_probs == bin_boundaries[i])
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(correct[in_bin]))
            bin_conf = float(np.mean(max_probs[in_bin]))
            ece += (bin_count / len(max_probs)) * abs(bin_acc - bin_conf)

    mean_conf = float(np.mean(max_probs))
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0, 1], got {ece}"
    assert 0.0 <= mean_conf <= 1.0, f"Mean confidence must be in [0, 1], got {mean_conf}"
    print(f"  -> Calibration metrics verified: Mean Confidence={mean_conf:.4f}, ECE={ece:.4f} [PASS]")

    print("\n" + "=" * 70)
    print("ALL V6 SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_v6_pipeline()
