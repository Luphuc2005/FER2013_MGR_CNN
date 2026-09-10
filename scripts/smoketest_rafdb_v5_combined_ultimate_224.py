#!/usr/bin/env python3
"""Comprehensive smoke test and regression verification for RAF-DB 224x224 experiment."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from config import load_config
from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline
from losses.classification import supervised_mgr_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for RAF-DB 224 experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_224.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Smoke test batch size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    print(f"[INFO] Loading config: {cfg_path}")
    cfg = load_config(cfg_path)
    img_size = int(cfg["data"]["image_size"])
    num_classes = int(cfg["data"]["num_classes"])
    batch_size = int(args.batch_size)

    assert img_size == 224, f"Expected image_size=224, got {img_size}"
    assert num_classes == 7, f"Expected num_classes=7, got {num_classes}"

    print(f"\n{'='*70}")
    print(" 1. BUILDING MODEL & LOADING PRETRAINED WEIGHTS")
    print(f"{'='*70}")
    model = ConvNeXtBaseFaceFERBaseline(cfg)

    # Check pretrained loading status
    print(f"\n[PRETRAINED LOAD STATUS]: {model.pretrained_load_status}")
    if model.pretrained_load_status not in ("loaded", "partial"):
        print("[WARNING] Pretrained weights were not fully loaded! Check file existence.")

    print(f"\n{'='*70}")
    print(" 2. SHAPE TRACE VERIFICATION (Input 224x224)")
    print(f"{'='*70}")
    dummy_input = tf.random.normal([batch_size, img_size, img_size, 3], dtype=tf.float32)
    dummy_labels = tf.constant([0, 3][:batch_size], dtype=tf.int32)

    # Forward pass to get endpoints
    endpoints = model.backbone(
        dummy_input, training=False, return_endpoints=True, stage3_adapter=model.stage3_adapter
    )
    feat = endpoints["stage4"]
    pooled = model.gap(feat)
    dropped = model.head_dropout(pooled, training=False)
    classifier_logits = model.classifier(dropped)
    outputs = model({"image": dummy_input}, training=False)

    print("\n[224 EXPERIMENT SHAPE TRACE]")
    print(f"input:            {tuple(dummy_input.shape)}")
    print(f"stem:             {tuple(endpoints['stem'].shape)}")
    print(f"stage1:           {tuple(endpoints['stage1'].shape)}")
    print(f"stage2:           {tuple(endpoints['stage2'].shape)}")
    print(f"stage3:           {tuple(endpoints['stage3'].shape)}")
    print(f"stage4:           {tuple(endpoints['stage4'].shape)}")
    v_proj = outputs.get("visual_projector")
    if v_proj is None and model.visual_projector is not None:
        v_proj = model.visual_projector(pooled)

    print(f"gap:              {tuple(pooled.shape)}")
    if v_proj is not None:
        print(f"visual_projector: {tuple(v_proj.shape)}")
    print(f"semantic_logits:  {tuple(outputs['semantic_logits'].shape)}")
    print(f"final_logits:     {tuple(outputs['logits'].shape)}")

    # Verification assertions
    assert tuple(dummy_input.shape) == (batch_size, 224, 224, 3), f"Input mismatch: {dummy_input.shape}"
    assert tuple(endpoints["stem"].shape) == (batch_size, 112, 112, 128), f"Stem mismatch: {endpoints['stem'].shape}"
    assert tuple(endpoints["stage1"].shape) == (batch_size, 112, 112, 128), f"Stage1 mismatch: {endpoints['stage1'].shape}"
    assert tuple(endpoints["stage2"].shape) == (batch_size, 56, 56, 256), f"Stage2 mismatch: {endpoints['stage2'].shape}"
    assert tuple(endpoints["stage3"].shape) == (batch_size, 28, 28, 512), f"Stage3 mismatch: {endpoints['stage3'].shape}"
    assert tuple(endpoints["stage4"].shape) == (batch_size, 14, 14, 1024), f"Stage4 mismatch: {endpoints['stage4'].shape}"
    assert tuple(pooled.shape) == (batch_size, 1024), f"GAP mismatch: {pooled.shape}"
    if v_proj is not None:
        assert tuple(v_proj.shape) == (batch_size, 768), f"Projector mismatch: {v_proj.shape}"
    assert tuple(outputs["semantic_logits"].shape) == (batch_size, 7), f"Semantic logits mismatch: {outputs['semantic_logits'].shape}"
    assert tuple(outputs["logits"].shape) == (batch_size, 7), f"Final logits mismatch: {outputs['logits'].shape}"

    print("\n[SHAPE TRACE ASSERTIONS PASSED!]")

    print(f"\n{'='*70}")
    print(" 3. DYNAMIC PART ATTENTION & SIGLIP2 SEMANTIC VERIFICATION")
    print(f"{'='*70}")
    attn_maps = outputs.get("part_attention_maps")
    if attn_maps is None and model.use_dynamic_part_attention and model.dynamic_part_attn is not None:
        stage3_feat = endpoints.get("stage3_adapter", endpoints["stage3"])
        _, _, _, attn_maps = model.dynamic_part_attn(stage3_feat, training=False)
    if attn_maps is not None:
        attn_shape = tuple(attn_maps.shape)
        print(f"Dynamic Part Attention maps shape: {attn_shape}")
        assert attn_shape == (batch_size, 28, 28, 3), f"Part attention maps shape mismatch: {attn_shape}"
        assert tf.math.reduce_all(tf.math.is_finite(attn_maps)), "Part attention maps have non-finite values!"
        print("[OK] Dynamic Part Attention maps shape (B, 28, 28, 3) and finite values verified.")

    if hasattr(model, "text_prototypes") and model.text_prototypes is not None:
        proto_shape = tuple(model.text_prototypes.shape)
        print(f"SigLIP2 text prototypes shape:     {proto_shape}")
        assert proto_shape == (7, 5, 768), f"Text prototypes shape mismatch: {proto_shape}"
        print("[OK] Text prototypes shape (7, 5, 768) verified.")

    alpha_t = outputs.get("adaptive_fusion_alpha")
    if alpha_t is None and model.use_adaptive_fusion_gate and model.adaptive_fusion_gate is not None:
        alpha_t = model.adaptive_fusion_gate(pooled, training=False)
    if alpha_t is not None:
        alpha_val = alpha_t.numpy()
        print(f"Adaptive fusion gate alpha range:  min={alpha_val.min():.4f}, max={alpha_val.max():.4f}")
        assert np.all(alpha_val >= 0.0) and np.all(alpha_val <= 0.20 + 1e-5), f"Alpha out of range: {alpha_val}"
        print("[OK] Adaptive fusion alpha(x) verified in [0, 0.20].")

    print(f"\n{'='*70}")
    print(" 4. FORWARD PASS, LOSS COMPUTATION & REGRESSION CHECK")
    print(f"{'='*70}")
    loss, loss_breakdown = supervised_mgr_loss(
        labels=dummy_labels,
        outputs=outputs,
        num_classes=num_classes,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.10)),
        use_hard_loss=bool(cfg["model"].get("use_hard_semantic_loss", True)),
        hard_pairs=cfg["model"].get("hard_pairs", {}),
        hard_margin=float(cfg["model"].get("hard_margin", 0.15)),
        lambda_hard=float(cfg["model"].get("lambda_hard", 0.10)),
    )
    print(f"Total Supervised Loss: {float(loss):.4f}")
    for k, v in loss_breakdown.items():
        if isinstance(v, tf.Tensor):
            print(f"  - {k}: {float(v):.4f}")
    assert tf.math.is_finite(loss), "Loss is not finite!"
    print("[OK] Forward pass and loss finite verified.")

    print(f"\n{'='*70}")
    print(" 5. SAM TRAINING STEP VERIFICATION (Both Passes)")
    print(f"{'='*70}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["lr"]))
    trainable_vars = model.trainable_variables
    rho = float(cfg["training"].get("sam_rho", 0.02))

    # First forward pass
    with tf.GradientTape() as tape:
        out1 = model({"image": dummy_input}, training=True)
        loss1, _ = supervised_mgr_loss(labels=dummy_labels, outputs=out1, num_classes=num_classes)
    grads1 = tape.gradient(loss1, trainable_vars)

    # Check finite gradients
    for g, v in zip(grads1, trainable_vars):
        if g is not None:
            assert tf.math.reduce_all(tf.math.is_finite(g)), f"Gradient for {v.name} has non-finite values!"

    # Compute SAM perturbation epsilon = rho * g / ||g||
    grad_norm = tf.linalg.global_norm([g for g in grads1 if g is not None])
    scale = rho / (grad_norm + 1e-12)
    e_r_list = []
    for g, v in zip(grads1, trainable_vars):
        if g is not None:
            e_r = g * scale
            v.assign_add(e_r)
            e_r_list.append(e_r)
        else:
            e_r_list.append(None)

    # Second forward pass
    with tf.GradientTape() as tape2:
        out2 = model({"image": dummy_input}, training=True)
        loss2, _ = supervised_mgr_loss(labels=dummy_labels, outputs=out2, num_classes=num_classes)
    grads2 = tape2.gradient(loss2, trainable_vars)

    # Restore variables
    for e_r, v in zip(e_r_list, trainable_vars):
        if e_r is not None:
            v.assign_sub(e_r)

    # Apply second step gradients
    optimizer.apply_gradients(zip([g for g in grads2 if g is not None], [v for g, v in zip(grads2, trainable_vars) if g is not None]))
    print(f"[SAM Step 1 Loss]: {float(loss1):.4f} | [SAM Step 2 Loss]: {float(loss2):.4f} | [Grad Norm]: {float(grad_norm):.4f}")
    assert tf.math.is_finite(loss2), "SAM second step loss is not finite!"
    print("[OK] SAM 2-step optimization pass verified.")

    print(f"\n{'='*70}")
    print(" 6. VALIDATION & HFLIP TTA FORWARD PASS VERIFICATION")
    print(f"{'='*70}")
    # Normal forward
    val_out = model({"image": dummy_input}, training=False)
    val_logits = val_out["logits"]

    # HFlip TTA forward
    flipped_input = tf.image.flip_left_right(dummy_input)
    val_flip_out = model({"image": flipped_input}, training=False)
    val_flip_logits = val_flip_out["logits"]

    w_orig = float(cfg["tta"]["original_weight"])
    w_flip = float(cfg["tta"]["flip_weight"])
    tta_logits = w_orig * val_logits + w_flip * val_flip_logits
    print(f"Original logits sample: {val_logits[0].numpy()[:3]}")
    print(f"Flipped logits sample:  {val_flip_logits[0].numpy()[:3]}")
    print(f"TTA fused logits sample: {tta_logits[0].numpy()[:3]}")
    assert tf.math.reduce_all(tf.math.is_finite(tta_logits)), "TTA fused logits have non-finite values!"
    print("[OK] HFlip TTA forward pass verified.")

    print(f"\n{'='*70}")
    print(" ALL 224 SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
