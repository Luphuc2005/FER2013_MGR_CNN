#!/usr/bin/env python3
"""Pre-flight validation suite for RAF-DB V8.3 Optimal.

Verifies:
  1. Static configuration assertions:
     - Model: rafdb_v8_3_optimal
     - Direct Global-Regional Fusion enabled (1024-dim, proj=256)
     - Dynamic Part Attention enabled (bottleneck=128)
     - Adaptive Fusion Gate enabled (alpha_max=0.10)
     - Multi-prototype SigLIP2 semantic supervision (5 protos, lambda_sem=0.06, lambda_hard=0.06)
     - SAM optimizer (rho=0.02) + AdamW (wd=0.05)
     - LR: head=0.00025, backbone=0.000008 (healthy learning rate)
     - Unfreeze: standard freeze_backbone_epochs=4, unfreeze at epoch 5 (progressive unfreeze disabled)
     - Robust minority stochastic augmentation (rotation 15.0, scale 0.88-1.12, translation 0.06, brightness 0.18, contrast 0.80-1.20)
     - Global augmentation: random erasing prob 0.35, max area 0.15
     - Class-aware oversampling: target 650 for minority classes [0, 1, 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config

CONFIG_NAME = "config_rafdb_v8_3_optimal.yaml"


def check_static_config(cfg_path: Path):
    cfg = load_config(cfg_path)
    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]
    a = cfg["augmentation"]
    ma = d.get("minority_aug", {})

    print("[CHECK 1/5] Static Config Assertions...")
    assert m["name"] == "rafdb_v8_3_optimal", f"Unexpected name {m['name']}"
    assert m["arch"] == "convnext_base_ms1m_arcface"
    assert m["use_dynamic_part_attention"], "Dynamic Part Attention must be enabled"
    assert m["dynamic_part_bottleneck"] == 128
    assert m["use_global_regional_fusion"], "Global Regional Fusion must be enabled"
    assert m["global_regional_projection_dim"] == 256
    assert m["use_adaptive_fusion_gate"], "Adaptive Fusion Gate must be enabled"
    assert m.get("adaptive_fusion_max_alpha") == 0.10, f"Expected alpha_max 0.10, got {m.get('adaptive_fusion_max_alpha')}"

    # Semantic branch & loss
    assert m["use_semantic_branch"] and m["use_clip_semantic"]
    assert m["multi_prototype"]
    assert m["lambda_sem"] == 0.06, f"Expected lambda_sem 0.06, got {m['lambda_sem']}"
    assert m["lambda_hard"] == 0.06
    assert m["hard_margin"] == 0.15

    # Gate anti-collapse schedule
    assert m["use_adaptive_granularity"]
    assert m.get("granularity_gate_step_schedule", False)
    assert m.get("gate_entropy_floor", 0.0) == 0.80
    assert m.get("lambda_gate_entropy", 0.0) == 0.01

    # Learning rates & optimizer
    assert t["lr"] == 0.00025, f"Expected head lr 0.00025, got {t['lr']}"
    assert t["visual_extractor_lr"] == 0.000008, f"Expected backbone lr 0.000008, got {t['visual_extractor_lr']}"
    assert t["optimizer"] == "sam" and t["base_optimizer"] == "adamw"
    assert t["sam_rho"] == 0.02
    assert t["weight_decay"] == 0.05
    assert t["label_smoothing"] == 0.10

    # Unfreeze strategy: unfreeze at epoch 5
    assert m["freeze_backbone_epochs"] == 4
    assert m["unfreeze_backbone"] is True
    assert not m.get("progressive_unfreeze", {}).get("enabled", False), "Progressive unfreeze should be disabled"

    # Global Augmentation
    assert a["horizontal_flip"] is True
    assert a["rotation_degrees"] == 15.0
    assert a["random_erasing_prob"] == 0.35, f"Expected erasing prob 0.35, got {a['random_erasing_prob']}"
    assert a["random_erasing_area_max"] == 0.15, f"Expected erasing max area 0.15, got {a['random_erasing_area_max']}"
    assert a["gamma_prob"] == 0.50

    # Minority Stochastic Augmentation
    assert ma["flip_prob"] == 0.50
    assert ma["rotation_range"] == 15.0, f"Expected rotation 15.0, got {ma['rotation_range']}"
    assert ma["brightness_delta"] == 0.18
    assert ma["contrast_lower"] == 0.80 and ma["contrast_upper"] == 1.20
    assert ma["scale_lower"] == 0.88 and ma["scale_upper"] == 1.12
    assert ma["translation_fraction"] == 0.06

    # Sampling strategy
    assert d["sampling_strategy"] == "class_aware_minority_oversample"
    assert d["target_minority_count"] == 650
    assert d["minority_classes"] == [0, 1, 2]

    print("  [PASSED] Static configuration assertions verified.")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / CONFIG_NAME))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    print(f"=== Validating RAF-DB V8.3 Optimal: {cfg_path.name} ===")
    check_static_config(cfg_path)
    print("=== All pre-flight checks passed successfully! ===")


if __name__ == "__main__":
    main()
