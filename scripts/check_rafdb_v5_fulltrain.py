#!/usr/bin/env python3
"""Contract check for RAF-DB v5 Combined Ultimate Final Full-Train Protocol."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def verify_fulltrain_contract(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    data_cfg = cfg["data"]

    assert model_cfg["name"] == "rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain", f"Unexpected name: {model_cfg['name']}"
    assert data_cfg["full_train"] is True, "data.full_train must be True"
    assert data_cfg["shuffle_buffer"] == 12271, "shuffle_buffer must be 12271"

    # Architecture & feature flags identical to v5 combined ultimate
    assert model_cfg["use_dynamic_part_attention"] is True
    assert model_cfg["dynamic_part_bottleneck"] == 128
    assert model_cfg["use_adaptive_fusion_gate"] is True
    assert model_cfg["adaptive_fusion_max_alpha"] == 0.20
    assert model_cfg["use_adaptive_granularity"] is True
    assert model_cfg["use_hard_semantic_loss"] is True
    assert model_cfg["drop_path_rate"] == 0.25
    assert model_cfg["classifier_dropout1"] == 0.40

    # Augmentation identical to v5 combined ultimate
    assert aug_cfg["rotation_degrees"] == 15.0
    assert aug_cfg["random_erasing_prob"] == 0.40
    assert aug_cfg["random_erasing_area_max"] == 0.15

    # Training & optimization identical
    assert train_cfg["optimizer"] == "sam"
    assert train_cfg["base_optimizer"] == "adamw"
    assert train_cfg["sam_rho"] == 0.02
    assert train_cfg["lr"] == 0.0003
    assert train_cfg["visual_extractor_lr"] == 0.00001
    assert train_cfg["weight_decay"] == 0.05
    assert train_cfg["scheduler"] == "cosine"
    assert train_cfg["epochs"] == 60
    assert train_cfg["patience"] == 0, "Patience must be 0 (no early stopping on full-train)"

    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain"

    print(f"[OK] Full-Train Contract verified successfully for: {cfg_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check v5 Full-Train contract.")
    parser.add_argument("--config", type=str, default="config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain.yaml")
    args = parser.parse_args()

    p = Path(args.config)
    if not p.is_absolute():
        p = ROOT / p
    verify_fulltrain_contract(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
