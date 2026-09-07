#!/usr/bin/env python3
"""Contract and sanity verification script for RAF-DB v5 competing variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def verify_variant_dynamic_attention(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    assert model_cfg["name"] == "rafdb_siglip2_semantic_stable_v5_dynamic_part_attention"
    assert model_cfg["use_dynamic_part_attention"] is True, "use_dynamic_part_attention must be True"
    assert model_cfg["dynamic_part_bottleneck"] == 128
    assert model_cfg["use_soft_regional_pooling"] is False
    assert model_cfg["use_semantic_branch"] is True
    assert model_cfg["use_clip_semantic"] is True
    assert model_cfg["multi_prototype"] is True
    assert model_cfg["ablation"] == "adaptive_clip_confusion"
    assert Path(cfg["data"]["data_path"]) == ROOT / "data/rafdb"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_dynamic_part_attention"
    print(f"[OK] Contract verified: {cfg_path.name}")


def verify_variant_adaptive_fusion(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    assert model_cfg["name"] == "rafdb_siglip2_semantic_stable_v5_adaptive_fusion_gate"
    assert model_cfg["use_adaptive_fusion_gate"] is True, "use_adaptive_fusion_gate must be True"
    assert model_cfg["adaptive_fusion_max_alpha"] == 0.20
    assert model_cfg["use_soft_regional_pooling"] is True
    assert model_cfg["use_semantic_branch"] is True
    assert model_cfg["use_clip_semantic"] is True
    assert model_cfg["multi_prototype"] is True
    assert model_cfg["ablation"] == "adaptive_clip_confusion"
    assert Path(cfg["data"]["data_path"]) == ROOT / "data/rafdb"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_adaptive_fusion_gate"
    print(f"[OK] Contract verified: {cfg_path.name}")


def verify_variant_combined_ultimate(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    assert model_cfg["name"] == "rafdb_siglip2_semantic_stable_v5_combined_ultimate"
    assert model_cfg["use_dynamic_part_attention"] is True, "use_dynamic_part_attention must be True"
    assert model_cfg["dynamic_part_bottleneck"] == 128
    assert model_cfg["use_soft_regional_pooling"] is False
    assert model_cfg["use_adaptive_fusion_gate"] is True, "use_adaptive_fusion_gate must be True"
    assert model_cfg["adaptive_fusion_max_alpha"] == 0.20
    assert model_cfg["use_semantic_branch"] is True
    assert model_cfg["use_clip_semantic"] is True
    assert model_cfg["multi_prototype"] is True
    assert model_cfg["ablation"] == "adaptive_clip_confusion"
    assert Path(cfg["data"]["data_path"]) == ROOT / "data/rafdb"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate"
    print(f"[OK] Contract verified: {cfg_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check v5 contracts.")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    args = parser.parse_args()

    cfg1 = ROOT / "config_rafdb_siglip2_semantic_stable_v5_dynamic_part_attention.yaml"
    cfg2 = ROOT / "config_rafdb_siglip2_semantic_stable_v5_adaptive_fusion_gate.yaml"
    cfg3 = ROOT / "config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml"

    if args.config:
        p = Path(args.config)
        if not p.is_absolute():
            p = ROOT / p
        if "combined_ultimate" in p.name:
            verify_variant_combined_ultimate(p)
        elif "dynamic_part_attention" in p.name:
            verify_variant_dynamic_attention(p)
        elif "adaptive_fusion_gate" in p.name:
            verify_variant_adaptive_fusion(p)
        else:
            print(f"[WARNING] Unrecognized config name: {p.name}")
    else:
        verify_variant_dynamic_attention(cfg1)
        verify_variant_adaptive_fusion(cfg2)
        verify_variant_combined_ultimate(cfg3)

    print("\nALL V5 CONTRACTS PASSED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
