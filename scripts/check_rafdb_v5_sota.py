#!/usr/bin/env python3
"""Contract check for RAF-DB v5 Combined Ultimate SOTA Tuned."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def verify_sota_contract(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]

    valid_names = [
        "rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota",
        "rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota_logit_adj",
    ]
    assert model_cfg["name"] in valid_names, f"Unexpected model name: {model_cfg['name']}"
    assert model_cfg["use_dynamic_part_attention"] is True, "Dynamic Part Attention must be True"
    assert model_cfg["dynamic_part_bottleneck"] == 128
    assert model_cfg["use_adaptive_fusion_gate"] is True, "Adaptive Fusion Gate must be True"
    assert model_cfg["adaptive_fusion_max_alpha"] == 0.20
    assert model_cfg["drop_path_rate"] == 0.10, "drop_path_rate must be 0.10"
    assert model_cfg["classifier_dropout1"] == 0.35, "classifier_dropout1 must be 0.35"
    assert model_cfg["lambda_hard"] == 0.05, "lambda_hard must be 0.05"
    assert 3 not in model_cfg["hard_pairs"].get(1, []), "Happy (3) should not be in hard negative pairs for Disgust (1)"

    assert train_cfg["weight_decay"] == 0.035, "weight_decay must be 0.035"
    assert train_cfg["visual_extractor_lr"] == 0.00003, "visual_extractor_lr must be 3e-5"

    assert aug_cfg["rotation_degrees"] == 5.0, "rotation_degrees should be 5.0 for aligned faces"
    assert aug_cfg["random_erasing_prob"] == 0.15, "random_erasing_prob should be 0.15"

    assert Path(cfg["data"]["data_path"]) == ROOT / "data/rafdb"
    expected_output_dirs = [
        ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota",
        ROOT / "outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota_logit_adj",
    ]
    assert Path(cfg["paths"]["output_dir"]) in expected_output_dirs, f"Unexpected output_dir: {cfg['paths']['output_dir']}"

    print(f"[OK] SOTA Contract successfully verified for: {cfg_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check v5 SOTA contract.")
    parser.add_argument("--config", type=str, default="config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota.yaml")
    args = parser.parse_args()

    p = Path(args.config)
    if not p.is_absolute():
        p = ROOT / p
    verify_sota_contract(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
