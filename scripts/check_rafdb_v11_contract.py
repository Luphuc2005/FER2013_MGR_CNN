#!/usr/bin/env python3
"""Contract and sanity verification script for RAF-DB V11 Champion SOTA 93%."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def verify_v11_contract(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    assert model_cfg["name"] == "rafdb_v11_champ_sota_93", f"Wrong name: {model_cfg['name']}"
    assert model_cfg.get("use_global_regional_fusion", False) is False, "Must keep direct 1024-dim ArcFace head (no compression)"
    assert model_cfg["use_dynamic_part_attention"] is True, "use_dynamic_part_attention must be True"
    assert model_cfg["dynamic_part_bottleneck"] == 128
    assert model_cfg["use_soft_regional_pooling"] is False
    assert model_cfg["use_adaptive_fusion_gate"] is True, "use_adaptive_fusion_gate must be True"
    assert model_cfg["adaptive_fusion_max_alpha"] == 0.20
    assert model_cfg["use_adaptive_granularity"] is True
    assert model_cfg["use_semantic_branch"] is True
    assert model_cfg["use_clip_semantic"] is True
    assert model_cfg["multi_prototype"] is True
    assert model_cfg["use_hard_semantic_loss"] is True
    assert abs(float(model_cfg["lambda_hard"]) - 0.12) < 1e-4
    assert abs(float(model_cfg["hard_margin"]) - 0.18) < 1e-4

    # Targeted hard pairs check
    hp = {int(k): v for k, v in model_cfg.get("hard_pairs", {}).items()}
    assert 4 in hp and 6 in hp[4], "Sad (4) must target Neutral (6)"
    assert 6 in hp and 4 in hp[6], "Neutral (6) must target Sad (4)"
    assert 1 in hp and 4 in hp[1] and 6 in hp[1], "Disgust (1) must target Sad (4) and Neutral (6)"
    assert 2 in hp and 5 in hp[2], "Fear (2) must target Surprise (5)"

    assert abs(float(training_cfg["lambda_rdrop"]) - 0.30) < 1e-4, "lambda_rdrop must be 0.30"
    assert training_cfg["optimizer"] == "sam"
    assert abs(float(training_cfg["sam_rho"]) - 0.02) < 1e-4

    assert Path(cfg["data"]["data_path"]) == ROOT / "data/rafdb"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/papers/rafdb_v11_champ_sota_93"

    print(f"[OK] RAF-DB V11 Champion Contract verified: {cfg_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RAF-DB V11 contract.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config_rafdb_v11_champ_sota_93.yaml"), help="Path to config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    verify_v11_contract(cfg_path)
    print("\nALL RAF-DB V11 CONTRACT CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
