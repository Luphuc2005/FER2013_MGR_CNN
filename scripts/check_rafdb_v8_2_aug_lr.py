#!/usr/bin/env python3
"""Check V8.2 config scope and actual LR resolver without importing TensorFlow.

--assets additionally requires the server dataset, weights and finite prototypes.
File presence is not checkpoint-load verification. No training runs here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from utils.optimizer_config import resolve_base_optimizer
from utils.semantic_schedule import resolve_lambda_sem

CONFIG = ROOT / "config_rafdb_v8_2_linear_aug_lr.yaml"
BASELINE = ROOT / "config_rafdb_v8_1_linear_stable.yaml"


def check_config():
    if any(key.startswith("MGR_") for key in os.environ):
        raise RuntimeError("Unset MGR_* environment overrides before checking this experiment.")
    old = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    new = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected = copy.deepcopy(old)
    expected["source"] = new["source"]
    expected["model"]["name"] = "rafdb_v8_2_linear_aug_lr"
    expected["paths"]["output_dir"] = "outputs/papers/rafdb_v8_2_linear_aug_lr"
    expected["augmentation"].update(gamma_min=0.80, gamma_max=1.25)
    expected["data"]["minority_aug"].update(
        brightness_delta=0.15, contrast_lower=0.85, contrast_upper=1.15,
        scale_lower=0.95, scale_upper=1.05, translation_fraction=0.03,
    )
    expected["training"].update(
        visual_extractor_lr=2e-6, min_lr=1e-7, head_decay_epochs=60,
    )
    assert expected == new, "Unexpected difference from V8.1 beyond augmentation/LR and run metadata."
    cfg = load_config(CONFIG)
    assert cfg["model"]["classifier_type"] == "linear"
    assert not any("arcmargin" in key for key in cfg["model"])
    assert cfg["training"]["lr"] == cfg["training"]["finetune_lr"] == 1e-4
    assert resolve_base_optimizer(cfg["training"]) == ("adamw", 0.035)
    assert cfg["training"]["optimizer"] == "sam"
    assert [resolve_lambda_sem(cfg, e) for e in (1, 8, 9, 16, 17, 30, 31, 60)] == [
        0.10, 0.10, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
    ]
    print("V8_2_CONFIG_SCOPE_OK augmentation_lr_only_relative_to=V8.1")
    return cfg


def check_lr_schedule(cfg):
    # Execute the two real pure functions; avoid train.py's TensorFlow/GPU imports.
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8-sig"))
    names = {"cosine_lr", "resolve_phase_lrs"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    namespace = {"np": SimpleNamespace(cos=math.cos, pi=math.pi)}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "train.py"), "exec"), namespace)
    resolve = namespace["resolve_phase_lrs"]
    old = load_config(BASELINE)
    print("epoch head_lr backbone_optimizer_lr baseline_head_lr baseline_backbone_lr")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_backbone = epoch > cfg["model"]["freeze_backbone_epochs"]
        head, backbone = resolve(cfg, epoch - 1, train_backbone)
        assert math.isfinite(head) and 1e-7 <= head <= 1e-4
        assert backbone == 0.0 if not train_backbone else 1e-7 <= backbone <= 2e-6
        if epoch in (1, 4, 5, 9, 12, 13, 24, 25, 29, 30, 33, 45, 60):
            old_head, old_backbone = resolve(old, epoch - 1, train_backbone)
            print(f"{epoch:2d} {head:.8g} {backbone:.8g} {old_head:.8g} {old_backbone:.8g}")
    assert math.isclose(resolve(cfg, 59, True)[0], 1e-7)
    assert resolve(cfg, 32, True)[0] > 1e-5
    print("V8_2_LR_RESOLVER_OK all_60_epochs_checked tensorflow_runtime=NOT_RUN")
    print("STAGE_MULTIPLIERS=gradient_scaling_not_independent_optimizer_LRs")


def check_assets(cfg):
    import numpy as np
    data = Path(cfg["data"]["data_path"])
    for split in ("train", "val", "test"):
        assert (data / f"{split}.csv").is_file() or (data / split).is_dir(), f"Missing {split} at {data}"
    weights = ROOT / cfg["model"]["convnext_base_pretrained_path"]
    assert weights.is_file() and weights.stat().st_size > 0, f"Missing weights: {weights}"
    prototypes = np.load(ROOT / cfg["model"]["clip_prototypes_path"], allow_pickle=False)
    assert prototypes.shape == (7, 5, 768) and np.isfinite(prototypes).all()
    print("ASSET_PRESENCE_AND_PROTOTYPE_SHAPE_OK checkpoint_restore=NOT_CHECKED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true")
    args = parser.parse_args()
    cfg = check_config()
    check_lr_schedule(cfg)
    if args.assets:
        check_assets(cfg)
    print("VAL_ACCURACY_IMPROVEMENT=UNVERIFIED")


if __name__ == "__main__":
    main()
