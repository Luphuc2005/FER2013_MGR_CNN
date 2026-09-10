"""Standalone Pure-Python Unit Test for V6 Progressive Unfreezing and Config logic.
Does NOT require TensorFlow.
"""

import unittest
import numpy as np
from config import load_config
from utils.optimizer_config import resolve_base_optimizer
from utils.logit_adjustment import compute_logit_adjustment


def _var_belongs_to_stage(var_name: str, stage_num: int) -> bool:
    lower = var_name.lower()
    if stage_num == 1:
        return "stage1_block" in lower or "stem_conv" in lower or "stem_norm" in lower
    elif stage_num == 2:
        return "stage2_block" in lower or "downsample_stage2" in lower
    elif stage_num == 3:
        return "stage3_block" in lower or "downsample_stage3" in lower
    elif stage_num == 4:
        return "stage4_block" in lower or "downsample_stage4" in lower
    return False


def resolve_progressive_unfreeze_mask(
    cfg: dict,
    epoch_number: int,
    backbone_var_names: list,
) -> tuple:
    prog_cfg = cfg.get("model", {}).get("progressive_unfreeze", {})
    if not prog_cfg.get("enabled", False):
        return {v: True for v in backbone_var_names}, [1, 2, 3, 4]

    schedule = prog_cfg.get("schedule", [])
    trainable_stages = []
    for phase in schedule:
        start = int(phase.get("start_epoch", 1))
        end = int(phase.get("end_epoch", 9999))
        if start <= epoch_number <= end:
            trainable_stages = [int(s) for s in phase.get("trainable_stages", [])]
            break

    mask = {}
    for v_name in backbone_var_names:
        is_trainable = any(_var_belongs_to_stage(v_name, s) for s in trainable_stages)
        mask[v_name] = is_trainable
    return mask, trainable_stages


def compute_stage_lr_scales(
    cfg: dict,
    backbone_var_names: list,
    trainable_stages: list,
) -> dict:
    prog_cfg = cfg.get("model", {}).get("progressive_unfreeze", {})
    stage_mults = prog_cfg.get("stage_lr_multipliers", {})
    scales = {}
    for v_name in backbone_var_names:
        scale = 1.0
        for s in trainable_stages:
            if _var_belongs_to_stage(v_name, s):
                scale = float(stage_mults.get(str(s), 1.0))
                break
        scales[v_name] = scale
    return scales


class V6PurePythonTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("config_rafdb_v6_anti_overfitting.yaml")

        # Mock variable names mirroring actual ConvNeXtBaseFRBackbone variable names
        self.mock_backbone_vars = [
            "convnext_base_fr_backbone/stem_conv/kernel:0",
            "convnext_base_fr_backbone/stem_conv/bias:0",
            "convnext_base_fr_backbone/stem_norm/gamma:0",
            "convnext_base_fr_backbone/stem_norm/beta:0",
            "convnext_base_fr_backbone/stage1_block0/depthwise_conv/kernel:0",
            "convnext_base_fr_backbone/stage1_block2/linear2/bias:0",
            "convnext_base_fr_backbone/downsample_stage2/conv/kernel:0",
            "convnext_base_fr_backbone/stage2_block0/depthwise_conv/kernel:0",
            "convnext_base_fr_backbone/stage2_block2/linear2/bias:0",
            "convnext_base_fr_backbone/downsample_stage3/conv/kernel:0",
            "convnext_base_fr_backbone/stage3_block0/depthwise_conv/kernel:0",
            "convnext_base_fr_backbone/stage3_block15/linear1/kernel:0",
            "convnext_base_fr_backbone/stage3_block26/linear2/bias:0",
            "convnext_base_fr_backbone/downsample_stage4/conv/kernel:0",
            "convnext_base_fr_backbone/stage4_block0/depthwise_conv/kernel:0",
            "convnext_base_fr_backbone/stage4_block2/linear2/bias:0",
        ]

    def test_v6_config_values(self):
        self.assertEqual(self.cfg["model"]["name"], "rafdb_v6_anti_overfitting")
        self.assertEqual(self.cfg["training"]["label_smoothing"], 0.05)
        self.assertEqual(self.cfg["training"]["weight_decay"], 0.05)
        self.assertEqual(self.cfg["training"]["visual_extractor_lr"], 0.00001)
        self.assertEqual(self.cfg["training"]["lr"], 0.0003)
        self.assertTrue(self.cfg["training"]["save_best_macro_f1"])

    def test_progressive_schedule_phase1(self):
        for ep in range(1, 9):
            mask, stages = resolve_progressive_unfreeze_mask(self.cfg, ep, self.mock_backbone_vars)
            self.assertEqual(stages, [])
            self.assertEqual(sum(mask.values()), 0)

    def test_progressive_schedule_phase2(self):
        for ep in range(9, 16):
            mask, stages = resolve_progressive_unfreeze_mask(self.cfg, ep, self.mock_backbone_vars)
            self.assertEqual(stages, [4])
            # Only stage 4 and downsample_stage4 should be True
            for name, is_trainable in mask.items():
                if "stage4" in name:
                    self.assertTrue(is_trainable, f"{name} should be trainable in phase 2")
                else:
                    self.assertFalse(is_trainable, f"{name} should be frozen in phase 2")

    def test_progressive_schedule_phase3(self):
        for ep in [16, 20, 45, 60]:
            mask, stages = resolve_progressive_unfreeze_mask(self.cfg, ep, self.mock_backbone_vars)
            self.assertEqual(stages, [3, 4])
            for name, is_trainable in mask.items():
                if "stage3" in name or "stage4" in name:
                    self.assertTrue(is_trainable, f"{name} should be trainable in phase 3")
                else:
                    self.assertFalse(is_trainable, f"{name} should be frozen in phase 3")

    def test_discriminative_lr_scales(self):
        scales = compute_stage_lr_scales(self.cfg, self.mock_backbone_vars, [3, 4])
        for name, scale in scales.items():
            if "stage4" in name:
                self.assertEqual(scale, 1.0)
            elif "stage3" in name:
                self.assertEqual(scale, 0.5)

    def test_ece_computation(self):
        # Deterministic test of ECE logic
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])  # 3/4 correct = 75% acc
        max_probs = np.array([0.8, 0.8, 0.8, 0.8])  # uniform confidence 80%
        # Diff = |0.75 - 0.80| = 0.05
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
        self.assertAlmostEqual(ece, 0.05, places=5)

    def test_logit_adjustment(self):
        labels = np.array([0]*100 + [1]*50 + [2]*25 + [3]*200 + [4]*80 + [5]*60 + [6]*120)
        report = compute_logit_adjustment(labels, tau=0.5, num_classes=7)
        self.assertEqual(len(report["offsets"]), 7)
        # Most frequent class (class 3 with 200) should have the highest offset (least negative)
        self.assertEqual(int(np.argmax(report["offsets"])), 3)
        # Least frequent class (class 2 with 25) should have the lowest offset (most negative)
        self.assertEqual(int(np.argmin(report["offsets"])), 2)


if __name__ == "__main__":
    unittest.main()
