#!/usr/bin/env python3
"""Sanity and mathematical verification of Logit Adjustment (ICLR 2021)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logit_adjustment import compute_logit_adjustment, configure_training_logit_adjustment, logit_adjustment_kwargs


class TestLogitAdjustment(unittest.TestCase):
    def test_offset_math(self):
        # Simulated RAF-DB train counts:
        # angry: 705, disgust: 717, fear: 281, happy: 4772, sad: 1982, surprise: 1290, neutral: 2524
        counts = [705, 717, 281, 4772, 1982, 1290, 2524]
        labels = np.repeat(np.arange(7), counts)
        total = float(len(labels))

        report = compute_logit_adjustment(labels, tau=0.5, num_classes=7)
        self.assertEqual(report["train_samples"], 12271)
        self.assertEqual(report["counts"], counts)

        offsets = np.array(report["offsets"])
        priors = np.array(report["priors"])

        # Fear (idx 2) is the rarest class; Happy (idx 3) is the most frequent
        self.assertEqual(np.argmin(priors), 2)
        self.assertEqual(np.argmax(priors), 3)

        # Offsets should be strictly monotonic with priors (smaller prior -> more negative offset)
        self.assertTrue(offsets[2] < offsets[3])

        # Mathematical verification: offset = tau * log(prior)
        expected_offsets = 0.5 * np.log(np.array(counts, dtype=np.float64) / total)
        np.testing.assert_allclose(offsets, expected_offsets, rtol=1e-5)

        # Margin difference between Fear and Happy: tau * (log(pi_happy) - log(pi_fear))
        margin_diff = offsets[3] - offsets[2]
        self.assertGreater(margin_diff, 1.0)
        print(f"\n[OK] Logit adjustment margin boost for Fear vs Happy: +{margin_diff:.3f}")

    def test_config_integration(self):
        cfg = {
            "data": {"num_classes": 7},
            "training": {
                "logit_adjustment": {"enabled": True, "tau": 0.5}
            }
        }
        labels = np.tile(np.arange(7), 100)
        configure_training_logit_adjustment(cfg, labels)
        self.assertIn("resolved_logit_adjustment_report", cfg["training"])

        kwargs = logit_adjustment_kwargs(cfg)
        self.assertIn("logit_adj_offsets", kwargs)
        self.assertEqual(len(kwargs["logit_adj_offsets"]), 7)

    def test_disabled_returns_empty(self):
        cfg = {"training": {"logit_adjustment": {"enabled": False}}}
        kwargs = logit_adjustment_kwargs(cfg)
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
