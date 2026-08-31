#!/usr/bin/env bash
# Script to run ConvNeXt-Base MS1M-ArcFace Baseline without SAM (AdamW)

set -e

echo "================================================================"
echo " Starting ConvNeXt-Base MS1M ArcFace Baseline (No SAM / AdamW) "
echo "================================================================"

python train.py --config config_convnext_base_ms1m_arcface_baseline_nosam.yaml

echo "================================================================"
echo " Training Completed Successfully!                               "
echo "================================================================"
