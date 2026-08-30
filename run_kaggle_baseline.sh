#!/usr/bin/env bash
# Script to run ConvNeXt-Base MS1M-ArcFace Baseline (76.26%) on Kaggle GPU

set -e

echo "========================================================"
echo " Starting ConvNeXt-Base MS1M ArcFace Baseline on Kaggle "
echo "========================================================"

python train.py --config config_kaggle_convnext_base_ms1m_arcface_baseline.yaml

echo "========================================================"
echo " Baseline Training Completed Successfully!              "
echo "========================================================"
