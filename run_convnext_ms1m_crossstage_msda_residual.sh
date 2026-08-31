#!/usr/bin/env bash
# Local/server launcher for the isolated ConvNeXt-B MS1M Cross-Stage MSDA Residual experiment.

set -euo pipefail

CONFIG="${CONFIG:-config_convnext_ms1m_crossstage_msda_residual.yaml}"

echo "================================================================"
echo " Starting ConvNeXt-B MS1M Cross-Stage MSDA Residual (AdamW only)"
echo "================================================================"
echo "CONFIG=$CONFIG"
echo "OUTPUT=outputs/tf_runs/convnext_ms1m_crossstage_msda_residual"

python -u scripts/train_convnext_ms1m_crossstage_msda_residual.py --config "$CONFIG"

echo "================================================================"
echo " Training completed"
echo "================================================================"
