#!/usr/bin/env bash
# Local/server launcher for the SAM variant of the isolated Cross-Stage MSDA Residual experiment.

set -euo pipefail

CONFIG="${CONFIG:-config_convnext_ms1m_crossstage_msda_residual_sam.yaml}"

echo "================================================================"
echo " Starting ConvNeXt-B MS1M Cross-Stage MSDA Residual (SAM+AdamW)"
echo "================================================================"
echo "CONFIG=$CONFIG"
echo "OUTPUT=outputs/tf_runs/convnext_ms1m_crossstage_msda_residual_sam"

python -u scripts/train_convnext_ms1m_crossstage_msda_residual.py --config "$CONFIG"

echo "================================================================"
echo " Training completed"
echo "================================================================"
