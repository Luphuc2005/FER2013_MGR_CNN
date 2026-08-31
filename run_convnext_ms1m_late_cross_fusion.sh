#!/bin/bash
# Local/Direct execution script for ConvNeXt-B MS1M Late Cross-Stage Attention + Direct Feature Fusion

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config_kaggle_convnext_ms1m_late_cross_fusion.yaml"

python scripts/train_convnext_ms1m_late_cross_fusion.py --config "$CONFIG" "$@"
