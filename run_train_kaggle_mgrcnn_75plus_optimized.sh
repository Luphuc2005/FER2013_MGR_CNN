#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_FILE="${1:-config_kaggle_mgrcnn_75plus_optimized.yaml}"

echo "================================================================="
echo "[INFO] Starting MGR-CNN 75%+ Optimized Training Pipeline"
echo "[INFO] Config: ${CONFIG_FILE}"
echo "================================================================="

python3 -u train.py --config "${CONFIG_FILE}" "${@:2}"
