#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export MGR_GPU_IDS="0,1"
export CUDA_VISIBLE_DEVICES="0,1"
export MGR_REQUIRE_TWO_GPUS=1
export MGR_MIN_GPUS=2
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

mkdir -p /kaggle/working/logs

STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="/kaggle/working/logs/train_2stage_200epochs_${STAMP}.log"

echo "[INFO] Running Automated 2-Stage Training (200 Epochs) for MGR-CNN on Kaggle..."
echo "[INFO] Logging to: ${TRAIN_LOG}"

python3 train.py --config config_kaggle_2stage_200epochs_7512.yaml 2>&1 | tee -a "${TRAIN_LOG}"
