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

echo "[INFO] Starting Fast 2-Stage Trial Run (5 Epochs) on Kaggle..."
python3 train.py --config config_kaggle_2stage_trial.yaml 2>&1 | tee -a /kaggle/working/logs/train_2stage_trial.log
