#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MGR_GPU_IDS=0
export MGR_REQUIRE_TWO_GPUS=0
export MGR_MIN_GPUS=1
export MGR_BATCH_SIZE_PER_GPU=8

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[INFO] Starting 1x V100 Local Expert SAM Training (100 Epochs)..."
"${PYTHON_BIN}" check_environment.py | tee "logs/check_environment_v100_local_expert_sam_${STAMP}.log"
"${PYTHON_BIN}" train.py --config config_1gpu_convnext_local_expert_sam_100epochs_v100.yaml 2>&1 | tee "logs/train_v100_local_expert_sam_${STAMP}.log"
