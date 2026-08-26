#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MGR_GPU_IDS="${MGR_GPU_IDS:-0,1,2,3}"
export MGR_REQUIRE_TWO_GPUS=0
export MGR_MIN_GPUS=4
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-16}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-4}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_2stage_4gpu_${STAMP}.log"

echo "[INFO] Running 2-Stage Automated Pipeline (75.12%+ 200 Epochs) on 4x GPU Node..."
echo "[INFO] Log file: ${TRAIN_LOG}"

"${PYTHON_BIN}" train.py --config config_2stage_4gpu_200epochs_7512.yaml 2>&1 | tee -a "${TRAIN_LOG}"
