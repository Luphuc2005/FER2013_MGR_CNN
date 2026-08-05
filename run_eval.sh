#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MGR_GPU_IDS="${MGR_GPU_IDS:-0,1}"
export MGR_REQUIRE_TWO_GPUS="${MGR_REQUIRE_TWO_GPUS:-0}"
export MGR_MIN_GPUS="${MGR_MIN_GPUS:-1}"

SPLIT="${1:-test}"
"${PYTHON_BIN}" evaluate.py --config config.yaml --split "${SPLIT}"
