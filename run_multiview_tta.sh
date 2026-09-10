#!/bin/bash
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="${1:-$ROOT/config_rafdb_siglip2_semantic_stable_v3.yaml}"
CHECKPOINT="${2:-}"

export NVIDIA_LIB=$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

if [ -n "$CHECKPOINT" ]; then
    "$FER_PY" scripts/evaluate_multiview_tta.py --config "$CONFIG" --checkpoint "$CHECKPOINT"
else
    "$FER_PY" scripts/evaluate_multiview_tta.py --config "$CONFIG"
fi
