#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FER_PY="$ROOT/fer2013_env/bin/python"
if [ ! -f "$FER_PY" ]; then
  FER_PY="python3"
fi

CONFIG="${1:-config_rafdb_siglip2_semantic_stable_v3.yaml}"
CHECKPOINT="${2:-outputs/papers/rafdb_siglip2_semantic_stable_v3/checkpoints/best/ckpt-52}"
SPLIT="${3:-test}"

export NVIDIA_LIB="$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia"
if [ -d "$NVIDIA_LIB" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"
fi

echo "============================================================"
echo " Sweeping SigLIP2 Semantic Fusion Alpha on RAF-DB"
echo " Config:     $CONFIG"
echo " Checkpoint: $CHECKPOINT"
echo " Split:      $SPLIT"
echo " Start:      $(date)"
echo ""
echo ">>> [PHASE 1: NO-TTA (Ảnh gốc thuần túy, xuất phát điểm 90.97%)] <<<"
"$FER_PY" scripts/sweep_rafdb_semantic_fusion.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --no-tta \
  --alphas 0.0 0.02 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30

echo ""
echo ">>> [PHASE 2: WITH TTA (Ảnh gốc + Lật ngang)] <<<"
"$FER_PY" scripts/sweep_rafdb_semantic_fusion.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --alphas 0.0 0.02 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30

echo "============================================================"
echo " Finished at $(date)"
echo "============================================================"
