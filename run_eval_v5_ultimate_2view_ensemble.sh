#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FER_PY="$ROOT/fer2013_env/bin/python"
if [ ! -f "$FER_PY" ]; then
  FER_PY="python3"
fi

CONFIG="${1:-config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml}"
OUTPUT_DIR="outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate"
CKPT_BEST_ACC="$OUTPUT_DIR/checkpoints/best"
CKPT_BEST_LOSS="$OUTPUT_DIR/checkpoints/best_loss"
SPLIT="${2:-test}"

export NVIDIA_LIB="$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia"
if [ -d "$NVIDIA_LIB" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"
fi

echo "=========================================================================="
echo " Running FAST 2-View TTA (Origin + Flip) Ensemble for v5 Combined Ultimate"
echo " Config:         $CONFIG"
echo " Best Acc Dir:   $CKPT_BEST_ACC"
echo " Best Loss Dir:  $CKPT_BEST_LOSS"
echo " Target Split:   $SPLIT"
echo " Start:          $(date)"
echo "=========================================================================="

"$FER_PY" scripts/evaluate_ensemble_multiview_tta.py \
  --config "$CONFIG" \
  --checkpoint-dirs "$CKPT_BEST_ACC" "$CKPT_BEST_LOSS" \
  --two-view-only \
  --batch-size 32 \
  --split "$SPLIT" \
  --output "$OUTPUT_DIR/eval_ensemble_2view_best_acc_and_loss_report.json"

echo "=========================================================================="
echo " Evaluation finished at $(date)"
echo "=========================================================================="
