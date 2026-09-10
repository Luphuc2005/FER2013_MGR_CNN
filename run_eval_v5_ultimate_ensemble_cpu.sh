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
BATCH_SIZE="${3:-32}"
SWEEP_STEP="${4:-0.05}"

# Force purely CPU mode
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS=$(nproc 2>/dev/null || echo 32)
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=4

echo "=========================================================================="
echo " Running Ensemble + Multi-View TTA + TTA Sweep (CPU Mode)"
echo " Model / Run:    v5 Combined Ultimate (Dynamic Part Attention + Fusion Gate)"
echo " Config:         $CONFIG"
echo " Best Acc Dir:   $CKPT_BEST_ACC"
echo " Best Loss Dir:  $CKPT_BEST_LOSS"
echo " Target Split:   $SPLIT"
echo " Batch Size:     $BATCH_SIZE"
echo " Sweep Step:     $SWEEP_STEP"
echo " CPU Cores:      $NCPUS"
echo " Start Time:     $(date)"
echo "=========================================================================="

"$FER_PY" scripts/evaluate_ensemble_multiview_tta.py \
  --config "$CONFIG" \
  --checkpoint-dirs "$CKPT_BEST_ACC" "$CKPT_BEST_LOSS" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --sweep-ratio \
  --sweep-step "$SWEEP_STEP" \
  --cpu \
  --output "$OUTPUT_DIR/eval_ensemble_best_acc_and_loss_cpu_report.json"

echo "=========================================================================="
echo " CPU Evaluation finished at $(date)"
echo " Report saved to: $OUTPUT_DIR/eval_ensemble_best_acc_and_loss_cpu_report.json"
echo "=========================================================================="
