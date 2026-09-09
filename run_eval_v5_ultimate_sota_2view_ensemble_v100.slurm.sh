#!/bin/bash
#SBATCH --job-name=EVAL_V5_SOTA_2VIEW
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_SOTA_2VIEW_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_SOTA_2VIEW_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota"
CKPT_BEST_ACC="$OUTPUT_DIR/checkpoints/best"
CKPT_BEST_LOSS="$OUTPUT_DIR/checkpoints/best_loss"

export NVIDIA_LIB=$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " Running FAST 2-View TTA (Origin + Flip) Ensemble for v5 SOTA Tuned"
echo " Config:         $CONFIG"
echo " Best Acc Dir:   $CKPT_BEST_ACC"
echo " Best Loss Dir:  $CKPT_BEST_LOSS"
echo " Mode:           Origin + Flip Only (No zoom/rotation sweep)"
echo " Batch Size:     32"
echo " Target Split:   TEST"
echo " Start:          $(date)"
echo "=========================================================================="

nvidia-smi

"$FER_PY" scripts/evaluate_ensemble_multiview_tta.py \
  --config "$CONFIG" \
  --checkpoint-dirs "$CKPT_BEST_ACC" "$CKPT_BEST_LOSS" \
  --two-view-only \
  --batch-size 32 \
  --split test \
  --output "$OUTPUT_DIR/eval_ensemble_2view_best_acc_and_loss_report.json"

echo "=========================================================================="
echo " Evaluation completed successfully at $(date)"
echo "=========================================================================="
