#!/bin/bash
#SBATCH --job-name=EVAL_V4_COMB_ENS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V4_COMB_ENS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V4_COMB_ENS_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v4_combined.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v4_combined"
CKPT_BEST_ACC="$OUTPUT_DIR/checkpoints/best"
CKPT_BEST_LOSS="$OUTPUT_DIR/checkpoints/best_loss"
REPORT_OUTPUT="$OUTPUT_DIR/eval_ensemble_best_acc_and_loss_report.json"

export NVIDIA_LIB=$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ROOT/logs"

echo "=========================================================================="
echo " Running Ensemble Evaluation for v4 Combined (Best Acc + Best Loss)"
echo " Config:         $CONFIG"
echo " Best Acc Dir:   $CKPT_BEST_ACC"
echo " Best Loss Dir:  $CKPT_BEST_LOSS"
echo " Target Split:   TEST"
echo " Batch Size:     64 (Tối ưu 32GB VRAM cho Multi-View TTA 6-views)"
echo " Start Time:     $(date)"
echo " Job ID:         ${SLURM_JOB_ID:-standalone}"
echo " Node:           $(hostname)"
echo "=========================================================================="

nvidia-smi

if [ ! -d "$CKPT_BEST_ACC" ]; then
    echo "[ERROR] Checkpoint best dir not found: $CKPT_BEST_ACC"
    exit 1
fi

CHECKPOINT_DIRS=("$CKPT_BEST_ACC")
if [ -d "$CKPT_BEST_LOSS" ]; then
    CHECKPOINT_DIRS+=("$CKPT_BEST_LOSS")
    echo "[INFO] Found best_loss directory. Running Joint Ensemble (Best Acc + Best Loss)!"
else
    echo "[WARNING] best_loss directory not found ($CKPT_BEST_LOSS). Running only Best Acc checkpoints."
fi

# Chạy Multi-View TTA + Sweep:
# - Đánh giá từng checkpoint (best và best_loss)
# - Đánh giá Ensemble riêng từng nhóm
# - Đánh giá GRAND JOINT ENSEMBLE gộp toàn bộ checkpoint tốt nhất
# - Sweep tỉ lệ trọng số Origin : Flip từ 0% đến 100% (bước 5%)
"$FER_PY" -u scripts/evaluate_ensemble_multiview_tta.py \
  --config "$CONFIG" \
  --checkpoint-dirs "${CHECKPOINT_DIRS[@]}" \
  --split test \
  --batch-size 64 \
  --sweep-step 0.05 \
  --output "$REPORT_OUTPUT"

echo "=========================================================================="
echo " Evaluation completed successfully at $(date)"
echo " Report saved to: $REPORT_OUTPUT"
echo "=========================================================================="
