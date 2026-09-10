#!/bin/bash
#SBATCH --job-name=EVAL_V5_LGSA_ENS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_LGSA_ENS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_LGSA_ENS_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_lgsa.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_lgsa"
CKPT_BEST_ACC="$OUTPUT_DIR/checkpoints/best"
CKPT_BEST_LOSS="$OUTPUT_DIR/checkpoints/best_loss"
REPORT_OUTPUT="$OUTPUT_DIR/eval_ensemble_best_acc_and_loss_report.json"

export NVIDIA_LIB=$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# Thư mục log
mkdir -p "$ROOT/logs"

echo "=========================================================================="
echo " Running Ensemble Evaluation (Best Acc + Best Loss) on GPU (32GB VRAM)"
echo " Config:         $CONFIG"
echo " Best Acc Dir:   $CKPT_BEST_ACC"
echo " Best Loss Dir:  $CKPT_BEST_LOSS"
echo " Target Split:   TEST"
echo " Batch Size:     64 (Tận dụng 32GB VRAM cho Multi-View TTA 6-views = 384 imgs/step)"
echo " Start Time:     $(date)"
echo " Job ID:         ${SLURM_JOB_ID:-standalone}"
echo " Node:           $(hostname)"
echo "=========================================================================="

nvidia-smi

# Kiểm tra thư mục checkpoint tồn tại
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

# Chạy script Ensemble Multi-View TTA:
# - Đánh giá từng checkpoint riêng lẻ (No-TTA, 2-View, 4-View, 5-View, 6-View)
# - Đánh giá Ensemble Top Checkpoints theo từng nhóm (best, best_loss)
# - Đánh giá GRAND JOINT ENSEMBLE kết hợp cả Best Acc và Best Loss
# - Tự động sweep tỉ lệ trọng số TTA Origin : Flip (0% -> 100%, bước 5%)
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
