#!/bin/bash
#SBATCH --job-name=EVAL_FERPLUS_CKPT33
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_FERPLUS_CKPT33_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_FERPLUS_CKPT33_%j.err

# GHI CHÚ: Nếu cluster yêu cầu bắt buộc phải có --gres=gpu mới submit được vào gpu-queue,
# hãy bỏ comment dòng dưới. Script bên dưới vẫn ép CUDA_VISIBLE_DEVICES="-1" để chạy 100% CPU.
# #SBATCH --gres=gpu:v100:1

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/ferplus_siglip2_confusion_majority8

# Ép chặt chạy CPU thuần túy, ẩn toàn bộ GPU để không tranh chấp tài nguyên với job training GPU
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"

# Mặc định đánh giá checkpoint 33 (có thể truyền tham số khác khi gọi: sbatch ... 33)
CKPT_TAG="${1:-33}"
CKPT_NAME="ckpt-${CKPT_TAG#ckpt-}"

CHECKPOINT="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/best/${CKPT_NAME}"

echo "============================================================"
echo " FERPlus Test Evaluation (CPU Standalone) - ${CKPT_NAME}"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID:-standalone}"
echo "Node                : $(hostname)"
echo "Allocated CPUs      : ${SLURM_CPUS_PER_TASK:-8}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time          : $(date)"
echo "ROOT                : $ROOT"
echo "Python              : $FER_PY"
echo "Config              : $CONFIG"
echo "Target Checkpoint   : $CHECKPOINT"
echo "============================================================"

[ -x "$FER_PY" ] || { echo "[ERROR] Python interpreter not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

# Tìm file checkpoint (trong best/, nếu chưa có thì tìm tự động trong last/ hoặc các thư mục con khác)
if [ ! -f "${CHECKPOINT}.index" ]; then
    echo "[WARNING] ${CHECKPOINT}.index not found in best/."
    echo "[INFO] Searching for ${CKPT_NAME} across all checkpoint subdirectories..."
    FOUND_CKPT=$(ls $ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/*/${CKPT_NAME}.index 2>/dev/null | head -n 1 | sed 's/\.index$//' || true)
    if [ -n "$FOUND_CKPT" ] && [ -f "${FOUND_CKPT}.index" ]; then
        echo "[INFO] Found ${CKPT_NAME} at: $FOUND_CKPT"
        CHECKPOINT="$FOUND_CKPT"
    else
        echo "[ERROR] Could not find ${CKPT_NAME}. Currently available checkpoints:"
        ls -la "$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints"/* 2>/dev/null || true
        exit 1
    fi
fi

echo "============================================================"
echo " Running Evaluation on FERPlus Test Split via CPU..."
echo "============================================================"
"$FER_PY" -u evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --cpu

echo "============================================================"
echo " FERPlus Test Evaluation Completed at $(date)"
echo " Log output saved to: logs/EVAL_FERPLUS_CKPT33_${SLURM_JOB_ID:-standalone}.out"
echo " Metrics saved to   : outputs/papers/ferplus_siglip2_confusion_majority8/test_metrics_${CKPT_NAME}_tta_hflip.json"
echo "============================================================"
