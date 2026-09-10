#!/bin/bash
#SBATCH --job-name=EVAL_FERPLUS_CPU_CKPT24
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_FERPLUS_CPU_CKPT24_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_FERPLUS_CPU_CKPT24_%j.err

# GHI CHÚ: Nếu cluster yêu cầu bắt buộc phải có --gres=gpu mới submit được vào gpu-queue,
# hãy bỏ comment dòng dưới. Script bên dưới vẫn ép CUDA_VISIBLE_DEVICES="-1" để chạy 100% CPU.
# #SBATCH --gres=gpu:v100:1

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/ferplus_siglip2_confusion_majority8

# Ép chặt chạy CPU thuần túy, tuyệt đối không cấp phát VRAM hay ảnh hưởng job training GPU
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
CHECKPOINT="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/best/ckpt-24"

echo "============================================================"
echo " FERPlus Test Evaluation (CPU Standalone) - Checkpoint Epoch 24"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID:-standalone}"
echo "Node                : $(hostname)"
echo "Allocated CPUs      : ${SLURM_CPUS_PER_TASK:-8}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time          : $(date)"
echo "ROOT                : $ROOT"
echo "Python              : $FER_PY"
echo "Config              : $CONFIG"
echo "Checkpoint Target   : $CHECKPOINT"
echo "Output Directory    : outputs/papers/ferplus_siglip2_confusion_majority8"
echo "============================================================"

[ -x "$FER_PY" ] || { echo "[ERROR] Python interpreter not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

# Kiểm tra file checkpoint ckpt-24
if [ ! -f "${CHECKPOINT}.index" ]; then
    echo "[WARNING] Checkpoint index file ${CHECKPOINT}.index not found in best/."
    echo "[INFO] Searching for ckpt-24 across all checkpoint subdirectories..."
    FOUND_CKPT=$(ls $ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/*/ckpt-24.index 2>/dev/null | head -n 1 | sed 's/\.index$//' || true)
    if [ -n "$FOUND_CKPT" ] && [ -f "${FOUND_CKPT}.index" ]; then
        echo "[INFO] Found ckpt-24 at: $FOUND_CKPT"
        CHECKPOINT="$FOUND_CKPT"
    else
        echo "[ERROR] Could not find ckpt-24. Existing checkpoints:"
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
echo " Log output saved to: logs/EVAL_FERPLUS_CPU_CKPT24_${SLURM_JOB_ID:-standalone}.out"
echo "============================================================"
