#!/bin/bash
#SBATCH --job-name=EVAL_FERPLUS_ENSEMBLE_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_FERPLUS_ENSEMBLE_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_FERPLUS_ENSEMBLE_CPU_%j.err

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
CKPT_DIR="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/best"

echo "============================================================"
echo " FERPlus Individual Checkpoints & Ensemble Evaluation (CPU)"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID:-standalone}"
echo "Node                : $(hostname)"
echo "Allocated CPUs      : ${SLURM_CPUS_PER_TASK:-8}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time          : $(date)"
echo "ROOT                : $ROOT"
echo "Python              : $FER_PY"
echo "Config              : $CONFIG"
echo "Checkpoint Dir      : $CKPT_DIR"
echo "============================================================"

[ -x "$FER_PY" ] || { echo "[ERROR] Python interpreter not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

echo "[INFO] Listing available checkpoints in $CKPT_DIR:"
ls -lh "$CKPT_DIR"/*.index || true

echo "============================================================"
echo " Running Evaluation on FERPlus Test Split (Individual + Ensemble)..."
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py \
    --config "$CONFIG" \
    --checkpoint-dir "$CKPT_DIR" \
    --split test \
    --cpu

echo "============================================================"
echo " FERPlus Ensemble Evaluation Completed at $(date)"
echo " Log saved to: logs/EVAL_FERPLUS_ENSEMBLE_CPU_${SLURM_JOB_ID:-standalone}.out"
echo "============================================================"
