#!/bin/bash
#SBATCH --job-name=EVAL_IND_ENS_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_IND_ENS_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_IND_ENS_CPU_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CPUS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export TF_NUM_INTRAOP_THREADS="$CPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

# Tự động nhận diện thư mục đầu ra (Ưu tiên tham số $1 -> thư mục mới nhất của seed42 -> fallback)
TARGET_DIR="${1:-}"
if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR=$(ls -td outputs/papers/siglip2-confusion-seed42* outputs/papers/siglip2-confusion* 2>/dev/null | head -n 1 || true)
fi

if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Could not find target output directory! Please provide path: sbatch $0 <output_dir>"
    exit 1
fi

if [[ "$TARGET_DIR" == *"seed42"* ]]; then
    CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed42.yaml"
else
    CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
fi

CKPT_DIR="$TARGET_DIR/checkpoints/best"

echo "============================================================"
echo " EVALUATION (PURE CPU - 32 THREADS): INDIVIDUAL + ENSEMBLE"
echo "============================================================"
echo "Job ID         : ${SLURM_JOB_ID:-standalone}"
echo "Node           : $(hostname)"
echo "CPUs allocated : $CPUS threads"
echo "CUDA DEVICES   : $CUDA_VISIBLE_DEVICES (Pure CPU Mode)"
echo "Start time     : $(date)"
echo "Config file    : $CONFIG"
echo "Target Dir     : $TARGET_DIR"
echo "Checkpoint Dir : $CKPT_DIR"
echo "============================================================"

[ -x "$FER_PY" ] || { echo "[ERROR] Python not found: $FER_PY"; exit 1; }
[ -d "$CKPT_DIR" ] || { echo "[ERROR] Checkpoints directory not found: $CKPT_DIR"; exit 1; }

echo -e "\n[INFO] Checkpoints available in $CKPT_DIR:"
ls -lh "$CKPT_DIR"/ckpt-*.index 2>/dev/null || true

# 1. Chạy đánh giá chi tiết từng Checkpoint và tính Softmax Ensemble bằng CPU
echo -e "\n============================================================"
echo " Running Individual Checkpoint Evaluations + Ensemble on CPU (TEST SPLIT)"
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py \
    --config "$CONFIG" \
    --checkpoint-dir "$CKPT_DIR" \
    --split test \
    --cpu \
    --w-orig 0.40 \
    --w-flip 0.60

echo "============================================================"
echo " CPU Evaluation Pipeline Completed Successfully at: $(date)"
echo "============================================================"
