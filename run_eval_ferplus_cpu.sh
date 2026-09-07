#!/bin/bash
# ==============================================================================
# Script evaluation tập test FERPlus bằng CPU cho checkpoint ckpt-24
# An toàn 100%: không chiếm GPU, không dừng/ảnh hưởng job GPU đang chạy.
# ==============================================================================

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"

# Ép chặt CPU mode, ẩn toàn bộ GPU khỏi process này
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
CHECKPOINT="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/best/ckpt-24"

echo "=================================================================="
echo "    FERPlus Majority-8 Test Evaluation (CPU Standalone Mode)"
echo "=================================================================="
echo "Date       : $(date)"
echo "Host       : $(hostname)"
echo "CUDA Env   : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Python     : ${FER_PY}"
echo "Config     : ${CONFIG}"
echo "Checkpoint : ${CHECKPOINT}"
echo "=================================================================="

# Kiểm tra file checkpoint và python env
[ -x "$FER_PY" ] || { echo "[ERROR] Python interpreter not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config not found: $CONFIG"; exit 1; }

if [ ! -f "${CHECKPOINT}.index" ] && [ ! -d "${CHECKPOINT}" ]; then
    echo "[WARNING] Checkpoint index not found at ${CHECKPOINT}.index"
    echo "Listing available checkpoints in directory:"
    ls -l "$(dirname "$CHECKPOINT")" || true
fi

# Chạy evaluate.py trên CPU
"$FER_PY" -u evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --cpu

echo "=================================================================="
echo " Evaluation finished at $(date)"
echo "=================================================================="
