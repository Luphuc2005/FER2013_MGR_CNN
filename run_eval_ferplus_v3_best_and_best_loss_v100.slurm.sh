#!/bin/bash
#SBATCH --job-name=FERPLUS_V3_EVAL
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FERPLUS_V3_EVAL_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FERPLUS_V3_EVAL_%j.err

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"

mkdir -p "$ROOT/logs"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
export MGR_TF_INTRA_OP_THREADS=12
export MGR_TF_INTER_OP_THREADS=4
export MGR_TF_DATA_NUM_PARALLEL_CALLS=12
export MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE=16
export MGR_PREFETCH_BUFFER=4

FER_PY="$ROOT/fer2013_env/bin/python"
if [ ! -x "$FER_PY" ]; then
    FER_PY="$(which python3 || which python)"
fi

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

EXP_DIR="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8_v3"
CKPT_BEST_DIR="$EXP_DIR/checkpoints/best"
CKPT_BEST_LOSS_DIR="$EXP_DIR/checkpoints/best_loss"

CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion_v3.yaml"
if [ ! -f "$CONFIG" ]; then
    CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
fi

echo "=========================================================================="
echo " HPC V100 EVALUATION: FERPlus Majority-8 (v3)"
echo " Best Checkpoint (Val Accuracy) & Best Loss Checkpoint"
echo "=========================================================================="
echo " Job ID:            ${SLURM_JOB_ID:-standalone}"
echo " Node:              $(hostname)"
echo " Allocated CPUs:    ${SLURM_CPUS_PER_TASK:-16}"
echo " CUDA Visible:      ${CUDA_VISIBLE_DEVICES:-}"
echo " Python:            $FER_PY"
echo " Config:            $CONFIG"
echo " Output Dir:        $EXP_DIR"
echo " Best Acc Dir:      $CKPT_BEST_DIR"
echo " Best Loss Dir:     $CKPT_BEST_LOSS_DIR"
echo " Start Time:        $(date)"
echo "=========================================================================="

nvidia-smi

# 1. Định vị Checkpoint Best Accuracy (ưu tiên ckpt-17)
BEST_ACC_CKPT=""
if [ -f "$CKPT_BEST_DIR/ckpt-17.index" ]; then
    BEST_ACC_CKPT="$CKPT_BEST_DIR/ckpt-17"
elif [ -d "$CKPT_BEST_DIR" ]; then
    BEST_ACC_CKPT=$(ls -1 "$CKPT_BEST_DIR"/*.index 2>/dev/null | sort -V | tail -n 1 | sed 's/\.index$//' || true)
fi

# 2. Định vị Checkpoint Best Loss
BEST_LOSS_CKPT=""
if [ -d "$CKPT_BEST_LOSS_DIR" ]; then
    BEST_LOSS_CKPT=$(ls -1 "$CKPT_BEST_LOSS_DIR"/*.index 2>/dev/null | sort -V | tail -n 1 | sed 's/\.index$//' || true)
fi

echo ""
echo "[INFO] Detected Best Accuracy Checkpoint: ${BEST_ACC_CKPT:-NONE}"
echo "[INFO] Detected Best Loss Checkpoint:     ${BEST_LOSS_CKPT:-NONE}"
echo ""

# ==========================================================================
# PHASE 1: ĐÁNH GIÁ ĐƠN LẺ TỪNG CHECKPOINT (No-TTA & TTA HFlip)
# ==========================================================================

if [ -n "$BEST_ACC_CKPT" ]; then
    echo "=========================================================================="
    echo " [PHASE 1.1] ĐÁNH GIÁ CHECKPOINT BEST ACCURACY: $(basename "$BEST_ACC_CKPT")"
    echo "=========================================================================="
    
    echo ">>> [1/2] Đánh giá No-TTA..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_ACC_CKPT" \
        --split test \
        --no-tta-hflip \
        --output "$EXP_DIR/eval_$(basename "$BEST_ACC_CKPT")_no_tta.json"

    echo ">>> [2/2] Đánh giá TTA (HFlip 50/50)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_ACC_CKPT" \
        --split test \
        --tta-hflip \
        --orig-weight 0.50 \
        --flip-weight 0.50 \
        --output "$EXP_DIR/eval_$(basename "$BEST_ACC_CKPT")_tta_hflip.json"
fi

if [ -n "$BEST_LOSS_CKPT" ]; then
    echo "=========================================================================="
    echo " [PHASE 1.2] ĐÁNH GIÁ CHECKPOINT BEST LOSS: $(basename "$BEST_LOSS_CKPT")"
    echo "=========================================================================="
    
    echo ">>> [1/2] Đánh giá No-TTA..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_LOSS_CKPT" \
        --split test \
        --no-tta-hflip \
        --output "$EXP_DIR/eval_best_loss_$(basename "$BEST_LOSS_CKPT")_no_tta.json"

    echo ">>> [2/2] Đánh giá TTA (HFlip 50/50)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_LOSS_CKPT" \
        --split test \
        --tta-hflip \
        --orig-weight 0.50 \
        --flip-weight 0.50 \
        --output "$EXP_DIR/eval_best_loss_$(basename "$BEST_LOSS_CKPT")_tta_hflip.json"
fi

# ==========================================================================
# PHASE 2: GRAND JOINT ENSEMBLE + MULTI-VIEW TTA SWEEP
# Kết hợp cả Best Accuracy và Best Loss checkpoints
# ==========================================================================
echo "=========================================================================="
echo " [PHASE 2] GRAND ENSEMBLE + MULTI-VIEW TTA EVALUATION"
echo "=========================================================================="

EVAL_DIRS=()
if [ -d "$CKPT_BEST_DIR" ]; then
    EVAL_DIRS+=("$CKPT_BEST_DIR")
fi
if [ -d "$CKPT_BEST_LOSS_DIR" ] && [ "$(ls -A "$CKPT_BEST_LOSS_DIR" 2>/dev/null)" ]; then
    EVAL_DIRS+=("$CKPT_BEST_LOSS_DIR")
fi

if [ ${#EVAL_DIRS[@]} -gt 0 ]; then
    REPORT_OUTPUT="$EXP_DIR/eval_grand_ensemble_best_acc_and_loss_report.json"
    
    echo "[INFO] Running Joint Ensemble over dirs: ${EVAL_DIRS[*]}"
    echo "[INFO] Report destination: $REPORT_OUTPUT"
    
    "$FER_PY" -u scripts/evaluate_ensemble_multiview_tta.py \
        --config "$CONFIG" \
        --checkpoint-dirs "${EVAL_DIRS[@]}" \
        --split test \
        --batch-size 64 \
        --sweep-step 0.05 \
        --output "$REPORT_OUTPUT"
else
    echo "[WARNING] No valid checkpoint directory found to run ensemble."
fi

echo "=========================================================================="
echo " HPC V100 EVALUATION HOÀN TẤT THÀNH CÔNG!"
echo " Kết thúc: $(date)"
echo "=========================================================================="
