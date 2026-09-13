#!/bin/bash
#SBATCH --job-name=FER_EVAL_15CKPTS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_EVAL_15CKPTS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_EVAL_15CKPTS_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

# Config and Experiment directory (Default to Job 6842: siglip2-confusion_v4, or override via env)
CONFIG="${CONFIG:-$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml}"
EXP_DIR="${EXP_DIR:-$ROOT/outputs/papers/siglip2-confusion_v4}"

echo "=========================================================================="
echo " 15-CHECKPOINT COMPREHENSIVE EVALUATION & ENSEMBLE PIPELINE (FER2013)"
echo " Evaluating best (acc), best_loss, best_macro_f1 & massive ensemble sweep"
echo "=========================================================================="
echo " Job ID:            ${SLURM_JOB_ID:-standalone}"
echo " Node:              $(hostname)"
echo " CUDA Visible:      ${CUDA_VISIBLE_DEVICES:-}"
echo " Python:            $FER_PY"
echo " Config:            $CONFIG"
echo " Exp Directory:     $EXP_DIR"
echo " Start Time:        $(date)"
echo "=========================================================================="

nvidia-smi

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# Optimize runtime
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_CUDNN_USE_AUTOTUNE=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

echo ""
echo "=== KIỂM TRA TOÀN BỘ 15 CHECKPOINTS TRONG CÁC THƯ MỤC ==="
echo "--- [1] checkpoints/best/ (Ranked by Val Accuracy):"
ls -lh "$EXP_DIR/checkpoints/best"/*.index 2>/dev/null || echo "Chưa có checkpoint trong checkpoints/best"

echo "--- [2] checkpoints/best_loss/ (Ranked by Lowest Val Loss):"
ls -lh "$EXP_DIR/checkpoints/best_loss"/*.index 2>/dev/null || echo "Chưa có checkpoint trong checkpoints/best_loss"

echo "--- [3] checkpoints/best_macro_f1/ (Ranked by Val Macro F1):"
ls -lh "$EXP_DIR/checkpoints/best_macro_f1"/*.index 2>/dev/null || echo "Chưa có checkpoint trong checkpoints/best_macro_f1"

echo "--- [4] checkpoints/periodic/ (Periodic Snapshots):"
ls -lh "$EXP_DIR/checkpoints/periodic"/*.index 2>/dev/null || echo "Chưa có checkpoint trong checkpoints/periodic"
echo "=========================================================================="

# ==========================================================================
# PHASE 1: TỰ ĐỘNG QUÉT & ĐÁNH GIÁ TỪNG CHECKPOINT TRONG TẤT CẢ 15 CHECKPOINTS
# - TTA sweep (tìm w_orig, w_flip tối ưu trên validation)
# - Tính Val Acc, Val Loss, Val Macro F1 cho từng checkpoint
# - Tính Test No-TTA, Test TTA, Test Macro F1 cho từng checkpoint
# - In bảng xếp hạng 15 checkpoint để người dùng so sánh lựa chọn
# - Xuất báo cáo chi tiết cho từng checkpoint vào $EXP_DIR/eval_individual_ckpts/
# ==========================================================================
echo ""
echo "=========================================================================="
echo " [PHASE 1] CHẠY TTA SWEEP & ĐÁNH GIÁ TOÀN BỘ CHECKPOINT (TỐI ĐA 15 CKPTS)"
echo "=========================================================================="

"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
    --config "$CONFIG" \
    --exp-dir "$EXP_DIR" \
    --step 0.05 \
    --num-comb-samples 50000 \
    --save-individual

# ==========================================================================
# PHASE 2: ĐÁNH GIÁ ĐƠN LẺ VỚI EVALUATE.PY (NO-TTA & TTA 50/50 CHO TỪNG FILE)
# Lưu kết quả riêng rẽ từng checkpoint vào $EXP_DIR/eval_per_checkpoint/
# ==========================================================================
echo ""
echo "=========================================================================="
echo " [PHASE 2] ĐÁNH GIÁ CHUẨN HÓA VỚI EVALUATE.PY (NO-TTA & TTA 50/50)"
echo "=========================================================================="

EVAL_DIR="$EXP_DIR/eval_per_checkpoint"
mkdir -p "$EVAL_DIR"

evaluate_single() {
    local ckpt_prefix="$1"
    local category="$2"
    local base_name
    base_name=$(basename "$ckpt_prefix")

    echo ">>> Đánh giá [$category]: $base_name"

    # 1. No-TTA Test
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --no-tta-hflip \
        --output "$EVAL_DIR/${category}_${base_name}_no_tta.json" 2>/dev/null || true

    # 2. TTA Test (50/50 HFlip)
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --tta-hflip \
        --orig-weight 0.50 \
        --flip-weight 0.50 \
        --output "$EVAL_DIR/${category}_${base_name}_tta_5050.json" 2>/dev/null || true
}

# Đánh giá 5 checkpoints trong best/
if [ -d "$EXP_DIR/checkpoints/best" ]; then
    for idx_file in $(ls -1 "$EXP_DIR/checkpoints/best"/*.index 2>/dev/null | sort -V); do
        evaluate_single "${idx_file%.index}" "best_acc"
    done
fi

# Đánh giá 5 checkpoints trong best_loss/
if [ -d "$EXP_DIR/checkpoints/best_loss" ]; then
    for idx_file in $(ls -1 "$EXP_DIR/checkpoints/best_loss"/*.index 2>/dev/null | sort -V); do
        evaluate_single "${idx_file%.index}" "best_loss"
    done
fi

# Đánh giá 5 checkpoints trong best_macro_f1/
if [ -d "$EXP_DIR/checkpoints/best_macro_f1" ]; then
    for idx_file in $(ls -1 "$EXP_DIR/checkpoints/best_macro_f1"/*.index 2>/dev/null | sort -V); do
        evaluate_single "${idx_file%.index}" "best_macro_f1"
    done
fi

echo ""
echo "=========================================================================="
echo " TẤT CẢ 15 CHECKPOINTS ĐÃ ĐƯỢC ĐÁNH GIÁ HOÀN TẤT!"
echo " 1. Báo cáo tổng hợp TTA & Ensemble 15 checkpoint : $EXP_DIR/val_tuned_tta_ensemble_report.json"
echo " 2. Báo cáo chi tiết từng checkpoint (TTA-tuned)  : $EXP_DIR/eval_individual_ckpts/"
echo " 3. Báo cáo đánh giá chuẩn hóa evaluate.py        : $EXP_DIR/eval_per_checkpoint/"
echo " End Time: $(date)"
echo "=========================================================================="
