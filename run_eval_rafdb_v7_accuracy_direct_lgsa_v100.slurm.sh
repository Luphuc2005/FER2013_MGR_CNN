#!/bin/bash
#SBATCH --job-name=EVAL_RAFDB_V7
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_V7_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_V7_%j.err

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

CONFIG="$ROOT/config_rafdb_v7_accuracy_direct_lgsa.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_v7_accuracy_direct_lgsa"

CKPT_BEST_DIR="$EXP_DIR/checkpoints/best"
CKPT_BEST_LOSS_DIR="$EXP_DIR/checkpoints/best_loss"
CKPT_BEST_F1_DIR="$EXP_DIR/checkpoints/best_macro_f1"

echo "=========================================================================="
echo " HPC V100 EVALUATION: RAF-DB V7 ACCURACY DIRECT LGSA"
echo " Comprehensive Evaluation: best, best_loss, best_macro_f1 & Grand Ensemble"
echo "=========================================================================="
echo " Job ID:            ${SLURM_JOB_ID:-standalone}"
echo " Node:              $(hostname)"
echo " Allocated CPUs:    ${SLURM_CPUS_PER_TASK:-16}"
echo " CUDA Visible:      ${CUDA_VISIBLE_DEVICES:-}"
echo " Python:            $FER_PY"
echo " Config:            $CONFIG"
echo " Output Dir:        $EXP_DIR"
echo " Start Time:        $(date)"
echo "=========================================================================="

nvidia-smi

echo ""
echo "=== DANH SÁCH CHECKPOINTS HIỆN CÓ ==="
echo "--- [1] best (Val Accuracy):"
ls -lh "$CKPT_BEST_DIR"/*.index 2>/dev/null || echo "Không tìm thấy checkpoint trong $CKPT_BEST_DIR"

echo "--- [2] best_loss (Lowest Val Loss):"
ls -lh "$CKPT_BEST_LOSS_DIR"/*.index 2>/dev/null || echo "Không tìm thấy checkpoint trong $CKPT_BEST_LOSS_DIR"

echo "--- [3] best_macro_f1 (Highest Val Macro F1):"
ls -lh "$CKPT_BEST_F1_DIR"/*.index 2>/dev/null || echo "Không tìm thấy checkpoint trong $CKPT_BEST_F1_DIR"
echo "=========================================================================="

# ==========================================================================
# PHASE 1: ĐÁNH GIÁ ĐƠN LẺ TỪNG CHECKPOINT (No-TTA & TTA HFlip)
# ==========================================================================
echo ""
echo "=========================================================================="
echo " [PHASE 1] ĐÁNH GIÁ TỪNG CHECKPOINT ĐƠN LẺ VỚI EVALUATE.PY"
echo "=========================================================================="

evaluate_single_ckpt() {
    local ckpt_prefix="$1"
    local tag="$2"
    local ckpt_base
    ckpt_base=$(basename "$ckpt_prefix")
    
    echo "----------------------------------------------------------------------"
    echo ">>> Đang đánh giá [$tag]: $ckpt_base"
    echo "----------------------------------------------------------------------"
    
    # 1. No-TTA
    echo "  -> [1/2] Đánh giá No-TTA..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --no-tta-hflip \
        --output "$EXP_DIR/eval_${tag}_${ckpt_base}_no_tta.json"

    # 2. TTA HFlip 50/50
    echo "  -> [2/2] Đánh giá TTA (HFlip 50/50)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --tta-hflip \
        --orig-weight 0.50 \
        --flip-weight 0.50 \
        --output "$EXP_DIR/eval_${tag}_${ckpt_base}_tta_hflip.json"
}

# 1. Đánh giá tất cả checkpoint trong `best`
if [ -d "$CKPT_BEST_DIR" ]; then
    for idx_file in $(ls -1 "$CKPT_BEST_DIR"/*.index 2>/dev/null | sort -V); do
        ckpt="${idx_file%.index}"
        evaluate_single_ckpt "$ckpt" "best_acc"
    done
fi

# 2. Đánh giá tất cả checkpoint trong `best_loss`
if [ -d "$CKPT_BEST_LOSS_DIR" ]; then
    for idx_file in $(ls -1 "$CKPT_BEST_LOSS_DIR"/*.index 2>/dev/null | sort -V); do
        ckpt="${idx_file%.index}"
        evaluate_single_ckpt "$ckpt" "best_loss"
    done
fi

# 3. Đánh giá tất cả checkpoint trong `best_macro_f1`
if [ -d "$CKPT_BEST_F1_DIR" ]; then
    for idx_file in $(ls -1 "$CKPT_BEST_F1_DIR"/*.index 2>/dev/null | sort -V); do
        ckpt="${idx_file%.index}"
        evaluate_single_ckpt "$ckpt" "best_macro_f1"
    done
fi

# ==========================================================================
# PHASE 2: GRAND JOINT ENSEMBLE + MULTI-VIEW TTA SWEEP
# Kết hợp cả Best Accuracy, Best Loss, và Best Macro-F1
# ==========================================================================
echo ""
echo "=========================================================================="
echo " [PHASE 2] GRAND JOINT ENSEMBLE & MULTI-VIEW TTA EVALUATION"
echo "=========================================================================="

CHECKPOINT_DIRS=()
if [ -d "$CKPT_BEST_DIR" ] && [ "$(ls -A "$CKPT_BEST_DIR" 2>/dev/null)" ]; then
    CHECKPOINT_DIRS+=("$CKPT_BEST_DIR")
fi
if [ -d "$CKPT_BEST_LOSS_DIR" ] && [ "$(ls -A "$CKPT_BEST_LOSS_DIR" 2>/dev/null)" ]; then
    CHECKPOINT_DIRS+=("$CKPT_BEST_LOSS_DIR")
fi
if [ -d "$CKPT_BEST_F1_DIR" ] && [ "$(ls -A "$CKPT_BEST_F1_DIR" 2>/dev/null)" ]; then
    CHECKPOINT_DIRS+=("$CKPT_BEST_F1_DIR")
fi

if [ ${#CHECKPOINT_DIRS[@]} -gt 0 ]; then
    REPORT_OUTPUT="$EXP_DIR/eval_grand_joint_ensemble_report.json"
    
    echo "[INFO] Khởi chạy Grand Joint Ensemble với các thư mục:"
    for d in "${CHECKPOINT_DIRS[@]}"; do
        echo "   -> $d"
    done
    echo "[INFO] File báo cáo tổng hợp: $REPORT_OUTPUT"
    echo ""
    
    "$FER_PY" -u scripts/evaluate_ensemble_multiview_tta.py \
        --config "$CONFIG" \
        --checkpoint-dirs "${CHECKPOINT_DIRS[@]}" \
        --split test \
        --batch-size 64 \
        --sweep-step 0.05 \
        --output "$REPORT_OUTPUT"
else
    echo "[WARNING] Không tìm thấy thư mục checkpoint hợp lệ để chạy Ensemble."
fi

echo ""
echo "=========================================================================="
echo " ĐÁNH GIÁ HOÀN TẤT THÀNH CÔNG!"
echo " Báo cáo đã được lưu tại: $EXP_DIR"
echo " Thời gian kết thúc: $(date)"
echo "=========================================================================="