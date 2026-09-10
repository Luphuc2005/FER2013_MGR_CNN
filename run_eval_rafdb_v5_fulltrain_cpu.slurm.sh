#!/bin/bash
#SBATCH --job-name=EVAL_RAFDB_V5_CPU
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_V5_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_V5_CPU_%j.err

# NOTE: Pure CPU evaluation on 'compute' partition (0 GPU, no QOSMaxGRES conflict with GPU jobs).

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain"

echo "=========================================================================="
echo " Evaluating RAF-DB v5 Full-Train on CPU (Test Set: 3,068 samples)"
echo " Config  : $CONFIG"
echo " Exp Dir : $EXP_DIR"
echo " Cores   : $NCPUS threads"
echo " Mode    : PURE CPU (0 MB VRAM, CUDA_VISIBLE_DEVICES=-1)"
echo " Job ID  : ${SLURM_JOB_ID:-standalone}"
echo "=========================================================================="

evaluate_checkpoint() {
    local ckpt_prefix="$1"
    echo ""
    echo "=========================================================================="
    echo ">>> EVALUATING CHECKPOINT (CPU): $ckpt_prefix"
    echo "=========================================================================="

    # 1. No-TTA Evaluation
    echo ""
    echo "[Mode 1/2] CPU Standard Evaluation (No TTA) on Test Set (3,068 samples)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --cpu \
        --no-tta-hflip

    # 2. TTA H-Flip Evaluation
    echo ""
    echo "[Mode 2/2] CPU Test-Time Augmentation (TTA H-Flip 50/50) on Test Set (3,068 samples)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --cpu \
        --tta-hflip
}

# Target checkpoint from CLI argument or auto-scan
TARGET_CKPT="${1:-}"

if [[ -n "$TARGET_CKPT" ]]; then
    if [[ "$TARGET_CKPT" == *.index ]]; then
        TARGET_CKPT="${TARGET_CKPT%.index}"
    fi
    if [[ ! -f "${TARGET_CKPT}.index" ]]; then
        if [[ -f "$EXP_DIR/checkpoints/periodic/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/periodic/${TARGET_CKPT}"
        elif [[ -f "$EXP_DIR/checkpoints/last/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/last/${TARGET_CKPT}"
        elif [[ -f "$ROOT/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$ROOT/${TARGET_CKPT}"
        else
            echo "[ERROR] Checkpoint not found: ${TARGET_CKPT}.index"
            exit 1
        fi
    fi
    evaluate_checkpoint "$TARGET_CKPT"
else
    echo ""
    echo "[INFO] Searching for available checkpoints in:"
    echo "  - $EXP_DIR/checkpoints/last"
    echo "  - $EXP_DIR/checkpoints/periodic"

    FOUND=0

    # 1. Check last checkpoint (e.g. ckpt-60)
    if [[ -d "$EXP_DIR/checkpoints/last" ]]; then
        for idx in $(find "$EXP_DIR/checkpoints/last" -name "*.index" | sort); do
            prefix="${idx%.index}"
            evaluate_checkpoint "$prefix"
            FOUND=$((FOUND + 1))
        done
    fi

    # 2. Check periodic checkpoints (ckpt-10, ckpt-20, ...)
    if [[ -d "$EXP_DIR/checkpoints/periodic" ]]; then
        for idx in $(find "$EXP_DIR/checkpoints/periodic" -name "*.index" | sort -V); do
            prefix="${idx%.index}"
            evaluate_checkpoint "$prefix"
            FOUND=$((FOUND + 1))
        done
    fi

    if [[ "$FOUND" -eq 0 ]]; then
        echo "[WARNING] No checkpoints found yet in $EXP_DIR/checkpoints!"
        echo "Make sure training has reached at least epoch 10 or finished epoch 60."
        exit 1
    fi
fi

echo ""
echo "=========================================================================="
echo " [FINISHED] All CPU evaluations completed successfully!"
echo " Results are saved in: $EXP_DIR"
echo "=========================================================================="