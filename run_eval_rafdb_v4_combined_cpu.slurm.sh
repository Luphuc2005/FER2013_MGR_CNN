#!/bin/bash
#SBATCH --job-name=EVAL_V4_COMB_CPU
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V4_COMB_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V4_COMB_CPU_%j.err

# NOTE: High-performance CPU evaluation (32 cores, 64GB RAM, 0 MB GPU VRAM used).

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v4_combined.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v4_combined"

echo "=========================================================================="
echo " Evaluating RAF-DB v4 Combined on High-Performance CPU (Test: 3,068 samples)"
echo " Config  : $CONFIG"
echo " Exp Dir : $EXP_DIR"
echo " Cores   : $NCPUS threads | RAM: 64GB"
echo " Mode    : PURE CPU (0 MB GPU VRAM, CUDA_VISIBLE_DEVICES=-1)"
echo " Job ID  : ${SLURM_JOB_ID:-standalone}"
echo "=========================================================================="

evaluate_checkpoint() {
    local ckpt_prefix="$1"
    echo ""
    echo "=========================================================================="
    echo ">>> EVALUATING CHECKPOINT (CPU 32 CORES): $ckpt_prefix"
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

# Resolve target checkpoint: argument or best/ckpt-44 or auto-scan
TARGET_CKPT="${1:-}"

if [[ -z "$TARGET_CKPT" ]]; then
    # Default to ckpt-44 (best epoch from Job 6537) if exists
    if [[ -f "$EXP_DIR/checkpoints/best/ckpt-44.index" ]]; then
        TARGET_CKPT="$EXP_DIR/checkpoints/best/ckpt-44"
    fi
fi

if [[ -n "$TARGET_CKPT" ]]; then
    if [[ "$TARGET_CKPT" == *.index ]]; then
        TARGET_CKPT="${TARGET_CKPT%.index}"
    fi
    if [[ ! -f "${TARGET_CKPT}.index" ]]; then
        if [[ -f "$EXP_DIR/checkpoints/best/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/best/${TARGET_CKPT}"
        elif [[ -f "$EXP_DIR/checkpoints/last/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/last/${TARGET_CKPT}"
        elif [[ -f "$EXP_DIR/checkpoints/periodic/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/periodic/${TARGET_CKPT}"
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
    echo "[INFO] Searching for best checkpoints in: $EXP_DIR/checkpoints"
    FOUND=0
    if [[ -d "$EXP_DIR/checkpoints/best" ]]; then
        for idx in $(find "$EXP_DIR/checkpoints/best" -name "*.index" | sort); do
            prefix="${idx%.index}"
            evaluate_checkpoint "$prefix"
            FOUND=$((FOUND + 1))
        done
    fi
    if [[ "$FOUND" -eq 0 && -d "$EXP_DIR/checkpoints/last" ]]; then
        for idx in $(find "$EXP_DIR/checkpoints/last" -name "*.index" | sort); do
            prefix="${idx%.index}"
            evaluate_checkpoint "$prefix"
            FOUND=$((FOUND + 1))
        done
    fi
    if [[ "$FOUND" -eq 0 ]]; then
        echo "[ERROR] No checkpoints found in $EXP_DIR/checkpoints!"
        exit 1
    fi
fi

echo ""
echo "=========================================================================="
echo " [FINISHED] All CPU evaluations completed successfully!"
echo " Results are saved in: $EXP_DIR"
echo "=========================================================================="