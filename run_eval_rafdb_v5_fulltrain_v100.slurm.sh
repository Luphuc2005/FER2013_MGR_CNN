#!/bin/bash
#SBATCH --job-name=EVAL_RAFDB_V5_FULL
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_V5_FULL_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_V5_FULL_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain"

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " Evaluating RAF-DB v5 Combined Ultimate Full-Train Checkpoints (Test Set)"
echo " Config  : $CONFIG"
echo " Exp Dir : $EXP_DIR"
echo " Job ID  : ${SLURM_JOB_ID:-standalone}"
echo "=========================================================================="
nvidia-smi

evaluate_checkpoint() {
    local ckpt_prefix="$1"
    echo ""
    echo "=========================================================================="
    echo ">>> EVALUATING CHECKPOINT: $ckpt_prefix"
    echo "=========================================================================="

    # 1. No-TTA Evaluation
    echo ""
    echo "[Mode 1/2] Standard Evaluation (No TTA) on Test Set (3,068 samples)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --no-tta-hflip

    # 2. TTA H-Flip Evaluation
    echo ""
    echo "[Mode 2/2] Test-Time Augmentation (TTA H-Flip 50/50) on Test Set (3,068 samples)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split test \
        --tta-hflip
}

# If a specific checkpoint is passed as argument, evaluate only that one
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
    # Auto-discover all available checkpoints in last/ and periodic/
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
echo " [FINISHED] All requested evaluations completed successfully!"
echo " Results are saved in: $EXP_DIR"
echo "=========================================================================="