#!/bin/bash
#SBATCH --job-name=EVAL_SINGLE
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_SINGLE_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_SINGLE_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_sota"

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " Evaluating Saved Checkpoints from Job 6681 (Single Model)"
echo "============================================================"

# Ưu tiên kiểm tra ckpt-7 nếu có
CKPT7="$EXP_DIR/checkpoints/best_loss/ckpt-7"
if [ -f "${CKPT7}.index" ]; then
    echo ""
    echo ">>> [1/2] Evaluating ckpt-7 (Lowest Validation Loss) ..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT7" \
        --split test \
        --tta-hflip
fi

# Quét và đánh giá toàn bộ các checkpoint còn lại trong best_loss
echo ""
echo ">>> [2/2] Scanning all checkpoints in best_loss ..."
for ckpt in $(find "$EXP_DIR/checkpoints/best_loss" -name "*.index" | sort); do
    prefix="${ckpt%.index}"
    if [ "$prefix" != "$CKPT7" ]; then
        echo "------------------------------------------------------------"
        echo " Evaluating: $prefix"
        echo "------------------------------------------------------------"
        "$FER_PY" -u evaluate.py \
            --config "$CONFIG" \
            --checkpoint "$prefix" \
            --split test \
            --tta-hflip || true
    fi
done

echo ""
echo "Finished all evaluations."
