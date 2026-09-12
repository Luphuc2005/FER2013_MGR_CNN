#!/bin/bash
#SBATCH --job-name=SOUP_RAFDB_V5
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SOUP_RAFDB_V5_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SOUP_RAFDB_V5_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " STEP 1: EVALUATE INDIVIDUAL CHECKPOINTS (ckpt-40, ckpt-50, ckpt-60)"
echo "=========================================================================="
for ckpt in ckpt-40 ckpt-50 ckpt-60; do
    echo ""
    echo ">>> Evaluating $ckpt..."
    "$FER_PY" -u evaluate.py --config "$CONFIG" --checkpoint "$EXP_DIR/checkpoints/periodic/$ckpt" --split test --tta-hflip
done

echo ""
echo "=========================================================================="
echo " STEP 2: CREATE & EVALUATE MODEL SOUP 1 (ckpt-50 + ckpt-60)"
echo "=========================================================================="
"$FER_PY" -u scripts/create_model_soup.py \
    --config "$CONFIG" \
    --checkpoints ckpt-50 ckpt-60 \
    --evaluate

echo ""
echo "=========================================================================="
echo " STEP 3: CREATE & EVALUATE MODEL SOUP 2 (ckpt-40 + ckpt-50 + ckpt-60)"
echo "=========================================================================="
"$FER_PY" -u scripts/create_model_soup.py \
    --config "$CONFIG" \
    --checkpoints ckpt-40 ckpt-50 ckpt-60 \
    --evaluate

echo ""
echo "=========================================================================="
echo " [FINISHED] All Full-Train checkpoint evaluations and soups completed!"
echo "=========================================================================="
