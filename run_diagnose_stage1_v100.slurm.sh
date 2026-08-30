#!/bin/bash
#SBATCH --job-name=FER_DIAGNOSE_STAGE1
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_DIAGNOSE_STAGE1_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_DIAGNOSE_STAGE1_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_stage1_rgb_smirk_3d_cnn_late_fusion.yaml"
BASELINE_CKPT="/home/ptbao/projects/FER2013_MGR_CNN/outputs/tf_runs/convnext_base_ms1m_arcface_baseline/checkpoints/best/ckpt-43"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " STAGE 1 SANITY CHECKS & DIAGNOSTIC SUITE"
echo "============================================================"

"$FER_PY" -u scripts/diagnose_stage1_late_fusion.py \
    --config "$CONFIG" \
    --baseline-checkpoint "$BASELINE_CKPT"

echo "============================================================"
echo " DIAGNOSTICS COMPLETED"
echo "============================================================"
