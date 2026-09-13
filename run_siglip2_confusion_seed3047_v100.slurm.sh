#!/bin/bash
#SBATCH --job-name=FER_SIGLIP2_SEED3047
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_SEED3047_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_SEED3047_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/siglip2-confusion-seed3047

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed3047.yaml"

echo "============================================================"
echo " FER2013 ConvNeXt-Base MS1M SigLIP 2 Confusion (Seed 3047)"
echo " CPU: 24 cores | 15 Checkpoints | Full TTA + Ensemble"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_CUDNN_USE_AUTOTUNE=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

# 1. Train Model
"$FER_PY" -u train.py --config "$CONFIG"

# 2. Comprehensive 15-Checkpoint TTA Sweep & Ensemble
echo "============================================================"
echo " Running TTA Sweep & Ensemble on all 15 Checkpoints..."
echo "============================================================"
"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$CONFIG" || true

echo "============================================================"
echo " Completed Seed 3047 Pipeline"
echo " End: $(date)"
echo "============================================================"
