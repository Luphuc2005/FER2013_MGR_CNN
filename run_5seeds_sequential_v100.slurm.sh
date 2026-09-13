#!/bin/bash
#SBATCH --job-name=FER_SIGLIP2_5SEEDS_SEQ
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_5SEEDS_SEQ_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_5SEEDS_SEQ_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
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

nvidia-smi

# 5 User-Requested Seeds: 0, 1, 43, 123, 3047
SEEDS=(0 1 43 123 3047)

for SEED in "${SEEDS[@]}"; do
    CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed${SEED}.yaml"
    OUTPUT_DIR="outputs/papers/siglip2-confusion-seed${SEED}"
    
    echo "============================================================"
    echo " STARTING PIPELINE FOR SEED: $SEED"
    echo " Config: $CONFIG"
    echo " Output: $OUTPUT_DIR"
    echo " Start:  $(date)"
    echo "============================================================"

    # 1. Train Model
    "$FER_PY" -u train.py --config "$CONFIG"

    # 2. Comprehensive 15-Checkpoint TTA Sweep & Ensemble
    echo "============================================================"
    echo " Running TTA Sweep & Ensemble on 15 Checkpoints for Seed $SEED..."
    echo "============================================================"
    "$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$CONFIG" || true

    echo "Finished SEED $SEED at $(date)"
done

echo "============================================================"
echo " All 5 Seeds (0, 1, 43, 123, 3047) Finished Successfully!"
echo " End: $(date)"
echo "============================================================"
