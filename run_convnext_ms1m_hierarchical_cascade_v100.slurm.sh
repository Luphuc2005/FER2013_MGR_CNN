#!/bin/bash
#SBATCH --job-name=convnext_ms1m_hierarchical_cascade
#SBATCH --output=logs/convnext_ms1m_hierarchical_cascade_%j.out
#SBATCH --error=logs/convnext_ms1m_hierarchical_cascade_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpunode
#SBATCH --time=24:00:00

set -euo pipefail

echo "=========================================="
echo " JOB INFO"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NotSet}"

PROJECT_DIR="/home/ptbao/projects/FER2013_MGR_CNN"
cd "${PROJECT_DIR}"

source "${PROJECT_DIR}/fer2013_env/bin/activate"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS="--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
export TF_DISABLE_XLA=1
export TF_DISABLE_XLA_COMPILATION=1
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

python scripts/train_convnext_ms1m_hierarchical_cascade_fusion.py --config config_convnext_base_ms1m_hierarchical_cascade_fusion.yaml
