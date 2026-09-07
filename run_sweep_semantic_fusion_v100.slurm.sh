#!/bin/bash
#SBATCH --job-name=RAFDB_SWEEP_FUSION
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SWEEP_FUSION_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SWEEP_FUSION_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="${1:-$ROOT/config_rafdb_siglip2_semantic_stable_v3.yaml}"
CHECKPOINT="${2:-$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v3/checkpoints/best/ckpt-52}"
SPLIT="${3:-test}"

export NVIDIA_LIB=$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " Sweeping SigLIP2 Semantic Fusion Alpha on RAF-DB (Slurm)"
echo " Config:     $CONFIG"
echo " Checkpoint: $CHECKPOINT"
echo " Split:      $SPLIT"
echo " Start:      $(date)"
echo "============================================================"

nvidia-smi

"$FER_PY" scripts/sweep_rafdb_semantic_fusion.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --alphas 0.0 0.02 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30

echo "============================================================"
echo " Completed successfully at $(date)"
echo "============================================================"
