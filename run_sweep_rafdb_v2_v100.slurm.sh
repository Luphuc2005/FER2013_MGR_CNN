#!/bin/bash
#SBATCH --job-name=RAFDB_V2_SWEEP
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V2_SWEEP_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V2_SWEEP_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion_v2.yaml"
CHECKPOINT="$ROOT/outputs/papers/rafdb_adaptive_siglip2_confusion_v2_v2/checkpoints/best/ckpt-60"

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " Running TTA Weight Sweep for RAF-DB v2 (ckpt-60)"
echo " Config: $CONFIG"
echo " Checkpoint: $CHECKPOINT"
echo "============================================================"

nvidia-smi

"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --step 0.05
