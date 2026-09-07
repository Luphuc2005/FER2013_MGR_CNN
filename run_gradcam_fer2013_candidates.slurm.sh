#!/bin/bash
#SBATCH --job-name=FER_GRADCAM_CANDIDATES
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_GRADCAM_CANDIDATES_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_GRADCAM_CANDIDATES_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/gradcam_fer2013_candidates

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

if [ ! -x "$FER_PY" ]; then
    echo "[WARNING] $FER_PY not found, falling back to python"
    FER_PY="python"
fi

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
if [ -d "$NVIDIA_LIB" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"
fi

echo "============================================================"
echo " Starting Grad-CAM Candidate Generation (SLURM Job: ${SLURM_JOB_ID:-standalone})"
echo " Node: $(hostname)"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo " Time: $(date)"
echo "============================================================"

nvidia-smi || true

"$FER_PY" -u generate_gradcam_fer2013_candidates.py "$@"

echo "============================================================"
echo " Grad-CAM Candidate Generation Finished: $(date)"
echo "============================================================"
