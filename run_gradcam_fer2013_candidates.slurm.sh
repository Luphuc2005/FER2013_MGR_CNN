#!/bin/bash
#SBATCH --job-name=FER_GRADCAM_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_GRADCAM_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_GRADCAM_CPU_%j.err

# GHI CHÚ: Nếu cluster yêu cầu bắt buộc phải có --gres=gpu mới submit được vào gpu-queue,
# hãy bỏ comment dòng dưới. Script bên dưới vẫn ép CUDA_VISIBLE_DEVICES="-1" để chạy 100% CPU.
# #SBATCH --gres=gpu:v100:1

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/gradcam_fer2013_candidates

# Ép chặt chạy CPU thuần túy, tuyệt đối không cấp phát VRAM hay ảnh hưởng job training GPU
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

if [ ! -x "$FER_PY" ]; then
    echo "[WARNING] $FER_PY not found, falling back to python"
    FER_PY="python"
fi

echo "============================================================"
echo " Starting Grad-CAM Candidate Generation on FER2013 (100% CPU)"
echo " Job ID              : ${SLURM_JOB_ID:-standalone}"
echo " Node                : $(hostname)"
echo " Allocated CPUs      : ${SLURM_CPUS_PER_TASK:-8}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo " Start Time          : $(date)"
echo "============================================================"

"$FER_PY" -u generate_gradcam_fer2013_candidates.py --cpu "$@"

echo "============================================================"
echo " Grad-CAM Candidate Generation Finished: $(date)"
echo "============================================================"
