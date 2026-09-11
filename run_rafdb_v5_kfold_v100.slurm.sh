#!/bin/bash
#SBATCH --job-name=RAFDB_V5_KFOLD
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_KFOLD_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_KFOLD_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs

FER_PY="$ROOT/fer2013_env/bin/python"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

if env | grep -q '^MGR_'; then
    echo 'Unexpected MGR_* overrides. Unset them before submitting this experiment.'
    exit 1
fi

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi
echo "Starting 1-Click RAF-DB V5 5-Fold Cross Validation & Ensemble Pipeline..."
echo "Job ID: ${SLURM_JOB_ID:-standalone}"

"$FER_PY" -u run_kfold_rafdb_v5.py

echo "================================================================="
echo "  RAF-DB V5 5-FOLD SOTA PIPELINE FINISHED SUCCESSFULLY!"
echo "================================================================="
