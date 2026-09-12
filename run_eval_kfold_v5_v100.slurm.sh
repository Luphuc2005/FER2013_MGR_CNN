#!/bin/bash
#SBATCH --job-name=EVAL_KFOLD_V5
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_KFOLD_V5_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_KFOLD_V5_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p logs

FER_PY="$ROOT/fer2013_env/bin/python"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi
echo "================================================================="
echo "  RUNNING 5-FOLD ENSEMBLE EVALUATION (RAF-DB V5)"
echo "  Job ID: ${SLURM_JOB_ID:-standalone}"
echo "  Start Time: $(date)"
echo "================================================================="

"$FER_PY" -u evaluate_kfold_ensemble.py \
    --base-config config_rafdb_v5_kfold_base.yaml \
    --kfold-dir data/rafdb/kfold_5 \
    --output-dir outputs/papers/rafdb_v5_kfold \
    --n-splits 5 \
    --step 0.05

echo ""
echo "================================================================="
echo "  5-FOLD ENSEMBLE EVALUATION FINISHED SUCCESSFULLY!"
echo "  End Time: $(date)"
echo "================================================================="
