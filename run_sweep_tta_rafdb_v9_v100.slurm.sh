#!/bin/bash
#SBATCH --job-name=SWEEP_TTA_V9
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_TTA_V9_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_TTA_V9_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="${1:-$ROOT/config_rafdb_v9_single_sota_93.yaml}"
CHECKPOINT="${2:-}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

[[ -x "$FER_PY" ]] || { echo "Missing Python: $FER_PY"; exit 1; }

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi
echo "Config=$CONFIG Checkpoint=${CHECKPOINT:-auto_rank1}"
echo "Running Validation-Tuned TTA Weight Sweep (Step: 0.05)..."
echo "Strategy: Find optimal Original/Flip weight on Validation set, then evaluate on Test set."

if [[ -n "$CHECKPOINT" ]]; then
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --step 0.05
else
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05
fi

echo "SWEEP_TTA_COMPLETE"