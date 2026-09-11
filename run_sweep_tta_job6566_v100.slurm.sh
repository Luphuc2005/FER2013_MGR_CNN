#!/bin/bash
#SBATCH --job-name=SWEEP_TTA_V5_CHAMP
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_TTA_V5_CHAMP_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_TTA_V5_CHAMP_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml"
CHECKPOINT="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate/checkpoints/best/ckpt-49"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

[[ -x "$FER_PY" ]] || { echo "Missing Python: $FER_PY"; exit 1; }

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi
echo "Config=$CONFIG"
echo "Checkpoint=$CHECKPOINT"
echo "Running Validation-Tuned TTA Weight Sweep on Champion V5 (Job 6566)..."
echo "Evaluates w_orig in [0.00, 1.00] with step 0.05 on Validation set first, then applies to Test set."

"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --step 0.05

echo "================================================================="
echo "  SWEEP TTA JOB 6566 COMPLETE!"
echo "================================================================="
