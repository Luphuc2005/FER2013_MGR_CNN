#!/bin/bash
#SBATCH --job-name=RAFDB_V11_SOTA93
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V11_SOTA93_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V11_SOTA93_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v11_champ_sota_93.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_v11_champ_sota_93"

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

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Output already nonempty: $OUTPUT_DIR. Refusing overwrite/resume."
    exit 1
fi

nvidia-smi
echo "Config=$CONFIG Output=$OUTPUT_DIR Job=${SLURM_JOB_ID:-standalone}"

echo "================================================================="
echo "  STEP 1: Verify RAF-DB V11 Contract & Hyperparameters"
echo "================================================================="
"$FER_PY" -u scripts/check_rafdb_v11_contract.py --config "$CONFIG"

echo "================================================================="
echo "  STEP 2: Train RAF-DB V11 Champion SOTA 93% Model"
echo "================================================================="
"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment

echo "================================================================="
echo "  STEP 3: Run Validation-Tuned TTA Weight Sweep (Step 0.05)"
echo "================================================================="
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05

echo "================================================================="
echo "  RAF-DB V11 CHAMPION SOTA PIPELINE FINISHED SUCCESSFULLY!"
echo "================================================================="
