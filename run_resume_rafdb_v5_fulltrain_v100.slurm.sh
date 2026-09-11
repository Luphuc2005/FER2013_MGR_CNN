#!/bin/bash
#SBATCH --job-name=RAFDB_V5_RESUME
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_RESUME_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_RESUME_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_fulltrain"
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

echo "=========================================================================="
echo " RESUMING RAF-DB v5 Combined Ultimate Full-Train"
echo " Config     : $CONFIG"
echo " Output Dir : $OUTPUT_DIR"
echo " Job ID     : ${SLURM_JOB_ID:-standalone}"
echo "=========================================================================="

# Check checkpoint directory
if [[ ! -d "$OUTPUT_DIR/checkpoints/last" ]]; then
    echo "[ERROR] Checkpoint directory not found at $OUTPUT_DIR/checkpoints/last"
    echo "Cannot resume. Please verify if the output directory exists on disk."
    exit 1
fi

echo "[INFO] Found existing checkpoint(s) in last/:"
ls -la "$OUTPUT_DIR/checkpoints/last"

nvidia-smi

# Resume directly from last checkpoint (e.g. ckpt-24) continuing through epoch 60
"$FER_PY" -u train.py --config "$CONFIG" --resume --no-auto-increment

echo "Finished resume training: $OUTPUT_DIR (v5 Combined Ultimate Final Full-Train)."
