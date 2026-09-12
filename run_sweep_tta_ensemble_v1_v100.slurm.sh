#!/bin/bash
#SBATCH --job-name=SWEEP_TTA_ENS_V1
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_TTA_ENS_V1_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_TTA_ENS_V1_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
mkdir -p "$ROOT/logs"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v1_clean_evolution.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_v1_clean_evolution"

# Checkpoint dir to evaluate (defaults to sweeping both best/ and best_loss/)
CHECKPOINT_DIR="${1:-}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " VALIDATION-TUNED TTA SWEEP & MULTI-CHECKPOINT ENSEMBLE (RAF-DB V1)"
echo " Config         : $CONFIG"
echo " Checkpoint Dir : ${CHECKPOINT_DIR:-'Auto: best/ (Val Acc) + best_loss/ (Val Loss)'}"
echo " Job ID         : ${SLURM_JOB_ID:-standalone}"
echo " Start Time     : $(date)"
echo "=========================================================================="

nvidia-smi

# Run validation-tuned TTA sweep on every checkpoint + compute Test ensemble
if [ -n "$CHECKPOINT_DIR" ]; then
    echo ">>> Evaluating custom checkpoint directory: $CHECKPOINT_DIR"
    "$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
        --config "$CONFIG" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --step 0.05 \
        --output "$OUTPUT_DIR/val_tuned_tta_ensemble_report.json"
else
    echo ">>> Evaluating ALL checkpoints in best/ (Acc) AND best_loss/ (Loss)..."
    "$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
        --config "$CONFIG" \
        --include-best-loss \
        --step 0.05 \
        --output "$OUTPUT_DIR/val_tuned_tta_ensemble_report.json"
fi

echo ""
echo "=========================================================================="
echo " [FINISHED] TTA Sweep & Ensemble completed successfully!"
echo " Results saved in: $OUTPUT_DIR/val_tuned_tta_ensemble_report.json"
echo " End Time: $(date)"
echo "=========================================================================="
