#!/bin/bash
#SBATCH --job-name=RAFDB_V1_EVOL
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V1_EVOL_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V1_EVOL_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v1_clean_evolution.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_v1_clean_evolution"

mkdir -p "$ROOT/logs" "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

if env | grep -q '^MGR_'; then
    echo '[WARNING] Unsetting MGR_* environment overrides for clean run...'
    unset $(env | grep -o '^MGR_[^=]*')
fi

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " RAF-DB V1 Clean Evolution (Single Model, Standard Split)"
echo " Fix 1: visual_extractor_lr=1e-5 (Eliminates semantic drift)"
echo " Fix 2: Logit Soft Fusion alpha=0.15 (ConvNeXt 85% + SigLIP2 15%)"
echo " Fix 3: Smooth LogSumExp Prototype Aggregation (Eliminates Gate Collapse)"
echo " Config     : $CONFIG"
echo " Output Dir : $OUTPUT_DIR"
echo " Job ID     : ${SLURM_JOB_ID:-standalone}"
echo "=========================================================================="

nvidia-smi

# Backup previous run if nonempty
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    BACKUP_DIR="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] Existing output found. Moving to: $BACKUP_DIR"
    mv "$OUTPUT_DIR" "$BACKUP_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Run pre-flight smoke test
echo ""
echo ">>> Running Pre-flight RAF-DB Smoke Test..."
"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"

# Run Full Training
echo ""
echo ">>> Starting RAF-DB V1 Clean Evolution Training (60 epochs max, early-stopping patience=20)..."
"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment

# Run Automated TTA Weight Sweep on Best Checkpoint
echo ""
echo ">>> Running Post-training TTA Weight Sweep (tuning on Val -> applying to Test)..."
BEST_CKPT="$OUTPUT_DIR/checkpoints/best/ckpt-$(cat "$OUTPUT_DIR/training_summary.json" 2>/dev/null | grep -o '"best_epoch": [0-9]*' | grep -o '[0-9]*' || echo "")"
if [[ -f "${BEST_CKPT}.index" ]]; then
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_CKPT" --step 0.05
else
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05
fi

echo ""
echo "=========================================================================="
echo " [SUCCESS] RAF-DB V1 Clean Evolution Completed Successfully!"
echo " Results saved in: $OUTPUT_DIR"
echo "=========================================================================="
