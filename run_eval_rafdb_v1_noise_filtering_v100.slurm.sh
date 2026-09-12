#!/bin/bash
#SBATCH --job-name=EVAL_V1_NOISE
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V1_NOISE_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V1_NOISE_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v1_noise_filtering.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_v1_noise_filtering"

mkdir -p "$ROOT/logs"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

if env | grep -q '^MGR_'; then
    echo '[WARNING] Unsetting MGR_* environment overrides...'
    unset $(env | grep -o '^MGR_[^=]*')
fi

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " RAF-DB V1 Noise Filtering: Evaluation on Best & Best-Loss Checkpoints"
echo " Config     : $CONFIG"
echo " Checkpoints: $EXP_DIR/checkpoints"
echo " Job ID     : ${SLURM_JOB_ID:-standalone}"
echo " Start Time : $(date)"
echo "=========================================================================="

nvidia-smi

# --------------------------------------------------------------------------
# 1. EVALUATE ALL CHECKPOINTS IN best/ (Ranked by Validation Accuracy)
# --------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo " [PART 1/3] Evaluating Checkpoints in 'best/' (Top Val Accuracy)"
echo "=========================================================================="
if [ -d "$EXP_DIR/checkpoints/best" ]; then
    for idx_file in $(find "$EXP_DIR/checkpoints/best" -name "*.index" | sort -V); do
        ckpt="${idx_file%.index}"
        echo ""
        echo ">>> Evaluating Best Acc Checkpoint: $(basename "$ckpt")"
        echo "    [No-TTA Evaluation]"
        "$FER_PY" -u evaluate.py \
            --config "$CONFIG" \
            --checkpoint "$ckpt" \
            --split test \
            --no-tta-hflip || true

        echo "    [TTA H-Flip (50/50) Evaluation]"
        "$FER_PY" -u evaluate.py \
            --config "$CONFIG" \
            --checkpoint "$ckpt" \
            --split test \
            --tta-hflip || true
    done
else
    echo "[WARN] No $EXP_DIR/checkpoints/best directory found!"
fi

# --------------------------------------------------------------------------
# 2. EVALUATE ALL CHECKPOINTS IN best_loss/ (Ranked by Validation Loss)
# --------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo " [PART 2/3] Evaluating Checkpoints in 'best_loss/' (Top Val Loss)"
echo "=========================================================================="
if [ -d "$EXP_DIR/checkpoints/best_loss" ]; then
    for idx_file in $(find "$EXP_DIR/checkpoints/best_loss" -name "*.index" | sort -V); do
        ckpt="${idx_file%.index}"
        echo ""
        echo ">>> Evaluating Best Loss Checkpoint: $(basename "$ckpt")"
        echo "    [No-TTA Evaluation]"
        "$FER_PY" -u evaluate.py \
            --config "$CONFIG" \
            --checkpoint "$ckpt" \
            --split test \
            --no-tta-hflip || true

        echo "    [TTA H-Flip (50/50) Evaluation]"
        "$FER_PY" -u evaluate.py \
            --config "$CONFIG" \
            --checkpoint "$ckpt" \
            --split test \
            --tta-hflip || true
    done
else
    echo "[WARN] No $EXP_DIR/checkpoints/best_loss directory found!"
fi

# --------------------------------------------------------------------------
# 3. COMPREHENSIVE TTA SWEEP & COMBINATORIAL ENSEMBLE (50,000+ combinations)
# --------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo " [PART 3/3] Running Validation-Tuned TTA Sweep & Checkpoint Ensemble Sweep"
echo " (Extracts logits once per checkpoint, sweeps optimal TTA & Combinations)"
echo "=========================================================================="
"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
    --config "$CONFIG" \
    --checkpoint-dir "$EXP_DIR/checkpoints/best" \
    --include-best-loss \
    --step 0.05 \
    --num-comb-samples 50000 \
    --output "$EXP_DIR/evaluation_best_and_best_loss_summary.json"

echo ""
echo "=========================================================================="
echo " [SUCCESS] Evaluation completed at $(date)"
echo " Detailed JSON report saved at: $EXP_DIR/evaluation_best_and_best_loss_summary.json"
echo "=========================================================================="
