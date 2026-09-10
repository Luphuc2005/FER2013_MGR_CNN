#!/bin/bash
#SBATCH --job-name=EVAL_V5_MOD_2V
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_MOD_2V_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_MOD_2V_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_moderate_aug.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_moderate_aug"
CKPT_BEST_ACC="$OUTPUT_DIR/checkpoints/best"
CKPT_BEST_LOSS="$OUTPUT_DIR/checkpoints/best_loss"
REPORT_OUTPUT="$OUTPUT_DIR/eval_ensemble_2view_best_acc_and_loss_report.json"

export NVIDIA_LIB=$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ROOT/logs"

echo "=========================================================================="
echo " Running FAST 2-View TTA Ensemble for v5 Ultimate Moderate Aug on GPU (32GB VRAM)"
echo " Config:         $CONFIG"
echo " Best Acc Dir:   $CKPT_BEST_ACC"
echo " Best Loss Dir:  $CKPT_BEST_LOSS"
echo " Target Split:   TEST"
echo " Batch Size:     128 (Siêu nhanh với 32GB VRAM)"
echo " Mode:           2-View Only (Origin + Flip + TTA Weight Ratio Sweep)"
echo " Start Time:     $(date)"
echo " Job ID:         ${SLURM_JOB_ID:-standalone}"
echo " Node:           $(hostname)"
echo "=========================================================================="

nvidia-smi

if [ ! -d "$CKPT_BEST_ACC" ]; then
    echo "[ERROR] Checkpoint best dir not found: $CKPT_BEST_ACC"
    exit 1
fi

CHECKPOINT_DIRS=("$CKPT_BEST_ACC")
if [ -d "$CKPT_BEST_LOSS" ]; then
    CHECKPOINT_DIRS+=("$CKPT_BEST_LOSS")
    echo "[INFO] Found best_loss directory. Running Joint Ensemble (Best Acc + Best Loss)!"
else
    echo "[WARNING] best_loss directory not found ($CKPT_BEST_LOSS). Running only Best Acc checkpoints."
fi

"$FER_PY" -u scripts/evaluate_ensemble_multiview_tta.py \
  --config "$CONFIG" \
  --checkpoint-dirs "${CHECKPOINT_DIRS[@]}" \
  --two-view-only \
  --split test \
  --batch-size 128 \
  --sweep-step 0.05 \
  --output "$REPORT_OUTPUT"

echo "=========================================================================="
echo " 2-View Evaluation completed successfully at $(date)"
echo " Report saved to: $REPORT_OUTPUT"
echo "=========================================================================="
