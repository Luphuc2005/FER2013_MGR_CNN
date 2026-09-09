#!/bin/bash
#SBATCH --job-name=EVAL_V5_2VIEW_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_2VIEW_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_V5_2VIEW_CPU_%j.err

# GHI CHÚ: Nếu cluster yêu cầu bắt buộc phải có --gres=gpu mới submit được vào gpu-queue,
# hãy bỏ comment dòng dưới. Script bên dưới vẫn ép CUDA_VISIBLE_DEVICES="-1" để chạy 100% CPU.
# #SBATCH --gres=gpu:v100:1

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=2

FER_PY="$ROOT/fer2013_env/bin/python"
if [ ! -f "$FER_PY" ]; then
  FER_PY="python3"
fi

CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate"
CKPT_BEST_ACC="$OUTPUT_DIR/checkpoints/best"
CKPT_BEST_LOSS="$OUTPUT_DIR/checkpoints/best_loss"

echo "=========================================================================="
echo " Running Fast 2-View Ensemble + TTA Sweep on CPU"
echo " Model / Architecture: v5 Combined Ultimate"
echo " Config:               $CONFIG"
echo " Best Acc Dir:         $CKPT_BEST_ACC"
echo " Best Loss Dir:        $CKPT_BEST_LOSS"
echo " Target Split:         TEST"
echo " CPUs Allocated:       $NCPUS"
echo " Start Time:           $(date)"
echo "=========================================================================="

"$FER_PY" -u scripts/evaluate_ensemble_multiview_tta.py \
  --config "$CONFIG" \
  --checkpoint-dirs "$CKPT_BEST_ACC" "$CKPT_BEST_LOSS" \
  --split test \
  --batch-size 32 \
  --two-view-only \
  --sweep-ratio \
  --sweep-step 0.05 \
  --cpu \
  --output "$OUTPUT_DIR/eval_ensemble_2view_best_acc_and_loss_cpu_report.json"

echo "=========================================================================="
echo " CPU Fast 2-View Evaluation completed successfully at $(date)"
echo " Report: $OUTPUT_DIR/eval_ensemble_2view_best_acc_and_loss_cpu_report.json"
echo " Log:    logs/EVAL_V5_2VIEW_CPU_${SLURM_JOB_ID:-standalone}.out"
echo "=========================================================================="
