#!/bin/bash
#SBATCH --job-name=EVAL_JOB6078_SIGLIP2
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_JOB6078_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_JOB6078_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/siglip2-confusion"

echo "============================================================"
echo " EVALUATION PIPELINE FOR JOB 6078 (NO RETRAINING)"
echo " TTA Sweep (Best Acc & Best Loss) + Top-5 Ensemble"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Target Output: $OUTPUT_DIR"
echo "============================================================"

nvidia-smi

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 1. Automated TTA Sweep on Best Accuracy Checkpoint (ckpt-29)
echo -e "\n[1/3] Running TTA Sweep on Best Accuracy Checkpoint..."
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

# 2. Automated TTA Sweep on Best Loss Checkpoint (ckpt-9 with .index stripped)
echo -e "\n[2/3] Running TTA Sweep on Best Loss Checkpoint..."
BEST_LOSS_CKPT=$(ls -d "$OUTPUT_DIR/checkpoints/best_loss"/ckpt-*.index 2>/dev/null | tail -n 1 | sed 's/\.index$//' || true)
if [ -n "$BEST_LOSS_CKPT" ]; then
    echo "Found best loss checkpoint prefix: $BEST_LOSS_CKPT"
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_LOSS_CKPT" --step 0.05 || true
else
    echo "[WARN] No best loss checkpoint found in $OUTPUT_DIR/checkpoints/best_loss"
fi

# 3. Automated Top-5 Checkpoint Softmax Ensemble + TTA Evaluation
echo -e "\n[3/3] Running Top-5 Checkpoint Softmax Ensemble + TTA..."
if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
    "$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true
else
    echo "[WARN] scripts/evaluate_top5_ensemble_siglip2.py not found!"
fi

echo "============================================================"
echo " Evaluation Finished: $(date)"
echo "============================================================"
