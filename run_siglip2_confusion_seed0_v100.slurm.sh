#!/bin/bash
#SBATCH --job-name=FER_SIGLIP2_SEED0
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_SEED0_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_SEED0_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/siglip2-confusion-seed0

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed0.yaml"

echo "============================================================"
echo " FER2013 ConvNeXt-Base MS1M SigLIP 2 Confusion (Seed 0)"
echo " CPU: 32 threads | Full TTA | Top-5 Ensemble"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Output: outputs/papers/siglip2-confusion-seed0"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 1. Train Model
"$FER_PY" -u train.py --config "$CONFIG"

# 2. Automated TTA Sweep on Best Accuracy Checkpoint
echo "============================================================"
echo " Running Automated TTA Sweep on Best Accuracy Checkpoint..."
echo "============================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

# 3. Automated TTA Sweep on Best Loss Checkpoint
echo "============================================================"
echo " Running Automated TTA Sweep on Best Loss Checkpoint..."
echo "============================================================"
BEST_LOSS_CKPT=$(ls -d $ROOT/outputs/papers/siglip2-confusion-seed0*/checkpoints/best_loss/ckpt-*.index 2>/dev/null | tail -n 1 | sed 's/\.index$//' || true)
if [ -n "$BEST_LOSS_CKPT" ]; then
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_LOSS_CKPT" --step 0.05 || true
fi

# 4. Automated Top-5 Checkpoint Softmax Ensemble + TTA Evaluation
echo "============================================================"
echo " Running Automated Top-5 Checkpoint Softmax Ensemble + TTA..."
echo "============================================================"
if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
    "$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true
fi

# 5. Comprehensive 15-Checkpoint TTA Sweep & Ensemble
echo "============================================================"
echo " Running TTA Sweep & Ensemble on all 15 Checkpoints..."
echo "============================================================"
"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$CONFIG" || true


echo "============================================================"
echo " Completed Seed 0 Pipeline (Training + TTA + Top-5 Ensemble)"
echo " End: $(date)"
echo "============================================================"
