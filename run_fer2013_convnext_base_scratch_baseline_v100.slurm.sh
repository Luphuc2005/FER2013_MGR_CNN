#!/bin/bash
#SBATCH --job-name=FER2013_SCRATCH
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER2013_SCRATCH_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER2013_SCRATCH_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/fer2013_convnext_base_scratch_baseline

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_fer2013_convnext_base_scratch_baseline.yaml"

echo "============================================================"
echo " FER2013 ConvNeXt-Base Scratch Baseline (No Face Pretrain)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Dataset: /home/ptbao/projects/FER2013_MGR_CNN/data/fer13-split"
echo "Output: outputs/papers/fer2013_convnext_base_scratch_baseline"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 1. Train Model
echo "============================================================"
echo " Starting Full FER2013 Scratch Baseline Training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

# 2. Automated TTA Sweep on Best Accuracy Checkpoint
echo "============================================================"
echo " Running Automated TTA Sweep on Best Accuracy Checkpoint..."
echo "============================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

# 3. Top-5 Ensemble Evaluation
echo "============================================================"
echo " Running Top-5 Checkpoint Softmax Ensemble Evaluation..."
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_baseline.py --config "$CONFIG" || true

echo "============================================================"
echo " FER2013 Scratch Baseline Completed Successfully!"
echo " End: $(date)"
echo "============================================================"
