#!/bin/bash
#SBATCH --job-name=AFFECTNET_BASELINE
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/AFFECTNET_BASELINE_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/AFFECTNET_BASELINE_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/affectnet_baseline

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_affectnet_convnext_base_ms1m_baseline.yaml"

echo "============================================================"
echo " AffectNet-7 ConvNeXt-Base MS1M Baseline"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Output: outputs/papers/affectnet_baseline"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 0. Smoke test first
echo "============================================================"
echo " Running AffectNet-7 Baseline Smoke Test..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_affectnet_pipeline.py --config "$CONFIG" --num-batches 5 || exit 1

# 1. Train Model
echo "============================================================"
echo " Starting AffectNet-7 Baseline Training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

# 2. Automated TTA Sweep
echo "============================================================"
echo " Running Automated TTA Sweep on Best Accuracy Checkpoint..."
echo "============================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

echo "============================================================"
echo " AffectNet-7 Baseline Pipeline Completed"
echo " End: $(date)"
echo "============================================================"
