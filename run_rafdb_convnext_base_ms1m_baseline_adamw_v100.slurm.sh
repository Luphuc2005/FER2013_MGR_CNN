#!/bin/bash
#SBATCH --job-name=RAFDB_BASE_ADAMW
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_BASE_ADAMW_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_BASE_ADAMW_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/rafdb_convnext_base_ms1m_baseline_adamw

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_convnext_base_ms1m_baseline_adamw.yaml"

echo "============================================================"
echo " RAF-DB ConvNeXt-Base MS1M AdamW Baseline (7 Classes)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Dataset: /home/ptbao/projects/FER2013_MGR_CNN/data/rafdb"
echo "Output: outputs/papers/rafdb_convnext_base_ms1m_baseline_adamw"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 0. Smoke Test
echo "============================================================"
echo " Running Pre-flight RAF-DB Smoke Test (1 Batch)..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"

# 1. Train Model
echo "============================================================"
echo " Starting Full RAF-DB AdamW Baseline Training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

# 2. Automated TTA Sweep on Best Accuracy Checkpoint
echo "============================================================"
echo " Running Post-Training TTA Weight Sweep..."
echo "============================================================"
"$FER_PY" -u scripts/sweep_tta_weights.py "$CONFIG"

echo "============================================================"
echo " RAF-DB AdamW Baseline Pipeline Completed Successfully!"
echo " End: $(date)"
echo "============================================================"
