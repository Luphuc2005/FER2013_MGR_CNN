#!/bin/bash
#SBATCH --job-name=FER_BASE_RAW_PAPER
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_BASE_RAW_PAPER_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_BASE_RAW_PAPER_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/convnext_base_ms1m_raw_paper_baseline

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_raw_paper_baseline.yaml"

echo "============================================================"
echo " FER2013 ConvNeXt-Base MS1M Raw Paper Baseline"
echo " (No Augmentation | No SAM | Standard AdamW & Cross-Entropy)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Output: outputs/tf_runs/convnext_base_ms1m_raw_paper_baseline"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

"$FER_PY" -u train.py --config "$CONFIG"

echo "============================================================"
echo " FER2013 ConvNeXt-Base MS1M Raw Paper Baseline Completed"
echo " End: $(date)"
echo "============================================================"
