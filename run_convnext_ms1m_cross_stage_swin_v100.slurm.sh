#!/bin/bash
#SBATCH --job-name=FER_CROSS_STAGE_SWIN
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_CROSS_STAGE_SWIN_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_CROSS_STAGE_SWIN_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/tf_runs/convnext_ms1m_cross_stage_swin/checkpoints

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-1}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MGR_GPU_IDS=${MGR_GPU_IDS:-0}
export MGR_REQUIRE_TWO_GPUS=0
export MGR_MIN_GPUS=1
export MGR_BATCH_SIZE_PER_GPU=${MGR_BATCH_SIZE_PER_GPU:-32}

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_ms1m_cross_stage_swin.yaml"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Configure NVIDIA CUDA/CUDNN libraries for TensorFlow GPU.
export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " FER2013 ConvNeXt-B MS1M Cross-Stage Swin Fusion (1x V100)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MGR_BATCH_SIZE_PER_GPU=$MGR_BATCH_SIZE_PER_GPU"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Output: outputs/tf_runs/convnext_ms1m_cross_stage_swin"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }
[ -f "$ROOT/pretrained/convnext_base_ms1m_arcface.pth" ] || { echo "[ERROR] Missing MS1M pretrained checkpoint: $ROOT/pretrained/convnext_base_ms1m_arcface.pth"; exit 1; }

"$FER_PY" -u scripts/train_convnext_ms1m_cross_stage_swin.py \
  --config "$CONFIG"

echo "============================================================"
echo " FER2013 Cross-Stage Swin training completed"
echo "End: $(date)"
echo "============================================================"
