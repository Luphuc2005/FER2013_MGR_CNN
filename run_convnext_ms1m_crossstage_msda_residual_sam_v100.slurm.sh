#!/bin/bash
#SBATCH --job-name=FER_MSDA_SAM
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_MSDA_SAM_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_MSDA_SAM_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
FER_PY=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python
CONFIG="$ROOT/config_convnext_ms1m_crossstage_msda_residual_sam.yaml"
OUTPUT_DIR="$ROOT/outputs/tf_runs/convnext_ms1m_crossstage_msda_residual_sam"

cd "$ROOT"
mkdir -p logs "$OUTPUT_DIR/checkpoints"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-1}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="${TF_XLA_FLAGS:---tf_xla_enable_xla_devices=false}"
export MGR_GPU_IDS=${MGR_GPU_IDS:-0}
export MGR_REQUIRE_TWO_GPUS=0
export MGR_MIN_GPUS=1
export MGR_BATCH_SIZE_PER_GPU=${MGR_BATCH_SIZE_PER_GPU:-32}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " FER2013 ConvNeXt-B MS1M Cross-Stage MSDA Residual SAM (1x V100)"
echo "============================================================"
echo "hostname=$(hostname)"
echo "job id=${SLURM_JOB_ID:-standalone}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MGR_BATCH_SIZE_PER_GPU=$MGR_BATCH_SIZE_PER_GPU"
echo "ROOT=$ROOT"
echo "Python=$FER_PY"
echo "config path=$CONFIG"
echo "output path=$OUTPUT_DIR"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }
[ -f "$ROOT/pretrained/convnext_base_ms1m_arcface.pth" ] || { echo "[ERROR] Missing MS1M pretrained checkpoint: $ROOT/pretrained/convnext_base_ms1m_arcface.pth"; exit 1; }

"$FER_PY" - <<'PY'
import tensorflow as tf
print("TensorFlow version:", tf.__version__, flush=True)
print("GPU detection:", tf.config.list_physical_devices("GPU"), flush=True)
PY

"$FER_PY" -u scripts/train_convnext_ms1m_crossstage_msda_residual.py   --config "$CONFIG"

echo "============================================================"
echo " FER2013 MSDA Residual SAM training completed"
echo "End: $(date)"
echo "============================================================"
