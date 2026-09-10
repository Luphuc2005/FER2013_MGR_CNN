#!/bin/bash
#SBATCH --job-name=FERPLUS_SIGLIP2_8CLS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FERPLUS_SIGLIP2_8CLS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FERPLUS_SIGLIP2_8CLS_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/ferplus_siglip2_confusion_majority8

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"

echo "============================================================"
echo " FERPlus Official Majority8 ConvNeXt-Base MS1M SigLIP2"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Output: outputs/papers/ferplus_siglip2_confusion_majority8"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " Running FERPlus official Majority8 smoke test..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_ferplus_pipeline.py --config "$CONFIG" --num-batches 1

echo "============================================================"
echo " Starting FERPlus official Majority8 full training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

echo "============================================================"
echo " FERPlus official Majority8 pipeline completed"
echo " End: $(date)"
echo "============================================================"
