#!/bin/bash
#SBATCH --job-name=FER_DUAL_MS1M_3D_GUIDED
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_DUAL_MS1M_3D_GUIDED_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_DUAL_MS1M_3D_GUIDED_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/dual_convnext_smirk_guided_attention_ms1m_fer_scratch/logs outputs/dual_convnext_smirk_guided_attention_ms1m_fer_scratch/checkpoints

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_dual_convnext_smirk_guided_attention_ms1m_fer_scratch.yaml"

echo "============================================================"
echo " RGB ConvNeXt-Base MS1M + SMIRK 3D-Guided Residual Attention"
echo " No FER checkpoint restore; FER classifier is random init"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

"$FER_PY" -u scripts/train_dual_convnext_smirk_guided_attention_ms1m_fer_scratch.py \
    --config "$CONFIG"

echo "[INFO] Running Top-5 Checkpoint Softmax Ensemble Evaluation..."
"$FER_PY" -u scripts/evaluate_top5_ensemble_dual_convnext.py \
    --config "$CONFIG"

echo "============================================================"
echo " Training & Top-5 Ensemble evaluation completed"
echo " End: $(date)"
echo "============================================================"
