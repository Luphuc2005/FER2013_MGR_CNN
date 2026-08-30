#!/bin/bash
#SBATCH --job-name=fer_smirk_aux
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/fer_smirk_aux_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/fer_smirk_aux_%j.err

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"
mkdir -p logs

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_smirk_auxiliary.yaml"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# Configure NVIDIA CUDA/cuDNN libraries for TensorFlow GPU
export NVIDIA_LIB="$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " FER2013 - CONVNEXT SMIRK 3D AUXILIARY SUPERVISION"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

# ------------------------------------------------------------------
# STEP 1: EXTRACT SMIRK 3D AUXILIARY TARGETS (TRAIN, VAL, TEST)
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo " STEP 1: EXTRACT SMIRK 3D AUXILIARY TARGETS"
echo "============================================================"
"$FER_PY" scripts/extract_smirk_auxiliary_targets.py \
  --config "$CONFIG" \
  --splits train val test \
  --batch-size 128

# ------------------------------------------------------------------
# STEP 2: RUN ABLATION EXPERIMENTS
# ------------------------------------------------------------------
for ABLATION in "baseline" "exp" "exp_jaw" "exp_jaw_head"; do
  echo ""
  echo "============================================================"
  echo " STEP 2: RUNNING ABLATION EXPERIMENT: $ABLATION"
  echo "============================================================"
  "$FER_PY" scripts/train_convnext_smirk_auxiliary.py \
    --config "$CONFIG" \
    --ablation "$ABLATION"
done

echo ""
echo "============================================================"
echo " ALL SMIRK 3D AUXILIARY EXPERIMENTS COMPLETED"
echo " End: $(date)"
echo "============================================================"
