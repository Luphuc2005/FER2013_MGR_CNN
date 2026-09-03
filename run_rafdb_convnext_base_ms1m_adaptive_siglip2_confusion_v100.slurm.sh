#!/bin/bash
#SBATCH --job-name=RAFDB_SIGLIP2_CONF
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SIGLIP2_CONF_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SIGLIP2_CONF_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/rafdb_adaptive_siglip2_confusion

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"

echo "============================================================"
echo " RAF-DB ConvNeXt-Base MS1M Adaptive Multi-Granularity SigLIP 2"
echo " + Confusion-Aware Hard Semantic Separation"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Dataset: /home/ptbao/projects/FER2013_MGR_CNN/data/rafdb"
echo "Output: outputs/papers/rafdb_adaptive_siglip2_confusion"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 0. Smoke Test (1 Batch pipeline check)
echo "============================================================"
echo " Running Pre-flight RAF-DB Smoke Test (1 Batch)..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"

# 1. Train Model
echo "============================================================"
echo " Starting Full RAF-DB Training..."
echo "============================================================"
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
BEST_LOSS_CKPT=$(ls -d $ROOT/outputs/papers/rafdb_adaptive_siglip2_confusion*/checkpoints/best_loss/ckpt-*.index 2>/dev/null | tail -n 1 | sed 's/\.index$//' || true)
if [ -n "$BEST_LOSS_CKPT" ]; then
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_LOSS_CKPT" --step 0.05 || true
fi

# 4. Top-5 Ensemble Evaluation
echo "============================================================"
echo " Running Top-5 Checkpoint Softmax Ensemble Evaluation..."
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true

echo "============================================================"
echo " RAF-DB SigLIP 2 Pipeline Completed Successfully!"
echo " End: $(date)"
echo "============================================================"
