#!/bin/bash
#SBATCH --job-name=EXPW_SIGLIP2
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EXPW_SIGLIP2_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EXPW_SIGLIP2_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/expw_convnext_base_ms1m_adaptive_siglip2_confusion

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_expw_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"

echo "============================================================"
echo " ExpW ConvNeXt-Base MS1M Adaptive SigLIP2 + Confusion Pipeline"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Images: /home/ptbao/projects/FER2013_MGR_CNN/data/expw_gdrive/data/image/extracted_full/origin"
echo "Train CSV: data/expw/expw_train.csv (68,845 samples)"
echo "Val CSV: data/expw/expw_val.csv (9,179 samples)"
echo "Test CSV: data/expw/expw_test.csv (13,769 samples)"
echo "Output: outputs/papers/expw_convnext_base_ms1m_adaptive_siglip2_confusion"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

# 0. Smoke Test Dataset Loader + 1 Batch
echo "============================================================"
echo " Running Pre-flight ExpW Dataset Loader & Model Smoke Test..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_expw_pipeline.py "$CONFIG"

# 1. Full Model Training on ExpW
echo "============================================================"
echo " Starting Full ExpW Model Training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

# 2. Automated TTA Weight Sweep on Best Accuracy Checkpoint (step 0.05)
echo "============================================================"
echo " Running Post-Training TTA Weight Sweep..."
echo "============================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05

# 3. Automated Top-5 Checkpoint Softmax Ensemble + TTA Evaluation
echo "============================================================"
echo " Running Top-5 Checkpoint Ensemble + TTA Evaluation..."
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG"

echo "============================================================"
echo " ExpW SigLIP2 Pipeline Completed Successfully!"
echo " End: $(date)"
echo "============================================================"
