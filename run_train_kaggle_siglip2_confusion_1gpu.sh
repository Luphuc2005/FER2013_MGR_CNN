#!/usr/bin/env bash
# End-to-End Pipeline for Kaggle 1-GPU: ConvNeXt-Base MS1M Adaptive SigLIP2 + Confusion-Aware Separation

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/kaggle/config_convnext_base_ms1m_adaptive_siglip2_confusion_1gpu_top5_tta.yaml}"
LOG_DIR="/kaggle/working/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MGR_GPU_IDS="${MGR_GPU_IDS:-0}"
export MGR_REQUIRE_TWO_GPUS=0
export MGR_MIN_GPUS=1

STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_siglip2_confusion_1gpu_${STAMP}.log"
SWEEP_LOG="${LOG_DIR}/sweep_tta_${STAMP}.log"

echo "============================================================"
echo " Starting SigLIP 2 Confusion-Aware Training on Kaggle 1-GPU "
echo " Config: ${CONFIG_PATH}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES} | Batch Size: 16"
echo "============================================================"

# Step 1: Train model & save Top 5 checkpoints
python3 train.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${TRAIN_LOG}"

echo "============================================================"
echo " Step 1 Completed! Training log saved at ${TRAIN_LOG}      "
echo " Step 2: Running TTA Weight Sweep Grid Search (0.00 -> 1.00)"
echo "============================================================"

# Step 2: Sweep TTA weights on Validation and Test sets
python3 sweep_tta_weights.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${SWEEP_LOG}"

echo "============================================================"
echo " Full Kaggle 1-GPU Pipeline Completed Successfully!         "
echo " - Checkpoints saved (Top 5 Best Accuracy & Top 5 Best Loss) "
echo " - TTA Sweep Log saved at ${SWEEP_LOG}                      "
echo "============================================================"
