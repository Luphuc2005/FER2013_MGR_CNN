#!/usr/bin/env bash
# Script to run FER2013 ConvNeXt-Base MS1M Adaptive SigLIP2 + Confusion-Aware Separation on Kaggle 2-GPU (Global BS 32)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/kaggle/config_convnext_base_ms1m_adaptive_siglip2_confusion_2gpu.yaml}"
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

echo "============================================================"
echo " Starting SigLIP 2 Confusion-Aware Training on Kaggle 1-GPU "
echo " Config: ${CONFIG_PATH}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES} | Batch Size: 16"
echo "============================================================"

python3 train.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${TRAIN_LOG}"

echo "============================================================"
echo " Step 1 Completed! Log saved at ${TRAIN_LOG}                "
echo " Step 2: Running TTA Weight Sweep Grid Search (0.00 to 1.00)..."
echo "============================================================"

SWEEP_LOG="${LOG_DIR}/sweep_tta_${STAMP}.log"
python3 sweep_tta_weights.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${SWEEP_LOG}"

echo "============================================================"
echo " Full Kaggle Pipeline Completed Successfully!               "
echo " - Checkpoints saved in outputs directory (Top 5 Best)      "
echo " - TTA Sweep Log: ${SWEEP_LOG}                              "
echo "============================================================"
