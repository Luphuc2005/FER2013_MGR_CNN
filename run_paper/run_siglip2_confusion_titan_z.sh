#!/usr/bin/env bash
# ============================================================
# FER2013 ConvNeXt-Base MS1M Adaptive SigLIP2 + Confusion Pipeline
# Dedicated Runner for Titan Z / Dual-GPU Server
# ============================================================

set -euo pipefail

# Ensure working directory is the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# 1. Environment & GPU Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MGR_GPU_IDS="${MGR_GPU_IDS:-0,1}"
export MGR_REQUIRE_TWO_GPUS=0
export MGR_MIN_GPUS=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Detect Python interpreter
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Python interpreter not found! Please activate your virtualenv."
    exit 1
fi

CONFIG="run_paper/config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/run_paper_siglip2_confusion_${STAMP}.log"

echo "============================================================"
echo " FER2013 SigLIP 2 Confusion-Aware Dual-GPU Execution "
echo "============================================================"
echo " Start:                $(date)"
echo " Project Root:         ${ROOT_DIR}"
echo " Config:               ${CONFIG}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo " Log file:             ${RUN_LOG}"
echo "============================================================"

nvidia-smi || true

# 1. Model Training (Auto-increments output_dir if previous run exists)
echo ""
echo "[1/3] Starting Model Training..."
"${PYTHON_BIN}" -u train.py --config "${CONFIG}" 2>&1 | tee -a "${RUN_LOG}"

# 2. Automated TTA Sweep on Best Accuracy & Best Loss Checkpoints
echo ""
echo "============================================================"
echo " [2/3] Running Automated TTA Weight Sweep..."
echo "============================================================"
"${PYTHON_BIN}" -u sweep_tta_weights.py --config "${CONFIG}" --step 0.05 2>&1 | tee -a "${RUN_LOG}" || true

BEST_LOSS_CKPT=$(ls -d ${ROOT_DIR}/outputs/papers/siglip2-confusion*/checkpoints/best_loss/ckpt-*.index 2>/dev/null | tail -n 1 | sed 's/\.index$//' || true)
if [ -n "$BEST_LOSS_CKPT" ]; then
    echo "[INFO] Running TTA Sweep on Best Loss Checkpoint: ${BEST_LOSS_CKPT}"
    "${PYTHON_BIN}" -u sweep_tta_weights.py --config "${CONFIG}" --checkpoint "${BEST_LOSS_CKPT}" --step 0.05 2>&1 | tee -a "${RUN_LOG}" || true
fi

# 3. Automated Top-5 Checkpoint Softmax Ensemble + TTA Evaluation
echo ""
echo "============================================================"
echo " [3/3] Running Automated Top-5 Checkpoint Softmax Ensemble + TTA..."
echo "============================================================"
if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
    "${PYTHON_BIN}" -u scripts/evaluate_top5_ensemble_siglip2.py --config "${CONFIG}" 2>&1 | tee -a "${RUN_LOG}" || true
fi

echo ""
echo "============================================================"
echo " FER2013 SigLIP 2 Paper Pipeline Completed Successfully!"
echo " End: $(date)"
echo " Log: ${RUN_LOG}"
echo "============================================================"
