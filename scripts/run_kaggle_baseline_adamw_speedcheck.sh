#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/kaggle/config_convnext_base_ms1m_arcface_baseline_2gpu.yaml"
LOG_DIR="outputs/kaggle_speedcheck_logs"
mkdir -p "$LOG_DIR"

echo "[SPEEDCHECK] 1x T4 AdamW full-backbone run"
MGR_GPU_IDS=0 \
MGR_REQUIRE_TWO_GPUS=0 \
MGR_MIN_GPUS=1 \
MGR_OUTPUT_DIR=outputs/speedcheck_adamw_1gpu \
python -u train.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/adamw_1gpu.log"

echo "[SPEEDCHECK] 2x T4 AdamW full-backbone run"
MGR_GPU_IDS=0,1 \
MGR_REQUIRE_TWO_GPUS=1 \
MGR_MIN_GPUS=2 \
MGR_OUTPUT_DIR=outputs/speedcheck_adamw_2gpu \
python -u train.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/adamw_2gpu.log"

echo "[SPEEDCHECK] Summary files:"
echo "  outputs/speedcheck_adamw_1gpu/training_history.csv"
echo "  outputs/speedcheck_adamw_2gpu/training_history.csv"
echo "Compare train_time_sec and train_samples_per_sec. Lower total epoch time is useful, but throughput tells the real training speed."
