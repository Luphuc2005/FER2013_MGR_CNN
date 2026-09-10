#!/bin/bash
set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"

FER_PY="$ROOT/fer2013_env/bin/python"
[ -x "$FER_PY" ] || FER_PY="python"

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

EXP_DIR="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8_v3"
CKPT_BEST_DIR="$EXP_DIR/checkpoints/best"
CKPT_BEST_LOSS_DIR="$EXP_DIR/checkpoints/best_loss"

CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion_v3.yaml"
[ -f "$CONFIG" ] || CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"

echo "=== Evaluating Best Acc and Best Loss Checkpoints ==="
echo "Config: $CONFIG"

# 1. Best Acc (ckpt-17)
if [ -f "$CKPT_BEST_DIR/ckpt-17.index" ]; then
    echo ">>> Running ckpt-17..."
    "$FER_PY" -u evaluate.py --config "$CONFIG" --checkpoint "$CKPT_BEST_DIR/ckpt-17" --split test --tta-hflip
fi

# 2. Grand Ensemble (best + best_loss)
"$FER_PY" -u scripts/evaluate_ensemble_multiview_tta.py \
    --config "$CONFIG" \
    --checkpoint-dirs "$CKPT_BEST_DIR" "$CKPT_BEST_LOSS_DIR" \
    --split test \
    --batch-size 64 \
    --sweep-step 0.05 \
    --output "$EXP_DIR/eval_grand_ensemble_best_acc_and_loss_report.json"