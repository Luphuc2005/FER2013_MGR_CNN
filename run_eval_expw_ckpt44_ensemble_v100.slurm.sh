#!/bin/bash
#SBATCH --job-name=EVAL_EXPW_CKPT44
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_EXPW_CKPT44_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_EXPW_CKPT44_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_expw_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/expw_convnext_base_ms1m_adaptive_siglip2_confusion"

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " Evaluation Pipeline: ExpW Epoch 44 Checkpoint & Ensemble"
echo "============================================================"
echo "Config: $CONFIG"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================================"

nvidia-smi

# Find Checkpoint Epoch 44
CKPT_44="$OUTPUT_DIR/checkpoints/best/ckpt-44"
if [ ! -f "${CKPT_44}.index" ]; then
    # Try finding in best_loss or last if not in best
    CKPT_44=$(ls $OUTPUT_DIR/checkpoints/*/ckpt-44.index 2>/dev/null | head -n 1 | sed 's/\.index$//' || true)
fi

echo "------------------------------------------------------------"
echo " [STEP 1] Running TTA Weight Sweep on Checkpoint Epoch 44..."
echo "------------------------------------------------------------"
if [ -n "$CKPT_44" ] && [ -f "${CKPT_44}.index" ]; then
    echo "[INFO] Found Checkpoint 44: $CKPT_44"
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$CKPT_44" --step 0.05
else
    echo "[INFO] Specific ckpt-44 index file not found in exact path, running TTA sweep on Best Checkpoint in manager..."
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05
fi

echo "------------------------------------------------------------"
echo " [STEP 2] Running Top-5 Checkpoint Softmax Ensemble + TTA..."
echo "------------------------------------------------------------"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG"

echo "============================================================"
echo " ExpW Evaluation Completed Successfully!"
echo "============================================================"
