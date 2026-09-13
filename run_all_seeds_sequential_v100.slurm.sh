#!/bin/bash
#SBATCH --job-name=FER_SIGLIP2_ALL_SEEDS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_ALL_SEEDS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_ALL_SEEDS_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi

# List of seeds to execute sequentially
SEEDS=(42 123 0 3407 2024 777)

for SEED in "${SEEDS[@]}"; do
    CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed${SEED}.yaml"
    OUTPUT_DIR="outputs/papers/siglip2-confusion-seed${SEED}"
    
    echo "============================================================"
    echo " STARTING PIPELINE FOR SEED: $SEED"
    echo " Config: $CONFIG"
    echo " Output: $OUTPUT_DIR"
    echo " Start:  $(date)"
    echo "============================================================"

    # 1. Train Model
    "$FER_PY" -u train.py --config "$CONFIG"

    # 2. Automated TTA Sweep on Best Accuracy Checkpoint
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

    # 3. Automated TTA Sweep on Best Loss Checkpoint
    BEST_LOSS_CKPT=$(ls -d "$ROOT/$OUTPUT_DIR"/checkpoints/best_loss/ckpt-*.index 2>/dev/null | tail -n 1 | sed 's/\.index$//' || true)
    if [ -n "$BEST_LOSS_CKPT" ]; then
        "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_LOSS_CKPT" --step 0.05 || true
    fi

    # 4. Automated Top-5 Checkpoint Softmax Ensemble + TTA Evaluation
    if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
        "$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true
    fi

    echo "Finished SEED $SEED at $(date)"
done

echo "============================================================"
echo " All 6 Seeds Finished Successfully!"
echo " End: $(date)"
echo "============================================================"
