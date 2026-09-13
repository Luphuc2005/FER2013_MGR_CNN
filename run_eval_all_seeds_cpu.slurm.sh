#!/bin/bash
#SBATCH --job-name=EVAL_ALL_SEEDS_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_ALL_SEEDS_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_ALL_SEEDS_CPU_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CPUS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export TF_NUM_INTRAOP_THREADS="$CPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

echo "============================================================"
echo " PURE CPU EVALUATION FOR ALL TRAINED SEEDS (${CPUS} Threads)"
echo " Top-5 Checkpoint Softmax Ensemble + TTA Evaluation"
echo "============================================================"
echo "Start: $(date)"

SEEDS=(42 123 0 3407 2024 777)

for SEED in "${SEEDS[@]}"; do
    CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed${SEED}.yaml"
    OUTPUT_DIR="$ROOT/outputs/papers/siglip2-confusion-seed${SEED}"
    
    if [ -d "$OUTPUT_DIR/checkpoints/best" ]; then
        echo "------------------------------------------------------------"
        echo " Evaluating SEED $SEED in $OUTPUT_DIR"
        echo "------------------------------------------------------------"
        "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true
        
        if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
            "$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true
        fi
    else
        echo "[SKIP] Seed $SEED checkpoints not found yet in $OUTPUT_DIR"
    fi
done

echo "============================================================"
echo " All Evaluations Finished at: $(date)"
echo "============================================================"
