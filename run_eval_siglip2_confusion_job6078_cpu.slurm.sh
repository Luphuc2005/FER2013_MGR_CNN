#!/bin/bash
#SBATCH --job-name=EVAL_JOB6078_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_JOB6078_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_JOB6078_CPU_%j.err

# GHI CHÚ: Ép chạy CPU 100% để KHÔNG tốn GPU / VRAM, tận dụng tối đa 32 CPU threads
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

# Ép chặt CPU thuần túy
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Tối ưu đa luồng CPU cho TensorFlow & NumPy
CPUS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export TF_NUM_INTRAOP_THREADS="$CPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/siglip2-confusion"

echo "============================================================"
echo " FER2013 Job 6078 Evaluation - Pure CPU (${CPUS} Threads)"
echo " TTA Sweep (Best Acc & Best Loss) + Top-5 Checkpoint Ensemble"
echo "============================================================"
echo "Job ID              : ${SLURM_JOB_ID:-standalone}"
echo "Node                : $(hostname)"
echo "Allocated CPUs      : $CPUS threads"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start Time          : $(date)"
echo "Output Directory    : $OUTPUT_DIR"
echo "============================================================"

[ -x "$FER_PY" ] || { echo "[ERROR] Python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config not found: $CONFIG"; exit 1; }

# 1. Sweep TTA trên Best Accuracy Checkpoint (ckpt-29)
echo -e "\n[1/3] Running TTA Sweep on Best Accuracy Checkpoint (CPU)..."
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

# 2. Sweep TTA trên Best Loss Checkpoint (ckpt-9)
echo -e "\n[2/3] Running TTA Sweep on Best Loss Checkpoint (CPU)..."
BEST_LOSS_CKPT=$(ls -d "$OUTPUT_DIR/checkpoints/best_loss"/ckpt-*.index 2>/dev/null | tail -n 1 | sed 's/\.index$//' || true)
if [ -n "$BEST_LOSS_CKPT" ]; then
    echo "Found best loss checkpoint: $BEST_LOSS_CKPT"
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_LOSS_CKPT" --step 0.05 || true
else
    echo "[WARN] No best loss checkpoint found."
fi

# 3. Top-5 Checkpoint Softmax Ensemble + TTA
echo -e "\n[3/3] Running Top-5 Checkpoint Softmax Ensemble + TTA (CPU)..."
if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
    "$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true
else
    echo "[WARN] scripts/evaluate_top5_ensemble_siglip2.py not found!"
fi

echo "============================================================"
echo " CPU Evaluation Finished Successfully at: $(date)"
echo "============================================================"
