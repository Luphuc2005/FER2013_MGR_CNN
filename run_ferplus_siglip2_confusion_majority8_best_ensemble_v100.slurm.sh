#!/bin/bash
#SBATCH --job-name=FERPLUS_BEST_ENS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FERPLUS_BEST_ENS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FERPLUS_BEST_ENS_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_ferplus_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
CHECKPOINT_DIR="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/checkpoints/best"
REPORT="$ROOT/outputs/papers/ferplus_siglip2_confusion_majority8/best_top5_tta_ensemble_report.json"
CHECKPOINTS=(ckpt-9 ckpt-11 ckpt-14 ckpt-24 ckpt-33)

cd "$ROOT"
mkdir -p "$ROOT/logs" "$(dirname "$REPORT")"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
export MGR_TF_INTRA_OP_THREADS=12
export MGR_TF_INTER_OP_THREADS=4
export MGR_TF_DATA_NUM_PARALLEL_CALLS=12
export MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE=16
export MGR_PREFETCH_BUFFER=4
export NVIDIA_LIB="$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

[ -x "$FER_PY" ] || { echo "[ERROR] Python environment not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config not found: $CONFIG"; exit 1; }
[ -d "$CHECKPOINT_DIR" ] || { echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"; exit 1; }

for checkpoint_name in "${CHECKPOINTS[@]}"; do
    checkpoint_prefix="$CHECKPOINT_DIR/$checkpoint_name"
    [ -f "${checkpoint_prefix}.index" ] || {
        echo "[ERROR] Missing checkpoint index: ${checkpoint_prefix}.index"
        exit 1
    }
    if ! compgen -G "${checkpoint_prefix}.data-*" > /dev/null; then
        echo "[ERROR] Missing checkpoint data shard: ${checkpoint_prefix}.data-*"
        exit 1
    fi
done

echo "============================================================"
echo " FERPlus Majority-8 SigLIP2: Best Top-5 TTA Ensemble"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Config: $CONFIG"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "TTA weights: original=0.50, hflip=0.50"
echo "Split: TEST"
echo "Report: $REPORT"
echo "============================================================"

nvidia-smi

"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py \
  --config "$CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --split test \
  --w-orig 0.50 \
  --w-flip 0.50 \
  --output "$REPORT"

echo "============================================================"
echo " FERPLUS BEST TOP-5 ENSEMBLE COMPLETED"
echo " Report: $REPORT"
echo " End: $(date)"
echo "============================================================"
