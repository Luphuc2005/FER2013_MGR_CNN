#!/bin/bash
#SBATCH --job-name=RAFDB_BESTLOSS_ENS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_BESTLOSS_ENS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_BESTLOSS_ENS_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v2.yaml"
CHECKPOINT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v2/checkpoints/best_loss"
REPORT="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v2/best_loss_5ckpts_tta_ensemble_report.json"

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

mapfile -t CHECKPOINT_INDEXES < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ckpt-*.index' -print | sort -V)
if [ "${#CHECKPOINT_INDEXES[@]}" -ne 5 ]; then
    echo "[ERROR] Expected exactly 5 best-loss checkpoints, found ${#CHECKPOINT_INDEXES[@]} in $CHECKPOINT_DIR"
    printf '  %s\n' "${CHECKPOINT_INDEXES[@]}"
    exit 1
fi

for index_file in "${CHECKPOINT_INDEXES[@]}"; do
    checkpoint_prefix="${index_file%.index}"
    if ! compgen -G "${checkpoint_prefix}.data-*" > /dev/null; then
        echo "[ERROR] Missing TensorFlow checkpoint data shard for: $checkpoint_prefix"
        exit 1
    fi
done

echo "============================================================"
echo " RAF-DB Semantic-Stable v2: Best-Loss Top-5 Ensemble"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "TF threads: intra=$MGR_TF_INTRA_OP_THREADS inter=$MGR_TF_INTER_OP_THREADS data_parallel=$MGR_TF_DATA_NUM_PARALLEL_CALLS private_pool=$MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE prefetch=$MGR_PREFETCH_BUFFER"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "Config: $CONFIG"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Checkpoints:"
printf '  - %s\n' "${CHECKPOINT_INDEXES[@]%.index}"
echo "TTA weights: original=0.60, hflip=0.40"
echo "Split: TEST"
echo "Report: $REPORT"
echo "============================================================"

nvidia-smi

"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py \
  --config "$CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --split test \
  --w-orig 0.60 \
  --w-flip 0.40 \
  --output "$REPORT"

echo "============================================================"
echo " BEST-LOSS TOP-5 ENSEMBLE COMPLETED"
echo " Report: $REPORT"
echo " End: $(date)"
echo "============================================================"
