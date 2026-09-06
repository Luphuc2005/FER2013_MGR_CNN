#!/bin/bash
#SBATCH --job-name=btfer_raf_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/btfer_raf_cpu_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/btfer_raf_cpu_%j.err

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"

mkdir -p logs outputs/papers/btfer_crossdomain/rafdb_adaptive_siglip2_confusion_v2_v4

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TF_NUM_INTEROP_THREADS=4

if [ -x "$ROOT/fer2013_env/bin/python" ]; then
    source "$ROOT/fer2013_env/bin/activate"
    FER_PY="$ROOT/fer2013_env/bin/python"
else
    FER_PY="$(command -v python)"
fi

echo "============================================================"
echo " BTFER zero-shot cross-domain CPU evaluation"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-16}"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start: $(date)"
echo "============================================================"

"$FER_PY" -u scripts/evaluate_btfer_cpu.py \
    --data-root "$ROOT/data/btfer/Batch_Ready 7" \
    --experiment-dir "$ROOT/outputs/papers/rafdb_adaptive_siglip2_confusion_v2_v4" \
    --output-dir "$ROOT/outputs/papers/btfer_crossdomain/rafdb_adaptive_siglip2_confusion_v2_v4" \
    --batch-size 64 \
    --intra-op-threads "${SLURM_CPUS_PER_TASK:-16}" \
    --inter-op-threads 4 \
    --save-confusion-png

echo "============================================================"
echo " BTFER CPU evaluation completed"
echo "End: $(date)"
echo "============================================================"
