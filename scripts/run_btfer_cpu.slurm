#!/bin/bash
#SBATCH --job-name=btfer_expw_ens_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/btfer_expw_ens_cpu_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/btfer_expw_ens_cpu_%j.err

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"

mkdir -p logs outputs/papers/btfer_crossdomain/expw_siglip2_confusion_top5_ensemble

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
echo " BTFER zero-shot ExpW top-5 checkpoint ensemble CPU evaluation"
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
    --config "$ROOT/config_expw_convnext_base_ms1m_adaptive_siglip2_confusion.yaml" \
    --data-root "$ROOT/data/btfer/Batch_Ready 7" \
    --experiment-dir "$ROOT/outputs/papers/expw_convnext_base_ms1m_adaptive_siglip2_confusion" \
    --output-dir "$ROOT/outputs/papers/btfer_crossdomain/expw_siglip2_confusion_top5_ensemble" \
    --checkpoint-dir "$ROOT/outputs/papers/expw_convnext_base_ms1m_adaptive_siglip2_confusion/checkpoints/best" \
    --ensemble-top-k 5 \
    --batch-size 64 \
    --intra-op-threads "${SLURM_CPUS_PER_TASK:-16}" \
    --inter-op-threads 4 \
    --save-confusion-png

echo "============================================================"
echo " BTFER ExpW top-5 ensemble CPU evaluation completed"
echo "End: $(date)"
echo "============================================================"
