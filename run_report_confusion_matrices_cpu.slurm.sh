#!/bin/bash
#SBATCH --job-name=FER_REPORT_CM_CPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_REPORT_CM_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_REPORT_CM_CPU_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

# 100% CPU Pure Inference / Reporting (Zero GPU requirement)
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
EXP_DIR="$ROOT/outputs/papers/siglip2-confusion_v4"

echo "============================================================"
echo " FER2013 Confusion Matrix & Classification Report (CPU ONLY)"
echo " Target Exp Dir : $EXP_DIR"
echo " Node           : $(hostname)"
echo " CUDA Visible   : $CUDA_VISIBLE_DEVICES (Pure CPU)"
echo " Start          : $(date)"
echo "============================================================"

# 1. Report and visualize CKPT-42 (Best Single Model: 76.68% Acc, 0.7593 Macro-F1)
echo "------------------------------------------------------------"
echo " [1] Generating Report & Heatmaps for CKPT-42 (100% CPU)..."
echo "------------------------------------------------------------"
"$FER_PY" -u scripts/report_confusion_matrices.py --exp-dir "$EXP_DIR" --checkpoint ckpt-42

# 2. Also generate report for CKPT-41 and CKPT-43 if available
for CKPT in ckpt-41 ckpt-43 ckpt-38; do
    if [ -f "$EXP_DIR/test_metrics_${CKPT}_tta_hflip.json" ] || [ -f "$EXP_DIR/eval_individual_ckpts/${CKPT}_eval.json" ]; then
        echo "------------------------------------------------------------"
        echo " Generating Report for $CKPT..."
        echo "------------------------------------------------------------"
        "$FER_PY" -u scripts/report_confusion_matrices.py --exp-dir "$EXP_DIR" --checkpoint "$CKPT" || true
    fi
done

echo "============================================================"
echo " Finished Confusion Matrix & Report Generation on CPU!"
echo " PNG Heatmaps saved in: $EXP_DIR/"
echo " End: $(date)"
echo "============================================================"
