#!/bin/bash
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

# Ép chặt chạy CPU thuần túy, tuyệt đối không cấp phát VRAM hay ảnh hưởng job training GPU
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

if [ ! -x "$FER_PY" ]; then
    echo "[WARNING] $FER_PY not found, falling back to python"
    FER_PY="python"
fi

echo "============================================================"
echo " Starting Grad-CAM Candidate Generation on FER2013 (100% CPU)"
echo " Node: $(hostname)"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " Time: $(date)"
echo "============================================================"

"$FER_PY" -u generate_gradcam_fer2013_candidates.py --cpu "$@"
