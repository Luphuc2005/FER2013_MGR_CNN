#!/bin/bash
#SBATCH --job-name=FER_SMIRK_XATTN
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SMIRK_XATTN_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SMIRK_XATTN_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1

# ============================================================
# PATHS & ENVIRONMENTS
# ============================================================

SMIRK_PY="$ROOT/smirk_env/bin/python"
FER_PY="$ROOT/fer2013_env/bin/python"

CONFIG="$ROOT/config_smirk_geometry_cross_attention.yaml"

if [ -d "$ROOT/external/smirk" ]; then
    SMIRK_ROOT="$ROOT/external/smirk"
elif [ -d "$ROOT/smirk" ]; then
    SMIRK_ROOT="$ROOT/smirk"
elif [ -d "/home/ptbao/projects/smirk" ]; then
    SMIRK_ROOT="/home/ptbao/projects/smirk"
else
    SMIRK_ROOT="${SMIRK_ROOT:-$ROOT/external/smirk}"
fi

SMIRK_CHECKPOINT="${SMIRK_CHECKPOINT:-$SMIRK_ROOT/pretrained_models/SMIRK_em1.pt}"
BASELINE_CKPT="${BASELINE_CKPT:-$ROOT/outputs/tf_runs/convnext_base_ms1m_arcface_baseline/checkpoints/best/ckpt-43}"

export SMIRK_ROOT
export SMIRK_CHECKPOINT
export PYTHONPATH="$SMIRK_ROOT:$ROOT:${PYTHONPATH:-}"

# Wrapper helper to run SMIRK PyTorch commands with clean environment or inside Apptainer if set
run_smirk_cmd() {
    if [ -n "${APPTAINER_IMAGE:-}" ] && command -v apptainer >/dev/null 2>&1; then
        apptainer exec --nv "$APPTAINER_IMAGE" "$SMIRK_PY" "$@"
    elif [ -n "${SINGULARITY_IMAGE:-}" ] && command -v singularity >/dev/null 2>&1; then
        singularity exec --nv "$SINGULARITY_IMAGE" "$SMIRK_PY" "$@"
    else
        env -u LD_LIBRARY_PATH -u LD_PRELOAD "$SMIRK_PY" "$@"
    fi
}

echo "============================================================"
echo " FER2013 - SMIRK 3D GEOMETRY CROSS ATTENTION"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "Start: $(date)"
echo
echo "ROOT=$ROOT"
echo "SMIRK_PY=$SMIRK_PY"
echo "FER_PY=$FER_PY"
echo "SMIRK_ROOT=$SMIRK_ROOT"
echo "SMIRK_CHECKPOINT=$SMIRK_CHECKPOINT"
echo "BASELINE_CKPT=$BASELINE_CKPT"
echo "CONFIG=$CONFIG"
echo "============================================================"

nvidia-smi

# ============================================================
# FILE CHECK
# ============================================================

[ -x "$SMIRK_PY" ] || {
    echo "[ERROR] SMIRK python not found: $SMIRK_PY"
    exit 1
}

[ -x "$FER_PY" ] || {
    echo "[ERROR] FER python not found: $FER_PY"
    exit 1
}

[ -d "$SMIRK_ROOT" ] || {
    echo "[ERROR] SMIRK repo not found: $SMIRK_ROOT"
    exit 1
}

[ -f "$SMIRK_CHECKPOINT" ] || {
    echo "[ERROR] SMIRK checkpoint not found: $SMIRK_CHECKPOINT"
    exit 1
}

[ -f "$CONFIG" ] || {
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
}

[ -f "$BASELINE_CKPT.index" ] || {
    echo "[ERROR] Baseline ckpt-43 not found: $BASELINE_CKPT"
    exit 1
}

# ============================================================
# STEP 0: FAIL-FAST ENVIRONMENT CHECK FOR SMIRK / PYTORCH / GPU
# ============================================================

echo
echo "============================================================"
echo " STEP 0: CHECK SMIRK / PYTORCH / PYTORCH3D / GPU"
echo "============================================================"

run_smirk_cmd - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("Torch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("[FAIL-FAST] PyTorch cannot access GPU.")

print("GPU Device Name:", torch.cuda.get_device_name(0))

try:
    import pytorch3d
    print("PyTorch3D: OK")
except ImportError as e:
    raise ImportError(f"[FAIL-FAST] PyTorch3D import failed: {e}")

print("SMIRK_GPU_ENV_OK")
PY

# ============================================================
# STEP 1: SMOKE TEST SMIRK 3D EXTRACTION
# ============================================================

echo
echo "============================================================"
echo " STEP 1: SMOKE TEST SMIRK 3D EXTRACTION - 16 TRAIN IMAGES"
echo "============================================================"

run_smirk_cmd -u scripts/extract_smirk_vlm_geometry_tokens.py \
    --config "$CONFIG" \
    --smirk-root "$SMIRK_ROOT" \
    --smirk-checkpoint "$SMIRK_CHECKPOINT" \
    --device cuda \
    --splits train \
    --batch-size 8 \
    --max-samples-per-split 16 \
    --force \
    --save-preview

# ============================================================
# STEP 2: FULL SMIRK 3D GEOMETRY CACHE
# ============================================================

echo
echo "============================================================"
echo " STEP 2: FULL 3D GEOMETRY CACHE - TRAIN / VAL / TEST"
echo "============================================================"

run_smirk_cmd -u scripts/extract_smirk_vlm_geometry_tokens.py \
    --config "$CONFIG" \
    --smirk-root "$SMIRK_ROOT" \
    --smirk-checkpoint "$SMIRK_CHECKPOINT" \
    --device cuda \
    --splits train val test \
    --batch-size 64 \
    --force

echo
echo "============================================================"
echo " 3D CACHE COMPLETED"
echo " Expected shapes:"
echo " SMIRK input:            [B,3,224,224]"
echo " FLAME vertices:         [B,V,3]"
echo " depth tokens:           [B,49,768]"
echo " normal tokens:          [B,49,768]"
echo " geometry cache:         [B,98,768]"
echo "============================================================"

# ============================================================
# STEP 3: TENSORFLOW CROSS-ATTENTION SMOKE TRAIN
# ============================================================

echo
echo "============================================================"
echo " STEP 3: CROSS-ATTENTION SMOKE TRAIN - 1 BATCH"
echo "============================================================"

"$FER_PY" -u scripts/train_smirk_geometry_cross_attention.py \
    --config "$CONFIG" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --batch-size 4 \
    --max-train-batches 1 \
    --max-eval-batches 1 \
    --smoke-only

# ============================================================
# STEP 4: FULL TRAINING + FINAL EVALUATION
# ============================================================

echo
echo "============================================================"
echo " STEP 4: FULL GEOMETRY CROSS-ATTENTION TRAINING"
echo " Baseline: ConvNeXt-MS1M ckpt-43"
echo "============================================================"

"$FER_PY" -u scripts/train_smirk_geometry_cross_attention.py \
    --config "$CONFIG" \
    --baseline-checkpoint "$BASELINE_CKPT"

echo
echo "============================================================"
echo " FER2013 SMIRK GEOMETRY CROSS-ATTENTION COMPLETED"
echo "============================================================"
echo "End: $(date)"
