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
# FIXED HOST ENVIRONMENT & MODEL PATHS
# ============================================================

SMIRK_PY="/home/ptbao/projects/FER2013_MGR_CNN/smirk_host_env/bin/python"
FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

CONFIG="$ROOT/config_smirk_geometry_cross_attention.yaml"

SMIRK_ROOT="/home/ptbao/projects/smirk"
SMIRK_CHECKPOINT="$SMIRK_ROOT/pretrained_models/SMIRK_em1.pt"
FLAME_MODEL="$SMIRK_ROOT/assets/FLAME2020/generic_model.pkl"
LANDMARKER_MODEL="$SMIRK_ROOT/assets/face_landmarker.task"

BASELINE_CKPT="$ROOT/outputs/tf_runs/convnext_base_ms1m_arcface_baseline/checkpoints/best/ckpt-43"

export SMIRK_ROOT
export SMIRK_CHECKPOINT
export PYTHONPATH="$SMIRK_ROOT:$ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo " FER2013 - SMIRK 3D GEOMETRY CROSS ATTENTION (HOST ENV)"
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
echo "FLAME_MODEL=$FLAME_MODEL"
echo "LANDMARKER_MODEL=$LANDMARKER_MODEL"
echo "BASELINE_CKPT=$BASELINE_CKPT"
echo "CONFIG=$CONFIG"
echo "============================================================"

nvidia-smi

# ============================================================
# MANDATORY FILE & ENVIRONMENT CHECKS
# ============================================================

[ -x "$SMIRK_PY" ] || {
    echo "[ERROR] SMIRK host python not found: $SMIRK_PY"
    exit 1
}

[ -x "$FER_PY" ] || {
    echo "[ERROR] FER TensorFlow python not found: $FER_PY"
    exit 1
}

[ -d "$SMIRK_ROOT" ] || {
    echo "[ERROR] SMIRK repository not found: $SMIRK_ROOT"
    exit 1
}

[ -f "$SMIRK_CHECKPOINT" ] || {
    echo "[ERROR] SMIRK checkpoint not found: $SMIRK_CHECKPOINT"
    exit 1
}

[ -f "$FLAME_MODEL" ] || {
    echo "[ERROR] FLAME model file not found: $FLAME_MODEL"
    exit 1
}

[ -f "$LANDMARKER_MODEL" ] || {
    echo "[ERROR] Face landmarker task file not found: $LANDMARKER_MODEL"
    exit 1
}

[ -f "$CONFIG" ] || {
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
}

[ -f "$BASELINE_CKPT.index" ] || {
    echo "[ERROR] Baseline ckpt-43 index file not found: $BASELINE_CKPT"
    exit 1
}

# ============================================================
# STEP 0: FAIL-FAST CHECK ON COMPUTE NODE V100
# ============================================================

echo
echo "============================================================"
echo " STEP 0: CHECK SMIRK_HOST_ENV / PYTORCH / GPU / DEPENDENCIES"
echo "============================================================"

"$SMIRK_PY" - <<'PY'
import sys
import torch

print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("[FAIL-FAST ERROR] PyTorch cannot access V100 GPU.")

print("GPU Device Name:", torch.cuda.get_device_name(0))

try:
    import pytorch3d
    print("PyTorch3D import: OK")
except ImportError as e:
    raise ImportError(f"[FAIL-FAST ERROR] PyTorch3D import failed: {e}")

try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ImportError as e:
    raise ImportError(f"[FAIL-FAST ERROR] OpenCV import failed: {e}")

try:
    import mediapipe
    print("MediaPipe version:", mediapipe.__version__)
except ImportError as e:
    raise ImportError(f"[FAIL-FAST ERROR] MediaPipe import failed: {e}")

print("SMIRK_HOST_ENV_VERIFICATION_SUCCESS")
PY

# ============================================================
# STEP 1: SMOKE TEST SMIRK 3D EXTRACTION (16 TRAIN IMAGES)
# ============================================================

echo
echo "============================================================"
echo " STEP 1: SMOKE TEST SMIRK 3D EXTRACTION - 16 TRAIN IMAGES"
echo "============================================================"

"$SMIRK_PY" -u scripts/extract_smirk_vlm_geometry_tokens.py \
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
# STEP 2: FULL SMIRK 3D GEOMETRY CACHE (TRAIN / VAL / TEST)
# ============================================================

echo
echo "============================================================"
echo " STEP 2: FULL 3D GEOMETRY CACHE - TRAIN / VAL / TEST"
echo "============================================================"

"$SMIRK_PY" -u scripts/extract_smirk_vlm_geometry_tokens.py \
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
