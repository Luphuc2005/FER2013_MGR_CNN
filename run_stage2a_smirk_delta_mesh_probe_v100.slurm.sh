#!/bin/bash
#SBATCH --job-name=FER_STAGE2A_DELTA_MESH
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_STAGE2A_DELTA_MESH_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_STAGE2A_DELTA_MESH_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/stage2a_smirk_delta_mesh_probe/cache outputs/stage2a_smirk_delta_mesh_probe/logs outputs/stage2a_smirk_delta_mesh_probe/checkpoints outputs/stage2a_smirk_delta_mesh_probe/visualizations

export PYTHONUNBUFFERED=1

# ============================================================
# FIXED HOST ENVIRONMENT & MODEL PATHS
# ============================================================

SMIRK_PY="/home/ptbao/projects/FER2013_MGR_CNN/smirk_host_env/bin/python"
FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

CONFIG="$ROOT/config_stage2a_smirk_delta_mesh_probe.yaml"

SMIRK_ROOT="/home/ptbao/projects/smirk"
SMIRK_CHECKPOINT="$SMIRK_ROOT/pretrained_models/SMIRK_em1.pt"
FLAME_MODEL="$SMIRK_ROOT/assets/FLAME2020/generic_model.pkl"

export SMIRK_ROOT
export SMIRK_CHECKPOINT
export PYTHONPATH="$ROOT:$SMIRK_ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo " STAGE 2A: SMIRK ΔMESH 3D-ONLY FER PROBE (V100 GPU)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo
echo "ROOT=$ROOT"
echo "SMIRK_PY=$SMIRK_PY"
echo "FER_PY=$FER_PY"
echo "SMIRK_ROOT=$SMIRK_ROOT"
echo "SMIRK_CHECKPOINT=$SMIRK_CHECKPOINT"
echo "CONFIG=$CONFIG"
echo "============================================================"

nvidia-smi

# ============================================================
# MANDATORY FILE & ENVIRONMENT CHECKS
# ============================================================

[ -x "$SMIRK_PY" ] || { echo "[ERROR] SMIRK host python not found: $SMIRK_PY"; exit 1; }
[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -d "$SMIRK_ROOT" ] || { echo "[ERROR] SMIRK repo not found: $SMIRK_ROOT"; exit 1; }
[ -f "$SMIRK_CHECKPOINT" ] || { echo "[ERROR] SMIRK checkpoint not found: $SMIRK_CHECKPOINT"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"

# ============================================================
# STEP 0: SMOKE TEST SMIRK & PYTORCH GPU
# ============================================================

echo
echo "============================================================"
echo " STEP 0: VERIFY SMIRK_HOST_ENV & PYTORCH V100 GPU"
echo "============================================================"

"$SMIRK_PY" - <<'PY'
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("[FAIL-FAST ERROR] PyTorch cannot access V100 GPU.")
print("GPU Device Name:", torch.cuda.get_device_name(0))
PY

# ============================================================
# STEP 1: SMOKE VERIFICATION TEST (16 TRAIN SAMPLES)
# ============================================================

if [ "$SKIP_EXTRACTION" -ne 1 ]; then
    echo
    echo "============================================================"
    echo " STEP 1: SMOKE VERIFICATION TEST - 16 SAMPLES & 7 CHECKS"
    echo "============================================================"

    "$SMIRK_PY" -u scripts/extract_stage2a_smirk_delta_mesh.py \
        --config "$CONFIG" \
        --smirk-root "$SMIRK_ROOT" \
        --checkpoint "$SMIRK_CHECKPOINT" \
        --device cuda \
        --splits train \
        --batch-size 8 \
        --smoke-only

    # ============================================================
    # STEP 2: FULL STAGE 2A DELTA MESH CACHE EXTRACTION
    # ============================================================

    echo
    echo "============================================================"
    echo " STEP 2: FULL STAGE 2A DELTA MESH CACHE (TRAIN / VAL / TEST)"
    echo "============================================================"

    "$SMIRK_PY" -u scripts/extract_stage2a_smirk_delta_mesh.py \
        --config "$CONFIG" \
        --smirk-root "$SMIRK_ROOT" \
        --checkpoint "$SMIRK_CHECKPOINT" \
        --device cuda \
        --splits train val test \
        --batch-size 128 \
        --force

    echo
    echo "============================================================"
    echo " STAGE 2A DELTA MESH CACHE COMPLETED"
    echo "============================================================"
else
    echo "[INFO] SKIP_EXTRACTION=1: Using existing Stage 2A cache."
fi

# ============================================================
# CONFIGURE NVIDIA CUDA/CUDNN LIBRARIES FOR TENSORFLOW GPU
# ============================================================

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo
echo "============================================================"
echo " TENSORFLOW V100 GPU FAIL-FAST CHECK"
echo "============================================================"

"$FER_PY" - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow Version:", tf.__version__)
print("Detected GPUs:", gpus)
if not gpus:
    raise RuntimeError("[FAIL-FAST ERROR] TensorFlow GPU NOT FOUND on V100.")
print("TENSORFLOW_GPU_VERIFICATION_SUCCESS")
PY

# ============================================================
# STEP 3: TENSORFLOW STAGE 2A GNN SMOKE TRAIN
# ============================================================

echo
echo "============================================================"
echo " STEP 3: STAGE 2A GNN SMOKE TRAIN (32 SAMPLES / 2 EPOCHS)"
echo "============================================================"

"$FER_PY" -u scripts/train_stage2a_smirk_delta_mesh_probe.py \
    --config "$CONFIG" \
    --smoke-only

# ============================================================
# STEP 4: FULL STAGE 2A GNN PROBE TRAINING
# ============================================================

echo
echo "============================================================"
echo " STEP 4: FULL STAGE 2A DELTA MESH GNN PROBE TRAINING"
echo "============================================================"

"$FER_PY" -u scripts/train_stage2a_smirk_delta_mesh_probe.py \
    --config "$CONFIG"

# ============================================================
# STEP 5: VISUALIZATION & DEBUG STATS
# ============================================================

echo
echo "============================================================"
echo " STEP 5: STAGE 2A DELTA MESH VISUALIZATION & DEBUG STATS"
echo "============================================================"

"$SMIRK_PY" -u scripts/visualize_stage2a_delta_mesh.py \
    --config "$CONFIG" \
    --smirk-root "$SMIRK_ROOT" \
    --checkpoint "$SMIRK_CHECKPOINT" \
    --split test \
    --samples-per-class 2

echo
echo "============================================================"
echo " STAGE 2A COMPLETED SUCCESSFULLY"
echo "============================================================"
echo "End: $(date)"
