#!/bin/bash
#SBATCH --job-name=FER_SMIRK_TRAIN
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SMIRK_TRAIN_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SMIRK_TRAIN_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1

# ============================================================
# FIXED HOST ENVIRONMENT & PATHS
# ============================================================

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_smirk_geometry_cross_attention.yaml"
BASELINE_CKPT="$ROOT/outputs/tf_runs/convnext_base_ms1m_arcface_baseline/checkpoints/best/ckpt-43"
CACHE_DIR="$ROOT/outputs/smirk_geometry_cross_attention/geometry_tokens"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo " FER2013 - SMIRK 3D GEOMETRY CROSS ATTENTION (TRAIN ONLY)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "Start: $(date)"
echo
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "BASELINE_CKPT=$BASELINE_CKPT"
echo "CONFIG=$CONFIG"
echo "CACHE_DIR=$CACHE_DIR"
echo "============================================================"

nvidia-smi

# ============================================================
# FILE & GEOMETRY CACHE CHECKS
# ============================================================

[ -x "$FER_PY" ] || {
    echo "[ERROR] FER TensorFlow python not found: $FER_PY"
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

for split in train val test; do
    token_file="$CACHE_DIR/${split}_smirk_vlm_geometry_tokens.npz"
    if [ ! -f "$token_file" ]; then
        echo "[ERROR] Cached 3D geometry tokens missing for split '$split': $token_file"
        echo "Please run full extraction pipeline first or ensure cache files are present."
        exit 1
    fi
    echo "[INFO] Found geometry cache: $token_file"
done

# ============================================================
# CONFIGURE NVIDIA CUDA/CUDNN LIBRARIES FOR TENSORFLOW GPU
# ============================================================

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo
echo "============================================================"
echo " STEP 0: TENSORFLOW V100 GPU FAIL-FAST CHECK"
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
# STEP 1: CROSS-ATTENTION SMOKE TRAIN (1 BATCH)
# ============================================================

echo
echo "============================================================"
echo " STEP 1: CROSS-ATTENTION SMOKE TRAIN - 1 BATCH"
echo "============================================================"

"$FER_PY" -u scripts/train_smirk_geometry_cross_attention.py \
    --config "$CONFIG" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --batch-size 4 \
    --max-train-batches 1 \
    --max-eval-batches 1 \
    --smoke-only

# ============================================================
# STEP 2: FULL CROSS-ATTENTION TRAINING & EVALUATION
# ============================================================

echo
echo "============================================================"
echo " STEP 2: FULL GEOMETRY CROSS-ATTENTION TRAINING"
echo " Baseline: ConvNeXt-MS1M ckpt-43"
echo " Batch Size: 64"
echo "============================================================"

"$FER_PY" -u scripts/train_smirk_geometry_cross_attention.py \
    --config "$CONFIG" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --batch-size 64

echo
echo "============================================================"
echo " FER2013 SMIRK GEOMETRY CROSS-ATTENTION COMPLETED"
echo "============================================================"
echo "End: $(date)"
