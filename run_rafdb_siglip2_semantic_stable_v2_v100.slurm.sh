#!/bin/bash
#SBATCH --job-name=RAFDB_SEM_STABLE_V2
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SEM_STABLE_V2_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SEM_STABLE_V2_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/papers/rafdb_siglip2_semantic_stable_v2

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v2.yaml"

echo "============================================================"
echo " RAF-DB SigLIP2 Semantic-Stable v2"
echo " backbone_lr=1e-5 | lambda_sem=0.10 -> 0.20"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Output: outputs/papers/rafdb_siglip2_semantic_stable_v2"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " Validating semantic-stable config contract..."
echo "============================================================"
"$FER_PY" -u scripts/check_semantic_stable_contract.py "$CONFIG"

echo "============================================================"
echo " Running Pre-flight RAF-DB Smoke Test (1 Batch)..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"

if [ "${SMOKE_ONLY:-0}" = "1" ]; then
    echo "[INFO] SMOKE_ONLY=1; semantic-stable contract and 1-batch smoke passed. Training was not started."
    exit 0
fi

echo "============================================================"
echo " Starting RAF-DB Semantic-Stable v2 Training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

echo "============================================================"
echo " Running Post-Training TTA Weight Sweep..."
echo "============================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05

echo "============================================================"
echo " Running Top-5 Checkpoint Ensemble + TTA Evaluation..."
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" --w-orig 0.60 --w-flip 0.40

echo "============================================================"
echo " RAF-DB Semantic-Stable v2 Pipeline Completed Successfully!"
echo " End: $(date)"
echo "============================================================"
