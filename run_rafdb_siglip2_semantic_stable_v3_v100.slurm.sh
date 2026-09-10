#!/bin/bash
#SBATCH --job-name=RAFDB_SEM_STABLE_V3
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SEM_STABLE_V3_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_SEM_STABLE_V3_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v3.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v3"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTPUT_DIR"

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

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow Python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

echo "============================================================"
echo " RAF-DB SigLIP2 Semantic-Stable v3"
echo " backbone_lr=1e-5 | lambda_sem=0.10 -> 0.15"
echo " lambda_hard=0.10 | hard_margin=0.30"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

nvidia-smi

echo "============================================================"
echo " Validating semantic-stable v3 config contract..."
echo "============================================================"
"$FER_PY" -u scripts/check_semantic_stable_v3_contract.py "$CONFIG"

echo "============================================================"
echo " Running pre-flight RAF-DB smoke test (1 batch)..."
echo "============================================================"
"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"

if [ "${SMOKE_ONLY:-0}" = "1" ]; then
    echo "[INFO] SMOKE_ONLY=1; contract and 1-batch smoke passed. Training was not started."
    exit 0
fi

echo "============================================================"
echo " Starting RAF-DB Semantic-Stable v3 training..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG"

echo "============================================================"
echo " Running post-training TTA weight sweep..."
echo "============================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05

echo "============================================================"
echo " Running top-5 accuracy checkpoint ensemble + TTA..."
echo "============================================================"
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py \
  --config "$CONFIG" \
  --checkpoint-dir "$OUTPUT_DIR/checkpoints/best" \
  --split test \
  --w-orig 0.60 \
  --w-flip 0.40 \
  --output "$OUTPUT_DIR/best_top5_tta_ensemble_report.json"

echo "============================================================"
echo " RAF-DB Semantic-Stable v3 pipeline completed successfully"
echo " End: $(date)"
echo "============================================================"
