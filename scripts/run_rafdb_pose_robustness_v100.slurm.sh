#!/bin/bash
#SBATCH --job-name=RAFDB_POSE_EVAL
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_POSE_EVAL_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_POSE_EVAL_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
POSE_PY="${POSE_PY:-$ROOT/pose_eval_env/bin/python}"
CONFIG="${CONFIG:-$ROOT/config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion_v2.yaml}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/rafdb}"
CHECKPOINT_DIR="$ROOT/outputs/papers/rafdb_adaptive_siglip2_confusion_v2_v4/checkpoints/best"
RUN_DIR="$ROOT/outputs/pose_robustness/rafdb_v2_v4_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=12
export NVIDIA_LIB="$ROOT/fer2013_env/lib/python3.9/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

[[ -x "$FER_PY" && -x "$POSE_PY" ]] || {
    echo 'Missing environment. Run: bash scripts/setup_rafdb_pose_eval.sh'
    exit 1
}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG"; exit 1; }
[[ -f "$CHECKPOINT_DIR/ckpt-35.index" ]] || { echo 'Missing primary ckpt-35'; exit 1; }
mkdir -p "$ROOT/logs" "$RUN_DIR"
echo "Job=${SLURM_JOB_ID:-standalone} CPUs=${SLURM_CPUS_PER_TASK:-16} Node=$(hostname)"
echo "Config=$CONFIG Checkpoints=$CHECKPOINT_DIR Output=$RUN_DIR"
nvidia-smi

POSE_ARGS=()
[[ -z "${TEST_CSV:-}" ]] || POSE_ARGS+=(--test-csv "$TEST_CSV")
CUDA_VISIBLE_DEVICES=-1 "$POSE_PY" -u scripts/estimate_rafdb_head_pose.py \
  --data-root "$DATA_ROOT" --output-dir "$RUN_DIR/pose" --workers 8 \
  --label-space "${LABEL_SPACE:-auto}" "${POSE_ARGS[@]}"

EVAL_ARGS=()
# Keep all-best evaluation requested earlier; primary remains ckpt-35.
[[ "${ALL_BEST:-1}" != 1 ]] || EVAL_ARGS+=(--all-best)
[[ "${SMOKE_ONLY:-0}" != 1 ]] || EVAL_ARGS+=(--smoke-only)
[[ "${NO_TTA:-0}" != 1 ]] || EVAL_ARGS+=(--no-tta)
"$FER_PY" -u scripts/evaluate_rafdb_pose.py \
  --config "$CONFIG" --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint ckpt-35 \
  --pose-dir "$RUN_DIR/pose" --output-dir "$RUN_DIR/evaluation" \
  --batch-size 32 "${EVAL_ARGS[@]}"
echo "Finished. Results: $RUN_DIR"
