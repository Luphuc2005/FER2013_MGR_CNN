#!/bin/bash
#SBATCH --job-name=RAFDB_V3_LOSS_SWEEP
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V3_LOSS_SWEEP_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V3_LOSS_SWEEP_%j.err
set -euo pipefail
ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v3.yaml"
CHECKPOINT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v3/checkpoints/best_loss"
REPORT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v3/best_loss_tta_sweep_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
export MGR_TF_INTRA_OP_THREADS=12
export MGR_TF_INTER_OP_THREADS=4
export MGR_TF_DATA_NUM_PARALLEL_CALLS=12
export MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE=16
export MGR_PREFETCH_BUFFER=4
SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi
echo "Stable v3 BEST LOSS / 5 members / TTA sweep step=${STEP:-0.05} / output=$REPORT_DIR"
"$FER_PY" -u scripts/sweep_rafdb_best_loss_ensemble.py \
  --config "$CONFIG" --checkpoint-dir "$CHECKPOINT_DIR" \
  --step "${STEP:-0.05}" --output-dir "$REPORT_DIR"
