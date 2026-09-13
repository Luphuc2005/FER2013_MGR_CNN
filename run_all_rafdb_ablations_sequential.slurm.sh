#!/bin/bash
#SBATCH --job-name=RAFDB_ABL_ALL
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_ABL_ALL_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_ABL_ALL_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/ablation/rafdb

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

echo "============================================================"
echo " RAF-DB 5-STAGE ABLATION STUDY (SEQUENTIAL EXECUTION)"
echo " Stages: 1 (Baseline) -> 2 (Single) -> 3 (Multi) -> 4 (Adaptive) -> 5 (Full)"
echo " Evaluation: TTA Sweep + 15 Checkpoints + Combinatorial Ensemble per Stage"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_CUDNN_USE_AUTOTUNE=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

STAGES=(
  "config_rafdb_ablation_1_baseline.yaml"
  "config_rafdb_ablation_2_siglip2_single_proto.yaml"
  "config_rafdb_ablation_3_siglip2_multigranularity.yaml"
  "config_rafdb_ablation_4_siglip2_adaptive_weighting.yaml"
  "config_rafdb_ablation_5_full_model.yaml"
)

for CFG in "${STAGES[@]}"; do
  echo "============================================================"
  echo " [$(date)] STARTING STAGE: $CFG"
  echo "============================================================"
  "$FER_PY" -u train.py --config "$ROOT/$CFG"
  echo " [$(date)] Running 15-Checkpoint TTA Sweep & Ensemble..."
  "$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$ROOT/$CFG" || true
  echo "============================================================"
  echo " [$(date)] FINISHED STAGE: $CFG"
  echo "============================================================"
done

echo "============================================================"
echo " ALL 5 RAF-DB ABLATION RUNS FINISHED SUCCESSFULLY!"
echo " Results stored in: outputs/ablation/rafdb/"
echo " End: $(date)"
echo "============================================================"
