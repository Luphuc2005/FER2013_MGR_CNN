#!/bin/bash
#SBATCH --job-name=V5_VAL_MACROF1
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/V5_VAL_MACROF1_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/V5_VAL_MACROF1_%j.err
set -euo pipefail
ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml"
RUN_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate"
OUTPUT_DIR="$RUN_DIR/val_macro_f1_selection_${SLURM_JOB_ID:?Submit with sbatch}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
if env | grep -q '^MGR_'; then
    echo "Unset MGR_* overrides before this evaluation."
    exit 1
fi
SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi
echo "Target Split: VALIDATION; select Macro F1; then test only winner. HFlip 50/50 fixed."
"$FER_PY" -u scripts/select_v5_checkpoint_val_macro_f1.py --config "$CONFIG" --checkpoint-dirs "$RUN_DIR/checkpoints/best" "$RUN_DIR/checkpoints/best_loss" --protocol hflip50 --output-dir "$OUTPUT_DIR"
