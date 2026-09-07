#!/bin/bash
#SBATCH --job-name=RAFDB_V4_COMBINED
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V4_COMBINED_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V4_COMBINED_%j.err
set -euo pipefail
ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v4_combined.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v4_combined"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
# Avoid inherited MGR overrides silently changing this experiment.
if env | grep -q '^MGR_'; then
    echo 'Unexpected MGR_* overrides. Unset them before submitting this fixed-config experiment.'
    exit 1
fi
SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Output already nonempty: $OUTPUT_DIR. Refusing overwrite/resume."
    exit 1
fi
nvidia-smi
echo "Config=$CONFIG Output=$OUTPUT_DIR Job=${SLURM_JOB_ID:-standalone}"
"$FER_PY" -u scripts/check_rafdb_v4_combined.py --config "$CONFIG" --smoke
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then
    echo 'Smoke only; no training started.'
    exit 0
fi
# Same trainer, SAM, schedules, validation checkpoint selection and original test.
# Trainer reports no-TTA and fixed-config 0.5/0.5 HFlip TTA; no test-weight sweep.
"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment
echo "Finished: $OUTPUT_DIR (validation-selected checkpoint; original RAF test)."
