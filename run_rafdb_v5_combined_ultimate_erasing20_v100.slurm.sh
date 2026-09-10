#!/bin/bash
#SBATCH --job-name=RAFDB_V5_RE20
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_RE20_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_RE20_%j.err
set -euo pipefail
ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_erasing20.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_siglip2_semantic_stable_v5_combined_ultimate_erasing20"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

if env | grep -q '^MGR_'; then
    echo 'Unexpected MGR_* overrides. Unset them before submitting this experiment.'
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
"$FER_PY" - "$CONFIG" <<'PY'
import copy
import sys
from config import load_config
base = load_config("config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml")
expected = copy.deepcopy(base)
expected["augmentation"]["random_erasing_prob"] = 0.20
expected["paths"]["output_dir"] += "_erasing20"
actual = load_config(sys.argv[1])
if actual != expected:
    raise ValueError("RE20 must differ from V5 only by erasing probability and output path.")
print("V5_RE20_CONFIG_OK: erasing 0.40 -> 0.20; original CE, monitor, SAM and TTA retained.")
PY

"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment
echo "Finished: $OUTPUT_DIR (V5 original CE; Random Erasing 0.20 ablation)."
