#!/bin/bash
#SBATCH --job-name=FER_ANALYZE_HARD_PAIRS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_ANALYZE_HARD_PAIRS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_ANALYZE_HARD_PAIRS_%j.err

set -e

cd /home/ptbao/projects/FER2013_MGR_CNN

source /home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/activate

export PYTHONUNBUFFERED=1

echo "=========================================="
echo " JOB INFO"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

for CUDA_DIR in \
    /usr/local/cuda-11.8 \
    /usr/local/cuda-11.7 \
    /usr/local/cuda-11.6 \
    /usr/local/cuda-11.5 \
    /usr/local/cuda-11.4 \
    /usr/local/cuda-11.3 \
    /usr/local/cuda-11.2 \
    /usr/local/cuda-11.1 \
    /usr/local/cuda-11.0 \
    /usr/local/cuda
do
    if [ -d "$CUDA_DIR/lib64" ]; then
        echo "Found CUDA candidate: $CUDA_DIR"
        export CUDA_HOME="$CUDA_DIR"
        export PATH="$CUDA_DIR/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done

SITE_PACKAGES=$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)

if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for d in "$SITE_PACKAGES"/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
        fi
    done
fi

echo
echo "=========================================="
echo " NVIDIA-SMI"
echo "=========================================="
nvidia-smi

echo
echo "=========================================="
echo " TENSORFLOW GPU CHECK"
echo "=========================================="

python - <<'PY'
import sys
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)

if not gpus:
    print("ERROR: TensorFlow cannot detect GPU.")
    sys.exit(1)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Memory growth warning:", e)

print("GPU CHECK PASSED")
PY

echo
echo "=========================================="
echo " START BASELINE HARD PAIRS ANALYSIS"
echo "=========================================="

python -u scripts/analyze_baseline_hard_pairs.py \
    --config config_convnext_base_ms1m_arcface_baseline.yaml \
    --run-dir outputs/tf_runs/convnext_base_ms1m_arcface_baseline

echo
echo "=========================================="
echo " ANALYSIS COMPLETED"
echo "=========================================="
echo "End: $(date)"
