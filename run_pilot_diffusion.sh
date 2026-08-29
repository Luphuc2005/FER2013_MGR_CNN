#!/bin/bash
#SBATCH --job-name=FER_PILOT_DIFFUSION
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_PILOT_DIFFUSION_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_PILOT_DIFFUSION_%j.err

set -e

cd /home/ptbao/projects/FER2013_MGR_CNN

source /home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/activate

export PYTHONUNBUFFERED=1

echo "=========================================="
echo " JOB INFO: PILOT TARGETED DIFFUSION"
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
echo " STEP 1: SELECT TRAIN SOURCES (50% Med, 30% Hard, 20% Clean)"
echo "=========================================="

python -u scripts/select_train_sources.py \
    --config config_convnext_base_ms1m_arcface_baseline.yaml \
    --run-dir outputs/tf_runs/convnext_base_ms1m_arcface_baseline \
    --output-json data/synthetic_diffusion/pilot_sources.json

echo
echo "=========================================="
echo " STEP 2: GENERATE & TRIPLE-GATE FILTER PILOT SAMPLES (750 ACCEPTED)"
echo "=========================================="

python -u scripts/generate_and_filter_pilot_diffusion.py \
    --config config_convnext_base_ms1m_arcface_baseline.yaml \
    --run-dir outputs/tf_runs/convnext_base_ms1m_arcface_baseline \
    --sources-json data/synthetic_diffusion/pilot_sources.json \
    --output-dir data/synthetic_diffusion/pilot_images \
    --metadata-json data/synthetic_diffusion/pilot_metadata.json

echo
echo "=========================================="
echo " STEP 3: TRAIN MODEL WITH PILOT TARGETED DIFFUSION"
echo "=========================================="

python -u train.py \
    --config config_convnext_base_ms1m_arcface_pilot_diffusion.yaml

echo
echo "=========================================="
echo " STEP 4: EVALUATE ON VALIDATION SET (NO-TTA)"
echo "=========================================="

python -u scripts/analyze_baseline_hard_pairs.py \
    --config config_convnext_base_ms1m_arcface_pilot_diffusion.yaml \
    --run-dir outputs/tf_runs/convnext_base_ms1m_arcface_pilot_diffusion

echo
echo "=========================================="
echo " PILOT PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "End: $(date)"
