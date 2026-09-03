#!/bin/bash
#SBATCH --job-name=ABLATION_3_MULTI
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/ABLATION_3_MULTI_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/ABLATION_3_MULTI_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/paperfinal_v2/ablation_3_siglip2_multigranularity

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_ablation_3_siglip2_multigranularity.yaml"

echo "============================================================"
echo " Ablation Step 3: + Multi-Granularity Prototypes"
echo " Output: outputs/paperfinal_v2/ablation_3_siglip2_multigranularity"
echo "============================================================"

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

"$FER_PY" -u train.py --config "$CONFIG"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true
