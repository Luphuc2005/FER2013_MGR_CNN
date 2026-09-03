#!/bin/bash
#SBATCH --job-name=PAPERFINAL_ALL_ABLATIONS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/PAPERFINAL_ALL_ABLATIONS_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/PAPERFINAL_ALL_ABLATIONS_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/paperfinal

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo " STARTING COMPLETE PAPER FINAL ABLATION STUDY PIPELINE"
echo "============================================================"

# --- Step 1: ConvNeXt-B MS1M Baseline ---
echo -e "\n[STEP 1/6] Running ConvNeXt-B MS1M Baseline..."
"$FER_PY" -u train.py --config "config_ablation_1_baseline.yaml"
"$FER_PY" -u sweep_tta_weights.py --config "config_ablation_1_baseline.yaml" --step 0.05 || true

# --- Step 2: + SigLIP2 Single Prototype ---
echo -e "\n[STEP 2/6] Running + SigLIP2 Single Prototype..."
"$FER_PY" -u train.py --config "config_ablation_2_siglip2_single_proto.yaml"
"$FER_PY" -u sweep_tta_weights.py --config "config_ablation_2_siglip2_single_proto.yaml" --step 0.05 || true

# --- Step 3: + Multi-Granularity Prototypes ---
echo -e "\n[STEP 3/6] Running + Multi-Granularity Prototypes..."
"$FER_PY" -u train.py --config "config_ablation_3_siglip2_multigranularity.yaml"
"$FER_PY" -u sweep_tta_weights.py --config "config_ablation_3_siglip2_multigranularity.yaml" --step 0.05 || true

# --- Step 4: + Adaptive Weighting ---
echo -e "\n[STEP 4/6] Running + Adaptive Weighting..."
"$FER_PY" -u train.py --config "config_ablation_4_siglip2_adaptive_weighting.yaml"
"$FER_PY" -u sweep_tta_weights.py --config "config_ablation_4_siglip2_adaptive_weighting.yaml" --step 0.05 || true

# --- Step 5: + Confusion-Aware Separation ---
echo -e "\n[STEP 5/6] Running + Confusion-Aware Separation..."
"$FER_PY" -u train.py --config "config_ablation_5_siglip2_confusion_aware.yaml"
"$FER_PY" -u sweep_tta_weights.py --config "config_ablation_5_siglip2_confusion_aware.yaml" --step 0.05 || true

# --- Step 6: Full Model (Top-5 Checkpoint Ensemble) ---
echo -e "\n[STEP 6/6] Running Full Model (Top-5 Ensemble + TTA)..."
"$FER_PY" -u train.py --config "config_ablation_6_full_model_top5_ensemble.yaml"
"$FER_PY" -u sweep_tta_weights.py --config "config_ablation_6_full_model_top5_ensemble.yaml" --step 0.05 || true
"$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "config_ablation_6_full_model_top5_ensemble.yaml" || true

echo "============================================================"
echo " ALL PAPER FINAL ABLATION EXPERIMENTS COMPLETED!"
echo "============================================================"
