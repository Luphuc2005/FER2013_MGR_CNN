#!/usr/bin/env bash
#SBATCH --job-name=fer_smirk_geom_xattn
#SBATCH --output=logs/smirk_geom_xattn_%j.out
#SBATCH --error=logs/smirk_geom_xattn_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config_smirk_geometry_cross_attention.yaml}"
SMIRK_ROOT="${SMIRK_ROOT:-external/smirk}"
SMIRK_CHECKPOINT="${SMIRK_CHECKPOINT:-${SMIRK_ROOT}/pretrained_models/SMIRK_em1.pt}"
BASELINE_CKPT="${BASELINE_CKPT:-outputs/tf_runs/convnext_base_ms1m_arcface_baseline/checkpoints/best}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SMIRK_ROOT
export SMIRK_CHECKPOINT
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
export TF_DISABLE_XLA=1
export TF_DISABLE_XLA_COMPILATION=1

mkdir -p logs outputs/smirk_geometry_cross_attention

echo "[INFO] FER2013 SMIRK geometry cross-attention on V100"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] SMIRK_ROOT=${SMIRK_ROOT}"
echo "[INFO] SMIRK_CHECKPOINT=${SMIRK_CHECKPOINT}"
echo "[INFO] BASELINE_CKPT=${BASELINE_CKPT}"

if [[ ! -d "${SMIRK_ROOT}" ]]; then
  echo "[ERROR] Missing SMIRK_ROOT=${SMIRK_ROOT}. Clone https://github.com/georgeretsi/smirk and run quick_install.sh first." >&2
  exit 1
fi
if [[ ! -f "${SMIRK_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing official SMIRK checkpoint=${SMIRK_CHECKPOINT}." >&2
  exit 1
fi

echo "[INFO] Smoke geometry cache: 16 samples per split"
"${PYTHON_BIN}" scripts/extract_smirk_vlm_geometry_tokens.py \
  --config "${CONFIG}" \
  --smirk-root "${SMIRK_ROOT}" \
  --smirk-checkpoint "${SMIRK_CHECKPOINT}" \
  --device cuda \
  --splits train val test \
  --batch-size 8 \
  --max-samples-per-split 16 \
  --force \
  --save-preview

echo "[INFO] Full geometry cache"
"${PYTHON_BIN}" scripts/extract_smirk_vlm_geometry_tokens.py \
  --config "${CONFIG}" \
  --smirk-root "${SMIRK_ROOT}" \
  --smirk-checkpoint "${SMIRK_CHECKPOINT}" \
  --device cuda \
  --splits train val test \
  --batch-size 64 \
  --force

echo "[INFO] Smoke train: one batch, no test checkpoint selection"
"${PYTHON_BIN}" scripts/train_smirk_geometry_cross_attention.py \
  --config "${CONFIG}" \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --batch-size 4 \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --smoke-only

echo "[INFO] Full train"
"${PYTHON_BIN}" scripts/train_smirk_geometry_cross_attention.py \
  --config "${CONFIG}" \
  --baseline-checkpoint "${BASELINE_CKPT}"
