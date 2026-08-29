#!/usr/bin/env bash
#SBATCH --job-name=fer_smirk_only
#SBATCH --output=logs/smirk_only_%j.out
#SBATCH --error=logs/smirk_only_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config_smirk_only.yaml}"
SMIRK_ROOT="${SMIRK_ROOT:-external/smirk}"
SMIRK_CHECKPOINT="${SMIRK_CHECKPOINT:-${SMIRK_ROOT}/pretrained_models/SMIRK_em1.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SMIRK_ROOT
export SMIRK_CHECKPOINT
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
export TF_DISABLE_XLA=1
export TF_DISABLE_XLA_COMPILATION=1

mkdir -p logs outputs/smirk_only

echo "[INFO] FER2013 SMIRK-only on V100"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] SMIRK_ROOT=${SMIRK_ROOT}"
echo "[INFO] SMIRK_CHECKPOINT=${SMIRK_CHECKPOINT}"

if [[ ! -d "${SMIRK_ROOT}" ]]; then
  echo "[ERROR] Missing SMIRK_ROOT=${SMIRK_ROOT}. Clone https://github.com/georgeretsi/smirk and run quick_install.sh first." >&2
  exit 1
fi
if [[ ! -f "${SMIRK_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing official SMIRK checkpoint=${SMIRK_CHECKPOINT}. Run SMIRK quick_install.sh first." >&2
  exit 1
fi

echo "[INFO] Smoke extract: shape/NaN check on 32 samples per split"
"${PYTHON_BIN}" scripts/extract_smirk_features.py \
  --config "${CONFIG}" \
  --smirk-root "${SMIRK_ROOT}" \
  --checkpoint "${SMIRK_CHECKPOINT}" \
  --device cuda \
  --splits train val test \
  --batch-size 16 \
  --max-samples-per-split 32 \
  --output-subdir features_smoke \
  --force

echo "[INFO] Smoke train: one batch"
"${PYTHON_BIN}" scripts/train_smirk_classifier.py \
  --config "${CONFIG}" \
  --feature-dir outputs/smirk_only/features_smoke \
  --run-subdir smoke \
  --epochs 1 \
  --batch-size 16 \
  --max-train-batches 1 \
  --max-eval-batches 1

echo "[INFO] Full SMIRK feature extraction"
"${PYTHON_BIN}" scripts/extract_smirk_features.py \
  --config "${CONFIG}" \
  --smirk-root "${SMIRK_ROOT}" \
  --checkpoint "${SMIRK_CHECKPOINT}" \
  --device cuda \
  --splits train val test \
  --batch-size 128 \
  --force

echo "[INFO] Train TensorFlow SMIRK-only classifier"
"${PYTHON_BIN}" scripts/train_smirk_classifier.py \
  --config "${CONFIG}"
