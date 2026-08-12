#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-config_kaggle_teacher_2gpu.yaml}"
LOG_DIR="/kaggle/working/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MGR_GPU_IDS="${MGR_GPU_IDS:-0,1}"
export MGR_REQUIRE_TWO_GPUS=1
export MGR_MIN_GPUS=2

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-16}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-4}"
export MGR_TF_INTRA_OP_THREADS="${MGR_TF_INTRA_OP_THREADS:-16}"
export MGR_TF_INTER_OP_THREADS="${MGR_TF_INTER_OP_THREADS:-4}"
export MGR_TF_DATA_NUM_PARALLEL_CALLS="${MGR_TF_DATA_NUM_PARALLEL_CALLS:-4}"
export MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE="${MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE:-4}"
export MGR_TF_DATA_DETERMINISTIC="${MGR_TF_DATA_DETERMINISTIC:-0}"
export MGR_PREFETCH_BUFFER="${MGR_PREFETCH_BUFFER:-1}"
export MGR_SHUFFLE_BUFFER="${MGR_SHUFFLE_BUFFER:-512}"
export MGR_EVAL_TTA_HFLIP="${MGR_EVAL_TTA_HFLIP:-1}"
export MGR_TRAIN_VAL_TTA_HFLIP="${MGR_TRAIN_VAL_TTA_HFLIP:-0}"
export MGR_PREDECODE_PIXELS="${MGR_PREDECODE_PIXELS:-1}"
export MGR_PRELOAD_MASKS="${MGR_PRELOAD_MASKS:-1}"
export MGR_ALLOW_MISSING_MASKS="${MGR_ALLOW_MISSING_MASKS:-0}"
unset MGR_MAX_TRAIN_SAMPLES
unset MGR_MAX_VAL_SAMPLES
unset MGR_MAX_TEST_SAMPLES

PRIMARY_BATCH="${MGR_PRIMARY_BATCH_SIZE_PER_GPU:-64}"
FALLBACK_BATCH="${MGR_FALLBACK_BATCH_SIZE_PER_GPU:-32}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_kaggle_2gpu_${STAMP}.log"

echo "[INFO] Project: ${ROOT_DIR}"
echo "[INFO] Python: $(${PYTHON_BIN} --version)"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Checking TensorFlow environment"
"${PYTHON_BIN}" check_environment.py | tee "${LOG_DIR}/check_environment_kaggle_2gpu_${STAMP}.log"

run_training() {
  local batch_size="$1"
  export MGR_BATCH_SIZE_PER_GPU="${batch_size}"
  echo "[INFO] Starting Kaggle 2-GPU training with batch_size_per_gpu=${MGR_BATCH_SIZE_PER_GPU}"
  "${PYTHON_BIN}" train.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${TRAIN_LOG}"
}

if run_training "${PRIMARY_BATCH}"; then
  echo "[INFO] Training finished with batch_size_per_gpu=${PRIMARY_BATCH}"
  exit 0
fi

if [ "${PRIMARY_BATCH}" != "${FALLBACK_BATCH}" ] && grep -Eiq "ResourceExhaustedError|out of memory|OOM|Killed" "${TRAIN_LOG}"; then
  echo "[WARN] GPU/RAM memory error detected. Retrying with batch_size_per_gpu=${FALLBACK_BATCH}" | tee -a "${TRAIN_LOG}"
  run_training "${FALLBACK_BATCH}"
  exit 0
fi

echo "[ERROR] Training failed. See ${TRAIN_LOG}" >&2
exit 1
