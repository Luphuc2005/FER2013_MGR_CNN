#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${ROOT_DIR}/logs"
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
export MGR_PREDECODE_PIXELS="${MGR_PREDECODE_PIXELS:-0}"
export MGR_PRELOAD_MASKS="${MGR_PRELOAD_MASKS:-0}"
export MGR_CACHE_DATA="${MGR_CACHE_DATA:-0}"
export MGR_USE_NUMACTL="${MGR_USE_NUMACTL:-0}"
unset MGR_MAX_TRAIN_SAMPLES
unset MGR_MAX_VAL_SAMPLES
unset MGR_MAX_TEST_SAMPLES

PRIMARY_BATCH="${MGR_PRIMARY_BATCH_SIZE_PER_GPU:-2}"
FALLBACK_BATCH="${MGR_FALLBACK_BATCH_SIZE_PER_GPU:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_${STAMP}.log"

echo "[INFO] Project: ${ROOT_DIR}"
echo "[INFO] Python: $(${PYTHON_BIN} --version)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] batch_size_per_gpu primary=${PRIMARY_BATCH} fallback=${FALLBACK_BATCH}"
echo "[INFO] tf.data parallel=${MGR_TF_DATA_NUM_PARALLEL_CALLS} prefetch=${MGR_PREFETCH_BUFFER} shuffle=${MGR_SHUFFLE_BUFFER}"
echo "[INFO] Checking TensorFlow environment"
"${PYTHON_BIN}" check_environment.py | tee "${LOG_DIR}/check_environment_${STAMP}.log"

run_training() {
  local batch_size="$1"
  export MGR_BATCH_SIZE_PER_GPU="${batch_size}"
  echo "[INFO] Starting training with batch_size_per_gpu=${MGR_BATCH_SIZE_PER_GPU}"
  if [ "${MGR_USE_NUMACTL}" = "1" ] && command -v numactl >/dev/null 2>&1; then
    echo "[INFO] Using numactl --interleave=all for dual CPU socket memory balancing"
    numactl --interleave=all "${PYTHON_BIN}" train.py --config config.yaml --resume 2>&1 | tee -a "${TRAIN_LOG}"
  else
    "${PYTHON_BIN}" train.py --config config.yaml --resume 2>&1 | tee -a "${TRAIN_LOG}"
  fi
}

if run_training "${PRIMARY_BATCH}"; then
  echo "[INFO] Training finished with batch_size_per_gpu=${PRIMARY_BATCH}"
  exit 0
fi

if [ "${PRIMARY_BATCH}" != "${FALLBACK_BATCH}" ] && grep -Eiq "ResourceExhaustedError|out of memory|OOM|Killed" "${TRAIN_LOG}"; then
  echo "[WARN] GPU memory error detected. Retrying with batch_size_per_gpu=${FALLBACK_BATCH}" | tee -a "${TRAIN_LOG}"
  run_training "${FALLBACK_BATCH}"
  exit 0
fi

echo "[ERROR] Training failed. See ${TRAIN_LOG}" >&2
exit 1
