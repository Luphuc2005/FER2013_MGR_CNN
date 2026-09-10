#!/bin/bash
#SBATCH --job-name=RAFDB_V8_SOTA
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V8_SOTA_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V8_SOTA_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v8_sota_arcmargin_balanced_semdecay.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_v8_sota_arcmargin_balanced_semdecay"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

[[ -x "$FER_PY" ]] || { echo "Missing Python: $FER_PY"; exit 1; }
if env | grep -q '^MGR_'; then
    echo 'Unset MGR_* overrides before submitting this experiment.'
    exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Output already nonempty: $OUTPUT_DIR. Refusing overwrite/resume."
    exit 1
fi
SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi
echo "Config=$CONFIG Output=$OUTPUT_DIR Job=${SLURM_JOB_ID:-standalone}"
echo "======================================================================"
echo "  RAF-DB V8 SOTA CANDIDATE: V5 BASELINE + ARCMARGIN + OVERSAMPLING"
echo "======================================================================"
echo "Step 1: Static Pre-flight Checks (Config, Splits, Pretrained, Schedules)..."
"$FER_PY" -u scripts/check_rafdb_v8_sota.py --config "$CONFIG"

echo "Step 2: Pipeline Smoke Test (1-batch end-to-end)..."
"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"

echo "Step 3: Runtime Pre-flight Checks (TF ArcMargin, Direct Fusion, Gate, Gradient Contract)..."
"$FER_PY" -u scripts/check_rafdb_v8_sota.py --config "$CONFIG" --runtime

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo "V8_SMOKE_ONLY_COMPLETE: all pre-flight and smoke checks passed; full training skipped."
    exit 0
fi

echo "Step 4: Launching Full V8 Training..."
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/launch_config.yaml"
if command -v git >/dev/null 2>&1; then
    git rev-parse HEAD > "$OUTPUT_DIR/launch_git_commit.txt" || true
    git diff --binary > "$OUTPUT_DIR/launch_worktree.patch" || true
fi

"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment

echo "======================================================================"
echo "V8_TRAIN_COMPLETE: $OUTPUT_DIR"
echo "Primary report: test_metrics_no_tta.json (Direct FER No-TTA)."
echo "Secondary report: test_metrics_tta_hflip.json (TTA HFlip 50/50)."
echo "======================================================================"
