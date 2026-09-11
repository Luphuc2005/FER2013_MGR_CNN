#!/bin/bash
#SBATCH --job-name=RAFDB_V9_RDROP
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V9_RDROP_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V9_RDROP_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v9_rdrop_sota_93.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_v9_rdrop_sota_93"
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
echo 'V9 + R-Drop SOTA 93%: Direct 1024-dim Fusion + Adaptive Semantic Gate + R-Drop (lambda=0.5)'
echo 'Natural split (11,043 train / 1,228 val); SAM (rho=0.02) + AdamW; backbone unfreezes at epoch 5.'
echo 'Selection via val_accuracy (No-TTA). Target single-model test accuracy >= 93%.'

"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"
if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo 'V9_RDROP_SMOKE_ONLY_COMPLETE; training not started.'
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/launch_config.yaml"
if command -v git >/dev/null 2>&1; then
    git rev-parse HEAD > "$OUTPUT_DIR/launch_git_commit.txt" || true
    git diff --binary > "$OUTPUT_DIR/launch_worktree.patch" || true
fi
"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment
echo "V9_RDROP_TRAIN_COMPLETE: $OUTPUT_DIR"
echo 'Primary report: test_metrics_tta_hflip.json. Secondary: test_metrics_no_tta.json.'

echo "======================================================================"
echo "Step: Running Validation-Tuned TTA Weight Sweep (Val -> Test)..."
echo "======================================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05
echo "SWEEP_TTA_COMPLETE: Results saved to $OUTPUT_DIR/tta_sweep_results.json"