#!/bin/bash
#SBATCH --job-name=RAFDB_V5_224
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_224_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_V5_224_%j.err
set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"
FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v5_224_sota_93.yaml"
OUTPUT_DIR="$ROOT/outputs/papers/rafdb_v5_224_sota_93"
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
echo 'V5 224x224 SOTA 93%: Native 224x224 + Dynamic Part Attention (Stage 3) + Adaptive Semantic Gate'
echo 'SAM (rho=0.02) + AdamW (5e-6 backbone); Gate Floor T_end=1.6; Semantic Decay (0.08 -> 0.01).'
echo 'Selection via val_accuracy (No-TTA). Target single-model test accuracy >= 93%.'

"$FER_PY" -u scripts/smoketest_rafdb_pipeline.py "$CONFIG"
if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo 'V5_224_SMOKE_ONLY_COMPLETE; training not started.'
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/launch_config.yaml"
if command -v git >/dev/null 2>&1; then
    git rev-parse HEAD > "$OUTPUT_DIR/launch_git_commit.txt" || true
    git diff --binary > "$OUTPUT_DIR/launch_worktree.patch" || true
fi

"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment
echo "V5_224_TRAIN_COMPLETE: $OUTPUT_DIR"
echo 'Primary report: test_metrics_tta_hflip.json. Secondary: test_metrics_no_tta.json.'

echo "======================================================================"
echo "Step: Running Validation-Tuned TTA Weight Sweep (Val -> Test)..."
echo "======================================================================"
"$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05
echo "SWEEP_TTA_COMPLETE: Results saved to $OUTPUT_DIR/tta_sweep_results.json"

echo ""
echo "======================================================================"
echo "                     FINAL ALL-IN-ONE SUMMARY                         "
echo "======================================================================"
"$FER_PY" -c '
import json, sys
from pathlib import Path
out_dir = Path("'"$OUTPUT_DIR"'")
sweep_file = out_dir / "tta_sweep_results.json"
if sweep_file.exists():
    with sweep_file.open("r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"  Validation Optimal w_orig: {d[\"val_optimal\"][\"w_orig\"]:.2f} (w_flip: {d[\"val_optimal\"][\"w_flip\"]:.2f})")
    print(f"  Validation Peak Accuracy:  {d[\"val_optimal\"][\"accuracy\"]*100:.2f}%")
    print(f"  Test Accuracy (No-TTA):    {d[\"test_no_tta\"][\"accuracy\"]*100:.2f}%")
    print(f"  Test Accuracy (Val-Tuned): {d[\"test_val_tuned\"][\"accuracy\"]*100:.2f}%")
    print(f"  Test Macro F1:             {d[\"test_val_tuned\"][\"macro_f1\"]:.4f}")
    print(f"  TTA Improvement Gain:      {d[\"test_tta_gain_pct\"]:+.2f}%")
' || true
echo "======================================================================"
echo "ALL STEPS COMPLETED SUCCESSFULLY: $OUTPUT_DIR"