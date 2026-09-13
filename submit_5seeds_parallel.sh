#!/bin/bash
# Submit all 5 requested seeds (0, 1, 43, 123, 3047) to Slurm in parallel

set -euo pipefail

echo "============================================================"
echo " Submitting all 5 Seeds (0, 1, 43, 123, 3047) to Slurm..."
echo " Model: ConvNeXt-Base MS1M SigLIP 2 Confusion (FER2013)"
echo "============================================================"

JOB0=$(sbatch run_siglip2_confusion_seed0_v100.slurm.sh | awk '{print $4}')
echo " Seed 0    submitted: Job ID $JOB0"

JOB1=$(sbatch run_siglip2_confusion_seed1_v100.slurm.sh | awk '{print $4}')
echo " Seed 1    submitted: Job ID $JOB1"

JOB43=$(sbatch run_siglip2_confusion_seed43_v100.slurm.sh | awk '{print $4}')
echo " Seed 43   submitted: Job ID $JOB43"

JOB123=$(sbatch run_siglip2_confusion_seed123_v100.slurm.sh | awk '{print $4}')
echo " Seed 123  submitted: Job ID $JOB123"

JOB3047=$(sbatch run_siglip2_confusion_seed3047_v100.slurm.sh | awk '{print $4}')
echo " Seed 3047 submitted: Job ID $JOB3047"

echo "============================================================"
echo " All 5 seeds queued!"
echo " Check status with: squeue -u \$USER"
echo "============================================================"
