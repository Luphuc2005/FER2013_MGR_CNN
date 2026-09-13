#!/bin/bash
# Submit all 5 RAF-DB Ablation jobs to Slurm queue in parallel

set -euo pipefail

echo "============================================================"
echo " Submitting all 5 RAF-DB Ablation Stages to Slurm..."
echo " Output Directory: outputs/ablation/rafdb/"
echo "============================================================"

JOB1=$(sbatch run_rafdb_ablation_1_baseline_v100.slurm.sh | awk '{print $4}')
echo " Stage 1 (Baseline)                  submitted: Job ID $JOB1"

JOB2=$(sbatch run_rafdb_ablation_2_siglip2_single_proto_v100.slurm.sh | awk '{print $4}')
echo " Stage 2 (Single Prototype)          submitted: Job ID $JOB2"

JOB3=$(sbatch run_rafdb_ablation_3_siglip2_multigranularity_v100.slurm.sh | awk '{print $4}')
echo " Stage 3 (Multi-Granularity)         submitted: Job ID $JOB3"

JOB4=$(sbatch run_rafdb_ablation_4_siglip2_adaptive_weighting_v100.slurm.sh | awk '{print $4}')
echo " Stage 4 (Adaptive Weighting & Gate) submitted: Job ID $JOB4"

JOB5=$(sbatch run_rafdb_ablation_5_full_model_v100.slurm.sh | awk '{print $4}')
echo " Stage 5 (Full AMGSA-FER Model)      submitted: Job ID $JOB5"

echo "============================================================"
echo " All 5 ablation jobs queued!"
echo " Check status with: squeue -u \$USER"
echo "============================================================"
