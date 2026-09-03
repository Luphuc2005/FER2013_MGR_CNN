#!/bin/bash
# Submit all 6 paperfinal ablation jobs to Slurm in parallel across available GPU nodes

echo "============================================================"
echo " SUBMITTING ALL 6 ABLATION JOBS IN PARALLEL TO SLURM QUEUE"
echo "============================================================"

sbatch run_ablation_1_baseline_v100.slurm.sh
sbatch run_ablation_2_siglip2_single_proto_v100.slurm.sh
sbatch run_ablation_3_siglip2_multigranularity_v100.slurm.sh
sbatch run_ablation_4_siglip2_adaptive_weighting_v100.slurm.sh
sbatch run_ablation_5_siglip2_confusion_aware_v100.slurm.sh
sbatch run_ablation_6_full_model_top5_ensemble_v100.slurm.sh

echo "============================================================"
echo " All 6 jobs submitted! Check status with: squeue -u \$USER"
echo "============================================================"
