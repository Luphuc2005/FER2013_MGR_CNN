#!/bin/bash
# Helper script to submit all 6 seed jobs to SLURM cluster

echo "Submitting 6 Seed Jobs to SLURM..."

sbatch run_siglip2_confusion_seed42_v100.slurm.sh
sbatch run_siglip2_confusion_seed123_v100.slurm.sh
sbatch run_siglip2_confusion_seed0_v100.slurm.sh
sbatch run_siglip2_confusion_seed3407_v100.slurm.sh
sbatch run_siglip2_confusion_seed2024_v100.slurm.sh
sbatch run_siglip2_confusion_seed777_v100.slurm.sh

echo "All 6 jobs submitted! Check queue with: squeue -u \$USER"
