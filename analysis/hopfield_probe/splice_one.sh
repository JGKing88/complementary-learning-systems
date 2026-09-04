#!/usr/bin/env bash
# One checkpoint's basin, as a Slurm array task. 20 tasks, all independent.
#
#SBATCH --job-name=splice1
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/splice1_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-19
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec
/home/jackking/.conda/envs/cls/bin/python -u \
    analysis/hopfield_probe/splice_basin.py "$SLURM_ARRAY_TASK_ID"
