#!/bin/bash -l
#SBATCH --job-name=evalgp
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/goal_pairs_eval_%j.out

# Enumerate the quadrant table for one saved goal-pairs checkpoint.
#   CKPT=<run_dir>/pairs_u6000.pt sbatch hopfield_nav/run_eval_goal_pairs.sh
set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nn-generalization-control}
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "$REPO"
source scripts/cls_env.sh
PYTHONUNBUFFERED=1 python -m hopfield_nav.eval_goal_pairs --ckpt "$CKPT" ${EXTRA:-}
