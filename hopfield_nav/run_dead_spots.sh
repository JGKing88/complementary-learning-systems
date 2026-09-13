#!/bin/bash -l
#SBATCH --job-name=dead-spots
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/dead_spots_%j.out

# analysis/nav_tri/dead_spots.py on one policy + env set. See its docstring.
#   CKPT=... SPLIT=place=ood [ENVS_FROM=...] sbatch hopfield_nav/run_dead_spots.sh
set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-ood-place}
CKPT=${CKPT:?policy checkpoint}
SPLIT=${SPLIT:-place=ood}
N_ENVS=${N_ENVS:-24}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
python -u -m analysis.nav_tri.dead_spots --ckpt "$CKPT" --split "$SPLIT" --n_envs "$N_ENVS" \
    ${ENVS_FROM:+--envs_from "$ENVS_FROM"} ${EXTRA:-}
