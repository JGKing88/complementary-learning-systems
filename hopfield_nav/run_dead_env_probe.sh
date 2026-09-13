#!/bin/bash -l
#SBATCH --job-name=dead-env
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/dead_env_%j.out

# analysis/nav_tri/dead_env_probe.py on one checkpoint + split. See its docstring.
#   sbatch hopfield_nav/run_dead_env_probe.sh
#   CKPT=... SPLIT=recorded sbatch hopfield_nav/run_dead_env_probe.sh
set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-ood-place}
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navp2_ood_corner_s42_22629938/navigate_u1200.pt}
SPLIT=${SPLIT:-place=ood}
STORED=${STORED:-0,1}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
python -u -m analysis.nav_tri.dead_env_probe --ckpt "$CKPT" --split "$SPLIT" --stored "$STORED" ${EXTRA:-}
