#!/bin/bash -l
#SBATCH --job-name=hnav-pattern
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal
#SBATCH --mem=80G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/pattern_%j.out

# Classify what an explore policy's trajectory actually IS -- random walk,
# circle, perimeter orbit, spiral, boustrophedon, or nothing at all -- against
# reference families run through the same simulator at the policy's own stride
# and turn width. See hopfield_nav/probes/motion_pattern.py.
#
#   CKPT=/path/navigate_uN.pt sbatch hopfield_nav/run_probe_pattern.sh
#
# CPU partition on purpose: the GPU QOS caps this account at 2, and those
# belong to the training wave.

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
CKPT=${CKPT:?CKPT is required}
OUT=${OUT:-}
N_ENVS=${N_ENVS:-4}
TRIALS=${TRIALS:-32}
MAX_STEPS=${MAX_STEPS:-400}
N_DIST=${N_DIST:-10}
SPLIT=${SPLIT:-val}

module load miniforge/24.3.0-0
source activate cls

cd "$REPO"
source scripts/cls_env.sh

python -u -m hopfield_nav.probes.motion_pattern \
    --ckpt "$CKPT" --n_envs "$N_ENVS" --trials "$TRIALS" \
    --max_steps "$MAX_STEPS" --n_dist "$N_DIST" --split "$SPLIT" \
    --seed "${SEED_OVERRIDE:-42}" --device cpu \
    ${OUT:+--output_json "$OUT"}
