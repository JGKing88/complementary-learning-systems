#!/bin/bash -l
#SBATCH --job-name=hnav-motion
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal
#SBATCH --mem=80G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/motion_%j.out

# Locate a trained explore policy on the memoryless-coverage table: measure its
# stride and turn statistics on the real evaluation path, then simulate a walker
# with those same statistics. The gap is what memory is buying.
#
#   CKPT=/path/navigate_uN.pt sbatch hopfield_nav/run_probe_motion.sh
#
# CPU partition on purpose: the GPU QOS caps this account at 2, and those
# belong to the training wave. A few thousand batched steps run fine here.

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
CKPT=${CKPT:?CKPT is required}
OUT=${OUT:-}
N_ENVS=${N_ENVS:-4}
TRIALS=${TRIALS:-32}
MAX_STEPS=${MAX_STEPS:-400}
N_DIST=${N_DIST:-10}

module load miniforge/24.3.0-0
source activate cls

cd "$REPO"
source scripts/cls_env.sh

python -u -m hopfield_nav.probes.policy_motion \
    --ckpt "$CKPT" --n_envs "$N_ENVS" --trials "$TRIALS" \
    --max_steps "$MAX_STEPS" --n_dist "$N_DIST" --device cpu \
    --split "${SPLIT:-val}" \
    ${OUT:+--output_json "$OUT"}
