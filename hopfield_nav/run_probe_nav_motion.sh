#!/bin/bash -l
#SBATCH --job-name=hnav-navmotion
# The probe itself runs in ~2 min; the short wall clock is what lets it
# backfill into a busy partition instead of queueing behind the training wave.
# The memory is not negotiable, though -- building the scaffold field peaks
# well above 32G, which OOM-kills before the first trial.
#SBATCH --time=0-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal
#SBATCH --mem=100G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/navmotion_%j.out

# Decompose nav `mean_steps` and `success_rate` on the real evaluation path:
# stride, alignment to the goal, alignment to the recall direction q, the step
# count those statistics predict, and -- for the failures -- how close they ever
# got to the goal.
#
#   CKPT=/path/navigate_uN.pt N_DIST=0 sbatch hopfield_nav/run_probe_nav_motion.sh
#
# N_ENVS must be <= the run's num_val_envs (10 for the navigate_ee_* runs).
#
# CPU partition on purpose: the GPU QOS belongs to the training wave.

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
CKPT=${CKPT:?CKPT is required}
OUT=${OUT:-}
N_ENVS=${N_ENVS:-4}
TRIALS=${TRIALS:-32}
MAX_STEPS=${MAX_STEPS:-400}
N_DIST=${N_DIST:-0}

module load miniforge/24.3.0-0
source activate cls

cd "$REPO"
source scripts/cls_env.sh

python -u -m hopfield_nav.probes.nav_motion \
    --ckpt "$CKPT" --n_envs "$N_ENVS" --trials "$TRIALS" \
    --max_steps "$MAX_STEPS" --n_dist "$N_DIST" --device cpu \
    --split "${SPLIT:-val}" --seed "${SEED_OVERRIDE:-42}" \
    ${OUT:+--output_json "$OUT"}
