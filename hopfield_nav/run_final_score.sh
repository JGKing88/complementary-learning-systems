#!/bin/bash -l
#SBATCH --job-name=hnav-score
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal
#SBATCH --mem=56G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/score_%j.out

# Re-score one checkpoint on all three target metrics with enough trials to
# trust the number, on the same val split the training eval uses.
#
#   CKPT=/path/navigate_uN.pt sbatch hopfield_nav/run_final_score.sh
#
# CPU on purpose: the GPU QOS caps this account at 2 per partition and those
# belong to the training runs. See probes/final_score.py for why a single
# logged eval is not good enough for choosing a deliverable.

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
CKPT=${CKPT:?CKPT is required}
N_ENVS=${N_ENVS:-10}
TRIALS=${TRIALS:-64}
MAX_STEPS=${MAX_STEPS:-400}
OUT=${OUT:-}

module load miniforge/24.3.0-0
source activate cls

cd "$REPO"
source scripts/cls_env.sh

python -u -m hopfield_nav.probes.final_score \
    --ckpt "$CKPT" --n_envs "$N_ENVS" --trials "$TRIALS" \
    --max_steps "$MAX_STEPS" --n_dist 0 10 \
    ${OUT:+--output_json "$OUT"}
