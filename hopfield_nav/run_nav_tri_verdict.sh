#!/bin/bash -l
#SBATCH --job-name=navtriverd
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=mit_normal
#SBATCH --mem=120G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_verdict_%j.out

# The strict scorecard for a candidate model. In-training evals are 6 envs x
# 16 trials and are monitoring-grade only -- the explore-min wave found that
# estimate biased high by ~0.02 and, worse, wrong about RANKING. Nothing in
# EXPERIMENTS_NAV_TRI is called a result until it has been through here.
#
#   CKPTS="/path/a.pt /path/b.pt" sbatch hopfield_nav/run_nav_tri_verdict.sh
#
# Runs on CPU: it is the same work the probes do, both GPU partitions cap
# concurrency at 2, and training holds those.
#
# EVAL_ALL=1 additionally runs hopfield_nav.eval_all on the FIRST checkpoint.
# That is the established protocol; behavior_probe reimplements it to get the
# diagnostic columns, and the two agreeing is what licenses reading the
# probe's numbers as a verdict rather than as a second opinion.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
ENVS=${ENVS:-10}
TRIALS=${TRIALS:-32}
NDIST=${NDIST:-"0 5 10"}
MAX_STEPS=${MAX_STEPS:-200}
OUTDIR=${OUTDIR:-/orcd/pool/003/jackking/cls_runs/results/nav_tri_verdict}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUTDIR"

first=$(echo $CKPTS | awk '{print $1}')
tag=$(basename "$(dirname "$first")")

echo "=== nav_tri verdict: $ENVS envs x $TRIALS trials, n_dist=$NDIST, "
echo "    $MAX_STEPS steps, deterministic policy ==="
echo "    reference lines: coverage -- billiard 0.352, run-and-tumble 0.274,"
echo "    random walk 0.178, ceiling 0.5025"
echo "    mean_steps at |a|=1 -- 10.1 at cos(q,goal)=0.99, 15.3 at cos=0.70"

python -u -m analysis.nav_tri.behavior_probe \
    --ckpt $CKPTS --device cpu --mode explore nav \
    --n_distractors $NDIST --trials "$TRIALS" --envs "$ENVS" \
    --max_steps "$MAX_STEPS" \
    --json "$OUTDIR/verdict_${TAG:-$tag}.json"

if [ "${EVAL_ALL:-0}" = 1 ]; then
    echo ""
    echo "=== cross-check: hopfield_nav.eval_all on $first ==="
    python -u -m hopfield_nav.eval_all --ckpt "$first" --device cpu \
        --num-val-envs "$ENVS" --num_trials "$TRIALS" \
        --max_steps "$MAX_STEPS" --n_distractors $NDIST \
        --no-nav-stoch --skip-realistic --repeat-trials 0 \
        --output-json "$OUTDIR/evalall_${TAG:-$tag}.json"
fi
