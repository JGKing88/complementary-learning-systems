#!/bin/bash -l
#SBATCH --job-name=readoutfield
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/readout_field_%j.out

# Map the recalled direction field q(x) and find its sinks. No policy and no
# rollouts -- this tests whether the loops in EXPERIMENTS_NAV_P2 section 13.5
# are traps in the READOUT rather than in any policy.
#
#   CKPT=/path/navigate_u2000.pt sbatch hopfield_nav/run_readout_field.sh
#
# --seed must match the exploit_diag run being compared against, or the
# memories differ and the sinks are not the ones whose orbits were measured.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
NDIST=${NDIST:-10}
TRIALS=${TRIALS:-8}
DRAW=${DRAW:-32}
SEED=${SEED:-42}
STEP=${STEP:-1.0}
EGAIN=${EGAIN:-}
HBETA=${HBETA:-}
ENVS=${ENVS:-}
TAG=${TAG:-field}
OUTDIR=${OUTDIR:-/orcd/pool/003/jackking/cls_runs/results/exploit_diag}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUTDIR"

echo "################ readout_field :: $TAG ################"
python -u -m analysis.nav_tri.readout_field \
    --ckpt "$CKPT" --n_distractors "$NDIST" --trials "$TRIALS" \
    --seed "$SEED" --step "$STEP" --draw_trials "$DRAW" ${EGAIN:+--encoder_gain $EGAIN} ${HBETA:+--hopfield_beta $HBETA} ${ENVS:+--envs $ENVS} \
    --json "$OUTDIR/${TAG}.json" --html "$OUTDIR/${TAG}.html"

echo "done: $OUTDIR/${TAG}.json  $OUTDIR/${TAG}.html"
