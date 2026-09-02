#!/bin/bash -l
#SBATCH --job-name=navtriprobe
#SBATCH --time=0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_probe_%j.out

# Run analysis/nav_tri/behavior_probe.py on one or more checkpoints.
# Needs a GPU because encoded_Phi is ~12 GB at Npos=1716 and the scaffold has
# to be rebuilt to project the recall into the local tangent basis.
#
#   CKPTS="/path/a.pt /path/b.pt" sbatch hopfield_nav/run_nav_tri_probe.sh
#   CKPTS=... MODE="explore" TRIALS=32 sbatch hopfield_nav/run_nav_tri_probe.sh
#
# PROBE picks the tool: behavior (default) | signal | temporal | state.
# `state` is analysis/nav_tri/state_probe.py -- is the agent storing useful
# information in its recurrent state, and does the policy use it:
#
#   PROBE=state CKPTS="/path/a.pt /path/b.pt" TRIALS=16 \
#       sbatch hopfield_nav/run_nav_tri_probe.sh

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
MODE=${MODE:-"explore nav"}
NDIST=${NDIST:-"0 10"}
TRIALS=${TRIALS:-32}
MAX_STEPS=${MAX_STEPS:-200}
ENVS=${ENVS:-}
OUTDIR=${OUTDIR:-/orcd/pool/003/jackking/cls_runs/results/nav_tri_probe}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUTDIR"

# PROBE=behavior (default) runs behavior_probe.py; PROBE=signal runs
# signal_separability.py, which needs no trained policy and reads the
# checkpoint only for its config and recorded world.
PROBE=${PROBE:-behavior}

# PROBE=state is the exception to the per-checkpoint loop below: state_probe
# takes every checkpoint in ONE process so the 12 GB scaffold is built once,
# and comparing arms is the normal way to read it (a state_influence of 0.08
# means nothing until you have seen what a control scores).
if [ "$PROBE" = state ]; then
    tag=$(basename "$(dirname "$(echo $CKPTS | awk '{print $1}')")")
    echo ""
    echo "################ state :: $tag (+$(($(echo $CKPTS | wc -w) - 1)) more) ################"
    python -u -m analysis.nav_tri.state_probe \
        --ckpt $CKPTS --mode ${SMODE:-explore} \
        --n_distractors ${SNDIST:-0} \
        --trials "$TRIALS" --max_steps "$MAX_STEPS" \
        ${ENVS:+--envs $ENVS} ${SAMPLED:+--sampled_rollout} \
        --json "$OUTDIR/state_${tag}.json"
    exit 0
fi

for ck in $CKPTS; do
    tag=$(basename "$(dirname "$ck")")_$(basename "$ck" .pt)
    echo ""
    echo "################ $PROBE :: $tag ################"
    if [ "$PROBE" = signal ]; then
        python -u -m analysis.nav_tri.signal_separability \
            --ckpt "$ck" --n_distractors $NDIST \
            --cells "${CELLS:-200}" ${ENVS:+--envs $ENVS} \
            --json "$OUTDIR/signal_${tag}.json"
    elif [ "$PROBE" = temporal ]; then
        python -u -m analysis.nav_tri.temporal_separability \
            --ckpt "$ck" --n_distractors $NDIST \
            --steps "${TSTEPS:-20}" --sets "${SETS:-8}" --traj "${TRAJ:-32}" \
            ${ENVS:+--envs $ENVS} \
            --json "$OUTDIR/temporal_${tag}.json"
    else
        python -u -m analysis.nav_tri.behavior_probe \
            --ckpt "$ck" --mode $MODE --n_distractors $NDIST \
            --trials "$TRIALS" --max_steps "$MAX_STEPS" \
            ${ENVS:+--envs $ENVS} \
            --json "$OUTDIR/${tag}.json"
    fi
done
