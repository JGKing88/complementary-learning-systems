#!/bin/bash
# Motion-probe every Nth checkpoint of a run, on its held-out envs.
#
#   RUN=navigate_ee_L1_20356313 EVERY=200 bash hopfield_nav/submit_motion_sweep.sh
#
# `excess_over_matched_walk` is the reason to do this rather than read the
# coverage curve: coverage cannot distinguish a policy that walks fast and
# dumb from one that walks slowly with structure, and only the second has
# headroom above the ~0.56 memoryless ceiling. Reading it per checkpoint is
# also how the collapse in docs/EXPERIMENTS_EXPLORE_EXPLOIT.md was localized.
set -u

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
RUN=${RUN:?RUN is required, e.g. navigate_ee_L1_20356313}
EVERY=${EVERY:-200}
N_ENVS=${N_ENVS:-4}
TRIALS=${TRIALS:-32}
SPLIT=${SPLIT:-val}

A=/orcd/pool/003/jackking/cls_runs/agent_ckpts/$RUN
OUTDIR=${OUTDIR:-/orcd/pool/003/jackking/cls_runs/results/motion_$RUN}
mkdir -p "$OUTDIR"

n=0
for f in "$A"/navigate_u*.pt; do
    [ -e "$f" ] || continue
    u=$(basename "$f" .pt); u=${u#navigate_u}
    [ $((u % EVERY)) -eq 0 ] || continue
    [ -s "$OUTDIR/u$u.json" ] && continue      # already probed
    CKPT="$f" N_ENVS="$N_ENVS" TRIALS="$TRIALS" OUT="$OUTDIR/u$u.json" \
        sbatch --job-name="m_${RUN##*_}_$u" \
        --export=ALL,SPLIT="$SPLIT" "$REPO/hopfield_nav/run_probe_motion.sh" \
        > /dev/null && n=$((n + 1))
done
echo "submitted $n motion probes for $RUN -> $OUTDIR"
