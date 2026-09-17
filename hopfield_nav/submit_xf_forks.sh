#!/bin/bash
# Phase 2 of the explore-first line (docs/EXPLORE_FIRST_PLAN.md §4, §7): the
# six forks of one explorer checkpoint, differing only in what holds the
# policy near the explorer. Submit once the explorer has landed.
#
#   XF_EXPLORER=<.../navigate_u700.pt> bash hopfield_nav/submit_xf_forks.sh
#   DRY=1 XF_EXPLORER=... bash hopfield_nav/submit_xf_forks.sh   # print only
#   ONLY="xf_naive xf_kl_1" XF_EXPLORER=... bash ...              # a subset
#
# 5 h, not 12: this shape (1 env x 4 repeats x 64 batch, h1024) runs at
# ~8 s/update, so 1000 updates plus 40 evals is ~3 h -- and a 12 h request on
# ou_bcs_normal sat behind the task line's 12 h jobs with a start estimate
# five days out (log, 2026-09-17). CKPT_EVERY=25 makes a TIMEOUT a normal
# outcome either way.
set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/explore-first}
PARTITION=${PARTITION:-ou_bcs_normal}
TIME=${TIME:-5:00:00}
SEED=${SEED:-42}
: "${XF_EXPLORER:?set XF_EXPLORER to the explorer checkpoint the forks start from}"
[ -f "$XF_EXPLORER" ] || { echo "ERROR: $XF_EXPLORER is not a file" >&2; exit 1; }

cd "$REPO"

# The central arm first, then the plain lever, then the supporting arms.
ARMS=${ONLY:-"xf_naive xf_naive_lr03 xf_ewc_1e3 xf_ewc_1e4 xf_kl_1 xf_kl_10"}

for v in $ARMS; do
    cmd=(sbatch --partition="$PARTITION" --time="$TIME" --job-name="${v}_s${SEED}"
         hopfield_nav/run_nav_p2.sh)
    echo "XF_EXPLORER=$XF_EXPLORER VARIANT=$v SEED=$SEED REPO=$REPO ${cmd[*]}"
    if [ -z "${DRY:-}" ]; then
        XF_EXPLORER="$XF_EXPLORER" VARIANT="$v" SEED="$SEED" REPO="$REPO" "${cmd[@]}"
    fi
done
