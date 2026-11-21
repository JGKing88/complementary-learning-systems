#!/bin/bash
# Launch one idea 4 agent arm (GRID_MLP_NAV_PLAN §7b.15) as a chain of 5-hour
# jobs: link 1 starts the run (run_nav_p2.sh) in a fixed SAVE_DIR, every later
# link continues it (run_nav_tri_continue.sh, --continue_from the run's own
# resume/latest.pt, optimizer and RNG streams restored). Run from the repo.
#
#   ARM=arm2 VARIANT=task3r_k2_h128_gmlp_i4_life_rp I4_MEMORY=<memory.pt> \
#     REPO=<code snapshot> LINKS=8 bash hopfield_nav/run_idea4_agent_chain.sh
#
# Optional: LOAD=<parent navigate_u*.pt> (arm 1: fork idea 1's controller),
# I4_RL_HIDDEN (arms 3, 4), SCHEDULE (default: idea 3's lifetime curriculum).
set -euo pipefail
: "${ARM:?}" "${VARIANT:?}" "${REPO:?}"
LINKS=${LINKS:-8}
SEED=${SEED:-42}
SCHEDULE=${SCHEDULE:-'lifetime:1000,envs=2,iters=2 ; lifetime:1000,envs=3,iters=2 ; lifetime:1000,envs=4,iters=2 ; lifetime:1000,envs=6,iters=1'}
RUN=navigate_i4_${ARM}_s${SEED}
SAVE=/orcd/pool/003/jackking/cls_runs/agent_ckpts/$RUN
SB=(--parsable --partition=pi_fiete,ou_bcs_normal --time=05:00:00 --exclude=node1802)
prev=$(REPO=$REPO USE_WANDB=0 VARIANT=$VARIANT SEED=$SEED SAVE_DIR=$SAVE \
       I4_MEMORY=${I4_MEMORY:-} I4_RL_HIDDEN=${I4_RL_HIDDEN:-} I3_STAGE1=${LOAD:-} \
       I3_LIFE_SCHEDULE="$SCHEDULE" sbatch "${SB[@]}" hopfield_nav/run_nav_p2.sh)
echo "$ARM link 1: job $prev  ($RUN)"
for k in $(seq 2 "$LINKS"); do
  prev=$(REPO=$REPO RUN=$RUN SCHEDULE="$SCHEDULE" \
         sbatch "${SB[@]}" --dependency=afterany:$prev hopfield_nav/run_nav_tri_continue.sh)
  echo "$ARM link $k: job $prev"
done
