#!/bin/bash
# A1 wave: grid mode. Plan baseline (mlp-2, mlp-4 x both actions x 2 seeds)
# plus a small width/depth/schedule sweep so the best A1 config is picked
# with evidence rather than a second round-trip. Each run is minutes on a GPU.
#
#   bash hopfield_nav/submit_goal_pairs_a1.sh            # submit all
#   DRY=1 bash hopfield_nav/submit_goal_pairs_a1.sh      # print only
set -euo pipefail
cd "$(dirname "$0")/.."

submit() {  # TAG then env overrides as KEY=VAL...
  local tag=$1; shift
  if [ "${DRY:-0}" = "1" ]; then echo "  $tag: $*"; return; fi
  env MODE=grid TAG="$tag" "$@" sbatch --job-name="$tag" hopfield_nav/run_goal_pairs.sh
}

# --- plan baseline: depth x action x seed ---------------------------------
for MM in continuous discrete; do
  for L in 2 4; do
    for S in 0 1; do
      submit "a1_${MM:0:4}_l${L}h256_s${S}" MOVEMENT=$MM LAYERS=$L HIDDEN=256 SEED=$S
    done
  done
done

# --- width / depth / schedule, continuous, seed 0 ---------------------------
submit a1_cont_l2h512_s0     MOVEMENT=continuous LAYERS=2 HIDDEN=512  SEED=0
submit a1_cont_l2h1024_s0    MOVEMENT=continuous LAYERS=2 HIDDEN=1024 SEED=0
submit a1_cont_l3h512_s0     MOVEMENT=continuous LAYERS=3 HIDDEN=512  SEED=0
submit a1_cont_l4h512_s0     MOVEMENT=continuous LAYERS=4 HIDDEN=512  SEED=0
submit a1_cont_l3h512cos_s0  MOVEMENT=continuous LAYERS=3 HIDDEN=512  SEED=0 LR_SCHEDULE=cosine
submit a1_cont_l3h512wd_s0   MOVEMENT=continuous LAYERS=3 HIDDEN=512  SEED=0 WD=1e-4
submit a1_cont_l3h512tanh_s0 MOVEMENT=continuous LAYERS=3 HIDDEN=512  SEED=0 NONLIN=tanh
