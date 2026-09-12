#!/bin/bash -l
#SBATCH --job-name=eval-ood
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/eval_ood_%j.out

# Post-hoc place-OOD evaluation of the ood_place / ood_place_rp arms
# (run_nav_p2.sh), with d0_base as the legacy reference.
#
# One eval_all invocation per (checkpoint, split list). Splits are minted from
# the checkpoint's own world record, so `place=ood` is the complement of the
# rect that run declared, clear of every train footprint by the margin;
# `place=held_out` is a fresh draw INSIDE the rect; `recorded` is the run's own
# base_val. d0_base declared no rect (legacy place=Anywhere), so it gets only
# recorded + held_out -- `place=ood` on it raises, correctly.
#
# `place=held_out` on the refresh arm is run on its own: the refresh union can
# leave fewer than N_ENVS legal slots, and a raise there must not take the
# `ood` numbers down with it.
#
#   UPDATE=600 sbatch hopfield_nav/run_eval_ood.sh
#   UPDATE=1200 N_ENVS=48 sbatch --partition=pi_fiete hopfield_nav/run_eval_ood.sh
#   SET=corner UPDATE=1200 sbatch hopfield_nav/run_eval_ood.sh
#
# SET picks the arm family: `half` (ood_place / ood_place_rp, rect 858x1716 at
# margin 80, plus d0_base as reference) or `corner` (ood_corner /
# ood_corner_rp, rect 500x500 at margin 50). The corner set gets no
# place=held_out at all: a 500 box has no room for a fresh 24-env draw clear
# of even the 26 fixed envs, let alone the refresh union.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-ood-place}
SET=${SET:-half}
UPDATE=${UPDATE:-600}
N_ENVS=${N_ENVS:-24}          # minted envs per split; `recorded` keeps its own 6
NUM_TRIALS=${NUM_TRIALS:-16}
MAX_STEPS=${MAX_STEPS:-200}
N_DISTRACTORS=${N_DISTRACTORS:-"0 10"}
VAL_SEED=${VAL_SEED:-0}

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh

CK=$CLS_CKPTS
OUT=${OUT:-$CLS_RESULTS/eval_results/ood_${SET}_u${UPDATE}}
mkdir -p "$OUT"

declare -A RUN=(
  [d0_base]=navigate_navp2_d0_base_s42_22133273
  [ood_place]=navigate_navp2_ood_place_s42_22599420
  [ood_place_rp]=navigate_navp2_ood_place_rp_s42_22599421
  [ood_corner]=navigate_navp2_ood_corner_s42_22629938
  [ood_corner_rp]=navigate_navp2_ood_corner_rp_s42_22629939
)

ev() {  # ev <arm> <tag> <split...>
  local arm=$1 tag=$2; shift 2
  local ckpt="$CK/${RUN[$arm]}/navigate_u${UPDATE}.pt"
  if [ ! -f "$ckpt" ]; then echo "!!! MISSING $ckpt"; return 0; fi
  local args=()
  for s in "$@"; do args+=(--split "$s"); done
  echo; echo "=== $arm u$UPDATE  splits: $*  ($(date))"
  python -m hopfield_nav.eval_all --ckpt "$ckpt" --device cuda \
      --num_trials "$NUM_TRIALS" --max_steps "$MAX_STEPS" \
      --n_distractors $N_DISTRACTORS --num-val-envs "$N_ENVS" \
      --val_seed "$VAL_SEED" --skip-realistic \
      --tag "${arm}_u${UPDATE}_${tag}" \
      --output-json "$OUT/${arm}_u${UPDATE}_${tag}.json" \
      "${args[@]}" || echo "!!! $arm $tag FAILED (exit $?)"
}

case "$SET" in
  half)
    ev ood_place    main     recorded place=held_out place=ood
    ev ood_place_rp main     recorded place=ood
    ev ood_place_rp heldout  place=held_out
    ev d0_base      main     recorded place=held_out
    ;;
  corner)
    ev ood_corner    main    recorded place=ood
    ev ood_corner_rp main    recorded place=ood
    ;;
  *) echo "unknown SET=$SET (half|corner)" >&2; exit 1 ;;
esac

echo; echo "=== done $(date)  results in $OUT"
