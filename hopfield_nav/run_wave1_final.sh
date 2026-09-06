#!/bin/bash -l
#SBATCH --job-name=w1final
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=mit_normal
#SBATCH --mem=120G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/w1_final_%j.out
#
# The complete wave-1 report, in one job.
#
# Reads every arm at its LARGEST COMMON CHECKPOINT, not at whatever update each
# happened to die on -- the arms run on nodes of different speed (node3207 is
# 31.6 s/update against node4200's 21.5) so they reach different update counts
# inside the same 6 h wall.
#
# Everything load-bearing comes from the PROBE (144 matched trials), never from
# the training eval: the training-time swept eval swings 39-45% point to point
# (median ratio 1.5x, max 2.2x), which is wide enough to have produced three
# spurious between-arm comparisons in this document already.
#
#   COMMON=675 sbatch wave1_final.sh
set -euo pipefail
REPO=/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric
cd "$REPO"
export PYTHONPATH="$REPO"
PY=/home/jackking/.conda/envs/cls/bin/python
CKD=/orcd/pool/003/jackking/cls_runs/agent_ckpts
OUT=/orcd/pool/003/jackking/cls_runs/results/nav_tri_probe
mkdir -p "$OUT"

SPEC=$CKD/navigate_navp2_p20_e_s42_21695407/navigate_u700.pt

declare -A ARM=(
  [d0_base]=navigate_navp2_d0_base_s42_22133273
  [d1_kanneal]=navigate_navp2_d1_kanneal_s42_22133274
  [d1_persr]=navigate_navp2_d1_persr_s42_22133275
  [d1_ms3]=navigate_navp2_d1_ms3_s42_22133276
)

# Largest checkpoint every arm has, unless overridden.
if [ -z "${COMMON:-}" ]; then
  COMMON=999999
  for a in "${!ARM[@]}"; do
    d=$CKD/${ARM[$a]}
    [ -d "$d" ] || continue
    m=$(ls "$d"/navigate_u*.pt 2>/dev/null | sed 's/.*_u//;s/\.pt//' \
        | sort -n | tail -1)
    [ -n "$m" ] && [ "$m" -lt "$COMMON" ] && COMMON=$m
  done
fi
echo "############ largest common checkpoint: u$COMMON ############"

CKPTS=""; LABELS=""
for a in d0_base d1_kanneal d1_persr d1_ms3; do
  f=$CKD/${ARM[$a]}/navigate_u${COMMON}.pt
  if [ -f "$f" ]; then CKPTS="$CKPTS $f"; LABELS="$LABELS ${a}_u${COMMON}"; fi
done
echo "arms: $LABELS"

for ND in 0 10; do
  J="$OUT/w1_final_u${COMMON}_d${ND}.json"
  echo "################ wave 1 final  n_dist=$ND ################"
  $PY -u -m analysis.nav_tri.explore_traj \
      --ckpt $CKPTS "$SPEC" \
      --labels $LABELS p20_e_u700 \
      --envs 6 --trials 24 --n_distractors "$ND" \
      --max_steps 200 --split place=held_out \
      --seed 42 --device cpu --no-deterministic \
      --json "$J"
  echo "---- swept + billiard efficiency ----"
  $PY -u -m analysis.nav_tri.swept_from_traj --json "$J" --radius 1.0 --vs_billiard
  echo "---- collapsed tail, split by chase_q ----"
  $PY -u -m analysis.nav_tri.tail_report "$J"
  echo "---- recurrence (is anything orbiting?) ----"
  $PY -u -m analysis.nav_tri.recurrence --json "$J"
done

# --- the EXPLOIT half ------------------------------------------------------
# The explore probe above says nothing about mode A vs mode B, which is the
# single most important distinction in the project (§2.1) and cannot be read
# from a success rate: the two failure modes cost about the same. It needs
# follow_q_fail x q_accuracy_fail, and align_true beside follow_q or the
# geometric identity at q_accuracy ~ 1 makes the number meaningless.
#
# The exploit reference is p19_kcap, NOT p10_pol_v1. p10_pol_v1 is the better
# exploit model (SS9.8) but it trained on the OLD ur_loss2_repel_low encoder, so
# it does not share a world with these arms and behavior_probe refuses -- the
# same guard that blocked wave 0.2. p19_kcap u800 is SS17.10's delivered model
# on the same w52 encoder, which is what makes the comparison legal.
#
# mean_steps is NOT directly comparable across models that choose different
# speeds. Read it against each model's own optimum,
# (mean_start_dist - goal_radius) / that model's realized step magnitude, which
# the probe reports as mean_start_dist and step_mag_mean.
EXPLOIT=$CKD/navigate_navp2_p19_kcap_s42_21656252/navigate_u800.pt
NAVCK="$CKPTS"
[ -f "$EXPLOIT" ] && NAVCK="$NAVCK $EXPLOIT"
echo "################ wave 1 final -- EXPLOIT half ################"
$PY -u -m analysis.nav_tri.behavior_probe \
    --ckpt $NAVCK \
    --mode nav --n_distractors 0 5 10 \
    --split place=held_out --trials 32 --envs 6 \
    --max_steps 200 --seed 42 --device cpu \
    --json "$OUT/w1_final_u${COMMON}_nav.json"
echo "ALL DONE"
