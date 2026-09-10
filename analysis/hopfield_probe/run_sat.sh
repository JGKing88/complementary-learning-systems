#!/usr/bin/env bash
# Saturate the 10% winner. Does att0.5 become a real attractor network?
#
# Sec 7's two conditions: the saturated update is `x <- sign(Wx)/sqrt(D)`, so a
# memory is a fixed point iff (a) the pattern is already near a hypercube corner
# -- an encoder property, measured as cos_bin -- and (b) capacity holds,
# sign(Wz) = sign(z). Beta alone buys (b) and not (a), which is why raising it
# alone was a net loss in Sec 2.
#
# cos_bin at gain 100 is 0.9546 for att0.5, against v35's 0.9682 at the gain its
# saturated arm used -- so condition (a) is essentially already met and beta can
# be saturated without touching gain. The g300 arm is the more-corner-like
# variant (cos_bin 0.9841) and costs res90 7 -> 5, past the reach optimum, so it
# is the control that separates "more corner" from "shorter chart".
#
# Predictions, before the run:
#   * the trajectory stops drifting -- goal_dist 0.00 held to s=15, against the
#     unsaturated arm's 0.00 / 0.00 / 0.00 / 1.21 / 1.41 / 1.41 at s=1..15.
#   * acc45 at s=15 rises from 96.5% to ~100%.
#   * continuous reach is UNCHANGED near 0.987. Sec 10.15 found the residual
#     failures are walkers parked outside the 0.5-cell arrival radius, not field
#     failures, and saturation cannot move that.
#   * exact_hit holds rather than collapsing the way it did in Sec 2 (74% -> 44%
#     on v35), because retrieval is scored against a CONTINUOUS cell bank and
#     att0.5 at cos_bin 0.95 is most of the way to a corner already.
#
#SBATCH --job-name=probe_sat
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_sat_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_sat

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

CK=()
for s in 42 43 44 45; do
    CK+=(--ckpt "$(ls -d "$W"/*_att0.5_seed=$s)/encoder_final.pt"
         --label "att0.5-s$s")
done

echo "=== arm 1/2: gain 100 (own, cos_bin 0.955) + beta 1e6 ==="
"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
      --encoder_gain 100 --beta 1e6 --out "$OUT/g100_sat"

echo "=== arm 2/2: gain 300 (cos_bin 0.984, res90 5) + beta 1e6 ==="
"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
      --encoder_gain 300 --beta 1e6 --out "$OUT/g300_sat"

echo "DONE $OUT"
