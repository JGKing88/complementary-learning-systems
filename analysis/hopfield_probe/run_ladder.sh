#!/usr/bin/env bash
# Where is the res90 floor? Everything else in Sec 10 is bounded by it.
#
# The screen says every knob is the same trade: attract down, rate_lambda up,
# and gain up all lower the far-field alias rate AND lower res90 together. So
# the frontier is set by how low res90 can go before direction breaks, and that
# number currently rests on four points from two encoders (fine at 8, 11, 14;
# collapses 10.0 -> 19.2 degrees at 6).
#
# Beta is left to default, which the probe sets to the encoder gain -- the
# beta = gain operating point, no saturation, as instructed.
#
# Arm 1 is the within-encoder ladder and is the one that matters: four seeds of
# Level 6 at gains that the screen puts at res90 8 / 6 / 5. Gain 100 (res90 10)
# is already archived at `l6_production` and is deterministic, so it is not
# re-run.
#
# Arm 2 spot-checks the same res90 values across DIFFERENT encoders, so that a
# floor found in arm 1 is not mistaken for something specific to raising gain.
#   att1     res90  9, alias 0.0073   -- better alias than L6 at the same gain
#   att0.5   res90  7, alias 0.0059
#   L5       res90  6, alias 0.0043   -- best legal alias above the floor
#   rate3    res90  3, alias 0.0055   -- the extreme; expected to fail
#
#SBATCH --job-name=probe_ladder
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_ladder_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_ladder

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

L6=()
for s in 42 43 44 45; do
    d=$(ls -d "$S"/w49_g100_knee/*eps1_rate0.5_seed=$s)
    L6+=(--ckpt "$d/encoder_final.pt" --label "L6-s$s")
done

for g in 300 1000 3000; do
    echo "=== arm 1: L6 gain $g, beta = gain ==="
    "$PY" -m analysis.hopfield_probe.run "${L6[@]}" "${COMMON[@]}" \
          --encoder_gain "$g" --out "$OUT/l6_g$g"
    "$PY" -m analysis.hopfield_probe.report.build "$OUT/l6_g$g"
done

echo "=== arm 2: cross-encoder res90 ladder at gain 100 ==="
"$PY" -m analysis.hopfield_probe.run \
      --ckpt "$S/w52_attract_fwhm/004_att1_seed=42/encoder_final.pt" \
          --label att1-res9 \
      --ckpt "$S/w52_attract_fwhm/000_att0.5_seed=42/encoder_final.pt" \
          --label att0.5-res7 \
      --ckpt "$S/w39_batch_pairs/008_sm50_b4096_seed=42/encoder_final.pt" \
          --label L5-res6 \
      --ckpt "$S/w46_g100_spread/"*rate3*seed=42/encoder_final.pt \
          --label rate3-res3 \
      "${COMMON[@]}" --encoder_gain 100 --out "$OUT/ladder_g100"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/ladder_g100"

echo "DONE $OUT"
