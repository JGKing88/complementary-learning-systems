#!/usr/bin/env bash
# The full suite on the 10% encoder, saturated two different ways.
#
# Sec 7 gives two conditions for a stored pattern to be a fixed point, and they
# are set by two different knobs:
#
#   (a) the pattern sits near a hypercube corner   <- the ENCODER gain
#   (b) sign(Wz) = sign(z) survives the recall map <- the HOPFIELD's beta
#
# The ladder runs both at the encoder's own gain (100 here), where `u = beta*s
# /sqrt(D)` is about 3e-3 and the tanh is numerically inert -- recall is power
# iteration and the stored pattern is not a fixed point at all. Sec 10.19
# measured (b) alone and found the basin SHRINKS, with no mechanism that
# survived checking. So this runs the whole suite, not just the basin, at:
#
#   arm A   beta = 1e6, encoder gain 100    -- condition (b) only
#   arm B   beta = 1e6, encoder gain 1e6    -- (a) and (b) together
#
# Arm B's encoder output is `tanh(1e6 * z)`, which is `sign(z)` to float
# precision: the pattern IS a corner, cos_bin = 1 exactly, so it removes the
# one explanation Sec 10.19 offered and could not support (that saturation
# leaves the state 0.04 of cosine away from the goal). If the basin still
# shrinks with the pattern exactly on a corner, the shrinkage is not about
# where the fixed point sits.
#
# Four training seeds each, so the tabs carry the same seed spread as the
# ladder's five rungs and no claim rests on one draw. One checkpoint per task:
# the work is independent and 8 short jobs beat one long one.
#
#SBATCH --job-name=probe_sat10
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/sat10_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-7
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_sat10

i=$SLURM_ARRAY_TASK_ID
arm=$(( i / 4 ))
seed=$(( 42 + i % 4 ))
CK=$(ls -d "$S"/w52_attract_fwhm/*_att0.5_seed=$seed)/encoder_final.pt

if (( arm == 0 )); then
    GROUP="10% β=1e6"
    EGAIN=100
else
    GROUP="10% gain=1e6, β=1e6"
    EGAIN=1000000
fi

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$GROUP · att0.5 · s$seed" \
    --encoder_gain "$EGAIN" --beta 1e6 \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --out "$OUT/t$i"

echo "DONE task $i  arm=$arm seed=$seed  -> $OUT/t$i"
