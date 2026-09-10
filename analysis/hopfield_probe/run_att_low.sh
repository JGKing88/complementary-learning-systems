#!/usr/bin/env bash
# att1 and att0.5 -- four seeds each, at their own gain and at 300.
#
# The cross-encoder ladder put att1 and att0.5 at continuous reach 0.993 at
# gain 100, beating Level 6 at gain 300 (0.971) and Level 6 saturated (0.981).
# Both are Level 6 with `attract_lambda` moved DOWN from 2.0, the direction the
# campaign never went -- w52/w53/w54 walked up to 64 because `r_min` rewards
# res90. On `r_min` att1 scores 10.0 and att0.5 6.0, against Level 6's 12.0, so
# both were correctly recorded as losses and are, on the nav objective, wins.
#
# That is one seed each, and this campaign has had one-seed results evaporate
# four times, so this is the confirmation before anything is built on it.
#
# Gain 300 is included because the ladder showed Level 6's optimum is interior
# at gain 300 -- but att1 at 300 sits at res90 6.5, past where Level 6's chart
# started to go, so this also tests whether that limit is about res90 itself or
# about reading an encoder through a chart it was not trained for.
#
# Beta defaults to the encoder gain throughout: no saturation.
#
#SBATCH --job-name=probe_attlow
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_attlow_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_attlow

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

CK=()
for s in 42 43 44 45; do
    for arm in att1 att0.5; do
        d=$(ls -d "$W"/*_${arm}_seed=$s)
        CK+=(--ckpt "$d/encoder_final.pt" --label "$arm-s$s")
    done
done

echo "=== arm 1/2: own gain 100, beta = gain ==="
"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
      --out "$OUT/g100"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/g100"

echo "=== arm 2/2: gain 300, beta = gain ==="
"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
      --encoder_gain 300 --out "$OUT/g300"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/g300"

echo "DONE $OUT"
