#!/usr/bin/env bash
# Probe the two w55 arms that beat Level 6 on the screen.
#
# att0.25 has the lowest alias rate measured in-brief (0.0051 at gain 100) and
# continues the attract trend, but sits at res90 5 -- and reach peaks near res90
# 7 and falls off below 6. Level 6 FORCED to res90 5 by gain scored 0.608, while
# L5 TRAINED at res90 6 scored 0.977, so trained-short and forced-short behave
# differently and this is the arm that separates them.
#
# Gain 30 is included for exactly that reason: lowering inference gain LENGTHENS
# the chart, so if att0.25 is past the peak, gain 30 should bring it back. It is
# the mirror of raising gain on Level 6, and no arm has ever been read below its
# trained gain.
#
# sm30 is the other mechanism: 327 patches of 30 cells at the same coverage,
# alias 0.0060 at res90 8 against Level 6's 0.0082 at res90 10 -- better on both
# axes at once, which nothing else in the screen manages.
#
# Beta = gain throughout.
#
#SBATCH --job-name=probe_w55
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w55_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w55_nav_objective
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w55

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

A25=()
SM=()
for s in 42 43 44 45; do
    A25+=(--ckpt "$(ls -d "$W"/*_att0.25_seed=$s)/encoder_final.pt"
          --label "att0.25-s$s")
    SM+=(--ckpt "$(ls -d "$W"/*_sm30_seed=$s)/encoder_final.pt"
         --label "sm30-s$s")
done

echo "=== arm 1/3: att0.25 at own gain 100 ==="
"$PY" -m analysis.hopfield_probe.run "${A25[@]}" "${COMMON[@]}" \
      --out "$OUT/att0.25_g100"

echo "=== arm 2/3: att0.25 at gain 30 (lengthen the chart) ==="
"$PY" -m analysis.hopfield_probe.run "${A25[@]}" "${COMMON[@]}" \
      --encoder_gain 30 --out "$OUT/att0.25_g30"

echo "=== arm 3/3: sm30 at own gain 100 ==="
"$PY" -m analysis.hopfield_probe.run "${SM[@]}" "${COMMON[@]}" \
      --out "$OUT/sm30_g100"

echo "DONE $OUT"
