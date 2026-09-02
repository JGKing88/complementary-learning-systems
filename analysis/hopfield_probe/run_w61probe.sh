#!/usr/bin/env bash
# The two leading 0.75% arms, four seeds, all three scaffold draws.
#
# Screen: y35_a2 (18 x 35) 0.0541, y50_a2 (9 x 50) 0.0600, y27_a4 (30 x 27)
# 0.0649, y27_a2 0.0705. The count-held arm -- 30 patches, the count that won at
# 1.25% -- came LAST, and needed 27-cell patches to hold that count. So the
# crossover I read at 1.25% is better explained as a patch-size floor near 35
# cells than as a count preference: 35 wins at both 1.25% and 0.75%, and 27 and
# 25 lose wherever they appear.
#
# Both leaders go through because the gap is 0.006 and the screen has lost to
# the probe three times (Sec 10.12, 10.13, 10.15) -- it filters, it does not
# rank.
#
# Three draws in one job: the question is where reach lands against 1.25%'s
# 0.870, which is the size of gap a single draw cannot resolve (Sec 10.10).
#
#SBATCH --job-name=probe_w61
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w61_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w61_cov0.75
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w61

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --device cpu --fwhm_fallback 0.25)

CK=()
for s in 42 43 44 45; do
    CK+=(--ckpt "$(ls -d "$W"/*_y35_a2_seed=$s)/encoder_final.pt"
         --label "y35_a2-s$s" --encoder_gain 100)
done
for s in 42 43 44 45; do
    CK+=(--ckpt "$(ls -d "$W"/*_y50_a2_seed=$s)/encoder_final.pt"
         --label "y50_a2-s$s" --encoder_gain 200)
done

for ps in 0 1 2; do
    echo "=== arm: 0.75% leaders, probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
          --seed "$ps" --out "$OUT/ps$ps"
done

echo "DONE $OUT"
