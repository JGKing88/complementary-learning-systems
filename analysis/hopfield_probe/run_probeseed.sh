#!/usr/bin/env bash
# Does the att0.5 result survive a different scaffold sample?
#
# Every reach number in this campaign comes from `--seed 0`: 8 worlds x 20 envs,
# one draw. Encoder-training seed spread has been measured repeatedly; the
# PROBE's own world and goal sampling never has. att0.5 leads Level-6-at-gain-300
# by 0.987 to 0.971, which is small enough that one scaffold draw should not
# decide it.
#
# Two more draws, on the two arms that matter, each at its own best gain:
#   att0.5 at gain 100  (its optimum -- gain 300 costs it 0.010)
#   L6     at gain 300  (its optimum -- gain 100 costs it 0.040)
# Beta = gain throughout, no saturation.
#
#SBATCH --job-name=probe_seeds
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_seeds_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W52=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm
W49=/orcd/pool/003/jackking/cls_runs/sweeps/w49_g100_knee
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_seeds

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --device cpu --fwhm_fallback 0.25)

ATT=()
L6=()
for s in 42 43 44 45; do
    ATT+=(--ckpt "$(ls -d "$W52"/*_att0.5_seed=$s)/encoder_final.pt"
          --label "att0.5-s$s")
    L6+=(--ckpt "$(ls -d "$W49"/*eps1_rate0.5_seed=$s)/encoder_final.pt"
         --label "L6-s$s")
done

for ps in 1 2; do
    echo "=== arm: att0.5 gain 100, probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${ATT[@]}" "${COMMON[@]}" \
          --seed "$ps" --out "$OUT/att0.5_ps$ps"
    echo "=== arm: L6 gain 300, probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${L6[@]}" "${COMMON[@]}" \
          --seed "$ps" --encoder_gain 300 --out "$OUT/l6_g300_ps$ps"
done

echo "DONE $OUT"
