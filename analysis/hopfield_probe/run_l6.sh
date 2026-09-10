#!/usr/bin/env bash
# Level 6 (w49_g100_knee eps1_rate0.5) at production settings and at the
# recommended gain 300 + saturated beta. Four seeds each.
#
# Sec 10.6 item 2 nominates this arm off the screen: far>0.25 = 0.0057 at gain
# 300, res90 8, the only constrained arm inside v35's box. Prediction on record
# before the run -- continuous reach 0.92-0.95, between v35 g100+sat (far
# 0.0036, reach 0.974) and att16 g300+sat (far 0.0125, reach 0.890).
#
# The baseline arm exists because Level 6 has never been probed at all, so
# without it a good number could not be attributed to the gain change rather
# than to the encoder itself.
#
# Settings are copied exactly from the archived arms (n_worlds 8, K to 20,
# seed 0) -- anything else and the comparison to production/att16/v35 is void.
#
#SBATCH --job-name=probe_l6
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_l6_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w49_g100_knee
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_l6

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu)

CKPT=()
for s in 42 43 44 45; do
    d=$(ls -d "$W"/*eps1_rate0.5_seed=$s)
    CKPT+=(--ckpt "$d/encoder_final.pt" --label "L6-s$s")
done

echo "=== arm 1/2: production settings (beta = gain = 100) ==="
"$PY" -m analysis.hopfield_probe.run "${CKPT[@]}" "${COMMON[@]}" \
      --out "$OUT/production"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/production"

echo "=== arm 2/2: gain 300 + saturated beta ==="
"$PY" -m analysis.hopfield_probe.run "${CKPT[@]}" "${COMMON[@]}" \
      --encoder_gain 300 --beta 1e6 --out "$OUT/g300_sat"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/g300_sat"

echo "DONE  $OUT/production/report/index.html  $OUT/g300_sat/report/index.html"
