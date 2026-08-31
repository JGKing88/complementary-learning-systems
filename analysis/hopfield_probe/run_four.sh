#!/usr/bin/env bash
# The coverage ladder, all four rungs, one invocation, each at its own gain.
#
#   10%    w52 att0.5        seed 43   gain 100   reach 0.978
#    5%    w57 half_a0.5     seed 43   gain  75   reach 0.977
#   2.5%   w58 q_a1          seed 45   gain 100   reach 0.965
#   1.25%  w60 sm35x_a2      seed 44   gain 100   reach 0.870
#
# Seeds are each arm's best by three-draw mean, not by draw 0.
#
# Re-run rather than reusing probe_three because `encoder_header` now records
# training coverage, which the overview's coverage panels read -- the older JSON
# predates the field.
#
# beta defaults to the encoder gain throughout: no saturation.
#
#SBATCH --job-name=probe_four
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_four_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_four

"$PY" -m analysis.hopfield_probe.run \
  --ckpt "$S/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt" \
      --label "10% · att0.5"     --encoder_gain 100 \
  --ckpt "$S/w57_cov5/001_half_a0.5_seed=43/encoder_final.pt" \
      --label "5% · half_a0.5"   --encoder_gain 75 \
  --ckpt "$S/w58_cov2.5/011_q_a1_seed=45/encoder_final.pt" \
      --label "2.5% · q_a1"      --encoder_gain 100 \
  --ckpt "$S/w60_cov1.25/014_sm35x_a2_seed=44/encoder_final.pt" \
      --label "1.25% · sm35x_a2" --encoder_gain 100 \
  --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
  --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
  --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
  --seed 0 --device cpu --fwhm_fallback 0.25 \
  --out "$OUT"

"$PY" -m analysis.hopfield_probe.report.build "$OUT"
echo "DONE report: $OUT/report/index.html"
