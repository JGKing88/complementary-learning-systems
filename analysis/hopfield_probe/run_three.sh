#!/usr/bin/env bash
# The three coverage winners in ONE invocation, each at its own optimal gain.
#
# One invocation is what gives the report an encoder selector (`multi_page`
# stacks a body per encoder behind `data-encoder`). Until now --encoder_gain was
# global, which would have forced the 5% arm through gain 100 when its optimum
# is 75 -- past its peak, res90 ~6. Hence the per-ckpt form.
#
#   10%    w52 att0.5      seed 43   gain 100 (its own)
#    5%    w57 half_a0.5   seed 43   gain  75 (NOT its trained 100)
#   2.5%   w58 q_a1        seed 45   gain 100 (its own)
#
# beta defaults to the encoder gain throughout: no saturation.
#
# Settings are copied exactly from every archived arm, so the numbers here are
# comparable to Sec 10.8-10.13 rather than being a fresh scale.
#
#SBATCH --job-name=probe_three
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_three_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_three

"$PY" -m analysis.hopfield_probe.run \
  --ckpt "$S/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt" \
      --label "10% · att0.5"   --encoder_gain 100 \
  --ckpt "$S/w57_cov5/001_half_a0.5_seed=43/encoder_final.pt" \
      --label "5% · half_a0.5" --encoder_gain 75 \
  --ckpt "$S/w58_cov2.5/011_q_a1_seed=45/encoder_final.pt" \
      --label "2.5% · q_a1"    --encoder_gain 100 \
  --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
  --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
  --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
  --seed 0 --device cpu --fwhm_fallback 0.25 \
  --out "$OUT"

"$PY" -m analysis.hopfield_probe.report.build "$OUT"
echo "DONE report: $OUT/report/index.html"
