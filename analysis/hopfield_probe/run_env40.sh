#!/usr/bin/env bash
# The same five encoders in a 40x40 environment, to test whether the basin
# ceiling is the encoder or the measurement.
#
# `r_exact_95` is a radius in cells and cannot exceed the largest goal-to-corner
# distance the eval environment offers. Across 198 probed encoders the maximum
# ever seen is 21.62 and 18 of them sit within 0.1 of it -- a pin, not a
# coincidence. So the flatness between the 10% and 5% rungs (21.12 vs 20.73) may
# be the measurement running out of room rather than the encoders being equal.
#
# Doubling the env to 40 lifts the cap to ~55 cells. Two outcomes, both useful:
#   * basins rise above 21.62 and spread out -> the ceiling was real, the top of
#     the ladder was compressed, and env_size is a lever on the metric.
#   * basins stay near their env-20 values -> 21 is an encoder property and the
#     pin at 21.62 is a coincidence of where these encoders happen to sit.
#
# Everything else is held to the archived settings. Note that reach and the
# angular errors are NOT comparable to the env-20 runs -- a 40x40 arena is a
# harder navigation problem with longer paths -- so only the basin column
# transfers, which is the one being asked about.
#
#SBATCH --job-name=probe_env40
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_env40_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_env40

"$PY" -m analysis.hopfield_probe.run \
  --ckpt "$S/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt" \
      --label "10% · att0.5"     --encoder_gain 100 \
  --ckpt "$S/w57_cov5/001_half_a0.5_seed=43/encoder_final.pt" \
      --label "5% · half_a0.5"   --encoder_gain 75 \
  --ckpt "$S/w58_cov2.5/011_q_a1_seed=45/encoder_final.pt" \
      --label "2.5% · q_a1"      --encoder_gain 100 \
  --ckpt "$S/w60_cov1.25/014_sm35x_a2_seed=44/encoder_final.pt" \
      --label "1.25% · sm35x_a2" --encoder_gain 100 \
  --ckpt "$S/w61_cov0.75/014_y50_a2_seed=44/encoder_final.pt" \
      --label "0.75% · y50_a2"   --encoder_gain 200 \
  --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
  --steps 1 2 3 5 10 15 --env_size 40 --Npos 1716 \
  --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
  --seed 0 --device cpu --fwhm_fallback 0.25 \
  --out "$OUT"

echo "DONE $OUT"
