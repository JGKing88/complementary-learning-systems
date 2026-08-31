#!/usr/bin/env bash
# The two leading 1.25% arms, four seeds, all three scaffold draws.
#
# Screen result: among the size-held arms (15 x 50) the attract optimum is 2.0,
# which is the prediction Sec 10.14 put on record -- 0.5 / 0.5 / 1.0 / 2.0 down
# the ladder. But sm35x_a2 (30 x 35, COUNT held at 30) beats all of them on the
# alias rate, 0.0286 against x_a2's 0.0353, reversing which geometry won at 2.5%.
# So both go through, at their own gains.
#
# Three draws in one job because a single-draw gap under ~0.02 has not been
# trustworthy anywhere in this campaign (Sec 10.10), and the interesting
# question here -- whether reach finally breaks from flat -- is a comparison
# against 2.5%'s 0.965, which is that size of gap.
#
# Per-ckpt --encoder_gain: 100 for sm35x_a2, 200 for x_a2.
#
#SBATCH --job-name=probe_w60
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w60_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w60_cov1.25
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w60

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --device cpu --fwhm_fallback 0.25)

CK=()
for s in 42 43 44 45; do
    CK+=(--ckpt "$(ls -d "$W"/*_sm35x_a2_seed=$s)/encoder_final.pt"
         --label "sm35x_a2-s$s" --encoder_gain 100)
done
for s in 42 43 44 45; do
    CK+=(--ckpt "$(ls -d "$W"/*_x_a2_seed=$s)/encoder_final.pt"
         --label "x_a2-s$s" --encoder_gain 200)
done

for ps in 0 1 2; do
    echo "=== arm: 1.25% leaders, probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
          --seed "$ps" --out "$OUT/ps$ps"
done

echo "DONE $OUT"
