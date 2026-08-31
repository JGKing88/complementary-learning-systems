#!/usr/bin/env bash
# The two best 5%-coverage arms, four seeds, all three scaffold draws.
#
# Screen at matched res90 7 put sm35_a0.5 (120 patches x 35 cells) at alias
# 0.0084 and half_a0.5 (59 x 50) at 0.0088, against the 10% incumbent's 0.0059.
# Each is probed at the gain that lands it at res90 7 -- 50 and 75 respectively,
# NOT its trained gain of 100 -- so the two are compared at matched chart length
# rather than at whichever gain happened to be picked.
#
# All three draws for both arms rather than one draw then a follow-up: Sec 10.10
# measured a fixed arm swinging up to 0.03 across draws, the two arms are 0.0004
# apart on the screen, and a second queue round-trip costs more than the runs.
#
# Beta is never passed; the probe defaults it to the encoder gain.
#
#SBATCH --job-name=probe_w57
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w57_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
W=/orcd/pool/003/jackking/cls_runs/sweeps/w57_cov5
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_w57

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --device cpu --fwhm_fallback 0.25)

SM35=()
HALF=()
for s in 42 43 44 45; do
    SM35+=(--ckpt "$(ls -d "$W"/*_sm35_a0.5_seed=$s)/encoder_final.pt"
           --label "sm35_a0.5-s$s")
    HALF+=(--ckpt "$(ls -d "$W"/*_half_a0.5_seed=$s)/encoder_final.pt"
           --label "half_a0.5-s$s")
done

for ps in 0 1 2; do
    echo "=== sm35_a0.5 @ gain 50, probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${SM35[@]}" "${COMMON[@]}" \
          --seed "$ps" --encoder_gain 50 --out "$OUT/sm35_a0.5_ps$ps"
    echo "=== half_a0.5 @ gain 75, probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${HALF[@]}" "${COMMON[@]}" \
          --seed "$ps" --encoder_gain 75 --out "$OUT/half_a0.5_ps$ps"
done

echo "DONE $OUT"
