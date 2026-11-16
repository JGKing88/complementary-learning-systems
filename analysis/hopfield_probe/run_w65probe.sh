#!/usr/bin/env bash
# Hopfield probe on the relu-output encoders (w64, w65) against att0.5.
#
# w64_relu_out is att0.5 with a relu output and gain 1; it collapsed to ~5 of
# 1024 active units (unique radius r_min 0, alias ~1.0). w65_relu_density adds a
# floor on effective active units at k 32/64/128, which held exactly and gave
# r_min 3-5, median 6-7, alias 0.93-0.95, against att0.5's 5.0/9.5/0.871. The
# screen has lost to the probe before (Sec 10.12, 10.13, 10.15), so this is
# the measurement that decides.
#
# Beta defaults to each checkpoint's stored gain: 1 for relu (no Hopfield gain,
# by instruction) and 100 for att0.5. Both are far below the D^1.5 ~ 3.3e4
# knee, where beta divides out of the normalised update exactly
# (beta_cancels_check.py), so the two recall regimes are identical.
#
# att0.5 s43/s44 are re-run here so every row comes from the same code. Same
# protocol as run_w61probe.sh: three scaffold draws (probe seeds 0, 1, 2).
#
#SBATCH --job-name=probe_w65
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/w65_relu/logs/probe_w65_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/relu-output-encoder
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/w65_relu

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --device cpu --fwhm_fallback 0.25)

CK=()
add() { CK+=(--ckpt "$1/encoder_final.pt" --label "$2"); }
for s in 43 44; do
    add "$(ls -d "$S"/w52_attract_fwhm/*_att0.5_seed=$s)" "att0.5-s$s"
    add "$(ls -d "$S"/w64_relu_out/*_att0.5_relu_seed=$s)" "relu-s$s"
    for k in 32 64 128; do
        add "$(ls -d "$S"/w65_relu_density/*_relu_k${k}_seed=$s)" "relu_k$k-s$s"
    done
done

for ps in 0 1 2; do
    echo "=== probe seed $ps ==="
    "$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
          --seed "$ps" --out "$OUT/ps$ps"
done

echo "DONE $OUT"
