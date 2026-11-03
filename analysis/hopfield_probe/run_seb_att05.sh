#!/usr/bin/env bash
# att0.5 with one env per gradient step (w64_seb_att0.5) against att0.5
# (w52_attract_fwhm), four seeds each, in one probe call so both sides run
# the same code. Settings are run_att_low.sh's gain-100 arm: own gain 100,
# beta = gain, no saturation.
#
#SBATCH --job-name=probe_seb_att05
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/probe_seb_att05/slurm_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/seb-att05

PY=/home/jackking/.conda/envs/cls/bin/python
REF=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm
NEW=/orcd/pool/003/jackking/cls_runs/sweeps/w64_seb_att0.5
OUT=/orcd/pool/003/jackking/cls_runs/results/probe_seb_att05

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

CK=()
for s in 42 43 44 45; do
    d=$(ls -d "$NEW"/*_seb_att0.5_seed=$s)
    CK+=(--ckpt "$d/encoder_final.pt" --label "seb-s$s")
    d=$(ls -d "$REF"/*_att0.5_seed=$s)
    CK+=(--ckpt "$d/encoder_final.pt" --label "att0.5-s$s")
done

"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" --out "$OUT/g100"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/g100"
echo "DONE $OUT"
