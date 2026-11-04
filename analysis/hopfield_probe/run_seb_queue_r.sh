#!/usr/bin/env bash
# w66: as w65 with rate_lambda scaled by B/4096; a 10k / 40k queue of past codes for the
# rate term, four seeds each. Same settings as run_seb_att05.sh (gain 100,
# beta = gain). The att0.5 and w64 reference rows are in
# results/probe_seb_att05/g100, from the same probe code.
#
#SBATCH --job-name=probe_seb_queue_r
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/probe_seb_queue_r/slurm_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/seb-att05

PY=/home/jackking/.conda/envs/cls/bin/python
NEW=/orcd/pool/003/jackking/cls_runs/sweeps/w66_seb_queue_rescaled
OUT=/orcd/pool/003/jackking/cls_runs/results/probe_seb_queue_r

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

CK=()
for q in q10k_r q40k_r; do
    for s in 42 43 44 45; do
        d=$(ls -d "$NEW"/*_${q}_seed=$s)
        CK+=(--ckpt "$d/encoder_final.pt" --label "$q-s$s")
    done
done

"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" --out "$OUT/g100"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/g100"
echo "DONE $OUT"
