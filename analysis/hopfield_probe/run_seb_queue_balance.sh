#!/usr/bin/env bash
# w69_seb_queue_balance: four arms x seeds 42-44, same settings as
# run_seb_att05.sh (gain 100, beta = gain). Reference rows: att0.5 and w64 in
# results/probe_seb_att05/g100, w66 in results/probe_seb_queue_r/g100.
#
#SBATCH --job-name=probe_seb_balance
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/probe_seb_queue_balance/slurm_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/seb-att05

PY=/home/jackking/.conda/envs/cls/bin/python
NEW=/orcd/pool/003/jackking/cls_runs/sweeps/w69_seb_queue_balance
OUT=/orcd/pool/003/jackking/cls_runs/results/probe_seb_queue_balance

COMMON=(--n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20
        --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716
        --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000
        --seed 0 --device cpu --fwhm_fallback 0.25)

CK=()
for a in bal40k bal10k q40k_lr1e-4 bal40k_lr1e-4; do
    for s in 42 43 44; do
        d=$(ls -d "$NEW"/*_${a}_seed=$s)
        CK+=(--ckpt "$d/encoder_final.pt" --label "$a-s$s")
    done
done

"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" --out "$OUT/g100"
"$PY" -m analysis.hopfield_probe.report.build "$OUT/g100"
echo "DONE $OUT"
