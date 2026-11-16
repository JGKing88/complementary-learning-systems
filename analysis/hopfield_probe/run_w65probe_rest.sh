#!/usr/bin/env bash
# Finish probe seed 2 of run_w65probe.sh.
#
# That job asked for 2 h on the strength of run_w61probe.sh (24 encoder-seeds
# in 20 min), but on node1802 each encoder-seed took ~260 s, so probe seed 2
# was cut off at the time limit. Submit with --dependency=afterany on that job:
# at start this probes every label with no ps2/<label>.json yet, into
# ps2_rest/ so that directory's manifest describes exactly its own files.
#
#SBATCH --job-name=probe_w65r
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/w65_relu/logs/probe_w65r_%j.out
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
add() {
    if [ -f "$OUT/ps2/$2.json" ]; then
        echo "have ps2/$2.json"
    else
        echo "todo $2"
        CK+=(--ckpt "$1/encoder_final.pt" --label "$2")
    fi
}
for s in 43 44; do
    add "$(ls -d "$S"/w52_attract_fwhm/*_att0.5_seed=$s)" "att0.5-s$s"
    add "$(ls -d "$S"/w64_relu_out/*_att0.5_relu_seed=$s)" "relu-s$s"
    for k in 32 64 128; do
        add "$(ls -d "$S"/w65_relu_density/*_relu_k${k}_seed=$s)" "relu_k$k-s$s"
    done
done

if [ ${#CK[@]} -eq 0 ]; then
    echo "nothing to do"
    exit 0
fi
"$PY" -m analysis.hopfield_probe.run "${CK[@]}" "${COMMON[@]}" \
      --seed 2 --out "$OUT/ps2_rest"
echo "DONE $OUT/ps2_rest"
