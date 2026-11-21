#!/usr/bin/env bash
# Spectrum of the trained encoder against the ideal (trained_spectrum.py): one GPU job that
# encodes the whole torus for each encoder and saves S(n) and the translation-test samples.
#
#   bash analysis/hopfield_probe/run_spectrum.sh            # submit
#   bash analysis/hopfield_probe/run_spectrum.sh ideal att42 # only these encoders
#
# Runs from THIS worktree: do not delete it while the job is live.
set -euo pipefail

WT=${SPEC_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export SPEC_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/spectrum}

if [[ -z "${SPEC_IN_JOB:-}" ]]; then
    mkdir -p "$OUT/logs"
    SPEC_IN_JOB=1 exec sbatch --job-name=spectrum \
        --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal_gpu --gres=gpu:1 --exclude=node3804 \
        --time=${SPEC_TIME:-2:00:00} --cpus-per-task=4 --mem=${SPEC_MEM:-48G} \
        --output="$OUT/logs/spectrum_%j.out" "$0" "$@"
fi

cd "$WT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ONLY=()
[[ $# -gt 0 ]] && ONLY=(--only "$@")
$PY -u -m analysis.hopfield_probe.trained_spectrum --out "$OUT" "${ONLY[@]}"
