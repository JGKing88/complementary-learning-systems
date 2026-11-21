#!/usr/bin/env bash
# Hopfield navigation probe (run_ideal.sh settings, Hebbian storage, whole arena) for the codes
# listed in analysis/field_kernel/sets/<SET>.txt, one "spec|label" per line. BASE in a spec
# expands to n_freq=512,seed=0,gain=100,relu_mix=1.
#
#   scripts/snapshot_code.sh <tag>                                # commit first
#   SET=a CODE=<tag> bash analysis/field_kernel/run_probe.sh [ARRAY]
#   python -m analysis.field_kernel.summary $OUT                  # the table
#
# REGION="0 0 500" confines the worlds to a corner (default: whole arena).
set -euo pipefail
SET=${SET:-a}
CODE=${CODE:?CODE=<snapshot tag under cls_runs/code>}
WT=/orcd/pool/003/jackking/cls_runs/code/$CODE
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/field_kernel/probe_$SET}
REGION=${REGION:-}
export SET CODE OUT REGION
PY=/home/jackking/.conda/envs/cls/bin/python
mapfile -t ENCS < <(grep -v '^\s*$' "$WT/analysis/field_kernel/sets/$SET.txt")

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name="fk_probe_$SET" --partition=pi_evelina9,ou_bcs_normal,mit_normal \
        --time=3:00:00 --cpus-per-task=16 --mem=32G \
        --array="${1:-0-$(( ${#ENCS[@]} - 1 ))}%${THROTTLE:-4}" \
        --output="$OUT/logs/probe_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  code $CODE  $(cat GIT_COMMIT)"
IFS='|' read -r CK label <<< "${ENCS[$i]}"
CK=${CK/BASE/n_freq=512,seed=0,gain=100,relu_mix=1}
REG=()
[[ -n "$REGION" ]] && REG=(--world_region $REGION)

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$label" ${REG[@]+"${REG[@]}"} \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --storage_rule hebb \
    --out "$OUT/t$i"
echo "DONE probe $i  $label -> $OUT/t$i"
