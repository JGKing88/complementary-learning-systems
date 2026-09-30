#!/usr/bin/env bash
# Recall dynamics and saturated recall for the ideal encoder (r = 16).
#
#   dyn   CPU array 0-3, one task per header in ideal_dynamics_check.HEADERS
#         (ideal r=16 beta=100 / beta=1e6, att0.5 s42 beta=100, arm B):
#         fixed points under hebb/proj, the alpha walk, the chord.
#   sat   CPU array 0-4: run.py with --beta 1e6 (recall tanh saturated), same
#         settings as run_ideal.sh. 0-3 = ideal r=16 over whole / corner /
#         centre / opposite; 4 = att0.5 s42 over the whole arena (paired).
#
#   bash analysis/hopfield_probe/run_ideal_dyn.sh dyn [ARRAY]
#   bash analysis/hopfield_probe/run_ideal_dyn.sh sat [ARRAY]
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${IDEAL_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export IDEAL_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/ideal_encoder}

MODE=${1:-dyn}
ARG=${2:-}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    if [[ "$MODE" == dyn ]]; then
        exec sbatch --job-name=ideal_dyn --partition=ou_bcs_normal \
            --time=5:00:00 --cpus-per-task=8 --mem=24G \
            --array="${ARG:-0-3}" \
            --output="$OUT/logs/dyn_%A_%a.out" "$0" dyn
    else
        exec sbatch --job-name=ideal_sat --partition=ou_bcs_normal \
            --time=4:00:00 --cpus-per-task=16 --mem=32G \
            --array="${ARG:-0-4}" \
            --output="$OUT/logs/sat_%A_%a.out" "$0" sat
    fi
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID

if [[ "$MODE" == dyn ]]; then
    "$PY" -m analysis.hopfield_probe.ideal_dynamics_check \
        --index "$i" --k 5 --max_steps 30 --out "$OUT/dynamics"
    echo "DONE dyn $i -> $OUT/dynamics"
    exit 0
fi

REGIONS=("|whole" "0 0 500|corner" "608 608 500|centre"
         "1216 1216 500|opposite")
if (( i < 4 )); then
    CK="ideal:r=16,n_freq=512,seed=0,gain=100"; elabel="ideal r=16 β=1e6"
    IFS='|' read -r region rlabel <<< "${REGIONS[$i]}"
else
    CK="$S/w52_attract_fwhm/000_att0.5_seed=42/encoder_final.pt"
    elabel="att0.5 s42 β=1e6"; region=""; rlabel="whole"
fi
REG=()
[[ -n "$region" ]] && REG=(--world_region $region)

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$elabel | $rlabel" ${REG[@]+"${REG[@]}"} \
    --beta 1e6 \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --out "$OUT/sat/t$i"

echo "DONE sat $i  $elabel  region=$rlabel -> $OUT/sat/t$i"
