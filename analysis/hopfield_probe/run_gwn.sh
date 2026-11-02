#!/usr/bin/env bash
# Sec 20: the grid wave network (gwn_encoder.py).
#
#   build   one GPU job: gwn_check step0, gwn_l1_learn (step 1), gwn_check
#           step2, step3 -- kernel metrics, saving the encoders the probe uses
#   probe   CPU array: run.py (run_ideal.sh settings, hebb) on the saved
#           encoders; the main network in 4 regions, the rest whole + corner.
#           Compare with ideal r=16 at ideal_encoder/probe/t12-t15.
#
#   bash analysis/hopfield_probe/run_gwn.sh build
#   bash analysis/hopfield_probe/run_gwn.sh probe [ARRAY]     # default 0-21
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${GWN_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export GWN_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/gwn}
E=$OUT/encoders
MODE=${1:-build}
ARG=${2:-}

# encoder | label | regions (w = whole, c = corner, all = 4)
PROBES=(
    "exact_400k_gauss|exact 400k gauss|all"
    "exact_all_gauss|exact all-triples gauss|wc"
    "exact_400k_cut|exact 400k cut|wc"
    "exact_400k_rational|exact 400k rational|wc"
    "exact_400k_linear|exact 400k linear|wc"
    "learned_pca_fwhm0.25|PCA L1 bump 0.25|wc"
    "learned_move100k_fwhm0.25|movement L1 bump 0.25|wc"
    "learned_pca_fwhm0.1|PCA L1 bump 0.10|wc"
    "learned_move100k_fwhm0.1|movement L1 bump 0.10|wc"
    "exact_400k_coef0.1|exact 400k coef error 0.1|wc"
)
TASKS=()
for p in "${PROBES[@]}"; do
    IFS='|' read -r enc lab regs <<< "$p"
    if [[ $regs == all ]]; then
        for r in "|whole" "0 0 500|corner" "608 608 500|centre" "1216 1216 500|opposite"; do
            TASKS+=("$enc|$lab|$r"); done
    else
        TASKS+=("$enc|$lab||whole" "$enc|$lab|0 0 500|corner"); fi
done

if [[ -z "${GWN_IN_JOB:-}" ]]; then
    mkdir -p "$OUT/logs"
    if [[ $MODE == build ]]; then
        GWN_IN_JOB=1 exec sbatch --job-name=gwn_build --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal_gpu \
            --gres=gpu:1 --exclude=node3804 --time=${GWN_TIME:-5:00:00} --cpus-per-task=2 \
            --mem=${GWN_MEM:-32G} --output="$OUT/logs/build_%j.out" "$0" build
    else
        GWN_IN_JOB=1 exec sbatch --job-name=gwn_probe --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal \
            ${GWN_DEP:+--dependency=$GWN_DEP} --time=4:00:00 --cpus-per-task=16 --mem=32G --array="${ARG:-0-$(( ${#TASKS[@]} - 1 ))}" \
            --output="$OUT/logs/probe_%A_%a.out" "$0" probe
    fi
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

if [[ $MODE == build ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_check step0 --out "$OUT"; echo "DONE step0"
    "$PY" -m analysis.hopfield_probe.gwn_l1_learn --out "$OUT"; echo "DONE step1"
    "$PY" -m analysis.hopfield_probe.gwn_check step2 --out "$OUT"; echo "DONE step2"
    "$PY" -m analysis.hopfield_probe.gwn_check step3 --out "$OUT"; echo "DONE step3"
    exit 0
fi

i=$SLURM_ARRAY_TASK_ID
IFS='|' read -r enc lab region rlabel <<< "${TASKS[$i]}"
REG=(); [[ -n "$region" ]] && REG=(--world_region $region)
"$PY" -m analysis.hopfield_probe.run \
    --ckpt "gwn:$E/$enc.pt" --label "$lab | $rlabel" ${REG[@]+"${REG[@]}"} \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 --storage_rule hebb \
    --out "$OUT/probe/t$i"
echo "DONE probe $i  $lab  region=$rlabel -> $OUT/probe/t$i"
