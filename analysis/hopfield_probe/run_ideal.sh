#!/usr/bin/env bash
# The analytic ideal encoder (ideal_encoder.py: 2D random Fourier features of a
# Gaussian kernel, read off the grid code, D=1024) at r = 2, 4, 8, 16, beside
# the ladder's production encoder (w52_attract_fwhm att0.5, seeds 42 and 43;
# gain 100, fwhm 0.25, 118 patches of 50, batch 4096, exclude_cross_env_pairs
# -- from ckpt["train_config"] / meta.json) through the identical pipeline.
#
#   probe   CPU array, index = 4 * encoder + region: run.py (exact, basin,
#           reach, acc45) with the worlds over the whole arena, or confined to
#           the three 500-cell squares of run_corner.sh (corner / centre /
#           opposite). Same settings as run_corner.sh and run_proj.sh, so the
#           att0.5 rows are comparable to THEORY Sec 3.7 and the Sec 7.1 hebb
#           controls.
#   scan    CPU array, one task per encoder: ideal_scan_check.py (d_eff, far
#           sd, alias rate, and C(1) / r_0.9 / r_mono / r_u16 / whole-scaffold
#           alias ceiling at 8 references in each of 5 distance bands).
#
#   bash analysis/hopfield_probe/run_ideal.sh probe [ARRAY]    # default 0-23
#   bash analysis/hopfield_probe/run_ideal.sh scan  [ARRAY]    # default 0-5
#   python -m analysis.hopfield_probe.ideal_summary "$OUT"      # the table
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

# sbatch runs a spooled copy of this file, so the worktree is resolved at
# submit time and carried into the job through the environment.
WT=${IDEAL_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export IDEAL_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/ideal_encoder}

MODE=${1:-probe}
ARG=${2:-}

ENCS=(
    "ideal:r=2,n_freq=512,seed=0,gain=100|ideal r=2"
    "ideal:r=4,n_freq=512,seed=0,gain=100|ideal r=4"
    "ideal:r=8,n_freq=512,seed=0,gain=100|ideal r=8"
    "ideal:r=16,n_freq=512,seed=0,gain=100|ideal r=16"
    "$S/w52_attract_fwhm/000_att0.5_seed=42/encoder_final.pt|att0.5 s42"
    "$S/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt|att0.5 s43"
    "ideal:r=32,n_freq=512,seed=0,gain=100|ideal r=32"
    "ideal:r=48,n_freq=512,seed=0,gain=100|ideal r=48"
    "ideal:r=16,n_freq=512,seed=0,gain=100,weights=/orcd/pool/003/jackking/cls_runs/results/ideal_net/lsq_weights_r16_seed0.npy|ideal r=16 lsq"
    "ideal:r=16,n_freq=512,seed=0,gain=100,freqs=/orcd/pool/003/jackking/cls_runs/results/ideal_net/topk/freqs_512.npy,weights=/orcd/pool/003/jackking/cls_runs/results/ideal_net/topk/weights_512.npy|table top512 refit"
)
REGIONS=("|whole" "0 0 500|corner" "608 608 500|centre"
         "1216 1216 500|opposite")

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    if [[ "$MODE" == probe ]]; then
        exec sbatch --job-name=ideal_probe --partition=ou_bcs_normal \
            --time=4:00:00 --cpus-per-task=16 --mem=32G \
            --array="${ARG:-0-23}" \
            --output="$OUT/logs/probe_%A_%a.out" "$0" probe
    else
        exec sbatch --job-name=ideal_scan --partition=ou_bcs_normal \
            --time=${SCAN_TIME:-3:00:00} --cpus-per-task=8 --mem=24G \
            --array="${ARG:-0-5}" \
            --output="$OUT/logs/scan_%A_%a.out" "$0" scan
    fi
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID

if [[ "$MODE" == scan ]]; then
    IFS='|' read -r CK elabel <<< "${ENCS[$i]}"
    "$PY" -m analysis.hopfield_probe.ideal_scan_check \
        --ckpt "$CK" --label "$elabel" --n_per_band 8 --seed 0 \
        --out "$OUT/scan"
    echo "DONE scan $i  $elabel -> $OUT/scan"
    exit 0
fi

enc_i=$(( i / 4 ))
reg_i=$(( i % 4 ))
IFS='|' read -r CK elabel <<< "${ENCS[$enc_i]}"
IFS='|' read -r region rlabel <<< "${REGIONS[$reg_i]}"
REG=()
[[ -n "$region" ]] && REG=(--world_region $region)

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$elabel | $rlabel" ${REG[@]+"${REG[@]}"} \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --out "$OUT/probe/t$i"

echo "DONE probe $i  $elabel  region=$rlabel -> $OUT/probe/t$i"
