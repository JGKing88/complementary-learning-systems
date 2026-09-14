#!/usr/bin/env bash
# Does an encoder trained on one 500x500 corner work outside it? Two jobs.
#
#   corner_check   one GPU: the full-arena cosine map at 40 reference positions
#                  per encoder, binned by distance from the corner (kernel
#                  width, unique radius, alias ceiling and where it sits).
#   probe          a CPU array: the probe suite (basin, direction, reach,
#                  alias) with the worlds confined to one of three 500-cell
#                  squares -- the training corner, the arena centre, and the
#                  opposite corner -- for each of the six trained encoders.
#
# Encoders: w62_corner corner500 / scatter100 at seeds 42, 43, and the same
# recipe on 118 scattered patches (w53 att16) as the full-budget reference.
# encoder_final.pt throughout: encoder_best.pt is chosen by a whole-arena
# unique-radius eval, which for corner500 would let the unseen region pick the
# checkpoint.
#
#   sbatch analysis/hopfield_probe/run_corner.sh check
#   sbatch analysis/hopfield_probe/run_corner.sh probe
#
# The probe array is 18 tasks: index = 3 * encoder + region.
set -euo pipefail

WT=/orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec
PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260914
TMP=/home/jackking/.claude/jobs/d05f5770/tmp

MODE=${1:-check}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Submit ourselves with the right resources for the mode.
    if [[ "$MODE" == check ]]; then
        exec sbatch --job-name=corner_check --partition=ou_bcs_normal \
            --time=1:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=16G \
            --output="$TMP/corner_check_%j.out" "$0" check
    else
        exec sbatch --job-name=corner_probe --partition=ou_bcs_normal \
            --time=4:00:00 --cpus-per-task=16 --mem=32G --array=0-17 \
            --output="$TMP/corner_probe_%A_%a.out" "$0" probe
    fi
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [[ "$MODE" == check ]]; then
    "$PY" -m analysis.hopfield_probe.corner_check --n_per_band 8 --seed 0 \
        --out "$OUT/corner_check"
    echo "DONE corner_check -> $OUT/corner_check"
    exit 0
fi

i=$SLURM_ARRAY_TASK_ID
enc_i=$(( i / 3 ))
reg_i=$(( i % 3 ))

ENCS=(
    "w62_corner/000_corner500_seed=42|corner500 · s42"
    "w62_corner/001_corner500_seed=43|corner500 · s43"
    "w62_corner/002_scatter100_seed=42|scatter100 · s42"
    "w62_corner/003_scatter100_seed=43|scatter100 · s43"
    "w53_attract_knee/004_att16_seed=42|scatter118 · s42"
    "w53_attract_knee/005_att16_seed=43|scatter118 · s43"
)
REGIONS=("0 0 500|corner" "608 608 500|centre" "1216 1216 500|opposite")

IFS='|' read -r rel elabel <<< "${ENCS[$enc_i]}"
IFS='|' read -r region rlabel <<< "${REGIONS[$reg_i]}"
CK="$S/$rel/encoder_final.pt"

# Same settings as run_proj.sh, so the numbers sit beside the ladder's.
# shellcheck disable=SC2086
"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$elabel · $rlabel" \
    --world_region $region \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --out "$OUT/corner_probe/t$i"

echo "DONE task $i  $elabel  region=$rlabel  -> $OUT/corner_probe/t$i"
