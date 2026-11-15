#!/usr/bin/env bash
# Non-negative ideal code: relu(Q z) (ideal_encoder.py relu_mix=1), the ideal's units remixed by a
# fixed orthogonal Walsh-Hadamard rotation and rectified, against the plain ideal through the
# identical probe (run_ideal.sh settings, Hebbian storage).
#
#   plain ideal r=16          reference, rerun here so every row shares one commit
#   relu_mix r=16             non-negative activity, production memory
#   relu_mix r=16 center      the code's mean vector removed (= mean-subtracted storage and cue
#                             in this linear memory); activity itself stays non-negative
#   relu_mix r=17.9 (center)  r raised so the peak's half-width matches the ideal's 19.4 cells
#
# SET=b (OUT relu_mix_b): one global inhibition constant (center_global=1) in place of the
# per-unit mean, and a sparser code with 10% of units active (active=0.1), with and without the
# constant, at r = 16 and at r = 23.2 (the sparse code's half-width matched to the ideal's 19.4).
#
# Index = 2 * encoder + region (whole, corner). Throttled to 4 tasks at a time.
#
#   bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]          # default 0-9
#   SET=b bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]
#   python -m analysis.hopfield_probe.ideal_summary "$OUT"        # the table
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${RELU_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RELU_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
SET=${SET:-a}
export SET
B="n_freq=512,seed=0,gain=100"

if [[ "$SET" == b ]]; then
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_mix_b}
ENCS=(
    "ideal:r=16,$B,relu_mix=1,center_global=1|relu r=16 global inh"
    "ideal:r=16,$B,relu_mix=1,active=0.1|relu10% r=16"
    "ideal:r=16,$B,relu_mix=1,active=0.1,center_global=1|relu10% r=16 global inh"
    "ideal:r=23.2,$B,relu_mix=1,active=0.1|relu10% r=23.2"
    "ideal:r=23.2,$B,relu_mix=1,active=0.1,center_global=1|relu10% r=23.2 global inh"
)
else
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_mix}
ENCS=(
    "ideal:r=16,$B|ideal r=16"
    "ideal:r=16,$B,relu_mix=1|relu_mix r=16"
    "ideal:r=16,$B,relu_mix=1,center=1|relu_mix r=16 center"
    "ideal:r=17.9,$B,relu_mix=1|relu_mix r=17.9"
    "ideal:r=17.9,$B,relu_mix=1,center=1|relu_mix r=17.9 center"
)
fi
REGIONS=("|whole" "0 0 500|corner")

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=relu_mix_probe --partition=pi_evelina9,ou_bcs_normal \
        --time=4:00:00 --cpus-per-task=16 --mem=32G \
        --array="${1:-0-9}%4" \
        --output="$OUT/logs/probe_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
IFS='|' read -r CK elabel <<< "${ENCS[$(( i / 2 ))]}"
IFS='|' read -r region rlabel <<< "${REGIONS[$(( i % 2 ))]}"
REG=()
[[ -n "$region" ]] && REG=(--world_region $region)

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$elabel | $rlabel" ${REG[@]+"${REG[@]}"} \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --storage_rule hebb \
    --out "$OUT/probe/t$i"

echo "DONE probe $i  $elabel  region=$rlabel -> $OUT/probe/t$i"
