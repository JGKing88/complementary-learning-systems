#!/usr/bin/env bash
# Encoder-Hopfield probe at the spec's Sec 9 operating point, then the report.
#
# Usage:
#   ./analysis/hopfield_probe/run_probe.sh              # full, Sec 9 defaults
#   SCALE=fast ./analysis/hopfield_probe/run_probe.sh   # ~15 min, same shape
#   sbatch analysis/hopfield_probe/run_probe.sh         # as a job
#
# The whole suite is roughly one node-hour per encoder at SCALE=full, which is
# the argument for running all four level-7 seeds rather than picking one.
#
#SBATCH --job-name=hopfield_probe
#SBATCH --output=%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load miniforge/24.3.0-0 || true
fi
PYTHON="${PYTHON:-/home/jackking/.conda/envs/cls/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

source scripts/cls_env.sh

# --- encoders --------------------------------------------------------------
# Sec 9 #1. All four level-7 seeds, not one: the spread across them is the
# result, and the suite is cheap enough that picking one would only lose
# information. untrained_mlp.pt carries no train_config, which is what
# --fwhm_fallback is for -- it cannot mask a stored value.
V35="${V35:-$CLS_ENCODERS/run_20260422_185816/encoder_best.pt}"
L7_DIR="${L7_DIR:-$CLS_RUNS/sweeps/w53_attract_knee}"
UNTRAINED="${UNTRAINED:-$CLS_ENCODERS/untrained_mlp.pt}"

ARGS=(--ckpt "$V35" --label v35)
for arm in 004_att16_seed=42 005_att16_seed=43 006_att16_seed=44 \
           007_att16_seed=45; do
    seed="${arm##*=}"
    ARGS+=(--ckpt "$L7_DIR/$arm/encoder_final.pt" --label "L7-s$seed")
done
ARGS+=(--ckpt "$UNTRAINED" --label untrained)
ARGS+=(--fwhm_fallback 0.25)

# --- scale -----------------------------------------------------------------
SCALE="${SCALE:-full}"
case "$SCALE" in
    full)   # Sec 9: 50 worlds of 50 envs, K to 50, steps to 15.
        ARGS+=(--n_worlds 50 --n_envs_per_world 50
               --k 1 2 3 5 10 20 50 --steps 1 2 3 5 10 15
               --n_cont_samples 200000 --n_cont_annulus 50000
               --n_alias 20000) ;;
    fast)   # Same shape, ~15 min. For iterating on the report.
        ARGS+=(--n_worlds 4 --n_envs_per_world 20
               --k 1 3 5 10 20 --steps 1 2 3 5 10 15
               --n_cont_samples 60000 --n_cont_annulus 20000
               --n_alias 5000) ;;
    *) echo "SCALE must be 'full' or 'fast', got '$SCALE'" >&2; exit 2 ;;
esac

ARGS+=(--env_size "${SIZE:-20}" --Npos "${NPOS:-1716}"
       --seed "${SEED:-0}" --device "${DEVICE:-cpu}")
[[ "${RESCUE:-0}" == "1" ]] && ARGS+=(--rescue)

OUT="${OUT:-$CLS_RESULTS/hopfield_probe/$(date +%Y%m%d_%H%M%S)}"
ARGS+=(--out "$OUT")

echo "scale=$SCALE  out=$OUT"
"$PYTHON" -m analysis.hopfield_probe.run "${ARGS[@]}"
"$PYTHON" -m analysis.hopfield_probe.report.build "$OUT"

echo
echo "report: $OUT/report/index.html"
