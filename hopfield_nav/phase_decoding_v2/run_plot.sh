#!/bin/bash -l
# Standalone plotter — re-renders bars from saved metrics.json files.
# Runs in seconds on CPU; no slurm required.
#
# Single run:
#   METRICS=path/to/run_dir bash run_plot.sh
#   METRICS=path/to/metrics.json OUT=mybars.png bash run_plot.sh
#
# Compare multiple runs (space-separated METRICS, space-separated LABELS):
#   METRICS="trained_dir random_dir" \
#   LABELS="trained random_init" \
#   OUT=trained_vs_random.png \
#   bash run_plot.sh
#
# Or invoke the module directly:
#   python -m hopfield_nav.phase_decoding_v2.plot \
#       --metrics A.json B.json --labels trained random --out cmp.png

##METRICS=/home/jackking/cls/hopfield_nav/phase_decoding_v2/results/exp1_phase_a_only_glamorous-field-97_phase_a_u360_stochastic_random_init_seed0 bash /home/jackking/cls/hopfield_nav/phase_decoding_v2/run_plot.sh

set -euo pipefail

module load miniforge/24.3.0-0 2>/dev/null || true
source activate cls 2>/dev/null || true

cd /home/jackking/cls

METRICS="${METRICS:?METRICS is required (space-separated paths to metrics.json or run dirs)}"
LABELS="${LABELS:-}"

if [ -z "${OUT:-}" ]; then
    # Default: bars.png next to the first metrics path. For multi-run, put it
    # in the parent of the first run dir so it doesn't get clobbered.
    FIRST="$(echo "$METRICS" | awk '{print $1}')"
    if [ -d "$FIRST" ]; then
        OUT="$FIRST/bars.png"
    else
        OUT="$(dirname "$FIRST")/bars.png"
    fi
fi

LABELS_FLAG=""
if [ -n "$LABELS" ]; then
    # shellcheck disable=SC2086
    LABELS_FLAG="--labels $LABELS"
fi

echo "[run_plot] metrics=$METRICS out=$OUT labels=$LABELS"

# shellcheck disable=SC2086
python -m hopfield_nav.phase_decoding_v2.plot \
    --metrics $METRICS \
    --out "$OUT" \
    $LABELS_FLAG
