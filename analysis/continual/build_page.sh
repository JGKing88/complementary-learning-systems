#!/bin/bash
# Regenerate the results data and the page, in one place.
#
# The two commands were previously typed out each time, which is how the
# `--identifiability` argument would come to be passed on one regeneration and
# forgotten on the next -- and a page that silently drops a section is worse
# than one that fails to build.
set -euo pipefail

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

# Callers reach this from a login shell, an sbatch script, and an agent session,
# and only two of those three have the environment's python on PATH.
PY="${PY:-$(command -v python || echo /home/jackking/.conda/envs/cls/bin/python)}"

OUT_DIR="${1:-$CLS_RESULTS}"
DATA="$OUT_DIR/continual_results.json"
PAGE="$OUT_DIR/continual_results.html"
mkdir -p "$OUT_DIR"

# The forgetting panel needs each arm's 8 seeds merged into one multi-iter
# history. Done here rather than by hand so the panel regenerates from the raw
# histories like every other number on the page -- a figure built from files
# somebody merged once is exactly the kind of thing that stops matching its
# runs. The arm list comes from results_data so there is one source of truth.
MERGED="$CLS_HISTORIES/merged"
mkdir -p "$MERGED"
STAIR_ARMS=$("$PY" -c "from analysis.continual.results_data import STAIRCASE_ARMS; print(' '.join(a for a,_,_ in STAIRCASE_ARMS))")
for A in $STAIR_ARMS; do
    N=$(ls "$CLS_HISTORIES"/wave1/"${A}"_s*.json 2>/dev/null | wc -l)
    if [[ "$N" -gt 0 ]]; then
        "$PY" -u -m analysis.continual.merge_histories \
            --inputs "$CLS_HISTORIES"/wave1/"${A}"_s*.json \
            --out "$MERGED/${A}.json" --run_name "$A" >/dev/null
    else
        echo "[build_page] note: no seed files for $A; panel will omit it"
    fi
done

"$PY" -u -m analysis.continual.results_data \
    --wave0_dir     "$CLS_HISTORIES/wave0" \
    --wave1_dir     "$CLS_HISTORIES/wave1" \
    --recorded_dir  "$CLS_HISTORIES" \
    --runs_root     "$CLS_RUNS" \
    --n20_dir       "$CLS_HISTORIES/n20" \
    --incontext_dir "$CLS_HISTORIES/incontext" \
    --identifiability "$CLS_RESULTS/task_identifiability.json" \
    --incontext_generalization "$CLS_RESULTS/incontext_generalization.json" \
    --incontext_upper_bound "$CLS_RESULTS/incontext_upper_bound.json" \
    --staircase_dir "$MERGED" \
    --out "$DATA"

# The discrete suite, built through the identical pipeline with its own
# directories and its own joint tag. --joint_tag is load-bearing: the T0.1 runs
# live under runs/rnn/ rather than in the histories directory, so pointing only
# the history dirs at the discrete wave would leave a Gaussian-head ceiling as
# the denominator of a Categorical page.
DATA_D="$OUT_DIR/continual_results_discrete.json"
if [[ -d "$CLS_HISTORIES/wave1d" ]]; then
    "$PY" -u -m analysis.continual.results_data \
        --wave0_dir     "$CLS_HISTORIES/wave0d" \
        --wave1_dir     "$CLS_HISTORIES/wave1d" \
        --recorded_dir  "$CLS_HISTORIES" \
        --runs_root     "$CLS_RUNS" \
        --joint_tag     wave0d \
        --out "$DATA_D"
else
    echo "[build_page] no discrete wave yet; page will carry one action space"
    DATA_D=""
fi

"$PY" -u -m analysis.continual.results_page --data "$DATA" \
    ${DATA_D:+--data_discrete "$DATA_D"} --out "$PAGE"
"$PY" -u -m analysis.continual.validate_page "$PAGE"

echo
echo "data: $DATA"
[[ -n "$DATA_D" ]] && echo "data (discrete): $DATA_D"
echo "page: $PAGE"
