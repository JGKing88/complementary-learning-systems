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
# The two action spaces ran different methods, so each panel needs its own arm
# list; both come from results_data so there is one source of truth.
merge_set () {   # merge_set <histories subdir> <merged out dir>
    local src="$CLS_HISTORIES/$1" dst="$2"
    mkdir -p "$dst"
    # The stems come from the runs themselves: each union arm's best usable
    # configuration in *this* directory, picked by the same rule the frontier
    # table uses. That is what lets one arm list serve both action spaces even
    # though the same method's best setting is a different string in each.
    local arms
    arms=$("$PY" -c "from analysis.continual.results_data import staircase_stems; print(' '.join(a for a,_,_ in staircase_stems('$src')))")
    local A N
    for A in $arms; do
        N=$(ls "$src/${A}"_s*.json 2>/dev/null | wc -l)
        if [[ "$N" -gt 0 ]]; then
            "$PY" -u -m analysis.continual.merge_histories \
                --inputs "$src/${A}"_s*.json \
                --out "$dst/${A}.json" --run_name "$A" >/dev/null
        else
            echo "[build_page] note: no seed files for $A in $1; panel omits it"
        fi
    done
}

MERGED="$CLS_HISTORIES/merged"
merge_set wave1 "$MERGED"

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
# Every page beyond the canonical one: key, its histories dir, the wave-0 dir
# its floor and ceiling come from, and the joint tag for T0.1. The from-scratch
# pages share wave0's axes -- the oracle and the joint ceiling are properties of
# the task and the architecture, not of where the weights started.
#
# A fifth field names the matching prev_action=ON histories, where they exist:
# the flag only takes effect without a checkpoint, so only the from-scratch
# pages can show what the channel is worth.
#
#   key            hist dir     wave0 dir   joint tag   prev_action dir
EXTRA_PAGES="
discrete:wave1d:wave0d:wave0d:
continuous_fs:wave1_fs:wave0:wave0:wave1_fsp
discrete_fs:wave1d_fs:wave0d:wave0d:wave1d_fsp
"

PAGE_ARGS=()
while IFS=: read -r KEY HDIR W0DIR JTAG PDIR; do
    [[ -z "$KEY" ]] && continue
    if [[ ! -d "$CLS_HISTORIES/$HDIR" ]]; then
        echo "[build_page] $KEY: no $HDIR yet; page omitted"
        continue
    fi
    N=$(ls "$CLS_HISTORIES/$HDIR"/*.json 2>/dev/null | wc -l)
    if [[ "$N" -eq 0 ]]; then
        echo "[build_page] $KEY: $HDIR is empty; page omitted"
        continue
    fi
    MDIR="$CLS_HISTORIES/merged_$KEY"
    merge_set "$HDIR" "$MDIR"
    PREV_ARG=()
    if [[ -n "${PDIR:-}" && -d "$CLS_HISTORIES/$PDIR" ]]; then
        PREV_ARG=(--prev_dir "$CLS_HISTORIES/$PDIR")
    fi
    DPATH="$OUT_DIR/continual_results_${KEY}.json"
    "$PY" -u -m analysis.continual.results_data \
        --wave0_dir     "$CLS_HISTORIES/$W0DIR" \
        --wave1_dir     "$CLS_HISTORIES/$HDIR" \
        --recorded_dir  "$CLS_HISTORIES" \
        --runs_root     "$CLS_RUNS" \
        --joint_tag     "$JTAG" \
        --staircase_dir "$MDIR" \
        "${PREV_ARG[@]}" \
        --out "$DPATH"
    PAGE_ARGS+=(--page "${KEY}=${DPATH}")
done <<< "$EXTRA_PAGES"

"$PY" -u -m analysis.continual.results_page --data "$DATA" \
    "${PAGE_ARGS[@]}" --out "$PAGE"
"$PY" -u -m analysis.continual.validate_page "$PAGE"

echo
echo "data: $DATA"
for a in "${PAGE_ARGS[@]}"; do
    [[ "$a" == --page ]] || echo "data: $a"
done
echo "page: $PAGE"
