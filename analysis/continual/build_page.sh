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

"$PY" -u -m analysis.continual.results_data \
    --wave0_dir     "$CLS_HISTORIES/wave0" \
    --wave1_dir     "$CLS_HISTORIES/wave1" \
    --recorded_dir  "$CLS_HISTORIES" \
    --runs_root     "$CLS_RUNS" \
    --n20_dir       "$CLS_HISTORIES/n20" \
    --incontext_dir "$CLS_HISTORIES/incontext" \
    --identifiability "$CLS_RESULTS/task_identifiability.json" \
    --out "$DATA"

"$PY" -u -m analysis.continual.results_page --data "$DATA" --out "$PAGE"
"$PY" -u -m analysis.continual.validate_page "$PAGE"

echo
echo "data: $DATA"
echo "page: $PAGE"
