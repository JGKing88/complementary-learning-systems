#!/usr/bin/env bash
# Run the live test suite (hopfield_nav/tests).
#
# This is the gate every refactor phase must leave green. Test paths and
# default options come from [tool.pytest.ini_options] in pyproject.toml, so
# `pytest` alone from the repo root is equivalent -- this script exists to pin
# the interpreter and to give sbatch/CI one thing to call.
#
# The root tests/ directory is NOT run here: it exercises only the legacy `cls`
# package, and does not currently collect (tests/test_vectorized_generate.py
# imports a top-level `train` module that no longer exists). It is retired
# together with cls/ in phase 7 of the 2026-08 refactor.
#
# Usage:
#   ./run_tests.sh                       # whole live suite
#   ./run_tests.sh -k at_goal            # any pytest args are forwarded
#   ./run_tests.sh hopfield_nav/tests/test_audit.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/jackking/.conda/envs/cls/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

exec "$PYTHON" -m pytest "$@"
