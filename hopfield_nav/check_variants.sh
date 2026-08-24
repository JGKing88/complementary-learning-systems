#!/bin/bash
# Dry-run every VARIANT in run_nav_tri.sh: does the case block resolve, and
# what schedule / rollout shape / noise does it actually produce?
#
# Worth having because the launcher is the one place a wave's design becomes
# real, a typo in a case arm surfaces only when a six-hour job starts, and
# several variants exist for hours before anything runs them.
#
#   bash hopfield_nav/check_variants.sh            # all
#   bash hopfield_nav/check_variants.sh w3_A w3_B  # some

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# Overridable: phase 2 has its own launcher, and pointing this at the wrong
# file makes it report "case arm did not match" for a variant that is perfectly
# fine -- a false negative that sent one debugging session after a grep bug
# that did not exist.
#   SCRIPT=hopfield_nav/run_nav_p2.sh bash hopfield_nav/check_variants.sh p4_tp15
SCRIPT="${SCRIPT:-$HERE/run_nav_tri.sh}"

if [ $# -gt 0 ]; then
    VARIANTS="$*"
else
    # Every literal case label, minus the catch-all. `w3_A|w3_B|...` splits.
    # [A-Za-z] not [a-z]: the wave-3 arms are w3_A..w3_D, and a lowercase-only
    # pattern silently listed zero of them -- which is exactly the class of
    # miss this script exists to catch, so it caught its own.
    VARIANTS=$(grep -oE '^  [A-Za-z0-9_|]+\)' "$SCRIPT" \
               | tr -d ' )' | tr '|' '\n' | grep -v '^\*$' | sort -u)
fi

fail=0
for v in $VARIANTS; do
    # Stop before the trainer runs: `cd "$REPO"` is the last line before the
    # source, so a REPO that does not exist makes the script exit right there
    # with everything already echoed.
    out=$(VARIANT="$v" REPO=/nonexistent-dry-run bash "$SCRIPT" 2>&1)
    if echo "$out" | grep -q 'unknown VARIANT'; then
        echo "FAIL $v: case arm did not match"; fail=1; continue
    fi
    sched=$(echo "$out" | grep -m1 'schedule   :' | sed 's/.*: //')
    shape=$(echo "$out" | grep -m1 'rollout    :' | sed 's/.*: //')
    noise=$(echo "$out" | grep -m1 'noise      :' | sed 's/.*: //')
    shaping=$(echo "$out" | grep -m1 'shaping    :' | sed 's/.*: //')
    if [ -z "$sched" ]; then
        echo "FAIL $v: no schedule echoed"; echo "$out" | tail -3; fail=1; continue
    fi
    printf '%-14s %-46s %s\n                %s\n                %s\n' \
           "$v" "$sched" "$shape" "$noise" "$shaping"
done
exit $fail
