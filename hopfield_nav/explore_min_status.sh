#!/bin/bash
# One-line-per-eval view of the explore-min wave.
#
#   bash hopfield_nav/explore_min_status.sh            # every eval of every job
#   LAST=1 bash hopfield_nav/explore_min_status.sh     # only each job's latest
#   ONLY="c2s42 e2s42" bash ...                        # just these variants
#
# LAST=1 exists because wave 2 is 13 jobs x 40 evals: the full view is 500+
# rows, which is the right thing for reading one run's trajectory and the wrong
# thing for answering "where is the ladder now".
#
# cov/union are read at the two eval distractor levels, so the gap between the
# two columns IS the distractor-robustness number -- the thing that would
# otherwise need a separate pass to see.

# Logs live under $CLS_RUNS and are shared by every checkout, so this resolves
# the same place from a worktree as from the shared tree.
cd "${REPO_DIR:-/home/jackking/cls}/hopfield_nav/logs" || exit 1

printf "%-5s %-9s %-6s %7s %7s %7s %7s %6s %6s\n" \
       variant job update cov_d0 cov_d10 uni_d0 uni_d10 "d-gap" "s/u"

for f in slurm_explore_min_*.out; do
    [ -e "$f" ] || continue
    job=${f#slurm_explore_min_}; job=${job%.out}
    # [A-Za-z0-9]: variant names are not all lowercase (e4L), and a lowercase
    # class silently matched nothing rather than erroring.
    variant=$(sed -n 's/^=== variant=\([A-Za-z0-9]*\) .*/\1/p' "$f" | head -1)
    [ -n "$variant" ] || variant="?"
    if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $variant "* ]]; then
        continue
    fi
    # Last observed cost per update, excluding eval (the trainer prints it).
    spu=$(grep -o 's/u=[0-9.]*' "$f" | tail -1 | cut -d= -f2)

    evals=$(grep -n "\] expl=" "$f")
    [ -n "${LAST:-}" ] && evals=$(printf '%s\n' "$evals" | tail -1)
    printf '%s\n' "$evals" | while IFS= read -r line; do
        [ -n "$line" ] || continue
        tag=$(printf '%s' "$line" | sed -n 's/.*\[\([a-z_0-9]*\)\] expl=.*/\1/p')
        covs=$(printf '%s' "$line" | grep -o "'mean_coverage': [0-9.]*" \
                   | cut -d' ' -f2)
        unis=$(printf '%s' "$line" | grep -o "'union_coverage': [0-9.]*" \
                   | cut -d' ' -f2)
        c0=$(printf '%s\n' "$covs" | sed -n 1p)
        c1=$(printf '%s\n' "$covs" | sed -n 2p)
        u0=$(printf '%s\n' "$unis" | sed -n 1p)
        u1=$(printf '%s\n' "$unis" | sed -n 2p)
        [ -n "$c0" ] || continue
        gap=$(awk -v a="$c0" -v b="${c1:-$c0}" 'BEGIN{printf "%.3f", a-b}')
        printf "%-5s %-9s %-6s %7.3f %7.3f %7.3f %7.3f %6s %6s\n" \
               "$variant" "$job" "${tag#navigate_}" \
               "$c0" "${c1:-0}" "${u0:-0}" "${u1:-0}" "$gap" "${spu:-?}"
    done
done

echo
echo "reference -- v35 (interleave, 20 GPU-h): cov 0.53/0.51/0.48 at u380,"
echo "             and cov 0.507/0.485/0.470 at its u240, for n_dist 0/5/10."
echo "note: mean_coverage here is over a PINNED 400-step eval, so it is the"
echo "      same measurement across variants and comparable to those."
