#!/bin/bash
# Where every nav_tri run stands, in one screen. See docs/EXPERIMENTS_NAV_TRI.md.
#
#   bash hopfield_nav/nav_tri_status.sh          # queue + latest eval per run
#   bash hopfield_nav/nav_tri_status.sh full     # + the whole coverage history

LOGDIR=/orcd/pool/003/jackking/cls_runs/logs
MODE=${1:-short}

echo "=== queue ==="
squeue -u "$USER" -n navtri -o "%.10i %.9P %.22j %.2t %.11M %.11L %R" 2>/dev/null

echo
echo "=== runs ==="
for f in $(ls -t $LOGDIR/nav_tri_*.out 2>/dev/null | head -40); do
    jid=$(basename "$f" .out | sed 's/nav_tri_//')
    var=$(grep -m1 '=== nav_tri variant=' "$f" | sed 's/.*variant=//;s/ .*//')
    [ -z "$var" ] && continue
    sched=$(grep -m1 '    schedule   :' "$f" | sed 's/.*: //')
    spu=$(grep -oE 's/u=[0-9.]+' "$f" | tail -1 | sed 's/s\/u=//')
    last=$(grep -oE 'u[0-9]+\(' "$f" | tail -1 | tr -d 'u(')
    state=$(sacct -j "$jid" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')

    # `expl` block of the newest eval line: mean_coverage at each distractor level.
    cov=$(grep -oE "\] expl=\{.*" "$f" | tail -1 \
        | grep -oE "[0-9]+: \{'mean_coverage': [0-9.]+" \
        | sed "s/: {'mean_coverage'://" | tr '\n' ' ')
    nav=$(grep -oE "\] nav=\{.*" "$f" | tail -1 \
        | grep -oE "[0-9]+: \{'success_rate': [0-9.]+, 'mean_speed': [0-9.]+, 'mean_steps': [0-9.]+" \
        | sed "s/: {'success_rate'://;s/, 'mean_speed': [0-9.]*, 'mean_steps':/ steps/" | tr '\n' ' ')

    printf "%-9s %-12s %-11s u%-6s %5ss/u  %s\n" \
           "$jid" "$var" "${state:-?}" "${last:-0}" "${spu:-?}" "$sched"
    [ -n "$cov" ] && printf "            cov(d:val) %s\n" "$cov"
    [ -n "$nav" ] && printf "            nav(d:sr steps) %s\n" "$nav"
    if [ "$MODE" = full ]; then
        grep -oE "\] expl=\{0: \{'mean_coverage': [0-9.]+" "$f" \
            | sed "s/.*mean_coverage': //" | tr '\n' ' ' | fold -w 120
        echo
    fi
done
