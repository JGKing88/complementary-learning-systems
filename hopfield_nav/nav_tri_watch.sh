#!/bin/bash
# Emit one line per nav_tri job that LEAVES the queue, with its end state, the
# last update it reached, and the head of any traceback. Nothing per-update:
# a long run's progress belongs in nav_tri_status.sh, which is pulled when
# wanted, not pushed 20 times an hour.
LOGDIR=/orcd/pool/003/jackking/cls_runs/logs
prev=$(squeue -u "$USER" -n navtri -h -o "%i" 2>/dev/null | sort)
while true; do
    sleep 120
    cur=$(squeue -u "$USER" -n navtri -h -o "%i" 2>/dev/null | sort)
    gone=$(comm -23 <(echo "$prev") <(echo "$cur"))
    for j in $gone; do
        [ -z "$j" ] && continue
        st=$(sacct -j "$j" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
        lg="$LOGDIR/nav_tri_${j}.out"
        var=$(grep -m1 'variant=' "$lg" 2>/dev/null | sed 's/.*variant=//;s/ .*//')
        lastu=$(grep -oE 'u[0-9]+\(' "$lg" 2>/dev/null | tail -1 | tr -d 'u(')
        cov=$(grep -oE "expl=\{0: \{'mean_coverage': [0-9.]+" "$lg" 2>/dev/null \
              | tail -1 | sed "s/.*mean_coverage': //")
        echo "JOB_END $j ${var:-?} state=${st:-?} last=u${lastu:-0} cov_d0=${cov:-na}"
        grep -m3 -E 'Traceback|^[A-Za-z]*Error|CUDA out of memory|Killed' "$lg" 2>/dev/null | sed 's/^/      /'
    done
    prev="$cur"
done
