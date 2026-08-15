#!/bin/bash
# Block until a probe job's log says it finished N checkpoints (or died.)
#   bash hopfield_nav/wait_probe.sh <jobid> [n_expected]
L=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_probec_${1}.out
N=${2:-1}
while true; do
    if [ -f "$L" ]; then
        # `grep -c` prints 0 and exits 1 on no match, so `|| echo 0` would
        # append a second line and break the integer test.
        done_n=$(grep -c 'wrote ' "$L" 2>/dev/null; true)
        done_n=${done_n:-0}
        [ "$done_n" -ge "$N" ] && { echo "probe $1 done ($done_n)"; exit 0; }
        grep -qE 'Traceback|slurmstepd|CANCELLED' "$L" && { echo "probe $1 FAILED"; exit 1; }
    fi
    sleep 30
done
