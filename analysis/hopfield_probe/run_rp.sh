#!/usr/bin/env bash
# Sec 19: random projection + threshold + linear readout (rp_encoder.py).
# One GPU job, run sequentially (each fit takes seconds to minutes; the queue
# is the slow part):
#
#   step1   rp_readout_check.py, the best-case readout (ridge to the ideal
#           r=16 code), every nonlinearity x coverage 1/5/20% for each bump
#           width (0.25, 0.15, 0.10) and H (8192, 2048, 32768).
#   step2   rp_sfa_check.py, the slowness readout: every nonlinearity and bump
#           width at H = 8192, the pure-wave control (rff8), then H = 32768 for
#           hard50 and sig1 at bump 0.10.
#   all     step1 then step2.
#
#   bash analysis/hopfield_probe/run_rp.sh all
#   python -m analysis.hopfield_probe.rp_summary step1 "$OUT/step1"
#   python -m analysis.hopfield_probe.rp_summary step2 "$OUT/step2"
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${RP_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RP_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/rp_readout}
MODE=${1:-all}

# RP_IN_JOB marks the batch copy (SLURM_JOB_ID is already set in an
# interactive allocation, so it cannot tell the two apart)
if [[ -z "${RP_IN_JOB:-}" ]]; then
    mkdir -p "$OUT/logs"
    # pi_fiete usually has a GPU free but its QOS caps the lab at 48 CPUs, so
    # ask for 2; the other partitions are fallbacks, whichever starts first.
    RP_IN_JOB=1 exec sbatch --job-name=rp_$MODE --partition=pi_fiete,ou_bcs_normal,mit_normal_gpu \
        --gres=gpu:1 --exclude=node3804 --time=${RP_TIME:-5:00:00} --cpus-per-task=2 \
        --mem=${RP_MEM:-64G} --output="$OUT/logs/${MODE}_%j.out" "$0" "$MODE"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$MODE" == step1 || "$MODE" == all ]]; then
    for H in 8192 2048 32768; do
        for F in 0.25 0.15 0.10; do
            "$PY" -m analysis.hopfield_probe.rp_readout_check --fwhm "$F" --H "$H" \
                --out "$OUT/step1"
            echo "DONE step1 fwhm $F H $H"
        done
    done
fi

if [[ "$MODE" == step2 || "$MODE" == all ]]; then
    for F in 0.25 0.15 0.10; do
        "$PY" -m analysis.hopfield_probe.rp_sfa_check --fwhm "$F" --H 8192 \
            --variants sig1,sig4,sig16,hard50,hard10 --save gauss16 --out "$OUT/step2"
        echo "DONE step2 fwhm $F H 8192"
    done
    "$PY" -m analysis.hopfield_probe.rp_sfa_check --fwhm 0.25 --H 8192 --variants rff8 \
        --out "$OUT/step2"
    echo "DONE step2 control rff8"
    "$PY" -m analysis.hopfield_probe.rp_sfa_check --fwhm 0.10 --H 32768 \
        --variants hard50,sig1 --coverages 0.05 --save gauss16 --out "$OUT/step2"
    echo "DONE step2 fwhm 0.10 H 32768"
fi
