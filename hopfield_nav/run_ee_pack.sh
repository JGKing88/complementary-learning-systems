#!/bin/bash -l
#SBATCH --job-name=hnav-eepack
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=220G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/eepack_%j.out

# Several variants, one GPU allocation, run concurrently.
#
#   PACK="X3 X4 P1" sbatch hopfield_nav/run_ee_pack.sh
#
# Why pack at all: measured on a live X2, this workload sits at **12% GPU
# utilization and 3.4 of 46 GB VRAM**. It is bound by the serial python/numpy
# step loop -- envs_per_world x steps_per_rollout model calls per update, each
# tiny -- not by the GPU. Meanwhile the QOS caps this account at 2 GPUs, so
# one-run-per-allocation caps the whole line at two concurrent experiments
# while leaving 88% of both GPUs idle. Packing converts that idle into
# concurrency, which is the scarce resource here.
#
# What it costs: the processes contend for CPU, so each is somewhat slower than
# it would be alone. Budget CPUs accordingly (~4 per run) and read the s/u each
# run prints rather than assuming a solo figure carries over. Host RAM is the
# hard limit: each run holds its own 12 GB encoded_Phi and peaks near 45 GB, so
# --mem must cover the whole pack (measured, sacct MaxRSS 44.5 GB for one).
#
# Each run writes its own log; this job's own output carries only the roster
# and the exit codes, so a pack is as readable as N separate jobs.

set -uo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
PACK=${PACK:?PACK is required, e.g. PACK="X3 X4"}

cd "$REPO"
source scripts/cls_env.sh
source hopfield_nav/ee_variants.sh

echo "=== pack [$PACK] on $(hostname), job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

declare -a PIDS=() NAMES=()
for V in $PACK; do
    LOG="$CLS_LOGS/ee_${V}_${SLURM_JOB_ID:-local}.out"
    echo "  $V -> $LOG"
    # The variant's own knobs, then the launcher. `env` rather than exporting
    # into this shell: two variants in one pack must not inherit each other's
    # overrides, and an exported INIT_LOG_STD from the first would silently
    # become the second's default.
    # shellcheck disable=SC2046
    env $(ee_env "$V") VARIANT="$V" REPO="$REPO" \
        bash hopfield_nav/run_ee.sh > "$LOG" 2>&1 &
    PIDS+=($!)
    NAMES+=("$V")
    # Stagger the starts: every run builds the same 12 GB encoded_Phi at
    # startup, and doing that simultaneously is the one moment the pack's
    # memory use is not the sum of its steady states.
    sleep 90
done

STATUS=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "=== ${NAMES[$i]}: OK ==="
    else
        rc=$?
        echo "=== ${NAMES[$i]}: FAILED (exit $rc) ==="
        STATUS=1
    fi
done
exit $STATUS
