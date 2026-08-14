#!/bin/bash -l
#SBATCH --job-name=navtriprobec
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=mit_normal
#SBATCH --mem=120G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_probec_%j.out

# The signal / temporal probes on CPU.
#
# They need no GPU: the only heavy step is one forward pass of the encoder over
# Npos^2 grid codes to build encoded_Phi, which parallelizes across cores, and
# after that the work is 1024x1024 matmuls on batches of a few hundred rows.
# The GPU partitions cap concurrent jobs at 2 each and the training runs hold
# those for six hours at a time, whereas mit_normal has 3000 cores and a
# 12-hour limit -- so a probe that would wait six hours for a GPU it barely
# uses runs here immediately instead.
#
#   PROBE=signal   CKPTS=... sbatch hopfield_nav/run_nav_tri_probe_cpu.sh
#   PROBE=temporal CKPTS=... sbatch hopfield_nav/run_nav_tri_probe_cpu.sh

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
PROBE=${PROBE:-signal}
NDIST=${NDIST:-"0 1 3 5 10"}
OUTDIR=${OUTDIR:-/orcd/pool/003/jackking/cls_runs/results/nav_tri_probe}

module load miniforge/24.3.0-0
source activate cls

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUTDIR"

for ck in $CKPTS; do
    tag=$(basename "$(dirname "$ck")")_$(basename "$ck" .pt)
    echo ""
    echo "################ $PROBE :: $tag (cpu) ################"
    case "$PROBE" in
      signal)
        python -u -m analysis.nav_tri.signal_separability \
            --ckpt "$ck" --device cpu --n_distractors $NDIST \
            --cells "${CELLS:-200}" ${ENVS:+--envs $ENVS} \
            --json "$OUTDIR/signal_${tag}.json" ;;
      temporal)
        python -u -m analysis.nav_tri.temporal_separability \
            --ckpt "$ck" --device cpu --n_distractors $NDIST \
            --steps "${TSTEPS:-20}" --sets "${SETS:-8}" --traj "${TRAJ:-32}" \
            ${ENVS:+--envs $ENVS} \
            --json "$OUTDIR/temporal_${tag}.json" ;;
      behavior)
        python -u -m analysis.nav_tri.behavior_probe \
            --ckpt "$ck" --device cpu --mode ${MODE:-"explore nav"} \
            --n_distractors $NDIST --trials "${TRIALS:-32}" \
            --max_steps "${MAX_STEPS:-200}" ${ENVS:+--envs $ENVS} \
            --json "$OUTDIR/${tag}.json" ;;
      *) echo "unknown PROBE=$PROBE" >&2; exit 1 ;;
    esac
done
