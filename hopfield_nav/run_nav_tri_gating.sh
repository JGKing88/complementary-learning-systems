#!/bin/bash -l
#SBATCH --job-name=navtrigate
#SBATCH --time=0-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=mit_normal
#SBATCH --mem=100G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_gate_%j.out

# Does the policy gate on ||q||? A 2x2 crossover, not a correlation.
#
#   CKPT=<path> sbatch hopfield_nav/run_nav_tri_gating.sh
#
# In a normal rollout the memory's CONTENTS and the recall's MAGNITUDE move
# together -- a stored goal gives ||q|| ~ 0.22, decoys alone give ~0.17 at ten
# distractors -- so no observational statistic can say which one the policy is
# responding to. --q_rescale breaks them apart by forcing ||q|| to a chosen
# value while leaving its direction alone.
#
#   cell           memory   ||q|| fed    reads out as
#   ------------------------------------------------------------------
#   nav/native     goal     0.22 (own)   follow_q, the baseline
#   nav/decoymag   goal     0.17         does following SURVIVE a weak signal?
#   expl/native    decoys   0.17 (own)   chase_q, the baseline
#   expl/goalmag   decoys   0.22         does chasing APPEAR at a strong one?
#
# If the policy gates on magnitude, following tracks the fed norm: nav/decoymag
# collapses toward expl/native and expl/goalmag rises toward nav/native.
# If it gates on something else -- the multistep dynamics, the sensory context,
# its own recurrent state -- following tracks the memory contents instead, and
# each rescaled cell stays close to its native partner.
#
# Caveat for any write-up: rescaling creates input combinations the policy never
# saw in training (a decoy direction at goal strength), so a behaviour change is
# evidence the channel is USED, not proof of what the policy does on-distribution.

set -euo pipefail
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navtri_w6_pers_s42_20499183/navigate_u1950.pt}
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/nav_tri_probe}
NDIST=${NDIST:-10}
GOALMAG=${GOALMAG:-0.22}
DECOYMAG=${DECOYMAG:-0.17}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
cd /orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric
source scripts/cls_env.sh
mkdir -p "$OUT"

run () {   # mode tag [rescale]
    echo ""
    echo "######## mode=$1  norm=$2 ########"
    python -u -m analysis.nav_tri.behavior_probe --ckpt "$CKPT" --device cpu \
        --mode "$1" --n_distractors "$NDIST" --trials 32 --envs 8 \
        --max_steps 200 ${3:+--q_rescale $3} \
        --json "$OUT/gate_$1_$2.json"
}

run nav     native
run nav     decoymag "$DECOYMAG"
run explore native
run explore goalmag  "$GOALMAG"
