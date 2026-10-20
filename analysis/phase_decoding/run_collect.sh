#!/bin/bash -l
#SBATCH --job-name=pa-collect
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ou_bcs_normal
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/phase_abstraction/logs/collect_%x_%j.out
# One collection run for the abstraction analysis (docs/PHASE_ABSTRACTION.md).
#
#   NAME=d0_A CKPT=... DEFINITION=A MAX_STEPS=60 POST_STEPS=0 \
#     sbatch --time=02:00:00 --job-name=pa-d0_A analysis/phase_decoding/run_collect.sh
#   RANDOM_AGENT=1 adds the random-weight control.
set -euo pipefail
module load miniforge/24.3.0-0 2>/dev/null || true
source activate cls 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
OUT=/orcd/pool/003/jackking/cls_runs/results/phase_abstraction/${NAME:?}
RA=""
if [ -n "${RANDOM_AGENT:-}" ]; then RA="--random_agent --random_init_seed 0"; fi
python -m analysis.phase_decoding.run_collect \
    --ckpt "${CKPT:?}" --definition "${DEFINITION:?}" \
    --split place=held_out,wall=held_out,goal=held_out \
    --num_arenas 100 --val_seed 0 --n_trials 40 \
    --max_steps "${MAX_STEPS:?}" --post_steps "${POST_STEPS:?}" \
    --n_dist_min 0 --n_dist_max 10 --stochastic --seed 0 \
    --validate_arenas 10 $RA --out "$OUT"
echo COLLECT_DONE
