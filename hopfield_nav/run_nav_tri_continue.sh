#!/bin/bash -l
#SBATCH --job-name=navtri
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_%j.out

# Pick a nav_tri run back up where it stopped and run it further.
#
#   RUN=navigate_navtri_w6_pers_s42_20499183 SCHEDULE='interleave:2000,empty_frac=0.5' \
#       sbatch hopfield_nav/run_nav_tri_continue.sh
#
# This is `--continue_from`, NOT `--load_checkpoint`. The difference matters
# and is the reason this file exists separately from run_nav_tri.sh:
#
#   --load_checkpoint  forks a NEW run from a parent's weights, dropping the
#                      Adam moments and the RNG streams. Right when the
#                      objective changes (wave 3's warm starts).
#   --continue_from    picks THIS run's own trajectory back up, restoring the
#                      optimizer moments, the global/distractor/per-env RNG
#                      streams and the refresher's position. Right when the
#                      objective is unchanged and the run simply ran out of
#                      wall-clock -- which is this case.
#
# It also refuses config overrides by design, so run_nav_tri.sh cannot be used:
# that script spells out every knob, and every one of them would be rejected.
# Only the schedule may change, and only by LENGTHENING it.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
CKPTS_ROOT=${CKPTS_ROOT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts}
RESUME="$CKPTS_ROOT/${RUN}/resume/latest.pt"

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd "$REPO"
source scripts/cls_env.sh

echo "=== nav_tri continue: $RUN ==="
echo "    resume : $RESUME"
echo "    schedule (lengthened): ${SCHEDULE}"

python -u -m hopfield_nav.train_navigate \
    --continue_from "$RESUME" \
    --schedule "$SCHEDULE" \
    --device "${DEVICE:-cuda}"
