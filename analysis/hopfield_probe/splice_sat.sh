#!/usr/bin/env bash
# The same basin measurement with the Hopfield SATURATED: beta = 1e6.
#
# Each encoder keeps its own gain (100 / 75 / 100 / 100 / 200) -- only the
# recall loop gain changes, which is the knob that decides whether the stored
# pattern is a fixed point at all (Sec 7's condition (b)). Sec 10.16 showed
# saturation turns the 10% encoder into a genuine attractor: the recalled state
# lands on the goal and stays there to s=15, where the linear arm drifts off by
# 1.41 cells. The basin is the natural place for that to show up.
#
# 20 tasks, one checkpoint each, written to a separate directory so the two
# recall regimes never share a ladder.
#
#SBATCH --job-name=splice_sat
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/splicesat_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-19
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec
/home/jackking/.conda/envs/cls/bin/python -u \
    analysis/hopfield_probe/splice_basin.py "$SLURM_ARRAY_TASK_ID" 1e6
