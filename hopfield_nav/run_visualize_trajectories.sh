#!/bin/bash -l
#SBATCH --job-name=hnav-vizt
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/vizt_%j.out

# Visualize trajectories at every checkpoint of a training run.
#
# Required:
#   CHECKPOINT_DIR   directory containing checkpoint .pt files. Any file whose
#                    basename ends in _u{N}.pt or _update{N}.pt is matched
#                    (covers hopfield_nav_update{N}.pt, phase_a_u{N}.pt,
#                     phase_b_u{N}.pt, phase_c_u{N}.pt).
#
# Optional (with defaults):
#   MODE             combined | explore_only | exploit_only   (default: combined)
#   TRIALS           number of trial columns                  (default: 6)
#   EXPLORE_STEPS    max steps for explore phase              (default: 200)
#   NAV_STEPS        max steps for nav phase                  (default: 100)
#   N_DISTRACTORS    distractors preloaded into Hopfield      (default: 0)
#   FORCE_STORE      set to any non-empty value to force-store on first goal-reach
#                    (combined mode only; default: off)
#   UPDATES          comma-separated update numbers (e.g. "200,400,600")
#                    default: every hopfield_nav_update*.pt in CHECKPOINT_DIR
#   OUT              output PNG path
#                    default: $CHECKPOINT_DIR/trajectories_<MODE>.png
#   DEVICE           cuda | cpu                               (default: cuda)
#   SEED             trial-plan seed                          (default: 42)
#   GOAL_RADIUS      override EnvConfig.goal_radius from ckpt (default: unset → use saved value)
#
# Examples:
#   CHECKPOINT_DIR=checkpoint/phase_a_only_hopeful-haze-46 MODE=combined ./run_visualize_trajectories.sh
#   CHECKPOINT_DIR=checkpoint/phase_a_only_hopeful-haze-46 MODE=explore_only TRIALS=8 ./run_visualize_trajectories.sh
#   CHECKPOINT_DIR=checkpoint/phase_a_only_hopeful-haze-46 MODE=exploit_only N_DISTRACTORS=10 ./run_visualize_trajectories.sh
#   CHECKPOINT_DIR=checkpoint/phase_a_only_hopeful-haze-46 FORCE_STORE=1 UPDATES=20,100,200 ./run_visualize_trajectories.sh
#   # Also works for PPO-style runs:
#   CHECKPOINT_DIR=checkpoints/r2_r2a_low_ent MODE=combined ./run_visualize_trajectories.sh

set -euo pipefail

if [ -z "${CHECKPOINT_DIR:-}" ]; then
    echo "ERROR: CHECKPOINT_DIR is required" >&2
    exit 1
fi

MODE=${MODE:-combined}
TRIALS=${TRIALS:-5}
EXPLORE_STEPS=${EXPLORE_STEPS:-500}
NAV_STEPS=${NAV_STEPS:-100}
N_DISTRACTORS=${N_DISTRACTORS:-0}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-4000}
GOAL_RADIUS=${GOAL_RADIUS:-1}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd /home/jackking/cls

source scripts/cls_env.sh
ARGS=(
    --checkpoint_dir "$CHECKPOINT_DIR"
    --mode "$MODE"
    --trials "$TRIALS"
    --explore_steps "$EXPLORE_STEPS"
    --nav_steps "$NAV_STEPS"
    --n_distractors "$N_DISTRACTORS"
    --device "$DEVICE"
    --seed "$SEED"
)
[ -n "${FORCE_STORE:-}" ] && ARGS+=(--force_store)
[ -n "${UPDATES:-}" ] && ARGS+=(--updates "$UPDATES")
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")
[ -n "${GOAL_RADIUS:-}" ] && ARGS+=(--goal_radius "$GOAL_RADIUS")

echo "=== visualize_trajectories  mode=$MODE  ckpt_dir=$CHECKPOINT_DIR ==="
python -m hopfield_nav.visualize_trajectories "${ARGS[@]}"
