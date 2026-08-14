#!/bin/bash -l
#SBATCH --job-name=navtri
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_%j.out

# One model, three metrics: coverage (explore) + success_rate and mean_steps
# (exploit). See docs/EXPERIMENTS_NAV_TRI.md -- that document holds the waves,
# the hypotheses and the results; this file only holds the knobs.
#
#   VARIANT=w1_base sbatch hopfield_nav/run_nav_tri.sh
#   VARIANT=w1_c20  sbatch --partition=pi_fiete --time=8:00:00 \
#                          hopfield_nav/run_nav_tri.sh
#
# sbatch's own flags override the directives above, which is how a variant
# reaches pi_fiete (7 d, a100) instead of mit_normal_gpu (6 h, l40s/h200).
#
# Anything can be overridden from the environment for a one-off:
#   VARIANT=w1_base SEED=43 sbatch hopfield_nav/run_nav_tri.sh
#
# Every knob is spelled out rather than inherited from the trainer's argparse
# defaults, for the reason run_repro_v35.sh gives: a run in a comparison series
# must be able to say what it ran, and an inherited default moves silently.

set -euo pipefail

JOB_LABEL=nav_tri
VARIANT=${VARIANT:-w1_base}
# The worktree this line of work lives in. Overridable so the script still
# works if the branch is ever merged down to the main checkout.
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}

# ===========================================================================
# FIXED BY INSTRUCTION -- not experimental axes. See EXPERIMENTS_NAV_TRI §1.2.
# ===========================================================================
ENCODER=${ENCODER:-"encoder_training/sweeps/ur_loss2_repel_low/029_repel_weight=2_per_env_radius_frac=0.1_seed=44/encoder_best.pt"}
RNN_CELL=${RNN_CELL:-rnn}
RNN_NONLINEARITY=${RNN_NONLINEARITY:-relu}
GOAL_RADIUS=${GOAL_RADIUS:-1.0}
CONTINUOUS_NORMALIZE=${CONTINUOUS_NORMALIZE:-0}
INPUT_HOPFIELD_RAW=${INPUT_HOPFIELD_RAW:-1}
WALL_RESOLUTION=${WALL_RESOLUTION:-4}
OBSERVATION_SIZE=${OBSERVATION_SIZE:-60}
EXPLORE_ENDS_ON_GOAL=${EXPLORE_ENDS_ON_GOAL:-1}
RESET_STATE_ON_TELEPORT=${RESET_STATE_ON_TELEPORT:-0}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-200}
WANDB_PROJECT=${WANDB_PROJECT:-train_navigate}
# Never. Jack calls this one cheating: it hands the policy the explore/exploit
# distinction as a labelled bit instead of making it infer it.
INPUT_GOAL_IN_MEMORY=0

# --- scaffold (v35's, unchanged) -------------------------------------------
FWHM_RATIO=${FWHM_RATIO:-0.25}
LAMBDAS=${LAMBDAS:-"11 12 13"}
NP=${NP:-400}
STATIC_VECTORHASH=${STATIC_VECTORHASH:-1}

# --- environment (v35's, unchanged except the fixed block above) -----------
SIZE=${SIZE:-20}
MOVEMENT_MODE=${MOVEMENT_MODE:-continuous}
GOAL_REWARD=${GOAL_REWARD:-5.0}
TIME_PENALTY=${TIME_PENALTY:-0.05}

# --- agent (v35's, except the RNN/ReLU trunk above) ------------------------
HOPFIELD_MODE=${HOPFIELD_MODE:-continuous}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
NUM_RNN_LAYERS=${NUM_RNN_LAYERS:-1}
INIT_LOG_STD=${INIT_LOG_STD:--1.8}
FREEZE_LOG_STD=${FREEZE_LOG_STD:-1}
INPUT_PREV_REWARD=${INPUT_PREV_REWARD:-1}
INPUT_PREV_ACTION=${INPUT_PREV_ACTION:-0}
INPUT_HOPFIELD_SIGNAL=${INPUT_HOPFIELD_SIGNAL:-1}
INPUT_SENSORY=${INPUT_SENSORY:-1}
INPUT_ENCODED_STATE=${INPUT_ENCODED_STATE:-0}
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-"1 2 3"}

# --- optimization (v35's) --------------------------------------------------
LR=${LR:-3e-4}
MOVE_ENT_COEF=${MOVE_ENT_COEF:-0.005}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.15}

# --- reward shaping (v35's) ------------------------------------------------
NOVELTY_REWARD=${NOVELTY_REWARD:-0.3}
NOVELTY_ANNEAL=${NOVELTY_ANNEAL:-0}
NOVELTY_SCALE_REMAINING=${NOVELTY_SCALE_REMAINING:-1}
NOVELTY_SCALE_CAP=${NOVELTY_SCALE_CAP:-10}
REVISIT_PENALTY=${REVISIT_PENALTY:-0}
WALL_PENALTY=${WALL_PENALTY:-0.1}
PERSISTENCE_BONUS=${PERSISTENCE_BONUS:-0.05}

# --- explore regime (v35's) ------------------------------------------------
EXPLORE_GOALS_OFF=${EXPLORE_GOALS_OFF:-1}
EPSILON_EXPLORE=${EPSILON_EXPLORE:-0.4}
EPSILON_ANNEAL_UPDATES=${EPSILON_ANNEAL_UPDATES:-200}

# --- distractors (v35's; training range = eval range) ----------------------
N_TRAIN_DISTRACTORS_MIN=${N_TRAIN_DISTRACTORS_MIN:-0}
N_TRAIN_DISTRACTORS_MAX=${N_TRAIN_DISTRACTORS_MAX:-10}
N_TRAIN_EMP_DISTRACTORS_MIN=${N_TRAIN_EMP_DISTRACTORS_MIN:-0}
N_TRAIN_EMP_DISTRACTORS_MAX=${N_TRAIN_EMP_DISTRACTORS_MAX:-10}

# --- rollout shape / world -------------------------------------------------
BATCH_ENVS=${BATCH_ENVS:-16}
NUM_WORLDS=${NUM_WORLDS:-1}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-80}
SEED=${SEED:-42}
# No refreshing of anything: same envs every update, per instruction.

# --- eval ------------------------------------------------------------------
# Pinned at the rollout length so mean_coverage is one measurement across
# every variant, and equal to what training optimizes.
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-200}
NUM_VAL_ENVS=${NUM_VAL_ENVS:-6}
N_VAL_TRIALS=${N_VAL_TRIALS:-16}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 10"}
EVAL_SCOPE=${EVAL_SCOPE:-expl}
EVAL_EVERY=${EVAL_EVERY:-25}
CKPT_EVERY=${CKPT_EVERY:-25}

DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-1}

# ===========================================================================
# VARIANTS
# ===========================================================================
case "$VARIANT" in

  # --- smoke: does the config parse, and how many seconds is an update? ----
  smoke)
    SCHEDULE=${SCHEDULE:-'explore:6'}
    EVAL_EVERY=6; CKPT_EVERY=6; NUM_VAL_ENVS=2; N_VAL_TRIALS=8
    ;;
  smoke_c20)
    SCHEDULE=${SCHEDULE:-'explore:12'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EVAL_EVERY=12; CKPT_EVERY=12; NUM_VAL_ENVS=2; N_VAL_TRIALS=8
    ;;

  # === WAVE 1 -- baseline, the cost/diversity ladder, and the noise regime ==
  #
  # w1_base is v35's shaping and rollout shape under the fixed settings of
  # §1.2. Everything else in the wave is one deliberate step off it.

  w1_base)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    ;;

  # Cost/diversity ladder. PPO pool (envs x batch) is held at 1280 = w1_base's,
  # so env-steps per update and gradient batch are identical and the ONLY
  # difference is how many distinct envs those trajectories come from --
  # while serial model calls per update, which is what wall-clock tracks,
  # fall 4x and 10x. If these match w1_base at equal updates, every later wave
  # gets 4-10x more updates per GPU-hour.
  w1_c20)
    SCHEDULE=${SCHEDULE:-'explore:2400'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EVAL_EVERY=100; CKPT_EVERY=100
    ;;
  w1_c8)
    SCHEDULE=${SCHEDULE:-'explore:6000'}
    ENVS_PER_WORLD=8; BATCH_ENVS=160
    EVAL_EVERY=250; CKPT_EVERY=250
    ;;

  # Noise regime. eps steps are dropped from the PPO movement surrogate
  # (collector.py:479-485), so eps=0.4 discards 40% of the policy gradient and
  # makes the behaviour policy 16% worse at coverage than the mean it scores
  # (docs/EXPERIMENTS_NAV_TRI §3.2). These ask whether that purchase is worth
  # its price, and whether sigma can buy the same exploration more cheaply.
  w1_eps01)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    EPSILON_EXPLORE=0.1
    ;;
  w1_sig)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-1.2
    ;;

  # Shaping. A billiard -- straight lines, turn at the wall -- is the reachable
  # target behaviour (cov 0.387 vs a random walk's 0.178), and persistence_bonus
  # is the only term that rewards its defining feature. At v35's ratio the
  # straightness term is worth 0.05/step against novelty's ~0.3/step, i.e. 6:1
  # against. This makes it 2:1.
  w1_pers)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    PERSISTENCE_BONUS=0.15
    ;;

  *)
    echo "ERROR: unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac

export WANDB_NAME=${WANDB_NAME:-navtri_${VARIANT}_s${SEED}_${SLURM_JOB_ID:-local}}

echo "=== nav_tri variant=$VARIANT seed=$SEED ==="
echo "    schedule   : $SCHEDULE"
echo "    rollout    : ${ENVS_PER_WORLD} envs x ${BATCH_ENVS} batch x ${STEPS_PER_ROLLOUT} steps"
echo "                 pool=$((ENVS_PER_WORLD * BATCH_ENVS)) trajectories, \
$((ENVS_PER_WORLD * BATCH_ENVS * STEPS_PER_ROLLOUT)) env-steps/update, \
$((ENVS_PER_WORLD * STEPS_PER_ROLLOUT)) serial calls/update"
echo "    shaping    : nov=$NOVELTY_REWARD scale=$NOVELTY_SCALE_REMAINING/$NOVELTY_SCALE_CAP \
wall=$WALL_PENALTY pers=$PERSISTENCE_BONUS revisit=$REVISIT_PENALTY"
echo "    noise      : eps=$EPSILON_EXPLORE/$EPSILON_ANNEAL_UPDATES \
init_log_std=$INIT_LOG_STD freeze=$FREEZE_LOG_STD ent=$MOVE_ENT_COEF"
echo "    trunk      : $RNN_CELL/$RNN_NONLINEARITY h=$HIDDEN_SIZE"

cd "$REPO"
source hopfield_nav/navigate_job.sh
