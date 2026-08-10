#!/bin/bash -l
#SBATCH --job-name=hnav-expl-min
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_explore_min_%j.out

# Train an explore-ONLY policy on as little data as possible.
#
# The question is not "can coverage be high" -- v35 reaches mean_coverage 0.53
# in ~20 GPU-hours -- it is how few environment steps buy it. So the axis that
# matters is coverage vs *env-steps consumed*, and env-steps per update is
# BATCH_ENVS x STEPS_PER_ROLLOUT, both of which this launcher exposes.
#
#   VARIANT=s1 sbatch hopfield_nav/run_explore_min.sh
#   VARIANT=d3 BATCH_ENVS=4 sbatch hopfield_nav/run_explore_min.sh
#
# Every knob is spelled out rather than inherited, for the same reason
# run_repro_v35.sh spells its out: the baseline is a specific historical run,
# and an inherited default would silently move it. Everything below is
# byte-identical to run_repro_v35.sh EXCEPT the block marked "deliberate
# departures", so any difference in the result is attributable.

JOB_LABEL=explore_min
VARIANT=${VARIANT:-s1}

# ===========================================================================
# DELIBERATE DEPARTURES FROM run_repro_v35.sh
# ===========================================================================
#
# 1. Pure explore, not interleave. v35 spends half its rollouts in the exploit
#    regime, which trains "recall a point, go to it, stop" -- the exact
#    opposite of covering ground, and the reason its coverage is still
#    climbing at u380 while nav saturated at u60. Dropping it doubles the
#    explore data per update AND removes a gradient pulling the other way.
#    This is expected to be the single largest effect in the wave, and it
#    costs nothing.
#
# 2. Distractors are what the explore regime must learn to ignore, and with
#    goals off there is never any reward for chasing a recalled point -- so
#    the "chase" behavior has no way to form in the first place. Training
#    range is held at the eval range (0..10) so the input distribution
#    matches. Robustness at n_dist=10 is a target, not a hope.
SCHEDULE=${SCHEDULE:-'explore:300'}

# 3. Eval is trimmed and pinned. nav/disc are undefined for a policy never
#    trained to reach or store a goal, and cost two thirds of an eval pass.
#    EVAL_MAX_STEPS is pinned at 400 so mean_coverage stays the SAME
#    measurement across the rollout-length sweep -- without it a 100-step run
#    would report coverage over 100 steps and the variants would not be
#    comparable to each other or to v35's 0.53.
EVAL_SCOPE=${EVAL_SCOPE:-expl}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-400}
NUM_VAL_ENVS=${NUM_VAL_ENVS:-4}
N_VAL_TRIALS=${N_VAL_TRIALS:-16}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 10"}
EVAL_EVERY=${EVAL_EVERY:-25}
CKPT_EVERY=${CKPT_EVERY:-25}
#    4x16 = 64 rollouts per distractor level is a monitoring-grade estimate,
#    not the verdict. The verdict is a 10-env / 32-trial offline pass under
#    the v35 protocol, run on checkpoints at the end.

# 4. epsilon anneals over the run, not over 200 updates. v35's 200-update
#    anneal was sized for a 400+ update run; on a short one it leaves the
#    behavior policy far from the deterministic policy that eval scores for
#    the entire run, and 40% random directions is precisely what breaks the
#    long straight sweeps the shaping is trying to reinforce.
EPSILON_EXPLORE=${EPSILON_EXPLORE:-0.4}
EPSILON_ANNEAL_UPDATES=${EPSILON_ANNEAL_UPDATES:-100}

# 5. Rollout shape is the experiment. v35 was 16 x 400 = 6400 env-steps/update.
BATCH_ENVS=${BATCH_ENVS:-16}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-200}

# 6. Reward shaping is the other experiment. v35's shape is the control.
#    NOTE ON WHAT IS ACTUALLY FREE HERE: advantages are normalized over the
#    full pool (updates/ppo.py), and an explore rollout with goals off is
#    fixed-length with no teleport, so a flat per-step term is a constant that
#    cancels in the advantage. Since novelty fires on new cells and revisit on
#    old ones, -c*1[old] == (n+c)*1[new] - c, and REVISIT_PENALTY is exactly
#    redundant with NOVELTY_REWARD -- unless NOVELTY_SCALE_REMAINING is on,
#    which makes novelty state-dependent while the penalty stays flat. Only
#    ratios to novelty are meaningful; the overall scale is normalized away.
NOVELTY_REWARD=${NOVELTY_REWARD:-0.3}
NOVELTY_ANNEAL=${NOVELTY_ANNEAL:-0}
NOVELTY_SCALE_REMAINING=${NOVELTY_SCALE_REMAINING:-1}
NOVELTY_SCALE_CAP=${NOVELTY_SCALE_CAP:-10}
REVISIT_PENALTY=${REVISIT_PENALTY:-0}
WALL_PENALTY=${WALL_PENALTY:-0.1}
PERSISTENCE_BONUS=${PERSISTENCE_BONUS:-0.05}

# 7. INIT_LOG_STD/FREEZE_LOG_STD read the same as v35's, but no longer BEHAVE
#    the same. `--freeze_log_std` was a no-op on train_navigate until
#    2026-08-07: set_phase_freeze(freeze_move=False) handed movement_log_std
#    its gradient straight back, because log_std is a movement parameter. v35
#    therefore trained a *learnable* log_std -- visible in its log as std
#    drifting 0.166 -> 0.294 by u250 -- and so did everything in its lineage.
#    With that fixed, these runs are the first where the flag bites, which is
#    the V10 configuration: a pinned narrow std, with epsilon as the only
#    exploration. Variant f1 turns the freeze back off to bracket it.

# ===========================================================================
# HELD IDENTICAL TO run_repro_v35.sh BELOW THIS LINE
# (identical as WRITTEN; see departure 7 for the one that now behaves
#  differently than the same flags did in v35)
# ===========================================================================

# --- encoder / scaffold ----------------------------------------------------
ENCODER=${ENCODER:-encoders/run_20260422_185816/encoder_best.pt}
FWHM_RATIO=${FWHM_RATIO:-0.25}
LAMBDAS=${LAMBDAS:-"11 12 13"}
NP=${NP:-400}
STATIC_VECTORHASH=${STATIC_VECTORHASH:-1}

# --- environment -----------------------------------------------------------
SIZE=${SIZE:-20}
OBSERVATION_SIZE=${OBSERVATION_SIZE:-12}
MOVEMENT_MODE=${MOVEMENT_MODE:-continuous}
# Inert under explore_goals_off, kept so the config diff against v35 is empty.
GOAL_REWARD=${GOAL_REWARD:-5.0}
GOAL_RADIUS=${GOAL_RADIUS:-1.0}
TIME_PENALTY=${TIME_PENALTY:-0.05}

# --- agent -----------------------------------------------------------------
HOPFIELD_MODE=${HOPFIELD_MODE:-continuous}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
NUM_RNN_LAYERS=${NUM_RNN_LAYERS:-1}
INIT_LOG_STD=${INIT_LOG_STD:--1.8}
FREEZE_LOG_STD=${FREEZE_LOG_STD:-1}
INPUT_PREV_REWARD=${INPUT_PREV_REWARD:-1}
INPUT_PREV_ACTION=${INPUT_PREV_ACTION:-0}
INPUT_HOPFIELD_RAW=${INPUT_HOPFIELD_RAW:-1}
INPUT_HOPFIELD_SIGNAL=${INPUT_HOPFIELD_SIGNAL:-1}
INPUT_SENSORY=${INPUT_SENSORY:-1}
INPUT_ENCODED_STATE=${INPUT_ENCODED_STATE:-0}
INPUT_GOAL_IN_MEMORY=${INPUT_GOAL_IN_MEMORY:-0}
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-"1 2 3"}

# --- optimization ----------------------------------------------------------
LR=${LR:-3e-4}
MOVE_ENT_COEF=${MOVE_ENT_COEF:-0.005}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.15}

# --- explore-regime behavior -----------------------------------------------
EXPLORE_GOALS_OFF=${EXPLORE_GOALS_OFF:-1}

# --- distractors -----------------------------------------------------------
N_TRAIN_DISTRACTORS_MIN=${N_TRAIN_DISTRACTORS_MIN:-0}
N_TRAIN_DISTRACTORS_MAX=${N_TRAIN_DISTRACTORS_MAX:-10}
N_TRAIN_EMP_DISTRACTORS_MIN=${N_TRAIN_EMP_DISTRACTORS_MIN:-0}
N_TRAIN_EMP_DISTRACTORS_MAX=${N_TRAIN_EMP_DISTRACTORS_MAX:-10}

# --- world -----------------------------------------------------------------
NUM_WORLDS=${NUM_WORLDS:-1}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-80}
SEED=${SEED:-42}

# --- logging ---------------------------------------------------------------
DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-hopfield-nav-explore-min}

export WANDB_NAME=${WANDB_NAME:-explore_min_${VARIANT}_${SLURM_JOB_ID:-local}}

echo "=== variant=$VARIANT  ${BATCH_ENVS}x${STEPS_PER_ROLLOUT} = \
$((BATCH_ENVS * STEPS_PER_ROLLOUT)) env-steps/update ==="
echo "    shaping: nov=$NOVELTY_REWARD scale=$NOVELTY_SCALE_REMAINING/\
$NOVELTY_SCALE_CAP wall=$WALL_PENALTY pers=$PERSISTENCE_BONUS \
revisit=$REVISIT_PENALTY eps=$EPSILON_EXPLORE/$EPSILON_ANNEAL_UPDATES"

# REPO_DIR comes from the submitter's environment (sbatch exports it by
# default), so a wave submitted from an agent worktree trains that worktree's
# code instead of whatever branch the shared checkout happens to be sitting on.
# It cannot be derived here: SLURM copies this script to a node-local spool
# directory, so $BASH_SOURCE points somewhere useless.
cd "${REPO_DIR:-/home/jackking/cls}"
echo "    repo: $PWD @ $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
source hopfield_nav/navigate_job.sh
