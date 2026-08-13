#!/bin/bash -l
#SBATCH --job-name=hnav-ee
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=150G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/ee_%j.out

# The explore / exploit line. One launcher for every variant in
# docs/EXPERIMENTS_EXPLORE_EXPLOIT.md; a variant is a set of environment
# overrides, and the defaults below are the shared base every run departs from.
#
#   VARIANT=X1 ENVS_PER_WORLD=80 BATCH_ENVS=16 SCHEDULE='explore:550' \
#       sbatch hopfield_nav/run_ee.sh
#
# Every knob is spelled out rather than inherited from the trainer's argparse
# defaults, for the reason run_repro_v35.sh spells its out: a run's config is
# the experiment, and an inherited default moves it silently when the trainer
# changes. The base is run_repro_v35.sh with the departures marked below.

JOB_LABEL=ee
VARIANT=${VARIANT:-X1}

# ===========================================================================
# DEPARTURES FROM run_repro_v35.sh -- all requested, all deliberate
# ===========================================================================
#
# 1. Encoder. The unique-radius / repel sweep winner rather than the 2026-04
#    encoder v35 used. Its monotone coding radius is 21 cells, i.e. larger than
#    the 20-cell arena, so cosine-to-a-stored-pattern falls monotonically with
#    distance everywhere inside an env -- which is the precondition for the
#    Hopfield displacement to be a usable gradient at all. Measured
#    consequence, from diagnostics/hopfield_separability: with the goal in
#    memory q points at it to a 4-degree median error at every distance.
ENCODER=${ENCODER:-/orcd/pool/003/jackking/cls_runs/sweeps/ur_loss2_repel_low/029_repel_weight=2_per_env_radius_frac=0.1_seed=44/encoder_best.pt}

# 2. Vanilla Elman trunk with ReLU, not the historical GRU/tanh.
RNN_CELL=${RNN_CELL:-rnn}
RNN_NONLINEARITY=${RNN_NONLINEARITY:-relu}

# 3. wall_resolution 4. The sensory cone is the ONLY localizing channel the
#    policy has (input_encoded_state is off), and at resolution 1 a stripe
#    boundary can only fall on a cell edge, so cells alias. Measured on this
#    arena at 60 rays: bit-identical cell pairs 5e-4 at resolution 1 against
#    1e-5 at 4, and minimum Hamming 0 against 1.2. A systematic sweep is not
#    representable over aliased cells however it is trained.
WALL_RESOLUTION=${WALL_RESOLUTION:-4}

# 4. goal_radius 1.0 and no step-size normalization: the policy mean sets step
#    magnitude directly, so it can take a long stride toward a far recall and
#    a short one when close. allow_offcell_store stays False so a store inside
#    the radius still writes the goal cell's own pattern.
GOAL_RADIUS=${GOAL_RADIUS:-1.0}
CONTINUOUS_NORMALIZE=${CONTINUOUS_NORMALIZE:-0}
ALLOW_OFFCELL_STORE=${ALLOW_OFFCELL_STORE:-0}

# 5. Eval is the verdict protocol, run in-training. The explore-min wave found
#    its cheap 4-env x 16-trial monitor biased coverage high by ~0.02 and, worse,
#    mis-RANKED its own variants against the strict pass. Iterating on a biased
#    estimate is how a wave reaches the wrong conclusion cheaply, so this pays
#    for the real thing every time: 10 envs x 32 trials x {0,5,10} distractors
#    at a pinned 400 steps. Trials are batched, so 32 of them cost no serial
#    calls over 1; only envs and steps do.
#
#    eval_scope=nav_expl drops goal-discovery, the one unbatched evaluator
#    (~10 min a pass here), which scores a store head this trainer freezes.
EVAL_SCOPE=${EVAL_SCOPE:-nav_expl}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-400}
NUM_VAL_ENVS=${NUM_VAL_ENVS:-10}
N_VAL_TRIALS=${N_VAL_TRIALS:-32}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 5 10"}
EVAL_EVERY=${EVAL_EVERY:-25}
CKPT_EVERY=${CKPT_EVERY:-${EVAL_EVERY}}

# 6. No refreshing: the same envs, walls and goals every update.
#    (ENV_GENERATOR / REFRESH_* are wired through navigate_job.sh and unset.)

# 7. The schedule is the variant. Nothing here presumes explore or exploit.
SCHEDULE=${SCHEDULE:-'explore:550'}

# ===========================================================================
# HELD FROM run_repro_v35.sh
# ===========================================================================

# --- scaffold --------------------------------------------------------------
FWHM_RATIO=${FWHM_RATIO:-0.25}
LAMBDAS=${LAMBDAS:-"11 12 13"}
NP=${NP:-400}
STATIC_VECTORHASH=${STATIC_VECTORHASH:-1}

# --- environment -----------------------------------------------------------
SIZE=${SIZE:-20}
OBSERVATION_SIZE=${OBSERVATION_SIZE:-60}
MOVEMENT_MODE=${MOVEMENT_MODE:-continuous}
GOAL_REWARD=${GOAL_REWARD:-5.0}
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
# NEVER on: it hands the policy the regime label instead of making it read the
# memory, which is the whole task.
INPUT_GOAL_IN_MEMORY=${INPUT_GOAL_IN_MEMORY:-0}
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-"1 2 3"}

# --- optimization ----------------------------------------------------------
LR=${LR:-3e-4}
# Inert while FREEZE_LOG_STD=1: a diagonal Gaussian's entropy depends only on
# log_std, so with log_std pinned this term is a constant with no gradient.
# Kept at v35's value so the config diff is empty; it only bites in variants
# that unfreeze.
MOVE_ENT_COEF=${MOVE_ENT_COEF:-0.005}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.15}

# --- reward shaping (explore regime only) ----------------------------------
NOVELTY_REWARD=${NOVELTY_REWARD:-0.3}
NOVELTY_ANNEAL=${NOVELTY_ANNEAL:-0}
NOVELTY_SCALE_REMAINING=${NOVELTY_SCALE_REMAINING:-1}
NOVELTY_SCALE_CAP=${NOVELTY_SCALE_CAP:-10}
REVISIT_PENALTY=${REVISIT_PENALTY:-0}
WALL_PENALTY=${WALL_PENALTY:-0.1}
PERSISTENCE_BONUS=${PERSISTENCE_BONUS:-0.05}

# --- explore-regime behavior -----------------------------------------------
EXPLORE_GOALS_OFF=${EXPLORE_GOALS_OFF:-1}
EPSILON_EXPLORE=${EPSILON_EXPLORE:-0.4}
EPSILON_ANNEAL_UPDATES=${EPSILON_ANNEAL_UPDATES:-200}

# --- distractors -----------------------------------------------------------
N_TRAIN_DISTRACTORS_MIN=${N_TRAIN_DISTRACTORS_MIN:-0}
N_TRAIN_DISTRACTORS_MAX=${N_TRAIN_DISTRACTORS_MAX:-10}
N_TRAIN_EMP_DISTRACTORS_MIN=${N_TRAIN_EMP_DISTRACTORS_MIN:-0}
N_TRAIN_EMP_DISTRACTORS_MAX=${N_TRAIN_EMP_DISTRACTORS_MAX:-10}

# --- rollout shape ---------------------------------------------------------
BATCH_ENVS=${BATCH_ENVS:-16}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-200}
NUM_WORLDS=${NUM_WORLDS:-1}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-80}
SEED=${SEED:-42}

# --- logging ---------------------------------------------------------------
DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-train_navigate}
export WANDB_NAME=${WANDB_NAME:-ee_${VARIANT}_${SLURM_JOB_ID:-local}}

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}

echo "=== variant=$VARIANT  schedule='$SCHEDULE' ==="
echo "    shape: ${ENVS_PER_WORLD} envs x ${BATCH_ENVS} batch x ${STEPS_PER_ROLLOUT} steps"
echo "           = $((ENVS_PER_WORLD * BATCH_ENVS * STEPS_PER_ROLLOUT)) env-steps/update,"
echo "             pool $((ENVS_PER_WORLD * BATCH_ENVS)) trajectories,"
echo "             $((ENVS_PER_WORLD * STEPS_PER_ROLLOUT)) SERIAL model calls/update <- wall-clock"
echo "    shaping: nov=$NOVELTY_REWARD scale=$NOVELTY_SCALE_REMAINING/$NOVELTY_SCALE_CAP \
wall=$WALL_PENALTY pers=$PERSISTENCE_BONUS revisit=$REVISIT_PENALTY \
eps=$EPSILON_EXPLORE/$EPSILON_ANNEAL_UPDATES"
echo "    policy: ${RNN_CELL}/${RNN_NONLINEARITY} h=$HIDDEN_SIZE \
log_std=$INIT_LOG_STD frozen=$FREEZE_LOG_STD lr=$LR clip=$PPO_CLIP_COEF"

cd "$REPO"
source hopfield_nav/navigate_job.sh
