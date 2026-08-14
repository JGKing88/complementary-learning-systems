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
  # Everything wave 3 needs that has never been run together: a warm start via
  # LOAD_CKPT, an interleave schedule with an annealing empty_frac, shuffled
  # regime assignment, and eval_scope=navexpl so the nav metrics appear.
  # Deliberately tiny -- 4 envs x 4 batch x 20 steps is 80 serial calls per
  # update, so the scaffold build dominates and this runs on CPU.
  #   LOAD_CKPT=/path/to/navigate_u450.pt VARIANT=smoke_w3 sbatch ...
  smoke_w3)
    SCHEDULE=${SCHEDULE:-'interleave:4,empty_frac=1.0->0.5,anneal=2'}
    ENVS_PER_WORLD=4; BATCH_ENVS=4; STEPS_PER_ROLLOUT=20
    REGIME_ASSIGNMENT=shuffle
    EVAL_SCOPE=navexpl; EVAL_EVERY=4; CKPT_EVERY=4
    NUM_VAL_ENVS=2; N_VAL_TRIALS=4; EVAL_MAX_STEPS=20
    ;;
  # Same, but exploit-only: the wave-2 config in miniature, to confirm the nav
  # metrics appear before six hours are committed to it.
  smoke_x)
    SCHEDULE=${SCHEDULE:-'exploit:4'}
    ENVS_PER_WORLD=4; BATCH_ENVS=8; STEPS_PER_ROLLOUT=40
    WALL_PENALTY=0; PERSISTENCE_BONUS=0; REVISIT_PENALTY=0
    INIT_LOG_STD=-1.2
    EVAL_SCOPE=navexpl; EVAL_EVERY=4; CKPT_EVERY=4
    NUM_VAL_ENVS=2; N_VAL_TRIALS=8; EVAL_MAX_STEPS=40
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
  # A harder push on the same axis, for round 2. sigma=0.50 is 6x the policy's
  # initial mean magnitude of 0.086, so the sampled steps span the range the
  # policy has to learn to occupy; §3.2 prices the coverage cost of that much
  # noise at ~0.03, and eval scores the mean anyway.
  w1_sig2)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7
    ;;

  # --- round 2, all on the step-magnitude axis the u75 probe identified -----
  #
  # The policy has to move its mean |a| from 0.086 to ~1.0, and until it does,
  # cells_per_step is capped near |a| however good the trajectory is. These are
  # the three ways to make that scalar move faster. NOT included, and why:
  #
  #   REVISIT_PENALTY -- looks like a direct penalty on failing to leave the
  #     cell, which is exactly the failure. It is not. With 1[old] = 1 - 1[new],
  #     n*1[new]*scale - c*1[old] = 1[new]*(n*scale + c) - c, and the constant
  #     cancels under pooled advantage normalization. So it only reweights
  #     novelty slightly flatter in `scale`; it creates no new gradient against
  #     immobility. (It IS non-degenerate with novelty_scale_remaining on --
  #     just not in the way that would help here.)
  #   PERSISTENCE_BONUS -- a cosine, hence scale-invariant. Says nothing about
  #     magnitude at all.
  w1_lr)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    EPSILON_EXPLORE=0.1; LR=1e-3
    ;;
  # DEMOTED, kept for completeness. The clip permits a mean shift of about
  # 0.15*sigma per gradient step = ~0.025 cells/update at sigma=0.165, and the
  # measured ascent is 0.0012 cells/update -- twenty times below it. So the
  # magnitude ascent is gradient-limited, not clip-limited, and raising the
  # clip lifts a ceiling nothing is touching. Run w1_lr instead: it scales the
  # step directly and has the same ~20x of headroom. See docs §3.4.1.
  w1_clip)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    EPSILON_EXPLORE=0.1; PPO_CLIP_COEF=0.3
    ;;
  # Anneal sigma down after it has done its job. A large sigma buys the
  # magnitude ascent (P0.6) but then permanently blurs the policy; the anneal
  # gets the ascent early and a sharp policy late, which neither fixed value
  # can. Uses --log_std_anneal_*, i.e. the composition of the two knobs on
  # Jack's list rather than a new one.
  w1_siganneal)
    SCHEDULE=${SCHEDULE:-'explore:450'}
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7
    LOG_STD_ANNEAL_START_UPDATE=150
    LOG_STD_ANNEAL_END_UPDATE=350
    LOG_STD_ANNEAL_TARGET=-1.8
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

  # === WAVE 2 -- the exploit ceiling ======================================
  #
  # Independent of wave 1: a different regime, a different metric. The point
  # is how close mean_steps gets to the reference table in §3.3.1 -- 10.1 at
  # the cos 0.99 the readout achieves with no distractors, 12.7 at the cos
  # 0.82 it achieves with ten -- NOT success_rate, which that table shows is
  # 1.000 even for a policy whose direction is wrong by 60 degrees.
  #
  # Shaping is zeroed rather than inherited. wall_penalty, persistence_bonus
  # and revisit_penalty are read off cfg, not off the regime, so they apply to
  # exploit rollouts too (P0.3.1); leaving them on would make this baseline a
  # measurement of the leak rather than of following the goal signal.
  #
  # Note two knobs are inert here and are NOT swept: with shaping at zero the
  # per-step reward is `-time_penalty + (goal_reward + time_penalty)*1{goal}`,
  # whose constant cancels in the advantage and whose remainder is a pure
  # scale -- and pooled advantage normalization removes scales. GOAL_REWARD
  # and TIME_PENALTY therefore only reach the policy through the *value* loss,
  # which is not normalized. See §3.5.
  # Rollout shape: 20 envs x 64 batch, NOT wave 1's 80 x 16. Same PPO pool of
  # 1280 trajectories and the same env-steps per update, a quarter of the
  # serial model calls. w1_c20 ran that shape against w1_base and was 26%
  # AHEAD at a matched u100 while costing half the wall-clock per update, so
  # taking it here is acting on evidence rather than saving time blindly.
  # Diversity matters less for exploit anyway: "follow the recall signal" is
  # an env-independent skill, where coverage is not.
  #
  # sigma is the axis, because wave 1 found it dominant for explore AND the
  # exploit regime hardcodes epsilon to 0 (exploit.py:93) -- so sigma is the
  # ONLY exploration the policy has here, not merely the best one. The three
  # values bracket it; -1.2 is the default because wave 1 shows -1.8 is too
  # tight and -0.7 is not yet scored.
  w2_x_base)
    SCHEDULE=${SCHEDULE:-'exploit:600'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    WALL_PENALTY=0; PERSISTENCE_BONUS=0; REVISIT_PENALTY=0
    INIT_LOG_STD=-1.2
    EVAL_SCOPE=navexpl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;
  w2_x_sig2)
    SCHEDULE=${SCHEDULE:-'exploit:600'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    WALL_PENALTY=0; PERSISTENCE_BONUS=0; REVISIT_PENALTY=0
    INIT_LOG_STD=-0.7
    EVAL_SCOPE=navexpl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;
  # v35's sigma, to bracket the axis from below and to say what the historical
  # recipe would have scored on mean_steps under this protocol.
  w2_x_siglo)
    SCHEDULE=${SCHEDULE:-'exploit:600'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    WALL_PENALTY=0; PERSISTENCE_BONUS=0; REVISIT_PENALTY=0
    INIT_LOG_STD=-1.8
    EVAL_SCOPE=navexpl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;
  # Exploit is a far easier objective than explore -- dense +5, a readout at
  # cos 0.99 up to 3 distractors -- so if it is optimization-limited rather
  # than signal-limited this is the cheapest fix. Promoted over a clip sweep
  # by the arithmetic in docs §3.4.1.
  w2_x_lr)
    SCHEDULE=${SCHEDULE:-'exploit:600'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    WALL_PENALTY=0; PERSISTENCE_BONUS=0; REVISIT_PENALTY=0
    INIT_LOG_STD=-1.2; LR=1e-3
    EVAL_SCOPE=navexpl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # NOTE: a `w2_wallres1` variant was designed here and deliberately dropped.
  # P0.9 found wall-distance decodability falls with wall_resolution (R^2 0.45 /
  # 0.27 / 0.17 at 1 / 4 / 8), which looked like the instructed value of 4 was
  # costing coverage. Feeding those R^2 values through the noisy-billiard
  # simulation first (docs §3.1) showed the behaviour saturates long before the
  # decoder does: 0.361 vs 0.352 coverage, a gap of 0.009. Not worth a run.
  # Kept as a comment so the question is not re-opened from the R^2 table alone.

  # --- wave 2, explore arm: the same recipe, but update-limited no longer ---
  #
  # Wave 1 at 80 envs runs 450 updates in six hours and its curve is still
  # climbing roughly linearly there -- extrapolating w1_eps01's late slope puts
  # u450 near 0.26 against a 0.352 target, i.e. the run is update-limited, not
  # recipe-limited. At 20 envs x 64 batch the same PPO pool costs half the
  # wall-clock per update (and w1_c20 was 26% AHEAD per update at that shape),
  # so six hours buys ~1500 updates instead of 450.
  #
  # These carry wave 1's winning noise regime and differ only in sigma, which
  # is the axis wave 1 identified.
  w2_e_long)
    SCHEDULE=${SCHEDULE:-'explore:1500'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-1.2
    EVAL_EVERY=50; CKPT_EVERY=50
    ;;
  w2_e_long2)
    SCHEDULE=${SCHEDULE:-'explore:1500'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7
    EVAL_EVERY=50; CKPT_EVERY=50
    ;;
  # The sigma anneal at the long horizon: high while the magnitude climbs, low
  # once it is solved and the straightness term becomes readable (docs §3.4.1).
  # Only meaningful over a run long enough to have two phases.
  w2_e_anneal)
    SCHEDULE=${SCHEDULE:-'explore:1500'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7
    LOG_STD_ANNEAL_START_UPDATE=400
    LOG_STD_ANNEAL_END_UPDATE=900
    LOG_STD_ANNEAL_TARGET=-1.8
    EVAL_EVERY=50; CKPT_EVERY=50
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
