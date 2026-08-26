#!/bin/bash -l
#SBATCH --job-name=navp2
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_p2_%j.out

# Phase 2: measure the signal, then find the ceilings. See
# docs/EXPERIMENTS_NAV_P2.md -- that document holds the workstreams, the
# hypotheses and the results; this file only holds the knobs.
#
# Split from run_nav_tri.sh rather than extending it because phase 2 changes
# the movement model. Bounded step size makes no phase-1 coverage or
# mean_steps number comparable to a phase-2 one (P2 doc 2.1), and one
# launcher holding both would invite exactly that comparison.
#
#   VARIANT=p4_x sbatch hopfield_nav/run_nav_p2.sh
#   VARIANT=p5_e    sbatch --partition=pi_fiete --time=8:00:00 \
#                          hopfield_nav/run_nav_p2.sh
#
# sbatch's own flags override the directives above, which is how a variant
# reaches pi_fiete (7 d, a100) instead of mit_normal_gpu (6 h, l40s/h200).
#
# Anything can be overridden from the environment for a one-off:
#   VARIANT=p4_x SEED=43 sbatch hopfield_nav/run_nav_p2.sh
#
# Every knob is spelled out rather than inherited from the trainer's argparse
# defaults, for the reason run_repro_v35.sh gives: a run in a comparison series
# must be able to say what it ran, and an inherited default moves silently.

set -euo pipefail

JOB_LABEL=nav_p2
VARIANT=${VARIANT:-smoke}
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
# Jack, phase 2: massive steps are unrealistic. The band also brackets the
# explore optimum -- billiard coverage peaks at |a| ~ 1.0-1.5 and FALLS above
# it, so this costs explore nothing and caps ideal mean_steps at 4.9.
MIN_ACTION_NORM=${MIN_ACTION_NORM:-0.5}
MAX_ACTION_NORM=${MAX_ACTION_NORM:-2.0}
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
INPUT_PREV_ACTION=${INPUT_PREV_ACTION:-1}
# Both channels, per Jack's Q1. The committed action and the realized
# displacement differ whenever the norm clamp or the arena clip bites, and
# the regime cues that compare a change in ||q|| against distance travelled
# need the realized one -- which is exactly where mode-B failures pile up
# (H-wall). The difference is itself information: a clip means a wall.
INPUT_PREV_DISPLACEMENT=${INPUT_PREV_DISPLACEMENT:-1}
# Off by default: phase 2's P1-P5 all ran with the hard env clamp, and
# turning these on silently would make new runs incomparable to them.
ACTION_SQUASH=${ACTION_SQUASH:-0}
STATE_DEPENDENT_STD=${STATE_DEPENDENT_STD:-0}
# Polar (P10). Off by default for the same reason as the two above. Note
# INIT_LOG_STD is INERT under ACTION_POLAR=1 -- there is no Gaussian sigma --
# and INIT_LOG_KAPPA=1.85 is the value that reproduces INIT_LOG_STD=-0.7's
# 0.497 sigma at mid-speed 1.25, so a polar arm starts with the same ~23.8 deg
# of directional noise as the p9 arm it is compared against.
ACTION_POLAR=${ACTION_POLAR:-0}
INIT_LOG_KAPPA=${INIT_LOG_KAPPA:-1.85}
INIT_SPEED_MU=${INIT_SPEED_MU:-0.5}
INIT_SPEED_NU=${INIT_SPEED_NU:-3.0}
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

  # --- smoke: does the config parse with the new channels and bounds? ------
  smoke)
    SCHEDULE=${SCHEDULE:-'exploit:4'}
    ENVS_PER_WORLD=4; BATCH_ENVS=8; STEPS_PER_ROLLOUT=40
    EVAL_SCOPE=navexpl; EVAL_EVERY=4; CKPT_EVERY=4
    NUM_VAL_ENVS=2; N_VAL_TRIALS=8; EVAL_MAX_STEPS=40
    ;;

  # === P4 -- the exploit ceiling ==========================================
  #
  # Is the Hopfield signal good enough, with distractors, for a model that does
  # nothing else? Target: follow_q as high as it goes, mean_steps toward the
  # ideal 4.9 at |a| = 2, success_rate high.
  #
  # sigma is re-bracketed rather than inherited. Phase 1 put it at 0.50, but
  # that was chosen when sigma was the ONLY channel through which the policy
  # learned step magnitude, from a start of 0.086. min_action_norm=0.5 removes
  # that job and leaves sigma as mostly angular noise, so the old optimum does
  # not transfer -- and assuming it did would repeat phase 1's own mistake of
  # reading a knob's value from a regime it no longer lives in.
  p4_x|p4_x_s12|p4_x_s18)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p4_x)     INIT_LOG_STD=-0.7 ;;   # sigma 0.50, phase 1's pick
      p4_x_s12) INIT_LOG_STD=-1.2 ;;   # sigma 0.30
      p4_x_s18) INIT_LOG_STD=-1.8 ;;   # sigma 0.165, v35's
    esac
    ;;

  # === P4b -- is TIME_PENALTY holding the speed down? ======================
  #
  # At u850 the exploit arm runs at mean_speed 1.50 against a permitted 2.0, and
  # 91% of the ideal mean_steps FOR ITS OWN SPEED -- so the remaining gap is
  # speed, not path quality. The incentive to hurry is time_penalty against
  # goal_reward: at 0.05 vs 2.0 the agent is indifferent between arriving and
  # taking 40 extra steps, which is enormous slack when a trial takes 7. Raising
  # the penalty tightens that to 20 steps at 0.10 and 13 at 0.15.
  #
  # The risk is give-up behaviour on distant starts: if expected steps exceed the
  # indifference point the agent would rather not try. Mean start distance is
  # 10.85, about 7 steps at speed 1.5, so 0.10 is comfortable and 0.15 is the
  # aggressive end. Watch success_rate, not just mean_steps -- a mean_steps win
  # bought by abandoning the far starts is not a win.
  p4_tp10|p4_tp15)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p4_tp10) TIME_PENALTY=0.10 ;;
      p4_tp15) TIME_PENALTY=0.15 ;;
    esac
    ;;

  # === P5 -- the explore ceiling, as calibration ==========================
  #
  # Not a search. Phase 1 already showed explore saturates the reactive line
  # and that chase_q goes to zero, so the science is done; what is not reusable
  # is the NUMBER. 0.385 appears in the phase-1 doc both as the scripted
  # billiard baseline at eps=0 and as the explore specialist, and the two
  # cannot be told apart from that document. Bounded steps move the ceiling
  # anyway -- 0.378 at |a| ~ 1.25. One clean reference run.
  p5_e)
    SCHEDULE=${SCHEDULE:-'explore:1500'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    EVAL_SCOPE=expl; EVAL_EVERY=50; CKPT_EVERY=50
    ;;

  # === P9 -- the action parameterization =================================
  #
  # Section 8.2 measured that every P4 arm commands ||mu|| far past the 2.0 cap
  # (2.19 to 4.91) and that the sigma bracket therefore varied clamp depth
  # rather than exploration -- nominal sigma spanned 3.0x while the effective
  # angular noise sigma/||mu|| spanned only 1.33x. Section 9.1 measured the
  # explore arm at ||mu|| = 8.18 against a realized 1.98.
  #
  # Two changes, and a third arm so they can be attributed:
  #
  #   p9_sq     radial tanh on the policy MEAN
  #   p9_sq_std the same, plus a per-state log_std head
  #
  # sigma-only is deliberately NOT run. It is predicted to be neutered by the
  # same ||mu|| compensation that flattened the sigma bracket, so a null result
  # there could not distinguish "does not help" from "could not act".
  #
  # Read from the new per-update mu_norm / sigma / ang_noise logs, not from the
  # end metrics. The pass/fail is whether the sigma head takes over the
  # state-dependent modulation: if it stays flat while ||mu|| does the
  # modulating, the residual 4x magnitude channel still binds and the answer is
  # a polar parameterization.
  p9_sq|p9_sq_std)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_SQUASH=1
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p9_sq_std) STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0 ;;
    esac
    ;;

  # Explore-side counterparts. Section 9.1's clamp pathology was measured on
  # p5_e, so the explore arm is where the coverage prediction can be checked:
  # billiard peaks at ||a|| ~ 1.25 and the clamped policy sits at 1.98, so a
  # policy free to choose its magnitude should land nearer the optimum.
  p9_e_sq|p9_e_sq_std)
    SCHEDULE=${SCHEDULE:-'explore:1500'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    ACTION_SQUASH=1
    EVAL_SCOPE=expl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p9_e_sq_std) STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0 ;;
    esac
    ;;

  # === P6 -- interleaved ==================================================
  #
  # Baseline arm only for now. The distractor-curriculum arm (Q4b) needs the
  # trainer to anneal n_train_distractors_max, which it cannot yet do -- adding
  # that is a P6 task, not a launcher entry. A variant that silently ran at a
  # fixed distractor count while claiming to be a curriculum is exactly what
  # spelling every knob out in this file is meant to prevent.
  p6_base)
    SCHEDULE=${SCHEDULE:-'interleave:2000,empty_frac=0.5'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; INIT_LOG_STD=-0.7; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    ;;

  # === P10 -- the polar action parameterization ===========================
  #
  # Section 9.3's control settled it: the state-dependent sigma head DISPLACED
  # NOTHING. ||mu|| modulated 1.234x on distractor count WITHOUT the head and
  # 1.220x with it, while sigma itself moved 1.086x and was flat against
  # distance to goal -- the one place P1 shows the readout actually collapses.
  # The policy kept buying directional exploration with speed because
  # sigma/||mu|| still varies 4x over [0.5, 2], wider than the 2.2x modulation
  # it exhibits. Bounding the mean was necessary and not sufficient.
  #
  # Polar removes the channel rather than competing with it: heading and speed
  # are separate factors, so neither can pay for the other.
  #
  #   p10_pol      exploit, learned speed
  #   p10_pol_v1   exploit, speed frozen at 1.0
  #   p10_e_pol    explore, learned speed
  #   p10_e_pol_v1 explore, speed frozen at 1.0
  #
  # All four run STATE_DEPENDENT_STD=1: under polar that makes kappa and nu
  # per-state heads, and whether KAPPA picks up the state-dependence sigma
  # refused to is the whole question. Controls already exist -- p9_sq and
  # p9_sq_std complete the 2x2 -- so no new control arm is needed.
  #
  # THE FALSIFIER: if ||mu|| (logged as mu_norm = mean speed) still modulates
  # ~1.23x across distractor levels with heading noise fully decoupled, then
  # that modulation was a genuine speed policy all along, the residual-channel
  # story is wrong, and section 9.3's conclusion needs retracting. The frozen
  # arms are the sharp version of the same test: with speed constant by
  # construction, everything the policy does must go through kappa.
  #
  # Read dir_norm too. It is the direction head's magnitude, a gauge freedom
  # (atan2 is scale-invariant) that nothing in the objective pressures; if it
  # decays toward dir_soft the heading is being held near-uniform.
  p10_pol|p10_pol_v1)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      # 1.0 sits at the LOW edge of the measured billiard plateau (peak ~1.25,
      # band 1.0-1.5), and makes ||a|| a unit so displacements and
      # q-magnitudes are directly comparable. Jack's pick.
      p10_pol_v1) FREEZE_SPEED=1.0 ;;
    esac
    ;;

  p10_e_pol|p10_e_pol_v1)
    SCHEDULE=${SCHEDULE:-'explore:1500'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    EVAL_SCOPE=expl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p10_e_pol_v1) FREEZE_SPEED=1.0 ;;
    esac
    ;;

  # Polar smoke: the unit tests never touch the rollout collector, the channel
  # assembly, the evaluators or checkpoint round-tripping.
  p10_smoke|p10_smoke_v1)
    SCHEDULE=${SCHEDULE:-'exploit:4'}
    ENVS_PER_WORLD=4; BATCH_ENVS=8; STEPS_PER_ROLLOUT=40
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    EVAL_SCOPE=navexpl; EVAL_EVERY=4; CKPT_EVERY=4
    NUM_VAL_ENVS=2; N_VAL_TRIALS=8; EVAL_MAX_STEPS=40
    case "$VARIANT" in
      p10_smoke_v1) FREEZE_SPEED=1.0 ;;
    esac
    ;;

  *)
    echo "ERROR: unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac

export WANDB_NAME=${WANDB_NAME:-navp2_${VARIANT}_s${SEED}_${SLURM_JOB_ID:-local}}

echo "=== nav_p2 variant=$VARIANT seed=$SEED ==="
echo "    schedule   : $SCHEDULE"
echo "    rollout    : ${ENVS_PER_WORLD} envs x ${BATCH_ENVS} batch x ${STEPS_PER_ROLLOUT} steps"
echo "                 pool=$((ENVS_PER_WORLD * BATCH_ENVS)) trajectories, \
$((ENVS_PER_WORLD * BATCH_ENVS * STEPS_PER_ROLLOUT)) env-steps/update, \
$((ENVS_PER_WORLD * STEPS_PER_ROLLOUT)) serial calls/update"
# goal_reward is echoed with the shaping, not with the env, because from wave 3
# on it is a shaping knob: inside one pooled advantage normalization it sets the
# ratio between the explore and exploit regimes. See docs §3.5.
echo "    shaping    : nov=$NOVELTY_REWARD scale=$NOVELTY_SCALE_REMAINING/$NOVELTY_SCALE_CAP \
wall=$WALL_PENALTY pers=$PERSISTENCE_BONUS revisit=$REVISIT_PENALTY \
goal=$GOAL_REWARD time=$TIME_PENALTY"
echo "    noise      : eps=$EPSILON_EXPLORE/$EPSILON_ANNEAL_UPDATES \
init_log_std=$INIT_LOG_STD freeze=$FREEZE_LOG_STD ent=$MOVE_ENT_COEF"
echo "    trunk      : $RNN_CELL/$RNN_NONLINEARITY h=$HIDDEN_SIZE"
echo "    movement   : |a| in [$MIN_ACTION_NORM, $MAX_ACTION_NORM] prev_action=$INPUT_PREV_ACTION prev_disp=$INPUT_PREV_DISPLACEMENT"
if [ "${ACTION_POLAR:-0}" = "1" ]; then
  echo "    action     : POLAR  init_log_kappa=$INIT_LOG_KAPPA (kappa=6.36, ~23.8 deg) \
speed=${FREEZE_SPEED:-learned mu0=$INIT_SPEED_MU nu0=$INIT_SPEED_NU} \
state_dep=$STATE_DEPENDENT_STD  [init_log_std is INERT]"
else
  echo "    action     : cartesian  squash=$ACTION_SQUASH state_dep=$STATE_DEPENDENT_STD"
fi

cd "$REPO"
source hopfield_nav/navigate_job.sh
