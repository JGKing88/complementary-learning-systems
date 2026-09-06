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
# Depth {1} only, from 2026-09-06 (Jack). §7.7 ablated the depth channel and
# {1} is NEVER worse than {1,2,3} for a regime classifier; §5.4 says why --
# the recall does not converge, it DRIFTS, so depths 2 and 3 are degraded
# states sampling the transient of a power iteration away from the answer.
# Dropping them also removes the four-channel trap in EXPLOIT_DIAGNOSTIC §7.
# `d1_ms3` keeps "1 2 3" so the change is measured, not assumed.
# NOTE: this MOVES a default, which this launcher's header warns about. Every
# run up to and including P35 trained with "1 2 3"; a P2 number is therefore
# not comparable to a post-2026-09-06 one on this axis.
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-"1"}

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
# §18.8: score persistence on realized displacement, not the commanded
# action. Default 0 = every run up to P20.
PERSISTENCE_REALIZED=${PERSISTENCE_REALIZED:-0}

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
  # -------------------------------------------------------------------------
  # P16 -- a SATURATING Hopfield, on the v35 encoder.
  #
  # Section 14 found the mechanism behind every exploit failure: q(x) develops a
  # spurious sink for ~19% of memory draws, and a sink is present in all 24
  # failures and absent in all 155 clean-field episodes. The reason it can
  # happen is that the recall is not an attractor at all -- `tanh(beta * W x)`
  # runs with an argument around 1e-4, so tanh is in its linear region and
  # retrieval is a weighted BLEND of the stored patterns. A blend can point
  # anywhere, including into a vortex, without any pattern winning.
  #
  # This arm makes both nonlinearities actually saturate:
  #
  #   ENCODER_GAIN=300   the code is normalize(tanh(gain * z)), so this drives
  #                      the embedding toward binary. It changes the code's
  #                      SHAPE, not its magnitude -- the normalize is after.
  #   HOPFIELD_BETA=1e6  the recall argument reaches ~1e2, so tanh saturates and
  #                      retrieval becomes a sign-thresholded attractor rather
  #                      than a blend.
  #
  # Both were previously impossible to set independently: beta had no flag and
  # defaulted to the encoder gain, and --encoder_gain never reached the model.
  #
  # The v35 encoder is used because Jack asked for it; it shares lambdas
  # (11 12 13) and size (20) with the P2 world, so the scaffold is unchanged.
  # NOTE it is a departure from the encoder marked FIXED BY INSTRUCTION above,
  # and it moves two things at once (encoder AND both gains), so this arm is not
  # a clean single-factor test against p10_pol_v1 -- it is a "does saturation
  # help at all" probe. If it does, the factors want separating.
  #
  # Otherwise byte-identical to p10_pol_v1, the frozen-speed exploiter that
  # anchors sections 12-15, so the diagnostics apply unchanged.
  # -------------------------------------------------------------------------
  # P17 -- the CLEAN single-factor arm: raise the encoder gain, change nothing
  # else.
  #
  # Section 15.3's factor grid found that of the three things p16_sat changed
  # (encoder, encoder gain, hopfield beta) only ONE does any work:
  #
  #   encoder gain 5 -> 300   spurious sinks 37/192 -> 2/192 on THIS encoder
  #   hopfield beta            no effect at all once the code is binary
  #                            (2/192 either way, identical to 3 decimals),
  #                            and actively harmful on a smooth code (37 -> 62)
  #   v35 encoder              19% -> 13% on its own; not needed
  #
  # So this arm keeps the P2 encoder marked FIXED BY INSTRUCTION and moves
  # exactly one knob. Everything else is p10_pol_v1, which makes it the first
  # arm in this line that is a clean comparison against a measured baseline.
  #
  # HOPFIELD_BETA IS PINNED, and must be. cfg.hopfield.beta defaults to the
  # encoder gain, so setting ENCODER_GAIN=300 alone would drag beta to 300 as
  # well -- two factors, and an untested cell. 5.0 is p10_pol_v1's own value,
  # which is what the 2/192 grid cell was measured at.
  #
  # Prediction on record: the readout field is already known to be clean here
  # (2/192 sinks, goal basin 0.991), and section 14 showed a sink is present in
  # every failure and absent in all 155 clean-field episodes. If the field is
  # what limits exploit, this should beat p10_pol_v1's 0.875 at ten distractors.
  # If it does not, a clean field is not sufficient and the limit is elsewhere
  # -- which section 14 already hinted at, since the agent escaped 13 of 37
  # sinks anyway.
  # -------------------------------------------------------------------------
  # P18 -- the w49_g100_knee encoder, both gains at 300.
  #
  # Jack's pick. That encoder was trained at gain 100 already (its checkpoint
  # and model-config gains agree at 100.0, unlike v35's 3.699/5.0 split), so
  # 300 sharpens a code that is already fairly binary rather than binarising a
  # smooth one. lambdas [11 12 13] and out_dim 1024 match the P2 world, so the
  # scaffold geometry is unchanged.
  #
  # Note HOPFIELD_BETA=300 is what beta would default to here anyway, since it
  # follows the encoder gain when unset. It is written out because the default
  # coupling is exactly the trap that made p16_sat a two-factor arm, and a run
  # in a comparison series should say what it ran.
  #
  # Beta at 300 rather than p16_sat's 1e6 matters: 1e6 inflated ||q||, which
  # drove dir_norm to ~1.2 against p17_gain's ~0.25, which ran kappa to ~148 and
  # locked the policy onto a ~4 degree heading before it had learned anything.
  # That arm was flat at 0.05 success for 200 updates and was cancelled. 300 is
  # far below that regime.
  p18_knee)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w49_g100_knee/008_eps1_rate0.5_seed=42/encoder_final.pt
    ENCODER_GAIN=300
    HOPFIELD_BETA=300
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    FREEZE_SPEED=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    ;;

  p17_gain)
    ENCODER_GAIN=300
    HOPFIELD_BETA=5.0
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    FREEZE_SPEED=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    ;;

  p16_sat)
    # Set unconditionally, NOT with :- . The fixed block at the top of this
    # file has already assigned ENCODER by the time a variant runs, so a
    # `${ENCODER:-...}` here is a no-op and the arm would silently train on the
    # P2 encoder while claiming to use v35's.
    ENCODER=encoders/run_20260422_185816/encoder_best.pt
    ENCODER_GAIN=300
    HOPFIELD_BETA=1e6
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    FREEZE_SPEED=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    ;;

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

  # === P11 -- why does convergence take 550 updates? =======================
  #
  # `p10_pol_v1` reaches 1.000 success and holds it for 29 straight evals, but
  # only after swinging 0.03 / 0.21 / 0.90 / 0.51 / 0.91 / 1.00 / 0.49 through
  # its first 550 updates. Three single-factor arms against that run as the
  # control -- identical in every other respect, so each answers one question.
  #
  #   p11_cur  distractor curriculum, max 0 -> 10 over 400 updates
  #   p11_tp   time_penalty 0.05 -> 0.02
  #   p11_eps  epsilon 0.1/200 -> 0.3/600
  #
  # NOT run as a factorial: three arms crossed is eight runs to attribute a
  # first-500-update effect, and the control already exists.
  #
  # Ranked by how well motivated they are, which is also how much to believe a
  # null from each:
  #
  # 1. CURRICULUM. The strongest signal in the p10 data is that success at ZERO
  #    distractors locks at u600 and never wavers, while success at TEN
  #    oscillates 0.90-0.98 for the remaining 1400 updates and is still doing so
  #    at u2000. That is two problems, one solved and one never solved. The
  #    probe puts trajectory q_accuracy at 0.45-0.60 at ten distractors against
  #    0.99 at zero, so roughly half of all training episodes teach from a
  #    direction signal near a coin flip, with no cue telling the policy which
  #    kind of episode it is in. Prediction: the chaotic phase shortens AND the
  #    ten-distractor band tightens, because the policy learns to trust `q`
  #    before being asked to handle a `q` it cannot trust.
  #
  # 2. TIME_PENALTY. 0.05 x 200 steps = -10 for any failure against +1.4 for a
  #    12-step success. Early on every episode fails and collects ~-10, so
  #    advantages are noise until the first successes appear -- the
  #    plateau-then-breakthrough-then-collapse shape. Only time_penalty moves,
  #    not goal_reward: under pooled advantage normalization only their RATIO
  #    matters, so varying both would be one confounded knob (see the shaping
  #    degeneracy note).
  #
  # 3. EPSILON. Weakest, and a null here means least: epsilon actions are masked
  #    out of the PPO surrogate by policy_action_mask, so they change which
  #    states are visited, never the gradient. Worth one arm because it is the
  #    knob that governs whether the goal gets found at all early on.
  # p11_cur_tp crosses the two arms that are actually well motivated. It is the
  # one combination worth paying for: if either alone shortens the chaotic
  # phase, the cross says whether they address the same bottleneck (no further
  # gain) or different ones (additive). The epsilon arm is not crossed -- a
  # knob that cannot touch the gradient does not earn a cell.
  p11_cur|p11_tp|p11_eps|p11_cur_tp)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    FREEZE_SPEED=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      # max ramps FROM n_train_distractors_max TO _max_end, so the start value
      # is the one that has to be 0.
      p11_cur) N_TRAIN_DISTRACTORS_MAX=0; N_TRAIN_DISTRACTORS_MAX_END=10
               N_TRAIN_EMP_DISTRACTORS_MAX=0; N_TRAIN_EMP_DISTRACTORS_MAX_END=10
               DISTRACTOR_CURRICULUM_UPDATES=400 ;;
      p11_tp)  TIME_PENALTY=0.02 ;;
      p11_eps) EPSILON_EXPLORE=0.3; EPSILON_ANNEAL_UPDATES=600 ;;
      p11_cur_tp)
               N_TRAIN_DISTRACTORS_MAX=0; N_TRAIN_DISTRACTORS_MAX_END=10
               N_TRAIN_EMP_DISTRACTORS_MAX=0; N_TRAIN_EMP_DISTRACTORS_MAX_END=10
               DISTRACTOR_CURRICULUM_UPDATES=400
               TIME_PENALTY=0.02 ;;
    esac
    ;;

  # === P12 -- variable speed WITHOUT fast speed ============================
  #
  # The existing pair confounds two things. `p10_pol_v1` is pinned at exactly
  # 1.0; `p10_pol` may range over [0.5, 2] and learns ~1.8. So "learned beats
  # frozen on steps" could mean the policy benefits from CHOOSING its speed, or
  # merely from being allowed to go FAST. Nothing sits in between.
  #
  # These bound the speed at [0.5, 1.0]: the policy still chooses, but can never
  # exceed the frozen arm's fixed value. Reading against both existing arms:
  #
  #   beats p10_pol_v1  ->  the choosing is what helps
  #   matches it        ->  the magnitude was the whole story
  #
  # Jack's call, and it deliberately narrows the [0.5, 2] band fixed earlier in
  # this file -- for these two arms only.
  #
  # p12_lo_curtp additionally carries the P11 curriculum + time_penalty
  # treatment, so if that combination turns out to matter it is not confined to
  # the wide-speed setting.
  #
  # EPSILON_EXPLORE stays 0.1 for exact config parity with p10_pol even though
  # it is provably inert on an exploit-only schedule -- the trainer now prints a
  # warning saying so, which is the honest record rather than a silent 0.
  p12_lo|p12_lo_curtp)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MAX_ACTION_NORM=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p12_lo_curtp)
        N_TRAIN_DISTRACTORS_MAX=0; N_TRAIN_DISTRACTORS_MAX_END=10
        N_TRAIN_EMP_DISTRACTORS_MAX=0; N_TRAIN_EMP_DISTRACTORS_MAX_END=10
        DISTRACTOR_CURRICULUM_UPDATES=400
        TIME_PENALTY=0.02 ;;
    esac
    ;;

  # --- P10b: is the frozen arm's failure kappa runaway, or speed 1.0? -------
  #
  # At u50 the frozen-speed exploit arm sits at success 0.031 against the
  # learned-speed arm's 0.677 -- a 22x gap -- with kappa at 23.2 against 7.0
  # and angular noise 13.1 deg against 22.9. It sharpened its heading hard
  # around a direction it had not learned.
  #
  # Speed 1.0 alone does not explain it: at 1.0 the ideal mean_steps over a
  # ~10.8-cell start distance is ~10 against 200 available, so the cap on
  # SPEED cannot drive success to 3%. Premature convergence can.
  #
  # The likely coupling, and the reason it is worth one job to test: in the
  # learned arm a bad heading can be hedged by SLOWING DOWN -- a shorter step
  # overshoots less. With speed pinned that hedge is gone, heading errors cost
  # more, and PPO's only remaining lever is to sharpen kappa, which removes
  # the exploration that would have found the right heading. If that is right,
  # speed and heading are independent in the PARAMETERIZATION but not in their
  # effect on the task, which is a P10 result rather than a nuisance.
  #
  # Bracketed rather than guessed at a single value: 4x and 10x the 0.005 that
  # the learned arm is stable at.
  p10_pol_v1_e20|p10_pol_v1_e50)
    SCHEDULE=${SCHEDULE:-'exploit:2000'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    FREEZE_SPEED=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=50; CKPT_EVERY=50
    case "$VARIANT" in
      p10_pol_v1_e20) MOVE_ENT_COEF=0.02 ;;
      p10_pol_v1_e50) MOVE_ENT_COEF=0.05 ;;
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

  # === P19 -- the w52 attract-0.5 encoder, and what a curriculum costs ======
  #
  # Jack's ask, in three parts: run this encoder; reach the best accuracy in as
  # FEW UPDATES as possible; learned speed in [0.5, 1.0]. He expects a
  # distractor curriculum to help, so curriculum LENGTH is the axis these arms
  # sweep and `p19_nc` is the control that says what it costs.
  #
  # ENCODER -- w52_attract_fwhm/001_att0.5_seed=43. Its gain schedule ends at
  # 100.0, so ENCODER_GAIN=100 and HOPFIELD_BETA=100 are what the checkpoint
  # and the default beta-follows-gain coupling would produce anyway. They are
  # written out because that silent coupling is exactly what made p16_sat a
  # two-factor arm (section 15.1): a run in a comparison series says what it
  # ran rather than inheriting it.
  #
  # The scaffold does not move with the encoder. lambdas [11 12 13], out_dim
  # 1024, local radius 20 and fwhm_ratio 0.25 all match the shared defaults
  # above, exactly as they did for the v35 and knee encoders.
  #
  # ONE THING TO WATCH. This encoder's unique radius is far shorter than the
  # knee encoder's -- r_min 5.0 / median 9.5 against 12.0 / 17.0, at the same
  # alias rate (0.871 vs 0.865). The arena is 20x20 and a typical start-goal
  # distance is ~10.8 cells, so the MEDIAN unique radius sits BELOW the distance
  # the agent usually has to cover. If that bites it will show up as far-field
  # q errors rather than the near-goal ones section 16.5 left open. The
  # policy-free field map (run_readout_field.sh) is the cheap way to see it, so
  # it is run on the FIRST checkpoint rather than after training finishes.
  #
  # SPEED -- learned in [0.5, 1.0], per instruction. MIN 0.5 and MAX 1.0 are
  # both spelled out, and FREEZE_SPEED is left unset so the policy chooses.
  # This is p12_lo's setting, which section 9.9 measured as costing nothing on
  # path quality (directness 1.081x, i.e. the beelines Jack is asking for) and
  # buying a lot of stability: minimum after breakthrough 0.844 against the
  # pinned arm's 0.490.
  #
  # THE CURRICULUM AXIS, and why it needs a control. Section 9.9 measured the
  # 400-update curriculum as winning stability outright -- never below 0.979
  # after breakthrough against the control's 0.490 -- but REACHING breakthrough
  # LATER, u300 against u150. So on the exact axis Jack is asking about, the
  # only curriculum this project has ever run LOST. That is a reason to bracket
  # it rather than assume it.
  #
  # The prediction on record: p11_cur broke through about 75% of the way
  # through its own ramp, so if breakthrough tracks ramp PROGRESS rather than
  # update count, a 100-update ramp should break through near u75 and beat the
  # no-curriculum control. If instead breakthrough is pinned near u150-u300
  # regardless of ramp length, the curriculum is simply a delay and p19_nc wins.
  # p19_nc is what makes that falsifiable.
  #
  #   p19_nc    no curriculum -- max=10 from update 1 (the shared default)
  #   p19_c100  max 0 -> 10 over 100 updates
  #   p19_c300  max 0 -> 10 over 300 updates
  #
  # exploit:800 with EVAL_EVERY=25, rather than the 2000/50 the P10-P18 arms
  # use: the question is entirely about the early curve, p17_gain was flat from
  # u50 to u1100, and the 6 h partition wall lands near u1100 anyway (both p17
  # and p18 TIMEOUT-ed there). 32 eval points clears by a wide margin the
  # >=4-point bar this project's eval noise requires for a directional claim.
  p19_nc|p19_c100|p19_c300|p19_b5|p19_e20|p19_kcap|p19_kcur)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'exploit:800'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    EVAL_SCOPE=navexpl; EVAL_EVERY=25; CKPT_EVERY=25
    case "$VARIANT" in
      # p19_b5 -- the SAME encoder, beta dropped 100 -> 5.0, nothing else moved.
      #
      # Diagnosed at u70 of p19_nc: hopfield_beta inflates ||q||, and dir_norm
      # 1.36 drives kappa to 147.7 -- the exact value that killed p16_sat at
      # beta 1e6. Angular noise collapses from ~21 deg to 4.7 deg, so the policy
      # locks a heading before it has learned which heading is right.
      #
      #   beta 5.0 (p17_gain)  dir_norm 0.26  kappa  8.5  beeline at u200
      #   beta 300 (p18_knee)  dir_norm 1.06  kappa 133    beeline at u500
      #   beta 100 (p19_nc)    dir_norm 1.36  kappa 148    flat through u150
      #
      # This also RETRACTS the p18_knee comment above, which justified beta=300
      # with "300 is far below that regime". It is not: p18_knee ran kappa to
      # 133 and paid 2.5x in updates-to-beeline. It did not fail outright, so
      # the cost was never attributed.
      #
      # 5.0 is p17_gain's value, i.e. the one cell of the section 15 grid where
      # kappa did not run away. Prediction on record: dir_norm ~0.25, kappa <15,
      # and the beeline reached far sooner than p19_nc's -- which is the whole
      # objective (§17.5). If kappa still runs away at 5.0 the cause is the
      # encoder, not beta, and that is worth knowing too.
      p19_b5) HOPFIELD_BETA=5.0 ;;
      # p19_e20 -- resist the early kappa sharpening with entropy pressure.
      #
      # §17.7 established that kappa runaway is set by the ENCODER's readout
      # scale at update 1 (dir_norm 0.35 for the w52/knee family against 0.127
      # for the P2-fixed one) and that NEITHER encoder_gain NOR hopfield_beta
      # moves it. So the lever has to act on the policy, not on the memory.
      #
      # MOVE_ENT_COEF is that lever and it has never been run: p10_pol_v1_e20
      # and _e50 exist in this file with no checkpoint directory anywhere. The
      # entropy bonus opposes exactly what kappa runaway is -- the collapse of
      # angular spread -- so it is the one knob aimed at the mechanism rather
      # than at a correlate of it.
      #
      # 0.02 is 4x the 0.005 default, the lower of the two values the unrun
      # e20/e50 pair bracketed. The upper one is not launched at the same time
      # because the GPU quota is 2 and p19_nc must keep running as the control.
      #
      # Everything else is p19_nc, INCLUDING beta=100 and gain=100 -- Jack's
      # specified config -- so this is a clean single-factor test against a
      # control that is still running.
      #
      # Prediction on record: kappa stays below ~30 through u50 (p19_nc hits
      # 119 there) and ang_noise stays above ~0.15 rad. If kappa still runs away
      # then entropy cannot reach it either, and the remaining candidate is the
      # readout magnitude itself -- which would mean normalising the hopfield
      # input, and INPUT_HOPFIELD_RAW is FIXED BY INSTRUCTION, so that is a
      # question for Jack rather than a knob to turn.
      p19_e20) MOVE_ENT_COEF=0.02 ;;
      # p19_kcap -- bound kappa, instead of pushing back on it with entropy.
      #
      # The diagnostic on p19_nc u225 (§17.8) is unambiguous: on FAILURES the
      # readout is essentially perfect -- q_acc 0.988 at 0 distractors, 0.962 at
      # 10 -- and the policy drives at follow_q -0.726, nearly straight backward,
      # pinned against a wall for 100% of failures. Nothing is wrong with the
      # encoder, the memory or the field. The policy locked a wall-ward heading
      # and ~5 deg of angular noise cannot break it out.
      #
      # And kappa is not "running away" -- it is SATURATED. --log_kappa_max
      # defaults to 5.0, so kappa_max = e^5 = 148.4, and p19_nc measures 147.66
      # at u70. It is sitting on the ceiling. So did p18_knee (133) and p16_sat.
      #
      # 2.5 -> kappa_max = 12.2, just above the 8.5 that p17_gain -- the fast
      # arm, beeline at u200 -- settled at naturally, and far below the clamp.
      # At kappa 12 the angular sd is ~0.29 rad (~16 deg) against p19_nc's 0.08
      # rad (4.7 deg), so the policy keeps the exploration it needs to discover
      # that its heading is wrong.
      #
      # A hard bound rather than entropy pressure because MOVE_ENT_COEF=0.02
      # (p19_e20) cut kappa 45% at u30 and bought NO success -- 0.083/0.104/
      # 0.115/0.104/0.125 against the control's flat 0.09. A soft push against a
      # saturating quantity is the wrong instrument.
      #
      # Everything else is p19_nc, INCLUDING Jack's beta=100 and gain=100 and
      # the default MOVE_ENT_COEF, so this is a clean single factor.
      #
      # Prediction on record: kappa pinned at 12.2, ang_noise >= 0.25 rad, and
      # follow_q POSITIVE on failures. If success still does not move with a
      # 0.99-accurate readout and a policy free to turn, then the fault is in
      # the reward or the action pipeline, not in exploration.
      p19_kcap) LOG_KAPPA_MAX=2.5 ;;
      # p19_kcur -- Jack's ORIGINAL question, finally in a readable regime.
      #
      # He opened by saying a distractor curriculum would probably help reach
      # best accuracy in fewest updates. Two attempts could not answer it:
      # p19_c100 ran with the policy locked at the kappa clamp (§17.8) so both
      # it and its control were flat at ~0.09 and the axis was confounded, and
      # p19_c300 never started (GPU quota).
      #
      # §17.9 produced a config that learns -- kappa capped at 12.2, 0.990 @10
      # distractors by u125. So this is p19_kcap PLUS the curriculum, against
      # p19_kcap itself as the control. Single factor, readable at last.
      #
      #   max ramps FROM n_train_distractors_max TO _max_end, so the START
      #   value is the one that has to be 0.
      #
      # 100 updates rather than p11_cur's 400 because the whole regime is now
      # ~4x faster: p19_kcap saturates at u125, so a 400-update ramp would not
      # finish until long after the arm it is meant to accelerate has converged.
      #
      # Prediction on record, and it is NOT that the curriculum wins. §9.9
      # measured the only curriculum this project has run as REACHING
      # breakthrough later (u300 vs u150) while winning stability outright
      # (never below 0.979 after, against 0.490). On Jack's objective -- the
      # beeline, fast AND stable -- those pull opposite ways, so the honest
      # expectation is: later first-crossing, higher minimum-after. If it is
      # later on BOTH, the curriculum is simply a delay in this regime and the
      # question is closed.
      p19_kcur) LOG_KAPPA_MAX=2.5
                N_TRAIN_DISTRACTORS_MAX=0; N_TRAIN_DISTRACTORS_MAX_END=10
                N_TRAIN_EMP_DISTRACTORS_MAX=0; N_TRAIN_EMP_DISTRACTORS_MAX_END=10
                DISTRACTOR_CURRICULUM_UPDATES=100 ;;
      # max ramps FROM n_train_distractors_max TO _max_end, so the START value
      # is the one that has to be 0.
      p19_c100) N_TRAIN_DISTRACTORS_MAX=0; N_TRAIN_DISTRACTORS_MAX_END=10
                N_TRAIN_EMP_DISTRACTORS_MAX=0; N_TRAIN_EMP_DISTRACTORS_MAX_END=10
                DISTRACTOR_CURRICULUM_UPDATES=100 ;;
      p19_c300) N_TRAIN_DISTRACTORS_MAX=0; N_TRAIN_DISTRACTORS_MAX_END=10
                N_TRAIN_EMP_DISTRACTORS_MAX=0; N_TRAIN_EMP_DISTRACTORS_MAX_END=10
                DISTRACTOR_CURRICULUM_UPDATES=300 ;;
    esac
    ;;

  # === P20 -- the explore side of the w52 encoder ===========================
  #
  # P19 delivered the exploit half on Jack's encoder (17.10). This is the
  # matching explore half, and it is deliberately p19_kcap with three things
  # changed and NOTHING else: the schedule, the eval scope, and kappa.
  #
  # Explore depends on the OPPOSITE property of the memory from exploit. The
  # exploit arm needs ||q|| to point at the goal; the explore arm needs ||q||
  # to be SMALL when only distractors are stored, so the policy does not chase
  # a phantom (phase 1's corner trap). That is the thing an encoder swap could
  # break, and it is why this run is not a formality.
  #
  #   p20_e       p19_kcap's config on an explore schedule, kappa uncapped
  #   p20_e_kcap  the same, plus LOG_KAPPA_MAX=2.5
  #
  # MAX_ACTION_NORM=1.0 is carried over from P19 per Jack's speed instruction,
  # NOT the 2.0 that p10_e_pol ran at. This makes raw `cells_per_step`
  # incomparable to p10_e_pol's 0.75 BY CONSTRUCTION -- cps depends on stride
  # length. The comparison that survives is `strategy_efficiency`, which
  # divides by a perfect billiard AT THE REALIZED SPEED (behavior_probe.py:540)
  # and is the explore-side analogue of 17.5's `directness`. Quote that, not
  # cps, against p10_e_pol's 1.113. 2.1 says billiard coverage peaks at
  # ||a|| ~ 1.0-1.5 and falls above it, so capping at 1.0 sits at the low edge
  # of the optimum and should cost little against 2.0 -- possibly help.
  #
  # explore:700 rather than p10_e_pol's 1500. Its series converged at u200-250
  # (cps 0.747 at u200, 0.750 at u250) and the remaining 1250 updates bought
  # +0.026 against an oscillation band of 0.61-0.78 -- i.e. nothing that clears
  # this project's eval-noise bar. 700 is 2.8x the convergence point, gives 28
  # eval points at EVAL_EVERY=25 (p10_e_pol had 30 across twice the updates),
  # and lands near 4.5 h against the 6 h partition wall.
  #
  # PREDICTIONS ON RECORD
  #
  # H1 -- the encoder does NOT hurt explore. 7.7.2 measured goal-present vs
  #   goal-absent ||q|| separability at AUC 0.930 on this encoder against 0.698
  #   on the P2 gain-5 code at ten distractors. The property explore depends on
  #   is BETTER here, not worse. Falsifier: cps at 10 distractors below cps at
  #   0, or chase_q materially above 0. p10_e_pol had cps10 ~ cps0 throughout
  #   and chase_q ~ 0.000.
  #
  # H2 -- LOG_KAPPA_MAX=2.5 HURTS explore, the OPPOSITE sign from P19. Coverage
  #   comes from persistent straight motion: p5_e measured straightness 0.945,
  #   and a billiard (straightness ~1) is the reactive ceiling. A von Mises at
  #   kappa has circular sd ~ 1/sqrt(kappa), so the cap floors per-step
  #   directional noise at 1/sqrt(12.2) = 0.286 rad = 16.4 deg, against the
  #   4.7 deg the p10_pol_v1 exploit arm learned far-field (9.8.1). Capping
  #   kappa is capping straightness. Falsifier: p20_e_kcap >= p20_e on
  #   strategy_efficiency. If the cap turns out to be neutral or good here too,
  #   then the e^5 default is simply wrong for this project and that is a
  #   larger finding than either arm.
  #
  # H3 -- converged by ~u250, as p10_e_pol was.
  #
  # REGIME_ASSIGNMENT and GOAL_REWARD are both provably inert on a pure explore
  # schedule -- n_pre_now is 0 or n_envs so both assignment branches agree
  # (train_navigate.py:366), and EXPLORE_GOALS_OFF=1 means no goal is stored.
  # They are set anyway, to p19_kcap's values, so the diff between the exploit
  # and explore arms is exactly the three lines it claims to be.
  p20_e|p20_e_kcap)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    case "$VARIANT" in
      p20_e_kcap) LOG_KAPPA_MAX=2.5 ;;
    esac
    ;;

  # === P27-P30 -- the retracing is PAID FOR. Four ways to stop paying. ======
  #
  # Jack: "it is totally going back over its own path a lot. i just don't get
  # how this could happen with the novelty reward". It happens because of the
  # OTHER reward. Priced from collector.py, early in an episode (scale ~1):
  #
  #   plough straight over ground already covered   0    + 0.20 = 0.20
  #   180 deg turn at the wall onto fresh ground    0.30 - 0.20 = 0.10
  #
  # Persistence is bonus*cos, so a reversal costs -bonus and the SWING across
  # a wall turn is 2*bonus = 0.40 -- more than the 0.30 a new cell pays. The
  # agent is not ignoring novelty; it is correctly following a reward in which
  # the lawnmower's defining move loses to retracing, two to one. Any turn past
  # ~120 deg loses. And revisit_penalty is 0, so nothing sits on the other side.
  #
  # Second defect, independent: novelty_scale_remaining multiplies novelty by
  # total/remaining up to 10x, and exists precisely to punish redundancy late.
  # At 200 steps and 0.77 cells/step the agent reaches ~155 of 400 cells, so
  # the scale never exceeds ~1.6. The mechanism never reaches the regime it
  # was written for.
  #
  # Four one-line arms off p20_e. Each is scored on swept coverage AND on the
  # state probe (PROBE=state), because coverage can move without the mechanism
  # changing -- both oracle arms (P25, P26) proved that.
  #
  #   p27_pers1s   persistence pays max(0, cos): keeps the reward for smooth
  #                motion, stops paying the agent not to turn around. The
  #                minimal fix -- the wall turn goes from 0.10 to 0.30 and
  #                beats retracing at every angle.
  #   p28_revisit  revisit_penalty 0.15. Puts something on the empty side of
  #                the ledger. 0.15 because the turn wins once rp > 0.10.
  #   p29_long     600-step rollouts, so novelty_scale_remaining actually
  #                engages. 3x the compute per update.
  #   p30_perslow  persistence 0.20 -> 0.05. Same knob as p27 but by level
  #                rather than by shape; separating the two says whether the
  #                turn PENALTY or the straightness REWARD is what binds.
  p27_pers1s|p28_revisit|p29_long|p30_perslow)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    case "$VARIANT" in
      p27_pers1s)  PERSISTENCE_ONE_SIDED=1 ;;
      p28_revisit) REVISIT_PENALTY=0.15 ;;
      p29_long)    STEPS_PER_ROLLOUT=600 ;;
      p30_perslow) PERSISTENCE_BONUS=0.05 ;;
    esac
    ;;

  # === P31-P33 -- stop the policy being a function of position and heading ==
  #
  # The goal, stated by Jack: the agent must not just be acting as a function
  # of position and heading. §30-§33 measured exactly that and it survived
  # every reward intervention:
  #
  #   * position occupies 2 of 1024 state directions and is read at ~7x a
  #     SIZE-MATCHED random subspace (§32.2) -- 2-3x any other content;
  #   * position's SHARE of the whole-state causal effect predicts orbit depth
  #     MONOTONICALLY across all five P20/P27-P30 arms (§33.3);
  #   * four reward-shaping arms moved neither the ceiling nor that share much,
  #     and the two that raised it (p29, p30) orbited hardest and covered least.
  #
  # Reward shaping is therefore the wrong lever. These attack the INPUT the
  # policy is leaning on. Training rollouts only -- evaluation always sees the
  # full input, because the question is whether a policy TRAINED under an
  # unreliable position signal learns to weight it less, not whether it can
  # cope with a handicap at test time.
  #
  #   p31_placedrop  place code zeroed on 30% of steps, per env. Position is
  #                  intermittently unavailable, so a policy that is only a
  #                  position field cannot collect reward on those steps.
  #   p32_headdrop   prev_action AND prev_displacement zeroed together on 30%.
  #                  Dropped as a pair: either alone still carries the
  #                  direction of travel. The other half of the goal.
  #   p33_revisit_hi revisit_penalty 0.15 -> 0.40. p28 was the ONLY arm that
  #                  moved the balance the productive way -- lowest position
  #                  share (0.224 vs the control's 0.275), highest occupancy
  #                  share (0.170 vs 0.128), and the only arm with no orbit --
  #                  while leaving coverage alone. If that relationship is
  #                  causal and dose-dependent, more of it should push further.
  #
  # Scored on swept coverage, the PROXIMITY revisit measure (revisit_frac is
  # exactly 1 - cells_per_step and so is coverage restated), recurrence, and
  # the state probe. The state probe is the one that answers the goal: whether
  # position's share of the action fell.
  p31_placedrop|p32_headdrop|p33_revisit_hi|p33_revisit_mid|p34_dropaux)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    case "$VARIANT" in
      p31_placedrop)  PLACE_DROPOUT=0.3 ;;
      p32_headdrop)   HEADING_DROPOUT=0.3 ;;
      p33_revisit_hi) REVISIT_PENALTY=0.40 ;;
      # 0.40 COLLAPSED: swept 0.0924 identical at every eval u25-u200, mean_r
      # pinned at -0.31. The penalty EXCEEDS the novelty reward of 0.30, so
      # during the early wall-pin phase -- where nearly every step lands on
      # covered ground -- reward is uniformly negative and there is no escape
      # gradient. The pin becomes absorbing. 0.15 worked because it stayed
      # below novelty; 0.25 is the largest dose that still does.
      p33_revisit_mid) REVISIT_PENALTY=0.25 ;;
      # 0.25 STALLED TOO: 0.0976 at u25 to 0.1030 at u200, +0.005 over 175
      # updates, while p28 at 0.15 was at 0.462 by u200. Predicted by the
      # break-even: positive reward needs coverage rate > rp/(0.3+rp), so
      # 0.15 -> 0.33 and 0.25 -> 0.45, against a pinned start of ~0.10. Every
      # increment raises the bar the agent must clear BEFORE reward turns
      # positive while making the pin more punishing. Escalating this knob is
      # a dead end; the usable band is 0 < rp <~ 0.15.
      #
      # p34_dropaux takes the freed slot and is the better-motivated arm:
      # dropout alone takes position AWAY without giving the policy anything
      # to use instead. p24_aux is the one intervention that demonstrably put
      # more visitation INTO the state (occupancy delta_flow 0.136 vs the
      # control's 0.035, 4x, §30.11) and on its own changed no behaviour.
      # Together they are the two halves of the goal: make position
      # unreliable, and ensure there is a map to fall back on.
      p34_dropaux)
        PLACE_DROPOUT=0.3
        AUX_VISITED_WEIGHT=0.5
        AUX_VISITED_RADIUS=3.0
        ;;
    esac
    ;;

  # === P35-P37 -- WAVE 2. Stop adjusting inputs; change the problem. =========
  #
  # Wave 1 (§35) was a behavioural null and taught two things:
  #
  #   * every lever so far moves what the state CONTAINS and never what the
  #     policy DOES with it -- occupancy's absolute influence on the action sat
  #     at 0.024-0.030 across the control, p24_aux, and all three wave-1 arms;
  #   * p34_dropaux HALVED position's absolute grip (0.031 vs 0.057) and
  #     changed no behaviour at all, so position dominance is not the binding
  #     constraint either.
  #
  # Jack, correcting a claim in the first draft of this: "the models retrace
  # their steps because they don't have history, which obviously hugely harms
  # novelty." Right, and it reframes the puzzle. History is NOT useless here --
  # the policy passes within a cell of its own track on ~33% of steps and every
  # one costs novelty. So memory would pay, the information is in the state,
  # and the policy still will not use it. That is a CREDIT problem, not an
  # information problem.
  #
  #   p35_alias      alias_mod=10: the four quadrants of the 20x20 arena emit
  #                  IDENTICAL place codes. What is true about position
  #                  sufficiency is narrow -- a lawnmower is a function of
  #                  position alone (row parity off the y-coordinate) and would
  #                  score ~0.9 (§29.3), so a near-optimal MEMORYLESS policy
  #                  exists and that is the basin PPO settles into. Aliasing
  #                  removes it: no memoryless policy can be optimal. Applies
  #                  at training AND evaluation -- a sensor property, not a
  #                  perturbation.
  #   p36_rpanneal   revisit_penalty 0.40 ramped from 0 over 400 updates. The
  #                  penalty is the only signal that makes the value of memory
  #                  IMMEDIATE rather than diffuse, and p28 at 0.15 was the
  #                  only arm ever to kill the orbit. Constant doses fail for a
  #                  reason §34.3 derived: positive reward needs coverage rate
  #                  > rp/(0.3+rp) and the agent starts pinned at ~0.10, so the
  #                  penalty raises the bar before reward turns positive. The
  #                  ramp separates the jobs -- escape the pin, then apply the
  #                  pressure. 400 updates because at 300 the effective penalty
  #                  hits the pinned break-even by u25, and the control needs
  #                  ~u75 to break the pin.
  #   p37_aliasaux   alias_mod=10 plus the aux head. If aliasing makes memory
  #                  necessary, this also makes a map available to be the
  #                  memory -- the pairing that made p34 the only arm to move
  #                  position's grip.
  p35_alias|p36_rpanneal|p37_aliasaux)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    case "$VARIANT" in
      p35_alias)    ALIAS_MOD=10 ;;
      p36_rpanneal) REVISIT_PENALTY=0.40; REVISIT_ANNEAL_UPDATES=400 ;;
      p37_aliasaux) ALIAS_MOD=10
                    AUX_VISITED_WEIGHT=0.5
                    AUX_VISITED_RADIUS=3.0 ;;
    esac
    ;;

  # === P21 -- does the pin clear when persistence stops paying for it? ======
  #
  # 18.7 measured that 100% of episodes at u25 AND u50 are wall-pinned, and
  # 18.8 priced it: at the pin the persistence bonus PAYS +0.196/step while
  # wall_penalty charges only -0.093, a ratio of 2.1. The bonus scores
  # cos(a_t, a_t-1) on the COMMANDED action, and a pinned agent commands a
  # rock-steady heading (straightness 0.981, the highest number in the
  # document) while realizing 0.09 of it. It is paid the full ballistic bonus
  # for standing still.
  #
  # p21_pr is p20_e with ONE bit flipped: --persistence_realized. Nothing else
  # moves, so p20_e's own eval series is the control and does not need
  # re-running.
  #
  # explore:300 rather than 700 because the claim is entirely about the early
  # curve. p20_e was 100% pinned at u50, 31% at u75, 0% at u150; 300 updates
  # with EVAL_EVERY=25 gives 12 points across and past that window.
  #
  # PASS: the pinned fraction at u25/u50 is materially below 100%, and the
  # coverage curve leaves the 0.05 floor earlier than p20_e's u75. Score it
  # with `analysis.nav_tri.explore_traj` on the u25/u50/u75 checkpoints, the
  # same way 18.7 was measured -- NOT from the coverage curve alone, which
  # cannot tell "unpinned" from "pinned but lucky".
  #
  # Prediction on record: the pin clears earlier, and final coverage is
  # UNCHANGED or slightly better. The realized and commanded cosines are equal
  # whenever neither the norm clamp nor the arena clip bites, which for the
  # converged p20_e policy is 97% of steps (clip_frac 0.031) -- so this should
  # be nearly a no-op late and matter only where the agent is stuck.
  #
  # Falsifier worth stating: if the pin clears but coverage ends LOWER, the
  # bonus was doing something at the walls that this removes, and the right
  # answer is a smaller persistence_bonus on realized rather than the swap.
  p21_pr)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:300'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    PERSISTENCE_REALIZED=1
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # === P22 -- is the sensory input helping or harming explore? =============
  #
  # 60 of the policy's 74 input dims (81%) are the wall raycast. 6.9 concluded
  # it supplies nothing explore needs -- "the lawnmower ceiling is not blocked,
  # because 4's B3 already hands the agent exact self-motion... if P5 plateaus
  # at billiard, the diagnosis is recurrent capacity or reward shape, not the
  # sensor". P5 DID plateau at billiard. So the sensor's contribution to
  # explore has never been measured directly, only argued about.
  #
  # Reasons it might still help, none of which 6 tested:
  #   - 6's numbers are all CROSS-ENV. Explore trains on 20 FIXED envs, where
  #     the +-1 code is injective by design (wall_resolution=4 exists so two
  #     positions in one cell read differently).
  #   - integrating prev_displacement is arithmetically exact, but a LEARNED
  #     integrator drifts over 200 steps; a signature re-anchors it.
  #   - "have I been here" is a matching operation against your own history --
  #     env-general, needing only within-episode discriminability.
  # Reason it might not: the cone is heading-coupled (psi = atan2 of realized
  # displacement), so it is a (position, heading) signature, not a place code.
  #
  # p22_nos is p20_e with INPUT_SENSORY=0 and nothing else moved, so p20_e is
  # the control and needs no re-run. 74 input dims -> 14.
  #
  # explore:700, matching p20_e point for point rather than the shorter
  # schedule the early curve would allow -- the question is about final
  # coverage, not about time-to-breakout.
  #
  # SCORE ON swept_coverage (19), which is now logged. Not cells_per_step.
  #
  # Prediction on record: NO significant loss, per 6.9. If that holds, 81% of
  # the input is dead weight the policy spends capacity gating off, and the
  # interleaved model should probably drop it too. If coverage DROPS, the
  # sensor is doing something 6 did not measure -- most likely within-episode
  # place recognition -- and that matters more for interleaving than for
  # explore, because regime inference has to key on something.
  #
  # Read a null carefully: sensor and memory are complements for recognition,
  # and this regime cannot write to memory (allows_store=False). A null here
  # does not license "the sensor is useless" in a regime that can store.
  p22_nos)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    INPUT_SENSORY=0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # === P23 -- lever A: anneal the kappa ceiling off (2 doc 24.2) ===========
  #
  # The interleaved model deploys DETERMINISTICALLY -- exploit's 1.013 beeline
  # (17.10) is a deterministic number -- so explore has to work deterministic
  # too. And the kappa cap is what exploit NEEDS (17.9: 0.375@u475 ->
  # 1.000@u125). But the cap is what makes explore's MEAN policy a constant
  # curl (18.6), which deterministic eval then exposes:
  #
  #   p20_e_kcap swept   step 200: det 0.538  sampled 0.623
  #                      step 999: det 0.748  sampled 0.935
  #   p20_e              step 999: det 0.911  sampled 0.939
  #
  # The capped policy CAN reach 0.935 -- with noise doing the work. Training is
  # sampled, so it collects that version and never experiences the trap.
  #
  # KEY FACT (20.1, measured): kappa does not affect a deterministic action AT
  # ALL -- a 4x change moved every behavioural statistic by <0.001. The cap is
  # purely a TRAINING-TIME device. So ramp it: on early where exploit needs it,
  # off late so the mean policy is optimized nearer the regime it is deployed
  # in.
  #
  # p23_kanneal is p20_e_kcap with the ramp and nothing else moved, so
  # p20_e_kcap is the control and needs no re-run. 2.5 -> 5.0 over 400 of 700
  # updates: the cap is fully off for the last 300, and 400 is past the u125
  # where the exploit lineage converged, so it should not cost the unlock.
  #
  # SCORE DETERMINISTIC. That is the deployed regime and the entire point.
  # Compare against p20_e_kcap's 0.538@200 / 0.748@1000 and p20_e's 0.644 /
  # 0.911.
  #
  # Prediction on record: deterministic explore recovers toward p20_e. This is
  # lever A and it is NOT Tier-2 -- it should produce a better FIELD, not a
  # policy that uses memory. 22's replay signature is expected to SURVIVE. If
  # coverage recovers and replay survives, A worked and B is still needed.
  p23_kanneal)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    LOG_KAPPA_MAX=2.5
    LOG_KAPPA_MAX_END=5.0
    LOG_KAPPA_ANNEAL_UPDATES=400
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # === P24 -- lever B: force Tier-2 with an auxiliary visitation head =======
  #
  # 22 measured that the explore policy REPLAYS on a state repeat -- it is a
  # fixed (position, heading) vector field and does not consult where it has
  # been. 24 priced what that costs: at the 200-step operational horizon p20_e
  # gets swept 0.644 and SAMPLING ADDS NOTHING (0.643), against ~0.9 for a
  # perfect lawnmower. Only memory recovers that; noise cannot.
  #
  # AUX_VISITED_WEIGHT puts a BCE head on the trunk predicting which of 8
  # compass cells at radius 3 the agent has already visited. Training-time
  # oracle only -- no change to the observation, the reward, or deployment.
  #
  # p24_aux is p20_e plus the head, nothing else moved, so p20_e is the control
  # and needs no re-run. Weight 0.5 is a starting guess: large enough to shape
  # the trunk against move_loss ~1e-2 and value_loss ~1, small enough that a
  # BCE near 1.2 does not dominate. If the head learns (aux_visited_loss falls)
  # but nothing else changes, the weight is the next thing to move.
  #
  # THE SUCCESS TEST IS NOT COVERAGE. Coverage could rise from a better field
  # alone, which is lever A's job, not this one. Tier-2 is achieved iff 22's
  # REPLAY SIGNATURE BREAKS: same position, same heading, DIFFERENT action,
  # because the hidden state now carries where the agent has been. Score with
  # the replay probe on the final checkpoint, then coverage second.
  #
  # Prediction on record: aux_visited_loss falls well below its ~1.18 start
  # (the task is learnable from features -- pinned by test). Whether that
  # transfers to the POLICY is the open question, and the honest prior is
  # uncertain: a representation being present does not mean the policy head
  # uses it, which is exactly what 7.7.2 said about chart_frac and never got
  # to test.
  p24_aux)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    AUX_VISITED_WEIGHT=0.5
    AUX_VISITED_RADIUS=3.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # === P25 -- DIAGNOSTIC: hand visitation to the policy directly (27.5) =====
  #
  # NOT A SHIPPABLE CONFIG. INPUT_VISITED is an ORACLE at test time. This arm
  # exists to split one question and should never become a baseline.
  #
  # 27: the auxiliary head LEARNED to predict 8-direction visitation from the
  # trunk's features (aux_visited_loss 0.632 -> 0.367), so those features
  # provably carry visitation -- and the policy head, reading the identical
  # vector, ignored it. 22's replay signature was unchanged (ratio 0.115 vs the
  # control's 0.125) and coverage FELL 13.7%.
  #
  # So the question is no longer "is the information available". It is:
  #
  #   does the policy fail to USE visitation, or fail to EXTRACT it?
  #
  # Handing the same 8-vector in as an INPUT collapses "use memory" from "learn
  # to read your own hidden state" down to "learn to weight an input".
  #
  #   coverage IMPROVES  -> the bottleneck is representation-to-policy. The fix
  #                         is architectural (how the policy reads state), not
  #                         more pressure, and B2/B3 are the wrong next levers.
  #   coverage FLAT      -> the policy cannot exploit visitation even when
  #                         handed it. Much deeper: Tier-2 is not reachable
  #                         this route at all, and the field is a local optimum
  #                         that better inputs do not escape.
  #
  # p25_visin is p20_e plus the channel, nothing else moved, so p20_e is the
  # control. Input widens 74 -> 82 dims.
  #
  # Score BOTH: swept_coverage (does it help at all) AND the replay probe (does
  # it stop replaying). Those can come apart -- a policy could use the input to
  # improve coverage while still being a fixed function of an ENLARGED state,
  # which would still not be Tier-2. Worth knowing either way.
  p25_visin)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    INPUT_VISITED=1
    AUX_VISITED_RADIUS=3.0
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # === P26 -- DIAGNOSTIC: does the agent just not know WHERE IT IS? (29.4) ==
  #
  # NOT A SHIPPABLE CONFIG. INPUT_ABS_POSITION is an oracle at test time.
  #
  # The behaviour we are missing is a boustrophedon: east on even rows, west on
  # odd, step north at the wall. That is MEMORYLESS -- a function of position
  # alone -- so 22's replay finding cannot be what caps coverage at 0.644
  # against a lawnmower's ~0.9. But a boustrophedon is position-DEPENDENT: it
  # has to know which ROW it is in. The agent has exact relative self-motion
  # (prev_displacement, integration error 2.3e-14) and NO ANCHOR. It knows how
  # far it has moved, never where it is.
  #
  # 29 ruled out the memory story from the other side: handing the policy local
  # visitation (8 bits at radius 3) bought SPEED -- converged at u300 not u700,
  # steadier, no wall-pin phase -- and the SAME ceiling, 0.629 vs 0.625-0.644.
  #
  # So two candidates remain and this separates them:
  #
  #   coverage jumps toward ~0.9  -> LOCALIZATION was the blocker. The policy
  #                                  could run a sweeping field and never knew
  #                                  where it was.
  #   coverage flat at ~0.64      -> OPTIMIZATION. The information was
  #                                  sufficient and PPO cannot reach the
  #                                  lawnmower basin from this initialization.
  #
  # This is a FAIRER oracle than 27.5's. Absolute position is only 2 dims but
  # it is SUFFICIENT for the target behaviour, so a null here is informative in
  # a way 29's null was not (8 bits on one ring could not express global
  # structure at all). And it is derivable in principle: the wall code is
  # position-specific -- that is what wall_resolution=4 exists for -- and 25
  # measured that removing it costs 20%. So this asks "if localization were
  # solved, would it help", without solving it.
  #
  # p26_abspos is p20_e plus the channel, nothing else moved. Input 74 -> 76.
  #
  # Score swept_coverage against p20_e's 0.625-0.644 at 200 steps and 0.911 at
  # 1000, and run the recurrence curve (28) to confirm it is not just trading
  # coverage for a new orbit.
  p26_abspos)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    SCHEDULE=${SCHEDULE:-'explore:700'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    INPUT_ABS_POSITION=1
    EVAL_SCOPE=expl; EVAL_EVERY=25; CKPT_EVERY=25
    ;;

  # === D0/D1 -- WAVE 1 of docs/DUAL_TRAINING.md. The interleaved run. =======
  #
  # P6 was specced in §10 and never ran, so phase 2 has TWO specialists at
  # their ceilings and nothing that does both. These arms are that run, plus
  # the three knobs DUAL_TRAINING §4.3 identifies as having CONFLICTING optima
  # between the regimes -- the first list of contradictions this project has
  # had, and the reason interleaving is not just a schedule question.
  #
  #   d0_base     the P6 baseline. Interleave in the SAME PPO update (phase-1
  #               finding 11: explore-first collapses to 0.068, exploit-first
  #               to 0.062, and `blocked` behaves like explore-first). It is
  #               the control for the other three, so it must run.
  #   d1_kanneal  LOG_KAPPA_MAX 2.5 -> 5.0 over 400. **The highest-value open
  #               test in the project.** Exploit NEEDS the cap (§17.9: 1.000
  #               @u125 against 0.375@u475); explore needs it lifted (§23/§24:
  #               the cap traps the mean policy in a closed orbit, a 20%
  #               det/sampled gap). §26 measured the anneal explore-safe --
  #               gap 20% -> 3.3% -- and §26.3 says outright it tested nothing
  #               about whether exploit still converges under it. This is D6.
  #   d1_persr    --persistence_realized. §18.8 priced the wall pin: the
  #               persistence bonus PAYS +0.196/step for it against
  #               wall_penalty's -0.093, because it scores the COMMANDED
  #               action and a pinned agent commands a perfect heading while
  #               realizing 0.09. Gate 1 says nothing downstream is
  #               interpretable until the pin clears. This is D7.
  #   d1_ms3      the multistep control -- see the block below.
  #
  # MULTISTEP. Jack, 2026-09-06: "i don't see multi step hopfield mentioned
  # and i think you should drop that." The evidence was already in and agrees:
  # §7.7 ablated it and depth {1} is NEVER worse than {1,2,3} for a regime
  # classifier (0.869 vs 0.858 at ten distractors t=8, 0.888 vs 0.884 at
  # t=64), and §5.4 explains why -- the recall does not converge, it drifts,
  # so depths 2 and 3 are strictly DEGRADED states sampling the transient of a
  # power iteration away from the answer. It also removes the four-channel
  # implementation trap in EXPLOIT_DIAGNOSTIC §7, where `q` reaches the policy
  # through the raw signal plus three multistep channels and intervening on
  # one leaves the other three contradicting it.
  #
  # So the default is now "1" (changed above) and `d1_ms3` keeps "1 2 3".
  # The drop is therefore MEASURED against d0_base rather than assumed -- the
  # honest qualification in §7.7 is that it ablated four SUMMARY statistics of
  # the depth channel, while the policy receives q^2 and q^3 as raw 2-D
  # vectors and could in principle use them some other way.
  #
  # Speed is left at p20_e's [0.5, 1.0] so the explore interference number is
  # comparable to §22.3.1's 0.644. The [0.5, 2.0] arm is deliberately NOT in
  # this wave: §12 showed step count tracks the speed CAP rather than
  # navigation quality, so it would move mean_steps for a reason that is not
  # about interleaving.
  d0_base|d1_kanneal|d1_persr|d1_ms3)
    ENCODER=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
    ENCODER_GAIN=100
    HOPFIELD_BETA=100
    # empty_frac=0.5 is phase 1's measured optimum (tri finding 13: 0.5->0.7
    # moves ALONG a coverage/steps frontier, +28% coverage for -41% steps,
    # never outward). 1200 updates fits the 6 h wall at ~12.6 s/update with
    # margin; CKPT_EVERY=25 so a TIMEOUT still leaves a usable series.
    SCHEDULE=${SCHEDULE:-'interleave:1200,empty_frac=0.5'}
    ENVS_PER_WORLD=20; BATCH_ENVS=64
    EPSILON_EXPLORE=0.1; GOAL_REWARD=2.0
    PERSISTENCE_BONUS=0.20
    # Non-negotiable: positional assignment let the policy gate on env
    # IDENTITY instead of on the recall signal (D3), which is the one way to
    # score well on both metrics without solving the problem.
    REGIME_ASSIGNMENT=shuffle
    ACTION_POLAR=1; STATE_DEPENDENT_STD=1; FREEZE_LOG_STD=0
    MIN_ACTION_NORM=0.5; MAX_ACTION_NORM=1.0
    # Exploit's unlock, kept for the base. d1_kanneal is the arm that asks
    # whether it can be released later without losing it.
    LOG_KAPPA_MAX=2.5
    # navexpl, not expl: an interleaved run that scores only one half cannot
    # see interference, which is the whole quantity of interest.
    EVAL_SCOPE=navexpl; EVAL_EVERY=25; CKPT_EVERY=25
    case "$VARIANT" in
      d1_kanneal) LOG_KAPPA_MAX_END=5.0; LOG_KAPPA_ANNEAL_UPDATES=400 ;;
      d1_persr)   PERSISTENCE_REALIZED=1 ;;
      d1_ms3)     INPUT_HOPFIELD_MULTISTEP="1 2 3" ;;
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
echo "    encoder    : $(basename "$(dirname "$ENCODER")")/$(basename "$ENCODER") \
gain=${ENCODER_GAIN:-<ckpt>} hopfield_beta=${HOPFIELD_BETA:-<encoder gain>}"
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
