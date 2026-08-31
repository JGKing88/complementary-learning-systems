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
