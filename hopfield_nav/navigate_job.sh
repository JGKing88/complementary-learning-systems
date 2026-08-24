# Shared body of run_explore.sh / run_exploit.sh / run_navigate.sh.
#
# Sourced, not executed, and only after the caller has `cd`-ed to the repo root
# -- SLURM copies the batch script to a node-local spool directory, so
# $BASH_SOURCE points somewhere useless and a relative path from the repo root
# is the only thing that resolves. Same reason scripts/cls_env.sh says so.
#
# This file is a pass-through, not a policy. Every one of `train_navigate`'s
# flags has an environment variable here, named after it in upper case, and
# **an unset variable is not passed at all**. That rule is what makes the two
# cases work with one table:
#
#   fresh run  -- an unpassed flag falls back to the trainer's own argparse
#                 default, which is where those defaults belong.
#   LOAD_CKPT  -- an unpassed flag is inherited from the parent checkpoint,
#                 because `--load_checkpoint` takes the parent's config as its
#                 base and lets only the flags actually on the command line
#                 override it. A launcher that spelled out a value it did not
#                 mean would silently overwrite the parent.
#
# So a launcher sets only what it wants to say. run_navigate.sh sets everything
# (guarded on LOAD_CKPT); run_explore.sh and run_exploit.sh set a handful.
#
# Only three are always passed: SCHEDULE (required), ENCODER (the trainer
# requires it) and DEVICE.
#
# Booleans take 1/0 (also true/false, yes/no, on/off) and become --flag or
# --no-flag. Lists take a space-separated string: LAMBDAS="11 12 13".

set -euo pipefail

if [ -z "${SCHEDULE:-}" ]; then
    echo "ERROR: SCHEDULE is required, e.g. SCHEDULE='explore:600'" >&2
    exit 1
fi

ENCODER=${ENCODER:-encoders/run_20260422_185816/encoder_best.pt}
DEVICE=${DEVICE:-cuda}

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
# wandb auth comes from ~/.netrc (machine api.wandb.ai). Run `wandb login`
# once if it is missing; never paste an API key into a tracked script.
unset CUDA_VISIBLE_DEVICES

source scripts/cls_env.sh

ARGS=(
    --encoder_checkpoint "$ENCODER"
    --schedule "$SCHEDULE"
    --device "$DEVICE"
)

# Each helper ends in `return 0` on purpose: `[ -n "" ] && ...` as the last
# command would make the function return non-zero, and under `set -e` that
# exits the job. An unset knob is the normal case, not a failure.
_arg()  { [ -n "${2:-}" ] && ARGS+=("--$1" "$2"); return 0; }
_list() { [ -n "${2:-}" ] && { ARGS+=("--$1"); for _v in $2; do ARGS+=("$_v"); done; }; return 0; }
_bool() {
    case "${2:-}" in
        1|true|True|yes|on)   ARGS+=("--$1") ;;
        0|false|False|no|off) ARGS+=("--no-$1") ;;
    esac
    return 0
}

# --- Encoder / scaffold ----------------------------------------------------
_arg  encoder_gain                    "${ENCODER_GAIN:-}"
_arg  fwhm_ratio                      "${FWHM_RATIO:-}"
_list lambdas                         "${LAMBDAS:-}"
_arg  Np                              "${NP:-}"
_bool static-vectorhash               "${STATIC_VECTORHASH:-}"

# --- Environment -----------------------------------------------------------
_arg  size                            "${SIZE:-}"
_arg  observation_size                "${OBSERVATION_SIZE:-}"
_arg  movement_mode                   "${MOVEMENT_MODE:-}"
_arg  goal_reward                     "${GOAL_REWARD:-}"
_arg  goal_radius                     "${GOAL_RADIUS:-}"
_arg  time_penalty                    "${TIME_PENALTY:-}"
_bool continuous_normalize            "${CONTINUOUS_NORMALIZE:-}"
_arg  max_action_norm                 "${MAX_ACTION_NORM:-}"
_arg  min_action_norm                 "${MIN_ACTION_NORM:-}"
_bool allow_offcell_store             "${ALLOW_OFFCELL_STORE:-}"
_arg  wall_resolution                 "${WALL_RESOLUTION:-}"
_bool egocentric_heading              "${EGOCENTRIC_HEADING:-}"
_bool reset_state_on_teleport         "${RESET_STATE_ON_TELEPORT:-}"

# --- Agent -----------------------------------------------------------------
_arg  hopfield_mode                   "${HOPFIELD_MODE:-}"
_arg  hidden_size                     "${HIDDEN_SIZE:-}"
_arg  num_rnn_layers                  "${NUM_RNN_LAYERS:-}"
_arg  init_log_std                    "${INIT_LOG_STD:-}"
_bool freeze_log_std                  "${FREEZE_LOG_STD:-}"
_bool input_prev_reward               "${INPUT_PREV_REWARD:-}"
_bool input_prev_action               "${INPUT_PREV_ACTION:-}"
_bool input_prev_displacement         "${INPUT_PREV_DISPLACEMENT:-}"
_bool input_hopfield_raw              "${INPUT_HOPFIELD_RAW:-}"
_bool input_hopfield_signal           "${INPUT_HOPFIELD_SIGNAL:-}"
_bool input_sensory                   "${INPUT_SENSORY:-}"
_bool input_encoded_state             "${INPUT_ENCODED_STATE:-}"
_bool input_goal_in_memory            "${INPUT_GOAL_IN_MEMORY:-}"
_list input_hopfield_multistep        "${INPUT_HOPFIELD_MULTISTEP:-}"
_arg  rnn_cell                        "${RNN_CELL:-}"
_arg  rnn_nonlinearity                "${RNN_NONLINEARITY:-}"

# --- Optimization ----------------------------------------------------------
_arg  lr                              "${LR:-}"
_arg  move_ent_coef                   "${MOVE_ENT_COEF:-}"
_arg  ppo_clip_coef                   "${PPO_CLIP_COEF:-}"

# --- Reward shaping --------------------------------------------------------
_arg  novelty_reward                  "${NOVELTY_REWARD:-}"
_bool novelty_anneal                  "${NOVELTY_ANNEAL:-}"
_bool novelty_scale_remaining         "${NOVELTY_SCALE_REMAINING:-}"
_arg  novelty_scale_cap               "${NOVELTY_SCALE_CAP:-}"
_arg  revisit_penalty                 "${REVISIT_PENALTY:-}"
_arg  wall_penalty                    "${WALL_PENALTY:-}"
_arg  persistence_bonus               "${PERSISTENCE_BONUS:-}"

# --- Explore-regime behavior -----------------------------------------------
_arg  regime_assignment               "${REGIME_ASSIGNMENT:-}"
_bool explore_goals_off               "${EXPLORE_GOALS_OFF:-}"
_bool explore_ends_on_goal            "${EXPLORE_ENDS_ON_GOAL:-}"
_bool randomize_goal_per_rollout      "${RANDOMIZE_GOAL_PER_ROLLOUT:-}"
_arg  epsilon_explore                 "${EPSILON_EXPLORE:-}"
_arg  epsilon_anneal_updates          "${EPSILON_ANNEAL_UPDATES:-}"

# --- Distractors -----------------------------------------------------------
_arg  n_train_distractors_min         "${N_TRAIN_DISTRACTORS_MIN:-}"
_arg  n_train_distractors_max         "${N_TRAIN_DISTRACTORS_MAX:-}"
_arg  n_train_emp_distractors_min     "${N_TRAIN_EMP_DISTRACTORS_MIN:-}"
_arg  n_train_emp_distractors_max     "${N_TRAIN_EMP_DISTRACTORS_MAX:-}"
_arg  n_train_distractors_max_end     "${N_TRAIN_DISTRACTORS_MAX_END:-}"
_arg  n_train_emp_distractors_max_end "${N_TRAIN_EMP_DISTRACTORS_MAX_END:-}"
_arg  distractor_curriculum_updates   "${DISTRACTOR_CURRICULUM_UPDATES:-}"

# --- log-sigma anneal ------------------------------------------------------
_arg  log_std_anneal_start_update     "${LOG_STD_ANNEAL_START_UPDATE:-}"
_arg  log_std_anneal_end_update       "${LOG_STD_ANNEAL_END_UPDATE:-}"
_arg  log_std_anneal_target           "${LOG_STD_ANNEAL_TARGET:-}"

# --- Rollout shape ---------------------------------------------------------
_arg  batch_envs                      "${BATCH_ENVS:-}"
_arg  steps_per_rollout               "${STEPS_PER_ROLLOUT:-}"
_arg  num_worlds                      "${NUM_WORLDS:-}"
_arg  envs_per_world                  "${ENVS_PER_WORLD:-}"
_arg  seed                            "${SEED:-}"

# --- Eval ------------------------------------------------------------------
_arg  num_val_envs                    "${NUM_VAL_ENVS:-}"
_arg  n_val_trials                    "${N_VAL_TRIALS:-}"
_list val_distractors                 "${VAL_DISTRACTORS:-}"
_arg  eval_every                      "${EVAL_EVERY:-}"
_arg  eval_scope                      "${EVAL_SCOPE:-}"
_arg  eval_max_steps                  "${EVAL_MAX_STEPS:-}"

# --- Artifacts / logging ---------------------------------------------------
_arg  ckpt_every                      "${CKPT_EVERY:-}"
_arg  save_dir                        "${SAVE_DIR:-}"
_arg  load_checkpoint                 "${LOAD_CKPT:-}"
_arg  wandb_project                   "${WANDB_PROJECT:-}"
[ "${USE_WANDB:-1}" != "0" ] && ARGS+=(--use_wandb)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM output log: /home/jackking/cls/hopfield_nav/logs/slurm_${JOB_LABEL:-navigate}_${SLURM_JOB_ID}.out"
fi
echo "=== ${JOB_LABEL:-navigate}  schedule='$SCHEDULE' ==="
if [ -n "${LOAD_CKPT:-}" ]; then
    echo "    resuming from $LOAD_CKPT"
    echo "    (config inherited from it; only the flags set above override)"
else
    echo "    fresh init, encoder=$ENCODER"
fi

# EXTRA is unquoted on purpose: it is a flag string, not one argument.
python -u -m hopfield_nav.train_navigate "${ARGS[@]}" ${EXTRA:-}
