# Shared body of run_explore.sh / run_exploit.sh / run_navigate.sh.
#
# Sourced, not executed, and only after the caller has `cd`-ed to the repo root
# -- SLURM copies the batch script to a node-local spool directory, so
# $BASH_SOURCE points somewhere useless and a relative path from the repo root
# is the only thing that resolves. Same reason scripts/cls_env.sh says so.
#
# The three launchers differ only in their #SBATCH header and their default
# SCHEDULE. Everything they pass to the trainer lives here, so a flag that needs
# changing changes in one place.
#
# LOAD_CKPT changes what this script passes, not just what it adds. Under
# `--load_checkpoint` the trainer takes the parent's config as its base and lets
# only the flags actually on the command line override it -- so a launcher that
# always spelled out the architecture would overwrite the parent with its own
# defaults and defeat the inheritance. On a resume, therefore, every knob below
# stays unset unless you set it yourself, and the architecture block is not
# passed at all.
#
# The caller must set SCHEDULE. Everything else is overridable from the
# environment; the defaults in parentheses apply to a *fresh* run only.
#
#   SCHEDULE           stage list, e.g. 'explore:200 ; exploit:100'   (required)
#   LOAD_CKPT          .pt to start from; empty = fresh random init   (empty)
#   ENCODER            stage-1 encoder checkpoint                     (run_20260422_185816)
#                      Always passed -- the trainer requires it.
#   SEED               random seed                                    (42)
#   LR                 Adam learning rate                             (3e-4)
#   NOVELTY_REWARD     reward per first-visit cell, explore regime    (0.3)
#   SIZE               grid size                                      (8)
#   ENVS_PER_WORLD     envs the schedule's fraction splits            (20)
#   STEPS_PER_ROLLOUT  steps per rollout                              (400)
#   BATCH_ENVS         parallel trajectories per env                  (16)
#   EVAL_EVERY         eval cadence, in updates                       (50)
#   CKPT_EVERY         checkpoint cadence; empty = follow EVAL_EVERY  (empty)
#   SAVE_DIR           output dir; empty = $CLS_RUNS/agent_ckpts/navigate_<run> (empty)
#   WANDB_PROJECT      wandb project                                  (hopfield-nav-navigate)
#   USE_WANDB          set to 0 to disable wandb                      (1)
#   DEVICE             cuda | cpu                                     (cuda)
#   EXTRA              extra flags, appended last so they win         (empty)

set -euo pipefail

if [ -z "${SCHEDULE:-}" ]; then
    echo "ERROR: SCHEDULE is required, e.g. SCHEDULE='explore:600'" >&2
    exit 1
fi

ENCODER=${ENCODER:-encoders/run_20260422_185816/encoder_best.pt}
DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-hopfield-nav-navigate}

RESUMING=0
[ -n "${LOAD_CKPT:-}" ] && RESUMING=1

if [ "$RESUMING" = "0" ]; then
    # Nothing to inherit, so the launcher supplies the whole recipe.
    SEED=${SEED:-42}
    LR=${LR:-3e-4}
    NOVELTY_REWARD=${NOVELTY_REWARD:-0.3}
    SIZE=${SIZE:-8}
    ENVS_PER_WORLD=${ENVS_PER_WORLD:-20}
    STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-400}
    BATCH_ENVS=${BATCH_ENVS:-16}
    EVAL_EVERY=${EVAL_EVERY:-50}
fi

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
[ -n "${SEED:-}" ]              && ARGS+=(--seed "$SEED")
[ -n "${LR:-}" ]                && ARGS+=(--lr "$LR")
[ -n "${NOVELTY_REWARD:-}" ]    && ARGS+=(--novelty_reward "$NOVELTY_REWARD")
[ -n "${SIZE:-}" ]              && ARGS+=(--size "$SIZE")
[ -n "${ENVS_PER_WORLD:-}" ]    && ARGS+=(--envs_per_world "$ENVS_PER_WORLD")
[ -n "${STEPS_PER_ROLLOUT:-}" ] && ARGS+=(--steps_per_rollout "$STEPS_PER_ROLLOUT")
[ -n "${BATCH_ENVS:-}" ]        && ARGS+=(--batch_envs "$BATCH_ENVS")
[ -n "${EVAL_EVERY:-}" ]        && ARGS+=(--eval_every "$EVAL_EVERY")
[ -n "${CKPT_EVERY:-}" ]        && ARGS+=(--ckpt_every "$CKPT_EVERY")
[ -n "${SAVE_DIR:-}" ]          && ARGS+=(--save_dir "$SAVE_DIR")
[ "$USE_WANDB" != "0" ]         && ARGS+=(--use_wandb --wandb_project "$WANDB_PROJECT")

if [ "$RESUMING" = "1" ]; then
    ARGS+=(--load_checkpoint "$LOAD_CKPT")
else
    ARGS+=(
        # Architecture. These are the settings the 2026 sweeps converged on and
        # they have to match the encoder, so they are not env-var knobs. On a
        # resume they come from the parent instead; see the header.
        --fwhm_ratio 0.25
        --observation_size 12
        --movement_mode continuous
        --hopfield_mode continuous
        --lambdas 11 12 13
        --Np 400
        --static-vectorhash
        --no-input_encoded_state
        --input_hopfield_signal
        --input_prev_reward
        --input_prev_action
        --input_hopfield_raw
        --input_sensory
        --init_log_std -0.5
        # Eval world
        --num_worlds 1
        --num_val_envs 10
        --n_val_trials 32
        --val_distractors 0 5 10
    )
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM output log: /home/jackking/cls/hopfield_nav/logs/slurm_${JOB_LABEL:-navigate}_${SLURM_JOB_ID}.out"
fi
echo "=== ${JOB_LABEL:-navigate}  schedule='$SCHEDULE' ==="
if [ "$RESUMING" = "1" ]; then
    echo "    resuming from $LOAD_CKPT"
    echo "    (config inherited from it; only the flags above override)"
else
    echo "    fresh init, encoder=$ENCODER, seed=$SEED"
fi

# EXTRA is unquoted on purpose: it is a flag string, not one argument.
python -u -m hopfield_nav.train_navigate "${ARGS[@]}" ${EXTRA:-}
