#!/bin/bash -l
#SBATCH --job-name=hnav-eval-all
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=64G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_eval_all_%j.out

# Run all four eval types (navigation det + stoch, goal discovery,
# exploration, realistic) on a group of checkpoints.

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

# ---------------------------------------------------------------------------
# Checkpoints to evaluate. One path per line.
# ---------------------------------------------------------------------------
CKPTS=(
    /home/jackking/cls/checkpoint/revived-water-17/hopfield_nav_update100.pt
)

# ---------------------------------------------------------------------------
# Shared eval parameters (override with env vars when invoking the script,
# e.g. `NUM_TRIALS=16 sbatch run_eval_all.sh`).
# ---------------------------------------------------------------------------
DEVICE="${DEVICE:-cuda}"
NUM_TRIALS="${NUM_TRIALS:-32}"
MAX_STEPS="${MAX_STEPS:-200}"
N_DISTRACTORS="${N_DISTRACTORS:-0 3 10}"      # space-separated
REALISTIC_STEPS=200
REALISTIC_SEED_OFFSET=300

# Repeat eval: N independent trials per env, fresh Hopfield each, primary phase only.
# Set REPEAT_TRIALS=0 to skip.
REPEAT_TRIALS="${REPEAT_TRIALS:-5}"
REPEAT_STEPS="${REPEAT_STEPS:-200}"
REPEAT_SEED_OFFSET=400

# Boolean flags: set to "1" to skip.
SKIP_REALISTIC=1
SKIP_NAV_STOCH=1

# Where to dump per-ckpt JSON result files (empty = don't write).
OUTPUT_DIR="${OUTPUT_DIR:-/home/jackking/cls/hopfield_nav/eval_results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

# Optional encoder override — leave empty to use the checkpoint's saved path.
ENCODER_OVERRIDE="${ENCODER_OVERRIDE:-}"

# Optional: override Npos (VectorHash side length; default = checkpoint or Πλ)
NPOS="${NPOS:-}"
# Optional: override cfg.num_val_envs saved in the checkpoint.
NUM_VAL_ENVS=5
# Set to 1 to pass --hopfield-oracle, 0 to pass --no-hopfield-oracle, empty = leave checkpoint as-is
HOPFIELD_ORACLE=0
# Set to 1 to pass --action-oracle, 0 to pass --no-action-oracle, empty = leave checkpoint as-is
ACTION_ORACLE=0

# Optional: 1 = --gbook-only, 0 = --no-gbook-only, empty = use checkpoint's vectorhash.gbook_only
GBOOK_ONLY=1

# ---------------------------------------------------------------------------
# Drive loop.
# ---------------------------------------------------------------------------
echo "=== eval_all: $(date) ==="
echo "device=$DEVICE num_trials=$NUM_TRIALS max_steps=$MAX_STEPS"
echo "n_distractors=($N_DISTRACTORS) realistic_steps=$REALISTIC_STEPS"
echo "repeat_trials=$REPEAT_TRIALS repeat_steps=$REPEAT_STEPS"
[ -n "$NPOS" ] && echo "Npos_override=$NPOS"
[ -n "$NUM_VAL_ENVS" ] && echo "num_val_envs_override=$NUM_VAL_ENVS"
[ -n "$HOPFIELD_ORACLE" ] && echo "HOPFIELD_ORACLE=$HOPFIELD_ORACLE"
[ -n "$ACTION_ORACLE" ] && echo "ACTION_ORACLE=$ACTION_ORACLE"
[ -n "$GBOOK_ONLY" ] && echo "GBOOK_ONLY=$GBOOK_ONLY"
echo "output_dir=$OUTPUT_DIR"
echo ""

for ckpt in "${CKPTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "!!! MISSING: ${ckpt}"
        continue
    fi

    # Derive a filename-safe tag from the checkpoint path for the JSON output.
    json_name=$(echo "$ckpt" | sed 's|/|__|g; s|\.pt$||')

    cmd=(python -m hopfield_nav.eval_all
        --ckpt "$ckpt"
        --device "$DEVICE"
        --num_trials "$NUM_TRIALS"
        --max_steps "$MAX_STEPS"
        --n_distractors $N_DISTRACTORS
        --realistic-steps "$REALISTIC_STEPS"
        --realistic-seed-offset "$REALISTIC_SEED_OFFSET"
        --repeat-trials "$REPEAT_TRIALS"
        --repeat-steps "$REPEAT_STEPS"
        --repeat-seed-offset "$REPEAT_SEED_OFFSET"
        --output-json "$OUTPUT_DIR/${json_name}.json"
        --plot-path "$OUTPUT_DIR/${json_name}_realistic_drift.png"
    )
    if [ -n "$ENCODER_OVERRIDE" ]; then
        cmd+=(--encoder "$ENCODER_OVERRIDE")
    fi
    if [ "$SKIP_REALISTIC" = "1" ]; then
        cmd+=(--skip-realistic)
    fi
    if [ "$SKIP_NAV_STOCH" = "1" ]; then
        cmd+=(--no-nav-stoch)
    fi
    if [ -n "$NPOS" ]; then
        cmd+=(--Npos "$NPOS")
    fi
    if [ -n "$NUM_VAL_ENVS" ]; then
        cmd+=(--num-val-envs "$NUM_VAL_ENVS")
    fi
    if [ "$HOPFIELD_ORACLE" = "1" ] || [ "$HOPFIELD_ORACLE" = "true" ] || [ "$HOPFIELD_ORACLE" = "True" ]; then
        cmd+=(--hopfield-oracle)
    elif [ "$HOPFIELD_ORACLE" = "0" ] || [ "$HOPFIELD_ORACLE" = "false" ] || [ "$HOPFIELD_ORACLE" = "False" ]; then
        cmd+=(--no-hopfield-oracle)
    fi
    if [ "$ACTION_ORACLE" = "1" ] || [ "$ACTION_ORACLE" = "true" ] || [ "$ACTION_ORACLE" = "True" ]; then
        cmd+=(--action-oracle)
    elif [ "$ACTION_ORACLE" = "0" ] || [ "$ACTION_ORACLE" = "false" ] || [ "$ACTION_ORACLE" = "False" ]; then
        cmd+=(--no-action-oracle)
    fi
    if [ "$GBOOK_ONLY" = "1" ] || [ "$GBOOK_ONLY" = "true" ] || [ "$GBOOK_ONLY" = "True" ]; then
        cmd+=(--gbook-only)
    elif [ "$GBOOK_ONLY" = "0" ] || [ "$GBOOK_ONLY" = "false" ] || [ "$GBOOK_ONLY" = "False" ]; then
        cmd+=(--no-gbook-only)
    fi

    echo ">>> ${ckpt}"
    "${cmd[@]}"
    echo ""
done

echo "=== done: $(date) ==="
echo "results written to: $OUTPUT_DIR"
