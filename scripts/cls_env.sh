# Shared output locations for the shell drivers. Source this AFTER cd'ing to
# the repo root (SLURM copies a batch script to a node-local spool directory,
# so $BASH_SOURCE inside an sbatch job does not point at the repo):
#
#     cd /home/jackking/cls
#     source scripts/cls_env.sh
#
# Every path here mirrors an accessor in cls_paths.py; the two agree by way of
# the CLS_RUNS environment variable, which relocates all outputs at once:
#
#     CLS_RUNS=/tmp/cls_smoke sbatch hopfield_nav/run_continuous.sh
#
# The in-repo directory names (encoders/, checkpoint/, ...) still exist as
# symlinks into $CLS_RUNS, so relative paths saved in old checkpoints keep
# resolving. New scripts should use the variables below instead.

: "${CLS_RUNS:=/orcd/pool/003/jackking/cls_runs}"
export CLS_RUNS

CLS_ENCODERS="$CLS_RUNS/encoders"
CLS_CKPTS="$CLS_RUNS/agent_ckpts"
CLS_CKPTS_RNN="$CLS_RUNS/checkpoint_rnn"
CLS_HISTORIES="$CLS_RUNS/histories"
CLS_SCAFFOLD_CACHE="$CLS_RUNS/scaffold_cache"
CLS_FIGURES="$CLS_RUNS/figures"
CLS_RESULTS="$CLS_RUNS/results"
CLS_LOGS="$CLS_RUNS/logs"

# wandb writes its run directories beneath WANDB_DIR. Without this it defaults
# to ./wandb, i.e. inside the source tree.
export WANDB_DIR="${WANDB_DIR:-$CLS_RUNS/wandb}"

# Drivers write straight into these without creating them first (they used to
# be committed, in-tree directories), so make sure they exist.
mkdir -p "$WANDB_DIR" "$CLS_HISTORIES" "$CLS_SCAFFOLD_CACHE" "$CLS_LOGS" \
         "$CLS_FIGURES/model_comparison" "$CLS_RESULTS"
