#!/bin/bash -l
#SBATCH --job-name=et_uradius
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --partition=pi_evelina9
#SBATCH --output=/home/jackking/cls/encoder_training/scripts/logs/slurm-%j.out

# ==========================================================================
# Unique coding radius over encoders/. Edit below, then:
#     sbatch submit_unique_radius.sh
#
# pi_evelina9 is one node and is often queued behind other users; ou_bcs_low
# has ~25 GPU nodes and usually starts sooner. SBATCH directives cannot read
# shell variables, so override at submit time:
#     sbatch -p ou_bcs_low submit_unique_radius.sh
#
# The evaluator streams the grid a batch at a time, so memory is set by the
# cosine maps (n_refs x Npos x Npos float32 = 235 MB at 20 refs / Npos=1716),
# not by the codebook -- gen_gbook_2d would have been 10.2 GB. The --mem above
# is headroom, not a requirement. GPU time is the binding cost.
# ==========================================================================

N_REFS=20
BORDER=100
SEED=0                     # reference positions, shared across all encoders
HEADLINE_TRIM=16
TRIMS="0 4 16 64"
MARGIN_RADII="5 10 25 50"
PROFILE_LEVELS="0.9 0.5 0.1"
BATCH_SIZE=16384

# Which checkpoints. Repeatable; globs are relative to the encoders dir.
#   "*/encoder_best.pt"     33
#   "*/encoder_final.pt"   485
#   "*.pt"                 302 top-level, mostly pre-refactor saves whose
#                          state_dicts no longer load -- they land in the CSV
#                          with status=error rather than stopping the sweep
PATTERNS=("*/encoder_best.pt" "*/encoder_final.pt")

OUT_DIR=""                 # empty -> <sweeps_dir>/unique_radius_<timestamp>
LIMIT=""                   # e.g. 5 for a smoke test
RESUME=0                   # 1 to append to OUT_DIR and skip finished ckpts

# ==========================================================================

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

# Repo root. NOT derived from ${BASH_SOURCE[0]}: slurm copies the batch script
# into its spool directory before running it, so inside the job that path is
# /var/spool/slurmd/... and walking up from it lands nowhere near the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which makes this
# work from a worktree without hardcoding; override explicitly with
#     WORKDIR=/path/to/repo sbatch submit_unique_radius.sh
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-/orcd/home/002/jackking/cls}}"
if [ ! -d "$WORKDIR/encoder_training" ]; then
    echo "ERROR: WORKDIR=$WORKDIR is not a repo root (no encoder_training/)."
    echo "       Submit from the repo root, or set WORKDIR explicitly."
    exit 1
fi
cd "$WORKDIR"
mkdir -p /home/jackking/cls/encoder_training/scripts/logs

ARGS=(--n-refs "$N_REFS" --border "$BORDER" --seed "$SEED"
      --headline-trim "$HEADLINE_TRIM" --batch-size "$BATCH_SIZE")
[ -n "$TRIMS" ]           && ARGS+=(--trims $TRIMS)
[ -n "$MARGIN_RADII" ]    && ARGS+=(--margin-radii $MARGIN_RADII)
[ -n "$PROFILE_LEVELS" ]  && ARGS+=(--profile-levels $PROFILE_LEVELS)
[ -n "$OUT_DIR" ]         && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$LIMIT" ]           && ARGS+=(--limit "$LIMIT")
[ "$RESUME" = "1" ]       && ARGS+=(--resume)
for pat in "${PATTERNS[@]}"; do ARGS+=(--pattern "$pat"); done

echo "Host:    $(hostname)"
echo "Workdir: $WORKDIR"
echo "Flags:   ${ARGS[@]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "----- start -----"

PYTHONPATH="$WORKDIR" python -u -m encoder_training.sweep_unique_radius "${ARGS[@]}"
RC=$?
echo "----- exit code: $RC -----"
exit "$RC"
