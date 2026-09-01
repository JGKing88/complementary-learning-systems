#!/bin/bash -l
#SBATCH --job-name=cl-wave2d
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave2d_%j.out
set -uo pipefail

# =============================================================================
# Wave 2 of the discrete suite: every method that carries a coefficient.
#
#   C  online EWC   lambda
#   F  SI           lambda
#   G  LwF          alpha
#   D  CLEAR        clone_coef
#   E  DER++        alpha
#
# Split from wave1d on the real dependency rather than on the continuous
# suite's numbering: these five cannot start until
# analysis/continual/calibrate_discrete.py has said where to sweep.
#
# The discrete objective is a Categorical(4) cross-entropy sitting near
# ln(4) = 1.39. The continuous suite's objective was a Gaussian negative
# log-likelihood of order 10. A coefficient carried across unchanged is
# therefore wrong by roughly that ratio, in the direction that makes every
# regulariser look *stronger* than it was tuned to be -- the mirror image of
# the DER++/CLEAR failure, where a range taken from a cross-entropy paper never
# reached 3% of a Gaussian objective and the conclusion was "the method does
# not help here" when the truth was "the knob was never turned on".
#
# So the ranges below are read off the calibration table, not off the
# continuous runs and not off the papers. The guard immediately after them
# exists because a placeholder that silently runs is worse than one that stops.
# =============================================================================

# --- FROM THE CALIBRATION TABLE ---------------------------------------------
# Set CALIBRATED=yes only after calibrate_discrete.py has run against the
# discrete checkpoint and these six lines have been updated from its output.
# The saved table lives at $CLS_RESULTS/calibrate_discrete.json and its printed
# form is echoed into $LOGS/calibration.txt by the launcher.
CALIBRATED=no
EWC_LAMS=""            # ratio ~1e-3 .. ~1e0, four decades
SI_LAMS=""
LWF_ALPHAS=""
CLEAR_COEFS=""
DERPP_ALPHAS=""
# -----------------------------------------------------------------------------

if [[ "$CALIBRATED" != "yes" ]]; then
    echo "[wave2d] FATAL: coefficient ranges have not been calibrated." >&2
    echo "[wave2d]        run:  python -m analysis.continual.calibrate_discrete \\" >&2
    echo "[wave2d]                  --ckpt \$CLS_CKPTS_RNN/pretrain_20x20_discrete/final.pt \\" >&2
    echo "[wave2d]                  --out \$CLS_RESULTS/calibrate_discrete.json" >&2
    echo "[wave2d]        then fill in the ranges above and set CALIBRATED=yes." >&2
    exit 1
fi

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1d"     # one histories dir for the whole discrete suite
LOGS="$REPO/hopfield_nav/logs/wave2d"
mkdir -p "$OUT" "$LOGS"

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_discrete/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave2d] FATAL: discrete pretrain checkpoint missing: $CKPT" >&2
    exit 1
fi
CKPT_MODE=$(python - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(((ck.get("cfg") or {}).get("agent") or {}).get("movement_mode", "unknown"))
PY
)
if [[ "$CKPT_MODE" != "discrete" ]]; then
    echo "[wave2d] FATAL: checkpoint movement_mode=$CKPT_MODE, expected discrete." >&2
    exit 1
fi

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=discrete
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu)

echo "[wave2d] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"
echo "[wave2d] ewc=$EWC_LAMS  si=$SI_LAMS  lwf=$LWF_ALPHAS"
echo "[wave2d] clear=$CLEAR_COEFS  derpp=$DERPP_ALPHAS"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" --no-world_spec \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

BASE=(--load_checkpoint "$CKPT" --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

# --- C: Online EWC ----------------------------------------------------------
for LAM in $EWC_LAMS; do
for S in $(seq 1 $SEEDS); do
    launch "C_ewc_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method online_ewc --method_args "lam=${LAM},gamma=1.0,fisher=true"
done; done

# --- F: Synaptic Intelligence ------------------------------------------------
for LAM in $SI_LAMS; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

# --- G: Learning without Forgetting -----------------------------------------
for A in $LWF_ALPHAS; do
for S in $(seq 1 $SEEDS); do
    launch "G_lwf_a${A}_s${S}" "${BASE[@]}" --seed "$S" \
        --method lwf --method_args "alpha=${A}"
done; done

# --- D: CLEAR ---------------------------------------------------------------
for CC in $CLEAR_COEFS; do
for S in $(seq 1 $SEEDS); do
    launch "D_clear_cc${CC}_s${S}" "${BASE[@]}" --seed "$S" \
        --method clear --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=${CC}"
done; done

# --- E: DER++ ---------------------------------------------------------------
for A in $DERPP_ALPHAS; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${A}_s${S}" "${BASE[@]}" --seed "$S" \
        --method derpp --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${A}"
done; done

echo "[wave2d] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave2d] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave2d] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave2d] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
