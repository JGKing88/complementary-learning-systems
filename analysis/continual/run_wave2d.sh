#!/bin/bash -l
#SBATCH --job-name=cl-wave2d
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=pi_fiete
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
# Filled in from slurm_cl_calib_disc_21779972.out (2026-09-01). The printed
# tables are in logs/wave2d/calibration_*.txt and the JSON in
# $CLS_RESULTS/calibrate_discrete_*.json. Each range brackets the value where
# penalty/objective crosses ~1e-1, and runs far enough up to catch the
# plasticity trap -- which is visible in the table as the bc_loss column
# climbing, e.g. EWC at lam=1e7 (bc 1.77 -> 2.34) and SI at lam=1e6
# (1.73 -> 2.51).
#
# The reason this had to be measured rather than rescaled: the coefficients do
# not move in one direction. EWC and SI need roughly 100x MORE than the
# continuous suite swept, while DER++ needs about 10x LESS -- its term was
# already at ratio 0.16 at the smallest value the continuous wave used. A
# blanket correction by the ratio of the two loss scales would have been wrong
# for every one of them, because that ratio only describes the denominator;
# the numerators are a parameter-space penalty for EWC/SI and an MSE on
# Categorical logits for DER++, and those do not rescale together.
CALIBRATED=yes
EWC_LAMS="1000 10000 100000 1000000 10000000"        # crosses 0.1 at 1e4
SI_LAMS="100 1000 10000 100000 1000000"              # crosses 0.1 at ~1e3
LWF_ALPHAS="0.1 1 10 100 1000"                       # crosses 0.1 at ~1
CLEAR_COEFS="0.1 1 10 100 1000"                      # crosses 0.1 at ~1
DERPP_ALPHAS="0.01 0.03 0.1 0.3 1"                   # already 0.16 at 0.1
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
