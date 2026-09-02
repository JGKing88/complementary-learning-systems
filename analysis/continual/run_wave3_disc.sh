#!/bin/bash -l
#SBATCH --job-name=cl-wave3d
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave3d_%j.out
set -uo pipefail

# Partition: ou_bcs_normal, NOT pi_fiete, whose QOS caps the whole group at
# cpu=48 against this job's 96 -- such a job never starts, it sits at
# QOSGrpCpuLimit looking exactly like an ordinary queue wait.

# =============================================================================
# The two parameter-isolation arms, in discrete. Deliberately NOT all of wave 3.
#
# Wave 3 was scoped out of the discrete suite, and the hypernetwork family --
# which is most of its cost -- stays out. What comes back is the pair the
# per-environment panels need: multi-head and XdG(+SI) are two of the six arms
# the panel list spans, and without them the discrete page draws four panels
# against the continuous page's six, so the two figures are not comparing the
# same methods.
#
#   M   multi-head, one movement head per environment
#   N   XdG, gating swept
#   N2  XdG + SI on the overlapping units
#
# The SI coefficient range comes from the discrete calibration table, where the
# penalty crosses ~10% of the objective at lam ~1e3 (analysis/continual/
# calibrate_discrete.py; slurm_cl_calib_disc_21779972.out). The continuous
# suite's best was g=0.5 at lam 1e4-1e5, so this brackets both readings rather
# than assuming either transfers.
#
# The gating fraction is dimensionless -- a share of hidden units held off --
# so unlike the coefficients it does carry across action spaces unchanged.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1d"     # one histories dir for the whole discrete suite
LOGS="$REPO/hopfield_nav/logs/wave3d"
mkdir -p "$OUT" "$LOGS"

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_discrete/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave3d] FATAL: discrete pretrain checkpoint missing: $CKPT" >&2
    exit 1
fi
CKPT_MODE=$(python - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(((ck.get("cfg") or {}).get("agent") or {}).get("movement_mode", "unknown"))
PY
)
if [[ "$CKPT_MODE" != "discrete" ]]; then
    echo "[wave3d] FATAL: checkpoint movement_mode=$CKPT_MODE, expected discrete." >&2
    exit 1
fi
echo "[wave3d] checkpoint verified discrete: $CKPT"

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=discrete
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu)
BASE=(--load_checkpoint "$CKPT" --lr 1e-3 --epochs 1 --n_minibatches 1
      --batch_envs 1)

echo "[wave3d] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" --no-world_spec \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- M: multi-head, oracle task id ------------------------------------------
for S in $(seq 1 $SEEDS); do
    launch "M_multihead_s${S}" "${BASE[@]}" --seed "$S" --arch multihead
done

# --- N: XdG -----------------------------------------------------------------
for G in 0.5 0.8; do
for S in $(seq 1 $SEEDS); do
    launch "N_xdg_g${G}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating "$G"
done; done

# --- N2: XdG + SI on the overlap --------------------------------------------
for LAM in 1000 10000 100000; do
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.5_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating 0.5 \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

echo "[wave3d] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave3d] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave3d] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave3d] all ${#PIDS[@]} tasks OK"
fi

exit 0
