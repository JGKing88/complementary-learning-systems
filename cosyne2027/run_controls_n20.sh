#!/bin/bash -l
#SBATCH --job-name=cosyne-n20
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=ou_bcs_normal
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/cosyne_n20_%j.out
set -uo pipefail

# The six continual-learning controls of the Cosyne main-results figure at
# N=20 arenas. Same cell, flags and checkpoint as analysis/continual/
# run_wave_prev.sh (continuous, pretrained, previous-action channel), only
# the arena count changes and only the configurations the figure uses run:
# each method's best at T_eval = 100 from the N=5 selection
# (cosyne2027/cl_results.py), not re-tuned at N=20. 6 configs x 8 seeds,
# one single-threaded process each. Evaluation grows as N^2 (every update
# evaluates every seen arena), ~2.5-4 h a run.
#
#   sbatch cosyne2027/run_controls_n20.sh

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/cosyne2027-figs}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/cosyne2027_n20"
LOGS="$CLS_RUNS/logs/cosyne_n20"
mkdir -p "$OUT" "$LOGS"

N_ENVS=20; SIZE=20; OBS=60; MOVEMENT=continuous
MAX_STEPS=200; ITERS=200; SEEDS=8
COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --no-world_spec)

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_prev/final.pt"
[[ -f "$CKPT" ]] || { echo "[n20] FATAL: missing $CKPT" >&2; exit 1; }
read -r CK_MODE CK_PREV <<< "$(python - "$CKPT" <<'PY'
import sys, torch
a = (torch.load(sys.argv[1], map_location="cpu",
                weights_only=False).get("cfg") or {}).get("agent") or {}
print(a.get("movement_mode", "?"), bool(a.get("input_prev_action")))
PY
)"
if [[ "$CK_MODE" != "continuous" || "$CK_PREV" != "True" ]]; then
    echo "[n20] FATAL: checkpoint is $CK_MODE / prev_action=$CK_PREV" >&2
    exit 1
fi
echo "[n20] checkpoint verified: continuous, prev_action on"
BASE=(--load_checkpoint "$CKPT" --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)
echo "[n20] repo=$REPO cpus=${SLURM_CPUS_PER_TASK:-$(nproc)} started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

for S in $(seq 1 $SEEDS); do
    launch "A_lr3e-4_rst0_ep1_s${S}" --load_checkpoint "$CKPT" --seed "$S" \
        --lr 3e-4 --epochs 1 --n_minibatches 1 --batch_envs 1
    launch "B_er_bufinf_rb4_s${S}" "${BASE[@]}" --seed "$S" \
        --method er --method_args "buffer_size=inf,replay_batches=4,sampling=balanced"
    launch "D_clear_cc0.01_s${S}" "${BASE[@]}" --seed "$S" \
        --method clear --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=0.01"
    launch "C_ewc_lam10000_s${S}" "${BASE[@]}" --seed "$S" \
        --method online_ewc --method_args "lam=10000,gamma=1.0,fisher=true"
    launch "N2_xdgsi_g0.5_lam10000_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating 0.5 \
        --method si --method_args "lam=10000,xi=0.1"
    launch "M_multihead_s${S}" "${BASE[@]}" --seed "$S" --arch multihead
done

echo "[n20] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[n20] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[n20] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo "[n20] all ${#PIDS[@]} tasks OK"
