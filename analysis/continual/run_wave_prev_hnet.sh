#!/bin/bash -l
#SBATCH --job-name=cl-wave-prev-J
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave_prev_J_%j.out
set -uo pipefail

# =============================================================================
# The 24 hypernetwork tasks that run_wave_prev.sh failed, re-run correctly.
#
# The first version carried `--hnet_base none` over from run_scratch.sh, where
# it is right because that wave has no checkpoint, and then added one. The
# agent refused:
#
#     hnet base='none' has no base vector to warm-start; it is the
#     from-scratch variant. Use base='learned' or 'frozen' with a
#     checkpoint, or drop --load_checkpoint.
#
# Which is the good failure: 24 tasks stopped rather than 24 runs quietly
# training something that was not a warm start and reporting it as one. The
# other 440 tasks in that wave are unaffected, and the discrete twin never had
# an HNET arm to get wrong.
#
# base='learned' matches the pretrained suite's own J arm in run_wave3.sh.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1_p"
LOGS="$REPO/hopfield_nav/logs/wave_prev"
mkdir -p "$OUT" "$LOGS"

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_prev/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave-prev-J] FATAL: checkpoint missing: $CKPT" >&2
    exit 1
fi
read -r CK_MODE CK_PREV <<< "$(python - "$CKPT" <<'PY'
import sys, torch
a = (torch.load(sys.argv[1], map_location="cpu",
                weights_only=False).get("cfg") or {}).get("agent") or {}
print(a.get("movement_mode", "?"), bool(a.get("input_prev_action")))
PY
)"
if [[ "$CK_MODE" != "continuous" || "$CK_PREV" != "True" ]]; then
    echo "[wave-prev-J] FATAL: checkpoint is $CK_MODE / prev=$CK_PREV;" >&2
    echo "[wave-prev-J]        expected continuous / True." >&2
    exit 1
fi
echo "[wave-prev-J] checkpoint verified: continuous, prev_action on"

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=continuous
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --no-world_spec)
BASE=(--load_checkpoint "$CKPT" --lr 1e-3 --epochs 1 --n_minibatches 1
      --batch_envs 1)

echo "[wave-prev-J] started $(date -Is)"
PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

for B in 10000 100000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "J_hnet_b${B}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch hnet --hnet_base learned --method hnet --method_args "beta=${B}"
done; done

echo "[wave-prev-J] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave-prev-J] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave-prev-J] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
else
    echo "[wave-prev-J] all ${#PIDS[@]} tasks OK"
fi
exit 0
