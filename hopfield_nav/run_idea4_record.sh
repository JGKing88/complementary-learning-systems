#!/bin/bash -l
#SBATCH --job-name=i4record
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4record_%j.out

# Record an agent with idea 4's memory on the continual protocol, for training
# the memory on the agent's own movement (analysis/idea4/train_replay.py,
# GRID_MLP_NAV_PLAN §7b.15). N lifetimes, P at a time, each on a freshly
# minted set of held-out arenas (--split, --val_seed s) with its own seed.
#
#   CKPT=<navigate_u*.pt> OUT=<dir> [MEM=<memory.pt>] [N=16] [P=8] [S0=1] \
#     [ITERS_PER_BLOCK=40] [SPLIT=place=held_out] REPO=<code snapshot> \
#     sbatch hopfield_nav/run_idea4_record.sh
#
# Writes <OUT>/rec_s<s>.npz (the movement) and <OUT>/cl_s<s>.json (the
# protocol's scores for that lifetime).

set -euo pipefail
REPO=${REPO:?set REPO to a code snapshot}
CKPT=${CKPT:?set CKPT}
OUT=${OUT:?set OUT}
MEM=${MEM:-}
N=${N:-16}
P=${P:-8}
S0=${S0:-1}
ITERS_PER_BLOCK=${ITERS_PER_BLOCK:-40}
SPLIT=${SPLIT:-place=held_out}

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
mkdir -p "$OUT"
scaffold=$(python -u -m analysis.continual.prep_scaffold \
    --ckpt "$CKPT" --cache_root "$CLS_SCAFFOLD_CACHE" --static_vectorhash | tail -n 1)
echo "[i4record] scaffold cache: $scaffold"
run_one() {
    local s=$1
    python -u -m analysis.continual.agenthash \
        --out "$OUT/cl_s$s.json" --run_name "record s$s" \
        --ckpt "$CKPT" --device cuda \
        --n_envs 6 --iters_per_block "$ITERS_PER_BLOCK" \
        --max_steps 200 --seed $((3000 + s)) --num_full_iters 1 \
        --split "$SPLIT" --val_seed "$s" \
        --static_vectorhash --scaffold_cache "$scaffold" --mmap \
        --lock_store_after_goal --oracle_store_at_goal \
        --oracle_lock_store_not_at_goal --goal_radius 1 \
        $([ -n "$MEM" ] && echo --recurrent_memory "$MEM") \
        --record_memory "$OUT/rec_s$s.npz" --stochastic_policy \
        > "$OUT/log_s$s.txt" 2>&1 || echo "[i4record] s$s failed: $(tail -n 3 "$OUT/log_s$s.txt")"
}
for s in $(seq "$S0" $((S0 + N - 1))); do
    run_one "$s" &
    while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 10; done
done
wait
ls "$OUT"/rec_s*.npz | wc -l
echo "[i4record] done $(date)"
