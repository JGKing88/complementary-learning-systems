#!/bin/bash -l
#SBATCH --job-name=i4agent
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4agent_%j.out

# Idea 4 as a full agent (GRID_MLP_NAV_PLAN §7b.14, stage 3): idea 1's trained
# controller with idea 4's trained recurrent memory in place of its sensory
# key-value table, on the continual protocol scored as idea 1 was in §10.8
# (run_cl_ood.sh settings: 6 recorded val envs, 40 iterations per block,
# 200-step cap, seed 3000, one oracle store per env, goal radius 1, sampled).
#
#   MEM=<idea 4 memory.pt> TAG=<name> [CKPT=<idea 1 navigate_u*.pt>] \
#     [N_ENVS=6] [ITERS_PER_BLOCK=40] [MAX_STEPS=200] [STOCHASTIC=1] [C_SCALE=0.5] [SKIP_STAYS=0] \
#     REPO=<code snapshot> sbatch hopfield_nav/run_idea4_agent_cl.sh
#
# Writes $CLS_RUNS/idea4/agent_cl/cl_<TAG>.json, the plotter's panels and the
# retention summary.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navp2_task3r_k2_h128_gmlp_s42_24342872/navigate_u3800.pt}
MEM=${MEM:?set MEM to an idea 4 memory.pt}
TAG=${TAG:?set TAG}
N_ENVS=${N_ENVS:-6}
ITERS_PER_BLOCK=${ITERS_PER_BLOCK:-40}
MAX_STEPS=${MAX_STEPS:-200}
SEED=${SEED:-3000}
STOCHASTIC=${STOCHASTIC:-1}
C_SCALE=${C_SCALE:-0.5}
SKIP_STAYS=${SKIP_STAYS:-0}     # 1: step the memory only when the agent's cell changes

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
OUT=$CLS_RUNS/idea4/agent_cl
mkdir -p "$OUT"
scaffold=$(python -u -m analysis.continual.prep_scaffold \
    --ckpt "$CKPT" --cache_root "$CLS_SCAFFOLD_CACHE" --static_vectorhash | tail -n 1)
echo "[i4agent] scaffold cache: $scaffold"
prefix="$OUT/cl_${TAG}"
python -u -m analysis.continual.agenthash \
    --out "$prefix.json" \
    --run_name "idea 1 controller + idea 4 memory ($TAG), one store per env$([ "$STOCHASTIC" = "1" ] && echo ", sampled")" \
    --ckpt "$CKPT" --device cuda \
    --n_envs "$N_ENVS" --iters_per_block "$ITERS_PER_BLOCK" \
    --max_steps "$MAX_STEPS" --seed "$SEED" --num_full_iters 1 \
    --split recorded --val_seed 0 \
    --static_vectorhash --scaffold_cache "$scaffold" --mmap \
    --lock_store_after_goal --oracle_store_at_goal \
    --oracle_lock_store_not_at_goal --goal_radius 1 \
    --recurrent_memory "$MEM" --recurrent_c_scale "$C_SCALE" \
    $([ "$SKIP_STAYS" = "1" ] && echo --recurrent_skip_stays) \
    $([ "$STOCHASTIC" = "1" ] && echo --stochastic_policy)
python -u -m analysis.continual.plotting --history "$prefix.json" --out_prefix "$prefix" \
    --smooth 10 --show_std 0
echo "--- retention: $TAG"
python -u -m analysis.continual.retention --history "$prefix.json"
echo "[i4agent] done $(date)"
