#!/bin/bash -l
#SBATCH --job-name=cl-pretrain-disc
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=mit_normal_gpu
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_pretrain_disc_%j.out
set -uo pipefail

# =============================================================================
# The discrete pretraining checkpoint. Blocking for P2/P3/P4 of the discrete
# suite, and the reason this phase exists at all:
#
#   `restore_arch_from_ckpt` (training/rnn_setup.py) OVERRIDES the CLI's
#   --movement_mode with whatever the checkpoint was trained as, because the
#   two imply different parameter shapes -- a Categorical(4) head against a
#   Gaussian mean+log_std. So `--movement_mode discrete --load_checkpoint
#   <continuous ckpt>` does not produce a discrete run; it silently produces a
#   continuous one. The discrete method waves cannot borrow the existing
#   pretrained arm, they need their own.
#
# The recipe is the continuous checkpoint's own cfg, read back out of
# pretrain_20x20/final.pt, with movement_mode flipped and nothing else touched:
#
#   mode=mixed  n_envs=32  n_updates=1000  batch_envs=32  steps_per_rollout=100
#   size=20  observation_size=60  hidden=128  layers=1
#   lr=1e-3  epochs=4  n_minibatches=4  max_grad_norm=1.0
#   input_prev_action=FALSE
#
# `input_prev_action` stays OFF deliberately. It looks like it contradicts plan
# decision #1, but the continuous suite works the same way: the pretrained arms
# pass no --input_prev_action and `restore_arch_from_ckpt` would overwrite it
# from the checkpoint anyway, so only the from-scratch arm (A2) ever runs with
# the channel on. Turning it on here would make the discrete pretrained arm
# structurally different from its continuous counterpart, and the whole point
# of this suite is that the two are comparable panel-for-panel.
#
# seed=0, distinct from the wave seeds 1..8, so the 32 pretraining envs are not
# the 5 environments any method is later tested on. (The continuous
# checkpoint's seed is unrecoverable -- its run.json is a backfilled stub with
# config=null -- so this is a choice, not a match.)
#
# **This is the one phase where a GPU genuinely pays.** batch_envs=32 gives a
# real batch; the sequential waves run at batch_envs=1 where a 128-unit GRU on
# a batch of one is dominated by kernel-launch overhead and env stepping, which
# is why run_wave0.sh asks for none.
# =============================================================================

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

OUT_DIR="$CLS_CKPTS_RNN/pretrain_20x20_discrete"
LOGS="$REPO/hopfield_nav/logs/wave0d"
mkdir -p "$OUT_DIR" "$LOGS"

echo "[pretrain-disc] repo=$REPO"
echo "[pretrain-disc] out=$OUT_DIR"
echo "[pretrain-disc] started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u -m hopfield_nav.train_rnn \
    --mode mixed \
    --n_envs 32 --n_updates 1000 \
    --batch_envs 32 --steps_per_rollout 100 \
    --size 20 --observation_size 60 \
    --movement_mode discrete --goal_radius 0.5 \
    --hidden_size 128 --num_rnn_layers 1 \
    --lr 1e-3 --epochs 4 --n_minibatches 4 --max_grad_norm 1.0 \
    --n_eval_trials 32 --eval_max_steps 200 --eval_every 50 \
    --seed 0 --device cuda \
    --save_dir "$OUT_DIR" 2>&1 | tee "$LOGS/pretrain_discrete.log"

STATUS=${PIPESTATUS[0]}
echo "[pretrain-disc] finished $(date -Is)  status=$STATUS"

if [[ -f "$OUT_DIR/final.pt" ]]; then
    echo "[pretrain-disc] checkpoint present: $OUT_DIR/final.pt"
    # The path is passed in rather than read from the environment:
    # cls_env.sh sets CLS_CKPTS_RNN as a shell variable but only exports
    # CLS_RUNS, so os.environ would not see it.
    python - "$OUT_DIR/final.pt" <<'PY'
import sys, torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
a = ck["cfg"]["agent"]
print(f"[pretrain-disc] movement_mode={a['movement_mode']} hidden={a['hidden_size']} "
      f"prev_action={a['input_prev_action']}")
print("[pretrain-disc] head:", [k for k in ck["agent_state_dict"] if "movement" in k])
PY
else
    echo "[pretrain-disc] FATAL: no final.pt written" >&2
    exit 1
fi

exit "$STATUS"
