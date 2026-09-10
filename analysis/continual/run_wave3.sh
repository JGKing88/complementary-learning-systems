#!/bin/bash -l
#SBATCH --job-name=cl-wave3
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave3_%j.out
set -uo pipefail

# =============================================================================
# Wave 3 -- parameter isolation. docs/CONTINUAL_CONTROLS_PLAN.md section 4.3.
#
# The family the suite was missing. Waves 1 and 2 covered replay (ER, CLEAR,
# DER++) and parameter regularisation (EWC, SI, LwF): methods that share one set
# of weights across every task and differ in what they train on or how far they
# let the weights move. Nothing so far gives a task its own parameters, and that
# is the family the recurrent continual-learning literature actually points at.
#
#   J   HNET, learned base   the headline. A task-conditioned hypernetwork
#                            generates the policy's weights; the output
#                            regulariser pins what it generates for past tasks.
#   K   HNET, frozen base    the pretrained weights can never move; only the
#                            task-conditioned part does. Cannot forget through
#                            the shared component at all.
#   L   HNET, no base        pure von Oswald from scratch, and the only variant
#                            whose parameter count matches the baseline policy.
#   L0  from-scratch control the reference L is read against. Wave 1 has no
#                            from-scratch arm, so L would otherwise be
#                            uninterpretable.
#   M   multi-head           shared trunk, one head per task, oracle task id.
#                            Bounds the family: its heads cannot interfere, so
#                            whatever it fails to retain is trunk forgetting.
#   N   XdG                  a fixed random subset of hidden units per task,
#                            applied inside the recurrence.
#
# All of J-N are given an ORACLE TASK ID at training and evaluation time. That
# is a real advantage over every arm in Waves 1 and 2, and over the Hopfield
# store, which needs neither a task id nor a boundary. These are upper bounds on
# their family, not peers, and the results page says so.
#
# -- On the beta range -------------------------------------------------------
# Not decades around the published value. `analysis/continual/calibrate_beta.py`
# measured the penalty against this suite's BC loss first, and the answer was
# that von Oswald's beta=1 would have been a no-op here:
#
#     beta      bc_loss      penalty     pen/bc
#     0.01       7.7513   1.25547e-05   1.62e-06
#     1          7.6542    0.00133543   1.75e-04      <- the paper's value
#     100        7.8278      0.0820925   1.05e-02
#     10000      7.6904        1.23124   1.60e-01
#     1e+06     10.8268        4.04052   3.73e-01     <- BC loss now rising
#
# A sweep over {0.01 ... 100} -- the obvious one -- would have had every arm
# contribute under 1% of the objective and would have concluded that the
# regulariser does not help. Wave 2 had to be re-run twice for exactly this
# mistake, with DER++ and then CLEAR. So the sweep runs 1e2..1e7, centred where
# the penalty is actually comparable to the loss it is competing with.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1"      # same directory: the summaries read all waves
LOGS="$REPO/hopfield_nav/logs/wave3"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
[[ -f "$CKPT" ]] || { echo "[wave3] FATAL: missing $CKPT" >&2; exit 1; }

SEEDS=8
# Identical to Wave 2's, so every number lands on the same axes as the arms it
# is being compared with. The only additions are --arch and its knobs.
BASE=(--n_envs 5 --iters_per_block 200 --max_steps 200 --size 20
      --observation_size 60 --movement_mode continuous --goal_radius 0.5
      --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
      --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu --no-world_spec
      --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)
COMMON=("${BASE[@]}" --load_checkpoint "$CKPT")

echo "[wave3] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- J: HNET with a learned base (the headline arm) -------------------------
# beta=0 is the ablation that matters most: a hypernetwork with no regulariser.
# If J_b0 already retains, the win is the parameterisation and not the method;
# if it does not, the regulariser is doing the work. Run as --method none rather
# than beta=0 so it also carries no snapshot, which is the honest memory cost of
# "no regulariser".
for S in $(seq 1 $SEEDS); do
    launch "J_hnet_b0_s${S}" "${COMMON[@]}" --seed "$S" --arch hnet --method none
done
for B in 100 1000 10000 100000 1000000 10000000; do
for S in $(seq 1 $SEEDS); do
    launch "J_hnet_b${B}_s${S}" "${COMMON[@]}" --seed "$S" --arch hnet \
        --method hnet --method_args "beta=${B},normalize=true"
done; done

# --- K: HNET with a frozen base ---------------------------------------------
# The pretrained weights are pinned; only the task-conditioned residual moves.
# There is no shared component left to forget through, so if the regulariser
# still buys retention here it is protecting the generator itself.
for S in $(seq 1 $SEEDS); do
    launch "K_hnetfrz_b0_s${S}" "${COMMON[@]}" --seed "$S" --arch hnet \
        --hnet_base frozen --method none
done
for B in 10000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "K_hnetfrz_b${B}_s${S}" "${COMMON[@]}" --seed "$S" --arch hnet \
        --hnet_base frozen --method hnet --method_args "beta=${B},normalize=true"
done; done

# --- L: HNET from scratch, and the control it is read against ---------------
# `--hnet_base none` has nowhere to put a checkpoint and refuses one, so these
# two arms are the only ones in the wave that start from random weights. L0 is
# a plain RNN under the same conditions; without it, L's numbers would have no
# reference in the whole suite.
for S in $(seq 1 $SEEDS); do
    launch "L_hnetscratch_b10000_s${S}" "${BASE[@]}" --seed "$S" --arch hnet \
        --hnet_base none --method hnet --method_args "beta=10000,normalize=true"
    launch "L0_scratch_none_s${S}" "${BASE[@]}" --seed "$S" --arch rnn --method none
done

# --- M: multi-head with an oracle task id -----------------------------------
for S in $(seq 1 $SEEDS); do
    launch "M_multihead_s${S}" "${COMMON[@]}" --seed "$S" --arch multihead \
        --method none
done

# --- N: XdG -----------------------------------------------------------------
# Swept from mild to severe rather than fixed at the paper's 0.8, because the
# warm start makes gating expensive here: the checkpoint was trained with every
# unit available, so masking 80% of them hands this arm a broken policy at
# step 0 while every other arm starts from a working one.
for G in 0.2 0.5 0.8; do
for S in $(seq 1 $SEEDS); do
    launch "N_xdg_g${G}_s${S}" "${COMMON[@]}" --seed "$S" --arch xdg \
        --xdg_gating "$G" --method none
done; done

# XdG composes with SI, which is how Masse et al. ran it and how Ehret et al.
# benchmarked it. Two lambdas from the range Wave 2d found useful for SI alone.
for LAM in 100 10000; do
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.5_lam${LAM}_s${S}" "${COMMON[@]}" --seed "$S" --arch xdg \
        --xdg_gating 0.5 --method si --method_args "lam=${LAM},xi=0.1"
done; done

echo "[wave3] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave3] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave3] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave3] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave3_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
