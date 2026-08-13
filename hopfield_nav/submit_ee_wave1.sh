#!/bin/bash
# Wave 1. Two independent questions, run concurrently because they share no
# state: what limits explore coverage, and what a pure exploiter can reach.
#
#   bash hopfield_nav/submit_ee_wave1.sh          # all six
#   bash hopfield_nav/submit_ee_wave1.sh X2 P1    # named subset
#
# See docs/EXPERIMENTS_EXPLORE_EXPLOIT.md for what each variant is asking.
set -u

PART=${PART:-mit_normal_gpu}
TIME=${TIME:-06:00:00}
SUB="sbatch --partition=$PART --time=$TIME"

# --- explore: is coverage limited by updates, by env diversity, or by the
#     horizon the policy is trained on? ------------------------------------
#
# X1/X2/X3 hold the PPO pool at 1280 trajectories and the env-steps per update
# at 256,000, and trade `envs_per_world` against `batch_envs`. Only the SERIAL
# call count -- envs x steps, the thing wall-clock actually tracks -- changes,
# so each rung down the ladder buys ~4x the updates in the same 6 hours. If
# coverage is update-limited the low rungs win outright; if it is
# diversity-limited they fall short and the ladder says where.
x1() { VARIANT=X1 ENVS_PER_WORLD=80 BATCH_ENVS=16  STEPS_PER_ROLLOUT=200 \
       SCHEDULE='explore:550'  EVAL_EVERY=25  $SUB hopfield_nav/run_ee.sh; }
x2() { VARIANT=X2 ENVS_PER_WORLD=20 BATCH_ENVS=64  STEPS_PER_ROLLOUT=200 \
       SCHEDULE='explore:1600' EVAL_EVERY=50  $SUB hopfield_nav/run_ee.sh; }
x3() { VARIANT=X3 ENVS_PER_WORLD=8  BATCH_ENVS=160 STEPS_PER_ROLLOUT=200 \
       SCHEDULE='explore:2800' EVAL_EVERY=100 $SUB hopfield_nav/run_ee.sh; }

# X4 is X2 with the rollout horizon matched to the eval horizon. Coverage is
# scored over 400 steps; every run in this lineage trained on 200, so the
# second half of every eval rollout is a horizon the policy has never been
# optimized at -- and a policy that has "finished" sweeping at step 200 spends
# the rest retracing. Same shape as X2 otherwise, so the pair is a clean read
# on horizon alone.
x4() { VARIANT=X4 ENVS_PER_WORLD=20 BATCH_ENVS=64  STEPS_PER_ROLLOUT=400 \
       SCHEDULE='explore:800'  EVAL_EVERY=25  $SUB hopfield_nav/run_ee.sh; }

# --- exploit: how well can following be done, and does it cold-start? -----
#
# P1 is the v35 policy-noise setting. Under a pure exploit schedule the only
# reward is reaching the goal, so with log_std pinned at -1.8 (sigma 0.165) a
# rollout's random-walk reach is ~0.165*sqrt(200) = 2.3 cells on a 20x20 arena:
# the goal may simply never be found, leaving a constant reward that advantage
# normalization turns into pure noise. v35 never met this because its schedule
# opened on explore, where novelty pays for any movement at all.
p1() { VARIANT=P1 ENVS_PER_WORLD=20 BATCH_ENVS=64 STEPS_PER_ROLLOUT=200 \
       SCHEDULE='exploit:400' EVAL_EVERY=25 $SUB hopfield_nav/run_ee.sh; }

# P2 buys the discovery with variance and then takes it back: sigma 0.61 for
# the first 50 updates (random-walk reach ~8.6 cells, enough to stumble onto a
# goal from anywhere), annealed to 0.165 by u250 so the final policy is as
# precise as P1's. This is what init_log_std is for, and the anneal is what
# stops the extra noise from costing steps-to-goal at the end.
p2() { VARIANT=P2 ENVS_PER_WORLD=20 BATCH_ENVS=64 STEPS_PER_ROLLOUT=200 \
       SCHEDULE='exploit:400' EVAL_EVERY=25 \
       INIT_LOG_STD=-0.5 LOG_STD_ANNEAL_START_UPDATE=50 \
       LOG_STD_ANNEAL_END_UPDATE=250 LOG_STD_ANNEAL_TARGET=-1.8 \
       $SUB hopfield_nav/run_ee.sh; }

if [ $# -eq 0 ]; then
    set -- X1 X2 X3 X4 P1 P2
fi
for v in "$@"; do
    case "$v" in
        X1) x1 ;; X2) x2 ;; X3) x3 ;; X4) x4 ;; P1) p1 ;; P2) p2 ;;
        *) echo "unknown variant: $v" >&2; exit 1 ;;
    esac
done
