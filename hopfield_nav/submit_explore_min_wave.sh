#!/bin/bash
# Wave 1 of the explore-only, minimum-data study. Ten jobs on two axes.
#
#   bash hopfield_nav/submit_explore_min_wave.sh          # submit
#   DRY=1 bash hopfield_nav/submit_explore_min_wave.sh    # print, submit nothing
#
# Axis 1 -- REWARD SHAPE (s1..s6), all at 16 x 200 = 3200 env-steps/update.
#   Only ratios to novelty are meaningful: advantages are pool-normalized and
#   an explore rollout with goals off is fixed-length, so overall reward scale
#   divides out and a flat per-step term cancels in the advantage. That last
#   point makes REVISIT_PENALTY exactly redundant with NOVELTY_REWARD when
#   NOVELTY_SCALE_REMAINING is off -- which is why s4/s5 pair revisit with the
#   scale on and off rather than sweeping it against a flat novelty.
#
# Axis 2 -- DATA PER UPDATE (d1..d4), all on v35's shape (= s1), spanning 16x
#   from 800 to 12800 env-steps/update. The question these answer: does
#   coverage collapse onto one curve against env-steps (data-limited) or
#   against updates (update-limited)? Nothing else in the wave distinguishes
#   those two, and they imply opposite ways to make training cheap.
#
# s1 is deliberately shared as both the shaping control and the middle rung of
# the data ladder.

set -euo pipefail
cd /home/jackking/cls

# ONLY="e2s42 c2s42" submits just those. Without it every variant goes,
# including ones already run -- the wave is a record of the whole design, not
# a description of what is currently outstanding, so a bare run resubmits
# s1/s2/d3 as well. See docs/EXPERIMENTS_EXPLORE_MIN.md for what is done.
submit() {
    local variant=$1; shift
    if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $variant "* ]]; then
        return 0
    fi
    if [ -n "${DRY:-}" ]; then
        echo "VARIANT=$variant $* sbatch hopfield_nav/run_explore_min.sh"
        return 0
    fi
    local jid
    jid=$(env VARIANT="$variant" "$@" sbatch --parsable \
              --job-name="hxm-$variant" hopfield_nav/run_explore_min.sh)
    echo "$variant -> $jid"
}

# --- Axis 1: reward shape, 16 x 200 -----------------------------------------

# s1  control: v35's exact shape, but pure explore.
submit s1

# s2  drop wall_penalty. The perimeter is 76 of 400 cells; a flat penalty on
#     standing there is a penalty on 19% of the ground that must be covered.
submit s2 WALL_PENALTY=0

# s3  lean on persistence. A sweep is "go straight, turn rarely", and cosine
#     persistence is the only dense signal available from step 1 -- before the
#     policy has any coverage competence for novelty to reinforce. Prior runs
#     that ranked persistence poorly were long ones, where PPO has time to
#     find straight motion on its own; the claim here is that it earns its
#     keep precisely in the low-data regime.
submit s3 WALL_PENALTY=0 PERSISTENCE_BONUS=0.15

# s4  revisit penalty ON TOP of scaled novelty. With the remaining-scale on,
#     novelty is state-dependent and the flat penalty no longer cancels, so
#     this is a real term: it is the escape gradient out of already-swept
#     ground, where novelty alone is silent.
submit s4 WALL_PENALTY=0 REVISIT_PENALTY=0.1

# s5  same, with the remaining-scale OFF -- the pre-scale shaping. Isolates
#     whether the late-rollout multiplier helps or destabilizes the value head.
submit s5 WALL_PENALTY=0 REVISIT_PENALTY=0.1 NOVELTY_SCALE_REMAINING=0

# s6  s2 with low epsilon. At eps=0.4, two of every five steps is a random
#     direction, which is exactly what breaks a straight sweep -- and eval
#     scores the deterministic policy. Cheap to test, and if it wins it wins
#     on every other variant too.
submit s6 WALL_PENALTY=0 EPSILON_EXPLORE=0.1

# f1  the log_std bracket. `--freeze_log_std` was a no-op on train_navigate
#     until it was fixed on 2026-08-07, so every run in v35's lineage trained
#     a learnable log_std whatever its launcher said. Every other job in this
#     wave is the first to actually get a frozen one; this one opts back out,
#     so the wave can tell the fix's effect from the shaping's.
submit f1 WALL_PENALTY=0 FREEZE_LOG_STD=0

# --- Axis 1b: is the explore regime better off with the goal live? -----------
#
# explore_goals_off=0 leaves the goal rewarding and, more to the point,
# TELEPORTING: on arrival the agent is relocated to a random cell and the goal
# stays put (world/vec_env.py:60). So a live goal is really two things at once
# -- a search-for-something-hidden curriculum, and a stream of random restarts
# inside the rollout. Both plausibly help coverage.
#
# Two caveats drive the settings here. The goal is FIXED per env for the whole
# run unless refreshed, and the sensory codebook makes "in env X walk to Y"
# memorizable -- which would buy goal-finding without buying exploration, so
# these randomize it per rollout. And the exploration EVAL scores an inert
# goal with no teleport, so whatever is learned has to survive losing both.
#
# Note this also breaks the fixed-length property the shaping analysis rests
# on: teleports zero the shaping mask, rollouts stop being interchangeable,
# and revisit_penalty stops being redundant with novelty here even with the
# remaining-scale off.

# g1  live goal at full strength (5.0, as v35's exploit regime used).
submit g1 WALL_PENALTY=0 EXPLORE_GOALS_OFF=0 RANDOMIZE_GOAL_PER_ROLLOUT=1

# g2  live goal, weak (1.0). Keeps the restart stream but stops +5 from
#     out-shouting novelty, so coverage stays the thing being optimized.
submit g2 WALL_PENALTY=0 EXPLORE_GOALS_OFF=0 RANDOMIZE_GOAL_PER_ROLLOUT=1 \
          GOAL_REWARD=1.0

# --- Axis 3: how few DISTINCT training envs still generalize? ----------------
#
# The knob that actually dominates both cost and generalization, and the one
# the first cut of this wave failed to vary. Note what `batch_envs` is NOT:
# the update loop runs one rollout for EVERY env in envs_per_world and pools
# them into a single PPO step (train_navigate.py:241), while batch_envs is the
# parallel-episode batch *inside* one env's rollout. So
#
#     env-steps/update       = envs_per_world x batch_envs x steps_per_rollout
#     SERIAL model calls/upd = envs_per_world x steps_per_rollout
#
# and it is the second that sets wall-clock, because the envs are looped, not
# batched together. Measured: 80 envs x 200 steps = 30.8 s/update, while
# cutting batch_envs 4x made it *slower* (48.3 s/u) rather than faster. So
# envs_per_world is a near-linear time lever in a way batch_envs is not.
#
# Generalization is a fair question here because the eval world is its own
# setup_world call with its own scaffold and its own randomly drawn envs --
# never trained on. Coverage on it is held-out coverage.
# Weighted toward the low end, and SEEDED there. With 1-4 envs the identity of
# the particular envs drawn dominates the result, so a single seed cannot
# support a claim either way about whether that many envs generalizes.
#
# 1000 updates, not 300, because at this size an update is ~0.8 s and the
# question is not "how far does it get in 300" but whether few envs is merely
# SLOWER or actually CAPPED. Those look identical on a short run and imply
# opposite things. Every run still evals at u300, so the fixed-update
# comparison against the 80-env control survives.
# e* -- turn envs down and change nothing else. The practical question.
for s in 42 43 44; do submit "e2s$s" ENVS_PER_WORLD=2 SEED=$s SCHEDULE='explore:1000'; done
submit e1s42 ENVS_PER_WORLD=1  SEED=42 SCHEDULE='explore:1000'
submit e4s42 ENVS_PER_WORLD=4  SEED=42 SCHEDULE='explore:1000'
submit e8    ENVS_PER_WORLD=8         SCHEDULE='explore:1000'
submit e16   ENVS_PER_WORLD=16        SCHEDULE='explore:1000'

# c* -- the same diversity ladder at a CONSTANT PPO pool of 1280 trajectories
# per update, by trading batch_envs against envs_per_world.
#
# Necessary because the e* runs confound two things. envs_per_world sets both
# how many distinct envs exist AND how many rollouts enter each PPO step, so
# e2 has a pool of 2x16 = 32 trajectories against the 80-env control's 1280.
# If e2 underperforms that could be a 40x smaller gradient batch rather than
# missing diversity, and those call for opposite fixes.
#
# batch_envs is free to spend here: it is the parallel-episode batch INSIDE one
# env's rollout, so raising it multiplies env-steps per update without adding
# a single serial model call. c1 does 200 serial calls per update against s1's
# 16000 for identical data. Pool size, env-steps/update and memory all match
# s1 exactly; only the number of distinct envs moves.
for s in 42 43 44; do submit "c2s$s" ENVS_PER_WORLD=2 BATCH_ENVS=640 SEED=$s SCHEDULE='explore:1000'; done
submit c1s42 ENVS_PER_WORLD=1 BATCH_ENVS=1280 SEED=42 SCHEDULE='explore:1000'
submit c4s42 ENVS_PER_WORLD=4 BATCH_ENVS=320  SEED=42 SCHEDULE='explore:1000'
submit c8s42 ENVS_PER_WORLD=8 BATCH_ENVS=160  SEED=42 SCHEDULE='explore:1000'

# --- Axis 2: data per update, v35 shape --------------------------------------

# d1  6400/update -- v35's own rollout shape, so this isolates "pure explore"
#     from every other change in the wave.
submit d1 STEPS_PER_ROLLOUT=400 BATCH_ENVS=16

# d2  1600/update -- quarter-length rollouts. Watch for the specific failure:
#     novelty_scale_remaining is driven by cells left THIS rollout, so a
#     100-step rollout on a 400-cell grid never gets past scale ~1.3 and the
#     endgame gradient never appears. Coverage should learn fast and cap early.
submit d2 STEPS_PER_ROLLOUT=100 BATCH_ENVS=16

# d3  800/update -- the floor. 4 envs is a very noisy PPO batch.
submit d3 STEPS_PER_ROLLOUT=200 BATCH_ENVS=4

# d5  400/update -- 4 envs AND 100-step rollouts, the combined floor.
#
#     This replaced the original d4 (64 envs, 12800/update) once d3 came in.
#     d4 existed to decide data-limited vs update-limited by bracketing the
#     ladder from above, and d3 settled it from below without needing the
#     upper rung: through u125, 4 envs tracked 16 envs update-for-update
#     (u75 .246/.240, u100 .312/.304, u125 .355/.291) on a QUARTER of the
#     data. That is an update-limited signature, so more data per update was
#     no longer a question worth a GPU. Pushing the floor down is.
submit d5 STEPS_PER_ROLLOUT=100 BATCH_ENVS=4
