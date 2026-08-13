# The variant table for the explore/exploit line: one place, two readers.
#
# `ee_env <NAME>` prints that variant's settings as `export KEY="value"` lines,
# for the caller to `eval` inside a subshell:
#
#     ( eval "$(ee_env X2)"; VARIANT=X2 bash hopfield_nav/run_ee.sh )
#
# `export ... = ...` rather than bare `KEY=VALUE` fed to `env`, because two of
# these values contain spaces -- `VAL_DISTRACTORS="0 10"` and C5's
# semicolon-separated schedule -- and an unquoted command substitution would
# word-split them into separate arguments. The subshell is what keeps one
# variant's overrides from becoming the next one's defaults.
#
# Anything not named here comes from run_ee.sh's defaults, which are the shared
# base. Keep each variant to the knobs it is actually asking about.

ee_env() {
    case "$1" in
    # === wave 1: explore ===================================================
    # X1/X2/X3 hold the PPO pool at 1280 trajectories and env-steps/update at
    # 256,000, trading envs_per_world against batch_envs. Only the SERIAL call
    # count (envs x steps) moves, and that is what wall-clock tracks -- so each
    # rung buys ~4x the updates for the same hours.
    #
    # Prior data (docs/EXPERIMENTS_EXPLORE_EXPLOIT.md) already places the
    # diversity floor between 2 and 4 envs, so the surviving value of this
    # ladder is X1: same shape as explore-min's s1, but 60 rays at
    # wall_resolution 4 against s1's 12 at resolution 1. s1 scored 0.517.
    X1) cat <<'EOF'
export ENVS_PER_WORLD=80
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:400"
export EVAL_EVERY=25
EOF
        ;;
    X2) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:1100"
export EVAL_EVERY=50
EOF
        ;;
    X3) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=160
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:2000"
export EVAL_EVERY=100
EOF
        ;;
    # X4 matches the rollout horizon to the eval horizon. Coverage is scored
    # over 400 steps and every run in this lineage trained on 200, so the back
    # half of every eval rollout is a horizon the policy was never optimized
    # at -- and one that has finished sweeping by step 200 spends it retracing.
    # Identical to X2 otherwise, so the pair reads horizon alone.
    X4) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=400
export SCHEDULE="explore:550"
export EVAL_EVERY=25
EOF
        ;;

    # === wave 1: exploit ===================================================
    # P1 is v35's policy noise under a pure-exploit schedule. It was expected
    # to struggle to find a goal at all with sigma pinned at 0.165; it reached
    # 1.000 nav success by u50, because exploit rollouts are not reward-sparse
    # here -- wall_penalty and persistence_bonus fire in them too.
    P1) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="exploit:400"
export EVAL_EVERY=25
EOF
        ;;
    # P2 starts at sigma 0.61 and anneals to 0.165 by u250. Through u50 it is
    # strictly worse than P1 (0.703 against 1.000) -- the wide sigma lets
    # samples hide the mean, which is the argument for freezing it low. The
    # anneal is the test of whether that is recoverable.
    P2) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="exploit:400"
export EVAL_EVERY=25
export INIT_LOG_STD=-0.5
export LOG_STD_ANNEAL_START_UPDATE=50
export LOG_STD_ANNEAL_END_UPDATE=250
export LOG_STD_ANNEAL_TARGET=-1.8
EOF
        ;;

    # === wave 2: composition =============================================
    # All at X2's shape. Every one logs `adv_share`, so the pooled-normalizer
    # reweighting is measured per update rather than assumed.
    #
    # C1 is the v35-like control: a constant 50/50 mix from scratch, at v35's
    # goal_reward of 5. Its job is to MEASURE the imbalance -- an exploit
    # rollout's reward is a 5.0 spike, an explore rollout's a 0.3 novelty
    # bonus, they share one divisor, and the ratio decides how much of each
    # objective actually reaches the weights.
    C1) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0.5"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # C2 is C1 with goal_reward set so the two shares come out near 1. The
    # value is read off C1's own log rather than guessed, so GOAL_REWARD must
    # be passed in alongside. After advantage normalization only the ratio
    # matters, so this is not "caring less about the goal" -- it is declining
    # to run the explore objective at a fraction of its strength.
    C2) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0.5"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # C3 / C4 are the two orderings, and they need a parent: pass LOAD_CKPT.
    # They fail in opposite ways, which is why both are worth running. An
    # explore-first policy has no q readout to preserve and has to grow one; an
    # exploit-first policy has one -- P1's coverage of 0.035 is that readout
    # being linear in ||q|| -- and has to learn to suppress it.
    C3) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=1.0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    C4) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # C5: blocked, never interleaved. Worth running because the two regimes may
    # simply be fighting inside every gradient step, and blocking is the
    # cheapest way to find out -- at the risk of each block undoing the last,
    # which the eval curve shows as a sawtooth.
    C5) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:200 ; exploit:200 ; explore:200 ; exploit:200 ; explore:200 ; exploit:200 ; explore:200 ; exploit:200 ; explore:200 ; exploit:200 ; explore:200 ; exploit:200 ; explore:200"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;

    # === wave 2: explore ==================================================
    # W1 removes persistence_bonus. coverage_reference says it is a force
    # toward small turn sigma, and small turn sigma is the worst region of the
    # memoryless table (0.40 against 0.56 diffusive) because a straight run
    # crosses the arena and then pushes into a wall, losing every step until
    # something turns it. v35 and the whole explore-min wave ran it at 0.05.
    # Identical to X2 otherwise.
    W1) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export PERSISTENCE_BONUS=0
EOF
        ;;
    # W2 removes wall_penalty, for a reason that only shows up once the
    # memoryless ceiling is known. Beating 0.56 needs a *stateful* strategy,
    # and the cheapest one available here is an inward spiral: walk the
    # perimeter, step in, walk the next ring. It needs almost no memory -- a
    # ring index and the distance to the wall ahead, which the foveal cone
    # reports directly -- and at stride 1 it covers all 400 cells in 400 steps.
    #
    # wall_penalty exists to break the "perimeter-orbit" basin, where an agent
    # racks up coverage by riding the edge. But an inward spiral *starts* with
    # exactly that, so the penalty taxes the one high-coverage strategy this
    # policy could plausibly learn. explore-min tested wall=0 and saw nothing
    # (0.510 against 0.517) -- with 12 rays at wall_resolution 1, where the
    # agent could not localize well enough to run a spiral anyway. That is no
    # longer true (§2), which is what makes this worth retesting.
    W2) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0
EOF
        ;;
    # W3 drops both, i.e. novelty and the time penalty alone. If a stateful
    # strategy is being suppressed by the shaping rather than not found, this
    # is the configuration with nothing in its way.
    W3) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0
export PERSISTENCE_BONUS=0
EOF
        ;;
    # W4 / W5 cut epsilon-greedy, which the motion probe turned into a first-
    # order concern. The override injects a **unit-magnitude** random step,
    # while the policy's own stride at this stage is ~0.25 cells. So at eps=0.4
    # the agent's training-time motion is dominated by steps it did not choose
    # and is not trained on -- they are masked out of the move loss -- while
    # the novelty they earn still flows into the value function and the
    # advantages. Eval then runs the policy alone, at a quarter of the stride
    # that produced the training reward.
    #
    # The risk in cutting it is the opposite one: eps is also what gets a
    # near-stationary policy off its starting cell early on. W4 keeps a little,
    # W5 removes it, and the pair brackets that.
    W4) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export EPSILON_EXPLORE=0.1
export EPSILON_ANNEAL_UPDATES=200
EOF
        ;;
    W5) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export EPSILON_EXPLORE=0
EOF
        ;;
    # === wave 2: long explore =============================================
    # The finding that redirects this half: the only policy in the lineage that
    # beats a matched random walk (e4L, excess +0.20, coverage 0.616) got there
    # at ~u1800 with the SAME shaping as everything else. The structured,
    # wall-avoiding sweep is a late behaviour, and every run that stopped near
    # u1000 stopped before it.
    #
    # So the shape to run is the cheap one, for a long time. e4L's own shape --
    # a handful of envs at batch 16 -- costs `envs x steps` serial calls per
    # update, which at 8 envs is 1,600 against X1's 16,000. That is what buys
    # 3000 updates inside a 6 h wall.
    L1) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # L2 matches e4L exactly on env count, to check that its result reproduces
    # on the new encoder and sensory resolution before anything is built on it.
    L2) cat <<'EOF'
export ENVS_PER_WORLD=4
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # L3 is L1 with the learning rate dropped for the second half. e4L reached
    # 0.616 at u1800 and 0.027 by u2025 -- value_loss climbing 0.66 -> 4.6, then
    # reward collapsing to the value of an agent pinned against a wall, with
    # move_loss at zero and no gradient out. A late LR cut is the cheapest guard
    # the schedule grammar already expresses, and it costs nothing if the
    # collapse was going to be avoided anyway.
    L3) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:1500 ; explore:1500,lr=1e-4"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # W6 raises wall_penalty instead of removing it, which is the opposite of
    # W2 and now the better-supported bet. §3f measured the cost of wall
    # collisions at ~0.30 coverage: e8 loses 38% of its steps to the clip, e4L
    # 0.07%, at identical shaping. A fully blocked step always lands the agent
    # on an edge cell, so wall_penalty does charge for it -- evidently not hard
    # enough to fix it inside 975 updates at 0.1 against a novelty of 0.3.
    # Raising it to parity is the direct test.
    #
    # W2 (wall_penalty=0) stays as the other bracket. Its motivation -- that
    # the penalty taxes the perimeter leg of a spiral -- is weakened by this,
    # but not refuted: the perimeter still has to be visited.
    W6) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
EOF
        ;;
    # L4 carries wave 1's horizon question into the cheap shape: rollouts as
    # long as the 400-step eval, so the back half of an eval rollout is not a
    # horizon the policy was never optimized at. Serial calls are envs x steps,
    # so this costs twice L1 per update and gets half the updates.
    L4) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=400
export SCHEDULE="explore:1500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # === staged, NOT launched: needs a knob outside the agreed list ========
    # M1 caps the action's L2 norm. Everything else here stays on the list;
    # this one does not, so it is defined and left unrun until asked for.
    #
    # The case for it is §3f's collapse: the run that destroyed the best
    # explore policy in this lineage did so through a runaway in the policy
    # MEAN -- commanded stride 1.17 -> 2.87 with the turn width tightening to
    # 0.13, ending with 80% of steps deleted by the wall clip and the agent
    # earning exactly the reward of standing still. freeze_log_std cannot
    # prevent it, because it pins the noise and the runaway is in the mean.
    #
    # 2.0 is chosen to be inert for every healthy policy measured here (stride
    # 0.3-1.3) and to sit below the 2.87 the collapsed one reached. It is not
    # continuous_normalize, which was ruled out: that fixes stride at a
    # constant, this only clips the top.
    M1) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export MAX_ACTION_NORM=2.0
EOF
        ;;
    *)  echo "ee_env: unknown variant '$1'" >&2; return 1 ;;
    esac
}
