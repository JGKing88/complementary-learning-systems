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
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:700,empty_frac=0.5"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # C2 is C1 with goal_reward set so the two shares come out near 1. The
    # value is read off C1's own log rather than guessed, so GOAL_REWARD must
    # be passed in alongside. After advantage normalization only the ratio
    # matters, so this is not "caring less about the goal" -- it is declining
    # to run the explore objective at a fraction of its strength.
    C2) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:700,empty_frac=0.5"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # C3 / C4 are the two orderings, and they need a parent: pass LOAD_CKPT.
    # They fail in opposite ways, which is why both are worth running. An
    # explore-first policy has no q readout to preserve and has to grow one; an
    # exploit-first policy has one -- P1's coverage of 0.035 is that readout
    # being linear in ||q|| -- and has to learn to suppress it.
    C3) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:700,empty_frac=1.0->0.5,anneal=200"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    C4) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:700,empty_frac=0->0.5,anneal=200"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
EOF
        ;;
    # C5: blocked, never interleaved. Worth running because the two regimes may
    # simply be fighting inside every gradient step, and blocking is the
    # cheapest way to find out -- at the risk of each block undoing the last,
    # which the eval curve shows as a sawtooth.
    C5) cat <<'EOF'
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:100 ; exploit:100 ; explore:100 ; exploit:100 ; explore:100 ; exploit:100 ; explore:100"
export EVAL_EVERY=50
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
export ENVS_PER_WORLD=20
export BATCH_ENVS=64
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:1100"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
export PERSISTENCE_BONUS=0
EOF
        ;;
    *)  echo "ee_env: unknown variant '$1'" >&2; return 1 ;;
    esac
}
