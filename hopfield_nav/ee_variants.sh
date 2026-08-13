# The variant table for the explore/exploit line: one place, two readers.
#
# `ee_env <NAME>` prints that variant's environment as KEY=VALUE lines. The
# submitter reads it to size a job; the pack runner reads it to launch one. A
# variant defined in only one of those two is the failure this file exists to
# prevent -- it is how a wave ends up with a run whose config is not the one
# its own log claims.
#
# Anything not named here comes from run_ee.sh's defaults, which are the shared
# base. Keep each variant to the knobs it is actually asking about.

ee_env() {
    case "$1" in
    # === wave 1: explore ===================================================
    # X1/X2/X3 hold the PPO pool at 1280 trajectories and env-steps/update at
    # 256,000, trading envs_per_world against batch_envs. Only the SERIAL call
    # count (envs x steps) moves, and that is what wall-clock tracks -- so each
    # rung buys ~4x the updates for the same hours. Update-limited coverage
    # means the low rungs win; diversity-limited means they fall short.
    X1) cat <<'EOF'
ENVS_PER_WORLD=80
BATCH_ENVS=16
STEPS_PER_ROLLOUT=200
SCHEDULE=explore:400
EVAL_EVERY=25
EOF
        ;;
    X2) cat <<'EOF'
ENVS_PER_WORLD=20
BATCH_ENVS=64
STEPS_PER_ROLLOUT=200
SCHEDULE=explore:1100
EVAL_EVERY=50
EOF
        ;;
    X3) cat <<'EOF'
ENVS_PER_WORLD=8
BATCH_ENVS=160
STEPS_PER_ROLLOUT=200
SCHEDULE=explore:2000
EVAL_EVERY=100
EOF
        ;;
    # X4 matches the rollout horizon to the eval horizon. Coverage is scored
    # over 400 steps and every run in this lineage trained on 200, so the back
    # half of every eval rollout is a horizon the policy was never optimized
    # at -- and one that has finished sweeping by step 200 spends it retracing.
    # Identical to X2 otherwise, so the pair reads horizon alone.
    X4) cat <<'EOF'
ENVS_PER_WORLD=20
BATCH_ENVS=64
STEPS_PER_ROLLOUT=400
SCHEDULE=explore:550
EVAL_EVERY=25
EOF
        ;;

    # === wave 1: exploit ===================================================
    # P1 is v35's policy noise under a pure-exploit schedule, where the goal
    # reward is the ONLY reward. With log_std pinned at -1.8 (sigma 0.165) a
    # rollout's random-walk reach is 0.165*sqrt(200) ~ 2.3 cells on a 20x20
    # arena, so the goal may never be found and the reward is a constant that
    # advantage normalization turns into noise. v35 never met this: its
    # schedule opened on explore, where novelty pays for any movement at all.
    P1) cat <<'EOF'
ENVS_PER_WORLD=20
BATCH_ENVS=64
STEPS_PER_ROLLOUT=200
SCHEDULE=exploit:400
EVAL_EVERY=25
EOF
        ;;
    # P2 buys discovery with variance, then takes it back: sigma 0.61 for the
    # first 50 updates (random-walk reach ~8.6 cells, enough to stumble on a
    # goal from anywhere) annealed to 0.165 by u250, so the final policy is as
    # precise as P1's. That is what init_log_std is for, and the anneal is what
    # keeps the extra noise from costing steps-to-goal at the end.
    P2) cat <<'EOF'
ENVS_PER_WORLD=20
BATCH_ENVS=64
STEPS_PER_ROLLOUT=200
SCHEDULE=exploit:400
EVAL_EVERY=25
INIT_LOG_STD=-0.5
LOG_STD_ANNEAL_START_UPDATE=50
LOG_STD_ANNEAL_END_UPDATE=250
LOG_STD_ANNEAL_TARGET=-1.8
EOF
        ;;
    *)  echo "ee_env: unknown variant '$1'" >&2; return 1 ;;
    esac
}
