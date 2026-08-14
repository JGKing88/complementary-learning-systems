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

    # P3 is P2's recipe under the pinned contract flags, and is the exploit run
    # that counts. P1/P2 launched before `main` was merged, on code where a
    # goal-teleport unconditionally zeroed the recurrent state -- i.e.
    # reset_state_on_teleport=True, which is not what this line runs. Their
    # eval numbers are unaffected (a nav trial ends at the goal, so no teleport
    # happens inside one), so what they establish still stands: the sigma
    # anneal beats a pinned narrow sigma on speed, reaching 0.975 success at
    # 22.7 steps against P1's 1.000 at 41.5.
    #
    # 16 envs rather than P2's 20: diversity is what makes structure transfer
    # (3f), and serial calls are envs x steps, so this is 3,200 an update
    # against 4,000 and buys a longer run.
    P3) cat <<'EOF'
export ENVS_PER_WORLD=16
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="exploit:1200"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
export INIT_LOG_STD=-0.5
export LOG_STD_ANNEAL_START_UPDATE=50
export LOG_STD_ANNEAL_END_UPDATE=400
export LOG_STD_ANNEAL_TARGET=-1.8
EOF
        ;;

    # P4 is P3 with the learning rate cut. P3 diverged -- value_loss 6.9 ->
    # 5069 between u50 and u90, nav 1.000/28.9 -> 0.925/70.0 -- and §3i
    # attributes it to `reset_state_on_teleport=False` removing what had been
    # an accidental regularizer: an exploit rollout teleports every ~29 steps,
    # and zeroing the recurrent state there kept it from running the full 200
    # through a recurrence whose spectral radius training drives to ~4.2.
    #
    # LR is the allowed knob that acts on the cause: it slows the growth of the
    # recurrent weights doing the expanding. If it does not hold, halve
    # STEPS_PER_ROLLOUT, which halves the window the state has to grow in.
    P4) cat <<'EOF'
export ENVS_PER_WORLD=16
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="exploit:1200"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
export LR=1e-4
export INIT_LOG_STD=-0.5
export LOG_STD_ANNEAL_START_UPDATE=50
export LOG_STD_ANNEAL_END_UPDATE=400
export LOG_STD_ANNEAL_TARGET=-1.8
EOF
        ;;

    # P5 is P3 with reset_state_on_teleport back ON, which needs the standing
    # permission to depart from the pinned value and is used only because §3i
    # identified that setting as the cause of P3's divergence. It is the
    # control that isolates it: P4 changes LR and holds the flag, P5 changes
    # the flag and holds LR, so between them the mechanism is either confirmed
    # or it is not.
    #
    # If P5 is stable and P4 is not, the reset is doing the work and the pinned
    # value is genuinely incompatible with an expanding ReLU recurrence over a
    # 200-step rollout. If both are stable, LR was enough and the pin stands.
    P5) cat <<'EOF'
export ENVS_PER_WORLD=16
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="exploit:1200"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
export RESET_STATE_ON_TELEPORT=1
export INIT_LOG_STD=-0.5
export LOG_STD_ANNEAL_START_UPDATE=50
export LOG_STD_ANNEAL_END_UPDATE=400
export LOG_STD_ANNEAL_TARGET=-1.8
EOF
        ;;

    # P6 raises the time penalty, which §3l identifies as the pressure that is
    # missing. Navigation's 22 steps decompose into stride 0.65 and aim 0.68 --
    # a straight path, just slow and 49 degrees off a signal accurate to 4 --
    # and at TIME_PENALTY 0.05 against GOAL_REWARD 5.0 a 22-step approach costs
    # 1.1 against a prize of 5.0. Arriving sooner is barely worth anything.
    #
    # 0.2 makes that 4.4 against 5.0, i.e. roughly half the value of the goal
    # is at stake in how fast it is reached. Also carries P5's reset, since
    # otherwise this run would be testing two things and the divergence would
    # mask the result.
    P6) cat <<'EOF'
export ENVS_PER_WORLD=16
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="exploit:1200"
export EVAL_EVERY=50
export VAL_DISTRACTORS="0 10"
export RESET_STATE_ON_TELEPORT=1
export TIME_PENALTY=0.2
export INIT_LOG_STD=-0.5
export LOG_STD_ANNEAL_START_UPDATE=50
export LOG_STD_ANNEAL_END_UPDATE=400
export LOG_STD_ANNEAL_TARGET=-1.8
EOF
        ;;

    # === wave 3: explore, aimed at straight-run length ====================
    # W7 raises persistence_bonus, and the motion probes are what license it.
    # The gap between the best historical explore policy and the best of mine
    # is not stride, wall collisions or turn bias -- it is how far they go
    # before turning:
    #
    #     e8  u975  coverage 0.557  straight_run 10.77  turn_sigma 0.37
    #     L2  u1400 coverage 0.398  straight_run  1.58  turn_sigma 1.39
    #
    # L2 has already learned wall avoidance (blocked 34.8% -> 5.6% between u800
    # and u1400) so straightness is no longer the trap it is for a policy that
    # runs into walls -- which is the condition §3d said made persistence
    # ambiguous. With that condition met, the ambiguity resolves.
    #
    # Built on L5's shape rather than L1's because 16 envs is the better
    # baseline (0.296 at u300 against L1's 0.137), so this is one knob from the
    # best explore run rather than from the control.
    W7) cat <<'EOF'
export ENVS_PER_WORLD=16
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:1500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export PERSISTENCE_BONUS=0.2
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
    # L5 goes UP the diversity ladder, trading updates for envs at fixed
    # wall-clock. The reason is measured: on training envs e4L (4) and e8 (8)
    # learn the same amount of structure, +0.21 and +0.24 excess, but on
    # held-out envs e4L keeps 0.076 and e8 keeps 0.170. Diversity buys
    # *transfer* of the structure, not the structure itself -- and 8 is not
    # obviously the top of that curve. Serial calls are envs x steps, so 16
    # envs costs twice L1 per update and gets half the updates; whether that
    # trade is still favourable is the question.
    L5) cat <<'EOF'
export ENVS_PER_WORLD=16
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:1500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
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
    # C6 IS A DUPLICATE OF C4 -- DO NOT RUN IT AGAIN.
    #
    # I read the C3 comment above ("they need a parent: pass LOAD_CKPT"), found
    # no LOAD_CKPT in run_ee.sh, and concluded C4 had run cold. It had not:
    # C4's launch set LOAD_CKPT in its environment and its log carries the same
    # "Loaded agent state from .../P2/navigate_u175.pt" line C6's does. Same
    # parent, same config, same seed, so the two runs are bit-identical --
    # verified, 11/11 tensors equal at u100 and evals matching to 16 digits.
    # Kept only so the name is not silently reused.
    #
    # The lesson: check the run's own log for what it loaded before inferring
    # from the launcher what it must have done.
    C6) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_P2_20351854/navigate_u175.pt
EOF
        ;;
    # C7 = C6 with the regime balance corrected. At the defaults a 200-step
    # explore rollout that finds 40 new cells earns 0.3*40 = 12 against
    # GOAL_REWARD=5 for arriving, and §3p says pooled normalization leaves only
    # that ratio meaningful -- so the gate closing is the reward working as
    # specified, not a failure to learn. 20.0 puts arrival back above a good
    # explore rollout without touching novelty, which is what keeps coverage.
    C7) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export GOAL_REWARD=20.0
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_P2_20351854/navigate_u175.pt
EOF
        ;;
    # C8 is the clean version of C7, and the reason C7 is not clean.
    #
    # §3s says only the RATIO of arrival to a rollout of novelty matters, and
    # C7 changes it by raising GOAL_REWARD 5 -> 20. But advantages are
    # pool-normalized while the VALUE LOSS is not: 4x the reward is ~16x the
    # value loss, and C7 duly runs at value_loss 100-120 against C4's ~20, on a
    # trunk shared with the policy head. So C7 varies the ratio and the value
    # gradient's share of the trunk at once, and if it comes out worse -- as at
    # u100, 0.409/83.8 against C4's 0.991/65.4 -- the two cannot be separated.
    #
    # C8 moves the same ratio the other way: NOVELTY_REWARD 0.3 -> 0.075, so a
    # 40-new-cell rollout earns 3 against GOAL_REWARD's unchanged 5. Identical
    # 4x shift in the quantity §3s names, with rewards getting SMALLER, so
    # value targets shrink instead of exploding.
    C8) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export NOVELTY_REWARD=0.075
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_P2_20351854/navigate_u175.pt
EOF
        ;;
    # W9 is the first explore variant aimed at a MEASURED failure rather than a
    # guessed one. `motion_pattern` on W6 u900: perimeter_frac 0.463 against a
    # wall-aware walk's 0.040, with turn_bias 0.010 (no circling), msd_slope
    # 1.000 (no confinement) and blocked_frac 0.02 (no bumping). The policy
    # simply runs along the edge -- 46% of its steps in the 19% of cells that
    # form the boundary, on a loop of 76 cells whose reference coverage is
    # 0.202.
    #
    # wall_penalty is already exactly the right knob: config.py:160 applies it
    # per step the agent OCCUPIES an edge cell, and its comment names the
    # "perimeter-walk basin". The W series has only ever tried 0, 0.1 and 0.3.
    # At W6's 0.3 against NOVELTY_REWARD 0.3 a *fresh* edge cell is exactly
    # break-even, which is why 0.3 improved coverage without evicting the
    # policy. 0.6 makes a fresh edge cell net -0.3 and a revisited one -0.6.
    #
    # If it over-corrects the policy abandons the 1-cell border and caps at
    # 324/400 = 0.81, still double the best coverage measured, so the downside
    # is bounded and the upside is the whole deficit.
    W9) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.6
EOF
        ;;
    # C10 = the best components, and the run §3ac says to make.
    #
    # The composition grid so far varied the reward from parent P2 (C4 control,
    # C7, C8) and then varied the parent once, at C8's reward (C9). The reward
    # interventions all lost to the control; the parent swap was worth 4x in
    # success. The cell nobody has run is the winning reward with the winning
    # parent: C4's recipe, unchanged, from P5 u200 (1.000 / speed 0.759)
    # instead of P2 u175 (0.975 / 0.655).
    #
    # C10-vs-C4 is therefore a clean single-variable test of the parent on the
    # best-known reward, and C10-vs-C9 a clean test of the reward on the best-
    # known parent. If §3ac is right this is the strongest composite available.
    C10) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_P5_20363067/navigate_u200.pt
EOF
        ;;
    # C11 = best parent x best explore shaping, the cell the grid never reached.
    #
    # C9/C10 varied the reward from P5 with the default WALL_PENALTY of 0.1,
    # because they were designed before the perimeter basin was measured. W10
    # then established that WALL_PENALTY 0.3 with NOVELTY_REWARD 0.15 -- a fresh
    # edge cell at -0.15 rather than break-even -- is the best explore recipe in
    # the line (0.411 at u1000, past W6's 0.388 peak and level with L2's 0.419).
    #
    # No composite has ever used it. C11 is C9's schedule and parent with W10's
    # shaping, so the explore half of the composite trains under the recipe that
    # actually works on the explore axis instead of the default.
    C11) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
export NOVELTY_REWARD=0.15
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_P5_20363067/navigate_u200.pt
EOF
        ;;
    # C12 is the brief's own ordering -- explore first, then anneal exploit in --
    # finally run with an explorer worth starting from.
    #
    # C3 was defined for this in wave 2 and never launched, because at the time
    # the best explorer was 0.39 and falling. Every composite since has gone the
    # other way: exploit parent, explore ramped in. Now that W10 u2100 covers
    # 0.515 -- above the memoryless ceiling -- the reverse ordering has a parent
    # that is actually good at the half the composites never learn.
    #
    # `empty_frac=1.0->0.5` starts at pure explore and anneals exploit up to an
    # even split, the mirror of C9/C10/C11's `0->0.5`. Shaping is W10's, so the
    # explore half keeps the recipe that produced the parent.
    C12) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=1.0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
export NOVELTY_REWARD=0.15
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_W10_20372559/navigate_u2100.pt
EOF
        ;;
    # C12b: C12 at a third the learning rate.
    #
    # C12 produced a NaN policy mean within two updates. The checkpoint itself
    # is sound -- W10 u2100 scores 0.515 on 640 CPU trials, so the forward pass
    # is fine over 400 steps -- which puts the blowup in the optimiser step, not
    # the weights. A ReLU Elman recurrence trained 2,100 updates has a large
    # spectral radius (P3 measured ~4.2 on this architecture), and the first
    # gradient under a changed reward structure is where that bites.
    #
    # LR 3e-4 -> 1e-4 is the standard first remedy and the only change; if it
    # NaNs again the next lever is MAX_ACTION_NORM, which the brief allows and
    # which clips the runaway directly rather than slowing everything down.
    C12b) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=1.0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
export NOVELTY_REWARD=0.15
export LR=1e-4
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_W10_20372559/navigate_u2100.pt
EOF
        ;;
    # C12c: C12b plus the action-norm clip, which is what the NaN actually needs.
    #
    # C12 NaN'd within two updates at LR 3e-4. C12b at 1e-4 survived to u100 --
    # and its u100 eval is the best composite in the log, coverage 0.531 with
    # success 0.703 -- then NaN'd anyway. So the learning rate delays the blowup
    # rather than preventing it, which says the problem is the magnitude the
    # policy mean reaches, not the size of the step toward it.
    #
    # MAX_ACTION_NORM=2.0 clips exactly that. The brief left it inert but
    # explicitly permitted changing it, and 2.0 was chosen in M1 to be inert for
    # every healthy policy measured here (stride 0.3-1.3) while sitting below
    # the 2.87 a collapsed one reached -- so it should bind only on the runaway.
    C12c) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=1.0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
export NOVELTY_REWARD=0.15
export LR=1e-4
export MAX_ACTION_NORM=2.0
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_W10_20372559/navigate_u2100.pt
EOF
        ;;
    # W10 is to W9 what C8 was to C7, and for the same reason.
    #
    # W9 raised WALL_PENALTY 0.3 -> 0.6 to make a fresh edge cell net-negative.
    # It made training worse, not just the metric: mean_r fell monotonically
    # -0.10 -> -0.51 over 100 updates and coverage was 0.030 at u100 against
    # W6's 0.053. A penalty that large dominates the return, so the advantage
    # signal is almost entirely "was I on an edge" and the novelty differences
    # that are supposed to steer exploration are drowned out.
    #
    # W10 gets the identical sign flip from the other side: keep WALL_PENALTY
    # at W6's 0.3 and halve NOVELTY_REWARD to 0.15. A fresh edge cell is then
    # +0.15 - 0.3 = -0.15 instead of break-even, with the rewards getting
    # SMALLER rather than larger. Same arithmetic, no value-scale blowup.
    W10) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
export NOVELTY_REWARD=0.15
EOF
        ;;
    # C9: C8's clean ratio fix, but from P5 u200 (1.000 / 16.4 steps) instead of
    # P2 u175 (0.975 / 22.7). Parent quality bounds everything downstream of it,
    # and P5 is a strictly better exploiter reached in fewer updates. Runs
    # alongside C8 so parent and ratio are separable: C8-vs-C4 isolates the
    # ratio, C9-vs-C8 isolates the parent.
    C9) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=200
export SCHEDULE="interleave:2500,empty_frac=0->0.5,anneal=500"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export NOVELTY_REWARD=0.075
export LOAD_CKPT=/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_ee_P5_20363067/navigate_u200.pt
EOF
        ;;
    # W8: train at the horizon the metric is measured at. Everything so far
    # trains on 200-step rollouts and is scored on 400-step ones, so a cycle
    # longer than the training rollout costs nothing during training and
    # everything at eval -- and §3w says a cycle is exactly what the policy
    # settles into. The one existing 400-step run supports it: L4 reached 0.291
    # at u400 where W6, same 8-env shape at 200 steps, was at 0.220.
    #
    # W6's WALL_PENALTY=0.3 is kept because it is the best explore shaping
    # measured (§3t), so this is W6's recipe at the eval's own horizon. Serial
    # cost doubles to 3,200 calls/update; the peak is what matters, not the
    # update count, and §3x says the peak arrives and then leaves.
    W8) cat <<'EOF'
export ENVS_PER_WORLD=8
export BATCH_ENVS=16
export STEPS_PER_ROLLOUT=400
export SCHEDULE="explore:3000"
export EVAL_EVERY=100
export VAL_DISTRACTORS="0 10"
export WALL_PENALTY=0.3
EOF
        ;;
    *)  echo "ee_env: unknown variant '$1'" >&2; return 1 ;;
    esac
}
