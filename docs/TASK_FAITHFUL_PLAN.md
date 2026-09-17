# Task-faithful navigation training — plan

Written 2026-09-16. Experiment log: `docs/EXPERIMENTS_TASK_FAITHFUL.md`.
Predecessor lines: `docs/EXPERIMENTS_SAMPLE_EFF.md` (§7 one arena, §8 fixed
goals), `docs/EXPERIMENTS_NAV_TRI.md` (the interleaved explore/exploit recipe
this replaces the *training* side of; the evaluation side is unchanged).

> **Status: plan approved 2026-09-16, implementation in progress.** Sections
> §3–§5 are the work; §6 is the first wave.

## 0. What this is

The `d0_base` recipe trains two hand-split regimes in the same PPO update:
*explore* rollouts (memory holds distractors only, novelty pays, the
trajectory ends at the goal) and *exploit* rollouts (the goal is pre-stored by
an oracle before the rollout, novelty off, goal reward and teleport). The
split is a training device — the task the agent is actually evaluated on
(the continual protocol in `analysis/continual/agenthash.py`) has no such
split: the agent is dropped into an arena, searches, the goal is stored when
it gets there, and from then on it navigates.

This line trains on that task directly:

1. The agent is introduced to an arena. Its Hopfield holds distractors only
   (U[0, 10] patterns from cells outside the arena, as now).
2. It searches. Rewards are the explore rewards: novelty 0.3 (remaining-
   scaled, cap 10), wall −0.1, persistence +0.2, time −0.05, goal +2.0.
3. On the first goal touch the goal cell's pattern is written to the Hopfield
   (oracle write, exactly once, never anywhere else; the agent's store head
   never writes), the agent is teleported to a random cell, and the rollout
   continues. From that step on the rewards are the exploit rewards: novelty
   off, everything else unchanged. Every further touch pays +2.0 and
   teleports.
4. Memory and recurrent state reset at the end of the 200-step rollout, not
   at goal touches. Each of the B trajectories in a rollout has its own
   Hopfield.
5. **Revisits** (`visits=K`, Jack: "kind of an explore/exploit split, but
   more realistic — worth trying"): the memory a trajectory built is kept for
   the next K−1 rollouts of the same env while the recurrent state and
   prev_reward reset each rollout. Visit 1 is search → store → navigate;
   visits 2..K arrive with whatever visit 1 left — goal + distractors if it
   found the goal, distractors only if it did not. This is exactly the
   continual protocol's revisit trial (`evaluation/protocols.py:110-112`
   resets `h_rnn` and `prev_reward` per trial while the Hopfield persists).
   K=1 is the pure protocol; both are run.
6. Never `input_goal_in_memory`. The agent has prev_reward (+2.0 on the step
   after the store) and the readout itself; it has to work out that the
   memory became trustworthy.

Primary training set: **3 arenas with fixed interior goals** (the §8 setting
that produced a q-follower under the old split). Controls: 1 arena fixed
interior goal, 1 arena with the goal redrawn per rollout.

## 1. Why the old result does not transfer for free

Under the old split, memorising "this arena's walls → heading to the goal
cell" paid nothing extra: exploit paid memoriser and q-follower equally, and
the explore regime actively punished going to the known goal (the trajectory
ended, forfeiting novelty). Here a memoriser beelines from t=0 and collects
~19 touches ≈ 38 reward per rollout, while a q-follower must search first
and collects ~12–16 ≈ 24–32. The pull toward the position map is stronger,
which is why one fixed-goal arena is a control and three is the primary.
The verdicts (§5) are the ones that see through it: `follow_q` on unseen
arenas and the continual protocol.

## 2. What already exists (verified 2026-09-16)

| piece | where | state |
|---|---|---|
| one Hopfield per trajectory, batched recall | `rollout/signal.py:198-205`; `collector.py:94` requires the list when stores are allowed | exists |
| oracle write at the goal | `collector.py:589` `auto_store_active & at_goal_mask`; writes the goal cell's pattern (`allow_offcell_store=False`) | exists, but gated on `auto_store_warmup` updates and fires on *every* touch |
| per-trajectory "goal stored" flag | `agent_goal_store_fired` (`collector.py:128`) | exists; seeded from a scalar `goal_in_memory_init` |
| teleport with state kept | `reset_state_on_teleport` default off since 08-12 | exists |
| regime-independent shaping | wall / persistence / time apply in both regimes; only novelty and ε are regime-specific | as required |
| explore trajectory ends at the goal | `explore_ends_on_goal` default on | must be off here |
| agent store head OR'd into the write | `effective_store = agent_store \| ...` (`collector.py:589`) | must be masked |
| schedule stage kinds | `training/stages.py` `explore` / `exploit` / `interleave` | add `task` |
| `--env_repeats` | K rollouts of each env per update, consecutive per env | reused: visits are consecutive repeats |

## 3. Implementation

All additive; the interleave path is untouched.

### 3.1 `training/stages.py`
- `KIND_FRACTIONS["task"] = 1.0` (every slot starts without the goal).
- New stage key `visits` (int ≥ 1, default 1; only meaningful for `task`).
  `Stage.visits`, `Knobs.visits`, `resolve` copies it, `format_schedule`
  emits it. Schedule strings: `task:4000`, `task:4000,visits=2`.
- Parser rejects `visits` on a non-task kind.

### 3.2 `training/task.py` — `TaskRegime`
- `allows_store = True` (the oracle write; the head never writes).
- `new_memory(vh, env, env_offset, knobs, B)` → list of B fresh Hopfields,
  each with `n ~ U[dist_min, dist_max]` distractors from
  `sample_distractors` (own draw per trajectory, from `dist_rng`).
- `spec(...)` → `RolloutSpec(hop=<list>, allow_store=True,
  novelty_reward=knobs.novelty, goals_active=True, epsilon=knobs.eps,
  goal_in_memory_init=False, ends_on_goal=False, task_mode=True)`.
- `RolloutSpec.task_mode: bool = False` added.

### 3.3 `rollout/collector.py` — `task_mode`
New kwargs `task_mode: bool = False`, `store_fired_init: np.ndarray | None`.
When `task_mode`:
- `auto_store_active = True`; the write fires where
  `at_goal_mask & ~agent_goal_store_fired` (first touch only, per
  trajectory); `agent_store` is forced False before it reaches
  `effective_store`, the store cost / bonus paths, and the recorded
  `store_actions` (the head is frozen anyway).
- Novelty and revisit shaping multiply by `~agent_goal_store_fired`
  (pre-store only, per trajectory). Wall, persistence, time, goal unchanged.
- ε-greedy candidates are masked by `~agent_goal_store_fired`.
- `agent_goal_store_fired` is seeded from `store_fired_init` when given
  (visits ≥ 2), else all False.
- A `TaskTracker` (`rollout/task_stats.py`) records per trajectory: whether
  it started with the goal stored, the step of its first touch, coverage
  (visited-cell count) at that step, the number of touches after the first,
  and the step of the last touch. Returned as `RolloutBatch.task` (dict of
  small arrays) with a `merge()` helper for the trainer and eval.
- The existing `RegimeDiagnostics` (`cos_aq` etc.) is split into pre-store
  and post-store instances by the flag; returned as `diag` (pre) and
  `diag_post`. In the trainer's log the post-store one takes the `expt/*`
  names and the pre-store one `expl/*`, so `regime_gap` keeps its meaning
  (how much more the policy follows q once there is a goal to follow).
- `RolloutBatch.store_fired_final: np.ndarray | None` — the flag at the end,
  handed to the next visit.

### 3.4 `train_navigate.py` composer
- `stage.kind == "task"` → every slot uses `task_regime`; `n_reps % visits
  == 0` enforced at startup. Slot `v = slot − local_idx·n_reps`; when
  `v % visits == 0` a fresh memory list is built, otherwise the previous
  rollout's Hopfield list and `store_fired_final` are passed on.
- Per-update log: `train/task/*` from `merge()` (found_frac, steps_first,
  cov_first, reaches_post, steps_per_reach, and the revisit_* trio when
  visits > 1); print line shows `found=… first=… post=…`.
- Banner: `regime: task (visits=K), distractors ~U[a,b] per trajectory`.
- `--no-explore_ends_on_goal` is not needed (the task spec sets it) but the
  banner states it.

### 3.5 Evaluation
- `evaluation/metrics.py: evaluate_task(...)` — runs the collector in
  task_mode on each val env (B = `n_val_trials`, sampled policy, the
  convention for uncertain policies), for each `val_n_distractors`, visits
  = 2, and reports per d: `found_rate`, `steps_first`, `cov_first`,
  `reaches_post`, `steps_per_reach`, `revisit_found`, `revisit_steps`.
  Same code path as training by construction.
- `do_eval` gains scope `task` (= task + expl + nav, so the old numbers
  stay comparable with d0_base) and prints `[tag] task={…}`.
- Verdict instruments unchanged: `run_se_probe.sh` (held-out explore_traj /
  behavior_probe nav with follow_q), `run_cl_ood.sh` (continual protocol,
  STOCHASTIC=1), `training_curve.py`, `compare_curves.py`.

### 3.6 Launcher — `run_nav_p2.sh`
Variant family `task*`, SE optimizer (lr 1e-4 × 10 × 8, target_kl 0.1),
polar head, κ cap 2.5, goal 2.0, 200 steps, batch 64:
- `task3_<lever>`: 3 envs, fixed goals (no refresh), `ENV_REPEATS=2`.
- `task1_<lever>`: 1 env fixed goal (seed picks the arena; s43 = (11,6)).
- `task1r_<lever>`: 1 env, `--redraw_goal_per_rollout`.
- Levers: `k1` / `k2` (visits), `h128` / `h1024`. `visits=2` requires
  `ENV_REPEATS` a multiple of 2; `task3_k2` uses 2 slots per env = one
  sequence per env per update (64 × 3 × 2 = 384 trajectories/update,
  76,800 env-steps, same as `fix3`).
- `--eval_scope task`.

## 4. Metrics

Training log, every update (`train/task/*`):
- `found_frac` — trajectories (visit 1) that touched the goal.
- `steps_first` — mean step of the first touch, over found.
- `cov_first` — visited cells at the first touch (search efficiency).
- `reaches_post` — touches after the first, per found trajectory.
- `steps_per_reach` — (last touch − first touch) / reaches_post.
- `revisit_found`, `revisit_steps_first`, `revisit_steps_per_reach` — the
  same for visit ≥ 2 trajectories that arrived with the goal stored.
- `expl/cos_aq` (pre-store chase_q), `expt/cos_aq` (post-store follow_q),
  `regime_gap`.

Eval (every 50 updates, 6 held-out arenas, sampled): `task_d/*` as above
plus the old `nav` (pre-stored goal, deterministic — the revisit from a
fresh state) and `expl` (empty memory coverage).

Verdicts (checkpoint-level): held-out `follow_q` and `align_true` at d = 0 /
5 / 10 (`behavior_probe --mode nav`), explore_traj success and tail, the
continual protocol at 200 iterations per arena (revisit steps and retention).

## 5. Success criteria and what falsifies what

- **Protocol works at all**: `found_frac` → ≥ 0.95 and `steps_per_reach`
  → ≤ 14 on the *train* arenas (fix1_h128 s43 exploits at 11–12).
- **Generalises**: held-out `follow_q` ≥ 0.8 at d=0 and the continual
  protocol's revisits at ≤ 13 steps with no forgetting — i.e. matches
  `fix3_h128` / `fix1_h128 s43` from §8.
- **K=1 vs K=2**: if K=1's held-out *revisit* numbers (fresh state, goal in
  memory) are worse than its visit-1 post-store numbers, the agent learned
  a reward latch rather than the ‖q‖ gate, and K=2 is required. If they
  match, the pure protocol suffices.
- **3 arenas vs 1**: if `task3` follows q and `task1` (fixed) does not, §1's
  incentive argument holds; if `task1r` (redraw) also fails, the protocol
  itself is the problem, not memorisation.

## 6. Wave 1

| arm | envs | goal | visits | trunk | notes |
|---|---|---|---|---|---|
| task3_k1_h128 | 3 | fixed | 1 | 128 | primary, pure protocol |
| task3_k2_h128 | 3 | fixed | 2 | 128 | primary, revisits |
| task3_k1_h1024 | 3 | fixed | 1 | 1024 | trunk control |
| task3_k2_h1024 | 3 | fixed | 2 | 1024 | trunk control |
| task1_k2_h128 (s43) | 1 | fixed (11,6) | 2 | 128 | memorisation control |
| task1r_k2_h128 | 1 | redraw | 2 | 128 | protocol control |

4000 updates, eval every 50, seed 42 (task1 s43). 12 h walls on
ou_bcs_normal (cap 8 GPUs), overflow to mit_normal_gpu (6 h + continue).
Wave 2 depends on wave 1: seeds for whichever arm clears §5, then the
sample-efficiency question (fewest samples to the §5 bar) and the levers
parked in §7.

## 7. Parked

- Novelty kept on after the store (a directness-vs-coverage lever; the
  default here is off, per "same rewards as explore/exploit").
- Longer rollouts (400 steps) to give the post-store phase more room; kept
  at 200 so the numbers stay comparable with everything before.
- Teleport target distribution (uniform now; "far from goal" would make the
  post-store phase harder).
- Distractors drawn from inside the arena (the continual protocol's
  distractors are other arenas' goals, so outside-arena is the faithful
  choice).
- Memory carried across updates rather than within one (visits > env_repeats).
