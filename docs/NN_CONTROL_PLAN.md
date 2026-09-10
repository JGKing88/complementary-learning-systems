# Goal-conditioned NN control: can a plain network navigate from encoded states?

Status: **plan**, 2026-09-10. Nothing below §2 is built. Branch
`worktree-nn-generalization-control`; the config edits in §2.1 marked
*done* are the only changes so far.

---

## 0. The question

The attractor route to a goal is: store the goal's code, recall it from the
current code, read the difference off as a direction. It works on any
(start, goal) pair without ever having seen the pair, because nothing about
the pair is learned.

This experiment asks whether a network with no attractor and no memory
mechanism can do the same thing when it is simply *told* the goal: given an
encoding of where it is and an encoding of where it is going, can it learn to
move between them — and does that hold on states, goals and environments it
never trained on?

The teacher makes the question exact. `GridEnv` has no obstacles, so the
optimal action is `normalize(g − p)` (`rollout/oracles.py`: *"shortest-path =
greedy Manhattan toward the goal"*; `bfs_action_batch_continuous` literally
returns the unit vector). So the whole task is:

> **Can the network compute `normalize(g − p)` from `enc(p)` and `enc(g)`,
> for encodings it has not seen?**

Everything else is packaging around that.

---

## 1. The task, precisely

### 1.1 One episode

A row is a (start `p`, goal `g`) pair of integer cells in one `GridEnv` of
size `S`. At each step the policy receives its input (§1.2), emits a movement,
the env moves it (clipped at the arena), and the episode ends when it is at
`g`. It is then re-seeded with a fresh `(p, g)` and, for a recurrent trunk, a
zeroed hidden state — each pair is an independent episode, exactly as
`collect_rollout_rnn` treats a goal-reach today.

Supervision is behaviour cloning against the teacher with DAgger: the student
acts, the teacher labels every step. At-goal steps are masked out of the loss
(existing contract).

### 1.2 Inputs, by mode

All channels are things the agent legitimately has. Knowing what the goal
looks like is the premise of the task, not a leak. There is **no reward
channel**: with the goal as an input the only reward event is arrival, which
carries nothing.

| mode | channels | what `enc(·)` is |
|---|---|---|
| **grid** | sensory(p), prev_action, **gbook(p)**, **gbook(g)** | the smoothed grid code, width `Ng`, at the cell's *global* scaffold position (env offset + local cell). `input_grid_state` already supplies `gbook(p)`. |
| **regular** | sensory(p), prev_action, **omni(g)** | the raw ray vector. `omni(g)` is all four cardinal views at `g` (`GridEnv.omni_obs_at`), `4·obs_size`, which removes "which way was it facing" from the spec. A `north` variant (one view) is a cheaper arm. |
| **xy** (ceiling) | sensory(p), prev_action, **(x,y)(p)/S**, **(x,y)(g)/S** | clean coordinates. If this fails, nothing else is interpretable. |

`prev_action` is kept because it was specified, and because it is the only
channel that reports a wall clip. It is not needed for the task, which is
memoryless once `g` is an input; §3.4 says how static evaluation handles it.

### 1.3 Outputs

Both action modes, as separate arms:

- **discrete** — `Categorical(4)` over cardinal moves. Teacher: greedy
  Manhattan; the *optimal set* is 1 action when `p` and `g` share an axis,
  else 2.
- **continuous** — `Normal(2)`, env run with `continuous_normalize=True` so
  the step is a unit vector and only direction is learned. Teacher: the unit
  vector `(g − p)/‖g − p‖`. No ties.

### 1.4 What "never trained on" means — three orthogonal holdouts

The split machinery already expresses two of them as traits
(`world/generate.py`, `world/domains.py`). The third is new.

| holdout | mechanism | grid mode tests | regular mode tests |
|---|---|---|---|
| **H-env** — held-out environments | `make_val_set(levels={wall: held_out, place: held_out})` | a scaffold region whose grid codes were never seen (`place`) | a barcode never seen — the observation *generating rule*, not a table (`wall`) |
| **H-goal** — cells never used as a goal | `goal_cells_train` / `goal_cells_val` (exists, `goal_val_frac`) | goal-input generalization within familiar envs | same |
| **H-region** — cells never used as a start **or** a goal | **new**: `region_cells ⊂ goal_cells_val`, excluded from starts too (`region_val_frac`) | scattered global codes never seen at all | observations never seen at all |

Region cells are the same *local* cells in every training env. In grid mode
their global codes differ per env (offsets differ), so the held-out set is
scattered across the scaffold — which is what we want. `H-env` then holds out
a *contiguous* region on top of that.

Evaluation reports the full **start × goal quadrant table**:
start ∈ {train, region} × goal ∈ {train, goal-heldout, region} = 6 cells,
on training envs and on held-out envs. This is what tells a start-side failure
from a goal-side failure from a both-sides failure.

---

## 2. Code changes

Ordered so each step is testable on its own. File paths are current as of
`b83ec51`.

### 2.1 Config vocabulary — `hopfield_nav/config.py`

- `RNN_CELLS = ("gru", "rnn", "mlp")`; `validate_recurrent_core` rejects
  `mlp` + `softplus` (that value names `SoftplusRNN`, not an activation). **done**
- `GOAL_SENSORY_MODES = ("none", "omni", "north")`. **done**
- `SENSORY_ENCODERS = ("linear", "conv", "xcorr")` — the ray-axis arms (§2.4).
- `RNNAgentConfig`:
  - `input_goal_grid_state: bool = False` **done**
  - `goal_sensory: str = "none"` **done**
  - `input_xy_state: bool = False` — current `(x,y)/S`, pairs with
    `goal_channel="abs"` for the ceiling arm.
  - `sensory_encoder: str = "linear"`, `sensory_encoder_channels: int = 16`,
    `sensory_encoder_kernel: int = 5`.
- `RNNTrainConfig`: `region_val_frac: float = 0.0`; `envs_per_update: int = 0`
  (0 = all train envs each update; the goal-conditioned trainer samples a
  subset because per-row goals make each env's rollout much richer).

### 2.2 Input layout — `hopfield_nav/policy/agent_rnn.py`, `hopfield_nav/rollout/rnn.py`

The RNN stack builds its input in `build_rnn_input` and sums widths in
`compute_rnn_input_dim`, in two places that agree by convention. The ray-axis
encoder (§2.4) needs to know *which columns* are ray vectors, which forces the
layout to be data. Do what `policy/channels.py` did for the other stack:

- `rnn_input_layout(cfg, sensory_dim, gbook_dim) -> list[tuple[str, int]]`
  in `agent_rnn.py`. Order is a compatibility surface — every existing
  checkpoint's first layer was trained against
  `sensory, prev_action, prev_reward, grid_state, goal_vec`; the new channels
  **append** in this order: `xy_state(2)`, `goal_grid_state(Ng)`,
  `goal_sensory(4·obs | obs)`.
- `compute_rnn_input_dim` becomes `sum(w for _, w in rnn_input_layout(...))`.
- `build_rnn_input(..., xy_state=None, goal_grid_state=None, goal_sensory=None)`
  appends in the same order; an enabled-but-missing channel raises (today it
  silently skips — `if cfg.input_grid_state and grid_state is not None`; that
  is the shape-preserving failure `channels.py` was written to kill, and the
  new channels should not inherit it).
- Value producers, in `rollout/rnn.py` next to `grid_state_vec`:
  - `gbook(g)` is `grid_state_vec(goals, env_offset, sgb)` — same function,
    goals in place of positions. No new code.
  - `goal_sensory_vec(env, goals, mode)`: `omni` →
    `env._codebook[gx, gy].reshape(B, -1)`; `north` →
    `env._codebook[gx, gy, cardinal_index(0.0)]`. A gather, no ray-casting.
  - `xy_vec(positions, size)` — the `abs` branch of `goal_channel_vec`
    applied to positions.

### 2.3 Trunks — `hopfield_nav/policy/recurrent.py`

- `FeedForwardCore(nn.Module)`: `num_layers` hidden layers of `hidden_size`,
  activation from `rnn_nonlinearity` (tanh | relu), dropout between layers
  only. `forward(x, h) -> (features (B,T,H), zeros (num_layers,B,H))`. It
  honours the four trunk contracts (`input_size`, `parameters()`,
  `(L,B,H)` state, T-step ≡ T single-steps — trivially, there is no state) so
  the rollout, `bc_rnn_update`, `initial_h`/`final_h` plumbing and the
  evaluator run unchanged. `build_recurrent_core` dispatches on
  `cell == "mlp"`. **Docstring and `--rnn_cell` help done; class not yet.**
- Depth is `--num_rnn_layers`, reused deliberately: one flag, same meaning
  ("how many stacked layers") in both trunks.

### 2.4 Ray-axis encoders — `hopfield_nav/policy/sensory_encoder.py` (new)

Applied inside `RNNAgent.forward` before the trunk, on the columns the layout
marks as ray vectors (`sensory`, and `goal_sensory` split into its 1 or 4
views). Three arms, one class:

- `linear` — identity. Today's behaviour; the default.
- `conv` — `Conv1d(1 → C, k)` over the ray axis, **the same module applied to
  every view** (siamese; the current view and the goal views are the same kind
  of thing), then flatten, replacing the raw columns. Adds the shift-sharing
  inductive bias and nothing else.
- `xcorr` — **no parameters**. The circular cross-correlation of the current
  ray vector with each goal view, `obs_size` lags each, appended to the raw
  input. Hands the trunk the quantity `conv` would have to learn, so it
  separates *cannot compute the correlation* from *cannot use it*. It is a
  fixed function of two observations the agent already has — not an oracle.

Ray-vector width is `observation_size`; `RNNAgent.__init__` receives the
layout rather than a bare `input_dim` (a keyword, default None → old path,
so nothing that constructs an `RNNAgent` today changes).

### 2.5 Per-row goals — `hopfield_nav/world/env.py`, `hopfield_nav/world/vec_env.py`, `hopfield_nav/rollout/oracles.py`

`VecEnv` shares one goal across its `B` rows. Per-row goals are what make a
memoryless policy's data rich — `B` goals per rollout instead of one — and
they are the natural unit for the quadrant table. Rather than refactor the
shared `VecEnv` (the Hopfield stack sits on it), the goal-conditioned
collector **owns its goals** and drives the env with `goals_active=False`,
so `step_batch` never consumes an at-goal step or teleports. Three small
supporting changes:

- `env._at_goal_l2(pos, goal, radius)`: accept `goal` of shape `(B,2)`
  (`ndim == 2` → row-wise). `(2,)` path untouched.
- `VecEnv.set_positions(positions, indices=None)` and the same on
  `ContinuousVecEnv`: with `indices`, only those rows' positions and headings
  are touched. Today both reset **every** row's heading to North, which would
  silently change the observation of every row that was not re-seeded.
  `indices=None` keeps today's behaviour exactly.
- `oracles.py`: `unit_vector_batch(positions, goals)`,
  `greedy_manhattan_batch(positions, goals, size, rng)` and
  `optimal_action_set_batch(positions, goals, size) -> (B,4) bool`. The
  scalar-goal functions stay for `train_rnn`.

### 2.6 The collector — `hopfield_nav/rollout/goal_conditioned.py` (new)

`collect_goal_conditioned(vec, env, agent, cells, *, sgb, env_offset, steps,
device, movement_mode, rng, h0=None) -> RNNRolloutBatch`

- `cells: CellSets` (§2.7) supplies `start_train` and `goal_train`; each row
  draws `p ∈ start_train`, `g ∈ goal_train`, `p ≠ g`.
- Per step: `obs_batch`, `positions` → teacher per row → assemble input
  (§2.2) → `agent.act` → `step_batch`. Rows whose post-step position is at
  their goal are re-seeded via `set_positions(..., indices=rows)`, their
  hidden state zeroed, and the at-goal step's label masked (existing
  `move_label_mask` contract).
- Returns the same `RNNRolloutBatch`, so `updates/bc_rnn.py::bc_rnn_update`
  is reused untouched. Layer: `rollout` (imports `policy`, `world`); never
  imports `updates` or `training`.

### 2.7 The split — `hopfield_nav/world/spec.py`, `hopfield_nav/world/generate.py`, `hopfield_nav/training/rnn_setup.py`

- `GeneratedSplit.region_cells: frozenset = frozenset()`. JSON round-trip
  with a default, so every existing `world.json` still loads.
- `generate_split(..., region_frac=0.0)`: after the goal partition, draw
  `region_cells ⊂ goal_cells_val` of size `round(region_frac · S²)` from
  `trait_rng(seed, "region")`. Invariant: **region ⊂ never-goal**, so H-region
  is strictly stronger than H-goal.
- `CellSets` (in `spec.py`): derived from a split —
  `start_train = all − region`, `goal_train = goal_cells_train`,
  `goal_heldout = goal_cells_val − region`, `region`. One object the collector
  and the evaluator both read, so they cannot disagree about which cells are
  which.
- `rnn_world` passes `region_frac=cfg.region_val_frac`. The declared path
  already returns `split.base_val` for held-out envs; H-env sets at other
  levels come from `make_val_set`.

### 2.8 The static evaluator — `hopfield_nav/evaluation/goal_pairs.py` (new)

`evaluate_pairs(agent, env, cells, *, sgb, env_offset, n_pairs, device,
movement_mode, rng) -> dict`

For each of the 6 quadrants: sample `n_pairs` `(p, g)`, build the input at
**episode-first-step state** (`prev_action = 0`, `h = 0`, heading North —
exactly the state every training episode starts in, so it is in-distribution
by construction), one deterministic forward, compare to the teacher:

- continuous → angular error in degrees (mean, median, fraction < 30°)
- discrete → fraction of chosen actions in the optimal set

Optionally a second pass with `prev_action = teacher(p, g)` — the
"arrived along the line" proxy — to check the first-step number is not an
artefact of the zero prev-action.

`evaluate_goal_rollouts(...)` reuses the collector with a deterministic
student and no re-seeding to report success rate and steps-to-goal on
held-out envs. Secondary: it confirms the static number turns into
behaviour, and it is the only place a GRU's use of history can show up.

### 2.9 CLI and launcher — `hopfield_nav/train_goal_nav.py`, `hopfield_nav/run_goal_nav.sh` (new)

A thin composer, the pattern `train_navigate.py` follows. Builds an
`RNNTrainConfig`, calls `rnn_world` (with `--env_generator`,
`--place_margin`), builds `sgb` when grid state is on, `RNNAgent`, Adam;
each update samples `envs_per_update` train envs, collects one
goal-conditioned rollout per env, runs `bc_rnn_update`; every `eval_every`
runs `evaluate_pairs` on train envs and on every held-out set; logs to wandb
(`train_goal_nav` project); writes checkpoints, `run.json` via
`run_manifest`, `world.json` via `write_rnn_world_spec`. Not a fourth mode of
`train_rnn.py`: that file's modes are the continual-learning protocol, and a
goal-conditioned mode would branch it in four places.

`run_goal_nav.sh`: `VARIANT=<name> sbatch`, the `run_nav_tri.sh` pattern;
`mit_normal_gpu`, 1 GPU, 4 CPU, 3 h.

### 2.10 Tests — `hopfield_nav/tests/test_goal_nav.py` (new) + existing suites

- Layout: widths sum to `compute_rnn_input_dim`; with all new channels off
  the assembled tensor is bit-identical to today's (extend the golden
  observation fixture rather than replace it).
- `FeedForwardCore`: T-step ≡ T single-steps; state shape `(L,B,H)`.
- `set_positions(indices=)`: untouched rows keep position **and heading**.
- `_at_goal_l2` with `(B,2)` goals.
- Collector: every start ∉ region, every goal ∈ `goal_train`, re-seed zeroes
  `h`, mask is 0 at at-goal steps, `episodes_completed` counts.
- Split: `region_cells ⊂ goal_cells_val`; `world.json` round-trip; a
  `world.json` without the field loads with an empty region.
- `optimal_action_set_batch`: aligned → 1 action, off-axis → 2.
- `xcorr`: recovers a known shift on a synthetic ray vector.
- Entry-point smoke test: add `train_goal_nav` to
  `scripts/check_entry_points.py`.
- `test_layering.py` must pass unchanged: new modules sit in `rollout`,
  `evaluation`, `policy`, `world`.

### 2.11 Not changed

`policy/channels.py` and the Hopfield stack; `collect_rollout_rnn`;
`evaluate_nav_all`; `train_rnn.py`'s three modes; `VecEnv.step_batch`
semantics; the scalar-goal oracles. Every existing checkpoint loads, every
existing `world.json` reads.

### 2.12 Order of work

1. §2.1 rest, §2.3, §2.2 — trunk + channels, testable with a synthetic batch.
2. §2.5, §2.7 — env/split support, unit-tested in isolation.
3. §2.6, §2.8 — collector and evaluator.
4. §2.9 — CLI; run W0 (§3.5) on CPU as the integration test.
5. §2.4 — encoders. Deferred to last because W1/W2 don't need them and W3
   may not run.

---

## 3. Experiment plan

### 3.1 Fixed settings

| | value | why |
|---|---|---|
| `size` | 20 | project working size |
| `lambdas` | 11, 12, 13 (`Npos = 1716`) | the working scaffold; every launcher uses it |
| `fwhm_ratio` | 0.25 | `RNNTrainConfig` default |
| `observation_size` | **60** (120 as a scaling arm) | ~9% exact twins at 60 vs ~27% at 12 (`docs/sensory_code.md`); precision is ~4 lags per unit `dx` at 60, ~15 at 240 |
| `wall_resolution` | **1** | raising it dissolves the shift structure regular mode depends on: pure-shift correlation ~0.85 at 1, ~0.38 at 8 |
| `egocentric_heading` | True (default) | first-step heading is North after `set_positions` |
| continuous | `continuous_normalize=True`, `continuous_scale=1.0` | unit step, direction only — the teacher's unit vector is the exact target |
| discrete | `speed=1` | |
| episode cap | `3·size = 60` steps | a straight line is ≤ 38 |
| envs | 64 train, 16 held-out (`wall=held_out, place=held_out`), 8 `same` (memorisation probe) | `place_margin=20`; 88 footprints of 20² on 1716² is 1.2% of the scaffold |
| cells | `goal_val_frac=0.2`, `region_val_frac=0.1` | 40 region cells ⊂ 80 never-goal cells; 320 train-goal cells, 360 start cells |
| batch | `batch_envs=64`, `steps_per_rollout=64`, `envs_per_update=8` | 32k labelled steps/update, ~500 episodes/update |
| trunk | `hidden_size=256` | |
| BC | `lr=1e-3`, `epochs=4`, `n_minibatches=4` | the DAgger consensus from the BC line |
| budget | 2000 updates | ~1.5–2 h at the measured ~7 s per 100k steps |
| seeds | 2 per headline arm | env draws are the variance, not the metric — static eval over 4096 pairs is tight |

### 3.2 Arms

Factors: input mode × trunk × action × (regular only) encoder.

| id | mode | trunk | encoder | action |
|---|---|---|---|---|
| **W0** | xy | mlp-2 | — | both |
| **W1** | grid | mlp-2, mlp-4, gru | — | both |
| **W2** | regular | mlp-4, gru | linear | both |
| **W3** | regular | mlp-4 | conv, xcorr | both |
| **W4** | the closest-but-failing arm | ×2 width, ×1.5 depth, `obs_size=120`, 256 envs | | |

`mlp-k` = `--rnn_cell mlp --num_rnn_layers k`. GRU is `num_rnn_layers=1`.
W0 = 2 runs, W1 = 12 (×2 seeds), W2 = 8, W3 = 4, W4 ≤ 4. ≈ 30 runs, ≈ 60 GPU-h.

### 3.3 Metrics and reference lines

Primary, per arm, per checkpoint: the 6-cell quadrant table on train envs
and on held-out envs, for

- continuous: **mean angular error** (deg); also median and frac < 30°.
- discrete: **optimal-set accuracy**.

Reference lines: teacher = 0° / 1.0. Uniform random = 90° / ≈0.37 (the
optimal set averages ~1.5 of 4). Always plotted.

Secondary: held-out-env rollout success rate and mean steps-to-goal, and the
`same`-env quadrant table (the memorisation probe — if `same` ≫ held-out,
the network learned the pool, not the task).

A checkpoint is picked by held-out-env **train×train** quadrant, never by the
held-out quadrants it is then reported on.

### 3.4 Decision rules

Stated before the runs so the result is read by the rule, not the other way.

- **Generalizes**: held-out-env, region×region quadrant within 10° (cont.) /
  0.05 (disc.) of the train-env train×train quadrant, and both are at
  ≤ 20° / ≥ 0.90.
- **Interpolates only**: train×train and train×goal-heldout pass; any
  region row fails. The net localises seen cells and does not extend.
- **Memorises**: train-env train×train passes; held-out envs near random.
- **Cannot represent**: train-env train×train fails to reach ≤ 20° / ≥ 0.90
  by 2000 updates on the xy arm → pipeline bug, stop.

For regular mode the ceiling is below 1.0 because of exact twins (~9% of
cells at `obs_size=60`); report the measured twin rate of the actual envs
(`positional_identifiability.py`) beside the table so a 0.92 is read as a
pass.

### 3.5 Waves, with kill criteria

**W0 — the pipeline works.** xy mode, mlp-2, both actions, 1 seed, 300
updates, CPU is fine. Must hit ≤ 5° / ≥ 0.98 on **every** quadrant, train
and held-out envs alike — coordinates carry no env identity, so any gap here
is a bug. Also confirms `same` ≈ held-out. *Kill*: anything else; fix before
W1.

**W1 — grid mode.** The primary question. 3 trunks × 2 actions × 2 seeds.
Read: (a) does mlp-4 generalize by §3.4; (b) is gru better than mlp on the
static table (it should not be — the task is memoryless; if it is, the
static eval is leaking history somewhere); (c) is gru better on the
*rollout* numbers (it may be — that gap is "uses history", and is the one
place recurrence can legitimately show). *Kill*: if mlp-2 already
generalizes, drop mlp-4/gru seeds 2.

**W2 — regular mode, linear encoder.** Pre-registered expectation: train-env
table passes, held-out-**wall** table fails for the MLP, gru static ≈ mlp
static. If it *passes* held-out walls, W3 is unnecessary and that is the
stronger result — record it and skip. *Kill*: none; this wave is informative
either way.

**W3 — regular mode, ray-axis encoders.** Only if W2 failed on held-out walls.
`xcorr` first: if it passes and `conv` does not, the failure was "cannot
compute the correlation"; if neither passes, the failure is downstream of the
code. *Kill*: skip entirely if W2 passed.

**W4 — scaling.** Only for an arm that is within ~2× of the threshold. Scale
one thing at a time (width, depth, rays, envs). *Kill*: if a 2× scale moves
the number < 20%, it is not a scale problem.

### 3.6 Predictions

Written down now so they can be wrong.

- **P1** xy: solved everywhere in < 200 updates.
- **P2** grid, mlp-4: solved on train envs; **generalizes** to held-out place
  and region by §3.4. The grid code is a smooth periodic function of
  position and the target is a smooth function of a difference — this is the
  case a plain network should get. mlp-2 will be noticeably worse on region
  quadrants. gru ≈ mlp on the static table.
- **P3** regular, linear: train-env table passes; held-out walls at or near
  random for the MLP. The single linear read of the ray vector cannot express
  a cross-correlation between two views (`docs/sensory_code.md`, Open), and
  without it a new barcode is a new lookup table.
- **P4** regular, xcorr: closes most of the held-out-wall gap. conv: closes
  some of it, more slowly.
- **P5** discrete and continuous rank the arms the same way; discrete is
  stricter in absolute terms.
- **P6** the both-sides quadrant (region × region) is always the worst cell,
  and the start-side row is worse than the goal-side column — the goal is a
  constant the trunk can condition on; the start is what has to be
  *decoded*.

### 3.7 What would change the conclusion

- If P2 fails on region but passes on goal-heldout, the net localises by
  lookup and the "instant generalization" claim for the attractor stands in
  its strongest form. That is a real result, not a failed experiment.
- If P3 *passes*, the warp structure is learnable from a linear read after
  all, and the `sensory_code.md` "Open" note should be amended.
- If gru beats mlp on the **static** table, the static evaluator is wrong —
  investigate before believing anything else.

---

## 4. Risks and open points

- **Aliasing floor in regular mode.** ~9% of cells have an exact twin at
  `obs_size=60`. Fixed by reporting the ceiling, not by raising
  `wall_resolution` (see §3.1). If the floor turns out to bind, go to 120 rays
  (W4), never to resolution.
- **Region cells are local, not global.** In grid mode the H-region holdout is
  scattered across the scaffold; H-env `place=held_out` is the contiguous
  one. Both are reported; they answer different questions and should not be
  averaged.
- **prev_action in the static eval.** First-step state is in-distribution by
  construction; the "arrived along the line" pass is the check that the
  first-step number is representative. If the two disagree by more than the
  seed spread, report both and say so.
- **Continuous at-goal.** `goal_radius=0.5` on the snapped cell; with unit
  normalised steps the agent lands on cells, so at-goal is exact equality in
  practice. Confirm in W0.
- **`place_margin`.** The RNN stack requires it explicitly. 20 is generous;
  the split diagnostics report the realised cosine margin — read it once.
- **Not in scope.** Multi-goal memory, capacity, interference — the axes the
  attractor is actually built for. Supplying the goal as an input removes
  them by design. If the MLP wins here, the attractor's claim relocates to
  those axes; it does not disappear.
