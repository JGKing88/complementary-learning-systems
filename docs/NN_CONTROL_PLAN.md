# Goal-conditioned NN control: can a plain network navigate from encoded states?

Status: **plan, revision 3**, 2026-09-10. Nothing below §5 is built. Branch
`worktree-nn-generalization-control`; the config edits in §5.1 marked *done*
are the only code so far.

Revision 2 replaced the single rollout-based design of revision 1 with two
experiments that answer two different claims, and pared the primary one down
to a supervised loop over sampled pairs. §1 says why. Revision 3 gives B a
fresh goal on every goal-reach (per-row goals, §4.3), which makes B's data
unit identical to A's and removes the goal-memorisation loophole outright;
it also states the evaluation in units (§4.4) and adds the continual plot
(§4.6).

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
returns the unit vector). So the task is:

> **Can the network compute `normalize(g − p)` from `enc(p)` and `enc(g)`,
> for encodings it has not seen?**

---

## 1. Two experiments, one table

There are two claims hiding in "can a NN do this", and they need different
controls.

**The instant claim.** Given one pair `(enc(p), enc(g))` and nothing else,
emit the direction. This is what the attractor does; no trajectory, no
history. The right control is a **memoryless** network trained on
**i.i.d. sampled pairs** — Experiment **A**. A rollout adds nothing a
memoryless net can use, and it adds one thing that hurts: trajectories walk
through held-out cells, which leaks the cell-level holdout.

**The given-time claim.** Put the network in the environment and let it
move. A recurrent net sees `(Δenc, action)` pairs as it goes, which is enough
to estimate *how the code changes under movement here* — the local frame the
attractor's readout hand-codes — and to build the observation→position map
of a new wall in-context (the §5.2 mechanism, measured at +0.33). It might
succeed where A fails, **by a different mechanism**, and that mechanism is
memory, which is what the rest of this project studies. Experiment **B**
tests it — and is factorialised so that a B win can be attributed to memory
rather than to the rollout data distribution.

Both are scored on **one table** — A's static quadrant table, evaluated for B
at the state of an episode's first step (`h = 0`, `prev_action = 0`). That is
B's *instant* number. B additionally reports direction quality **against step
in episode**, which is its *eventual* number. A flat curve means B is
memoryless-equivalent and A's result stands. A rising curve is in-context
mapping — a real finding about memory, not a rebuttal of A.

A runs first: it is minutes per run, it owns every within-env holdout, and
its table decides which cells of B are informative.

---

## 2. The task, precisely (shared by A and B)

### 2.1 Environments

`GridEnv`, size `S = 20`, no obstacles, walls carrying a ±1 barcode at
`wall_resolution = 1`, `observation_size = 120` rays over a 120° cone. Envs
are placed on the `lambdas = [11, 12, 13]` scaffold (`Npos = 1716`) by the
declared-domain generator (`world/generate.py`) with `place_margin = 20`.

### 2.2 Encodings

| name | `enc(c)` for a cell `c` | width | produced by |
|---|---|---|---|
| **gbook** | smoothed grid code at the cell's *global* scaffold position (env offset + local cell) | `Ng = 434` | `smooth_gbook`; `rollout/rnn.py::grid_state_vec` |
| **omni** | all four cardinal ray-cast views at the cell, concatenated | `4 · 120 = 480` | `GridEnv.omni_obs_at` — a codebook gather |
| **xy** | `(x, y) / S` | 2 | — |

**Grid mode** is `[gbook(p), gbook(g)]`. **Regular mode** is
`[omni(p), omni(g)]`. **xy mode** is `[xy(p), xy(g)]`, the ceiling. There is
no sensory channel in grid mode — it would be a second encoding of `p`, and
one that carries env identity, so a success would be unattributable and a
held-out-env failure could be the barcode's fault rather than the code's.
There is no reward channel: with the goal as an input, arrival is the only
reward event and it carries nothing. `prev_action` is absent from A by
construction (§3) and is a **factor** in B (§4).

`omni` rather than the single egocentric view, for `p` as well as `g`: it
makes both encodings heading-free, so the task is symmetric and "which way
was it facing" is not a hidden variable. The codebase already calls this
"the heading-invariant observation". The single-view version is a realism
arm for later, and B uses omni too so that A and B differ only in history.

### 2.3 Output and teacher

Both action modes, as separate arms:

- **discrete** — 4 cardinal logits. Teacher: the **optimal set** — actions
  that reduce Manhattan distance; 1 action when `p` and `g` share an axis,
  else 2. Loss and metric both use the set, so tie-breaking never enters.
- **continuous** — a 2-vector. Teacher: the unit vector `(g − p)/‖g − p‖`.
  No ties. (B runs the env with `continuous_normalize = True` so a step is a
  unit vector and only direction is learned.)

### 2.4 Holdouts

Three, orthogonal. Two are existing traits; the third is new.

| holdout | mechanism | grid mode tests | regular mode tests | A | B |
|---|---|---|---|---|---|
| **H-env** — held-out environments | `make_val_set(levels={wall: held_out, place: held_out})` | a scaffold region whose grid codes were never seen (`place`) | a barcode never seen — the observation *generating rule*, not a table (`wall`) | ✓ | ✓ |
| **H-goal** — cells never used as a goal | `goal_cells_train` / `goal_cells_val` (exists, `goal_val_frac`) | goal-input generalization within familiar envs | same | ✓ | — |
| **H-region** — cells never a start **or** a goal | **new**: `region_cells ⊂ goal_cells_val` (`region_val_frac`) | scattered global codes never seen at all | observations never seen at all | ✓ | — |

Region cells are the same *local* cells in every training env; in grid mode
their global codes differ per env, so the held-out set is scattered across
the scaffold, and H-env `place` is the contiguous one on top. B cannot honor
cell-level holdouts (its trajectories pass through them), so H-goal and
H-region belong to A alone.

### 2.5 The table

For every env set (train envs, H-env envs, and a `same`-level set as the
memorisation probe), the **start × goal quadrant table**:

start ∈ {train, region} × goal ∈ {train, goal-heldout, region} — 6 cells.

Each cell: continuous → **mean angular error** (deg), median, fraction
< 30°; discrete → **optimal-set accuracy**. Reference lines: teacher
= 0° / 1.0; uniform random = 90° / ≈ 0.37 (the optimal set averages ~1.5 of
4). The table is the deliverable of both experiments.

---

## 3. Experiment A — memoryless, i.i.d. pairs

### 3.1 Model

`RNNAgent` with the `mlp` trunk (§5.3): `num_rnn_layers` hidden layers of
`hidden_size`, the existing discrete / continuous heads. Input is the pair
encoding from §2.2 and nothing else. There is no GRU arm: with i.i.d.
samples there is nothing to recur over, and a GRU here is an MLP with extra
parameters.

### 3.2 Data

Per training env, precompute once: `gbook` at every cell (`S² × Ng`), `omni`
at every cell (`S² × 480`), and the `CellSets` (§5.5). A training batch is,
for every train env, `pairs_per_env` draws of `p ∈ start_train`,
`g ∈ goal_train`, `p ≠ g` — an index gather, no env stepping. At 64 envs
× 512 pairs that is 32k samples per update, and an update is one forward.

No DAgger: the policy's "state" is the cell `p`, and uniform sampling covers
every state it could ever be in — strictly more than rollouts, which pile up
along straight lines to goals.

### 3.3 Loss

- discrete: cross-entropy against a **uniform distribution over the optimal
  set** (soft target; never an arbitrary tie-break).
- continuous: MSE between the head's mean and the unit vector. Cosine loss
  would match the metric exactly but has a degenerate gradient at zero mean;
  MSE reaches the same optimum.

### 3.4 Evaluation

The sampler pointed at a different cell set. During training, every
`eval_every` updates: 4096 pairs per quadrant per env set. At the end:
**full enumeration** of every `(p, g)` in every quadrant (train quadrant is
360 × 320 ≈ 115k pairs per env — one batched forward), so the final table
has no sampling variance.

Checkpoint selection is by the H-env **train × train** cell, never by the
held-out cells it is then reported on.

Secondary, on H-env envs only: roll the trained policy out (deterministic,
no teacher) for success rate and steps-to-goal, to confirm the static number
turns into behaviour. Existing env stepping, nothing new.

### 3.5 Arms

| id | mode | trunk | encoder (regular only) | action |
|---|---|---|---|---|
| **A0** | xy | mlp-2 | — | both |
| **A1** | grid | mlp-2, mlp-4 | — | both |
| **A2** | regular | mlp-4 | linear | both |
| **A3** | regular | mlp-4 | conv, xcorr | both |
| **A4** | the closest-but-failing arm | ×2 width, ×1.5 depth, `obs_size = 240`, 256 envs | | |

Encoders (§5.4) act on the ray-vector columns before the trunk. `linear` is
identity. `conv` is a siamese `Conv1d` over the ray axis, the same module on
all 8 views. `xcorr` has no parameters: the circular cross-correlation of
each current view with the goal view of the **same heading** (N with N, E
with E …), `4 × 120` lags, appended to the raw input — the quantity `conv`
would have to learn, handed over, to separate *cannot compute it* from
*cannot use it*.

### 3.6 Script

`hopfield_nav/train_goal_pairs.py`, its own composer. It builds an
`RNNTrainConfig` so that `rnn_world`, `restore_arch_from_ckpt` and
`write_rnn_world_spec` work unchanged, then: precompute per-env tensors →
loop {sample, forward, loss, step} → periodic static eval → checkpoints,
`run.json`, `world.json`, wandb (`train_goal_pairs`). Nothing from
`updates/` or `rollout/`: `bc_rnn_update` is built around
`RNNRolloutBatch`, and A's loss is one line. Launcher `run_goal_pairs.sh`.

---

## 4. Experiment B — recurrent, rollouts

### 4.1 What it tests

Whether a network **given time in the environment** can reach goals in
held-out envs that A could not reach instantly — and, if so, whether that is
because of memory or because of the rollout data.

### 4.2 Arms — the factorial that makes a B win attributable

| B arm | trunk | `prev_action` | data | isolates |
|---|---|---|---|---|
| **B-full** | GRU | on | rollouts | the full hypothesis |
| **B-rec** | GRU | off | rollouts | recurrence alone — multiple looks, nothing to integrate against |
| **B-dist** | MLP | off | rollouts | **the data distribution** — on-path states, DAgger recovery — with no memory |

B-dist is the control that matters. If B-dist ≈ A on the table, any B-full
gain over A is memory. If B-dist is already better than A, the gain was the
data, not the history, and the "history helps" story is dead before B-full
is read.

Both action modes; grid and regular. Regular uses `omni(p)` (§2.2) so the
only difference from A is history.

### 4.3 Data — lifetimes of independent episodes

The unit vocabulary, because the design lives in it:

- **episode** — one (start `p` → goal `g`) attempt. Ends at goal-reach or at
  the 60-step cap.
- **chunk** — `steps_per_rollout` (64) consecutive steps of one env, the unit
  of one BC update. Contains however many episodes fit; an imperfect policy
  in a new env may not finish one.
- **lifetime** — `resample_envs_every` chunks on **one env** with the hidden
  state carried the whole way. At its end the env is swapped and `h` zeroed.

`train_rnn.py` mixed mode with `carry_across_episodes` already gives
lifetimes. B adds one rule on top of it:

> **On goal-reach, the row gets a fresh start *and* a fresh goal. Its hidden
> state is kept.**

The goal is drawn from `goal_cells_train`, the start uniformly, `p ≠ g`.
That is one flag, `--resample_goal_on_reach`, and it makes B's data unit
identical to A's: an episode is one `(p, g)` pair in both. The *only*
difference is that B's episodes are consecutive in one env with `h` carried
across them — which is the cleanest possible statement of "B = A + history".

Why both halves of the rule matter:

- **Fresh goal.** With one goal per lifetime, a recurrent net can succeed by
  finding the goal once and remembering *where it was* — the §5.2 mechanism,
  already measured at +0.33 — without ever reading `enc(g)`. That would make a
  B success uninterpretable. A new goal every episode removes the strategy:
  the only thing that persists across episodes is the **env**, so the only
  thing worth holding in `h` is the map.
- **Keep `h`.** Zeroing the state at goal-reach (what `collect_rollout_rnn`
  does without `carry_across_episodes`) leaves one episode of history —
  ~15 steps of a good policy, one chunk of a bad one — which is not enough to
  learn anything about an env. That regime is the one in which no in-context
  effect was ever found; the effect appeared only once lifetimes existed.

So: goal-learning in-context is made impossible; map-learning in-context is
what B measures.

**The BC update** is truncated BPTT: each chunk is re-run with gradients from
its detached `initial_h` (`RNNRolloutBatch.initial_h`, whose docstring says
why: starting from zeros would cap the usable horizon at one chunk). Forward,
the network sees the whole lifetime; backward, credit is assigned within 64
steps. Map-learning is local — a few `(Δenc, action)` pairs fix a local
frame — so this should not bind, and the §5.2 result was obtained under the
same truncation. It is the same for all three B arms, so the B-full vs B-dist
comparison cancels it.

Per-row goals require `VecEnv` to hold a `(B, 2)` goal array rather than one
tuple (§5.8). Under the default the rows are identical and behaviour is
bit-for-bit today's. `omni` is the sensory channel (`--sensory_mode omni`),
for comparability with A. B does not attempt the cell-level holdouts (§2.4).

### 4.4 Evaluation — three readouts, one metric

Everything below scores the **policy's own action against the teacher**,
`normalize(g − p)`: angular error (continuous) or optimal-set membership
(discrete). The teacher is never in the loop at eval — the student acts on its
own policy throughout, which is off-teacher, unlike DAgger training. Using one
metric everywhere is what lets the three readouts be laid side by side.

**Readout 1 — instant: A's static table at `h = 0`.** B's trained weights,
every `(enc(p), enc(g))` pair, zero hidden state, zero `prev_action`, one
forward. Identical protocol, cell sets and env sets to A;
`evaluation/goal_pairs.py` is imported, not reimplemented. This is what B can
do with **no experience of the env** — its score on the attractor's own claim.

**Readout 2 — eventual: direction quality vs. experience.** The procedure,
exactly, because the reading depends on it. It is `evaluate_in_context`'s
lifetime loop (`evaluation/incontext.py:67`, the §5.2 evaluator) with per-row
goals and a per-step score.

*Setup.* One H-env env. `n_lifetimes = 64` parallel rows via `make_vec`,
each an independent lifetime — one sample of "meeting this env for the first
time". Every row draws a start uniformly and a goal from `goal_cells_train`.
`h = 0`, `prev_action = 0`. Weights frozen.

*One tick, per row:*

1. Rows at goal or at `steps_in_ep ≥ 60` close their episode: `ep_idx += 1`,
   `steps_in_ep = 0`, `reset_indices` → fresh start **and fresh goal**. `h`
   untouched. (Existing logic; the only change is the goal redraw.)
2. Build the input: `enc(p)`, `enc(g)` for *this row's* goal, `prev_action`
   if the arm has it.
3. `agent.act(x, h, deterministic=True)` → action `a`, `h_next`.
4. **Score `a` against the teacher** — continuous: `angle(a, g − p)` in
   degrees; discrete: `a ∈ optimal_set(p, g)` as 0/1. The label is computed,
   used for the score, and discarded — it never touches the action.
5. Record `(row, ep_idx, steps_in_ep, score)`, for live rows only.
6. `step_batch(a)`; `steps_in_ep += 1`; `prev_action = a`.

`n_episodes = 20` per lifetime; budget `20 × 61` ticks; rows that finish sit
out and their `h` is not advanced. Repeat over every H-env env; report the
mean and the per-env spread.

*Aggregation.* Every score has an `(episode, step)` coordinate. Keep the full
2-D table with per-cell counts (an empty bin reads as empty, not zero), and
its two marginals:

- **by episode** — `mean(score | ep_idx = k)`, `k = 0..19`. With a fresh
  goal every episode this is exactly *goals seen so far in this env*, and
  nothing else. **This is the map-learning curve, and the one that decides
  B.** Pre-registered "rising": `mean(ep ≥ 5) − mean(ep 0)` beyond the seed
  spread.
- **by step** — `mean(score | steps_in_ep = t)`, `t = 0..59`. Read to the
  cap, not to 15: an imperfect policy in a new env runs long.

*Three shapes the 2-D table separates,* which the marginals alone would blur:

| shape | meaning |
|---|---|
| flat in both axes | memoryless-equivalent; A's result stands |
| rises with step, resets at each episode | **episode-local** — uses the last few `(Δenc, action)` pairs, forgets at teleport. Real history use, but not a map. The B-rec signature. |
| rises with episode, flat-ish within | **map-learning** — knowledge of the env accumulates across goals |

*Why it is valid.* The teacher scores only; the student acts on its own
policy every step — off-teacher, unlike DAgger training. The score is dense
(every step has one), so unlike the §5.2 evaluator's `memory_lift` it needs
no conditioning on whether the previous episode succeeded. And **episode 0,
step 0 is readout 1 by construction** — same state, same forward, same
score — so that cell of the table must equal the static table's entry for the
matching quadrant. If it does not, one of the two evaluators is wrong.

*Code.* `evaluation/rnn.py::evaluate_lifetime_direction(env, agent,
n_lifetimes, n_episodes, max_steps, device, *, sgb, env_offset, goal_pool,
rng)`. ~40 lines on top of the existing 80: goal redraw in the reset, per-row
goal in the input, score-and-record after `act`, and the live-row mask on
both the score and `h` (the existing loop advances `h` for dead rows on a
stale observation; harmless there, wrong here).

**Readout 3 — behaviour.** Success rate and steps-to-goal over the same
lifetimes (`evaluate_nav_all` machinery). Confirms that direction quality
turns into reaching goals. Secondary.

Pre-registered definition of "rising" for readout 2: episode ≥ 5 score
better than episode 0 score by more than the seed spread, on H-env envs.

### 4.5 What B can and cannot conclude

| readout 1 (instant) | readout 2 (vs episode) | B-dist | reading |
|---|---|---|---|
| B-full ≈ A | flat | ≈ A | history does not help; A is the whole story |
| B-full ≈ A | **rising** | ≈ A | **in-context mapping** — a memory result; A's instant result stands |
| B-full > A | — | ≈ A | not history — the static evaluator is leaking; investigate first |
| — | — | **> A** | the rollout **data** helped; fix A's sampler before reading any B arm |

B is read only in cells where A **failed**. Where A generalizes, B cannot beat
it on the instant table (A is at the ceiling), and its only possible finding
there is "history hurts", which is minor.

### 4.6 The classic continual-learning plot, from the same machinery

The project's headline figure is success-vs-time with one block per env:
train on env 0, then env 1, …, and at every point score **every env seen so
far**, so forgetting appears as each earlier env's curve falling when a later
block starts. `train_rnn.py` sequential mode already produces it for the
no-goal RNN (`train_sequential` records `trace[(global_step, block,
{env: nav_det})]`), and `evaluate_sequential_episodes` produces the Hopfield
version.

The goal-conditioned models slot into it with no new protocol — only a change
in what the model is given:

- **A in sequential mode.** `train_goal_pairs.py --mode sequential`: envs
  introduced one per block, the pair sampler restricted to the current block's
  env, and at every eval the static table on every env seen so far. Score per
  env is the train × train cell (or success from a short rollout, to match
  the existing plot's y-axis). Since A never steps the env and the only
  per-env state is the weights, this is the pure **weight-forgetting** curve
  for a memoryless goal-conditioned net: did learning env 3's codes
  overwrite env 0's? It is a ~30-line addition to the script.
- **B in sequential mode.** `train_rnn.py --mode sequential` with the goal
  channels and `--resample_goal_on_reach`. Already exists; nothing to add.
  Its per-env score at revisit is readout 3 on that env with a fresh `h`, or
  — the more interesting variant — with `h` carried from the last time it was
  in that env, which separates *weight* forgetting from *activation*
  retention.

Three curves on the existing axes: the attractor (stores, no weight update
per env), A-sequential (weights only), B-sequential (weights + activations).
The attractor's claim is that its curve is flat because nothing was learned
per env; A's and B's show what a network that *does* learn per env pays for
it. The Hopfield stack's `evaluate_sequential_episodes` is not reused — its
protocol is built around store events — but the x-axis, y-axis and block
structure are the same, and `analysis/continual/` plots all three from the
same `trace` shape.

---

## 5. Code changes

Ordered so each step is testable alone. Paths current as of `b83ec51`.

### 5.1 Config vocabulary — `hopfield_nav/config.py`

- `RNN_CELLS = ("gru", "rnn", "mlp")`; `validate_recurrent_core` rejects
  `mlp` + `softplus`. **done**
- `GOAL_SENSORY_MODES = ("none", "omni", "north")`. **done**
- `SENSORY_MODES = ("ego", "omni")` — the current-observation channel's form.
- `SENSORY_ENCODERS = ("linear", "conv", "xcorr")`.
- `RNNAgentConfig`:
  - `input_goal_grid_state: bool = False` **done**; `goal_sensory: str = "none"` **done**
  - `input_sensory: bool = True` — today it is hard-wired on
    (`compute_rnn_input_dim`: "Sensory always on"). Grid mode turns it off.
  - `sensory_mode: str = "ego"` — `omni` makes the channel `4 · obs_size`.
  - `input_xy_state: bool = False` — `xy(p)`; pairs with `goal_channel="abs"`
    (which is already `xy(g)`) for the ceiling arm.
  - `sensory_encoder: str = "linear"`, `sensory_encoder_channels: int = 16`,
    `sensory_encoder_kernel: int = 5`.
- `RNNTrainConfig`: `region_val_frac: float = 0.0`; `pairs_per_env: int = 512`
  (A); `resample_goal_on_reach: bool = False` (B).

### 5.2 Input layout — `hopfield_nav/policy/agent_rnn.py`, `hopfield_nav/rollout/rnn.py`

The layout becomes data, as `policy/channels.py` did for the other stack,
because the ray-axis encoder has to know which columns are ray vectors and
because `sensory` is now optional and variable-width.

- `rnn_input_layout(cfg, obs_size, gbook_dim) -> list[tuple[str, int]]`.
  Order is a compatibility surface — every existing checkpoint was trained
  against `sensory, prev_action, prev_reward, grid_state, goal_vec`; the new
  channels **append**: `xy_state(2)`, `goal_grid_state(Ng)`,
  `goal_sensory(480 | 120)`. `sensory` off removes the first slot; a
  checkpoint with it on is unaffected.
- `compute_rnn_input_dim` = the sum over the layout.
- `build_rnn_input(...)` takes the new channels as keywords and appends in
  layout order. An enabled-but-missing channel **raises** — today it silently
  skips (`if cfg.input_grid_state and grid_state is not None`), which is the
  shape-preserving failure `channels.py` was written to kill.
- Producers in `rollout/rnn.py`: `gbook(g)` is `grid_state_vec(goals, …)`
  unchanged; `goal_sensory_vec(env, goals, mode)` and
  `sensory_vec(env, positions, mode)` are codebook gathers
  (`omni` → `env._codebook[x, y].reshape(B, −1)`; `north` → view
  `cardinal_index(0.0)`); `xy_vec(positions, size)` is the `abs` branch of
  `goal_channel_vec` applied to positions.
- `RNNAgent.__init__(cfg, input_dim, *, layout=None)`: keyword, default
  None → today's path.

### 5.3 Trunk — `hopfield_nav/policy/recurrent.py`

`FeedForwardCore(nn.Module)`: `num_layers` hidden layers of `hidden_size`,
activation from `rnn_nonlinearity` (tanh | relu), dropout between layers
only. `forward(x, h) -> (features (B,T,H), zeros (L,B,H))`. It honours the
four trunk contracts — `input_size`, `parameters()`, `(L,B,H)` state,
T-step ≡ T single-steps (trivially) — so B-dist runs through the rollout and
`bc_rnn_update` unchanged. `build_recurrent_core` dispatches on
`cell == "mlp"`. **Docstring and `--rnn_cell` help done; class not yet.**

### 5.4 Ray-axis encoders — `hopfield_nav/policy/sensory_encoder.py` (new)

Applied in `RNNAgent.forward` before the trunk, on the columns the layout
marks as views (`sensory` split into 1 or 4, `goal_sensory` into 1 or 4).
`linear` identity; `conv` siamese `Conv1d(1 → C, k)` on every view, flatten,
replaces the raw columns; `xcorr` parameter-free circular cross-correlation
of each current view with the same-heading goal view, appended. Built last
(§5.9): A1/A2 do not need it and A3 may not run.

### 5.5 The split — `hopfield_nav/world/spec.py`, `hopfield_nav/world/generate.py`, `hopfield_nav/training/rnn_setup.py`

- `GeneratedSplit.region_cells: frozenset = frozenset()`; JSON round-trip
  with a default so every existing `world.json` loads.
- `generate_split(..., region_frac=0.0)`: after the goal partition, draw
  `region_cells ⊂ goal_cells_val`, size `round(region_frac · S²)`, from
  `trait_rng(seed, "region")`. Invariant: **region ⊂ never-goal**.
- `CellSets` (in `spec.py`): `start_train = all − region`,
  `goal_train = goal_cells_train`, `goal_heldout = goal_cells_val − region`,
  `region`. One object read by the sampler and the evaluator, so they cannot
  disagree.
- `rnn_world` passes `region_frac=cfg.region_val_frac`. The declared path
  already returns `split.base_val` for H-env; other levels via
  `make_val_set`.

### 5.6 Pair sampler and static evaluator — `hopfield_nav/evaluation/goal_pairs.py` (new)

- `EnvTensors(env, offset, sgb, device)`: `gbook`, `omni`, `xy` for every
  cell, computed once.
- `sample_pairs(cells, starts: str, goals: str, n, rng) -> (p, g)` and
  `enumerate_pairs(cells, starts, goals)`.
- `pair_inputs(tensors, cfg, p, g) -> (B, D)` — assembles the input in
  layout order, using the same producers as §5.2 so A and B build
  bit-identical tensors for the same `(p, g)`.
- `pair_targets(p, g, movement_mode)` — unit vectors, or the `(B, 4)` optimal
  set.
- `evaluate_pairs(agent, tensors, cells, *, movement_mode, n_per_quadrant |
  enumerate, device) -> dict` — the 6-cell table. `h = 0`,
  `prev_action = 0`, which is what makes it valid for B.

Layer: `evaluation` (imports `policy`, `rollout`, `world`).

### 5.7 Experiment A's script — `hopfield_nav/train_goal_pairs.py`, `hopfield_nav/run_goal_pairs.sh` (new)

§3.6. ~150 lines including the CLI. Adds itself to
`scripts/check_entry_points.py`.

### 5.8 Experiment B's additions

**Per-row goals** — `hopfield_nav/world/env.py`, `hopfield_nav/world/vec_env.py`, `hopfield_nav/rollout/oracles.py`:

- `_at_goal_l2(pos, goal, radius)`: a `goal.ndim == 2` branch, row-wise.
  Three lines; the `(2,)` path is untouched. (`goal_arr[0]` on a `(B, 2)`
  array is the first *row*, so it does not broadcast by accident — the
  branch is needed.)
- `VecEnv` and `ContinuousVecEnv`: `_goal` becomes a `(B, 2)` array,
  initialised by tiling `base_env._goal`. `reset_all` / `reset_indices`
  exclude the *row's* goal. New `set_goals(goals, indices=None)`. When
  `resample_goal_on_reach` is set, `reset_indices` also draws a fresh goal
  for each reset row from a caller-supplied cell pool (`goal_cells_train`),
  before drawing the start. `step_batch` is goal-agnostic already — it only
  reads the goal through `at_goal` — so it does not change. Under the default
  every row holds the same goal and behaviour is bit-for-bit today's; the
  Hopfield stack never sees a difference. `best_action_batch` has no callers
  and is left alone.
- `bfs_action_batch_discrete` / `_continuous`: accept `(B, 2)` goals. The
  loops already index per row; it is an `np.atleast_2d` and a broadcast.
- `collect_rollout_rnn` and `goal_channel_vec` read `vec._goal[b]` per row.
  ~10 lines.

**Trainer and evaluators** — `hopfield_nav/train_rnn.py`, `hopfield_nav/rollout/rnn.py`, `hopfield_nav/evaluation/rnn.py`, `hopfield_nav/evaluation/incontext.py`:

- The three call sites that assemble the input (`collect_rollout_rnn`,
  `evaluate_nav_all`'s step, `evaluate_in_context`) compute and pass the new
  channels. `restore_arch_from_ckpt` restores the new agent fields.
- `--resample_goal_on_reach` (§4.3) and `--sensory_mode omni`.
- `evaluation/rnn.py::evaluate_lifetime_direction(...)` — readout 2: runs
  lifetimes on H-env envs with the training regime's chunk and lifetime
  lengths, records the per-step score of the policy's own action, returns it
  binned by step-in-episode and by episode-in-lifetime. Built on
  `evaluate_in_context`'s lifetime loop, which already exists for §5.2.
- The periodic eval calls `evaluate_pairs` (readout 1) on the same env sets
  A uses.

**Continual (§4.6)** — `hopfield_nav/train_goal_pairs.py --mode sequential`:
~30 lines — introduce envs one per block, restrict the sampler to the current
env, evaluate the table on every env seen so far, emit the same
`trace[(global_step, block, {env: score})]` shape `train_sequential` writes.
B's sequential mode exists already.

### 5.9 Order of work

1. §5.1 rest, §5.3, §5.2 — trunk and layout; unit-testable with a synthetic
   batch.
2. §5.5 — split; unit-testable.
3. §5.6, §5.7 — sampler, evaluator, A's script. Run **A0** as the integration
   test.
4. **A1, A2** run. While they run:
5. §5.8 — per-row goals first (unit-tested: default is bit-identical), then
   B's additions. Run **B** arms.
6. §5.4 — encoders, only if A2 fails on held-out walls. Run **A3**.

### 5.10 Tests — `hopfield_nav/tests/test_goal_pairs.py` (new) + existing

- Layout: widths sum to `compute_rnn_input_dim`; with every new flag at its
  default the assembled tensor is bit-identical to today's (extend the golden
  fixture, do not replace it); enabled-but-missing raises.
- `FeedForwardCore`: T-step ≡ T single-steps; state shape `(L,B,H)`.
- Split: `region ⊂ goal_cells_val`; round-trip; a `world.json` without the
  field loads with an empty region.
- Sampler: every `p ∈ starts`, `g ∈ goals`, `p ≠ g`; enumeration count is
  `|starts| · |goals| − |starts ∩ goals|`.
- `pair_inputs` equals `build_rnn_input` on the same `(p, g)` with
  `prev_action = 0` — the A/B bridge, pinned.
- Optimal set: aligned → 1 action, off-axis → 2; random-policy accuracy
  ≈ 0.37 on enumeration.
- Per-row goals: `_at_goal_l2` with `(B, 2)`; `VecEnv` with
  `resample_goal_on_reach` off reproduces today's rollout bit-for-bit
  (extend the golden fixture); with it on, every reset row's goal is in the
  pool and differs from its start; `bfs_action_batch_*` with `(B, 2)`
  equals a per-row loop over the scalar version.
- `xcorr`: recovers a known shift on a synthetic view.
- Entry-point smoke; `test_layering.py` unchanged.

### 5.11 Not changed

`policy/channels.py` and the Hopfield stack; `VecEnv.step_batch` and the
at-goal contract; `bc_rnn_update`; `train_rnn.py`'s sequential/finetune
protocols. `VecEnv`'s goal becomes an array but every row is identical under
the default, so the Hopfield stack is bit-for-bit unaffected. Every existing
checkpoint loads (layout with defaults is today's layout); every existing
`world.json` reads. The per-row-goal *collector* of revision 1 is gone — A
does not step the env — but per-row goals themselves came back for B, as the
smallest change that gives every episode its own goal.

---

## 6. Experiment plan

### 6.1 Fixed settings

| | value | why |
|---|---|---|
| `size` | 20 | project working size |
| `lambdas` | 11, 12, 13 (`Npos = 1716`, `Ng = 434`) | the working scaffold |
| `fwhm_ratio` | 0.25 | `RNNTrainConfig` default |
| `observation_size` | **120** (240 in A4) | a codebook gather either way, so it costs nothing; ~9% exact single-view twins at 60 vs ~27% at 12, lower at 120 and lower still under omni. Displacement precision rises with rays (~4 lags per unit `dx` at 60, ~15 at 240) |
| `wall_resolution` | **1** | raising it dissolves the shift structure regular mode depends on (pure-shift correlation ~0.85 at 1, ~0.38 at 8) |
| envs | 64 train; H-env 16 (`wall = held_out, place = held_out`); `same` 8 | `place_margin = 20`; 88 footprints of 20² is 1.2% of the scaffold |
| cells | `goal_val_frac = 0.2`, `region_val_frac = 0.1` | 40 region ⊂ 80 never-goal; 320 train-goal, 360 start cells |
| A batch | 64 envs × 512 pairs = 32k / update | one forward |
| A budget | 2000 updates, `hidden_size = 256`, Adam `lr = 1e-3` | minutes per run |
| B batch | `batch_envs = 64`, `steps_per_rollout = 64`, 8 envs / update | 32k labelled steps / update |
| B budget | 2000 updates; BC `lr = 1e-3`, `epochs = 4`, 4 minibatches | ~2 h at the measured ~7 s per 100k steps |
| continuous env (B) | `continuous_normalize = True`, `scale = 1.0` | unit step, direction only |
| episode cap (B, rollouts) | 60 steps, **on** | a straight line is ≤ 38; an imperfect policy in a new env needs the cap to produce episode boundaries at all |
| lifetime (B) | match §5.2's `resample_envs_every` | the one lifetime length at which an in-context effect has been measured |
| seeds | A: 2 per arm (cheap). B: 1, then 2 for any arm that is read | env draws are the variance; A's final table is enumerated |

### 6.2 Waves and kill criteria

**A0 — the pipeline.** xy, mlp-2, both actions, 300 updates, CPU fine. Must
hit ≤ 5° / ≥ 0.98 on **every** cell of every env set — coordinates carry no
env identity, so any gap is a bug. `same` ≈ H-env. *Kill*: anything else;
fix before A1.

**A1 — grid mode.** The primary question. mlp-2, mlp-4 × both actions × 2
seeds = 8 runs. *Kill*: if mlp-2 generalizes by §6.3, mlp-4 seed 2 is dropped.

**A2 — regular mode, linear.** mlp-4 × both × 2 seeds = 4 runs. Informative
either way. If held-out walls **pass**, A3 is unnecessary and that is the
stronger result.

**A3 — regular, encoders.** Only if A2 failed on held-out walls. `xcorr`
first, then `conv`. 4 runs.

**B1 / B2 — grid / regular.** The three arms of §4.2 × both actions = 6 runs
per mode, 1 seed. Run after A1/A2 have tables, read only in cells A failed.
Second seed only for an arm that is read.

**A4 — scaling.** Only for an A arm within ~2× of threshold; one factor at a
time. *Kill*: a 2× scale that moves the number < 20% is not a scale problem.

Totals: A ≈ 22 runs at minutes each; B ≤ 12 runs at ~2 h; encoders and
scaling conditional.

### 6.3 Decision rules (stated before the runs)

Applied to the A table; B is read against it by §4.5.

- **Generalizes**: H-env region × region within 10° / 0.05 of train-env
  train × train, and both at ≤ 20° / ≥ 0.90.
- **Interpolates only**: train × train and train × goal-heldout pass; any
  region row fails.
- **Memorises**: train-env train × train passes; H-env near random; `same`
  ≫ H-env.
- **Cannot represent**: A0 fails → pipeline bug, stop.

For regular mode report the measured twin rate of the actual envs
(`positional_identifiability.py`) beside the table, so the ceiling is known
and a 0.92 is read as a pass.

### 6.4 Predictions

- **P1** A0: solved everywhere in < 200 updates.
- **P2** A1, mlp-4: generalizes to H-env `place` and to region. The grid code
  is a smooth periodic function of position and the target is a smooth
  function of a difference. mlp-2 worse on region cells.
- **P3** A2: train-env table passes; H-env `wall` at or near random. A linear
  read of the ray vector cannot express a cross-correlation between two
  views (`docs/sensory_code.md`, Open); without it a new barcode is a new
  lookup table.
- **P4** A3: `xcorr` closes most of the held-out-wall gap; `conv` some of it.
- **P5** B-dist ≈ A in every cell (the sampler is not the problem).
- **P6** B-full, regular, held-out walls: readout 1 ≈ A; readout 2 rising
  by episode — in-context mapping, the §5.2 mechanism. B-rec between.
- **P7** B-full, grid, held-out place: if P2 holds, B has nothing to add. If
  P2 fails, B-full rises by episode — it estimates the local frame from
  `(Δgbook, action)`.
- **P8** Discrete and continuous rank arms identically; discrete stricter.
- **P9** Region × region is always the worst cell; the start row is worse
  than the goal column — the goal is a constant to condition on, the start
  has to be decoded.

### 6.5 What would change the conclusion

- P2 fails on region but passes on goal-heldout → the net localises by lookup
  and the attractor's instant-generalization claim stands in its strongest
  form. A real result.
- P3 passes → the warp is learnable from a linear read; amend
  `sensory_code.md`.
- P5 fails (B-dist > A) → A's sampler is missing something the rollout
  distribution has; fix A before reading any B arm.
- B-full first-step > A on the static table → the static evaluator is
  leaking; investigate before believing anything.

---

## 7. Risks and open points

- **Aliasing floor in regular mode.** Report the ceiling; never raise
  `wall_resolution`. If the floor binds, 240 rays (A4).
- **Region is local, H-env place is global.** Both reported; not averaged.
- **B's truncation.** Credit is assigned within one 64-step chunk. If the
  episode-in-lifetime curve rises only for the first chunk's worth of
  episodes and then plateaus, the truncation is binding and
  `steps_per_rollout` should go up before anything is concluded. The §5.2
  result under the same truncation says this is unlikely.
- **Continuous at-goal in B.** `goal_radius = 0.5` on the snapped cell; unit
  steps land on cells; confirm exact equality in the first B run.
- **`place_margin`.** Required explicitly by the RNN stack; 20 is generous.
  Read the split diagnostics' realised cosine margin once.
- **Not in scope.** Multi-goal memory, capacity, interference — the axes the
  attractor is built for. Supplying the goal as an input removes them by
  design. If A wins, the attractor's claim relocates to those axes; it does
  not disappear.
