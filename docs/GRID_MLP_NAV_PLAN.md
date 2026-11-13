# Grid-MLP navigator — a competitor model to Agent-HaSH (plan)

Started 2026-09-28. Branch `worktree-sensory-key-retrieval`.

## 0. Question

Agent-HaSH navigates by storing the goal in a Hopfield memory over a learned
grid-code embedding and *learning* to follow the recalled displacement. The
grid MLP (NN-control line, `docs/NN_CONTROL_PLAN.md`) takes two grid codes and
outputs the direction between them, and is trained the same way the encoder
is (random walks + odometry, no goals). **If a navigator built on the grid MLP
matches Agent-HaSH under the same continual protocol, it is a competitor
model.**

Four models are planned. Idea 1 (§1) uses a nonparametric key-value store,
and idea 2 (§6) a meta-learned network whose weights stay plastic at
evaluation. Both share one controller with Agent-HaSH's arrangement, so they
and Agent-HaSH differ only in what fills the memory-readout input. Idea 3 (§7)
has no memory module: a GRU holds the goals in its hidden state. Idea 4 (§7b)
keeps ideas 1 and 2's arrangement with a recurrent network as the memory
module, so its goals live in activity, as the control for why plastic weights
help. It holds one goal, the newest, because a recurrent network trained by
gradient descent does not learn to bind a goal to a view, which an
outer-product write does in one step (§7b.5).

## 1. Model (idea 1)

A key-value goal memory keyed on *sensory* input, whose value is the goal's
*grid state*. The frozen grid MLP turns (current grid state, recalled goal
grid state) into a direction, and that direction is an **input** to the
controller, which learns whether to follow it — the same arrangement as
Agent-HaSH, where the recalled displacement `q` is a policy input. There is no
explicit explore / exploit gate.

| Part | What it is |
|---|---|
| `g_t` | `gbook` at the agent's true scaffold position (`lambdas 11/12/13`, `Npos 1716`, `Ng 434`) — the same scaffold Agent-HaSH runs on |
| Grid MLP | frozen Phase-1 decode (random walks + odometry, displacement-balanced pairs; 0.5° held-out) — `MLP(g_t, g_goal)` → unit direction. Input `[gbook(p), gbook(g)]` |
| Memory | ONE memory shared by every env (as in `analysis/continual/agenthash.py`), one goal per env |
| Write | oracle store on first arrival at the goal: key = four-heading view at the goal, re-indexed by absolute direction (180 two-degree slices); value = `g(goal)` |
| Read, every step | query = the agent's *live* single view plus its absolute heading, compared to each key on the slices the view covers; argmax cosine; returns `(g_goal, s)` with `s` the top similarity; empty memory → `s = 0` |
| Memory readout to the controller | `d = MLP(g_t, g_goal)` (unit; zeros when memory is empty) and **`c = s`, always as its own input**. Flag `scale_q_by_c` (default **off**) feeds `c · d` in place of `d` |
| Controller | RNN, hidden 128, **not plastic at evaluation** |
| RNN inputs | live view, `d`, `c`, previous action, previous reward |
| Heads | as d0_base / `task3r_k2_h128`: direction (von Mises), step length (Beta on [0.5, 1.0]) |

**Following is learned.** The controller must learn to follow `d` when `c`
says the recall is the own env's goal (≈ 0.5, §2) and to ignore it when memory
is empty (`c = 0`) or holds only another env's goal (`c` ≈ 0.2–0.3).

**`scale_q_by_c`.** Agent-HaSH feeds raw `q` (`input_hopfield_raw`), whose
magnitude is its gate, so a small readout is a weak pull even to a linear
readout. With the flag off (default) `d` is unit and the controller has to
learn the product "follow `d` only when `c` is high" itself; `c` stays
interpretable, and idea 1's `c` (a similarity) and idea 2's (a probability)
need not share a scale. The flag is the Agent-HaSH-parity variant, to try if
following lags.

**Gradient into the memory.** The direction is always a sampled action
conditioned on `d`, so PPO's gradient reaches whatever produced `d`. Idea 1's
memory has no parameters, so this matters only for idea 2 (§6).

## 2. Why the memory works: offline evidence

All offline, every cell of every env, d0_base env config (size 20,
`observation_size` 60, `wall_resolution` 4). Code `analysis/sensory_key/`,
results `/orcd/pool/003/jackking/cls_runs/sensory_key/`.

**Question 1:** with every env's goal stored, does argmax return the querying
env's own goal? **Question 2:** does the top score separate "my env's goal is
stored" from "it is not" (AUC)?

### 2.1 Current walls: at chance

Four-heading view as key and query (`retrieval_test.py`): accuracy = 1/N
exactly (0.165 / 0.109 / 0.041 / 0.015 at N = 6 / 10 / 30 / 100), AUC
0.50–0.54. The view has no spatial smoothness: neighbour-cell cosine 0.056
against a random-vector sd of 0.065. The key matches only *at* the goal.
Agent-HaSH does not have this problem because its encoder embedding is trained
to have a smooth spatial kernel.

### 2.2 No near-wall texture fixes it

`sweep_walls.py`. Smoothed barcodes (correlation length 0.5–16 cells,
continuous or ±1), per-wall constants, constant + barcode, and multi-frequency
sums (constant + 8 / 4 / 2-cell + fine, six weightings). Best: each wall one
random constant — N=6 0.79, N=30 0.46, far from the goal 0.19 — and those
views make 98% of cells near-duplicates, so position is lost. The obstacle is
geometric: from two distant cells the same ray lands on wall points up to ~20
cells apart, so only a component longer than the room survives, ~4 numbers
per env.

### 2.3 A distal panorama does

A per-env skyline at infinity: 180 random ±1 values, one per 2° slice of
*absolute* direction, read by every ray at its absolute angle and **summed**
onto the near-wall value. Identical from every cell; near walls still carry
position (0% near-duplicate cells).

| Panorama weight | N=6 | N=30 | N=100 | own vs best-foreign sim (N=100) |
|---|---|---|---|---|
| 0 | 0.17 | 0.04 | 0.015 | 0.00 vs 0.21 |
| 0.5 | 0.90 | 0.67 | 0.49 | 0.20 vs 0.21 |
| **1** | **1.00** | **1.00** | **1.00** | **0.50 vs 0.20** |

(four-heading key and query, AUC equal to accuracy to two places at weight 1.)

### 2.4 With a single live view

`single_view_test.py`, weight 1. The agent only ever sees one view.

| Key (stored at goal) | Query (every step) | N=6 | N=30 | N=100 | own vs foreign (N=100) |
|---|---|---|---|---|---|
| single view, random heading | single view, random heading | 0.19 | 0.04 | 0.016 | 0.00 vs 0.32 |
| single view | single view, same heading | 0.99 | 0.98 | 0.97 | 0.50 vs 0.32 |
| **four-heading view** | **single live view, on the slices it covers** | **0.996** | **0.985** | **0.961** | **0.50 vs 0.32** |

The last row is the design: only the store is idealised (a look-around at
the goal); every query is the live view. It needs the agent's absolute
heading, which it has from its own last (world-frame) move.

## 3. Environment

`EnvConfig.distal_amp` / `--distal_amp` (commit `794f429`): 0 is off and
bit-identical; the panorama has its own seed stream so walls and goals are
unchanged; applied inside `raycast_codes`, so the codebook, live casts and
both vec envs see it. Tests `hopfield_nav/tests/test_distal_panorama.py`.

**Both models run at `--distal_amp 1.0`.** The panorama also hands a single
egocentric view a compass, which changes the task — the comparison is only
fair on identical envs.

Movement is unchanged from d0_base / `task3r_k2_h128`: continuous, polar head,
step length learned in [0.5, 1.0] cells (`min/max_action_norm`).

## 4. Training

**Protocol: `task3r_k2_h128`** — the Agent-HaSH recipe that worked best
(`docs/EXPERIMENTS_TASK_FAITHFUL.md`; CL 3000/3000, zero forgetting). Search →
store once → continue; novelty off after the store; 3 envs, goals redrawn;
memory kept for K = 2 rollouts; hidden 128. **Oracle store for now** — no
store head. No goal-in-memory input (task-faithful rule).

**Foreign goals in training memory.** Each rollout's memory is pre-filled with
goals from other envs. Without them the controller never sees a foreign recall
with `c` ≈ 0.3, which is the case it has to learn to ignore.

**Later variant:** the d0_base recipe, for both models.

## 5. Evaluation

- Continual protocol as `analysis/continual/agenthash.py`: one memory, envs
  added in sequence, every earlier env revisited; sampled policy
  (`--stochastic_policy`).
- Metrics: nav_det, disc, expl; retention / forgetting.
- **Baseline:** Agent-HaSH retrained as `task3r_k2_h128` at `--distal_amp 1.0`
  (the existing checkpoints saw no panorama).
- Diagnostics: alignment of the taken direction with `d`, binned by `c`
  (the learned gate); steps that follow `d` with only a foreign goal in
  memory; the MLP's angular error along followed paths.

## 6. Idea 2: a memory written by gradient steps

Idea 2 keeps everything from idea 1 except the memory. Idea 1 stores each goal
as a row in a table and looks it up by similarity. Idea 2 stores goals in the
weights of a small network, `M`. Writing a goal means taking a few gradient
steps on `M`. Reading means running `M` on the current view.

The question is whether a memory written this way can keep old goals as well
as a table does. A table never forgets, because each env gets its own row. A
network that learns env after env usually forgets, because the gradient steps
for a new env also change the weights that held the earlier goals. Idea 2
tests whether meta-learning can prevent that.

None of idea 2 is built yet.

### 6.1 How the memory works

`M` is a small MLP. Its input is the same as idea 1's query, the live view
with the panorama and the agent's absolute heading. It has two outputs.

- `ĝ` is its guess of this env's goal grid state. `M` outputs the phase of
  each of the 3 grid modules as a cos, sin pair, 12 numbers in all, and these
  are turned into a grid code. That way the output is always a valid grid
  state.
- `c` is its confidence that it holds a goal for this env.

Reading happens every step. The controller receives `d = MLP(g_t, ĝ)` and `c`,
exactly as in idea 1. The controller never changes during evaluation. Only `M`
does.

Writing happens each time the agent reaches the goal, on every visit to an
env. At that moment the agent's own grid state is the goal's grid state, so it
has the right answer without a teacher. `M` takes a few gradient steps so that
the views the agent saw on this visit map to that grid state, with `c` pushed
to 1. This stores the same thing Agent-HaSH and idea 1 store, in weights
instead of a table.

Writing on every visit, not just the first, matters because one visit sees
little of the arena. In the task evals the agent has seen 15 to 20% of the
cells (`cov_first` 0.15 to 0.18) when it first finds the goal, yet `M` must
recognise the env from any cell. Each later visit adds views from other parts
of the arena. Idea 1 never had this problem, because its key is one view
stored at the goal and compared by direction.

### 6.2 Why it needs meta-learning

If `M` starts from random weights and takes ordinary gradient steps, writing
env 5 damages what it stored for envs 1 to 4. Meta-learning sets two things in
advance so that this happens less.

- θ₀, the weights `M` starts every lifetime from.
- α, the learning rate a write uses. It can be one number, or one per weight.

Every other change to `M` happens during the writes themselves. This is the
default design because it has the fewest parts.

### 6.3 Training

Training repeats one unit, a lifetime. A lifetime brings in T new envs, T
about 3 to 10, and it has the structure of the continual protocol
(`analysis/continual/agenthash.py`). After each new env, the agent revisits
every env it has already learned. A lifetime runs the same way in training and
in evaluation.

1. Reset `M` to θ₀.
2. Bring in the next new env. The agent searches it. `M` has never been
   written for this env, so it should report `c` near 0.
3. Each time the agent reaches the goal, `M` takes k gradient steps with
   learning rate α. That is a write. k stays small, 1 to 3, because a learned
   per-weight α lets one step do the work of several, and each step adds to
   the cost of the outer loop.
4. The agent then revisits every env from earlier in the lifetime, one visit
   each, writing again whenever it reaches a goal.
5. Repeat from step 2 until all T envs are in.

The revisits in step 4 are what test idea 2. A visit right after a write, as
in the `task3r` protocol's consecutive visits, repeats the easy case. A visit
to env 1 after envs 2 to 5 have been written shows whether those writes erased
it. The revisits also give `M` its extra views of each arena.

At the end of the lifetime, an outer loop scores `M`. For every env in the
lifetime, it checks whether `M` still returns that env's goal from views the
writes did not use. It also checks that `c` is high for envs `M` wrote and low
for envs it did not. The final weights are θ₀ changed by every write in the
lifetime, and each write is differentiable. So the error can be backpropagated
through all the writes to θ₀ and α, and an Adam step updates both. For a small
MLP, a lifetime of up to about 10 envs and a few steps per write, the exact
second-order gradient is affordable. A first-order approximation is the
fallback.

This design has two costs. More of each lifetime is spent revisiting envs
whose goal is already stored, which shifts training toward exploit; in §10.6
the variants that exploit best already search worst. And every write is one
more step the outer loop backpropagates through, so a lifetime with many
writes may need first-order gradients or the `R` + `H` split of §6.5, where
backpropagating through writes to a linear head is cheap.

The controller trains with PPO on the same lifetimes, using the
`task3r_k2_h128` recipe's PPO settings with the oracle store. PPO's gradient does not reach
`M`. The rollout buffer records `d` and `c` as fixed inputs, so PPO never has
to recompute `M` while its weights change within a lifetime.

The steps before each store also train the controller on the case "my goal is
not stored". Idea 1 needed memory pre-filled with other envs' goals to create
that case. Here it arises by itself, once a lifetime is long enough that `M`
already holds some other envs' goals.

### 6.4 Order of work

1. Train `M` alone first, with no controller and no RL, on random-walk
   lifetimes with the true goals as labels. Then measure retention. After
   writing N envs in sequence, for N = 6, 30 and 100, how often does `M` still
   return each env's goal? Idea 1's table scores 0.96 to 1.00 on the same test
   (§2.4). If `M` falls far short, stop here.
2. Only if it passes, train the controller with PPO together with `M`.

Result (§10.9). Step 1 fails. Written once per env, the best `M` keeps 0.26 of
goals at N = 30 and 0.11 at N = 100, so step 2 was not started.

### 6.5 Variants for later

- Write with PPO instead of a supervised step. This is MAML-RL (Finn et al.
  2017). The memory would then be written from reward alone, which is a
  stronger claim. It isn't the default for three reasons. A PPO gradient from
  one episode carries little information about where the goal is, and MAML-RL
  needed about 20 rollouts per write even on simple tasks. The exact label is
  free at the store. And meta-gradients through policy-gradient steps have
  high variance.
- Split `M` into a fixed part and a plastic part. This is OML (Javed and White
  2019). A representation network `R` is meta-learned and never changes at
  evaluation. Only a small linear head `H` on top of it gets written. If `R`
  gives each env its own nearly separate features, which the panorama makes
  easy, writing one env barely touches the others. The other standard design
  is ANML (Beaulieu et al. 2020), where a meta-learned mask decides which
  weights each write may change.

### 6.6 Open choices

The size of `M`, one learning rate or one per weight, k within 1 to 3, the
range of lifetime lengths T in training, how many earlier envs to revisit when
T is large (all of them, or a random subset), and exact versus first-order
gradients.

## 7. Idea 3: a plain GRU, with the goal held in its state

Idea 3 is the most naive model. It has no memory module. A GRU gets the same
inputs as idea 1 minus the memory readout, and has to hold each goal in its
hidden state. The question is whether a memory module is needed at all.

It is also the control the continual-learning comparison is missing. The
existing RNN baseline (`train_rnn.py`) learns envs one at a time in its
weights, which is the forgetting setup the continual-learning controls test.
Nothing in the repo yet tests goals held in activations across the protocol.
If a plain GRU manages it, it is a serious competitor. If it fails in a
characteristic way, that failure is the argument for a memory module.

None of idea 3 is built yet.

### 7.1 Model

- The controller is a GRU with the same polar action head as the other models.
- Its inputs are the live view (with the panorama, `distal_amp` 1), the grid
  code `g_t` (the smoothed grid code at the agent's position, 434 numbers),
  the previous action and the previous reward. There is no memory readout
  and no goal-in-memory bit.
- It gets `g_t` so that the only difference from idea 1 is where the goal
  lives. Without it, the GRU would also have to learn path integration, which
  makes stage 1 harder and the result harder to interpret.
- The hidden state is never reset within a lifetime. In stage 1 it carries
  across the K visits to one env; in stage 2 it carries across the whole
  continual protocol.

### 7.2 Stage 1: one env

The agent explores, finds the goal, holds it in its state, and keeps going back
to it. This is the RL² setting (Duan et al. 2016; Wang et al. 2016), which
recurrent agents learn routinely, and this repo's teacher-swap test
(2026-09-01, a frozen RNN adapts in context) points the same way.

Training is the `task3r_k2_h128` recipe with two changes.
1. There is no memory, so there is no store. The first goal touch still ends
   novelty, as the store does in `task3r`.
2. The hidden state carries across the K visits. Today `task3r` resets it
   every visit, because the Hopfield memory carries the goal. PPO then has to
   backpropagate across visit boundaries, so a whole visit sequence trains
   as one sequence. That is a contained code change in the collector and the
   PPO update.

It passes if revisits find the goal nearly every time, in close to idea 1's
11 to 12 steps.

### 7.3 Stage 2: the continual protocol

One GRU runs the whole continual protocol, revisits included, and has to keep
every env's goal in its state while it learns new envs.

Model size is not the obstacle. Six goals is a few dozen numbers, and a
512-unit state holds that easily. Data is not the obstacle either, since
lifetimes can be generated without limit. Time is. At evaluation size the
protocol is 840 mini-episodes. Env 1's goal is written near the start and
needed again tens of thousands of steps later. Backpropagation through time
cannot span that, and with truncated backpropagation the gradient never
connects the moment a goal is stored to the moment it is needed. So the
network gets no signal to protect env 1's goal while it learns env 5.

The plan works around that.
1. Train on short lifetimes where full backpropagation fits: 2 or 3 envs, a
   few revisits each, and a step cap after the goal is found. That is a few
   hundred to about a thousand steps per sequence.
2. Use a curriculum on the number of envs and on the gap between finding a
   goal and revisiting it.
3. Make length generalisation the test. Train short, evaluate on the full
   6-env protocol with `run_cl_ood.sh` (as in §10.8), and plot retention
   against the time since the goal was found. If the GRU learns stable
   storage, retention stays flat at long gaps. If it learns decaying storage,
   retention falls off, and that curve is the result.

### 7.4 Order of work and open choices

1. Stage 1: carry the hidden state across visits in the collector and PPO,
   then train on the `task3r` recipe.
2. Stage 2 on short lifetimes, at 2 and 3 envs.
3. Length generalisation on the full protocol, scored like §10.8 so all three
   models sit side by side.

Still open are the hidden size (128 as in `task3r`, or larger), GRU or LSTM,
the training sequence length, and the curriculum schedule.

### 7.5 What was built (2026-11-02)

Jack chose to train the plain GRU on many places. Idea 3 has no grid MLP, so
to walk back to a remembered goal the GRU has to learn by RL to turn its own
grid code and a stored goal into a direction. On task3r's 3 fixed arenas it
could learn that only at those 3 places, and the held-out arenas sit at other
places on the scaffold. The NN-control line found that a decode learned at a
few scattered places does not transfer (A1m). So the training arenas move to
new places every update. The question this asks is whether a plain GRU can do
the whole task, not only whether it can hold a goal. If it fails, the goal
probe below says which part failed.

Model. A GRU, not the ReLU RNN that idea 1 and Agent-HaSH use, because here
the state is the only memory and gates are what let a state hold a value.
Inputs are the current reward, the previous reward, action and displacement,
the 60-ray view with the panorama, and the grid code at the agent's cell (434
numbers). There is no memory readout and no distractors. Hidden 128 (as
task3r) and 512.

Code (commits a582262, 5132b08, 7bb8034).
- `--memory_backend none` skips every memory read and write
  (`memory/backend.py`). The first goal touch still ends the search shaping,
  as the store does in task3r.
- `--input_grid_code` adds the `grid_code` channel, last in the layout.
- Stage 1. The GRU state carries across the visits of a task sequence, in
  training and in `evaluate_task`. PPO trains each sequence as one trajectory
  (`training/task.py: merge_visits`), and a test checks that replaying it from
  a zero state reproduces the collected log-probs exactly.
- Stage 2. A `lifetime:N,envs=E,iters=K` stage runs, for each row, a small
  copy of the continual protocol: E arenas introduced one at a time, K passes
  over the seen arenas per block, each trial ending at the goal or the 200-step
  cap (`rollout/lifetime.py`). The state runs through the whole lifetime. A row
  that finishes a trial early keeps its state unchanged until the others
  finish. Each row's real steps are packed into one PPO trajectory, and the
  lifetime is one episode, so returns run across trials.
- Evaluation. Every in-training eval runs `evaluate_task` (two visits, state
  carried) and `evaluate_lifetime` (3 held-out arenas, 2 passes per block).
  nav= and disc= do not apply (there is no memory to pre-load or store into)
  and refuse the backend. The continual protocol is `agenthash.py`, which now
  carries the state from each trial to the next through all 840 trials
  (`run_cl_ood.sh` arms `i3_h128`, `i3_h512`).
- `analysis/idea3/goal_probe.py` fits linear readouts of the goal's grid
  phases and of the displacement to the goal from the GRU state, on held-out
  trajectories, before and after the goal was first found. In lifetime mode
  it also asks whether the goals of other arenas are held.

Launcher levers on `task3r_*`: `_i3` (the model above), `_rp` (arenas and
goals redrawn every update across the whole scaffold, 80-cell margin from the
held-out arenas; `--refresh_goal 1` replaces the per-sequence redraw, which is
the same thing here since each arena gets one visit sequence per update),
`_h512`.

### 7.6 Training idea 3 by supervision (plan, 2026-11-04)

The RL runs of §7.5 and §10.13 failed on the training, not necessarily on the
model (Jack, 2026-11-04: "our training process, not the model, is probably
quite far off"). This section is the replacement, and it need not match how
ideas 1 and 2 were trained.

Why PPO from scratch was the wrong tool. Storing the goal pays only once the
policy steers by it, and steering cannot start before the goal is stored, so
PPO never left the ground. The hard function, turning (grid code now, held
goal code) into a direction, got a noisy scalar a few times per episode,
where Phase 1 needed dense supervised pairs balanced over displacement to
learn it at all. And each update saw 3 arenas, with the same 3 wall layouts
for a whole run. Almost everything the GRU needs has a known right answer at
every step, so it can be taught directly.

The plan, three stages, each a stage of one `train_navigate` schedule.

1. Pretrain the memory and the direction step, supervised. A teacher drives
   the agent through the real task (the same collector, so the inputs are
   exactly what the agent sees later): before the goal is found it searches
   (the novelty teacher, toward the nearest unvisited cell), after it goes
   straight to the goal, and the task teleports it after each touch, as now.
   Teacher moves carry direction noise, so the previous action does not give
   the next one away. At every step after the first touch the GRU's
   displacement head must output the vector to the goal, and its policy must
   assign high likelihood to the teacher's move. Backpropagation from every
   later step reaches the touch, so storing the goal is taught directly, and
   every teleport gives a fresh displacement from a random cell. Walls and
   places are redrawn every update, and more arenas are used per update.
2. Imitation on the agent's own states (DAgger). The probability that the
   teacher, not the student, acts anneals from 1 to 0; the labels stay the
   teacher's. This removes the gap between the teacher's states and the
   student's. Then the same on short continual-protocol lifetimes
   (`lifetime` stages), where the label after a find is the direction to the
   current arena's goal, so several goals have to be held at once.
3. PPO fine-tune from the imitation checkpoint, if search or speed need it.

The teacher knows the true goal, but only to produce training labels. The
trained model never sees it and is a plain GRU at test (Jack agreed,
2026-11-04). This is not the goal-in-memory input ruled out for the
task-faithful line, which is an input at test.

What it shows. If this trains a GRU that returns to goals at new arenas and
keeps them through the continual protocol, a plain GRU can do the task and
the earlier failure was the training. If it fails even with dense supervision,
the architecture is the limit.

Evaluation as before: the in-training task and lifetime evals, the goal
probe, `follow_head`, and the continual protocol with the state carried
(`cl_summary.py`, side by side with idea 1 and Agent-HaSH).

Built (commit f66ea77).
- `collect_rollout(teacher_beta=..., teacher_noise_deg=...)`: the teacher
  labels every step (toward the nearest unvisited cell before the row's first
  goal touch, straight at the goal after, at 0.9 of the speed range), makes a
  `beta` share of the moves with its direction turned by Gaussian noise (30°
  default; the label stays clean), and records the vector to the goal and
  whether the goal was known. `merge_visits`, `pack` and `pad_common` carry
  the new fields.
- `updates/sup.py: sup_update` replays each sequence whole and fits the
  policy to the labels (the polar head's heading likelihood plus 0.2 of its
  speed likelihood, on steps not at the goal), the displacement head to the
  goal vector on known steps, and the value head to the rollout's own
  returns. 2 passes per update.
- Schedule keys `sup=1` and `beta=a->b` on `task` and `lifetime` stages.
  Lever `_sup` runs the plan as one schedule, 3000 updates:
  `task:800,visits=2,sup=1,beta=1->0 ; task:700,visits=2,sup=1,beta=0 ;
  lifetime:800,envs=3,iters=2,sup=1,beta=0 ; lifetime:700,envs=6,iters=1,sup=1,beta=0`,
  6 arenas x 16 rows x 2 visits per update, places, walls and goals redrawn
  every update (`_rp_rw`). The PPO fine-tune (stage 3) forks from its end.
- Tests: teacher labels point exactly at the goal after the find and are the
  executed moves at beta 1 without noise; the fields survive joining and
  packing; repeated `sup_update` steps reduce both losses; a CLI run of a
  supervised schedule completes. A GPU smoke run (27894170) took 17 s per
  supervised task update and 48 s per 6-arena lifetime update with an
  untrained student.

| run | model | job |
|---|---|---|
| `task3r_k2_h512_i3_rp_rw_sup` | GRU 512, one layer | 27895152 |
| `task3r_k2_h512_i3_rp_rw_l3_sup` | GRU 512, three layers | 27895153 |

Both from `code/i3_f66ea77`, 1-day limit (`--continue_from` if a run needs
more).

Wave 1 (those two runs) learned to copy the previous move. While the teacher
drove, the student's direction error after a find was 31°, about what copying
a move with 30° noise gives (24°), and it rose to chance as the teacher handed
over (99° at beta 0.33). Its displacement loss stayed 0.35 to 0.9. Wave 2
(27906089, 27906090; lr 1e-3, 4 passes, displacement weight 2, noise 60°, 300
pure-teacher updates first) plateaued in the same place during the pure-teacher
phase: displacement loss 0.31 to 0.38 and 30 to 36° after a find, flat from
u70 to u300, for 1 and 3 layers alike.

Stage 1 in its pure form works. `analysis/idea3/pretrain_decode.py` builds
synthetic sequences on the GPU from table lookups, with the collector's input
layout (views zero, their input weights zeroed in the saved model): a fresh
arena place and goal per sequence, noisy goal pursuit, a teleport at each
touch with the previous displacement zero as in the collector, a new visit
every 200 steps. The displacement head learns the vector to the goal at every
step after the first touch. The step right after a teleport can only be
answered from the grid codes, so it is scored on its own. 20,000 steps of
256 sequences x 400 steps, lr 1e-3, about 1.5 h on one GPU (jobs 27912919,
27912920).

| fresh places and goals, step 20,000 | all steps after a find | step after a teleport (grid decode only) |
|---|---|---|
| GRU 512, one layer | R² 0.997, 2.2° | R² 0.995, 2.8° (median 1.9°) |
| GRU 512, three layers | R² 0.998, 1.7° | R² 0.997, 2.2° (median 1.6°) |

Both sat near 47° on the step after a teleport for the first 3,000 to 6,000
steps and then dropped within a few hundred steps, to 4 to 6° by step 6,000 to
8,000. So a plain GRU can hold a goal and turn the held goal code and the
current grid code into a direction about as accurately as idea 1's frozen
grid MLP (0.4 to 2°), at places and goals it has never seen. The RL failure
and the teacher-driven waves' plateau were the training, not the model: they
never gave the decode enough dense, varied examples to get past the plateau
(the teacher-driven waves saw about 2,400 steps a second against roughly
18,000 here, and far fewer teleports). Checkpoints
`/orcd/pool/003/jackking/cls_runs/idea3/pretrain_h512.pt` and
`pretrain_h512_l3.pt`.

Stages 2 and 3 forked from the pretrained models (jobs 27921631 one layer,
27921633 three layers; `code/i3_60d0bd8`, `LOAD_CKPT` = the pretrained
checkpoint), lr 3e-4 to keep the decode:
`task:200,visits=2,sup=1,beta=1 ; task:600,...,beta=1->0 ; task:400,...,beta=0 ;
lifetime:600,envs=3,iters=2,sup=1,beta=0 ; lifetime:600,envs=6,iters=1,sup=1,beta=0`.
In the real task, teacher-driven, the decode carries over at once: by u200
the displacement loss is 0.045 (one layer) and 0.020 (three layers), and the
student's heading after a find is 12° and 7° off the teacher's, against 0.31
to 0.38 and 30 to 36° for wave 2 without pretraining.

The in-training evals run the student alone on the held-out arenas (new walls,
places and goals), sampled. Over u50 to u200, still before the handover:

| student alone | revisit found | revisit steps | steps per return in a visit | lifetime, known goal reached |
|---|---|---|---|---|
| three layers | 0.91 to 1.00 | 11.7 to 13.8 | 12.1 to 12.4 | 0.51 to 0.94 |
| one layer | 0.33 to 0.46 | 19 to 76 | 20 to 32 | 0.14 to 0.32 |
| idea 1, late (§10.1, §10.5) | 1.00 | 11.5 to 11.8 | about 12 | |

The three-layer GRU returns to a goal it found as fast as idea 1 does, at
arenas it has never seen, and already holds goals across 3-arena lifetimes
it has not trained on. Its search is weaker than idea 1's (first visits find
the goal 0.23 to 0.45 of the time against 0.5 to 0.6); imitating the novelty
teacher's search is hard (73° off its moves).

At the end of the two-visit stages (u1200, teacher fully handed over) the
three-layer student revisits 0.92 to 0.95 of the time in 16 to 17 steps (a
little slower than at u200; the goal probe at u200 and u800 shows the decode
did not erode, held-out trained-head R² 0.66 then 0.78, the agent's moves
following the goal direction at cosine 0.73 then 0.84). The one-layer student
stays behind (revisits 0.57 to 0.66 in 37 to 42 steps).

The full continual protocol at u1200, before any lifetime training (state
carried through all 840 trials, the §10.8 settings): the three-layer GRU holds
one goal and applies it everywhere. Arena 0 alone succeeds 0.90; when arena 1
arrives, arena 0 goes to 1.00 and arena 1 to 0.05, so in a new arena the GRU
keeps steering for arena 0's goal instead of searching. Later arena 4's goal
takes over (0.65, then 0.93) and arena 0 collapses (0.07). Scored on every
arena (`cl_summary.py`): own block 0.33, after the find 0.32, worst arena
change −0.83 (one layer: 0.28, 0.15, −0.90). This is the baseline for the
lifetime stages, which train exactly this: a new arena means an unknown goal,
and several goals have to be held at once.

The real-task lifetime stage did not teach it and undid the single-arena
skill. Over u1200 to u1800 (3-arena lifetimes, student alone in training) the
three-layer student's revisits on held-out arenas fell from 0.95 in 16 steps to
0.55 to 0.63 in 64 to 79, and in the 3-arena lifetime eval a known goal was
reached 0.41 to 0.62 of the time in 77 to 87 steps (from 0.79 in 41). In
training lifetimes its heading after a find stayed 53 to 64° off the teacher.
The one-layer run was stopped at u1850; the three-layer one runs on to u2400
as a comparison.

So stage 2 got a synthetic version too (`pretrain_decode.py --mode lifetime`,
generator in `hopfield_nav/training/synth_decode.py`): per sequence 2 to 6
arenas at random scaffold places, each with a goal, trials in the protocol's
order (block i: one or two passes over arenas 0..i), each from a random cell
with no previous step and ending at the goal or after 50 steps, rows packed.
In an arena whose goal the row has not reached the walker wanders first, so
its moves do not give the goal away. Targets: the vector to the current
arena's goal when it is known, and whether it is known (a pretraining-only
head). Started from the single-arena model, 10,000 steps of 128 lifetimes, lr
5e-4, about 2 h (job 27969855).

| step 10,000, fresh places and goals | 3 arenas | 6 arenas |
|---|---|---|
| knows whether this arena's goal was found | 0.998 | 0.994 |
| direction at the first step of a trial in a known arena | 10.9° (median 6.2°) | 23.1° (median 12.1°) |
| all known steps | R² 0.97, 7.2° | R² 0.88, 13.7° |

A plain three-layer GRU holds up to 6 goals keyed by arena place, tells a new
arena from a known one, and on entering a known arena from a random cell
recalls that arena's goal and decodes its direction, with a median error of 6°
(3 arenas) and 12° (6 arenas). The 6-arena figure was still falling; 20,000
more steps at lr 3e-4 run as job 27975446.

Then the real-task stages again, from this model, with rehearsal: after every
update, 2 optimizer steps on 64 synthetic lifetimes (`--synth_rehearsal_steps`,
lever `_reh`), so the real-task stages cannot wear down what pretraining taught.
`task3r_k2_h512_i3_rp_rw_l3_sup_reh`, job 27975445, `code/i3_01ef8f8`:
`task:300,visits=2,sup=1,beta=1->0 ; task:200,visits=2,sup=1,beta=0 ;
lifetime:400,envs=3,iters=2,sup=1,beta=0.5->0 ; lifetime:600,envs=6,iters=1,sup=1,beta=0`,
lr 3e-4.

By u500 (end of its two-visit stages) the student alone, held-out arenas,
revisits 0.99 to 1.00 in 12 to 17 steps and reaches a known goal in 3-arena
lifetimes 0.90 to 0.95 of the time in 16 to 26 steps. The full continual
protocol at u500 (840 trials, state carried) is much worse:

| u500, protocol | a0 | a1 | a2 | a3 | a4 | a5 |
|---|---|---|---|---|---|---|
| own block | 1.00 | 0.05 | 0.38 | 0.90 | 0.05 | 0.10 |
| final block | 1.00 | 0.00 | 0.65 | 0.80 | 0.03 | 0.10 |

Success after the find 0.52 (`cl_summary`), against 0.32 for the model
without lifetime pretraining and 1.00 for idea 1. Three things checked:

1. A protocol input bug, specific to idea 3. `run_mini_episode` placed the
   agent with `set_position`, which does not reset the env's last
   displacement, so a trial's first prev_displacement input was the env's last
   move from its previous trial. The collector starts every rollout at zero.
   `fresh_prev_displacement` (on for memory backend none, off otherwise, so
   earlier models' protocol and goldens are unchanged) fixes it. Effect small:
   success after the find 0.52 at 40 passes per block (unchanged), 0.59 to
   0.68 at 2. The same three arenas fail.
2. Length. Passes per block 2 / 5 / 10 / 40 give success after the find 0.59 /
   0.70 / 0.71 / 0.52: not the main cause in the real protocol.
3. Memory or policy? `analysis/idea3/synth_val_arenas.py` runs synthetic
   lifetimes over the run's own 6 held-out arenas (their places and goals; no
   views, no policy) and scores recall per arena: median direction error at
   the first step of a trial in an arena found before.

| median error at a trial start | a0 | a1 | a2 | a3 | a4 | a5 |
|---|---|---|---|---|---|---|
| synthetic-lifetime model, 2 passes per block | 6° | 10° | 15° | 28° | 33° | 26° |
| same, 10 passes per block | 14° | 41° | 29° | 64° | 54° | 63° |
| real-task fork u500, 2 passes per block | 17° | 105° | 95° | 88° | 68° | 109° |

Two problems. Length does matter for the memory itself: trained on at most
2 passes per block, recall degrades with 10 (the protocol has 40). And the
real-task stages wore the memory down despite rehearsal: after them only
arena 0 is still recalled, and the rehearsal loss sat near 0.2 throughout
against 0.017 at the end of pretraining. One likely route is the view input,
whose weights start at zero and grow in the real task, changing the trunk's
dynamics on real inputs in a way synthetic rehearsal (no views) cannot
constrain.

Fixes, both running.
- Long synthetic lifetimes: `--iters_choices 2 5 10 20 --tbptt 600`
  (truncated BPTT, windows of 600 steps, state carried detached), from the
  synthetic-lifetime model, 1,500 lifetimes of 64, lr 3e-4 (job 27988918).
- `--freeze_trunk` (lever `_ft`): in the real-task stages only the heads
  train, so the memory cannot be worn down and the view weights stay zero
  (the trunk then sees the same inputs as in pretraining). The cost is that
  search is a linear readout of features not trained for search. First fork
  from the 2-pass synthetic model (job 27989094, `code/i3_ed115bd`; heads at
  lr 1e-3: two-visit stages with the handover, then 6-arena lifetimes). The
  rehearsal fork was stopped at u550.

Results of those two.
- Long synthetic lifetimes fixed length for the memory. On a held-out set of
  6 arenas (the frozen fork's), synthetic recall at a trial start, median:
  7 / 10 / 12 / 25 / 48 / 46° for arenas 0 to 5 at 40 passes per block, the
  same as at 2 passes (7 / 10 / 12 / 23 / 32 / 37°). The fifth and sixth
  arenas are held only roughly.
- A frozen trunk with linear heads cannot gate "follow the goal if known, else
  search" (a product of two signals); its revisits slowed from 18 to 39 steps
  as it trained (stopped). With a fresh 2-layer MLP head
  (`analysis/idea3/add_head_mlp.py`, job 27998044, from the long-lifetime
  trunk) the heads can gate, but the frozen trunk does not fit the student's
  own trajectories: its heading after a find is 8° off while the teacher
  drives and 30 to 35° once the student does, and the displacement loss goes
  0.04 to 0.3. Held-out revisits 0.67 to 0.86 in 20 to 45 steps.
- So the trunk has to adapt to real trajectories without wearing the memory
  down. `--freeze_view_input` (lever `_fv`) masks the gradient of the trunk's
  view-input weights, which then stay at pretraining's zeros, while the rest of
  the trunk trains (lr 1e-4, rehearsal 4 steps per update). Job 28010359,
  `code/i3_v10` (commit 77838e3), from the long-lifetime trunk.

The protocol before the real-task lifetime stage measures the wrong thing.
On the frozen-MLP fork at u400, arena 0 is held throughout (0.88 to 1.00 per
block) and arena 1 almost never solved (0.03 in its own block), yet synthetic
recall on these arenas is good (arena 1 at 10°). Until then the policy has only
seen one arena per sequence, never a second arena while holding another's
goal; the lifetime stage is what teaches that. Both forks are scored on the
protocol at the end of their lifetime stage.

The frozen trunk with the MLP head got worse in its real-task lifetime stage
(held-out revisits 0.29 to 0.31 in 78 to 83 steps over u400 to 800, from 0.80
in 33); freezing the trunk is a dead end. The view-held trainable fork is the
best idea 3 model so far:

| `_reh_fv`, held-out arenas, student alone | u0-200 | u200-400 | u400-600 | u600-750 |
|---|---|---|---|---|
| revisit found | 0.99 | 1.00 | 0.99 | 1.00 |
| revisit steps | 10.9 | 10.9 | 14.8 | 22.2 |
| 3-arena lifetime, known goal reached | 0.82 | 0.91 | 0.97 | 0.97 |
| steps to a known goal | 21 | 16 | 14 | 22 |

Its revisits match idea 1's speed (11.5 to 11.8) through u400. In the full
protocol at u750, mid-way through its lifetime stage, the second arena is now
solved (own block 0.90; every earlier checkpoint 0.03 to 0.05):

| u750, protocol | a0 | a1 | a2 | a3 | a4 | a5 |
|---|---|---|---|---|---|---|
| own block | 0.97 | 0.90 | 0.23 | 0.35 | 0.38 | 0.60 |
| final block | 0.53 | 0.78 | 0.35 | 0.30 | 0.45 | 0.60 |

Success after the find 0.65 in 57 steps (idea 1 1.00 in 13), worst arena
change −0.45 (arena 0, from 0.97 to 0.53 as later arenas arrive): interference
between held goals, as the synthetic recall of the fifth and sixth arena
already suggested.

#### More 2-pass lifetimes do not help (2026-11-06 to 11-08)

The run hit its 16 h limit near u870. A continuation (`--continue_from`, job
28088456, `code/i3_v10`) added a 600-update lifetime stage at teacher share 0
and reached u1290 before its own limit. Its in-training evals stayed flat over
the 8 points from u900 to u1250. In 3-arena lifetimes known goals were reached
0.93 to 0.99 of the time in 17 to 25 steps, and new goals were found 0.18 to
0.36 of the time. The protocol, one run per checkpoint, got no better:

| protocol, held-out arenas | u750 | u850 | u1050 | u1250 |
|---|---|---|---|---|
| own block | 0.57 | 0.73 | 0.43 | 0.63 |
| success after the find | 0.65 | 0.71 | 0.40 | 0.58 |
| steps after the find | 57 | 53 | 40 | 54 |
| worst arena change | −0.45 | −0.53 | −0.43 | −0.75 |

At u850 the loss is in arena 0. It holds 0.95 with one more arena and falls to
0.47 once a third arrives.

The real-task stages wear down the long retention that long-block pretraining
gave the trunk. This is synthetic recall on the same six held-out arenas
(`analysis/idea3/synth_val_arenas.py`, no views, no policy), as the median
direction error at a trial start for arenas 0 to 5, in degrees:

| checkpoint | 2 passes per block | 40 passes per block |
|---|---|---|
| `pretrain_long_h512_l3` | 7/10/12/23/32/37 | 7/10/12/25/48/46 |
| u400 | 11/12/14/28/40/40 | 14/29/22/45/59/58 |
| u850 | 11/16/19/39/46/58 | 18/30/34/57/48/67 |
| u1250 | 13/18/18/41/48/38 | 24/46/33/56/69/53 |

Recall at 2 passes changes little, and recall at 40 passes decays as training
goes on. Both the real lifetimes (6 arenas, 2 passes, 42 trials) and the
synthetic rehearsal (2 passes, hard-coded) are short. The protocol runs 40
passes per block (840 trials), and arena 0 starts failing near trial 120.
After pretraining, nothing trains a goal to survive that long. This is not
the whole story, because arena 0 is still the best-recalled arena here but the
one the protocol loses, so the policy on real trajectories adds a second
failure.

The fix now running rehearses long lifetimes. `--synth_rehearsal_iters 2 5 10
20 --synth_rehearsal_tbptt 600` (lever `_rl`, commit d96bd25) draws
pretraining's passes per block and runs each lifetime in windows of 600 steps,
with the gradients summed into one optimizer step. A `--load_checkpoint` fork
from u850 cannot place its arenas. It must clear all 7,491 of the parent's
refreshed placements, and 6 clear offsets remain. So `--continue_from` now
accepts the two rehearsal flags (commit dd88c88), and job 28372096
(`code/i3_v12`) continues the run from u1250 with long rehearsal on the same
held-out arenas. It is scored by synthetic recall at 40 passes, then by the
protocol.

#### Long rehearsal holds old goals and learns new ones worse (2026-11-09)

Long rehearsal brings 40-pass recall back on arenas 0 to 3, over four
checkpoints. Arena 4 goes the other way and is near chance by u1450 even at 2
passes (78°), an arena-specific decode failure rather than a retention one.

| 40 passes, trial start, arenas 0 to 5 | degrees |
|---|---|
| `pretrain_long_h512_l3` | 7/10/12/25/48/46 |
| u1250 (long rehearsal starts) | 24/46/33/56/69/53 |
| u1300 | 18/35/28/44/70/52 |
| u1350 | 19/35/23/42/69/55 |
| u1400 | 13/24/20/40/77/49 |
| u1450 | 17/27/21/40/95/54 |

The protocol was run with two seeds (3000, 3001; second seed in
`idea3/cl_s3001`) on u850 and u1450:

| protocol | u850 s3000 | u850 s3001 | u1450 s3000 | u1450 s3001 |
|---|---|---|---|---|
| success after the find | 0.71 | 0.67 | 0.59 | 0.57 |
| own block | 0.73 | 0.63 | 0.40 | 0.44 |
| arena 0, final block | 0.47 | 0.35 | 0.97 | 0.93 |
| worst arena change | −0.53 | −0.48 | −0.23 | −0.58 |

In both seeds, long rehearsal fixes the loss of arena 0, and new arenas are
used worse in their own block (arena 1 0.15 to 0.20 there, then 0.65 to 0.80
in later blocks; arenas 4 and 5 0.10 to 0.15). Success after the find falls
from about 0.69 to about 0.58. The worst arena change swings 0.35 between
seeds at u1450, because the forgetting moves from arena 0 to arenas 2 and 3,
so one protocol run cannot rank checkpoints on it.

The run hit its time limit at u1690. The last two checkpoints, two seeds each:

| protocol | u850 | u1450 | u1550 | u1650 |
|---|---|---|---|---|
| success after the find | 0.71 / 0.67 | 0.59 / 0.57 | 0.55 / 0.73 | 0.65 / 0.56 |
| steps after the find | 53 / 60 | 47 / 39 | 42 / 35 | 41 / 41 |
| worst arena change | −0.53 / −0.48 | −0.23 / −0.58 | −0.25 / −0.40 | −0.18 / −0.25 |

Synthetic recall at 40 passes is 16/25/21/43/57/46° at u1550 and
15/22/20/47/59/42° at u1650, so arena 4 is back from chance. Long rehearsal
cut forgetting and made revisits faster (about 40 steps after the find
against 55), but success after the find did not move. The seeds differ by up
to 0.18 on it (u1550), so two runs per checkpoint is the minimum for any
claim. At u1650 both seeds learn and hold arenas 0 and 1 (final block 0.75 to
0.93) and solve arenas 2 to 5 at 0.12 to 0.57 in every block, their own
included.

#### Recall falls with the order a goal arrives in, set by pretraining

Introducing the same six arenas in reverse order (`synth_val_arenas.py
--order 5 4 3 2 1 0`) shows that error follows the position, not the arena.
Median error at a trial start, 40 passes, by position introduced:

| position | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `pretrain_long`, forward | 7 (a0) | 10 (a1) | 12 (a2) | 25 (a3) | 48 (a4) | 46 (a5) |
| `pretrain_long`, reversed | 7 (a5) | 11 (a4) | 20 (a3) | 12 (a2) | 26 (a1) | 28 (a0) |
| u1650, forward | 15 (a0) | 22 (a1) | 20 (a2) | 47 (a3) | 59 (a4) | 42 (a5) |
| u1650, reversed | 14 (a5) | 33 (a4) | 31 (a3) | 29 (a2) | 39 (a1) | 77 (a0) |

Arena 4 is recalled at 11° when it comes first and at 48° when it comes
fifth. The GRU writes the first goals well and each later one worse.
Pretraining set this, and the real-task stages make the later positions worse
still. The training process accounts for it. In protocol order the first arena
is in every block and the last only in the final one, and lifetimes draw 2 to
6 arenas, so the sixth position exists in one lifetime in five. In a 6-arena,
2-pass lifetime position 0 has 1,876 known steps and position 5 has 76, so
the sixth goal gets a 25th of the gradient.

`pretrain_decode.py --balance_positions` weights the displacement loss so
every position carries the same total weight (mean weight per known step 1).
Job 28468381 (`code/i3_v13`, commit c8d219d) fine-tuned `pretrain_long` with
it at pretraining's other settings (2/5/10/20 passes, window 600, lr 3e-4,
1,500 lifetimes of 64), giving `idea3/pretrain_long_bal_h512_l3.pt`. It
halves the error of the later positions and costs the early ones nothing:

| 40 passes, by position | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `pretrain_long`, forward | 7 | 10 | 12 | 25 | 48 | 46 |
| balanced, forward | 7 | 10 | 12 | 18 | 25 | 16 |
| `pretrain_long`, reversed | 7 | 11 | 20 | 12 | 26 | 28 |
| balanced, reversed | 7 | 11 | 14 | 11 | 18 | 12 |

Recall stays flat from 2 to 40 passes. A smaller position effect remains (7°
first, up to 25° later).

The real-task stages are rerun from it with long, balanced rehearsal
(`--synth_rehearsal_balance`, lever `_bal`, commit 3500428): variant
`task3r_k2_h512_i3_rp_rw_l3_sup_reh_fv_rl_bal`, job 28482187 (`code/i3_v14`),
the best model's schedule plus a 600-update lifetime stage. The real-task
lifetimes are still unbalanced (6 arenas at 2 passes give the first arena 12
trials and the last 2).

#### Synthetic recall does not predict the protocol (2026-11-10)

The balanced run is no better in the protocol. Two seeds, with u500 the end
of the task stages:

| protocol | fv u850 | bal u500 | bal u750 |
|---|---|---|---|
| success after the find | 0.71 / 0.67 | 0.26 / 0.32 | 0.73 / 0.59 |
| steps after the find | 53 / 60 | 33 / 18 | 37 / 46 |
| own block | 0.73 / 0.63 | 0.23 / 0.18 | 0.70 / 0.34 |

At u500 only arena 0 is solved, yet synthetic recall at u500 holds every
arrival position within 10 to 24° (40 passes, reversed order
10/18/17/19/19/24). The lifetime stage then makes the protocol work for
arenas 0 and 1 while synthetic recall of the later positions erodes (u750,
reversed, 15/28/27/31/49/43).

Steering by the agent's own displacement head in known arenas
(`agenthash --follow_disp_head`) does worse than the policy (u500 0.19/0.20,
u750 0.41/0.51), but straight-line steering in walled arenas confounds that.
The clean measure lets the policy act and records the head's error against
the straight line to the goal on the protocol's own trajectories
(`--probe_disp_head`, `run_cl_ood.sh PROBE_HEAD=1`). Median error per arena,
goal 2 or more cells away, degrees:

| real protocol trajectories | trial start | all known steps |
|---|---|---|
| bal u500 | 18/100/47/55/135/119 | 41/103/125/135/166/131 |
| bal u750 | 14/28/29/42/94/118 | 10/34/70/62/94/117 |
| fv u850 | 15/53/40/64/74/93 | 69/49/33/66/69/69 |

On real trajectories the memory is the bottleneck, not the policy. At u500 the
head is at chance for arena 1 and worse than chance for arenas 2 to 5, which
fits a head still pointing at arena 0's goal (y = 3; the others lie at y = 10
to 18). Protocol success per arena tracks this error: the arenas the policy
solves are the ones whose head error is under about 30°. The lifetime stage
is what teaches the real-trajectory memory to hold arena 1 (and partly 2 and
3). For fv u850 the head error on arena 0 grows block by block (8, 22, 74,
87°), which is its forgetting.

Synthetic recall overstates the memory because the generator leaks the goal.
In an arena whose goal is not yet found, the walker wanders 0 to 30 steps and
then heads straight for the goal, so its moves and grid-code path reveal the
goal before the find. In a known arena it pursues the goal every trial, which
keeps refreshing the state from its own moves. A model can then hold a goal
as "where my moves head" rather than writing it once at the find. On the
protocol the student's own moves do not head for an unlearned goal, so that
memory is absent. `Synth.lifetime(prefind="random")` (commit 104c125) keeps the
walker random-walking until it touches the goal by chance. With 200-step
trials it finds about as many goals as the pursuing walker (known trial starts
per position 155/78/92/43/26/7 against 159/132/90/59/37/12).

The pre-find leak is not the cause. With the non-leaking walker, synthetic
recall at u500 is almost unchanged (10 passes, trial start 12/18/15/26/43/27°
against 12/18/13/25/38/19°), so the write at the find works. The cause is the
pursuit after the find. With `pursue_known=False` the walker random-walks in
found arenas too, so no later trial re-reads the goal from moves that head
to it, and the goal has to survive from the one write:

| 10 passes, trial start, degrees | walker pursues found goals | never pursues |
|---|---|---|
| `pretrain_long_bal` | 7/9/11/19/23/17 | 27/59/39/77/65/93 |
| bal u500 | 12/18/15/26/43/27 | 33/78/56/58/102/126 |
| bal u500, real protocol probe | | 18/100/47/55/135/119 |

Without pursuit, synthetic recall at u500 reproduces the real-protocol
failure. Every pretraining stage (single-arena, lifetime, long-block,
balanced) used a walker that heads straight to a found goal on every later
trial. The GRU learned to keep a goal by re-reading it from those
trajectories, not by holding the write. In the protocol one failed revisit
leaves nothing to re-read, and the failures cascade. The real lifetime stage
helps only the arenas whose revisits the policy already wins.

The no-pursuit recall is therefore the proxy to train against.
`pretrain_decode.py` gains `--cap`, `--prefind` and `--pursue_known` (the
chance that a trial in a found arena pursues the goal), and its evals add
`E6k2np` (6 arenas, 2 passes, 200-step trials, no pursuit). Two arms
fine-tune `pretrain_long_bal` with 200-step trials, the non-leaking pre-find
walk, balanced positions, 2/5/10 passes, window 600, lr 3e-4, 1,500
lifetimes of 64 (`code/i3_v15`, commit 88f8901): `pursue_known 0`
(job 28580665, `idea3/pretrain_np0_h512_l3.pt`) and `0.5` (job 28580667,
`pretrain_np05_h512_l3.pt`). If no-pursuit recall comes near the pursued
numbers, the real-task stages are rerun from the better one.

Both arms reach their level by about step 100 (in-run `E6k2np`, median trial
start, 21 to 22°) and hold it. Half pursuit is the better arm. On the
held-out arenas, trial start, degrees:

| | no pursuit, 10 passes | pursued, 40 passes |
|---|---|---|
| `pretrain_long_bal` | 27/59/39/77/65/93 | 7/10/12/18/25/16 |
| `pretrain_np0` | 14/22/21/40/77/71 | 18/47/25/63/59/82 |
| `pretrain_np05` | 11/19/22/40/48/49 | 11/17/18/38/35/34 |

Without pursuit, the first three arenas come down from 27 to 59° to 11 to 22°
and the last three from 65 to 93° to 40 to 49°. Pursued recall of the later
positions costs 16 to 25° to 34 to 38°, a trade the protocol favours. The
real-task stages are rerun from `pretrain_np05` with the same walker in the
rehearsal (`--synth_rehearsal_cap 200 --synth_rehearsal_prefind random
--synth_rehearsal_pursue_known 0.5`, lever `_np`, passes 2/5/10, 4 steps per
update; commit 8fd2eaa): variant
`task3r_k2_h512_i3_rp_rw_l3_sup_reh_fv_rl_bal_np`, job 28593952
(`code/i3_v16`), the balanced run's schedule.

At u500 (end of the task stages) it is no better. Protocol, two seeds:
success after the find 0.26 / 0.40 (balanced run 0.26 / 0.32); arena 1 is
0.00 to 0.07 in its own block. Steering by the head does no better (0.35 /
0.40). The head probe at a trial start is 25/101/57/88/–/119° (arena 4 never
found); its all-steps medians look good (42/24/30/18/36°) but long failed
trials, with the agent pressed against a wall while the head points through
it, inflate that measure, so trial start is the one to read. No-pursuit
synthetic recall at u500 (10 passes) is 28/51/56/69/58/77°, against
11/19/22/40/48/49° for `pretrain_np05`: the task stages wore the fix down
despite 4 steps per update of no-pursuit rehearsal.

The task stages carry the same leak. For their first 300 updates the teacher
drives (share 1 falling to 0), so the inputs are the teacher's moves, which
head straight for the goal: the pursuit shortcut again, in real data. A
variant keeps the teacher at share 0 from the start (the student drives, the
teacher only labels), so no move hands the goal to the state; an untrained
policy's trajectories are close to the no-pursuit walk. Job 28640809, same
checkpoint, rehearsal and code, schedule `task:500,visits=2,sup=1,beta=0 ;
lifetime:1000,envs=6,iters=2,sup=1,beta=0` (lr 1e-4).

The teacher was not the cause. With the student driving from the start,
held-out revisits are still 1.00 in 11 to 12.6 steps from u50, but no-pursuit
recall erodes as much as in the teacher-driven run (10 passes, trial start,
u250: share 0 18/33/25/50/59/61°, teacher-driven 20/31/39/45/65/45°, from
11/19/22/40/48/49°). The erosion comes from the real-task training itself,
most likely the single-arena task stages: 500 updates in which nothing
requires holding a goal across arenas. The share-0 run was stopped at about
u300. Job 28656514 (same checkpoint, rehearsal and code) skips them and trains
on 6-arena real lifetimes from the start with the student driving:
`lifetime:1500,envs=6,iters=2,sup=1,beta=0` (lr 1e-4).

The original-schedule run from `pretrain_np05` (28593952) hit its time limit
150 updates into its lifetime stage. At u650 it lands where every variant has:

| protocol, two seeds | fv u850 | np run u650 |
|---|---|---|
| success after the find | 0.71 / 0.67 | 0.57 / 0.68 |
| steps after the find | 53 / 60 | 51 / 48 |
| worst arena change | −0.53 / −0.48 | −0.33 / −0.35 |

Forgetting is lower, and both seeds hold arenas 0 and 1 to the end (final
block 0.93 to 0.97 and 0.82 to 0.97), but arenas 2 to 5 stay at 0.12 to 0.47
in every block. Head probe at a trial start 20/52/76/66/59/120°; no-pursuit
recall (10 passes) 20/53/40/53/53/92°. Long rehearsal, position balancing,
no-pursuit pretraining and a student-driven task stage all end at two goals
held and about 0.6 to 0.7 success after the find.

The lifetime-only run (28656514) at u100, two seeds: success after the find
0.56 / 0.49 in 32 steps (faster than any earlier run's 48 to 60), arena 0
held at 1.00 in every block in both seeds (worst arena change −0.40 / 0.00),
arena 1 failing (0.05 to 0.40), arenas 2 and 3 solved in one seed (0.70 to
0.85). Head probe at a trial start 24/71/93/67/82/113°; no-pursuit recall
(10 passes) 22/66/36/54/77/81°. One early point; u300 follows.

The cluster cancelled the run at u230 (2026-11-11 08:12, during a Slurm
controller outage; no reason recorded). It resumes from its u200 state with
`--continue_from` (job 28685740, `code/i3_v16`). At u200, two seeds: success
after the find 0.62 / 0.82 in 32 / 29 steps, worst arena change −0.33 /
−0.23. Seed 3001 is the best idea 3 protocol so far: arenas 0 and 1 held at
0.97 to 1.00 to the end, arenas 2 and 4 at 0.47 to 0.95, so three to four
goals held, not two. Seed 3000 holds only arenas 0 and 1. No-pursuit recall
(10 passes) 22/44/27/43/77/59°. Two points (u100, u200) show a direction, not
a result; the run continues to about u550.

| lifetime-only run, two seeds | u100 | u200 | u300 |
|---|---|---|---|
| success after the find | 0.56 / 0.49 | 0.62 / 0.82 | 0.76 / 0.72 |
| steps after the find | 32 / 32 | 32 / 29 | 31 / 30 |
| worst arena change | −0.40 / 0.00 | −0.33 / −0.23 | −0.38 / −0.63 |

At u300 both seeds beat the earlier best (fv u850, 0.71 / 0.67 in 53 to 60
steps), and both hold more than two goals: arena 0 at 1.00, arena 2 at 0.85
to 0.95 for most of the protocol, arenas 3 and 4 at 0.45 to 0.95 in places.
Arena 1 is weak in its own block (0.03 to 0.17) and good only later; some
arenas still decay by the final block. No-pursuit recall (10 passes)
22/61/31/53/70/63°. Idea 1 is 1.00 in 13 steps.

u400 did not hold it: success after the find 0.60 / 0.45 in 42 / 49 steps
(seed means 0.53, 0.72, 0.74, 0.52 at u100 to u400), with arena 0 decaying
again in seed 3001 (0.88 to 0.30) and arena 2 never solved there. The rise
to u300 is not yet a gain over the plateau; the series is being filled in at
every 50 updates (u350, u450, u500) before any claim.

| success after the find | u100 | u200 | u300 | u350 | u400 | u450 | u500 |
|---|---|---|---|---|---|---|---|
| seed 3000 | 0.56 | 0.62 | 0.76 | 0.65 | 0.60 | 0.56 | 0.58 |
| seed 3001 | 0.49 | 0.82 | 0.72 | 0.48 | 0.45 | 0.63 | 0.56 |
| steps after the find, mean | 32 | 31 | 30 | 50 | 45 | 30 | 44 |

Seven checkpoints, no lasting trend, centred near 0.6. The run was stopped
at about u520 (2026-11-11).

#### Where idea 3 stands (2026-11-11)

One goal in one arena is solved: on held-out arenas the student alone
revisits 0.98 to 1.00 of the time in 10 to 12 steps, idea 1's speed, in every
run since §7.6's pretraining. First finds are weak (about 40% within 200
steps), which Jack ruled not the interesting question.

Holding goals across arenas is not solved. Every training variant here ends
at 0.45 to 0.82 success after the find (two seeds each; typical 0.6 to 0.7)
in 28 to 60 steps, holding two goals well and later ones poorly, against idea
1's 1.00 in 13 steps:

| variant | success after the find |
|---|---|
| view-held run, u850 | 0.71 / 0.67 |
| + long rehearsal, u1450 to u1650 | 0.55 to 0.73 |
| balanced-position pretraining, u750 | 0.73 / 0.59 |
| no-pursuit pretraining, original schedule, u650 | 0.57 / 0.68 |
| no-pursuit, lifetimes only, u100 to u500 | 0.45 to 0.82 |

What the work established:
- Synthetic recall with a walker that pursues found goals overstates the
  memory. The GRU can keep a goal by re-reading it from trajectories that
  head to it. The valid proxy is no-pursuit recall (`synth_val_arenas.py
  --cap 200 --prefind random --no_pursue_known`), which reproduces the
  protocol probe (`agenthash --probe_disp_head`).
- Pretraining can build a memory that holds goals without pursuit
  (`pretrain_np05`: 11/19/22/40/48/49° at 10 passes), but every real-task
  training tried wears it down (to about 20/50/35/50/60/70°), with or
  without the teacher driving and with or without single-arena stages.
- Recall falls with arrival order because protocol-order lifetimes give the
  sixth goal a 25th of the gradient; balancing fixes that on the synthetic
  test only.

Untested levers, if the line is reopened: a larger recurrent state (hidden
1024 or 2048, the capacity question); real-task lifetimes with the loss
balanced by position and with more passes per block; several training seeds
per variant, since checkpoint-to-checkpoint swing is 0.2 to 0.3.

#### Next step: one generic recipe, varied only by hidden size (agreed 2026-11-12)

The training above became a curriculum built for one architecture: a scripted
walker, its pursuit probability, teacher stages, view freezing, long and
balanced rehearsal. Each piece patched the last symptom. A result from it
cannot be read as a property of recurrent networks, in either direction: a
success would belong to the curriculum, and the plateau cannot be called the
GRU's limit. Jack (2026-11-12): the setup is "incredibly contrived".

The next idea 3 result uses one simple, generic recipe, applied unchanged to
every arm. The only thing varied is the hidden size, so the question becomes
capacity, not curriculum.

Recipe, identical for every arm:
1. Optional single pretraining stage: the existing single-arena grid-decode
   pretraining (`analysis/idea3/pretrain_decode.py --mode single`, its
   original settings, no lifetimes, no walker variants). It is kept only
   because nothing learned the grid decode from the real task in feasible
   compute (RL arms, §7.5). If it is used, it is used for every arm with the
   same settings; an arm without it is the cleaner control if compute allows.
2. One real-task stage: supervised DAgger on 6-arena protocol-order lifetimes
   (`lifetime:N,envs=6,iters=2,sup=1,beta=0`). The student drives from the
   first update and the teacher only labels moves. One auxiliary loss, the
   goal displacement when the goal is known, at one fixed weight. All weights
   train.
3. Dropped: synthetic rehearsal, view freezing (`_fv`), long or balanced
   rehearsal (`_rl`, `_bal`), the no-pursuit walker (`_np`), teacher-driven
   stages, single-arena task stages.

Arms: a plain 1-layer GRU at hidden 256, 512, 1024 and 2048, the same input
set as §7 and the same env-step budget per arm. Two training seeds per size if
the budget allows.

Read-out, per arm: the continual protocol with two seeds at four or more
checkpoints (success after the find, steps after the find, per-arena
retention), and the head probe at a trial start (`agenthash
--probe_disp_head`). Report the series, not single checkpoints. The answer is
how many of the six goals a plain recurrent state holds as it grows, against
idea 1 and Agent-HaSH at 1.00.

### 7.7 Stage 1 runs (RL, superseded by §7.6)

Stage 1 runs, 4000 updates, launched from a frozen copy of the code at
`/orcd/pool/003/jackking/cls_runs/code/i3_5132b08` so later edits cannot reach
them. A 20-update smoke run (27664925) took 8.6 s per update.

| run | job |
|---|---|
| `task3r_k2_h128_i3_rp` | 27665224 |
| `task3r_k2_h512_i3_rp` | 27665225 |
| `task3r_k2_h512_i3_rp_ag` (auxiliary goal head, below) | 27679116 |

By u1000 neither plain run uses a found goal. Revisits find it as often as
first visits (0.3 to 0.4, 20 evals each) and take as long (67 to 90 steps),
and returns after a teleport within the first visit take 50 to 77 steps,
search speed. Idea 1 revisited every time in about 12.5 steps by then. The
goal probe on the u1000 checkpoints (held-out goals, 614k steps each) says the
goal is not stored at all.

| u1000 | goal phase R², before find / after | own cell phase R² | direction to goal error, after find |
|---|---|---|---|
| GRU 128 | −0.13 / −0.13 | 0.26 | 69° |
| GRU 512 | −0.24 / −0.24 | 0.64 to 0.77 | 67° |

The 128-unit GRU also lost most of its own position (0.68 to 0.82 at u20).
Storing the goal pays only once the policy steers by it, and steering cannot
start before it is stored. The plain runs continue to u4000. A third arm adds
`--aux_goal_weight 0.1` (lever `_ag`): a linear head on the GRU features
predicts the grid code the agent saw at its last rewarded step. The target is
the agent's own input at that moment, and the head is never read by the
policy or at evaluation, so the tested model is still a plain GRU. If this
arm learns to return, storage was the bottleneck and RL can learn to turn
grid codes into a direction. If it stores the goal and still cannot return,
that step is the obstacle. Launched from `code/i3_43e1dc4`.

By u500 the goal head has done its job but the agent still cannot return.
Its loss fell from 13.6 to 1.1, and the goal probe on the u500 checkpoint
reads the goal's phase at R² 0.50 after the find (−0.26 before). The
displacement to the goal is still not readable (R² −0.20, direction error 47°
after the find against 51° before). Behaviour got worse in one way. After a
find the plain runs keep searching and touch the goal again 0.3 to 0.5 times
per visit. This arm almost stops (0.0 to 0.1), though the returns it does
make are fast (5 to 25 steps). The state after a find now differs from the
state before it, so the policy is in a regime where search no longer runs and
steering by grid codes has not been learned. So storage was one bottleneck,
and turning (grid code now, stored goal code) into a direction is the second.

A fourth arm adds `--aux_disp_weight 1.0` (lever `_ad`, with `_ag`): a second
head predicts the displacement to the cell of the last reward, worked out
exactly from the two grid codes (`updates/ppo.py: code_displacement`, tested
against the code itself, scaffold wrap included). That is the
code-pair-to-displacement supervision the Phase-1 grid MLP was trained with,
from the agent's own inputs. The head is never read by the policy. Job
27685876, from `code/i3_4c79274`. (The probe's cell-from-code inversion was
checked exact on 500 random cells, so its direction numbers stand.)

The displacement head learns on the training rollouts (loss 0.12 to 0.24 by
u1500, against about 1.3 for always predicting zero), but behaviour does not
change: revisits find the goal 0.34 of the time in 68 to 88 steps, and returns
within a visit take 50 to 65 steps. The probe on the u1500 checkpoint, held-out
arenas, after the find, now also scores the agent's own moves.

| u1500, GRU 512 + both heads, after the find | value |
|---|---|
| goal phase R² | 0.53 |
| displacement R² / direction error | 0.09 / 62° |
| cosine of action with true goal direction | 0.005 |
| cosine of action with the direction decoded from its own state | −0.04 |

The goal is held, the direction to it reads only weakly on new arenas, and the
policy does not follow either. The plain GRU 128 ended at u4000 (153.6M env
steps) the same as it was at u1000: first visits find the goal 0.32 to 0.36
of the time, revisits 0.34 to 0.39 in 90 to 97 steps, and in the 3-arena
lifetime a known goal is reached no more often than an unknown one.

A negative result needs a positive control, so a fifth run trains the plain
GRU 512 on task3r's 3 fixed arenas, goals redrawn each sequence
(`task3r_k2_h512_i3`, job 27718237). There it only has to return at 3 places
and can use the fixed walls. If it learns to return, the setup can teach
RL² goal return and the failure above is about doing it at new places. If it
cannot, the training setup itself is the obstacle. Option A as a control,
not as the experiment.

## 7b. Idea 4: a recurrent memory module, goals held in activity (plan, 2026-11-12)

Idea 4 is a memory-only control: how many goals can persistent activity hold?
It keeps ideas 1 and 2's arrangement and changes only the memory module, which
becomes a small recurrent network. Its weights are fixed after training, so
every stored goal has to live in its activity. The grid code, the frozen grid
MLP, the controller and the oracle store are unchanged, so ideas 1, 2 and 4
and Agent-HaSH differ only in how the memory stores a goal. With them it is
the control for why an associative memory with plastic weights helps at all,
over persistent activity alone.

Idea 4 differs from idea 3 (§7). Idea 3's GRU has to decode directions, hold
goals and choose actions in one state. Idea 4's network only holds goals, and
no actions are learned for its main result.

### 7b.1 Model

| | idea 1 | idea 2 | idea 4 |
|---|---|---|---|
| memory | table of (view key, goal) rows | small MLP | small recurrent network (GRU) |
| where goals live | table rows | weights, changed by gradient steps | activity; weights fixed after training |
| write | add a row at the goal | a few gradient steps on the visit's views | feed the visit's views with the goal as input |
| read, every step | nearest key to the live view | run the MLP on the live view | step the network on the live view |

- Inputs each step: the same query as ideas 1 and 2 (live view with the
  panorama, plus absolute heading). During a write, also the goal's grid
  phases (cos and sin for each of the 3 modules, 12 numbers) and a write flag;
  at other steps these are zero.
- State: the GRU hidden state, never reset within a lifetime.
- Outputs each step: `ĝ`, the goal phases, turned into a grid code as in idea
  2, and `c`, its confidence that it holds this env's goal.
- Downstream, unchanged: `d = MLP(g_t, ĝ)` through the frozen grid MLP, and the
  controller receives `d` and `c` as in ideas 1 and 2.

### 7b.2 Step 1: memory alone (the main result)

No controller and no actions. It reuses idea 2's harness and go/no-go test
(`analysis/idea2/`, §6.4, §10.9) with the recurrent network in place of the
plastic MLP.

- Training: supervised, on random-walk lifetimes of a few envs. A write is a
  40-step random walk in the env with the goal phases fed in. After the
  writes, fresh views of each written env are queried, and the loss asks for
  that env's goal phases, with `c` high for written envs and low for unseen
  ones. Gradients flow by backpropagation through time over the lifetime.
- Test: write N held-out envs once each, for N = 6, 30 and 100, then decode
  each env's goal from 64 fresh views at random cells and headings. A view is
  correct when the decoded cell is within 1.5 cells of the goal (chance about
  0.02).
- Reads change a recurrent memory, so the test scores reads two ways: each
  from a copy of the state right after the last write (storage), and in
  sequence as navigation would (storage plus the erosion reading causes).
- Sweep: the hidden size of the network, from idea 2's `M` size up to 2048,
  for a capacity curve. The training lifetimes stay short (a few envs), as in
  idea 2, so N = 30 and 100 also test how storage extends past training.

Reference numbers on the same test (§10.9):

| memory | N = 6 | N = 30 | N = 100 |
|---|---|---|---|
| idea 1, table | 0.96 to 1.00 | | |
| nearest neighbour on fixed keys | 1.000 | 0.998 | 0.992 |
| linear delta rule on fixed keys | 0.870 | 0.417 | 0.144 |
| idea 2, best meta-learned plastic MLP | 0.68 to 0.72 | 0.26 | 0.11 |

### 7b.3 Step 2, only if step 1 is competitive

Train the controller with PPO on top of the recurrent memory, as §6.3 planned
for idea 2.

### 7b.4 Open choices

The hidden sizes in the sweep, the number of envs per training lifetime, the
truncation window for backpropagation through time, and whether the network
also gets the write signal's timing as a separate input (a flag) or only
through the value input.

### 7b.5 Step 1 result: activity holds one goal, the newest (2026-11-12)

Idea 4 fails step 1, so step 2 is not run. Every recurrent memory trained
holds one goal, the one written last. Written once each, the best run
decodes 1.00 at N = 1, 0.67 at N = 2, 0.22 at N = 6 and chance from N = 10.
On the same test idea 2's best plastic memory keeps 0.68 to 0.72 at N = 6 and
idea 1 keeps 0.96 to 1.00. Capacity is not the limit. A stripped-down test
with two random keys shows that no recurrent cell tried learns to pick a goal
by matching the query against a stored key, at widths up to 1024 and over
1.9M training lifetimes. An outer-product fast-weight memory with the same
encoder and read MLP learns six pairs to cosine 0.99 within 5000 steps. So
plastic weights help because an outer-product write binds a key to a goal in
one step, and a recurrent network trained by gradient descent does not learn
to do that binding in its activity.

What was built (branch `worktree-sensory-key-retrieval`, commits `719f230` to
`27ab82a`).
- `hopfield_nav/memory/recurrent.py`, the memory module. Idea 2's view
  encoder (360 to 256 to 256) feeds a GRU, together with the goal phases and a
  write flag during a write. A read MLP on the GRU output and the encoded view
  returns the goal phases and the `c` logit. `--chrono` sets chrono
  initialisation of the update gate (Tallec and Ollivier 2018).
- `analysis/idea4/gpu_views.py`, a torch port of `raycast_codes` and `canvas`.
  It matches numpy exactly on 4000 views, so training views are made on the
  GPU.
- `analysis/idea4/train_memory.py` trains on random-walk lifetimes of T envs
  (T from 2 to 20) by backpropagation through the whole lifetime. Each env is
  written once by a 40-step walk. After each write, copies of the state read
  16 fresh views of every written env and two unseen envs. Then the state
  reads a walk of 0 to 40 steps in one env and carries the result forward.
  The test is idea 2's "once" test (§10.9), with the same envs, goals, walks
  and views at the same seeds, scored two ways: "copy" reads each view from a
  copy of the state after the last write, and "seq" reads all views in
  sequence. `--curriculum` starts at T = 2 and lengthens lifetimes when older
  goals are held. `--full_pano` replaces every view with the env's whole
  panorama, so write and read keys are identical.
- `analysis/idea4/recall_probe.py`, plain associative recall. Each lifetime
  draws fresh random keys (+-1 vectors, like panoramas) and goals (random
  phase pairs), writes each key with its goal, then queries each key.
- `hopfield_nav/run_idea4.sh` (width sweep) and `hopfield_nav/run_idea4_probe.sh`.
  Outputs are in `/orcd/pool/003/jackking/cls_runs/idea4/`, logs
  `idea4_*`, `i4probe_*`, and `python -m analysis.idea4.summarise` prints the
  tables.

Runs on the arena task (1-layer GRU, Adam 1e-3, 32 lifetimes per step).

| run | steps | result |
|---|---|---|
| widths 256, 512, 1024, T 2 to 20 | 6k to 20k (512 ran the full 20k) | the newest goal only, from step 250 on; `c` at chance |
| width 2048 | | NaN within 50 steps; one overflowing gradient clipped to inf x 0 (a guard now skips such steps; not rerun) |
| width 256, curriculum | 20k | never passed T = 2; final test below |
| width 512, curriculum, whole panorama as every view | 8k | the same as with real views |

Final tests after 20k steps, copy reads, mean of 3 test seeds (sequence reads
in brackets):

| run | N = 1 | N = 2 | N = 3 | N = 6 | N = 10 | N = 30 | N = 100 |
|---|---|---|---|---|---|---|---|
| width 256, curriculum | 1.00 (1.00) | 0.67 (0.50) | 0.22 (0.11) | 0.22 (0.11) | 0.10 (0.03) | 0.03 (0.04) | 0.01 (0.02) |
| width 512 | 1.00 (1.00) | 0.67 (0.67) | 0.11 (0.11) | 0.17 (0.17) | 0.07 (0.07) | 0.01 (0.02) | 0.04 (0.04) |

Holding one goal of N scores about 1/N, which is what both runs score from
N = 3 on.

Plain associative recall (`recall_probe.py`, 64 lifetimes per step, cosine
per pair after the given steps; outputting the bisector of two goals scores
2/pi = 0.64 on both):

| memory | pairs | steps | cosine |
|---|---|---|---|
| GRU 256, Adam 1e-3 / 3e-4 / 3e-4 with chrono init | 2 | 30k | 0.63 to 0.64 |
| GRU 1024, Adam 3e-4 | 2 | 30k | 0.64 |
| GRU 256, two layers | 2 | 30k | 0.64 |
| GRU 256, each query read for 4 steps | 2 | 30k | 0.64 |
| GRU 256, 8-bit keys | 2 | 30k | 0.64 |
| GRU 64, 1-bit keys (0.82 is possible) | 2 | 6k | 0.64 |
| LSTM 256 | 2 | 30k | 0.64 |
| GRU 256 with multiplicative integration (Wu et al. 2016) | 2 | 30k | 0.64 |
| GRU 64, the same two keys in every lifetime | 2 | 2k | 1.00 |
| outer-product fast weights, H 256 | 6 | 5k | 0.99 |
| outer-product fast weights, H 256 | 30 | 10k | 0.93 |

Reading.
1. The recurrent memory stores and reads back one goal without trouble
   (1.00 at N = 1). On the arena task it settles on returning the newest goal
   whatever the view. In the probe it returns the bisector of the two goals.
   In both cases the read ignores the query.
2. The failure is binding, not storage or capacity. With the same keys in
   every lifetime a GRU holds two goals perfectly within 2000 steps, because
   it only has to keep them in order. With keys drawn fresh in each lifetime,
   even 1-bit keys, it never starts to match. Matching needs a product
   between the query and stored content. A fast-weight write stores the
   outer product of key and goal and the read takes an inner product with the
   query, so the product is built in. A gated RNN can only approximate it
   through its gates, and gradient descent did not find that approximation at
   any width, cell type or learning rate tried. Associative-recall tasks in
   the literature point the same way (fast weights against LSTMs in Ba et
   al. 2016; the H3 paper, Fu et al. 2023, on models without multiplicative
   interactions). Their exact numbers were not checked here.
3. For the control question of §7b this is the answer. Persistent activity
   in a network trained by gradient descent holds one goal, because it does
   not learn to bind goals to views. A memory with plastic weights binds
   them in one write. Idea 2 shows plastic weights alone are not enough at
   large N (crosstalk), and idea 1's argmax read is what reaches 0.96 to 1.00.

Not tested. Training far longer (10^7 or more lifetimes), a recurrent cell
with explicit products of input and state large enough to hold an outer
product in its activity (that is a fast-weight memory by another name), and
a rerun at width 2048 with the guard.

## 8. Decisions and open items

| | |
|---|---|
| Protocol | `task3r_k2_h128`, oracle store — **decided** |
| Grid MLP | Phase-1 decode — **decided**; checkpoint path to find on `worktree-nn-generalization-control` |
| Panorama | summed, weight 1 — **decided** |
| Memory (idea 1) | four-heading key at store, live-view query by absolute direction — **decided** |
| Controller | no explicit gate: `d = MLP(g_t, ĝ)` and `c` as inputs, direction and step length learned — **decided** |
| `c` | always its own input; `scale_q_by_c` flag, default off — **decided** |
| Plasticity | controller never plastic at evaluation — **decided** |
| Idea 2 `M` | MLP; phase outputs + confidence; panorama view input — **agreed** |
| Idea 2 training | Default in §6.2 and §6.3. One plastic MLP, meta-learn θ₀ and α, supervised write at the store, PPO does not train `M`. **Decided.** Variants in §6.5, small choices in §6.6. |
| Idea 3 | GRU with the live view, `g_t`, previous action and reward; no memory readout. Hidden state carried across visits (stage 1), then across the continual protocol (stage 2). **Decided**; rest open (§7.4). |
| d0_base recipe | later variant |
| Learned store head | later, once the oracle version works |

## 9. Implementation order

1. Memory module (argmax key-value, direction-indexed keys, heading-sliced
   query) with a unit test reproducing §2.4. Done.
2. Rollout wiring: `d` and `c` inputs through the frozen MLP, oracle store,
   foreign-goal pre-fill, `scale_q_by_c` flag. Done (stage 1, `8a51297`).
3. Smoke run. Done.
4. Agent-HaSH `task3r_k2_h128` baseline at `--distal_amp 1.0`. Done (§10).
5. Stage 2, the sensory_kv backend in the other evaluators. Done (`74f79f2`).
   `evaluate_navigation`, `evaluate_goal_discovery`, `evaluate_exploration`,
   `evaluate_sequential_episodes` and `analysis/continual/agenthash.py` now
   run idea 1. Under sensory_kv the store payload is a `KVEntry`, so every
   existing store line serves both memories, and the evaluators' distractor
   draws consume the random stream exactly as the Hopfield draws do, so both
   models get the same trials. `evaluate_realistic` and `evaluate_repeat`
   still refuse the backend. The Hopfield path is unchanged: the stub
   goldens, the task-run golden and a new continual-protocol golden
   (`tests/golden_agenthash_run.py`) all match.
6. Evaluate both models on nav_det, disc, expl and the continual protocol.
   Not run yet.
7. Idea 2 (§6.4): `M` meta-trained alone on random-walk lifetimes, a
   retention-vs-N go/no-go, then joint training with the controller. The
   go/no-go is done and fails (§10.9), so joint training was not started.
8. Idea 3 (§7.4): stage 1 with the hidden state carried across visits, then
   stage 2 on short lifetimes, then length generalisation on the full
   protocol. Code for all three is built (§7.5); stage 1 is training.

## 10. Idea 1 results — first head-to-head (2026-09-30)

Implementation: stage 1 on this branch (`hopfield_nav/memory/`, commit
`8a51297`); Hopfield path regression-tested bit-identical
(`tests/golden_task_run.py`). Both runs: `task3r_k2_h128` recipe, 4000 updates,
`--distal_amp 1.0`, one 16 h job each on `ou_bcs_normal`, no segmenting.

| run | job | wall |
|---|---|---|
| idea 1, `task3r_k2_h128_gmlp` | 24342872 | 11.5 h |
| Agent-HaSH, `task3r_k2_h128_distal` | 24342873 | 12.2 h |
| Agent-HaSH, `task3r_k2_h128`, no panorama (reference) | 22883646 | to u3450 |

Task eval (held-out arenas, sampled, visits = 2), means over update windows
(10–20 eval points each; `python -m analysis.sensory_key.compare_runs`).
`steps/reach` = steps between goal reaches after the store; `revisit` = the
second visit, memory kept.

**0 distractors**

| window | run | found | steps/reach | revisit found | revisit steps to goal | cos(a, readout) post-store |
|---|---|---|---|---|---|---|
| u0–500 | idea 1 | 0.41 | **27.3** | **0.94** | **31.5** | **0.58** |
| | Agent-HaSH | 0.27 | 57.0 | 0.68 | 68.3 | −0.01 |
| u500–1000 | idea 1 | 0.55 | **12.8** | 1.00 | **14.2** | **0.91** |
| | Agent-HaSH | 0.47 | 16.6 | 1.00 | 19.3 | 0.77 |
| u2000–3000 | idea 1 | 0.58 | 11.9 | 1.00 | 11.8 | 0.94 |
| | Agent-HaSH | 0.58 | 12.2 | 1.00 | 12.6 | 0.93 |
| u3000–4000 | idea 1 | 0.61 | 12.0 | 1.00 | 11.8 | 0.93 |
| | Agent-HaSH | 0.63 | 12.1 | 1.00 | 12.6 | 0.94 |

**10 distractors**

| window | run | found | steps/reach | revisit found | revisit steps to goal | cos(a, readout) post-store |
|---|---|---|---|---|---|---|
| u0–500 | idea 1 | 0.35 | **25.6** | **0.92** | **31.4** | **0.57** |
| | Agent-HaSH | 0.26 | 62.9 | 0.60 | 78.4 | −0.04 |
| u1000–2000 | idea 1 | 0.57 | **12.2** | 1.00 | **12.2** | **0.93** |
| | Agent-HaSH | 0.47 | 13.3 | 1.00 | 16.3 | 0.89 |
| u3000–4000 | idea 1 | 0.56 | 12.2 | 1.00 | **11.9** | 0.93 |
| | Agent-HaSH | 0.60 | 12.7 | 1.00 | 13.9 | 0.91 |

**Reading of the training curves.**
- **Exploit is learned much faster with the grid MLP.** By u500 idea 1
  follows its readout (cos 0.58) while Agent-HaSH does not yet (≈ 0);
  steps/reach 27 vs 57, revisit found 0.94 vs 0.68. Agent-HaSH catches up by
  ~u2000 at 0 distractors.
- **The panorama is not what helps.** Agent-HaSH with and without it (22883646)
  is the same late and slightly *slower* early with it, so the compass the
  panorama provides does not explain idea 1's lead.

### 10.1 Like-for-like at the end of training, with d0_base (2026-09-30)

`python -m analysis.sensory_key.eval_task_ckpts` (jobs 24487511, 24489724):
`evaluate_task` on ONE env set -- the 6 `base_val` envs the task3r runs
recorded -- each checkpoint with its own env config (panorama or not), 16
trials per env, visits = 2, sampled; 5–10 eval seeds per checkpoint. Late
checkpoints pooled per run (± = spread across checkpoints). d0_base u725 is
one checkpoint, h1024, trained under its own (non-task) recipe. Results
`/orcd/pool/003/jackking/cls_runs/sensory_key/eval_task_ckpts*.json`.

| run (checkpoints) | revisit steps, d = 0 | revisit steps, d = 10 | found, d = 0 / 10 | cos(a, readout) post, 0 / 10 |
|---|---|---|---|---|
| d0_base (u725) | 12.03 | 12.21 | 0.62 / 0.60 | 0.87 / 0.85 |
| task3r, no panorama (u3250–3450) | 12.45 ± 0.27 | 13.15 ± 0.24 | 0.68 / 0.65 | 0.94 / 0.91 |
| Agent-HaSH + panorama (u3800–4000) | 12.35 ± 0.51 | 13.41 ± 1.25 | 0.66 / 0.62 | 0.94 / 0.92 |
| **idea 1 (u3800–3950)** | **11.56 ± 0.11** | **11.55 ± 0.22** | 0.60 / 0.57 | 0.93 / 0.93 |

- **Exploit: idea 1 is best, by a modest margin**: ~0.5 step (4%) fewer
  than d0_base at d = 0 and ~0.7 step (5%) fewer at d = 10; ~1.6 steps (12%)
  fewer than the matched task3r Hopfield runs at d = 10. Its revisit time does
  not grow with distractors at all (11.56 → 11.55); every Hopfield run's does.
- **Search: idea 1 is slightly worse** (found 0.60 / 0.57 against 0.66–0.68 /
  0.62–0.65 for the task3r Hopfield runs; ≈ d0_base) **and less stable**: its
  u4000 checkpoint collapsed to 0.33 / 0.34 (10 eval seeds; its own
  in-training eval reads 0.33 there too), after 0.53–0.70 for all of
  u2500–3950. Revisit and following stayed normal at u4000 -- the collapse is
  in search only. No Hopfield run shows a drop like it. One seed: whether this
  is a real instability needs a second seed.
- d0_base's 12.21 at d = 10 matches its exploit-probe 12.14 (`DUAL_TRAINING.md`)
  -- the two protocols agree on it.

### 10.2 How much sooner (same recipe, same experience per update)

First update at which the 3-eval rolling mean of the in-training task eval
crosses a target (idea 1 / Agent-HaSH + panorama / Agent-HaSH no panorama).
d0_base is not comparable here (h1024, its own recipe).

| target | d = 0 | d = 10 |
|---|---|---|
| revisit found ≥ 0.99 | **u250** / u550 / u400 | **u250** / u550 / u400 |
| revisit steps ≤ 20 | u550 / u700 / u550 | **u500** / u850 / u600 |
| revisit steps ≤ 15 | **u750** / u1250 / u1000 | **u650** / u1850 / u1250 |
| revisit steps ≤ 13 | **u950** / u1850 / u1850 | **u1050** / u3200 / u2600 |
| cos(a, readout) post ≥ 0.9 | **u750** / u1200 / u1150 | **u750** / u1800 / u1450 |

Loose targets: close. Tight ones: ~2× sooner at d = 0, 2.5–3× at d = 10.
Exploit only -- search is not faster. One seed per run.

**Why the early gap is open.** Hopfield recall already gives a near-exact
direction (d0_base `q_accuracy` 0.98 / 0.97 at 0 / 10 distractors), so it is
not that Agent-HaSH must learn to turn recall into a direction. Untested
candidates: raw `q`'s varying magnitude (its gate) versus idea 1's unit `d`
with `c` apart; and linear recall blending distractors into `q` (d0_base
`follow_q` 0.92 → 0.82 with 10) versus argmax returning one clean goal.

**Not yet measured.** The continual protocol (`agenthash.py`: N envs in
sequence, retention) and nav / disc / expl need stage 2 for the sensory_kv
backend. One seed each.

### 10.3 Why idea 1 learns exploit sooner: the inputs themselves (2026-09-30)

The two controllers get the same *information* -- a 2-D goal direction to
follow when trustworthy, same view, same previous action / reward, same
action head -- so the difference has to be in how that input behaves.
`analysis/sensory_key/readout_compare.py` (job 24498430) measures exactly what
each controller is handed, with no policy: every cell of the 6 task3r val
arenas, 8 goals each, Hopfield `q` (the distal run's encoder / beta) vs idea
1's `(d, c)` queried with one live view per cell. Results
`/orcd/pool/003/jackking/cls_runs/sensory_key/readout_compare.json`.

| distance | q error (d = 0 / 10) | d error (0 / 10) | median ‖q‖ | presence AUC ‖q‖, d = 10 | presence AUC c, d = 10 |
|---|---|---|---|---|---|
| 1–2 | 5.1° / 9.4° | 0.6° / 1.9° | **0.13** | **0.67** | 0.996 |
| 3–5 | 5.8° / 7.8° | 0.5° / 0.7° | 0.25 | 0.92 | 0.995 |
| 6–10 | 6.1° / 7.4° | 0.4° / 1.1° | 0.35 | 0.97 | 0.994 |
| > 10 | 7.0° / 8.8° | 0.4° / 1.0° | 0.38 | 0.98 | 0.994 |

(presence AUC: own goal stored vs only distractors stored, from the model's own
presence signal.) Empty memory gives exactly 0 for both, so that case is equal.

1. **Direction**: both good (< 2% of cells > 45° off), `d` ~10× more precise.
2. **Scale**: ‖q‖ shrinks 3× toward the goal; `d` is unit everywhere. Agent-HaSH
   must learn to follow a vector whose strength depends on where it is --
   weakest where precision matters most.
3. **Gate with distractors**: ‖q‖ cannot tell own goal from distractors near
   the goal (AUC 0.67 within 2 cells) because ‖q‖ is small there either way;
   `c` separates at 0.99 everywhere.

Consistent with §10.2: at d = 0 (no gate needed) idea 1 is ~2× sooner -- (1)
and (2); at d = 10, (3) adds and the gap grows to 2.5–3×. Correlational.
**Causal test**: Agent-HaSH task3r with `--no-input_hopfield_raw` (unit q,
still 0 when empty). If it closes most of the d = 0 gap, the scale is the cause.

### 10.4 Bug found: an empty Hopfield read a spurious q (fixed 2026-10-01)

With a list of per-trajectory Hopfields where only some were empty,
`signal.hopfield_signal_at` and `multistep_q` projected the empty rows with
recall = 0, giving `q = project(0 − x)` -- a position-only vector, about as
large as a real recall -- and masked only the *normalized* signal. Every
Agent-HaSH run feeds raw `q` (and multistep), so a still-searching trajectory
with an empty memory was handed that vector whenever another trajectory in the
batch held a memory. A known quirk, pinned by a characterization test during a
behavior-preserving refactor; fixed in `d2ec758` (q zeroed for empty rows).

**Affected**: task-regime training (trajectories with 0 distractors, before
their store -- both task3r Agent-HaSH baselines, 22883646 and 24342873) and
`evaluate_task` at 0 distractors for every Hopfield model once any trajectory
had stored (the §10.1 d = 0 rows for d0_base / task3r). **Not affected**:
d = 10 evals (no empty rows), d0_base's training (one shared Hopfield per
rollout), `agenthash` continual, batched nav / expl evals, and idea 1.

So §10.1–10.2's Agent-HaSH task numbers carry it. Reruns on the fixed code:
`task3r_k2_h128_distal` (24504515) and the unit-q + ‖q‖ variant
`task3r_k2_h128_distal_qmag` (24504516; `--no-input_hopfield_raw
--input_q_magnitude`, the Hopfield analog of idea 1's (d, c)).

### 10.5 Fixed-code head-to-head, and the causal test (2026-10-02)

All on the fixed code (§10.4), `task3r_k2_h128` at `distal_amp 1`, one seed each:
idea 1 (24342872, unaffected by the bug), Agent-HaSH raw `q` rerun
(`_distal`, 24504515), Agent-HaSH unit `q` + ‖q‖ (`_distal_qmag`, 24504516;
`--no-input_hopfield_raw --input_q_magnitude` -- the Hopfield analog of idea
1's (d, c)).

**Learning speed** (first update where the 3-eval rolling mean of the
in-training task eval crosses; idea 1 / raw q / unit q + ‖q‖):

| target | d = 0 | d = 10 |
|---|---|---|
| revisit found ≥ 0.99 | u250 / u700 / u300 | u250 / u600 / u300 |
| revisit steps ≤ 15 | u750 / u1300 / **u550** | **u650** / u1400 / u900 |
| revisit steps ≤ 13 | u950 / u1850 / **u850** | **u1050** / u3000 / u1150 |
| cos(a, readout) post ≥ 0.9 | **u750** / u1650 / u800 | **u750** / u2050 / u850 |

**End of training** (`eval_task_ckpts`, job 24560734: the 6 task3r val envs,
u3800–4000 pooled, 5 eval seeds each, ± across checkpoints;
`eval_task_ckpts_fixed.json`):

| run | revisit steps d = 0 | revisit steps d = 10 | found d = 0 / 10 | pre-store chase, d = 10 |
|---|---|---|---|---|
| idea 1 | **11.60 ± 0.13** | **11.48 ± 0.24** | 0.54 / 0.52 (0.60 / 0.57 without u4000) | **0.007** |
| Agent-HaSH, unit q + ‖q‖ | 11.68 ± 0.38 | 12.41 ± 1.03 | 0.56 / 0.60 | 0.042 |
| Agent-HaSH, raw q | 12.26 ± 0.38 | 13.25 ± 0.79 | **0.65 / 0.62** | 0.023 |

**Reading.**
1. **Most of idea 1's learning-speed advantage was the representation of
   Agent-HaSH's readout, not the memory.** Split ‖q‖ out of `q` and Agent-HaSH
   learns exploit as fast as idea 1 -- slightly faster at d = 0, close behind
   at d = 10 -- and 2–3× faster than with raw `q`. Raw `q` folds the magnitude
   (its gate) into the vector and shrinks 3× toward the goal (§10.3).
2. **At d = 0 the end state ties**: unit q + ‖q‖ 11.68 vs idea 1 11.60.
3. **With 10 distractors idea 1 is still best at the end** (11.5 vs 12.4 vs
   13.2) and chases foreign goals least before storing (0.007 vs 0.042 /
   0.023). This residual is the memory: near the goal ‖q‖ cannot tell the own
   goal from a distractor (presence AUC 0.67 within 2 cells, §10.3), while the
   argmax store's `c` separates them at 0.99.
4. **Search: raw-q Agent-HaSH is the best searcher** (found 0.65 / 0.62 vs
   0.54–0.60). Idea 1's search also collapsed once (u4000) -- one seed.

### 10.6 Projection storage, and idea 1 with magnitude folded in (2026-10-03)

Two more runs, same recipe, fixed code, one seed each:
`task3r_k2_h128_distal_qmag_proj` (24570179; Agent-HaSH unit q + ‖q‖ with
`--hopfield_storage_rule proj`, the projection rule -- stored patterns exact
fixed points, no cross-talk) and `task3r_k2_h128_gmlp_scq_noc` (24569423;
idea 1 with only `c·d`, no `c` channel -- magnitude folded into the direction,
like raw q). Expectations going in: projection storage should be the best
Agent-HaSH, especially with distractors; idea 1 with `c·d` should look like
raw-q Agent-HaSH.

**Learning speed** (first update the 3-eval rolling mean crosses; idea 1 /
idea 1 c·d / raw q / q + ‖q‖ / q + ‖q‖ proj):

| target | d = 0 | d = 10 |
|---|---|---|
| revisit steps ≤ 15 | u750 / u1000 / u1300 / **u550** / **u550** | **u650** / u1000 / u1400 / u900 / u950 |
| revisit steps ≤ 13 | u950 / u1100 / u1850 / **u850** / u1250 | **u1050** / u1450 / u3000 / u1150 / u1450 |
| cos(a, readout) post ≥ 0.9 | u750 / u1050 / u1650 / u800 / **u700** | u750 / u1050 / u2050 / u850 / **u700** |

**End of training** (`eval_task_ckpts`, u3800–4000 pooled, 5 eval seeds each,
± across checkpoints; jobs 24560734 + 24643905):

| run | revisit steps d = 0 | revisit steps d = 10 | found d = 0 / 10 |
|---|---|---|---|
| idea 1 (d + c) | **11.60 ± 0.13** | **11.48 ± 0.24** | 0.54 / 0.52 |
| idea 1, c·d only | 12.08 ± 0.15 | 11.90 ± 0.29 | 0.63 / 0.61 |
| Agent-HaSH raw q | 12.26 ± 0.38 | 13.25 ± 0.79 | **0.65 / 0.62** |
| Agent-HaSH q + ‖q‖ | 11.68 ± 0.38 | 12.41 ± 1.03 | 0.56 / 0.60 |
| Agent-HaSH q + ‖q‖, proj | 12.28 ± 0.52 | 12.87 ± 0.99 | 0.64 / 0.62 |

**Reading.**
1. **Projection storage is not the best Agent-HaSH at the end.** Early it is
   as fast as q + ‖q‖ (fastest to follow its readout of all five), but its
   final exploit is no better: 12.28 at d = 0 (≈ raw q) and 12.87 ± 0.99 at
   d = 10 (between q + ‖q‖ and raw q, inside their spread). It searches like
   raw q (0.64 / 0.62). The near-goal own-vs-distractor confusion (§10.3) is
   about how strongly the *query* overlaps each stored pattern, which the
   projection rule does not change -- a plausible reason it does not fix the
   d = 10 gap. Not tested.
2. **Idea 1 with c·d sits between, and keeps its distractor advantage.**
   Folding the magnitude in slows its learning (1.3–1.5× vs idea 1) but it is
   still 1.3–2× faster than raw q; at the end it is comparable to raw q at
   d = 0 (12.08 vs 12.26) but much better at d = 10 (11.90 vs 13.25). So the
   d = 10 edge belongs to the argmax memory, independent of representation.
3. **A representation trade-off appears.** The two variants that feed
   magnitude *apart* (idea 1 d + c, Agent-HaSH q + ‖q‖) exploit best at d = 0
   (11.6–11.7) but search worst (found 0.54–0.56); the folded ones (raw q,
   c·d) and proj search better (0.63–0.65) and exploit ~12.1–12.3. One seed
   each -- worth a second seed before reading much into it.

### 10.7 Two seeds, over training: idea 1 vs Agent-HaSH q + ‖q‖ + projection (2026-10-03)

Second seeds (seed 43): idea 1 `task3r_k2_h128_gmlp` (24660241; 20 h limit,
replaced 24659510 that would have finished minutes inside 16 h) and
`task3r_k2_h128_distal_qmag_proj` (24659511). Seed 42 = 24342872 / 24570179.

Figure: `/orcd/pool/003/jackking/cls_runs/sensory_key/figs/perf_over_training_d10.png`
(and `_d0.png`), made by `python -m analysis.sensory_key.plot_training`. One
0–1 axis, both models, two quantities from the in-training task eval:
- **path optimality** (solid) -- optimal steps / steps taken on the revisit
  (goal stored, fresh start), ratio of means; the optimal count is exact, over
  every start cell of the run's own val arenas (start uniform over cells ≠ goal,
  reached within the 1-cell goal radius, ≤ 1 cell per step), E[opt] = 10.1 /
  10.6 steps for the seed-42 / seed-43 worlds;
- **found on first visit** (dashed) -- fraction finding the goal within the
  200-step search, before it is stored.
Seed mean, band = seed range, rolling mean over 5 evals.

Late window (u3000–4000, mean over evals; per seed in brackets):

| | path optimality d = 0 | d = 10 | found d = 0 | d = 10 |
|---|---|---|---|---|
| idea 1 | **0.87** [0.86, 0.88] | **0.88** [0.86, 0.90] | 0.52 [0.61, 0.42] | 0.48 [0.56, 0.40] |
| Agent-HaSH q + ‖q‖ proj | 0.82 [0.85, 0.80] | 0.76 [0.81, 0.70] | 0.51 [0.60, 0.43] | 0.53 [0.59, 0.46] |

**Reading.** Idea 1 takes more direct paths to a stored goal in both seeds;
the gap is small without distractors and large with 10 (seed ranges do not
overlap: 0.86–0.90 vs 0.70–0.81) -- the argmax memory's distractor advantage
(§10.5–10.6) holds across seeds. Search is the same for both models (found
0.48–0.53), and varies more between seeds (≈ 0.2) than between models. The
seed-43 Agent-HaSH run dips at the very end (u4000 eval: revisit 19.8 steps at
d = 0), so its end-point labels on the figure (0.70 / 0.74) understate its
late-window level; the window means above are the fair comparison.

### 10.8 Stage 2: nav, expl, disc and the continual protocol (2026-10-31)

Both models from §10.7 (idea 1 and Agent-HaSH q + ‖q‖ with projection storage),
seeds 42 and 43, each scored at u3800 and u3950. Each cell is the mean of
those four values, with their range in brackets. `eval_all` used 32 trials per
env on each run's 6 recorded val envs (jobs 27456617 to 27456620). The
continual protocol is `run_cl_ood.sh` with the sampled policy: 6 envs, 40
iterations per block, a 200-step cap, seed 3000, one store per env, 600
revisits per run (jobs 27456628 to 27456634 and 27456636). Pooled by
`python -m analysis.sensory_key.summarise_stage2`; raw files in
`/orcd/pool/003/jackking/cls_runs/sensory_key/eval_all/` and `.../cl/`.

| metric | idea 1 | Agent-HaSH q + ‖q‖ proj |
|---|---|---|
| nav det success, d = 0 / 10 | 1.000 / 1.000 | 0.995 / 0.993 |
| nav det steps, d = 0 | **11.15** [10.94, 11.34] | 13.01 [11.33, 15.64] |
| nav det steps, d = 10 | **11.09** [10.81, 11.42] | 14.15 [13.04, 15.73] |
| nav stoch steps, d = 0 / 10 | **11.98 / 11.94** | 14.30 / 16.37 |
| expl swept coverage, d = 0 / 10 | 0.54 / 0.54 | 0.55 / 0.53 |
| expl goal found, d = 0 / 10 | 0.42 / 0.36 | 0.45 / 0.43 |
| disc goal reached, d = 0 / 10 | 0.16 / 0.15 | 0.50 / 0.48 |
| continual, first visit reached | 0.968 [0.938, 0.983] | 0.957 [0.904, 0.988] |
| continual, revisit reached | **1.000** (2400 of 2400) | 0.997 [0.990, 1.000] |
| continual, revisit steps | **12.85** [12.54, 13.23] | 15.58 [13.34, 19.30] |
| continual, worst per-env retention change | **0.000** | −0.013 [−0.050, 0.000] |

Reading.
1. Navigation to a stored goal. Both models nearly always succeed, and idea 1
   takes fewer steps: 11.1 against 13.0 to 14.2 with the deterministic policy.
   At 10 distractors the two ranges do not overlap. This agrees with the
   path optimality in §10.7.
2. Continual protocol. Neither model forgets much. Idea 1 reached the goal on
   all 2400 revisits with no loss in any env, and it reaches it faster (12.8
   against 15.6 steps; the ranges do not overlap). Agent-HaSH lost 5% of one
   env's revisits in one run.
3. Exploration is a tie. Swept coverage is the same, and Agent-HaSH finds the
   goal slightly more often, inside a wide spread across runs.
4. Discovery favours Agent-HaSH (0.48 to 0.50 against 0.15 to 0.16), but it
   is not a fair test of either model. It lets the agent's own store head
   write memory, and `task3r` training never trains that head; both models
   were trained with the oracle store. A likely cause of idea 1's low score,
   not yet tested: an untrained head writes the agent's current cell into
   memory, idea 1's own-env key then matches that cell's entry with high `c`,
   and the agent follows a confident direction to a cell that is not the
   goal. A Hopfield memory blends such a write with everything else it holds.
   A fair disc comparison needs a trained store head (§8).

### 10.9 Idea 2 offline, retention against N (2026-10-31)

Idea 2 fails the go/no-go of §6.4, so it stops before any controller training.
Written once per env, the best plastic memory keeps 0.68 to 0.72 of goals at
N = 6, 0.26 at N = 30 and 0.11 at N = 100. Idea 1 keeps 0.96 to 1.00 on the
same test (§2.4). The cause is crosstalk. Every write lands in the same
weights, so reading one env's goal also returns a share of every other stored
goal, and the total share grows with N. Meta-learning did not remove it. The
best meta-learned memory scores below a linear head on hand-built canvas keys
that has no training at all.

Setup. Every run trains `M` alone on random-walk lifetimes
(`analysis/idea2/meta_train.py`, envs from `analysis/idea2/lifetimes.py`, distal
panorama at amplitude 1). A write is one 40-step random walk in the env,
labelled with the goal's grid phases. The test writes N held-out envs in
sequence, then decodes each env's goal from 64 fresh views at uniform random
cells and headings. A view counts as correct when the decoded cell is within
1.5 cells of the goal, so chance is about 0.02, and returning one goal for
every env scores 1/N. Each run's last checkpoint was scored with
`--eval_ckpt`. `python -m analysis.idea2.summarise` prints the tables below
from `/orcd/pool/003/jackking/cls_runs/idea2/*/eval_step*.json`.

Two protocols. The first version of the test was wrong. After each new env it
rewrote every earlier env, so at N = 100 env 1 was written 100 times. That is
rehearsal. A plain network with no meta-learning scored 0.71 to 0.85 at
N = 100 under it, with its oldest envs above its newest. The go/no-go uses
"once", which writes each env one time, as idea 1's store does. The revisit
numbers are kept because the §6.3 lifetime revisits, but they do not test
memory.

References, with no meta-learning ("once"):

| reference | N = 6 | N = 30 | N = 100 |
|---|---|---|---|
| offline ceiling, one walk per env (`ceiling.py`) | 0.96 | 0.72 | 0.46 |
| offline ceiling, five walks per env | 1.00 | 0.97 | 0.91 |
| linear head on centred canvas keys, one delta-rule write per env (`fixed_keys.py`) | 0.870 | 0.417 | 0.144 |
| the same head stored all at once by least squares, no forgetting | 0.875 | 0.415 | 0.115 |
| nearest neighbour on the same keys | 1.000 | 0.998 | 0.992 |
| random-start network, best fixed rate | 0.17 to 0.25 | 0.04 to 0.05 | 0.04 to 0.07 |

The offline ceiling trains the 256-wide network on all N envs at once with
Adam, so nothing is forgotten. It is the best any learning rule can do with
this network and these views.

Meta-learned memories ("once"). Each cell is the accuracy, with the oldest and
newest quarter of envs in brackets.

| run | step | N = 6 | N = 30 | N = 100 | c AUC, N = 6 / 100 |
|---|---|---|---|---|---|
| whole MLP, revisits in training, T 2 to 6 | 7000 | 0.167 (0.00 / 1.00) | 0.000 (0.00 / 0.00) | 0.060 (0.00 / 0.10) | 0.96 / 0.63 |
| whole MLP, T 2 to 10 | 6000 | 0.000 (0.00 / 0.00) | 0.000 (0.00 / 0.00) | 0.048 (0.00 / 0.08) | 0.94 / 0.60 |
| whole MLP, 200-step visits | 4000 | 0.333 (0.00 / 0.00) | 0.016 (0.00 / 0.01) | 0.034 (0.00 / 0.06) | 0.98 / 0.61 |
| whole MLP, k = 3 | 4000 | 0.180 (0.00 / 1.00) | 0.044 (0.00 / 0.01) | 0.012 (0.01 / 0.00) | 0.93 / 0.59 |
| R + H, ReLU keys | 4500 | 0.357 (0.05 / 0.75) | 0.083 (0.02 / 0.18) | 0.052 (0.03 / 0.12) | 0.63 / 0.53 |
| R + H, signed keys | 5000 | 0.664 (0.66 / 0.77) | 0.234 (0.14 / 0.35) | 0.110 (0.04 / 0.28) | 0.86 / 0.58 |
| R + H, signed keys, 200-step visits | 4000 | **0.716** (0.58 / 1.00) | 0.240 (0.11 / 0.41) | 0.108 (0.04 / 0.30) | 0.90 / 0.56 |
| R + H, signed keys, then T 20 to 100 | 3000 + 2000 | 0.680 (0.67 / 0.64) | **0.262** (0.17 / 0.36) | **0.113** (0.06 / 0.23) | 0.80 / 0.55 |

Every run after the first trains without revisits, on T = 2 to 10 envs for the
whole MLP and T = 2 to 20 for R + H. The R + H runs write only the head, with
a squared-error write loss on unit-length keys (`--plastic_tensors 2
--normalize_hidden --write_loss mse`); "signed" adds `--linear_key`. The last
row continues the signed-key run's step-3000 checkpoint on lifetimes of 20 to
100 envs, scoring 8 random written envs after each write (`--init_ckpt
--score_envs 8`). Each run stopped once its evals stopped rising. Under the
revisit protocol the best meta-learned memory reaches 0.78 at N = 6, 0.37 at
N = 30 and 0.14 at N = 100.

Reading.
1. A whole-network write is not specific to one env. All four whole-MLP runs
   stored at most the newest goal and were at chance from N = 30. At step 1500,
   an env's goal decoded on only 9% of views right after its own write, and
   after a lifetime the output barely changed from view to view. These runs
   did learn `c`. It separates written from unseen envs with AUC 0.93 to 0.98
   at N = 6.
2. The cosine write loss limits itself. Its gradient shrinks as the raw output
   grows, so after one large write the later writes barely move the output,
   and the network returns env 1's goal for every env. That is the "oldest
   1.00, newest 0.00" pattern of the high-rate baselines. A squared error on
   the raw phases (`--write_loss mse`) stores the newest env reliably.
3. R + H (§6.5) helps, but only with signed keys. With a ReLU on the key layer
   every key lies in the positive orthant, where any two overlap, and the
   memory plateaued at 0.36 at N = 6. Without that ReLU it reached 0.66 to
   0.75, and the oldest env is kept nearly as well as the newest.
4. Longer visits and longer lifetimes leave N = 30 and N = 100 where they
   were. 200-step visits gave 0.240 and 0.108 against 0.234 and 0.110 at
   40 steps. Training on lifetimes of up to 100 envs moved N = 30 from 0.228
   to 0.262 and N = 100 from 0.093 to 0.113.
5. The limit is crosstalk, and fixed keys show its size
   (`analysis/idea2/fixed_keys.py`). A linear head reads, for each stored env,
   the view's cosine with that env's key times that env's goal, and sums them.
   A single view fills 60 of the 180 direction slots, so its cosine with its
   own env's stored key is only 0.38. Its cosine with another env's key is
   small, 0.075 root mean square, but the read adds N − 1 of them. The noise
   is 0.16, 0.41 and 0.75 at N = 6, 30 and 100, and the delta rule's accuracy
   falls with it (0.87, 0.42, 0.14). Nearest neighbour only needs the own-env
   cosine to be the largest, so it stays at 0.99 to 1.00. This is not
   forgetting. Storing all N goals at once by least squares, so that every
   stored key maps exactly to its goal, gives the same 0.88, 0.42 and 0.12.
   The whole-MLP runs did forget in the classic way (newest 1.00, older 0.00),
   but for a linear head the limit is the read, so replay, EWC or slower
   writes would not help. Better-separated keys or a nonlinear read would.
6. Familiarity is easy and goals are hard. Every meta-learned memory learned
   `c`, but its AUC falls to 0.53 to 0.63 by N = 100, for the same crosstalk
   reason.

What a working idea 2 would need. A plastic linear head reaches idea 1's
retention only if two things change together. Keys must look the same from
every view of an env, so the own-env cosine approaches 1. And keys must be
much wider than N, since random 256-dimensional keys already overlap by about
1/16. The other route is a nonlinear read over the stored keys, which is
idea 1's argmax memory. A whole-network write would need ANML-style masks
(§6.5) to stay local. None of these was run. Code is on branch
`worktree-sensory-key-retrieval` (commits `d2467d0` to `94e49ce`).

### 10.10 Idea 2 with a continual-learning write, XdG and SI (2026-11-01)

The whole-MLP memory forgets catastrophically. Trained on all envs at once, a
1024-wide MLP keeps 0.97, 0.81 and 0.67 at N = 6, 30 and 100 with one walk per
env (`ceiling.py --hidden 1024`); written one env at a time it is at chance
from N = 30. This tries the standard fix, and it does not close that gap.
XdG and SI stop old weights being overwritten, but here the forgetting comes
mostly from new weights acting on old inputs, which they do not cover.

Setup (`python -m analysis.idea2.xdg_si`, no meta-learning). `M` is a
1024-wide two-layer MLP from a random start. Each env is written once by
50 SGD steps on one 40-step walk, squared error on the raw goal phases, and
every write stores its goal (0.98 to 1.00 right after it at c = 0). XdG gates
plasticity. On entering a new env a random fraction p of each hidden layer's
units is drawn, and only the weights into those units, and out of the last
layer's chosen units, may change. Every unit is used when reading, so the read
needs no env identity. SI (Zenke et al. 2017) penalises changes to weights in
proportion to how much earlier writes relied on them, with strength c. It uses
an exact proximal step, since an explicit step on its penalty diverges. The
control is the original XdG, which gates activity too, in the write and in the
read. That read is told which env it is in, so it is an oracle and not a valid
memory. Same envs and test views as §10.9, "once" protocol.

| write | N = 6 | N = 30 | N = 100 |
|---|---|---|---|
| no gating, no SI | 0.167 | 0.039 | 0.038 |
| SI only, best of c = 1 to 100 | 0.362 | 0.083 | 0.067 |
| plasticity gating p = 0.2, best of c = 0 to 100 | 0.310 | 0.067 | 0.051 |
| plasticity gating p = 0.05, best of c = 0 to 100 | 0.167 | 0.053 | 0.057 |
| activity gating p = 0.2, no SI (oracle) | 0.667 | 0.583 | 0.556 |
| activity gating p = 0.2, SI c = 10 (oracle) | 1.000 | 1.000 | 0.965 |
| activity gating p = 0.05, SI c = 10 (oracle) | 1.000 | 1.000 | 1.000 |

Reading.
1. The forgetting comes through new weights. Plasticity gating leaves env 1's
   units alone and SI holds the shared weights in place, but later envs train
   other units, and at read time those units also fire for env 1's views and
   push its output toward their own goals. SI and XdG protect parameters, not
   what the network outputs on old inputs, so they cannot stop this. Stronger
   SI only trades plasticity away. At c = 100 a write stores 0.12 to 0.53 of
   its goal, and retention is no better.
2. Silencing the other envs' units removes it. With activity gating and SI,
   every goal survives at N = 100, but that read is told the env (oracle).
   Joint training gets part of the way by another route. It sees env 1's
   views while it learns env 2, so it can shape env 2's features to stay quiet
   on them. A sequential write never sees env 1's views.
3. Without an oracle, the network would have to pick the units from the
   current view. A gate computed from a single view does not do this well.
   With a 20% random-projection gate, a test view shares 0.47 of its active
   units with its best-matching write view from its own env and 0.28 with
   other envs' (chance 0.20), because views at different headings see
   different parts of the panorama.
4. Methods that constrain outputs on old inputs, replay or functional
   regularisation, target exactly this, but they need stored views of earlier
   envs. Storing views and matching the current one against them is idea 1,
   which keeps 0.99 at N = 100. Commit `b14da98`.

### 10.11 Idea 2 with experience replay (2026-11-01)

Replay removes the forgetting. With every env's walk views stored and replayed,
the whole MLP written one env at a time matches joint training of the same
network, and does slightly better at N = 30 and 100. It still trails idea 1 at
N = 100 (0.75 against 0.99), because joint training itself is capped by
generalising from one 40-step walk per env. And it needs many stored views per
env; one stored view barely helps.

Setup. As §10.10 (1024-wide MLP from a random start, each env written once by
50 SGD steps on one walk, no gating, no SI), plus replay. Each step adds R
stored views of earlier envs, each labelled with its env's goal, weighted
equally with the current env's views. After its write, B of each env's 40 walk
views go into the buffer. `python -m analysis.idea2.xdg_si --p 1 --c 0
--replay 8 40 200 --buffer_per_env 40 8 1` (job 27592897). The range in each
cell is over R = 8, 40, 200, which barely matters.

| views stored per env | N = 6 | N = 30 | N = 100 |
|---|---|---|---|
| none (no replay) | 0.167 | 0.039 | 0.038 |
| 1 | 0.29 to 0.32 | 0.13 | 0.19 |
| 8 | 0.72 to 0.79 | 0.63 to 0.71 | 0.51 to 0.55 |
| 40, every walk view | 0.92 to 0.95 | 0.87 to 0.89 | 0.72 to 0.75 |
| joint training, same network, one walk per env | 0.97 | 0.81 | 0.67 |
| nearest neighbour on centred canvas keys (§10.9) | 1.00 | 1.00 | 0.99 |

Reading.
1. With full storage the oldest envs are kept as well as the newest (0.77 to
   0.80 against 0.69 to 0.70 at N = 100), so what remains is not forgetting.
   It is the joint-training limit. One walk per env does not cover the views
   the test uses. The 256-wide ceiling rose from 0.46 to 0.91 at N = 100 with
   five walks per env (§10.9), so longer visits would lift this too.
2. Storage is the cost. Below 8 views per env, retention falls fast, and the
   oldest envs go first (0.43 to 0.50 at N = 100 with 8 views). One stored
   view covers 120 degrees, a third of the panorama, so replaying it does not
   pin the env's output at other headings.
3. Idea 1 stores one key per env, an all-around view at the goal, and reads
   by nearest neighbour. Replay here stores 40 partial views per env and
   still trails at N = 100. §10.12 replays only idea 1's store.
   Commit `7ca1249`.

### 10.12 Replay with idea 1's storage (2026-11-01)

This asks whether the MLP, trained by replay, can match idea 1 when it keeps
exactly what idea 1 keeps. It cannot. With one all-around snapshot per env it
keeps 0.57 at N = 100, against idea 1's 0.96 to 1.00 (§2.4). So idea 1's
direct lookup is the better use of the same storage, and idea 2 closes.

Setup. As §10.11, but after each env's write the buffer keeps only one
snapshot, taken from the goal cell and covering all 180 direction slots
(ray-cast at 6 headings and merged), which is the information in idea 1's key.
Each replay sample is an earlier env and a random heading, and its input is
the 120-degree view at that heading cut out of the env's snapshot, labelled
with the env's goal (`--buffer_mode omni`). A diagnostic replays exact
ray-cast views from the goal cell at random headings instead
(`--buffer_mode goalcell`). That is the same position and information without
the snapshot's 2-degree slots, so it needs the env at replay time and is not
equal storage. Jobs 27607617 and 27607620. The range in each cell is over 8
and 40 replayed per step.

| replayed for each earlier env | N = 6 | N = 30 | N = 100 |
|---|---|---|---|
| views cut from idea 1's snapshot (equal storage) | 0.70 to 0.71 | 0.63 to 0.64 | 0.57 to 0.58 |
| exact views from the goal cell (diagnostic) | 0.88 to 0.89 | 0.84 to 0.85 | 0.74 |
| all 40 walk views (§10.11) | 0.92 to 0.95 | 0.87 to 0.89 | 0.72 to 0.75 |
| idea 1 (§2.4) | 0.96 to 1.00 | 0.96 to 1.00 | 0.96 to 1.00 |

Reading.
1. At equal storage the replayed MLP keeps 0.57 at N = 100, and its oldest
   envs fall to 0.44 to 0.45. Idea 1 keeps 0.96 or more from the same
   snapshot.
2. Part of the gap is the snapshot's 2-degree slots. A view cut from it
   samples every slot at one fixed angle, while a real view samples each slot
   at an angle that shifts with heading, and nearby walls change within a
   slot (cut and real views differ by 0.06 on values of about 0.33). The MLP
   trained on cut views generalises worse to real ones. Idea 1's cosine match
   tolerates the same jitter. Exact views from the goal cell lift retention to
   0.74 at N = 100, as good as replaying all 40 walk views.
3. Even then replay stops near 0.74 at N = 100, the one-walk generalisation
   limit of §10.11. Idea 1's nearest-neighbour read does not have that limit.
   Commit `6ea5007`.

### 10.13 Idea 3 stage 1, and the continual protocol (2026-11-02)

A plain GRU trained on arenas that move every update (§7.5, option B) does
not learn to use a goal it has found. Both plain runs went 4000 updates
(153.6M env steps), and after a find they do no better than searching again.

Continual protocol, the §10.8 settings (`run_cl_ood.sh`, sampled, 6 arenas, 40
iterations per block, 200-step cap, seed 3000, each run's own held-out arenas),
with idea 3's state carried through all 840 trials. `retention.py` scores
revisits only in arenas learned in their own block, which leaves idea 3 with
no row, so every model is scored here on every arena by
`python -m analysis.idea3.cl_summary` (idea 1 and Agent-HaSH from their §10.8
histories). "After the find" is every trial in an arena after the agent first
reached its goal there.

| run | own block | revisit found | revisit steps | after the find | worst arena change |
|---|---|---|---|---|---|
| idea 1 (s42 u3800, s43 u3950) | 0.94 to 0.98 | 1.00 | 12.6 to 13.1 | 1.00 | 0.00 |
| Agent-HaSH q + ‖q‖ proj (same) | 0.90 to 0.98 | 0.99 to 1.00 | 13.3 to 16.2 | 0.99 to 1.00 | −0.05 to 0.00 |
| idea 3 GRU 128 (u3800, u3950) | 0.31 to 0.33 | 0.32 to 0.35 | 70 to 75 | 0.32 to 0.34 | −0.15 |
| idea 3 GRU 512 (u3800, u3950) | 0.34 to 0.40 | 0.40 | 79 to 87 | 0.38 to 0.40 | −0.15 |

Idea 3's revisits succeed at its search rate and take as long as a search, so
there is no retention to lose. That makes "worst arena change" sampling noise
rather than forgetting. Results in `/orcd/pool/003/jackking/cls_runs/idea3/cl/`.

Why, from the goal probe (§7.5). The plain GRU never stores the goal (goal
phase R² at chance after the find). With an auxiliary head that rewards
storing it, the goal is stored (R² 0.50 to 0.53) but the direction to it reads
poorly on new arenas, and the agent's moves ignore it (cosine with the goal
direction 0.005). Turning a stored grid code into a direction at places not
seen before is what RL does not find. Idea 1 gets that step from the frozen
grid MLP, trained in Phase 1 on code pairs and odometry.

Both auxiliary-head arms also ended at u4000 without returning (held-out
task eval, u3500 to 4000).

| arm | first visit found | revisit found | revisit steps |
|---|---|---|---|
| GRU 512 + goal head | 0.47 | 0.32 | 72 |
| GRU 512 + goal and displacement heads | 0.34 | 0.38 | 93 |

The fixed-arena positive control (`task3r_k2_h512_i3`, §7.5) has not learned
to return either. At u2500 (96M env steps), on its own 3 training arenas,
returns after a find take 50 to 80 steps and revisits 67 to 101, flat since
u500. There it could look the answer up, since the walls and the grid codes
of each arena are the same every update. So the failure is not only about
working at new places: PPO does not teach this GRU to return to a goal it has
just found, in this task, within 100 to 150M env steps. It is not the
visit-carry code either. A return after a teleport within one visit uses only
the ordinary within-rollout state, and those are not learned.

The control ended at u4000 the same way (returns after a find 36 to 80
steps, revisits 57 to 93, on its own arenas).

Reading. The hard step is not holding the goal (an auxiliary head gets it
stored) but turning the held goal and the current grid code into a direction,
which RL does not find even at 3 fixed places. Idea 1 is handed that step by
the frozen grid MLP. Stage 2 (several goals across arenas) was not launched,
because no stage-1 model returns to one goal.

Depth (Jack, 2026-11-03). One GRU step combines the new input with the held
state through a single tanh layer, and the heads read the result linearly. So
comparing the current grid code with a held goal code gets one nonlinear
layer, where the grid MLP that does it for idea 1 has 5 ReLU layers of 768.
The units holding the goal also have to keep their update gates closed, so
the comparison has to be computed in other units that reach the goal through
that one layer. Two arms add depth, both GRU 512 with both auxiliary heads on
moving arenas, from `code/i3_dab7e79`:

| arm | what | job |
|---|---|---|
| `task3r_k2_h512_i3_rp_mlp3_ag_ad` | a 3-layer ReLU MLP between the GRU and every head, reading the GRU output and the step's input (`--head_mlp_layers 3`); the GRU holds, the MLP compares | 27796255 |
| `task3r_k2_h512_i3_rp_l3_ag_ad` | 3 stacked GRU layers (`NUM_RNN_LAYERS=3`) | 27796256 |

The auxiliary heads read the features the policy reads (the MLP output in the
first arm), so the displacement supervision trains the comparison network
directly.

At u1000 to 1500 the stacked GRU is the first idea 3 arm whose revisits beat
its first visits on held-out arenas (found 0.55 to 0.56 against 0.36 to 0.38;
lifetime eval, a known goal reached 0.63 to 0.70 against 0.22 to 0.29 on a
first visit). They are not faster (75 steps), which looks like a search
biased toward the goal rather than a beeline. It stores the goal best of any
arm (goal loss 0.5) and its displacement head reaches loss 0.09 on training
rollouts, about 90% of the variance. The MLP-head arm never stores the goal
(goal loss stuck near 8) and shows no gain.

The goal probe on the stacked GRU at u1500, held-out arenas, after the find:
goal phase R² 0.58, but displacement R² −0.32 (direction error 68°), and the
agent's moves barely follow the goal direction (cosine 0.03). So the
displacement it computes on training arenas does not carry over. A likely
reason: `_rp` moves the 3 training arenas every update but keeps their 3 wall
layouts, and with 3 layouts the view can tell the GRU where it is, standing in
for the grid code. That shortcut fails on new walls, and it was open to every
arm so far. A sixth arm redraws the walls every update as well (lever `_rw`,
`task3r_k2_h512_i3_rp_rw_l3_ag_ad`, job 27828068, from `code/i3_69c3c8b`;
rebuilding 3 arenas costs 0.4 s, 11.3 s per update in all), so the grid code
is the only position signal that transfers.

Both depth arms ran to u4000. The MLP head never helped (revisits 0.20 to
0.31, at or below first visits). The stacked GRU's gain was partial and
faded.

| stacked GRU, held-out arenas | u1000 to 1500 | u3500 to 4000 |
|---|---|---|
| revisit found vs first visit (task eval) | 0.55 vs 0.37 | 0.32 vs 0.45 |
| known goal vs first visit (lifetime eval) | 0.66 vs 0.25 | 0.45 vs 0.29 |
| revisit steps | 75 | 84 |

Its revisits never got faster than a search, so at best it searched with a
pull toward the goal.

In the wall-refresh arm every update's arenas are new (place and walls), so
its training-side losses already score unseen arenas. By u1190 its
displacement loss is 0.16, so it does compute the direction to the goal from
grid codes at arenas it has never seen, yet its revisits (0.49 to 0.51) barely
beat its first visits (0.43 to 0.48).

How good is the direction it holds, and would following it be enough? The
goal probe now also applies the agent's own trained displacement head to the
held-out states. (Its ridge readouts, refit on only 6 arenas, underestimate:
R² 0.09 refit against 0.48 for the trained head on the same states.)

| stacked GRU at u1500, held-out arenas, after the find | trained head R² | direction error |
|---|---|---|
| walls fixed (`_l3`) | 0.26 | 51° |
| walls redrawn every update (`_rw_l3`) | 0.48 | 45° |

So the wall shortcut was real: redrawing walls nearly doubles how well the
direction carries to new arenas. But 45° on average is rough.
`analysis/idea3/follow_head.py` then runs the two-visit task with every move
after the find steered by the agent's own head, at full length, nothing else
changed. That is the most the policy could get from the direction it holds.

| held-out arenas, two visits | steps per return within a visit | revisit found | revisit steps |
|---|---|---|---|
| `_rw_l3` u1500, its policy | 56 | 0.52 | 73 |
| `_rw_l3` u1500, steered by its head | 34 | 0.36 | 65 |
| `_l3` u4000, its policy | 57 | 0.38 | 88 |
| `_l3` u4000, steered by its head | 47 | 0.17 | 78 |
| idea 1, for scale | about 12 | 1.00 | about 12 |

Following its own direction shortens returns within a visit but nowhere near
idea 1's, and makes revisits fail more often (a direction 45° off at full
speed drives into walls). So the direction is too rough to navigate by, not
only unused.

Answer to the depth question. Depth helps the GRU store the goal and compute
part of the direction, and the stacked GRU is the only arm with any revisit
gain. It does not make the GRU return. The direction it computes at new arenas
is far less accurate than the grid MLP's (45° against 0.4 to 2°), and the
policy does not follow it. The Phase-1 grid MLP reached sub-degree accuracy
only with dense, displacement-balanced supervised pairs (raw walks plateau at
8 to 10°, §10 of the NN-control plan); RL rollouts plus an auxiliary head give
far less of that signal. The `_rw_l3` arm ended at u4000 the same way:
over u2000 to 4000 its revisits found the goal 0.37 to 0.47 of the time
against 0.55 to 0.61 on first visits, in 74 to 81 steps. All RL arms are
closed; the supervised training of §7.6 replaces them.
