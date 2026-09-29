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

Two models share everything except the memory: **idea 1** (§1) uses a
nonparametric key-value store; **idea 2** (§6) uses a meta-learned network whose
weights stay plastic at evaluation. Both share one controller with Agent-HaSH's
arrangement, so the three differ only in what fills the memory-readout input.

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

## 6. Idea 2: meta-learned plastic memory

Same env, grid MLP and controller as idea 1; the argmax store is replaced by a
network `M` whose **weights are the memory**.

| Part | What it is |
|---|---|
| `M` | an MLP (not an RNN: the memory must live in the weights, and the controller already carries within-episode state) |
| Input | the live view (with panorama) plus absolute heading, as idea 1's query |
| Output | the goal's grid state as module phases — (cos, sin) of the 2-D phase for each of the 3 modules, 12 numbers — decoded to `gbook`, so it is always a valid grid state; plus a confidence `c` ("do I know this env?") |
| Write | no write operation: at the oracle store, `M`'s weights take a few gradient steps so this env's views map to `g_goal` (and `c` to 1) |
| Read | every step, `ĝ, c = M(view, heading)`; the controller gets `d = MLP(g_t, ĝ)` and `c` exactly as in idea 1 |
| Plastic at eval | `M` only. The controller is frozen at evaluation |

`M` is **meta-trained** so that a few inner-loop gradient steps store a new
env→goal association without erasing the earlier ones — plain SGD on a
sequence of envs would forget. Precedents: OML (Javed & White 2019; a
meta-learned representation feeding a plastic head) and ANML (Beaulieu et al.
2020; a meta-learned mask gating which weights an update may touch).

The question idea 2 answers: idea 1's memory cannot forget by construction;
can a *parametric, gradient-written* memory be meta-learned to retain as well?

### 6.1 Where the training signal comes from

With the direction as a sampled action conditioned on `d` (§1), PPO's
gradient reaches `M` through `d` and `c`. It teaches `c` well (the controller's
decision to follow depends on it) and `ĝ` only noisily (one advantage per step,
through the frozen MLP), and not at all while the controller still ignores
`d`. With an oracle store the exact target `g(goal)` is known — the inner loop
needs it anyway to write the memory — so the proposal is to train **jointly**:

- controller: PPO (`task3r_k2_h128` protocol, oracle store);
- `M`: a supervised meta-loss on the same rollouts — after the inner updates at
  each store, `M`'s error on every env seen so far, plus `c` on seen vs unseen
  envs — and PPO's gradient through `d` and `c`;
- optional warm start: `M` meta-trained alone on random-walk views with oracle
  goal labels, which also gives a cheap **retention-vs-N go/no-go** (N = 6 /
  30 / 100 envs learned in sequence) before any RL. Idea 1's memory scores
  0.96–1.00 there (§2.4).

Pure-RL training of `M` (no supervised meta-loss) stays possible as a later,
stronger claim.

### 6.2 Training — still to settle

- the outer-loop unit: how many envs per sequence, and how that relates to the
  N used at evaluation;
- the inner loop: which views (the episode so far, the four-heading view at
  the goal), how many steps, learned per-parameter learning rates or not,
  which parameters are plastic (all, a head only as in OML, or masked as in
  ANML);
- second-order through the inner steps vs first-order, and how far back the
  outer gradient reaches;
- how PPO's epochs interact with weights that changed during collection;
- where unseen-env (`c` → 0) and foreign-goal cases come from.

## 7. Decisions and open items

| | |
|---|---|
| Protocol | `task3r_k2_h128`, oracle store — **decided** |
| Grid MLP | Phase-1 decode — **decided**; checkpoint path to find on `worktree-nn-generalization-control` |
| Panorama | summed, weight 1 — **decided** |
| Memory (idea 1) | four-heading key at store, live-view query by absolute direction — **decided** |
| Controller | no explicit gate: `d = MLP(g_t, ĝ)` and `c` as inputs, direction and step length learned — **decided** |
| `c` | always its own input; `scale_q_by_c` flag, default off — **decided** |
| Plasticity | controller never plastic at evaluation — **decided** |
| Idea 2 `M` | MLP; phase outputs + confidence; panorama view input — **agreed**; training §6.2 **to discuss** |
| d0_base recipe | later variant |
| Learned store head | later, once the oracle version works |

## 8. Implementation order

1. Memory module (argmax key-value, direction-indexed keys, heading-sliced
   query) with a unit test reproducing §2.4.
2. Rollout wiring: `d` and `c` inputs through the frozen MLP, oracle store,
   foreign-goal pre-fill, `scale_q_by_c` flag.
3. Smoke run.
4. Agent-HaSH `task3r_k2_h128` baseline at `--distal_amp 1.0`.
5. Continual evaluation of both.
6. Idea 2, once §6.2 is settled: `M` meta-training (warm start + retention-vs-N
   gate first), then joint training.
