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

This is idea 1 of two. Idea 2 is not yet written down.

## 1. Model (idea 1)

A key-value goal memory keyed on *sensory* input, whose value is the goal's
*grid state*. Exploiting means handing (current grid state, recalled goal grid
state) to the frozen grid MLP and taking the step it gives; exploring means the
policy chooses its own move. A learned gate chooses between them every step.

| Part | What it is |
|---|---|
| `g_t` | `gbook` at the agent's true scaffold position (`lambdas 11/12/13`, `Npos 1716`, `Ng 434`) — the same scaffold Agent-HaSH runs on |
| Grid MLP | frozen Phase-1 decode (random walks + odometry, displacement-balanced pairs; 0.5° held-out) — `MLP(g_t, g_goal)` → unit direction. Input `[gbook(p), gbook(g)]` |
| Memory | ONE memory shared by every env (as in `analysis/continual/agenthash.py`), one goal per env |
| Write | oracle store on first arrival at the goal: key = four-heading view at the goal, re-indexed by absolute direction (180 two-degree slices); value = `g(goal)` |
| Read, every step | query = the agent's *live* single view plus its absolute heading, compared to each key on the slices the view covers; argmax cosine; returns `(g_goal, s)` with `s` the top similarity; empty memory → zeros, `s = 0` |
| Controller | RNN, hidden 128 |
| RNN inputs | live view, `g_goal`, `s`, the MLP direction `MLP(g_t, g_goal)`, previous action, previous reward |
| Heads | gate (Bernoulli: explore / exploit); direction (von Mises, as d0_base); step length (Beta on [0.5, 1.0], as d0_base) |

**Step.** Explore: the policy's direction × learned step length. Exploit: the
MLP's direction × learned step length. The MLP decides *direction only*;
step length stays learned on both branches, as in Agent-HaSH.

**Gate.** Learned, no masking. It must learn to refuse exploit when memory is
empty (`s = 0`) *and* when the recalled goal belongs to another env (`s` ≈
0.2–0.3 vs ≈ 0.5 for the own goal, §2).

**PPO log-prob.** `log p(gate) + log p(step length) + [explore] · log p(direction)`.
Exploit steps are deterministic in direction given the gate, so they carry no
direction term.

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
goals from other envs. Without them the gate never sees a foreign recall with
`s` ≈ 0.3, which is the case it has to learn to reject.

**Later variant:** the d0_base recipe, for both models.

## 5. Evaluation

- Continual protocol as `analysis/continual/agenthash.py`: one memory, envs
  added in sequence, every earlier env revisited; sampled policy
  (`--stochastic_policy`).
- Metrics: nav_det, disc, expl; retention / forgetting.
- **Baseline:** Agent-HaSH retrained as `task3r_k2_h128` at `--distal_amp 1.0`
  (the existing checkpoints saw no panorama).
- Diagnostics: gate decision vs `s`; exploit steps taken with only a foreign
  goal in memory; the MLP's angular error along exploit paths.

## 6. Decisions and open items

| | |
|---|---|
| Protocol | `task3r_k2_h128`, oracle store — **decided** |
| Grid MLP | Phase-1 decode — **decided**; checkpoint path to find on `worktree-nn-generalization-control` |
| Panorama | summed, weight 1 — **decided** |
| Memory | four-heading key at store, live-view query by absolute direction — **decided** |
| Exploit step | MLP direction × learned step length — proposed |
| d0_base recipe | later variant |
| Learned store head | later, once the oracle version works |
| Idea 2 | not yet written |

## 7. Implementation order

1. Memory module (argmax key-value, direction-indexed keys, heading-sliced
   query) with a unit test reproducing §2.4.
2. Rollout wiring: gate, exploit path through the frozen MLP, oracle store,
   foreign-goal pre-fill.
3. Hybrid PPO log-prob.
4. Smoke run.
5. Agent-HaSH `task3r_k2_h128` baseline at `--distal_amp 1.0`.
6. Continual evaluation of both.
