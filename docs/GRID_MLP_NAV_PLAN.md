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

### 6.1 Default training: naive meta-learning

`M` is one small MLP with weights θ, all of them plastic. The **only**
meta-learned quantities are θ₀ (the initial weights every lifetime starts
from) and α (the inner learning rate — a scalar, or one per parameter as in
Meta-SGD).

**One lifetime** — a sequence of T envs, no replay. This is also exactly what
happens at evaluation:

```
θ ← θ₀                                   # fresh memory
for env t = 1 … T:
    run the episode (controller frozen within the lifetime):
        every step: ĝ, c = M_θ(view, heading) → d = MLP(g_t, ĝ) → controller acts
    on reaching the goal (oracle store):
        target  = g_t                    # the agent's OWN grid state now = the goal's
        data    = the views seen so far this episode (all from env t)
        repeat k times:  θ ← θ − α ∇θ L(θ; data → target, c → 1)
    continue; the K = 2 revisits use the updated θ
```

**The inner loop is supervised with a self-generated label.** At the store
the agent stands on the goal, so its current grid state *is* the goal's grid
state — no teacher is involved. The update says "everything I have been
seeing in this env maps to where I am now": the same value Agent-HaSH and
idea 1 write, written into weights instead of a table.

**Outer loop.** After a lifetime, the final θ is scored on held-out views from
every env in it (does it still output each env's goal?) plus `c` — high on
envs it wrote, low on envs it never wrote. θ_T is θ₀ pushed through every
inner update, each differentiable in θ and α, so `∇_{θ₀, α} L_outer` is
backprop through the whole chain; an Adam step updates θ₀ and α. Exact
second-order is affordable for a small MLP, T up to ~10 and a few inner
steps; first-order is the fallback.

**Unseen-env cases come for free.** Before env t's store `M` has never been
written for it, so every pre-store step is an "unseen env" example (`c` → 0)
whose `ĝ` is effectively some other env's goal — the case idea 1 needed an
explicit foreign-goal pre-fill for, provided lifetimes are long enough that
`M` already holds a few envs.

**Controller.** PPO on the same lifetimes (`task3r_k2_h128` protocol, oracle
store), reading `d` and `c` as inputs. By default PPO does not backprop into
`M`: `d` and `c` go into the rollout buffer as fixed observations, so PPO's
epochs never recompute `M`'s changing weights. (With the direction a sampled
action conditioned on `d`, a PPO path into `M` exists — it teaches `c` well
and `ĝ` only noisily; a later variant.)

**Warm start and go/no-go.** Meta-train θ₀ and α alone on random-walk
lifetimes with no controller, and measure retention against N (6 / 30 / 100
envs written in sequence) before any RL. Idea 1's memory scores 0.96–1.00
there (§2.4).

### 6.2 Variant: PPO inner loop

The inner update can instead be a policy-gradient step,
`θ ← θ + α ∇θ J_PPO(episode)` — MAML-RL (Finn et al. 2017). It supports a
stronger claim: a memory written by reward alone. Not the default because:
the step has to write the goal from one episode, and a PPO gradient carries
the goal's location only through rewards and sampled actions' log-probs
(MAML-RL used ~20 rollouts per inner step on simple tasks); the exact target
is available at the store for free; and meta-gradients through
policy-gradient steps are high-variance (hence E-MAML / ProMP), with
continual retention stacked on top. Worth trying once the supervised default
works.

### 6.3 Variant: fixed representation + plastic head (OML)

Split `M` into `R`, a representation network that is meta-learned but never
updated at evaluation, and `H`, a small linear head that is the only plastic
part (Javed & White 2019). If `R` learns sparse, near-orthogonal features per
env — the panorama makes that easy — writing env t into `H` barely disturbs
env t−1: `R` effectively learns its own key-value store, and forgetting
becomes something the meta-learning can shape. Second-order through inner
steps on a linear head is cheap, so the outer gradient can span the whole
lifetime, and per-parameter learning rates come almost free. ANML (a
meta-learned mask gating which weights an update may touch; Beaulieu et al.
2020) is the other standard structure.

### 6.4 Remaining choices

`M`'s size; scalar or per-parameter α; k; the range of T in training (and how
retention extends to N = 30 / 100 at evaluation); second-order vs
first-order.

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
| Idea 2 `M` | MLP; phase outputs + confidence; panorama view input — **agreed** |
| Idea 2 training | naive default (§6.1): one plastic MLP, meta-learn θ₀ and α, supervised inner loop at the store, PPO not into `M` — **decided**; variants PPO inner loop (§6.2), `R` + `H` (§6.3); small choices §6.4 |
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
6. Idea 2 (§6.1): `M` meta-trained alone on random-walk lifetimes →
   retention-vs-N go/no-go → joint training with the controller.

## 9. Idea 1 results — first head-to-head (2026-09-30)

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

### 9.1 Like-for-like at the end of training, with d0_base (2026-09-30)

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

### 9.2 How much sooner (same recipe, same experience per update)

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

### 9.3 Why idea 1 learns exploit sooner: the inputs themselves (2026-09-30)

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

Consistent with §9.2: at d = 0 (no gate needed) idea 1 is ~2× sooner -- (1)
and (2); at d = 10, (3) adds and the gap grows to 2.5–3×. Correlational.
**Causal test**: Agent-HaSH task3r with `--no-input_hopfield_raw` (unit q,
still 0 when empty). If it closes most of the d = 0 gap, the scale is the cause.

### 9.4 Bug found: an empty Hopfield read a spurious q (fixed 2026-10-01)

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
had stored (the §9.1 d = 0 rows for d0_base / task3r). **Not affected**:
d = 10 evals (no empty rows), d0_base's training (one shared Hopfield per
rollout), `agenthash` continual, batched nav / expl evals, and idea 1.

So §9.1–9.2's Agent-HaSH task numbers carry it. Reruns on the fixed code:
`task3r_k2_h128_distal` (24504515) and the unit-q + ‖q‖ variant
`task3r_k2_h128_distal_qmag` (24504516; `--no-input_hopfield_raw
--input_q_magnitude`, the Hopfield analog of idea 1's (d, c)).

### 9.5 Fixed-code head-to-head, and the causal test (2026-10-02)

All on the fixed code (§9.4), `task3r_k2_h128` at `distal_amp 1`, one seed each:
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
   (its gate) into the vector and shrinks 3× toward the goal (§9.3).
2. **At d = 0 the end state ties**: unit q + ‖q‖ 11.68 vs idea 1 11.60.
3. **With 10 distractors idea 1 is still best at the end** (11.5 vs 12.4 vs
   13.2) and chases foreign goals least before storing (0.007 vs 0.042 /
   0.023). This residual is the memory: near the goal ‖q‖ cannot tell the own
   goal from a distractor (presence AUC 0.67 within 2 cells, §9.3), while the
   argmax store's `c` separates them at 0.99.
4. **Search: raw-q Agent-HaSH is the best searcher** (found 0.65 / 0.62 vs
   0.54–0.60). Idea 1's search also collapsed once (u4000) -- one seed.
