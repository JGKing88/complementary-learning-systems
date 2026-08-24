# Phase 2 — measure the signal before training on it

Phase 1 (`docs/EXPERIMENTS_NAV_TRI.md`) produced one model that scores on all
three metrics and, at the end, a diagnosis: the failures split into two modes
with opposite causes, and only one of them is a policy problem. Phase 2 takes
that seriously and **measures the signal first**. Three analyses establish what
is knowable; three training runs then establish what is reachable.

Read §0 to resume. Read §4 before launching anything — there is a train/eval
mismatch that must be fixed first.

---

## 0. Where I am

| | |
|---|---|
| **Status** | Spec written. Nothing launched. §4 fixes are blocking. |
| **Branch / worktree** | `nav-tri-metric` at `.claude/worktrees/nav-tri-metric` |
| **Predecessor** | `docs/EXPERIMENTS_NAV_TRI.md` — read its §0 findings 1–22 |
| **Open decisions** | §11 — four forks put to Jack; spec assumes the recommended default in each |
| **Still running from phase 1** | `w7_x_matched` (job 21033482), `--q_scale` dose response (21032524) |

**Workstream order.** P1/P2/P3 are analysis and run concurrently — none needs a
trained model. P4/P5 are single training runs that can start as soon as §4
lands. P6 needs P3's classifier, so it goes last.

| | workstream | needs | gate |
|---|---|---|---|
| **P1** | where `q` fails to point at the goal | scaffold only | none |
| **P2** | is relative displacement decodable from sensory? | scaffold only | none |
| **P3** | ideal observer for "is the goal in memory?" | scaffold only | none |
| **P4** | exploit specialist → ceiling | §4 | none |
| **P5** | explore specialist → ceiling (calibration) | §4 | none |
| **P6** | interleaved | P3 classifier | P3, P4, P5 |

---

## 1. The ask

Jack, 2026-08-24. Organized, not paraphrased away.

1. **Robustly characterize where `q` fails.** For distractors 0–10, high-sample
   measurement of how close `q` is to the true goal direction, and how often it
   points at a *known distractor* instead. Distributions over locations within
   an env, over envs, over seeds. **Make plots.**
2. **Sensory analysis, new question.** After training on many envs to learn the
   geometric structure: in a *new* env, given two sensory inputs, can the
   vector between them be decoded? And with some experience in the new env?
3. **A thorough ideal-observer analysis for explore vs exploit.** Under exactly
   what conditions is there signal about whether the goal is in memory — and
   what *probing behaviour* is needed or helpful to get that signal? Jack's
   candidate cues: (i) decreasing `‖q‖`, (ii) consistent `q` direction,
   (iii) something in the sensory input — and if not, should sensory input be
   structured differently?
4. **Train an exploit model to its ceiling**, to learn in isolation whether the
   Hopfield signal is good enough to exploit well even with distractors.
   `follow_q` should be *really* high — the model should essentially always be
   following `q` — with low `mean_steps` and high `success_rate`. Explicit
   failure-mode analysis: stuck on walls, etc.
5. **Train an explore model to its ceiling.** Probably easy; compare against
   the calculated ceiling. Record in this doc either way.
6. **Then interleaved training.** Metrics that matter: `mean_steps`,
   `success_rate`, `follow_q`, whether the model tracks optimal exploit
   behaviour, coverage / exploration-type analysis, `chase_q` (whether the
   agent follows `q` during explore — what `q` points at does not matter), some
   way to see whether the agent is collecting the information it needs to
   disambiguate (leans on the ideal observer), and how much interleaving costs
   against the two specialist ceilings. On phase 1's corner-trap mechanism:
   understand **when and why during training** it happens.

**Two setup changes**, both Jack's:

- **Bound step size to [0.5, 2].** Massive steps are unrealistic anyway.
- **Add `--input_prev_action`.**

Everything from phase 1 §1.2 that was fixed by instruction stays fixed: the
`ur_loss2_repel_low/029` encoder, RNN + ReLU, `goal_radius` 1, no step
normalization, raw Hopfield input, `wall_resolution` 4, `observation_size` 60,
200-step rollouts, `explore_ends_on_goal`, `reset_state_on_teleport` off, no
env refreshing, and **never** the `goal_stored_in_memory` input bit.

---

## 2. What changed, and three findings that reshape the plan

### 2.1 Step bounds are already implemented — and they bracket the explore optimum

`--min_action_norm` / `--max_action_norm` exist and are plumbed end to end
(`config.py:69-74`, `train_navigate.py:1073-1081`, `world/env.py:566-569`,
`world/vec_env.py:377-389`). `continuous_scale` is 1.0, so **action norm is in
grid cells** and `[0.5, 2]` means what it says. ε-actions are unit magnitude,
so they sit inside the band untouched. No new code — a launcher change.

The bound is better-chosen than it looks. Billiard coverage as a function of
step magnitude, recomputed for this phase
(`coverage_baselines.billiard_cells_per_step`, size 20, 200 steps, 128 trials):

| `‖a‖` | 0.5 | 0.8 | 1.0 | 1.5 | **2.0** | 2.5 | 3.47 |
|---|---|---|---|---|---|---|---|
| cells/step | 0.494 | 0.669 | 0.754 | **0.757** | 0.715 | 0.639 | 0.616 |
| coverage @200 | 0.247 | 0.334 | 0.377 | **0.378** | 0.357 | 0.319 | 0.308 |
| ideal `mean_steps` | 19.7 | 12.3 | 9.8 | 6.6 | **4.9** | 3.9 | 2.8 |

**Coverage is non-monotonic in step size and peaks at `‖a‖ ≈ 1.0–1.5`, inside
the band.** Big steps skip cells. So:

- The bound costs explore nothing — it contains the optimum.
- It caps exploit at ideal `mean_steps` ≈ 4.9 rather than 2.8. Realistic, and
  still far below anything measured.
- **Phase 1's best model ran at `‖a‖ = 3.47`, past the coverage optimum.** Its
  coverage 0.315 against a billiard line of 0.308 at that magnitude means it
  was not under-covering — it was *over-striding*, paying coverage for stride
  to win `mean_steps` (phase 1 finding 22).

That last point is the real consequence: **part of phase 1's explore/exploit
tension was an artifact of unbounded step size.** Once stride cannot be traded
for `mean_steps`, the frontier changes shape and has to be re-measured. No
phase-1 coverage or `mean_steps` number is comparable to a phase-2 one.

### 2.2 `INIT_LOG_STD`'s job changes completely

Phase 1 finding 1 — the largest single result of that phase — was that σ is the
only channel through which the policy learns step *magnitude*, which started at
0.086 against an optimum of 1.0. `min_action_norm = 0.5` sets a floor in the
env, so:

- The wave-1 pathology is gone at initialization. `‖a‖ ≥ 0.5` from step one.
- σ's remaining role is mostly **angular** noise, not magnitude.
- **The σ = 0.50 optimum does not transfer.** It was selected under a regime
  where σ *created* magnitude. Expect the new optimum lower. Re-bracket it.

### 2.3 A blocking train/eval mismatch

`make_vec` (`world/vec_env.py:436`) does not accept `min_action_norm` /
`max_action_norm` and cannot forward them. The training collector builds
`ContinuousVecEnv` **directly** and passes both (`rollout/collector.py:133-134`),
but every evaluation path goes through `make_vec`:

- `evaluation/batched.py:83` and `:221` — the batched nav and explore evals
- `analysis/nav_tri/behavior_probe.py:94` and `:221`
- `evaluation/rnn.py:29`

`make_env`, the single-env factory, *does* pass them (`world/env.py:619-620`),
so the gap is exactly the batched path — which is what training-time eval and
every probe in `analysis/nav_tri/` use.

**Set the bounds today and training clamps while every reported number does
not.** Silent, and it would poison every comparison in this phase. Fixed in §4
before anything runs.

---

## 3. What phase 1 established that constrains this phase

Full list in `EXPERIMENTS_NAV_TRI.md` §0. The ones that bind here:

1. **Two exploit failure modes, same cost, opposite cause** (findings 20–21).
   Mode A trusts a broken readout (`follow_q_fail` high, `q_accuracy_fail`
   low) — encoder-limited. Mode B ignores a usable one (`follow_q_fail` ≈ 0,
   `q_accuracy_fail` 0.437) — regime-detection-limited, a policy problem.
   **P3 exists to put a ceiling on how much of mode B is even detectable.**
2. **`success_rate` cannot distinguish them** — both models miss about the same
   fraction at d=10. Every failure claim in this phase must carry
   `follow_q_fail` *and* `q_accuracy_fail`.
3. **No `dir_acc` number is interpretable without its world** (finding 19). Two
   worlds gave 1.8% and 23.3% of cells below cos 0.5 at d=10 — a 13× swing that
   8 envs does not average out. **P1 is the fix: distributions over many
   worlds, never a mean over eight.**
4. **Signal-separability needs ≥8 independent distractor draws** (finding 18).
   Two draws produced two different wrong conclusions.
5. **`MOVE_ENT_COEF` is a no-op under `FREEZE_LOG_STD=1`** (finding 3).
6. **No exploit conclusion before ~500 updates** (finding 16); the eval
   oscillates between `mean_steps` 17 and 159 at a fixed eval seed.
7. **Interleave in the same PPO update** (finding 11). Explore-first,
   exploit-first and blocked all collapse. P6 starts from interleaving.
8. **20 envs × 64 batch ≈ 80 × 16 per update at 2.9× less wall-clock**
   (finding 9). Keep 20 × 64.
9. **`WALL_PENALTY` must not be raised** (finding 5) — the perimeter is 19% of
   the arena and must be visited.

### 3.1 Reference ladder for this phase

| quantity | value | source |
|---|---|---|
| coverage hard ceiling, 200 steps | 0.5025 | one new cell per step |
| coverage billiard ceiling, `‖a‖ ∈ [0.5,2]` | **0.378** at `‖a‖ ≈ 1.25` | §2.1 |
| coverage lawnmower ceiling | → 0.5025 | needs position knowledge — **P2 decides if this reopens** |
| mean start distance | 10.85 | phase 1 probe |
| ideal `mean_steps` at `‖a‖ = 2` | **4.9** | (10.85 − goal_radius) / 2 |
| ideal `mean_steps` at `‖a‖ = 1` | 9.8 | same |

**P2 sets which coverage ceiling is the target.** Billiard is reactive — it
needs only "am I about to hit a wall". Lawnmower needs to know where it has
already been. Phase 1's P0.9 found absolute position decodable from the sensory
cone at only R² ≤ 0.13 and called 0.387 the practical ceiling. **Jack's question
2 asks about *relative* displacement between two observations, which is a
different and probably much easier quantity.** If it decodes well, the agent can
path-integrate from sensory, the lawnmower line reopens, and the explore target
moves from 0.378 toward 0.50.

---

## 4. Blocking fixes — land these before any run

- [ ] **B1. `make_vec` must forward the action-norm bounds.** Add
      `min_action_norm` / `max_action_norm` parameters, pass them into
      `ContinuousVecEnv`, and update all five call sites to read
      `cfg.env.min_action_norm` / `cfg.env.max_action_norm`. Regression test:
      build a vec env from a config with bounds set, step it with an action of
      norm 5, assert the realized displacement has norm 2.
- [ ] **B2. Assert train/eval agreement.** A test that walks one config through
      both `make_env` and `make_vec` and asserts the movement bounds match.
      This class of bug has now appeared once; the test is what stops it
      recurring silently.
- [ ] **B3. Phase-2 launcher block** — `MIN_ACTION_NORM=0.5`,
      `MAX_ACTION_NORM=2.0`, `INPUT_PREV_ACTION=1`, plus pass-throughs. The
      phase-1 launcher pins `INPUT_PREV_ACTION=0`.
- [ ] **B4. Decide `prev_action` semantics** — §11 Q1. The collector currently
      feeds `result["move_action"]`, the *committed* action
      (`rollout/collector.py:652`), which differs from the realized displacement
      whenever the norm clamp or the arena clip bites.

---

## 5. P1 — where does `q` fail to point at the goal?

**Question.** Over distractors 0–10, what is the distribution of the angle
between `q(x)` and the true goal direction, and how often does `q` point at a
*known distractor* instead? Distributions, not means — finding 19 says a mean
over 8 envs is not a measurement.

**Method.** Purely a property of the scaffold and encoder; no policy involved.
For each env, each cell `x`, with the goal stored plus `k` distractors drawn
from other envs, compute `q(x) = W_x·(recall(x) − x)` and record:

| statistic | definition |
|---|---|
| `dir_err(x)` | angle between `q(x)` and `goal − x` |
| `qnorm(x)` | `‖q(x)‖` |
| `lock_target(x)` | `argmin` over {goal} ∪ {distractors} of angle to `q(x)` |
| `distractor_lock(x)` | `lock_target ≠ goal` — Jack's "does it point at a known distractor" |
| `lock_margin(x)` | angle gap between best and second-best target |

**Sampling axes** — the point is robustness, so each is a real axis, not a
nuisance:

| axis | levels |
|---|---|
| distractor count | 0,1,2,3,4,5,6,7,8,9,10 |
| cell within env | all 400 |
| env draw | 32 |
| distractor draw | 8 per (env, count) |
| scaffold / codebook seed | 4 |

≈ 4.5 M recalls, batched on GPU. Encoder held fixed at the one specified by
instruction.

**Plots** (`analysis/nav_p2/q_failure_map.py`, PNGs under `results/nav_p2/`):

1. Per-env arena heatmap of `dir_err`, goal and distractors marked — where in
   space does it break?
2. CDF of `dir_err` by distractor count, one band per env — the finding-19
   plot, showing between-world spread explicitly.
3. `distractor_lock` rate vs distractor count, with per-env spread.
4. `dir_err` vs distance-to-goal, and vs distance-to-wall.
5. `qnorm` distributions, goal-present vs goal-absent, by distractor count —
   the separability picture that phase 1 measured at only 8 envs.

**Decision rule.** If `distractor_lock` is rare (<5%) and `dir_err` is a small
concentrated tail, mode A is a *localized* defect and an exploit specialist
should approach the ideal `mean_steps` of 4.9. If it is broad, mode A is a hard
ceiling and P4's result is predicted before it runs — which is the point of
running P1 first. Feed the measured distribution through
`analysis/nav_tri/exploit_reference.py` to turn it into a predicted
`mean_steps`, and treat P4 as the test of that prediction.

### 5.1 RESULT — the readout is good, and phase 1's headline about it was one world

Three scaffold seeds x 64 fresh envs x 8 distractor draws x all 400 cells, at
the real Npos=1716. At ten distractors, the three seeds agree closely:

| seed | lock=goal | lock=distractor | lock=mixture | `dir_cos` given lock=goal | frac `dir_cos`<0.5 |
|---|---|---|---|---|---|
| 0 | 0.9944 | 0.0021 | 0.0034 | 0.9653 | 0.0144 |
| 1 | 0.9912 | 0.0004 | 0.0084 | 0.9619 | 0.0188 |
| 2 | 0.9865 | 0.0020 | 0.0114 | 0.9644 | 0.0143 |

**The Hopfield locks onto the goal at ten distractors 98.7-99.4% of the time,
and locks onto a distractor 0.04-0.2% of the time.** Jack's question -- how
often does `q` point at a known distractor instead of the goal -- has the answer
*almost never*. What little failure there is at ten distractors is mostly the
third category, a spurious mixture belonging to no stored pattern (0.3-1.1%).

And when it does lock on the goal, `dir_cos` is 0.962-0.965, a mean error of
about 15 degrees, essentially flat from d=1 to d=10.

**This overturns the phase-1 statement that the encoder points ~46 degrees
wrong at ten distractors.** That came from a mean `dir_acc` of 0.696 on a single
eval world of 8 envs. The per-env spread here shows why: the median env has
0.4-0.9% of cells below cos 0.5, and the worst has 9.8-12.0%. Worlds like the
one phase 1 measured exist; they are the tail, not the typical case, and an
8-env mean cannot tell which one it drew. Finding 19 said no `dir_acc` number is
interpretable without its world -- this is that, quantified.

Separability degrades as expected but less than phase 1 reported: the
goal-present / goal-absent `|q|` ratio falls 4.66 -> 2.42 from one distractor to
ten, against phase 1's 4.8 -> 1.28.

**Consequence for the whole phase.** §3 item 1 inherited phase 1's conclusion
that mode A -- trusting a broken readout -- is encoder-limited and the dominant
constraint. If `q` locks correctly 99% of the time and points within 15 degrees
when it does, **mode A cannot be what caps exploit.** Either the failures are
concentrated in the small bad tail, in which case P4 should reach close to the
ideal 4.9 steps on most trials and fail hard on a few, or they are not
readout failures at all. P4 is now a sharper test than it was designed to be:
the readout has been measured, so a shortfall against `exploit_reference`'s
prediction cannot be blamed on the encoder.

### 5.2 Figures, and two things only the figures showed

`analysis/nav_p2/q_failure_plots.py` -> `results/nav_p2/figs/`. Six figures:
lock outcome, between-world spread, the lock decomposition, arena maps,
geometry, and `|q|` separability.

Two findings came out of drawing the data that the summary table had hidden.

**1. CORRECTED — direction error is governed by recall fidelity, not by
geometry.** The first version of this section said the bad readout was "a halo
around the goal" and explained it as a one-cell displacement projecting to
mostly noise. Jack asked what that meant. It does not mean anything: a *short*
displacement is the case where the tangent-plane linearization is at its
**best**, not its worst, so the stated mechanism was backwards. Working it out
properly:

| `cos_goal` (how close the settled state is to the exact goal pattern) | share of cells | % with `dir_cos` < 0.5 |
|---|---|---|
| ≥ 0.99 | 50.1% | **0.01%** |
| 0.90 – 0.99 | 49.1% | 2.87% |
| 0.50 – 0.90 | 0.8% | 21.8% |

**99.7% of all bad-direction cells have `cos_goal` < 0.99.** When the Hopfield
settles exactly on the stored pattern the direction is essentially never wrong.
The failure is imperfect recall — the settled state is a slight admixture of
other patterns — and nothing else.

Distance enters only as a **lever arm**. `‖q‖` scales with distance to the goal
(median 0.041 at one cell, 0.393 beyond twelve), so a *fixed* amount of recall
error turns into a *larger angular* error when the true displacement is short.
That is why the failure rate is 11.7% within 1.5 cells of the goal against
0.44% beyond twelve. Holding `‖q‖` fixed and varying distance shows the same
thing from the other side: what predicts a bad direction is `‖q‖` *disagreeing*
with the true distance, in either direction, not the distance itself.

**And the halo was overstated.** The near-goal rings have a much higher failure
*rate* but contain few cells: **only 24.8% of all sub-0.5 cells lie within 2.5
cells of the goal**, and the ring from 5 to 8 cells contributes more (23.9%)
than any other. So "the bad readout is a halo around the goal" is wrong as a
description of where the failures are; the correct statement is that the rate is
highest there while the mass is spread over the arena.

What survives from the original claim is the wall half, which was measured
rather than reasoned: **`dir_cos` is flat against distance-to-wall**, so `q` is
not degraded near boundaries. That remains a negative result for the readout
half of H-wall (§7.4).

**A prediction this hands to P3.** If direction error is governed by recall
fidelity, then the agent has an *observable* proxy for it. `cos_goal` is not
available to the policy, but a pattern that has settled exactly is a **fixed
point of the recall map** and one sitting at cos 0.94 is not — which is exactly
what the group-C statistic `c1 = ‖recall(x) − recall²(x)‖` measures, and the
multistep iterates are already computed and already fed to the policy. So `c1`
should predict `dir_cos`, giving the policy a per-step confidence signal on its
own readout. That is a sharp, cheap test and it should be run early in P3.

**2. A wrong lock is not a random direction.** The decomposition CDF shows all
three groups hugging cos 1.0; a randomly-directed `q` would trace the diagonal
and none of them does. Cells classified as "spurious mixture" are mostly recalls
that sit near the goal pattern without reaching the 0.9 threshold, so they still
carry most of the direction. Even the 938 genuinely distractor-locked cells are
mostly still pointed roughly goalward.

So the `lock_thresh` of 0.9 is doing more work than the label suggests, and the
honest statement is stronger than the one §5.1 made: **there is no sizeable
population of catastrophically misdirected `q` at ten distractors at all.** The
whole tail below cos 0.5 is 1.6% of cells, spread across all three lock
categories rather than concentrated in the wrong-lock ones.

---

## 6. P2 — is relative displacement decodable from sensory input?

**Question (Jack's).** After learning geometric structure across many envs, in a
*new* env, given two sensory observations, can the vector between them be
decoded? And does a little experience in the new env help?

**Why it matters.** §3.1: it decides whether the explore ceiling is billiard
(0.378) or lawnmower (→0.50). It also gives a **`q`-independent check on
self-motion**, which §7.4's wall hypothesis needs.

**Method.** Train `f(s_1, s_2) → Δpos` on envs 1..N, test on held-out envs.

- Architectures: ridge on `[s1, s2, s1−s2, s1⊙s2]`; a 2-layer MLP. The gap
  between them says whether the structure is linear in the cone.
- Report R² and angular error, broken out by `‖Δpos‖`, by distance-to-wall, and
  by absolute position — a decoder that only works in the middle of the arena is
  a different result from one that works everywhere.
- Controls: same-env train/test (upper bound), shuffled pairs (chance).

**Adaptation to a new env**, `k ∈ {0, 4, 16, 64, 256}` steps:

- *supervised anchors* — `k` labelled `(s, pos)` pairs. Unrealistic; an upper
  bound on what adaptation could buy.
- *self-motion self-supervision* — a trajectory of `(s_t, a_t)` with unknown
  absolute position. **This is the realistic one**: the agent knows its own
  displacement, so it can fit the local `s → pos` structure with no labels. It
  is what an actual explore rollout provides for free.

**Decision rule.** If held-out-env angular error is small at `‖Δpos‖ ≈ 1–2`, the
agent can path-integrate from sensory, the lawnmower ceiling reopens, and P5's
target rises. If it needs adaptation, report how many steps — that is a direct
statement about how much of an episode must be spent localizing. If it fails
even with 256 anchors, the sensory representation is the bottleneck and Jack's
"should we structure sensory input differently?" becomes the live question.

---

## 7. P3 — ideal observer: when is there signal that the goal is in memory?

The centrepiece, and the one that bounds mode B. Jack's three candidate cues are
groups (A), (B) and (D) below; group (C) is an addition.

### 7.1 Two questions, both worth answering

- **Q_ep, episode-level** — does memory contain a pattern stored in *this* env?
  This is what "explore vs exploit" means to the agent.
- **Q_step, step-level** — is the *current* recall `recall(x_t)` the goal rather
  than a foreign pattern? Even inside an exploit episode, some positions fall in
  a distractor's basin.

Q_step is the operational one: mode B is a per-step failure to follow a usable
recall. Q_ep is the regime question. Measure both.

### 7.2 Candidate statistics — all functions of what the policy can see

**A. Magnitude and its dynamics** *(Jack's "decreasing ‖q‖")*

| | statistic |
|---|---|
| `a1` | `‖q_t‖` |
| `a2` | `Δ‖q_t‖ = ‖q_t‖ − ‖q_{t−1}‖` |
| `a3` | **self-motion residual** `r_t = ‖q_t‖ − ‖q_{t−1}‖ + ⟨d_{t−1}, q̂_{t−1}⟩` |
| `a4` | running mean and variance of `r` over the episode |

`a3` is the sharpened version of Jack's cue. A real stored target sits at a
fixed location, so moving `d` toward it shrinks `‖q‖` by exactly the projection
of `d` on `q̂`, and `r ≈ 0`. A phantom's recall *moves with you* as the basin
shifts, so `r` is large. **This is why `prev_action` matters, and why it should
be the realized displacement** (§11 Q1): `d` is what actually happened, not what
was commanded.

**B. Direction consistency** *(Jack's second cue)*

| | statistic |
|---|---|
| `b1` | `cos(q_t, q_{t−1})` |
| `b2` | **allocentric spread** — with `ĝ_t = Σ_{s<t} d_s + q_t`, the trace of the covariance of `{ĝ_s}_{s≤t}` |
| `b3` | drift rate `‖ĝ_t − ĝ_{t−1}‖` |

`b2` is the 2-D version of `a3` and strictly stronger than `b1`: for a real goal
`ĝ` is *constant*, so its spread is a direct test. `b1` can look consistent
while `ĝ` drifts steadily.

**C. Hopfield fixed-point quality** — *not on Jack's list, and likely the
strongest static cue*

| | statistic |
|---|---|
| `c1` | `‖recall(x) − recall²(x)‖` |
| `c2` | `cos(q^{(1)}, q^{(3)})` — do multistep recalls agree? |

A genuinely stored pattern is a **fixed point** of the recall map; a spurious
mixture is not. This is the classic spurious-state detector, it needs no
movement at all, and `--input_hopfield_multistep 1 2 3` already computes the
iterates — so the policy is *already being fed the raw material* and it costs
nothing to measure. **If `c1` separates well at a single step, regime detection
needs no probing and Jack's probing-behaviour question largely dissolves.**
Worth knowing early either way.

**D. Sensory consistency** *(Jack's third cue)*

| | statistic |
|---|---|
| `d1` | residual between the observation predicted by the recalled pattern and the actual observation |
| `d2` | mismatch between P2's sensory-decoded displacement and the commanded action |

`d1` needs a `pattern → obs` decoder, fitted offline; it is a property of the
scaffold, not the agent. `d2` falls out of P2 and is the wall diagnostic.

### 7.3 The analysis

1. **Per-statistic separability** — AUC for Q_ep and Q_step as a function of
   distractor count {0,1,2,3,5,7,10} and steps observed
   `t ∈ {1,2,4,8,16,32,64}`, reported as a **distribution over envs and seeds**,
   per finding 19.
2. **The ideal-observer bound** — a classifier (logistic regression, plus
   gradient-boosted trees for the nonlinear ceiling) on the full feature vector,
   cross-validated across *held-out envs*. Yields `AUC(t, n_dist)`. **This is
   the number that bounds mode B.** If AUC at `t = 3` is 0.99 the policy's
   regime failures are inexcusable and P6 should fix them; if it is 0.65 the
   policy is near-optimal and the fix must be elsewhere.
3. **Feature ablation** — drop each of A/B/C/D; the AUC loss is that group's
   unique contribution. Answers which cue actually carries the discrimination,
   which is Jack's question 3 restated.
4. **Probing behaviour** — the same classifier, but on trajectories generated by
   fixed probe policies:

   | probe | rationale |
   |---|---|
   | `still` | information floor with minimal motion (`‖a‖ = 0.5` forced) |
   | `straight` | constant heading |
   | `billiard` | the reactive explore baseline |
   | `along_q` | maximizes expected `‖q‖` change — best for group A |
   | `perp_q` | maximum parallax — best for group B |
   | `anti_q` | control |
   | `random` | control |

   Report `AUC(t)` per probe, **steps to reach AUC 0.95**, and information per
   unit distance travelled — probing has a coverage cost, so the comparison is
   only fair per unit of motion. This answers "what probing behaviour is needed
   or helpful" with a number instead of an intuition.
5. **Score the trained agents' own trajectories** with the frozen classifier.
   This *is* Jack's "is the agent collecting the information it needs" metric
   for P6, and it costs one extra evaluation pass.
6. **Wall interaction** — condition everything on distance-to-wall and on
   whether the step was clipped. See §7.4.

**Deliverables.** `AUC(t)` curves per cue per distractor level with per-env
bands; a steps-to-0.95 table per probe; the ablation table; and a single
headline number — the ideal-observer AUC at the distractor levels we train on,
which every mode-B claim in P6 gets measured against.

### 7.4 A hypothesis this analysis is built to test — H-wall

`prev_action` currently carries the **committed** action
(`rollout/collector.py:652`), not the realized displacement. The two differ
whenever the arena clip bites — and, once §2.1's bounds are on, whenever the
norm clamp bites. But `a3` and `b2`, the two sharpest regime cues, both need the
*realized* displacement: they compare how much `q` changed against how far you
actually moved.

**So near a wall the agent's best regime cue is corrupted, in the direction of
declaring a real goal a phantom.** Phase 1 measured `fail_frac_at_edge = 0.389`
for the combined model — 39% of its mode-B failures end against a wall, which is
what this predicts. Test: ideal-observer AUC conditioned on distance-to-wall and
on clip events, with and without realized displacement in the feature set. If it
holds, §11 Q1 is not a preference — it is the fix.

---

## 8. P4 — exploit specialist to its ceiling

**Question.** Is the Hopfield signal good enough, even with distractors, for a
model that does nothing else to exploit really well?

**Target.** `follow_q` as high as it goes — the model should essentially always
follow `q`. `mean_steps` toward the ideal 4.9 at `‖a‖ = 2`. `success_rate` high.
*(Jack's message says "success rate should be low"; read as high, since low
`mean_steps` and high success are the exploit goals throughout.)*

**Predicted before it runs.** P1's `dir_err` distribution through
`analysis/nav_tri/exploit_reference.py` gives a predicted `mean_steps` per
distractor level. P4 tests that prediction — agreement means the readout is the
binding constraint and mode A is understood; a large shortfall means the policy
is leaving signal on the table.

**Config.** Phase-1 `w7_x_matched` plus §4: `exploit:2000`, 20 × 64 × 200,
bounds `[0.5, 2]`, `--input_prev_action`, `GOAL_REWARD=2.0`. σ **re-bracketed**
per §2.2 — the phase-1 optimum does not transfer. Bracket `INIT_LOG_STD` over
{−1.8, −1.2, −0.7} at minimum.

**Failure-mode analysis** — explicit, per Jack, and split by phase-1 finding 20:

| | statistic |
|---|---|
| mode A vs mode B split | `follow_q_fail` × `q_accuracy_fail` |
| stuck on walls | `fail_frac_at_edge`, `clip_frac`, time-at-wall |
| corner trap | occupancy of the four corner cells vs uniform |
| distractor lock | fraction of failures ending within `goal_radius` of a distractor — P1's `lock_target` applied to trajectories |
| oscillation | sign changes of `⟨d_t, q̂_t⟩`, i.e. approach/retreat cycling |
| give-up | `follow_q` late-bin collapse (phase-1 per-step bins) |

The last two are new. Phase 1 found the specialist walks into a phantom and
*stops* 5.6 away; whether that is a stall, an oscillation or a give-up is not
yet distinguished, and each implies a different fix.

---

## 9. P5 — explore specialist to its ceiling (calibration)

**Jack asked whether this is already done and can be skipped. The science is
done; the number is not safely reusable; the run is cheap.**

What phase 1 established: an explore-only arm at `‖a‖ = 0.796` reached
`strategy_efficiency = 0.735` with `edge_frac` 0.167 *below* the 0.19 uniform
occupancy, `clip_frac` 0.086, and `chase_q ≈ −0.016` — i.e. **the explore-side
distractor problem is solved**; coverage at ten distractors (0.2477) equalled
coverage at zero (0.2468). That result stands and does not need redoing.

Three reasons not to skip the run:

1. **The 0.385 figure is ambiguous in my own notes.** It appears in
   `EXPERIMENTS_NAV_TRI.md` §3.2 as the *scripted billiard baseline at ε = 0*,
   and in §0 as "the explore-only specialist". I cannot tell them apart from the
   document. A phase where every training result is scored against the explore
   ceiling cannot rest on a number of uncertain provenance.
2. **The step bounds move the ceiling** (§2.1) and re-scale the whole coverage
   axis. Phase-1 coverage numbers are not comparable to phase-2 ones.
3. **P2 may reopen the ceiling entirely** (§3.1). If relative displacement
   decodes, the target moves from 0.378 toward 0.50 and "at ceiling" means
   something different.

So P5 runs as a **calibration and reference run, not a search**: one arm at the
σ P4 selects, bounds on, `--input_prev_action`, long enough to plateau. Expect
0.36–0.38. Deliverable: a clean, unambiguous explore ceiling under phase-2
settings, with `strategy_efficiency` beside it so "the agent moves too slowly"
and "the agent moves badly" stay separable.

---

## 10. P6 — interleaved

Runs last; needs P3's frozen classifier. Starts from phase-1 finding 11
(interleave within the same PPO update; explore-first, exploit-first and blocked
all collapse) and finding 14 (select by `joint_curve.py`, not the last
checkpoint).

**Metric panel**, Jack's list plus what phase 1 says is needed to read it:

| | metric | reads |
|---|---|---|
| exploit | `mean_steps`, `success_rate`, `follow_q` | against P4's ceiling |
| exploit | `follow_q_fail` × `q_accuracy_fail` | **mode A vs mode B** — finding 20 |
| exploit | per-step `follow_q` bins | lock-on latency; the bimodal signature |
| exploit | tracking optimal behaviour | `path_efficiency`, `step_mag_mean`, and `dir_err`-predicted `mean_steps` from P1 |
| explore | coverage, `strategy_efficiency`, `cells_per_step` | against P5's ceiling |
| explore | `chase_q` | following `q` during explore — what `q` points at does not matter |
| explore | `edge_frac`, `clip_frac`, corner occupancy | the collapse mode |
| both | **ideal-observer AUC of the agent's own trajectories** | is it collecting the information it needs? — P3 item 5 |
| both | interference = specialist ceiling − interleaved, per metric | the cost of sharing one policy |

**The "when and why does the corner trap appear" question needs new
instrumentation.** Phase 1 could not answer it because eval runs every 25–50
updates and does not split by regime, so the collapse was only ever seen after
the fact. Add **per-update, per-regime rollout diagnostics**: `chase_q`,
`follow_q`, `edge_frac`, `clip_frac`, `‖a‖`, and the ideal-observer AUC of the
rollouts themselves, logged every update and split explore/exploit. Cheap —
these are already computed inside the rollout — and it turns the corner trap
from a post-hoc diagnosis into a time series with an onset.

Phase 1's mechanism to confirm or refute: exploit installs persistent
`q`-following; in an explore rollout `q` points at distractors; the agent drives
into a wall (`edge_frac` 0.82, `clip_frac` 0.65). The prediction is that
`chase_q` rises *before* `edge_frac` does, and that both move within a few
updates of an increase in exploit weight.

**Design arms** — see §11 Q4 for which to prioritize.

---

## 11. Decisions — resolved 2026-08-24

Four forks put to Jack; all four answered. Recorded here because each one
changes what the code does, not just what the spec says.

- **Q1. `prev_action` semantics → BOTH channels.** The policy gets the
  committed action *and* the realized displacement as separate inputs. The
  committed action alone leaves H-wall (§7.4) untestable from the policy side;
  the realized displacement alone throws away the fact that a clip *happened*,
  which is itself information — a clip means a wall is there. Two extra input
  dimensions. See B4.
- **Q2. Auxiliary regime head → analysis probe only, NOT in the loss.** Fit a
  frozen probe on the policy's hidden state to measure how much regime
  information it already carries, outside the objective. The ground-truth
  "goal is in this env's memory" bit never enters training, as an input or a
  target. So regime detection stays fully emergent and P3's ideal-observer AUC
  is the only ceiling we get — which makes P3 load-bearing rather than merely
  informative.
- **Q3. Probe scope → scripted menu, agents' own trajectories, AND a learned
  prober.** The fullest option. The scripted menu answers "what movement buys
  information fastest"; scoring the trained agents' trajectories with the
  frozen classifier is P6's "is it collecting what it needs" metric; and the
  learned prober — a small policy trained to maximize the classifier's
  log-likelihood — says whether a better probe exists than the seven we thought
  of. If the learned prober beats every scripted probe by a wide margin, that
  is itself a finding: the information is there but not in a form any simple
  heuristic extracts.
- **Q4. Interleave arms → baseline plus distractor curriculum.** (a) fixed
  `empty_frac` with shuffled assignment, re-run under the new bounds, as the
  reference; (b) a curriculum ramping distractors 0 → 10. Not the auxiliary
  head (excluded by Q2) and not the ideal-observer-gated variant.

**Consequence of Q2 worth stating plainly.** With no supervised regime signal,
the only lever on mode B is what the reward and the curriculum make learnable.
That raises the stakes on P3: if the ideal-observer AUC is high and the trained
agent's own trajectories score far below it, the gap is a *learning* failure
and the curriculum (Q4b) is the intended fix. If the AUC itself is low, mode B
is not a policy failure at all and the phase-1 conclusion in §3 item 1 needs
revising.

**Also flagged, not blocking.** Jack asked "if not, should we structure sensory
input differently?" — that is downstream of P2. If relative displacement does not
decode from the current 60-ray cone, the input-representation question becomes
live, and it is a bigger change than any knob in this document. Deferred until
P2 reports.
