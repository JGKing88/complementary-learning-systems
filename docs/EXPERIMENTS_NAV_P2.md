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
| **Running** | P4 exploit ceiling `p4_x` / `p4_x_s12`, P5 explore calibration `p5_e` |
| **Done** | §4 blocking fixes, P1 (§5) with figures, and the recall-mechanism thread §5.3-5.7 |

**Open items** (priority order):

- [ ] **Read P4/P5 when they land** and fill §8/§9. P4 at u450 was already at
      `mean_steps` 10.09 (d=0) / 14.10 (d=10) with `mean_speed` 1.25, so the
      relevant ideal is ~7.9 steps rather than 4.9 — speed is the larger lever.
- [ ] **Test the modern-Hopfield substitution BOTH ways before adopting it**
      (§5.7). It gives exact retrieval of the continuous patterns — all 11 as
      fixed points, a cos-0.70 cue restored to 1.0000, no spurious states —
      which should collapse the direction-error tail. But the explore side
      depends on `|q|` being *small* when only distractors are stored, and that
      separability currently comes from the recall being a blurry blend. A sharp
      retrieval could give a confident `q` pointing at a phantom and drive
      `chase_q` up, which is the phase-1 corner-trap mechanism. Run
      `q_failure_map` with the modern update and compare `dir_cos` **and**
      goal-present/goal-absent `|q|` separability. Not a recommendation until
      both are measured.
- [ ] **Fix `hopfield/core.py`'s docstring** — "clean memory attractors converge
      in 1-2 steps; diffuse landscapes wander" is false (§5.4).
- [ ] **P2 and P3 not started.** P3 is the critical path for P6, and §5.3-5.7
      changed what its group-C statistics mean.
- [ ] `signal_separability` draws **one** distractor set per (env, count).
      Raise it — that single fact produced the 23.3% that misdirected §6.7.1 of
      the phase-1 doc (§5.2.1).

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

## 4. Blocking fixes — ALL LANDED 2026-08-24

18 new tests in `hopfield_nav/tests/test_action_norm_bounds.py`; the 213-test env suite passes; a smoke run confirms `input_dim` 74 with both prev-action channels live and `cells_per_step` starting at 0.347 where phase 1's untrained policy sat near 0.05.

- [x] **B1. `make_vec` must forward the action-norm bounds.** Add
      `min_action_norm` / `max_action_norm` parameters, pass them into
      `ContinuousVecEnv`, and update all five call sites to read
      `cfg.env.min_action_norm` / `cfg.env.max_action_norm`. Regression test:
      build a vec env from a config with bounds set, step it with an action of
      norm 5, assert the realized displacement has norm 2.
- [x] **B2. Assert train/eval agreement.** A test that walks one config through
      both `make_env` and `make_vec` and asserts the movement bounds match.
      This class of bug has now appeared once; the test is what stops it
      recurring silently.
- [x] **B3. Phase-2 launcher block** — `MIN_ACTION_NORM=0.5`,
      `MAX_ACTION_NORM=2.0`, `INPUT_PREV_ACTION=1`, plus pass-throughs. The
      phase-1 launcher pins `INPUT_PREV_ACTION=0`.
- [x] **B4. Decide `prev_action` semantics** — §11 Q1. The collector currently
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

### 5.2.1 CORRECTION — the 23.3% was draw variance, not a bad world

Both phase-1 worlds were re-measured with this tool, replaying their recorded
placements rather than drawing fresh ones:

| world | phase-1 value | re-measured here | envs | draws/env |
|---|---|---|---|---|
| `w6_pers` ("world A") | 1.8% | **1.31%** | 6 | 8 |
| smoke ("world B") | **23.3%** | **1.03%** | 2 | 8 |

Same scaffold (Npos 1716), same encoder, same beta, same recorded world loaded
from `world.json`. The only methodological difference is that
`signal_separability` draws **one** distractor set per (env, count) — two draws
total across its two envs — while this draws eight per env.

**So 23.3% was a two-draw fluke, and the "13x between-world variance" in
`EXPERIMENTS_NAV_TRI` §6.7.1 attributed it to the wrong axis.** Recorded
placement is not systematically worse than fresh either: `w6_pers` at 1.31%
sits inside the fresh-world distribution (pooled 1.43-1.88%, per-env median
0.41-0.94%).

Between-world variance *is* real — the per-env spread at ten distractors runs
from ~0 to 12% across 192 environments — but it is several times smaller than
the number that was used to argue for it. Phase-1 finding 19 should read: **no
`dir_acc` number is interpretable without its world *and its draw count*, and
the draw count is the larger term.** That is phase-1 finding 18 again, which
had already been established after two earlier wrong conclusions from
two-draw measurements; this is the third time.

### 5.3 The recall does not converge — it drifts. Two corrections.

Jack asked whether a spurious fixed point is still a fixed point. It is, and
that broke the argument in §5.2's closing prediction, so the question was
tested rather than argued (`analysis/nav_p2/recall_convergence.py`, 25,600
cells x 12 recall steps at ten distractors).

**The answer is neither branch that question offered** — but the first reading
of it, "the recall never reaches a fixed point", was also wrong, and §5.4
corrects it. It converges fine; twelve steps was simply the transient. What the
twelve-step window shows is that iterating walks steadily *away* from the goal
pattern. For the cleanest cells (cos ≥ 0.99 after one step):

| recall steps | cos to goal | `dir_cos` | % `dir_cos` < 0.5 |
|---|---|---|---|
| 1 | 0.995 | 0.998 | **0.00%** |
| 3 | 0.959 | 0.996 | 0.22% |
| 5 | 0.899 | 0.995 | 0.62% |
| 8 | 0.801 | 0.990 | 1.75% |
| 12 | 0.615 | 0.982 | **5.54%** |

Pooled over all cells, direction quality goes from **1.46% bad at one step to
12.27% bad at twelve**, and 100% of the imperfect-after-one-step cells are
still moving at step 12 (step residual 0.04-0.08, never settling).

**Correction 1 — the group-C rationale in §7.2 is wrong.** That section
justified `c1 = ‖recall(x) − recall²(x)‖` as a spurious-state detector on the
grounds that "a genuinely stored pattern is a fixed point of the recall map; a
spurious mixture is not." Both halves are false here. This is a **classical
Hebbian Hopfield with tanh** (`X ← normalize(tanh(β W X))`), where mixture
states *are* genuine attractors — Jack's point — and in this operating regime
nothing reaches a fixed point anyway, the good cells included.

What `c1` actually measures is **drift rate**. It is not worthless: the residual
after one step is 0.148 for the imperfect group against 0.099 for the clean
one, so it does carry signal. But the reason is not the one given, and the
statistic should be described as what it is. §7.2 group C is amended
accordingly, and its "if `c1` separates at a single step the probing question
dissolves" line no longer follows from anything.

**Correction 2 — the system is not being used as an attractor network.**
(Understated: §5.4 shows it is not an attractor network *at all*.)
`hopfield/core.py` documents `recall_batch_trajectory` as letting the policy
see convergence dynamics: *"clean memory attractors converge in 1-2 steps;
diffuse landscapes wander."* Measured here, that is false — nothing converges
and more iteration is monotonically worse. The Hopfield is functioning as a
**one-shot associative map**, and the trainer's `steps=1` is not a default that
was never tuned but very close to optimal.

Consequence for the policy's inputs: `--input_hopfield_multistep 1 2 3` feeds
`q` computed at three recall depths, and depths 2 and 3 are *degraded* states,
not better-converged ones. That is not necessarily useless — the rate at which
a recall degrades could still discriminate goal-present from goal-absent, which
is a question for P3 — but the channel is not delivering what its name and
docstring claim, and no design decision should rest on that reading again.

**A cheap thing to test in P4**: nothing here suggests `steps=1` should change,
but the finding is that the readout is best at the *first* step and degrades
monotonically. Worth confirming `steps=1` beats `steps=2` end-to-end before the
exploit ceiling is called, since it is a one-token change and the analysis says
it should matter.

### 5.4 It is a linear associative memory. The tanh is inert and beta is a no-op.

Jack pushed back on §5.3: a symmetric-weight Hopfield has an energy function, so
it has to settle. He is right, and "the recall never reaches a fixed point" was
the wrong conclusion from a twelve-step window. Four checks
(`analysis/nav_p2/recall_dynamics.py`), each able to falsify the explanation:

**1. The tanh is numerically inert.** `Hopfield` sets `scale = 1/D` with
`D = 1024`, so for a cue near a stored pattern `‖W x‖ ≈ 1/D`. Measured, the tanh
argument `|β W x|` has median **9.97e-05** and max 7.68e-04, where tanh needs an
argument of order 1 to bend. The largest relative deviation of `tanh(u)` from
`u` over the whole batch is **2.3e-07**.

**2. Removing the nonlinearity changes nothing.** Median
`cos(with tanh, without tanh)` after twelve steps is **1.00000000**.

**3. It converges — to the wrong thing.**

| step | cos to goal | cos to top eigenvector of W | residual |
|---|---|---|---|
| 1 | 0.991 | 0.086 | 3.6e-01 |
| 5 | 0.829 | 0.202 | 6.7e-02 |
| 12 | 0.673 | 0.583 | 8.1e-02 |
| 50 | 0.066 | **1.0000** | 3.2e-04 |
| 200 | 0.064 | 1.0000 | **5.5e-08** |
| 4000 | 0.064 | 1.0000 | 5.5e-08 |

Settled by ~50 steps and numerically exact by 200. The fixed point is the
**leading eigenvector of `W`**, which sits at cos 0.064 to the goal — very
nearly orthogonal to the thing being recalled.

**4. The spectrum predicts the rate.** `λ₂/λ₁ = 0.838`, i.e. a 10× error
reduction every 13 steps — so the twelve-step window sampled precisely the
transient and saw a monotone drift with no settling. Nothing mysterious.

#### What this actually means

`X ← normalize(tanh(β W X))` with the tanh in its linear region is
`X ← normalize(W X)`, which is **power iteration**. So this is a **linear
associative memory**, not a Hopfield attractor network: there are no basins, no
nonlinear attractors, and the classical spurious-state theory Jack invoked does
not describe it. There is exactly one fixed point that matters and it is junk.

The retrieval happens entirely in the **first** application: `W x` is a matched
filter, `≈ (1/D) ξ_goal (ξ_goalᵀ x)` plus cross-terms from the other stored
patterns. Every subsequent step is power iteration toward the leading
eigenvector, i.e. toward the direction most shared among the stored patterns —
a blend. That is why the readout degrades monotonically with recall depth
(§5.3), why `steps=1` is optimal, and why it is not merely optimal but the only
setting that retrieves anything.

#### `hopfield.beta` is a no-op

This follows rigorously from check 1. With `tanh(u) = u` to seven digits,

    normalize(tanh(β W x)) = normalize(β W x) = normalize(W x)

for any `β > 0` — the scalar cancels in the normalization. So `cfg.hopfield.beta`,
which the trainer sets from the encoder's recorded gain, **has no effect on
anything** in this configuration. It would only start to matter at `β` of order
`10⁴`, where the pre-activation reaches tanh's knee.

That is the third dead knob this project has found (phase-1 finding 3,
`MOVE_ENT_COEF` under `FREEZE_LOG_STD`; and `--freeze_log_std` itself, which did
nothing on `train_navigate` for the whole v35 lineage). **Any future claim that
a Hopfield hyper-parameter matters should be checked against this** before a
sweep is spent on it.

#### Consequences to carry forward

1. **Do not describe this system as an attractor network** in write-ups. It is a
   linear matched filter, and the difference is not cosmetic — it determines
   which theory applies.
2. **`hopfield/core.py`'s docstring is wrong** and should be fixed: it says
   "clean memory attractors converge in 1-2 steps; diffuse landscapes wander."
   Everything converges, to the same useless place, at a rate set by `λ₂/λ₁`.
3. **The multistep channel is a decay-rate probe, not a convergence probe.**
   `--input_hopfield_multistep 1 2 3` samples the transient. How fast a cue
   rotates toward the leading eigenvector depends on how much of it lies along
   the stored pattern versus the shared blend — so it may still discriminate
   goal-present from goal-absent. P3 should test that on its merits rather than
   on the convergence story, which is now known to be false.
4. **A real lever exists where a dead one was assumed.** If the nonlinearity is
   wanted — genuine attractor behaviour, with basins that could actually clean
   up a noisy cue — it needs `scale` or `β` raised by ~3-4 orders of magnitude.
   Whether that helps is untested and is a legitimate experiment; what is not
   legitimate is assuming the current network does it.

### 5.5 Why "linear associative memory" and not "attractor network"

Jack asked why the label. A linear map on the sphere has attractors too, so
"it is linear" settles nothing. The criterion that matters for *memory* is
whether each stored pattern is a **stable fixed point with a basin around it**
— that is what lets the dynamics clean up a corrupted cue, and a net storing 11
patterns should have at least 11 such states. Three tests
(`analysis/nav_p2/attractor_test.py`), none of which uses the word "linear":

**1. The stored patterns are not fixed points.** Start *exactly* at a stored
pattern and iterate:

| steps | 1 | 2 | 5 | 20 | 50 | 200 |
|---|---|---|---|---|---|---|
| cos to its own starting pattern | 0.994 | 0.977 | 0.838 | 0.168 | 0.154 | 0.154 |

A stable fixed point would hold at 1.0000. These leave immediately.

**2. But one step *does* complete a corrupted pattern — very well.** Starting
from a stored pattern corrupted to cos 0.70:

| steps | 0 | **1** | 2 | 3 | 5 | 20 | 200 |
|---|---|---|---|---|---|---|---|
| cos to the clean pattern | 0.700 | **0.987** | 0.956 | 0.897 | 0.838 | 0.186 | 0.154 |

**0.70 → 0.987 in a single application**, then monotone decay. This is the
result that makes the label meaningful rather than pejorative: the
associative-memory function works, and works well. It just happens *in the
matched filter*, not in a relaxation.

**3. Eleven patterns, one attractor.** 512 random starts, 400 steps: **one**
distinct limit (up to sign), at `|cos|` 1.000000 to the top eigenvector of `W`.

#### The answer

Not "it has no attractors" — it has exactly one. The point is that **the
attractor is not a memory**, and the memories are not attractors. Every stored
pattern is a saddle: with symmetric `W`, each eigenvector is a fixed point but
only the largest is stable, so any perturbation grows along the dominant
direction. Retrieval therefore cannot live in the dynamics, and it doesn't —
it lives entirely in `W x = (1/D) Σ_k ξ_k (ξ_kᵀ x)`, a correlation between the
cue and every stored pattern. That is the textbook linear associative memory
(Anderson / Kohonen correlation-matrix memory), where retrieval quality is set
by **cross-talk between stored patterns** rather than by basin geometry.

The distinction is load-bearing, not semantic. It predicts, correctly, that
iteration cannot improve a retrieval (§5.3, §5.4), that `steps=1` is the only
setting that retrieves anything, and that the way to improve recall is to
reduce cross-talk between stored patterns — which is an **encoder** property,
the same axis phase 1's `ur_loss2_repel` sweep was already pushing on.

**One measured number sharpens that last point.** The max absolute overlap
between the 11 stored patterns here is **0.4447** — nowhere near orthogonal. In
a correlation-matrix memory that is exactly what limits retrieval, and it is
why the top eigenvector still sits at `|cos|` 0.83 to the nearest stored
pattern. Cross-talk, not capacity or dynamics, is the binding constraint.

### 5.6 Why no basins at all — gain, not normalization, and then a second reason

Jack asked whether the normalization is what kills the basins. Swept `beta` over
six orders of magnitude with normalization on and off
(`analysis/nav_p2/gain_sweep.py`), 11 stored patterns, D=1024:

| beta | median \|beta·Wx\| | stored is a fixed pt? | # attractors from 256 starts | corrupted cue restored to |
|---|---|---|---|---|
| **5** (trainer) | 1.1e-04 | 0.154 | **1** | 0.154 |
| 500 | 1.1e-02 | 0.154 | 1 | 0.154 |
| 5,000 | 1.1e-01 | 0.150 | 1 | 0.150 |
| 20,000 | 4.4e-01 | 0.107 | 1 | 0.107 |
| 100,000 | 2.2e+00 | 0.086 | 1 | 0.086 |
| 1,000,000 | 2.2e+01 | 0.185 | **8** | 0.244 |

**Not the normalization.** With normalization still on, raising the gain does
eventually produce multiple attractors — 8 of them at `beta = 1e6`, where the
pre-activation finally reaches 22 and the tanh genuinely saturates. So
normalization does not prevent basins. What it does is decide *what the
degenerate attractor is* when there are none: with it, the top eigenvector;
without it, the state decays to the **origin**, since `λ₁ = 1.4e-3 < 1`. That
collapse is visible in the `normalize=False` rows below the transition.

*(A measurement fault worth recording: the first version of this script reported
"61 attractors" in exactly those collapsed rows. `F.normalize` maps zero to
zero, so every pair of dead states has cosine 0 and each start counted as its
own attractor — an artifact that reads as a rich landscape when it is the
precise opposite. The script now detects the collapse and says so.)*

**But there is a second reason, and it is the more fundamental one.** Look at
the "stored is a fixed pt" column: it **never approaches 1.0 at any gain**. Even
at `beta = 1e6` with the tanh deeply saturated it is 0.185, and the attractors
that do appear sit at `|cos|` 0.66 from the nearest stored pattern. Raising the
gain creates basins — but **not around the memories**.

The reason is that a saturating `tanh` puts its fixed points at the corners of
the hypercube, `±1` in every coordinate. That is what makes classical Hopfield
work: the stored patterns *are* binary, so they *are* corners. Here the stored
patterns are **continuous encoder outputs**, which are not at corners, so
saturation necessarily moves them somewhere else. A continuous vector cannot be
a fixed point of a saturating elementwise nonlinearity unless it happens
already to be saturated.

**So there is no gain at which this becomes an attractor memory for its own
patterns.** Low gain gives one attractor and no basins; high gain gives basins
at hypercube corners that are not the stored patterns. The matched-filter regime
is not a suboptimal choice among available regimes — for continuous-valued
stored patterns it is the only regime in which retrieval works at all, which is
consistent with one step restoring a cos-0.70 cue to 0.987 (§5.5) while every
iterated regime fails.

Cross-talk compounds it: max overlap between the stored patterns is 0.4447, so
even binary patterns at this correlation would not retrieve cleanly.

**Consequence.** The earlier suggestion (§5.4, point 4) that raising `scale` or
`beta` by 3-4 orders of magnitude is "a legitimate experiment" is **withdrawn**
— it is now measured, and it does not give attractor retrieval of these
patterns. Making this an attractor network would require binarizing the stored
patterns (or an architecture whose fixed points are not corners, e.g. a modern
softmax Hopfield, whose fixed points *are* the stored patterns by construction).
That is a real design fork, but it is a different network, not a hyper-parameter.

### 5.7 The network is not broken — it is mismatched. And there is a drop-in fix.

Jack: "so classical continuous Hopfield networks just don't work?" No. Three
conditions on the same 11 patterns (`analysis/nav_p2/architecture_test.py`):

| | beta | stored is a fixed pt | # attractors | corrupted 0.70 restored to |
|---|---|---|---|---|
| **B.** classical + continuous *(current setup)* | 1e5 | 0.086 | 1 | 0.086 |
| | 1e7 | 0.226 | 21 | 0.148 |
| **A.** classical + **binarized** patterns | 1e5 | **1.0000** | **11** | **1.0000** |
| | 1e6 | 1.0000 | 34 | 1.0000 |
| **C.** **modern** / dense assoc. + continuous | 8 | **1.0000** | **11** | **1.0000** |
| | 512 | 1.0000 | 11 | 1.0000 |

**The classical model works exactly as designed.** Binarize the patterns and
raise the gain and it retrieves perfectly: all 11 stored patterns become fixed
points, a cue corrupted to cos 0.70 is restored to cos 1.0000, and pushing the
gain higher produces 34 then 47 attractors — the spurious mixture states
classical theory predicts, and the ones Jack expected to see. Nothing is broken.

**The mismatch is the pattern type.** A saturating elementwise nonlinearity has
its fixed points at hypercube corners; classical Hopfield works because its
stored patterns *are* corners. This project stores continuous encoder outputs,
which are not, so they can never be fixed points of that dynamics at any gain.
Capacity is not involved — 0.138·D = 141 here against 11 patterns stored.

*(Caveat, flagged before the run: binarizing is not a perfectly controlled
change. It also decorrelates — max overlap drops 0.4447 → 0.2910 — and moves
each pattern substantially, median cos(ξ, sign ξ) = 0.7996. So A demonstrates
that the architecture retrieves binary patterns, not that binarizing *these*
patterns would preserve what the encoder encodes.)*

**And there is a drop-in that handles the continuous patterns exactly.** The
modern / dense associative memory update `ξ ← Xᵀ softmax(β X ξ)` gives all 11
patterns as exact fixed points from `beta = 8` upward, restores a cos-0.70 cue
to 1.0000, and finds **exactly 11 attractors and no spurious ones** even at
`beta = 512`. Its fixed points are the stored patterns by construction, which
is precisely the property the classical net cannot have here.

#### What this would and would not buy the project — do not assume it is a win

Tempting conclusion: swap in a modern Hopfield and the readout becomes exact.
P1 found direction error is governed almost entirely by recall fidelity (§5.2),
so exact recall should drive `dir_cos` to ~1 and the sub-0.5 tail to ~0.

**But the explore side may get worse, and that is not a detail.** The whole
explore/exploit discrimination rests on `|q|` being *small* when only
distractors are stored (§5.5, ratio 2.3-4.7). That separability currently comes
from the recall being a **blurry blend** in the goal-absent case. A sharp
modern Hopfield would retrieve the nearest distractor *exactly and confidently*
— which could produce a large, clean `q` pointing at a phantom, and drive
`chase_q` up. Phase 1 established that persistent q-following during explore is
exactly what causes the corner-trap collapse.

So this is a **real design fork, not a free upgrade**, and it is measurable with
tooling that already exists: run `q_failure_map` with the modern update
substituted and compare both the `dir_cos` distribution *and* the goal-present /
goal-absent `|q|` separability. Both numbers have to improve, or at least the
second must not collapse, before it is worth changing the network the whole
project is built on.

Recorded as an open item, not a recommendation.

### 5.8 Why the system works at all — measured

Everything in §5.3-5.7 reduces the recall to a single product,
`W x = (1/D) Σ_k ξ_k (ξ_k · x)`: a sum of stored patterns weighted by their
similarity to where the agent stands. So the whole pipeline rests on one
property of the **encoder** — that same-env similarity beats cross-env
similarity by enough that one product returns the goal. Measured
(`analysis/nav_p2/why_it_works.py`, 8 envs x 4 draws x 400 cells):

**1. The signal — `ξ(p) · ξ(goal)` within an env**

| grid distance | 0-1.5 | 3-5 | 5-8 | 8-12 | 12-20 | 20-30 |
|---|---|---|---|---|---|---|
| median | 0.9993 | 0.9936 | 0.9832 | 0.9573 | 0.8869 | 0.7534 |

The encoder maps an entire environment into a **tight cone**: even corner to
corner, similarity is 0.75.

**2. The cross-talk — similarity to patterns from other envs**

median **−0.0002**, p90 0.044, p99 0.273, **max 0.9823**

Typically *orthogonal*, which is what the repulsion objective was trained to do.
But the tail is heavy and that is where the whole story lives (below).

**3. The margin that makes one product enough** — `ξ·ξ_goal − max_k ξ·ξ_k`:
median **0.862**, p10 0.641, and a distractor out-weighs the goal in only
**0.26%** of cases. That number is the retrieval failure rate, and it matches
P1's independently-measured lock failures (0.15% distractor + 0.78% mixture).

**4. After the tangent projection** — median `|q|` to the goal **0.3006**, to
the nearest distractor **0.0670**, a **4.49x** separation. And the prediction:
a displacement to another env's pattern is an unrelated direction in D
dimensions, so it keeps `√(2/D) = 0.0442` of its norm, and `‖ξ_d − ξ_x‖ ≈ √2`,
giving 0.0625 against the measured 0.0670. **The explore/exploit magnitude
separation is a dimensionality effect and is quantitatively predicted.**

#### The account

The system does **two separate jobs, and the encoder does both**:

1. **Retrieval.** Same-env similarity ~0.99 against cross-env ~0 means the goal
   term dominates the weighted sum, so one product returns the goal pattern.
2. **Geometry.** Retrieval says *which* pattern, not *where*. The tangent
   projection `q = W_xᵀ(ξ_g − ξ_x)` recovers the grid displacement, because the
   encoder is a smooth chart of the arena.

**This is why attractor dynamics was never needed.** Basins exist to complete a
*corrupted* cue. The cue here is `ξ_x`, a clean embedding of the agent's actual
position — not a noisy copy of anything. The question is "which stored pattern
is most like where I am, and where is it relative to me", which is a
nearest-neighbour lookup plus relative geometry, and a correlation matrix
answers it in one product. Iterating adds nothing because there is no corruption
to remove; it re-applies `W` and amplifies whatever is most shared across all
patterns, which is the top eigenvector and carries no information about which
pattern was cued.

#### The one thing that fails, and it is not diffuse

Cross-talk has median −0.0002 but **max 0.9823**. The failures are not general
interference — they are **rare near-collisions**, where a distractor drawn from
somewhere else in the scaffold encodes almost identically to an in-env position.
p99 is 0.273 while the max is 0.982: an extremely heavy tail on an otherwise
clean distribution.

That explains the whole failure structure measured in P1 — 0.26% of cells where
a distractor out-weighs the goal, against 99%+ clean retrieval — and it makes
the target precise. **The mean is already perfect; only the worst case hurts.**
Improving retrieval means suppressing the tail of the cross-env similarity
distribution, not lowering its average.

That is exactly the objective the `ur_loss2_repel` encoder sweeps were built
around — and it retroactively justifies scoring those sweeps by **worst-case
coding radius** rather than a mean, which is how they were already being read.

### 5.9 Storing tanh(g·ξ): where the memories DO become fixed points

Jack's experiment: apply the nonlinearity **before** storage, `p = tanh(g·ξ)`,
and find the gain and capacity at which those stored patterns are fixed points
(`analysis/nav_p2/stored_gain_capacity.py`). It works, and the phase boundary is
sharp.

**Median cos(iterate(p), p)** — ≥ 0.99 means the stored patterns are fixed
points. Patterns drawn from random scaffold positions, D = 1024, dynamics gain
swept per cell (below):

| storage gain g | M=5 | 11 | 25 | 50 | 100 | 141 | 200 | 400 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.530 | 0.333 | 0.242 | 0.097 | 0.110 | 0.055 | 0.054 | 0.058 |
| 10 | 0.499 | 0.461 | 0.289 | 0.076 | 0.099 | 0.053 | 0.054 | 0.057 |
| 30 | 0.941 | 0.885 | 0.747 | 0.050 | 0.100 | 0.047 | 0.059 | 0.057 |
| **100** | **0.990** | **0.986** | **0.976** | **0.923** | 0.118 | 0.054 | 0.057 | 0.060 |
| **300** | **0.997** | **0.996** | **0.993** | **0.986** | 0.161 | 0.059 | 0.062 | 0.063 |
| **1000** | **0.999** | **0.998** | **0.997** | **0.996** | 0.336 | 0.062 | 0.064 | 0.062 |

**Turn-on gain is g ≈ 100**, marginal at 30, absent at ≤ 10. At g = 100 the
patterns are 83% saturated and sit at cos 0.954 to their own binarization.

**Capacity is between 50 and 100 patterns** — M = 50 holds at every gain ≥ 100,
M = 100 fails everywhere. That is below the classical 0.138·D = 141, which is
the expected penalty for correlated patterns: §5.8 measured cross-similarity
with median ≈ 0 but a heavy tail (p99 0.27, max 0.98), and near-collisions cost
capacity disproportionately.

**Basins come with the fixed points.** Table B — a cue corrupted to cos 0.70 —
reproduces table A cell for cell (0.990/0.986/0.976/0.923 at g=100). Wherever
the patterns are stable they are also *recoverable*, so these are real basins
and not just marginal stability.

**The dynamics gain needs only to clear its own threshold.** With
`p = tanh(g·ξ)` of squared norm `S`, `W p ≈ (S/D) p`, so the loop gain is
`β·S/D` and a nonzero fixed point requires `β > D/S` — 1024 at g=1, ~1.3 at
g=100. In the working cells the winning multiple of that threshold is only
**2–5×**. *(The first run of this sweep used a single β = 1.0, which sits below
threshold in every cell, and reported that nothing worked anywhere. The
threshold is a function of the storage gain, so a fixed β cannot test this
grid.)*

**Per-step normalization does not change the picture** — the same run with the
codebase's `normalize_each` gives the same boundary (g ≈ 100–300, M ≤ 50),
confirming §5.6: normalization decides what the degenerate attractor is, not
whether basins exist.

#### What this does and does not settle

It settles the mechanism, but a caveat Jack caught matters for stating it
correctly. An earlier version of this section said the network is linear
"because the patterns are stored at unit norm, making `W x ~ 1/D`". That picks
one arbitrary half of a product and calls it the cause. `W = (1/D) Σ p pᵀ`, so
storing `λp` instead of `p` scales `W` by `λ²`, and `β·W` is **invariant** under
`(p → λp, β → β/λ²)` — exactly, since zeroing the diagonal commutes with the
scaling. Unit-norm and natural-norm storage are the *same dynamics at rescaled
β*. There is one quantity, the loop gain `β·S/D`, and two knobs that set it.

**And fixing the loop gain buys nothing on its own**, which §5.6 had already
measured: sweeping β from 5 to 1e6 on the raw continuous patterns never took
stability above 0.226. Raising the loop gain converts decay-to-zero into a
nonzero fixed point; it does not make a continuous vector into a corner.

So there are **two independent conditions**, and they were being conflated:

| condition | knob | compensable | what it buys |
|---|---|---|---|
| loop gain `β·S/D > 1` | β *or* storage norm — the same knob twice | yes, by either | a nonzero fixed point instead of decay to zero |
| stored pattern near a corner | storage gain `g` only | **no** | that the fixed point is *your memory* |

The current system fails the first (β = 5 at unit norm), which is why it is a
linear matched filter. But it would still fail the second even with the first
repaired, which is exactly why §5.6's β sweep failed. Only pre-saturating the
patterns satisfies the second, and that is what this section measures.

**But it does not follow that this should be adopted, and the reason is §5.8.**
The encoder does *two* jobs here: retrieval (same-env similarity ≈ 0.99 vs
cross-env ≈ 0) and **geometry** (`q = W_xᵀ(ξ_g − ξ_x)` recovers the grid
displacement because the encoding is a smooth chart). At g = 100 the stored
pattern is at cos 0.954 to its own binarization — most of the continuous
structure is gone. Retrieval would become exact; whether the *tangent
projection* still decodes direction from a saturated pattern is an entirely
separate question, and it is the one that matters, because direction is what the
policy actually consumes.

There is also an interface problem this raises. If the stored patterns are
`tanh(g·ξ)` but the cue is the raw `ξ_x`, then the recall returns
`tanh(g·ξ_goal)` and `q = W_xᵀ(tanh(g·ξ_g) − ξ_x)` mixes a saturated vector with
an unsaturated one — not obviously a meaningful displacement. A coherent version
would tanh the cue as well and re-derive the tangent basis in the saturated
space, which is a different geometry, not a parameter change.

**Next test, cheap and decisive**: run `q_failure_map` with `tanh(g·ξ)` storage
and matched cueing at g ∈ {30, 100, 300}, and read `dir_cos` and the
goal-present/goal-absent `|q|` separation. If direction survives saturation, the
attractor regime is genuinely available and brings exact retrieval with it. If
it does not, then the linear matched filter is not a limitation of this design
but the only regime in which the geometry survives — which would be the
strongest statement this phase could make about the architecture.

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

**CORRECTED — see §5.3.** This group was justified as a spurious-state
detector: a stored pattern is a fixed point of the recall map, a mixture is
not. Both halves are false here. The network is a classical Hebbian Hopfield
with tanh, where mixture states *are* genuine attractors, and in this operating
regime nothing reaches a fixed point at all — iterating walks steadily away
from the goal pattern, and pooled direction quality degrades from 1.46% bad at
one step to 12.27% at twelve.

`c1` and `c2` therefore measure **drift rate**, not spuriousness. That still
carries signal — the one-step residual is 0.148 for imperfect-fidelity cells
against 0.099 for clean ones — and it is still free, since the iterates are
already computed and already fed to the policy. So keep both statistics in the
feature set and let the ablation price them, but do not expect them to separate
"the answer is right" from "the answer is a blend", and drop the earlier claim
that a single-step separation would dissolve the probing question.

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
