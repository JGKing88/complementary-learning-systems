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
| **Done** | §4 blocking fixes, P1 (§5) with figures, the recall-mechanism thread §5.3-5.9, and **P2 (§6)** |

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
- [x] **P2 done (§6).** The cross-env geometry *is* learnable — up to R² 0.658
      at 19.8° with both cones pinned North — and at the agent's own headings,
      at the walk persistence the `p5_e` explore checkpoint was **measured** to
      have, the best decoder adds +0.254 over its own shuffled control. But on
      **direction**, which is what path integration needs, the cones help only
      from four steps on, and at one step are far worse than the agent's own
      heading (8.0° vs 1.5°) — **and four steps on is exactly where integrating
      `prev_displacement` is already exact**. Adaptation buys nothing at
      any `k` up to 256. And the question stopped gating the ceiling when §4's
      B3 turned on `input_prev_displacement`, which hands the agent **exact**
      self-motion (integration error 2.3e-14). Three things carry forward:
      **P5's explore target is set by what the policy does with
      `prev_displacement`, not by the cone**; `WALL_RESOLUTION` is a measured
      trade-off (4 → 1 is worth 4× the cross-env geometry and costs in-env
      uniqueness); and a **range** sensor over the same rays decodes at 0.869
      where the ±1 code gets 0.132, which is the number Jack's
      "structure the input differently?" question was waiting on.
- [ ] **P3 not started.** It is the critical path for P6, and §5.3-5.7 changed
      what its group-C statistics mean.
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
| coverage lawnmower ceiling | → 0.5025 | needs position knowledge — **§6: reopened, but by `prev_displacement`, not by the cone** |
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

**Answered in §6, and not the way this table expected.** Relative displacement
is *not* decodable in a held-out env — but the question stopped deciding the
ceiling when §4's B3 turned on `input_prev_displacement`, which hands the agent
exact self-motion. The lawnmower line is open on information grounds; whether
P5 reaches it is a policy-capacity question.

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

Measured in `analysis/nav_p2/displacement_decodability.py` (cross-env transfer,
six framings, all controls), `analysis/nav_p2/displacement_adaptation.py`
(k-shot in a new env, plus the two measurements that explain the result) and
`analysis/nav_p2/policy_turn_stats.py` (where the trained policy actually sits
on §6.5's axis — the only one of the three that needs the scaffold and a GPU).
Launcher `hopfield_nav/run_nav_p2_disp.sh`, seven probes. Otherwise no encoder,
no scaffold, no GPU: like P0.9 this is a property of the sensor. 64 training
envs and **48 held-out test envs** unless stated, and every number below is a
median over the test envs with [p10, p90] where it is quoted.

### 6.1 RESULT — the geometry does transfer, the agent cannot use it, and the ceiling question it was written to settle has already been answered elsewhere

Three findings, and the answer to Jack's question is the conjunction of the
first two rather than either alone.

**1. The shared geometry is real and does transfer — in a canonical
orientation.** With both cones pinned facing North, a framing with no heading
anywhere, a decoder trained on other envs works on held-out ones, and works
better the more favourable the configuration: **R² 0.13 at 51° of median
angular error** at the launcher's own settings (`wall_resolution=4`, 64 envs,
ridge or MLP), rising to **R² 0.658 at 19.8°** with the coarsest wall code, 128
training envs and a 2-layer MLP — against a chance level of 89.9° and a
snap-oracle ceiling of 0.945 / 8.6°. So the answer to "can the geometric
structure be learned across environments" is **yes**, and §6.3–6.4 say what
carries it and what suppresses it.

**2. At the agent's own headings, the cones add nothing.** That is the finding
that decides the question, and it survives every attempt made here to pose it
generously. With no heading supplied the decoders are at chance (R² ≤ 0.001,
89–90°). With heading supplied they appear to do well — and the `side-only`
control shows that is the heading talking: pooled over lags, heading alone
scores 0.233 / 27.4° and `ridge/spec`, `ridge/xcorr`, `ridge/bilin` and
`mlp/raw` score 0.233, 0.237, 0.229 and 0.240, every one within 0.007 of it,
with every shuffled control equal to its unshuffled twin. In the best-posed
version (§6.5 — cones aligned by `Δψ`, answer in the agent's own frame, the
coarse wall code) they do better: at the persistence the `p5_e` explore
checkpoint **was measured to walk with**, the best decoder adds **+0.254 of R²
over its own shuffled control**, nearly doubling what heading alone supplies.
But the **direction** estimate — the part path integration needs — splits by
lag, and splits the wrong way for this project. At **one step** the cones make
it much worse (8.0° against heading-only's 1.5°), because at one step the
heading *is* the displacement direction. Only from four steps on do they
improve it — and over four or eight steps the agent's own integration of
`prev_displacement` is exact anyway. **There is no lag at which the cones
supply a direction the agent does not already have more precisely.**

**3. The premise moved under phase 2's own §4 fixes.** P2 was written to decide
whether the explore ceiling is billiard (0.378) or lawnmower (→0.50), on the
reasoning that lawnmower needs to know where the agent has already been and
that sensory decoding was the only route to it. That was true in phase 1, where
`INPUT_PREV_ACTION=0` and `input_prev_displacement` defaulted off, so the agent
had no self-motion channel at all. **§4's B3 turned both on.**
`prev_displacement` is `ContinuousVecEnv._last_displacement` read straight off
the env (`rollout/collector.py:659`) — the realized post-clip move, exactly.
Integrating it over 200 steps × 4000 episodes reproduces position relative to
the start to **2.3 × 10⁻¹⁴**.

So the agent is already handed, exactly and for free, the quantity P2 asked
whether it could decode. **The lawnmower ceiling is not gated on sensory
decodability. It is gated on whether the policy can hold and use a position it
is already being given** — a capacity question for P5/P6, not a sensory one.

One caveat, with a number. Position-relative-to-start is exact from step 1;
position in the *arena frame* is not, and the `_serpentine_track` baseline that
defines the lawnmower line assumes the arena bounds (it walks to a corner and
sweeps columns of exactly `size−1`). Those bounds come from wall contacts,
which are directly observable because a clipped step makes `prev_action` and
`prev_displacement` differ — which is precisely why §11 Q1 decided to feed
both. Under random actions 12.0% of steps are clipped, two non-parallel walls
are touched in 80.7% of episodes at a median of **79 steps** (p10 20, p90 165),
and all four in only **1.9%**. The frame is cheap to half-pin and expensive to
pin completely, which argues for a sweep the agent runs in its own frame and
repairs on contact rather than a planned boustrophedon.

### 6.2 The heading is the answer, so supplying it is a leak, not a control

The spec asked for a "free-heading, headings supplied" arm as the realistic
case. It cannot be run that way at lag 1, and the reason is structural rather
than incidental: heading is not an independent variable in this env.
`ContinuousVecEnv` sets `ψ = atan2(dx, dy)` of the *realized displacement*
(`world/vec_env.py:461`). At lag 1 the heading at the second observation **is**
the direction of the displacement being decoded.

This was measured rather than reasoned about, by adding a `side-only` arm that
sees the heading terms and nothing from the cones. In the `ego` framing at lag
1 it reaches **R² 0.875 at 1.3° of median angular error** — better than every
sensory arm anywhere in this study. The sensory arm sitting beside it scored
0.861 / 2.4°, i.e. *worse*, and its shuffled control — `s2` permuted, the
pairing destroyed — scored 0.862 / 2.3°, i.e. identical. Without that control
this section would have reported a 2.4° angular error as its headline.

Pooled over all four lags the same thing holds and is easier to read: the leak
is 0.233 / 27.4°, and `ridge/spec`, `ridge/xcorr`, `ridge/bilin` and `mlp/raw`
land at 0.233, 0.237, 0.229 and 0.240 — every one of them within 0.007 of
heading-only, and every shuffled control equal to its unshuffled twin to three
decimals. **Under uniform turns the two cones add nothing whatsoever to what
the agent's own heading already tells it**, at any lag.

So: **lag 1 is interpretable only in the `fixed` framing**, where no heading
enters anywhere. From lag 2 the headings stop determining the cumulative
displacement and the supplied-heading framings become real measurements — but
the leak does not vanish, it decays (0.875 → 0.432 → 0.213 → 0.103 across lags
1, 2, 4, 8), so the `side-only` row is printed per lag and every claim below is
made against it.

### 6.3 What can cross an env boundary at all, and the number it gets

The wall code is a fresh random ±1 draw per env (`world/env.py:329`), so the
map position → cone is a different hash in every arena and nothing read off the
raw pattern can transfer. The spec's own feature set makes the point: ridge on
`[s1, s2, s1−s2, s1⊙s2]` scores **−0.000 / 89.5°, exactly chance**, at every
resolution and in every framing.

What *can* transfer is the second-order agreement structure.
`E[s1[i]·s2[j]] = 1` when ray *i* of view 1 and ray *j* of view 2 land on the
same wall segment and 0 otherwise, whatever the codebook is — so `s1 s2ᵀ` is a
codebook-independent geometric measurement, and it is the only channel a
cross-env decoder has. Two features were built on it: `xcorr`, the
shift-invariant projection (per-ray products plus cross-correlation over ±24
ray lags), and `bilin`, the full 60×60 outer product.

`fixed` framing, `wall_resolution=4` (the launcher default), 48 held-out envs:

| decoder | R² [p10, p90] | median ang. err [p10, p90] | <45° |
|---|---|---|---|
| snap-oracle **CEILING** | 0.945 [0.944, 0.946] | 8.6 [8.5, 8.8] | 91.5% |
| NN table, in-env, all 400 cells | 0.943 [0.822, 0.946] | 8.6 [8.5, 8.9] | 91.3% |
| mlp/raw | 0.132 [0.094, 0.184] | 51.3 [47.2, 55.6] | 45.3% |
| mlp/bilin | 0.122 [0.053, 0.206] | 47.8 [41.3, 55.0] | 47.8% |
| ridge/bilin | 0.091 [0.059, 0.130] | 55.1 [50.0, 61.0] | 43.0% |
| ridge/xcorr | 0.084 [0.056, 0.107] | 56.9 [52.3, 62.9] | 41.5% |
| same-env ridge/xcorr | 0.088 [0.053, 0.132] | 59.1 [52.8, 67.5] | 39.9% |
| ridge/spec (the spec's features) | −0.000 [−0.002, 0.001] | 89.5 [85.7, 93.0] | 25.7% |
| shuffled control (bilin) | 0.001 [−0.000, 0.002] | 86.0 [84.1, 87.8] | 26.5% |
| constant **CHANCE** | −0.000 [−0.001, −0.000] | 89.9 [88.5, 91.8] | 24.9% |

The gap between ridge and the MLP is small here (0.09 → 0.13) — but do not read
that as "the structure is linear". At this resolution there is barely any
structure for either to find; where there *is* (§6.4, resolution 1 with 128
training envs) the MLP reaches 0.658 against ridge's 0.392, and the gap is the
larger part of the result. The honest reading of this table is that at
`wall_resolution=4` the cross-env signal is close to the floor for every
decoder tried.

The two rows at the top are the important ones. **A nearest-neighbour table
over all 400 cells of the env in question saturates the ceiling**, 0.943
against 0.945. Within an env the pair of cones does determine the displacement,
essentially perfectly. So this is not a claim that the information is missing
from two glimpses. It is a claim about *transfer*: the part of the map that is
env-specific is a random hash and cannot be learned once and reused, and at this
resolution the part that is shared geometry carries R² ~0.1.

Note also that the same-env linear decoder (0.088) is no better than the
cross-env one (0.084). A linear decoder gets nothing from being told which env
it is in — all it can use is the shared geometry either way — which is a second,
independent statement of the same fact.

### 6.4 `wall_resolution` is the knob that decides how much geometry survives

`fixed` framing, best cross-env arm, 48 held-out envs, over the `WALL_RESOLUTION`
values the launcher can take (default 4):

| wall_res | ridge/xcorr | ridge/bilin | shuffled control | in-env NN table | snap ceiling |
|---|---|---|---|---|---|
| 1 | 0.359 / 30.7° | **0.387 / 28.6°** | 0.007 / 76.0° | 0.242 | 0.945 / 8.6° |
| 2 | 0.190 / 41.7° | 0.210 / 39.4° | 0.003 / 83.5° | 0.563 | 0.945 / 8.7° |
| 4 (default) | 0.084 / 56.9° | 0.091 / 55.1° | 0.001 / 86.0° | 0.943 | 0.945 / 8.6° |
| 8 | 0.040 / 70.0° | 0.039 / 67.0° | 0.005 / 85.4° | **0.945** | 0.945 / 8.7° |

**Monotone across the whole range, and worth a factor of ten from 8 to 1.**
The mechanism is spatial frequency: at `wall_resolution=1` a segment is a whole
cell, so adjacent rays land on the same or neighbouring segments and the cone
has runs whose shift under translation is readable; at 8 a segment is an eighth
of a cell and every ray lands on its own, so the cone is a fine hash and the
shift is not. The in-env table moves the opposite way over the same range
(0.242 at resolution 1, 0.945 at 8), which is the same trade-off seen from the
other side: a finer code is a better fingerprint and a worse ruler.

This is a live trade-off rather than a free win. `wall_resolution=4` exists so
that two positions inside one cell read differently (`world/env.py:100-106`),
which is what the continuous movement mode needs for the *encoder*. What it
costs is the cross-env geometry, and P2 is the first measurement of that cost.

Pushed to the most favourable configuration the sensor allows — resolution 1,
**128** training envs, and a 2-layer MLP instead of ridge — the `fixed` framing
reaches **R² 0.658 [0.617, 0.708] at 19.8°** (0.650 / 16.6° for the MLP on
bilinear features), against the 0.945 / 8.6° ceiling. At one step it is 0.584 /
19.4° against a lag-1 ceiling of 0.803 / 13.9°. That is a real decoder, not a
trace: three quarters of the way to the ceiling in R², and it is what justifies
calling the cross-env geometry *learnable* in §6.1. The three levers that get
it there — coarser code, more training envs, a nonlinear decoder — are all
things this project can change. What none of them touches is §6.5.

Two things make that number trustworthy rather than a lucky fit. A separate job
reran the same configuration and got **0.657 / 19.7°** against 0.658 / 19.8° —
independent processes, independent env draws, same answer. And the MLP's *own*
shuffled control — an MLP fit with the pairing destroyed — sits at **0.019 /
80.7°**, so essentially none of the 0.657 is coming from anything but the two
cones. (It is not exactly zero, and should not be: `s1` alone weakly constrains
`Δpos`, because a cell near a wall cannot be followed by a step into it. That
boundary effect is precisely why the shuffled control, not the constant
predictor, is the baseline these increments are quoted against.)

### 6.5 Free heading destroys it, and two separate things are doing the damage

With the cones at the poses actually occupied and no heading supplied (`free`),
every decoder at every resolution sits at chance: R² ≤ 0.001, median angular
error 89–90°. Two candidate explanations, and they had to be separated.

**(a) The decoder is not being told enough to answer in world frame.** Posing
this correctly took two attempts and the first one is worth recording. The
`derot` framing re-indexes view 2's cone by `Δψ` so ray *i* of both views looks
along the same world bearing — the operation an agent that knows its own
heading would perform. Its sign is invisible in the downstream score, so a
self-test pins it: two views from the same cell at different headings must
agree exactly after alignment, and it reports **1.000 for the correct sign and
0.516 — chance for ±1 — for the flipped one**. The first implementation had the
sign backwards; the guard now runs at the top of every job. But de-rotation
alone still scored near chance, and the reason is that the framing was
ill-posed: after alignment the features are expressed relative to view 1's
bearing, `ψ1` is never supplied, and the target was still in world frame. The
decoder was being asked to rotate by an angle it had not been given.

`derot_ego` fixes it — same alignment, target in view 1's frame, which is also
what an agent chaining its own displacements would compute. Even at 128
training envs and with an MLP, the ill-posed `derot` reaches only 0.029;
`derot_ego` is the version worth reading.

And there is a second trap in it, which the control rows catch. In the
egocentric frame the target is not isotropic — the agent moves forward by
construction — so the baselines are strong: heading alone reaches 27.4° under
uniform turns, and under a persistent walk even the *constant* predictor gets
into the twenties. **An angular error in an ego-frame framing means nothing on
its own.** The quantity to read is each decoder's increment over *its own
shuffled control*, which keeps the heading terms and destroys only the pairing:

`derot_ego`, resolution 1, 48 training envs, 32 held-out test envs, pooled over
lags 1–8. "cones add" is `ridge/bilin` minus its own shuffled control:

| per-step turn sd | median &#124;Δψ&#124; | shared rays | disjoint | constant | heading only | ridge/bilin | its shuffled | **cones add** |
|---|---|---|---|---|---|---|---|---|
| uniform | 91.2° | 14 / 60 | 34.9% | 87.6° | 0.235 / 27.4° | 0.260 / 27.3° | 0.233 / 28.2° | **+0.027** |
| 45° | 30.6° | 45 / 60 | 8.8% | 43.1° | 0.280 / 15.6° | 0.309 / 17.2° | 0.273 / 15.1° | **+0.036** |
| 20° | 11.5° | 54 / 60 | 7.1% | 20.2° | 0.281 / 6.8° | 0.387 / 11.0° | 0.279 / 8.9° | **+0.108** |
| 10° | 5.0° | 57 / 60 | 5.3% | 10.8° | 0.248 / 3.5° | 0.438 / 7.8° | 0.265 / 6.0° | **+0.173** |
| **`p5_e` u700 (measured)** | **14.6°** | **53 / 60** | **0.1%** | — | — | — | — | — |

**The last row is measured, not swept** (`analysis/nav_p2/policy_turn_stats.py`,
the phase-2 explore checkpoint `p5_e` at u700, 6 envs × 16 trials × 200 steps,
19 008 consecutive pairs). It rolls the trained policy out, takes the realized
displacements, and computes exactly the quantities the table is indexed by. The
answer settles which row of the sweep the project is actually standing on: the
policy turns a **median of 14.6°** per step, its two consecutive cones share
**53 of 60 rays**, and **0.1%** of consecutive pairs are disjoint — a lower
disjoint fraction than any synthetic walk in the table, because the policy also
avoids the wall clips that snap a random walker's heading. It sits at the
`20°` row, at the favourable end.

**How representative is that row?** The checkpoint runs
`--persistence_bonus 0.20` and `--epsilon_explore 0.1 --epsilon_anneal_updates
200`, which looks at first like an unusually straight-walking configuration —
`run_nav_p2.sh` documents `0.05` and `0.4`. It is not: **every variant block in
that launcher sets exactly `PERSISTENCE_BONUS=0.20` and `EPSILON_EXPLORE=0.1`**
(lines 171–236), and the `0.05` / `0.4` at the top are fallbacks no variant
uses. All five P4 runs and the P5 run share those values. So 14.6° is
representative of the phase-2 lineage as actually launched, not an outlier
within it — and since ε anneals over 200 updates, any phase-2 policy past u200
has none of it left, as this one at u700 does not.

The caveat that remains is narrower than it first looked: this is one
checkpoint, from the explore schedule, at one update. It is not a statement
about a policy someone deliberately runs at the unused `0.05` / `0.4`
fallbacks, nor about the interleaved P6 agent, which does not exist yet.

(Two more details from the same run: the median step norm is **2.00**, so the
policy saturates `MAX_ACTION_NORM` on essentially every step; and the smoothness
is local — by lag 8 the median `|Δψ|` is 122.7° and 57.9% of pairs are disjoint,
so the policy is locally ballistic and globally not.)

So the row that matters for this checkpoint is `20°`, not `uniform`. Four
things to read off it, and the first is not what I expected.

**The cones do contribute, and the contribution grows steeply with overlap.**
The increment goes +0.027 → +0.036 → +0.108 → +0.173 as the disjoint fraction
falls from 35% to 5%, and in the bottom row the cones nearly *double* the R²
over heading alone (0.438 against 0.248). So §6.5(b) is a large real effect, not
a plausible story — a straighter walk genuinely does make the two glimpses
mutually informative.

**A non-linear decoder roughly doubles the increment, and needs its own
control to say so.** The table above is ridge on 48 training envs. Running the
same framing on **128** with a 2-layer MLP, and an MLP fit on shuffled pairs
beside it (the ridge rows are repeated at 128 envs so the comparison is
like-for-like; they move by 0.003 against the 48-env table, which is the
sample-size effect):

| turn sd | framing | arm | R² | its shuffled | **cones add** | ang. err | heading-only |
|---|---|---|---|---|---|---|---|
| uniform | `derot_ego` | ridge/bilin | 0.265 | 0.236 | +0.029 | 26.9° | 27.5° |
| uniform | `derot_ego` | mlp/raw | 0.332 | 0.238 | **+0.094** | 26.5° | 27.5° |
| uniform | `ego` | mlp/raw | 0.324 | 0.238 | +0.086 | 29.4° | 27.5° |
| 20° | `derot_ego` | ridge/bilin | 0.395 | 0.290 | +0.105 | 10.5° | 6.8° |
| 20° | `derot_ego` | mlp/raw | 0.610 | 0.407 | +0.203 | 9.4° | 6.8° |
| 20° | **`ego`** | **mlp/raw** | **0.656** | 0.402 | **+0.254** | 9.7° | 6.8° |

So the +0.108 in the table above is a floor: the best decoder tried gets
**+0.254** at the operating point. The MLP control was worth building rather
than borrowing the ridge's, and the numbers say why — `shuf/mlp-raw` reaches
**0.402** where the ridge's shuffled control sits at 0.290 and heading-only at
0.282. A non-linear decoder reads the heading terms far better than a linear
one. Scoring `mlp/raw`'s 0.656 against the ridge's control would have claimed
+0.37 and overstated the cones by nearly half.

**And my hand-built de-rotation turned out to be unnecessary, and at the
operating point harmful.** `ego` differs from `derot_ego` only in *not*
re-indexing view 2's cone by `Δψ` — both are handed `Δψ` as a feature. Aligning
helps under uniform turns (+0.094 against +0.086) and *costs* at the operating
point (+0.203 against +0.254), because shifting by `round(Δψ / ray)` zero-fills
the wedge with no counterpart and discards it; when `Δψ` is small there is
little to gain and real information to lose. **A non-linear decoder handed `Δψ`
learns the rotation better than an explicit shift applies it.** The framing was
worth building — it is what proved the earlier world-frame `derot` was
ill-posed rather than uninformative (§6.5a) — but the headline number should
come from `ego`.

**Where the increment goes depends on the lag, and the split is the finding.**
Not "all magnitude", which is what the ridge-only view suggested. Per-lag
angular error for the best arm (`ego` / `mlp/raw`) against its own shuffled
control and against heading-only, at the operating point:

| | lag 1 | lag 2 | lag 4 | lag 8 |
|---|---|---|---|---|
| `‖Δpos‖` median | 0.95 | 2.02 | 4.01 | 7.45 |
| mlp/raw | 8.0° | 8.4° | 10.1° | **13.5°** |
| its shuffled | 5.1° | 6.6° | 9.9° | 17.8° |
| heading only | **1.5°** | 4.8° | 9.6° | 19.5° |

**At one step the cones make the direction estimate worse — much worse — and
from four steps on they make it better.** The same split holds for
`derot_ego` (8.0° → 15.0° against heading-only's 1.5° → 19.5°) and under
uniform turns (lag 1: 11.4° against 1.4°; lag 8: 42.9° against 58.2°), so it is
not an artifact of one framing or one persistence. The mechanism is not subtle:
at lag 1 the heading *is* the displacement direction, so nothing can beat it and
the cones only add noise; by lag 8 the heading is one step's worth of a
cumulative move and stops determining it, so the cones have something to
contribute.

**And that is exactly the wrong way round for this project.** The regime where
the cones help on direction — multi-step displacement — is the regime where the
agent's own integration is *exact and free*, because summing `prev_displacement`
over eight steps is as accurate as over one (§6.1: 2.3e-14 over two hundred).
The regime where sensory would have to substitute, a single step, is the one
where it is worst. There is no lag at which the cones supply a direction the
agent does not already have more precisely.

**(b) The cones frequently do not see the same world.** The aperture is 120°
and heading is the direction of travel, so two consecutive views point wherever
two consecutive actions pointed. Under uniform random actions the median
`|Δψ|` between consecutive steps is **91.6°**, the median number of rays the
two views share is **14 of 60**, and **35.0% of consecutive pairs see
completely disjoint parts of the world** — for a third of the pairs there is no
shared world in the two observations at all, so nothing can relate them.

That is a fact about the aperture and the movement model, not about the code,
and it is separable from (a) because walk persistence moves it. The policy is
paid to go straight (`PERSISTENCE_BONUS=0.05`), so a uniform random walk is the
pessimistic end of this axis and the table above sweeps across it.

**So (a) and (b) are both real, and the sweep separates them.** Under a random
walk the two cones are usually not looking at the same world and contribute
almost nothing. Turn persistence up until they are, and they contribute a lot —
but what they contribute is a magnitude correction to a displacement the agent
is already handed exactly, while making its direction estimate worse. That is a
sharper result than "the sensor is uninformative", and it is only visible
because the shuffled control keeps the heading terms.

And the range sensor of §6.7 shows the damage is not the ±1 code's doing. That
sensor decodes at 0.869 / 11.1° in the `fixed` framing, near the 0.945 ceiling.
Under free heading it drops to **0.035 / 81.7°**. A sensor that is otherwise
almost at the ceiling loses about 95% of its R² the moment the heading is free.
**Whatever is destroying the free-heading case is not the hash, because it
destroys an un-hashed sensor just as thoroughly.**

### 6.6 Experience in the new env buys nothing, and one table says why

`k ∈ {0, 4, 16, 64, 256}` steps of experience in the held-out env, 24 test
envs, `fixed` framing (the favourable one), `wall_resolution=4`:

| decoder | k=0 | k=4 | k=16 | k=64 | k=256 |
|---|---|---|---|---|---|
| cross-env only | 0.086 / 57.0 | — | — | — | — |
| NN table (label-free) | — | −0.50 / 90.0 | −1.40 / 90.0 | −5.65 / 90.0 | −12.91 / 77.3 |
| self-sup ridge | — | — | −0.001 / 84.4 | −0.001 / 83.4 | −0.188 / 81.1 |
| cross-env + self-sup | — | — | 0.059 / 63.9 | 0.064 / 60.3 | 0.023 / 65.9 |

(R² / median angular error in degrees. The `k=0` row is an independent refit of
the §6.3 `ridge/xcorr` decoder on 48 training envs rather than 64, and it lands
at 0.086 / 57.0 against 0.084 / 56.9 — a reproducibility check across separate
jobs that is worth having.) **256 steps — more than a full explore episode —
leaves the decoder no better than zero steps.** In the `free` framing every cell
is at chance. So the answer to "and maybe with some experience in the new env"
is: no, and there is no number of steps that changes it.

The NN row's flat 90.0° is not a coincidence: those are collapses, where the
table returns the same entry for both queries and the predicted displacement is
exactly zero. Dropping such samples is the obvious implementation and it
silently flatters the decoder — they are scored 90°, exactly uninformative, and
`frac_zero_pred` reports the rate. The first version of this study dropped them
and crashed on an all-zero arm, which is how the problem was found.

Two remarks on how the arms are posed. First, the spec separated *supervised
anchors* (an unrealistic upper bound) from *self-motion self-supervision* (the
realistic regime). For decoding a **displacement** they are the same
computation: an anchor table keyed by absolute position and the same table
keyed by position-relative-to-the-trajectory-start differ by a constant, and a
difference of two matched entries cancels it. The unrealistic upper bound is
reachable with no labels at all — and it is the row that fails hardest. Second,
ridge is the wrong in-env decoder, which is why §6.3 reports the table: within a
single env a linear fit gets 0.088 while the full 400-cell table gets 0.943.

Why `k` anchors do not help is one table — median cosine between cones at ψ=0,
by grid distance:

| sensor | wall_res | same cell | d=1 | d≤2 | d≤3 | d≤5 | d≤8 | d≤13 | d>13 | **p99.75 far** |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 1.000 | 0.283 | 0.150 | 0.050 | 0.033 | 0.033 | 0.000 | 0.000 | **0.550** |
| code | 2 | 1.000 | 0.133 | 0.117 | 0.067 | 0.033 | 0.033 | 0.000 | 0.000 | **0.433** |
| code | 4 | 1.000 | 0.100 | 0.017 | 0.033 | 0.033 | 0.017 | 0.000 | 0.000 | **0.400** |
| code | 8 | 1.000 | 0.067 | 0.033 | 0.017 | 0.017 | 0.000 | 0.000 | 0.000 | **0.367** |
| range | any | 1.000 | 0.990 | 0.972 | 0.933 | 0.826 | 0.544 | −0.123 | −0.647 | **1.000** |

The `same cell` column reads exactly 1.000 by construction and is there as the
guard that the quantity is what it claims to be. The last column is the bar a
true match has to clear: with ~400 candidate cells the best wrong one sits at
the p99.75 of the far-pair similarity.

**The ±1 cone carries no locality.** At `wall_resolution=4` a cell one step away
sits at 0.100 while the best distractor sits at 0.400, so the neighbour of a
stored view essentially never wins the argmax — and coarsening the code does not
fix it, because the distractor bar falls just as fast. A stored view therefore
localizes the agent only when it is standing on **exactly** the cell it was
stored from. "Experience in a new env" means the cells you have literally
occupied, not the region you have explored, and that is why the table decoder
gets *worse* with more anchors: more entries means the wrong match is drawn from
a wider spread of positions, so the error grows.

The range sensor is the control that shows this is the code's doing and not the
cone's: its similarity decays *smoothly* — 0.990 at one cell, still 0.826 at
five. Yet its table also fails (best 0.001 / 90.0° at k=4), for the opposite
reason given in §6.7: ranges alias, so its p99.75-far similarity is **1.000**
and a wrong cell ties the right one exactly. One sensor has no locality, the
other has no uniqueness, and a table needs both.

Put that next to §5.8. The **encoder** holds `ξ(p)·ξ(goal)` at 0.9993 within 1.5
cells and 0.75 corner to corner — a tight, smooth chart of the arena. The **raw
cone it is built from** is at 0.100 one cell away. Whatever generalizes in this
system is manufactured by the encoder; none of it is present in the sensor.

### 6.7 The geometry is there — the ±1 hash is what removes it

Every arm was re-run on a **range** sensor: the same 60 rays, the same cone,
the same plane intersections, returning the distance to the wall instead of the
±1 code of the segment hit (`raycast_range`). That is the lidar this env does
not have, and it bounds what restructuring the sensory input could buy.

`fixed` framing, `wall_resolution=4`, 48 held-out envs:

| decoder | ±1 code | range |
|---|---|---|
| snap-oracle **CEILING** | 0.945 / 8.6° | 0.945 / 8.6° |
| mlp/raw | 0.132 / 51.3° | **0.869 / 11.1°** |
| ridge/bilin | 0.091 / 55.1° | 0.712 / 19.5° |
| ridge/spec (the spec's features) | −0.000 / 89.5° | 0.775 / 15.7° |
| shuffled control | 0.001 / 86.0° | 0.018 / 83.0° |

At one step (lag 1, `‖Δpos‖` 1.19) the range sensor reaches 0.749 / 16.4°
against a lag-1 ceiling of 0.802 / 14.0°. **The geometry a cross-env decoder
needs is fully present in the cone; the ±1 hash is what removes it** — a factor
of six in R² and a factor of five in angular error, on identical rays.

Two details worth carrying. First, even `[s1, s2, s1−s2, s1⊙s2]` — the spec's
own feature set, which is *exactly chance* on codes — reaches 0.775 on ranges.
Nothing clever is required once the signal is not hashed; the elaborate
codebook-independent features exist only because the code needed them. Second,
the range sensor's in-env table is *worse* than its own regression (0.398, with
17.2% of its predictions exactly zero, against 0.712 for ridge) because ranges
**alias**: two cells the same distance from the wall the cone faces, both clear
of the side walls, produce identical profiles. Smoothness and injectivity pull
against each other, and the ±1 code sits at the injective end. That is the real
design tension behind `wall_resolution`, stated as a measurement rather than a
preference.

The two sensors also fail in opposite halves of the arena. The code decodes
best nearest the wall the cone faces (§6.8); the range sensor decodes best
furthest from it (0.946 in the two `y_lo` quadrants against 0.79 in the two
`y_hi`), which is what a profile with more dynamic range at distance would
predict. Neither is a defect — they are different measurements of the same
geometry.

### 6.8 Where the surviving signal lives

Breakdowns of the best cross-env arm, `fixed` framing at resolution 1 (the
configuration where there is enough signal to break down at all):

| decoder | xlo_ylo | xlo_yhi | xhi_ylo | xhi_yhi |
|---|---|---|---|---|
| ridge/bilin, 64 envs | 0.324 / 33.3° | 0.451 / 25.1° | 0.319 / 33.6° | 0.442 / 25.3° |
| mlp/raw, 128 envs | 0.646 / 21.6° | 0.670 / 18.2° | 0.639 / 21.7° | 0.664 / 18.2° |

The cone faces North, so the two `y_hi` quadrants are the half of the arena
nearer the wall it is looking at, and they decode better — a nearer wall
subtends more of the cone per unit of translation. The effect is large for the
linear decoder (0.45 against 0.32) and mostly flattens for the MLP with twice
the training envs (0.67 against 0.64). **The anisotropy is a property of the
weak decoder, not of the observation**, which is a useful thing to know before
reading anything into a spatial breakdown of a marginal signal.

The distance-to-nearest-wall breakdown runs the other way — 0.267 / 37.7° at
0.5–1.5 rising monotonically to 0.512 / 22.5° at 7.5–10 — and the two are not in
conflict. `d_wall` is the distance to the *nearest* wall in any direction while
the cone is directional, so its near bin mixes cells facing a close wall with
cells backed against one and looking down the length of the arena. **The
quadrant split is the interpretable one; `d_wall` is confounded for a
directional sensor**, and it is reported here only so the confound is on the
record rather than quietly informing a conclusion.

### 6.9 Decision

**The explore target stays at billiard 0.378 on sensory grounds, and moves for a
different reason.**

- **Does relative displacement decode well enough in a held-out env that the
  lawnmower ceiling reopens?** No. It decodes *somewhere* — up to R² 0.658 at
  19.8° with both cones pinned North, a coarse code and 128 training envs, so
  the cross-env geometry is genuinely learnable. At the agent's own headings,
  and at the walk persistence the `p5_e` explore checkpoint was *measured* to
  have, the best decoder adds +0.254 of R² over its own shuffled control — a
  real channel, not a trace. But on **direction**, which is what path
  integration needs, it helps only at four steps and beyond, and at one step it
  is far worse than the agent's own heading (8.0° against 1.5°). Four steps and
  beyond is exactly where integrating `prev_displacement` is already exact, so
  there is no lag at which the cones supply a direction the agent lacks. At the
  launcher's own `wall_resolution` even the pinned-North number is only R² 0.13
  at 51°. Nothing here supports path integration from sensory.
- **How many steps of adaptation would it need?** None works. 256 steps of
  experience in the new env leaves it where zero steps left it, and the locality
  table says why: a stored cone identifies the cell it was stored from and
  nothing within one step of it.
- **But the lawnmower ceiling is not blocked**, because §4's B3 already hands
  the agent exact self-motion. The information a lawnmower needs is present in
  the observation from step 1. **P5's target should therefore be set by what the
  policy can do with `prev_displacement`, not by what can be read off the cone**
  — and if P5 plateaus at billiard, the diagnosis is recurrent capacity or
  reward shape, not the sensor.
- **Jack's follow-up question — "should we structure sensory input differently?"
  — is live, and now has a number.** A range sensor over the same 60 rays, the
  same cone and the same geometry decodes displacement in a held-out env at
  **R² 0.869 / 11.1°** against a ceiling of 0.945 / 8.6° — six times the R² of
  the ±1 code on identical rays. The geometry is in the cone; the hash is what
  removes it. `WALL_RESOLUTION` is the same lever from the other end, worth a
  factor of 4 between 4 and 1. Neither change is free: the code is injective
  where the range profile aliases, and `wall_resolution=4` is there so that two
  positions inside one cell read differently.
- **The one thing no sensor change fixes** is §6.5. With a 120° aperture and
  heading locked to the direction of travel, 35% of consecutive glimpse pairs
  are disjoint under random actions — but the `p5_e` explore checkpoint is
  persistent enough that only **0.1%** of its pairs are, so the aperture need
  not bind. What binds instead is the **heading coupling**. Because ψ is
  `atan2` of the realized displacement, at one step the heading already *is* the
  answer's direction and the cones cannot improve on it; by the time they can
  (four steps and beyond) the agent's own integration is exact. A cone pinned to
  the direction of travel can only ever report on where the agent is going, not
  on where it is. That is the thing to change if visual odometry is ever wanted
  here — not the code, and not the aperture.

**What this hands the other workstreams.** §7.4's wall hypothesis asked for a
`q`-independent check on self-motion: `prev_displacement` is exactly that, and
it is exact — and §7.2's `d2` statistic, specified as "P2's sensory-decoded
displacement vs the commanded action", should be replaced by
`prev_action − prev_displacement`, which is the same diagnostic computed exactly
instead of estimated. And §6.6's locality table is a constraint on any future
proposal to use stored sensory views as a place memory: they are usable as an
exact match and as nothing else.

### 6.10 What P2 did not resolve

Four things, in the order they would change a conclusion.

1. ~~The trained policy's turn distribution.~~ **Partly measured and folded
   into §6.5** — `analysis/nav_p2/policy_turn_stats.py`, one rollout of the
   `p5_e` explore checkpoint. It was the largest open item on this list and it
   moved the answer: that policy turns a median of 14.6° per step with 0.1% of
   its consecutive view pairs disjoint, so it sits at the *favourable* end of
   the sweep, where the cones add +0.108 rather than +0.027 — the sweep alone
   would have understated the sensory channel by 4×. Its `persistence_bonus
   0.20` / `epsilon_explore 0.1` are shared by **every** variant block in
   `run_nav_p2.sh`, so it is representative of the lineage rather than an
   outlier in it.

   An attempt to bracket the axis with a second checkpoint **failed, and the
   failure is worth recording**. Running the same probe on the P4 *exploit*
   specialist (`p4_x` u1200) in the explore regime returns a degenerate row —
   median `|Δψ|` 0.0° with p90 180.0°, because **45.5% of its steps do not move
   at all** and a still step carries the previous heading forward. Its median
   step norm is 0.50, the `MIN_ACTION_NORM` floor. That is not a
   low-persistence walk to compare against; it is a policy operating far
   outside the regime it was trained for, with an empty memory and nothing to
   navigate toward. The `frac_still` diagnostic is what made it visible rather
   than letting "median turn 0.0°" read as perfect persistence. **A real
   bracket still needs an explore-schedule policy trained at a different
   persistence, which does not exist yet**, and the same measurement is owed
   for the interleaved P6 agent.

   (The 45.5%-still figure is a finding for P4/P6 in its own right: the exploit
   specialist has no explore behaviour to fall back on when memory is empty. It
   stands still at the minimum step norm. Anything that interleaves the two
   regimes inherits that.)
2. ~~How much more a non-linear decoder gets in the ego framing.~~ **Resolved
   and folded into §6.5.** The MLP more than doubles the ridge increment
   (+0.254 against +0.105 at the operating point), and building it its own
   shuffled control rather than borrowing the ridge's mattered: `shuf/mlp-raw`
   reaches 0.402 where the ridge's control sits at 0.290, because a non-linear
   decoder reads the *heading* terms much better too. Borrowing would have
   claimed +0.37 and overstated the cones by nearly half. The same run also
   showed the explicit de-rotation to be unnecessary — a non-linear decoder
   handed `Δψ` learns the rotation better than the hand-built shift applies it.
   It also overturned the ridge-only
   reading that the increment was "all magnitude" — it is magnitude at one step
   and direction from four steps on, which is §6.5's sharpest result.
3. **The decoder class more broadly.** `bilin` spans the complete second-order
   statistic, which is the codebook-independent sufficient statistic for the
   *expected* cross-view structure, and the MLP adds nonlinearity on top of it.
   Higher-order codebook-independent structure exists in principle and was not
   exhausted; an end-to-end recurrent decoder over a whole trajectory is a
   different and strictly larger hypothesis class than anything run here. The
   in-env table result (0.943, saturating the ceiling) shows the information is
   present within an env, so the ceiling on transfer is not obviously tight.
4. **Whether a range sensor would survive contact with the rest of the system.**
   §6.7 measures only displacement decoding. The ±1 code was chosen to make
   nearby cells *distinguishable*, which is what the encoder and the Hopfield
   store need, and the range profile aliases badly (p99.75 far-pair similarity
   **1.000**). Swapping the sensor would have to be re-scored against
   `unique_radius`, not against this section.

---

## 7. P3 — ideal observer: when is there signal that the goal is in memory?

The centrepiece, and the one that bounds mode B. Jack's three candidate cues are
groups (A), (B) and (D) below; group (C) is an addition.

### 7.1 Three questions — REVISED after §5

- **Q_ep, episode-level** — does memory contain a pattern stored in *this* env?
  This is what "explore vs exploit" means to the agent.
- **Q_step, step-level** — is the *current* recall `recall(x_t)` the goal rather
  than a foreign pattern?
- **Q_trust, step-level — ADDED, and the one that matters most.** Is this
  recall's *direction* reliable right now?

**Note from P2 (§6) — group B is stronger than this spec assumed.** The
allocentric spread `b2` accumulates `Σd`, and the original worry was drift. P2
measured it: integrating `input_prev_displacement` over 200 steps × 4000
episodes reproduces position to **2.3e-14**. So `b2` is drift-free by
construction, not approximately — the policy is handed an exact position and the
question is only whether it uses it. Conversely **group D is weaker than
assumed**: P2 found a stored cone identifies only the cell it came from
(similarity 0.100 at one cell against 0.400 for the best of ~400 wrong cells),
so any observation-prediction residual (`d1`) will be a fingerprint check rather
than a graded signal. Weight the ablation's expectations accordingly.

**Why Q_trust was added.** The spec originally had only the first two, on the
reasoning that mode B is a per-step failure to follow a usable recall. But §5.2
measured what actually determines usability, and it is neither of them: cells
whose recall sits at `cos_goal ≥ 0.99` have a **0.01%** rate of bad direction,
and **99.7%** of all bad-direction cells fall below that threshold. Direction is
the only thing the policy consumes, so "should I follow `q` right now" is the
operational question, and it is *not* the same as "is this the goal" — §5.2 also
found that a recall which locked on the wrong pattern still usually points
roughly the right way.

`cos_goal` is available as a ground-truth label at analysis time and is not
available to the policy, so it is a clean supervised target. Q_trust is the
headline; Q_ep and Q_step stay as secondary targets because the regime question
is what P6 needs.

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

`c1` and `c2` therefore measure **drift rate**, not spuriousness. Re-derived
from the linear theory rather than from attractor folklore: `recall(x) ∝ W x`,
`W` has an eigen-decomposition, and iterating amplifies the top eigenvector at a
rate set by how the cue's energy is distributed over that spectrum. So the
residual is a measure of the cue's **spectral concentration on the stored set** —
how cleanly it picks out one pattern rather than sitting in a region several
patterns share.

That is genuinely informative, and it is measured: the one-step residual is
**0.148** for imperfect-fidelity cells against **0.099** for clean ones. It is
also free, since the iterates are already computed and already fed to the
policy. So both statistics stay in the feature set.

**But note the consequence for the ablation.** Spectral concentration on the
stored set is closely related to `‖q‖`, which is group A. **A and C are probably
substantially redundant**, and a leave-one-out ablation would report both as
contributing nothing when in fact they carry the same real signal twice. §7.3
item 3 handles this with leave-one-in and a joint A∪C drop.

Drop the earlier claim that a single-step separation would dissolve the probing
question — it followed from the spurious-state framing, which is gone.

**D. Sensory consistency** *(Jack's third cue)*

| | statistic |
|---|---|
| `d1` | residual between the observation predicted by the recalled pattern and the actual observation |
| `d2` | mismatch between the **realized** displacement and the commanded action |

`d1` needs a `pattern → obs` decoder, fitted offline; it is a property of the
scaffold, not the agent. `d2` was specified as "P2's sensory-decoded
displacement vs the commanded action" and **§6 replaces it with something
strictly better**: there is no usable sensory-decoded displacement, but
`prev_displacement` carries the realized move exactly, so `d2` is
`prev_action − prev_displacement`, which is non-zero exactly when the arena
clipped the step. That is the wall diagnostic, it is exact rather than
estimated, and both channels are already fed (§4 B3).

### 7.3 The analysis

0. **Sanity anchor, wired in before anything else.** §5.8 predicts the
   goal-absent `‖q‖` analytically: an unrelated direction in D dimensions keeps
   `√(2/D) = 0.0442` of its norm under the tangent projection, and with a
   displacement norm of ≈ √2 that gives **0.0625** against a measured 0.0670,
   with goal-present at 0.3006. So group A's separability has an *expected*
   value derivable from first principles. **If the measured AUC comes in far
   below what that separation implies, the measurement is wrong, not the
   signal.** Three instrumentation faults in §5 produced confident wrong answers
   before being caught; this check is cheap and goes in first.
1. **Per-statistic separability** — AUC for all three targets as a function of
   distractor count {0,1,2,3,5,7,10} and steps observed
   `t ∈ {1,2,4,8,16,32,64}`, reported as a **distribution over envs and seeds**,
   per finding 19.

   **`n_dist = 0` is degenerate and must not be pooled into a headline.** With
   no distractors the goal-absent memory is *empty*, so `q = 0` exactly — a
   perfect cue, for a trivial reason that has nothing to do with the signal
   being measured. It is a real condition (training samples 0–10), so report it,
   but separately.
2. **The ideal-observer bound** — a classifier (logistic regression, plus
   gradient-boosted trees for the nonlinear ceiling) on the full feature vector,
   cross-validated across *held-out envs*. Yields `AUC(t, n_dist)`. **This is
   the number that bounds mode B.** If AUC at `t = 3` is 0.99 the policy's
   regime failures are inexcusable and P6 should fix them; if it is 0.65 the
   policy is near-optimal and the fix must be elsewhere.
3. **Feature ablation** — drop each of A/B/C/D; the AUC loss is that group's
   unique contribution. Answers which cue actually carries the discrimination,
   which is Jack's question 3 restated.

   **Expect A and C to be redundant, and design for it.** §5.4 showed the
   dynamics is linear power iteration, so group C measures how fast a cue
   rotates toward the top eigenvector — which is a function of its spectral
   concentration on the stored set, and that is closely related to `‖q‖` in
   group A. A leave-one-out ablation reports *unique* contribution and will show
   both as near-zero if they carry the same information. **Also run the
   leave-one-in (each group alone) and the A∪C joint drop**, or genuine signal
   will be hidden by its own duplication.
3b. **Does recall depth earn its input channel?** `--input_hopfield_multistep
   1 2 3` feeds `q` at three depths, and §5.3–5.4 established that depths 2 and
   3 are strictly *degraded* states rather than better-converged ones, sampling
   the transient of a power iteration. The decay *rate* may still discriminate,
   but that is now a claim to test rather than assume: compare the classifier
   with depths {1} against {1,2,3}. **If depth adds nothing, that is a policy
   input channel that can be removed** — a real result, not a null one.
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
6. **Wall interaction — RE-SCOPED.** The original plan was to condition
   everything on distance-to-wall to test H-wall. Half of that is already
   answered and was answered negatively: §5.2 measured `dir_cos` as **flat**
   against distance-to-wall, so the readout is *not* degraded near boundaries.
   What survives is narrower and sharper — a **channel ablation**. Compute the
   self-motion residual `a3` and the allocentric spread `b2` twice, once from
   the *commanded* action and once from the *realized* displacement, and compare
   their separability near walls and clips. Both are policy inputs now
   (§4 B4), so this is a concrete comparison between two available signals
   rather than a hypothesis about `q`. See §7.4.

**Deliverables.** `AUC(t)` curves per cue per distractor level with per-env
bands; a steps-to-0.95 table per probe; the ablation table including the
leave-one-in and A∪C variants; the depth-{1} vs depth-{1,2,3} comparison; and a
single headline number — the **Q_trust** ideal-observer AUC at the distractor
levels we train on, which every mode-B claim in P6 gets measured against.

### 7.4 A hypothesis this analysis is built to test — H-wall

`prev_action` currently carries the **committed** action
(`rollout/collector.py:652`), not the realized displacement. The two differ
whenever the arena clip bites — and, once §2.1's bounds are on, whenever the
norm clamp bites. But `a3` and `b2`, the two sharpest regime cues, both need the
*realized* displacement: they compare how much `q` changed against how far you
actually moved.

**So near a wall the agent's best regime cue is corrupted, in the direction of
declaring a real goal a phantom.** Phase 1 measured `fail_frac_at_edge = 0.389`
for the combined model — 39% of its mode-B failures end against a wall.

**Status after §5 — half refuted, half sharpened.** The readout half is dead:
`dir_cos` is flat against distance-to-wall (§5.2), so `q` is not degraded near
boundaries and the wall failures are not explained by a worse signal there.
Phase 1's own matched control also removed the explore-leakage reading — an
exploit-only model that has never explored fails at walls *more* often (0.472
against 0.389), so walls are hard for any policy here.

What survives is the displacement half, and it is now a clean comparison rather
than a hypothesis, because both signals are policy inputs (§4 B4): compute `a3`
and `b2` from the **commanded action** and from the **realized displacement**,
and compare their separability conditioned on distance-to-wall and on clip
events. If the commanded version degrades near walls and the realized one does
not, the mechanism is confirmed and the fix is already shipped — the question
becomes whether the policy *uses* the right channel, which is a P6 question.

### 7.5 RESULT — the bound, and the answer to the gating question

`analysis/nav_p2/ideal_observer.py` generates the evidence, `..._fit.py` fits
it, `..._score.py` applies the result to trained policies, and
`io_features.py` holds the cue statistics in one implementation all three
import. Launcher `hopfield_nav/run_nav_p3_io.sh`. Every table below is
reproducible from `results/nav_p2/`:

| file | what it holds |
|---|---|
| `io_probe.npz`, `io_probe_gen.log` | the 7-probe feature tensor and its anchor / group-D controls |
| `io_prober.npz`, `io_prober_fit.log` | the 16-probe parametric sweep (§7.8) |
| `io_probe_fit.log`, `io_probe_main.json` | per-statistic and ideal-observer tables (§7.5, §7.7) |
| `io_ablation_walls.log` | the cue ablation and the wall conditioning (§7.7, §7.10) |
| `io_single_full.log` | the full single-feature ranking at ten distractors |
| `io_trust30deg.log`, `io_trustpresent_pooled.log` | the two Q_trust sensitivity arms |
| `io_agents_score.log`, `io_agents2_agents.npz` | the trained-policy pass (§7.9) |

**The grid.** 48 freshly drawn envs × 7 distractor levels {0,1,2,3,5,7,10} × 8
distractor draws × 2 regimes × 2 start cells × 7 probe policies =
**75,264 episodes of 64 steps**, features saved at twelve values of *steps
observed*. A second run adds a 16-probe parametric sweep (§7.8). Every
classifier is cross-validated over **held-out envs**, six folds of eight:
splitting rows at random would let it memorise a wall code and report an AUC no
policy in a new arena could realise.

**The instrument was checked before it was read.** Four checks, all of which
had to pass:

| check | result | what it rules out |
|---|---|---|
| frame self-test | oracle `cos(q, goal−x)` **0.9952**, axis-swapped **−0.3220** | `gram_schmidt_2d_batch` is handed `d_forward` first and returns it *second*; if `q` and the realized displacement were in transposed frames, `a3` and `b2` would be noise and nothing else here would say so |
| analytic anchor (§7.3 item 0) | reproduces §5.8 at every level, below | a broken readout reading as a weak signal |
| label permutation | **0.452 – 0.549** over all 126 estimable headline cells | leakage across the fold boundary |
| constant features | **0.500** exactly | a tie-handling bug in the AUC itself |

The last one was not free. Out-of-fold scores must be pooled by *average* rank:
each fold's model carries its own intercept, so raw scores are not comparable
across folds, and a naive rank normalisation breaks ties in row order — which is
grouped by regime. The constant-feature control read 0.518 until ties were
averaged. It is the leakage detector, so it has to be exact.

**The anchor reproduces §5.8 across every level**, which is the point of it:

| `n_dist` | median `‖q‖` goal-present | goal-absent | ratio |
|---|---|---|---|
| 0 | 0.3135 | 0.0000 | — (degenerate) |
| 1 | 0.3127 | 0.0488 | 6.41 |
| 2 | 0.3128 | 0.0518 | 6.04 |
| 3 | 0.3117 | 0.0581 | 5.36 |
| 5 | 0.3100 | 0.0701 | 4.43 |
| 7 | 0.3080 | 0.0782 | 3.94 |
| 10 | 0.3011 | 0.0857 | 3.51 |

§5.8 measured 0.3006 and 0.0670 and predicted the second at
`√(2/D)·√2 = 0.0625`. Both land, and the goal-absent value grows with the
number of distractors exactly as a maximum over more unrelated directions
should.

#### The base rates settle most of it before any classifier runs

Q_trust asks whether following `q` right now is a good idea. Its *prior*,
measured on `billiard` trajectories pooled over all twelve step counts, with
steps standing inside `goal_radius` excluded in **both** regimes so the mask
cannot carry the label:

| `n_dist` | regime | `P(cos ≥ 0.5)` | `P(cos ≥ 0.866)` (30°) | `P(cos ≥ 0.966)` (15°) | median `cos` |
|---|---|---|---|---|---|
| 1 | goal present | 0.9985 | 0.9943 | 0.9564 | 0.9970 |
| 1 | goal absent | 0.3150 | 0.1521 | 0.0791 | −0.0485 |
| 3 | goal present | 0.9980 | 0.9933 | 0.9427 | 0.9966 |
| 3 | goal absent | 0.3204 | 0.1500 | 0.0758 | −0.0576 |
| **10** | **goal present** | **0.9855** | **0.9620** | **0.8827** | **0.9957** |
| **10** | **goal absent** | **0.3196** | **0.1653** | **0.0798** | **−0.0220** |

The goal-absent row is a control that came out perfect. For a *uniformly*
directed 2-D vector the three probabilities are 1/3, 1/6 and 0.083; the measured
values are 0.320, 0.165 and 0.080, with a median cosine of −0.02. **A recall
that locked onto a foreign pattern is directionally indistinguishable from
noise** — which is what §5.8's dimensionality argument predicts and what makes
the goal-absent rows readable against a known floor.

The goal-present row is the one that matters for P6. **At ten distractors, with
the goal in memory, `q` points within 30° of it on 96.2% of steps and within
15° on 88.3%.** The median is 5.3° of error.

#### The ideal-observer AUC

Full policy-visible feature set — groups A ∪ B ∪ C, 21 statistics, every one a
function of `q` at the three recall depths already fed to the policy plus
`prev_action` and `prev_displacement`, both policy inputs since §4 B3.
Gradient-boosted trees on `billiard` trajectories, six-fold CV over held-out
envs; the per-env column is the median over the 48 envs with [p10, p90].

**Q_trust** (`cos(q, goal − x) ≥ 0.5`):

| `n_dist` | t=1 | t=2 | t=4 | t=8 | t=16 | t=32 | t=64 | chance | per-env at t=64 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.861 | 0.860 | 0.873 | 0.869 | 0.898 | 0.896 | 0.899 | 0.545 | 0.936 [0.833, 0.985] |
| 3 | 0.824 | 0.837 | 0.862 | 0.877 | 0.887 | 0.905 | 0.889 | 0.483 | 0.893 [0.810, 0.959] |
| 7 | 0.753 | 0.798 | 0.848 | 0.869 | 0.883 | 0.881 | 0.891 | 0.491 | 0.893 [0.822, 0.952] |
| **10** | **0.760** | 0.799 | 0.855 | 0.858 | 0.874 | 0.886 | **0.884** | 0.525 | 0.885 [0.783, 0.958] |

**Q_ep** (is anything from this env in memory at all):

| `n_dist` | t=1 | t=2 | t=4 | t=8 | t=16 | t=32 | t=64 | chance | per-env at t=64 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.983 | 0.987 | 0.991 | 0.994 | 0.994 | 0.999 | 1.000 | 0.508 | 1.000 [0.999, 1.000] |
| 3 | 0.939 | 0.968 | 0.986 | 0.993 | 0.995 | 0.998 | 0.998 | 0.512 | 1.000 [0.996, 1.000] |
| 7 | 0.863 | 0.937 | 0.979 | 0.983 | 0.990 | 0.996 | 0.998 | 0.513 | 1.000 [0.995, 1.000] |
| **10** | **0.872** | 0.938 | 0.979 | 0.985 | 0.989 | 0.990 | **0.996** | 0.501 | 1.000 [0.973, 1.000] |

Logistic regression tracks the trees closely for Q_trust (0.788 → 0.874 at ten
distractors) and lags them mid-episode for Q_ep (0.887 at t=4 against 0.979),
so Q_ep's cue is mildly non-linear — consistent with §7.7, where the working
statistic is a running *order statistic* of `‖q‖`.

**The ceiling control earns its place at t = 1 and nowhere else.** Adding the
two *D-dimensional* recall statistics — `‖r¹ − r²‖` and `cos(r¹, r³)` before
the tangent projection, an input channel that could be fed but is not — lifts
Q_ep at ten distractors from 0.872 to **0.937** at a single step, and from
0.979 to 0.981 at four steps. By t = 8 the gap is gone. So the 2-D projection
does throw away something real, and what it throws away is worth about
**0.065 of AUC on the first step only**; a couple of steps of motion recovers
it for free.

A sensitivity check at a stricter Q_trust threshold (`cos ≥ 0.866`, which
brings the pooled positive class to 57%) gives 0.872 → 0.926 at three
distractors and 0.805 → 0.917 at ten. The headline is not an artefact of a
97%-positive class.

**Q_trust restricted to goal-present episodes** — the version that speaks
directly to mode B, since during exploit the goal *is* in memory — reads 0.978
at t = 1 and 0.953 at t = 64 at ten distractors. **It should not be quoted as a
headline**, and the reason is the label-permutation control printed beside it,
which on those same cells ranges from **0.535 to 0.728** rather than sitting at
0.500: with `P(y) = 0.983`, 768 rows contain about 13 negatives, and the AUC is
not estimable at that class balance. This is the one place in §7 where a
control failed, and the number it guards is the one a mode-B claim would most
like to cite. What *is* solid there is the base rate above, which needs no
classifier at all.

**The degenerate condition, reported separately** as §7.3 item 1 requires. At
`n_dist = 0` the goal-absent memory is empty, `q = 0` exactly, and Q_ep is
separable at AUC **1.000** at every step count — for a reason that has nothing
to do with the signal being measured. Q_trust has no negative rows there at all
(`dir_cos` is undefined when `q = 0`), which is why `valid_trust` is a separate
mask from `valid`: one shared mask would have deleted every negative row of the
degenerate condition and left a one-class problem reading as a perfect AUC.

#### The gating answer

**Mode B is a learning problem, not an information problem.** Three independent
numbers say so, and they do not depend on each other:

1. **The prior alone almost answers it.** When the goal is in memory at ten
   distractors, `q` points within 30° of it on **96.2%** of steps. A policy that
   ignores `q` is not being cautious about a signal it cannot verify; it is
   declining a signal that is right 24 times in 25.
2. **When it *is* worth not following, that is detectable.** An ideal observer
   on the policy's own inputs separates the unusable steps at AUC **0.76 after a
   single step** and **0.88 after eight**, against a label-permuted control at
   0.52. The warning is available from the inputs the policy already receives.
3. **The regime question is more available still.** Q_ep is **0.87** from one
   step and **0.99** from four, at ten distractors, with per-env p10 of 0.973 at
   t=64 — so this is not a mean hiding a bad tail.

So P6 may treat mode-B failures as failures of credit assignment, of
exploration in policy space, or of representation — not as a policy correctly
reporting an absence of evidence. **The one caveat is §7.9, and it is
specific**: a regime detector *fitted on exploration data* will invert on a
successful exploiter, so the gating signal has to be built from statistics that
survive the agent's own approach to the goal.

### 7.6 Q_step is Q_ep, and can be retired

§7.1 kept `Q_step` — is the current recall the goal rather than a foreign
pattern — as a separate target. Measured, it is not one:

| `n_dist` | `P(lock = goal ǀ goal present)` | `P(lock = goal ǀ goal absent)` |
|---|---|---|
| 1 | 0.999 | 0.000 |
| 3 | 0.999 | 0.000 |
| 7 | 0.992 | 0.000 |
| 10 | 0.989 | 0.000 |

The goal-absent column is **identically zero**, trivially, since the goal is not
stored; the goal-present column is 0.989 at ten distractors, which is §5.1's
lock rate seen from the other side. So `Q_step` and `Q_ep` disagree on 1.1% of
rows at the hardest level, and every table computed for the two is the same
table to three decimals. It is reported for completeness and should not be
carried into P6 as a third question.

### 7.7 Which cue carries the discrimination — `‖q‖` always, `b2` when the arena allows

Question 3 restated. Single-feature AUC, per-env median over 48 envs,
`billiard`, ten distractors; sign-corrected, so a cue that predicts the negative
class still scores above 0.5.

| cue | statistic | Q_ep t=1 | t=8 | t=64 | Q_trust t=1 | t=64 |
|---|---|---|---|---|---|---|
| **D** | `d1_chart` — recall's residual from this env's pattern subspace | **1.000** | **1.000** | **1.000** | 0.876 | 0.867 |
| **D** | `d1_valid1` — its decoded-observation validity | **1.000** | **1.000** | **1.000** | 0.879 | 0.883 |
| *oracle* | `cos_goal` — not policy-visible | 1.000 | 1.000 | 1.000 | 0.910 | 0.892 |
| **A** | `a6_q_max` — running max of `‖q‖` | 0.887 | 0.898 | **0.963** | 0.792 | **0.848** |
| **A** | `a5_q_mean` — running mean of `‖q‖` | 0.887 | 0.887 | 0.958 | 0.792 | 0.848 |
| **A** | `a5_q_std` — running s.d. of `‖q‖` | 0.594 | 0.738 | 0.949 | 0.587 | 0.841 |
| **A** | `a1_qnorm` — `‖q‖` itself | 0.887 | 0.881 | 0.881 | 0.792 | 0.807 |
| **B** | `b3_drift_mean` — mean drift of the allocentric goal estimate | 0.500 | 0.725 | 0.906 | 0.500 | 0.803 |
| **B** | `b1_cos_mean` — mean `cos(q_t, q_{t−1})` | 0.500 | 0.820 | 0.854 | 0.500 | 0.780 |
| *oracle* | `o_c1D` — `‖r¹ − r²‖`, unprojected | 0.781 | 0.750 | 0.762 | 0.708 | 0.674 |
| **C** | `c1_q12_rel` — `‖q¹ − q²‖ / ‖q¹‖` | 0.701 | 0.703 | 0.713 | 0.719 | 0.633 |
| **C** | `c2_cos13` — `cos(q¹, q³)` | 0.664 | 0.643 | 0.682 | 0.606 | 0.612 |
| **B** | `b2_spread` — allocentric spread, **realized** *(see point 2 — this row is a mixture)* | 0.500 | 0.680 | 0.587 | 0.500 | 0.584 |
| **A** | `a3_resid` — **self-motion residual, realized** | 0.500 | 0.570 | 0.568 | 0.500 | 0.572 |
| **A'** | `x_a3_cmd` — the same from the commanded action | 0.500 | 0.570 | 0.564 | 0.500 | 0.577 |
| **D** | `d2_clip` — `‖a − d‖`, the wall diagnostic | 0.500 | 0.531 | 0.531 | 0.500 | 0.533 |

Three things fall out of that table.

**1. The discrimination is `‖q‖`, and extra steps help by sampling more of it —
not by generating motion evidence.** `‖q‖` alone scores 0.887 at one step and
still 0.881 at sixty-four; the *running maximum* over the episode climbs to
0.963 and the *running standard deviation* from 0.594 to 0.949. Those are order
statistics over the cells visited, and the mechanism is §5.8's: with the goal
stored, `‖q‖` scales with distance to it, so over an episode it takes a wide
range of values with a large maximum, while with only foreign patterns it is a
small number with little structure whatever you do. §7.8 tests this directly —
`still`, which visits essentially one cell, saturates at Q_ep 0.945 while
`billiard`, which sweeps the arena, reaches 0.997. `‖q‖` is the cue that is
*always* available; point 2 finds a sharper one that is not.

**2. Jack's sharpened cue `a3` is worth about seven points of AUC over chance on
a passively-moving agent, and that is all — but `b2` is a different story, and
the pooled number above hides it.** `a3`, how much `‖q‖` *should* have shrunk if
the target were fixed, reaches **0.568** for Q_ep and **0.572** for Q_trust at
ten distractors, and the commanded-action version is the same to three decimals.
§7.8 shows why: on a billiard trajectory the projection of the step onto `q̂` is
small and sign-varying, so the test has no power.

`b2`, the allocentric spread, reads 0.680 at t = 8 and 0.587 at t = 64 in the
table above, which looks like the same verdict. **It is not — that pooled number
is an average over two populations that behave completely differently, and the
wall conditioning in §7.10 separates them.** Split on whether the episode has
met an arena clip, at ten distractors and t = 8, `n` = 847 clean and 686
clipped:

| statistic | no clip yet | has clipped |
|---|---|---|
| `b2_spread` — allocentric spread | **1.000** | 0.611 |
| `b3_drift_mean` — drift of the same estimate | **1.000** | 0.674 |
| `b4_dev` — current deviation from the running mean | **1.000** | 0.619 |
| `a6_q_max` — running max of `‖q‖` | 0.943 | 0.894 |
| `a1_qnorm` — `‖q‖` | 0.892 | 0.871 |

**On a billiard episode that has not yet touched a wall, the allocentric spread
is a perfect regime detector by eight steps.** That is the cue working exactly
as §7.2 argued it should: with the goal stored, `ĝ = Σd + q` is a *constant*, so
its variance is zero; with a foreign pattern it wanders. Group A is the more
*robust* cue — 0.87 to 0.94 either way — but group B is the sharper one when the
arena lets it work. The pooled decay from 0.764 at t = 4 to 0.587 at t = 64 is
not the statistic degrading; it is the clean population disappearing, since by
t = 64 every billiard episode has met a wall (`n` = 0 in the no-clip column).
It is the same failure mode as §5.1's eight-env mean and §5.2.1's two-draw
fluke, on a new axis: a mean over a mixture told the opposite of the truth.

**3. Group D — Jack's sensory-consistency cue — is perfect, and is not
available.** Both group-D fingerprints separate the regimes at AUC **1.000** at
every distractor level and every step count including t = 1. That deserves its
controls stated plainly:

| control | value |
|---|---|
| decoded-observation magnitude on real observations | 1.000 (a saturated ±1 barcode) |
| the same on held-out in-env cells | 0.31 |
| the same on distractor patterns | **30.6** |
| chart residual on in-env cells | 0.43 |
| chart residual on distractor patterns | **0.90** |

The decoder does not *shrink* on a foreign pattern, it **extrapolates**, by a
factor of a hundred, because the pattern lies outside the span it was fitted on.
So `d1` does not measure "does the recall predict the observation I am seeing";
it measures **"could this env's chart have produced the recalled pattern at
all"**, and because the env's 400 patterns occupy a low-dimensional subspace of
D = 1024 while a scaffold cell from elsewhere does not, that question has a
yes/no answer. §7.1 predicted group D would be "a fingerprint check rather than
a graded signal", and it is exactly that — but the fingerprint is perfect, not
weak.

**This is the one genuinely missing input channel P3 found, and its cost should
be stated with it.** The decoder is fitted *inside the env*: the wall code is a
fresh random draw per arena (`world/env.py:329`), so nothing learned in one env
transfers, and the agent is not handed this. It is not fantasy either — §6.3
measured a within-env nearest-neighbour table saturating the displacement
ceiling, 0.943 against 0.945 — but acquiring it is a representation-learning
problem, not a flag that can be switched on. What P3 establishes is the value of
doing so: an exact regime signal from the first step, against 0.87 for the best
thing the policy currently sees.

#### The ablation, including the two variants §7.3 item 3 insisted on

GBT, held-out-env CV, `billiard`. §7.2 predicted A and C would be largely
redundant and that a leave-one-out alone would report both as worthless. It was
right, and the leave-one-in column is what shows it:

Q_trust, ten distractors, t = 8 (`n` = 1533):

| feature set | AUC | Δ vs A∪B∪C |
|---|---|---|
| A alone | 0.854 | −0.005 |
| B alone | 0.837 | −0.021 |
| C alone | 0.753 | −0.106 |
| D alone | 0.878 | +0.020 |
| **A∪B∪C (policy)** | **0.858** | 0.000 |
| A∪B∪C∪D | 0.883 | +0.025 |
| A∪B (drop C) | **0.869** | **+0.011** |
| A∪C (drop B) | 0.845 | −0.013 |
| B∪C (drop A) | 0.843 | −0.016 |
| B alone (drop A∪C) | 0.837 | −0.021 |

Every leave-one-out is within 0.02 of the full set, which taken alone would say
no cue matters. Leave-one-in says otherwise: **A alone recovers 99% of the full
set, B alone 98%, and A ∪ C jointly contribute only 0.021 over B alone.** The
three groups are three views of one quantity. Q_ep at three distractors is the
same story more starkly — A alone 0.987, B alone 0.982, full set 0.993 at t = 8,
and at t = 1 group B is *exactly* 0.500 because every one of its statistics
needs a previous step.

**Recall depth does not earn its input channel (§7.3 item 3b).** Group C is
precisely the depth-2 and depth-3 derived statistics, so `A∪B (drop C)` is the
depth-{1}-only classifier. It scores **0.869 against 0.858** for depth-{1,2,3}
at ten distractors and t = 8, **0.888 against 0.884** at t = 64, and 0.763
against 0.760 at t = 1. Depth {1} is *never worse* and is usually slightly
better — the extra depths add variance without adding information, which is
what §5.3–5.4 predicted once the dynamics was known to be power iteration away
from the answer. The honest qualification: this ablates *my four summaries* of
the depth channel, and the policy receives `q²` and `q³` as raw 2-D vectors, so
it could in principle use them some other way. But there is no evidence here
that it should, and `--input_hopfield_multistep 1` is now the defensible
default.

### 7.8 What probing behaviour buys, and the surprise

Seven scripted probes plus a sixteen-point parametric sweep over
`a ∝ α·q̂ + β·q̂⊥ + γ·ĥ` (`results/nav_p2/io_prober.npz`), all over identical
memories and identical per-cell tables, so the comparison is of behaviour and of
nothing else. GBT on the policy-visible set, ten distractors.

**Q_trust** — the question probing actually affects:

| probe | (α, β, γ) | t=1 | t=2 | t=4 | t=8 | t=16 | t=64 | steps→0.95 | path / net at t=64 |
|---|---|---|---|---|---|---|---|---|---|
| `still` (oscillate in place) | — | 0.761 | 0.829 | 0.835 | 0.836 | 0.834 | 0.829 | never | 63.0 / 1.0 |
| `random` | — | 0.760 | 0.815 | 0.846 | 0.851 | 0.848 | 0.852 | never | 60.9 / 5.9 |
| `billiard` | — | 0.750 | 0.812 | 0.844 | 0.864 | 0.878 | 0.889 | never | 61.5 / 11.7 |
| `straight` | (0,0,1) | 0.751 | 0.788 | 0.838 | 0.853 | 0.896 | 0.894 | never | 15.5 / 13.9 |
| `perp_q` | (0,1,0) | 0.734 | 0.795 | 0.856 | 0.871 | 0.886 | 0.884 | never | 28.8 / 11.7 |
| `anti_q` | (−1,0,0) | 0.761 | 0.834 | 0.878 | 0.863 | 0.860 | 0.878 | never | 19.0 / 10.2 |
| **`along_q`** | **(1,0,0)** | 0.729 | 0.844 | 0.917 | **0.945** | 0.973 | **0.986** | **9.1** | 38.0 / 10.3 |
| `along_q` + persist | (1,0,1) | 0.741 | 0.830 | 0.891 | 0.939 | 0.984 | **0.993** | 10.0 | 38.2 / 11.0 |
| `along_q` + ½ perp | (1,0.5,0) | 0.763 | 0.848 | 0.900 | 0.944 | 0.973 | 0.981 | **9.7** | 39.0 / 10.9 |
| 2 `along_q` + perp | (2,1,0) | 0.768 | 0.854 | 0.908 | 0.945 | 0.970 | 0.982 | **9.4** | 38.3 / 10.7 |
| ½ `along_q` + perp | (0.5,1,0) | 0.760 | 0.843 | 0.882 | 0.915 | 0.941 | 0.938 | 23.2 | 37.2 / 11.1 |
| `−along_q` + perp | (−1,1,0) | 0.750 | 0.827 | 0.864 | 0.864 | 0.850 | 0.836 | never | 20.4 / 12.1 |

**The answer to "what probing behaviour buys information fastest" is: walk along
`q`.** It is the only family that reaches AUC 0.95 at all — at t ≈ 9, against
*never* for every probe that does not follow `q` — and it is also the best per
unit of distance travelled: **0.069** of AUC-above-chance per cell of path at
t = 8, against 0.054 for billiard, 0.052 for random and 0.048 for `still`. It
wins on both axes at once, which is not the usual shape of such a trade, and it
travels *less* far than billiard because following a converging signal curves
the path.

The sweep also says what the weights need to be. Everything with α ≥ 1 lands
between 9 and 14 steps to 0.95 whatever β and γ are; α = 0.5 needs 23 steps;
α ≤ 0 never gets there. **The `along_q` component is the whole effect and the
other two are cosmetic** — which is a useful negative for anyone designing a
probing bonus.

**Two of the spec's expectations are refuted here.**

*`perp_q` was predicted to be the best probe for group B on a parallax argument,
and it buys nothing* — 0.884 at t = 64 against billiard's 0.889 and `still`'s
0.829. Maximum parallax is the wrong idea because the discriminating quantity is
not a triangulation; it is whether the target you are converging on is really
there.

*`still` was expected to be the information floor, and it is, but not by much*:
0.829 against billiard's 0.889 after sixty-four steps. An agent that never moves
still gets to 0.83 on `‖q‖` alone. The gap between standing still and the best
passive exploration is six points; the gap between the best passive exploration
and walking along `q` is **ten points**, and it is the only one that closes the
problem.

**And this is where Jack's first cue is vindicated in its proper form.** §7.7
found `a3` — "`‖q‖` should shrink by exactly your projection onto `q̂`" — almost
worthless on a billiard trajectory. The reason is now plain: on a billiard
trajectory the projection of the step onto `q̂` is small and sign-varying, so the
test has no power. Walk *along* `q` and the same test is the whole story. The
measured consequence is stark: following `q` for 64 steps in the goal-absent
regime drives `P(direction is good)` from its chance value of 0.33 down to
**0.06**, while in the goal-present regime 37% of trials have reached the goal
outright and been excluded. **The cue was right; it needed the behaviour that
makes it testable.**

**For Q_ep, probing is nearly irrelevant**, and that is worth stating because it
is the regime question P6 cares about. Every probe crosses AUC 0.95 within 2–3
steps at ten distractors: `billiard` at 2.7, `along_q` at 2.3, `anti_q` at 2.1,
`random` at 3.3. The one exception is `still`, which saturates at **0.945** and
never crosses — the direct confirmation of §7.7's claim that the extra
information in later steps is *more cells sampled*, not motion evidence. An
agent that moves at all, in any direction, knows within three steps whether
anything from this env is in its memory.

**One artefact to flag, because it looks like a result.** `anti_q`'s goal-absent
trust rate *rises* to 0.695 by t = 64, which reads as "running away from a
phantom improves your direction". It does not: walking along `−q` until you hit
a boundary leaves you at a wall with `q` pointing inward, and the goal is in the
interior, so `cos(q, goal − x)` is mechanically high. It is arena geometry, not
memory — and it is why `anti_q`'s AUC does not rise with it.

### 7.9 The trained agents' own trajectories — the agent is collecting it

§7.3 item 5. Three trained policies scored on a fresh 32-env world drawn with a seed
disjoint from the probe world, so no env is shared between fitting and scoring;
8 draws × 2 starts × 4 distractor levels × 2 regimes, `goals_active = False` in
both regimes so that episode *length* cannot become a function of the label.

Each cell carries three numbers and all three are needed. **frozen** is the
probe-fitted classifier applied as-is. **refit** is the same feature set
refitted on the agent's own rows with held-out-env CV — whether the information
is *there*. **probe** is `billiard` at the same (level, `t`).

`navigate_navp2_p4_x_s42_21102411/navigate_u1250.pt`, the P4 exploit
specialist (still training; see §7.11), at **ten distractors**:

| target | arm | t=1 | t=2 | t=4 | t=8 | t=16 | t=32 | t=64 |
|---|---|---|---|---|---|---|---|---|
| Q_ep | frozen | 0.857 | 0.802 | 0.582 | 0.574 | 0.494 | 0.391 | **0.208** |
| Q_ep | **refit** | 0.841 | 0.934 | 0.969 | 0.985 | 0.994 | 0.995 | **0.998** |
| Q_ep | probe (billiard) | 0.872 | 0.938 | 0.979 | 0.985 | 0.989 | 0.990 | 0.996 |
| Q_trust | frozen | 0.743 | 0.693 | 0.496 | 0.575 | 0.358 | 0.259 | **0.114** |
| Q_trust | **refit** | 0.728 | 0.785 | 0.850 | 0.917 | 0.969 | 0.982 | **0.983** |
| Q_trust | probe (billiard) | 0.760 | 0.799 | 0.855 | 0.858 | 0.874 | 0.886 | 0.884 |

Three results, and the first would have been reported wrongly without the
other two.

**1. The frozen classifier does not merely lose power on the agent's own
trajectories — it inverts.** Q_ep goes 0.857 → 0.208 and Q_trust 0.743 → 0.114,
which are confident *wrong* answers, not shrugs. The mechanism is that the
exploit policy succeeds: §5.2 established that `‖q‖` scales with distance to the
goal — median 0.041 at one cell against 0.393 beyond twelve — so `‖q‖` next to
the goal is **smaller than the goal-absent value of 0.086**. An agent that walks
to the goal spends its episode in the one region of the arena where the current
value of the regime cue is not merely weak but reversed, and a classifier fitted
on arena-sweeping trajectories reads that as "no goal in memory".

**2. The information is nevertheless all there — the agent is collecting it.**
Refitted on the agent's own rows, Q_ep reaches **0.998** and Q_trust **0.983**,
both at or above the billiard reference (0.996 and 0.884). Q_trust is *better*
than billiard by a full ten points, and §7.8 says why: the exploit policy
follows `q`, so it is executing the `along_q` probe as a side effect of
exploiting, and `along_q` is the behaviour that reaches 0.986.

**So Jack's "is the agent collecting the information it needs" question has a
clean answer: yes, and by accident.** The exploit policy's own behaviour is the
optimal probing behaviour. What it is not doing is *using* the result — which is
mode B, and §7.5 already said that is a learning problem.

**The consequence for P6 is concrete.** A gate or auxiliary head trained on
exploration rollouts will invert on a successful exploiter. Any regime signal
must be built from statistics that survive the approach to the goal — the
running maximum `a6_q_max` and the running standard deviation `a5_q_std` do,
because they remember the large `‖q‖` from early in the episode; the current
`a1_qnorm` does not. That is a specific, testable design constraint and it is
the main thing §7 hands forward.

**3. The explore specialist is the control that makes the reading airtight.**
`navigate_navp2_p5_e_s42_21102414/navigate_u850.pt` at ten distractors:

| target | arm | t=1 | t=4 | t=8 | t=16 | t=64 |
|---|---|---|---|---|---|---|
| Q_ep | frozen | 0.857 | 0.785 | 0.867 | 0.863 | **0.904** |
| Q_ep | refit | 0.841 | 0.953 | 0.976 | 0.987 | 0.997 |
| Q_trust | frozen | 0.743 | 0.498 | 0.568 | 0.629 | 0.585 |
| Q_trust | **refit** | 0.728 | 0.796 | 0.843 | 0.853 | **0.838** |

Its mean distance to the goal stays at 10.1 – 11.6 for the whole episode
(against the exploiter's 7.6 – 9.1) and **not one of its 1024 rows is masked**,
because it never arrives. Consequently the frozen classifier does *not* invert
on it: 0.904 at t = 64 for Q_ep, against the exploiter's 0.208. That confirms
the inversion is caused by approaching the goal and nothing else.

And on the target that probing affects, the explore specialist collects
**less** information than the exploiter and less than a billiard: Q_trust refit
**0.838** against the exploiter's 0.983 and billiard's 0.884. It moves, but not
along `q`. **The behaviour that gathers the most evidence about whether the
memory is real is the behaviour that acts on the memory** — which is a
convenient fact for P6 and an inconvenient one for any design that separates
"verify" from "exploit" into different phases.

*(The second P4 seed, `..._s12_s42_21102413/navigate_u1300.pt`, sits between:
frozen Q_ep 0.857 → 0.488 rather than → 0.208, refit 0.995, Q_trust refit
0.975 — a policy that exploits less reliably and therefore spends less of its
episode next to the goal.)*

### 7.10 The channel ablation at walls (§7.3 item 6) — H-wall's last half

The surviving half of H-wall was a comparison, not a hypothesis: compute `a3`
and `b2` from the commanded action and from the realized displacement, and see
whether the commanded version degrades where the arena clip bites.

**The channel comparison is a null, and a clean one.** On billiard trajectories
at ten distractors and t = 8, single-feature AUC, per-env median:

| condition | `a3` realized | `a3` commanded | `b2` realized | `b2` commanded | n |
|---|---|---|---|---|---|
| no clip yet | 0.630 | 0.630 | **1.000** | **1.000** | 847 |
| has clipped | 0.647 | 0.623 | 0.611 | 0.592 | 686 |
| `wall_dist ≤ 1` | 0.629 | 0.636 | 0.673 | 0.745 | 465 |
| `wall_dist ≥ 4` | 0.667 | 0.667 | 0.800 | 0.800 | 564 |

On unclipped episodes the two channels are *identical by construction* — with no
clip the commanded action **is** the realized displacement — and both are
perfect. On clipped episodes the realized version is better by 0.02 for both
statistics, which is the sign H-wall predicted and about a tenth of the size it
needed to matter. In the near-wall split the *commanded* version is mildly
**better** (0.745 against 0.673), the opposite sign, on ~10 rows per env: read
that as noise, not as a reversal.

**But the wall does destroy the cue, by a mechanism H-wall did not name.** `b2`
goes from **1.000 to 0.611** the moment an episode has met a clip, and the
commanded/realized distinction explains none of that gap. What explains it is
that `b2` needs *net displacement*: `ĝ = Σd + q` is constant when the goal is
stored and drifts in proportion to how far you have moved when it is not, so an
episode pinned against a boundary loses the drift that made the two regimes
distinguishable. **The arena clip attacks the cue by suppressing motion, not by
corrupting the channel.**

That reframes rather than revives H-wall. The original claim was that
`prev_action` carried the wrong quantity and that near a wall the agent's best
regime cue was therefore corrupted "in the direction of declaring a real goal a
phantom". The channel half is refuted: §4 B4's fix is correct, costs nothing,
and buys 0.02. The consequence half survives in a different form: a wall really
does cost the sharpest regime cue almost all of its power, and no input channel
can fix that because the missing ingredient is displacement. What can fix it is
*behaviour* — §7.8's `along_q` generates net displacement by construction, and
the exploit policy already does it (§7.9).

The wall diagnostic itself, `d2_clip = ‖a − d‖`, sits at **0.531 flat in `t`** —
an excellent detector of walls that carries essentially nothing about whether
the goal is in memory, which is exactly what it should do.

**H-wall's readout half stays dead** (§5.2: `dir_cos` flat against
distance-to-wall), and phase 1's wall failures are still not a readout problem.

### 7.11 What P3 did not resolve

- **The learned prober is parametric, not trained.** §7.3 item 4's fullest
  option was a learned prober; what ran is a sixteen-point search over
  `α·q̂ + β·q̂⊥ + γ·ĥ`, a policy search over a three-parameter family rather
  than an RL agent. The family contains all four named probes as corners and has
  a clear interior optimum (α dominant), so a trained prober would have to beat
  `along_q` by finding something outside a linear combination of the three
  available directions. That is possible and untested.
- **`Q_trust ǀ goal-present` is not estimable at this class balance.** With
  98.3% positives at ten distractors on billiard trajectories the negative class
  is about 13 rows, and its label-permutation control comes back at 0.535 –
  0.728 instead of 0.500 — the control failing, not the estimate merely being
  noisy. A real number needs a stratified design or a much larger draw, and it
  is the number a mode-B claim would most like to cite directly.
- **Group D's acquisition cost is unmeasured.** The in-env pattern decoder is
  worth AUC 1.000 against 0.87 — the largest single effect in §7 — and nothing
  here says how much in-env experience is needed to fit one well enough to keep
  that. §6.3's within-env nearest-neighbour table used all 400 cells.
- **The agent-trajectory result is on policies that were still training.** The
  P4 checkpoints are `u1250` and `u1300` of runs that had not converged and P5
  is at `u850`. The inversion is mechanistic and should survive, but the
  magnitudes will move.
- **The probe study fixes `‖a‖ = 1`.** Every probe walks one cell per step, so
  the comparison is fair between probes but says nothing about step size, which
  §2.1's bounds make a live knob. Since §7.10 shows the group-B cue is a
  function of *net displacement*, step size is the obvious next axis and is
  untested.
- **`b2`'s perfect score is on a shrinking population.** By t = 64 no billiard
  episode is still unclipped, so "the cue is perfect at eight steps on clean
  episodes" is a statement about the 55% of episodes that have not yet met a
  wall in a 20x20 arena at one cell per step. Whether a policy can arrange to
  stay in that population, and what it costs in coverage, is a P6 question.
- **Everything is at `steps = 1`**, which §5.4 established is the only setting
  that retrieves. The depth-{1,2,3} channel is answered as a contribution to a
  classifier (§7.7), not as a claim about what other recall depths would do.

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
3. **P2 has reported (§6) and it does change the target — for a different
   reason than expected.** Relative displacement does *not* decode from the
   cone, but `input_prev_displacement` hands the agent exact self-motion, so
   the information a lawnmower needs is present from step 1 and the 0.50 line
   is open on information grounds. **P5 is therefore a test of whether the
   policy can use a position it is already being given**, and a plateau at
   0.378 is a capacity or shaping result, not a sensory one. Expect 0.36–0.38;
   read anything above it as evidence the recurrence is integrating.

So P5 runs as a **calibration and reference run, not a search**: one arm at the
σ P4 selects, bounds on, `--input_prev_action`, long enough to plateau. Expect
0.36–0.38. Deliverable: a clean, unambiguous explore ceiling under phase-2
settings, with `strategy_efficiency` beside it so "the agent moves too slowly"
and "the agent moves badly" stay separable.

### 8.1 RESULT — the exploit ceiling, and what moved it

Five arms, all run to the six-hour wall (~u1500) except `p4_tp10`, which
diverged. Selection is the lowest `mean_steps` at ten distractors **subject to
success ≥ 0.85**, because the two trade off and the time-penalty bracket was
designed to risk exactly that trade (`analysis/nav_p2/p4_summary.py`):

| arm | σ | time pen. | u | succ@0 | steps@0 | succ@10 | steps@10 | speed@10 |
|---|---|---|---|---|---|---|---|---|
| `p4_x` | 0.50 | 0.05 | 1500 | 1.000 | **6.25** | 0.885 | 9.29 | 1.50 |
| `p4_s12` | 0.30 | 0.05 | 1150 | 1.000 | 6.84 | 0.917 | 9.27 | 1.45 |
| `p4_s18` | 0.165 | 0.05 | 900 | 1.000 | 7.70 | **0.927** | 9.60 | 1.31 |
| `p4_tp10` | 0.50 | 0.10 | 450 | 1.000 | 8.49 | 0.948 | 9.31 | 1.40 |
| `p4_tp15` | 0.50 | 0.15 | 800 | 1.000 | 6.66 | 0.875 | **8.01** | **1.52** |

**Against phase 1 this is a large gain.** The matched exploit-only control under
unbounded steps reached 10.16 at d=0 and 12.70 at d=10 with success 0.833.
Phase 2 reaches **6.25 / 8.01** — roughly **38% fewer steps at both distractor
levels, with higher success**. Bounded step size plus the two `prev_action`
channels account for it; no shaping knob was touched.

**σ appears to trade speed for reliability** — success at ten distractors rises
monotonically as σ falls (0.885 → 0.917 → 0.927) while speed falls with it
(1.50 → 1.45 → 1.31) and `mean_steps` gets worse. **The description holds; the
mechanism given here was wrong, and §8.2 replaces it.** This paragraph
originally explained the trend as σ buying accuracy at the cost of the
exploration needed to find the speed limit. Measuring the commanded magnitude
showed that is not what separates the arms.

**The time-penalty hypothesis is confirmed, and it buys steps by giving up
success.** `p4_tp15` posts the best `mean_steps` at ten distractors (8.01) and
the highest speed (1.52), which is the predicted effect: at `time_penalty` 0.05
against `goal_reward` 2.0 the agent is indifferent between arriving and taking
forty extra steps, and tightening that to thirteen makes it hurry. But its
success is the lowest of the five (0.875). **That is the give-up failure the
arm was launched to watch for, and it happened** — a `mean_steps` win taken over
the near starts that were kept. Recorded as a real trade, not a win.

**`p4_tp10` diverged** at ~u1200 with a NaN policy mean
(`Expected parameter loc ... to satisfy Real()`). Its last valid eval had the
best success of any arm (0.948) at u450, so the intermediate penalty may be the
sweet spot, but a diverged run cannot support that and it is not claimed. A
rerun at lower LR would settle it.

**Where the remaining gap is.** The best arms run at speed ~1.5 against a
permitted 2.0. At their own speed they are close to the ideal
`(10.85 − 1) / 1.5 ≈ 6.6`; at the cap the ideal would be 4.9. So the residual is
still speed, and §9.1 explains why the policy cannot find it.

### 8.2 CORRECTION — the σ bracket was a clamp-depth bracket

Probing all four arms for the **commanded** magnitude `‖μ‖`, against `σ =
exp(init_log_std)`, gives the effective angular noise `σ/‖μ‖` — the quantity
that actually governs directional exploration for a Gaussian policy on a
displacement:

| arm | σ | time pen. | ‖μ‖ @ d=0 | ang @ d=0 | ‖μ‖ @ d=10 | ang @ d=10 |
|---|---|---|---|---|---|---|
| `p4_x` | 0.497 | 0.05 | 4.91 | 5.8° | 2.19 | **13.0°** |
| `p4_s12` | 0.301 | 0.05 | 2.83 | 6.1° | 1.31 | **13.1°** |
| `p4_s18` | 0.165 | 0.05 | 2.19 | 4.3° | 0.96 | **9.9°** |
| `p4_tp15` | 0.497 | **0.15** | 3.71 | 7.7° | 1.48 | **19.2°** |

**Every arm saturates the clamp at zero distractors** — `‖μ‖` from 2.19 to 4.91
against a cap of 2.0. This is not an outlier, it is what this policy class does
here, and §9.1's clamp pathology is therefore not explore-specific. An earlier
reading of these runs inferred from `mean_speed ≈ 1.5` that exploit sat *inside*
the bound; that was the realized magnitude, and assuming the commanded one
matched it was the error.

**The policy compensates for σ by scaling `‖μ‖`, partially.** At fixed time
penalty, nominal σ spans **3.0×** while the effective angular noise spans only
**1.33×**. So most of what σ changed was the commanded magnitude, not the
exploration — which means the bracket did not vary what it was named after.

*(A stronger claim was made first, from two arms, that compensation was exact —
`p4_x` and `p4_s12` agree to 13.0° against 13.1°. Four arms show it is not. The
place it breaks is informative: `p4_s18` would need `‖μ‖ ≈ 0.73` to match and
sits at 0.96, against a `min_action_norm` of 0.5. The policy compensates until
the floor stops it, which is why the lowest-σ arm is the one that genuinely got
less angular noise.)*

**The time penalty moves angular noise more than σ does.** `p4_tp15` has
identical σ to `p4_x` and lands at 19.2° against 13.0°. So neither bracket was
a clean manipulation: the σ arms mostly varied clamp depth, and the
time-penalty arms substantially varied exploration.

**So the corrected mechanism for §8.1's trend is:** σ sets `‖μ‖`, `‖μ‖` sets how
deep into the clamp the policy sits, and clamp depth sets speed. A policy
commanding 4.91 against a 2.0 cap runs at the limit on nearly every step and
cannot modulate; one commanding 2.19 still has room to vary. That is what
separates the arms, not exploration.

**One further observation, consistent across all four arms:** angular noise
roughly doubles from zero to ten distractors (5.8° → 13.0°, 6.1° → 13.1°,
4.3° → 9.9°, 7.7° → 19.2°). The policy widens its directional exploration when
the readout is less trustworthy — sensible behaviour, and what P3's Q_trust
framing predicts. But with σ fixed globally the *only* channel available for it
is `‖μ‖`, so the policy buys its state-dependent exploration by paying in speed.
**That is the argument for a state-dependent σ head**, and it is independent of
the clamp question.

**Scope.** One checkpoint per arm in one probe world. The `‖μ‖` values are large
and consistent; the arm *rankings* built on them are not — the probe world makes
`p4_s12` best on `mean_steps` at ten distractors where the trainer's eval
favoured `p4_tp15`. Rankings need more worlds before they are quoted, per
findings 18 and 19.

### 9.1 RESULT — the explore ceiling, and a clamp pathology

`p5_e` converged and held: coverage **0.379–0.400** from u600 to u1000, after
oscillating between 0.21 and 0.39 before that. Behaviour probe at u850
(`analysis/nav_tri/behavior_probe.py`, 8 envs × 32 trials):

| | |
|---|---|
| `mean_coverage` | 0.390 |
| `cells_per_step` | 0.781 |
| `realized_mag_mean` | **1.98** |
| billiard reference at 1.98 | 0.702 |
| **`strategy_efficiency`** | **1.113** |
| `straightness` | 0.945 |
| `edge_frac` | 0.121 (uniform is 0.19) |
| `chase_q` | ≈ 0.000 |

**The explore policy beats a perfect billiard by 11% at its own speed.** That
matters because billiard is the *reactive* ceiling — it needs only "am I about
to hit a wall" — so exceeding it means the policy is using something more. P2
identified what: `input_prev_displacement` hands it an exact position
(integration error 2.3e-14). On the billiard → lawnmower scale (0.351 → 0.5025)
it is about a quarter of the way.

`chase_q ≈ 0` confirms the explore-side distractor problem stays solved, as in
phase 1.

#### The clamp traps the policy at the speed limit

The same probe shows the policy **commands** `‖a‖ = 8.18` while **realizing**
1.98 — it saturates `max_action_norm` on 100% of steps. Beyond the clamp the
gradient with respect to magnitude is exactly zero, so nothing pulls it back.

**And 2.0 is not the coverage optimum.** Billiard peaks at `‖a‖ ≈ 1.25` (0.378)
and falls to 0.351 at 2.0 (§2.1). At its measured 1.11 efficiency a policy
choosing 1.25 would reach roughly **0.42** rather than 0.390. So the hard clamp
is *costing* explore coverage by parking the policy in a zero-gradient region at
the boundary.

The mechanism is clean and worth stating because it generalizes: novelty reward
pushes `‖μ‖` up early, it overshoots the clamp, the magnitude gradient vanishes,
and the novelty signal that would have located the interior optimum never
reaches the parameter again.

**`max_action_norm` is not changed** — the [0.5, 2] band is fixed by
instruction. Recorded as a design note for a later phase: a **soft** bound
(a tanh-squash on the magnitude, with the log-prob correction) preserves
gradient everywhere and would let novelty find the interior optimum, which a
hard clamp structurally cannot.

#### An instrumentation fault this exposed

The first probe reported `strategy_efficiency` **3.97**, which would have meant
the policy covers four times what a perfect billiard manages. It does not.
`step_mag_mean` is the *commanded* magnitude, so the billiard reference was
being taken at 8.18 — where billiard scores terribly because eight-cell strides
skip most of the arena — and dividing by a deliberately bad reference
manufactured a spectacular number. The metric predates phase 2 and silently
assumed commanded equals realized, which was true until `max_action_norm`
existed. It now references the realized displacement and reports both, so the
clamp's bite is visible rather than inferred.

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
input differently?" — that was deferred until P2 reported. **It has (§6.7), and
the question is now live with a number.** Displacement does not decode from the
±1 cone, but it decodes at R² 0.869 / 11.1° from the *same 60 rays returning
range instead of a code* — against a ceiling of 0.945 / 8.6°. The geometry is in
the cone; the hash removes it. `WALL_RESOLUTION` is the same lever from the
other end and is a one-line change (4 → 1 is worth 4× the cross-env signal).
Neither is free: the ±1 code is injective where the range profile aliases, and
`wall_resolution=4` exists so two positions inside one cell read differently.
And neither helps the free-heading case (§6.5), which is the one that binds.
