# Phase 2 — measure the signal before training on it

Phase 1 (`docs/EXPERIMENTS_NAV_TRI.md`) produced one model that scores on all
three metrics and, at the end, a diagnosis: the failures split into two modes
with opposite causes, and only one of them is a policy problem. Phase 2 takes
that seriously and **measures the signal first**. Three analyses establish what
is knowable; three training runs then establish what is reachable.

Read §0 to resume. Read §4 before launching anything — there is a train/eval
mismatch that must be fixed first.

---

## 0.0 What exploration is FOR — read before optimizing it

Settled with Jack 2026-09-01, after asking whether the target is "act like a
mouse" or "impose a mouse's constraints and see what emerges".

**Neither, quite. The target is EFFECTIVE STABLE SEARCH.** The central question
of this project is not whether animals explore this way, so mouse-likeness is
not the objective — and it could not be one anyway, since it cannot be written
as a reward without a behavioural dataset to imitate.

**Exploration is instrumental.** It exists so the memory system has something to
store and the other two metrics have something to work with.

**So why learn it at all?** A scripted billiard is ten lines, costs zero
updates, has no wall-pin basin and no κ runaway — and `p5_e` beat a perfect
billiard by only 11% while `p10_e_pol` plateaued *at* it. The answer is the
**one-model constraint**: the tri-metric claim is a single policy that explores,
navigates and discriminates while *inferring* which regime it is in (which is
why `input_goal_in_memory` is cheating). A scripted explorer cannot be that
policy. Exploration is learned because of the one-model claim, not because we
care how mice do it.

**Consequences, which are the point of writing this down:**

1. **The bar is instrumental, not maximal.** "Good enough and stable enough not
   to poison interleaved training" — concretely `strategy_efficiency >= 1`
   (matches a scripted billiard at its own speed) with low eval variance.
   `p20_e` is at **1.038**. That bar is met. Further explore optimization has
   low marginal value; the leverage is in the interleaved regime, where the
   actual claim lives.
2. **The speed cap is a declared realism constraint with a known price.**
   "Massive steps are unrealistic" is biological-realism reasoning applied to an
   instrumental subsystem, and §19.2 measures it at ~30% of expected discovery
   time. Keeping it is fine. Believing it is free is not — §18.2 did, and is
   retracted.
3. **Do not chase mouse-likeness by accident.** If it ever becomes a real
   target it needs a behavioural dataset to score against, and exploration stops
   being instrumental and becomes the result. That is a different project.

---

## 0. Where I am

| | |
|---|---|
| **Metric** | **`swept_coverage` is the headline explore metric from 2026-09-01 (§19).** Union of `goal_radius` discs along the path = P(goal findable). `mean_coverage` counts snapped cells, which hides the speed axis: it says speed barely matters, swept area says speed dominates. §2.1/§18.2's "the speed cap is free" is retracted. `union_swept_coverage` is the spread diagnostic. |
| **Status** | **P10 polar landed (§9.4–9.8).** Two of four arms finished; the exploit-frozen model is the phase-2 best. |
| **Branch / worktree** | `nav-tri-metric` at `.claude/worktrees/nav-tri-metric` |
| **Predecessor** | `docs/EXPERIMENTS_NAV_TRI.md` — read its §0 findings 1–22 |
| **Open decisions** | §11 — four forks put to Jack; spec assumes the recommended default in each |
| **Running** | **P20 (§18) — DONE.** Both arms COMPLETED 700/700. **`p20_e` 21695407 is the delivered explore model**: `mean_coverage` **0.390** at realized speed **0.964**, `strategy_efficiency` **1.038**, `chase_q` **0.000** — matches `p5_e`'s coverage at half the speed, on a **fresh `held_out` draw**. The κ cap that unlocked exploit COSTS explore 12.1%, but **not** via straightness (§18.4 refutes its own mechanism) — via `edge_frac`, 0.061 vs 0.127. |
| **Running** | **P19 (§17) — DONE.** Both arms COMPLETED 800/800. **`p19_kcap` 21656252 is the delivered model**: Jack's w52 encoder, gain=beta=100, learned speed [0.5, 1.0], plus `LOG_KAPPA_MAX=2.5`. **Accuracy 1.000 from u125; beeline from u150, worst 1.090, final 1.013; 27 consecutive evals ≥0.990.** Curriculum LOSES on both halves (§17.11). |
| **Charts** | **Explore trajectories + failure modes** [f59ee221](https://claude.ai/code/artifact/f59ee221-a39d-4af4-8f18-0fb8a5f824f4) · One published page per run — `p10_pol_v1` [3bc9ad4e](https://claude.ai/code/artifact/3bc9ad4e-0655-43ca-b870-0516f4487bdc) · `p10_pol` [388023ce](https://claude.ai/code/artifact/388023ce-a725-4253-a53b-c9979a77baf2) · `p10_e_pol` [00bd7fd3](https://claude.ai/code/artifact/00bd7fd3-bb60-4e22-a968-c62822c5cdb3) · `p10_e_pol_v1` [8fd3ecf0](https://claude.ai/code/artifact/8fd3ecf0-c429-40d6-bde6-008ca25b5a40) · `p11_cur` [4de8dfa7](https://claude.ai/code/artifact/4de8dfa7-9403-43c8-b4f9-b14669ae603e) · `p11_tp` [4dbbe6e9](https://claude.ai/code/artifact/4dbbe6e9-c8e3-41fb-9b38-36a1443bf420) · `p11_cur_tp` [6c3a0503](https://claude.ai/code/artifact/6c3a0503-dba1-405a-a90c-d33c491ee5b2) · `p12_lo` [6b09232a](https://claude.ai/code/artifact/6b09232a-bcd2-4609-9c1d-97d9757d0f5a) · `p12_lo_curtp` [835846df](https://claude.ai/code/artifact/835846df-d30d-46f7-b979-3fe41fdfff7e) |
| **Finished** | **`p10_pol_v1` 21300389** — 2000/2000, **1.000 success @ 10.95 steps (1.10× optimal)**, the phase-2 best exploit model. **`p10_e_pol` 21300390** — 1500/1500, **cps 0.75** against a billiard ceiling of 0.775. |
| **Done** | §4 blocking fixes, P1 (§5) with figures, the recall-mechanism thread §5.3-5.9, **P2 (§6)**, and **P10 (§9.4–9.8)** |
| **⚠ Read before quoting any §9.6–9.8 number** | Every behavioural number there is on the **`recorded`** split — the run's own `base_val`, never trained on but the set it was scored against at every eval, and the only set the probe could build until 2026-08-27. It is **not** a fresh draw. `--split` now exists on the probe; nothing has been re-run with it. |

**Open items** (priority order):

- [x] **DONE — §23. Re-scored explore with `deterministic=False`.** The κ-cap gap is **3.2%, not 14%**; §18.4's magnitude is retracted. §22's vector-field finding survives sampling. Explore should be scored sampled from here on. Original item: Every explore number in this document is the *noiseless mean policy*; the training reward was earned by *sampled* trajectories, and for a search task the noise is functional. §18.4's 12% κ-cap gap could shrink or invert, since the capped arm's whole difference is spread that a deterministic eval discards. No retraining needed.

- [ ] **Run `p21_pr` — staged, not launched.** §18.7 measured **100% of episodes at u25/u50 wall-pinned**, and §18.8 priced it: the **persistence bonus pays +0.196/step for the pin** against `wall_penalty`'s −0.093, because it scores the *commanded* action and a pinned agent commands a perfect heading while realizing 0.09. `--persistence_realized` (default off) fixes that without taxing the walls; `p21_pr` is `p20_e` with that one bit flipped, `explore:300`, and `p20_e` is its own control. Score it with `explore_traj` on u25/u50/u75, not from the coverage curve.

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

> **RETRACTED for the explore claim — §19.2.** The bracketing argument is measured on **cell** coverage, which counts snapped cell centres and so charges a long stride for ground it actually swept. Under **swept area** at the real detection radius, coverage is *monotone increasing* in speed and the [0.5, 1.0] cap costs ~30% of expected discovery time. The step bounds are still the right call for physical realism; they are not free.

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

### 7.7.1 What group D is really saying — the projection deletes the signal

Jack's question, and it is the right one to ask of that 1.000: *is D saying
that if we passed the raw encoding — the actual Hopfield recall — the policy
would have the signal?*

**Broadly yes, and the code says exactly how much is thrown away.**

#### What the policy receives

`rollout/signal.py::project_to_signal` computes the full recall at
`embed_dim` = 1024, then:

```
W = vectorhash.gram_schmidt_projection(positions, env_offset)   # (B, 2, 1024)
q = vectorhash.project_displacement(embeddings, recalled, W)    # (B, 2) E/N
```

`hopfield_signal_at` sets `signal_dim = 4 if hopfield_mode == "discrete" else
2`. So what reaches the policy is **two numbers** per recall — the recalled
displacement in a local 2-D frame, raw under `input_hopfield_raw` so `‖q‖`
survives — for the main signal plus depths 1/2/3. About **8 of the agent's 74
input dims**. The remaining **1022 dimensions of the recall are projected away
and never reach the policy at all.**

#### D measures precisely the discarded part

`d1_chart` is the recall's residual against this env's chart subspace — the
top-64 right singular vectors of the env's 400 encoded cells
(`io_features.py::chart_basis`), i.e. how much of the recalled vector lies
*outside* what this environment can explain. The 2-D Gram–Schmidt frame is a
local slice of that same chart: `q` keeps the in-chart coordinates and discards
the orthogonal remainder.

So group D's AUC 1.000 against `‖q‖`'s 0.87 is not a claim that the policy has
a perfect cue and ignores it. It is the statement that **the projection to `q`
deletes a signal that the raw recall carries.**

#### Three qualifications, in the order they bite

1. **`d1` is not the raw vector — it is the raw vector measured against an
   env-specific basis.** Handing the policy 1024 dims makes the information
   *present*; extracting the statistic still requires inferring the current
   env's chart subspace online. That is a representation-learning problem, not
   a feature read, and it is the same gap §7.11 records as "group D's
   acquisition cost is unmeasured".

2. **There is a far cheaper version, and it should be tested first.** `W` is
   already computed every step, so the orthogonal residual
   `‖recalled − Wᵀq‖` — or better the ratio `‖q‖ / ‖recalled‖`, *how much of
   this recall does the local chart explain* — is **one extra scalar**, needs
   no env-specific fit, and is the decoder-free shadow of `d1`. If the
   separation survives that compression it is nearly free. If it does not, the
   1024-dim route is the only one and the acquisition cost becomes the real
   question.

3. **Whether it survives compression is measurable offline, with no training.**
   Same machinery as `ideal_observer`: compute `‖q‖ / ‖recalled‖` on
   goal-present and goal-absent draws and read its AUC against `d1_chart`'s
   1.000 and `a1_qnorm`'s 0.887. A CPU job of minutes.

**The honest prior is uncertain**, and against the cheap version: the 2-D frame
is a much smaller subspace than the 64-dim chart, so its residual is dominated
by in-chart directions the frame simply does not span, and the
goal-present/absent contrast may wash out. That is the thing to measure, not to
assume either way.

> **MEASURED — §7.7.2. The prior above is wrong.** `chart_frac` reaches AUC
> **0.974 / 0.988** at ten distractors (P2 gain-5 / w52), which **beats** the
> env-fitted `d1_chart` (0.942 / 0.972) and beats `‖q‖` by **+0.276** on the
> encoder §7 was measured on. The contrast does not wash out.

#### Consequence for re-running P3 on a new encoder

If the chart residual is what carries the signal, then the encoder's
**effective dimensionality** bears on it directly — a fixed top-64 basis
explains less of a code that spreads over more dimensions. The w52 encoder runs
at gain 100 against the gain-5 code P3 was measured on, so it is *more* binary
and *higher*-dimensional. That makes `d1`'s 1.000 **more** likely to move under
an encoder swap, not less, and it is the number in §7 most exposed to one.

> **MEASURED — §7.7.2. Also wrong.** w52 is *better* on all three statistics,
> not worse. `d1_chart` does fall to 0.972 at ten distractors — but it falls to
> 0.942 on the gain-5 code as well, so the drop is a **distractor-count**
> effect, not an encoder one. And `‖q‖` separability, the thing the worry was
> really about, is **0.930 on w52 against 0.698 on gain-5**.

### 7.7.2 RESULT — the compression works, and it beats the fitted basis

§7.7.1 proposed the cheap test and predicted, explicitly, that it would probably
fail: "the 2-D frame is a much smaller subspace than the 64-dim chart, so its
residual is dominated by in-chart directions the frame simply does not span,
and the goal-present/absent contrast may wash out."

**That prediction is wrong.** Measured on both encoders, 200 random cells per
env, 6 envs, `--chart_k 64`, seed 0 (job 21691512, CPU):

| n_dist | encoder | `auc_qmag` | **`auc_chart_frac`** | `auc_d1_chart` |
|---|---|---|---|---|
| 1 | P2 gain-5 | 0.980 | **0.999** | 1.000 |
| 1 | w52 gain-100 | 0.993 | **0.999** | 0.998 |
| 3 | P2 gain-5 | 0.964 | **0.999** | 1.000 |
| 3 | w52 gain-100 | 0.991 | **0.998** | 1.000 |
| **10** | **P2 gain-5** | **0.698** | **0.974** | 0.942 |
| **10** | **w52 gain-100** | **0.930** | **0.988** | 0.972 |

`chart_frac` = `‖q‖ / ‖recall − x‖`, the fraction of the recalled displacement
the **local 2-D tangent frame** explains. `d1_chart` = residual outside the
env's 64-dim SVD chart, the §7.7 statistic that needs a per-env fit.

**The one scalar matches or beats the fitted basis everywhere**, and by three
points at ten distractors on both encoders. The means say why — at n=10 on the
P2 encoder, `frac_goal` **0.638** against `frac_dist` **0.125**, a 5× gap, with
the goal-absent value in the right ballpark for §7's own √(2/D) ≈ 0.044
prediction for an unrelated direction in D = 1024.

#### Three consequences

1. **§7.11's "group D's acquisition cost is unmeasured" largely dissolves.**
   That gap existed because `d1` needed a basis fitted from the env's 400
   cells. `chart_frac` needs no fit: `W` is already built every step
   (`project_to_signal`) and the recall is already computed. **One extra
   scalar, no in-env experience required.**

2. **It is a large gain over what the policy actually receives.** Against
   `‖q‖`, the statistic in the observation today: **+0.276 AUC** on the P2
   encoder (0.698 → 0.974) and +0.058 on w52, at ten distractors. This is
   precisely the signal §7.7.1 identified as deleted by the projection, and it
   survives compression to a single number.

3. **The encoder swap IMPROVES separability, and §7.7.1's exposure claim was
   also wrong.** w52 is better on all three statistics. `d1_chart` does fall
   from 1.000 to 0.972 at ten distractors — but it falls to **0.942 on the P2
   encoder as well**, so that is a distractor-count effect, not an encoder
   effect. The worry that a more binary, higher-dimensional code would collapse
   the goal-absent separability is not supported: `auc_qmag` at n=10 is 0.930
   on w52 against 0.698 on the gain-5 code.

#### What this does NOT establish

- **The absolute numbers are not §7.7's.** This module samples 200 random cells
  per env; §7.7's 0.887 for `a1_qnorm` is a per-env median over billiard
  *trajectories*. `auc_qmag` here reads 0.698 on the same encoder. The
  within-run comparisons are valid — all three statistics share one sampling —
  the cross-reference to the published table is not, and no claim here rests
  on it.
- **This is static, per-cell, and policy-free.** It says the information is
  present and cheap to expose. It does not say a policy trained with the extra
  channel would use it, which is a training experiment and untried. Note also
  `feedback_hopfield_nav_bc_inputs`: the input set was frozen for the bc-AQ
  line, so adding a channel is a decision, not an obvious win.
- **It needs `‖recall − x‖`**, the norm of the full 1024-dim displacement.
  Trivial to compute where the recall already exists, but it is a rollout
  change, not free at the policy's input layer today.

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
  **§7.7.1 reframes what this cost buys** and **§7.7.2 largely dissolves it**:
  the policy receives only the 2-D projection `q`, and the scalar
  `‖q‖ / ‖recall − x‖` — needing no env-specific basis at all — scores
  **0.974 / 0.988** at ten distractors against the fitted `d1_chart`'s
  **0.942 / 0.972**. The cheap version wins, so the 1024-dim route and its
  acquisition cost are not required. What remains open is whether a *policy*
  trained with the channel uses it, which is a training question.
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

### 9.2 CORRECTION — the clamp was not why the policy sat at the cap

§9.1 argued that the hard clamp *cost* explore coverage by parking the policy in
a zero-gradient region at the boundary, and predicted that a policy free to
choose its magnitude would move toward billiard's `‖a‖ ≈ 1.25` optimum and reach
roughly 0.42. Both halves are wrong.

Four arms with the radial tanh squash on the mean (§ P9), which bounds `‖μ‖`
smoothly with live gradient everywhere:

| arm | `mu_norm` | `sigma` | `ang_noise` | coverage |
|---|---|---|---|---|
| `p5_e` — hard clamp | **8.18** commanded, 1.98 realized | 0.497 | ~0.06 | 0.379–0.400 |
| `p9_e_sq` — soft bound | **1.98** | 0.497 | 0.251 | 0.372–0.376 |
| `p9_e_sq_std` — soft + state σ | **1.97** | 0.311–0.319 | 0.158–0.162 | 0.367–0.386 |

**The policy chooses ~1.98 with full gradient available.** It was never trapped;
it prefers the maximum. So the clamp did not cause the speed choice, and
removing the dead zone does not change it.

The prediction rested on an assumption that was never tested: that the policy's
measured strategy efficiency of 1.11 would *hold* at a different magnitude. That
efficiency is measured against billiard, and this policy already **beats**
billiard by 11% — so there was no reason its own optimum should sit where
billiard's does. Extrapolating one policy's efficiency across magnitudes is not
a prediction, it is an assumption wearing one.

**What the change did achieve, and it is not nothing:**

1. **`‖μ‖` is bounded.** 1.97–1.98 against a commanded 8.18, so the
   commanded/realized gap is closed. That removes the confound §8.2 found in
   *both* phase-2 brackets — future manipulations of σ or the time penalty will
   vary what they are named after.
2. **σ is a live knob for the first time.** The state arms moved it 0.497 →
   ~0.31 and the resulting angular noise genuinely differs between arms (0.251
   against 0.158–0.162), where §8.2 showed a 3× nominal σ range collapsing to
   1.33× effective.
3. **Performance is unchanged.** Coverage 0.367–0.386 against 0.379–0.400;
   exploit similar. So this is a **measurement fix, not a capability gain**, and
   should be described as one.

#### What is still unanswered, and why the diagnostic cannot answer it

The stated pass/fail was whether the σ head **takes over the state-dependent
modulation** that §8.2 found the policy performing through `‖μ‖`. The per-update
`sigma` is a **batch mean**, and a batch mean cannot distinguish "learned a lower
global σ" from "learned a σ that varies with state". The instrument answers a
different question from the one it was built for.

Answering it needs σ measured *conditioned on distractor count* — the axis along
which §8.2 measured the angular noise doubling. Until then, the honest statement
is that σ is now movable and has moved, not that it has become state-dependent.

### 9.3 RESULT — the sigma head learned a lower constant, not a state-dependent one

§9.2 recorded that the pass/fail could not be read from the training logs,
because the per-update `sigma` is a batch mean. Measured properly
(`behavior_probe --mode nav`, sigma binned on both axes, 8 envs x 32 trials),
against the global-sigma arm as a control whose sigma is constant by
construction:

| n_dist | state σ | state ‖μ‖ | state ang | ctrl σ | ctrl ‖μ‖ | ctrl ang |
|---|---|---|---|---|---|---|
| 0 | 0.2938 | 1.595 | 10.6° | 0.4966 | 1.584 | 18.0° |
| 3 | 0.3042 | 1.419 | 12.3° | 0.4966 | 1.408 | 20.2° |
| 10 | 0.3190 | 1.307 | 14.0° | 0.4966 | 1.284 | 22.2° |

**Axis 1 — distractor count. σ is state-dependent, and the head displaced
nothing.** σ rises 1.086× from zero to ten distractors, which is real
state-dependence in the predicted direction. But the control settles it: `‖μ‖`
modulates **1.234× without the head and 1.220× with it**. The policy performs
the same magnitude modulation either way, and the head merely **added** its
1.086× on top.

*(An earlier reading of the state arm alone reported this as "σ takes over 29%
of the modulation, `‖μ‖` does 71%". That share-of-total framing is wrong and the
control is what shows it: a share implies displacement, and none occurred. The
correct statement is additive, not fractional.)*

**Axis 2 — distance to the goal. σ is flat, and the prediction fails.** Within
each distractor level σ varies by ~3% across distance bins with no trend — at
d=0 it reads 0.2857 / 0.2799 / 0.2770 / 0.2855 from nearest to farthest, and at
d=10 it is *slightly lower* near the goal (0.303) than far (0.312). Meanwhile
angular noise rises sharply toward the goal in **both** arms — 13.7° → 8.6°
near-to-far in the state arm, 26.6° → 14.4° in the control.

§9.1's prediction was that a sigma tracking readout trustworthiness should
**rise** within two cells of the goal, where P1 measured the readout collapsing
(10th percentile of `cos(q, goal)` → 0.37). It does not. The near-goal increase
in angular noise is achieved **entirely by slowing down**, identically in an arm
that has no sigma head at all.

That is a sharper negative than a flat result would have been. The head *can*
condition on state — it does so for distractor count — but does not use
proximity, even though proximity is where the readout actually degrades and the
information is present in its inputs.

**What the head actually did.** Its dominant effect is a **1.69× lower global
sigma** (0.294 against the control's 0.497), which roughly halves angular noise
everywhere. So a head introduced to make exploration state-dependent mostly
learned a better constant.

#### Consequence

The residual magnitude channel flagged in `action_head.py` — that `σ/‖μ‖` can
still vary 4× over `‖μ‖ ∈ [0.5, 2]`, which exceeds the 2.2× modulation the
policy exhibits — **binds**. The policy keeps using it in preference to the
channel provided. Bounding the mean was necessary to make σ meaningful, and it
was not sufficient to make σ the channel the policy reaches for.

That is the case for the polar parameterization, now on evidence rather than on
principle: magnitude and heading as separate distributions, so directional
exploration cannot be bought by changing speed.

---

### 9.4 P10 — the polar action parameterization

**Launched 2026-08-26.** Exploit arms `p10_pol` (21295699) and `p10_pol_v1`
(21295703) on `pi_fiete`; explore arms `p10_e_pol` (21295996) and
`p10_e_pol_v1` (21295997) on `ou_bcs_normal`, moved there because pi_fiete's
group GRES quota had them pending behind the exploit pair. Code in
`hopfield_nav/policy/polar_head.py`; 42 tests in `tests/test_polar_head.py`.

#### The parameterization

```
theta ~ VonMises(theta_bar, kappa)          allocentric world-frame heading
r     ~ lo + (hi-lo) * Beta(mu*nu, (1-mu)*nu)
a     = r * (cos theta, sin theta)
```

Four heads off the same trunk: **two means and two spreads**. The heading mean
is the *existing* `movement_mean` Linear read as a direction and `atan2`-ed, so
a checkpoint forked into a polar run keeps its learned direction.

**Allocentric, not egocentric**, because `q = W_x(recall(x) - x)` is itself a
world-frame displacement — the heading the policy is matching is world-frame.
An egocentric turn would make it re-derive `target - previous heading` from
information it already has, and the frame would drift whenever the env clamp
made the realized heading differ from the commanded one.

**`(mu, nu)`, not `(alpha, beta)`.** Identical family; this is only how alpha
and beta are computed. The reason is freezability: in `(alpha, beta)` there is
no spread parameter to freeze, because holding alpha fixed and learning beta
moves the mean *and* the spread. `--freeze_log_std` has been load-bearing here
and already spent the whole v35 lineage silently doing nothing (§ memory
`project_hopfield_nav_log_std_freeze_bug`), so a freeze has to be a
`requires_grad` flag on one named scalar. `nu >= 2` forbids a U-shaped speed
density for **every** mu — a U-shape needs `nu < min(1/mu, 1/(1-mu)) <= 2` — so
one constant floor buys unimodality with no coupled restriction on mu.

**`--freeze_speed 1.0` deletes the speed factor** rather than taking a
degenerate limit: its log-prob and entropy slots are exactly zero. A
zero-variance Normal or infinite-concentration Beta would give `+inf` and
`-inf`. Inexpressible under the Cartesian head at any parameter setting, which
is itself an argument for polar.

#### PPO correctness

`log_prob` is taken on `(theta, u)` and **omits** both the polar→Cartesian
Jacobian `-log r` and the affine rescale `-log span`. Each depends only on the
sampled action, never on a parameter, so both cancel exactly in the importance
ratio — verified in `test_omitted_jacobians_cancel_in_the_ratio` against a
reference that carries them explicitly.

Entropy is `H(VonMises) + H(Beta)`: the **polar** entropy, not the Cartesian
one, which differs by `E[log r]` — precisely the term that would let an entropy
bonus buy directional randomness with speed. The Beta's entropy does depend on
mu, but *symmetrically* about mid-speed, so it cannot be increased by going
faster. (A log-normal speed would have been worse: its entropy carries a bare
`+m`, monotone in the mean. That is why it was rejected.)

#### Two bugs found by measuring, not by reading

**1. `atan2` hides a gauge freedom.** The importance-ratio spread is *entirely*
the heading factor — measured max `|Δ log p|` of 0.283 against the speed
factor's 0.005. `atan2` has gain `kappa/‖v‖`, and `‖v‖` is unpressured by
anything in the objective (theta is scale-invariant), so it random-walks toward
the singularity. Measured gain: 26 at `‖v‖=0.24`, 636 at 0.01, 6360 at 0.001,
unbounded at 0. **Same shape as what killed the first `p9_e_sq_std` run at
u120.** Fixed by shrinking kappa as `‖v‖²/(‖v‖²+s²)`, which makes a short
direction vector mean a *low concentration* — bounded, and the correct limit.

**2. `dir_soft = 0.05` was an active participant, not a backstop.** The first
smoke run showed the real 1024-unit trunk emits `‖v‖ ≈ 0.071` at init, where
0.05 cut kappa from 6.36 to 4.25 — 23.8° of directional noise silently became
29.9°, breaking the calibration against the p9 arms. Retuned to **0.01**: 1%
distortion there, less as `‖v‖` grows, gradient still capped at ~318.

| `dir_soft` | κ_eff at ‖v‖=0.071 | circ sd | peak ‖∂logp/∂v‖ |
|---|---|---|---|
| 0.05 | 4.251 | 29.9° | 63.6 |
| 0.02 | 5.892 | 24.8° | 159 |
| **0.01** | **6.236** | **24.0°** | **318** |
| 0.005 | 6.328 | 23.8° | 636 |

The softening **bounds** the decay; it does not prevent it. Hence `dir_norm` in
the per-update log — watch it.

#### Calibration: both parameterizations share one axis

Deliberate, so §9.3's table and the P10 runs plot together:

| column | Cartesian | polar |
|---|---|---|
| `mu_norm` | `‖μ‖` | mean speed |
| `sigma` | radial noise | speed sd |
| `ang_noise` | `σ/‖μ‖` | von Mises circular sd |
| `kappa` | — (absent, not 0 or NaN) | κ |
| `dir_norm` | — | `‖v‖`, the gauge |

`init_log_kappa = 1.85` (κ = 6.34) is the value reproducing `init_log_std=-0.7`'s
σ = 0.497 at mid-speed 1.25 — both ≈ 23.8°, so a polar arm starts with the same
directional exploration as the p9 arm it is compared against. And §9.3's
measured 10.56° of `σ/‖μ‖` is κ = 29.4, which reads back as **10.66°** — the two
conventions agree to ~1%.

#### The prediction, and what would falsify it

All four arms run `STATE_DEPENDENT_STD=1`, which under polar makes kappa and nu
per-state heads. Controls already exist — `p9_sq` and `p9_sq_std` complete the
2×2 — so no new control arm is needed.

**Predicted:** kappa picks up the state-dependence sigma refused to — lower at
high distractor count *and* near the goal, where P1 shows the readout
collapsing — while mean-speed modulation across distractor levels falls below
the control's 1.234×.

**Falsifier:** if mean speed still modulates ≈1.23× with heading noise fully
decoupled, that modulation was a genuine speed policy all along, the
residual-channel story is wrong, and §9.3's conclusion needs retracting. The
`_v1` arms are the sharp version: with speed constant by construction,
everything the policy does must go through kappa.

Read with `analysis/nav_tri/behavior_probe.py`, which now records κ per step
and reports circular sd in the same `ang_*` columns, binned by distractor count
and distance to goal exactly as §9.3 was.

#### The first launch died in update 2 — and the cause was not polar

Both exploit arms (21295699, 21295703) completed **exactly update 1** and
crashed in update 2 with *every* entry of the heading NaN. Recorded because the
first hypothesis was wrong in an instructive way.

**The learned-speed and the frozen-speed arm died identically**, which rules out
the Beta — the frozen arm has no speed factor at all. Ruled out by direct
measurement, not by argument:

| suspect | test | result |
|---|---|---|
| Beta sampler degenerate draws | 5.1M draws at the real per-update volume | 0 exact-0, 0 exact-1, 0 NaN |
| Beta `log_prob` overflow | ‖a‖ from 0 to 1e6, α down to 0.1 | finite everywhere |
| `vm_entropy` gradient | κ from 1e-6 to 148 | finite everywhere |
| polar backward | ‖v‖ from 0 to 1, through *both* `log_prob` and `entropy` | finite, bounded |

The fault is in the PPO loop around them, and it is **generic — polar only made
a latent hazard live**:

1. `exp()` of a log-ratio. A von Mises at its κ ceiling can put ~2κ ≈ 296
   between two log-probs; `exp(296)` is `inf` in float32. A Gaussian's
   quadratic form is gentler, which is why this never fired before.
2. **`inf * 0` is NaN.** The masked reductions used `* mask`, and the masked
   steps are exactly the ε / auto-nav ones whose ratio the existing comment in
   `ppo.py` already says explodes. The mask that exists to *remove* those steps
   is what converted them into a NaN.
3. `clip_grad_norm_` scales by `max_norm / (total_norm + 1e-6)`. With
   `total_norm = inf` that factor is 0, and `inf * 0` is NaN — so one bad
   sample did not merely dominate the update, it wrote NaN into a parameter
   **permanently**.

Fixed in `ppo.py`: clamp the log-ratio before `exp` (`exp(20)` = 4.9e8, far past
anything `clip_coef` admits, so no healthy step changes); `torch.where` instead
of `* mask` in every masked reduction, because selection does not propagate a
non-finite through a zero; and **skip the optimizer step when the returned grad
norm is non-finite** rather than taking it. Verified numerically identical on
the healthy path against the pre-fix reproduction.

It also now prints which term went bad, once per update, and counts survivals
as `nonfinite_steps` in the log — a problem being survived should be visible,
not swallowed. **If `nonfinite_steps` is persistently nonzero, that is a real
problem being tolerated and wants its own diagnosis.**

**Still unproven:** the exact triggering sample. A synthetic reproduction at the
real shapes (320×200, ε-greedy overrides, realistic episode termination) did
*not* reproduce it in 8 updates, so the trigger needs the real observations.
The diagnostic exists so the next occurrence names itself rather than costing
another full pass.

##### CORRECTION — those three mechanisms are not the cause

The diagnostic fired on the very next launch and **refutes all three as the
explanation here.** They are real hazards and stay closed; they are not this.

Four reports across three arms, every one with the forward entirely clean:

| arm | ratio_max | logratio_max | adv_absmax | move_loss | value_loss |
|---|---|---|---|---|---|
| exploit, learned | 11.5 | 3.44 | 8.38 | 0.060 | 0.294 |
| exploit, frozen | 11.1 | 6.79 | 7.89 | 0.058 | 0.288 |
| explore, frozen | 1.006 | **0.006** | 6.84 | −0.014 | 2.21 |
| explore, learned | 13.0 | 2.57 | 5.40 | 0.019 | 6.04 |

No ratio anywhere near the ±20 clamp, no non-finite forward value, and no loss
term consistently large. The third row is decisive on its own: `logratio_max =
0.006` is *no policy change at all*, and it still produced a non-finite
gradient. **The overflow is created inside the backward, not carried in from
the forward.** So the fixes bought survival and information, not the diagnosis.

A second self-inflicted delay is recorded because it is the kind that repeats:
the first parameter-level report came back `nonfinite_params=[]
largest_finite=[]` — *both* empty, which reads as "no gradients anywhere". The
diagnostic was inspecting `.grad` after `optimizer.zero_grad(set_to_none=True)`
had already nulled it. A diagnostic placed after the cleanup it is diagnosing
reports nothing and looks like a finding.

**It is a rare transient, and the runs outgrow it.** Both explore arms passed
u10 learning normally (`mean_r` rising, `nonfinite_steps = 0` at the eval
points, one event each), and `dir_norm` grows on its own — 0.106 → 0.238 and
0.254 — i.e. the direction head moves *away* from the small-`‖v‖` region where
the heading gradient peaks.

##### THE PATH — confirmed by the parameter-level report

Pinned, not inferred, once the report read the gradients *before*
`clip_grad_norm_` overwrote them. Both arms name the identical set:

| arm | non-finite | which | finite |
|---|---|---|---|
| exploit, learned speed | **8 / 14** | `rnn`×4, `movement_mean`×2, `log_kappa_head`×2 | `value_head`×2, Beta speed×4 |
| exploit, frozen speed | **8 / 10** | the same eight | `value_head`×2 |

The frozen arm has four fewer parameters and the *same* non-finite set, so this
is unambiguous: **the von Mises path**. `movement_mean` feeds `atan2`;
`log_kappa_head` feeds κ through the same `shrink(‖v‖)`; the Beta speed path and
the value head are untouched; the RNN is downstream contamination via
`features`.

At this point I guessed the *mechanism* from the confirmed *path* and got it
wrong, which is recorded because the guess was plausible and cost a launch:
`direction` comes off a vanilla ReLU RNN unrolled 200 steps, activations can
grow like `‖W_hh‖^200`, and every form of the shrink computation squares `‖v‖`,
which overflows float32 past ≈1.8e19. That rewrite (divide out the max-abs
component first, so no unnormalized square exists; θ unchanged because atan2 is
scale-invariant, shrink rewritten as `sq_n/(sq_n + (s/vmax)²)`) is a **real
hazard on the far side of the range and stays**. It did not fix the crash, and
the next launch said so immediately.

##### ROOT CAUSE — the κ floor sat inside torch's VonMises NaN region

`torch.distributions.VonMises.log_prob` has a **NaN gradient with respect to
concentration for κ < 1e-5**, while its forward stays perfectly finite.
`_log_modified_bessel_fn` evaluates *both* branches and selects with
`torch.where`; the large-κ branch computes `3.75/x`, which overflows for tiny
x, and `where`'s backward then does `inf * 0`.

| κ | 1e-8 | 1e-7 | 1e-6 | 3e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|---|---|---|---|---|
| `dlogp/dκ` | NaN | NaN | NaN | NaN | ok | ok | ok | ok |

**The floor was 1e-6**, chosen as "uniform to every digit that matters", and it
landed squarely inside that region. Samples with a tiny direction vector
floored κ, `log_prob`'s backward produced NaN on `log_kappa_head` and
`movement_mean`, and `clip_grad_norm_` spread it. It accounts for every
observation: finite losses at `logratio_max` of 0.006 and 0.047 (no policy
movement at all), non-finite gradients confined to the von Mises path, the
Beta path and value head clean, both arms naming the identical set, and the RNN
as downstream contamination.

Floor is now **1e-2** — three orders of magnitude of margin, still uniform for
any purpose here (circular sd 186.5°, a 2% density modulation between the modal
and antimodal directions).

**Verified:** zero non-finite events on all four arms after the fix, and the
exploit arms — which had died at update 2 without exception across five
launches — ran past u30 clean and learning (`mean_r` 0.087 → 0.107).

Degeneracy became a **magnitude** test tied to `dir_soft` rather than an
exact-zero test: an exact-zero test reopened the κ-floor band — where the floor
binds while `atan2` is still live, breaking the `‖v‖²` proportionality the bound
rests on — at exactly 1000 for `‖v‖ = 1e-9`.

**Why the tests missed it:** the suite checked the gradient w.r.t. the
*direction input* as `‖v‖` shrank, and the *head parameters* only at `‖v‖ ≈ 1`,
where κ is nowhere near its floor. The crash needed both at once — a floored κ
**and** a backward reaching `log_kappa_head`. Two tests now cover it: one at
the source, which asserts the NaN region still exists so a future torch fix
gets the floor revisited deliberately rather than silently, and one sweeping
`‖v‖` through the degenerate region while checking head parameters.

**Method note.** Three separate times a diagnostic read state after the step
that destroyed it — `zero_grad(set_to_none=True)`, then `clip_grad_norm_`
(which rescales in place, and with a NaN norm smears NaN over *every*
parameter), then a `bad[:4]` truncation that could not distinguish "4 of 4 RNN
tensors" from "4 of 14 parameters". Each produced an empty or uniform result
that read as a finding. Instrumentation has an ordering contract too.

**Watch alongside it — and it is already happening.** The polar entropy is ~1/3
the Cartesian value at matched noise (H(vonMises)+H(Beta) ≈ 0.49 against a
Gaussian's 1.45), so `MOVE_ENT_COEF=0.005` — tuned for Cartesian — has less
purchase here. **The frozen-speed arm has it worst: with no speed factor its
entropy is the heading term alone, i.e. half the regularized quantity.**

Observed on the fixed code, `p10_pol_v1` at u40: κ 4.5 → **18.3**,
`move_entropy` → **−0.001**, angular noise 31° → 14°. Not yet a failure — 18.3
is well below the 148 ceiling, and a sharp heading is what exploit wants — but
the trajectory is a collapse, not a settle.

If κ pins near its ceiling, **raise `ent_coef` rather than lowering
`log_kappa_max`**: the ceiling is a bound on the symptom, the coefficient acts
on the cause. Note the two arms are not comparable on `ent_coef` as it stands,
because the frozen arm's entropy is missing a term the learned arm has; a
follow-up wanting them matched should scale `ent_coef` on the frozen arm rather
than assume 0.005 means the same thing in both.

Exploit arms relaunched with the fix as **21299767 / 21299771**. The two
explore arms (21295996 / 21295997) were **left running** rather than restarted:
they already carry the skip-on-non-finite guard, report `nonfinite_steps = 0`
at every eval, and were 50 updates in. The only code they lack is the overflow
rewrite, which is algebraically identity-preserving, so the arms stay
comparable — the explore pair simply skips a rare minibatch where the exploit
pair no longer needs to.

#### First eval, u50 (early — 1500/2000 updates to go)

| arm | coverage | cells/step | speed | κ | ang noise |
|---|---|---|---|---|---|
| `p10_e_pol` (learned speed) | 0.193 | 0.386 | **1.478** | 4.70 | 28.8° |
| `p10_e_pol_v1` (speed ≡ 1) | 0.161 | 0.321 | 1.000 | 3.25 | 39.0° |

Reference ladder: 0.36 = uniform random walk, 0.775 = billiard. Too early to
read, but noted because it is the first datapoint on what freezing speed costs:
the free arm picked 1.478, inside the measured billiard band, and leads on
coverage.

#### u50 EXPLOIT — a reading that did not survive; see the reversal below

| arm | success | speed | κ | ang noise | `dir_norm` |
|---|---|---|---|---|---|
| `p10_pol` (learned speed) | **0.677** | 1.413 | 7.01 | 22.9° | 0.338 |
| `p10_pol_v1` (speed ≡ 1) | **0.031** | 1.000 | **23.16** | 13.1° | 0.548 |

*(The `mean_steps` column reads backwards — 42.0 against 10.0 — because
`mean_steps` is computed over SUCCESSES ONLY, so the frozen arm's 10.0 is a
biased sample of its 3% easiest trials. The mean_steps trap, again.)*

Speed 1.0 alone cannot explain 3%: at speed 1 the ideal `mean_steps` over a
~10.8-cell start distance is ~10 against 200 available, so capping speed caps
*efficiency*, not reachability. **Premature convergence can** — κ ran to 23
while success sat at 3%, i.e. the policy sharpened its heading hard around a
direction it had not learned.

**Hypothesis for why freezing speed CAUSES that.** In the learned arm a bad
heading can be hedged by slowing down: a shorter step overshoots less. With
speed pinned that hedge is gone, heading errors cost more, and PPO's only
remaining lever is to sharpen κ — which removes the very exploration that would
have found the right heading. If it holds, **speed and heading are independent
in the parameterization but not in their effect on the task**, which is a P10
result rather than a nuisance.

A wrong intermediate reading is recorded because it was tempting: `move_entropy
= −0.001` looked like an entropy collapse, but von Mises differential entropy on
the circle is `0.5·log(2πe/κ)`, which crosses zero at κ = 2πe ≈ 17.1. A negative
value means κ > 17, not that exploration stopped. Separately, "the frozen arm
has less entropy to regularize" was also wrong: the bonus gradient w.r.t. κ is
`−ent_coef · dH_vm/dκ` in *both* arms, and the Beta term is additive in the
speed parameters, contributing nothing to κ's gradient.

**P10b — the decisive test.** `p10_pol_v1_e20` (21302000) and `p10_pol_v1_e50`
(21302001): identical frozen-speed exploit arms at `MOVE_ENT_COEF` 0.02 and
0.05, i.e. 4× and 10× the 0.005 the learned arm is stable at. If more entropy
pressure recovers success, the failure is κ runaway and freezing speed is
survivable with retuning. If it does not, freezing speed is itself the cost, and
the hedging story above is the explanation.

#### The dissociation — freezing speed helps explore and wrecks exploit

All four arms at u50, which is the strongest structure in the data so far:

| arm | metric | value | κ | ang noise |
|---|---|---|---|---|
| `p10_e_pol` explore, learned | cov / cps | 0.163 / 0.326 | 3.99 | 31.7° |
| `p10_e_pol_v1` explore, frozen | cov / cps | **0.207 / 0.414** | 5.46 | 26.4° |
| `p10_pol` exploit, learned | success | **0.677** | 7.01 | 22.9° |
| `p10_pol_v1` exploit, frozen | success | **0.031** | **23.16** | 13.1° |

Freezing speed is **better** for explore and **catastrophic** for exploit — and
the frozen *explore* arm's κ is a healthy 5.46, no runaway whatsoever. So
pinning the speed does not by itself drive κ up; it does so **only in exploit**.

That is evidence *for* the hedging hypothesis, not against it: the mechanism
predicts the pathology exactly where there is a target to overshoot. Explore has
no goal, so slowing down buys nothing, so removing the option costs nothing and
κ stays put. Exploit has a goal, overshoot is costly, the hedge is gone, and κ
runs.

The explore side also has a cleaner reading than "speed 1.0 beats 1.5": the
frozen arm's headings are *sharper* (26.4° against 31.7°), and a sharper heading
means a straighter trajectory, which is what covers new cells. Worth separating
those two before crediting the speed — the free arm chose 1.5 AND a blunter
heading, and only one of those is the speed's doing.

**And the u50 explore ordering does not survive.** Full trajectories:

| arm | u50 | u100 | u150 | u200 | u250 | u300 |
|---|---|---|---|---|---|---|
| `p10_e_pol` learned | 0.163 | **0.126** | 0.333 | 0.373 | 0.375 | 0.355 |
| `p10_e_pol_v1` frozen | 0.207 | 0.274 | **0.125** | | | |

Both arms take a **dip and recover** — the learned one at u100, the frozen one
at u150. Reading the frozen arm's 0.125 as a collapse was the peak-calling trap
from `project_hopfield_nav_explore_exploit` in reverse; training reward climbed
straight through it (0.287 → 0.330 → 0.335), which was the tell. Nothing in
either explore arm is decided before ~u200.

**Where the explore side actually stands:** `p10_e_pol` reaches
**cps 0.75 at u250** against the billiard reference of **0.775**, at speed 1.732
and κ 19.7. Near-billiard efficiency. The u300 value of 0.710 is inside eval
noise of that, so this is not a peak call.

---

### 9.5 REVERSAL — freezing speed to 1 WINS on exploit

Everything in §9.4's u50 exploit reading is wrong. Full trajectories:

| arm | u50 | u100 | u150 | u200 | u250 | u300 |
|---|---|---|---|---|---|---|
| `p10_pol` learned speed | 0.677 | 0.844 | 0.969 | 0.979 | 0.990 | |
| `p10_pol_v1` **speed ≡ 1** | 0.031 | 0.208 | 0.896 | 0.510 | 0.906 | **1.000** |

| at its best | success | mean_steps | κ | ang noise |
|---|---|---|---|---|
| learned speed (u250) | 0.990 | 27.2 | ≈13 | 16.4° |
| **frozen speed (u300)** | **1.000** | **19.6** | 37.8 | 9.7° |

**CAVEAT on that table — it compares two peaks, which is the thing this
project's notes warn against.** At u300 the learned arm fell to **0.698**, so its
full series is 0.677 / 0.844 / 0.969 / 0.979 / 0.990 / 0.698 against the frozen
arm's 0.031 / 0.208 / 0.896 / 0.510 / 0.906 / 1.000. Both swing by 30+ points
between consecutive evals and **neither has converged**. The defensible claim is
that both reach ≥0.99 at their peaks and the frozen arm was slower to start, not
that it "wins on both metrics".

Three claims to retract, all made from a single early eval point:

1. *"Freezing speed costs 22× in success."* It costs a slower start. By u300 it
   is ahead on success **and** takes 28% fewer steps.
2. *"κ ran to 23 while success sat at 3% — premature convergence."* Backwards.
   κ kept climbing to 37.8 and success went to **1.000**. The sharpening was the
   policy **learning to point accurately**, which is precisely what produces the
   straighter, shorter paths. A rising κ is the signature of success here, not
   of collapse.
3. *The hedging hypothesis* — that pinning speed removes the option of slowing
   down to hedge a bad heading, leaving κ as the only lever. It predicted a
   failure that did not occur. It may still describe the early transient, but it
   is not load-bearing for anything and should not be cited.

`p10_pol_v1_e20` / `_e50` were launched to discriminate κ-runaway from
freezing-itself and are **cancelled**: the premise is refuted, and more entropy
pressure would blunt exactly the κ that is producing the result.

### 9.6 THE P10 ANSWER — the falsifier resolves, and κ does what σ would not

Behaviour probes on both exploit checkpoints: `p10_pol_v1` u300 (frozen speed,
1.000 success) and `p10_pol` u250 (learned speed, 0.990).

**Read the learned-speed arm first.** An earlier revision of this section
concluded, from the frozen arm alone, that "a constant spread is what the policy
wants". That is **wrong** — the learned-speed arm shows a clean proximity
gradient in κ. The frozen arm is the special case, not the general one, and
generalizing from it was the same one-arm error as §9.5's.

#### The falsifier — resolved, and the §9.3 story survives

| across n_dist 0→10 | §9.3 Cartesian | P10 polar |
|---|---|---|
| mean magnitude / speed | **1.234×** | **1.018×** |
| spread (σ → κ) | 1.086× | 1.067× |

§9.4 set the falsifier as: *if mean speed still modulates ≈1.23× with heading
noise decoupled, the residual-channel story is wrong and §9.3 needs retracting.*
It modulates **1.018×**. The channel closed. The policy **was** buying
directional variation with speed, and given a proper channel it stopped.

#### κ does what σ refused to — on the axis that matters

Directional noise by distance to goal, learned-speed arm:

| n_dist | d0–2 | d2–4 | d4–8 | d8+ | ratio |
|---|---|---|---|---|---|
| 0 | **18.82°** | 17.31° | 16.05° | **13.98°** | **1.35×** |
| 1 | 18.68° | 17.14° | 15.88° | 13.75° | 1.36× |
| 3 | 18.77° | 17.13° | 16.09° | 14.12° | 1.33× |
| 5 | 18.65° | 17.27° | 16.49° | 14.35° | 1.30× |
| 10 | 18.06° | 17.69° | 17.25° | 17.24° | 1.05× |

Monotone, ~1.34×, reproduced at four distractor levels: **more directional
noise near the goal**. That is §9.1's prediction — the one §9.3 measured as flat
for σ and called a sharp negative — now satisfied. And it cannot be §9.3's
artifact: there the near-goal angular rise was `σ/‖μ‖` and came entirely from
slowing down, whereas this is the circular sd computed from κ directly, with
mean speed flat at 1.37–1.40 throughout.

**It collapses at n_dist=10** (1.05×), where `q_accuracy` is 0.613. With the
readout globally untrustworthy the policy stops treating near-goal as special
and instead raises noise everywhere — mean 16.37° at n_dist=0 against 17.27° at
10. Selective caution gives way to uniform caution.

#### The frozen-speed arm has NO gradient — and that is the open question

| n_dist | d0–2 | d8+ | ratio |
|---|---|---|---|
| 0 | 10.06° | 9.74° | 1.03× |
| 10 | 10.17° | 10.07° | 1.01° |

Flat at every level, at a much sharper operating point overall (9.8° against the
learned arm's 16.4°).

##### RESOLVED — it is sharpness, not the frozen speed

The learned-speed arm regressed to κ ≈ 32 by u300, which is a
**sharpness-matched control against the frozen arm with speed still free**.
Probed (21311943):

| checkpoint | κ (mean ang) | speed | gradient d0–2 → d8+ |
|---|---|---|---|
| learned u250 | ≈13 (16.4°) | free | **1.35×** |
| learned u300 | ≈32 (11.2°) | free | **1.15×** |
| frozen u300 | ≈38 (9.8°) | pinned | **1.03×** |

Monotone in sharpness, and the learned arm at matched κ behaves like the frozen
arm. **The gradient decays as the policy sharpens, regardless of whether speed
is free.** The "joint widening" hypothesis — that pinning speed suppresses a
coordinated widening of both channels — is unsupported and should be dropped.

That also qualifies §9.6's headline: κ is state-dependent on proximity **while
the policy is still relatively broad**, and the modulation shrinks as it
converges. It is a property of the training regime, not a fixed property of the
parameterization.

| n_dist | 0 | 1 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| directional noise (circ sd) | 9.83° | 9.88° | 9.85° | 9.99° | **10.12°** |
| success | 1.000 | 0.995 | 0.984 | 0.964 | 0.901 |
| `q_accuracy` | 0.989 | 0.984 | 0.958 | 0.833 | **0.711** |
| `follow_q` | 0.558 | 0.506 | 0.499 | 0.443 | 0.427 |
| `mean_steps` | 20.6 | 23.0 | 22.9 | 23.3 | 26.8 |

κ modulates 1.029× across 0→10 here — flat, on both axes. Taken alone this
looked like "the policy does not want state-dependent noise"; the learned-speed
arm above shows that conclusion does not generalize.

#### A second adaptation, present in both arms — reliance, not spread

`q_accuracy` falls 0.989 → 0.711 as distractors rise, and `follow_q` falls
0.558 → 0.427 with it. **The policy does adapt to a degrading readout — by
trusting it less.** That is state-dependent behaviour of exactly the kind §9.1
sought, expressed through the policy **mean** rather than its spread.

(Not a claim about rate: follow_q falls 0.765× against q_accuracy's 0.719× over
the full range, but faster than it at n_dist=5 and slower at 10. It tracks the
readout; it does not provably outpace it.)

"Be more uncertain where the readout is bad" and "rely on the readout less
where it is bad" are both valid responses to the same information, and this
policy does **both**: κ near the goal (learned-speed arm) and `follow_q` against
distractor count (both arms). §9.1 looked only for the first, which is why §9.3
read as a flat negative.

#### Consequence for the polar case

The polar parameterization is vindicated on its own terms. It closed the
magnitude channel (1.234× → 1.018×), and with that channel closed the spread
channel **is** used state-dependently on the axis P1 says matters — which is
exactly the mechanism §9.3 predicted and could not demonstrate. It also enables
the frozen-speed arm, the best exploit model in phase 2 (§9.5).

Two follow-ups, both now motivated by evidence:

- **Why the frozen arm has no gradient** — the `log_kappa_max` test above. It
  matters because P6 will want one model doing both regimes, and freezing speed
  is the right exploit choice and the wrong explore one (§9.5).
- **What governs `follow_q`.** It is the larger effect (0.558 → 0.427, and
  0.407 → 0.392 in the learned arm) and nothing in the action parameterization
  addresses it. It is where the policy encodes confidence in the readout.

---

### 9.7 `follow_q` needs a baseline — and mode A survives the correction

**Reading order, because this section reversed twice.** The original text called
mode A from the aggregate `follow_q`; that reasoning was invalid (see the
baseline argument below) and was retracted. The retraction then proposed that
the low `q_accuracy` was near-goal geometric degeneracy — **that hypothesis is
also dead**, refuted by the distance-binned probe in §9.7.1. The mode-A
conclusion is re-established in §9.7.2 on the divergence test, which is the only
comparison that can support it. Net: the original call was right, none of the
original reasoning was.

**This section originally claimed these models "barely follow the readout" and
that one had fallen into mode B. Both claims are wrong.** They came from
comparing `follow_q` against 1.0. The correct baseline is `align_true`, the
policy's alignment with the TRUE goal direction — that is the only comparison
that separates *"does not use the readout"* from *"does not move very
directly."*

| checkpoint | `align_true` | `follow_q` | `q_accuracy` | `path_efficiency` |
|---|---|---|---|---|
| polar frozen u300 | 0.548 | **0.558** | 0.989 | 0.557 |
| polar learned u300 | 0.003 | **−0.001** | 0.993 | 0.464 |
| Cartesian `p9_sq_std` u900 | 0.835 | **0.843** | 0.987 | 1.256 |

`follow_q ≈ align_true` in every case at n_dist=0 — but **that equality proves
nothing**, and citing it as evidence was a second error. The three metrics are
the three pairwise cosines among the action `a`, the recall `q`, and the true
goal direction `g`: `follow_q` = cos(a,q), `align_true` = cos(a,g),
`q_accuracy` = cos(q,g). When `q_accuracy` ≈ 0.99, `q` sits 8.5° from `g`, so
`follow_q` and `align_true` are *forced* to be nearly equal no matter what the
policy does. The comparison is only informative where `q` and `g` diverge.

**At n_dist=10 they diverge, and the answer is unambiguous:**

| frozen arm, n_dist=10 | all trials | failed trials |
|---|---|---|
| `q_accuracy` | 0.711 | **0.097** |
| `follow_q` | **0.427** | **0.373** |
| `align_true` | 0.279 | **−0.182** |

`follow_q` (0.427) **exceeds** `align_true` (0.279): the action tracks `q` more
closely than it tracks the true goal. The policy is using the readout.

And on failures it followed `q` faithfully (0.373) while `q` pointed ~84° wrong
(0.097), and therefore moved *away* from the goal (−0.182). **That is mode A —
trusting a broken readout — which is ENCODER-limited.** The original section
named mode B, i.e. the opposite mode. This model's residual ~10% failure at ten
distractors is the recall's fault, not the policy's.

The u300 learned arm is not ignoring `q`: `align_true` is also ≈0, i.e. it is
not moving toward the goal either. It is a bad policy in a transient dip
(success 0.698, one of the oscillation's low points), reaching the goal by
covering ground — 46 steps × 1.59 = 73 cells of path in a 400-cell arena.

**The arithmetic that should have caught this immediately.** 1.000 success at
15–20 steps from ~10.8 cells out, at 1 cell/step, is a directed trajectory;
blind search of a 400-cell arena cannot do it. Directness 0.55 × 20.6 steps
≈ 11 cells of progress ≈ the 9.95 required. The numbers were always consistent
with full use of `q`; only the baseline was wrong.

#### 9.7.1 The distance-binned probe — geometry hypothesis refuted

`q_accuracy` by distance to goal, frozen arm (probe 21324515):

| n_dist | d0–2 | d2–4 | d4–8 | d8+ |
|---|---|---|---|---|
| 0 | **0.976** | 0.989 | 0.992 | 0.992 |
| 1 | 0.958 | 0.984 | 0.991 | 0.992 |
| 5 | 0.892 | 0.932 | **0.751** | 0.883 |
| 10 | 0.756 | 0.845 | 0.722 | 0.734 |

**The readout is excellent right up to the goal at low distractor counts —
0.976 in the nearest bin at n_dist=0.** So the near-goal geometric degeneracy
proposed in the retraction does not exist. Degradation is driven by
*distractors*, and at n_dist=10 it is spread roughly uniformly across distance
(0.72–0.85), not concentrated anywhere.

#### 9.7.2 Mode A, on the divergence test

Where `q` and `g` diverge, `follow_q` **exceeds** `align_true` — the policy
tracks the readout more closely than the truth, and is misdirected when the
readout is wrong:

| | q_accuracy | follow_q | align_true |
|---|---|---|---|
| n_dist=5, d4–8 | 0.751 | **0.460** | 0.299 |
| n_dist=10, d4–8 | 0.722 | **0.485** | 0.244 |
| n_dist=10, d8+ | 0.734 | **0.570** | 0.452 |

That is mode A — trusting a broken readout — **encoder-limited**. The residual
~10% failure at ten distractors is the recall's fault, not the policy's.

The nearest bin inverts: at n_dist=10, d0–2 has `follow_q` 0.187 **below**
`align_true` 0.292. Near the goal the policy stops following `q` and is better
aligned with the truth than with the readout — a distinct regime, see 9.7.3.

#### 9.7.3 The terminal approach is the largest policy-side inefficiency

Alignment by distance, both arms at n_dist=0, both at 1.000 success:

| bin | `q_accuracy` | frozen `align_true` | learned `align_true` |
|---|---|---|---|
| d8+ | 0.99 | **0.817** | 0.710 |
| d4–8 | 0.99 | **0.524** | 0.287 |
| d2–4 | 0.99 | **0.419** | 0.271 |
| d0–2 | 0.97 | **0.449** | 0.218 |

`q` is 97–99% accurate at every distance and **both** policies stop tracking it
over the final two cells. That is where the path length goes: ~11 cells of
journey taken in 15–20 steps.

**Frozen speed is exonerated — it is better at every bin**, roughly double the
learned arm's alignment in the mid-range. It wins `mean_steps` (15.4 vs 18.7)
on *directness*, with half the stride, which is the opposite of the phase-1
pattern where stride bought `mean_steps` at the cost of productive motion.

Likely mechanism: a 1.85-cell step against a 1.0-cell goal radius overshoots and
reverses, and reversals average per-step alignment toward zero. **Testable and
cheap: freeze speed at the 0.5 floor and see whether near-goal alignment
improves further**, trading transit steps for terminal precision.

#### 9.7.5 RESOLVED at u900 — no polar deficit, and the P1 "gap" is not a gap

**The matched-budget comparison, which was the whole open question:**

| checkpoint (all u900) | `follow_q` | `q_accuracy` | success | steps | speed |
|---|---|---|---|---|---|
| Cartesian `p9_sq_std` | 0.843 | 0.987 | 1.000 | 8.05 | 1.98 |
| **polar frozen** | **0.819** | 0.989 | 1.000 | 13.4 | 1.00 |
| polar learned | 0.656 | 0.989 | 1.000 | 10.5 | 1.83 |

**The polar "memory-following deficit" was training budget, nothing else.** At
matched u900 the frozen arm reads 0.819 against 0.843 — equal for practical
purposes. `follow_q` in that arm climbed **0.558 → 0.819** between u300 and
u900. Every claim in the earlier drafts of §9.7 about polar degrading
memory-following is dead, and the confound flagged when the comparison was
first made is exactly what it turned out to be.

**And the P1 discrepancy dissolves.** `q_accuracy` at n_dist=10, *same six envs,
same encoder*, four policies:

| frozen u300 | learned u300 | frozen u900 | learned u900 |
|---|---|---|---|
| 0.711 | 0.716 | 0.599 | **0.509** |

Four values for one underlying readout. **The probe's `q_accuracy` is a property
of the TRAJECTORY, not of the memory** — it is the readout quality where a
particular policy happened to go. P1 samples all 400 cells uniformly and answers
"how good is the readout" (~0.95 at ten distractors); the probe answers "how
good was it along this rollout". Neither is wrong; they are different questions,
and several messages were spent hunting a bug that does not exist.

The direction is consistent: the frozen arm follows `q` more (0.74–0.82 against
0.52–0.66) and *sees* better `q_accuracy` along its path. A policy that tracks
the readout reaches the goal directly and spends few steps anywhere bad; one
that does not wanders into exactly those regions. **Trajectory `q_accuracy` is
partly an OUTCOME of policy quality, not an input to it.**

**Consequence for the mode-A call in §9.7.2.** `q_accuracy_fail` being low on
failed trials is partly circular — failed trials are by construction the ones
that wandered. The divergence test (`follow_q` > `align_true`) survives, because
it is a within-step comparison of two directions at the same state. But
"the readout is bad, therefore encoder-limited" needs **P1-style uniform
sampling** to support it, not the trajectory number. Mode A is indicated, not
established.

#### 9.7.6 CORRECTION — sharpness does not explain the near-goal gradient

§9.6 concluded the near-goal κ gradient decays monotonically with policy
sharpness (1.35× at κ≈13, 1.15× at κ≈32, 1.03× at κ≈38). **At u900 the frozen
arm is at κ≈93 — far sharper — and its gradient is back to 1.24×**
(7.67° near against 6.20° far), with the learned arm at 1.52°/1.52×. Sharper,
larger gradient. The trend is refuted.

What the u300 frozen checkpoint actually was is a mid-training state where the
gradient happened to be absent. Three points drawn from two different arms at
one moment each were read as a mechanism; they were never a series. Same error
as the four eval-point reversals, in a different metric.

#### 9.7.4 (superseded by 9.7.5) OPEN — the probe and P1 disagree about the readout

P1's uniform-grid measurement at ten distractors: lock=goal 98.7–99.4%,
`dir_cos` 0.963 given lock, i.e. an expected mean near **0.95**. The probe reads
**0.711**, uniformly across distance bins. They agree at n_dist=0 (0.989 vs
~0.99), so the probe is not broken, and it is not geometry.

Remaining candidates: trajectories visit systematically harder cells than
uniform sampling does (near walls, say), or the two measurements condition on
the recall differently — P1 reports `dir_cos` **given lock=goal** plus a
separate lock rate, the probe reports an unconditional cosine.

**The test:** have the probe record P1's decomposition — lock target, lock rate,
`dir_cos` given lock — along the trajectory. That splits the 0.711 into "locks
less often here" versus "points worse when locked", and either answer resolves
it. Until then neither number should be quoted as *the* readout accuracy.

#### What the data actually shows — a path-quality gap, not a memory gap

The Cartesian arm is far more **direct** (`align_true` 0.835) than the polar
frozen arm (0.548): 1.60× optimal path length against 2.07×. Both use `q`
fully. That is a real difference and still confounded by budget (u900 vs u300)
and speed (1.98 vs 1.00) — the matched-budget probe remains the test.

**The open question worth asking** is therefore not "why does the policy ignore
`q`" but **"why is the path only 55% direct when `q` is 98.9% accurate?"** The
per-step breakdown says alignment is *learned within an episode*: `follow_q` by
step index runs 0.373 / 0.514 / 0.581 / 0.671 / 0.732 / 0.770, with
`align_true` tracking it at every index. The policy starts poorly aligned after
a teleport and improves over ~6 steps. That transient, not readout neglect, is
where the path length is going.

---

#### (superseded) original section text follows

Reading the probes for what the policy is *using*, not just how it is spread:

| checkpoint | success | `q_accuracy` | `follow_q` |
|---|---|---|---|
| `p10_pol` u250 | 0.990 | 0.988 | **0.407** |
| `p10_pol_v1` u300 | 1.000 | 0.989 | **0.558** |
| `p10_pol` u300 | 0.698 | **0.993** | **−0.001** |

The third row holds at **every** distractor level, so it is not one condition:
`follow_q` = −0.001 / −0.031 / −0.018 / −0.021 / +0.015 for n_dist 0→10, against
`q_accuracy` 0.993 → 0.813. The same probe also puts mean speed at 1.5864–1.5911,
a **1.003×** modulation — the decoupling of §9.6 holds even more tightly at u300
than the 1.018× measured at u250.

A tempting inference to avoid: *"a policy ignoring q should be insensitive to
distractors, since distractors only corrupt q."* The data does **not** support
it. Success falls 0.698 → 0.641 (×0.918) here, against ×0.901 for the frozen arm
at `follow_q` ≈ 0.5 and ×0.922 for the learned arm at u250 at `follow_q` ≈ 0.41.
All three degrade about equally with distractors regardless of how much they use
`q`, which is itself worth explaining and is not explained here.

The third row is **mode B in its purest form** (`project_nav_tri_failure_modes`:
ignoring a usable readout, policy-limited rather than encoder-limited). Between
u250 and u300 the learned-speed arm **abandoned the Hopfield readout entirely** —
`follow_q` went to zero while `q_accuracy` stayed at 0.993, i.e. a near-perfect
signal left completely on the table — and success fell 0.990 → 0.698 with
`mean_steps` rising to 46.3.

**But the first two rows are the uncomfortable part.** Even the *good*
checkpoints follow `q` only weakly — 0.407 and 0.558 against readouts accurate
to 0.99. **Both polar exploit arms reach high success while largely not using
the memory.** They are navigating by something else: systematic search, wall
following, whatever the trunk has learned from the sensory and prev-action
channels.

That reframes the headline. "1.000 success in 19.6 steps" is a strong
*navigator* and not obviously a strong *memory-follower*, and this project is
about the latter. A success rate that can be achieved without the readout is not
measuring the readout.

**Whether this is polar's doing is NOT yet established.** §9.3's Cartesian
`p9_sq_std` arm reads `follow_q = 0.843` at n_dist=0 against polar's 0.407 and
0.558 — and `mean_steps` 8.05 against 19.6 and 27.2.

But that comparison is **not currently fair**, on two counts, and should not be
cited until it is:

1. **Training budget.** `p9_sq_std` was probed at **u900**; these polar arms at
   **u250–u300**, i.e. roughly a third of the training. `follow_q` may simply
   develop late.
2. **Speed.** `p9_sq_std` ran at a realized ‖a‖ ≈ 1.98 (commanded 8.18, clamped),
   so its 8.05 steps is ~1.46× its own ideal; the frozen polar arm at speed 1.0
   has an ideal of ~10.8 and took 19.6, i.e. ~1.81×. Normalizing shrinks the gap
   substantially without closing it.

**The right test is a matched-budget probe** — re-probe both polar exploit arms
at u900 and compare `follow_q` against `p9_sq_std` at the same update. If the
gap survives that, polar improved the action parameterization while degrading
memory-following, and that trade is the most important open question in phase 2.
If it does not, §9.7's first two rows are simply an early-training reading.

---

**THE METHODOLOGICAL POINT — four reversals in one session, all the same error.**

| # | claim, from N eval points | what happened |
|---|---|---|
| 1 | "freezing speed helps explore" (N=1, u50) | gone by u150 |
| 2 | "the frozen explore arm collapsed" (N=1, u150) | transient |
| 3 | "freezing speed costs 22× on exploit — premature convergence" (N=1, u50) | reached **1.000** by u300 |
| 4 | "the frozen explore arm is genuinely collapsing" (N=3 declining, + falling train reward, u200) | recovered to **0.311** at u300 |

Reversal 4 is the instructive one: **three consecutive declining evals AND a
falling training reward was still not enough.** The series reads
0.207 → 0.274 → 0.125 → 0.072 → 0.311.

Two GPU jobs (`p10_pol_v1_e20/_e50`) were launched on reversal 3's wrong
diagnosis and cancelled.

Operating rule for this project, stated so it can be checked rather than
remembered: **no directional claim from fewer than four eval points spanning
≥200 updates.** Report the series, not the latest value. The prior existed
(`project_hopfield_nav_explore_exploit`, the peak-calling trap) and restating it
was not enough — every one of these was made after writing that warning down.

Current standing (both arms healthy, neither converged):

`p10_pol_v1` (frozen speed, exploit) through u550 — the series, per the rule
above:

| u | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| success | .03 | .21 | .90 | .51 | .91 | 1.00 | .49 | .99 | 1.00 | 1.00 | .98 |
| steps | 10 | 48 | 66 | 47 | 34 | 19.6 | 43 | 16.0 | 16.3 | **15.4** | 35 |

**Its level is ~1.000 success at 15–16 steps** (u400–500), not the 19.6 quoted
earlier from the single u300 point — the series is better than the point.
Against an ideal of ~10.8 steps at speed 1.0 that is **1.43× optimal**. The
dips at u200/u350/u550 are the oscillation, not a trend; κ meanwhile climbs
25.8 → 39.4 → 33.3 → 40.7 → 77.7, itself non-monotone.

| arm | update | metric |
|---|---|---|
| `p10_e_pol` learned | u550 | cov 0.378, **cps 0.755** (billiard 0.775) |
| `p10_e_pol_v1` frozen | u300 | cov 0.311, cps 0.621 |
| `p10_pol` learned | u300 | success 0.698 (peak 0.990 at u250) |
| `p10_pol_v1` frozen | u400–500 | **success ~1.000 at 15–16 steps** |

---

### 9.8 FINAL — the delivered models

Both completed arms ran to their scheduled ends cleanly.

#### `p10_pol_v1` — exploit, speed frozen at 1.0 — **2000/2000**

Final eval, 96 trials per level:

| | success | mean_steps | vs optimum |
|---|---|---|---|
| 0 distractors | **1.000** (96/96) | **10.95** | **1.10×** |
| 10 distractors | 0.958 (92/96) | 13.64 | 1.37× |

Still improving at the end — last four evals 11.60 / 11.47 / 11.30 / 10.95 —
so 2000 updates did not exhaust it. **The phase-2 best exploit model.** For
contrast the Cartesian `p9_sq_std` took 8.05 steps but at speed 1.98, i.e.
1.60× its own optimum against this one's 1.10×.

Behaviour probe on the final checkpoint (21365319):

| n_dist | q_accuracy | follow_q | align_true | success | steps |
|---|---|---|---|---|---|
| 0 | 0.989 | **0.911** | 0.908 | 1.000 | 11.68 |
| 1 | 0.927 | 0.852 | 0.824 | 0.990 | 11.33 |
| 3 | 0.833 | 0.803 | 0.678 | 0.969 | 11.82 |
| 5 | 0.725 | 0.697 | 0.502 | 0.948 | 11.98 |
| 10 | **0.450** | **0.630** | 0.257 | 0.901 | 17.42 |

**`follow_q` 0.911 EXCEEDS the Cartesian baseline's 0.843.** The polar model
follows the readout *better*, not worse — §9.7's original worry is not merely
retracted but inverted. The arm's own progression is 0.558 (u300) → 0.819
(u900) → **0.911** (u2000): it was always a training-budget reading.

At ten distractors `follow_q` (0.630) **exceeds** `q_accuracy` (0.450) — more
committed to the readout than the readout deserves, with `align_true`
collapsing to 0.257 as a direct result. The sharpest mode-A evidence in the
dataset. Steps are flat at ~11.3–12.0 through *five* distractors and only break
at ten, so the model is effectively distractor-immune up to five.

**Note the two success numbers disagree**: the training eval says 0.958 at ten
distractors (92/96), the probe says 0.901 (192 trials). Same checkpoint, same
envs, both `deterministic=True`, comparable start distances — the difference is
which starts were drawn and how many. They are ~1.9 sd apart (p≈0.06). The
honest statement is **0.90–0.96, bracketed**, with the probe's the more precise.
They are independent estimates and were quoted interchangeably at first, which
they should not have been.

#### `p10_e_pol` — explore, learned speed — **1500/1500**

Final five evals: cps 0.758 / 0.776 / 0.711 / 0.769 / 0.721 → **~0.75, coverage
~0.37**. Against the ladder (0.36 uniform random walk, 0.775 billiard, 0.955
lawnmower) this is **at the billiard ceiling**. The next rung is a different
behaviour class — systematic sweeping, which needs remembering where it has
been — not something more of this objective reaches.

#### 9.8.1 CORRECTION — the near-goal κ gradient GROWS with training

§9.6 said it decays with policy sharpness; §9.7.6 already flagged that as
refuted. The final checkpoint settles the direction:

| checkpoint | κ | d0–2 | d8+ | ratio |
|---|---|---|---|---|
| u300 | ≈38 | 10.06° | 9.74° | 1.03× |
| u900 | ≈93 | 7.67° | 6.20° | 1.24× |
| **u2000** | — | **7.76°** | **4.71°** | **1.65×** |

Same arm, monotone across 1700 updates. **The policy LEARNS to be more
directionally uncertain near the goal** — 7.8° there against 4.7° far out. That
is §9.1's prediction satisfied, and it explains why every "sharpness suppresses
it" story failed: u300 was simply too early. The far-field 4.7° says the policy
is extremely committed when it knows where it is going and deliberately hedges
on final approach — a strategy, not the pathology read into it earlier.

#### 9.8.2 The caveat that applies to all of §9.6–9.8

**Everything above is measured on the `recorded` split** — the run's own
`base_val`, which it was scored against at every eval. Never trained on, but not
a fresh draw, and the probe could not build anything else until `--split` was
added on 2026-08-27. Nothing here has been re-run on a minted set.

Before any of these numbers are treated as the models' performance rather than
their validation-set performance, re-probe with
`--split place=held_out` (which sets all three traits to `held_out`). The
grammar and minting path are shared with `eval_all`, so the two are directly
comparable.

---

### 9.9 P11 / P12 — why 550 updates, and what "learned speed" was worth

Eight runs, all 2000/2000 (explore arms 1500/1500), all against `p10_pol_v1`.
Write-ups are published as charts; links in §0.

#### P11 — three shaping knobs on the convergence question

Stability reported as two threshold-free facts: the update at which a run first
reaches 0.85 success, and the minimum it touches afterward. (An earlier draft
counted "collapses below 0.55" and applied it inconsistently — the control's
count swept in pre-breakthrough updates while the treatments' did not, which
flattered the treatments.)

| arm | first ≥0.85 | min after | locks | final steps | final @ 10 |
|---|---|---|---|---|---|
| control `p10_pol_v1` | u150 | 0.490 | u600 | 10.95 | 0.958 |
| **`p11_cur`** curriculum | u300 | **0.979** | **u400** | 11.40 | 0.948 |
| `p11_tp` time_penalty 0.02 | **u100** | 0.365 | u500 | 11.58 | 0.875 |
| `p11_cur_tp` both | u200 | 0.646 | u450 | 12.09 | 0.958 |

**None of these knobs changes what the policy becomes** — all four finish
within 10.95–12.09 steps. They change only how turbulently it gets there.

- **Curriculum wins on stability outright.** Never falls below 0.979 after its
  breakthrough, against the control's 0.490. Costs a later start (u300 vs u150).
- **Cheaper failure is the largest single-knob effect and the wrong one.**
  24× more success at u50 (0.760 against 0.031) — but at 59 steps per goal
  against the control's 10. It is stumbling onto targets inside a 200-step
  budget, not steering. It then has the *worst* stability of the four. **Early
  goal discovery is not the convergence bottleneck.**
- **The cross is intermediate, not additive**, on both stability measures, and
  worst of all four on final efficiency. Two knobs pushing the same quantity in
  opposite directions.

`p11_eps` is a **no-op** and was cancelled: `exploit.py` hard-codes
`epsilon=0.0`, so `--epsilon_explore` on an exploit-only schedule is accepted,
echoed and discarded. It produced numbers bit-identical to the control. This
also retracts a claim made earlier in the analysis — ε-annealing was cited as a
possible contributor to the u150–u250 swings, and **there was never any ε in
any exploit run in this phase**. The trainer now warns at startup.

#### P12 — the control P10 needed and did not have

`p10_pol_v1` is pinned at 1.0; `p10_pol` may range over [0.5, 2] and learns
~1.8. Nothing sat in between, so "learned beats frozen" confounded *choosing* a
speed with *being allowed to go fast*.

| speed setting | chooses | final steps | directness | min after 0.85 |
|---|---|---|---|---|
| pinned at 1.0 | — | 10.95 | 1.043× | 0.490 |
| **learned [0.5, 1.0]** | 0.94 | 11.79 | 1.081× | **0.844** |
| learned [0.5, 1.0] + treatments | 0.975 | 11.25 | 1.042× | 0.646 |
| learned [0.5, 2.0] | ~1.8 | **7.17** | 1.076× | 0.490 |

**Choosing does not buy steps.** Capped at 1.0 the policy takes 11.79 steps
against the pinned arm's 10.95 — slightly worse. Every setting walks a path
within 4–8% of the straight line. **Step count tracks the speed cap**, and
§9.5's headline (7.17 against 10.95) was permission to move 80% faster, not a
better navigator.

**Choosing does buy stability, unpredicted.** Same cap, same everything else,
and the minimum after breakthrough is 0.844 against 0.490. A candidate reading:
an adjustable step size gives PPO a low-risk way to absorb a bad update, where a
pinned policy can only change heading. Suggested by this run, not tested by it.

**Consequence for §9.5.** The frozen-vs-learned framing used throughout P10 was
measuring the bound, not the freedom, and freezing is just the degenerate case
of a tight bound. The genuine frozen-speed findings that survive are: it costs
nothing on explore (§9.8), and it is not the cause of the near-goal κ gradient
(§9.7.6).

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

---

## 12. P13 — per-episode failure forensics

> **Summary of §12 and §13.** Six phase-2 exploit models, 3456 observational
> episodes plus 30 paired intervention arms.
>
> 1. **The exploiter has exactly one failure mode, and it is memory
>    interference.** Every checkpoint is 192/192 with no distractors. All 213
>    failures are at ten. A goal-only memory rescues 155 of 155 failures and
>    breaks none.
> 2. **The goal pattern never loses the recall** — margin > 0 on 101/101
>    failures and 667/667 successes. Interference is a degraded *level*
>    (median 0.60 against 0.93 for matched successes), a blend the goal
>    dominates, and the contamination is enough to rotate the direction into a
>    miss.
> 3. **The policy is not at fault.** It follows the recalled direction
>    faithfully — `follow_q_fail` 0.435 against `align_true_fail` −0.176 — and
>    scrambling that direction while holding magnitude and smoothness fixed
>    collapses success from 0.87 to 0.26.
> 4. **Failures are lost on the approach, not at the doorstep.** A clean
>    readout beyond two cells rescues 90%; within two cells, 37%; and *no*
>    episode is rescued by the near fix alone. The near-goal readout collapse
>    is real but secondary.
> 5. **1500 further updates change nothing** — neither how often nor why it
>    fails.
>
> One correction to an earlier reading of ours: the near-goal widening of the
> heading distribution is **not** the policy sensing an unreliable readout
> (§12.7). It happens just as much with a perfect readout.

Method spec: `docs/EXPLOIT_DIAGNOSTIC.md`. Tool: `analysis/nav_tri/exploit_diag.py`
(rollouts → per-episode table → `derive()`) and `analysis/nav_tri/exploit_report.py`
(table → HTML). Launcher: `hopfield_nav/run_exploit_diag.sh`. Report:
https://claude.ai/code/artifact/790509c2-253b-423e-a46d-e039ac004665

Motivation: §9's numbers come from `behavior_probe`, which pools every step of
every episode behind a fail mask. `follow_q_fail = 0.43` reads the same whether
every failure was mildly bad or half were catastrophic, and the mixture is what
decides what to fix. This labels each failure on two independent axes —
**symptom** (what the trajectory did, from `d(t)` alone) and **cause** (which
link of memory → readout → policy → execution broke, from the cosines against a
distance-matched success baseline).

Runs: nine cells at 10 distractors and nine at 0, `recorded` split, 32 trials ×
6 envs = 192 episodes each, 3456 episodes total. Jobs 21616671 / 21616672 /
21616673. **Checkpoints must be grouped by action head** — the agent is built
once from the first config, so `freeze_speed` and `min/max_action_norm` have to
agree. The eight P2 runs are three incompatible groups, and the tool now
refuses a mixed list rather than silently reading a [0.5, 1] policy as [0.5, 2].

### 12.1 Failures exist only under memory competition

Every checkpoint is **192/192 at zero distractors**. All 213 failures across the
nine 10-distractor cells. Success at 10 sits at 0.849–0.917 with no arm
separating: p10_pol_v1 0.875, p11_cur 0.870, p11_tp 0.875, p11_cur_tp 0.854,
p12_lo 0.870, p12_lo_curtp 0.849.

### 12.2 The dominant cause is recall contamination, not decode failure

Pooled over the four frozen-speed arms (101 failures):

| symptom | ro_memory | ro_decode | policy | execution | row |
|---|---|---|---|---|---|
| straddled | 13 | 5 | 7 | 0 | 25 |
| approached_left | 7 | 0 | 7 | 6 | 20 |
| blocked | 30 | 0 | 1 | 0 | 31 |
| timeout_converging | 10 | 0 | 4 | 0 | 14 |
| never_approached | 1 | 0 | 10 | 0 | 11 |
| **total** | **61** | **5** | **29** | **6** | **101** |

**The goal pattern wins the recall competition on 101/101 failures and 667/667
successes.** It never loses. What separates a failure is the *level*: margin
median 0.60 against 0.93 for successes at the same distance, distributions that
barely overlap (success p10 0.836 ≈ failure p90 0.834). The recall is a blend
the goal merely dominates, and the contamination rotates the direction.

This corrected a real error in the tool. The first version split
`readout_memory` from `readout_decode` on an absolute floor (`margin < 0.02`),
which asks "did a distractor win" — the answer is never — and so labelled
~65/101 as decode errors. Switching to a **matched deficit** flipped it to 61
memory / 5 decode. The whole tool is built on matched baselines and this was the
one place an absolute cutoff survived; the doc now records why.

Symptom and cause are genuinely independent, which is the case for both axes:
**`blocked` is 30/31 contamination** (the agent drives into geometry because the
recalled heading is wrong) while **`never_approached` is 10/11 policy**. Pooled
cosines average those together.

### 12.3 Near-goal failure is a heading problem, not a geometry problem

Threshold-free, pooled over the frozen arms (n=101, R=1.0):

- 37% reached inside R+1 — touched the doorstep and did not enter
- 54% reached inside R+2
- 36% never got within R+3

Of the 37 that touched the doorstep, **5 had clip_frac > 0.5 and the median was
0.000** — no wall contact. So the near-goal failure is heading error, as §2 of
the spec predicted; it is not the step-length straddle, which the geometry at
R=1.0 mostly forbids (the ball is twice the minimum step).

`clip_frac` is strongly bimodal — 50 episodes below 0.1, 37 above 0.9, four in
between — so the 0.5 cut for `blocked` sits in an empty gap rather than through
a mode.

**Caveat, and the threshold machinery caught it.** The *leading* symptom is not
stable across the (band width, dwell) grid for three of four arms; `straddled`
ranges 3–9 depending on the setting. The near-goal claim above is stated in
threshold-free terms for that reason. "Straddled is the modal failure" is not a
claim this data supports.

**And read this section next to §13.3.** These are the failures that *end* near
the goal, which is not the same as failures *caused* near the goal. The
intervention shows the damage is done on the approach: fixing the readout only
beyond two cells rescues 90% of failures, fixing it only within two cells
rescues 37%, and no episode is rescued by the near fix alone.

### 12.4 Mode A, and the readout degrades near the goal

Pooled at 10 distractors for p10_pol_v1: `follow_q_fail` 0.435 against
`align_true_fail` **−0.176**. The policy follows the readout faithfully and the
readout points the wrong way — the agent moves *away* from the goal on average.
Read alone, 0.435 looks like a mediocre policy; the matched baseline is what
flips the interpretation. This is the §9.7 trap in its sharpest form yet.

On *successes* — so not a failure artifact — `q_accuracy` by distance runs
0.634 (d ∈ [1,2)) → 0.832 → 0.929 → 0.953 → 0.986 → 0.991 → 0.993. The readout
is near-perfect far out and degrades sharply inside two cells. Heading circular
sd widens over the same range, 0.095 far out to 0.155 near the goal.

**That is NOT the policy knowing the readout is unreliable — retracted, see
§12.7.** The widening happens just as much with no distractors at all.

`explained_frac = 1.000` throughout, so the linear-recall decomposition
describes what the network actually computed and the margin numbers are safe to
read.

### 12.5 Training does not change what the model fails at

`p11_cur` at 10 distractors, u500 → u2000 (4 points; series, not a fit):

| update | success | fails | straddled | blocked | ro_memory | policy |
|---|---|---|---|---|---|---|
| 500 | 0.917 | 16 | 3 | 7 | 11 | 3 |
| 1000 | 0.885 | 22 | 4 | 8 | 16 | 5 |
| 1500 | 0.896 | 20 | 3 | 8 | 15 | 5 |
| 2000 | 0.870 | 25 | 7 | 8 | 18 | 4 |

The success spread is 0.047 against a binomial SE of 0.023 at n=192 — about two
SE, and non-monotonic. **No directional claim.** As a fraction of failures the
mixture is flat (ro_memory 0.69 / 0.73 / 0.75 / 0.72), so 1500 further updates
change neither how often nor why it fails. Consistent with §9.9: none of the
P11/P12 knobs moves the endpoint.

### 12.6 What this points at

The main lever is **encoder/memory separation at 10 distractors** — the policy
does follow the signal it is given, and no config change fixes a contaminated
recall.

But there is a **second, policy-side lever**, which an earlier draft of this
section wrongly closed off: the heading spread does *not* widen where the
readout breaks (§12.7.1). Inside two cells the readout falls to 73% of its
clean accuracy and the spread response is 1.06 — nil. Making the policy widen
there is a live option, not something already solved.

**The intervention that was listed here as the top next step has been run — see
§13.** It confirmed the readout attribution (155/155 failures rescued by a
clean memory, 0 broken) and added the finding that the damage is done on the
approach rather than at the doorstep. What remains:

1. **Separate clean memory from clean decode.** `A_clean` and `B_oracle` both
   saturate at 1.000, so this difficulty cannot tell them apart. Needs a harder
   regime — more distractors, or a smaller capture radius — where neither hits
   the ceiling.
2. **Forced widening** (§12.7.1) — clamp κ down near the goal at eval and see
   whether a policy that widened where the readout breaks would do better. The
   one policy-side lever the interventions did not test.
3. **Re-probe on a minted split.** Every number in §12 and §13 is `recorded`.

### 12.7 Retraction — the policy does not widen because it distrusts the readout

§12.4 first read the near-goal rise in heading spread as calibration: the
readout degrades inside two cells, the spread widens there, so the policy
"knows". **That is wrong**, and the zero-distractor cells already in the same
run falsify it.

At 0 distractors the readout never degrades — `q_accuracy` is 0.970 at d<2
against 0.992 far out, a near/far ratio of 0.99 — and the policy **still**
widens near the goal in every one of the six runs:

| run | near/far spread, d=0 | near/far spread, d=10 |
|---|---|---|
| p10_pol_v1 | 1.47 | 1.61 |
| p11_cur | 1.87 | 2.30 |
| p11_tp | 2.40 | 2.42 |
| p11_cur_tp | 1.94 | 1.92 |
| p12_lo | 1.82 | 1.98 |
| p12_lo_curtp | 1.78 | 2.37 |

The near-goal widening is a function of DISTANCE, not of signal quality. The
confound §12.4 missed is that on a direct trajectory distance is collinear with
time-in-episode, and the whole effect survives with the signal held perfect.

### 12.7.1 What survives: a confidence signal that is silent where it matters

Comparing the same model against itself at the same distance, 10 distractors
over 0, mean across the six runs:

| d ≥ | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 13 |
|---|---|---|---|---|---|---|---|---|
| spread ratio | **1.06** | 1.37 | 1.57 | **1.63** | 1.41 | 1.13 | 1.08 | 0.99 |
| q_accuracy ratio | **0.73** | 0.85 | 0.90 | 0.92 | 0.97 | 1.00 | 1.00 | 1.00 |

The policy *does* widen under distractors — so a confidence signal exists — but
the two rows are **anti-aligned at the near bin**. Where the readout is most
degraded (inside two cells, down to 73% of its clean accuracy) the spread
response is essentially nil at 1.06. The response peaks at d∈[3,5), where the
readout has barely moved.

So the confidence signal goes silent in exactly the band where the readout
breaks — and that is the band where failures happen (§12.3: 37% of failures
reach inside R+1).

Two controls, both of which strengthen the result rather than weaken it:

- **Start distance is matched**: 10.95 at d=0 against 10.68–10.89 at d=10, so
  the comparison is not confounded by geometry.
- **d=10 successes take MORE steps** (e.g. p11_tp 17.6 vs 12.0). If that extra
  dithering is spent near the goal it should have manufactured elevation in the
  near bin; the ratio is 1.06 regardless.

**Alternative not ruled out.** κ is computed from the trunk's direction-vector
norm (`polar_head.PolarHead.forward`), so the mid-range elevation could be
passive magnitude propagation — ‖q‖ falls under interference and κ follows —
rather than a learned calibration. Behaviourally the conclusion is the same.
The §13 arms do **not** separate them: `B_oracle` preserves ‖q‖ but also
changes the trunk input, so κ could move either way. The clean test is
`behavior_probe --q_scale`, which already exists — it multiplies ‖q‖ while
preserving direction, so if κ tracks it the response is passive.

**Consequence for §12.6.** The read that "the policy already widens where the
signal is weak" is withdrawn. It does not. Widening the heading distribution
near the goal under interference is now a live policy-side lever, not something
already solved — alongside the encoder work §12.6 points at.

---

## 13. P14 — interventions: the causal test

Spec `docs/EXPLOIT_DIAGNOSTIC.md` §7. Jobs 21618482 (frozen arms) / 21618483
(learned speed), six runs at u2000, 10 distractors, six arms each, 155 baseline
failures. Report: https://claude.ai/code/artifact/790509c2-253b-423e-a46d-e039ac004665

§12 is conditioned on a failure having happened, so it is correlational
throughout. These re-roll the same episode with one factor changed. **Pairing is
exact** — `deterministic=True` actions, fixed starts, RNG re-seeded per arm;
verified by two independent baseline runs producing bit-identical rows — so a
flip is an effect, not a sample, and no matched-baseline machinery is needed.

### 13.1 Results

| arm | rescued / 155 | broken |
|---|---|---|
| `A_clean` — goal-only memory everywhere | **155 (100%)** | 0 |
| `B_oracle` — true bearing, magnitude kept | **155 (100%)** | 0 |
| `C_far` — goal-only memory beyond 2 cells | 139 (90%) | 1 |
| `C_near` — goal-only memory within 2 cells | 57 (37%) | 0 |
| `D_placebo` — fixed random rotation | 16 (10%) | **709** |

**The control passes emphatically.** `D_placebo` holds magnitude, smoothness and
within-episode consistency fixed and changes only the target: success falls from
~0.87 to ~0.26. So `B_oracle`'s gain is goal-specificity, not "any consistent
signal". Broken episodes end a median 6–7 cells out with `clip_frac` 0.95 and
`q_acc` −0.45 — driving confidently into a wall, which is what a fixed wrong
bearing predicts.

### 13.2 The symptom taxonomy is causally validated

Share of each baseline symptom rescued, pooled:

| symptom | n | `C_near` | `C_far` |
|---|---|---|---|
| straddled | 42 | **98%** | 100% |
| approached_left | 28 | 36% | 89% |
| blocked | 46 | **0%** | 78% |
| timeout_converging | 21 | 24% | 86% |
| never_approached | 18 | 6% | 100% |

`C_near` rescues 98% of straddles and **0% of blocked**. The spatial arms were
defined by a distance gate and the symptoms by the shape of `d(t)`, with no
knowledge of each other — the agreement is a check, not a construction. A
near-band fix repairs exactly the near-band symptom.

### 13.3 The damage is done on the approach, not at the doorstep

| rescued by | n | share |
|---|---|---|
| both arms | 57 | 37% |
| `C_near` only | **0** | **0%** |
| `C_far` only | 82 | 53% |
| neither — needs clean everywhere | 16 | 10% |

**Zero episodes are rescued only by `C_near`**, so the rescue is strictly
nested: everything a clean near-goal readout saves, a clean approach readout
also saves, plus 82 more. Arrive in good shape and you get in even with a
degraded terminal readout; a clean terminal readout cannot save an episode that
never arrives. The near-goal `q_accuracy` collapse (0.634, §12.4) is real but
**secondary** — it accounts for the 10% that need clean everywhere.

### 13.4 What this settles, and what it does not

Settled: contamination is causally sufficient AND necessary for every failure
measured — 155/155 rescued, 0 broken — and the policy genuinely uses the
direction of `q`, since scrambling it is catastrophic. §12.2's attribution
survives the test that could have falsified it.

**Not news, and §12 overstated it.** Distractors here are **memory-only**:
`sample_distractors` draws encoded patterns from grid positions *outside* the
test env, never physical obstacles. So `A_clean` reproduces the
`n_distractors=0` condition, which already scores 192/192 — its recovery to
1.000 is close to a consistency check. What it adds is pairing: the
observational `d=0` vs `d=10` comparison is unpaired, because drawing
distractors consumes RNG and shifts every start. The load-bearing arms are
`C` and `D`.

Still open: `B_oracle` and `A_clean` both saturate at 1.000, so they cannot be
separated — whether a clean *decode* would beat a clean *memory* is untestable
at this difficulty. It needs a harder regime (more distractors, or a smaller
capture radius) where neither saturates.

### 13.5 What an approach failure actually looks like

§13.3 says the damage is done on the approach but not what the agent *does*.
The **motion axis** (spec §2.1) answers it: `straightness` is the mean cosine
between consecutive realised displacements, `edge_frac` the share of steps on a
perimeter cell, `wander` the path travelled per cell of ground gained.

**Failures come in exactly two motions and nothing else.** Pooled over 155
baseline failures on six runs:

| motion | n | share | straightness | edge | path/gain | d_min |
|---|---|---|---|---|---|---|
| `pinned` | 61 | 39% | +0.59 | **0.94** | 10 | 4.00 |
| `looping` | 94 | 61% | +0.70 | **0.00** | 23 | 1.87 |

Zero `committed`, zero `meandering`, zero `oscillating`. The two classes
separate cleanly on wall contact (0.94 against 0.00) and on how close they get
(4.0 cells against 1.9).

So: **two fifths are driven into the perimeter and held there; three fifths
orbit.** The orbiting ones hold a consistent turn — straightness +0.70 — and
travel 23 cells of path per cell of ground gained. It is a smooth circle, not a
random walk and not an oscillation.

**There are no interior obstacles in this world.** The arena is an open box and
the only thing that shortens a step is the boundary clip, so `pinned` means
held against the *perimeter*. That is the same basin the explore work found.

### 13.5.1 The motion axis carries information the symptom axis did not

| | straddled | approached_left | blocked | timeout_conv | never_appr |
|---|---|---|---|---|---|
| `pinned` | 7 | 8 | 46 | 0 | 0 |
| `looping` | 35 | 20 | 0 | 21 | 18 |

`blocked` is 100% `pinned` **by construction** — both rules test
`clip_frac > 0.5` — so that cell is not evidence. The informative rows are the
others: `never_approached` and `timeout_converging` are **100% looping**, and
`straddled` is 35/42. Those three symptoms described the shape of `d(t)` and
carried no information about the motion; they are all orbits.

Split by which spatial arm rescues them:

| | n | motion | edge | d_min |
|---|---|---|---|---|
| far-only (approach damage) | 82 | looping 41 / pinned 41 | 0.74 | 4.00 |
| both arms | 57 | looping 50 / pinned 7 | 0.01 | 1.41 |
| neither | 16 | pinned 13 / looping 3 | 0.93 | 6.32 |

The episodes a near-goal fix can also save are orbits at the doorstep — 1.41
cells out, zero wall contact. The ones no partial fix saves are pinned far out
at 6.3 cells with edge 0.93.

### 13.5.2 The placebo is a positive control for the classifier

`D_placebo` gives the agent a fixed random bearing, so mechanically it should
drive straight and hit the perimeter. It does: over 848 failures,
`pinned` 651 / `committed` 170 / `looping` 27, straightness **+0.96**, edge
**0.95**.

**`committed` appears only under the placebo** — never once in a real failure.
So the classifier can detect straight-line motion when it is there, and the
finding that no real failure is a confident drive in a wrong direction is a
measurement rather than a gap in the labels.

Note the observational cells in §12 predate this axis and carry no motion
fields; the intervention `baseline` arms cover the same six runs at ten
distractors, which is where every failure is. The `p11_cur` training sweep has
not been re-run with motion.

---

## 14. P15 — the loops are spurious attractors in the readout

Tool: `analysis/nav_tri/readout_field.py`. Launcher:
`hopfield_nav/run_readout_field.sh`. Job 21620981. Report:
https://claude.ai/code/artifact/3e986fa7-72d0-4fb0-b9b6-09358ee75805

§13.5 found that 61% of failures are loops — a tight ball, radius 1.41 ± 0.18,
never centred on the goal, held for the whole horizon — and that separately
trained policies land on the *same* phantom. That pointed at the readout rather
than any policy, but only circumstantially.

`q(x)` is a vector field over cells, and one memory can be evaluated at every
cell at once (`Hopfield.recall_batch`). So the hypothesis is testable with **no
policy and no rollouts**: evaluate `q` everywhere, then integrate it from all
400 cells using the env's own update rule — fixed step, snapped sampling,
clipped at the arena — and see where the flow ends up.

### 14.1 A sink in `q` is necessary for failure

p10_pol_v1, 192 memories, joined to the rollout outcomes on `(env, trial)`.
The join is only meaningful if the memories match, so the RNG alignment is
**checked, not assumed**: the field's drawn start position agrees with the
rollout's on 192/192.

| | agent failed | agent reached | failure rate |
|---|---|---|---|
| field has a sink | **24** | 13 | 65% |
| field is clean | **0** | 155 | **0%** |

**Zero failures in 155 episodes whose field was clean.** Every one of the 24
failures had a sink. A sink is not sufficient — the agent escaped 13 of 37,
since it is not a pure `q`-follower and has sensory input and RNN state — but it
is present every single time the agent misses.

The share of cells flowing to the goal separates almost completely: median 1.00
on reached episodes, **0.09** on failed ones.

### 14.2 The agent orbits where the field says it will

| env / trial | agent orbited | field sink | apart |
|---|---|---|---|
| 0 / 1 | (14.9, 6.5) | (14.9, 6.5) | **0.06** |
| 0 / 28 | (17.4, 0.0) | (17.4, 0.0) | **0.06** |
| 5 / 7 | (6.6, 0.0) | (5.4, 0.0) | 1.15 |
| 1 / 12 | (8.7, 4.7) | (8.6, 7.3) | 2.59 |

Median agreement **0.61 cells**; chance for points placed at random in a 20×20
arena is ~7.7. Nothing about the agent went into computing the sink.

### 14.3 It is the memory, not the goal

The same goal is clean under most distractor draws and trapped under others —
for goal (17,4), trials 0/3/4/5/6/7 flow 100% to the goal with no sink while
trials 1/2 have 3–4% flow and a sink with a basin of ~385/400. Overall 37/192
memories carry a sink.

So the attractor is a property of **the particular set of stored patterns**, not
of the goal or the arena. That is exactly consistent with §13.2: interference
degrades the recall margin without the goal ever losing the competition, and a
contaminated blend does not need to flip the winner to bend the direction field
into a vortex.

### 14.4 What this changes

This is a mechanism, and it makes the earlier results one story:

- §12.2's contamination is causal *because* it folds `q` into a sink.
- §13.3's "damage done on the approach" follows: the basin covers most of the
  arena, so the trajectory is captured early and far out.
- §13.5's loops are the orbits of that sink, and its radius (~1.4 cells) is the
  ball the agent circles.

**A failure is now predictable before running the agent.** `readout_field.py`
scores a memory in seconds with no rollout, and no clean-field memory has ever
produced a failure. That makes it a screening tool for encoder work: the target
is not "make the goal win" — it already always wins — but **"make draws that do
not fold the field"**.

Open: only p10_pol_v1 has been mapped. The field is policy-independent, so the
sinks should be identical for every arm sharing this encoder and seed — worth
confirming on one other run, which would also re-derive §13.5.3's cross-model
agreement from the field rather than from trajectories.

---

## 15. P16 — a saturating Hopfield (`p16_sat`)

Job 21622130, `VARIANT=p16_sat sbatch hopfield_nav/run_nav_p2.sh`.

§14 found that every exploit failure has a spurious sink in `q(x)`, and the
reason such a sink can exist at all is that **the recall is not an attractor**.
The update is `tanh(beta · W x)` with an argument around 1e-4, so `tanh` sits in
its linear region and retrieval is a weighted *blend* of the stored patterns. A
blend can point anywhere — including into a vortex — without any pattern
winning, which is exactly what §13.2 measured (the goal always wins, the margin
is merely degraded).

This arm saturates both nonlinearities:

| knob | value | what it does |
|---|---|---|
| `--encoder_gain` | 300 | the code is `normalize(tanh(g·z))`, so this binarises the embedding. **Shape, not magnitude** — the normalize comes after. Measured: 88.5% of components land within 5% of ±1/√D against 5.2% at the default, norm unchanged at 1.000. |
| `--hopfield_beta` | 1e6 | the recall argument reaches ~1e2, so `tanh` saturates and retrieval becomes sign-thresholded rather than a blend. |

Encoder is v35's (`encoders/run_20260422_185816/encoder_best.pt`), which shares
lambdas (11 12 13) and size (20) with the P2 world, so the scaffold is
unchanged. Everything else is byte-identical to `p10_pol_v1` — the frozen-speed
exploiter that anchors §12–15 — so the diagnostics apply unchanged.

### 15.1 Three things this required fixing

**Neither gain was settable as named.** `--encoder_gain` only ever wrote
`cfg.hopfield.beta` and never reached the encoder, and there was **no
`--hopfield_beta` flag at all** — beta defaulted to the encoder gain and had no
independent route. Both added; an explicit `--encoder_gain` now applies to the
model as well, while the default path is unchanged so no existing run moves.

**The v35 encoder carries two different gains, and always has.** The
checkpoint's top-level `gain` is **3.699** (which became beta) and its
model-config `gain` is **5.0** (what `forward` actually encodes with). Every run
on this encoder has encoded at 5.0 while setting beta to 3.699. Recorded in
`encoder_io.load_encoder`.

**A silent-wrong-config bug in the variant.** Written first as
`ENCODER=${ENCODER:-...}`, which is a no-op because the fixed block at the top
of `run_nav_p2.sh` has already assigned `ENCODER` — the arm would have trained
on the P2 encoder while claiming v35's. Now set unconditionally, and the startup
banner echoes the encoder and both gains so the config is confirmed from two
independent points in the stack rather than assumed.

### 15.2 Caveat on what this can conclude

The arm moves **three** things at once against `p10_pol_v1` — encoder, encoder
gain, and beta. It is a "does saturation help at all" probe, not a
single-factor test. If it helps, the factors want separating; if it does not,
that is informative regardless since the mechanism predicted it should.

The sharp check is not the success rate but `readout_field.py`: a saturating
recall should not be able to produce a smeared blend, so **the sinks should be
gone or far rarer**. That is measurable on this checkpoint with no rollouts.

### 15.3 Which knob mattered — and the mechanism story was wrong twice

The field needs no policy, so both gains can be swept on **existing**
checkpoints in minutes (`readout_field.py --encoder_gain / --hopfield_beta`).
That turned "wait for a training run" into a factor grid. 192 memories per cell,
same seed, same envs.

Beta is the P2 encoder's own 5.0 on the P2 rows and v35's 3.699 on the v35
rows -- each encoder's default, since beta defaults to the encoder gain.

| encoder | enc gain | beta | trapped | basin mean | worst |
|---|---|---|---|---|---|
| P2 | 5.0 | 5.0 | 37/192 (19%) | 0.848 | 0.005 |
| P2 | 300 | 5.0 | **2/192 (1%)** | 0.991 | 0.035 |
| P2 | 5.0 | 1e6 | **62/192 (32%)** | 0.735 | 0.000 |
| P2 | 300 | 1e6 | 2/192 (1%) | 0.991 | 0.035 |
| v35 | 5.0 | 3.70 | 25/192 (13%) | 0.898 | 0.013 |
| v35 | 300 | 3.70 | 3/192 (2%) | 0.999 | 0.757 |
| v35 | 300 | 1e6 | 2/192 (1%) | 0.999 | 0.948 |

**`encoder_gain` is the whole fix.** Binarising the code takes 37→2 on the P2
encoder and 25→3 on v35. The encoder swap on its own accounts for only
19%→13%.

**`hopfield_beta` does nothing once the code is binary, and is harmful when it
is not.** The controlled 2×2 on one encoder:

| P2 code | beta 5.0 | beta 1e6 |
|---|---|---|
| smooth (gain 5) | 37 | 62 (**+25**) |
| binary (gain 300) | 2 | 2 (**+0**) |

### CORRECTED — beta does change the field, just not the trapping

An earlier draft of this section read the two binary rows as "identical to three
decimals" and concluded beta does nothing. **Jack pushed back, and he was
right.** Comparing the stored `q` fields directly rather than the summary:

| | beta 5.0 vs 1e6, binary code |
|---|---|
| cells with cos < 0.999 | **35.6%** |
| minimum cosine | **−1.000** (some cells exactly reversed) |
| \|q\| ratio | 0.25× to 36× |
| median \|q\| where they disagree | 0.197, against 0.266 where they agree |

So beta materially rewrites the field, at a third of all cells, including sign
flips, and *not* only at cells where `q` is negligible. What it does **not**
change is the basin structure: goal basin 0.990924 against 0.990846, one
memory out of 192 with a different basin, and identical sink counts throughout.

The flow is a dynamical system — local direction differences wash out while the
overall gradient still points at the goal. "Same trapping" is a statement about
the integrated outcome, not about `q`.

**And the mechanism I gave for it was wrong.** I argued that with a binary code
`sign(p) ∝ p` makes saturation idempotent. That confuses the *patterns* with the
*recall*: `H = Σᵢ pᵢ⟨pᵢ, x⟩` is a sum of binary vectors and is not itself
binary, so `sign(H) ≠ H/‖H‖`. There is no idempotence, which is exactly why the
field moves.

What survives, stated at the right strength:

* **`encoder_gain` is the fix.** 37→2 on the P2 encoder, 25→3 on v35. Nothing
  else comes close, and this is unaffected by any of the above.
* **`hopfield_beta` changes the field everywhere, and changes the OUTCOME only
  when the code is smooth** — 37→62 there, versus no basin change on a binary
  code. Its effect is conditional on the code's geometry, not absent.

That conditionality is also why `p17_gain` pins `HOPFIELD_BETA=5.0` rather than
letting it follow the encoder gain: beta is a real factor on the field, so
leaving it free would have made the "clean single-factor arm" a two-factor one.

### 15.4 Two retractions

This section's own hypothesis was wrong, twice, and the grid caught both.

**"The recall is a linear blend; saturate it into an attractor."** §15's framing
and the `p16_sat` rationale. False: saturation alone takes trapping from 19% to
32%. Retrieval being a blend was never the problem.

**"The two knobs are complementary — binary patterns plus a sign update."**
The correction after the first retraction. Also false, in its second half: once
the patterns are binary the sign update is a no-op, so `hopfield_beta` is not
part of the fix at all.

What survives is simpler than either: **the geometry of the code is what
matters.** A near-binary code makes patterns drawn from elsewhere in the
scaffold reliably near-orthogonal to the query, so the contaminating terms
`⟨p_distractor, x⟩` stay uniformly small. A smooth code has a fatter overlap
tail, and occasionally a distractor correlates strongly enough to bend the field
into a vortex. That claim is *not yet directly measured* — it is the natural
reading of the grid, and the sharp test is the overlap distribution at gain 5
against gain 300, which no run has produced.

**Consequence for §14.6.** "Make draws that do not fold the field" now has a
concrete lever: raise the encoder gain. It costs nothing at training time and
needs no new encoder.

**Consequence for the P16 arm.** `p16_sat` sits in the best cell of the grid, so
the configuration is right, but `hopfield_beta=1e6` is carrying no weight in it.
A cleaner arm would be the P2 encoder at `encoder_gain=300` alone — one factor,
same field quality (2/192), and directly comparable to `p10_pol_v1`.

---

## 16. P17 / P18 — raising the encoder gain, end to end

Jobs 21623992 (`p17_gain`) and 21625093 (`p18_knee`). Both stopped on the
6-hour wall clock at u1100 and u1050 respectively — not crashes, and neither
completed the 2000-update schedule.

| arm | encoder | enc gain | beta |
|---|---|---|---|
| `p10_pol_v1` | P2 | 5 | 5 |
| `p17_gain` | P2 (unchanged) | **300** | 5 (pinned) |
| `p18_knee` | w49_g100_knee | **300** | 300 |

`p17_gain` is the clean single-factor arm: same encoder as the baseline, one
knob moved. `p18_knee` additionally swaps the encoder, which was already
trained at gain 100 — so 300 sharpens an already fairly binary code rather than
binarising a smooth one.

### 16.1 Converged quality

u550 onward, where all three are past their knee. 96 trials per eval,
`recorded` split.

| arm | @0 | @10 | worst @10 | perfect @10 | steps @10 |
|---|---|---|---|---|---|
| p10_pol_v1 | 0.998 | 0.919 | 0.781 | **0/12** | 16.9 |
| p17_gain | 0.997 | **0.997** | 0.969 | **10/12** | 12.1 |
| p18_knee | **1.000** | 0.995 | **0.990** | 6/11 | **11.6** |

Both new arms' converged **mean** at ten distractors exceeds the baseline's
**best single eval ever** — 0.979, across all 40 of its evals to u2000. The
baseline never once scored 96/96 at ten distractors; `p17_gain` does it in 10
of 12 evals and `p18_knee` in 6 of 11. Both are ~1.4x faster in steps.

### 16.2 They converge very differently

| arm | first eval ≥ 0.95 @10 | worst eval after that |
|---|---|---|
| p10_pol_v1 | u300 | 0.542 |
| p17_gain | **u50** | 0.760 |
| p18_knee | u550 | **0.990** |

`p17_gain` is at 1.000/1.000 by update 50 — where the baseline was at
0.031/0.052 — but wobbles early (0.760 at u150). `p18_knee` climbs slowly and
almost linearly (0.125 at u50, 0.583 at u450, 0.917 at u500) and is then the
most stable arm in the whole phase: **its worst eval after u550 is 0.990**.

So the knee encoder trades convergence speed for stability and ends with the
lowest step count. Which of those matters is a choice, not a result.

### 16.3 What this closes

The chain from §13 runs end to end:

1. 61% of exploit failures are orbits (§13.5).
2. Those orbits are sinks in the readout field, and a sink is present in every
   failure and absent in all 155 clean-field episodes (§14).
3. `encoder_gain` 5→300 removes the sinks, 37/192 → 2/192, with no policy
   involved and replicated on two encoders (§15.3).
4. **Removing them removes the failures**: 0.919 → 0.997 at ten distractors,
   at ~1.4x the speed, one factor changed.

### 16.4 What is still not explained

**Why raising the encoder gain cleans the field.** The overlap-tail hypothesis
— that a binary code keeps `⟨p_distractor, x⟩` uniformly small — was measured
and **falsified**: distractor overlap p99 moves only 0.170 → 0.137 and the
goal:distractor ratio is essentially unchanged (25.6 → 25.1). The recall
composition is the same; the field is not.

Four mechanism stories have now failed in this section (§15.4 has the other
three). The empirical result is robust — replicated on two encoders, confirmed
in training, policy-independent — and the account of it is absent. The
remaining untested suspect is *local* geometry: `q` comes from projecting
`recall − x` onto a Gram-Schmidt tangent basis, and gain changes how the code
varies between adjacent cells even when global overlaps do not.

Both runs are truncated at ~u1100 and every number is on the `recorded` split.

### 16.5 The residual failures are NOT readout failures — §14's rule does not survive

Field and diagnostic on both final checkpoints, joined on `(env, trial)` with
the RNG alignment verified (0 start mismatches on 192/192 in both arms).

**The field is exactly policy-independent.** `p17_gain` at u1100 gives 2/192
trapped, basin mean 0.9909, worst 0.0350 — identical to four decimals to the
same encoder+gain measured on an *untrained* checkpoint, and the same two
memories `(2,12)` and `(3,29)`. 1100 updates of PPO moved it not at all, which
is what "no policy involved" should mean.

**The same two draws trap under both encoders.** `p18_knee` uses a different
encoder and still traps on `(2,12)` and `(3,29)`, plus two of its own. Some
distractor sets fold the field regardless of the encoding, which bounds what any
encoder fix can buy.

| arm | trapped | basin mean | worst |
|---|---|---|---|
| p10_pol_v1 | 37/192 | 0.848 | 0.005 |
| p17_gain | 2/192 | 0.991 | 0.035 |
| p18_knee | 4/192 | 0.987 | 0.025 |

**And now the part that changes the story.** Both arms fail 3 of 192 at ten
distractors, and **4 of those 6 failures sit on a perfectly clean field**
(goal basin 1.00):

| arm | case | field | symptom | motion | d_min |
|---|---|---|---|---|---|
| p17_gain | env3 t29 | trapped (0.04) | never_approached | committed | 3.27 |
| p17_gain | env4 t14 | **clean (1.00)** | straddled | pinned | 1.26 |
| p17_gain | env5 t28 | **clean (1.00)** | straddled | pinned | 1.12 |
| p18_knee | env4 t14 | **clean (1.00)** | straddled | pinned | 1.20 |
| p18_knee | env5 t7 | trapped (0.28) | straddled | looping | 1.54 |
| p18_knee | env5 t19 | **clean (1.00)** | approached_left | looping | 1.64 |

§14 established that a sink is *necessary* for failure — 24/24 failures had one,
0/155 clean-field episodes failed. **That does not hold here.** It was not wrong
for its regime; it does not generalise once the sinks are removed.

What it was masking: the clean-field failures are `straddled` + `pinned` at
`d_min` 1.1–1.3 against a capture radius of 1.0, with `edge_frac` 0.63–0.67. The
readout is correct — the field routes 100% of the arena to the goal — and the
agent still reaches the doorstep held against the **perimeter** and cannot close
the last ~0.2 cells. That is execution and geometry, not readout.

`env4 trial14` fails in **both** arms on a clean field, so it is a specific hard
goal-near-wall configuration rather than noise. And the trapped memories were
frequently solved anyway (1 of 2 for `p17_gain`, 3 of 4 for `p18_knee`),
consistent with §14's "necessary but not sufficient".

**So the binding constraint has moved.** Readout contamination dominated
failures at 0.919 success; at 0.997 the residual is near-goal pinning against
the arena boundary. Fixing more of the readout will not recover it. The levers
that would are geometric — the capture radius against the minimum step, and how
goals close to a wall are approached — and none of them has been tested.

---

## 17. P19 — the w52 attract-0.5 encoder, and how few updates it needs

**Status: DONE, 2026-08-31.** Delivered model **`p19_kcap`** (21656252) —
accuracy 1.000 from u125, beeline from u150, worst directness 1.090, final
1.013. The one change from `p19_nc` is **`LOG_KAPPA_MAX` 5.0 → 2.5**; the
encoder, both gains and the speed band are as Jack specified. Read §17.9–17.11
for the result; §17.6 and §17.7 are a mechanism story that was **retracted**
and are kept only for the record.

Six arms, in order, and why each ended:

| arm | job | what it tested | outcome |
|---|---|---|---|
| `p19_nc` | 21651001 | control, Jack's config | cancelled u475 — 0.375, κ pinned at the 148 clamp |
| `p19_c100` | 21651003 | curriculum, 100-update ramp | cancelled u125 — confounded, policy locked |
| `p19_c300` | 21651005 | curriculum, 300-update ramp | cancelled unstarted — GPU quota |
| `p19_b5` | 21653544 | `hopfield_beta` 100 → 5 | cancelled u10 — **falsified §17.6 in ten updates** |
| `p19_e20` | 21653877 | `MOVE_ENT_COEF` 0.005 → 0.02 | cancelled u125 — κ −45%, success unchanged |
| **`p19_kcap`** | **21656252** | **`LOG_KAPPA_MAX` 5.0 → 2.5** | **COMPLETED 800/800 — the answer** |
| `p19_kcur` | 21659098 | `p19_kcap` + curriculum | COMPLETED 800/800 — loses on both halves |

### 17.1 The ask, and the one axis it turns on

Jack, in three parts: run
`w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt`; reach the best accuracy
in **as few updates as possible**; learned speed in [0.5, 1.0]. He expects a
distractor curriculum to help, and adds the success criterion: *successful
agents will make beelines towards goal*.

Everything except the curriculum is fixed by that ask, so curriculum **length**
is the only axis these arms sweep — with a no-curriculum control, because §9.9
already measured the curriculum LOSING on exactly this axis.

`ENCODER_GAIN=100` and `HOPFIELD_BETA=100` are what the checkpoint's gain
schedule (`gain_start` 1.0 → `gain_end` 100.0) and the default
beta-follows-gain coupling produce anyway. They are written out because that
silent coupling is what made `p16_sat` a two-factor arm (§15.1).

The scaffold does not move with the encoder: lambdas [11 12 13], `out_dim`
1024, local radius 20, `fwhm_ratio` 0.25 all match the shared defaults, exactly
as they did for the v35 and knee encoders. `goal_radius` 1.0 and
`reset_state_on_teleport` 0 as always.

### 17.2 What is different about this encoder — a short unique radius

| encoder | `attract_lambda` | `r_min` | median | alias | `gain_end` |
|---|---|---|---|---|---|
| `w49_g100_knee/008` (p18) | 2.0 | 12.0 | 17.0 | 0.865 | 100 |
| **`w52_attract_fwhm/001`** (P19) | **0.5** | **5.0** | **9.5** | 0.871 | 100 |

The alias rate is the same; the **radius is less than half**. The arena is
20×20 and a typical start–goal distance is ~10.8 cells, so this encoder's
*median* unique radius sits **below the distance the agent usually has to
cover**. If that bites, it should appear as **far-field** `q` errors, not the
near-goal ones §16.5 left open.

That is cheap to check without waiting for training, because the readout field
`q(x)` depends on the encoder, the Hopfield and the stored memories and **not
on the policy** — so the first checkpoint characterises it as well as the last.
`run_readout_field.sh` is queued behind `p19_nc`'s u25 checkpoint with the same
flags as the p17/p18 field runs (trials 8, draw 32, seed 42, 10 distractors),
so its sink count lands on the same /192 denominator as their **2/192** and
**4/192**.

### 17.3 The arms

|  | curriculum | `n_train_distractors_max` |
|---|---|---|
| `p19_nc` | none — the control | 10 from update 1 |
| `p19_c100` | 100 updates | 0 → 10 |
| `p19_c300` | 300 updates | 0 → 10 |

Shared: `exploit:800`, 20 envs × 64 batch, `GOAL_REWARD` 2.0,
`PERSISTENCE_BONUS` 0.20, polar action with state-dependent σ, **speed learned
in [0.5, 1.0]** (`FREEZE_SPEED` unset), evals and checkpoints every 25.

`exploit:800` rather than the 2000 the P10–P18 arms use, and the 6 h partition
wall lands near u1100 anyway — both p17 and p18 TIMEOUT-ed there rather than
finishing their schedules. 32 eval points clears the ≥4-point bar this
project's eval noise requires for any directional claim.

**CORRECTION.** An earlier draft of this section justified the 800 budget by
saying `p17_gain` "was flat from u50 to u1100". **That is false**, and it is
peak-calling of exactly the kind §9.9 and the eval-noise rule exist to prevent.
Its actual series:

| u | 50 | 100 | 150 | 200 | 400 | 750 | 1100 |
|---|---|---|---|---|---|---|---|
| succ@10 | **1.000** | 0.885 | **0.760** | 1.000 | 1.000 | 0.969 | 1.000 |
| steps@10 | 19.5 | 31.8 | 21.7 | 12.8 | **11.2** | 12.2 | 12.7 |
| speed | 0.75 | 0.59 | 0.64 | 0.86 | **0.95** | 0.86 | 0.90 |

It *touches* 1.000 at u50, **collapses to 0.760 by u150**, is reliably at 1.000
from ~u200, and does not stop dipping below 0.99 until **u800**.

§16.2 is not affected — it already reports the honest pair (first eval ≥ 0.95
*and* the worst eval after it) and names the 0.760 wobble outright. The error
was confined to this section's justification, and to a verbal summary that
compressed §16.2 into "converges 11× faster (u50 vs u550)" while dropping the
second half of the pair. **First-touch is not convergence**, and quoting it
alone is the failure mode §9.9 adopted the two-number rule to prevent.

The 800 budget survives the correction, for a better reason than the one it was
given: success saturates near u200 and the dip-free point is u800, so 800 spans
both. And the second half of the run is not idle — `steps@10` falls 19.5 → 11.2
and `mean_speed` climbs 0.75 → 0.95 long after success is pinned at 1.000. **On
this task success saturates first and the beeline arrives ~200-350 updates
later**, which is directly relevant to Jack's ask: "best accuracy in as few
updates as possible" and "beelines towards goal" are not reached at the same
update, and the second is the later one.

### 17.4 Predictions on record

**The curriculum is not free, and the only measurement we have says it loses on
this axis.** §9.9: the 400-update curriculum won stability outright — never
below 0.979 after breakthrough against the control's 0.490 — but reached
breakthrough *later*, u300 against u150.

- **If breakthrough tracks ramp PROGRESS** (p11_cur broke through ~75% of the
  way through its own ramp), `p19_c100` should break through near **u75** and
  beat `p19_nc`. That is the case for Jack's expectation.
- **If breakthrough is pinned near u150–u300 regardless of ramp length**, the
  curriculum is simply a delay and `p19_nc` wins outright.

`p19_nc` is what makes that falsifiable, and §9.9 is the reason it had to be in
the launch rather than assumed.

**On beelines.** §9.9 measured every speed setting walking within 4–8% of the
straight line, and learned [0.5, 1.0] specifically at directness **1.081×** —
so the beeline criterion is expected to be met, and *step count tracks the
speed cap*, not path quality. At a 1.0 cap over a ~10.8-cell start distance the
ideal is ~11 steps; `p12_lo` got 11.79. **A `mean_steps` much above ~12 means
something is wrong, not that the agent is slow.**

**What would count as the answer to Jack's question:** the update at which each
arm first reaches its plateau, and the minimum it touches afterwards — the two
threshold-free numbers §9.9 settled on, reported as a series rather than as a
peak, per the eval-noise rule.

### 17.5 The objective, restated by Jack — the BEELINE, fast and stable

> "report both separately it's good to know. but the goal should be to get
> 'beelines to goal' as quickly (and stably) as possible"

So accuracy is the **precondition**, not the target. Every P19 arm from here is
reported as **two plateaus**, each with the threshold-free pair §9.9 settled on
(first update reaching the level, and the worst value at or after it):

| plateau | metric | crossing |
|---|---|---|
| **ACC** | `success_rate` @ 10 distractors | first ≥ 0.95 |
| **BEELINE** | `directness` | first ≤ 1.10 |

#### The beeline metric, and why it is not `mean_steps`

`mean_steps` tracks the **speed cap** (§9.9): the same path walked slower costs
more steps. The speed-invariant quantity is the distance actually walked,

    path = mean_steps × mean_speed        [cells]
    directness = path / 10.5

whose floor is the mean straight-line start–goal distance. **10.5 cells** is
inferred two independent ways from §9.9's own table and confirmed by a third:

| source | arithmetic | → |
|---|---|---|
| `p10_pol_v1` | 10.95 steps × 1.00 speed / 1.043 directness | 10.50 |
| `p12_lo` | 11.79 × 0.94 / 1.081 | 10.25 |
| `p17_gain` best observed path | 11.2 × 0.95 | 10.64 |

Good enough to rank arms; **not** a per-episode directness — for that, use the
probe, which computes it against each episode's own start–goal distance.

**Consequence, recorded because it killed a candidate arm:** directness is
speed-invariant *by construction*, so `init_speed_mu` — which sets the initial
speed as a NORMALIZED position in the band, i.e. 0.5 → 0.75 cells in the
[0.5, 1.0] band, not 1.25 as its config comment (written for the [0.5, 2.0]
band) says — moves `mean_steps` but **cannot move directness**. It was
considered as a "reach the beeline sooner" lever and rejected on that basis.
What moves directness is heading quality, and §16 showed the encoder dominates
that.

#### The survivorship trap

`mean_steps` and `mean_speed` are computed over **successful episodes only**, so
at low success only the nearest goals are reached and path looks spuriously
good. `p19_nc` at u100 shows 9.3 steps / path 7.2 cells at success 0.073 — a
"better than converged" path that means nothing. **Directness is therefore only
quoted where succ@10 ≥ 0.90** and suppressed otherwise, rather than footnoted.

#### The two reference runs, scored on the new objective

| run | ACC first ≥0.95 | worst after | **BEELINE first ≤1.10** | **worst after** |
|---|---|---|---|---|
| `p17_gain` | u50 | 0.760 | **u200** | 1.092 |
| `p18_knee` | u550 | 0.990 | u500 | **1.041** |

The sharpest statement of the accuracy/beeline gap: at u50 `p17_gain` was
**accurate but not beelining** — success 1.000 at directness **1.383**, walking
38% further than the straight line. By u200 it was both (1.056), refining to
~1.01 by u400.

On Jack's objective the two split: `p17_gain` reaches the beeline **2.5×
sooner**, `p18_knee` holds it **tighter**. Same gain, same schedule, same
shaping — **the encoder is what separates them**, which is why P19's own curve
is the measurement that matters.

#### `p19_c300` cancelled

`QOSMaxGRESPerUser` caps this account at **2 concurrent GPUs**, so the three
arms serialise and Jack's own `w58_cov` encoder sweep queues behind them.
`p19_c300` was dropped, unstarted: a 300-update ramp only reaches full task
difficulty at u300, which is *by construction* the arm least able to reach a
beeline quickly, and §9.9 already measured longer ramps breaking through later.
`p19_nc` vs `p19_c100` still answers whether a curriculum helps at all; a longer
ramp is worth re-running only if it does.

The readout-field job was moved to **CPU** (`mit_normal`) for the same reason —
it is a policy-free sweep over a 20×20 grid with no rollouts, so it does not
need a GPU and should not spend the quota the training arms need.

### 17.6 RESULT — the field is clean, and `hopfield_beta` is what costs updates

Two measurements, taken while the arms were still running, that together
relocate the bottleneck for the third time in this phase.

#### The readout field is clean — the short unique radius does NOT bite

§17.2 predicted that a median unique radius of 9.5, below the ~10.8-cell
typical travel distance, would show up as far-field `q` errors. **It does not.**
Field mapped on `p19_nc`'s u25 checkpoint (policy-free, so the first checkpoint
is as good as the last), restricted to draws 0–7 so every run is on the same 6
envs, same seed, same draw stream, same denominator:

| encoder | trapped | rate | goal basin |
|---|---|---|---|
| **w52 attract-0.5 @ gain 100** | **1/48** | **2.1%** | 0.980 |
| `p17_gain` v35 @ gain 300 | 0/48 | 0.0% | 1.000 |
| `p18_knee` w49 @ gain 300 | 1/48 | 2.1% | 0.985 |
| `p10_pol_v1` BASELINE @ gain 5 | 5/48 | 10.4% | 0.905 |

Identical to `p18_knee`, 5× cleaner than the baseline, and the single memory it
traps on — **`(env5, trial7)`** — is the *same draw* that traps `p18_knee` and
the baseline. Nothing about the failure is unique to this encoder.

Re-run at the full 32 draws per env, so the w52 number stands on the same 192
denominator as the references rather than on a 48-draw subset:

| encoder | trapped | rate | goal basin | worst |
|---|---|---|---|---|
| **w52 attract-0.5 @ gain 100** | **5/192** | **2.6%** | 0.984 | 0.013 |
| `p17_gain` v35 @ gain 300 | 2/192 | 1.0% | 0.991 | 0.035 |
| `p18_knee` w49 @ gain 300 | 4/192 | 2.1% | 0.987 | 0.025 |
| `p10_pol_v1` BASELINE @ gain 5 | 37/192 | 19.3% | 0.848 | 0.005 |

Same conclusion at 4× the resolution: w52 is marginally the loosest of the three
good encoders and **7× cleaner than the baseline**. Draw **`(3, 29)`** traps all
three of them and `(5, 7)` traps two — these are hard memory configurations, not
a property of any one encoder.

So at u25 the signal is already there: 47 of 48 draws route essentially the
whole arena to the goal. An agent sitting at 0.10 success is **not** being
misled by its memory.

#### κ runaway, driven by `hopfield_beta`

`p19_nc`'s own training metrics, against the two reference runs:

| run | beta | κ@u10 | κ@u40 | κ@u70 | `dir_norm` | `ang_noise` | **beeline** |
|---|---|---|---|---|---|---|---|
| `p17_gain` | **5.0** | 4.8 | 8.4 | 8.8 | 0.26 | 20.6° | **u200** |
| `p18_knee` | 300 | 17.4 | 70.1 | 119.1 | 1.06 | 5.4° | u500 |
| `p19_nc` | 100 | 16.3 | 97.5 | **147.7** | **1.36** | **4.7°** | flat ≥u150 |

The chain: **beta inflates ‖q‖** → `dir_norm` 0.26 → 1.36 → **κ runs away** →
angular exploration collapses from ~21° to ~5°. The policy commits to a heading
before it has learned which heading is right. κ = 147.7 at u70 is the *same
value* that killed `p16_sat` at beta 1e6 (§15), which was cancelled flat at
0.05.

**RETRACTION.** The `p18_knee` variant comment justifies beta=300 with "300 is
far below that regime". **That is wrong.** `p18_knee` ran κ to 133 and paid
**2.5× in updates-to-beeline** (u500 against `p17_gain`'s u200). It did not fail
outright, so the cost was never attributed to beta — it was filed under "the
knee encoder trades convergence speed for stability" (§16.2). On this evidence
the trade was never the encoder's; it was beta's.

This makes beta the *third* thing in this phase whose effect was misread by
looking at only one of its consequences — after "beta is a no-op" (§5.4,
corrected §15.4) and "a sink is necessary for failure" (§14, corrected §16.5).

#### `p19_b5` — the single-factor test

`p19_c100` was cancelled to free the GPU slot: with neither arm learning, the
curriculum axis is confounded by the pathology and cannot be read. `p19_b5` is
`p19_nc` with **beta 100 → 5.0 and nothing else moved**, so it is a clean
single-factor test against a control that is still running.

`p19_nc` is deliberately left running at beta=100 — it is the config Jack
specified, and it is the control.

**Prediction on record:** `dir_norm` ~0.25, κ < 15, and the beeline reached far
sooner than `p19_nc`'s. If κ still runs away at beta 5.0 then the cause is the
encoder rather than beta, which is worth knowing too.

**If it holds**, the lesson generalises past this encoder: `hopfield.beta`
defaults to the encoder gain, so **every arm trained on a gain-100 or gain-300
encoder has silently been paying this cost**, and the default coupling — already
identified in §15.1 as what made `p16_sat` a two-factor arm — is more expensive
than "an unstated default" made it sound.

### 17.7 RETRACTION — beta is NOT the cause. The encoder's readout scale is.

`p19_b5` answered in ten updates, and it **falsifies §17.6's mechanism**. Beta
100 → 5.0, everything else identical:

| u | `p19_nc` (beta 100) | `p19_b5` (beta 5.0) |
|---|---|---|
| 1 | κ 5.618, dir 0.336 | κ 5.638, dir 0.357 |
| 10 | κ 16.343, dir 0.682 | κ 16.223, dir 0.671 |

Identical to three digits. **Beta does nothing here.** The trainer's own startup
line confirms the arm ran at `hopfield beta 5`, so this is not a plumbing
failure — it is the hypothesis being wrong.

#### What `dir_norm` actually is, and why the causal story was backwards

`polar_head.py:393` — `dir_norm = sq.sqrt() * vmax`, the norm of the **policy's
own direction vector**, logged as a gauge. It is the policy's *confidence*, an
OUTPUT of the head, not ‖q‖ arriving from the memory. §17.6 read a policy
statistic as a memory statistic and built a causal chain on it. κ and `dir_norm`
are two views of the same quantity, so "dir_norm drives κ" was never even a
mechanism — it was one number explaining itself.

#### What the data actually says — it is the ENCODER, at update 1

| run | encoder | gain | beta | κ@u1 | **dir@u1** | κ@u10 |
|---|---|---|---|---|---|---|
| `p10_pol_v1` | P2 fixed | 5 | 5 | 4.37 | **0.128** | 8.0 |
| `p17_gain` | P2 fixed | **300** | 5 | 4.38 | **0.126** | 4.8 |
| `p18_knee` | knee | 300 | **300** | 5.65 | **0.351** | 17.4 |
| `p19_nc` | w52 | 100 | **100** | 5.62 | **0.336** | 16.3 |
| `p19_b5` | w52 | 100 | **5** | 5.64 | **0.357** | 16.2 |

Three facts, and they settle it:

1. **The split exists at u1** — essentially at initialisation, before training
   can have done anything. It is a property of the signal, not of learning.
2. **It tracks encoder family exactly.** P2-fixed ⇒ dir ≈ 0.127. The newer
   sweep encoders (knee, w52) ⇒ dir ≈ 0.35, a **2.75×** gap.
3. **Neither knob touches it.** `encoder_gain` 5 → 300 moves dir 0.128 → 0.126
   (`p10_pol_v1` vs `p17_gain`); `hopfield_beta` 100 → 5 moves it 0.336 → 0.357.

So the newer encoders hand the policy a **larger-magnitude readout from the
first step**, the direction head produces correspondingly larger logits, and κ
starts higher and compounds. Angular exploration collapses before the policy has
learned which heading is right.

#### What survives, and what does not

- **SURVIVES:** the κ runaway itself, and its cost. `p17_gain` (κ ~8) reaches
  the beeline at u200; `p18_knee` (κ→133) at u500; `p19_nc` (κ→148) is flat at
  u175. The §16.2 line "the knee encoder trades convergence speed for stability"
  is *right after all* — it IS the encoder — but for the readout-scale reason
  above, not for anything about basins or stability.
- **RETRACTED:** everything in §17.6 attributing this to beta, including the
  retraction §17.6 itself made of the `p18_knee` comment. That comment's claim
  that beta=300 is harmless is, on this evidence, **correct**.
- **UNCHANGED:** the field results (§17.6, first half). Those are direct
  measurements of `q(x)` and do not depend on any of this.

#### Consequence for the objective

`p19_nc` is not broken — at u175 (0.104) it is tracking `p18_knee`, which was at
0.115 at u150 and 0.188 at u200 before breaking through around u500. The w52
encoder behaves like the knee encoder because it *is* the same kind of encoder.

The lever is therefore whatever resists early κ sharpening, and **the one arm
this project already wrote for that was never run**: `p10_pol_v1_e20` /
`_e50` (`MOVE_ENT_COEF` 0.02 / 0.05 against the default 0.005) exist in the
launcher with no checkpoint directory. That is the untested lever, and it is the
one aimed at Jack's objective.

**A note on method.** This is the second mechanism story in two hours built on a
quantity I had not read the definition of — the first being the §15.4 beta
correction. The measurement that killed it (`p19_b5`) cost ten updates. **The
single-factor arm should have been launched before the write-up, not after it.**

#### 17.7.1 ‖q‖ measured directly — a partial account, stated as such

The claim above ("the newer encoders hand the policy a larger readout") was
still an inference from `dir_norm`, a policy statistic. It is measurable
directly and costs nothing: the field JSONs store `field` = the **raw,
unnormalised `q`** at every cell, so ‖q‖ is already on disk for each encoder.

| encoder | gain | beta | mean ‖q‖ | ratio | `dir@u1` ratio |
|---|---|---|---|---|---|
| P2 fixed | 5 | 5 | 0.2589 | 1.00 | 1.00 |
| P2 fixed | 300 | 5 | 0.2390 | 0.92 | 0.98 |
| knee | 300 | 300 | 0.3189 | **1.23** | **2.74** |
| w52 | 100 | 100 | 0.3423 | **1.32** | **2.62** |

**What this supports.** The readout genuinely IS larger for the two runaway
encoders. Rank order agrees 4/4 with `dir@u1`, and the within-family control is
clean: P2-fixed at gain 5 vs 300 moves ‖q‖ by 0.92× and `dir@u1` by 0.98×
together — independent confirmation that `encoder_gain` does not touch either.

**What it does NOT support, and this is the point.** ‖q‖ differs by **1.3×**
while `dir@u1` differs by **2.7×**. Magnitude alone does not account for the
size of the effect; something amplifies it — candidates include the three
`multistep_q` channels scaling together, and the *consistency* of `q` across
steps mattering more than its norm. **Neither has been measured.**

So: ‖q‖ is *a* contributor with the right sign and a perfect rank match over
four points, and is **not** a complete mechanism. Recorded at that strength
deliberately — §5.4, §14, §15.4 and §17.6 were each a story that outran its
measurement, and the pattern is expensive enough to stop repeating.

### 17.8 RESULT — the readout is perfect and the policy drives backwards

`exploit_diag` on `p19_nc` u225, 48 episodes per condition, CPU (the GPU quota
was full):

| condition | | n | `q_acc` | `follow_q` | `align_true` | `d_min` |
|---|---|---|---|---|---|---|
| 0 distractors | SUCCESS | 3 | +0.991 | +0.902 | +0.861 | 0.63 |
| 0 distractors | **FAIL** | 45 | **+0.988** | **−0.726** | −0.742 | 5.98 |
| 10 distractors | SUCCESS | 5 | +0.988 | +0.918 | +0.945 | 0.83 |
| 10 distractors | **FAIL** | 43 | **+0.962** | **−0.691** | −0.725 | 6.31 |

**On the failing episodes the readout is essentially perfect** — `cos(q,
goal−pos)` = 0.988 with no distractors, 0.962 with ten — and the policy moves at
`follow_q` **−0.726**, nearly straight backwards. Motion is **`pinned` for
100% of failures** (45/45 and 43/43), symptom mostly `blocked`, and `d_min` ~6
cells: the agent jams against a wall and drives into it for the whole horizon.

This is Mode B — ignoring a usable readout — far past anything §12 recorded,
where `follow_q` was merely *below* its `align_true` baseline rather than
**negative**. Nothing is wrong with the encoder, the memory, or the field. §17.6
and §17.7's field results said the signal was fine; this says the policy has it
and throws it away.

#### κ is not "running away" — it is SATURATED at the clamp

`--log_kappa_max` defaults to **5.0**, so κ_max = e⁵ = **148.41**. `p19_nc`
measures **147.66** at u70. It is sitting *on the ceiling*, and so were
`p18_knee` (133) and `p16_sat` (~148, cancelled). Every "runaway" in this phase
is the same clamp being reached.

That reframes the fix. `p19_e20` (`MOVE_ENT_COEF` 0.005 → 0.02) cut κ 45% at u30
— 40.6 against 72.2 — and bought **no success at all**: 0.083 / 0.104 / 0.115 /
0.104 / 0.125 against the control's flat ~0.09. **A soft pressure against a
saturating quantity is the wrong instrument.** Arm cancelled at u125.

`p19_kcap` bounds it instead: `LOG_KAPPA_MAX` 5.0 → **2.5**, κ_max **12.2**,
just above the 8.5 that `p17_gain` — the fast arm, beeline at u200 — settled at
on its own. At κ 12 the angular sd is ~0.29 rad (~16°) against `p19_nc`'s 0.08
rad (4.7°). Everything else is `p19_nc`, including Jack's beta=100 and gain=100,
so it is a clean single factor.

#### The control is slow, not stuck — and that took 13 eval points to see

`p19_nc` sat at ~0.09 for **twelve consecutive evals** (u25–u300), then at u325
went to **0.240 / 0.198**, ~2.4× on both metrics at once. That is the shape
`p10_pol_v1` had (0.052 → 0.146 → 0.875 at u150) and `p18_knee` had (climbing
from u200 to 0.917 at u500).

**So the w52 encoder is viable, just slow** — which is exactly the finding that
twelve flat points would have got wrong if the arm had been cancelled. Two
readings were made and withdrawn along the way: "tracking `p18_knee`" (from two
points, and it was flat where p18 climbed) and "`p19_e20` is slowly climbing"
(from three, and it was not). **The rule that held up is the one §9.9 already
adopted: report the series, and do not call a trend under ~5 points.**

Being slow is still a failure against the stated objective — `p17_gain` reached
the beeline at u200, and `p19_nc` had not reached 0.25 success by u325. The
question `p19_kcap` answers is whether the cap recovers the difference.

### 17.9 RESULT — the κ ceiling was the whole problem

`LOG_KAPPA_MAX` 5.0 → 2.5 (κ_max 148.4 → 12.2). **One knob, nothing else moved**
— same encoder, `ENCODER_GAIN` 100, `HOPFIELD_BETA` 100, default
`MOVE_ENT_COEF`, learned speed [0.5, 1.0].

| u | 25 | 50 | 75 | 100 | 125 |
|---|---|---|---|---|---|
| `p19_nc` κ≤148 | 0.083 | 0.083 | 0.073 | 0.073 | 0.104 |
| **`p19_kcap` κ≤12.2** | 0.094 | 0.052 | **0.344** | **0.635** | **0.990** |

A monotone five-point climb to **0.990 @ 10 distractors and 1.000 @ 0 at u125**,
while the control needed 450 updates to reach 0.323.

Against every reference in the phase, on first-reach:

| arm | first ≥0.95 @10 |
|---|---|
| **`p19_kcap`** | **u125** |
| `p10_pol_v1` | u150 (0.875) |
| `p17_gain` | u50 touched, reliable ~u200 |
| `p18_knee` | u550 |
| `p19_nc` | not reached by u450 |

#### The mechanical evidence, which is not eval noise

Per-update training statistics, dense and unambiguous:

| u | `p19_nc` κ / ang | `p19_kcap` κ / ang |
|---|---|---|
| 20 | 33.9 / 0.21 | **12.08** / 0.294 |
| 40 | 97.5 / 0.14 | **12.03** / 0.295 |
| 70 | 147.7 / 0.082 | **12.04** / 0.295 |

κ pins at 12.03 and angular exploration holds flat at 0.295 rad (16.9°) while
the control collapses to 0.082 rad (4.7°). `dir_norm` *falls* (0.587 → 0.486)
instead of inflating. Every prediction §17.8 put on record is confirmed.

#### What this says about the whole phase

`--log_kappa_max` defaults to 5.0 = **e⁵ = 148.4**, and this is where
`p16_sat` (~148, cancelled flat), `p18_knee` (133) and `p19_nc` (147.66) all
ended up. **Three arms across three encoders were each sitting on the same
clamp**, and in every case the effect was read as something else — as beta
(§17.6, retracted), as the encoder "trading convergence speed for stability"
(§16.2), and as a property of saturation (§15).

The cost was never small. `p18_knee` reached the beeline at u500 against
`p17_gain`'s u200; `p19_nc` had not reached 0.35 by u450 where `p19_kcap`
reached 0.99 by u125. **`p17_gain` is the only arm in the phase that never
approached the clamp** — it settled at κ ≈ 8.5 on its own — and it is also the
only fast one. That correlation was visible in §16.2's own table and was
attributed to the encoder.

#### The beeline has NOT arrived, and that is the remaining work

At u125, `steps@10` = 41.0 at `mean_speed` 0.39 → path ≈ 16.0 cells against the
10.5-cell straight line, **directness 1.52**. The agent reaches the goal ~99% of
the time by a route half again too long. Per §17.5 that is the expected ordering
— accuracy saturates first, the beeline follows — and `p17_gain` took ~150 more
updates to close the same gap (31.8 → 21.7 → 12.8 steps).

Note also `mean_speed` 0.39 sits **below `min_action_norm` = 0.5**, so it is
realized displacement after wall clipping, not commanded speed: the agent is
still scraping walls, just no longer pinned against them.

**Not called converged.** `p17_gain` touched 1.000 at u50 and fell to 0.760 by
u150, so the §9.9 pair — first crossing AND the minimum after it — is what
decides this, and only the first half is in.

### 17.10 THE ANSWER — beeline at u150, on Jack's encoder and Jack's gains

Scored on the §17.5 objective — both plateaus, each with §9.9's threshold-free
pair:

| arm | ACC first ≥0.95 | worst after | **BEELINE first ≤1.10** | worst after |
|---|---|---|---|---|
| **`p19_kcap`** | **u125** | 0.990 | **u150** | **1.090** |
| `p17_gain` | u50 touched | 0.760 | u200 | 1.092 |
| `p18_knee` | u550 | 0.990 | u500 | **1.041** |
| `p19_nc` control | not by u475 | — | not by u475 | — |

(`p19_kcap` scored over its completed 800-update run, 32 evals — see the
stability note below, which the mid-run draft of this section got wrong.)

`p19_kcap`'s trajectory:

| u | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|
| succ@10 | 0.094 | 0.052 | 0.344 | 0.635 | 0.990 | **1.000** |
| steps@10 | 11.1 | 8.8 | 17.1 | 75.1 | 41.0 | **16.5** |
| speed | 0.76 | 0.83 | 0.68 | 0.20 | 0.39 | 0.67 |
| path (cells) | 8.4 | 7.3 | 11.6 | 14.7 | 16.0 | **11.1** |
| directness | — | — | — | — | 1.525 | **1.054** |

**It reaches the beeline sooner than `p17_gain` (u150 vs u200) and holds it
marginally tighter (1.090 vs 1.092)** — and does so on the w52 encoder at Jack's
specified `ENCODER_GAIN` 100 and `HOPFIELD_BETA` 100, with learned speed in
[0.5, 1.0]. The delivered config is `p19_kcap`: `p19_nc` plus
`LOG_KAPPA_MAX=2.5`.

Where it clearly beats `p17_gain` is the **success floor: 0.990 against
0.760**, over 27 consecutive evals. The two are near-identical on worst-case
directness; the win is speed-to-beeline and never losing accuracy.

Note the u100 row — 75.1 steps at speed 0.20, path 14.7 — is the survivorship
regime §17.5 warned about *inverted*: success is climbing, and the newly
reachable goals are the FAR ones, so path inflates before it collapses. It is
not a regression; directness at u125 (1.525) → u150 (1.054) is the correction.

**Stability — RESOLVED at completion, and the mid-run number was optimistic.**
This section was first written at u150, where "worst after" rested on a single
point and read **1.054**. Over the completed 800-update run (32 evals) it is
**1.090** — a single excursion at u700; from u650 on the arm runs 1.010–1.017.
Success never drops below 0.990 across **27 consecutive evals from u150 to
u800**.

The pattern is worth naming because it recurred three times in P19: a minimum
taken over few points is mechanically better than one taken over many, so every
early "worst after" flatters its arm. §17.11 records the case where this
actually reversed a conclusion.

#### What actually delivered it

One knob, and it was never an experimental axis — it was a **default**.
`--log_kappa_max` = 5.0 → κ_max = e⁵ = 148.4. Lowering it to 2.5 (κ_max 12.2)
took a run that reached 0.375 in 475 updates to 1.000 in 150.

Everything else tried on this encoder moved nothing:

| tried | effect on κ | effect on success |
|---|---|---|
| `hopfield_beta` 100 → 5 | none (16.2 vs 16.3 at u10) | none |
| `encoder_gain` (5 → 300, on the P2 encoder) | none (dir 0.128 → 0.126) | — |
| `MOVE_ENT_COEF` 0.005 → 0.02 | −45% at u30 | **none** |
| **`log_kappa_max` 5.0 → 2.5** | **pinned at 12.03** | **0.09 → 1.000** |

### 17.11 The curriculum answer — it costs 50 updates and buys ~1%

Jack opened P19 expecting a distractor curriculum to help reach best accuracy in
fewest updates. Two earlier attempts could not answer it — `p19_c100` ran with
the policy locked at the κ clamp so it and its control were both flat at ~0.09,
and `p19_c300` never started. `p19_kcur` is the same test in a regime that
learns: `p19_kcap` **plus** a 0→10 ramp over 100 updates, single factor.

Point counts differ (13 evals past the crossing for `p19_kcap`, 5 for
`p19_kcur`), and a 13-point minimum is mechanically worse than a 5-point one, so
**both are scored on their first five evals past each crossing**:

| | ACC first ≥0.95 | worst succ (first 5) | BEELINE first ≤1.10 | worst direct (first 5) |
|---|---|---|---|---|
| **`p19_kcap`** no curriculum | **u125** | 0.990 | **u150** | 1.054 |
| `p19_kcur` curriculum | u175 | **1.000** | u200 | **1.043** |

**The curriculum delays both plateaus by exactly 50 updates** — +40% on
accuracy, +33% on the beeline — and repays it with a **~1%** tighter hold.

> **CORRECTED at completion.** Both arms finished their full `exploit:800`
> schedules (COMPLETED, 32 evals each, ~4h41m). **The tightness gain was an
> artifact of the short window and reverses with the full series:**
>
> | | ACC first | worst succ | BEELINE first | **worst direct** | final direct |
> |---|---|---|---|---|---|
> | **`p19_kcap`** none | **u125** | 0.990 | **u150** | **1.090** | 1.013 |
> | `p19_kcur` curriculum | u175 | 0.990 | u200 | 1.118 | 1.031 |
>
> At 5 evals past the crossing the curriculum looked tighter (1.043 vs 1.054);
> over all 32 it is **looser** (1.118 vs 1.090). This is precisely the
> asymmetry the matched-window scoring was introduced to guard against — and
> the guard was not enough, because *both* windows were short. **The rule that
> actually works is to compare completed runs.**
>
> So the answer is not a trade-off: **no curriculum wins on BOTH halves** — it
> crosses 50 updates sooner *and* holds tighter. Jack's opening hypothesis is
> falsified in this regime, cleanly.
>
> `p19_kcap`'s worst is one excursion at u700; from u650 on it runs 1.010–1.017
> at success 1.000, i.e. **27 consecutive evals from u150 to u800 with success
> ≥ 0.990.**

On the §17.5 objective (the beeline, fast AND stable) that is a bad trade, and
**no curriculum wins**. The direction reproduces §9.9 exactly (there: u300
against u150, with stability 0.979 against 0.490), so this is now measured
**twice in two different regimes with the same sign**. The magnitude of the
stability gain is much smaller here — 1% against §9.9's 0.49 — because with κ
capped there is no longer a turbulent phase for the curriculum to smooth out.
**That is the more interesting half of the result:** §9.9's curriculum was
mostly compensating for instability that the κ clamp was causing, and once the
clamp is fixed there is little left for it to buy.

#### Delivered

`p19_kcap` — job 21656252, `navigate_navp2_p19_kcap_s42_21656252/`.

| | |
|---|---|
| encoder | `w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt` (Jack's) |
| `ENCODER_GAIN` / `HOPFIELD_BETA` | 100 / 100 (Jack's) |
| speed | learned in [0.5, 1.0] (Jack's) |
| **`LOG_KAPPA_MAX`** | **2.5** — the one change from the default 5.0 |
| run | **COMPLETED 800/800**, exit 0, 4h41m, 32 evals |
| accuracy | **1.000 @ 0 and @ 10 distractors, from u125**; floor **0.990** over 27 consecutive evals u150–u800 |
| beeline | **from u150**; worst 1.090 (one excursion at u700), **1.013 final**, 1.010–1.017 from u650 |
| mean_speed | **0.99** at u800, against the 1.0 cap |

Rollout shape, from the run's own `run.json`: 1 world, **20 envs × 64 rollouts
per env = 1280 trajectories/update**, 200 steps each (256,000 env-steps/update,
4,000 serial policy calls). PPO: `ppo_epochs` 4, `n_minibatches` 4 — advantages
normalised across the full pool, minibatches over whole trajectories (the trunk
is an RNN). `refresh_envs_each_update = False`, so **the 20 envs and their goals
are fixed for the whole run**; start position and the distractor draw (~U[0,10])
are re-sampled every rollout.

Against the phase's previous best, `p17_gain`: beeline at **u150 against u200**,
worst-case directness essentially tied (**1.090 against 1.092**), and a success
floor of **0.990 against 0.760**. The win is speed-to-beeline and never losing
accuracy, not a tighter path.

**Caveat worth carrying.** With `refresh_envs_each_update=False` the model saw
**20 fixed (env, goal) pairs** for all 800 updates. Scores are on the
`recorded` val split, not those goals — but `project_hopfield_nav_v22` records
that enabling the per-update goal refresh halved seed variance elsewhere. If
this result needs hardening against the seed lottery, that is the one-flag
rerun.
Faster *and* tighter *and* steadier, on a different encoder, from one default.

---

## 18. P20 — the explore side of the w52 encoder

P19 delivered the exploit half on Jack's encoder. This is the matching explore
half. Two arms, launched 2026-08-31, **21695407 `p20_e`** and **21695408
`p20_e_kcap`**.

### 18.1 Why this is not a formality

Explore depends on the **opposite** property of the memory from exploit, and
the two are not guaranteed to move together under an encoder swap:

| | needs | failure looks like |
|---|---|---|
| **exploit** | `q` points at the goal | drives the wrong way, or ignores a good readout (§12) |
| **explore** | `‖q‖` is **small** when only distractors are stored | chases a phantom — phase 1's corner trap |

The w52 code is sharper and higher-gain than the P2 gain-5 code every explore
result in this document was measured on, so a sharp retrieval of a *distractor*
is exactly the thing that could give the policy a confident `q` pointing
nowhere. That is the open item §0 has carried since §5.7 ("a sharp retrieval
could give a confident `q` pointing at a phantom and drive `chase_q` up").

**§7.7.2 already put a number on it, and it points the other way.**
Goal-present vs goal-absent `‖q‖` separability at ten distractors is **AUC
0.930 on w52 against 0.698 on the gain-5 code**. The property explore depends
on is *better* on this encoder. That is the prediction this run tests
behaviourally rather than statically.

### 18.2 The arms — three lines of diff, and nothing else

Both arms are `p19_kcap` with the schedule, the eval scope and κ changed:

| | `p19_kcap` (exploit) | `p20_e` | `p20_e_kcap` |
|---|---|---|---|
| schedule | `exploit:800` | `explore:700` | `explore:700` |
| `EVAL_SCOPE` | `navexpl` | `expl` | `expl` |
| `LOG_KAPPA_MAX` | **2.5** | *default 5.0* | **2.5** |
| encoder / gain / beta | w52 / 100 / 100 | same | same |
| `‖a‖` band | [0.5, 1.0] | same | same |
| polar, learned speed, state-dep std | yes | same | same |
| `ENVS_PER_WORLD` × `BATCH_ENVS` | 20 × 64 | same | same |

`REGIME_ASSIGNMENT` and `GOAL_REWARD` are both **provably inert** on a pure
explore schedule — `n_pre_now` is 0 or `n_envs` so both assignment branches
agree (`train_navigate.py:366`), and `EXPLORE_GOALS_OFF=1` means no goal is
stored — but they are set to `p19_kcap`'s values anyway so the diff is exactly
the three rows that claim to be the diff.

#### The metric to quote is `strategy_efficiency`, not `cells_per_step`

`MAX_ACTION_NORM=1.0` carries over from P19, per Jack's speed instruction.
`p10_e_pol` — the phase-2 explore best, cps **0.75** — ran at **2.0**. Since
`cells_per_step` depends on stride length, **the raw comparison to 0.75 is
invalid by construction**, in the same way §17.5's `mean_steps` was before
`directness` replaced it.

`strategy_efficiency` is the fix and it already exists: `cps` divided by a
perfect billiard **at the realized speed** (`behavior_probe.py:540`). Quote
that against `p5_e`'s **1.113**. §9.1's instrumentation fault is the reason to
be careful here — the same metric once read 3.97 because it referenced the
*commanded* magnitude.

§2.1 says billiard coverage peaks at `‖a‖ ≈ 1.0–1.5` and falls above, so
capping at 1.0 sits at the low edge of the optimum: it should cost little
against 2.0, and may help.

> **RETRACTED — §19.2.** Measured on **cell** coverage, which hides the
> speed axis. Under swept area the cap costs ~30% of expected discovery time.
> The table below is a correct reading of the wrong metric.

**MEASURED — it helps, and it raises the target.** `billiard_cells_per_step`
at this arena and horizon:

| `‖a‖` | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | **1.98** | 2.0 |
|---|---|---|---|---|---|---|---|
| billiard cps | 0.495 | 0.658 | **0.757** | 0.783 | 0.761 | **0.702** | 0.687 |

The reactive ceiling at the new cap (**0.757**) is **higher** than at the old
one (0.687), because 2.0 is past the peak and 1.0 is not. So the [0.5, 1.0]
band Jack's speed instruction imposes is not a concession on the explore side —
it is nearer the coverage optimum than the band `p10_e_pol` ran in.

This also fixes the target. `p10_e_pol`'s cps 0.75 at `‖a‖ ≈ 2.0` is
`strategy_efficiency` **1.09**, consistent with `p5_e`'s independently measured
1.113 at 1.98. **An arm that matches that efficiency at speed 1.0 should read
cps ≈ 0.84**, not 0.75. That is the number to hold these runs to; anything
compared against 0.75 flatters them by ~12%.

#### Why 700 updates and not 1500

`p10_e_pol`'s own series converged at u200–250 and then went nowhere:

| u200 | u250 | … | u1250 | u1500 |
|---|---|---|---|---|
| 0.747 | 0.750 | oscillating 0.61–0.78 | 0.776 | 0.721 |

**+0.026 across 1250 updates**, against an oscillation band of 0.17. That does
not clear this project's eval-noise bar (`feedback_eval_point_threshold`). 700
updates is 2.8× the convergence point and gives **28 eval points at
`EVAL_EVERY=25`** (§18.7 measures what that budget bought: 78% of the rise by
u150, and the first ~150 updates go on escaping a wall-pin basin rather than on
learning to explore) — more than `p10_e_pol` got across twice the updates — and
lands near 4.5 h against the 6 h partition wall, with checkpoints every 25.

### 18.3 Predictions on record

**H1 — the encoder does NOT hurt explore.** Grounded in §7.7.2's 0.930 vs
0.698. **SCORED — §18.4: CONFIRMED.** `chase_q` 0.000, coverage at ten
distractors equal to coverage at zero. *Falsifier:* cps at ten distractors materially below cps at zero, or
`chase_q` materially above 0. `p10_e_pol` had `cps10 ≈ cps0` at every one of
its 30 evals and `chase_q ≈ 0.000`.

**H2 — `LOG_KAPPA_MAX=2.5` HURTS explore. This is the opposite sign from
P19,** and it is the reason both arms are being run rather than just the capped
one. Explore coverage comes from persistent straight motion: `p5_e` measured
`straightness` **0.945**, and a billiard (straightness ≈ 1) is the reactive
ceiling it beat by 11%. A von Mises at κ has circular sd ≈ `1/√κ`, so the cap
**floors** per-step directional noise at `1/√12.2` = 0.286 rad = **16.4°**,
against the **4.7°** the `p10_pol_v1` exploit arm learned far-field (§9.8.1).
Capping κ is capping straightness.

> **SCORED — §18.4. Direction right, mechanism WRONG.** The cap does cost
> explore: −12.1% `strategy_efficiency`. But `straightness` came out **higher**
> on the capped arm (0.979 vs 0.958), the opposite of what the argument above
> requires. The real mechanism is `edge_frac` — 0.061 vs 0.127 — a κ ceiling
> is a ceiling on *turn sharpness*, so the capped policy cannot run the wall
> and under-visits the perimeter. Straightness is not a coverage proxy.

*Falsifier:* `p20_e_kcap ≥ p20_e` on `strategy_efficiency`. **If the cap is
neutral or good here too, that is the larger finding** — it would mean the
`e⁵` default is simply wrong for this project across both regimes, not a
regime-specific fix, and §17.9's result would generalize well beyond the arm it
was found on.

**H3 — converged by ~u250**, as `p10_e_pol` was. **SCORED — §18.4:
approximately right** (~0.68 by u250) but the uncapped arm kept improving to
0.777 by u700.

The three predictions are independent: H1 is about the memory, H2 about the
policy's action distribution, H3 about budget. §7.7.1 got two predictions wrong
in a row in this document (§7.7.2), so these are recorded to be scored, not to
be right.

### 18.4 RESULT — the κ cap costs 12% of explore, and NOT for the reason predicted

Both arms COMPLETED 700/700. Behaviour probe on the final checkpoints, job
21719662, `--mode explore`, 8 envs × 32 trials, **`--split place=held_out`** —
a fresh draw, not the `recorded` split every §9.6–9.8 number sits on (§9.8.2).

| | **`p20_e`** (κ uncapped) | **`p20_e_kcap`** (κ ≤ 12.2) | Δ |
|---|---|---|---|
| `mean_coverage` | **0.390** | 0.338 | −13.3% |
| `cells_per_step` | **0.780** | 0.676 | |
| **`strategy_efficiency`** | **1.038** | **0.912** | **−12.1%** |
| `billiard_ref` | 0.752 | 0.742 | |
| `realized_mag_mean` | 0.964 | 0.951 | |
| **`straightness`** | **0.958** | **0.979** | **+2.2% — WRONG WAY** |
| `edge_frac` | **0.127** | **0.061** | **−52%** |
| `chase_q` (0 / 10 dist) | 0.000 / 0.013 | 0.000 / −0.006 | |

Training-eval agreement is good: the final four evals gave 0.777 against the
probe's 0.780 for `p20_e` and 0.694 against 0.676 for `p20_e_kcap`, on
different draws.

#### H1 — CONFIRMED. The sharper encoder does not break explore.

`chase_q` is **0.000** at zero distractors on both arms and **0.013 / −0.006**
at ten. Coverage at ten distractors equals coverage at zero (0.387 vs 0.390).
The concern §0 has carried since §5.7 — that a sharp retrieval would give the
policy a confident `q` pointing at a phantom and drive `chase_q` up — **does
not materialize on this encoder.** §7.7.2's static prediction (goal-absent
`‖q‖` separability 0.930 on w52 against 0.698 on the gain-5 code) is now
backed behaviourally.

> **MAGNITUDE RETRACTED — §23.** Measured sampled rather than
> deterministic, the gap is **3.2%, not 14%**: the capped arm goes 0.333 →
> 0.375 while `p20_e` stays flat. Its own noise breaks the closed orbits that
> trap its mean policy (state repeats 1905 → 524). The cap costs something; it
> costs about a quarter of what this section reports.

#### H2 — direction CONFIRMED, mechanism REFUTED

The cap costs **12.1% of `strategy_efficiency`**, and `p20_e` won **all 16**
of the last 16 matched eval points (u325–u700), so the effect is real and well
past this project's noise bar. Across all 24 matched points from u125 the means
are **0.703 against 0.671**; across the final four, **0.777 against 0.694** —
the gap widens with training rather than closing.

**But the stated mechanism is wrong, and the probe says so directly.** §18.3
argued: "A von Mises at κ has circular sd ≈ 1/√κ, so the cap floors per-step
directional noise at 16.4°… **capping κ is capping straightness**." Measured
`straightness` is **higher** on the capped arm — 0.979 against 0.958. The
prediction has the sign backwards on its own mechanism variable.

> **SHARPENED — §18.6.** The account below is right in direction but weaker
> than the trajectory data supports. The capped policy learned a **constant
> one-signed turn** of +0.120 rad/step (a full revolution every 52 steps),
> non-overlapping with the uncapped arm's +0.001. It traces an annulus, which
> is why it both misses the perimeter and retraces the middle.

**What actually separates them is `edge_frac`: 0.061 against 0.127, less than
half.** Uniform occupancy is 0.19 and `p5_e` measured 0.121, so `p20_e` sits
where the good explore policies sit and the capped arm is the outlier — it
**avoids the perimeter**. The coherent reading, and it is consistent with every
number in the table: a κ ceiling is a ceiling on how *sharply* the policy can
turn. A policy that cannot turn tightly cannot run the wall — it must arc away
on approach — so it under-visits the perimeter ring and its trajectories are
straighter precisely *because* it never makes the hard corner turns the
uncapped policy makes. Straightness goes up, coverage goes down.

**The general lesson is that `straightness` is not a coverage proxy.** The
billiard reference has straightness ≈ 1 and is the *reactive ceiling*, which
invited exactly the inference §18.3 made. Here the straighter policy is the
worse one. Any future argument of the form "this should help coverage because
it makes motion more persistent" now has a counterexample in this document.

#### The cap buys stability, and that is a real trade

Final four evals, and this was not predicted either:

| | u625 | u650 | u675 | u700 | range |
|---|---|---|---|---|---|
| `p20_e` | 0.749 | 0.794 | 0.796 | 0.769 | 0.047 |
| `p20_e_kcap` | 0.695 | 0.697 | 0.689 | 0.695 | **0.007** |

The capped arm is **~7× tighter**. So κ capping is the same shape of trade as
§17.11's curriculum — steadier, worse — and **the κ setting for an interleaved
model is a genuine choice, not a free win**: exploit needs the cap (§17.9 took
it from 0.375-at-u475 to 1.000-at-u125), and explore pays 12% for it.

#### H3 — approximately right

Both arms reached ~0.68 by u250, as `p10_e_pol` did, then improved slowly:
`p20_e` to 0.777 by u700, `p20_e_kcap` flat at ~0.694 from u500 on. So "u250"
is right to within ~8% for the level but the last 400 updates were not wasted
on the uncapped arm. `p10_e_pol`'s series showed the same slow tail.

### 18.5 What `p20_e` is worth against the phase's explore best

**`p20_e` is the delivered explore model**:
`$CLS_RUNS/agent_ckpts/navigate_navp2_p20_e_s42_21695407/navigate_u700.pt`

| | `p5_e` | `p10_e_pol` | **`p20_e`** |
|---|---|---|---|
| encoder | P2 gain-5 | P2 gain-5 | **w52 gain-100** |
| `‖a‖` band | [0.5, 2.0] | [0.5, 2.0] | **[0.5, 1.0]** |
| realized speed | 1.98 | ~2.0 | **0.964** |
| `mean_coverage` | 0.390 | ~0.37 | **0.390** |
| `strategy_efficiency` | 1.113 | ~1.09 | **1.038** |
| split | `recorded` | `recorded` | **`held_out`** |

**It matches `p5_e`'s absolute coverage exactly (0.390) at half the speed**,
and beats `p10_e_pol`'s ~0.37. It still beats a perfect billiard, by 3.8%.

**But the efficiency margin is lower — 1.038 against 1.113 — and that
comparison is not clean in either direction.** Two confounds, pulling opposite
ways and neither quantified:

- **Against `p20_e`**: it runs at 0.964 where the billiard reference is
  **0.752**, while `p5_e` ran at 1.98 where the reference is **0.702**. A
  higher bar divides a similar numerator. This is a consequence of Jack's speed
  band, not of the encoder.
- **For `p20_e`**: this is the **first P-series probe on a fresh
  `place=held_out` draw**. `p5_e`'s 1.113 is on the set it was scored against
  at every eval. The direction of that bias is known — a fresh draw is
  harder — so 1.038 and 1.113 are not comparable and the gap is an upper bound
  on the real one.

The honest statement is **absolute coverage matched at half the speed, margin
over billiard not established either way**. Closing that needs `p5_e` re-probed
with `--split place=held_out`, which is the §9.8.2 open item and one command.

#### The speed cap was free, as §18.2 predicted

`realized_mag_mean` is **0.964** against a 1.0 ceiling — the policy saturates
its speed bound, exactly as §9.1/§9.2 measured it doing at 2.0. But at 1.0 that
costs almost nothing: billiard peaks at 1.25 (0.783) and 1.0 gives 0.757,
against 0.687 at 2.0. **Jack's [0.5, 1.0] band puts the saturating policy
nearer the coverage optimum than the band the earlier explore runs used**, and
the matched coverage at half the speed is that prediction landing.

### 18.6 The two explore failure modes, and a sharper mechanism for §18.4

288 matched episodes (`analysis/nav_tri/explore_traj.py`, 6 envs × 24 trials ×
2 distractor levels, `place=held_out`). Every episode is rolled from the
identical start with identical memory contents on both checkpoints, so a
difference between two arenas is the policy alone. Page:
[f59ee221](https://claude.ai/code/artifact/f59ee221-a39d-4af4-8f18-0fb8a5f824f4)

#### Mode 1 — the wall pin. Rare, catastrophic, and coverage does not name it.

**One episode in 144** on `p20_e`:

| | this episode | that policy's median |
|---|---|---|
| coverage | **0.052** (21 cells of 400) | 0.390 |
| realized speed | **0.10** | 0.96 |
| `clip_frac` | **0.91** | 0.03 |
| `edge_frac` | **0.94** | 0.13 |

The agent reaches a wall and stays there. It keeps *commanding* full stride
while the boundary clip absorbs 91% of it, so realized speed collapses to a
tenth. This is `project_hopfield_nav_perimeter_basin` in a single trajectory.

**The diagnostic is the commanded/realized gap, not coverage.** Low coverage is
the symptom and it is shared with ordinary bad episodes; `clip_frac` 0.91 at
`speed` 0.10 is the signature and nothing else in 288 episodes comes near it.

#### CORRECTION — both arms loop. Jack spotted this in the trajectories.

The heading below originally read "the circler", which implies looping is
specific to the capped arm. **It is not.** Measured on the same 288 episodes
with a return-based detector (the path comes back within 1.0 cell of somewhere
it was ≥15 steps earlier) and against a **billiard null** at each arm's own
speed, because in a 20×20 box a 200-step path re-crosses itself by geometry
alone:

| | re-crossings / episode | vs null | share of steps on old ground | vs null | handedness flips |
|---|---|---|---|---|---|
| `p20_e` (uncapped) | **12.5** | **+4.4** | 0.310 | +0.062 | **7.9** |
| billiard null @0.96 | 8.1 | — | 0.248 | — | — |
| `p20_e_kcap` (capped) | 8.9 | +2.4 | **0.445** | **+0.140** | **0.0** |
| billiard null @0.95 | 6.5 | — | 0.305 | — | — |

**100% of episodes in BOTH arms re-cross their own path, and the uncapped arm
does it MORE often** — 12.5 events against 8.9, and well above its own billiard
null. My §18.6 framing was wrong to present looping as the capped arm's
pathology.

**What is actually different is dwelling, and handedness.** The uncapped arm
crosses its path more but spends *less* of the episode on old ground (0.310
against 0.445); its excess over the null is +0.062 against the capped arm's
**+0.140, more than double**. It cuts across and keeps going. And it flips
handedness ~8 times an episode where the capped arm flips **zero** times.

> **The cost is not looping. It is not leaving.**

**A detector caveat that nearly cost me a second wrong claim.** The turn-based
detector — |Σ dθ| ≥ 2π inside a window — is contaminated by wall bounces, which
are large instantaneous turns: the **billiard null scores 59% on it**. It
cannot be read on its own as evidence of looping, only as "rotational
character". The return-based detector is the one to trust, and it is the one
the table above uses.

#### Mode 2 — a CONSTANT-RATE circle. What §18.4's 12% actually is.

> **SHARPENED again — Jack: "even the uncapped does circling sometimes."**
> Right, and `signed_turn_mean` hid it for the second time, the same way it hid
> the looping the first time: it is an EPISODE MEAN, so a path that circles one
> way then the other cancels to ~0. Windowed over 40 steps, 144 episodes each:
>
> | | windows with \|turn\| > π | of those, positive | episodes with BOTH a +π and a −π window |
> |---|---|---|---|
> | `p20_e` | **24.4%** | 55% | **143/144 (99%)** |
> | `p20_e_kcap` | 99.3% | 100% | 0/144 (0%) |
>
> **Both arms circle**, and the uncapped one accumulates half a rotation in a
> quarter of its windows — in both directions, in 99% of episodes. The
> difference is **CONSISTENCY, not presence**: windowed turn is mean −0.005 /
> sd 2.69 for `p20_e` against mean **+4.86 / sd 0.77** for the capped arm —
> 16% relative spread. The capped policy is a **metronome**: a near-constant
> 0.121 rad/step, one revolution every 52 steps, every window, every episode.
>
> So the thing needing explanation is not a handedness preference but why it
> locked onto a constant turn RATE — a limit cycle in the learned policy.
>
> **THIRD correction, and "metronome circler" overstates the visual — Jack:
> "it does really good looking nav sometimes."** It does, and by every LOCAL
> measure the capped arm is *straighter than the control*:
>
> | | steps < 15°/step | longest straight run | abs turn | **signed turn** |
> |---|---|---|---|---|
> | `p20_e` | 76.7% | median 19, max 30 | 11.1° | **0.03°** |
> | `p20_e_kcap` | **84.3%** | **median 22, max 37** | **8.4°** | **6.9°** |
>
> The drift is **6.9°/step — below the 15° that counts as running straight**.
> Locally it is clean straight running; it simply never cancels, and closes a
> revolution every 52 steps. Nothing in the paths looks like a tight circle.
>
> The honest statement is the ratio of systematic to total turning:
> **`p20_e_kcap` is 6.9° of 8.4° = 82% systematic; `p20_e` is 0.03° of 11.1° =
> 0.3%.** The uncapped arm turns MORE but symmetrically; the capped arm turns
> LESS but almost entirely one way. Call it a **slow systematic curl inside
> locally straight running**, not a circle.
>
> That is also the functional cost: bidirectional turning redirects into new
> regions, while a curl that never reverses confines the agent to one annulus
> — `edge_frac` 0.061 and dwelling 0.445, the 12% coverage gap.
>
> **METHOD NOTE.** This finding was corrected three times, each time by Jack,
> and each time the cause was the same: an aggregate answering a subtly
> different question than the one asked. `signed_turn_mean` (an episode mean)
> hid the looping, then hid the bidirectional circling; `straightness` (an
> unsigned cosine) hid the curl. **For turning behaviour, report the signed /
> unsigned / windowed triple together — no one of them is safe alone.**

Given that both arms loop, the property specific to the capped policy is
**handedness**. It turns the same direction on every step of every episode:

| `signed_turn_mean` (rad/step) | mean | median | min | max |
|---|---|---|---|---|
| `p20_e` (uncapped) | +0.001 | −0.001 | −0.030 | **+0.036** |
| `p20_e_kcap` (capped) | **+0.120** | +0.118 | **+0.094** | +0.148 |

**The distributions do not overlap.** The slowest-turning capped episode still
turns faster than the fastest-turning uncapped one, with a gap of 0.058
rad/step across 144 episodes each. This is not a tendency; it is a property of
the policy.

**But read it with the box above:** what does not overlap is the *episode
mean*. The uncapped arm turns hard in both directions and cancels; it is not a
policy that goes straight.

**And the bias is global, not environmental.** 100% of 144 trials turn the same
way, across six arenas, with between-env spread (0.008) no larger than
within-env (0.009). It lives in the weights: not the arena, not sampling
(evaluation is deterministic, §20.1), not a per-episode accident. `p20_e` is
cleanly unbiased by the same measure — 49% positive, per-env means within
±0.001 of zero.

**Mechanism: MEASURED, §28.** A ~54-step orbit, read straight off the recurrence curve (dip depth 10.66, 98/100 trajectories) against the curl's predicted 51.9. The account below stands.

**Originally recorded as unknown:** A concentration cap contains no left or right, so
this is spontaneous symmetry breaking, and no account in §18.4 or §18.6
explains it. One systematic asymmetry does exist and is worth knowing:
`vec_env.reset_all` sets `_heading_rad = 0.0`, so every episode starts facing
due east. That cannot be the whole story — `p20_e` shares the init and shows no
bias — but it is a candidate seed. **The decisive test is a different seed:**
if the capped config circles the other way, the direction is arbitrary and the
mechanism is "the cap destabilises the symmetric solution"; if it circles the
same way, something systematic is biasing it and the polar head's θ
parameterization and that east-facing init are where to look.

**This REPLACES §18.4's mechanism sentence.** That section said a κ ceiling
"is a ceiling on how sharply the policy can turn," so the capped policy "must
arc away on approach" and under-visits the perimeter. The direction was right
but the statement was weaker and vaguer than the data supports. What the capped
policy actually learned is a **constant-rate turn** — 0.120 rad/step is a full
revolution every 52 steps, so roughly four circuits per 200-step episode. A
constant-rate turn traces an **annulus**, which explains both halves of the
§18.4 table at once: it never reaches the perimeter (`edge_frac` 0.061) *and*
it dwells in the middle (0.445 of steps on old ground against 0.310).

It also explains the straightness inversion that refuted the original
prediction. `straightness` is a mean **unsigned** cosine, so a steady circle
and an unbiased walk are indistinguishable to it — the capped arm scores 0.979
because a gentle constant curve is locally very straight. `signed_turn_mean`
is the statistic that separates them, and it is the one `behavior_probe`'s own
docstring says exists for exactly this case ("a policy that circles at a
constant rate and one that jitters symmetrically can score the same").

#### A methodological correction — `revisit_frac` is coverage restated

Measured correlation with coverage is **−1.0000** in both arms, and it is
algebraic rather than empirical: at a fixed 200 steps, unique cells =
200·(1−revisit), so

> **coverage = (1 − `revisit_frac`) / 2**, exactly — verified to 0.00000 max
> absolute error across all 288 episodes.

So `revisit_frac` carries **no information coverage does not**, and any future
result quoting both as if they corroborate each other is double-counting one
measurement. It stays in the trajectory captions because it is legible there,
not because it is evidence.

#### What did NOT show up

`chase_q` at ten distractors is **+0.022** (uncapped) and **−0.016** (capped).
Neither arm chases a phantom recall. Both failure modes above are **motor**,
not memory — which is the behavioural counterpart of H1 and consistent with
§7.7.2's separability result.

### 18.7 Where the updates actually go — 100% of early episodes are wall-pinned

Jack, on reading §18: *"I don't understand how it could take 700 updates to
learn such a simple policy."* Two things are wrong with the premise, and the
second one is the interesting one.

**It did not take 700.** 78% of the total rise is done by u150 and the run is
within 10% of its best by u250. 700 was a budget chosen before the run, not a
learning time.

**And the first ~150 updates are not spent learning to explore.** Rolling the
early checkpoints (job 21745999, 4 envs × 16 trials, matched starts,
`place=held_out`):

| ckpt | coverage | realized speed | `clip_frac` | `edge_frac` | `straightness` | **pinned episodes** |
|---|---|---|---|---|---|---|
| u25 | 0.048 | **0.088** | **0.914** | 0.928 | **0.985** | **64/64 (100%)** |
| u50 | 0.049 | **0.105** | **0.906** | 0.926 | 0.981 | **64/64 (100%)** |
| u75 | 0.194 | 0.520 | 0.445 | 0.533 | 0.944 | 20/64 (31%) |
| u100 | 0.215 | 0.609 | 0.374 | 0.468 | 0.962 | 13/64 (20%) |
| u150 | 0.326 | 0.868 | 0.070 | 0.241 | 0.951 | **0/64 (0%)** |
| u700 | 0.386 | 0.963 | 0.031 | 0.126 | 0.958 | 0/64 (0%) |

("pinned" = the §18.6 wall-pin signature, `clip_frac` > 0.5 with realized speed
< 0.5.)

**Every episode at u25 and u50 is wall-pinned.** The policy *commands* ~0.79
and *realizes* 0.09 — the boundary clip absorbs **91%** of every step — while
spending 93% of the episode on the perimeter ring. It is pressed into the wall,
not exploring.

**The un-pinning IS the learning curve.** Pinned fraction goes
100% → 31% → 20% → 0% across u50–u150, and coverage tracks it exactly
(0.049 → 0.194 → 0.215 → 0.326). Once free at u150 the model is already at
0.326, and the remaining **550 updates buy +18%** (0.326 → 0.386).

So the honest budget is: **~150 updates to escape the wall-pin basin, then
~100 more to reach the plateau, then a long slow tail.** The thing that looks
expensive is not the exploring.

#### This resolves the u50 anomaly, and adds a third straightness counterexample

§18's own per-update trace showed `ang_noise` 0.191 at u50 — a persistence
length of 27 steps, more than enough to cross a 20-cell arena — alongside
coverage 0.05. That looked impossible for a policy "still learning to go
straight", and it is: the policy is not walking at all.

It also produces the strongest version of §18.4's lesson. **`straightness` at
u25 is 0.985 — the highest value anywhere in this document** — and it belongs
to a policy that covers 4.8% of the arena while jammed against a wall. A pinned
agent does not turn. Three times now (§18.4's capped arm, §18.6's circler, and
here) a *higher* straightness has accompanied *worse* coverage.

#### What this suggests trying, and what it does not establish

> **SUPERSEDED — §18.8.** The suggestion below (raise `WALL_PENALTY`) is
> **not** the recommendation any more. The reward decomposition shows the pin
> is *paid for* by the **persistence bonus** at +0.196/step against
> `wall_penalty`'s −0.093 — and that raising `wall_penalty` to the >0.24 it
> would need also taxes legitimate perimeter work, pushing toward the
> edge-avoiding policy §18.4 measured at 12% worse coverage. The fix is
> `--persistence_realized`.

The obvious lever is the one aimed at the basin rather than at the objective:
`WALL_PENALTY` is **0.1** in this config, and if 100% of early episodes are
pinned then that term is not doing the job it exists for. A bracket on it —
or an epsilon/κ schedule that keeps the early policy from committing into the
boundary — is where the 150 updates are, and it is a much cheaper experiment
than another 700-update arm.

**Not established:** that the pin is *caused* by κ sharpening rather than
merely co-occurring with it. `ang_noise` falls 0.477 → 0.191 over u1–u50, which
is the same over-commitment mechanism §17.9 measured on the exploit side, and
it is the natural story — but a pinned policy also *sees* almost no variation,
so the causal arrow could run either way. Distinguishing them needs an
intervention, not another observation.

### 18.8 The pin is PAID FOR — and by the persistence bonus, not the wall

Jack, following §18.7: *"so the next step might be to raise WALL_PENALTY?"*
The reward decomposition says no, or at least not that knob first.

Every explore-phase shaping term, priced per step:

| ckpt | novelty | **persistence** | wall | time | **predicted** | **logged `mean_r`** |
|---|---|---|---|---|---|---|
| u50 **pinned** | +0.030 | **+0.196** | −0.093 | −0.050 | +0.084 | **+0.074** |
| u150 free | +0.236 | +0.190 | −0.024 | −0.050 | +0.352 | **+0.334** |
| u700 final | +0.292 | +0.192 | −0.013 | −0.050 | +0.421 | **+0.416** |

(novelty = `0.3 · N · s̄ / 200` with `s̄` the mean of `min(10, 400/(400−k))`
over the N novel visits; the rest read straight off `collector.py`.)

The model reproduces the logged reward across a **5.7× range** with error
≤0.018, so the shaping is understood and the pin can be priced.

**At the pin the persistence bonus PAYS +0.196/step while `wall_penalty`
charges only −0.093 — a ratio of 2.1.** The pin is not an unpunished state. It
is a *rewarded* one, and the term rewarding it is the one meant to encourage
ballistic exploration.

#### Why: persistence is scored on the COMMANDED action

`collector.py` computes `cos(a_t, a_{t−1})` from `result["move_action"]` — what
the policy asked for, before the norm clamp and the arena clip. A wall-pinned
agent asks for a rock-steady heading (`straightness` **0.981**, the highest
number in this document) and realizes **0.09** of it. It collects the full
straight-line bonus for standing still.

**This is the same commanded-vs-realized confusion §9.1 already caught once**,
where referencing the commanded magnitude made `strategy_efficiency` read 3.97
for a policy merely sitting at its speed cap. Same bug class, different
consumer, still live in the reward function rather than in a metric.

#### Why raising `wall_penalty` is the wrong first move

1. It would have to reach **>0.24** (from 0.10) merely to make the pin
   unprofitable — it is bidding against +0.196/step of persistence income.
2. At 0.24 it charges the *healthy* policy **−0.031/step** for legitimate
   perimeter work. `p20_e` sits at `edge_frac` 0.126 and uniform occupancy is
   0.19, so the ring is not a place a good explorer avoids.
3. **We have already measured what an edge-avoiding explore policy looks
   like.** It is `p20_e_kcap`: `edge_frac` 0.061 and **12% less coverage**
   (§18.4). Raising `wall_penalty` pushes toward a failure mode this
   workstream has already characterised.

#### The change — `--persistence_realized`, default OFF

Score the bonus on the realized displacement instead. A pinned agent's cosine
collapses (its realized step is ~0, and a zero step scores 0 rather than the
1.0 a 0/0 cosine would give); an unobstructed policy is untouched, because
realized == commanded whenever neither the clamp nor the clip bites — which for
converged `p20_e` is 97% of steps (`clip_frac` 0.031). It removes the pin's
income **without taxing the walls at all**.

**Default `False`, deliberately.** Every run from P1 to P20 trained under the
commanded-action version, and this launcher's own header says an inherited
default that moves silently is the thing spelling every knob out is meant to
prevent. Turning it on is a variant, not a new baseline.

Implementation notes worth keeping:

- The shaping term uses its **own** `prev_disp_shaping` buffer rather than the
  existing `prev_disp_t`.

  > **CORRECTION — Jack: "the rollout ends on teleport anyway, so
  > `prev_disp_t` not resetting shouldn't matter." He is right, and I
  > overstated it.** The first draft of this note called the missing reset a
  > latent bug. It is not reachable in anything this project runs, for **two
  > independent reasons**:
  >
  > 1. `_reset_mask` is `at_goal & contract.teleport & contract.reset_state`,
  >    and `reset_state_on_teleport = False` is **fixed by instruction** —
  >    confirmed in `p20_e`'s own `run.json`. The mask is all-False, so the
  >    reset block **never executes**, and `prev_reward_t` / `prev_action_t`
  >    are not reset either. The asymmetry is textual, not behavioural.
  > 2. In the explore regime there is nothing to teleport *from*:
  >    `EXPLORE_GOALS_OFF=1` means the env reports **no at-goal rows at all**
  >    (`collector.py:142`).
  >
  > A later correction to reason 2: `explore.py:93` computes
  > `ends_on_goal = (not goals_off) and ends_on_goal`, so with goals off
  > `--explore_ends_on_goal` is **vacuous** — and `ExploitRegime` never sets
  > `ends_on_goal` at all, taking `RolloutSpec`'s default `False`
  > (`stages.py:109`). **So the row-freezing path is live in nothing this
  > phase runs**, and citing it as a reason was wrong. Exploit genuinely
  > teleports and continues; reason 1 is what covers it.
  >
  > Note this makes the *current* behaviour coherent rather than merely
  > harmless: the reset is gated on `reset_state` precisely so the enrichment
  > buffers share the hidden state's fate, and zeroing them while the
  > recurrence deliberately carries across would be the inconsistent choice.
  > **The asymmetry becomes real only if `RESET_STATE_ON_TELEPORT` is ever set
  > to 1**, when `prev_reward_t` / `prev_action_t` would be zeroed and
  > `prev_disp_t` silently would not.
  >
  > So nothing leaks today. The separate buffer stays because it costs nothing
  > and is correct under **any** contract, including if
  > `reset_state_on_teleport` is ever turned on — not because the shared one
  > is broken.
- Tests: `TestPersistenceRealized` in `test_audit.py`. The first case is a
  regression test on the *old* behaviour — it asserts the commanded version
  keeps paying ~1.0/step to an agent jammed against a wall — so the bug stays
  documented rather than merely fixed. Both drive due east for 12 steps in a
  6-wide arena, because `collect_rollout` calls `vec.reset_all()` and a staged
  start position does not survive; any start therefore ends pinned, and the
  two ends of the same rollout give both assertions.

#### `p21_pr` — staged, NOT launched

`p20_e` with one bit flipped, `explore:300`, everything else identical, so
**`p20_e`'s own eval series is the control and needs no re-run**.

**Prediction on record:** the pin clears earlier than `p20_e`'s u75, and final
coverage is unchanged or slightly better. **Score it with
`analysis.nav_tri.explore_traj` on the u25/u50/u75 checkpoints**, the way §18.7
was measured — not from the coverage curve, which cannot distinguish "unpinned"
from "pinned but lucky".

**Falsifier:** if the pin clears but coverage ends *lower*, the bonus was doing
something real at the walls that this removes, and the answer is a smaller
`persistence_bonus` on realized rather than the swap.

---

## 19. The explore metric — swept area replaces cell coverage

Jack, opening an RL-theory discussion on explore: *"we want to train the agent
to do optimal behavior to find a hidden goal, no?"* Yes — and taking that
seriously changes the metric, because `mean_coverage` measures something else.

**From 2026-09-01 the headline explore metric is `swept_coverage`.**

### 19.1 What it is

At every step the agent occupies a continuous position and would detect a goal
anywhere within `goal_radius` of it — that is exactly what `at_goal` tests, an
**L2 ball on the continuous position** (`env.py:170`, whose docstring already
warns against "the snap-square vs L2-ball mismatch"). `swept_coverage` is the
fraction of the arena covered by the **union** of those discs along the path.

For a uniformly-placed goal it is **P(the goal was findable this episode)** —
not a proxy for it, equal to it.

`mean_coverage` counts unique *snapped cells* the agent's position landed on.
That has a detection radius too; it is just an accidental one — the half-width
of a grid cell — and it disagrees with the real one wherever the stride is long.

### 19.2 Why it matters: cell coverage hides the speed axis

Billiard, 200 paths, T=200, r=1.0:

| speed | cell cov@200 | **swept cov@200** | E[T] by cells | **E[T] by swept** |
|---|---|---|---|---|
| 0.50 | 0.246 | 0.391 | 174 | 157 |
| 1.00 | 0.383 | 0.633 | 159 | **127** |
| 1.25 | **0.399** | 0.712 | 157 | 114 |
| 1.50 | **0.401** | 0.771 | 157 | 104 |
| 2.00 | 0.384 | 0.839 | 158 | **88** |
| 3.00 | 0.397 | 0.881 | 157 | **70** |

**Cell coverage says speed barely matters** — E[T] flat at ~157 across a 6×
range, with a shallow peak at 1.25–1.5. **Swept area says speed is the dominant
variable** — E[T] falls 127 → 88 → 70 from speed 1.0 to 3.0. A long stride
sweeps the same corridor but lands on fewer cell *centres*, and cell-counting
charges it for ground it actually crossed.

#### RETRACTION — §2.1 and §18.2's "the speed cap is free"

§2.1 says the [0.5, 2.0] band "brackets the explore optimum — billiard coverage
peaks at |a| ~ 1.0–1.5 and FALLS above it, so this costs explore nothing", and
§18.2 went further, calling the [0.5, 1.0] cap a measured *improvement*
(billiard cps 0.757 at 1.0 against 0.687 at 2.0).

**Both were computed on cell coverage and neither survives the swept version.**
Under the metric that matches how the goal is actually detected, coverage is
monotone increasing in speed and the cap costs roughly **30% of expected
discovery time**. The [0.5, 1.0] band is a deliberate price for physical
realism — Jack's "massive steps are unrealistic" — which is a fine reason, but
it is a price and §18.2 reported it as a win.

### 19.3 Why the endpoint, and not the time integral

Expected discovery time is `E[min(T, 200)] = Σ_t (1 − swept(t))`, the area
*above* the curve — the more fundamental quantity, since it rewards covering
early rather than merely covering. Measured on the real rollouts, it ranks
everything identically to the endpoint:

| | swept@200 | E[T] |
|---|---|---|
| `p20_e` | **0.625** | **124.7** |
| `p20_e_kcap` | 0.537 | 130.3 |

and across six checkpoints (u25 → u700) the two orderings are the same list.
The mean curves do cross (t = 14, 23, 70 — the capped arm leads briefly early)
but never enough to flip a verdict. Our policies share one curve *shape* and
differ only in rate, and when shape is fixed the endpoint determines the
integral.

So the endpoint wins on simplicity and cost — it needs only the final union
mask, not a running fraction at every step.

**The condition under which this stops being safe, and it is not
hypothetical:** `novelty_scale_remaining` pays up to **10× more for late cells
than early ones**, which is precisely a pressure to bend curve shape without
moving its end. If that knob changes, re-check against `E[T]` before trusting
the endpoint.

### 19.4 Why it is not radius-free, though Jack is right to want that

Jack: *"this poses an issue because this should be independent of goal_radius
ideally."* It cannot be, and cell coverage is not either — it silently fixes
`r ≈ 0.5` at the grid's half-width. There is no scale-free notion of "how much
did you observe", and no meaningful `r → 0` limit: passing arbitrarily close to
every point takes unbounded time.

**An attempt at a radius-invariant behavioural statistic failed and is recorded
so it is not retried.** `sweep_efficiency = swept / (2r · path_length)` should
be 1.0 for a perfect lawnmower and radius-free to first order. Measured:

| speed | r=0.5 | r=1.0 | r=2.0 |
|---|---|---|---|
| 0.50 | 0.899 | 0.811 | 0.644 |
| 1.00 | 0.814 | 0.665 | 0.439 |
| 2.00 | 0.646 | 0.440 | 0.247 |
| 3.00 | 0.507 | 0.315 | 0.171 |

It varies by 0.25–0.40 across `r` and is not speed-free either. Two real
geometric causes, neither fixable by bookkeeping: **boundary waste** (the disc
overhangs the wall; loss scales as `r · perimeter / area` = `0.2r` here, which
predicts the observed r-drop), and **arena saturation** (the ratio is bounded
by `arena / 2rL`; at speed 3, r=1 that ceiling is 0.335 and the measurement is
0.315 — sitting *on* it). Normalising by the bound gives 0.81/0.67/0.88/0.94,
non-monotone, so that does not rescue it either.

The radius here is the task's own `goal_radius`, which is the honest choice: a
bigger goal really is easier to find.

### 19.5 `union_swept_coverage`

Jack's ask, the swept analogue of `union_coverage`: the OR of the per-trial
masks within an env, averaged over envs. It answers "given B attempts, how much
of the arena was reachable at all" — a **spread / mode-collapse diagnostic**,
not a single-episode search number. A policy collapsed onto one route scores
`union == per_trial` exactly; that case is pinned by
`test_a_collapsed_policy_gains_nothing_from_more_trials`.

Expect it to saturate *harder* than the cell version (already 0.951 / 0.923 /
0.876), since discs are strictly larger than cells. Good collapse detector,
poor discriminator between healthy policies. `swept_coverage` stays the number
that separates them.

### 19.6 What this does not change

`mean_coverage`, `cells_per_step`, `union_coverage` and `redundancy` are all
still computed and logged. Every number in §9, §18 and before them stands as
measured — they were correct readings of cell coverage. What changes is which
one leads, and the specific conclusions that turned on the speed axis (§19.2).

**Golden note.** Adding the two keys changed
`evaluators.npz[evaluate_exploration__agg_keys/agg_vals]` from shape (18,) to
(22,) — 2 keys × 2 distractor levels. `gen_golden --check` across all five
golden files showed **no other difference**, in any file or any other array, so
the change is provably additive: no existing evaluator number moved.
Regenerated with `--only evaluators`.

---

## 20. What κ actually does — and the evaluation it never reaches

Jack, on the intermittent-search proposal: *"wait but kappa is learned and a
function of state right?"* Yes. `STATE_DEPENDENT_STD=1` and the trunk is an
RNN, so "state" carries history: the policy can hold high κ through a
relocation and drop it through a scan with no new machinery. **§19's claim that
the parameterization is "confined to a single scale by construction" is
retracted.** The question is not expressivity but use.

### 20.1 κ does not affect evaluation at all

Every eval in this document runs `deterministic=True` — `world_setup.py:614`
calls `evaluate_exploration` without the flag, which defaults True, and
`behavior_probe.py:245` does the same. A deterministic action is the
distribution's **mean**; κ only sets the spread, which is never drawn.

Measured, on 64 matched trials of `p20_e_kcap` under two different κ ceilings:

| | κ p50 | coverage | `edge_frac` | `straightness` | `signed_turn` | speed |
|---|---|---|---|---|---|---|
| ceiling 5.0 | **44.6** | 0.3340 | 0.0548 | 0.9782 | 0.1194 | 0.9506 |
| ceiling 2.5 | **11.6** | 0.3335 | 0.0547 | 0.9782 | 0.1191 | 0.9505 |

**κ moves 4× and every behavioural statistic is identical to four decimals.**

So the κ story throughout §17–§19 is a story about **training**, never about
evaluated behaviour: the cap changes which trajectories PPO collects, which
changes the mean-direction policy that is learned, and the 12% coverage gap is
that learned difference. It is not the cap "capping straightness at eval" —
straightness at eval is a property of the mean policy.

### 20.2 A harness bug, found by this and harmless because of it

`explore_traj.py` and `behavior_probe.py` both built **every** agent from
`cks[0]`'s config while loading each checkpoint's own weights. With `p20_e`
listed first, `p20_e_kcap` was instantiated under `log_kappa_max=5.0` instead
of its own 2.5, and its weights emitted the κ its training clamp had been
suppressing — 44.6 against a 12.2 ceiling, which is impossible, since
`shrink = sq/(sq+soft_rel) ∈ [0,1)` can only *reduce* κ.

`behavior_probe` guards **world** keys against exactly this ("does not share a
world … Probe them separately") and had no guard for **agent** keys. Same blind
spot in both files. Fixed: each agent is built from its own config, and any
differing agent knob is printed rather than silently absorbed.

**It invalidated nothing.** Because evaluation is deterministic, the wrong κ was
never used — the table above is the measurement of that. An earlier note
claiming §18.4 and §18.6 were "taken off an agent the run never produced" was
wrong in its consequence and is withdrawn; those sections stand. The fix
matters going forward, for stochastic probes and for any κ-derived statistic —
which is exactly what §20.3 is.

### 20.3 The policy modulates κ, and never leaves the ballistic band

`p20_e` (uncapped), 64 trials × 200 steps, correct config:

| | value |
|---|---|
| κ p05 → p95 | **22 → 97** (4.3×) |
| within-episode sd / between-episode sd | 0.035 / **0.0025** |
| autocorrelation time of κ | **~3 steps** |

So the modulation is real and it is *within* episodes — 14× more within than
between. But translate it into behaviour: **κ=97 is 5.8° of turn per step,
κ=22 is 12°.** Both are committed. A scanning or tumbling mode needs κ of
order 1 — near-uniform, 60°+ — and **the policy never goes there.** It
modulates between "very straight" and "slightly less straight", and the
modulation decorrelates in ~3 steps against a run length of ~κ ≈ 50.

`p20_e_kcap` has no modulation at all: κ p05 = p50 = p95 = **11.6** against a
12.2 cap, autocorrelation τ ≈ 1 step. It is welded to its ceiling — which is
the cleanest evidence in the document that the cap binds.

**Conclusion: the action space can express intermittent search and the learned
policy occupies a narrow, uniformly ballistic corner of it.** Adding a mixture
component would not help — nothing prevents low κ today and the policy does not
use it. That points at optimization or reward, which is where §6.9 pointed too.

### 20.4 OPEN — is a deterministic eval the right measurement for explore?

Every explore number in this document — `cells_per_step`, the new
`swept_coverage`, §18.4's 12% gap, the whole P20 comparison — is the
**noiseless mean policy**, not the policy that was trained.

For exploit that is defensible: the mean is the intended behaviour. **For
explore it is questionable.** The policy earns its training reward through
*sampled* trajectories, and directional noise is functionally part of how a
searcher covers ground. Removing it measures a policy that was never trained
and would never be deployed. `project_incontext_teacher_swap`, recorded the
same day in a different workstream, puts it bluntly: *"ALWAYS evaluate
uncertain policies sampled, not deterministic."*

**And it could move a live conclusion.** The κ-capped arm's whole difference is
that it carries more directional noise — noise that a deterministic eval
discards. Under a sampled eval its extra spread might be *functional* for
coverage, so §18.4's 12% gap could shrink or invert. That is a hypothesis, not
a finding; the test is `evaluate_exploration(..., deterministic=False)` on both
final checkpoints, which needs no retraining.

---

## 21. The adaptation-novelty proposal — killed by a pre-diagnostic

Jack objected that a `Δswept` reward is "too direct" and asked for something
semi-biologically-plausible. The proposal was **firing-rate adaptation on the
place code**: replace the oracle novelty reward — which reads a ground-truth
`visited_cells` array the agent cannot see — with

    a_t = λ·a_{t−1} + (1−λ)·φ_t          (adaptation trace)
    n_t = 1 − cos(φ_t, a_{t−1})          (the un-adapted response)

where `φ_t` is the encoder output at the current position, already computed
every step at `collector.py:263`. Repetition suppression, local, online, no
oracle, one extra vector.

`analysis/nav_tri/adaptation_probe.py` tests it offline, with no training.
**It fails**, and for the opposite reason to the one predicted.

### 21.1 The kernel is graded — the stated risk was wrong

The worry on record was that at `encoder_gain=100` the code would be a near
lookup table, so `1 − cos` would read ~1.0 everywhere and no λ could make the
signal graded. Measured over every cell pair in 3 held-out envs:

| `|x−y|` | 0–2 | 2–4 | 4–5 | 5–7 | 7–10 | 10–14 | 14–30 |
|---|---|---|---|---|---|---|---|
| cos(φ(x), φ(y)) | **0.995** | 0.970 | 0.944 | 0.902 | 0.830 | 0.723 | **0.568** |

Smooth decay across the whole arena, contrast 0.294, no cliff. **The gain-100
code is not a hash-like lookup table**, which also corrects a loose claim made
elsewhere in discussion that the vectorhash decorrelates nearby positions.

### 21.2 But the signal does not track novelty — R² 0.016

On real `p20_e` rollouts (3 envs × 8 trials × 200 steps), against the binary
first-visit flag (which fires on 0.764 of steps):

| λ | mean | sd | corr(new) | **R²(new)** | new-cell mean | revisit mean | gap |
|---|---|---|---|---|---|---|---|
| 0.90 | 0.066 | 0.032 | 0.098 | 0.010 | 0.0678 | 0.0604 | 0.007 |
| 0.95 | 0.096 | 0.050 | 0.106 | 0.011 | 0.0990 | 0.0867 | 0.012 |
| 0.99 | 0.143 | 0.090 | 0.128 | **0.016** | 0.1489 | 0.1219 | **0.027** |

The predicted failure was `R² → 1` — the signal being binary novelty in
disguise. The actual result is `R² → 0`: it is **nearly orthogonal to
novelty**. At the best λ the new-cell/revisit gap is 0.027 against a signal sd
of 0.090 — **0.3 sd of discrimination.**

### 21.3 Why, and it is structural rather than a tuning failure

Because the code varies *smoothly* with position (§21.1), the leaky trace is a
low-pass filter of position: `a` converges to the code at the recent **mean
position**. So `1 − cos(φ_t, a)` measures *how far you are from where you have
recently been*, not *whether you have been here*. It is a displacement signal,
not a novelty signal — it cannot tell "ten cells from my recent mean in a
fresh direction" from "ten cells from my recent mean, back over old ground."

> **Novelty is a set-membership question, and a single averaged trace cannot
> represent a set.** Averaging destroys the shape of the visited region.

### 21.4 The condition for reviving it: a SPARSE code

The adaptation trick works when the place code is **sparse**. Sparse codes
superpose without interference, so a leaky sum over visited positions *is* an
occupancy map and `1 − cos` genuinely reads set membership — which is how it
works in a real place-cell system at a few percent activity.

This encoder is the opposite: adjacent cells at cos **0.9946**, opposite
corners of a 20×20 arena still at **0.568**. A dense, highly correlated code
averages into mush.

**So the blocker is neither λ nor `encoder_gain` — it is code sparsity, an
encoder property.** Re-run `adaptation_probe` if the encoder's sparsity or
unique-radius setting ever changes; the module exists for that.

### 21.5 What survives

The *objection* that motivated it stands: the novelty reward reads an oracle
the agent cannot see, and that is the implausible part, not the idea of a
novelty bonus (which is well attested — SN/VTA responds to novelty absent
reinforcement, and a CA3→CA1 comparator is a proposed intrinsic mechanism).
What is refuted is this particular substrate. A learned-predictor novelty
(RND/ICM-style) is untouched by the sparsity argument, because it does not
rely on superposing codes.

---

## 22. The explore policy is a vector field, not a memory-based explorer

Jack, looking at the trajectory wall: *"why does it do the same movement again
in the exact same spot?"* The answer turns out to be the clearest mechanistic
result in this workstream, and it retracts two claims made while getting to it.

Pages: [0 distractors](https://claude.ai/code/artifact/aad8b116-ff1e-403f-8100-a6d12b205c54)
· [10 distractors](https://claude.ai/code/artifact/7ec1e4d3-9df9-457e-a514-0d241fcbfe76)

### 22.1 On a state repeat, it replays

Find every pair of timesteps where the agent is back within 0.5 cells of an
earlier position with heading within 15°, at least 20 steps later, and measure
how far the two continuations diverge. Against random pairs of timesteps in the
same trajectory as the null:

| divergence after k steps | `p20_e` repeat | random null |
|---|---|---|
| k=1 | 0.12 cells | 1.25 |
| k=10 | **1.28** | 10.22 |
| k=24 | **3.05** | 15.93 |

438 repeats in 46 of 100 trajectories (`p20_e_kcap`: **1905** in 94 of 100).
The continuations stay ~8× closer than chance for 25 steps.

**The RNN hidden state is definitely different on the second visit — it carries
the entire history, including having been there — and it barely changes the
action.** The policy is close to a fixed function of (position, heading): a
**learned vector field**. Under deterministic evaluation, same input, same
output, forever.

### 22.2 RETRACTION — "it steers toward novelty" was an itinerary artifact

An earlier measurement compared, at each turn, the 90° cone ahead of the
heading taken against the cone from turning the other way, and found the chosen
side less-visited: 0.197 vs 0.276, with 69.6% of turns choosing the emptier
side, ~50 se. **That measurement stands. The interpretation does not.**

A fixed itinerary produces exactly that correlation with no memory at all: the
field generates the visitation, so wherever the field takes you next is by
construction somewhere you have not been *yet on this itinerary*. The test
could not separate memory from itinerary.

§22.1 separates them, because at a state repeat the visitation history is
completely different and **the action is the same anyway**. So the policy does
not consult where it has been. **Tier 2 — coarse spatial recency — really is
unexploited**, which is where the analysis had it before this correction of a
correction.

### 22.3 What this explains

- **Exact retracing.** The field has closed orbits; enter one and you loop.
- **Consistency across arenas.** One field, applied everywhere — which is
  reasonable, since every arena here is the same 20×20 box differing only in a
  random ±1 wall code (Jack's point). There is nothing to adapt to.
- **The "loose lawnmower" look.** The field is a decent space-filling flow.
  That is a real achievement; it is just not a memory-based one.
- **`p20_e_kcap`'s 4× higher repeat count** (1905 vs 438). Its field is the
  constant curl of §18.6 — a closed orbit almost everywhere.
- **Why retracing is not U-turns.** `p20_e` makes **9× fewer** sharp reversals
  than a billiard (0.21% of steps vs 1.88%) and makes them at grazing
  incidence (0.35) rather than head-on (0.93). It has learned to slide along
  walls rather than bounce off them.

### 22.3.1 The orbits do NOT bound coverage — measured to 1000 steps

Jack: *"shouldn't the state be more variable to enable better deterministic
exploration?"* The worry, and the argument I gave for it, was that a
deterministic policy is an autonomous dynamical system, must converge to an
attractor, and is therefore coverage-bounded. **Measured, that is wrong for
`p20_e`.** 8 envs x 8 trials, 1000 steps:

| step | swept DET | swept SAMPLED | cells DET | cells SAMP |
|---|---|---|---|---|
| 200 | 0.644 | 0.643 | 0.386 | 0.389 |
| 400 | 0.814 | 0.833 | 0.591 | 0.607 |
| 999 | **0.911** | 0.939 | 0.805 | 0.848 |

It covers **91% of the arena deterministically** and is still gaining +0.026
over the last 300 steps. The deterministic/sampled gap is ~3% at 1000 steps and
**zero at 200**.

**The argument was too strong.** It assumed a short limit cycle; a
deterministic continuous-state system can equally have a quasi-periodic,
space-filling orbit, and this field is one. §22's replay is real — trajectories
re-cross and locally repeat — but the repeats do not close into small loops and
so do not bound coverage.

**Consequence: deterministic deployment is fine for `p20_e`**, and the state
does not need to be more variable for coverage's sake. `p20_e_kcap` is the
genuine exception: its constant curl IS a closed orbit, which is exactly why
deterministic eval traps it at 0.333 while its own noise recovers 0.375 (§23).
One arm has a space-filling field; the other has a circle.

**What survives:** a monotone internal accumulator remains the mechanism for
memory-driven exploration, since it makes state repeats impossible. But the
current policy does not need it to explore well — which is also why there is no
gradient pressure to learn it. Those are the same fact.

### 22.3.1 The orbits do NOT bound coverage — measured to 1000 steps

Jack: *"shouldn't the state be more variable to enable better deterministic
exploration?"* The argument I gave for that — a deterministic policy is an
autonomous dynamical system, must converge to an attractor, and is therefore
coverage-bounded — **is wrong for `p20_e`.** 8 envs × 8 trials, 1000 steps:

| step | swept DET | swept SAMPLED | cells DET | cells SAMP |
|---|---|---|---|---|
| 200 | 0.644 | 0.643 | 0.386 | 0.389 |
| 400 | 0.814 | 0.833 | 0.591 | 0.607 |
| 999 | **0.911** | 0.939 | 0.805 | 0.848 |

It covers **91% deterministically** and is still gaining +0.026 over the last
300 steps. The det/sampled gap is ~3% at 1000 steps and **zero at 200**.

**The argument assumed a short limit cycle.** A deterministic continuous-state
system can equally have a quasi-periodic, space-filling orbit, and this field is
one. §22's replay is real — trajectories re-cross and locally repeat — but the
repeats do not close into small loops, so they do not bound coverage.

### 22.4 The caveat that may be the whole story

**This is measured under `deterministic=True`.** During training actions are
*sampled*, and noise perturbs the state enough to break a closed orbit. The
trained policy would not replay like this.

So §20.4's open item is not a methodological nicety: **we may be characterising
a failure mode that exists only because we turned the noise off.** Re-running
these same 100 rollouts with `deterministic=False` settles it and costs
nothing. Until then, "the explore policy is a vector field with closed orbits"
is a statement about the *evaluated* policy, not necessarily the trained one.


---

## 23. RESOLVED — the κ cap costs 3%, not 12%. Deterministic eval was the rest.

§20.4 flagged that every explore number in this document is the noiseless mean
policy, and predicted that the κ-capped arm — whose entire difference is extra
directional spread — might be systematically penalised by that. Measured, on
the same 100 matched rollouts, `--no-deterministic`:

| | deterministic | **sampled** |
|---|---|---|
| `p20_e` coverage | 0.3881 | 0.3870 |
| `p20_e_kcap` coverage | 0.3332 | **0.3745** |
| **gap** | **14.1%** | **3.2%** |

**The cap costs ~3%, not the 12–14% §18.4 reported.** The rest was our
evaluation discarding the noise that makes the capped policy work.

### 23.1 The mechanism is §22's replay, and the noise breaks it

| `p20_e_kcap` | deterministic | sampled |
|---|---|---|
| state repeats | **1905** in 94/100 | **524** in 91/100 |
| replay ratio at k=10 | 0.112 | 0.285 |
| `abs_turn` per step | 0.1445 | 0.3014 |

Deterministically the capped policy is trapped in the closed orbits of §18.6's
constant curl. Its own sampling noise — the noise the cap *forces it to keep* —
perturbs the state enough to escape them. The doubling of `abs_turn` is that
noise doing the work.

So the cap is not mainly a coverage handicap. It is a handicap **to the mean
policy**, and the policy is not deployed as its mean.

### 23.2 §22 survives, for the arm that matters

`p20_e` barely moves: coverage 0.3881 → 0.3870, repeats 438 → 408, replay ratio
0.125 → 0.160. It **still replays on a state repeat under sampling**, so
"learned vector field, not a memory-based explorer" is a property of the policy
and not of the evaluation. Its noise is smaller (κ ≈ 50 ⇒ ~8°/step against the
capped arm's ~16°) and its field has few closed orbits to escape.

### 23.3 What this changes

- **§18.4's headline is retracted to ~3%.** The direction still holds — the cap
  costs something — but the magnitude was mostly measurement.
- **§18.6's constant curl is still real** and still what makes the capped
  policy's mean useless. Sampling rescues the *behaviour*, not the field.
- **Deterministic evaluation systematically penalises policies whose noise is
  functional**, which for a search task is a real bias, not a nicety. That is
  `project_incontext_teacher_swap`'s lesson arriving in a second workstream.
- **Explore should be scored sampled from here on**, or at minimum both ways.
  Every `swept_coverage` number in §18–§19 is the deterministic figure.

**Not established:** whether the training-time evals (the `cells_per_step` and
`swept_coverage` series behind §18.4's 16-of-16 comparison) move the same way.
Those are also deterministic (`world_setup.py:614`), so the *series* comparison
inherits the same bias — but a converged-checkpoint result does not
automatically transfer to every point on a learning curve.


---

## 24. The Tier-2 prize, priced

Jack: *"we do need determinism for when we mix this with exploit no? so I do
want to get tier-2 capability somehow."* Right on both counts, and it makes
`p20_e_kcap` the case that matters rather than `p20_e`: exploit deploys
deterministically (that is where §17.10's 1.013 beeline comes from), and the κ
cap is what exploit *needs* (§17.9: 0.375@u475 → 1.000@u125).

### 24.1 The capped policy's deterministic deficit widens with horizon

`p20_e_kcap`, 8 envs × 8 trials, 1000 steps:

| step | swept DET | swept SAMPLED | gap |
|---|---|---|---|
| 200 | 0.538 | 0.623 | 14% |
| 400 | 0.645 | 0.806 | 20% |
| 999 | **0.748** | **0.935** | **20%** |

Against `p20_e`'s 0.911 / 0.939 (3%). **The capped policy can reach 0.935 — with
noise doing the work.** If it used memory instead it would get there
deterministically. That 25% is the Tier-2 prize, and it is real gradient
pressure — except **training never sees it**, because training is sampled and
collects the 0.935 version. The flaw is only exposed in the regime we deploy in
and never in the one we optimize in.

### 24.2 Two separable problems

**A — the capped policy's deterministic deficit.** Probably fixable without
Tier-2 at all. κ is irrelevant at deterministic deployment (§20.1, measured);
the cap is purely a training-time device for policy-space exploration. So
**anneal `LOG_KAPPA_MAX` 2.5 → 5.0 over training**: on early when exploit needs
it, off late so the policy sharpens and its *mean* is optimized nearer the
deployed regime. One knob, mirroring `EPSILON_ANNEAL_UPDATES`.

**B — Tier-2 proper.** Note where the prize actually is. At the **200-step
operational horizon** `p20_e` gets swept **0.644** deterministically and
sampling adds *nothing* (0.643). A perfect lawnmower at speed 0.96 with r=1.0
sweeps a 2-wide corridor over ~192 cells of path ≈ **0.9**. So the field leaves
~40% on the table at the horizon we train and deploy at, and **no amount of
noise recovers it — only memory does.**

Planned route for B: an **auxiliary visitation head** predicting "is the cell
ahead already visited?" from the hidden state, supervised on the
`visited_cells` array the collector already maintains. Training-time oracle
only; no change to reward, action space, or deployment. It forces the
representation directly rather than hoping reward pressure produces it.

**The success test for B is NOT coverage.** Coverage could rise from a better
field alone. Tier-2 is achieved iff **§22's replay signature breaks** — same
position and heading, different action, because the hidden state now carries
where the agent has been.


---

## 25. P22 — the sensory ablation. The sensor helps, and §6.9 was wrong.

`p22_nos` = `p20_e` with `INPUT_SENSORY=0` and nothing else moved: 74 input
dims → 14. Job 21776318, COMPLETED 700/700, `p20_e` as its own control.

| window | `p22_nos` | `p20_e` | delta |
|---|---|---|---|
| u1–200 (breakout) | 0.274 | 0.424 | **−35.6%** |
| u200–500 | 0.567 | 0.713 | −20.4% |
| u500–700 (tail) | 0.667 | 0.750 | −11.0% |
| **all 28 matched evals** | **0.512** | **0.641** | **−20.2%** |

The ablation wins **2 of 28** points. Final four: **11.2% lower** and **3.4×
more variable** (range 0.162 against 0.047); over the whole back half its eval
sd is **1.9×** the control's. Its `swept_coverage` final-4 mean is 0.5615,
against the 0.625–0.644 measured for `p20_e` on the trajectory sets.

### 25.1 §6.9's prediction is refuted

§6.9 concluded: *"the lawnmower ceiling is not blocked, because §4's B3 already
hands the agent exact self-motion… if P5 plateaus at billiard, the diagnosis is
recurrent capacity or reward shape, not the sensor."* That reasoning was
endorsed here and used to argue the sensor was ~81% dead weight. **It is not.**

What §6 measured was **cross-env displacement decoding**, and it measured it
correctly. What it did not measure is what the sensor does *within* an episode,
which is where the contribution turns out to be.

### 25.2 The shape of the contribution is learning SPEED, then stability

The gap narrows monotonically — 35.6% → 20.4% → 11.0% — so the sensor mostly
buys **time to competence** rather than final coverage. It also buys
**stability**: without it the run collapses intermittently (u250 fell to cps
0.156, near the wall-pinned 0.048 of §18.7) and never becomes as steady.

Both matter more for the interleaved model than for explore alone. §0.0 set the
bar as "good enough **and stable enough** not to poison interleaved training",
and an arm that periodically falls back toward the pin basin is exactly what
would poison it.

### 25.3 What this does not say

It does not say the sensor is used for **place recognition**. §22 measured that
the policy replays on a state repeat, i.e. it does not consult where it has
been — so whatever the sensor buys, it is not visitation memory. The most
likely reading, consistent with both results, is that it helps the policy
**shape a better field faster** (and stay out of the §18.7 wall-pin basin),
not that it supports Tier-2.


---

## 26. P23 — lever A worked. The closed orbit is gone; Tier-2 is not.

`p23_kanneal` = `p20_e_kcap` with `LOG_KAPPA_MAX` ramped 2.5 → 5.0 over 400 of
700 updates, nothing else moved. Job 21787292, COMPLETED 700/700.

### 26.1 The result

| | swept@200 det | swept@999 det | det vs sampled @999 |
|---|---|---|---|
| `p20_e_kcap` (control) | 0.538 | 0.748 | **20%** |
| **`p23_kanneal`** | **0.599** | **0.877** | **3.3%** |
| `p20_e` (uncapped ref) | 0.644 | 0.911 | 3% |

Training evals, final four: swept **0.624**, cps **0.758** — **+9.3%** over the
capped control and **−2.4%** off the uncapped one.

**The structural result is the third column.** §23 measured the capped arm's
deterministic/sampled gap at 20%: its mean policy was trapped in §18.6's
constant curl and only its own noise escaped. Annealed, that gap collapses to
**3.3%**, matching `p20_e`. **The closed orbit is gone.**

This is what §24.2 predicted from §20.1's measurement that κ does not affect a
deterministic action: the cap is a *training-time* device, so keeping it early
(where §17.9 showed exploit needs it) and lifting it late costs almost nothing
in deterministic explore.

### 26.2 It is NOT Tier-2, exactly as predicted

Replay probe on the final checkpoint: divergence after a state repeat is
**1.196 cells at k=10 against 10.127 for random pairs — ratio 0.118**, against
`p20_e`'s 0.125. **The policy still replays.** The hidden state still does not
change the action.

So A delivered a **better vector field**, not a policy that uses memory. That
was the prediction on record in the variant, and it is why B remains the
interesting arm: §24's 200-step headroom (0.644 → ~0.9 for a perfect
lawnmower) is untouched by this.

### 26.3 Caveats

- **Stability.** Final-4 range is 0.074, against the capped control's
  remarkably tight 0.007 and `p20_e`'s 0.047 — the least steady of the three.
  §0.0's bar is "good enough **and stable enough**", so this matters if
  interleaving turns out to be stability-limited.
- **A transient at the ramp endpoint.** swept dipped to 0.397 at u400, exactly
  where the ramp completes, and recovered within one window. A longer or
  smoother schedule is the fix if it recurs.
- **Not tested here: what this does to EXPLOIT.** The whole justification for
  keeping the cap early is §17.9's exploit unlock. This arm is explore-only,
  so it shows the anneal is safe for explore and says nothing about whether
  exploit still converges at u125 under it. That is the interleaved run's
  question, and it should be checked before the anneal is adopted as default.


---

## 27. P24 — lever B FAILED. The representation was there and the policy ignored it.

`p24_aux` = `p20_e` plus the auxiliary visitation head (8 compass cells at
radius 3, BCE weight 0.5), nothing else moved. Job 21789497, COMPLETED 700/700.

### 27.1 The success test, which was set before launching

**Not coverage.** Coverage could rise from a better field alone, which is lever
A's job. Tier-2 is achieved iff §22's **replay signature breaks** — same
position, same heading, *different* action.

| replay divergence at k=10 (cells; lower = more replay) | vs random | ratio |
|---|---|---|
| `p20_e` (control) | 1.275 / 10.216 | 0.125 |
| `p23_kanneal` (lever A) | 1.196 / 10.127 | 0.118 |
| **`p24_aux` (lever B)** | **1.170 / 10.184** | **0.115** |

**It did not break.** Identical to the control. The policy still emits the same
action at the same (position, heading) despite the hidden state carrying a full
episode of different history.

### 27.2 And it cost

| | swept@200 det | swept@999 det | cps final-4 | volatility u300+ |
|---|---|---|---|---|
| `p20_e` | **0.644** | **0.911** | **0.777** | sd 0.030 |
| `p24_aux` | 0.607 | 0.838 | 0.671 (**−13.7%**) | sd 0.097 (**3.3×**) |

Across all 28 matched evals: −9.8%, winning 11 of 28 points.

### 27.3 The informative part: the representation WAS there

`aux_visited_loss` fell 0.632 → **0.367**. The head predicts 8-direction
visitation *from the trunk's own features*, so those features provably contain
visitation information — and the policy head, reading the identical vector,
ignores it.

> This is exactly the gap §7.7.2 named about `chart_frac` and could not test:
> *"It does not say a policy trained with the extra channel would use it."*
> Now tested, on a different quantity. **It does not.**

**A note on `aux_visited_loss` as a metric:** it is confounded with coverage.
Early in an episode almost nothing is visited, so the target is nearly all-zero
and trivially predictable; as coverage improves the target becomes genuinely
mixed and the balanced BCE *rises*. It ended at 0.893, higher than it started.
It says the head is wired and training, and nothing about progress. It should
not have been the headline signal in the run watch.

### 27.4 What this rules out, and what it changes about the fallbacks

This is **not** a pressure problem and **not** a representation-availability
problem:

- There is already **0.644 → ~0.9** of reward headroom at the 200-step horizon
  (§24), so PPO has gradient available and is not taking it.
- The information is already in `features`, and the policy ignores it.

So the field is a strong **local optimum**, and a memory-conditioned policy is
a qualitatively harder function to find. **B2 (close the train/deploy gap) and
B3 (make the ceiling bite) both address pressure, and pressure is not what is
missing.** Running either next would be answering a question that is not the
one open.

### 27.5 The diagnostic that splits what remains

Hand the 8-direction visitation vector to the policy as an **actual input
channel**. That collapses "use memory" from *learn to read your own hidden
state* down to *learn to weight an input*. It is an oracle at test time and
therefore **not a shippable answer** — it is a diagnostic, and it splits the
remaining possibilities cleanly:

- **Coverage improves** → the bottleneck is representation-to-policy. The fix
  is architectural (how the policy reads state), not more pressure.
- **Coverage does not improve** → the policy cannot exploit visitation *even
  when handed it*, which is a much deeper result and would say Tier-2 is not
  reachable by this route at all.


---

## 28. The recurrence curve — the primary orbit diagnostic

`analysis/nav_tri/recurrence.py`. **`mean |p(t) − p(t+τ)|` as a function of τ.**
An orbit of period T shows a clear MINIMUM at τ = T; a path that merely wanders
shows no post-rise dip. Jack, after four rounds of me building indirect
statistics: *"i am looking at the trajectories lol… capped is clearly just
looping."*

### 28.1 The measurement

100 trajectories per arm, 200 steps, deterministic:

| τ | 20 | 30 | 40 | **50** | **54** | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| `p20_e_kcap` | 13.67 | **14.47** | 10.98 | **5.30** | **4.06** | 6.18 | 11.84 | 13.91 |
| `p20_e` | 11.82 | 12.77 | 11.93 | 10.22 | — | 8.12 | 9.33 | 11.08 |

`p20_e_kcap` is **14.5 cells from where it was 30 steps ago and back within 4
cells 54 steps later** — dip depth **10.66**, in 98/100 trajectories. `p20_e`
has no post-rise dip at all in the mean curve.

τ = 54 against the curl's predicted period 2π/0.121 = **51.9**. So §18.6's
rotation-period account was right.

### 28.2 It corrects a retraction

§27 recorded the mechanism as "open" because I validated the period story
against **revisit lag** and got a null (peak lag 70.8 vs 76.7 for the two arms,
no within-arm correlation with `signed_turn`). **That was a false negative, and
the retraction was wrong.**

Cell-based revisit lag cannot see this orbit because the orbit **precesses**:
the agent returns to the same *region*, within 4–5 cells, which is a different
snapped cell. I had half-noticed this — §23.1 already said the capped arm
"sweeps an annulus, not a single closed curve" — and then chose an instrument
blind to exactly that.

**The lesson is instrument choice, not more statistics.** `signed_turn_mean`
needed three corrections; `straightness` is unsigned and reads a 6.9°/step curl
as straight; windowed |Σdθ| is contaminated by wall bounces (billiard null 59%);
revisit lag is blind to precession. The recurrence curve has none of those: no
null, no window size, no signed/unsigned ambiguity, and it works on the
continuous position.

### 28.3 How to read it

- **Trust the AGGREGATE curve**, not the per-trajectory count. Individual noisy
  minima sit at scattered lags and cancel in the mean, which is exactly what
  distinguishes a real orbit from noise. `p20_e` shows 91/100 "orbiting"
  per-trajectory while its aggregate correctly shows none; the module prints a
  warning when those disagree.
- `DIP_CELLS = 3.0` (15% of the arena). 1.0 was far too lenient.
- `period_iqr` is the consistency check: a real orbit has the same period
  across trajectories, which is why the dips reinforce rather than cancel.

Tests: `test_recurrence.py`, 9 cases — recovers a known period, finds a
**precessing** orbit (the §27 failure), finds a 6.9°/step curl that reads as
straight, and does **not** fire on a straight run or a random walk.


---

## 29. P25 — the visitation oracle buys SPEED, not ceiling

`p25_visin` = `p20_e` plus `--input_visited`, nothing else moved. Job 21812138,
COMPLETED 700/700. **Not a shippable config** — the channel is an oracle at
test time.

### 29.1 Result

| | swept (final-4) | cps (final-4) | reaches control's converged level |
|---|---|---|---|
| `p20_e` (control) | 0.625–0.644 | 0.777 | u700 |
| **`p25_visin`** | **0.629** | **0.774** | **u300** |

**Identical ceiling, ~2.3× faster, and steadier** (u225–300 range 0.022 against
the control's 0.047 at convergence). It also skips the §18.7 wall-pin phase
entirely: swept 0.408 at u25, where the control is 100% pinned at 0.048.

### 29.2 What the oracle actually was — read this before citing the result

Jack: *"wait how did the oracle work?"* It is **8 bits**: the 8 compass cells
at radius **3** from the agent's snapped position, each "have I been there
before?", binary, no recency.

That is **not a visited map**. It is 8 samples of a 400-cell set on one small
ring. It cannot express "the northwest quadrant is untouched" or "I am in row 4
of a sweep."

**So the claim this licenses is narrow:** *local* visitation at radius 3 is not
the binding constraint on coverage. A richer signal — a coarse occupancy grid,
or a direction-to-nearest-unexplored vector — was never tested. An earlier note
here called this "the strongest possible version of a less memoryless state";
that was wrong and is withdrawn.

### 29.3 What DOES survive: memorylessness cannot be the cap

Structural, not experimental, so the thin oracle does not touch it.

**A boustrophedon is memoryless.** Sweeping a known rectangle is a function of
position alone — east on even rows, west on odd, step north at the wall. Row
parity is read off the y-coordinate, not off memory. So a *memoryless* field
can score ~0.9 at 200 steps.

**Therefore memorylessness cannot be what caps us at 0.644, because the target
behaviour is itself memoryless.** §22's replay finding explains the orbiting
story (§18.6, §28) and does not explain the coverage ceiling.

### 29.4 The two candidates left, and the test that separates them

Not memory (§29.3), not orbiting (`p20_e` has no recurrence dip, §28), not the
κ cap (`p20_e` is uncapped). Remaining:

1. **Localization.** A boustrophedon needs to know *which row you are in* —
   absolute position, not relative displacement. The agent has exact
   self-motion and no anchor. §25's ablation is suggestive: removing the wall
   code cost 20%, and it is the only absolute-position signal available.
2. **Optimization.** The wandering field is a strong local optimum and the
   lawnmower basin may be unreachable by PPO regardless of what is available.

**The test: hand the policy its absolute position, 2 dims.** Same machinery as
`input_visited`. Coverage jumps toward 0.9 → localization was the blocker;
coverage unchanged → optimization. That is the experiment that should have
followed the boustrophedon argument, which was available before P25 was
launched.

## 30. The state probe — is the agent storing anything, and does it use it?

`analysis/nav_tri/state_probe.py`. Runs on any checkpoint, any channel set,
either mode:

    PROBE=state CKPTS="a.pt b.pt" TRIALS=16 sbatch hopfield_nav/run_nav_tri_probe.sh

Jack asked for "a test or metric that can tell us clearly whether the agent is
learning to store useful information in state." The reason this is two numbers
and not one is §27: **content without use is possible, and we have already
produced it.** `p24_aux` drove `aux_visited_loss` 0.632 → 0.367 — the trunk
learned to represent local visitation — while the policy reading that same
vector went on ignoring it (replay ratio 0.115 vs the control's 0.125). A
decoding-only diagnostic would have scored that a success.

### 30.1 CONTENT — ridge probes, reported as ΔR²

Linear probes from the hidden state `h` to `pos`, `start_pos`, `elapsed`,
`coverage`, `visited8`, `heading`. Two design points do all the work:

**Scored as ΔR² = R²([obs, h]) − R²(obs), never as R²(h).** The trunk sees the
observation and passes it forward, so `h` decodes heading, the recall signal,
and (under `input_abs_position`) position outright whether or not it is
remembering anything. Only what `h` adds *beyond the current observation* is
storage. `R²(h)` is printed anyway, because seeing it high next to a ΔR² of
zero is what makes the distinction concrete.

**Fit and scored on disjoint TRIALS.** Consecutive hidden states are nearly
identical, so a split on timesteps lets the probe memorise its own test set and
report memory that is not there.

Two of the targets are controls. `heading` is in `prev_action` outright, so
R²(obs) ≈ 1 says the machinery works. `start_pos` is the opposite: constant
within an episode, present in no channel, so anything above 0 there is path
integration and nothing else.

### 30.2 USE — hold the observation fixed, splice the state

The policy is f(obs, h). Swap each half from a different episode and measure how
far the **deterministic** action moves (deterministic because that is what the
evaluation protocol executes, and because κ does not enter it, §20.1):

| | what it measures |
|---|---|
| swap BOTH | the natural spread of the action — **the scale**; everything else is a fraction of it |
| swap STATE only | `state_influence` — the headline |
| swap STATE, same step index | the same, with the clock controlled out |
| swap OBS only | `obs_influence` |
| zero the state | the episode-start counterfactual |
| own-episode donor τ back | the lag curve — the on-manifold version |

`state_share = Δstate / (Δstate + Δobs)` is the one-line summary: what fraction
of the action is driven by memory rather than by what is in front of the agent.

The cross-episode donor may be **off-manifold** for this observation, so the
action could move for reasons that are not "the state is used" —
`state_influence` is an **upper bound**. That is the useful direction when the
finding is that it is near zero, and the lag curve is the on-manifold control
when it is not.

### 30.3 How to read the two together

| content | use | reading |
|---|---|---|
| high | ~0 | the trunk represents history and **the policy ignores it** — §27's lever-B failure. The fix is in the readout, not the representation. |
| ~0 | — | nothing is being stored. The fix is upstream of the trunk: input, horizon, or objective. |
| high | high | the state carries history *and* changes the action — Tier-2 (§24). |

Thresholds in the printed verdict are ΔR² > 0.05 and `state_influence` > 0.15;
they are labels on a continuum, and the table is the thing to read.

### 30.4 Relation to the replay probe (§22)

The replay ratio is the coarse version of the same causal question: it compares
two *naturally occurring* visits to the same (position, heading) and asks
whether the continuations diverge. The splice is sharper because it holds the
observation **exactly** fixed and varies only `h`, instead of relying on finding
near-repeats — and because it comes with a scale (the both-swap) that makes the
number comparable across agents with different action magnitudes.

### 30.5 First run — four arms at u700 (job 21826076)

6 envs x 16 trials x 200 steps, explore, deterministic. Trunk is a 1024-unit
vanilla RNN, obs 74 (82 for `p25_visin`).

**The tool's own controls fire correctly**, which is the first thing to check:
`heading` scores R²(obs) = 1.000 on every arm (the observation probe works) and
ΔR² = −0.000 (it correctly adds nothing); `visited8` scores R²(obs) = 1.000 on
`p25_visin` **and only there**, which is the arm that takes it as an input
channel.

| ΔR² | `p20_e` | `p20_e_kcap` | `p24_aux` | `p25_visin` | |
|---|---|---|---|---|---|
| pos | 0.592 | 0.123 | 0.058 | 0.144 | [!] |
| start_pos | 0.000 | −0.004 | 0.017 | −0.009 | |
| elapsed | 0.331 | 0.132 | **0.838** | 0.086 | [!] |
| coverage | 0.362 | 0.170 | **0.879** | 0.086 | [!] |
| visited8 | 0.085 | 0.043 | 0.119 | −0.000 | [!] |
| heading | −0.000 | −0.000 | −0.000 | −0.000 | |

| USE | `p20_e` | `p20_e_kcap` | `p24_aux` | `p25_visin` |
|---|---|---|---|---|
| state_influence | 0.210 | 0.191 | **0.311** | 0.145 |
| same-step swap | 0.200 | 0.183 | 0.235 | 0.135 |
| state_share | 0.183 | 0.175 | **0.281** | 0.132 |
| lag τ=20 | 0.264 | 0.246 | **0.404** | 0.165 |

### 30.6 The state is a LOCALIZER AND A CLOCK, not a memory

For `p20_e`: total decodability R²(both) is 0.659 for position, 0.331/0.362 for
elapsed/coverage — and **−0.002 for `start_pos`**, on every one of the four
arms. The agent does not know where its episode began.

`elapsed` and `coverage` are near-duplicates as targets (coverage grows
monotonically with t), and they track each other to within 0.03 on every arm,
so read them as one thing: a clock.

So `h ≈ f(current position, time)`. Both are state-like; neither is *history*.
A policy that is a function of (position, t) is still a vector field — a
time-varying one. **This is consistent with §22's replay finding and sharpens
it**: the state is not empty, it is a localizer plus a clock, and the reason
the policy replays is that the episode-specific part of it is what is missing.

It also bears on §29.4 before `p26_abspos` lands: the agent already carries a
position estimate at R²(both) = 0.66, so "the policy could not know where it
was" is weaker than §29.4 assumed. Not settled — 0.66 is not 1.0, and a linear
probe is a lower bound on what the policy could use — but the abs-position
oracle is not handing over something absent.

### 30.7 §27's "the policy ignored it" is too strong — with an off-by-one caveat

`p24_aux` has **more** content and **more** use than the control: elapsed
0.331 → 0.838, coverage 0.362 → 0.879, state_influence 0.210 → 0.311,
state_share 0.183 → 0.281, lag-τ=20 0.264 → 0.404. The policy did not ignore
the state; it read it more.

What the aux head actually built was a much better **clock**. The thing it was
trained on barely moved: `visited8` R²(both) 0.138 → 0.156. And a better clock
does not help you sweep, which is why §27 saw no coverage gain.

**The caveat, and it is load-bearing:** the aux head reads `features`, which for
a 1-layer trunk is `h_{t+1}` — the state *after* seeing obs_t. This probe reads
`h_t`, the state going *in*, because the action at t is f(obs_t, h_t) and that
is the causal question. So the `visited8` row is **not** the same quantity as
§27's `aux_visited_loss` and does not by itself refute it. Recording `h_next`
alongside `h_in` would make that comparison exact; it is a small change and has
not been made.

The elapsed/coverage and all the USE numbers are unaffected by this — they are
measured on the same `h_t` across all four arms.

### 30.8 An independent hint at §28's orbit, and the confound the run exposed

`p20_e_kcap`'s lag curve falls hardest at long τ: 0.246 at τ=20 → **0.116** at
τ=50, a 53% drop, against `p20_e`'s 0.264 → 0.198 (25%). §28 measured `kcap`'s
orbit period at ~54, and at τ ≈ the period the state has come back near itself,
so splicing it in changes little. Two unrelated measurements pointing at the
same number. **Suggestive, not established** — every arm falls somewhat at
τ=50 — and a finer grid over τ=35–70 would settle it cheaply.

**The confound, found by running it:** R²(obs) for position is 0.067 on `p20_e`
and 0.728 on `p24_aux` **with the same 74 input channels**. The observation
baseline is not a property of the channel set — it is a property of the states
the agent visits, and a narrow, orbit-like distribution makes position linearly
decodable from the sensory code in a way a broad one does not. So where the
baseline moves, a ΔR² gap is a **headroom** difference, not a storage
difference. `cross_report` now flags those rows `[!]` and points at R²(both),
rather than leaving it to be noticed.

### 30.9 `--content_h` — closing the off-by-one

§30.7's caveat is now a flag rather than a limitation. `rollout(record_state=True)`
records both hidden states, and `--content_h` picks which one the CONTENT
probes read:

| | reads | right for |
|---|---|---|
| `in` (default) | `h_t` | the causal question — the action at t is f(obs_t, h_t), so this is matched to the USE half, which always uses `h_t` |
| `out` | `h_{t+1}` | comparing against a supervised head bolted onto the trunk, which reads `features` (verified: `ppo.py` calls `agent(mb_obs, return_features=True)` and `agent.visited_logits(features)`) |

USE always splices `h_t`; splicing the post-step state would answer a question
nobody asked.

**Every §30.5 number is `--content_h in`.** The `visited8` row there is
therefore still not §27's quantity — re-running the four arms with
`--content_h out` is what would settle whether the aux head's 0.632 → 0.367
shows up as linear decodability. That has not been run.

### 30.10 The baseline ladder — three controls, each catching the one before it

`delta` alone is too weak a bar, and each rung was added because the previous
one produced a claim that did not survive scrutiny. Every rung is the same
mechanism: put the cheaper explanation into the BASELINE and ask what `h` still
adds beyond it.

| column | baseline | rules out |
|---|---|---|
| `deltaR2` | obs | — |
| `delta_clk` | + a perfect clock | the state is a step counter |
| `delta_anc` | + current position (linear) | it is a position code |
| `delta_flow` | + any smooth f(position) **per env** | §22's deterministic field, under which **the past is a function of the present** |

`delta_flow` uses random Fourier features of position in a separate block per
environment — separate because the walls differ, so the flow does too. The
self-checks are that `elapsed` scores exactly 0 at the clock rung and `pos`
exactly 0 at the anchor rung, both by construction.

### 30.11 Result — only `p24_aux` has a spatial map

`delta_flow`, the column that survives every control:

| target | `p20_e` | `p20_e_kcap` | `p24_aux` | `p25_visin` |
|---|---|---|---|---|
| occupancy (4x4) | 0.035 | 0.011 | **0.136** | 0.009 |
| pos_lag5 | 0.023 | 0.009 | 0.006 | 0.012 |
| pos_lag10 | 0.130 | 0.027 | 0.035 | 0.047 |
| pos_lag20 | 0.157 | 0.026 | 0.088 | 0.041 |

Verdicts: `p20_e` content YES, `kcap` **no**, `p24_aux` YES, `p25_visin` **no**.

**The control has no map.** `p20_e` occupancy runs 0.207 → 0.030 → 0.044 →
0.035: it collapses the moment the baseline gets a clock and never recovers.
What looked like knowing where it had been was knowing what time it was. This
is the well-powered version of §30.6's `start_pos` claim — 19,200 samples
rather than 67 — and it says the same thing far more convincingly.

**`p24_aux` is the exception, and it is a real one.** Occupancy runs 0.408 →
0.132 → 0.133 → **0.136** — essentially unchanged across all three controls,
and 4x the control arm. Not the clock, not position, not the flow.

### 30.12 CORRECTION to §30.7 — the aux head built a map, not just a clock

§30.7 concluded "what the aux head actually built was a much better clock; the
thing it was trained on barely moved." **The second half was measured on the
wrong target.** `visited8` is 8 bits on one ring at radius 3 and understates
the representation badly; on the coarse occupancy map the same arm scores
`delta_flow` 0.136 against the control's 0.035.

Both halves of §27 therefore now read the other way:

* the representation IS there — a genuine spatial occupancy map that survives
  the clock, position and flow controls;
* the policy DOES read the state more than the control does —
  `state_influence` 0.310 vs 0.209, `state_share` 0.281 vs 0.181, lag-τ=20
  0.404 vs 0.263;
* **and coverage still did not improve** (§27).

So lever B failed for neither of the two reasons the diagnostic was built to
separate. It is a third mode: **a real, used representation that was not strong
enough, or not of the right kind, to change the behaviour.** That is the finding
that should drive what gets tried next, and it was invisible to every measure
before this one.

### 30.13 UNRESOLVED — `pos_lag20` on the control

`p20_e` scores `pos_lag20` at 0.379 → 0.390 → 0.390 → **0.157**: it drops 60%
at the flow rung but does not vanish, and it is 6x every other arm.

**Do not read it as trajectory memory yet.** The lag profile is 0.023 at k=5,
0.130 at k=10, 0.157 at k=20 — it **increases with lag**, and any decaying
memory trace must do the opposite. A residual that grows with lag is the
signature of a baseline degrading at long lag, not of a state that remembers
further back. Two readings survive:

1. genuine 20-step path memory, which would contradict §22; or
2. 96 random features per env are not rich enough to capture a backward flow
   that is also **not a function** where the wall clip is absorbing — several
   pasts map to one present, so something must disambiguate them.

**The cheap decider, not yet run:** extend `POS_LAGS` to (2, 5, 10, 20, 30, 50,
100) and read the profile. Memory must fall monotonically with lag; a baseline
artefact will keep climbing. No new machinery, one flag.

### 30.14 The targeted splice — which content the policy actually reads

The whole-state swap (§30.2) replaces position, clock and map at once, so it
says the state matters and never WHICH content matters. `--splice_targets`
fits a linear readout subspace of `h` for a named target on the TRAIN trials,
then on held-out trials swaps **only that subspace** for a donor episode's,
leaving the orthogonal complement untouched.

Two controls, both required:

* a **random subspace of the same rank** — any k-dimensional perturbation of
  `h` moves the action somewhat, so `ratio > 1` is the claim, never `d_sub`;
* **`ratio-pos`**, the same thing with the position directions projected out
  of the target subspace. The decoders are not orthogonal — visitation near a
  wall is partly a statement about where you are — and every arm reads
  position at ~10x, so a map subspace can inherit its punch from position.

| | `p20_e` | `kcap` | `p24_aux` | `p25_visin` |
|---|---|---|---|---|
| pos ratio (rank 2) | **11.99** | 10.53 | 10.21 | 9.78 |
| pos frac of full swap | 0.259 | 0.320 | 0.189 | 0.291 |
| occupancy ratio (rank 16) | 1.74 | 1.46 | **3.04** | 1.32 |
| occupancy **ratio-pos** | 1.53 | 1.42 | **3.52** | 0.98 |
| occupancy frac of full swap | 0.139 | 0.121 | 0.134 | 0.126 |

**Every policy here reads position, and little else.** Two directions out of
1024 carry a quarter to a third of the entire state's causal influence, at
10-12x a random 2-plane, on all four arms regardless of what else is in there.

**Only `p24_aux` reads a map**, at 3.52x after position is removed against
0.98-1.53 elsewhere — and residualising RAISES it, so it is not position in
disguise.

### 30.15 Lever B, finally settled — all five claims measured

| claim | measure | verdict |
|---|---|---|
| the map exists | occupancy `delta_flow` 0.136 vs 0.035 | **yes** |
| the policy reads it | `ratio-pos` 3.52 vs 1.53 | **yes** |
| it is a major input | 0.134 of the full swap from 16 dims, vs position's 0.189 from 2 — ~11x less per direction | **no** |
| the aux head raised its priority | share of full swap 0.134 vs the control's 0.139 | **no** |
| coverage improved (§27) | — | **no** |

So lever B failed for **neither** of the two reasons this diagnostic was built
to separate. Not "no representation" (§27's reading) and not "content the
policy ignores" (§30.7's reading, and the framing this tool was designed
around). It is a third mode: **a real, causally-read representation that never
became a priority.** The aux head made the state matter more overall
(`state_influence` 0.310 vs 0.209) and put a genuine map in it, and the map's
share of the policy's state-driven behaviour did not move at all.

That reframes what a lever C should do. Adding representation is not the
problem and neither is making it readable; the problem is that position
dominates the state's influence on the action by an order of magnitude per
direction, and nothing tried so far changes that ratio.

## 31. P26 — absolute position ANSWERS §29.4, and the answer is optimization

`p26_abspos` = `p20_e` plus a 2-dim normalised (x, y) channel, input 74 → 76,
nothing else moved. Job 21824106, COMPLETED 700/700. **Not shippable** — the
channel is an oracle at test time.

### 31.1 Matched scoring, because the logs record different things

`swept_coverage` postdates `p20_e`, so its training log has none. Both
checkpoints were rolled on the SAME 6 envs, same starts, same seed (96 trials,
200 steps) and the one dump reduced by `analysis/nav_tri/swept_from_traj.py`:

| | swept @200 | sd | union |
|---|---|---|---|
| `p20_e` | **0.637** | 0.047 | 1.000 |
| `p26_abspos` | **0.452** | 0.039 | 0.947 |

`p20_e`'s matched 0.637 lands inside the 0.625–0.644 the doc already had for
it, which is the reducer validating itself against an independent measurement.

### 31.2 Speed, then a WORSE ceiling

`cells_per_step`, the metric both logs recorded:

| update | 25 | 50 | 75 | 100 | 200 | 300 | 450 | 700 |
|---|---|---|---|---|---|---|---|---|
| `p20_e` | 0.093 | 0.099 | 0.425 | 0.478 | 0.704 | 0.642 | 0.760 | **0.773** |
| `p26_abspos` | **0.119** | **0.225** | **0.636** | **0.671** | 0.646 | 0.629 | 0.648 | 0.611 |

Ahead through u100 — it skips the §18.7 wall-pin phase the control spends its
first 50 updates in — then flat from u75 to u700 while the control climbs past
it. Final-4 swept 0.479 against `p25_visin`'s 0.629 and the control's 0.637.

**§29.4 is answered: OPTIMIZATION, not localization.** And §30 says why the
oracle was redundant: the agent already carried a position estimate at
R²(both) 0.66 and already read position from 2 of 1024 directions at 10–12×
a random 2-plane. Handing it a cleaner copy of the one thing it already had
and already prioritised moved the learning curve, not the destination.

### 31.3 It is WORSE, and the recurrence curve says why

| | aggregate dip | at τ | trajectories orbiting |
|---|---|---|---|
| `p20_e` | 4.83 | 102 | 84/96 |
| `p26_abspos` | **11.28** | **46** | 96/96 |

`p26_abspos` swings 13.79 cells at τ=20 down to **4.11 at τ=50**, back to 13.80
at τ=70, down again at ~100 and ~140 — minima spaced ~48 apart, the signature
of a closed orbit, and 2.3× the control's depth.

**The oracle made it orbit.** That is coherent rather than surprising: §22
established the policy is a deterministic vector field, §30.14 that every arm
reads position from 2 directions at ~10×, and a deterministic field indexed on
position has closed orbits as its generic attractor. Strengthening the position
signal strengthened exactly the mechanism that produces them. Absolute position
did not merely fail to help — it made the failure mode worse.

### 31.4 CORRECTION to §28 — `p20_e` has a weak dip, not none

§28 states "`p20_e` has no post-rise dip at all in the mean curve." **Its own
table contradicts that**: 12.77 at τ=30 falling to 8.12 at τ=60 is a depth of
4.65, above the `DIP_CELLS = 3.0` the same section sets. This run reproduces
the curve (12.50 → 8.49, depth 4.83, τ=102).

The substantive contrast §28 drew still holds and is what should be quoted: a
dip of 4.8 spread over an IQR of 63–102 is weak and incoherent, against
`p20_e_kcap`'s 10.66 at a period of 57 in 98/100 trajectories. But "no dip at
all" is not what the data says, and the sentence should read **weak and broad,
not absent**.

## 32. The USE side needed its own ladder — and it overturns §30.15

Jack, on the targeted splice: *"how can you do a position swap if the state you
are bringing in is from an agent at the same position? unless it's not."*

It was not. `_donor` draws from any (trial, step) of another episode, so the
donor stands somewhere else. That is fine for most subspaces and **not** fine
for position, and chasing why cost §30.14's headline its meaning.

### 32.1 Four controls, each catching the previous one

| control | what it removes | position, `p20_e` |
|---|---|---|
| none | — | 11.99 |
| donor matched on POSITION | the state saying "I am at B" while the held-fixed observation says "I am at A". Only position suffers this: the observation carries no visitation signal, so swapping occupancy is merely uninformative, never self-contradictory. | 9.93 |
| …and on HEADING | the same contradiction one level down. Two agents in a cell heading opposite ways are not comparable, and `prev_action` puts heading in the observation. | 7.86 |
| …and SUBSPACE SIZE divided out | **the big one.** A readout subspace for something strongly encoded is HIGH-VARIANCE; a random 2-plane in 1024 dims holds ~2/1024 of the variance. `d_sub/d_random` was partly measuring *the edit was bigger*. `size/rnd` for position is **1.8–2.7 across every arm**. | **6.85 ± 1.69** |

`ratio_sens` is action-change per unit of state-change, which is what "does the
policy weight these directions" actually means.

### 32.2 The corrected table, with the uncertainty it always needed

`ratio_sens`, 1σ from the random baseline across 8 draws:

| subspace | `p20_e` | `kcap` | `p24_aux` | `p25_visin` |
|---|---|---|---|---|
| position (rank 2) | 6.85 ± 1.69 | 3.83 ± 0.58 | 5.52 ± 1.50 | 5.13 |
| occupancy (rank 16) | 2.21 ± 0.23 | 1.66 ± 0.24 | **2.62 ± 0.26** | 1.82 |
| visited8 (rank 8) | 2.93 ± 0.37 | 1.52 ± 0.20 | 1.97 ± 0.43 | 1.33 |

The band covers only the random baseline's spread, not sampling variance over
the 96 episodes nor variance in the fitted subspace, so it is a **floor** on
the uncertainty.

### 32.3 §30.14's "10–12×, an order of magnitude per direction" is WITHDRAWN

Position is 1.8–2.7× larger than a random subspace on every arm. Corrected, it
sits at **3.8–6.9×** random, and against the other subspaces it is a factor of
**2–3, not 10**. The conclusion drawn from the inflated number — *"adding
representation cannot help because position swamps everything"* — was resting
on the gap, and a 2–3× gap is a far more movable thing. Nothing about a lever C
follows from these numbers the way §30.15 claimed it did.

### 32.4 §30.12 and §30.15 are WITHDRAWN. §27 was right.

`p24_aux` occupancy **2.62 ± 0.26** against the control's **2.21 ± 0.23**:
difference 0.41, combined σ 0.35, **1.2σ**. Not a difference. Before the size
control it read 3.04 vs 1.74 and I called it a 2× separation and a settled
result.

What survives, and it is worth stating precisely because it took four controls
to isolate:

* **CONTENT is real and large.** `p24_aux` has 4× more occupancy decodable from
  its state than the control (`delta_flow` 0.136 vs 0.035). Decodability has no
  subspace-size confound; none of this touches it.
* **USE is not detectably different.** The policy's weighting of the map
  directions did not measurably move.

**That is §27's original reading — "the representation was there and the policy
ignored it" — and §30.12's correction of it was wrong.** The "third failure
mode" of §30.15 does not exist; there is no evidence the map became a
causally-read input at all.

The lesson is not about the aux head. **Every ratio in §30.14 was a ratio
against a control chosen for its rank and nothing else**, and rank is the one
property that does not determine how much of the state a subspace holds.
