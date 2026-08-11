# Explore-only, on as little training as possible

**Goal.** An agent that explores well — high `mean_coverage`, high
`union_coverage`, flat across distractor counts — trained on the smallest
amount of data and wall-clock that will produce it. Navigation is explicitly
out of scope: this policy is never asked to reach or store a goal.

**Constraints, as set 2026-08-07.** ≤2 GPU-hours per run. 20×20 grid, shorter
rollouts welcome. Robust at `n_dist=10`. Reward shaping only — no changes to
the agent's inputs or architecture. Later relaxed twice, by request:
`explore_goals_off` may be toggled, and `batch_envs` may be raised freely.

Launcher: `hopfield_nav/run_explore_min.sh`. Wave:
`hopfield_nav/submit_explore_min_wave.sh`. Status:
`hopfield_nav/explore_min_status.sh`.

**Baseline to beat.** `gentle-terrain-124` / the v35 repro
(`docs/EXPERIMENTS_SCHEDULE_REPRO.md`): `mean_coverage` 0.53 / 0.51 / 0.48 at
`n_dist` 0/5/10 at u380, and 0.507 / 0.485 / 0.470 at u240 — reached in roughly
20 GPU-hours on an interleaved explore+exploit schedule.

**Status 2026-08-07.** Wave 1 done and scored under the full v35 protocol:
three runs at mean coverage **0.495–0.514 against v35's 0.507** — a tie on
coverage — reached in **2.6 h instead of ~20 h**, with the distractor gap
collapsed from 0.050 to ≈0.000. So the wins are cost and distractor
robustness, not coverage. Reward shaping turned out not to be where the
headroom was; rollout/env structure is.

**Status 2026-08-10.** Wave 2 — the diversity ladder — is **submitted**, from a
branch off `main` carrying only wave 1's tooling (`1be48b4`), not the
`env-generator` line. Thirteen jobs: `e1/e2×3/e4/e8/e16` and
`c1/c2×3/c4/c8`, all `explore:1000`. Submitting it surfaced a second bug, in
the launcher rather than the model — see below — which would have silently
trained the wrong branch.

**The result so far: a handful of environments is enough, and 80 is waste.**
`e8` settles at **0.538 ± 0.015** in **57 minutes** — above s1's best single
eval (0.517) and its verdict score (0.495), which cost 2 h 40 m on 80 envs, and
above v35's 0.507 at ~20 h. `e4` reaches 0.467 in 27 minutes. One and two envs
are genuinely crippled (0.15, 0.34); 4, 8 and 16 all land in 0.43–0.54 with no
orderable trend on one seed each. Pool size is non-monotone in env count —
essential at 2 envs, an overfitting liability at 4. All of this is on the cheap
monitoring eval; **the verdict pass is owed and decides.**

---

## What the target metrics can and cannot be

`evaluate_exploration` reports seven numbers. Only **two** of them are free.

| metric | relation |
|---|---|
| `cells_per_step` | `mean_coverage × 400 / max_steps`. Identical to `mean_coverage` at `max_steps=400`. |
| `union_per_rollout` | `union_coverage / num_trials`. |
| `redundancy` | `≈ union_coverage / (N · mean_coverage)`, N = trials. |

The `redundancy` identity was checked against v35's own logs and reproduces to
three decimals (u140 n_dist=0: reported 0.0777, predicted 0.0772; u100: 0.1134
vs 0.1122). It is not exact only because it averages a per-env ratio rather
than taking a ratio of averages.

The consequence matters for how this wave is scored: **`redundancy` cannot be
maximized alongside `mean_coverage`.** Its ceiling is `min(1, 1/(N·cov))`,
which *falls* as coverage rises — at N=32, cov 0.53 caps redundancy at 0.059,
cov 0.85 caps it at 0.037. Asking for both is asking for rollouts that are
simultaneously long and disjoint, which 32 rollouts on 400 cells cannot be.

So the scored targets are **`mean_coverage` ↑, `union_coverage` ↑, and the
`n_dist` 0→10 gap → 0**, with `redundancy` reported as a derived diagnostic.

---

## The cost model

This governs every sizing decision below, and two thirds of it was wrong in
the first cut of this wave.

The update loop runs **one rollout per env, for every env in
`envs_per_world`**, and pools them all into a single PPO step
(`train_navigate.py:241`). `batch_envs` is *not* envs per update — it is the
parallel-episode batch **inside** one env's rollout. Therefore:

```
env-steps / update        = envs_per_world × batch_envs × steps_per_rollout
PPO pool (trajectories)   = envs_per_world × batch_envs
SERIAL model calls / upd  = envs_per_world × steps_per_rollout
```

Wall-clock tracks the **third** line, because envs are looped, not batched
together. Measured on an l40s, 80 envs × 200 steps = 16,000 serial calls =
**30.8 s/update**, i.e. ~1.9 ms per serial call.

Two consequences that are not obvious from the flag names:

- **Cutting `batch_envs` does not save time.** d3 cut it 4× (16→4) and got
  *slower* — 48.8 s/u vs 30.8 — because the serial call count is unchanged and
  each call just underuses the GPU more. It saves data, not wall-clock.
- **`batch_envs` is nearly free to raise** — *false above ~batch 16, corrected
  2026-08-10.* It adds no serial call, but the calibration of ~1.9 ms per call
  was taken at batch 16, where the GPU is idle enough that a bigger batch rides
  along free. It does not hold at wave-2 sizes: `c2s42` (2 envs × batch 640)
  and `e2s42` (2 envs × batch 16) issue the **same 400 serial calls per
  update**, and run at **~4.8 s/u against ~0.9 s/u**. Same call count, 5×
  the wall-clock. Past saturation the per-call cost scales with the batch, so
  the honest cost model is

  ```
  wall-clock/update ≈ (envs_per_world × steps_per_rollout) × f(batch_envs)
  ```

  with `f` flat only while the GPU is underused. `batch_envs` buys pool size at
  a real price; it is `envs_per_world` that is the cheap lever, not both.

`eval_scope=expl` brought an eval pass down to **5.0 s**, so eval is no longer
a meaningful share of anything and per-update cost is essentially all rollout
collection.

---

## What is free in the reward shaping, and what is not

Advantages are normalized over the full pool (`updates/ppo.py:142`), and an
explore rollout under `explore_goals_off` is fixed-length with no teleport.
Two things follow.

**Only ratios to novelty are meaningful.** Scaling every shaping term by a
constant scales the reward by that constant, which normalization removes.

**`revisit_penalty` is exactly redundant with `novelty_reward`.** Novelty fires
on new cells and revisit on old ones, so
`n·1[new] − c·1[old] = (n+c)·1[new] − c`. The `−c` is a constant per step over
a fixed number of steps, so it cancels in the advantage. Revisit becomes a real
term only when:

- `novelty_scale_remaining` is on — novelty is then state-dependent while the
  penalty stays flat; or
- the goal is live — teleports zero the shaping mask and rollouts stop being
  equal-length.

This is why the planned shaping variants pair revisit with the remaining-scale
on and off, rather than sweeping revisit against a flat novelty, which would
have burned GPU hours measuring nothing.

**Distractors.** With `goals_off` there is never a reward for moving toward a
recalled point, so the chase behavior has no gradient by which to form. The
prediction was that pure explore dissolves the distractor problem rather than
merely improving it. See results.

---

## Two bugs this wave found

### The launcher trained the wrong checkout (found 2026-08-10, wave 2)

All three explore-min scripts hard-coded `cd /home/jackking/cls`, so a wave
submitted from an agent worktree did not train that worktree — it trained
whatever branch the *shared* checkout happened to be sitting on. Submitting
wave 2 from a branch off `main` would in fact have run `env-generator`'s
per-rollout Hopfield derivation and store-head objective freeze, and the
resulting numbers would have been attributed to the diversity ladder.

Nothing in the logs would have shown it, which is what let it survive wave 1
unnoticed — wave 1 happened to be submitted from the shared checkout, so it is
not affected, but only by luck.

Fixed by having the submitter resolve its own repo root from `$BASH_SOURCE` and
export it as `REPO_DIR`, which `sbatch` passes to every job; the batch scripts
read it from the environment, the only channel available to them, since SLURM
runs them from a node-local spool copy where `$BASH_SOURCE` is useless. Each
job now prints `repo: <path> @ <sha>` at startup so the checkout and commit are
in the experiment record rather than implied.

### `--freeze_log_std` did nothing (found 2026-08-07, wave 1)

`--freeze_log_std` was a **no-op on `train_navigate`** until 2026-08-07.
`NavAgent.__init__` set `requires_grad=False`; then `train_navigate.py:90`
called `set_phase_freeze(freeze_move=False, …)` and `move_params()` includes
`movement_log_std`, handing the gradient straight back. No phase freezes
movement, so the flag never bit.

Visible in the v35 repro log as `std` drifting **0.166 (u1) → 0.294 (u250)**
under `FREEZE_LOG_STD=1`. Every run in that lineage trained a learnable
log_std whatever its launcher said — including the configuration V10 credits
with fixing coverage drift under deterministic eval.

Fixed in `set_phase_freeze`; pinned by `hopfield_nav/tests/test_log_std_freeze.py`.
Runs from 2026-08-07 are the first where the flag works, so they are not
comparable to earlier ones on this axis. Variant `f1` exists to bracket it.

---

## Results — wave 1, complete

Three runs, pure `explore:300`, 20×20, all COMPLETED. Eval pinned at 400 steps
so coverage is the same measurement across variants and comparable to v35.

`mean_coverage` at `n_dist=0`, in-training eval (4 val envs × 16 trials):

| update | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | **300** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **s1** 80 envs ×16, v35 shape | .068 | .155 | .240 | .304 | .291 | .347 | .372 | .423 | .444 | .490 | .500 | **.517** |
| **s2** 80×16, no wall_penalty | .051 | .154 | .180 | .241 | .323 | .394 | .400 | .395 | .405 | .465 | .469 | **.510** |
| **d3** 80 envs ×4 | .052 | .128 | .246 | .312 | .355 | .385 | .393 | .440 | .449 | .484 | .518 | **.515** |

Final, at u300:

| run | cov d0 | cov d10 | union d0 | union d10 | 0→10 gap | s/update | elapsed |
|---|---|---|---|---|---|---|---|
| s1 | 0.517 | 0.514 | 0.991 | 0.988 | 0.004 | 31.1 | 2:39:57 |
| s2 | 0.510 | 0.515 | 0.984 | 0.985 | −0.005 | 30.4 | 2:35:52 |
| d3 | 0.515 | 0.517 | 0.956 | 0.961 | −0.002 | 49.6 | 4:16:12 |
| *v35* | *0.53 (u380)* | *0.48* | — | — | *0.050* | *185* | *~20 h* |

### 1. v35-class coverage for roughly a seventh of the wall-clock

s1 finishes in **2 h 40 m** at a strict-protocol mean of **0.495** (the
in-training 0.517 was optimistic — see the verdict pass). v35 reaches a mean
of 0.507 in ~20 GPU-hours, and was at 0.507 / 0.470 at its u240 (~12.4 h). So
pure explore **matches** the baseline's coverage at about **7× less
wall-clock** — it does not exceed it — and was still climbing at u300 where
the run was cut.

The data saving is close to exactly the exploit share v35 spends — dropping the
exploit regime returns that data to coverage, no more and no less. No synergy
bonus, and no cost either.

### 2. Learning is update-limited, not data-limited — confirmed to u300

d3 cut `batch_envs` 4× and finished at **0.515 against s1's 0.517**. It tracked
s1 the whole way (u75 .246/.240, u100 .312/.304, u125 .355/.291, u250
.484/.490) on a **quarter of the env-steps**. The gradient from 4 parallel
episodes per env is already sufficient; the other 12 buy nothing.

The early read held all the way to the end, which is what licenses the `c*`
ladder below: if the pool can be shrunk 4× at no cost, the reverse trade —
spending `batch_envs` to buy back the pool that a small `envs_per_world` gives
up — is worth testing.

Note the asymmetry that makes this a data result and not a time result: d3 took
**longer** in wall-clock (4:16 vs 2:40) despite a quarter of the data, exactly
as the cost model predicts. Same 16,000 serial calls per update, less work per
call, worse GPU utilization — and it sat on a contended a100 while s1/s2 had
l40s to themselves, so the 49.6 vs 31.1 s/u gap is partly node, not method.

### 3. Distractor robustness: solved, not merely improved

The `n_dist` 0→10 gap is **≤0.011 at every eval of every run**, and −0.005 to
+0.004 at u300 — i.e. indistinguishable from zero, with the sign varying.
v35's gap was 0.050. The mechanism was predicted in advance: under
`explore_goals_off` there is no reward for moving toward a recalled point, so
chase behavior has no gradient by which to form.

This is the wave's cleanest result. Pending confirmation at 10 envs × 32 trials.

### 4. The shaping axis produced nothing

s1 (wall 0.1) vs s2 (wall 0) finish at 0.517 vs 0.510 — a 0.007 gap, on one
seed, after the two crossed each other twice en route (s2 ahead at u125–150,
behind at u200–275). No support for the V18-era claim that `wall_penalty` caps
coverage, and no support for the opposite either. **The honest reading is that
`wall_penalty` does not matter at this scale**, and that the shaping knobs are
not where the remaining headroom is.

That is why shaping sits *below* diversity in the plan below, having started
above it. It is also consistent with the analysis above: under advantage
normalization most of the shaping space is degenerate, and the terms that
survive are fewer than the flag list suggests.

### 5. The log_std freeze works, end to end

`std` printed 0.165 at u1 and 0.165 at u300 in all three runs — against the v35
repro's drift to 0.294 under the same flag. First confirmation in a real run
that the fix does what the unit test asserts. Every number in this wave was
therefore measured with a genuinely pinned policy std, which no earlier run in
this lineage was.

### Verdict pass — done, and it moves the numbers

Job `19872591` (`hopfield_nav/run_explore_min_verdict.sh`, 34 min) scored all
three u300 checkpoints under the v35 protocol: 10 envs x 32 trials, `n_dist`
{0, 5, 10}, 400 steps. **These supersede every in-training number above.**

| run | cov d0 | d5 | d10 | mean | union d0 | redundancy | 0->10 gap |
|---|---|---|---|---|---|---|---|
| d3 | 0.513 | 0.519 | 0.511 | **0.514** | 0.985 | 0.060 | 0.002 |
| s2 | 0.503 | 0.501 | 0.505 | **0.503** | 0.994 | 0.062 | -0.003 |
| s1 | 0.496 | 0.496 | 0.495 | **0.495** | 0.984 | 0.063 | 0.001 |
| *v35* | *0.53* | *0.51* | *0.48* | *0.507* | - | - | *0.050* |

**The 4x16 monitoring eval was biased high by ~0.02.** s1 read 0.517 in
training and 0.496 here; s2 0.510 -> 0.503; d3 0.515 -> 0.514. The bias is not
uniform, so the cheap estimate is not even reliable for *ranking*: it put s1
first and d3 last, and the strict protocol reverses that. Treat the
4-env/16-trial numbers as a progress signal only, never as a result. This is
exactly what the protocol was held in reserve for.

**Revised headline: the wave MATCHES v35's coverage, it does not beat it.**
Mean coverage 0.495-0.514 against v35's 0.507 - a tie, on one seed, with the
spread between our own three variants (0.019) larger than the gap to the
baseline. The claim that survives is about *cost*, not coverage: v35-class
coverage for **2.6 h instead of ~20 h**. Any statement elsewhere in this
document implying the wave exceeded 0.53 is reading the optimistic
in-training estimate and is superseded here.

**The distractor result survives the strict protocol intact** - gap 0.002 /
-0.003 / 0.001 against v35's 0.050, now measured at 10 envs x 32 trials across
all three distractor levels. This is the wave's one unambiguous improvement
over the baseline, and it is a large one.

**`redundancy` lands on its predicted ceiling.** Reported 0.060-0.063 against
`union/(N*cov)` = 0.984/(32*0.496) = 0.062 for s1. It is pinned by the other
two metrics exactly as derived above, with no slack left to optimize -
confirming that the original ask for "high coverage AND high redundancy" was
unsatisfiable rather than merely unsatisfied.

`goal_find_rate` 0.63-0.82 is incidental: the goal is inert in this evaluator,
so it records only that a thorough sweep tends to cross it.

### What wave 1 did not answer

- **Diversity.** Every run used `envs_per_world=80`. The whole wave varied the
  axis that turned out not to matter (`batch_envs`) and never touched the one
  that drives both cost and generalization.
- **Rollout length.** All three ran 200 steps. `d2`/`d5` untested.
- **Whether 0.517 is a ceiling or just u300.** All three were still climbing.
  A longer run at the winning config is the cheapest available gain.

---

## Results — wave 2, the diversity ladder (in progress, 2026-08-10)

Thirteen jobs, all `explore:1000`, submitted from a branch off `main` at
`1be48b4` + the launcher fix. The per-user cap is `gres/gpu=2`, so the ladder
drains two at a time; rungs are recorded here as they land.

**All numbers below are the 4-env × 16-trial in-training eval**, which wave 1
showed is biased high by ~0.02 and unreliable even for *ranking*. They are
progress signal. The verdict pass is still owed.

### Read trailing means, not final evals

This eval is noisy enough that a single reading is not a level, and it misled
this document twice before the rule was adopted: `c2s43` reads **0.452** at
u1000 with 0.315 / 0.322 / 0.387 immediately before it, and `e4`-vs-`c4` looks
like 0.504-vs-0.447 on final evals but is a **tie at 0.467** on the last 200
updates. Runs are therefore summarized by the mean of their last 8 evals.

| run | envs | pool | final | **mean₈** | sd₈ | max | @max |
|---|---|---|---|---|---|---|---|
| `e4s42` | 4 | 64 | 0.504 | **0.467** | 0.022 | 0.504 | u1000 |
| `c4s42` | 4 | 1280 | 0.447 | **0.467** | 0.013 | 0.531 | u325 |
| `c2s43` | 2 | 1280 | 0.452 | 0.363 | 0.064 | 0.452 | u1000 |
| `c2s42` | 2 | 1280 | 0.346 | 0.348 | 0.024 | 0.402 | u600 |
| `c2s44` | 2 | 1280 | *running* | *0.319* | 0.022 | 0.354 | u375 |
| `e1s42` | 1 | 16 | 0.152 | 0.158 | 0.011 | 0.189 | u775 |
| `c1s42` | 1 | 1280 | 0.152 | 0.146 | 0.007 | 0.196 | u400 |
| `e2s43` | 2 | 32 | 0.137 | 0.146 | 0.018 | 0.264 | u475 |
| `e2s42` | 2 | 32 | 0.071 | 0.113 | 0.034 | 0.215 | u125 |
| `e2s44` | 2 | 32 | 0.076 | 0.097 | 0.020 | 0.201 | u800 |

Note `sd₈` as a stability read in its own right: `c4` 0.013 and `c1` 0.007 are
settled, `c2s43` 0.064 and `e2s42` 0.034 are not. And note that s1/d3 cannot be
compared on mean₈ at all — they ran only 300 updates, so their last 8 evals
cover their climb rather than a plateau. Against them, use the verdict pass.

**The 2-env seed spread is small, not large.** On finals it looks like 0.346 /
0.452 / 0.301 — a 0.151 range. On mean₈ it is 0.348 / 0.363 / 0.297, a range of
0.066. The 2-env level is ~0.34, and the design's three seeds earned their
place by showing that the apparent spread was mostly eval noise.

### Headline: **eight environments beats eighty**, at a third of the wall-clock

| run | envs | batch | pool | mean₈ | max | wall-clock |
|---|---|---|---|---|---|---|
| `e1s42` | 1 | 16 | 16 | 0.158 | 0.189 | ~9 m |
| `c1s42` | 1 | 1280 | 1280 | 0.146 | 0.196 | 103 m |
| `e2s42/43/44` | 2 | 16 | 32 | 0.113 / 0.146 / 0.097 | ≤0.264 | 21 m ea |
| `c2s42/43/44` | 2 | 640 | 1280 | 0.348 / 0.363 / 0.297 | ≤0.452 | 80 m ea |
| `e4s42` | 4 | 16 | 64 | 0.467 | 0.504 @u1000 | **27 m** |
| `c4s42` | 4 | 320 | 1280 | 0.467 | 0.531 @u325 | 87 m |
| **`e8`** | **8** | **16** | **128** | **0.538** | **0.561** @u950 | **57 m** |
| `c8s42` | 8 | 160 | 1280 | 0.492 | 0.539 @u650 | 125 m |
| `e16` | 16 | 16 | 256 | 0.435 | 0.490 @u725 | 105 m |
| *s1* | *80* | *16* | *1280* | *— (u300 only)* | *0.517 @u300* | *2 h 40* |

`e8` settles at **0.538 ± 0.015** — a genuine plateau from u625 to u1000, not a
spike — with union coverage 0.96–0.99 and the distractor gap at zero. It is the
highest settled level either wave has produced, above s1's best single eval
(0.517) and well above s1's verdict-protocol score (0.495), in **57 minutes
against 2 h 40 m**.

So the diversity curve does *not* flatten at 4, as an earlier version of this
section concluded from `e4` alone:

| envs | settled level | binding constraint |
|---|---|---|
| 1 | 0.15 | diversity — an 80× pool moves it by 0.001 |
| 2 | 0.34 | diversity — `c2` plateaus; without pool, instability on top |
| 4 | 0.467 | unclear; pool no longer helps, more envs may |
| **8** | **0.538** | — best observed |
| 16 | 0.435 | — |
| 80 | ≥0.517 at u300, never run to u1000 | — |

**It saturates by 4–8 envs, and then stops being orderly.** The 2→4 step is
worth 0.13 and is unambiguous. Past that, 4 / 8 / 16 land at 0.467 / 0.538 /
0.435 — a 0.10 spread with no monotone trend, on **one seed each**. The 2-env
seed triple showed a 0.066 spread on mean₈ at a *more* stable configuration, so
a 0.10 gap between single runs is not enough to order 4, 8 and 16, and the
claim that "8 is optimal" is not supported. `e16`'s trajectory also swings from
0.203 to 0.490 mid-run, which is not the profile of a converged setting.

What the ladder does support: **one and two envs are crippled, four or more is
enough, and beyond eight you pay wall-clock (105 m at 16 envs against 57 m at
8) for nothing visible.** Ordering the top of the ladder would need seeds, and
that is the obvious wave-3 job.

**A caveat this wave cannot remove.** s1 ran 300 updates and was still climbing
when cut; `e8` ran 1000. So this is "cheap config run long beats expensive
config run short" — a real and useful claim, since the wave's question is
minimum cost — but it is *not* a ceiling comparison. Nobody has run 80 envs to
u1000, which would cost ~8.7 h. At matched u300, 80 envs is still ahead
(0.517 against `e8`'s ~0.42).
| 80 | 0.517 | — |

### What the 1- and 2-env rungs say about pool size

The `c*` ladder was built to separate diversity from PPO-pool size, and the
answer is that **pool only matters once diversity is not the binding
constraint**:

- **At 1 env it buys nothing — exactly nothing.** `c1` finishes at **0.151**
  and `e1` at **0.152**, on pools of 1280 and 16. An 80× larger gradient batch
  produced a difference of 0.001, for 103 minutes of wall-clock against 9. One
  env caps the run long before the gradient does. This is the cleanest null in
  either wave.
- **At 2 envs it buys stability, not the ceiling.** `e2` collapses; `c2` with
  the same two envs is stable for 1000 updates. But `c2` still tops out at
  ~0.37 ± 0.03 — u450 0.392, u600 0.402, u750 0.361, u1000 0.346, drifting
  mildly *down*. So 2 envs is **capped, not slow**, the distinction the
  1000-update budget was bought to make and one that could not have been made
  at u300, where `c2` reads 0.327 and merely looks behind.
- **At 4 envs the big pool actively hurts, by overfitting.** See below — this
  is the wave's second real result.

### At 4 envs, the large pool buys speed and then costs generalization

`c4s42` (4 envs, pool 1280) against `e4s42` (4 envs, pool 64) — same four
training envs, same seed, only the pool differs:

| update | `c4` (pool 1280) | `e4` (pool 64) |
|---|---|---|
| u175 | 0.438 | 0.254 |
| u325 | **0.531** ← peak | 0.374 |
| u475 | 0.413 | 0.401 |
| u700 | 0.507 | 0.443 |
| u1000 | 0.447 | **0.504**, still climbing |
| wall-clock | 87 m (5.2 s/u) | **27 m** (1.6 s/u) |

> **Read the trailing means, not the final evals.** On the last-8-eval average
> `e4` and `c4` are **tied at 0.467**. The 0.504-vs-0.447 gap in the `u1000`
> row is single-eval noise, and an earlier version of this section drew a
> ranking from it that the data does not support. What differs between the two
> is the *shape*, which is robust, and that is what the rest of this section is
> about.

`c4` learns far faster per update and **peaks at 0.531 — above s1's 0.517** —
then falls back to a 0.467 plateau. `e4` rises steadily to the same 0.467 and,
unlike `c4`, has its maximum at the last eval — it is still climbing where `c4`
has turned over. The training reward says which is which:

| | training `mean_r` u150 → u900 | held-out coverage |
|---|---|---|
| `c4` | 0.226 → **0.336** (rising) | peaks u325, then **falls** |
| `e4` | 0.092 → 0.271 (rising) | rises monotonically throughout |

`c4` fits its four training envs harder than `e4` does and generalizes worse
for it: rising train, falling held-out, the unambiguous overfitting signature —
and the same one `e2` showed in its first phase. **`e4`'s small pool is doing
regularization.** 64 trajectories give a noisy gradient, the noise keeps the
policy from settling into the four training codebooks, and the run climbs
slower but does not turn over.

**`c8` confirms it, and shows diversity does not rescue the big pool.** At 8
envs the pool-1280 run behaves exactly like `c4`: training reward climbing
monotonically 0.230 → 0.336 (the same terminal value `c4` reached), held-out
coverage peaking at **0.539 @u650** and then falling to a 0.492 plateau, while
`e8` on a pool of 128 climbs past it to 0.538 with its maximum at u950. Four
times the diversity only **delays the peak** — u325 at 4 envs, u650 at 8 — it
does not remove the turnover. And note both `c4` and `c8` peak at ~0.53: the
big pool reaches the same place and then gives it back.

So the effect of pool size is **non-monotone and depends on env count**, and
the small pool wins by more as diversity grows:

| envs | small pool | pool 1280 | result |
|---|---|---|---|
| 1 | 0.158 (pool 16) | 0.146 | **nothing** — diversity binds first |
| 2 | 0.113 (pool 32) | 0.348 | **large gain** — stability binds first |
| 4 | 0.467 (pool 64) | 0.467, peak 0.531 | **tie**, peak given back |
| 8 | **0.538** (pool 128) | 0.492, peak 0.539 | **small pool wins** |

There is no single "bigger pool is better". Two envs need pool to survive; four
envs get nothing from it, and pay for it in wall-clock and in a peak they then
give back.

The practical consequence is that the two cheapest configurations in the study
land at the same plateau but differ in kind:

- **`e4`, 27 min** → plateau 0.467, max 0.504 *at the final eval*, monotone, no
  early stopping needed, still climbing when cut.
- **`c4`, 87 min** → same 0.467 plateau, but reaches **0.531 at u325** on the
  way. Early-stopped there it is ~28 min for the highest single number either
  wave has produced; run to u1000 it gives that back.

Both are enormously cheaper than s1 (0.517 in 2 h 40 m) and v35 (0.507 in
~20 h) for coverage in the same band — but all of these are 4×16 monitoring
evals, and wave 1 showed those are biased ~0.02 high and unreliable for
ranking. **The verdict pass decides it, not this table.**

### The 2-env collapse is reproducible, and the `e*` ladder is non-monotonic

All three seeds are in, and none of them is a fluke: `e2` ends at **0.071 /
0.137 / 0.076, mean 0.095 ± 0.03**. The dramatic collapse is not universal, but
the failure is:

| seed | peak | after the peak |
|---|---|---|
| s42 | 0.215 @u125 | collapse to 0.07 |
| s43 | 0.264 @u475 | collapse to 0.10, partial recovery, ends 0.137 |
| s44 | 0.168 @u700 | no collapse — drifts down from a low peak |

The two runs that climbed past ~0.21 both fell off; the one that never got
there merely wandered. So the fragile regime is entered by *becoming competent*
on a tiny pool, not by initialization luck.

Two corrections to earlier drafts of this section follow.

**"A 32-trajectory pool is unusable" was too strong**, and was written from
`e2s42` alone. `e1` is stable on a pool of *16* and `e4` climbs cleanly to
0.504 on 64, so pool size is not a monotone danger. The defensible claim is
narrower: *at 2 envs and pool 32, a run that reaches ~0.2 destabilizes, 2 times
out of 2*.

**And the `e*` ladder is not monotone in `envs_per_world`** — 1 env ends at
0.152, 2 envs at 0.095, 4 envs at 0.504. Adding the second env makes things
*worse*. That is not a diversity effect; it is `e1` sitting stably below the
fragile band while `e2` climbs into it. It is also a warning about the `e*`
design generally: because it moves envs and pool together, its rungs are not
comparable to each other, and only the `c*` rungs isolate diversity.

### `e2`'s collapse has two phases, and only the first was predicted

The doc anticipated overfitting to the training envs' codebooks, diagnosed as
held-out coverage falling while training `mean_r` rises. That happens, but only
first:

| window | held-out cov | training `mean_r` | reading |
|---|---|---|---|
| u125 → u300 | 0.215 → 0.151 | 0.128 → 0.181 | overfitting, as predicted |
| u400 → u500 | 0.160 → 0.074 | 0.155 → **−0.021** | outright optimization collapse |

The second phase is new and is not overfitting: training reward goes *negative*,
below its own u1 value. The run does not memorize its two envs and stop
generalizing — it stops optimizing at all. Worth separating, because
overfitting argues for more envs and a collapse argues for a bigger batch, and
here the bigger batch is what actually fixed it.

### The distractor result survives at 2 envs

The `n_dist` 0→10 gap stays within ±0.04 of zero across every eval of both
runs, including `e2`'s collapse. Whatever the pool and diversity are doing,
they do not touch the mechanism — with `explore_goals_off` there is no gradient
by which chase behavior can form, so there is nothing for distractors to
exploit even in a run that is falling apart.

### Costs, measured

`e2s42` ran 1000 updates in **20 m 49 s** (0.9 s/u). `c2s42` took ~80 min
(4.6 s/u) for the same 1000 updates and the same env-steps — the 5× is the
`batch_envs` cost correction above, not extra data.

### Still outstanding

`c1` (running), `c4`, `c8`, `e8`, `e16`, and the `s43`/`s44` seeds at 2 envs.

The open question has moved. It is no longer "where does the floor lift" —
`e4` lifts it — but:

1. **Is `e4`'s 0.504 a ceiling or just u1000?** It was still climbing when cut,
   as s1 was at u300. A longer `e4` is now the cheapest available gain in the
   whole study: 27 minutes bought 0.504.
2. ~~**Does `c4` beat `e4`?**~~ **Answered: no, not by u1000** — it peaks
   higher and sooner (0.531 @u325) but overfits and ends at 0.447. The minimal
   recipe is `e4`; the *fastest-to-0.53* recipe is `c4` with early stopping.
3. ~~**Is `e2`'s collapse seed-specific?**~~ **Answered: no**, 3 seeds, mean
   0.095.
4. **Do 8 and 16 envs add anything measurable over 4?** If not, 4 envs is the
   recommendation and `e8`/`e16` only bound it from above. Now sharper given
   the overfitting result: more envs should *reduce* `c4`-style overfitting, so
   `c8` is the test of whether a large pool stops hurting once there is enough
   diversity to absorb it.
5. **Does `e4L` keep climbing?** It should, on the evidence — `e4` shows no
   overfitting signature at all — which would make the 4-env ceiling higher
   than anything measured so far.

---

## The plan: what to queue next, and why

Everything below is cancelled-but-designed as of 2026-08-07. Submit with
`bash hopfield_nav/submit_explore_min_wave.sh`, or cherry-pick.

### Priority 1 — how few distinct envs still generalize

The open question, and the one that is also the largest cost lever, since
`envs_per_world` is the only term in the serial-call count that is not already
minimal.

Generalization is a fair reading of the existing numbers: the eval world is a
separate `setup_world` call (`train_navigate.py:382`) with its own VectorHash
scaffold and its own randomly drawn envs, codebooks and goals. Nothing in it is
trained on, so the reported coverage **is already held-out coverage**.

**`e*` — turn envs down, change nothing else.** `e2s42/43/44`, `e1s42`,
`e4s42`, `e8`, `e16`.

**`c*` — the same ladder at a constant PPO pool of 1280 trajectories.**
`c2s42/43/44`, `c1s42`, `c4s42`, `c8s42`.

| variant | envs | batch_envs | pool | env-steps/upd | serial calls/upd |
|---|---|---|---|---|---|
| c1 | 1 | 1280 | 1280 | 256,000 | 200 |
| c2 | 2 | 640 | 1280 | 256,000 | 400 |
| c4 | 4 | 320 | 1280 | 256,000 | 800 |
| c8 | 8 | 160 | 1280 | 256,000 | 1,600 |
| s1 *(done)* | 80 | 16 | 1280 | 256,000 | 16,000 |

The `c*` ladder exists because `e*` alone is confounded. `envs_per_world` sets
both how many distinct envs exist *and* how many rollouts enter each PPO step,
so `e2` has a pool of 32 trajectories against s1's 1280. A shortfall could be a
40× smaller gradient batch rather than missing diversity — and those imply
opposite fixes. Trading `batch_envs` against `envs_per_world` holds pool size,
env-steps per update and memory identical to s1, leaving diversity as the only
difference.

Three seeds at 2 envs because at that size *which* two envs were drawn is
plausibly the largest term in the result; one seed could not support a claim in
either direction.

**1000 updates, not 300**, because at 0.4–3 s/update it is nearly free, and
because "few envs is slower" and "few envs is capped" look identical at 300
updates and mean opposite things. Every run still evals at u300, preserving the
fixed-update comparison against s1.

**What each outcome would mean.**

- `c2` matches `s1` → diversity is not the binding constraint. The cheap recipe
  is 2 envs with a large `batch_envs`: same data, **400 serial calls per update
  instead of 16,000**.
- `c2` matches `s1` but `e2` falls short → both effects are real and separable;
  few envs is fine provided the batch is bought back.
- `c2` falls short of `s1` → genuine diversity floor, and the ladder locates it.
- Eval coverage peaks then declines while training `mean_r` keeps rising →
  overfitting to the training envs' codebooks. A live risk at 1–2 envs over
  1000 updates, and detectable precisely because eval is held out.

### Priority 2 — rollout length, the remaining wall-clock lever

**`d2`** — `steps_per_rollout=100`, batch 16. The other factor in the serial
call count. Watch for a specific failure: `novelty_scale_remaining` is driven
by cells left *this rollout*, so a 100-step rollout on a 400-cell grid never
gets past a scale of ~1.3 and the endgame gradient never appears. Expect fast
early learning and an early cap. Interacts with train/eval length mismatch,
since eval is pinned at 400 steps.

**`d5`** — 4 envs × 100 steps = the combined floor, if `d2` survives.

### Priority 3 — questions the wave has not yet touched

**`f1`** — `--no-freeze_log_std`. Brackets the bug fix above. Worth running
before trusting any cross-era comparison, since every other run in this wave is
the first to get a genuinely frozen log_std.

**`s6`** — `epsilon_explore=0.1` instead of 0.4. At 0.4, two of every five steps
is a random direction, which is what breaks the long straight sweeps the
shaping is trying to reinforce — while eval scores the deterministic policy. If
it wins it wins on every other variant too.

**`g1` / `g2`** — `explore_goals_off=0`, at goal_reward 5.0 and 1.0, both with
`randomize_goal_per_rollout=1`. A live goal is two things at once: a
search-for-something-hidden curriculum, and a stream of random restarts, since
arrival teleports the *agent* to a random cell while the goal stays put
(`world/vec_env.py:60`). Goal randomization is required or the fixed sensory
codebook makes "in env X walk to Y" memorizable, which buys goal-finding
without buying exploration. Note this also breaks the fixed-length property the
shaping analysis rests on, so `revisit_penalty` is non-redundant here even with
the remaining-scale off.

### Deprioritized, and why

- **`s3` / `s4` / `s5`** (persistence-heavy; revisit × remaining-scale) — the
  shaping axis has shown nothing across s1/s2, so these sit below diversity.
- **`d4`** (64 envs, 12,800 env-steps/update) — existed to bracket
  data-vs-update-limited from above; d3 settled it from below.
- **`d1`** (400-step rollouts) — the most expensive variant, and longer
  rollouts are the wrong direction given the cost model. Its one remaining use
  is testing whether matching train length to the 400-step eval matters.

---

## Verdict protocol

In-training evals use 4 val envs × 16 trials at `n_dist` ∈ {0, 10} — a
monitoring-grade estimate, deliberately cheap. **No conclusion is final on
those numbers.** The verdict is an offline pass on saved checkpoints under the
v35 protocol — 10 envs, 32 trials, `n_dist` ∈ {0, 5, 10}, 400 steps — so the
comparison against 0.53 / 0.51 / 0.48 is like-for-like:

```bash
python -m hopfield_nav.eval_all --ckpt <ckpt> --device cuda \
    --num-val-envs 10 --num_trials 32 --max_steps 400 \
    --n_distractors 0 5 10 --no-nav-stoch --skip-realistic --repeat-trials 0 \
    --output-json <out.json>
```

Single seed everywhere except the 2-env runs. Treat every gap below ~0.03 as
noise until it is seeded.
