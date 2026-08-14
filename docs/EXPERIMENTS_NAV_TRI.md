# One model, three metrics: coverage + success rate + steps-to-goal

Working document for the `nav-tri-metric` branch. Started 2026-08-14.

**Read `## 0. Where I am` first** — it is the resume point, rewritten every
wave, and it is the only section that goes stale on purpose.

---

## 0. Where I am

| | |
|---|---|
| **Branch / worktree** | `nav-tri-metric` at `.claude/worktrees/nav-tri-metric` |
| **Wave in flight** | **Wave 1** (6 runs, submitted 2026-08-14) — §6 |
| **wandb project** | `train_navigate` |
| **Launcher** | `hopfield_nav/run_nav_tri.sh` (`VARIANT=<name> sbatch …`) |
| **Status helper** | `bash hopfield_nav/nav_tri_status.sh` |
| **Env python** | `$HOME/.conda/envs/cls/bin/python` (login node has no `python`) |

Resume checklist, in order:

1. `bash hopfield_nav/nav_tri_status.sh` — what is queued, running, finished.
2. Read §6 for the wave in flight: its hypothesis, its variant table, and the
   rule stated *in advance* for what each outcome would mean.
3. Fill the results table in that wave's section. Then write the "Conclusions"
   subsection — hypothesis confirmed / refuted / underpowered — before
   designing anything new.
4. Design the next wave in §6 with hypothesis and decision rule written down
   *before* submitting.

**Open items carried forward** (things known to be owed, in priority order):

- [x] `signal_separability` at the real Npos=1716 — **done, and it overturned
      the Npos=300 result.** See the CORRECTION in P0.7.
- [ ] Re-run `temporal_separability` at the real Npos=1716 (P0.8 is Npos=300
      and is therefore an *under*estimate). Queued on `mit_normal` (CPU).
- [ ] Behaviour-probe every wave-1 checkpoint at u450 and fill the §6 table.
- [x] Regime assignment was **positional**, so at a fixed `empty_frac` the same
      envs were always exploit and the policy could gate on env identity rather
      than on the recall signal. **Fixed**: `--regime_assignment shuffle`
      (default stays `index` so older configs still mean what they meant).
      Wave 3 uses `shuffle`. Also fixed a latent logging bug it exposed — the
      pre/emp reward split sliced a world-major list and so mixed regimes
      whenever `num_worlds > 1`.

**Scheduler facts that shape how waves are launched:**

- `mit_normal_gpu` — 6 h limit, and **`QOSMaxGRESPerUser` caps concurrent GPUs
  at 2**. `sinfo` showing idle nodes does not mean a third job will start.
- `pi_fiete` — 7 d, 8×a100 on `node3807`, but that node is **shared with other
  partitions and its memory runs near full** (988 G of 1023 G allocated at
  submit time), so jobs pend on memory rather than GPUs. Hence the 64 G request
  and the `precompute_encoded_phi` fix.
- **CPU training is not viable, tested and dropped.** The workload is B=16 rows
  through a small RNN, which uses a GPU poorly, so `mit_normal`'s 3000 cores
  looked like a way past the 4-GPU ceiling. Measured: 32 cores took ~15 min
  just to build `encoded_Phi` (≈3.5 min on a GPU) and had not finished a single
  update at 23 min, against 38 s/update on an l40s. Killed. **The probes,
  however, run fine on CPU and should go there by default** — they need one
  encoder pass and then small matmuls, and `run_nav_tri_probe_cpu.sh` starts
  immediately instead of queueing six hours behind a training run.
- Runs are sized to be checkpoint-safe: `CKPT_EVERY` is set so a job killed at
  the wall-clock limit still leaves a usable series. A TIMEOUT is a normal
  outcome here, not a failure.

---

## 1. The ask

From Jack, 2026-08-14. Restated as a spec.

**Objective.** A *single* model that scores well on all three of:

| metric | regime | source |
|---|---|---|
| coverage | explore | `evaluate_exploration` → `mean_coverage` |
| success rate | exploit | `evaluate_navigation` → `success_rate` |
| steps to goal | exploit | `evaluate_navigation` → `mean_steps` |

Trainer is `hopfield_nav/train_navigate.py`. Nothing else.

**Method Jack asked for.**

- Work in **waves**. Each wave states a question or hypothesis *first*, then
  runs a set number of runs, then records observations, then moves on.
- **Probe before deciding.** Where a decision can be checked cheaply — by
  simulation, by an analysis of rollouts, by reading the code — check it rather
  than guessing. Subagents and throwaway code are both fair game.
- Decisions must be **motivated by problem structure and RL principles**, not
  by a random search over the knob list.
- Keep runs **under ~6 hours** so iteration is fast. Partitions: `pi_fiete`
  (7 d, a100) and `mit_normal_gpu` (6 h, l40s/h200).
- Record everything here: the instructions, the waves, the hypotheses, the
  findings, the tests and where their code lives, and how to pick up mid-task.

**Order of attack Jack proposed** (explicitly flagged as a guess to be adapted
if it fails):

1. Train a model that explores very well.
2. Train a model that exploits very well.
3. Take a good explorer and anneal exploit envs in during training.
4. If that fails: try interleaving throughout, or blocking (never interleaved),
   or exploit-first. Adapt.

### 1.1 Knobs I may turn

`GOAL_REWARD`, `TIME_PENALTY`, `HIDDEN_SIZE`, `LR`, `PPO_CLIP_COEF`,
`NOVELTY_REWARD`, `NOVELTY_ANNEAL`, `NOVELTY_SCALE_REMAINING`,
`NOVELTY_SCALE_CAP`, `REVISIT_PENALTY`, `WALL_PENALTY`, `PERSISTENCE_BONUS`,
`EPSILON_EXPLORE`, `EPSILON_ANNEAL_UPDATES`, `BATCH_ENVS`, `ENVS_PER_WORLD`,
`STEPS_PER_ROLLOUT`, `SCHEDULE`, `MOVE_ENT_COEF`, `FREEZE_LOG_STD`,
`INIT_LOG_STD`.

The last three carry a warning: understand what they do before moving them,
and run an experiment if that is what it takes. §3.4 does that.

### 1.2 Fixed by instruction — not experimental axes

Everything here supersedes `run_repro_v35.sh`; anything not listed is v35's.

| setting | value | why (Jack's instruction) |
|---|---|---|
| encoder | `encoder_training/sweeps/ur_loss2_repel_low/029_repel_weight=2_per_env_radius_frac=0.1_seed=44/encoder_best.pt` | named directly |
| `--rnn_cell` | `rnn` | "use RNN" (v35 was a GRU) |
| `--rnn_nonlinearity` | `relu` | "and ReLU" |
| `--goal_radius` | `1.0` | named |
| `--continuous_normalize` | `0` | "don't normalize step size" |
| `--input_hopfield_raw` | `1` | "use raw hopfield input" |
| `--wall_resolution` | `4` | named |
| `--observation_size` | `60` | named (v35: 12 in explore-min, 60 in v35) |
| `--explore_goals_off` | `1` *(start)* | "reward at goal off for explore training… but that could be a knob" |
| `--explore_ends_on_goal` | `1` | named (inert while `goals_off`) |
| `--reset_state_on_teleport` | `0` | "only try true if you have a very very good reason" |
| refreshing | all off | "same envs in every update"; wall/goal refresh allowed later with a reason, **env locations never** |
| `--steps_per_rollout` | `200` | named |
| `--wandb_project` | `train_navigate` | named |
| `SCHEDULE` | not v35's | "ignore SCHEDULE from v35" |
| `--input_goal_in_memory` | `0` | **never** — Jack calls it cheating |

**Input channels are frozen.** Not on the knob list, and a standing instruction
from the bc-AQ line (memory `feedback_hopfield_nav_bc_inputs`): do not propose
new input channels when remediating; tune regime and hyper-parameters. If the
input set ever looks like the binding constraint, say so as a finding — do not
quietly change it.

### 1.3 Jack's hypotheses, recorded as given

- To explore, the model must learn to **ignore** the Hopfield signal (which
  during explore comes only from distractors); to exploit it must **follow** it
  exactly. So the real skill is *following the Hopfield only when the recalled
  pattern belongs to the env it is currently in*.
- Multi-step Hopfield recall may be what makes that decidable, because it
  supplies a directional/convergence signal. (`--input_hopfield_multistep 1 2 3`
  is already on.)
- **Success rate alone is not a measure of exploitation.** With 200 steps a good
  explorer stumbles onto the goal. `mean_steps` is the discriminating metric.
- Good `mean_steps` should be near the mean start-goal distance. §3.3 computes
  that number.

---

## 2. The environment, as it actually is

Established by reading the source, and it changes what the target behaviours
are. **The arena has no interior obstacles.**

- Movement is `pos_f = clip(pos_f + action * scale, 0, size-1)`
  (`world/vec_env.py:410`). `scale = continuous_scale = 1.0`. Nothing blocks a
  step; the only constraint is the clip at the boundary.
- The four "walls" are planes at `x, y = -0.5` and `size-0.5`. Their content is
  a `(4, size * wall_resolution)` array of ±1 codes (`world/env.py:328`) which
  the foveal cone ray-casts (`world/env.py:141-166`). **Walls are landmark
  texture for localization, not geometry.**
- Position is float; cell identity — and therefore coverage — is
  `clip(round(pos_f), 0, size-1)` (`world/vec_env.py:275`).
- Heading is the direction actually travelled (`world/vec_env.py:423`), and it
  rotates the sensory cone. A step absorbed by the clip leaves heading
  unchanged.

Consequences that drive every decision below:

1. **A step can add at most one new cell.** So `mean_coverage ≤ (T+1)/size²`.
   At `size=20, T=200` that is **0.5025**, a hard ceiling.
2. **Optimal explore is a lawnmower sweep**; optimal exploit is a straight line.
3. **Pressing into a wall is catastrophic**, because the env clips rather than
   sliding or reflecting: a policy that goes straight and does not turn burns
   every remaining step in one cell. This is the mechanical basis of the
   "corner trap" recorded in `project_hopfield_nav_v18d20_d30`.
4. The agent gets **no `prev_action` and no `encoded_state`** channel, so it
   cannot path-integrate its own displacement. Localization has to come from
   the wall-texture cone. `wall_resolution=4` exists to make that cone
   position-specific *within* a cell.

---

## 3. Reference lines — what the numbers can be

### 3.1 Behaviour classes, measured

`analysis/nav_tri/coverage_baselines.py` — pure geometry, no GPU, seconds to
run. It reimplements the two lines of `vec_env.step_batch` that matter and
scores scripted policies under exactly the coverage definition
`evaluate_exploration` uses.

```bash
$HOME/.conda/envs/cls/bin/python -m analysis.nav_tri.coverage_baselines \
    --steps 200 --trials 64
```

At `size=20`, `T=200`, 64 trials, unit step:

| policy | `mean_coverage` | cells/step |
|---|---|---|
| **ceiling** (one new cell every step) | **0.5025** | 1.000 |
| serpentine / lawnmower | 0.478 | 0.955 |
| **billiard** — straight, specular bounce at wall | **0.387** | 0.775 |
| persistent walk σ=1.0 rad/step | 0.286 | 0.572 |
| persistent walk σ=0.5 | 0.240 | 0.481 |
| uniform random walk | 0.178 | 0.356 |
| persistent walk σ=0.2 | 0.108 | 0.216 |
| persistent walk σ=0.05 (≈ straight, no turn) | **0.050** | 0.100 |

The non-monotonicity is the point: **straighter is better only if you turn at
the wall.** σ=0.05 is worse than a uniform random walk by 3.5×, because it
drives into a boundary and the clip eats the rest of the episode. So
"persistence" is only valuable when paired with wall-triggered turning — which
is exactly the `PERSISTENCE_BONUS` × `WALL_PENALTY` pair.

#### The ladder the agent can actually stand on

The rungs above assume perfect information. Two more, added once P0.9 measured
what the sensory cone supplies, are the ones to score against:

| policy | `mean_coverage` | cells/step |
|---|---|---|
| ceiling | 0.5025 | 1.000 |
| lawnmower — **needs position, and P0.9 says that is not decodable** | 0.478 | 0.955 |
| billiard, perfect wall knowledge | 0.366 | 0.732 |
| billiard, wall distance at R²=0.45 (`wall_resolution=1`) | 0.361 | 0.722 |
| **billiard, wall distance at R²=0.27 (`wall_resolution=4`, instructed)** | **0.352** | **0.704** |
| billiard, R²=0.17 (`wall_resolution=8`) | 0.349 | 0.699 |
| **run-and-tumble, no wall sensing at all, `p_turn ≈ 0.25`** | **0.274** | 0.548 |
| uniform random walk | 0.178 | 0.356 |

Three things follow, and they set the targets for every explore wave:

1. **The practical target is ≈0.35**, not 0.387 and certainly not 0.478.
2. **Wall sensing is worth +0.076 coverage (+28%)** over turning at random —
   real, and the largest single behavioural gain available.
3. **`wall_resolution` barely matters**: 1 vs 4 is 0.361 vs 0.352, a gap of
   0.009, because the behaviour saturates long before the decoder does. P0.9
   showed the *decodability* differs by a lot (R² 0.45 vs 0.27); this shows it
   does not cash out. **So the instructed `wall_resolution=4` is fine and is
   not worth a variant.**
4. A memoryless run-and-tumble at `p_turn ≈ 0.25` — hold a heading for ~4
   steps, then re-orient, with no sensing whatsoever — already reaches 0.274.
   That is the floor a competent policy must clear, and it needs no state at
   all beyond the heading the env supplies for free.

**Billiard is the realistic near-term target**, because it is *reactive*: it
needs only "am I about to hit a wall". The lawnmower needs to know where it has
already been.

> P0.9 later made this quantitative and turned it from a guess into a ceiling:
> wall proximity is decodable from the sensory cone at R² ≈ 0.27, absolute
> position at R² ≤ 0.13. **The lawnmower line is not reachable in a held-out
> env; 0.387 is the practical ceiling**, and the ray-cast cone reports wall
> distance only *implicitly*, not "directly" as this paragraph first assumed.

### 3.2 What each noise source costs coverage

Same script, billiard base:

| `epsilon_explore` | 0 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| `mean_coverage` | 0.385 | 0.377 | 0.365 | 0.356 | 0.323 |

| `init_log_std` | −1.8 | −1.2 | −0.8 | −0.5 | 0.0 |
|---|---|---|---|---|---|
| σ per component | 0.165 | 0.301 | 0.449 | 0.607 | 1.000 |
| `mean_coverage` | 0.384 | 0.373 | 0.352 | 0.351 | 0.351 |

Read carefully: **eval scores the deterministic policy, so neither of these is
a direct cost to the reported metric.** What the table prices is the damage to
the *behaviour* policy PPO learns from — ε=0.4 makes the sampled trajectory 16%
worse than the policy that produced it, and the novelty reward those steps earn
is credited to a policy that did not choose them. `init_log_std=-1.8` is
essentially free by this measure (−0.3%), which is a point in favour of buying
exploration with σ rather than with ε.

### 3.3 What good `mean_steps` is

Start and goal are both drawn uniformly on the 20×20 grid, so:

- mean start→goal Euclidean distance = **10.50**
- with `goal_radius = 1.0` and unit steps, an oracle beeline takes
  **mean 9.50 steps** (median 9.30)

So **`mean_steps ≈ 9.5` is the floor** and anything near ~10–12 is excellent.
The v35 lineage's reference point was 22.9 — about 2.4× the floor.

#### 3.3.1 `success_rate` is nearly uninformative here — and the reference table

`analysis/nav_tri/exploit_reference.py` simulates a policy that does nothing but
follow a direction of a given cosine accuracy, at a given step magnitude, under
the real nav protocol (open arena, ends on arrival, `goal_radius=1`, 200-step
budget, action σ = 0.165).

```bash
$HOME/.conda/envs/cls/bin/python -m analysis.nav_tri.exploit_reference
```

**`success_rate` is 1.000 in every single row** — including cos-accuracy 0.50,
i.e. a direction wrong by 60° on average. In an obstacle-free 20×20 arena with
200 steps, essentially any goalward drift arrives. This confirms Jack's warning
in the strongest possible form: **success rate cannot distinguish a good
navigator from a bad one in this setup, and should be read as a sanity check
only.** `mean_steps` carries the entire exploit signal.

`mean_steps`, successes only, as `metrics.py:351-354` computes it:

| cos(q, goal) | \|a\|=0.75 | \|a\|=1.0 | \|a\|=1.5 | \|a\|=2.0 |
|---|---|---|---|---|
| 1.00 (oracle) | 13.2 | **10.0** | 6.8 | 5.3 |
| 0.99 | 13.3 | **10.1** | 6.9 | 5.4 |
| 0.95 | 14.0 | 10.6 | 7.3 | 5.9 |
| 0.90 | 14.9 | 11.4 | 7.9 | 6.5 |
| 0.82 | 16.6 | **12.7** | 9.0 | 7.5 |
| 0.70 | 19.8 | 15.3 | 11.0 | 9.3 |
| 0.50 | 28.6 | **22.4** | 16.4 | 14.2 |

**How to use this table.** Run `behavior_probe.py` on a checkpoint, read off its
`q_accuracy` and `step_mag_mean`, and look up the row. At or near the reference
⇒ the policy already follows the signal as well as the signal permits, and nav
tuning is pointless because the *readout* is the limit. Well above it ⇒ the
policy is failing to follow a signal that is demonstrably there, which is an
optimization problem worth fixing.

Applied to the numbers already in hand: P0.7 measures the readout at cos **0.99
at every distractor count, 0 through 10**, so the reachable `mean_steps` at unit
steps is **≈10.1 regardless of distractors** — the same row as an oracle. The
v35 lineage's reference point of 22.9 sits on the cos ≈ 0.50 row, i.e. roughly
**2× more steps than its readout permitted**. That gap is the exploit headroom,
and it is entirely a policy gap.

**Target for this line of work: `mean_steps` ≤ 12 at mean |a| ≈ 1**, with
`success_rate` ≥ 0.98 as a sanity check rather than an achievement.

**`mean_steps` is gameable and must always be reported with step magnitude.**
`continuous_normalize=False` leaves `|a|` free, and at `|a| = 2` the oracle
figure drops to 4.75 without the policy having got any better at navigating.
Every nav number in this document is therefore quoted next to the measured mean
`|a|`, and a run whose `mean_steps` fell while `|a|` rose is recorded as a step-
size change, not a navigation improvement. (It also trades against coverage —
§3.1's magnitude sweep peaks at exactly `|a| = 1`.)

### 3.4 The three knobs Jack flagged

**`INIT_LOG_STD` / `FREEZE_LOG_STD`.** The movement head is a 2-D Gaussian;
`init_log_std` sets `log σ` per component and `freeze_log_std` decides whether σ
takes gradient. σ is in *cells*: at −1.8, σ = 0.165 cell, which on a unit step
is ≈9.5° of heading jitter.

Two things make this knob load-bearing rather than cosmetic:

- σ is the **only** exploration the policy itself owns (ε is an external
  override). Frozen high = permanent exploration and a permanently blurred
  eval-vs-train gap; frozen low = sharp behaviour but PPO sees almost no
  variation to learn from. Learnable σ collapses toward 0 once returns are
  positive, which ends exploration early.
- **`--freeze_log_std` was a no-op on `train_navigate` until 2026-08-07**
  (`docs/EXPERIMENTS_EXPLORE_MIN.md`, memory
  `project_hopfield_nav_log_std_freeze_bug`). The whole v35 lineage trained a
  *learnable* σ whatever its launcher said. So v35's `-1.8 / freeze=1` is not a
  configuration anyone has actually validated — it is a configuration that was
  silently overridden. Treat both as open.

**`MOVE_ENT_COEF`.** The PPO entropy bonus on the movement distribution. For a
Gaussian, differential entropy is `Σ log σ + const` — it depends on **σ alone,
not on the mean.** So:

- with `freeze_log_std=1`, `move_ent_coef` multiplies a **constant** and its
  gradient is exactly zero. It is a no-op.
- with `freeze_log_std=0`, it is a *pressure to keep σ large*, and nothing else.

This is a genuine interaction and the reason the three are flagged together:
`MOVE_ENT_COEF` is meaningless except as the counter-pressure that stops a
learnable σ from collapsing. Recorded as a prediction to verify (§4, T3).

---

### 3.4.1 Why σ moves the magnitude, and whether the clip is in the way

The policy has to move its mean step from 0.086 cells to ~1.0 (P0.6). Three
things govern how fast, and only two of them turn out to matter.

**σ, through credit assignment, not through gradient size.** For a Gaussian,
`∇_μ log π = (a − μ)/σ²`, so *smaller* σ gives a **larger** per-sample
gradient — the opposite of the observed effect. What σ actually buys is the
*range of magnitudes the policy ever samples*: at σ=0.165 around a mean of
0.09, the policy never executes a step longer than ~0.6, so it has no evidence
that 1.0 is better and must crawl. At σ=0.50 the sampled steps span the whole
useful range in one update. This is exploration in **action space**, and it is
why `w1_sig` doubles `w1_eps01` at u50.

**The PPO clip, which is NOT binding.** For a mean shift δ, the ratio is
`exp((a−μ)δ/σ² − δ²/2σ²)`, so on a typical sample `log ratio ≈ δ/σ` and
`clip_coef = 0.15` permits `δ ≲ 0.15 σ` per gradient step — about **0.025
cells/update** at σ=0.165. The *measured* rate is 0.086 → 0.178 over 75
updates = **0.0012 cells/update**, twenty times below the clip ceiling.

> **So the ascent is gradient-limited, not clip-limited**, and
> `PPO_CLIP_COEF` is demoted: raising it lifts a ceiling nothing is touching.
> `LR` is promoted by the same arithmetic — it scales δ directly and has ~20×
> of headroom before the clip engages. That is the reasoning behind `w1_lr`
> and against `w1_clip`, and it is arithmetic rather than a guess.

### 3.5 Which knobs are provably inert, and when

Pooled advantage normalization (`updates/ppo.py:210-214`) removes any constant
offset and any overall scale from the reward. That makes several entries on the
knob list dead in specific regimes, and it is cheaper to know that than to sweep
them.

| knob | inert when | why |
|---|---|---|
| `MOVE_ENT_COEF` | `FREEZE_LOG_STD=1` | Gaussian entropy is `Σ log σ + const`; freezing σ makes the whole term a constant with zero gradient. Verified numerically (P0.4). |
| `TIME_PENALTY` | explore regime with `explore_goals_off` | the rollout is fixed-length with no teleport, so `−time_penalty` is the same constant every step and cancels. |
| `GOAL_REWARD` | explore regime with `explore_goals_off` | no goal event exists (`vec_env.py:397-398` forces the mask to zeros). |
| `GOAL_REWARD`, `TIME_PENALTY` | exploit regime **with shaping at zero** | reward becomes `−t + (g+t)·1{goal}`; the constant cancels and `(g+t)` is a pure scale. They survive only through the **value** loss, which is not normalized — so they act as an implicit `vf_coef`, not as a preference. |
| `REVISIT_PENALTY` | explore, `NOVELTY_SCALE_REMAINING=0`, fixed-length rollout | `n·1{new} − c·1{old} = (n+c)·1{new} − c`; exactly redundant with novelty. Non-degenerate once the remaining-scale is on (novelty state-dependent, penalty flat) or the goal is live (rollouts stop being equal-length). |
| `NOVELTY_SCALE_CAP` | 200-step rollouts | the scale is `size²/remaining`, and 200 steps can leave at most ~200 of 400 cells unvisited, so it never exceeds ~2 and a cap of 10 never binds. |

Only **ratios** among the surviving shaping terms matter, never their common
scale.

## 4. Cost model, and how runs are sized

From `docs/EXPERIMENTS_EXPLORE_MIN.md`, re-derived against
`train_navigate.py:328-349`. One update collects **one rollout per env, for
every env in `envs_per_world`**, and pools them all into a single PPO step.
`batch_envs` is the parallel-episode batch *inside* one env's rollout.

```
env-steps / update       = envs_per_world × batch_envs × steps_per_rollout
PPO pool (trajectories)  = envs_per_world × batch_envs
SERIAL model calls / upd = envs_per_world × steps_per_rollout      <-- wall-clock
```

Measured previously on an l40s: 80 envs × 200 steps = 16,000 serial calls =
**30.8 s/update** (~1.9 ms/call). So a 6-hour run at 80 envs is ≈700 updates.

Therefore:

- **`batch_envs` is nearly free to raise** — more data and a bigger PPO pool at
  zero serial cost.
- **`envs_per_world` is the wall-clock lever**, and also the diversity lever.
  Trading it against `batch_envs` at constant pool size is the cheap way to buy
  updates. Untested as of 2026-08-07 — the `c*` ladder in the explore-min doc
  was designed and never submitted. Wave 1 settles it.

### Tests written for this document

| what | where | why |
|---|---|---|
| behaviour-class coverage baselines, noise pricing, oracle `mean_steps` | `analysis/nav_tri/coverage_baselines.py` | §3.1–3.3. Pure geometry, no GPU, seconds. |
| what a checkpoint actually *does* — behaviour class, failure mode, and the readout-vs-policy split | `analysis/nav_tri/behavior_probe.py` | P0.6. Instrumented copy of the two eval protocols. |
| is "goal in memory" decidable from the observation? | `analysis/nav_tri/signal_separability.py` | P0.7. No policy involved — a property of encoder + scaffold + Hopfield. |
| how many steps of `q` does it take to decide? (ideal-observer AUC vs T) | `analysis/nav_tri/temporal_separability.py` | P0.8. Bounds what any architecture could extract. |
| what nav numbers a perfect follower of a given-accuracy signal gets | `analysis/nav_tri/exploit_reference.py` | §3.3.1. Turns `q_accuracy` into a predicted `mean_steps`, so a checkpoint's nav gap is diagnosable as readout vs policy. |
| what is decodable from the 60-ray sensory cone at all | `analysis/nav_tri/sensory_decodability.py` | P0.9. Pure sensor property; no encoder, no scaffold, seconds. |
| probe launcher, **CPU — prefer this one** | `hopfield_nav/run_nav_tri_probe_cpu.sh` | `PROBE=signal\|temporal\|behavior CKPTS="…" sbatch …`. The probes need no GPU: the only heavy step is one encoder pass to build `encoded_Phi`. Both GPU partitions cap concurrency at 2 and training holds those for six hours, while `mit_normal` has 3000 cores — so this runs immediately instead of waiting out a training run. |
| probe launcher, GPU | `hopfield_nav/run_nav_tri_probe.sh` | same interface; only worth it when a GPU slot is genuinely idle |
| training launcher | `hopfield_nav/run_nav_tri.sh` | `VARIANT=<name> sbatch …`; variants in the `case` block |
| queue + latest metric per run | `hopfield_nav/nav_tri_status.sh` | pulled, not pushed |
| one line per job that leaves the queue | `hopfield_nav/nav_tri_watch.sh` | for `Monitor`; deliberately silent per-update |

Both probes take `--npos` to shrink the scaffold so every code path runs on a
CPU in a minute (`encoded_Phi` is 12 GB at the real Npos=1716). **That mode is
for validating the tool, never for a number** — it changes the geometry being
measured, and it refuses the recorded `world.json` because the offsets would
index a different scaffold.

### Code changes made on this branch

| change | why |
|---|---|
| `navigate_job.sh`: added `WALL_RESOLUTION`, `EGOCENTRIC_HEADING`, `RESET_STATE_ON_TELEPORT`, `EXPLORE_ENDS_ON_GOAL`, `RNN_CELL`, `RNN_NONLINEARITY` | six flags `train_navigate` accepts that the pass-through was missing, four of them required by §1.2. The file's stated contract is that every flag has an env var. |
| `world/scaffold.py`: `precompute_encoded_phi` writes into a preallocated array instead of `concatenate`-ing a list of parts, and frees `sgb`/`flat` early | peak RSS was **44.5 GB for a 12 GB answer**, which on a memory-contended node is the difference between scheduling and queueing. Bit-identical output; the 67 golden/signal/smoke tests pass unchanged. |

---

### 4.1 Things outside the knob list that would matter — flagged, not used

Recorded rather than acted on, because §1.1 is an explicit list and these are
not on it. Each is a one-line change if Jack wants it.

- **`--min_action_norm 1.0`** would fix the immobility of P0.6 outright: the env
  scales any sub-threshold action up to unit length, so the policy would move a
  full cell from update 1 and would only have to learn *direction*. As it
  stands, a large part of the explore run is spent learning magnitude — a
  scalar — through a noisy policy gradient. This is the single largest
  intervention available and it is one flag. It does change the action space,
  and it caps the "take bigger steps" route to a low `mean_steps` (§3.3), which
  is arguably a *feature* given that route games the metric.
- **`--input_prev_action`** would let the RNN path-integrate its own
  displacement, which is what a lawnmower sweep needs and what the current input
  set cannot support (§2.4). Deliberately not proposed: the input set is frozen
  by standing instruction (§1.2). Flagged because if coverage plateaus around
  the billiard line (0.387) rather than climbing toward the lawnmower line
  (0.478), **that is the reason**, and no amount of shaping will fix it.
- **Randomizing which envs get which regime.** `train_navigate.py:335` assigns
  regimes positionally, so at a fixed `empty_frac` the same envs are always
  exploit. See the open item in §0.

## 5. Evaluation protocol used here

- **In-training (monitoring).** `EVAL_MAX_STEPS=200` pinned so coverage is
  always the same measurement, and equal to the training rollout length.
  Distractor levels `0 10`. Cheap env/trial counts. **Never a verdict.** The
  explore-min wave found the cheap estimate biased high by ~0.02 and — worse —
  wrong about *ranking*.
- **Verdict.** Offline pass over saved checkpoints, 10 val envs × 32 trials,
  `n_dist ∈ {0, 5, 10}`, 200 steps, deterministic policy.
- Every nav result is quoted as **(success_rate, mean_steps, mean |a|)**
  together, per §3.3.

---

## 6. Waves

### Wave 0 — probes, before anything was trained

Three things were established before a single GPU-hour was spent, because all
three change what the later waves should even try.

**P0.1 — the arena is empty.** §2. Read off `world/vec_env.py` and
`world/env.py`. Consequence: the target behaviours are a lawnmower and a
straight line, and the coverage ceiling at 200 steps is 0.5025.

**P0.2 — behaviour-class reference ladder.** §3.1–3.3, via
`analysis/nav_tri/coverage_baselines.py`. Consequence: a *billiard* — straight,
turn at the wall — is the realistic target at coverage 0.387, it is reactive
enough to be learnable from the ray-cast cone alone, and "go straight" without
"turn at the wall" is catastrophically worse than a random walk (0.050 vs
0.178) because the env clips.

**P0.3 — the trainer's mechanics.** Read out of the source; the four that
changed a decision:

1. **Shaping leaks across regimes.** Only `novelty_reward` and `goals_active`
   are set per regime (`train_navigate.py:340-341`). `revisit_penalty`,
   `wall_penalty` and `persistence_bonus` are read straight off `cfg.hopfield`,
   so they apply to **exploit** rollouts too, despite docstrings in `config.py`
   saying "explore phase only". At exploit's reward scale (+5 per goal, ~20
   goals per rollout) `wall_penalty` is a −3.8 nuisance and `persistence_bonus`
   is a +10 *help* — small, but it means the shaping chosen for explore is not
   free to choose in the combined phase.
2. **ε steps are dropped from the PPO movement surrogate**
   (`collector.py:479-485`, `updates/ppo.py:267-268`). So ε does not teach the
   policy to act randomly — it *discards* that fraction of the policy gradient
   and widens the state distribution. ε=0.4 therefore costs 40% of the movement
   gradient per update.
3. **Advantages are normalized once, over the whole pool**
   (`updates/ppo.py:210-214`). Only shaping *ratios* matter; a constant per-step
   term cancels exactly. `time_penalty` is such a constant under `goals_off`,
   and `goal_reward` is inert there, so neither is an explore-phase knob at all.
4. **`mean_steps` is over successful trials only, and reads 0.0 — not NaN —
   when there are no successes** (`metrics.py:351-354`). It is
   survivorship-biased downward and must never be read without `success_rate`
   beside it.

**P0.4 — `MOVE_ENT_COEF` is a no-op under `FREEZE_LOG_STD=1`.** Predicted from
the Gaussian entropy identity (§3.4), then confirmed numerically in the smoke
run: `move_entropy = -0.762` at `init_log_std = -1.8`, exactly
`2(½log 2πe + log σ)`. The entropy term depends on σ alone; freezing σ makes it
a constant with zero gradient. **So `MOVE_ENT_COEF` cannot be evaluated
independently of `FREEZE_LOG_STD`, and every v35-lineage run that set both was
sweeping a dead knob.**

**P0.5 — measured cost.** 80 envs × 200 steps = **38.3 s/update** on an
`mit_normal_gpu` l40s with `observation_size=60`, `wall_resolution=4` and the
RNN/ReLU trunk. A 6-hour budget is therefore ≈450 updates at that rollout
shape. `input_dim = 70`.

**P0.6 — the untrained policy is IMMOBILE, and ε is the only thing that moves
it.** Measured with `analysis/nav_tri/behavior_probe.py` on a 6-update
checkpoint: `step_mag_mean = 0.086` cells, `revisit_frac = 0.91`, coverage
0.031. The initial failure mode is not the corner trap — the policy simply does
not move. Two consequences, and the second is the important one:

- **The policy's Gaussian σ is twice its mean.** At `init_log_std = -1.8`,
  σ = 0.165 against a mean magnitude of 0.086, so early behaviour is
  noise-dominated. The movement head has to grow its output ~12× to reach the
  |a| = 1 that §3.1 shows is the coverage optimum.
- **ε actions are unit-magnitude by construction** (`collector.py:302-317`
  emits `[cos θ, sin θ]`, *not* scaled by the policy's own magnitude). So at
  ε=0.4 two of every five steps is a full-size move while the policy's own are
  ~0.09. Confirmed in the live runs at u1–u10: `w1_base` (ε=0.4) sits at
  `mean_r = -0.0018` and `w1_eps01` (ε=0.1) at `-0.0396`; backing out the
  shaping terms, that is ≈0.23 new cells/step against ≈0.03.

  **But ε steps are masked out of the movement surrogate (P0.3.2).** So ε
  supplies the *motion* and none of the *gradient that would produce motion*.
  σ supplies both, because the policy is scored on the action it actually
  sampled. That asymmetry is the mechanistic reason to expect `w1_sig` to beat
  `w1_eps01`, and it reframes `INIT_LOG_STD` from "exploration temperature" to
  "the channel through which the policy learns its own step size".

**P0.7 — the readout is excellent at few distractors and degrades sharply by
ten.** `analysis/nav_tri/signal_separability.py`, no policy involved. Per env,
sampling 400 cells and comparing memory with the goal present against the same
distractors without it. **Npos=1716 (the real scaffold), 8 envs = 8 independent
distractor draws. This table is the authoritative one — see the two corrections
below it.**

| `n_dist` | `q_goal_mean` | `q_dist_mean` | `dir_acc_goal` = cos(q, goal−cell) | `recall_is_goal_frac` | **`AUC(\|q\|)`** |
|---|---|---|---|---|---|
| 0 | 0.255 | 0.000 | **0.992** | 1.00 | — |
| 1 | 0.269 | 0.056 | **0.994** | 1.00 | 0.955 |
| 3 | 0.252 | 0.062 | **0.994** | 1.00 | 0.936 |
| 5 | 0.259 | 0.120 | 0.972 | 1.00 | 0.836 |
| 10 | 0.218 | 0.170 | **0.696** | 1.00 | **0.619** |

- **Up to ~3 distractors the readout is essentially perfect** (cos 0.99) and the
  regime is separable from `|q|` alone (AUC 0.94–0.96). Via §3.3.1 that is
  `mean_steps` ≈ 10.1 at unit steps — the oracle row.
- **By 10 distractors both collapse**: direction accuracy 0.70, separability
  0.62. Via §3.3.1 the reachable `mean_steps` at d=10 is ≈**15.3**, not 10.1.
  So **the exploit target is distractor-dependent and must be quoted per
  level**, and a run scored only at d=0 will look better than it is.
- **`recall_is_goal_frac` stays 1.00 throughout, and that metric is a trap.**
  The attractor is always nearer the goal pattern than to any distractor, yet
  the *direction* still degrades — because recall is
  `normalize(tanh(β W x))`, a soft mixture over all stored patterns, not a
  verbatim retrieval. It lands closest to the goal while being pulled off it.
  **`dir_acc_goal` is the metric that matters; the margin is not.**
- `--input_hopfield_raw 1` is what puts the separating quantity in the
  observation at all — the normalized signal discards `|q|`. Jack's instruction
  to use the raw signal is load-bearing.

> **CORRECTION 1 (superseded).** A first version of this section, on a shrunken
> **Npos=300** scaffold, reported `AUC(|q|)` 0.35–0.49 and "no single-step cue".
> That was an artifact of the small scaffold: the exclusion region distractors
> are drawn from is ~30× smaller there, so they sit close enough to interfere.
>
> **CORRECTION 2 (superseded).** A second version, at the real Npos but with
> **2 envs = 2 distractor draws**, reported `dir_acc_goal` 0.99 and `AUC(|q|)`
> 0.90–0.97 at *every* level, and concluded the distractor problem was "solved
> upstream by the `repel_weight=2` encoder". **That was a two-sample fluke.**
> At 8 draws the degradation with distractor count is clear and monotonic.
>
> **The lesson, recorded because it cost two wrong conclusions: the number of
> independent distractor draws dominates the variance of every number in this
> table.** Never read it at fewer than ~8. The independent `walk` column of P0.8
> (16 draws) agreed with the 8-draw figure (0.714 vs 0.619 at d=10) and
> disagreed with the 2-draw one (0.956), which is what exposed the fluke —
> cross-checking two probes that measure the same quantity by different routes
> is what caught this, and is worth keeping up.

**P0.9 — the agent cannot tell where it is, and only weakly whether a wall is
near.** `analysis/nav_tri/sensory_decodability.py`. Needs no encoder or
scaffold: this is a property of the sensor. Decode a target from one 60-ray
observation at a random (position, heading), R² on held out 30%, decoder fit
*per env* (so this is the generous within-env case):

| `wall_resolution` | dist-to-wall: ridge / MLP / **autocorr** | pos_x | pos_y | heading |
|---|---|---|---|---|
| 1 | −0.01 / 0.30 / **0.45** | 0.13 | 0.07 | 0.16 |
| **4** *(instructed)* | −0.01 / 0.21 / **0.27** | 0.03 | 0.06 | 0.05 |
| 8 | −0.01 / 0.07 / **0.17** | 0.02 | 0.02 | 0.06 |

The "autocorr" decoder is a ridge on lag-*k* ray agreement plus the sign-change
rate — the one route that is **codebook-independent** and therefore usable in a
held-out env: how fast the pattern varies across the cone is geometry, not
code. Standing near a wall, adjacent rays land on adjacent segments and agree;
standing far, they land on distant segments and are independent.

- **Absolute position is not decodable** — R² ≤ 0.13 by any decoder at any
  resolution. The wall code is random ±1 per segment, so the observation is a
  *hash* of position, not a smooth function of it: unique (which is what
  `wall_resolution` was raised to achieve) but not invertible by anything that
  generalizes. **A lawnmower sweep needs to know where it has been, so the
  lawnmower line (coverage 0.478) is out of reach in a held-out env, and the
  billiard line (0.387) is the real practical ceiling.**
- **Wall proximity is present but weak** (R² 0.27 at the instructed resolution),
  and only through the codebook-independent autocorrelation route. Enough for a
  reactive turn — see below — and the RNN can integrate across steps, which
  this single-observation probe does not credit.
- **`wall_resolution` trades cell-uniqueness against smooth geometry**, which is
  why every column *falls* as it rises: a finer hash is more unique and less
  invertible. Its documented purpose (`config.py:110-124`) is to stop distinct
  cells sharing a bit-identical observation, which serves env identity — a thing
  the policy cannot exploit in a held-out env anyway.

  > **But it does not cash out, so the instruction is fine.** Feeding these R²
  > values into the noisy-billiard simulation in §3.1 gives coverage 0.361 at
  > `wall_resolution=1` against **0.352 at the instructed 4** — a gap of 0.009,
  > because the behaviour saturates long before the decoder does. An earlier
  > version of this section proposed a `w2_wallres1` variant to measure the
  > cost of the instruction; **that variant was dropped**, because simulating
  > it first showed there is no cost worth a GPU-hour. What the same simulation
  > *does* show is that having wall sensing at all is worth +0.076 coverage
  > over turning at random, so the weak-but-nonzero R² is doing real work.

**P0.8 — the cue is temporal, and it only appears if the agent ACTS on the
signal.** `analysis/nav_tri/temporal_separability.py` puts an ideal-observer
bound on the question: extract features from the first *T* values of `q` along
a trajectory (magnitude statistics, direction persistence, the scatter of the
implied targets `x_t + q_t`, and the slope of `|q|` in time), fit a logistic
regression, report held-out AUC. No policy is trained, so this is an upper
bound on what **any** architecture could extract. `n_dist=3`, Npos=300
(validation-grade):

| trajectory | T=1 | T=2 | T=5 | T=10 |
|---|---|---|---|---|
| **walk** — uniform random steps, i.e. deciding *while exploring* | 0.569 | 0.724 | 0.715 | 0.759 |
| **follow** — steps along `q`, i.e. probing the signal | 0.675 | 0.816 | 0.889 | **0.933** |

The mechanism is the one predicted: a real in-env goal is a **fixed point**, so
walking toward it makes `|q|` shrink and the implied targets `x_t + q_t`
cluster; a distractor is not, so neither happens. Passive observation gets to
≈0.76 and stalls. **Acting on the signal is what makes it decidable**, and ten
steps of following buys AUC 0.93 — 5% of a 200-step episode.

#### P0.8 at the real scaffold — the finding sharpens

Re-run at Npos=1716, 8 memory draws × 32 trajectories per env per condition:

| mode | `n_dist` | T=1 | T=2 | T=5 | T=10 | T=20 |
|---|---|---|---|---|---|---|
| walk | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| walk | 3 | 0.876 | 0.894 | 0.888 | 0.888 | 0.891 |
| walk | 10 | 0.714 | 0.776 | 0.793 | 0.779 | 0.731 |
| **follow** | 3 | 0.735 | 0.813 | 0.928 | 0.966 | **0.995** |
| **follow** | 10 | 0.817 | 0.865 | 0.969 | 0.981 | **0.982** |

(`n_dist=0` is 1.000 by construction: the goal-absent memory is *empty*, so
`q` is identically zero.)

**Passive observation plateaus; acting does not.** Watching while you walk gets
to ≈0.89 at three distractors and ≈0.78 at ten, and **stays there** — twenty
steps of passive evidence is worth no more than two. Following the signal
climbs monotonically to **0.98–0.995**. The mechanism is the one predicted: a
real in-env goal is a fixed point, so moving toward it shrinks `|q|` and
clusters the implied targets `x_t + q_t`, while a phantom does neither — and
that evidence only exists if the agent moves along `q`.

So the single-step cue of P0.7 is real but **not sufficient on its own at high
distractor counts**, and the probe-and-verify behaviour is what closes the gap
from ~0.78 to ~0.98. Ten steps of probing costs 5% of a 200-step episode.

> Note the two probes disagree at `n_dist=10`: P0.7's `AUC(|q|)` reads 0.956
> where this reads 0.714 for the same quantity at T=1. P0.7 drew distractors
> **once per env over 2 envs**, i.e. two draws total, while this draws 8 per
> env — so P0.7's figure is thin and this one is the more trustworthy of the
> two. A re-run of P0.7 at 8 envs is in flight to settle it; until it lands,
> **read the single-step separability as ≈0.7–0.9, not ≈0.95**.

**Consequences for the ordering.** The risk to explore-first is *not* removed by
P0.7's correction:

- Under `explore_goals_off` with distractors, following `q` earns nothing and
  costs steps, so a long pure-explore phase trains the policy to ignore the
  recall channel — and, worse, to never *probe* it. Probing is precisely the
  behaviour that carries separability from 0.78 to 0.98.
- **Wave 3 therefore keeps an interleaved arm beside the requested
  explore-first arm**, rather than running it only if explore-first fails.

---

### Wave 1 — the baseline, the cost/diversity ladder, and the noise regime

Submitted 2026-08-14. Pure explore (`explore_goals_off=1`, `eval_scope=expl`,
`EVAL_MAX_STEPS=200`, `VAL_DISTRACTORS="0 10"`, 6 val envs × 16 trials).

**Why these three questions and not shaping.** The reward landscape already
prefers the right answer. Scoring the scripted policies of §3.1 under v35's
actual shaping (novelty 0.3 with remaining-scale, wall 0.1, persistence 0.05)
gives per-step returns of **+0.356 for a billiard, +0.098 for a random walk,
+0.081 for a perimeter orbit, −0.02 for drive-into-a-wall**. The billiard wins
by 3.6× over the random walk and the two known failure basins are already
priced below it. So the explore problem is an **optimization** problem, not an
objective-design problem — which is also what the explore-min wave concluded
empirically ("the shaping axis produced nothing"). Wave 1 therefore spends
itself on optimization budget and exploration noise, and buys only one ticket
in the shaping lottery.

| variant | change from `w1_base` | updates | envs × batch | serial calls/upd |
|---|---|---|---|---|
| `w1_base` | — (v35 shaping + rollout shape, under §1.2) | 450 | 80 × 16 | 16,000 |
| `w1_c20` | pool held at 1280, envs ÷4 | 2400 | 20 × 64 | 4,000 |
| `w1_c8` | pool held at 1280, envs ÷10 | 6000 | 8 × 160 | 1,600 |
| `w1_eps01` | `EPSILON_EXPLORE` 0.4 → 0.1 | 450 | 80 × 16 | 16,000 |
| `w1_sig` | ε 0.1 **and** `INIT_LOG_STD` −1.8 → −1.2 | 450 | 80 × 16 | 16,000 |
| `w1_pers` | `PERSISTENCE_BONUS` 0.05 → 0.15 | 450 | 80 × 16 | 16,000 |

**Q1 — is diversity or gradient-batch the binding constraint?** `envs_per_world`
sets both how many distinct envs exist and how many rollouts enter each PPO
step, so lowering it alone confounds two effects with opposite fixes. The ladder
holds the PPO pool at 1280 trajectories and 256,000 env-steps per update — the
only thing that changes is how many distinct envs those come from — while the
serial call count, which is what wall-clock tracks, falls 4× and 10×. Eval is
held out (a separate world with its own scaffold and envs), so a diversity
failure shows up as a train/eval gap rather than being invisible.

- *Decision rule.* If `w1_c20` matches `w1_base` at u450, **every later wave
  runs at 20 envs** and gets 4× the updates per GPU-hour. If it matches and
  `w1_c8` does not, the floor is between 8 and 20. If `w1_c20` falls short at
  equal updates, diversity is binding and the 80-env shape stays.
- The ladder runs long (2400 / 6000) on purpose: "few envs is slower" and "few
  envs is capped" are indistinguishable at a fixed update count and mean
  opposite things. All three still report at u450 for the fixed-update read.

**Q2 — is ε=0.4 worth its price?** Three separate mechanisms say it is
expensive, and they are independent of each other:

1. **It discards 40% of the movement gradient.** ε steps are masked out of the
   PPO surrogate (P0.3.2), so four in ten steps teach the policy nothing about
   what to do.
2. **It corrupts the policy's estimate of its own heading.** The input set has
   **no `prev_action` channel**, so the only way the policy knows which way it
   is currently travelling is that *it chose* the action — the RNN's hidden
   state carries the intention forward. An ε step replaces the executed action
   without telling the policy, so after it the internal heading is wrong, and
   the only way back is the sensory cone, which P0.9 decodes heading from at
   R² ≈ 0.05. **Going straight is exactly the behaviour this breaks**, and
   going straight is the defining feature of every good rung on the §3.1
   ladder.
3. **It corrupts `persistence_bonus`.** That term is
   `cos(a_t, a_{t−1})` on the *executed* actions (`collector.py:620-638`), so
   at ε=0.4 roughly 64% of consecutive pairs contain at least one random
   vector, and the straightness reward the shaping is paying for is mostly
   noise credited to a policy that did not produce it.

Against all that, ε is the only mechanism producing a *large* re-orientation,
and at σ=0.165 the policy's own noise is ~9.5° of heading jitter per step — and
P0.6 showed ε is also what supplies most of the early *motion*, since its
actions are unit-length while the policy's start at 0.086.

- *Decision rule.* `w1_eps01` > `w1_base` → ε was over-bought; carry 0.1
  forward. `w1_eps01` < `w1_base` → the large re-orientations are doing real
  work, and the next question is whether they are still needed late (anneal
  schedule) rather than whether to have them.

**Q3 — can σ buy the same exploration more cheaply than ε?** σ noise is *inside*
the policy, so PPO keeps the gradient (no masking) and the sampled action is
scored honestly; §3.2 prices σ=0.30 at −3% coverage against ε=0.4's −16%.
`w1_sig` moves both at once, so it is only interpretable against `w1_eps01`:
the pair isolates σ with ε held at 0.1.

**Q4 — does straightness need to be paid for directly?** At v35's ratio, novelty
is worth ~0.30/step to a billiard and persistence at most 0.05/step — 6:1
against the term that rewards the target behaviour's defining feature.
`w1_pers` makes it 2:1. Given the return analysis above this is the one shaping
ticket the wave buys, and the expectation is that it does nothing.

**What is deliberately NOT in this wave.** `LR`, `PPO_CLIP_COEF`,
`HIDDEN_SIZE`, `NOVELTY_*`, `REVISIT_PENALTY`, `WALL_PENALTY`, `TIME_PENALTY`,
`GOAL_REWARD` — the last two are provably inert under `explore_goals_off`
(P0.3.3), and the rest wait on the behaviour diagnosis from
`analysis/nav_tri/behavior_probe.py`, which says *which* failure the baseline
has before anything is tuned to fix it.

#### Results — wave 1

*(pending)*

| variant | cov d0 @u450 | cov d10 | cells/step | s/u | notes |
|---|---|---|---|---|---|
| `w1_base` | | | | 38.3 | |
| `w1_c20` | | | | | |
| `w1_c8` | | | | | |
| `w1_eps01` | | | | | |
| `w1_sig` | | | | | |
| `w1_pers` | | | | | |

#### Live notes — wave 1 (to be folded into Conclusions)

- **u75, behaviour probe on both checkpoints — Q2 answered mechanistically, and
  a bigger finding underneath it.** `analysis/nav_tri/behavior_probe.py`,
  explore mode, matched update index:

  | | `w1_base` (ε=0.4) | `w1_eps01` (ε=0.1) |
  |---|---|---|
  | `mean_coverage` | 0.0418 | **0.0666** |
  | **`step_mag_mean`** | 0.178 | **0.251** |
  | `straightness` | 0.472 | **0.575** |
  | `edge_frac` | 0.288 | 0.200 |
  | `clip_frac` | 0.157 | 0.104 |
  | `revisit_frac` | 0.916 | 0.867 |
  | `chase_q` (d=10) | −0.022 | — |

  Lower ε gives a **41% larger step**, a straighter path, less time on the
  perimeter and less wall-clipping. That is the predicted mechanism showing up
  in the behaviour and not merely in the score: at ε=0.1 nine steps in ten
  produce policy gradient instead of six, and the policy's own heading estimate
  survives (Q2, mechanisms 1 and 2).

  `chase_q ≈ 0` in both: neither policy chases the distractor recall. The
  explore half of the disambiguation is not a problem.

- **u50, and σ is confirmed as the dominant lever.** At a matched u50:
  `w1_sig` (σ=0.30, ε=0.1) **0.0839** against `w1_eps01` (σ=0.165, ε=0.1)
  0.0468 and `w1_base` (σ=0.165, ε=0.4) 0.0398. **2.1× the baseline and 1.8×
  the ε change alone**, from one knob, exactly as the magnitude analysis
  predicts: ε is masked out of the movement surrogate, so σ is the only
  channel through which the policy learns its own step size.

  Acted on: **`w1_c20` cancelled at u130 to free a slot for `w1_sig2` (σ=0.50).**
  The ladder asks "do 20 envs match 80?" *on the ε=0.4, σ=0.165 config that
  `w1_sig` has now superseded*, and a cost saving measured on a recipe about to
  be abandoned answers nothing — the ladder gets re-run on the winning recipe.
  Its partial curve (0.0723 at u100) is kept in the table as a data point, not
  as an answer to Q1. **Q1 is therefore deferred, not answered, by wave 1.**

- **THE BINDING CONSTRAINT AT u75 IS STEP MAGNITUDE, NOT STRATEGY.** Both
  policies are at |a| ≈ 0.18–0.25 against the coverage optimum of 1.0 (§3.1).
  At |a| = m < 1 the agent needs 1/m steps to leave a cell, so `cells_per_step`
  is capped at roughly **m** no matter how good the trajectory is. Simulated
  exactly (§3.1, magnitude sweep): a perfect billiard gets `cells_per_step` of
  0.179 / 0.284 / 0.507 / 0.790 at |a| = 0.15 / 0.25 / 0.5 / 1.0. **No amount
  of shaping or schedule can beat this cap; only a larger |a| can.**

  `behavior_probe` now reports **`strategy_efficiency`** = observed
  `cells_per_step` ÷ the billiard reference *at the same magnitude*, which
  splits the two failures apart. At u75: `w1_base` 0.084 against a 0.21 ceiling
  = **40%**, `w1_eps01` 0.133 against 0.285 = **47%**. So magnitude explains
  most of the shortfall and there is a further ~2× path-quality gap on top.
  Both must close to reach the 0.352 target: at 47% efficiency and |a| = 1.0 a
  policy would land at coverage 0.185, not 0.35.

  This promotes `INIT_LOG_STD` from "exploration temperature" to *the* knob:
  ε is masked out of the movement surrogate, so σ is the only channel through
  which the policy can learn its own magnitude (P0.6). Acted on immediately —
  `w1_sig2` (σ = 0.50, ε = 0.1) queued, displacing `w1_c8`, whose cost question
  `w1_c20` largely answers. Magnitude is growing on its own (0.086 at u6 →
  0.25 at u75 for `w1_eps01`), so it may resolve by u450; the question is the
  *rate*, and σ sets it.

- **u25, and the sign is right on Q2.** `w1_eps01` reads `mean_coverage` 0.0428
  against `w1_base`'s 0.0297, while `w1_base` has the *higher* training reward
  (`mean_r` −0.0018 vs −0.0396). That is P0.6 exactly: ε moves the agent, which
  inflates the novelty the rollout collects, but eval scores the **deterministic
  mean policy**, and ε-steps are masked out of the movement surrogate — so the
  reward ε buys never becomes a policy that moves. Far too early to conclude at
  u25 of 450; recorded because it is a prediction landing, not a result.
- **`pi_fiete` is ~1.6× slower than `mit_normal_gpu` here**, node contention not
  hardware: `w1_sig` runs 58.4 s/u on node3807's a100 against `w1_base`'s
  36.9 s/u on an l40s. **Never compare `s/u` across partitions**, and expect
  `pi_fiete` runs to reach ~u370 rather than u450 in six hours. Comparisons in
  the results table are therefore taken at a **matched update index**, not at
  each run's end.
- **The ladder is ~2× cheaper per update, not 4×.** `w1_c20` runs 19.2 s/u
  against 38.3 for 80 envs — but on the *contended* node, so ~12 s/u
  like-for-like, i.e. ≈3×. The serial-call count fell 4×, so the shortfall is
  the PPO step, which does not scale with `envs_per_world` (the pool is held
  constant by construction). **The cost model in §4 undercounts PPO**: it is
  roughly 8 s of the 38 at this rollout shape, and it is a floor on how cheap
  any variant can get.

#### Conclusions — wave 1

*(pending)*

---

### Wave 2 — the exploit ceiling (designed, fires as slots free)

Independent of wave 1 — a different regime and a different metric — so it does
not wait on it. `SCHEDULE='exploit:400'`, `EVAL_SCOPE=all`, eval every 20.

**The question.** §3.3.1 says the readout supports `mean_steps` of **10.1**
(no distractors) to **12.7** (ten) at unit steps, and that `success_rate` is
1.000 for any policy with even a weak goalward drift. So the only question
worth asking is: **how far above the reference row does a trained policy sit,
and why?** The v35 lineage sat at 22.9, the cos ≈ 0.50 row — a 2× policy gap
against a readout measured at cos 0.99.

**Shaping is zeroed, not inherited.** `wall_penalty`, `persistence_bonus` and
`revisit_penalty` leak into exploit rollouts (P0.3.1), so a baseline that left
them on would measure the leak rather than goal-following.

| variant | change | why |
|---|---|---|
| `w2_x_base` | shaping all 0 | the clean control: goal reward only |
| `w2_x_pers` | `PERSISTENCE_BONUS=0.05` | the one leaked term whose sign is plausibly *positive* for nav — a beeline is a straight line |
| `w2_x_sig` | `INIT_LOG_STD` −1.8 → −1.2 | ε is hardcoded 0 in the exploit regime (`exploit.py:93`), so σ is the **only** channel through which the policy can learn its step magnitude, and P0.6 says it starts 12× too small |
| `w2_x_lr` | `LR` 3e-4 → 1e-3 | exploit is a much easier objective than explore (dense +5, near-perfect signal); if it is optimization-limited, this is the cheapest fix |

**Every result is read as the triple (`success_rate`, `mean_steps`, mean |a|)**
and placed against §3.3.1's table using the `q_accuracy` the behaviour probe
measures for that same checkpoint. Decision rule: at the reference ⇒ the readout
is the limit and exploit is done; well above ⇒ a policy gap, and wave 3 gets an
exploit-tuning arm before it gets a combination arm.

Not swept, per §3.5: `GOAL_REWARD` and `TIME_PENALTY`, which with shaping at
zero reach the policy only through the un-normalized value loss.

---

### Wave 3 — combining the two (pre-registered; fires after waves 1–2)

**Written before any wave-1 or wave-2 result is in**, so the decision rules
cannot be fitted to the outcome. Everything here runs with
`--regime_assignment shuffle` (see §0) and the shaping and noise settings that
waves 1–2 select.

The four orderings Jack named, as schedules:

| arm | schedule | warm start |
|---|---|---|
| **A** *(the requested order)* | `interleave:450,empty_frac=1.0->0.5,anneal=150` | best wave-1 explore checkpoint |
| **B** *(interleave throughout)* | `interleave:450,empty_frac=0.5` | fresh |
| **C** *(blocked, never interleaved)* | `explore:225 ; exploit:225` | fresh |
| **D** *(exploit first)* | `interleave:450,empty_frac=0.0->0.5,anneal=150` | best wave-2 exploit checkpoint |

A and D warm-start via `--load_checkpoint`, which is what makes "anneal the
other regime in" cheap: the first phase has already been paid for by waves 1–2,
so each arm costs one 450-update run rather than two.

**The prediction, and why.** P0.8 says the regime is decidable at AUC ≈0.9 from
`|q|` alone at ≤3 distractors, but only ≈0.72–0.79 at ten unless the agent
*probes* — takes a few steps along `q` and watches whether it converges. Under
`explore_goals_off`, probing earns nothing and costs coverage, so a long
pure-explore phase actively trains it away.

- **A** therefore starts from a policy that has learned to ignore the recall
  channel and must unlearn it. Predicted: works at low distractor counts, and
  the d=10 nav numbers lag.
- **B** never trains the wrong prior, and an interleaved rollout is the only
  setting where probing has positive expected value — half the envs pay off.
  **Predicted best on the composite.**
- **C** is the control that isolates "interleaving matters" from "seeing both
  regimes matters": it sees both, never together.
- **D** brackets A from the other side.

**Decision rule, stated now.** Score all four on the §5 verdict protocol and
compare the triple (coverage, `mean_steps` at d=0, `mean_steps` at d=10) against
the reference lines. If B ≥ A on the composite, **the requested ordering is
refuted and interleaving becomes the recipe**; report that plainly rather than
continuing to tune A. If A ≥ B, the ordering stands and the P0.8 concern was
priced too high. If C matches both, then interleaving *per se* is irrelevant and
only regime exposure matters — the cheapest outcome, and the one that would let
waves 1 and 2 simply be concatenated.

**The failure mode to watch for is interference, not either metric alone.** A
model that scores 0.35 coverage and `mean_steps` 30, or 10.5 steps and 0.10
coverage, has not combined anything. `analysis/nav_tri/behavior_probe.py`
diagnoses which: `chase_q` in explore mode should stay near 0 (the policy
ignores phantom recalls) while `follow_q` in nav mode should be high (it
follows real ones). A model that fails by chasing distractors while exploring
shows high `chase_q`; one that fails by ignoring the goal shows low `follow_q`
with high `q_accuracy`.
