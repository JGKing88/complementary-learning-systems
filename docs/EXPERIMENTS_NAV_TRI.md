# One model, three metrics: coverage + success rate + steps-to-goal

Working document for the `nav-tri-metric` branch. Started 2026-08-14.

**Read `## 0. Where I am` first** — it is the resume point, rewritten every
wave, and it is the only section that goes stale on purpose.

---

## 0. Where I am

| | |
|---|---|
| **Branch / worktree** | `nav-tri-metric` at `.claude/worktrees/nav-tri-metric` (pushed) |
| **Runs live** | `w2_e_long` (σ=0.30) · `w2_e_long2` (σ=0.50) · `w2_e_long3` (σ=1.0, the bracket) · `w2_x_sig2` (first exploit) — all 20 envs × 64 batch |
| **Wave state** | Wave 1 **complete**, all four questions answered (see its Conclusions). Wave 2 explore + exploit both running. Wave 3 designed and pre-registered, not launched. |
| **Best so far** | coverage **0.280** (`w2_e_long`, u250) against a 0.352 target and the v35 recipe's 0.121 at u175 |

**The recipe that currently wins**, for anyone picking this up cold:

```
20 envs × 64 batch × 200 steps   (pool 1280; 12.6 s/update)
EPSILON_EXPLORE=0.1              (not v35's 0.4)
INIT_LOG_STD=-0.7  (σ≈0.50)      (not v35's -1.8; this is the big one)
FREEZE_LOG_STD=1
everything else as v35 / §1.2
```
| **wandb project** | `train_navigate` |
| **Launcher** | `hopfield_nav/run_nav_tri.sh` (`VARIANT=<name> sbatch …`) |
| **Status** | `bash hopfield_nav/nav_tri_status.sh` |
| **Results tables** | `python -m analysis.nav_tri.collect_results --prefix w1` |
| **Env python** | `$HOME/.conda/envs/cls/bin/python` (login node has no `python`) |

Resume checklist, in order:

1. `bash hopfield_nav/nav_tri_status.sh` — what is queued, running, finished.
2. `python -m analysis.nav_tri.collect_results --prefix w1` — the curves, with
   the reference lines printed underneath.
3. Read §6 for the wave in flight: its hypothesis, its variant table, and the
   rule stated *in advance* for what each outcome would mean.
4. Fill that wave's results table, then write its "Conclusions" —
   confirmed / refuted / underpowered — **before** designing anything new.
5. Design the next wave in §6 with hypothesis and decision rule written down
   *before* submitting.

### The single most important finding so far

**The explore metric is step-magnitude-limited, and `INIT_LOG_STD` is the knob
that moves it.** The policy starts at mean |a| = 0.086 against an optimum of
1.0, and `cells_per_step` is capped near |a| however good the trajectory is. ε
cannot fix it — ε steps are masked out of the PPO movement surrogate, so they
move the agent without teaching it to move. σ is the only channel. Measured
1.8× coverage from σ alone at matched u50, 2.1× against the full v35 config.
See §3.4's verdict and the wave-1 live notes.

**Open items carried forward** (priority order):

- [ ] Fill the wave-1 results table at u450 and write its Conclusions.
- [ ] Behaviour-probe the wave-1 finals
      (`PROBE=behavior CKPTS="…" sbatch hopfield_nav/run_nav_tri_probe_cpu.sh`)
      and record `step_mag_mean` + `strategy_efficiency` beside every coverage
      number — a coverage figure without them cannot be diagnosed.
- [ ] **Smoke the exploit config before committing 6 h to it.** Nothing on the
      exploit side has been run yet — every exploit number in this document is
      simulation (§3.3.1) or readout probe (P0.7). One
      `SCHEDULE='exploit:6' EVAL_SCOPE=navexpl` run first, to confirm the nav
      metrics actually appear and `navexpl` behaves, then launch the wave.
- [ ] Launch wave 2 (exploit) at the σ wave 1 selects; variants already in the
      launcher.
- [ ] **Caveat on the σ result: `w1_sig` ran on `pi_fiete` (a100) while
      `w1_eps01` ran on `mit_normal_gpu` (l40s)**, so that comparison is
      confounded with node — different float rounding acts like a different
      seed. The 1.8× effect is far larger than that could plausibly produce,
      and the u50 behaviour probe checks the *mechanism* (does σ actually
      enlarge `step_mag_mean`?) which is node-independent. `w2_e_long` vs
      `w2_e_long2` will give a clean same-node, same-shape σ comparison.
- [x] **Q1 (cost/diversity) — answered.** 20 envs ≈ 80 envs *per update*
      (within ±15%, sign flips between the two comparisons) and 2.9× cheaper
      per update, so it wins outright per GPU-hour. All long runs use 20 × 64.
      `w1_sig2` stays at 80 envs as the over-fitting control.
- [ ] Cross-check `behavior_probe` against `hopfield_nav.eval_all`
      (`EVAL_ALL=1 … run_nav_tri_verdict.sh`) before any probe number is quoted
      as a verdict — the probe reimplements the eval protocols.
- [x] `signal_separability` at the real Npos=1716 — done; **overturned twice**,
      see the two CORRECTIONs in P0.7. Read at ≥8 distractor draws only.
- [x] `temporal_separability` at the real Npos — done, see P0.8.
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
learnable σ from collapsing.

> **Verified.** The smoke run printed `move_entropy = -0.762` at
> `init_log_std = -1.8`, which is exactly `2(½log 2πe + log σ)` — entropy is a
> function of σ alone, so under `FREEZE_LOG_STD=1` it is a constant with zero
> gradient. **Every run in the v35 lineage that set both was sweeping a dead
> knob.** All runs here keep `MOVE_ENT_COEF` at v35's 0.005 and simply do not
> treat it as an axis.

#### Verdict on these three, after wave 1

`INIT_LOG_STD` turned out to be **the single most important knob in the whole
explore problem**, and for a reason that is not "exploration temperature":

- The policy starts at mean |a| = 0.086 and needs ~1.0 (P0.6), and
  `cells_per_step` is capped near |a| whatever the trajectory looks like. So
  the explore metric is *magnitude-limited* long before it is strategy-limited.
- ε cannot fix that, because ε steps are masked out of the movement surrogate —
  they move the agent without teaching it to move. **σ is the only channel
  through which the policy learns its own step size.**
- Measured: at a matched u50, σ=0.30 gives coverage 0.0839 against σ=0.165's
  0.0468 at the same ε — **1.8× from one knob**, and 2.1× against the full v35
  configuration.
- And it is nearly free on the other side of the ledger: σ=0.50 costs 5% of
  `mean_steps` (§3.4.1), so one σ serves both regimes.

`FREEZE_LOG_STD` stays at 1 throughout. Learnable σ collapses once returns go
positive, which would end the magnitude ascent exactly when it is paying off;
and with the freeze genuinely working only since 2026-08-07, a pinned σ is also
the configuration whose behaviour is now predictable. The `w1_siganneal` variant
is the principled middle: a *scheduled* σ, high while the magnitude climbs and
low once the straightness term becomes readable (§3.4.1).

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

**And a large σ is nearly free for the exploit metric, so there is no
trade-off to manage.** The obvious worry is that σ good for explore is too
blunt for navigation — a jittery path is a longer path, and `mean_steps` is the
exploit metric. Priced with `exploit_reference.py --sigma 0.5`: at cos 0.99 and
|a| = 1, `mean_steps` is **10.61 at σ=0.50 against 10.11 at σ=0.165** — a 5%
cost, against σ's ~2× effect on explore learning speed. `success_rate` stays
1.000 throughout.

> **SUPERSEDED — see the wave-2 exploit notes in §6.** This simulates a
> *perfect* follower, which walks straight in and stops. A learning policy has
> a **terminal-approach** problem the simulation cannot express: as it nears
> the goal `|q| → 0`, the recall direction becomes noise-dominated, and σ=0.50
> is ~0.5 cells of jitter against a `goal_radius` of 1.0 — so it overshoots and
> orbits. Measured on the first exploit run, `success_rate` fell 0.969 → 0.510
> between u25 and u50 as |a| grew. **The 5% figure below is a lower bound that
> does not bind; do not use it to justify one shared σ.** The σ *anneal* is the
> resolution, and is a requirement rather than a refinement.

**The one real cost of a large σ: it destroys the `persistence_bonus` signal
while the step is still small.** That term is `cos(a_t, a_{t−1})` on the
*executed* actions (`collector.py:620-638`). With mean magnitude `m` and
per-component noise σ, two consecutive executions of the *same* intended
direction have expected cosine ≈ `m² / (m² + σ²)`:

| | σ=0.165 | σ=0.30 | σ=0.50 |
|---|---|---|---|
| at m = 0.25 (where the runs are now) | 0.70 | 0.41 | **0.20** |
| at m = 1.0 (the target) | 0.97 | 0.92 | 0.80 |

So at σ=0.50 and today's step size the straightness reward is ~80% noise, and
it recovers to ~80% signal only once the magnitude reaches 1. Two consequences:

- **Shaping straightness and raising σ fight each other early and stop fighting
  late.** That is a strong argument for the `w1_siganneal` variant — buy the
  magnitude ascent with a large σ, then decay it so the straightness term
  becomes readable — and it predicts that `PERSISTENCE_BONUS` is worth
  revisiting *after* magnitude is solved, not before. (Wave 1's `w1_pers` was
  therefore mistimed, which is a better reason to have deprioritized it than
  the one originally given.)
- It also means the ~45% `strategy_efficiency` measured at u75 is not
  necessarily a hard limit: the deterministic policy's own straightness is only
  0.47–0.57, and the term meant to fix that is currently being drowned.

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

> **`GOAL_REWARD` is emphatically NOT inert in an interleaved run**, and the
> table above nearly hid that. Both regimes' rollouts go into **one** pooled
> normalization, so `goal_reward` sets the *ratio* between an explore rollout's
> smooth ~0.23/step novelty and an exploit rollout's +5.0 spikes — i.e. which
> regime's gradient survives normalization. It is inert *within* a regime and
> decisive *between* them. See the wave-3 collapse.
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

**Solved, from two measurements at different `envs_per_world` on an l40s** under
this project's settings (`observation_size=60`, `wall_resolution=4`, RNN/ReLU,
`hidden_size=1024`, pool held at 1280):

| shape | serial calls/update | measured s/update |
|---|---|---|
| 80 envs × 16 batch | 16,000 | **37.0** |
| 20 envs × 64 batch | 4,000 | **13.8** |

Two points, two unknowns:

```
s_per_update  =  1.93 ms x (envs_per_world x steps_per_rollout)  +  6.1 s
                 \___ rollout collection ___/                       \_ PPO _/
```

The 1.93 ms/call reproduces the explore-min wave's independent estimate. **The
6.1 s PPO term does not shrink with `envs_per_world`** — the pool is held
constant by construction — so it is a hard floor: even at one env an update
costs ~6.5 s, and the most the ladder can ever buy is **5.7×**, of which 20
envs already captures 2.7×.

Sizing follows directly. A 6-hour budget is ≈**450 updates at 80 envs** and
≈**1,560 at 20**.

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
- **Randomizing which envs get which regime.** *(Done — see §0.)*
- **`--rnn_cell gru` instead of `rnn`/`relu`.** Flagged because the measured
  strategy gap is specifically a *state-holding* failure and a vanilla Elman
  cell is the architecture least equipped for it. To travel in a straight line
  with no `prev_action` channel, the policy must re-emit the same world-frame
  vector for several consecutive steps, i.e. hold a heading in the hidden state
  while the sensory cone underneath it changes every step. An ungated ReLU
  recurrence has nothing protecting that state from the input; a GRU's update
  gate is exactly the mechanism for it. Measured: `run_len_mean` 1.6–2.1 steps
  against the ~4 that is optimal, across every variant and unchanged by σ.
  **Not acted on — `rnn`/`relu` is an explicit instruction (§1.2)**, and the
  within-instruction remedies (σ anneal so `persistence_bonus` becomes
  readable; raising `PERSISTENCE_BONUS` once the magnitude is solved) are
  untried. Recorded so that if run length stays near 2 after both, the
  architecture is the honest explanation and not a tuning failure.

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
  **Q1 is therefore deferred, not answered, by wave 1** — but the partial curve
  is encouraging rather than neutral: at a matched u100, `w1_c20` (20 envs)
  reads **0.0723 against `w1_base`'s (80 envs) 0.0574**, the two differing in
  nothing but `envs_per_world` × `batch_envs` at a held pool. So 20 envs was
  **26% ahead per update while costing half the wall-clock per update**, i.e.
  the question to re-ask on the winning recipe is not "does it match?" but "how
  much better is it?". One seed, one comparison point — but the sign is the
  useful one, and it makes the re-run a priority rather than a formality.

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

- **u50 σ-ladder probe — the node confound is resolved, and σ acts *purely*
  through magnitude.** All three checkpoints probed in one process, matched
  update, explore mode, `n_dist=0`:

  | | `w1_base` ε.4/σ.165 | `w1_eps01` ε.1/σ.165 | `w1_sig` ε.1/**σ.30** |
  |---|---|---|---|
  | **`step_mag_mean`** | 0.143 | 0.175 | **0.344** |
  | **`strategy_efficiency`** | 0.447 | 0.418 | 0.486 |
  | `straightness` | 0.442 | 0.460 | 0.575 |
  | `abs_turn_mean` (rad) | 0.994 | 0.962 | 0.794 |
  | `run_len_mean` (steps) | 1.59 | 1.69 | 2.07 |
  | `cells_per_step` | 0.079 | 0.089 | **0.181** |

  1. **Confound resolved.** `w1_sig` ran on a different node than the other
     two, so its score alone could not separate σ from a seed-like rounding
     difference. It doubles the *learned step magnitude*, 0.175 → 0.344, and no
     amount of float rounding does that. The mechanism is σ.
  2. **σ is a pure magnitude effect.** `strategy_efficiency` — coverage divided
     by what a perfect billiard gets *at that same magnitude* — is **0.42–0.49
     for all three**, essentially flat. So the entire 2× coverage gain is the
     magnitude gain and none of it is better pathing. That is a cleaner
     decomposition than expected and it says the two problems are separable:
     σ fixes one and does nothing to the other.
  3. **The remaining gap has a name: the policy turns far too often.**
     `run_len_mean` is 1.6–2.1 steps against the ~4 that §3.1's run-and-tumble
     sweep puts at the coverage optimum, and `abs_turn_mean` of 0.79–0.99 rad
     is much nearer a uniform random walk's π/2 than a billiard's 0. Closing
     it is worth ~2× on top of whatever magnitude delivers — and the term that
     rewards it, `persistence_bonus`, is precisely the one §3.4.1 shows is
     drowned while σ is large and the step is small. **This is the case for the
     σ anneal, now made from measurement rather than from algebra.**

- **u100–u110, and the wave is cut short deliberately.** Standings at a matched
  u100: `w1_base` 0.0574, `w1_c20` 0.0723, `w1_eps01` 0.0807, and `w1_sig`
  already 0.0839 **at u50**. The ordering is settled and the mechanism is
  understood, while the curves are still climbing roughly linearly — so the
  runs are **update-limited, not recipe-limited**, and their remaining hours
  buy a more precise ranking of a recipe that is already being superseded.

  So: **`w1_eps01` cancelled at u110** and the freed (fast) slot given to
  `w2_e_long` — the same noise regime at 20 envs × 64 batch, 1500 updates
  instead of 450. What is kept and why:

  | run | kept? | why |
  |---|---|---|
  | `w1_base` | **yes, to the end** | the v35 control. Without an endpoint there is no "we beat the baseline by X at matched updates". |
  | `w1_sig` (σ=0.30) | yes | carries the σ axis to u450 at 80 envs |
  | `w1_sig2` (σ=0.50) | yes | ditto, and brackets σ from above |
  | `w1_eps01` (σ=0.165) | **no** | the ε question is answered by score *and* by the u75 behaviour probe; its σ=0.165 rung is duplicated by `w1_base` at the same σ |

  A side benefit worth naming: `w1_sig`/`w1_sig2` at 80 envs against
  `w2_e_long`/`w2_e_long2` at 20 envs, **at matched σ**, re-answers the
  deferred Q1 (cost vs diversity) as a by-product rather than as a separate
  wave.

- **Q1 (cost vs diversity), answered as a by-product.** Two comparisons at a
  held PPO pool of 1280, differing only in `envs_per_world` × `batch_envs`:

  | comparison | 80 envs | 20 envs | at |
  |---|---|---|---|
  | σ=0.165, ε=0.4 | `w1_base` 0.0574 | `w1_c20` **0.0723** | u100 |
  | σ=0.30, ε=0.1 | `w1_sig` **0.0839** | `w2_e_long` 0.0729 | u50 |

  The sign flips between them, so the honest reading is **20 envs ≈ 80 envs per
  update, within ±15%** — and per *GPU-hour* 20 envs wins outright, because the
  cost model puts it at 2.9× more updates for the same wall-clock. **Every
  long run from here uses 20 × 64.** The residual risk is over-fitting to 20
  codebooks over 1500 updates, which held-out eval would show as a plateau or
  decline while an 80-env run kept climbing — `w1_sig2` is kept running at 80
  envs precisely as that control.

- **The wave is rebalanced again, on the same principle.** The long 20-env runs
  are what produce a *model*; the 80-env runs are controls, and two suffice.
  `w1_sig` cancelled at u70 (its σ=0.30 is carried further and cheaper by
  `w2_e_long`); the slot goes to `w2_e_long2`, σ=0.50 at 20 envs. Surviving
  80-env runs: `w1_base` (v35 control) and `w1_sig2` (diversity control).

#### Conclusions — wave 1

All four questions answered. The wave was **cut short on purpose** once each
was settled — three runs were cancelled mid-flight and their slots reassigned,
which is recorded above with the reasoning for each. Numbers below are
in-training evals at matched update indices, i.e. **monitoring-grade**; none
has been through the §5 verdict protocol yet.

**Q1 — diversity or gradient batch? → Neither is binding; take the cheap
shape.** 20 envs × 64 batch matches 80 × 16 per update to within ±15% (the
sign flips between the two comparisons available) at 2.9× less wall-clock per
update. **Adopted for every long run.** `w1_sig2` continues at 80 envs as the
over-fitting control.

**Q2 — is ε=0.4 worth its price? → No, decisively.** ε=0.1 beat ε=0.4 at every
matched update (0.081 vs 0.057 at u100), and the u75 behaviour probe showed
*why*: a 41% larger step, a straighter path, less wall-clipping. Three
independent mechanisms were identified in advance and all point the same way —
ε discards 40% of the movement gradient, corrupts the policy's estimate of its
own heading (there is no `prev_action` channel, so the RNN only knows its
heading because it chose it), and reduces `persistence_bonus` to mostly noise.
A fourth appeared unbidden: at ε=0.4 the *training reward is flat while eval
coverage climbs*, so ε also destroys `mean_r` as a progress signal.

**Q3 — can σ buy the same exploration more cheaply? → σ is not a substitute
for ε, it is a bigger lever than ε.** σ=0.30 gave 2.1× the baseline's coverage
at matched u50 from one knob; σ=0.50 leads again at u25. The probe shows it
acts **purely through step magnitude** (0.175 → 0.344) with
`strategy_efficiency` flat at 0.42–0.49 — so magnitude and path quality are
*separable* problems and σ solves exactly one of them. Priced on the other
side: σ=0.50 costs 5% of `mean_steps`, so one σ can serve both regimes.

**Q4 — does straightness need paying for directly? → Untested, and
deliberately so.** `w1_pers` was cancelled before it ran. The reason improved
during the wave: not merely "shaping showed nothing before", but that
`persistence_bonus` carries `m²/(m²+σ²)` signal, which at the wave's actual
step sizes is ~20–40% — it would have been measured in the one regime where it
cannot work. It is now the *primary* candidate for the next wave, because Q3
localized the remaining gap to exactly what it rewards.

**The finding that outranks all four**, and which the wave was not designed to
look for: **the explore metric is step-magnitude-limited before it is
strategy-limited.** `cells_per_step` is capped near |a|, the policy starts at
|a| = 0.086 against an optimum of 1.0, and no shaping or schedule can lift
that cap. Everything Q2 and Q3 measured is downstream of it.

**Where it leaves the target.** Best in-training coverage so far is **0.124 at
u75** (`w1_sig`), against `w1_base`'s 0.043 at the same update — **2.9× the
v35 recipe** — with a practical target of 0.352 and 1400 updates still to run
on the long arms. The two gaps that remain are quantified and separable:
magnitude (|a| 0.34 → 1.0) and turn rate (run length 2.1 → ~4).

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

**Rollout shape is 20 envs × 64 batch, not wave 1's 80 × 16** — same PPO pool
and env-steps, a quarter of the serial calls, half the wall-clock per update.
Taken on evidence: `w1_c20` ran exactly that shape and was 26% *ahead* of
`w1_base` at a matched u100. Diversity also matters less here — "follow the
recall signal" is an env-independent skill, where coverage is not.

**σ is the axis**, because wave 1 found it dominant for explore *and* the
exploit regime hardcodes ε=0 (`exploit.py:93`), so σ is the **only**
exploration a nav policy has, not merely the best one.

| variant | change | why |
|---|---|---|
| `w2_x_base` | σ=0.30, shaping all 0 | the control: goal reward only, at the σ wave 1 favours |
| `w2_x_sig2` | σ=0.50 | brackets σ from above |
| `w2_x_siglo` | σ=0.165 | v35's value — brackets from below, and says what the historical recipe scores on `mean_steps` under this protocol |
| `w2_x_lr` | `LR` 3e-4 → 1e-3 | exploit is far easier than explore (dense +5, readout at cos 0.99 up to 3 distractors); if it is optimization- rather than signal-limited, this is the cheapest fix. Promoted over a clip sweep by §3.4.1 |

`PERSISTENCE_BONUS` was going to take a ticket here — a beeline is a straight
line, so its sign is plausibly positive for nav. **Dropped**: §3.4.1 shows the
term is `m²/(m²+σ²)` signal, i.e. ~40% noise at σ=0.30 while the step is still
small, so it would be measured in the regime where it cannot work. It belongs
in a later wave, after magnitude is solved.

**Every result is read as the triple (`success_rate`, `mean_steps`, mean |a|)**
and placed against §3.3.1's table using the `q_accuracy` the behaviour probe
measures for that same checkpoint. Decision rule: at the reference ⇒ the readout
is the limit and exploit is done; well above ⇒ a policy gap, and wave 3 gets an
exploit-tuning arm before it gets a combination arm.

Not swept, per §3.5: `GOAL_REWARD` and `TIME_PENALTY`, which with shaping at
zero reach the policy only through the un-normalized value loss.

---

#### Live notes — wave 2, explore arm

- **The recipe works, and the σ trend is monotone.** At matched updates,
  20 envs × 64 batch, ε=0.1:

  | run | σ | u50 | u100 |
  |---|---|---|---|
  | `w1_base` *(v35 recipe, 80 envs)* | 0.165 | 0.0398 | 0.0574 |
  | `w2_e_long` | 0.30 | 0.0729 | **0.1660** |
  | `w2_e_long2` | 0.50 | **0.1364** | — |

  `w2_e_long` at u100 is **2.9× the v35 recipe at the same update**, and
  `w2_e_long2` is **1.9× `w2_e_long`** at u50 — the latter being the *clean*
  σ comparison (same rollout shape, same partition), which retires the node
  confound noted in §0 rather than merely arguing around it.

- **Bracketing σ from above, because monotone-so-far is not monotone.** 0.165 <
  0.30 < 0.50 at every point measured; the next question is where it turns.
  `w2_e_long3` runs **σ=1.0**, which samples |a| out to ~3 around a mean near
  0.3 and is plausibly past the point where the advantage signal is swamped.

  It is **explore-only, and that is what makes a fixed σ=1.0 a fair test**:
  §3.3.1 prices σ=1.0 at 12.74 `mean_steps` against σ=0.50's 10.49 and
  σ=0.165's 10.00 — a **27% exploit cost, against 5% for σ=0.50** — so σ=1.0
  could never be carried into a combined model un-annealed. What this run can
  settle is only whether the *explore* metric still wants more, and that is
  worth knowing before choosing the anneal's starting point.

- `w1_sig2` cancelled at u40 to free the slot. Its job was the over-fitting
  control for 20-env runs — insurance that only pays out if those runs plateau,
  and can be bought then. Bracketing σ is on the critical path now.

- **σ=0.50 confirmed at a second matched point, and the v35 control retired.**

  | update | `w1_base` σ=0.165 (80 envs) | `w2_e_long` σ=0.30 | `w2_e_long2` σ=0.50 |
  |---|---|---|---|
  | u50 | 0.0398 | 0.0729 | **0.1364** |
  | u100 | 0.0574 | 0.1660 | **0.2383** |
  | u150 | 0.0949 | 0.2103 | — |
  | u175 | 0.1206 | — | — |
  | u200 | — | 0.2441 | — |

  σ=0.50 reaches at **u100** what σ=0.30 needs **u200** for, and what the v35
  recipe does not reach at all in the 190 updates it ran. **`w1_base`
  cancelled at u190**, its curve recorded above as the v35 reference: the
  headline comparison is *already* won by 4× at matched updates, and refining
  it to u450 would buy precision on a question that is settled while the
  **exploit half of Jack's goal remains entirely untested**. The freed slot
  goes to `w2_x_sig2`, the first exploit run.

  The claim this leaves is the stronger form anyway: **our recipe at u100
  (0.238) beats the v35 recipe's best in 190 updates (0.121) by 2×, in half
  the updates and at a third of the wall-clock per update.**

- **σ bracketed from above: σ=1.0 is past the peak.** At matched u50 —
  σ=0.165 → 0.0398, σ=0.30 → 0.0729, σ=0.50 → **0.1364**, σ=1.0 → **0.0503**.
  A clean inverted-U with **σ=0.50 at the maximum**. Worth the run: the trend
  was monotone at every point previously measured, and extrapolating it would
  have cost 2.7×. (One point; σ=0.30 also trailed at u25 before overtaking, so
  confirmation at u100 before acting.)

- **u200 behaviour probe on `w2_e_long` — the magnitude problem is nearly
  solved, and the distractor problem is solved outright.**

  | metric | u50 (σ=0.30) | **u200 (σ=0.30)** | optimum |
  |---|---|---|---|
  | `step_mag_mean` | 0.344 | **0.796** | 1.0 |
  | `strategy_efficiency` | 0.486 | **0.735** | 1.0 |
  | `cells_per_step` | 0.181 | 0.494 | 0.79 |
  | `straightness` | 0.575 | 0.623 | 1.0 |
  | `run_len_mean` | 2.07 | 2.09 | — |
  | `revisit_frac` | — | 0.506 | — |
  | `edge_frac` | — | 0.167 | 0.19 = uniform |
  | `chase_q` (d=10) | — | −0.016 | 0 |

  1. **|a| has gone 0.086 → 0.796.** The magnitude ceiling that dominated wave 1
     is nearly gone, which retires the concern that no shaping could ever bite.
  2. **Strategy improved too, 0.49 → 0.735** — and *not* by lengthening runs:
     `run_len_mean` is unchanged at ~2.1. The agent is beating a run-and-tumble
     of its own turn rate (which would give ~0.44 cells/step at |a| = 0.8
     against its 0.494), so its turns are **structured, not random** — it has
     learned something billiard-like, turning where turning pays.
  3. **`edge_frac` 0.167 is *below* the 0.19 a uniform occupancy gives**, and
     `clip_frac` is 0.086. **The perimeter-orbit basin and the corner trap —
     the two failure modes this lineage has repeatedly fallen into — are both
     absent.** `wall_penalty` at v35's 0.1 is doing its job and needs no change.
  4. **The explore-side distractor problem is solved.** `chase_q ≈ 0` and
     coverage at ten distractors (0.2477) is *identical* to coverage at zero
     (0.2468). The policy simply ignores a recall that is not its goal, which
     is exactly the half of Jack's disambiguation that explore is responsible
     for.

  **Where that leaves the target.** Multiplying the two remaining gaps —
  |a| 0.796 → 1.0 (billiard reference 0.672 → 0.79) and efficiency
  0.735 → 0.9 — predicts coverage **0.356**, i.e. the 0.352 target, and there
  are 1300 updates left on this arm alone. σ=0.50 is running ahead of it.

#### CONCLUSION — wave 2, explore arm. Target exceeded.

`mean_coverage` at `n_dist=0`:

| update | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 1000 | 1100 |
|---|---|---|---|---|---|---|---|---|---|
| σ=0.30 | 0.166 | 0.244 | 0.278 | 0.306 | 0.334 | 0.350 | 0.321 | **0.374** | 0.374 |
| σ=0.50 | 0.238 | 0.298 | 0.344 | 0.323 | 0.354 | 0.356 | **0.375** | — | — |

**Both pass the 0.352 practical target**, and σ=0.50 does it in ~700 updates
against σ=0.30's ~1000 — consistent with the σ ranking throughout.

**Final numbers** (`w2_e_long` ran its full 1500; `w2_e_long2` hit the
wall-clock limit at u1150, which the checkpoint cadence makes a normal ending
rather than a loss):

| run | σ | cov d=0 | cov d=10 | cells/step |
|---|---|---|---|---|
| `w2_e_long` | 0.30 | 0.3738 | 0.3751 | 0.748 |
| **`w2_e_long2`** | **0.50** | **0.3845** | **0.3842** | **0.769** |

Two things to read off:

- **0.769 cells/step is 97% of the perfect-information billiard's 0.790.**
  Against the §3.1 ladder the policy is at the billiard ceiling, not
  approaching it. What lies above — the lawnmower at 0.478 coverage — P0.9
  showed is unreachable without decodable position, so this is close to the
  practical maximum for this observation set.
- **Coverage at ten distractors EQUALS coverage at zero** (0.3842 vs 0.3845;
  0.3751 vs 0.3738, i.e. d=10 marginally *higher* in one). The explore-side
  distractor problem is not merely reduced, it is gone.

For scale: the v35 recipe reached **0.121 in the 190 updates it ran**, and the
explore-min wave's best was ~0.51 *at 400 eval steps*, i.e. ~0.28 at this
protocol's 200. **0.375 is the best coverage this lineage has produced.**

#### What the best explorer actually DOES

Jack asked directly: random walk, zipper, circles, wall-hugging? **None of
them.** `behavior_probe` on `w2_e_long2` u1150, explore mode:

| metric | d=0 | d=10 | reads as |
|---|---|---|---|
| `mean_coverage` | 0.3809 | 0.3811 | — |
| `cells_per_step` | 0.762 | 0.762 | — |
| **`strategy_efficiency`** | **1.061** | 1.057 | *exceeds* the billiard reference |
| **`step_mag_mean`** | **1.951** | 1.957 | ~2-cell steps, **not** the 1.0 predicted |
| `straightness` | 0.848 | 0.851 | near-ballistic (0.62 at u200) |
| `abs_turn_mean` | 0.427 rad | 0.422 | ≈24° per step |
| `run_len_mean` | 3.59 | 3.67 | ≈ the ~4-step optimum of §3.1 |
| `revisit_frac` | 0.238 | 0.238 | was 0.506 at u200 |
| `edge_frac` | 0.157 | 0.158 | **below** uniform's 0.19 |
| `clip_frac` | 0.064 | 0.064 | rarely drives into a wall |
| `chase_q` | 0.000 | 0.020 | ignores the recall channel |

**The behaviour class is ballistic sweeping with structured turns**: long
straight runs of ~3.6 steps at ~2 cells per step, turning *before* the wall
rather than clipping into it, never orbiting the perimeter, and completely
ignoring a recall signal that is not its goal. Every failure mode this lineage
has recorded — perimeter orbit, corner trap, distractor chasing, random walk —
is absent.

Two things worth pulling out:

- **`strategy_efficiency` 1.06 > 1**, i.e. the policy beats the specular
  billiard *at its own step magnitude*. That is not impossible and is not a
  bug in the metric: a specular billiard in a square box falls into periodic
  orbits for many launch angles and re-treads them, which the reference
  inherits and the policy avoids. The reference is a good bar, not a ceiling.
- **It chose |a| ≈ 1.95, where §3.1's magnitude sweep says 1.0 is optimal**
  (0.790 cells/step against 0.740 at |a| = 2). So the policy is at a *slightly
  wrong* magnitude and more than compensates with better pathing. **This
  invalidates reading `mean_steps` against the |a| = 1 row of §3.3.1** — at
  |a| ≈ 2 the oracle is **4.75 steps, not 10.0** — so every nav number must be
  read against the magnitude the policy actually has. The probe reports it for
  exactly this reason.

So both halves are solved separately:

| metric | best | target | status |
|---|---|---|---|
| coverage | **0.375** | 0.352 | **exceeded** |
| `success_rate` | **1.000** | ≥0.98 | met |
| `mean_steps` | **12.1** | ~10.1 | 1.2× the readout limit |

Wave 3 — whether *one* model can hold all three — is now the whole question.

#### A train/eval mismatch the exploit arm exposed

Recorded separately because it is a property of the **trainer**, not of this
wave, and it affects every exploit and interleaved run.

`w2_x_sig2`'s *training* reward rises monotonically (`mean_r` −0.044 at u1 →
+0.043 at u80) while its *eval* nav degrades (`mean_steps_all` 43 at u25 → 113
at u75). Training and evaluation are not measuring the same task:

| | training rollout | nav eval |
|---|---|---|
| on reaching the goal | **teleport and continue** | **episode ends** |
| RNN state after a goal | **carried** (`reset_state_on_teleport=False`) | n/a |
| goals per episode | ~4 in 200 steps | exactly 1 |
| state at the measured reach | warm for ~3 of 4 | **always cold** |

So training optimizes "reach many goals in a long rollout, mostly from a warm
hidden state", and eval scores "reach one goal from a cold start". Only about
one reach in four is trained under the condition that is scored, and a policy
that exploits the warm state is rewarded for it.

**This is the first concrete reason to consider
`--reset_state_on_teleport true`**, which Jack's instructions permit only with
"a very very good reason". It is not yet good enough: the divergence has a
competing explanation — the policy overshooting the goal radius as |a| grows,
which is the σ story — and those imply different fixes. `w2_x_siglo` (σ=0.165)
and a u75 behaviour probe separate them:

- overshoot ⇒ `step_mag_mean` will have grown well past 1 by u75, and σ=0.165
  will not degrade;
- warm-state dependence ⇒ magnitude will look reasonable and σ=0.165 will
  degrade the same way.

**Overshoot is refuted.** The u75 probe reads `step_mag_mean` **0.233**, *down*
from u25's 0.324 — it shrank rather than growing past 1. Meanwhile `follow_q`
collapsed **0.702 → 0.316** while `q_accuracy` held at **0.982**: the policy is
un-learning to follow a signal that is still nearly perfect.

**But that leaves two live explanations, not one, and the second is of my own
making.**

| hypothesis | mechanism | why it fits |
|---|---|---|
| **warm-state** | training carries the RNN state across the post-goal teleport, so ~3 of 4 trained reaches are warm; eval is always cold | training reward stable while cold-start eval degrades |
| **over-fitting to 20 envs** | with fixed goals and a fixed sensory codebook, "in env *k* go to cell *(x,y)*" is memorizable — ~8,000 (observation → action) pairs across 20 envs, well within a 1024-unit RNN. The policy then does not *need* `q` | `follow_q` collapsing while training reward holds and **held-out** eval degrades |

Both predict exactly what was observed, and they are **not** distinguished by
anything measured so far. This is the over-fitting risk flagged when Q1 adopted
20 envs — and the control that would have caught it (`w1_sig2` at 80 envs) is
the run cancelled to free a slot, which was a mistake worth naming.

> ### CONCLUSION — wave 2, exploit arm. σ=0.50 wins, and there is no conflict.
>
> Both runs finished 600 updates. At the matched endpoint:
>
> | σ | d=0 success / `mean_steps` | d=10 success / steps / **all** |
> |---|---|---|
> | **0.50** | 1.000 / **12.1** | 0.906 / 18.9 / **35.9** |
> | 0.165 | 1.000 / 19.6 | 0.833 / 24.2 / **53.5** |
>
> **σ=0.50 is 1.6× better at d=0 and 1.5× better at d=10.** Against §3.3.1's
> references — 10.1 at cos 0.99 and 15.3 at cos 0.70 — **the exploit half is
> essentially solved**: 12.1 steps at `success_rate` 1.000 is 1.2× the
> readout-limited optimum, and the v35 lineage's reference point was 22.9.
>
> **So there is no explore/exploit conflict on σ.** The claim of one, two
> corrections below, was itself an artifact of reading an oscillating series at
> u100. σ=0.50 is simply right for both regimes, which is what the original
> simulation said before three intermediate readings talked me out of it.
>
> **The one durable finding here is about the metric, not the knob:** the
> exploit eval oscillates enormously during training — `mean_steps_all` swings
> between 17 and 159 for σ=0.50 and between 22 and 107 for σ=0.165, with a
> *fixed* eval seed, so it is real model movement. **No exploit conclusion is
> safe before ~500 updates**, and the two corrections below are both what
> happens when one is drawn at 100.
>
> ### CORRECTION — the section below was written at u100 and is wrong
>
> It read "σ=0.165 is 2× better on the honest metric, **and it does not
> degrade**", and concluded that exploit needs a permanently smaller σ. The
> full curves refute the strong form. `mean_steps_all`, d=0:
>
> | update | σ=0.50 | σ=0.165 |
> |---|---|---|
> | u25 | 43 | 120 |
> | u100 | 77 | **54** |
> | u175 | 86 | **29** |
> | u250 | 27 | **23** |
> | u350 | 112 | **24** |
> | u600 | **12.1** | *(running)* |
>
> **Both oscillate violently** — σ=0.165 dips to 107 at u200, σ=0.50 swings
> between 159 and 17 — and because the eval seed is fixed at 42, that is real
> model movement, not sampling noise. So "σ=0.50 degrades" was **a phase of an
> oscillation mistaken for a trend**, read off three consecutive points.
> σ=0.50 went on to reach **`mean_steps` 12.1 at `success_rate` 1.000**, the
> best exploit number of the project and near the 10.1 reference.
>
> What survives: σ=0.165 is **more stable** and better at every matched update
> through u350. What does not: that σ=0.50 fails, or that exploit needs a
> permanently small σ. **The matched-endpoint comparison is still owed** —
> σ=0.165 reaches u600 around 13:45.
>
> The lesson is the one this document keeps re-learning: **three points are not
> a trend when the underlying series oscillates**, and I had the fixed eval
> seed available to tell me the oscillation was real.

At u100 the comparison read:

| σ | `success_rate` | `mean_steps` | **`mean_steps_all`** |
|---|---|---|---|
| 0.50 (`w2_x_sig2`) | 0.667 | 62.0 | **~112** |
| **0.165 (`w2_x_siglo`)** | **0.979** | 51.1 | **54.2** |

The **reward-density** argument below still stands as an account of why the
large-σ run is *less stable* — it is the reason its behaviour policy can earn
reward while its mean wanders — but not as a reason it cannot win:

> Explore pays **every step** — novelty fires on each new cell — so a large σ
> is cheap: every noisy step still returns a graded signal, and the noise buys
> the action-space coverage that teaches magnitude.
>
> Exploit pays **only at the goal**, roughly 4 times in 200 steps. With mean
> |a| = 0.23 against σ = 0.50 the *executed* trajectory is mostly noise, so the
> reaches that earn reward are largely luck, and the gradient credits the mean
> only weakly. Cutting σ brings the behaviour back to the policy being scored.

**This is a genuine explore/exploit conflict on a single parameter, and the
first one found.** It cannot be resolved by picking one σ, which retires the
"one σ serves both" idea for good. Two consequences for wave 3:

1. **The σ anneal becomes structural, not cosmetic**: high σ while coverage is
   being learned, low σ once goal-following is what matters.
2. **It aligns with arm A.** An `empty_frac 1.0 → 0.5` schedule moves the run
   from explore-dominated to exploit-inclusive exactly as a σ anneal moves
   σ from high to low. The requested ordering and the σ schedule are the *same*
   curriculum, which is an argument for arm A that did not exist when wave 3
   was pre-registered — and it is recorded here as a revision of that
   prediction, not as a retrofit after seeing the result.

**RESOLVED — and it is neither of those two.** The `warmcold` probe, on
held-out eval envs, under the training contract:

| checkpoint | cold first reach | warm later reaches | speedup |
|---|---|---|---|
| u25 | 43.6 | 37.0 | **1.20×** |
| u75 | 60.1 | 43.2 | **1.29×** |

Warm-state dependence is **real but small**, and — decisively — **both** cold
*and* warm degraded between u25 and u75 (43.6 → 60.1 and 37.0 → 43.2). A
mismatch that only affects cold starts cannot explain a degradation that hits
warm ones too. Over-fitting is out for the same reason `w2_x_siglo` is: at
σ=0.165 the *same* 20 envs and the *same* fixed goals produce **no degradation
at all** (success 0.979 at u100).

**One explanation covers every observation, and it is the σ story from a
different angle:**

> With `freeze_log_std=1`, the **behaviour** policy is `N(μ, σ)` and the
> **scored** policy is `μ` alone. Training reward measures the former, eval
> measures the latter. When σ is large *and* reward is sparse, the behaviour
> policy collects its reward largely **through the noise** — so the training
> metric stays healthy while μ itself is only weakly constrained, and drifts.

Every observation follows: training reward stable while eval degrades ✓;
`follow_q` and `step_mag_mean`, both measured on μ, collapsing ✓; overshoot
absent because μ *shrank* ✓; σ=0.165 immune ✓; warm speedup small and not the
driver ✓.

**Consequences.**

- **`--reset_state_on_teleport` stays at `false`.** The gap it would close is
  1.2–1.3× and is not what is breaking these runs. Jack's bar of "a very very
  good reason" is not met, and the honest thing is to say so rather than take
  the change because it was available. Recorded as a *measured* 20–29%
  train/eval mismatch that remains, and would be worth revisiting only if
  something later depends on it.
- **The exploit recipe is σ=0.165**, and the explore recipe is σ=0.50. The
  conflict is real and quantified from both sides.
- **A general caution for this trainer:** with a frozen σ, *training reward is
  a measure of the behaviour policy and eval is a measure of the mean*, and the
  two come apart as σ grows — fastest where reward is sparse. `mean_r` rising
  is therefore **not** evidence that a run is healthy.

#### Live notes — wave 2, exploit arm

- **Config validated before committing GPU-hours.** `smoke_x` (4 updates,
  4 envs, CPU) confirms `nav=` metrics appear, `eval_scope=navexpl` skips
  goal-discovery as intended, and an eval costs 3.6 s. It also reproduced the
  documented trap in the wild: after 4 updates the policy reaches nothing and
  the log reads `'mean_steps': 0.0` with `'total_successes': 0` — **0.0, not
  NaN**, exactly as §P0.3.4 warns. Any automated ranking on `mean_steps` alone
  would put a totally failed policy first.

- **The trap fired for real on the second eval, and it is why `mean_steps_all`
  now exists.**

  | update | `success_rate` | `mean_steps` | **`mean_steps_all`** |
  |---|---|---|---|
  | u25 | 0.969 | 38.0 | **43** |
  | u50 | 0.510 | 28.4 | **112** |

  Read on `mean_steps` alone that is a 25% improvement; it is a large
  regression, with the mean taken over an easier surviving half.
  `collect_results` now derives and bolds
  `(successes·mean_steps + failures·max_steps) / trials`, so the tooling
  enforces what P0.3.4 previously only warned about.

- **u25 behaviour probe — the exploit gap is a POLICY gap, and its failure mode
  is TERMINAL.**

  | | d=0 | d=10 |
  |---|---|---|
  | `success_rate` | 1.000 | 0.771 |
  | `q_accuracy` *(is the signal right?)* | **0.983** | 0.892 |
  | `follow_q` *(does the policy follow it?)* | **0.702** | 0.445 |
  | `align_true` — successes / failures | 0.697 / — | 0.712 / **0.125** |
  | `step_mag_mean` | 0.324 | 0.229 |
  | `final_dist_fail` | — | **2.81** |
  | `fail_frac_at_edge` | 0.000 | 0.029 |

  1. **The readout is not the limit.** `q_accuracy` 0.98 against `follow_q`
     0.70 — the signal is nearly perfect and the policy follows it at cos 0.70.
     Via §3.3.1, cos 0.70 at |a| = 1 is `mean_steps` 15.3; at the actual
     |a| = 0.32 that predicts ~47 against the observed 39.3. **The whole of the
     exploit gap is magnitude plus follow-accuracy, both policy-side.**
  2. **Failures do not get lost — they get close and cannot close.**
     `final_dist_fail` is **2.8 cells** with only 2.9% of failure time at an
     edge, so this is not the corner trap. `align_true` splits 0.712 for
     successes against **0.125** for failures: near the goal the agent stops
     moving goalward at all.
  3. **The mechanism, and it corrects an earlier conclusion of mine.** As the
     agent approaches, `|q| → 0` and the recall *direction* becomes
     noise-dominated — while σ=0.50 adds ~0.5 cells of jitter against a
     `goal_radius` of 1.0. So the terminal approach is exactly where a large σ
     hurts, and it explains the u50 collapse: as |a| grew the policy began
     overshooting and orbiting.

     **§3.4.1's "a large σ is nearly free for the exploit metric" is wrong as
     stated.** That figure came from `exploit_reference.py`, which simulates a
     *perfect* follower and therefore has no terminal-approach problem at all.
     The real cost is not the 5% it predicted. **`w2_x_siglo` (σ=0.165) is now
     running to measure it**, and the practical consequence is that **the σ
     anneal is not a refinement but a requirement** for the combined model:
     large σ early for the magnitude ascent, small σ late for terminal
     precision.

### Wave 3 — combining the two (pre-registered; fires after waves 1–2)

**Written before any wave-1 or wave-2 result is in**, so the decision rules
cannot be fitted to the outcome. Everything here runs with
`--regime_assignment shuffle` (see §0) and the shaping and noise settings that
waves 1–2 select.

The four orderings Jack named, as schedules. **Re-sized for 20 envs × 64 batch**
(the shape Q1 selected): 12.6 s/update, so 1200 updates is ~4.2 h and fits the
6-hour limit with room for the eval overhead.

| arm | schedule | warm start |
|---|---|---|
| **A** *(the requested order)* | `interleave:1200,empty_frac=1.0->0.5,anneal=400` | best explore checkpoint |
| **B** *(interleave throughout)* | `interleave:1200,empty_frac=0.5` | fresh |
| **C** *(blocked, never interleaved)* | `explore:600 ; exploit:600` | fresh |
| **D** *(exploit first)* | `interleave:1200,empty_frac=0.0->0.5,anneal=400` | best exploit checkpoint |

All run `--regime_assignment shuffle` and `EVAL_SCOPE=navexpl`.

A and D warm-start via `--load_checkpoint`, which is what makes "anneal the
other regime in" cheap: the first phase is already paid for, so each arm costs
one run rather than two. **`--load_checkpoint` drops the Adam moments by
design** (memory `project_hopfield_nav_continue`), which is acceptable here —
the objective changes at the boundary, so stale moments would be wrong anyway —
but it means arm A's curve will dip for a few updates before recovering, and
that dip must not be read as interference.

**Warm-start from an intermediate checkpoint, not the final one.** The explore
arms run to u1500 but their curve is well above the target by ~u400; taking a
u600 checkpoint lets wave 3 start hours earlier at a small cost in explore
quality, and the *combination* question is what wave 3 is for.

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

#### Live notes — wave 3

- **A structural asymmetry between the arms that the pre-registration missed.**
  At `empty_frac=0.5` only **half** the rollouts are explore, so an
  interleaved-throughout arm accumulates coverage at roughly half the rate of a
  pure-explore run. Arm A front-loads that phase and arrives with it already
  paid for; arm B has to earn it at half speed *while also* learning to
  navigate. Visible immediately — A at u150 has coverage **0.312** (carried
  down from the 0.375 it was warm-started with), B at u250 has **0.062**.

  This is not a result, it is a **handicap in the comparison**, and it was not
  accounted for when the arms were designed. B is not slower at *combining*; it
  is slower at the explore half because it is given half as much of it. The
  fair reading of A vs B is therefore at **equal explore exposure**, not equal
  updates — or simply at the end, once both have had enough of each.

- Arm A's coverage fell 0.375 → 0.312 across the switch. Expected, and it is
  the sum of three things, none of which is interference: half the envs are now
  exploit, `--load_checkpoint` drops the Adam moments, and the objective it was
  optimizing changed. Whether it *recovers* while nav improves is the actual
  question.

- **ARM A COLLAPSES WHEN THE ANNEAL COMPLETES. This is the interference the
  wave was designed to detect.**

  | arm A | u50 | u150 | u250 | u350 | **u400** |
  |---|---|---|---|---|---|
  | coverage | 0.367 | 0.312 | 0.333 | 0.326 | **0.068** |

  and at u400 nav reads **0.948 / 26.2 / 35.3**. It held coverage in the
  0.30–0.37 band for 350 updates while `empty_frac` annealed 1.0 → 0.56, then
  **lost 80% of it in one 50-update step** as the fraction reached 0.50. A
  cliff, not a decline — and it bought navigation with it. **The model traded
  the explore metric for the exploit one**, which is exactly the failure the
  pre-registration named.

  Standings at their latest evals:

  | arm | coverage | nav d=0 (sr / steps / **all**) | reading |
  |---|---|---|---|
  | **A** explore-first | **0.068** | 0.948 / 26.2 / **35.3** | nav bought at the cost of coverage |
  | **B** interleaved | 0.128 | 0.958 / 31.3 / **38.4** | both, but both weak |
  | **C** blocked | **0.312** | 0.583 / 86.5 / **133.8** | still in its explore phase; nav not yet trained |

- **The likely mechanism, and it points at a knob on Jack's list.** PPO
  normalizes advantages **once over the pooled buffer**
  (`updates/ppo.py:210-214`), and the two regimes contribute rewards of very
  different *shape*:

  > An explore rollout earns ~0.23/step, smoothly, from novelty. An exploit
  > rollout earns −0.05/step punctuated by **+5.0** spikes at each goal. After
  > GAE, the exploit rollouts' advantages are far larger in magnitude, so when
  > the two are pooled and normalized together **the exploit gradient dominates
  > and the explore signal is scaled toward zero.**

  That predicts the cliff: as `empty_frac` falls, exploit rollouts take a
  growing share of the pool until they dominate the normalization outright, at
  which point explore stops being trained at all.

  **`GOAL_REWARD` is the knob** (v35's 5.0 against novelty's 0.3), and it is on
  the list. §3.5 records it as *inert* in explore-only and exploit-only runs —
  a constant offset and a pure scale respectively — but **in an interleaved
  run it is neither**: it sets the ratio between two regimes sharing one
  normalization, which is exactly the quantity that decides which one the
  gradient serves. That is a case §3.5 did not cover, and it is now the
  leading candidate for wave 4.

#### CONCLUSION — wave 3. Interleaving wins; the requested ordering is refuted.

The pre-registered rule was: *"If B ≥ A on the composite, the requested
ordering is refuted and interleaving becomes the recipe; report that plainly
rather than continuing to tune A."* Reporting it plainly.

| arm | ordering | coverage | nav d=0 (sr / steps / **all**) | nav d=10 **all** |
|---|---|---|---|---|
| **A** | explore-first, anneal exploit in *(requested)* | 0.068 | 0.948 / 26.2 / 35.3 | 63.4 |
| **B** | **interleaved throughout** | **0.198** | **1.000 / 12.6 / 12.6** | **32.5** |
| **C** | blocked, never interleaved | 0.223 ↓ | 0.906 / 27.6 / 43.8 | 64.2 |
| **D** | exploit-first, anneal explore in | 0.062 | 0.990 / 10.2 / 12.2 | 46.4 |

**B beats A on *both* metrics at once** — 2.9× the coverage and 2.8× better
`mean_steps_all` — and it is the only arm that holds both. So:

- **The requested explore-first ordering is refuted.** It reaches good
  navigation and destroys coverage doing it (§ the corner-trap diagnosis
  above). The prediction registered before the wave — that a pure-explore
  phase installs a prior the combined phase must unlearn — is what happened,
  though by a sharper mechanism than "unlearning": exploit training installs
  *persistent q-following*, which in an explore rollout drives the agent into
  a wall.
- **D is the mirror image and confirms the reading.** Exploit-first gives the
  best navigation of any arm (10.2 steps) and the worst coverage (0.062).
  Whichever regime a run starts in, it keeps that skill and loses the other.
  **Only simultaneous exposure holds both.**
- **C separates the two candidate explanations.** It sees both regimes but
  never together, and it behaves like A: coverage climbs to 0.351 during its
  explore block then *falls* through the exploit block (0.351 → 0.278 →
  0.223), while nav stays mediocre. So it is **not** "seeing both regimes"
  that matters — it is seeing them **in the same PPO update**.

**Arm B is a genuine single model for all three metrics**: coverage 0.198,
`success_rate` 1.000, `mean_steps` 12.6. But it is not yet good enough —
0.198 coverage against the explore-only 0.385 means **the combined model pays
about half its coverage** for navigation. That residual interference is what
wave 4 attacks.

**The failure mode to watch for is interference, not either metric alone.** A
model that scores 0.35 coverage and `mean_steps` 30, or 10.5 steps and 0.10
coverage, has not combined anything. `analysis/nav_tri/behavior_probe.py`
diagnoses which: `chase_q` in explore mode should stay near 0 (the policy
ignores phantom recalls) while `follow_q` in nav mode should be high (it
follows real ones). A model that fails by chasing distractors while exploring
shows high `chase_q`; one that fails by ignoring the goal shows low `follow_q`
with high `q_accuracy`.
