# Continual control suite — running log

Companion to [CONTINUAL_CONTROLS_PLAN.md](CONTINUAL_CONTROLS_PLAN.md). Newest
entries at the bottom. Every entry records what was done, what it produced, and
what it changed about the plan — including the things that turned out to be
wrong, which are the entries worth reading.

---

## 2026-08-30 — plan v1

Wrote the plan. Surveyed the existing comparison, extracted the current numbers
from `analysis/continual/histories/`, and read the continual-learning
literature for what belongs in the suite.

The recorded state at the time: the Hopfield agent at ~1.0 on every env; the
RNN control at ~0.19 retained and 0.99 on the env it is currently training on.
One control, no continual-learning method of any kind.

---

## 2026-08-30 — plan v2, after review

Six decisions came back. Two were corrections to v1, and both are worth keeping
visible rather than quietly editing away.

**`batch_envs=1` is a deliberate regime, not a defect.** v1 called it "1/16 of
the intended gradient budget." Wrong twice. Arithmetically: at `batch_envs=1`
the batch dimension is 1, so `n_minibatches` cannot split anything and is a
no-op — the only live knob is `epochs`, making it 4×, not 16×. Conceptually:
"budget" was the wrong frame. One rollout → one update is what makes the x-axis
read as *episodes consumed*, 200 for the RNN against 1 for the store, and that
is the axis the whole cost frontier hangs on. Kept as the headline regime; the
one real residual — gradient variance from a single autocorrelated trajectory —
became a `batch_envs=16` sensitivity condition instead of a fix.

**The continual-RL framing is cut.** The update is plain cross-entropy on
oracle actions: no reward in the loss, no value head, no policy gradient. v1
leaned on the continual-RL literature because that is what the searches
surfaced. The CRL taxonomy and the Continual World numbers do not transfer.
CLEAR survives, restated by mechanism — replay plus distillation to the past
self, which is a supervised loss once V-trace and the value term are stripped.

**Movement mode stays `continuous`.** v1 recommended `discrete`. Checked: every
headline `agenthash` run is continuous, so discrete would compare two different
tasks unless the Hopfield side were re-run. Discrete *is* easier, and easier-
for-the-control is the right direction, but not at that price.

Also: `input_prev_action` on, N=5 headline with an N=20 scaling panel, suite cut
from ~20 methods to 9, §5.1 and §5.2 promoted to first-class.

---

## 2026-08-30 — the prev-action channel was broken, not merely off

Turning on `--input_prev_action` crashed immediately:

```
RuntimeError: input.size(-1) must be equal to input_size. Expected 62, got 60
```

Both `rollout/rnn.py` and `evaluation/rnn.py` assembled the previous-action
channel only when a previous action existed — which is false at `t=0`. So the
first forward of every rollout and every eval fed the trunk an input two columns
narrower (continuous) or four narrower (discrete) than `compute_rnn_input_dim`
had sized it for.

**`input_prev_action` and `input_prev_reward` have therefore never been usable
on this stack at all**, which is why every recorded continual history has them
off. Fixed: `prev_action_channel()` returns an all-zero "no previous action"
row at t=0, distinct from every one-hot and from any real displacement.
`prev_reward` starts at a genuine zero. Runs with both flags off are
bit-identical — the values are only read when their cfg flag is set, and the
golden fixtures agree.

19 regression tests in `test_prev_action_channel.py`, including one asserting
that step *t*'s channel really carries step *t−1*'s action, so a "fix" that
always passed zeros would not pass.

Also wired `--init_log_std` / `--freeze_log_std` through
`analysis/continual/baseline.py` (plan §3.1 W5). They exist on `train_rnn` but
were never exposed here, so every continual run to date used the dataclass
default of `0.0` — σ = 1.0 against a unit-magnitude action, learnable, and
unsettable from the run script.

---

## 2026-08-30 — Wave 0 launched (slurm 21626914)

`analysis/continual/run_wave0.sh` on `ou_bcs_normal`, 64 parallel single-
threaded CPU tasks:

- **T0.1** joint ceiling, 4 widths × 2 depths × 3 seeds via `train_rnn --mode mixed`
- **T0.4** from-scratch sequential, 2 arms (`noprev` / `prev`) × 20 seeds
- **T0.3** oracle reachability, inline, seconds

T0.2 (per-env experts) deliberately not built: it only measures capacity
interference as `T0.1 − T0.2`, which is not worth the code unless T0.1 lands
low.

### T0.3 answered immediately: the oracle ceiling is 1.0000

Over 25 envs (5 seeds × 5 envs), the BFS teacher reaches on **every** trial,
with a worst case of **24 steps against the 200-step cap**. So `reached` really
is capped at 1.0 and the recorded ~0.19 retention is not an artifact of the
step limit.

Side finding: the eval has ~8× more step budget than it needs. If the N=20
panel becomes the long pole, shortening `max_steps` is a cheaper lever than
reducing the eval cadence — though it changes the task definition against the
existing histories, so it is a last resort.

### Compute is CPU-bound, and much cheaper than assumed

Measured **~1000 environment-steps/s per CPU core** at `batch_envs=1`. A
128-unit GRU on a batch of 1 barely touches an accelerator, so the scaling axis
is CPU fan-out, not GPUs.

| protocol | env-steps / seed | wall / seed |
|---|---|---|
| N=5, 200 upd/env | 800 k | ~13 min |
| N=20, 200 upd/env | 9.2 M | ~2.6 h |

One 192-CPU `ou_bcs_normal` node runs ~150 N=5 seeds concurrently. This likely
explains why the existing 30-seed job needed a 12 h allocation: 30 processes
contending on one GPU. Queue time, not compute, is what binds the 24 h target.

---

## 2026-08-30 — the continual-method scaffold

`hopfield_nav/continual/` — the methods, distinct from `analysis/continual/`,
which is the figure pipeline for the same experiment.

`ContinualMethod` is six hooks, each corresponding to a place the literature's
methods actually intervene: `on_block_start`, `extra_batches`, `penalty`,
`aux_loss`, `after_update`, `on_block_end` — plus `state_bytes`, which is not
bookkeeping but one of the five axes of the cost frontier.

Two things fell out of the design that are worth recording:

- **Replay needs no hook in the update at all.** `bc_rnn_update` already takes
  a *list* of rollouts and concatenates them, so a method contributes replayed
  trajectories simply by having them appended. Loss weighting across new and
  replayed data is then by supervised-token count, which is what we want.
- **The dependency between `updates` and `continual` runs one way.**
  `bc_rnn_update` takes plain callables, so it never imports a method;
  `training/rnn_sequential.py` composes the two. That is what lets both sit at
  layer 4 without a cycle. Added to the layering table.

**ER** (`replay.py`) stores whole trajectories, not timesteps — a replayed
timestep torn out of its trajectory would be supervised in a recurrent context
the agent could never have been in. Sampling is per-env balanced by default,
because the stream is *ordered by env* and a uniform draw late in training is
dominated by recent envs, which is the same recency bias the method exists to
fix. `buffer_size=inf` is the default: the interesting result is where perfect
memory sits on the cost frontier, not whether a small buffer degrades.

**Online EWC** (`regularize.py`) computes the *true* Fisher — actions sampled
from the model — not the empirical Fisher of the training loss. The estimator
is trajectory-level (one backward per trajectory, squared, averaged) rather
than per-timestep, which is stated in the module docstring rather than buried:
the exact per-timestep diagonal would need hundreds of thousands of backward
passes per block, and for a recurrent policy the trajectory likelihood is
arguably the more natural object anyway. `fisher="empirical"` is available as
an ablation so the difference can be measured rather than asserted.

### Three test failures, all informative

The behavioural tests caught three real things, which is the argument for
writing them that way rather than as smoke tests:

1. **The closed-form penalty test was wrong, not the code.** `penalty()`
   correctly skips frozen parameters — `movement_log_std` under
   `freeze_log_std=True` cannot move, so penalising it is meaningless — and the
   test's expected value counted them. Pinned that behaviour explicitly in a
   new test rather than just fixing the arithmetic.
2. **`gamma` decayed only the keys the new Fisher estimate contained.** A
   parameter missing from the estimate (a frozen one) would stay pinned at its
   old importance forever while everything around it decayed. Now every
   existing entry is decayed, then the new one added.
3. **The "large lambda restrains drift" test was measuring the wrong thing,
   twice.** First it compared against a `None` method — but estimating the
   Fisher samples from the model and consumes the global torch RNG, so a naive
   run diverges for reasons unrelated to the penalty. The control has to be
   `OnlineEWC(lam=0.0)`, which walks the same code path and draws the same
   numbers. Second, it measured distance from *init*, but `_anchor` is
   overwritten at every block end, so the anchor it read was the end of block 1
   rather than the point block 1 was supposed to stay near. Both fixed; the
   test now asserts the two runs reach the same block-0 anchor before comparing
   drift, so a broken control fails loudly instead of silently passing.

Full suite green.

---

## 2026-08-30 — Wave 0 results, and a verdict that was wrong

### T0.4: the from-scratch floor is far below the pretrained baseline

20 seeds per arm, 5 envs × 200 updates, `reached` over the last 20 % of the
final block:

| arm | retained (envs 0…3) | current env | forgetting | stability gap | never reached criterion |
|---|---|---|---|---|---|
| `noprev` (legacy surface) | 0.054 ± 0.012 | 0.579 ± 0.082 | 0.422 ± 0.032 | 0.172 ± 0.024 | 71 % |
| `prev` (settled surface) | 0.046 ± 0.010 | 0.510 ± 0.069 | 0.406 ± 0.032 | 0.168 ± 0.024 | 74 % |

Two things follow.

**Pretraining is doing real work, and now we know it.** The recorded pretrained
baseline retains ~0.19 and scores ~0.99 on the current env. From scratch it is
0.05 and ~0.55. So pretraining roughly quadruples retention and nearly doubles
current-env performance. Plan §3.2 P2 worried that pretraining might be doing
nothing and that nobody would know, because the from-scratch control had never
been run. It is not nothing.

**`input_prev_action` makes no detectable difference here.** 0.054 vs 0.046
retained, 0.579 vs 0.510 on the current env — both differences are within one
SEM of each other and the arms overlap. The decision to turn it on is harmless
but is not a win, and no direction should be read from these numbers.

Also worth flagging: at 200 updates/env from scratch, **71–74 % of envs never
reach the 0.9 criterion at all**, and the current env only reaches ~0.55. The
from-scratch control is not merely forgetting; it is barely learning. That
makes the *pretrained* arm the meaningful control for the suite, and it is
exactly what the Tier-1 tuning (W1–W6) exists to fix.

### T0.1: the joint ceiling came back low — and the obvious reading was wrong

The first run gave ~0.45–0.54 across every capacity from hidden=128 to 1024,
against an oracle of 1.000. Read at face value that is a **capacity** result:
one network cannot hold five envs at once, no continual method could exceed
~0.5, and a large part of the recorded "forgetting" would not be forgetting at
all. That is a big claim, and it is what the summary script printed.

It is not what happened. Two things say so:

1. **Capacity does not move the number.** 128 → 0.45, 256 → 0.54, 512 → 0.51,
   1024 → 0.36. Flat and noisy, with no monotone trend. If capacity were
   binding, more of it would help.
2. **Every curve is still climbing where the budget ends.** hidden=128 goes
   0.47 → 0.66 over its last 100 updates; the end-slope is +0.06 to +0.15 for
   every configuration. Nothing has plateaued.

The cause is arithmetic. At `batch_envs=1` and `epochs=1`, 1000 updates is
**1000 gradient steps** — against roughly 1M timesteps of collected data. The
run was optimisation-starved, not capacity-limited, and a "ceiling" that is
still rising when the budget runs out is not a ceiling.

**Two fixes.**

`wave0_summary` now measures the end-slope of the eval curve and refuses to
issue a capacity verdict while the run is still improving; it reports
INCONCLUSIVE and says the number is a lower bound. The check runs *before* the
capacity branch, because a still-climbing curve explains a low ceiling without
any capacity story and reporting one anyway would be inventing a result.

`run_wave0b.sh` re-runs T0.1 properly: **epochs 1 → 8** (eight passes over the
same five rollouts — 8× the gradient steps at zero extra environment cost,
which is the cheap axis when the bottleneck is optimisation) and **1000 → 8000
updates**, for 64,000 joint gradient steps against the original 1,000. It also
adds the lr axis, never swept before, and narrows capacity to {128, 512} ×
{1, 2} since capacity was already shown not to bind.

This run deliberately does **not** respect the online regime. T0.1 is an
*offline* reference — the best a single network can do given every env at once
— so hobbling it with the streaming protocol's one-step-per-rollout rule would
understate the ceiling, which is the one thing a ceiling must not do.
`n_minibatches` stays 1 so every gradient step sees all five envs, which is
what makes it joint rather than round-robin.

Submitted as slurm 21627945.

---

## 2026-08-30 — Wave 1 launched (slurm 21628160)

`analysis/continual/run_wave1.sh`, 96 CPUs, five arms:

| arm | what | runs |
|---|---|---|
| **A** | Tier-1 tuning of the *pretrained* control: lr × optimizer-reset × epochs | 12 configs × 8 seeds |
| **A-batch** | the W1 sensitivity condition, `batch_envs=16` | 8 |
| **A2** | Tier-1 tuning of the *from-scratch* control: `init_log_std` × lr (W5) | 6 configs × 8 seeds |
| **R** | `method=none` at exactly B and C's configuration | 8 |
| **B** | Experience Replay: buffer ∈ {inf, 200, 50, 10} × replay_batches ∈ {1, 4} | 8 configs × 8 seeds |
| **C** | Online EWC: λ over six decades | 6 configs × 8 seeds |

Three design points worth recording.

**The pretrained arm is primary, and T0.4 is why.** From-scratch came back at
0.05 retained with only ~0.55 on the env it is currently training on and 71–74 %
of envs never reaching criterion. Method differences measured on a control that
is barely learning would be noise on top of a broken baseline. The pretrained
arm reaches ~0.99 on the current env, which is where a retention difference can
actually show.

**`init_log_std` is swept only on the from-scratch arm.** `movement_log_std` is
a `Parameter`, so `load_state_dict` overwrites whatever `--init_log_std` asked
for whenever a checkpoint is loaded. Sweeping it on the pretrained arm would
sweep a value that never takes effect — precisely the kind of silently-inert
knob W5 was about in the first place.

**Arm R exists because B and C are meaningless without it.** A method has to be
read against `none` at *its own* configuration, not against the recorded
default from a different sweep.

Wave 1 does not wait on the corrected T0.1: the joint ceiling says how to
*interpret* these numbers, not what to run.

### `--method_args` coercion was wrong, and the smoke test caught it

`fisher=true` came back as the boolean `True` and `OnlineEWC` rejected it:

```
ValueError: fisher must be 'true' or 'empirical', got True
```

The parser was guessing types from the text alone, so it could not tell
`fisher=true` (the *string* naming one of two Fisher estimators) from
`normalize_fisher=true` (the boolean). Coercion needs the target type, and only
`build_method` knows it.

`parse_method_args` now returns raw strings and `build_method` coerces each
value against its parameter's default — `str` defaults keep the string, `bool`
defaults take true/false/1/0/yes/no and reject anything else, `int` and `float`
defaults cast, and `inf`/`none` are recognised before the numeric branch. A bad
boolean now names the argument and its default in the error rather than failing
somewhere inside the method. Four tests cover it, including the exact
`fisher` vs `normalize_fisher` case that broke.

Also implemented **W2** (`--reset_optimizer_each_block`), which clears Adam's
moment estimates at each task boundary in place, leaving parameter groups and
the learning rate untouched. Off by default, because that is what every
recorded history did.

---

## 2026-08-30 — Wave 2 methods implemented (ahead of the wave)

Built while Wave 0b and Wave 1 were on the cluster, so Wave 2 can launch the
moment Wave 1's tables land. The `ContinualMethod` interface held up: all four
fit the existing hooks, and the only addition needed was wiring the `on_step`
callback that was already stubbed in `bc_rnn_update` through to
`after_step`.

**SI** (`regularize.py`). The contrast with EWC is *where importance comes
from*. EWC stops at a boundary and asks a curvature question about the
endpoint; SI never stops, crediting each parameter with the loss reduction it
personally produced along the path, then normalising by how far it actually
travelled. Consequences worth having next to EWC: it needs no separate Fisher
pass, so it is nearly free where EWC pays a backward per trajectory per block;
and the accumulation itself needs no task boundary — only the fold-in does.

**LwF** (`distill.py`). No buffer, no per-parameter state — a model copy and
nothing else, which makes it the cheapest point on the memory axis by a wide
margin. Inactive in block 0 on purpose: there is nothing to preserve yet, and
snapshotting there would only pin the policy to its initialisation.

**CLEAR** and **DER++** (`distill.py`). Both are ER plus an output-space
anchor, and the interesting difference is *when the anchor is taken*. CLEAR
snapshots one converged past self and distils every replayed state against it.
DER++ freezes a target the instant a trajectory enters the buffer, so different
entries are anchored to different, older versions of the policy — a spread of
the optimisation trajectory rather than a single point. Both are boundary-free;
they disagree about what "the past" means. DER++ stores distribution
*parameters* rather than a frozen network, so its state is the buffer plus two
small tensors per entry instead of a model copy.

Design note: the distillation terms all go through `aux_loss`, not `penalty`,
because they regularise the model's outputs *on specific states* and therefore
need the data that `penalty` never sees. The frozen model's outputs are
constant within an update but `aux_loss` is called once per minibatch step, so
they are computed once and cached — and the cache is cleared in `after_update`,
because a stale target distilled against new data would be silently wrong.

The tests target the failures that would otherwise be invisible: KL against an
unchanged model must be exactly zero (otherwise the term is measuring something
other than divergence), SI's path integral must actually accumulate (a driver
that stops calling `after_step` turns SI into a no-op that still looks like a
method in the history), and DER++'s targets must stay index-aligned with its
buffer under reservoir eviction (misalignment distils each trajectory against
another one's target, and every "is the loss positive" test still passes).

Full suite green.

---

## 2026-08-30 — Wave 1 died on a concurrency bug in `WorldSpec.write`

246 of 272 runs failed within 12 minutes:

```
FileNotFoundError: '.../histories/wave1/world.json.tmp' -> '.../histories/wave1/world.json'
```

`WorldSpec.write` staged through a **fixed** temp name, `world.json.tmp`. Safe
for one writer; quietly broken for several. Every run writes its `world.json`
into the `--out` directory, so 272 concurrent processes all created the same
staging file — the first `os.replace` consumed it and every later one raised.
The failure happened *after* the environments were built and *before* any
training, so it cost a node-hour and produced nothing.

Wave 0 had the same shape (40 concurrent T0.4 runs into one directory) and got
away with it: the race is probabilistic and 40 writers spread over a slower
startup mostly missed each other. 272 launched at once did not.

**Two fixes, because there were two problems.**

*The race.* `write` now stages through `tempfile.mkstemp` in the destination
directory. Unique across threads as well as processes — a pid suffix would have
fixed the processes and left threads broken — and the rename stays atomic, so a
reader still never sees a half-written file. Mutation-checked: with the old
shared name, 48 concurrent writers give **27 failures**; with `mkstemp`, **0**.

*The semantics.* Even with unique staging, 272 runs at different seeds all
overwrite one `world.json`, which then describes whichever finished last and
none of the others — worse than absent. `baseline.py` gained
`--world_spec/--no-world_spec` (default on, preserving behaviour) and the
sweeps pass `--no-world_spec`.

### What the 26 survivors already showed

Not enough seeds to conclude anything, but one thing was immediately visible and
is the reason `current_env` sits next to `retained` in every table:

| config | retained | current env |
|---|---|---|
| online EWC, λ=1e5 | **0.384** | **0.024** |
| online EWC, λ=1e4 | 0.204 | 0.500 |
| ER, buffer=10, replay×4 | 0.329 | 0.951 |
| no method (reference) | 0.043 | 0.317 |

EWC at λ=1e5 "wins" retention by refusing to learn anything at all. A
leaderboard on `retained` alone would rank it first. This is exactly the
degenerate solution the plasticity column exists to expose, and it will need
saying explicitly in the results.

Wave 1 relaunched as slurm 21628688 with the fix. Full suite green.

---

## 2026-08-30 — Wave 0 complete, and §5.2 built and launched

### Wave 0 final (slurm 21626914, all 64 tasks OK)

The full capacity sweep, 3 seeds each:

| hidden | layers | final | end-slope |
|---|---|---|---|
| 128 | 1 | 0.452 ± 0.150 | +0.073 |
| 128 | 2 | 0.383 ± 0.101 | +0.075 |
| 256 | 1 | 0.438 ± 0.099 | +0.084 |
| 256 | 2 | 0.535 ± 0.150 | +0.095 |
| 512 | 1 | 0.465 ± 0.114 | +0.148 |
| 512 | 2 | 0.508 ± 0.109 | +0.132 |
| 1024 | 1 | 0.271 ± 0.110 | +0.048 |
| 1024 | 2 | 0.588 ± 0.125 | +0.159 |

**Every configuration is still rising**, and capacity does not order the
results (1024×1 is the *worst* row). The verdict stays INCONCLUSIVE and the
numbers are lower bounds. The corrected run at 64× the gradient budget is the
one that will answer it.

### §5.2 — the in-context, zero-weight-update control (slurm 21629579)

The measurement Jack flagged as important, and the one that could change the
framing. Three pieces:

**The collector gained `carry_across_episodes`.** Normally an env that reaches
its goal is *frozen* for the rest of the rollout, so each row is one
independent episode and the recurrent state never carries anything between
them. With the flag on, a reaching env is instead **teleported to a fresh start
and the hidden state is kept**, so one row becomes a lifetime — a sequence of
episodes in the same environment linked only by recurrent activity.

**`evaluation/incontext.py` measures success against episode index.** One
environment, N episodes back to back, `h` zeroed only at the start of a
lifetime. The goal is never observed and never moves, so an agent that solves
episode 5 faster than episode 1 can only be doing it by remembering where the
goal was, in activations, with no weight change.

**The episodic control arm is not optional.** A rising curve on its own proves
very little — a policy that drifts toward the middle of the arena, or simply
explores well, would produce one for reasons unrelated to memory. The control
is trained identically and differs *only* in whether state survived a
goal-reach, so the difference between the curves is the part attributable to
carrying anything. Evaluation is on held-out envs at a seed the pretraining
never saw, which is also what guards against the arms just memorising the pool.

The tests target the two ways this could be silently broken while still looking
clean: the collector still freezing reachers (a "lifetime" that is one episode
plus 190 frozen steps) and the evaluator zeroing `h` between episodes (no
channel for anything to carry). Both would produce a flat curve — the same
answer as a genuine negative — so both are asserted directly rather than
inferred.

Job: 32-env pool, 2000 updates, hidden=256, 3 seeds per arm, evaluated on 8
held-out envs × 64 lifetimes × 10 episodes.

---

## 2026-08-31 — T0.1 answered: the joint ceiling is ~0.99, and it was never capacity

The corrected run (slurm 21627945, epochs 1→8, 8000 updates) is still in
flight, but the answer is already unambiguous. Mean `nav_det` across all five
envs, seed 1:

| hidden | layers | lr | updates so far | mean nav_det |
|---|---|---|---|---|
| 128 | 1 | 1e-3 | 5500 | **0.988** |
| 128 | 2 | 1e-3 | 4100 | **0.994** |
| 512 | 1 | 1e-3 | 1500 | 0.944 |
| 512 | 2 | 1e-3 | 600 | 0.925 |
| 128 | 1 | 3e-3 | 5000 | 0.525 |
| 128 | 2 | 3e-3 | 3400 | 0.125 |
| 512 | 1 | 3e-3 | 1500 | 0.206 |
| 512 | 2 | 3e-3 | 600 | 0.050 |

**The joint ceiling is ≈ 0.99.** The first run's 0.45–0.59 was an
optimisation artifact and nothing else. Three things follow, and they matter
for how every other number in this suite is read.

**Capacity was never the constraint.** The best joint result comes from
`hidden=128, layers=1` — the *smallest* configuration tested, and exactly the
one the recorded baseline uses. A 128-unit GRU holds all five environments
simultaneously at 99 %. The original verdict would have claimed the opposite:
that the network cannot represent five envs at once and that much of the
recorded "forgetting" is really a capacity limit. That claim would have been
wrong, it would have gone into the paper, and the only thing that stopped it
was noticing the eval curves were still climbing.

**The retention gap is entirely forgetting.** The same architecture, at the
same capacity, on the same five environments, scores **0.99 trained jointly**
and **0.196 trained sequentially**. Nothing about representational capacity
explains that difference. Tier 2 is fully interpretable and T0.2 (per-env
experts) is not needed — it would only have been informative if the ceiling had
come in low.

**lr matters more than capacity, and 3e-3 is unstable.** Every lr=3e-3 row is
between 0.05 and 0.53 while every lr=1e-3 row is between 0.92 and 0.99. The
Wave-1 Tier-1 sweep covers 3e-4 / 1e-3 / 3e-3 on the sequential side, so
whether the same instability shows up there is about to be measured rather
than assumed.

Headroom for continual methods on the headline protocol: **0.196 → 0.99**.

---

## 2026-08-31 — Wave 1 complete (slurm 21628688, all 272 tasks OK)

Against a joint ceiling of **~0.99** and a matched reference of **0.044**:

### A — the naive control, tuned (plan §3.1)

| config | retained | current env |
|---|---|---|
| **lr=3e-4, optimizer reset, ep=1** | **0.081 ± 0.026** | 0.820 |
| lr=3e-4, no reset, ep=1 | 0.066 ± 0.021 | 0.790 |
| lr=1e-3, no reset, ep=1 *(the recorded default)* | 0.044 ± 0.013 | 0.753 |

Tuning nearly **doubles** the control's retention, 0.044 → 0.081. So the
recorded baseline *was* a mild strawman and now is not. The two knobs that did
it: **a lower learning rate** and **W2, resetting Adam's moments at each task
boundary** — the reset arm beats the no-reset arm at matched lr and epochs in
five of six pairs. Neither changes the conclusion: 0.081 against a 0.99 ceiling
is still nothing.

### A-batch — the W1 sensitivity condition, and it vindicates the regime

`batch_envs=16` gives **current env 0.997** (the best plasticity anywhere in the
suite) and **retained 0.056** — statistically indistinguishable from
`batch_envs=1`'s 0.044.

So the gradient noise of a single autocorrelated trajectory is **not** doing
the forgetting. That was the one real residual left after the v2 correction,
and it is now measured rather than argued. `batch_envs=1` stays, and the
episodes-consumed axis stays interpretable.

### A2 — from scratch, and W5 was a real gap

| init_log_std | best retained | current env |
|---|---|---|
| −1.0 | 0.069 ± 0.032 | 0.854 |
| −1.5 | 0.048 ± 0.016 | 0.723 |
| 0.0 *(the unreachable default)* | 0.034 ± 0.017 | 0.582 |

σ = 1.0 against a unit-magnitude action was costing about half the retention
and a quarter of the plasticity. The flag that could not be set from the run
script mattered.

### B — Experience Replay

| config | retained | current env | stored |
|---|---|---|---|
| **buffer=∞, replay×4** | **0.419 ± 0.044** | 0.768 | 53.6 MB |
| buffer=200, replay×4 | 0.404 ± 0.032 | 0.793 | 10.7 MB |
| buffer=50, replay×4 | 0.256 ± 0.046 | 0.832 | 2.7 MB |
| buffer=∞, replay×1 | 0.143 ± 0.031 | 0.710 | 53.6 MB |

**Replay ratio dominates buffer size.** Every `rb=4` row beats every `rb=1` row,
while ∞ versus 200 barely matters (0.419 vs 0.404 at a fifth of the storage).

### C — Online EWC, and the plasticity trap in full

| λ | retained | current env |
|---|---|---|
| 1e5 | 0.274 ± 0.032 | **0.296** ← degenerate |
| 1e4 | 0.149 ± 0.033 | 0.524 |
| 1e2 | 0.095 ± 0.033 | 0.762 |
| 1e0 | 0.050 ± 0.009 | 0.747 |

Ranked on retention alone, λ=1e5 is the best regulariser in the suite. It is
also barely learning. The best *usable* setting is λ=1e4 at 0.149 — real, and
far behind replay.

### The plan's §0.1 prediction was wrong, and that is the interesting part

§0.1 said: *"a replay buffer with an unbounded budget will probably close most
of this gap."* It did not. Unbounded ER reaches **0.419 against a 0.99
ceiling** — under half the distance, with 53.6 MB of stored trajectories.

The reason is visible in the sweep itself. At `replay_batches=4` an update sees
one new trajectory against four replayed ones, which is not joint training; it
is training on the current env with a correction term. Perfect memory is
necessary but nowhere near sufficient — what matters is how much of each
gradient step is spent on the past.

So Wave 2 gets a new arm **I**: unbounded ER at replay ratios 8, 16 and 32. If
ER converges on the ceiling by rb=32, §0.1 was right about the destination and
wrong about the price, and the honest statement becomes "replay matches the
store when it replays 32× per step and stores every trajectory it has ever
seen". If it plateaus well below, replay has a real ceiling here — a stronger
result than the one that was expected.

Wave 2 launched as slurm 21631698 (CLEAR, DER++, SI, LwF, frozen trunk, and
arm I).

---

## 2026-08-31 — the joint ceiling, settled: 0.985, converged

The corrected run finished, and with the reader fixed (below) the answer is
clean:

| hidden | layers | lr | budget | final | end-slope | |
|---|---|---|---|---|---|---|
| 128 | 1 | 1e-3 | **8000** | **0.9854 ± 0.0055** | +0.012 | **converged** |
| 128 | 1 | 3e-3 | 8000 | 0.1896 ± 0.0611 | −0.083 | degrading |
| 128 | 1 | 1e-3 | 1000 | 0.4521 ± 0.1497 | +0.073 | still rising |
| … | | | 1000 | 0.27–0.59 | +0.05…+0.16 | all still rising |

**The joint ceiling is 0.985**, at the smallest capacity tested and at exactly
the configuration the recorded baseline uses. Sequentially, the same network on
the same envs retains 0.044 untuned and 0.081 tuned. The gap is forgetting, in
full.

lr=3e-3 does not merely underperform — it **diverges**, ending 8000 updates at
0.19 with a negative slope. That is the same lr the sequential sweep found
worst, so the instability is a property of the optimisation, not of the
continual protocol.

### Two reader bugs, caught by inspecting the table before publishing it

**The budget was not in the key.** `load_joint` keyed on
`(hidden, layers, lr)`, so the 1000-update run and the 8000-update run both
landed on `(128, 1, 1e-3)` and were averaged into **0.719** — a number
describing neither, sitting between an under-trained 0.45 and a converged 0.99.
That would have gone straight onto the results page as the joint ceiling.
`n_updates` is now part of the key: a budget is not a nuisance parameter here,
it is the thing the second run changed.

**A negative slope was being called "converged."** The convergence test was
`slope <= 0.02`, which is true of a run getting rapidly *worse*. lr=3e-3 at
−0.083 was labelled converged and would have been read as a genuine capacity
result. The test is now two-sided: only a flat tail is convergence, a rising
one is a lower bound, and a falling one is divergence.

Neither bug would have crashed anything. Both would have produced a
plausible-looking table with a wrong number in it, which is the failure mode
this whole exercise keeps running into: the errors that matter here are the
ones that still render.

---

## 2026-08-31 — Wave 2 complete (slurm 21631698, all 144 tasks OK)

Against the joint ceiling of **0.985** and the matched reference of **0.044**:

| method | best config | retained | current env | stored |
|---|---|---|---|---|
| **ER, replay ×32** | `buffer=∞` | **0.579 ± 0.039** | 0.796 | 53.6 MB |
| ER, replay ×16 | `buffer=∞` | 0.546 ± 0.027 | 0.753 | 53.6 MB |
| ER, replay ×8 | `buffer=∞` | 0.508 ± 0.033 | 0.762 | 53.6 MB |
| ER, replay ×4 | `buffer=∞` | 0.419 ± 0.044 | 0.768 | 53.6 MB |
| CLEAR | `cc=1.0` | 0.201 ± 0.036 | 0.622 | 53.9 MB |
| online EWC | `λ=1e4` | 0.149 ± 0.033 | 0.524 | 0.6 MB |
| DER++ | `α=1.0` | 0.143 ± 0.031 | 0.710 | 56.8 MB |
| naive, tuned | `lr=3e-4, reset` | 0.081 ± 0.026 | 0.820 | — |
| SI | `λ=10` | 0.074 ± 0.022 | 0.765 | 0.6 MB |
| **frozen trunk** | `+ER` | **0.043 ± 0.010** | 0.399 | 53.6 MB |

Plus two more entries in the plasticity trap: **LwF at α=10** (retained 0.235,
current **0.238**) and **online EWC at λ=1e5** (0.274, current **0.296**). Both
would top a leaderboard ranked on retention.

### Replay ratio keeps paying, and still does not close the gap

Arm I answered the question Wave 1 raised. Retention against replay ratio, all
at an unbounded buffer:

| ratio | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| retained | 0.143 | 0.419 | 0.508 | 0.546 | 0.579 |

Monotone, clearly decelerating (+0.089, +0.038, +0.033 per doubling), and at
**0.579 against a 0.985 ceiling** after replaying 32 stored trajectories for
every new one. Plan §0.1 said a perfect buffer would close most of the gap. It
does not: perfect memory plus a 32:1 replay ratio gets 59 % of the way, and the
remaining distance is not obviously reachable by more of the same.

### Frozen trunk (P4) is a clean negative, and it bears on §5.1

Adapting only the 260-parameter movement head **hurts on both axes**: retention
0.043 (below the 0.044 reference) and current-env 0.399 against 0.753. The
trunk is where this task's competence lives; a head that small cannot express
a new environment's goal, so confining plasticity there costs plasticity
without buying stability.

That is a direct, if partial, argument about OML (§5.1). OML's mechanism is
head-only online adaptation over a *meta-learned* trunk. This result says the
head-only half, over a **normally-pretrained** trunk, is actively harmful — so
any benefit OML delivers here would have to come entirely from the
meta-learning changing what the trunk represents, not from the restriction
itself. That raises the bar for §5.1 considerably and is worth knowing before
building it.

### DER++ and CLEAR were under-tuned by us, in the way the plan warns about

DER++ returned **bit-identical statistics at α = 0.1, 0.5 and 1.0**, and one of
them matched plain ER at `rb=1` exactly. `aux_loss` fires and scales with α, so
the wiring is fine. Measuring the term against the BC loss at the real
configuration (`move_loss ≈ 17.8`) explains it:

| method | coefficient | aux / move |
|---|---|---|
| DER++ | α=0.1 | **0.003** |
| DER++ | α=1.0 | **0.029** |
| DER++ | α=100 | 2.9 |
| CLEAR | cc=1.0 | 0.40 |
| LwF | α=1.0 | 0.11 |

**The whole DER++ sweep sat below 3 % of the loss.** The method was effectively
off across its entire range, and reporting 0.143 as "DER++'s result" would have
been reporting a strawman.

The cause is a units mismatch. Buzzega's α=0.5 is calibrated against a
cross-entropy over CIFAR logits; here the primary loss is a Gaussian NLL of
magnitude ~18, so the same constant buys a thirtieth of the influence. A
coefficient copied from a paper only means anything alongside that paper's loss
scale.

Both sweeps were also **monotone-increasing to their top value** — the
signature of a range that stops before the method starts working. The plan
makes exactly this argument about EWC's λ, and we then made the mistake
ourselves on two other methods.

Wave 2b (slurm 21633232) re-runs DER++ at α ∈ {10, 100, 1000} and CLEAR at
cc ∈ {3, 10, 30}, with ranges chosen by **ratio to the primary loss** (~0.03 to
~10) rather than by citation. SI is included at λ=1000 for completeness,
although its Wave-2 sweep already turned over (λ=10 → 0.074, λ=100 → 0.066), so
its peak is genuinely inside the range that ran.

---

## 2026-08-31 — a confound in the §5.2 design, named while it is still cheap

Watching the in-context pretraining, the two arms diverge on the *ordinary*
episodic eval, not just the in-context one:

| updates | lifetime arm | episodic arm |
|---|---|---|
| 1 | 0.037 | 0.043 |
| 250 | 0.158 | 0.137 |
| 500 | **0.373** | — |

The lifetime arm is simply the better policy, and there is a mechanical reason
that has nothing to do with memory: **the arms do not receive equal
supervision.** A lifetime rollout teleports its reachers and keeps collecting,
so more of its 200 steps stay unmasked; an episodic rollout freezes a reacher
and masks out the remainder. The lifetime arm therefore gets more gradient
signal per update. The suite's own test `test_the_two_modes_actually_differ`
asserts exactly this — it was written as a guard that the flag was wired up,
and it is also a statement of the confound.

**Consequence: the arms are comparable on the adaptation *slope*, not on the
absolute level.** `adaptation` (last episode minus first) is scale-relative and
is insulated from how good each policy is overall; the raw success rates are
not. Both module docstrings now say so, so nobody reads the wrong number off
the figure later.

The clean fix, if this measurement turns out to be load-bearing, is to match
**total supervised steps** rather than updates between the arms. That is a
different job specification, not a code change, and it is not worth spending
before seeing whether the slope shows anything at all.

---

## 2026-08-31 — Wave 2b, and DER++ turns out to have been a no-op

### CLEAR, with a range that reaches

The corrected sweep found its shape immediately:

| clone_coef | retained | current env |
|---|---|---|
| 10 | 0.326 ± 0.019 | **0.250** ← trap |
| 30 | 0.290 ± 0.012 | **0.095** ← trap |
| 3 | 0.242 ± 0.029 | 0.424 |
| **1.0** | **0.201 ± 0.036** | **0.622** |
| 0.1 | 0.165 ± 0.029 | 0.686 |

The usable peak is around `cc=1`–`3`; above that CLEAR walks straight into the
plasticity trap. Wave 2's range stopped one step short of the turnover, so the
correction was worth making even though the usable answer barely moved.

SI at λ=1000 also improves on its Wave-2 peak (0.125 against 0.074 at λ=10) at
a current-env of 0.665, so its sweep had *also* stopped early — less badly than
DER++'s, and enough to matter.

### DER++ was never running

Wave 2b returned DER++ **bit-identical across α ∈ {10, 100, 1000}**, exactly as
Wave 2 had across {0.1, 0.5, 1.0}. Four orders of magnitude with no effect is
not a coefficient-scale problem, and it was not one.

`_dist_params` detached — and it was used for **both** roles: storing the
anchor at insertion time, where detaching is correct, and computing the live
prediction inside `aux_loss`, where it is fatal. The loss came out nonzero,
scaled correctly with α, and carried `requires_grad=False`. It added a
**constant** to the objective and contributed nothing to any gradient. DER++
ran as plain ER for two entire waves.

```
DER++  value=9.20e-01  requires_grad=False  grad_sum=None     <- before
DER++  value=1.19e+00  requires_grad=True   grad_sum=21.8     <- after
CLEAR  value=5.44e-01  requires_grad=True   grad_sum=8.97
LwF    value=1.43e-01  requires_grad=True   grad_sum=1.75
```

**Every value-based test passed the whole time.** `test_derpp_error_grows_as_
the_model_moves` asserted the loss was nonzero and grew as the model drifted —
both true, and both irrelevant. The one thing it did not check was whether the
number was attached to anything. I had written exactly that test for EWC
(`test_ewc_penalty_is_differentiable_wrt_the_parameters`) and never the
equivalent for the three distillation losses.

Fixed by splitting the helper into `_stored_params` (detached, for the anchor)
and `_live_params` (attached, for the prediction), with the docstring saying
which is which and why. New tests: `test_aux_loss_is_differentiable` over all
three methods, `test_derpp_stored_target_is_detached_but_the_prediction_is_not`
pinning the two roles apart, and `test_derpp_gradient_scales_with_alpha`
asserting that α reaches the *gradient* rather than only the reported value.

The 48 stale DER++ histories are deleted rather than left to be averaged in —
they describe a method that was not running. Wave 2c (slurm 21634287) re-runs
α ∈ {0.1, 1, 10, 100}.

**The pattern across all six defects so far is the same.** None of them crash.
The prev-action channel produced a shape error only when a never-used flag was
switched on; `WorldSpec.write` raced only at scale; the joint-ceiling verdict
was a plausible number from an unconverged run; the budget-blind key averaged
two incompatible runs; and DER++ reported a healthy-looking loss that did
nothing. Every one of them renders.

---

## 2026-08-31 — §5.1 (meta-pretraining): analysed, and deliberately not run

Jack flagged §5.1 and §5.2 as important. §5.2 is running. This records why
§5.1 is **not** being built, because "we ran out of time" would be the wrong
reason and is not the actual one.

OML's mechanism is a two-part split: a meta-learned representation network
(the trunk) that is frozen at meta-test, and a small prediction network (the
head) that adapts online. The claim is that meta-training can shape the trunk
so that head-only online SGD stops interfering with itself.

**P4 measured the head-only half directly, and it is actively harmful.**
Adapting only the 260-parameter movement head over a normally-pretrained trunk
gives retention 0.043 — *below* the 0.044 reference — and current-env 0.399
against 0.753. Restricting plasticity to the head costs plasticity without
buying any stability.

That does not refute OML, because OML would change what the trunk represents.
But it does relocate the entire burden: any benefit here would have to come
from meta-learning making the goal linearly decodable from the trunk's hidden
state within a handful of gradient steps on 260 parameters. The goal is never
observed and differs per environment, so that is a strong requirement, and P4
says the readout that would have to carry it is currently nowhere near able to.

**The architecture also will not support the cheap version of the question.**
The obvious next measurement — does retention improve as the adapted subset
grows? — needs intermediate points between "head" and "everything". With a
single GRU layer plus a linear head there are only two: 260 parameters or
73,000. There is no principled middle, so the plasticity-restriction axis
cannot be swept without changing the model, and changing the model would
un-match it from every other run in the suite.

**Decision.** §5.1 stays unbuilt, with this written down rather than left as an
omission. If it is wanted later, the honest prerequisites are: a two-layer
trunk so the restriction axis has intermediate points, and a first-order OML
outer loop over the existing sequential protocol as the inner loop. Both are
tractable; neither is worth doing before someone decides the P4 result is not
already the answer.

## 2026-08-31 — N=20 partial, and a plasticity finding

With 2–3 of 4 seeds in:

| config | N=5 retained | N=20 retained | N=20 current env |
|---|---|---|---|
| ER, replay ×16 | 0.546 | 0.361 | 0.317 |
| ER, replay ×4 | 0.419 | 0.259 | 0.256 |
| online EWC, λ=1e4 | 0.149 | 0.126 | 0.305 |
| naive, tuned | 0.081 | 0.040 | 0.342 |

Everything degrades, the ordering survives, and the ratio between replay and
the naive control *widens* (6.7× at N=5, 9× at N=20).

The part worth flagging is the last column. **Current-env performance collapses
too** — 0.26–0.34 at N=20 against 0.75–0.82 at N=5. At twenty environments the
agent is not merely forgetting the old ones; it is failing to learn the one in
front of it. That is the signature the plan named as the trigger for Family G
(plasticity maintenance — continual backprop, L2-init, shrink-and-perturb),
which was cut from the suite on the grounds that "at N=5 the control is not
plasticity-limited". At N=20 it is. Recorded as the concrete follow-up rather
than acted on now, since the panel is still filling in.

---

## 2026-08-31 — DER++, with a gradient (slurm 21634287, all 32 tasks OK)

| α | retained | current env | |
|---|---|---|---|
| **10** | **0.326 ± 0.037** | 0.546 | usable best |
| 100 | 0.277 ± 0.043 | 0.360 | entering the trap |
| 1 | 0.168 ± 0.030 | 0.701 | |
| 0.1 | 0.164 ± 0.025 | 0.738 | |

**0.143 → 0.326.** The fix more than doubled the method, and DER++ is now the
best non-pure-replay entry in the suite — ahead of CLEAR (0.201 usable), online
EWC (0.149) and SI (0.125). It had been reported as the *second worst* replay
variant while it was silently running as plain ER.

Two things this settles about the earlier confusion. The coefficient range was
never the problem — α=0.1 and α=1 now differ, as they should, and the method
turns over into the plasticity trap by α=100, so the range was fine all along.
And the Wave-2/Wave-2b split into two identical groups was an artifact of the
no-op: with no gradient, the run depended only on whatever else differed
between the two jobs.

### The usable frontier, after all corrections

Best configuration per method with current-env ≥ 0.5:

| method | retained | current env | stored |
|---|---|---|---|
| ER, replay ×32 | **0.579** | 0.796 | 53.6 MB |
| ER, replay ×16 | 0.546 | 0.753 | 53.6 MB |
| ER, replay ×8 | 0.508 | 0.762 | 53.6 MB |
| ER, replay ×4 | 0.419 | 0.768 | 53.6 MB |
| DER++, α=10 | 0.326 | 0.546 | 56.8 MB |
| CLEAR, cc=1 | 0.201 | 0.622 | 53.9 MB |
| online EWC, λ=1e4 | 0.149 | 0.524 | 0.6 MB |
| SI, λ=1000 | 0.125 | 0.665 | 0.6 MB |
| naive, tuned | 0.081 | 0.820 | — |
| frozen trunk | 0.043 | 0.399 | 53.6 MB |
| *joint ceiling* | *0.998* | | |
| *Hopfield store* | *0.994* | 1.000 | *no data* |

Every replay-family method needs tens of megabytes. Every parameter-space
method fits in 0.6 MB and reaches a fifth of what replay does. Nothing reaches
the ceiling, and every row above spends 200 gradient steps and 200 episodes per
environment against the store's 0 and 1.

---

## 2026-08-31 — a review pass on our own code, and two more defects in EWC

Six defects in, all of the same shape — plausible output, no crash — so I read
the methods package looking for the seventh rather than waiting for it. Both
findings are in `OnlineEWC.after_update`, the sampler that decides which of a
block's rollouts the Fisher is estimated on.

**It was unseeded.** It reached for the `random` module, whose global state
neither `torch.manual_seed` nor `np.random.seed` touches. Every other method in
the suite carries its own `np.random.RandomState(seed)`; EWC alone was
irreproducible, silently, and nothing would ever have surfaced it — two
identical commands simply returned different numbers.

**It was not a reservoir.** It drew `j` against the *buffer size* rather than
the number of items seen, giving a constant acceptance of `k/(k+1)` ≈ 0.97.
Correct reservoir sampling needs the acceptance to decay as `k/n`. Mutation-
checked over 400 updates at `k=8`:

```
OLD (buffer-size draw)     mean retained index = 389.7   tail-only would be 396
NEW (count-based)          mean retained index = 209.6   uniform would be 200
```

So the buffer held approximately **the last 32 rollouts of each block**, which
is exactly what its own comment said it avoided: *"a bounded, uniformly-spread
sample of the block rather than the last N updates."* The code and the comment
had disagreed since the method was written.

Whether this changes EWC's numbers is genuinely unclear — the tail of a block
and a uniform sample of it are both defensible answers to "states this task
visited", and the Fisher may not care much. But the run that produced 0.149 was
not the run the code described, and EWC is a headline method. The 48 affected
histories are deleted and Wave 2d (slurm 21634899) re-runs the full λ sweep,
plus SI at λ=1e4 since its range had also stopped short.

Three tests now cover it: reproducibility across seeds, uniformity of the
retained sample (the one the mutation check validates), and that the buffer
stays bounded and resets per block.

**Seven defects, one shape.** Not one of them crashed. The prev-action channel
errored only when a never-used flag was switched on; `WorldSpec.write` raced
only at scale; the joint-ceiling verdict was a plausible number from an
unconverged run; the budget-blind key averaged two incompatible budgets; DER++
reported a healthy loss carrying no gradient; two coefficient sweeps stopped
before their methods started working; and EWC sampled the tail while
documenting the opposite. The failure mode of this codebase is not exceptions.
It is numbers that look right.

---

## 2026-08-31 — the N=20 scaling panel (slurm 21631792, all 20 tasks OK)

Four seeds per configuration, twenty environments, everything else matched to
the headline protocol:

| config | N=5 retained | N=20 retained | N=20 current env | N=20 stored |
|---|---|---|---|---|
| ER, replay ×16 | 0.546 | **0.384 ± 0.023** | 0.323 | 214.4 MB |
| ER, replay ×4 | 0.419 | 0.282 ± 0.034 | 0.402 | 214.4 MB |
| online EWC, λ=1e4 | 0.149 | 0.131 ± 0.009 | 0.329 | 0.6 MB |
| naive, tuned | 0.081 | 0.046 ± 0.012 | 0.287 | — |
| reference | 0.044 | 0.038 | 0.323 | — |

**The ordering survives**, which is the main thing the panel was for. Replay >
regularisation > tuned naive > reference, at both stream lengths.

**The gap widens.** Against the *tuned* control, replay goes from 6.7× at N=5
to 8.3× at N=20. Replay degrades proportionally less than everything else
(0.546 → 0.384 keeps 70 %; the tuned control keeps 57 %).

**Replay's storage grows with the stream and the store's does not.** 53.6 MB at
five environments, **214.4 MB at twenty** — linear in the number of updates,
because an unbounded buffer keeps everything. The Hopfield store's memory is a
fixed `O(D²)` matrix: it does not grow with N at all. At five environments that
distinction is a footnote; by twenty it is a factor of four and still climbing.
This is the axis on which the two approaches diverge fastest, and it is only
visible in a scaling panel.

**And plasticity collapses for everyone.** Current-env performance is 0.29–0.40
at N=20 against 0.75–0.82 at N=5, across every arm including the reference. At
twenty environments the agent is not merely forgetting the old environments; it
is failing to learn the one in front of it.

That is exactly the trigger the plan named for **Family G** (continual
backprop, L2-init, shrink-and-perturb), cut from the suite on the explicit
grounds that "at N=5 the control is not plasticity-limited (`reached ≈ 0.99` on
the current env)". At N=5 that was true and the cut was right. At N=20 it is
false, and the plasticity-maintenance family becomes a live part of the suite
rather than a deferred one. Recorded as the concrete next wave.

---

## 2026-08-31 — EWC and SI, re-run on a uniform Fisher sample (slurm 21634899)

All 56 tasks OK. The sampler fix moved EWC, modestly but in the right
direction on both axes:

| λ | retained (before → after) | current env (before → after) |
|---|---|---|
| 1e5 | 0.274 → 0.286 | 0.296 → 0.332 *(still degenerate)* |
| **1e4** | **0.149 → 0.168** | **0.524 → 0.598** |
| 1e3 | 0.091 → 0.098 | 0.695 → 0.674 |
| 1e2 | 0.095 → 0.094 | 0.762 → 0.717 |

Estimating the Fisher on a uniform sample of the block rather than its tail
gains about 13 % relative retention at the usable setting **and** raises
plasticity from 0.524 to 0.598 — better on both, which is the direction a
better importance estimate should move things. The effect is not large, but the
run now matches what the code documents, which was the point.

SI's extended range confirms its shape: λ=1e4 posts 0.185 at current-env 0.287
(the trap again), so its usable best stays λ=1e3 at 0.125.

### The suite, complete and all-corrected

Best configuration per method with current-env ≥ 0.5, against a joint ceiling
of **0.998** and an oracle of **1.000**:

| method | retained | current env | stored | needs |
|---|---|---|---|---|
| *Hopfield store* | *0.994* | *1.000* | *no data* | *nothing* |
| ER, replay ×32 | **0.579 ± 0.039** | 0.796 | 53.6 MB | nothing |
| ER, replay ×16 | 0.546 ± 0.027 | 0.753 | 53.6 MB | nothing |
| ER, replay ×8 | 0.508 ± 0.033 | 0.762 | 53.6 MB | nothing |
| ER, replay ×4 | 0.419 ± 0.044 | 0.768 | 53.6 MB | nothing |
| DER++, α=10 | 0.326 ± 0.037 | 0.546 | 56.8 MB | nothing |
| CLEAR, cc=1 | 0.201 ± 0.036 | 0.622 | 53.9 MB | nothing |
| online EWC, λ=1e4 | 0.168 ± 0.033 | 0.598 | 0.6 MB | boundaries |
| SI, λ=1e3 | 0.125 ± 0.030 | 0.665 | 0.6 MB | boundaries |
| naive, tuned | 0.081 ± 0.026 | 0.820 | — | nothing |
| LwF, α=1 | 0.058 ± 0.016 | 0.665 | 0.3 MB | boundaries |
| frozen trunk + ER | 0.043 ± 0.010 | 0.399 | 53.6 MB | nothing |
| reference (no method) | 0.044 ± 0.013 | 0.753 | — | nothing |

And the configurations that score well only by declining to learn — reported
separately, because a leaderboard on retention alone would rank three of them
above every honest replay result:

| config | retained | current env |
|---|---|---|
| CLEAR, cc=10 | 0.326 | 0.250 |
| CLEAR, cc=30 | 0.290 | 0.095 |
| online EWC, λ=1e5 | 0.286 | 0.332 |
| DER++, α=100 | 0.277 | 0.360 |
| LwF, α=10 | 0.235 | 0.238 |
| SI, λ=1e4 | 0.185 | 0.287 |

Every method in the suite has one. That is not a quirk of any single algorithm;
it is what happens when a stability knob is turned far enough, and it is why
the plasticity column is not optional.

---

## 2026-08-31 — §5.2 answered: activation memory does not do this job

Three seeds, eight held-out environments each, 64 lifetimes of 10 episodes,
weights frozen throughout.

| | memory_lift | P(next \| hit) | P(next \| miss) | curve |
|---|---|---|---|---|
| **lifetime** (state carried) | **+0.029 ± 0.018** | 0.117 | 0.088 | 0.100 → 0.086 |
| episodic control | −0.003 ± 0.012 | 0.109 | 0.112 | 0.115 → 0.104 |
| *positive control* | *+0.559* | *0.907* | *0.348* | *0.31 → 0.82* |

**+0.029 against a detectable +0.559 — about 5 % of the available signal, and
within 1.6 SEM of zero at three seeds.** Attributable to carrying state:
+0.032. Both real curves are flat across ten episodes.

The conditional framing is what makes this readable. These policies solve only
~10 % of first episodes on held-out environments, so a flat *mean* curve would
have been ambiguous between "no memory" and "rarely had anything to remember".
`memory_lift` asks the sharper question — *among the lifetimes that did find the
goal, was the next episode any easier?* — and the answer is essentially no:
11.7 % against 8.8 %, where an agent that actually remembered would post 90.7 %
against 34.8 %.

**This is the result §0.2 named as the one that would genuinely hurt, and it
did not happen.** A frozen recurrent policy, pretrained across 32 environments
with its hidden state carried across episode boundaries, does not get better at
a new environment by having been in it. The comparison that runs on the store's
own terms — no weight updates for either model — is the one the store wins most
clearly.

### What this does and does not license

It licenses: *on this task, activation memory in a 256-unit GRU does not
substitute for an associative store.* A referee cannot answer that with "you
needed a bigger buffer", because there is no buffer on either side.

It does not license: *recurrent policies cannot do in-context navigation.* The
policies here are weak in absolute terms (~10 % on held-out environments), the
pool was 32 environments, and the hidden state is 256 units. A stronger
in-context learner — more pretraining environments, a larger recurrent state,
an architecture built for it — is not ruled out by this and would be the honest
next control if anyone doubts the result.

The supervision confound recorded earlier turned out not to matter for the
headline: the lifetime arm is the better policy on the *training pool* (0.494
against 0.268 by update 750) and the *worse* one on held-out environments
(curve 0.100 against 0.115). It generalises worse, not better. Since
`memory_lift` is conditional and within-arm, neither fact touches it.

---

## 2026-08-31 — done

Every measurement in the plan that was scoped for this pass has run. The
results page is generated from `results.json` by
`analysis/continual/results_page.py`, so it cannot drift from the runs behind
it:

    python -m analysis.continual.results_data --wave0_dir ... --out results.json
    python -m analysis.continual.results_page --data results.json --out page.html
    python -m analysis.continual.validate_page page.html

Published at https://claude.ai/code/artifact/f3a7987b-cd1f-476a-b2f2-6780191a2511

### Jobs

| job | what | outcome |
|---|---|---|
| 21626914 | Wave 0 (T0.1 / T0.3 / T0.4) | 64/64 OK |
| 21627945 | corrected joint ceiling | ceiling **0.998**, converged |
| 21628688 | Wave 1 (Tier-1, ER, EWC) | 272/272 OK |
| 21631698 | Wave 2 (CLEAR, DER++, SI, LwF, frozen, high-ratio ER) | 144/144 OK |
| 21631792 | N=20 scaling panel | 20/20 OK |
| 21633232 | Wave 2b (coefficient ranges) | 56/56 OK |
| 21634287 | Wave 2c (DER++ after the gradient fix) | 32/32 OK |
| 21634899 | Wave 2d (EWC/SI after the sampler fix) | 56/56 OK |
| 21629579 | §5.2 in-context | 6 pretrainings + 3 evals OK |
| 21643814 | §5.2 re-scored with `memory_lift` | 3/3 OK |

### The eight defects, and what they have in common

1. `input_prev_action` had never worked — the channel was absent at `t=0`, so
   the flag errored on the first forward and every recorded history had it off.
2. `WorldSpec.write` raced itself through a fixed temp name; 246 of 272 runs
   died at scale.
3. A capacity verdict issued from an unconverged run — the summary called
   ~0.5 a ceiling while every curve was still climbing.
4. The joint-ceiling reader keyed without the training budget and averaged a
   1000-update run with an 8000-update one into 0.719, a number describing
   neither.
5. A negative end-slope was labelled "converged" when it means diverging.
6. DER++'s distillation term carried `requires_grad=False` — a healthy-looking
   loss contributing no gradient, so the method ran as plain ER for two waves.
7. Two coefficient sweeps (DER++, CLEAR) stopped before their methods began
   working, chosen from papers with a different loss scale.
8. EWC's block sampler was unseeded *and* not a reservoir, holding the tail of
   each block while documenting the opposite.

**Not one of them raised an exception in normal operation.** Every one produced
a plausible number. Three were caught by a test written to be falsifiable
rather than confirmatory, three by inspecting a table before publishing it, and
two by reading code that had no failing symptom at all. The lesson this
codebase keeps teaching is that its failure mode is not crashes — it is output
that looks right, and the only defences that work against that are a positive
control, a mutation check, and reading the number before believing it.

---

## 2026-08-31 — the joint sweep finishes, and the verdict logic was wrong

The corrected T0.1 job completed with **2 of 24 tasks failed**, both
`h512, lr=3e-3`:

```
ValueError: Expected parameter loc ... to satisfy the constraint Real(),
but found invalid values: tensor([[[nan, nan]]])
```

That is the lr=3e-3 instability the sweep had already flagged, taken to its
conclusion: at hidden=128 that learning rate merely degrades (0.19, slope
−0.083); at hidden=512 it diverges to NaN and the run dies. A result, not a
defect — and worth stating, because "2 tasks failed" in a log otherwise full of
clean runs invites the assumption that something broke.

The finished sweep also shows the larger models turning over: `h512, l2,
lr=1e-3` reaches 0.842 at 8000 updates with slope **−0.040**, degrading.
The converged ceiling belongs to the *smallest* configurations.

### The verdict logic required every row to converge

`wave0_summary` printed **INCONCLUSIVE** with a converged 0.998 sitting in its
own table. The check was `all(abs(slope) <= 0.02 for every row)`, so the
earlier 1000-update runs — which are still rising, and are supposed to be —
vetoed a verdict they have no standing to veto. A lower bound cannot invalidate
a measurement above it.

Now the verdict asks whether a *converged* row establishes a ceiling, reports
which configuration it came from, and says how many rows have not settled:

```
oracle 1.000 | joint ceiling 0.998 (converged: hidden=128, layers=2,
                                    lr=0.001, 8000 updates) | floor 0.046
(13 of 16 configurations had not settled; those are lower bounds, not ceilings)
-> The network CAN hold all envs at once. The retention gap is genuinely
   forgetting, and Tier 2 is interpretable. Headroom: 0.952.
```

The results page had this right all along — it filters converged rows
individually — so the published numbers never carried the error. Only the
command-line summary did, which is the sort of split that puts two different
answers in front of two different readers.

**That is nine defects, and the ninth is the same shape as the other eight:**
no crash, a plausible-looking output, and a conclusion that would have been
reported with confidence.

---

## 2026-08-31 — Wave 3 never ran, and the page said "nine methods"

Jack asked where HNET went. It went nowhere: **Wave 3 was never built or
run.** There is no `hypernet.py`, no `isolate.py`, no `run_wave3.sh`, and
`CONTINUAL_METHODS` holds seven entries — `none` plus six methods.

Missing, all of it structural:

- **HNET**, the plan's own designated headline competitor (§4.3: *"the most
  important single competitor in this document"*)
- **Multi-head with oracle and inferred task ID**, which bounds the entire
  parameter-isolation family in one run
- **XdG**, and **GPM** as the optional stretch

And the results page claimed **"across nine methods"** in its closing claim
statement. Six continual-learning methods ran — ER, online EWC, CLEAR, DER++,
SI, LwF. The nine came from counting *arms* in the `MECHANISM` table, which
also includes the naive control, its batch variant, and the frozen-trunk
configuration. A count that happened to match the plan's number for a
completely different reason, which is why nothing flagged it.

### Why it slipped

Not a decision — there is no entry anywhere saying Wave 3 was dropped. Wave 2
finished, and then three corrections landed in a row: a coefficient range taken
from a paper with a different loss scale, a distillation term carrying no
gradient, and an importance sampler that was not a reservoir. **Each correction
re-ran an arm that already existed.** They consumed the slot Wave 3 would have
occupied, and the suite was declared complete when the corrections finished
rather than when the plan was finished.

Re-running existing work displaced new work and nothing noticed — including the
wave-gating discipline in the plan, which specifies exit criteria for each wave
but has nothing to say about a wave that is simply never entered. Every other
defect in this log was caught by a check; this one was caught by a reader
asking where a method went.

### What the claim is bounded by

The suite covers **replay and parameter-regularisation**. It says nothing about
methods that allocate separate parameters per task, and HNET is the one with
the strongest prior reason to do well here — Ehret et al. benchmarked online
EWC, SI, masking, generative replay and coresets across four *recurrent*
benchmarks and found hypernetworks beat weight-importance methods
consistently. This is a recurrent policy.

The page now carries a **§10 "What has not run"** section naming all three
methods, why each matters, and this explanation. The closing claim reads "six
methods" and gained a bullet stating the boundary explicitly.

---

## 2026-08-31 — Wave 3 built and launched

Jack said to build it. Three architectures, one method, 144 runs
(job **21653228**), and two measurements taken *before* the sweep rather than
after it.

### What was built

`--arch` is now a second axis beside `--method`, because a hypernetwork with no
regulariser and a plain RNN with one are different runs and a table keyed on
the method alone would file them under the same name.

| arch | mechanism | trainable params |
|---|---|---|
| `rnn` | the baseline; one shared network | 73,220 |
| `hnet` + `--hnet_base learned` | generator output added to a free base vector warm-started from the checkpoint | 146,204 |
| `hnet` + `--hnet_base frozen` | base pinned at the checkpoint; only the task-conditioned part moves | 72,984 |
| `hnet` + `--hnet_base none` | pure von Oswald, from scratch | 72,984 |
| `multihead` | shared trunk, one movement head per task | 73,480 |
| `xdg` | fixed random subset of hidden units per task, masked inside the recurrence | 73,220 |

Plus `--method hnet`, the output regulariser: it pins what the generator
*emits* for past tasks rather than where the generator's own parameters sit.
That is the interesting difference from EWC in a recurrent policy, where a
small weight change compounds over 200 timesteps. Its memory is one generator
snapshot — 0.58 MB with a learned base, 0.29 MB with a frozen one, fixed in the
number of tasks — against unbounded replay's ~50 MB.

The forward pass runs the generated weights through a *template* `RNNAgent` via
`torch.func.functional_call`, so the policy is the baseline's own code rather
than a second implementation of it. A hand-written functional GRU would be a
second copy of the thing the control is supposed to share with the baseline,
and any drift between them would surface as a method effect. A test pins the
equivalence directly: with the generator's output zeroed and the base set to an
`RNNAgent`'s weights, the two produce identical distributions and hidden states.

### The two things measured first

**The beta range came from measurement, not from the paper.** `calibrate_beta`
runs the real protocol and prints the penalty beside the BC loss:

| beta | BC loss | penalty | ratio |
|---|---|---|---|
| 0.01 | 7.75 | 1.3e-05 | 1.6e-06 |
| **1** *(von Oswald's value)* | 7.65 | 0.0013 | **1.7e-04** |
| 100 | 7.83 | 0.082 | 1.0e-02 |
| 10,000 | 7.69 | 1.23 | 1.6e-01 |
| 1,000,000 | 10.83 | 4.04 | 3.7e-01 |

The obvious sweep — decades around the published value — would have had **every
arm contribute under 1% of the objective**, and the conclusion would have been
"the regulariser does not help here". That is the third time this suite has
come to that exact edge: DER++ and CLEAR each cost a re-run for it. The wave
sweeps 1e2–1e7 instead, and the BC loss rising at 1e6 is the plasticity cost
becoming visible, which is where the useful range ends.

**The oracle task id is a real advantage, and now there is a number for it.**
Every Wave 3 arm is told which env it is in; Waves 1 and 2 are not, and the
Hopfield store is not. The plan (§4.3) asks for multi-head in an *inferred*
condition too, so the gap between them measures how much of the problem is task
inference. Rather than plumb a classifier through the protocol,
`task_identifiability` measures what that gap would report: fit observations to
env index, sweeping linear and MLP readouts over windows of 1–64 observations,
split **by trajectory** so correlated neighbouring frames cannot land on both
sides.

Best over every window and readout: **0.426** against a chance of 0.200
(5 envs, 1280 random-walk trajectories). The environments are barely
identifiable from what the agent sees. So the oracle task id is worth a great
deal here, and every Wave 3 arm is an upper bound on its family rather than a
peer of the boundary-free methods — the tables and the page say so in a column,
not a footnote.

The first version of this measurement was wrong in a way worth recording: it
split shuffled *frames* rather than trajectories, so neighbouring timesteps of
one random walk sat on both sides of the split, and it tried only a linear
readout on a single observation. It returned 0.266 — a low number from the
weakest classifier available, which is the least informative outcome there is.
The rewrite sweeps up to an MLP over 64-step windows before concluding.

### Two guards, because both failures would have been silent

A task-conditioned policy asked to act before `set_task` **raises** rather than
defaulting to task 0. Defaulting would evaluate all five envs under one task's
parameters and produce a curve indistinguishable from catastrophic forgetting —
with nothing downstream able to tell the difference.

And the driver **refuses** to fold replayed batches into a task-conditioned
update. One BC update is one forward pass under one task's parameters, so
replayed trajectories from other blocks would be trained through this block's
head, destroying exactly the isolation the agent exists to provide while still
producing a plausible-looking result. The multi-head + ER arm was dropped for
this reason rather than run and quietly misinterpreted.

51 tests, 1086 in the suite, all passing. The load-bearing ones assert
gradients rather than loss values — the hnet penalty is checked to *move the
generator*, and to restrain past-task weights monotonically in beta, which is
precisely what DER++ failed while passing every value-based test.
