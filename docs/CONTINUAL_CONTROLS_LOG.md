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
