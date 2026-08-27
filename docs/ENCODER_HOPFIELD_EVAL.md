# Evaluating an encoder through its Hopfield readout

Status: **implemented**, 2026-08-27. This document fixes what gets measured,
on what code path, with what conventions, and what the numbers are allowed to
be compared against. The code is `analysis/hopfield_probe/` (§8) — `analysis/`
because it needs both `encoder_training` and `hopfield_nav.world`, and
`encoder_training` cannot import upward without a cycle
(`hopfield_nav/encoder_io.py` already imports `encoder_training.models`).

    ./analysis/hopfield_probe/run_probe.sh          # Sec 9 defaults
    SCALE=fast ./analysis/hopfield_probe/run_probe.sh

Tests are `hopfield_nav/tests/test_hopfield_probe.py` — under `hopfield_nav`
rather than beside the package because `pyproject.toml` points pytest at
`hopfield_nav/tests` only, and a test the gate does not run is not a gate.

**Three things the implementation changed, and why.** They are corrections to
this document, not deviations from it.

1. **Test D's goal is absorbing.** Modelling arrival as the goal being a fixed
   point of the field is wrong: the goal cell's own near-zero `q` classifies to
   some cardinal and steps the agent straight back off it, so nothing is ever
   terminal there and `reach_rate` collapses to ~0.03 no matter how good the
   field is. An agent that arrives stops. See §7.
2. **`wrap_to_pi` is `[-pi, pi)`**, not the `(-pi, pi]` §4 and §5 claimed. Only
   the antipodal case can tell them apart and every aggregate takes `abs`, but
   a sign-of-error plot would notice.
3. **`--fwhm_fallback`, not `--fwhm_override`,** is the flag for
   `untrained_mlp.pt` (§1.1). A fallback fills in only where the checkpoint
   carries nothing, so one flag can cover a batch that mixes it with real
   encoders without silently masking their stored values.

---

## 0. Why this exists, and what it is not

We currently score encoders two ways, and neither answers the question the
navigation stack actually asks of them.

| existing metric | what it measures | why it is not enough |
|---|---|---|
| `encoder_training.unique_radius` | worst-case cosine-decay radius of the similarity map | pure geometry of `encoded_Phi`. No Hopfield, no interference between stored patterns, no `q`. |
| `hopfield_nav.eval_all` nav_det / disc / expl | end-to-end task success of a trained policy | confounds encoder, Hopfield, projection, and a learned RNN. An encoder change moves it for reasons you cannot attribute. |

What the stack actually needs from an encoder is a **direction field**. The
agent never sees a recalled memory; it sees `q`, a 2-D (East, North) vector
built by recalling from the current embedding, subtracting the current
embedding, and projecting the difference onto a local scaffold basis
(`hopfield_nav/rollout/signal.py:46`). If `q` points at the goal, navigation is
tractable regardless of policy; if it does not, no policy can fix it.

So the object under test is the map

```
(grid position p, memory contents M, recall steps s)  ->  q(p; M, s) in R^2
```

and the four tests below take it apart in the order the errors compound:

- **Test A** — is the stored goal a fixed point of the Hopfield dynamics at
  all, and over what real-space disc does a cue relax to it?
- **Test B** — at every *grid* cell, how far off is `q`'s angle from the true
  bearing to the goal?
- **Test C** — the same question at *continuous* positions, which reach
  `encoded_Phi` only through the env's `round()` snap. Test B is the floor;
  C − B is the cost of the snap alone.
- **Test D** — follow the field. Angle error is local; whether the trajectories
  it induces actually arrive is global, and B and C cannot see it.

Test C is deliberately not folded into B. The point of running both is to
avoid quoting one number that mixes associative-memory error with
quantisation error, which behave differently — the first is roughly
distance-independent, the second blows up as you approach the goal.

**Every test is run at several memory loads** (number of stored patterns) and
**at several recall-step counts**, and **averaged over many env samplings**. A
single env is a single wall seed, goal cell and scaffold offset, and each of
those three moves the answer.

---

## 1. The code path under test, exactly

This is the path a real rollout takes. The harness must call these functions,
not reimplement them, or the numbers describe a different system.

**1.1 Embedding of a grid cell.** Local env coords → global scaffold coords →
smoothed grid code → encoder → L2-normalised embedding.

```
gx, gy = local + env_offset                     # scaffold.py:311 get_encoded_state
g      = smooth_gbook(gbook, lambdas, fwhm)[:, gx, gy]   # gridcode/smoothing.py:64
z      = encoder(g)                             # models.py:41, output is F.normalize'd
```

`GridEncoder.forward` applies `gain` **before** the output nonlinearity and
L2-normalises after, so every embedding is a unit vector and `gain` controls
saturation, not scale. Both cues and stored patterns are therefore on the unit
sphere; there is no cue/memory scale mismatch to worry about.

**`fwhm_ratio` and `gain` are properties of the encoder, not of this harness.**
Both are inherited from the checkpoint and neither is ever passed on the
command line. `gain` already resolves this way in production
(`encoder_io.load_encoder`: top-level `ckpt["gain"]` wins, then `cfg.gain`).
`fwhm_ratio` does **not**: it lives in `ckpt["train_config"]["fwhm_ratio"]`,
which `EncoderModelConfig` filters out, and `train_navigate.py:972` supplies it
from a CLI default of 0.25 instead.

> **Landmine, verified 2026-08-27.** `encoder_io.validate_config` takes
> `encoder_gain` and `fwhm_ratio` as arguments and **checks neither** — its
> body compares `lambdas` and nothing else. So an encoder trained at one
> smoothing width can be evaluated at another with no error and no warning,
> and the resulting embeddings are simply not the ones the encoder was fitted
> to. This harness must read `fwhm_ratio` from the checkpoint and **hard-error
> if it is absent** rather than falling back to 0.25. `untrained_mlp.pt` has no
> `train_config` at all, so it is the one checkpoint that needs an explicit
> override — passed loudly, recorded in the result header, never defaulted.

**1.2 Storage.** `Hopfield.input_memory` (`hopfield/core.py:43`) normalises the
pattern, does `W += (1/D) z zᵀ`, then **zeroes the diagonal**. With `K`
patterns, `W = (1/D)(Σ_k z_k z_kᵀ − diag(Σ_k z_k ⊙ z_k))`.

**1.3 Recall.** `x_{t+1} = normalize((1−α)x_t + α·tanh(β·W x_t))`
(`hopfield/core.py:59`), run for `steps` iterations. Production defaults:
`α = 1.0`, `steps = 1`, `β = encoder_gain` (`train_navigate.py:488`).

> **This is a linear readout, not attractor dynamics, at the current operating
> point.** `‖W x‖ ~ K/D`, so at `D = 1024` the argument of `tanh` is order
> `β·10⁻³`. At `steps = 1` recall is `normalize(Σ_k z_k ⟨z_k, x⟩ − diag-term)`
> — a soft nearest-neighbour readout. As `steps → ∞` normalised linear
> iteration is power iteration, which converges to the **same** top eigenvector
> from every cue, so distinct per-goal attractors cannot exist in that limit
> for `K > 1`. Test A is written to measure this rather than assume it: `steps`
> is a first-class axis, and a "no fixed points beyond the top eigenvector"
> result is a *finding*, not a bug in the harness.

**1.4 Multi-step `q` is a production input, not a diagnostic.**
`multistep_q` (`rollout/signal.py:178`) projects the recall *trajectory* at
each requested iteration count and hands each one to the policy as a separate
2-D channel; `run_repro_v35.sh` sets `INPUT_HOPFIELD_MULTISTEP="1 2 3"`. So
"how accurate is `q` at step `s`" is a question about a channel the agent
actually consumes, and Tests B and C sweep `s` for that reason. Note the
production path computes the whole trajectory against a **single cached basis**
`W` — the basis does not move with the recall — which is what the harness must
reproduce.

**1.5 Local basis.** `VectorHash.gram_schmidt_projection`
(`scaffold.py:366`) reads two neighbour displacements at the *global* cell,
clipped to `[1, Npos−2]`:

```
d_fwd = Phi[gx,   gy+1] - Phi[gx, gy]     # North, +y
d_rgt = Phi[gx+1, gy  ] - Phi[gx, gy]     # East,  +x
W     = gram_schmidt_2d_batch(d_fwd, d_rgt)   # utils.py:11 -> (2, D), row0=East, row1=North
```

Gram–Schmidt is **asymmetric**: `d_fwd` (North) is normalised and kept exactly;
East is whatever is left after projecting North out of `d_rgt`. If the two
displacement directions are not orthogonal in embedding space — and there is no
reason they should be — this bakes in an angular distortion that favours the
North axis. §6.2 is the control that measures it.

**1.6 The signal.** `q = W @ (recalled − current)` (`scaffold.py:403`), giving
`(East, North) = (Δx, Δy)`. Bearing is `atan2(q_north, q_east)`, matching
`classify_direction_batch` (`utils.py:35`). The discrete agent then bins `q` to
a cardinal; the continuous agent normalises it.

**1.7 Continuous positions.** `ContinuousVecEnv` keeps `_pos_f` (float) as the
source of truth and derives `_pos = clip(round(_pos_f), 0, size−1)`
(`vec_env.py:272`). **Embeddings, the local basis, and therefore `q` are all
read at the snapped cell.** The true bearing, however, is from `_pos_f`. Test C
is exactly the size of that gap.

---

## 2. Shared harness

### 2.1 Lazy encoding — do not build `encoded_Phi`

`precompute_encoded_phi` materialises `(Npos, Npos, out_dim)` — **12 GB** at
`Npos = 1716, out_dim = 1024`, and `gbook` itself is `(434, 1716, 1716)`. This
harness never needs the whole field: it needs the cells inside a handful of env
footprints, their N/E neighbours, and a few thousand distractor cells.

The grid code at a position is a pure function of `(x mod λ, y mod λ)` per
module (`gridcode/codebook.py:38`), and smoothing is per-module and
position-local (`smoothing.py:77`), so a lazy

```python
encode_positions(encoder, lambdas, fwhm_ratio, xs, ys) -> (N, D)
```

is exact and costs `O(N)`. It must be **verified bit-identical** against
`precompute_encoded_phi` on a small `Npos` in a unit test, because everything
downstream inherits it.

### 2.2 Env sampling

An env is `(wall seed, local goal cell, scaffold offset)`. Reuse
`hopfield_nav.world.generate` so the envs are drawn from the same process the
training/eval splits use, and record the `EnvSpec` list so any result can be
replayed. Draw `n_envs` of them per configuration; report mean, and the spread
across envs, never a single env's number.

> Per `feedback_eval_point_threshold`: these evaluations swing hard. Any
> directional claim needs the across-env distribution, not just its mean.

### 2.3 Memory loading — what "number of stored goals" means

Three constructions are defensible and they are **not** the same experiment.
The harness supports all three behind one flag. **`multi_env_goals` is the
headline** (decided 2026-08-27); the other two are secondary and run on request.

| mode | contents of `W` | what it models |
|---|---|---|
| **`multi_env_goals`** (headline) | one goal from each of `K` different envs, each at its own scaffold offset | continual / many-rooms capacity: `K` real goals competing, which is the question "how many goals can this encoder hold at once" actually asks |
| `goal+distractors` | this env's goal, plus `K−1` patterns drawn from arbitrary cells **outside** its footprint, via `rollout/distractors.py:23` | production as it is scored today. Matches `n_train_distractors` and `--val_distractors`. |
| `same_env_goals` | `K` goals inside the **same** env footprint | worst-case interference — the patterns are maximally similar |

**What one draw is, under `multi_env_goals`.** A *world* is `K` envs placed in
one scaffold by `place_envs` (`scaffold.py:145`, `placement="spread"`), each
with its own wall seed, local goal and offset. One Hopfield holds all `K` goal
patterns. Then **every one of the `K` envs is scored in turn as the test env**,
against the same `W` — one world yields `K` measurements, not one, which is
what makes `K = 50` affordable. The reported unit is still the env; the world
is the sampling unit for the `W` it sits in.

This makes the `K` axis do two things at once, and the doc has to be honest
about it: raising `K` raises the memory load *and* packs the envs closer
together in the scaffold, because `place_envs` spreads a fixed `Npos` over more
of them. Those are separable and should be separated — hold `n_envs_per_world`
at the largest `K` and store only the first `K` goals, so the *placement* is
identical across the sweep and only the *load* moves. Report the mean
env-to-env offset distance per `K` either way, so a reader can see whether the
two were successfully decoupled.

Under this mode, "distractor" throughout §3–§5 means **another env's goal**,
not an arbitrary off-footprint cell — a strictly harder and more meaningful
confusion, since those are the patterns the system is actually being asked to
keep apart.

Storage order is shuffled per trial (as `metrics.py:320` does), because
`zero_diag` makes `W` order-dependent in a small but real way.

`K = 1` is the degenerate-but-important case: one memory, no interference, so
it isolates the encoder+projection geometry with the Hopfield still in the
loop. It should very nearly reproduce the oracle control of §6.1, and a gap
between them at `K = 1` is a bug in one of the two.

### 2.4 What is held fixed

One scaffold (`lambdas`, `Npos`, `fwhm_ratio`) and one encoder checkpoint per
run; everything else varies. Two encoders are only ever compared at an
identical scaffold, env draw, and RNG seed — the env lottery is larger than
most encoder differences.

### 2.5 Encoder header — copied, not recomputed

An earlier draft proposed measuring an "alias floor" here to calibrate a cosine
threshold `tau`. **That rationale is gone**: §3.2 no longer uses a threshold, so
there is nothing to calibrate, and the harness computes no floor of its own.

What survives is a header block, and it is *copied out of the checkpoint*
rather than measured — the L7 checkpoints already carry a full
`ckpt["unique_radius"]` dict from `encoder_training.eval_unique_radius`:

- `alias_ceiling_max` / `cos_floor_mean` — the highest and the mean cosine
  between far-apart cells. For the L7 encoders these are **0.88 and 0.02**:
  cells are near-orthogonal *on average* while some far pair still scores 0.88.
  Under §3.2's new definition — nearest **cell**, not nearest stored goal —
  that gap is the whole mechanism behind a large `retrieved_dist`, which is why
  it stays in the header even though `tau` is gone.
- `r_min` / `r_median`, `n_refs`, `headline` — so a reader can put this run's
  numbers next to the encoder's coding radius without opening another file.
- `gain`, `fwhm_ratio`, `output_nonlinearity`, `out_dim`, param count.

Checkpoints without a stored `unique_radius` (v35, untrained) leave those
fields null rather than triggering a recomputation. This harness does not
re-derive another module's metric.

Two things it *does* measure once per (encoder, world), because nothing else
records them:

- `diag_frac` — `‖diag(Σ_k z_k z_kᵀ)‖ / ‖Σ_k z_k z_kᵀ‖` before `zero_diag`
  removes it. That term is large exactly when the embedding is sparse, so this
  says how much of the self-recall signal storage throws away.
- `tanh_arg` — the distribution of `β·(W x)` over real cues. This is the
  number §1.3's linearity claim rests on, and it is not the same for every
  encoder: v35 runs at `β = 3.70` and the L7 encoders at `β = 100`, a 27×
  difference that lands them in potentially different regimes of the same
  `tanh`. Measure it; do not assume it.

---

## 3. Test A — are stored goals attractors, and how big is the basin?

### 3.1 Fixed-point test

For each stored pattern `z_k`, run recall from `z_k` itself and report:

- `residual(k) = 1 − cos(recall(z_k), z_k)` — a fixed point has residual 0.
- `sign_flip(k)` — `cos < 0`. `−z` is as much a fixed point as `z` under a
  symmetric `W` with normalisation, and a flipped recall inverts `q`. This must
  be counted, not silently absorbed by taking `|cos|`.
- `top_overlap(k) = argmax_j cos(recall(z_k), z_j)` — did it stay on itself, or
  fall onto another memory?

Swept over `steps ∈ {1, 2, 3, 5, 15}`. The headline is
**`frac_self_consistent(K, steps)`** = fraction of stored patterns whose recall
stays nearest to themselves.

**Collapse diagnostic.** Alongside it, report `mean_pairwise_cos(K, steps)` —
the mean cosine between the recall endpoints of *different* cues. If §1.3 is
right this rises toward 1 as `steps` grows: every cue landing on the same
vector. One number that says "the dynamics have one attractor, not `K`".

### 3.1a Rescue mode — can *any* setting give attractor behaviour?

**Off by default.** Everything else in this document evaluates the encoder at
the production operating point. This mode asks a different question — whether
the Hopfield layer is fixable at all — and its results say nothing about
encoder quality, so it must never be mixed into the headline tables. Enabled
with `--rescue`, run on one encoder and a small `K` grid, reported in its own
file.

The grid, in rough order of expected leverage:

| knob | production | rescue values | why it might matter |
|---|---|---|---|
| `Hopfield.scale` | `1/num_units` | `1`, `1/√D`, `1/K` | **the main suspect.** `scale` is what makes `‖W x‖ ~ 10⁻³` and therefore what makes `tanh` inert. Raising it is the one change that puts the nonlinearity into its working range at all. |
| `beta` | `= gain` (3.7 or 100) | `1, 10, 10², 10³, 10⁴` | the other half of the same product. Sweep jointly with `scale` — only `β·scale` matters, so sweep the product and say so. |
| `zero_diag` | `True` | `False` | the diagonal is the self-reinforcing term; removing it is what stops a stored pattern being a fixed point of the linear map. Cheapest single thing to try. |
| `alpha` | 1.0 | `0.1, 0.3, 0.5` | damped updates. Cannot create attractors the map does not have, but changes whether iteration finds them. |
| `normalize_each` | `True` | `False` | normalisation is what turns the iteration into power iteration. Without it the dynamics can converge on magnitude, not just direction. |
| `steps` | 1 | up to 100 | with the above changed, "converged" may mean something different. |

The success criterion is Test A's own `frac_self_consistent` and
`mean_pairwise_cos` at `steps = 50`: **`frac_self_consistent ≈ 1` with
`mean_pairwise_cos` low** is genuine per-goal attractor behaviour and would be
a real result. High `mean_pairwise_cos` at any setting is the collapse of §1.3
and means the knob did not help.

If a setting does work, it does not automatically become the default: §3.2 and
§4 have to be re-run under it to check the *basin* and the *direction field*
survived, since a memory can be a perfect fixed point with a basin of radius
zero, which would be worse for navigation than what we have.

### 3.2 Real-space basin radius

This is the number the test is actually for. The cue is not a noised goal
vector — it is the embedding of **another cell**, which is what makes the
radius live in real space.

**Retrieval is decided against every cell, not against the stored goals.**
Asking "is the recall nearer the goal than any *other stored goal*" is far too
easy — with `K = 3` that is a 3-way choice. The question that matters is
whether the recall lands nearer the goal cell than **any other cell in the
world**, which is what makes it a position readout rather than a memory index.
So `retrieved` is an argmax over a cell bank, and it returns a *cell*, not a
memory index:

```
x0            = encode(c + offset)
x_final       = recall(x0, steps, beta, alpha)
retrieved(c)  = argmax_{u in CELLBANK} cos(x_final, encode(u))     # a CELL
cos_goal(c)   = cos(x_final, z_goal)                               # reported directly
exact_hit(c)  = (retrieved(c) == goal cell)                        # no tau
retr_dist(c)  = || retrieved(c) - g ||_2      when retrieved is in the test env
d(c)          = || c - g ||_2
```

**`CELLBANK`** is every cell of all `K` env footprints in the world
(`K · size²` = 20 000 cells at `K = 50, size = 20`), plus `n_alias = 20 000`
uniformly-drawn scaffold cells to catch far aliases — the `alias_ceiling_max`
of 0.88 in §2.5 says those exist. It is deliberately *not* all 2.94M scaffold
cells: that is a 400 × 2.94M × 1024 matmul per env and it would dominate the
whole suite's cost for a tail that a random sample already estimates. The bank
is fixed per world, so the cosines are one batched matmul.

`tau` is gone from the primary path. `exact_hit` is a threshold-free predicate
and replaces it everywhere; the `cos_goal` distribution is reported outright so
that anyone who wants a threshold can pick one after the fact.

**`retr_dist` is only a distance when the retrieved cell is in the test env.**
When recall lands in another env's footprint or on an alias cell, real-space
distance to `g` is not a meaningful quantity — those are different rooms. Those
cases are counted in their own categories (`retrieved_env`, `retrieved_alias`),
never folded into a distance average. This is the main place a careless
implementation would manufacture a reassuring number.

Report, per `(K, steps)`:

- **`cos_goal(d)`** — mean and the 10/50/90 percentiles of `cos(x_final,
  z_goal)` against distance-to-goal. Reported as a first-class curve, not as
  an input to something else: it is the continuous quantity underneath every
  binary predicate here, and it degrades smoothly where `exact_hit` cliffs.
- **`exact_hit(d)`** — fraction of cells at distance `d` whose recall retrieves
  the goal cell exactly. This replaces the old `hit(d)`.
- **`retr_dist(d)`** — mean and 90th percentile of how far, in cells, the
  retrieved cell sits from the goal. `exact_hit` says whether it was right;
  this says how wrong it was when it was not, which is the difference between
  a readout that is noisy and one that is lost.
- **`r_exact_all` / `r_exact_95`** — the radii below, computed on `exact_hit`.

  - `r_exact_all` — largest `r` such that **every** cell with `d ≤ r` is an
    exact hit. Stop at the first failing radius, not the last passing one; the
    condition is not monotone in `r`. This mirrors the convention
    `unique_radius` already uses (`encoder_training/unique_radius.py`) so the
    two are readable together.
  - `r_exact_95` — largest `r` such that ≥95% of cells within it are exact
    hits. `r_exact_all` has cliff behaviour and two encoders can both report 0.
- **`r_by_direction`** — `r_exact_all` restricted to each of 8 angular sectors
  from the goal. Anisotropy is a known live issue in this scaffold
  (`EXPERIMENTS_UNIQUE_RADIUS.md` §6.11f), and a radius averaged over
  directions can hide a direction where it is 0.
- **`outcome(d)`** — the full categorical breakdown at each distance, which
  is where a miss actually gets diagnosed. Five mutually exclusive outcomes:

  | outcome | meaning | what it implicates |
  |---|---|---|
  | `exact` | retrieved cell **is** the goal cell | — |
  | `near` | retrieved cell is in the test env, `retr_dist ≤ 2` | readout is noisy but positional |
  | `far_same_env` | in the test env, `retr_dist > 2` | local decay too flat |
  | `other_env` | inside another env's footprint | memory interference between rooms |
  | `alias` | an off-footprint scaffold cell | the `alias_ceiling` tail of §2.5 |

  Splitting `near` from `far_same_env` is the point of having `retr_dist` at
  all: a readout that is consistently one cell off is a usable direction
  signal, and one that lands across the room is not, and both were a single
  "miss" under the old definition.
- **`confusion(j)`** — for `other_env` outcomes, *which* env. Under
  `multi_env_goals` this is a `K × K` matrix per world, and its structure is
  the point: if confusions concentrate on the envs whose scaffold offsets are
  nearest, the failure is scaffold aliasing and more encoder capacity will not
  fix it; if they are uniform, it is memory interference and it will.

**Figures:** see `ENCODER_HOPFIELD_EVAL_VIZ.md` §4 (`test_a.html`).

---

## 4. Test B — `q` accuracy on the grid

For each env, each `K`, each `steps ∈ S`, and **every local cell** `c`
(including cells at distance 0 and 1 from the goal, which are the degenerate
ones):

```
z_c   = encode(c + offset)
Wb    = gram_schmidt_projection([c], offset)         # scaffold.py:366, computed ONCE
traj  = recall_batch_trajectory(z_c, S, beta, alpha) # hopfield/core.py:104
for s in S:
    q_s        = Wb @ (traj[s] - z_c)                # scaffold.py:403
    theta_pred = atan2(q_s.north, q_s.east)
    theta_true = atan2(g_y - c_y, g_x - c_x)
    err(c, s)  = wrap_to_pi(theta_pred - theta_true) # signed, in (-pi, pi]
```

One trajectory call per cell yields every `s`, matching what `multistep_q`
does in production (§1.4) — including the single cached basis.

Reported per cell, and aggregated, **for each `s` separately**:

- **`|err|` mean and median per cell** → the heatmap. **In goal-relative
  coordinates**, not env coordinates: the goal moves from env to env, so a
  `size × size` map indexed by absolute cell has a different goal in every
  sample and averaging it is meaningless. Re-index every cell by its offset
  from that env's goal, `(c − g)`, giving a `(2·size−1)²` = 39×39 grid centred
  on the goal, and accumulate across all envs and worlds into that. The goal
  sits at the centre by construction and the map is directly interpretable as
  "how wrong is `q` when the goal is `Δ` away". One panel per `(K, s)`,
  diverging colormap on signed `err`, sequential on `|err|`, chance (90°)
  pinned on the colourbar.

  Two companions, because the goal-relative view deliberately destroys two
  things worth seeing:
  - **env-absolute map**, aggregated across envs and ignoring goal position.
    Answers a different question — is `q` worse near walls and in corners? —
    which goal-relative coordinates average away.
  - **one single-env panel** at a representative env, in absolute coordinates
    with its actual goal marked, as a sanity anchor. Aggregates hide
    structure; one raw example is what catches a harness bug.
- **`|err|` vs. distance-to-goal** — mean, median, and IQR band, binned in
  1-cell bins. This is the curve that decides whether the field is usable.
- **`acc_45(d)`** — fraction of cells with `|err| < 45°`, i.e. the correct
  cardinal under `classify_direction_batch`. This is the number the *discrete*
  agent actually consumes, and it is not recoverable from mean angle error.
- **`acc_90(d)`** — fraction with `|err| < 90°`, i.e. "moving along `q` reduces
  distance to goal at all". The weakest condition under which greedy following
  makes progress.
- **`‖q‖` vs. distance** — the magnitude channel. `input_hopfield_raw=1` in the
  v35 lineage feeds unnormalised `q` to the policy, so whether `‖q‖` carries
  distance information is a real question about the encoder, not a diagnostic
  afterthought. Note `‖q‖` shrinks with `s` as the recall saturates; report it
  per `s` and do not compare magnitudes across `s` without saying so.
- **`err` by sector** — mean signed error in each of 8 sectors, to expose the
  Gram–Schmidt North bias predicted in §1.5.

**The `steps` summary.** One plot the whole section exists to produce:
`acc_45` (and mean `|err|`) vs. `s`, one line per `K`. It answers whether the
extra multistep channels carry information the `s=1` channel does not — which
is the standing open question about `INPUT_HOPFIELD_MULTISTEP`. Three shapes
are possible and they mean different things: **flat** (the extra channels are
redundant), **falling** (iteration destroys the readout, consistent with §1.3),
**rising then falling** (there is a real optimum and `steps=1` is not it).

Chance level is a uniform bearing: `E|err| = 90°`, `acc_45 = 0.25`,
`acc_90 = 0.5`. Print it on every plot.

**Degenerate cells.** At `c = g`, `q ≈ 0` and `theta_true` is undefined. Report
that cell separately (as `‖q‖` only) and exclude it from angular aggregates;
do not let it silently become a 0° or a NaN.

---

## 5. Test C — `q` accuracy at continuous positions

Identical to Test B except the position is continuous and reaches the encoder
only through the env's snap. Same `steps` sweep.

```
p          = continuous position, float (2,)
c          = clip(round(p), 0, size-1)               # vec_env.py:272 -- the real snap
z_c        = encode(c + offset)                      # NOTE: at the SNAPPED cell
Wb         = gram_schmidt_projection([c], offset)    # also at the snapped cell
q_s        = Wb @ (recall(z_c, s) - z_c)
theta_true = atan2(g_y - p_y, g_x - p_x)             # from the CONTINUOUS position
err(p, s)  = wrap_to_pi(atan2(q_s.north, q_s.east) - theta_true)
```

Sampling: uniform over `[-0.5, size-0.5]²` (the region the env clips to), at a
high rate — proposed `n_samples_per_env = 200_000`, plus a **dense annulus
refinement** at `d < 3` cells where the effect concentrates, since uniform
sampling puts almost no mass there.

> **Cost note.** `q` at a continuous position depends on `p` *only* through
> `snap(p)`, so there are only `size²` distinct `q` values per `(env, K, s)`.
> Compute those once and reuse them; the "high sampling rate" then costs a
> lookup per sample, not a recall. This makes 200k samples/env free and is the
> reason the sampling rate can be pushed as high as we like.

Everything from §4 is reported again, with distance bins **much finer near
zero** (proposed edges: 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, …). Plus the
two quantities that only exist here:

- **`excess(d) = |err|_C(d) − |err|_B(snap(d))`** — the snap-attributable
  error, with the Hopfield contribution differenced out. This is the headline
  of Test C: **how much do continuous positions screw up `q`, over and above
  what the Hopfield already gets wrong** — reported per `s`, since there is no
  reason the two error sources compose the same way at every step count.
- **`err_geom(d)`** — the *analytic* snap error with a perfect readout:
  `angle(g − c) − angle(g − p)`. This is the ceiling on how good Test C could
  possibly be, and it diverges as `p → g`. Plotting `err_geom`, `excess`, and
  `|err|_C` together is what separates "the encoder degrades near the goal"
  from "quantisation degrades near the goal", which are otherwise
  indistinguishable and have opposite fixes.

**Heatmap:** 2-D histogram of mean `|err|` over the continuous plane at
sub-cell resolution (proposed 8 bins/cell), which makes the snap-cell structure
visible directly — the field is piecewise constant within a cell by
construction, and the plot should show that.

**No goal-radius masking.** An earlier draft proposed excluding an L2 ball of
`goal_radius` from the near-goal aggregates. That was wrong and is removed:
`goal_radius` is a reward-shaping knob on the training config that can be
changed between runs, and nothing about an encoder's quality may depend on the
current value of one. The near-goal region is reported in full, in fine bins,
with `n` per bin — anyone comparing against a particular training config can
mask afterwards from the raw arrays.

> **Aside worth deciding on (§9, #19).** The snap error is an artifact of the
> codebook being defined only at integer positions. Grid phase is
> `x mod λ`, which is perfectly well defined for real `x`, and `smooth_gbook`
> already places a Gaussian bump at a phase — so a *continuous-phase* encoding
> is a few lines away and would remove this error class entirely. Measuring it
> here tells you how much that change is worth before building it.

---

## 6. Controls

Without these, none of §3–§5 is attributable.

**6.1 Oracle `q` (no Hopfield).** Replace `recall(z_c)` with `z_goal` exactly —
this is `oracle_signal_at` (`rollout/signal.py:152`). It has no `K` and no
`steps` axis.

> **It is a ceiling, not a ground truth.** The objection is correct and worth
> stating in the doc: `z_goal − z_c` is a *large* displacement in embedding
> space, and `W` is a **local** tangent frame built from two one-cell neighbour
> displacements (§1.5). Projecting a far displacement onto a local frame is a
> first-order approximation on a curved manifold, and it degrades with
> distance for reasons that have nothing to do with the encoder's quality. So
> oracle `q` will look bad at range, and that is not a bug.
>
> It is still the right control, *because* the Hopfield path has exactly the
> same pathology: `recall(z_c)` is approximately `z_goal`, so `recalled − z_c`
> is the same large displacement through the same local frame. The oracle is
> therefore "what `q` would be if recall were perfect, under the projection we
> actually use" — the best achievable, not the truth. `|err|_hopfield −
> |err|_oracle` isolates recall error from projection error precisely because
> the projection error is common to both.

**6.1b Local oracle — is the basis itself sound?** The control §6.1 cannot be,
and the one the objection above actually motivates. Use a **one-cell**
displacement: the neighbour cell `c'` one step along the straight line toward
the goal, `q_local = W @ (encode(c') − encode(c))`, and score it against the
bearing to `c'`. This exercises the Gram–Schmidt frame at exactly the scale it
was constructed at, so it separates two things §6.1 confounds:

- `q_local` accurate, oracle `q` bad at range → the basis is fine and the
  embedding manifold is curved. The readout is intrinsically local, and the
  fix is a policy that takes short steps, not a better encoder.
- `q_local` already bad → the basis is broken at its own scale, and every
  number in §4 and §5 inherits that. This would be the most consequential
  single finding available from the whole suite.

Run `q_local` at every cell, report it as an `|err|` vs. distance curve like
any other, and it costs two encoder lookups per cell.

**Where the controls appear.** In the summary table for every configuration,
and as reference lines on the *aggregate* curves — `|err|` vs. distance, and
`acc_45` vs. `steps`. **Not** on every heatmap and not in every panel: that
would double the figure count for a line that does not vary within a panel.
The viz doc (`ENCODER_HOPFIELD_EVAL_VIZ.md`) fixes exactly where they show up.

**6.2 Gram–Schmidt order swap.** Rerun §4 with `d_rgt` as `e1` instead of
`d_fwd`. A large difference means the reported angles are substantially an
artifact of the basis construction, not of the encoder. Cheap; run once per
encoder, not per `(K, s)`.

**6.3 Empty memory.** `K = 0`. `q` is defined as zero by the production path
(`signal.py:102`), so this is a plumbing check: it must produce exactly the
chance-level line, and any deviation means the harness is leaking the goal.

**6.4 Untrained encoder.** The same suite on
`encoder_training/save_untrained_encoder.py` output. Sets the floor that a
trained encoder must clear, on the same env draw.

**6.5 Linear-`tanh` control.** Rerun §3 and the `steps` sweep of §4 with
`use_tanh=False`. Given §1.3 this should change nothing at the production
operating point; if it does, `beta` is doing more than the
`project_hopfield_is_linear` note assumes and the whole `steps` story needs
revisiting.

---

## 7. Test D — does the field actually flow to the goal?

In scope (decided 2026-08-27). Angle error per cell is a local statistic;
navigation is a global property of the vector field, and §4 cannot see it. A
field with 30° mean error and one sink in the wrong corner is a very different
object from one with 30° mean error and no spurious sinks.

No policy, no encoder calls beyond what §4 already computed: from each starting
cell, repeatedly step along `q̂` for `max_steps = 4·size` and record whether the
trajectory enters the goal's L2 ball. Two variants, because they answer
different questions:

- **discrete** — step to the neighbour picked by `classify_direction_batch`
  (`utils.py:35`), clipped at walls. This is literally what a perfect discrete
  agent would do.
- **continuous** — step `continuous_scale · q̂` from a float position, snapping
  for lookup exactly as §5 does. This is what a perfect *continuous* agent
  would do, and it can stall in ways the discrete one cannot.

Reported per `(K, steps)`:

- **`reach_rate`** overall and vs. start distance.
- **`mean_steps`** to reach, over successes only — and the success count
  beside it, always. Per `project_nav_tri_failure_modes`, `mean_steps` computed
  over a shrinking success set is a trap: a field that only succeeds from
  nearby will post an excellent `mean_steps`.
- **spurious sinks** — cells the field converges to that are not the goal,
  with their basin sizes. A map of these per env is the most directly
  actionable figure in the document.
- **limit cycles** — detected by state repetition (discrete) or by
  non-decreasing distance over a window (continuous).

Because it consumes the `q` field §4 already built, Test D costs essentially
nothing on top and runs by default.

---

## 8. Outputs and layout

```
analysis/hopfield_probe/
    __init__.py
    encode.py        # lazy encode_positions + equivalence test vs precompute_encoded_phi
    harness.py       # env sampling, memory loading, shared config dataclass
    attractor.py     # Test A
    qfield.py        # Tests B and C (one implementation, two position sources)
    controls.py      # Sec 6
    flow.py          # Test D -- greedy flow over the q field built by qfield.py
    report/          # figures + HTML pages. See ENCODER_HOPFIELD_EVAL_VIZ.md
    run.py           # CLI: python -m analysis.hopfield_probe.run --ckpt ...
    tests/
```

**The reporting layer is specified separately**, in
`docs/ENCODER_HOPFIELD_EVAL_VIZ.md`, and depends on this document only through
the result JSON. Nothing in `attractor.py` / `qfield.py` / `flow.py` may import
from `report/`: the tests must be runnable headless on a compute node, and the
pages regenerable from saved results without recomputing anything.

Tests B and C are **one implementation**. C differs only in where positions
come from and what `theta_true` is measured from; forking them guarantees they
drift apart and stop being comparable, which would destroy `excess(d)`.

Results write one JSON per `(encoder, scaffold, K, steps)` with raw per-cell
arrays, plus a top-level summary CSV — so replotting never requires
recomputation, in the shape `analysis/encoder_sweep.py` already reads.

---

## 9. Parameters

Settled 2026-08-27 unless marked **open**. Defaults follow `run_repro_v35.sh`,
which is the live operating point. `$CLS_RUNS` is
`/orcd/pool/003/jackking/cls_runs` (`scripts/cls_env.sh`).

| # | parameter | value | note |
|---|---|---|---|
| 1 | encoders | `$CLS_RUNS/encoders/run_20260422_185816/encoder_best.pt` (v35 lineage); `$CLS_RUNS/sweeps/w53_attract_knee/00{4,5,6,7}_att16_seed=4{2,3,4,5}/encoder_final.pt` (level 7); `$CLS_RUNS/encoders/untrained_mlp.pt` (floor) | level 7 is 4 seeds — run all 4, report the spread, do not pick one |
| 2 | `lambdas` / `Npos` | **inherited** (`[11,12,13]`) / 1716 | `lambdas` is a model field and `validate_config` does check it |
| 3 | `fwhm_ratio` | **inherited from `ckpt["train_config"]`** (0.25 for all four); hard-error if absent | §1.1. Never a CLI default. `untrained_mlp.pt` has no `train_config` and needs an explicit, recorded override |
| 4 | `gain` | **inherited** — v35 **3.6987**, L7 **100.0**, untrained 5.0 | §2.5. Note v35's `encoder_best.pt` was saved mid-anneal at epoch 675/1000, so its gain is 3.70, not the 5.0 in its model config |
| 5 | env `size` | 20 | v35 |
| 6 | `K` (stored goals) | `[1, 2, 3, 5, 10, 20, 50]` | goes past production's 0–10 so the capacity knee is inside the plot |
| 7 | memory mode | `multi_env_goals` | §2.3. `goal+distractors` secondary, on request |
| 8 | `n_envs_per_world` | 50, fixed at max `K` | §2.3 — keeps placement constant while load varies |
| 9 | worlds (draws) | 50 | so `K=50` gives 2500 env measurements, `K=1` gives 50 |
| 10 | `steps` sweep `S` | `[1, 2, 3, 5, 10, 15]` in A, B, C, D | production feeds 1,2,3 to the policy |
| 11 | `beta` | **inherited from the checkpoint** (`= gain`: 3.70 for v35, 100 for L7); `use_tanh=False` control | §6.5. Never a CLI value |
| 12 | `alpha` | 1.0 | pinned |
| 13 | `tau` | **removed.** `exact_hit` is threshold-free | §3.2 |
| 14 | `CELLBANK` for retrieval | all `K` footprints + 20k random scaffold cells | §3.2 |
| 15 | continuous samples/env | 200k uniform + 50k annulus at `d<3` | free, given the cost note in §5 |
| 16 | continuous heatmap resolution | 8 bins/cell | |
| 17 | §4 heatmap coordinates | goal-relative (39×39), + env-absolute + one single-env panel | §4 |
| 18 | Test D | in scope, runs by default | §7 |
| 19 | Test A rescue mode | **off by default**, `--rescue` | §3.1a |
| 20 | seeds | 3 | across-world spread is the bigger term |
| 21 | continuous-phase encoding variant | **open** — not built | §5 aside. Decide after seeing `excess(d)`; that curve is what says whether it is worth building |

### 9.1 Cost

The dominant term is Test B: `worlds × K × size² × |S|` recalls =
`50 × 50 × 400 × 6` = 6M, batched, at `D = 1024`. That is a few minutes of
GPU matmul, not a job. Test C adds nothing (§5 cost note), Test A is
`worlds × K × size²`, Test D is pure numpy on an already-computed field. The
real cost is `encode_positions`: `worlds × K × (size+1)²` encoder forward
passes = ~1.1M, which is one batched pass. **The whole suite is one node-hour
per encoder, not a sweep** — which is the argument for running all 4 level-7
seeds rather than picking one.

---

## 10. Traps

1. **`fwhm_ratio` is not validated by the thing that looks like it validates
   it.** `encoder_io.validate_config` accepts `fwhm_ratio` and `encoder_gain`
   and checks neither — only `lambdas`. Read both from the checkpoint, assert
   them into the result header, and never accept a CLI default (§1.1).
2. **Do not precompute `encoded_Phi`.** 12 GB, and unnecessary (§2.1).
3. **`q` is `(East, North) = (Δx, Δy)`.** A transpose here silently mirrors
   every angle and the aggregates still look plausible. Unit-test against a
   hand-built case where the goal is due East.
4. **`retr_dist` is undefined outside the test env** (§3.2). Averaging a
   real-space distance over retrievals that landed in another room manufactures
   a small, reassuring, meaningless number. Keep the categories separate.
5. **§4 heatmaps must be goal-relative.** The goal moves between envs, so an
   env-absolute per-cell average is a different quantity in every sample (§4).
6. **The Gram–Schmidt basis is clipped to `[1, Npos−2]`** and reads neighbours
   that may lie *outside* the env footprint. That is production behaviour and
   must be kept, but it means envs placed near the scaffold edge are a distinct
   population — record the offset with every result.
7. **Storage order matters** because `zero_diag` makes `W` order-dependent.
   Shuffle it, and average over the shuffle.
8. **Sign flips are real** (§3.1). Never take `|cos|`.
9. **`c = g` is degenerate** in Tests B and C. Handle it explicitly (§4).
10. **Distance bins are not uniform-mass.** In Test C, uniform-area sampling
   gives `∝ d` samples per bin, so the near-goal bins — the ones the whole test
   is about — are the noisiest. Hence the annulus refinement, and report `n`
   per bin on every curve.
11. **The basis is cached across the recall trajectory** in production
   (§1.4). Recomputing it per step would make the harness's multistep numbers
   describe a system that does not exist.
12. **Never compare across env draws.** Two encoders, one env list, one seed.
