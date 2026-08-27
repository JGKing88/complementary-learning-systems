# Evaluating an encoder through its Hopfield readout

Status: **spec**, 2026-08-27. Nothing here is implemented yet. This document
fixes what gets measured, on what code path, with what conventions, and what
the numbers are allowed to be compared against. Proposed code lands in
`analysis/hopfield_probe/` (see §8) — `analysis/` because it needs both
`encoder_training` and `hopfield_nav.world`, and `encoder_training` cannot
import upward without a cycle (`hopfield_nav/encoder_io.py` already imports
`encoder_training.models`).

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

and the three tests below take it apart in the order the errors compound:

- **Test A** — is the stored goal a fixed point of the Hopfield dynamics at
  all, and over what real-space disc does a cue relax to it?
- **Test B** — at every *grid* cell, how far off is `q`'s angle from the true
  bearing to the goal?
- **Test C** — the same question at *continuous* positions, which reach
  `encoded_Phi` only through the env's `round()` snap. Test B is the floor;
  C − B is the cost of the snap alone.

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
The harness supports all three behind one flag; §9 asks which is the headline.

| mode | contents of `W` | what it models |
|---|---|---|
| `goal+distractors` | this env's goal, plus `K−1` patterns drawn from cells **outside** this env's footprint, via `rollout/distractors.py:24` | production. Matches `n_train_distractors` and `--val_distractors`. |
| `multi_env_goals` | one goal from each of `K` different envs, each at its own offset; score the one belonging to the test env | continual / many-rooms capacity |
| `same_env_goals` | `K` goals inside the **same** env footprint | worst-case interference — the patterns are maximally similar |

Storage order is shuffled per trial (as `metrics.py:320` does), because
`zero_diag` makes `W` order-dependent in a small but real way.

### 2.4 What is held fixed

One scaffold (`lambdas`, `Npos`, `fwhm_ratio`) and one encoder checkpoint per
run; everything else varies. Two encoders are only ever compared at an
identical scaffold, env draw, and RNG seed — the env lottery is larger than
most encoder differences.

### 2.5 The alias floor, recorded once per (encoder, env)

Every cosine threshold in §3 is meaningless without knowing what cosine two
*unrelated* cells already score. Report, per env:

- `cos_floor` — mean and 99th-percentile cosine between the goal pattern and a
  few thousand cells far outside the env footprint. This is
  `unique_radius`'s `alias_ceiling` measured on the patterns this harness
  actually stores, and it is the number `tau` has to clear.
- `output_nonlinearity` and `gain` from the checkpoint config. A `sigmoid`
  head puts every embedding in the positive orthant, which lifts `cos_floor`
  toward 1 and compresses every discrimination in this document. It is a
  property of the checkpoint, not of the test, and must appear in the header of
  every result file rather than being discovered afterwards.
- `diag_frac` — `‖diag(W)‖ / ‖W‖` before it is zeroed. `zero_diag` removes a
  term that is large exactly when the embedding is sparse, so this says how
  much of the self-recall signal storage throws away.

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

### 3.2 Real-space basin radius

This is the number the test is actually for. The cue is not a noised goal
vector — it is the embedding of **another cell**, which is what makes the
radius live in real space.

For the test env's goal `g` (local coords), for every local cell `c`:

```
x0        = encode(c + offset)
x_final   = recall(x0, steps, beta, alpha)
retrieved(c) = argmax_k cos(x_final, z_k)          # which memory did it land on
hit(c)       = (retrieved(c) == goal index) and cos(x_final, z_goal) >= tau
d(c)         = || c - g ||_2                        # cells
```

Report, per `(K, steps)`:

- **`r_all`** — largest `r` such that **every** cell with `d ≤ r` is a hit.
  Stop at the first failing radius, not the last passing one; the condition is
  not monotone in `r`. This mirrors the convention `unique_radius` already
  uses (`encoder_training/unique_radius.py`) so the two are readable together.
- **`r_95`** — largest `r` such that ≥95% of cells within it are hits.
  `r_all` has cliff behaviour and two encoders can both report 0.
- **`hit_rate(d)`** — the full curve, which is what actually gets plotted.
  Reporting only a threshold crossing throws away the shape.
- **`r_by_direction`** — `r_all` restricted to each of 8 angular sectors from
  the goal. Anisotropy is a known live issue in this scaffold
  (`EXPERIMENTS_UNIQUE_RADIUS.md` §6.11f), and a radius averaged over
  directions can hide a direction where it is 0.
- **`spurious(d)`** — of the misses, what fraction landed on a *distractor*
  vs. on neither (a mixture). Different failures, different fixes.

**Figures:** per-`(K, steps)`, a `size × size` map of `retrieved(c)` (goal /
each distractor / mixture, categorical colour) with the goal marked; and one
`hit_rate` vs. `d` line per `K` on shared axes, one panel per `steps`.

**Note on `tau`.** A cosine threshold is a free parameter and will be argued
about. Mitigation: report `r` as a function of `tau` for
`tau ∈ {0.5, 0.7, 0.9}`, and treat the pure `argmax_k` (nearest-memory,
threshold-free) version as the primary.

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

- **`|err|` mean and median per cell** → the heatmap. One `size × size` panel
  per `(K, s)`, diverging colormap on signed `err` and a sequential one on
  `|err|`; goal marked; chance (90°) pinned on the colourbar.
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

**Goal radius.** `run_repro_v35.sh` runs `GOAL_RADIUS=1.0`, so the agent counts
as at-goal anywhere in an L2 ball of radius 1 — a region where `err_geom` is
already enormous. Report `excess` and `|err|_C` both including and excluding
the at-goal ball: error inside a region the agent never has to navigate out of
is not a cost, and averaging it in makes every encoder look worse than it is.

> **Aside worth deciding on (§9, Q6).** The snap error is an artifact of the
> codebook being defined only at integer positions. Grid phase is
> `x mod λ`, which is perfectly well defined for real `x`, and `smooth_gbook`
> already places a Gaussian bump at a phase — so a *continuous-phase* encoding
> is a few lines away and would remove this error class entirely. Measuring it
> here tells you how much that change is worth before building it.

---

## 6. Controls

Without these, none of §3–§5 is attributable.

**6.1 Oracle `q` (no Hopfield).** Replace `recall(z_c)` with `z_goal` exactly —
this is `oracle_signal_at` (`rollout/signal.py:152`). Run the full Test B and
Test C pipeline on it. The resulting error is **pure encoder + projection
geometry**, with the associative memory removed. It has no `K` and no `steps`
axis, so it is a single horizontal line under every curve in §4 and §5. Every
number should be quoted against it; `|err|_hopfield − |err|_oracle` is the
Hopfield's own contribution, and if the oracle is already bad, the encoder is
the problem and no amount of Hopfield tuning will help. *This is the single
most load-bearing control in the document.*

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

## 7. Optional Test D — does the field actually flow to the goal?

Angle error per cell is a local statistic; navigation is a global property of
the vector field. Cheap end-to-end summary, no policy involved: from each
starting cell, repeatedly step along `q` (unit step, greedy) for
`max_steps = 4·size` and record whether the trajectory reaches the goal.

Reports `reach_rate` overall and as a function of start distance, plus the
locations of **sinks that are not the goal** (cells where the field converges
elsewhere) and **limit cycles**. A field with 30° mean error and one sink in
the wrong corner is a very different object from one with 30° mean error and no
spurious sinks, and §4 cannot distinguish them.

This is optional in the sense that §3–§5 stand without it, and it should be
skipped if the goal is a fast encoder-ranking metric.

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
    flow.py          # Test D (optional)
    plot.py          # every figure
    run.py           # CLI: python -m analysis.hopfield_probe.run --ckpt ...
    tests/
```

Tests B and C are **one implementation**. C differs only in where positions
come from and what `theta_true` is measured from; forking them guarantees they
drift apart and stop being comparable, which would destroy `excess(d)`.

Results write one JSON per `(encoder, scaffold, K, steps)` with raw per-cell
arrays, plus a top-level summary CSV — so replotting never requires
recomputation, in the shape `analysis/encoder_sweep.py` already reads.

---

## 9. Parameters to decide

Proposed defaults follow `run_repro_v35.sh`, which is the live operating point.
Marked **?** are the ones that change the design rather than just the cost.

| # | parameter | proposed | note |
|---|---|---|---|
| 1 | encoder checkpoint(s) | `encoders/run_20260422_185816/encoder_best.pt` + best level-7 `w53_attract_knee` | **?** one, or a ranking sweep |
| 2 | `lambdas` / `Npos` | `[11,12,13]` / 1716 | v35 |
| 3 | `fwhm_ratio` | 0.25 | v35 |
| 4 | `gain` | checkpoint's | |
| 5 | env `size` | 20 | v35 |
| 6 | `K` (stored patterns) | `[1, 2, 3, 5, 10, 20, 50]` | **?** upper end |
| 7 | memory mode | `goal+distractors` | **?** §2.3 |
| 8 | `n_envs` per config | 50 | **?** cost driver |
| 9 | `steps` sweep `S` | `[1, 2, 3, 5, 10, 15]` in A, B and C | production feeds 1,2,3 to the policy |
| 10 | `beta` | `= gain`; plus `use_tanh=False` control | §6.5 |
| 11 | `alpha` | 1.0 | pinned |
| 12 | `tau` (Test A) | argmax primary; `{0.5,0.7,0.9}` reported | §3.2 |
| 13 | continuous samples/env | 200k uniform + 50k annulus at `d<3` | **?** — free, given the cost note in §5 |
| 14 | continuous heatmap resolution | 8 bins/cell | |
| 15 | continuous-phase encoding variant | not built | **?** §5 aside |
| 16 | Test D | build it? | **?** §7 |
| 17 | seeds | 3 | across-env spread is the bigger term |
| 18 | goal-radius exclusion | report both | §5 |

---

## 10. Traps

1. **Do not precompute `encoded_Phi`.** 12 GB, and unnecessary (§2.1).
2. **`q` is `(East, North) = (Δx, Δy)`.** A transpose here silently mirrors
   every angle and the aggregates still look plausible. Unit-test against a
   hand-built case where the goal is due East.
3. **The Gram–Schmidt basis is clipped to `[1, Npos−2]`** and reads neighbours
   that may lie *outside* the env footprint. That is production behaviour and
   must be kept, but it means envs placed near the scaffold edge are a distinct
   population — record the offset with every result.
4. **Storage order matters** because `zero_diag` makes `W` order-dependent.
   Shuffle it, and average over the shuffle.
5. **Sign flips are real** (§3.1). Never take `|cos|`.
6. **`c = g` is degenerate** in Tests B and C. Handle it explicitly (§4).
7. **Distance bins are not uniform-mass.** In Test C, uniform-area sampling
   gives `∝ d` samples per bin, so the near-goal bins — the ones the whole test
   is about — are the noisiest. Hence the annulus refinement, and report `n`
   per bin on every curve.
8. **The basis is cached across the recall trajectory** in production
   (§1.4). Recomputing it per step would make the harness's multistep numbers
   describe a system that does not exist.
9. **Never compare across env draws.** Two encoders, one env list, one seed.
