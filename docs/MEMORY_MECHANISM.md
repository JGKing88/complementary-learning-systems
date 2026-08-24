# What kind of memory is this?

A standalone account of what the `hopfield_nav` memory actually computes, why it
works, why it is not an attractor network, and what it would take to make it
one. Every number here is measured; the scripts are listed at the end.

This is not an experiment log — for the chronology, including the wrong turns,
see `EXPERIMENTS_NAV_P2.md` §5.3–5.9.

---

## Summary

1. **The system is a linear associative memory — a matched filter — not an
   attractor network.** Retrieval happens entirely in a single matrix-vector
   product. Iterating does not improve it and actively degrades it.
2. **It works well.** A cue corrupted to cos 0.70 is restored to **0.987** in
   one step, and the correct pattern wins the retrieval **99.7%** of the time
   at ten stored memories.
3. **It works because of the encoder, not the memory.** The encoder does two
   separate jobs — it makes same-environment similarity ≈ 0.99 against
   cross-environment ≈ 0, and it is a smooth chart of the arena so that a
   tangent-plane projection turns the retrieved pattern into a *direction*.
4. **There are no basins, for a reason no hyper-parameter can fix.** A
   saturating `tanh` has its fixed points at hypercube corners; the stored
   patterns are continuous encoder outputs, which are not corners.
5. **The attractor regime does exist, but it is a different network.** Storing
   `tanh(g·ξ)` at `g ≥ 300` gives genuine basins and exact retrieval, at a
   capacity of 50–100 patterns — but at that gain the pattern is ~98% of the way
   to its own binarization, and it is untested whether the geometry that
   produces the navigation signal survives.

---

## 1. What the system computes

The memory stores `M` patterns `ξ_k` (encoder outputs, unit norm, D = 1024) in a
Hebbian weight matrix, and recalls with one step:

```
W  = (1/D) · Σ_k ξ_k ξ_kᵀ          (zero diagonal)
recall(x) = normalize(tanh(β · W x))
```

Expanding the product is the whole story:

```
W x = (1/D) · Σ_k ξ_k (ξ_k · x)
```

**A sum of every stored pattern, weighted by how similar it is to the cue.**
That is a correlation, or matched filter — the classical linear associative
memory of Anderson and Kohonen. Nothing about it requires the network to be
iterated, and §3 shows iterating makes it worse.

### The encoder does two jobs, and the memory does neither

**Retrieval** works because the encoder separates environments. Measured:

| | value |
|---|---|
| `ξ(p) · ξ(goal)`, same env, 1 cell apart | 0.9993 |
| same env, 5–8 cells apart | 0.9832 |
| same env, opposite corners (20–30) | 0.7534 |
| **cross-environment, median** | **−0.0002** |

The encoder maps an entire environment into a tight cone and pushes other
environments to orthogonal. So in `Σ_k ξ_k (ξ_k · x)`, the goal's term dominates
and one product returns it. The margin `ξ·ξ_goal − max_k ξ·ξ_distractor` has
median **0.862**, and a distractor out-weighs the goal in only **0.26%** of
cells.

**Geometry** is the second job, and it is what the agent actually consumes.
Retrieval says *which* pattern, not *where*. The navigation signal is

```
q = W_xᵀ (recall(x) − x)
```

the projection of the retrieved displacement onto the two-dimensional tangent
plane of the embedding manifold at the agent's cell. Because the encoder is a
smooth chart of the arena, `ξ_goal − ξ_x ≈ J_x (p_goal − p_x)` locally, so this
recovers the grid displacement. Measured direction accuracy: cos **0.96** to the
true goal direction.

### Why explore and exploit are separable

The same geometry gives the agent its other necessary signal — whether the thing
it recalled is even in this environment.

| | median `‖q‖` |
|---|---|
| displacement to the goal pattern | **0.3006** |
| displacement to the nearest distractor | **0.0670** |

A distractor lives in another environment, so `ξ_d − ξ_x` is an essentially
unrelated direction in 1024 dimensions, and an unrelated direction keeps only
`√(2/D) = 0.0442` of its norm when projected onto a 2-D plane. With
`‖ξ_d − ξ_x‖ ≈ √2` that predicts **0.0625** against the measured 0.0670.

**The explore/exploit separation is a dimensionality effect**, quantitatively
predicted from first principles, and it costs nothing to obtain.

---

## 2. Why the dynamics is linear

The `tanh` is numerically inert:

| | |
|---|---|
| median \|β·W·x\| | **9.97e-05** |
| max \|β·W·x\| | 7.68e-04 |
| max relative deviation of `tanh(u)` from `u` | **2.3e-07** |
| median cos(trajectory with tanh, without tanh) after 12 steps | **1.00000000** |

`tanh` needs an argument of order 1 to bend. At 1e-4 it is the identity, so
`normalize(tanh(β W x))` is `normalize(W x)` — power iteration.

### One quantity, two knobs

The governing quantity is the **loop gain** `β·S/D`, where `S = ‖p‖²` is the
squared norm of the stored patterns. It is worth being precise about this
because it is easy to misattribute:

`W = (1/D) Σ p pᵀ`, so storing `λp` instead of `p` scales `W` by `λ²`. Therefore
`β·W` is **exactly invariant** under `(p → λp, β → β/λ²)` — and zeroing the
diagonal commutes with the scaling, so this is an identity, not an
approximation.

**Storage norm and `β` are the same knob.** Saying the system is linear "because
patterns are stored at unit norm" picks one arbitrary half of a product. The
correct statement is that `β·S/D ≈ 1e-4 ≪ 1`, and either factor can move it.

### `hopfield.beta` is a no-op

This follows from the above. With `tanh(u) = u` to seven digits,

```
normalize(tanh(β W x)) = normalize(β W x) = normalize(W x)
```

for any `β > 0` — the scalar cancels in the normalization. The trainer sets
`cfg.hopfield.beta` from the encoder's recorded gain, and it has no effect on
anything. It would begin to matter at `β ~ 10⁴`.

---

## 3. What iterating actually does

The network does reach a fixed point — a symmetric-weight system must — but it
is not a memory, and getting there destroys the retrieval.

| recall steps | cos to the goal pattern | cos to top eigenvector of `W` | step residual |
|---|---|---|---|
| 1 | 0.991 | 0.086 | 3.6e-01 |
| 5 | 0.829 | 0.202 | 6.7e-02 |
| 12 | 0.673 | 0.583 | 8.1e-02 |
| 50 | 0.066 | **1.0000** | 3.2e-04 |
| 200 | 0.064 | 1.0000 | **5.5e-08** |

Settled by ~50 steps and numerically exact by 200, onto the **leading
eigenvector of `W`** — which sits at cos **0.064** to the goal, essentially
orthogonal to what was being recalled. The rate is set by the spectrum:
`λ₂/λ₁ = 0.838`, a 10× error reduction every 13 steps.

The practical consequence, measured end to end on the navigation readout:

| recall steps | 1 | 12 |
|---|---|---|
| cells with direction error worse than cos 0.5 | **1.46%** | **12.27%** |

**`steps = 1` is not an untuned default — it is the only setting that retrieves
anything.** Retrieval lives entirely in the first product; every subsequent step
is power iteration toward the direction most shared among all stored patterns,
which by construction carries no information about which one was cued.

---

## 4. Why there are no basins

Three measurements, none of which relies on the word "linear":

| test | result |
|---|---|
| start **exactly** at a stored pattern, iterate 20 steps | cos to its own start falls **0.994 → 0.168** |
| 512 random starts, 400 steps, count distinct limits | **1**, for 11 stored patterns |
| that limit vs the top eigenvector of `W` | \|cos\| = **1.000000** |

So the memories are not attractors and the attractor is not a memory. Every
stored pattern is a **saddle**: with symmetric `W` each eigenvector is a fixed
point, but only the largest is stable, and any perturbation grows along the
dominant direction.

### Two independent conditions

Basins in a Hopfield network are *made by saturation*. `sign()` in the binary
case, a saturating `tanh` in the continuous one — saturation is what maps a
whole neighbourhood onto one state. Spurious mixtures are made the same way:
`sign(ξ¹+ξ²+ξ³)` is a fixed point precisely because `sign` maps it to itself.
With the nonlinearity inert there is no mechanism to create *any* extra stable
state, memory or spurious.

That gives two conditions, and conflating them is the main trap here:

| condition | knob | compensable? | what it buys |
|---|---|---|---|
| loop gain `β·S/D > 1` | `β` **or** storage norm — the same knob twice | **yes**, by either | a nonzero fixed point instead of decay to zero |
| stored pattern near a **corner** | storage gain `g` only | **no** | that the fixed point *is your memory* |

The current system fails the first, which is why it is a matched filter. But it
would fail the second even with the first repaired — sweeping `β` from 5 to 10⁶
on the raw continuous patterns never took stability above **0.226**. Raising the
loop gain converts decay-to-zero into a nonzero fixed point; it cannot make a
continuous vector into a corner.

### What normalization actually does

Per-step normalization is not what prevents basins. With it still on, raising
the gain far enough (β = 10⁶, pre-activation 22) does produce 8 attractors. What
normalization decides is *what the degenerate attractor is* when there are none:

| | below the gain transition |
|---|---|
| with `normalize_each` | state converges to the top eigenvector |
| without | state decays to the **origin** (`λ₁ = 1.4e-3 < 1`) |

### The classical model is not broken

Binarize the patterns and it works exactly as designed:

| condition | β | stored is a fixed point | # attractors | cue at cos 0.70 restored to |
|---|---|---|---|---|
| classical + continuous *(current)* | 1e5 | 0.086 | 1 | 0.086 |
| **classical + binarized** | 1e5 | **1.0000** | **11** | **1.0000** |
| classical + binarized | 1e6 | 1.0000 | 34 | 1.0000 |

All 11 patterns become fixed points, a corrupted cue is restored exactly, and
pushing the gain higher produces the spurious mixture states classical theory
predicts. **The failure is a mismatch between the architecture and the pattern
type, not a defect in the architecture.** Capacity is not involved either:
0.138·D = 141 against 11 stored.

*(That control is not perfectly clean — binarizing also decorrelates, dropping
max pattern overlap 0.4447 → 0.2910, and moves each pattern a long way, median
cos(ξ, sign ξ) = 0.7996. It shows the architecture retrieves binary patterns,
not that binarizing these particular patterns preserves what they encode.)*

---

## 5. Where the attractor regime does live

Two routes reach it, and both work.

### Route A — saturate the patterns before storing them

Store `p = tanh(g·ξ)`. The gain `g` interpolates continuously between the
encoder's vector and its binarization, moving the pattern toward a corner.

Recovery from a cue corrupted to cos 0.70 (≥ 0.99 means retrieved):

| storage gain `g` | M=5 | 11 | 25 | 50 | 100 | 141 |
|---|---|---|---|---|---|---|
| 10 | 0.499 | 0.461 | 0.228 | 0.091 | 0.098 | 0.052 |
| 30 | 0.941 | 0.885 | 0.758 | 0.040 | 0.100 | 0.045 |
| **100** | **0.990** | **0.986** | **0.976** | 0.923 | 0.108 | 0.055 |
| **300** | **0.997** | **0.996** | **0.993** | **0.986** | 0.146 | 0.062 |
| **1000** | **0.999** | **0.998** | **0.997** | **0.996** | 0.318 | 0.062 |

- **Turn-on gain is `g ≈ 100`**, marginal at 30, absent below.
- **Capacity is 50–100 patterns**, below the classical 141 — the expected
  penalty for correlated patterns.
- **Basins, not marginal stability**: the stability table matches this recovery
  table cell for cell.
- **Per-step normalization does not move the boundary.** Both the codebase's
  dynamics and the classical unnormalized one land in the same place, differing
  only at the very edge (g=100, M=50: 0.923 unnormalized vs 0.071 normalized).
- The dynamics gain only needs **2–5×** its own `D/S` threshold.

### Route B — change the update rule

The modern / dense associative memory update `ξ ← Xᵀ softmax(β X ξ)` has the
stored patterns as fixed points *by construction*, for continuous patterns:

| β | stored is a fixed point | # attractors | cue at cos 0.70 restored to |
|---|---|---|---|
| 8 | **1.0000** | **11** | **1.0000** |
| 512 | 1.0000 | **11** | 1.0000 |

Exact retrieval, exactly one attractor per stored pattern, and **no spurious
states even at β = 512** — the property the classical net cannot have here.

---

## 6. Whether you would want it

Both routes buy exact retrieval. Neither is obviously worth taking, and the
reason is §1: **the encoder does two jobs, and only one of them is retrieval.**

The navigation signal is `q = W_xᵀ(recall(x) − x)` — a *geometric* quantity that
reads the continuous structure of the embedding. At the storage gain where
basins appear (`g ≥ 300`), the stored pattern sits at cos ~0.98 to its own
binarization: most of that continuous structure is gone.

So the trade is:

- **gained** — retrieval becomes exact, so the direction error attributable to
  recall fidelity disappears. That matters, because recall fidelity governs
  essentially all of it: cells whose recall sits at cos ≥ 0.99 to the goal have
  a **0.01%** rate of bad direction, and 99.7% of all bad-direction cells fall
  below that threshold.
- **risked** — the tangent projection may no longer decode direction from a
  saturated pattern, in which case the exactness is worthless.

There is also an interface problem in Route A. If the stored pattern is
`tanh(g·ξ_goal)` but the cue is the agent's raw `ξ_x`, then
`q = W_xᵀ(tanh(g·ξ_g) − ξ_x)` subtracts an unsaturated vector from a saturated
one, which is not a meaningful displacement. A coherent version saturates the
cue as well and re-derives the tangent basis in the saturated space — a
different geometry, not a parameter change.

**The decisive test is cheap and has not been run**: recompute the direction
accuracy and the goal-present / goal-absent `‖q‖` separation under saturated
storage with matched cueing, at `g ∈ {30, 100, 300}`. If direction survives, the
attractor regime is genuinely available. If it does not, then the matched filter
is not a limitation of this design but **the only regime in which the geometry
survives** — which would be the strongest statement available about the
architecture.

---

## 7. What this does and does not say about the encoder

**It does not say the encoder must produce saturable codes.** That question is
downstream of the test in §6 and should not be pre-empted.

**It does say something independent and well-supported.** All retrieval failures
come from the extreme tail of the cross-environment similarity distribution:

| cross-env similarity | value |
|---|---|
| median | −0.0002 |
| p90 | 0.044 |
| p99 | 0.273 |
| **max** | **0.9823** |

The mean is already perfect — the repulsion objective achieved exactly what it
optimizes. The failures are **rare near-collisions**, where a pattern from
elsewhere in the scaffold encodes almost identically to an in-environment
position. That is the 0.26% of cells where a distractor out-weighs the goal, and
it matches the independently measured lock-failure rate.

The repel term in `encoder_training/losses.py` is a **mean**:

```python
repel = (K_pred[far] ** 2).mean()
```

Outliers at the far tail contribute almost nothing to a mean-squared penalty, so
there is no gradient pressure on them. Meanwhile the *evaluation* already uses
worst-case quantities — `unique_radius` is built on `bin_min`,
`inner_min(R) − outer_hi(R)`, and `alias_crossing_radius`. **The sweeps are
scored on worst case and trained on the mean.**

A tail-sensitive repel term — a hinge `relu(|cos| − τ)²`, a top-k mean, or
logsumexp — would put gradient where the failures are while leaving the 99%
already near zero untouched.

**Two honest qualifications.** First, this fixes the *matched filter*; it creates
no basins and does not touch the dynamics regime at all. Second, whether a plain
hinge suffices depends on how rare the pairs that actually break retrieval are:
with ~32k far-pairs per batch, a threshold around 0.2 would see hundreds of
violators, but if the damaging collisions are far rarer than that, hard-negative
mining is needed rather than a reshaped penalty. That distribution has not been
measured above p99 and should be before the loss is changed.

---

## How each claim was measured

All under `analysis/nav_p2/`, run against any navigation checkpoint (read only
for the encoder path, scaffold parameters, and Hopfield β).

| script | what it establishes |
|---|---|
| `why_it_works.py` | §1 — the similarity structure, margin, and `‖q‖` separation |
| `recall_dynamics.py` | §2, §3 — tanh inertness, the spectrum, the long-run limit |
| `recall_convergence.py` | §3 — readout degradation with recall depth |
| `attractor_test.py` | §4 — stability of stored patterns, attractor count, one-step completion |
| `gain_sweep.py` | §4 — the β sweep with and without normalization |
| `architecture_test.py` | §4, §5 — binarized control and the modern-Hopfield route |
| `stored_gain_capacity.py` | §5 — the storage-gain × capacity phase diagram |

Two measurement faults were found and fixed while producing these, both worth
knowing about because they produced confident wrong answers:

- **A padded zero read as convergence.** The final step's residual was filled
  with 0 for want of a successor, and a zero residual reads as "settled" — the
  precise opposite of what it meant. It is NaN now.
- **A collapsed state read as a rich landscape.** Below the gain transition
  without normalization, every state decays to zero; `F.normalize` maps zero to
  zero, so every pair of dead states has cosine 0 and each start counted as its
  own attractor. It reported "61 attractors" for a landscape that had collapsed
  to a point. The collapse is now detected and reported as such.

A third fault was in the experiment design rather than the instrumentation: the
first storage-gain sweep used a single dynamics gain `β = 1.0` and concluded
nothing worked at any gain or capacity. The threshold `D/S` moves with the
storage gain — 1024 at `g = 1`, ~1.3 at `g = 100` — so a fixed `β` sits below
threshold everywhere. **A single-value control cannot test a grid whose
threshold moves with the other axis.**
