# Why does grid → encoder → Hopfield work, and is it the best we can do?

## 0. The case for doing this analytically

*Written for someone not already inside the project. §1 onward assumes far more.*

**What the system is.** An agent moving through space needs to remember where
things are and go back to them. Ours does this with three pieces: grid cells
supply a periodic code for the agent's current position, a learned encoder
embeds that code as a high-dimensional vector, and an associative memory stores
the embedded codes of goal locations. To navigate, the agent cues the memory
with where it is now and reads a heading out of what comes back.

**The problem.** We want the encoder to map the grid code to embeddings that
work as memories in an attractor network: within some radius R of a remembered
goal, cues relax onto that goal. And when they do, we want to read the
real-space direction from cue to goal off the relaxation itself. Both halves are
precisely specified — the input code is fixed and known, the desired behaviour
is a concrete geometric property — so whether it is achievable is a question
that can be **answered**, not just explored.

### Questions independent of the grid code

**Can one embedding do both jobs?** Our empirical results suggest not. An
attractor network wants its memories to be binary — that is what makes them
fixed points — but a binary code carries no usable directional information,
because it can only change by flipping coordinates, so displacement over k cells
accumulates like a random walk rather than a straight line. Right now our
"attractor network" is actually **a one-step linear matched filter**, which
**returns a similarity-weighted sum of the stored goals and renormalises — so
the stored goals are not fixed points at all, and iterating degrades recall
instead of cleaning it up.** Nor can it descend gradually toward a goal the way
a Hopfield network is usually pictured: one pass leaves the state inside the
low-dimensional subspace spanned by the stored goals, so its intermediate states
are *blends of goals* rather than codes of intermediate positions, and there is
nothing in them to read a direction from. It is not an attractor network, and it
navigates well. *Why it matters:* "make it a genuine attractor" is an open design
direction we keep half-pursuing. If the incompatibility is real, it stops being
one, and the effort moves to changing the readout instead.

Existing theory is worth leaning on here. Classical Hopfield results give a real
guarantee — with N units you can store on the order of 0.1N patterns and recover
each exactly from a corrupted cue — but that guarantee is about cleaning up
**bit flips on a hypercube**. Our noise is not bit flips; it is displacement
along a 2D surface embedded in a high-dimensional sphere. Whether "relax to the
nearest memory" survives that change of geometry should be provable either way,
and the modern continuous-pattern versions of Hopfield networks come with
explicit separation conditions that would tell us.

**What dimension does this need?** Either way, we need real-space distance
between positions to track cosine similarity between their codes. Which
similarity-versus-distance profiles are achievable at all, and in how few
dimensions, is a classical question — which functions are positive-definite on a
sphere, and how many near-orthogonal directions fit in D dimensions. *Why it
matters:* we use 1024 dimensions, chosen by default. Three provably cannot do
it; where the real minimum sits between those two numbers is the difference
between having capacity to spare and sitting near a limit.

**What sets R?** R is one of our two success criteria, and we currently improve
it by trial. We do not know whether it is simply a property of how fast code
similarity falls off with distance — a near-field quantity — or whether it also
depends on how many goals are stored and how they interfere; our measurements
don't obviously support the simple story. *Why it matters:* a formula would say
which knob moves R and by how much, instead of one training run per guess.

### Questions about the grid code

**What is the encoder actually for?** A grid module with continuous phase traces
out a flat torus on the sphere, and a product of co-prime modules extends that
without repeating over an enormous range. That is already an embedding in which
real displacement maps proportionally onto code displacement — precisely the
property the direction readout needs. The grid code's one real defect is
aliasing: similarity returns at the module periods. So the encoder may not be
building spatial structure at all; it may only be suppressing aliases, while
risking damage to the geometry it was handed. Three things follow worth knowing:
is there a simpler or more direct way to remove the aliasing, how much of the
grid code's geometry does our encoder actually preserve, and would some other
structured input serve just as well? *Why it matters:* this is where nearly all
of our engineering effort has gone. If alias suppression is the encoder's only
real job, much of the space we have been searching is aimed at something the
input already provides.

### Why analytically

Scoring one candidate encoder costs a training run plus a full evaluation suite,
and every question above is currently answered by doing that many times over.
That reliably tells us which of the encoders we happened to train is best; it
never tells us whether a better one exists. The setup is unusually well
specified for a learned system — a fixed, structured input and a purely
geometric objective — which is exactly the case where theory should be able to
say what is possible before we spend more compute finding out.

> The precise versions of these questions are §6, with the calculations that
> would answer them in §7. §0 deliberately uses none of the vocabulary the
> campaign invented along the way.

---

A running conversation, started **2026-09-08**. Not a results log — the results
live in `EXPERIMENTS_HOPFIELD_PROBE.md` (probe), `EXPERIMENTS_UNIQUE_RADIUS.md`
(coding radius) and `EXPERIMENTS_NAV_TRI.md` (policy). This document asks what
those results *mean*, what a theory would have to explain, and which of those
explanations we could actually get.

**Success, as defined for this document:** a large **basin** (radius within
which a cue retrieves the right goal, `r_exact_all` over a scaffold disc) and a
high **continuous reach rate** (fraction of starts from which following `q`
arrives at the goal). Everything else — `r_min`, res90, alias rate, `acc45` — is
an instrument, not the target.

**Nomenclature.** "Hopfield" here means the project's recurrent memory module.
It is a one-shot outer-product associative memory with a saturating recurrence,
run for 1–15 steps; at the production operating point it is not a Hopfield
network in the usual sense (see §1.3 below) and nothing in this document depends
on it being one.

### Evidence tiers

Used inline throughout, because the single most expensive recurring error in
this campaign has been treating a tier-**[m]** number as tier-**[M]**.

| tag | meaning |
|---|---|
| **[M]** | measured over ≥4 training seeds *and* ≥3 scaffold draws, or an effect far larger than both spreads |
| **[m]** | measured once — one seed, one scaffold, or one arm. Five separate one-seed readings have already been retracted in this campaign |
| **[D]** | derived: follows from algebra we have written down and checked numerically |
| **[G]** | guess: plausible, load-bearing, and *not* tested |

---

## 1. The system, in one place

### 1.1 The forward path

```
(x, y)                              scaffold position, Npos = 1716
  │
  ├─ grid_codes(λ = [11,12,13], fwhm_ratio = 0.25)      → c ∈ R^434   (Σλ²)
  │      multi-module periodic code, Gaussian-smoothed
  │
  ├─ u = MLP(c)                                          → u ∈ R^1024
  │
  ├─ z = normalize(tanh(gain · u))                       → z ∈ S^1023
  │      ENCODER nonlinearity. gain = 100 in production.
  ▼
```

Storage of K goals, `Z ∈ R^{K×1024}`:

```
W = s · (ZᵀZ − diag),        s = 1/D = 1/1024        (zero_diag = True)
```

Recall of a cue `x`, for `steps` iterations:

```
x ← normalize( (1−α)·x + α·tanh(β · W x) ),          α = 1 in production
```

Readout at position `p`, goal code `z_goal`:

```
d_fwd = z(p + North) − z(p)
d_rgt = z(p + East)  − z(p)
basis = gram_schmidt(d_fwd, d_rgt)
q     = basis @ (z_goal − z(p))         ← 2-vector: bearing and (nominally) distance
```

The policy consumes `q` at `steps = 1`.

### 1.2 The knobs, sorted by what they touch

| knob | where | production | moves |
|---|---|---|---|
| `attract_lambda` | training loss | 0.5 | `d_eff` (down as λ rises) |
| `rate_lambda` (coding rate) | training loss | on | `d_eff` (up) |
| coverage / patch count | training data | 10% | capacity |
| `fwhm_ratio` | grid code | 0.25 | ~nothing measurable [M] |
| `lambdas` | grid code | 11,12,13 | scaffold period; never varied |
| encoder `gain` | inference *and* training | 100 | `d_eff` / chart length |
| `β` | recall | = gain | *nothing*, below D^1.5 |
| `α` | recall | 1 | time constant, only once β is above the knee |
| `steps` | recall | 1 | — |

### 1.3 The two nonlinearities, and why only one of them does anything

Both are `tanh`, and their arguments differ by ~650×.

* **Recall**: argument `β·(Wx)ᵢ`. `W` is a sum of K rank-1 terms over unit
  vectors with the diagonal zeroed, so `(Wx)ᵢ ~ D^−1.5 = 3.05e−5`. At β = 100
  the argument is 0.0033. **[D]**
* **Encoder**: argument `gain·u`, with median `|u|` = 0.0215 and no `1/D`
  suppression. At gain 100 the argument is 2.15 — already most of the way to a
  hypercube corner. **[D]**

Consequence, and it is stronger than "the tanh is weak": with α = 1 the update
is `x ← normalize(tanh(β·W x))`, so wherever tanh is linear,
`normalize(β·Wx) = normalize(Wx)` and **β divides out of the normalisation
exactly**. Measured identical to six decimals across β = 1 … 1000; the knee is
at `β ≈ D^1.5 = 32768`, where the median argument first reaches 1.08. **[M]**
(`beta_cancels_check.py`, PROBE §10.20.)

So the production system is a **linear matched filter followed by a
renormalisation**, and the only nonlinearity that shapes the code is the
encoder's.

---

## 2. The heuristics we operate on

These are the working rules the campaign has converged to. Grouped by which
half of the objective they serve, because — see §3 — they do not serve both.

### 2.0 Effective dimension — a sufficient statistic for the far field only

> **Corrected 2026-09-08.** An earlier draft of this section said `d_eff` was
> "the one training variable". That is wrong, and obviously so: a code with
> `d_eff` = 1024 is i.i.d. random per position — no aliases at all, and no
> chart, no local basis, no `q`, reach zero. The similarity structure is not a
> nuisance to be minimised, it is half the object.

`d_eff` = participation ratio of the code covariance, `(Σλ)²/Σλ²`; how many of
the 1024 output directions the code actually occupies.

**Far-field cosine spread is `1/√d_eff` to within a few percent over a 20×
range** — 131 → 345 in `d_eff`, 0.0825 → 0.0534 in far-cos sd against a
prediction of 0.0874 → 0.0538. **[M]** (PROBE §10.11.)

So distant pairs behave like random vectors in a `d_eff`-dimensional space, and
`d_eff` is a sufficient statistic **for the far field**. The near field — how
fast the code decorrelates over one to ten cells, which is what makes `q` a
usable finite difference — is a second, independent property, and `d_eff` says
nothing about it directly.

#### Both are functionals of one object: the code's spatial power spectrum

Take the code as a map `φ` from scaffold position to `S^1023`, and assume it is
statistically stationary — `⟨φ(p), φ(p′)⟩ = C(p − p′)` — which the grid scaffold
makes reasonable. Expand `φ(p) = Σ_ω a_ω e^{iω·p}` over the scaffold's Fourier
modes and write `P(ω) = ‖a_ω‖²`. Then:

* **near field**: `C(Δ) = Σ_ω P(ω) cos(ω·Δ) / Σ_ω P(ω)` — the autocorrelation is
  the Fourier transform of `P`. res90 is set by the *width* of `P`.
* **far field**: for generic `Δ` the phases are effectively random, so `C(Δ)` is
  a weighted sum of `M` random signs with sd `1/√PR(P)`; and if distinct spatial
  frequencies map to (near-)orthogonal output directions, the code covariance's
  eigenvalues *are* `P(ω)`, so `PR(P) = d_eff`. **[D, unverified numerically]**

That is a **derivation** of the `1/√d_eff` law rather than a fit to it, and it
says precisely what §2.1 and §2.2 are trading: the *shape* of `P`, of which
`d_eff` is one scalar summary (its participation ratio) and res90 is another
(its width). The two are not independent — in 2D the number of modes available
below frequency `ω_max` is `~π ω_max²`, so `d_eff · res90²` is bounded — but
they are also not the same number, and the bound is not obviously tight: over
the attract ladder the product runs 14.6k / 19.0k / 21.1k / 25.7k at
`att` 0.5 / 1 / 2 / 16, i.e. it *falls* as `d_eff` rises, against a ceiling of
order `L²/4π ≈ 234k` for a broadband 2D spectrum on a scaffold of period
`L = 1716`. **[m]** Something is leaving a factor of ~10 in mode count unused,
and the obvious suspect is that a 3-module grid input (λ = 11, 12, 13) cannot
reach an arbitrary spectrum.

What survives from the earlier draft: `attract_lambda`, `rate_lambda`, gain and
patch count all move `P` along roughly **one** direction, which is why they do
not compound. That is a fact about the *knobs we have*, not about the code.

### 2.1 Reach heuristics

**R1 — Keep a spread term; it is not a regulariser, it is the mechanism.**
With `rate_lambda = 0` the code collapses to **16 of 1024** directions and the
alias rate goes to 0.21 against 0.004–0.06 for everything else — a 25–50×
effect, the largest number in the screen. **[M]**

**R2 — Push `attract_lambda` *down*, not up.** Reach is monotone in it:
0.806 / 0.931 / 0.972 / 0.987 at λ = 16 / 2 / 1 / 0.5. **[M]** Three sweep waves
(w52→w54) climbed the other way, to 64, because `r_min` rewards that direction.

**R3 — Never select on `r_min`, res90, or coding radius.** `r_min ≈ res90 ·
√(ln(1/C)/ln(1/0.9))` is a **product of two opposing functions of `d_eff`**, so
it has an interior optimum in the same variable that reach improves in
monotonically — one variable, two metrics, two different peaks. **[D+M]**
Independently: attract 16 → 32 leaves `r_min` flat at 12.0 while worst-pair
overlap goes 0.578 → 0.699 and every probe metric drops. **[m]** Cross-talk, not
radius, is what the readout reads.

**R4 — Gain has an interior optimum, and training beats inference at reaching
it.** Four statements, because the compressed version of this was unreadable.

*(a) Raising the encoder gain shortens the chart.* `tanh(g·u)` pushes toward
±1, flattening the graded coordinates that made neighbouring positions similar.
res90 = 10 / 8 / 6 / 5 at gain 100 / 300 / 1000 / 3000. **[M]**

*(b) A shorter chart buys alias suppression and costs direction, and the buying
stops first.* Alias 0.0082 → 0.0056 → 0.0052 → ~0.005 (saturates); mean angular
error 7.3° → 9.0° → 19.3° → 41.7° (does not). Reach therefore has an interior
maximum: 0.931 / 0.971 / **0.954** / 0.608. Past gain 300 it is not a trade any
more, just a cost. **[M]**

*(c) Lowering `attract_lambda` shortens the chart the same way, so the two
knobs are substitutes.* res90 goes 18 → 7 as attract goes 64 → 0.5, and each
arm's best *inference* gain falls as its trained attract falls: `att2` wants
300, `att0.5` wants 100. An encoder that already spent the budget in training
gains nothing by spending it again at inference — which is why the best config
in the campaign is also the one needing no override. **[M]**

*(d) But at matched res90 they are not equivalent, and training wins.* `L5`
*trained* to res90 6 reaches 0.977; `L6` *forced* to res90 6 by inference gain
reaches 0.954, and forced to res90 5, 0.608. **[m]** Same res90, very different
reach — so res90 is not a sufficient statistic either. Plausibly: training
reshapes `P(ω)`, whereas gain applies a pointwise nonlinearity that decimates
whichever coordinates happened to be small. **[G]**

**R5 — res90 ≳ 5 is a hard floor, and it is the least understood number here.**
Below it the direction field collapses (|err| 42° at res90 5) even though the
one-cell cosine is still 0.996, which the Gram-Schmidt basis ought to find
sufficient. **[M]** for the collapse, **[G]** for every account of why.

### 2.2 Basin / capacity heuristics

**B1 — Coverage buys capacity, not reach.** Over a 4× range (10% → 2.5%) reach
is flat at 0.978 / 0.977 / 0.965 while retrieval falls 98 → 88%, dead goals at
K=20 go 0.08 → 0.25 → 0.42, and the basin shrinks. **[M]** The corrected basin
ladder is **27.0 / 23.0 / 19.2 / 11.5 / 13.5** cells at 10 / 5 / 2.5 / 1.25 /
0.75%, with the bottom two rungs not separable. Practical floor: **2.5%**.

**B2 — Saturation is a square, not a ladder.** There are two independent
saturation knobs — the **encoder** gain, which decides whether the *code* is
graded or binary, and **β**, which decides whether the *recall map* saturates —
so the possibilities form a square, not a ladder. Three of its corners are
measured on the 10% winner (`run_sat10.sh`), and **this table is where the names
"arm A" and "arm B" used throughout this document are defined**: **[M]**

| | encoder gain | β | `cos_self` | basin | exact | acc45 | reach |
|---|---|---|---|---|---|---|---|
| **production** | 100 | 100 | 0.813 | 27.0 | 0.982 | 0.995 | 0.987 |
| **arm A** — recall only | 100 | 1e6 | 0.957 | 24.5 | — | — | 0.973 |
| **arm B** — both | 1e6 | 1e6 | 1.0000 | 28.2 | 0.999 | 0.392 | 0.103 |

*Production* saturates neither: a graded code, and a recall map in its linear
regime. *Arm A* saturates the recall alone — the code stays graded, and the
dynamics gain genuine attracting fixed points sitting at hypercube corners
**near** each memory. *Arm B* saturates both — `z = sign(u)` makes the code
binary, and each stored pattern becomes an **exact** fixed point of the recall
(`cos_self` = 1.0000). The fourth corner (encoder saturated, β low) is not
measured and is not interesting: a binary code read through a linear filter.

Half-saturation *costs* basin (a binarised state compared against a continuous
bank); full saturation gives the perfect fixed point and a *larger* basin, and
**destroys the direction field** — acc45 0.995 → 0.392, reach 0.987 → 0.103.
The production corner is the only one where memory and direction both work
today, which is why arm B is the checkpoint §7.1's escape (iii-c) is aimed at:
its memory is already perfect and only its readout fails.

**B3 — β is a switch, not a knob** (§1.3). What crossing it buys is
step-invariance, which the policy does not use at `steps = 1`.

### 2.3 Measurement discipline

Every one of these was learned by getting it wrong first.

**M1 — One training seed is not an arm.** Five one-seed readings retracted so
far (fwhm 0.5, `att1`'s 0.993, and the whole pre-§10.18 basin ladder).

**M2 — One scaffold draw is not an arm.** A *fixed* arm swings 0.959–0.988 over
three draws. Gaps under ~0.02 between top arms do not resolve on one draw.
**[M]**

**M3 — The screen is a filter, not a ranking.** Twice, independently, the arm
with the better alias rate lost the probe.

**M4 — Any censoring metric will invert your ladder.** The env-bounded basin
censored at ~27 and the duplicate-bank bug corrupted 6 of 16 values per
encoder; between them 2.5% drew its worst seed and 1.25% its best and the
ladder came out backwards.

---

## 3. What the heuristics do *not* cover

*(Opened 2026-09-08. This is the live section.)*

**3.1 — Nothing in §2.1 was ever validated against the basin.** Every heuristic
with multi-seed, multi-scaffold support is a *reach* heuristic. The corrected
basin (§2.2's 27.0) exists only along the **coverage** axis and the
**saturation** axis. Along `attract_lambda` — the axis R2 says to move, and the
axis that sets `d_eff` — every basin number on record (§10.9's "20.7–21.4") came
from the env-bounded probe, which censors at ~27 and therefore *cannot report*
the corrected 10% value of 27.0. **We do not know whether lowering attract helps
or hurts the basin.** It has never been measured.

This matters for where the marginal return is: reach at the production encoder
is 0.987 with a seed spread of ±0.005 — near enough to a ceiling that R1–R5 have
little left to buy. The basin is 27 cells and has no known ceiling at all.

> **Superseded in part by §3.4.** At production coverage the residual reach loss
> is ~80–90% starts whose retrieval was *exact*, so the binding constraint on
> reach is neither the basin nor the memory — it is the direction field.

**3.2 — The basin metric mixes two mechanisms, and only one of them is reach's.**
`basin_mode_check.py` decomposes every recorded basin map by *what the failing
cue retrieved instead*. Two structurally different failures exist:

* **`other_goal`** — the recalled state is nearest a different **stored goal**.
  Cross-talk; the far-field mechanism, governed by `d_eff`; the same thing that
  sets the alias rate and therefore reach.
* **`near`** — the recalled state is nearest another **cell of the disc within
  2 cells of the goal**. No other memory is involved. The memory returned
  essentially the right thing and the argmax picked a neighbour. `d_eff` has no
  obvious claim on this at all.

Composition of failures in the annulus `[r_exact_all, r_exact_all + 4)` — the
cues that actually terminate the guaranteed basin — at K = 5: **[m]**

| encoder | median `r_exact_all` | near | far | other_goal |
|---|---|---|---|---|
| 10% (production) | 29.0 | **0.50** | 0.00 | **0.50** |
| 5% | 23.0 | 0.73 | 0.00 | 0.27 |
| 2.5% | 20.5 | 0.87 | 0.00 | 0.13 |
| 1.25% | 14.0 | **1.00** | 0.00 | 0.00 |
| 0.75% | 14.5 | 0.60 | 0.12 | 0.27 |
| 10% β = 1e6 | 25.0 | 0.90 | 0.04 | 0.06 |
| 10% gain = 1e6, β = 1e6 | 29.0 | 0.02 | 0.02 | **0.96** |

Far from the goal the composition is the expected one — beyond r = 40 the
production encoder's failures are 0.90 `other_goal`. It is specifically **at the
edge of the guarantee** that half the failures are 1–2 cell misses.

Two consequences:

1. > ### ✗ Retracted — `r_exact` is the right metric, and this was wrong
   >
   > The claim below was that a `near` retrieval is cheap, so the operational
   > basin is `r_goal` rather than `r_exact_all`. **Measured, and it is false**
   > (§3.4). The `~2°` figure is computed at r ≈ 29 and is irrelevant: the walk
   > has to pass through the *terminal* neighbourhood, and there the same
   > 1-cell target error is a large relative error against a small
   > `z_goal − z_here`. With `ARRIVAL_RADIUS = 0.5` a cue 0–4 cells from the
   > goal whose retrieval is off by one arrives **34.5%** of the time against
   > 99.5% for an exact retrieval.
   >
   > `r_exact_all` is right *because* it is a guarantee measured outward from
   > radius 0: it certifies a **contiguous exact core** around the goal, and a
   > contiguous exact core containing the terminal neighbourhood is exactly
   > what arrival needs. `r_goal` certifies nothing about the last four cells.
   > The tolerance table is kept below because §3.4 reads against it.

   **The published basin under-reports what navigation gets, and it does so
   more at low coverage.** ~~The retrieved code is consumed as
   `q = basis @ (z_goal − z_here)`; retrieving the cell one north of the goal
   is not a failed retrieval, it is a target one cell off, which at r ≈ 29 is a
   ~2° bearing error.~~ Re-measuring the same discs — paired per map, since the
   two columns can share a median with no map sharing a pair: **[m]**

   | encoder | `r_exact → r_goal`, per map | median gap |
   |---|---|---|
   | 10% | 26→26, 31→31, 29→29 | **0.0** |
   | 5% | 24→32, 23→28, 22→21 | 5.0 |
   | 2.5% | 19→29, 22→23 | 5.5 |
   | 1.25% | 19→24, 9→27 | **11.5** |
   | 0.75% | 13→27, 6→19, 16→16, 17→24 | 10.0 |
   | 10% β = 1e6 | 25→26, 25→31, 26→32, 24→26 | 4.0 |
   | 10% gain = 1e6, β = 1e6 | 31→30, 28→27, 29→29 | −1.0 |

   **For the production encoder the two coincide exactly on all three maps.**
   Its exact-cell basin *is* its cross-talk basin. The gap opens as coverage
   falls, reaching 11.5 cells at 1.25% — where one map has an exact basin of 9
   and a cross-talk basin of 27. So the *steepness* of §10.18's basin ladder is
   substantially a precision effect, not a memory effect.

2. **`r_tol2` ≈ `r_tol4` ≈ `r_goal` in every row.** Once 2 cells of slop are
   allowed, the *next* failure is not a 5-cell miss — it is a jump to a
   different stored goal. There is no intermediate regime. The memory either
   returns the goal to within a cell or two, or it returns a completely
   different memory; §10.1's "reach is nearly binary" at the environment level
   reappears here at the level of a single cue.

So the honest answer to "do basin and reach share a variable" is **neither yes
nor no**, but not for the reason first given. The published `r_exact_all`
carries a **precision** term alongside the cross-talk term — zero at 10%
coverage, ~11 cells at 1.25%. That term is real, it is *not* reach's variable,
and §3.4 shows it is *not* free either. B1's "coverage buys capacity" survives
on the dead-goal rate at K = 20, which is measured independently.

Caveat: 2–4 maps per group, one (world, env) pair each, so **[m]**. The
direction is monotone in coverage across all five rungs and the saturation
contrast is 0.02 against 0.90 `near`, both far larger than that noise, but the
per-rung magnitudes are not to be quoted.

**3.4 — A `near` retrieval is free at range and lethal at the goal.**
`near_miss_cost_check.py` crosses each cell's *retrieval outcome* against
whether the continuous flow from that cell *arrives*, on one memory per world
so both halves refer to the same object. It reproduces §10.14's reach exactly
(10% at four seeds: 0.993 / 0.987 / 0.987 / 0.984 against a published 0.987,
range 0.984–0.993), which is the check that the cross-tab is measuring the
published system. **[M]** for the contrast, four seeds each:

Arrival rate given the retrieval outcome, by **start distance from the goal**:

| encoder | outcome | 0–4 | 4–8 | 8–12 | 12–18 | 18–30 |
|---|---|---|---|---|---|---|
| 10% s43 | exact | 0.998 | 0.996 | 0.992 | 0.987 | 0.972 |
| 10% s43 | near | — | — | 1.000 | 0.992 | 0.791 |
| 1.25% s44 | exact | 0.995 | 0.987 | 0.980 | 0.973 | 0.948 |
| 1.25% s44 | **near** | **0.345** | **0.354** | 0.657 | 0.847 | 0.900 |
| 1.25% s43 | **near** | **0.504** | **0.522** | 0.708 | 0.811 | 0.742 |

The `near` row is not a constant discount, it is a **range-dependent** one:
~0.35 in the terminal neighbourhood, ~0.90 at range. Close to the goal
`z_goal − z_here` is small, so a one-cell error in the target is a large
*relative* error and the unit-length step misses the 0.5-cell arrival disc;
far away the same error is a couple of degrees, and the walk repairs it once it
enters the exact core.

**Production has no `near` retrievals inside 8 cells at all** — the "—" entries
above are empty cells, not small samples. Its exact core is contiguous out to
26–31 cells, so every near-miss it has is at range, where they are nearly free.
That is *why* production works, and it is why the metric has to be a guarantee
from radius 0 rather than a tolerance.

**And a second reading, which points somewhere else entirely.** Decomposing the
*lost* reach by retrieval outcome:

| | share `exact` | arrival if exact | fraction of lost reach that is `exact` |
|---|---|---|---|
| 10% s42 | 0.982 | 0.993 | **0.91** |
| 10% s43 | 0.982 | 0.990 | **0.79** |
| 1.25% s44 | 0.825 | 0.979 | 0.19 |
| 1.25% s42 | 0.685 | 0.987 | 0.19 |

At production coverage **~80–90% of the residual reach loss comes from starts
whose retrieval was exactly right**. The memory handed back the correct goal
code and the walk still did not arrive — a direction-field failure, not a
memory failure. At 1.25% that inverts: the memory is the problem (`near` alone
is ~45% of the loss at s44).

This changes §3.1's conclusion about where the marginal return is. At 10%
coverage neither the basin nor retrieval is what is capping reach at 0.987 —
**the readout is**, which is the same object §10.20 found the saturated arm
destroying and R4(d) found training-vs-inference distinguishing.

**3.3 — A bug found while doing 3.2, now fixed.** `basin_probe` filtered
`cx, cy, d` by the scaffold-edge `keep` mask but not `dx, dy`, so for any goal
within 64 cells of the scaffold edge the map payload paired each cue's outcome
with a *different* cue's offset. Nothing crashed — the radii come from `d` and
were always correct, and `_basin_map`'s `zip` silently truncated to the shorter
list. **60 of 140 recorded maps** were affected. Radii, and therefore every
basin number in PROBE §10.18, are unaffected. Fixed, with
`test_basin_map_offsets_track_clipping` pinning the invariant (under identity
recall every cue must land on its own offset). §3.2 uses only the 21 clean
K = 5 maps.

---

## 4. Cheap next measurements

> **Superseded as a framing by §6.** These four were accumulated one
> per turn; §5.5 derives a taxonomy they are instances of. Kept because the
> specific measurements they name are still the cheapest next steps.

**§4.1 — Is `d_eff = PR(P)` and `C = FT(P)` actually right?** §2.0 asserts the
code covariance's eigenvalues are the spatial power spectrum, under a
genericity assumption that distinct spatial frequencies map to orthogonal
output directions. Cheap to check: compute `P(ω)` directly, compare `PR(P)`
against the measured `d_eff` and `FT(P)` against the measured `C(Δ)`, on the
same checkpoints §10.11 used. If it holds, the `1/√d_eff` law stops being an
empirical fit.

**§4.2 — What spectra can a 3-module grid input reach?** `d_eff · res90²` runs
14.6k–25.7k against a broadband ceiling of ~234k. Is the missing factor of ~10
a property of the losses, or of the input code?

**§4.3 — What sets the radius of the *exact* core?** §3.4 makes `r_exact_all`
the metric that matters, and it is a statement about local precision, not about
cross-talk: within the core the recalled state's nearest cell must be the goal
and not its neighbour, and neighbouring cells sit at cosine ~0.998 of each
other. So the core radius is where the recall's perturbation first exceeds a
margin of order `1 − C(1)`. Both quantities should be computable — the
perturbation from `K`, `d_eff` and `C(r)`; the margin from `C(1)`, which is the
near field. If so, the exact basin is predictable from the same power spectrum
as everything else, and it is the one place where the *near* field enters a
capacity-like quantity.

**§4.4 — What is actually capping reach at 0.987?** §3.4 says ~80–90% of the
residual loss at production coverage is starts that retrieved exactly. That is
the direction field, and the campaign has no account of it: §10.11's open
question 3 (why direction collapses below res90 ≈ 5) is the same gap seen from
the other end. Where do those starts sit, and do they fail by bearing error,
by a sink, or by a limit cycle? `discrete_flow` already records sinks and limit
cycles and they have never been read against this.

---

## 5. The formal setup

*(Opened turn 4. §4's questions are re-derived from this in §5.5; the loose list
they started as is superseded.)*

### 5.1 Objects

| | | |
|---|---|---|
| `X` | `(Z_L)²`, `L = lcm(11,12,13) = 1716` | the scaffold: a discrete flat 2-torus, \|X\| = 2.94 M |
| `g : X → R^434` | ⊕ over modules λ of a Gaussian bump on `Z_λ × Z_λ` | the grid code. `‖g(p)‖` is constant; `g` is **exactly translation-equivariant** — `g(p+a)` is a cyclic permutation of `g(p)` |
| `φ : X → S^{D−1}` | `normalize ∘ tanh(γ·) ∘ MLP ∘ g`, `D = 1024`, γ = gain | the encoder. **Not** equivariant: the MLP mixes coordinates arbitrarily |
| `M = φ(X) ⊂ S^{D−1}` | | the **code manifold** — a 2-dimensional (discrete) surface on the sphere |
| `Z = [φ(y_1) … φ(y_K)]` | K goals drawn from X | the memory |
| `W` | `D^{-1}(ZᵀZ − diag)` | |
| `R(x)` | `normalize(tanh(β W x))`, `= normalize(Wx)` below the knee | one recall step |
| `B(p)` | `GS(φ(p+e₂) − φ(p), φ(p+e₁) − φ(p))` | the local frame |
| `q(p, v)` | `B(p)(v − φ(p))` | the readout |

Everything below is written in terms of **one function**, the similarity
kernel

```
C(a) = ⟨φ(p), φ(p + a)⟩            (assuming stationarity; see §2.0)
```

and its displacement form `N(a) = ‖φ(p+a) − φ(p)‖ = √(2(1 − C(a)))`.

### 5.2 The three conditions

Write `y` for the goal of the agent's own environment, `r = p − y` for the
agent's displacement from it, `R_op` for the largest `‖r‖` the system must work
at (≈ 30 cells for a size-20 arena), and `S_k = φ(y) − φ(y_k)` for the
separations between co-stored goals.

**(J1) Addressing.** One recall step gives
`W φ(p) ∝ C(r)·φ(y) + Σ_{k≠y} C(p − y_k)·φ(y_k)`. Retrieval names the right
memory when the first term dominates:

```
C(r)  >  max_{k ≠ y} C(p − y_k)                                        (J1)
```

A **max of K−1 far-field samples against one near-field signal**. §10.3's
fitted "one competitor above cos 0.25" is an empirical reading of exactly this,
and the far-field samples are `≈ N(0, 1/√d_eff)` (§2.0), so (J1) is a
tail-of-the-max problem in `d_eff` and `K`.

**(J2) Localisation.** Retrieval names the right *cell* when the recalled
state's nearest neighbour in `M` is `φ(y)` and not `φ(y ± e)`. Writing the
cross-talk residual as `ε`, this needs

```
⟨ε, φ(y ± e) − φ(y)⟩  <  ½‖φ(y ± e) − φ(y)‖²  =  1 − C(1)              (J2)
```

**The margin is the near field and the noise is the far field.** This is the
sharpest form of the tension: the quantity that has to be *large* for
localisation, `1 − C(1)`, is the quantity the readout wants *small*.

**(J3) Differentiation.** `q` is a first-order finite difference, so it needs
the chart to be **locally flat over the operating range**. Concretely, `q`
carries bearing *and* distance iff

```
‖φ(p + r) − φ(p)‖  ∝  ‖r‖      for ‖r‖ ≤ R_op                          (J3)
```

i.e. `N` is **ballistic** rather than diffusive. `N(k) ∝ √k` is what a generic
curved manifold gives — successive unit displacements are mutually orthogonal —
and it makes `⟨φ(y)−φ(p), d̂⟩` saturate after one cell, which is §10.20's
measured failure. Ballistic `N` is equivalent to the image being an
approximately **flat 2-plane patch** of radius `R_op`: zero extrinsic curvature
over that scale.

#### What "locally flat" means, and why a binary code cannot be it

**Operationally.** Take unit steps along the code manifold and ask whether the
displacements **stack** or **decorrelate**:

```
flat       φ(p + k·e) − φ(p)  ≈  k · (φ(p+e) − φ(p))       a straight line, ‖·‖ ∝ k
curved     each step's direction is fresh                   a random walk,   ‖·‖ ∝ √k
```

Measured (§10.20), `‖Δk‖ / (k‖Δ1‖)` at k = 2 / 4 / 8: production **0.965 /
0.910 / 0.814** — 81% of perfect flatness even eight cells out. Binary **0.701 /
0.492 / 0.345**, which is `1/√k` to three decimals. Maximally curved.

**Why a binary code is forced into it.** A binary code can only change by
flipping coordinates, and **a coordinate can only flip once** — flipping it back
undoes the move. So once a bit has flipped it cannot contribute further
displacement in that direction, and every subsequent step must recruit *fresh*
bits, which are orthogonal directions. Hence the random walk. That is
`H(k) ≤ k·H(1)` in words, i.e. (T1). A graded code has no such constraint: a
coordinate can keep moving in the same direction, so displacements add along a
line.

**A second picture, sometimes more useful.** Flat means the code points are laid
out locally like a **lattice in a plane** — so the point halfway between two of
them is itself (near) a code. Diffusive means they are laid out like **random
points on a sphere**, near-equidistant beyond a few cells — and the midpoint of
two random points on a sphere is near *no* point of the set. Check: the binary
code's `cos(k) = 1 − 2mk/D` hits zero at k ≈ 28 and stays there, which is the
near-equidistant regime exactly.

**Flatness then buys two things, which are one property seen twice.**

* **Chords stay on the surface.** In a plane, the straight line between two
  points lies in the plane. In a curved set it cuts through the interior. This
  is what decides whether a partial step *decodes to an intermediate position* —
  and its failure is measured exactly, cos dipping to `√(1 − H/D)` = 0.918.
* **The tangent plane still approximates far points.** `q` is a first-order
  Taylor estimate built from a one-cell step. On a flat patch it stays valid out
  to `R_op`; on a curved one it saturates after one cell.

**What curvature costs, precisely — three things, and only two of them bite.**
No intermediate states to traverse (fatal for Q1); no *distance* signal, since
`‖Δk‖ ∝ √k` means the gradient's magnitude stops growing (so `‖q‖` cannot encode
range); but **not** the bearing, which needs only that similarity *decrease* with
distance, and monotonicity survives curvature intact. That last is why (iii-c)
recovers arm B, and it is the distinction §10.20 originally missed.

### 5.3 The system, in one sentence

> **The encoder must embed a 2-torus in `S^{D−1}` as a union of nearly-flat 2D
> patches of radius `R_op`, such that patches around different goals are nearly
> orthogonal, and adjacent cells within a patch are separated by more than the
> cross-talk noise.**

(J3) is a *local flatness* condition, (J1) a *global packing* condition, (J2) a
*resolution* condition sitting between them. That is the whole system, and
naming them separately is what §2's heuristics never did — R1–R5 are all (J1)
with (J3) as an afterthought, and (J2) was invisible until §3.2.

### 5.4 Where the tension actually is — and where it isn't

Two observations that change the shape of the problem.

**(a) The grid code already satisfies (J3), exactly.** A single module with
continuous phase is the Clifford torus `(cos θ₁, sin θ₁, cos θ₂, sin θ₂)/√2`:
intrinsically flat, and `N(k)² = 2 − cos k₁ − cos k₂ ≈ ‖k‖²/2` — **ballistic by
construction**, with zero extrinsic curvature. Multiple co-prime modules are a
product of such tori: still flat, still ballistic, and non-aliasing out to
`lcm(λ)`. So the *near field is not the encoder's problem to solve* — it is
handed to it, and the encoder's job is to not break it. What the raw grid code
fails is (J1): its far field is a **lattice**, with exact revivals at every
`k ≡ 0 mod λ`, so `C` is not small at those separations at all. **[D]**

**(b) A random MLP cannot fix (J1) — it can only reshape the kernel
pointwise.** For a wide random network on constant-norm inputs, the composite
kernel is a scalar function of the input kernel: `C_φ(a) = f(C_g(a))`, with `f`
determined by the activation (Cho & Saul's arc-cosine kernels; Daniely et al.).
Two consequences. First, **an exact alias survives any `f`**: if `C_g(a) = 1`
then `C_φ(a) = 1`. Second, along the whole ladder of things the campaign
called knobs, a pointwise `f` trades (J1) against (J3) in a fixed way — a
power-law `f(c) = c^m` suppresses the far field as `ε^m` while shortening the
chart only linearly in `m`, which is a *very* favourable trade and is probably
why gain and the rate term work as well as they do. **[G — the conjecture; the
kernel identity itself is [D]]**

> **The single cheapest experiment in this document.** If (b) is right, then
> for an **untrained** encoder a scatter of `C_φ(a)` against `C_g(a)` over all
> displacements `a` must **collapse onto one curve**. For a trained encoder it
> must not, and the deviation *is* what training bought. §4.1's untrained
> control already exists. One afternoon, and it separates "the encoder is a
> kernel reshaping" from "the encoder learns geometry".

### 5.5 The question taxonomy

Four kinds of question, distinguished by what an answer would look like. Jack's
turn-4 list is written into it to check the taxonomy covers the space.

**I. Realizability — what codes exist at all?** Answers are existence proofs
and dimension bounds; no training involved.

* *What `D` is needed for a strictly monotone distance↔similarity code on a 2D
  scaffold?* → `C` must be a positive-definite function on `X`, so this is
  Schoenberg's theorem and the rank of the kernel: `D ≥ #{ω : P(ω) > 0}`. The
  1D case is solvable in `D = 2` (`C(k) = cos 2πk/L`); the 2D isotropic case is
  the obstruction Jack recalls at `D = 3`.
* *Is a network with basins in this way even possible?* → is there any `φ`
  satisfying (J1)–(J3) simultaneously at the required `K` and `R_op`, and what
  is the largest `R_op` for given `D`, `K`? A **packing** question: how many
  flat 2D patches of radius `R` fit in `S^{D−1}` at coherence ≤ μ (Grassmannian
  packing, Welch and Levenshtein bounds).
* *Can you have a true attractor network **and** a good reach rate?* → (J3)
  needs `M` to have graded structure near `φ(y)`; a genuine fixed point needs
  `φ(y)` to be a hypercube corner (§7, §10.20). **This is the one question the
  campaign has already answered empirically and negatively** — the question is
  whether that is a theorem or a property of this readout (see IV).

**II. Sufficiency — what does each condition demand of the code?** Answers are
inequalities relating `C`, `d_eff`, `K`, `R_op`.

* *Is perfect distance↔similarity enough for a good reach rate?* → no, by
  inspection of (J1): monotone `C` says nothing about the *tail* of `C` at
  large separations, which is what K−1 competitors sample. Spread is a
  statement about that tail, and §2.0 makes it `1/√d_eff`.
* *What guarantees a basin of radius `R`?* → (J1) ∧ (J2) for all `‖r‖ ≤ R`,
  which is §4.3.
* *Where does the 0.25 threshold come from, and the res90 ≈ 5 floor?* → the
  two fitted constants in §2; both should fall out of (J1) and (J3).

**III. Attainability — what training reaches such a code?** Answers are about
losses, sampling and coverage.

* *Coverage, sampling, loss shape, regularisation.* The empirical answers are
  §2.1's R1–R5 and §2.2's B1. The theory question is which of them are
  properties of the *optimum* and which of the *optimiser*.
* *Does an equivariant architecture do better?* → `g` is equivariant and `φ` is
  not; equivariance would make `C` exactly stationary, which is the assumption
  every result in §2.0 rests on. `encoder_training/equivariant.py` exists and
  has never been read against the probe.

**IV. Mechanism / optimality — is this the best we can do?**

* *What about the grid code enables this?* → §5.4(a): it supplies (J3) for
  free. The follow-on is whether any structured spatial input with a flat,
  ballistic local geometry would do (a torus of any period; a random Fourier
  feature bank with a shell spectrum), and whether the co-prime module
  structure matters beyond setting `lcm(λ)`.
* *Is this actually a Hopfield network?* → below the knee it is a linear
  matched filter (§1.3), so the classical capacity theory does not apply and
  the modern/dense-associative-memory separation theorems (Krotov & Hopfield;
  Ramsauer et al.) describe a system we are not running. Naming this correctly
  changes which literature is even relevant.
* *Is the (J1)-vs-(J3) trade forced?* → forced by information (any code on a 2D
  scaffold), by the input (three grid modules), by the readout (a first-order
  finite difference), or by the loss? **These have completely different
  implications and the campaign has never distinguished them.**

### 5.6 Theory that already exists and bears on this

Listed because the input and the objective are both unusually well specified,
which is exactly the situation where reaching for existing results beats
inventing.

| body | bears on |
|---|---|
| Schoenberg's theorem; positive-definite functions on spheres and tori | I — which `C(·)` profiles are realizable, and in what `D` |
| Spherical codes; Rankin, Welch, Levenshtein bounds; Grassmannian packing | I, II — how many near-orthogonal patches fit; the ceiling on `K` at given `D`, `R_op` |
| Grid-code capacity and resolution (Fiete, Burak & Brookings 2008; Sreenivasan & Fiete 2011; Mathis, Herz & Stemmler) | IV — what the co-prime module structure buys, and the Fisher-information view of res90 |
| Classical Hopfield capacity (Amit–Gutfreund–Sompolinsky) and the correlated-pattern extensions | I, II — but only if we move above the knee |
| Dense associative memory / modern Hopfield (Krotov & Hopfield 2016; Ramsauer et al. 2020) | I, IV — the separation condition, and what a real attractor would cost |
| Dual activations / arc-cosine kernels (Cho & Saul 2009; Daniely, Frostig & Singer 2016) | III, IV — §5.4(b), what an untrained encoder can and cannot do |
| Manifold capacity (Chung, Lee & Sompolinsky) | I, II — packing curved manifolds rather than points |
| Steerable / equivariant CNNs (Cohen & Welling) | III — the architecture question |

---

## 6. The questions

*(Opened turn 6.)* Seven, stated using **only the objects of §5.1** — position,
grid code, encoder, memory, readout — and the two success criteria. Nothing
here presupposes `d_eff`, res90, the alias rate or any other quantity the
campaign invented along the way; those belong to the *routes* in §7, not to the
questions. Each carries the context that makes it worth asking.

---

### Q1. Can the memory be a real attractor network and still support navigation?

**What the memory does now.** Each environment's goal is encoded as a vector,
and the set of them is stored in an outer-product weight matrix. To recall, a
cue vector is passed once through that matrix and renormalised. At the
operating point the saturating nonlinearity in that pass is numerically
inactive, so the operation is linear: the recalled vector is, up to a small
correction, a **weighted sum of the stored goal codes, weighted by how similar
the cue is to each**. Nothing about it is dynamical. The stored goals are not
fixed points of the map, and running it for more steps makes the result worse
rather than better.

**What a real attractor network would mean.** That each stored goal code is a
*stable fixed point*: recall started from a nearby cue moves onto it and stays.
That is worth wanting on its own terms — it makes recall robust to a corrupted
cue, makes the number of recall steps stop mattering, and gives each goal a
basin in the proper dynamical sense rather than as a measured radius.

**Why it might be incompatible with navigation.** The recalled vector is never
used as an identifier. It is used as one end of a subtraction: the agent's
heading comes from the **difference** between the recalled goal code and the
code at the agent's current position, expressed in a frame built from the codes
of two immediately adjacent positions. For that to give a usable direction —
and a usable distance — the code has to vary smoothly and *proportionally* with
displacement, over the whole range the agent must cover. Stable fixed points
constrain the code's geometry; so does proportional variation; and they
constrain the *same* geometry. Whether both can hold at once is the question.

**What an answer looks like.** What stable fixed points require of the code,
what the direction computation requires of it, and a proof that the two can or
cannot hold together. → **§7.1, whose *Synthesis* subsection is where this
question has landed.** In short: the answer splits. An attractor whose
fixed points sit *near* each memory is compatible with navigation and already
exists (measured: reach 0.973 against production's 0.987, in exchange for
step-invariance). An attractor whose fixed point **is** the memory is provably
incompatible, for any code and any encoder. The interesting content of Q1 is
that gap, and what the near-miss costs.

---

### Q2. From how far away can a stored goal be recovered, and what sets that distance?

**Context.** One of the two success criteria. Measured at ~27 cells for the
production encoder and 11–23 for weaker ones, with no account of why. It is the
number that decides whether an agent dropped anywhere in a room can find its
goal — and it has to be an **exact-cell** radius rather than an approximate one,
because a target one cell off is fatal on the final approach (§3.4).

**An answer looks like.** A formula for the radius in terms of how many goals
are stored, the code's local geometry, and its global statistics. → **§7.2**,
which already lands within ~30% and has ten predictions sitting in data we
have.

---

### Q3. What is the encoder actually for?

**Context.** The input is not raw position. It is a grid code — an already
highly structured, exactly translation-equivariant representation with
well-understood coding properties, and one this lab has theory for. The
campaign has treated the encoder as the thing that *builds* spatial structure
and has spent many sweeps tuning it on that assumption. It may instead be
inheriting nearly all of the structure and repairing one specific defect. If
so, the knob space we have explored is aimed at the wrong target.

**An answer looks like.** A decomposition: which properties navigation needs are
already present in the grid code, which are not, and hence what the encoder's
job is — with the follow-on of whether another structured spatial input would
serve, and whether the co-prime modules buy anything beyond setting the period.
→ **§7.5, §7.6.**

---

### Q4. What must any code satisfy for this to work, and what does that cost in dimension?

**Context.** Everything we know is about encoders we happen to have trained. We
have never asked what is admissible in principle, and without that we cannot
tell "our encoder is bad" from "no encoder could do better" — which is the
whole of *is this the best we can do*. The instinct that a 3-dimensional code
cannot do this is a special case.

**An answer looks like.** Conditions on the code's similarity structure for
retrieval and for direction-reading, stated separately, plus the smallest
output dimension in which a code meeting them exists. → **§7.3, §7.6.**

---

### Q5. Is the tension between separating far positions and relating near ones forced?

**Context.** Retrieval needs distant positions to have unrelated codes; the
readout needs nearby positions to have smoothly related ones. Every knob we
have moves both at once and in opposition — which is why the knobs do not
compound, and why an obvious selection metric sent three sweep waves the wrong
way. We have assumed the trade without asking where it comes from, and there
are four candidates with four different consequences: forced by information
(any code on a 2D domain at all), by the input, by the readout we chose, or
only by our loss.

**An answer looks like.** For each candidate, either a bound showing the trade
is unavoidable or a construction showing it is not. → **§7.4, §7.5.**

---

### Q6. Is the direction readout the right one?

**Context.** Direction is a first-order finite difference — a *linear* probe of
a code we deliberately made *nonlinear*. Two things point at it. At production
coverage, 80–90% of the remaining navigation failures are starts where the
memory returned exactly the right goal and the walk still failed (§3.4), so the
readout and not the memory is what is capping us. And Q1 turns on the readout
too: it is the readout's requirement that a true attractor violates.

**An answer looks like.** How much information about displacement is present in
the difference between retrieved and current codes, and how much a
finite-difference basis recovers — hence whether another readout has headroom,
and what it costs. → **§7.1's second escape.**

---

### Q7. What can training reach? — the boundary of the analytic program

**Context.** All of our operational knowledge lives here: which loss, how much
coverage, what sampling, whether an equivariant architecture would help. None
of it is derived and most of it probably cannot be. Q1–Q6 can say which codes
are admissible and which are optimal; they cannot say that gradient descent on
our loss finds them.

**An answer looks like.** Mostly measurement. What theory contributes is
telling us *what* to measure: once Q4 or Q5 names a target code, "does our loss
produce it?" becomes a test against a predicted object rather than another
sweep.

---

## 7. Routes: the specific calculations

The concrete decomposition of §6, ordered by value × tractability. Each is
stated with enough derivation to show it is a calculation and not a hope, plus
what would falsify it. **T1 and T3 are close to done; T2 already produces a
number that can be checked against §10.18 today.**

Notation from §5.1 throughout: `C(a)`, `N(a) = √(2(1−C(a)))`, `P(ω)` the spatial
power spectrum, `d_eff = PR(P)`, `K` stored goals, `D = 1024`.

---

### T1. An attractor is fine; an attractor *at the memory* is not — Q1, Q6

> **Corrected turn 7, and the correction matters.** This was first written as
> "a true attractor and a working direction field are incompatible", full stop.
> That is too strong, and the counterexample is already measured. §10.20's
> **arm A** (§2.2 defines the arms) — recall saturated (β = 1e6), encoder left
> continuous (gain 100) —
> has genuine attracting fixed points (`cos(recall(x), x)` = 0.9989,
> `cos(recall¹⁵(z), x)` = 0.9981 against 0.813 unsaturated) **and** a working
> direction field (reach 0.973, against production's 0.987). Its fixed point is
> a hypercube corner **near** each memory, at `cos_self` = 0.957, not the memory
> itself. The code stays continuous, so (J3) survives.
>
> What is actually impossible is the *stronger* requirement — that the stored
> code **be** the fixed point, `cos_self` = 1. The proof below is a proof of
> that, and its premise has to be stated to see why.

**Status: proven for the strong form.** Assume (i) each stored code is *itself*
a fixed point of the recall map, which above the knee forces `φ(y)` onto a
hypercube corner (§7's condition (a); §10.20's arm B attains it, `cos_self` =
1.0000); (ii) any cell can be a goal, so (i) must hold at every position — the
encoder is one function and cannot be a corner only at K chosen points; and
(iii) the readout is the first-order finite difference of §5.1.

By (i)+(ii) every code is a corner, `φ(p) ∈ {±1}^D/√D`, and then

```
N(k)² = 4·H(k)/D                        H = Hamming distance
```

`H` is a metric, so along a path of `k` unit steps `H(k) ≤ k·H(1)`, giving

```
N(k) ≤ √k · N(1)                                                    (T1)
```

**Diffusive, always.** (J3) asks for `N(k) = k·N(1)`, which combined with (T1)
forces `k² ≤ k`, i.e. `k ≤ 1`. So a fully binarised code **cannot** satisfy (J3)
beyond a single cell — not for this encoder, not for a better-trained one, not
for any binary code whatever. The continuous code escapes because its
displacement is a coherent vector sum rather than a count of flips.

> **What (T1) does and does not say — sharpened turn 18.** It bounds the
> **magnitude** of displacement, so a binary code cannot encode *how far*: the
> similarity gradient's size stops growing with distance. It says **nothing
> about bearing.** A gradient that is constant in `k` is still a perfectly good
> direction field, and Stage 1 measured exactly that — arm B reaching 0.984 on a
> fully binary code. The campaign read "does not grow" as "does not exist" for
> months. (J3) as stated in §5.2 is therefore too strong for what navigation
> actually needs: it demands proportionality, when *monotonicity* buys the
> bearing and only the distance readout needs more.

This converts §10.20's measured `‖Δk‖/(k‖Δ1‖)` = 0.701 / 0.492 / 0.345 ≈ `1/√k`
from a fact about one checkpoint into the only thing that could have happened,
and it says exactly where the three escapes are — one per premise:

* relax (i) — **accept a fixed point near the memory rather than at it.** This
  is arm A, it is already measured, and it costs basin 27.0 → 24.5 and reach
  0.987 → 0.973 in exchange for step-invariance. The cheapest escape by far,
  and the one the campaign already has in hand.
* relax (ii) — make goals a restricted subset of positions, so the code has to
  be a corner only there. Untested and probably unusable: goals are arbitrary
  cells of arbitrary environments.
* relax (iii) — a readout that is not a finite difference of the *stored* code.
  The only escape that could give an exact fixed point **and** a direction
  field. Four candidate forms, worked through below; §10.20's parked "basis
  from pre-nonlinearity activations" is **not** one of the live ones.

**Falsified by:** a binary code with `‖Δk‖/k‖Δ1‖` bounded away from `1/√k` over
a decade of `k`. (T1) says there is none.

#### Synthesis — where Q1 has landed

> We want an attractor whose relaxation — and ideally whose dynamics — passes
> through intermediate encoded states and comes to rest at a fixed point. The
> **graded** code gives us the intermediate states, because its manifold is
> locally **flat**, so a chord between two codes stays near the surface and
> decodes to real positions. But it cannot make a memory a stable fixed point:
> all `K` patterns are near-degenerate eigenvectors, so each is only
> *marginally* stable and the state drifts to a blend. The **binary** code gives
> us the exact fixed point but cannot traverse, because a blend of two corners
> has the sign pattern of whichever endpoint dominates — the decode flips
> discontinuously at the halfway point, with the cosine to the manifold dipping
> to `√(1 − H/D)`. A **continuous attractor network** is the structure that has
> both, though it needs an added symmetry-breaking term to drift toward a
> *chosen* goal rather than resting wherever it lands.

Three things that phrasing is carrying, each measured above:

* **Flat, not 2D.** Both codes are 2-parameter families; that is not the
  difference. The continuous one is flat over ~30 cells and the binary one is
  maximally curved — successive unit steps near-orthogonal — which is (T1)
  restated. Flatness and traversability are the same fact, not two.
* **Marginal, not absent.** See below.
* **A CAN gives two of three.** Invariance of the manifold (traversal stays on
  it) and stability of every point on it (arrival holds). It does **not** give
  drift toward one particular stored goal — that is what translation symmetry
  along the manifold forbids, and breaking it is the actual design work. Note
  also the near-circularity: the manifold you would want the CAN to carry *is*
  the position manifold, which is what the grid code already is.

##### What "near-degenerate eigenvectors, marginally stable" means

Below the knee the recall is `x ← normalize(Wx)` — **pure power iteration**, and
`W ≈ (1/D)·ZᵀZ` is, up to a 0.5% diagonal correction, `1/D` times the
**projector onto the span of the stored patterns**.

If the patterns were exactly orthogonal, that is *all* it would be: every
direction in the span stretched by the same factor. Power iteration would then
have nothing to select on, and **every stored pattern — and every blend of them
— would be an exact fixed point.** Perfect degeneracy.

What breaks the tie is the pattern **overlaps**. Write the Gram matrix
`G = ZZᵀ = I + E`, with `E` the off-diagonal cosines, of order `1/√d_eff` ≈ 0.06.
`W`'s nonzero eigenvalues are `(1 + eᵢ)/D` where `eᵢ` are `E`'s eigenvalues, so
they are all near 1 and split only by `O(‖E‖)`. Two consequences:

1. **A stored pattern is nearly, but not exactly, fixed.** Measured
   `cos(recall(z), z)` = 0.9966 at one step (§10.20). The force pushing it off is
   proportional to the eigenvalue splitting, which is tiny — hence *marginal*.
2. **Iterating amplifies the tie-break exponentially, and the winner has nothing
   to do with the cue.** Power iteration converges as `(λ₂/λ₁)ᵗ ≈ (1 − gap)ᵗ`
   toward `E`'s top eigenvector, which is some fixed combination of all `K`
   patterns. Measured `cos(recall¹⁵(z), x)` = 0.813.

So the **first** step is retrieval — it weights the stored patterns by the cue's
overlaps with them. **Every step after that is not retrieval**; it is the
tie-breaking dynamics grinding toward the globally favoured blend, forgetting the
cue as it goes. That is why more steps makes recall *worse* (§0), why `q` decays
with recall depth, and why the state drifts off the manifold (cos 0.989 → 0.729)
while its decoded position barely moves.

**And this is exactly the CAN parallel.** A CAN *engineers* exact degeneracy —
translation symmetry along the manifold — so the state is marginally stable in a
controlled direction and stays put until something moves it. Production has the
same mathematical structure by *accident*, in the `K`-dimensional memory span
rather than along the code manifold, with the degeneracy broken by uncontrolled
overlaps. Same shape; one designed, one not.

#### Why lowering `α` is not a fourth escape

The obvious next thought is to keep the saturated recall but take a *small step*
toward it — `x ← normalize((1−α)x + α·tanh(βWx))` with `α ≪ 1` — so the state
moves in the direction of the goal without ever binarising. It does not work,
and the reason is worth recording because it is not obvious.

`tanh(βWx)` is **not** normalised before the mix (`hopfield/core.py`), so above
the knee its norm is `√D = 32` against the cue's 1: the two terms are
comparable only near `α ≈ 0.03`, not at `α ≈ 0.5`. Writing `v = tanh(βWx₀)` and
`m = ‖(1−α)x₀ + αv‖`,

```
x_out − x₀ = (α/m)·v + ((1−α)/m − 1)·x₀
```

so `q = B(x_out − x₀)` **always lies in the span of the same two 2-vectors,
`Bv` and `Bφ(p)`**, for every `α`. Working the small-`α` limit through gives
`q ≈ 32α·B(v̂ − c·φ(p))` with `c = ⟨φ(p), v̂⟩ ≈ cos_bin·C(r)`, against `α = 1`'s
`q = B(v̂ − φ(p))`. The only difference is that the `−z_here` half of the
readout is subtracted with weight `c` instead of 1 — and since
`1 − cos_bin·C(r)` stays small where the signal is small, the resulting bias is
under 4% of the signal at every radius.

**`α` is a scalar gain on `q` plus a sub-4% bias. It cannot rotate a wrong
direction into a right one.** Which is exactly §4.3's empirical finding — "a
time constant, not a destroyer" — and §4.2's, that the rescue sweep's `α`
optimum did not transfer. `α` sets how fast the state walks to the fixed point,
never which fixed point or which way `q` points.

There is a real idea underneath the α proposal, and it is escape (iii). But it
needs stating carefully, because *"read the direction the state moves under
recall"* is **already what is implemented**: `project_q` is literally
`basis @ (recalled − current)` and `recalled = R(current)`, so `q` is the
one-step recall displacement projected on the local frame. The instinct is
right and the code already has it; what breaks under full saturation is that
**both ends of the subtraction are binary**.

#### What escape (iii) would actually be

The code `z` has two consumers with opposite requirements: the **memory** wants
it binary (exact fixed points), the **readout** wants it graded (a chart).
Escape (iii) is any scheme that stops making one object serve both. Four
candidates, and they are not equally alive.

**(iii-a) Basis from the pre-nonlinearity activations, target still `z`.** The
version parked in §10.20: `basis = GS(u(p+N) − u(p), u(p+E) − u(p))` with `q`
still `basis @ (z_goal − z(p))`. **This is dead, by §10.20's own measurement.**
The frame was never the problem — `‖d_fwd‖` is 0.267 binarised against 0.086
continuous, *larger*, and zero at none of 1500 positions. What breaks is the
accumulation: `q_north = ⟨Δk, d̂_fwd⟩` is flat at 0.267 → 0.239 for binary
against 0.086 → 0.417 continuous. Fixing a frame that was already
well-conditioned changes nothing. **[M]** — and this should be recorded in
§10.20, which still lists it as the open route.

**(iii-b) Heteroassociation — a binary key with a graded value.** Store
`z_k = sign(u_k)` as the key and `u_k` as the value. The autoassociative
recurrence runs on keys only, so its fixed points are exact; a second,
heteroassociative step emits `u_goal`; and the readout becomes
`q = GS(Δu) @ (u_goal − u(p))`, a difference of two **continuous** codes.
(J1) is satisfied on the key side and (J3) on the value side, by different
objects, which is exactly the decoupling. Cost: a second matrix (or an explicit
value store), and the value is only as good as the key retrieval — but key
retrieval is now a clean binary nearest-neighbour problem with exact fixed
points, which is *better* than the current continuous matched filter, not worse.
This is the key/value split of attention, i.e. the modern-Hopfield form in
§5.6 — worth noting that the principled fix is a known architecture.

**(iii-c) A scalar potential — descend `‖z_goal − z(p)‖` instead of projecting
it.** Binarisation destroys *bearing*, not *distance*: with
`‖z_goal − z(p)‖² = 4H/D` and `H ∝ k`, the **norm** still encodes displacement,
monotonically, as `√k`. So take the readout to be a finite difference of the
**scalar** rather than of the code — evaluate `‖z_goal − z(p ± e)‖` at the
neighbours the agent can already encode, and step downhill. No extra storage,
no second matrix, and it works on a fully binary code.

Its range limit falls out immediately: `H` saturates at `D/2` once positions
decorrelate, so the potential is informative only while `k·m ≪ D/2`. With
`m ≈ 18.4` flips per cell (§10.20) that is `k ≪ D/2m ≈ 28` cells — which is the
arena. Viable here, and it would fail on a much larger one.

**This is the cheapest thing in the document to test**, and it tests on data we
already have: arm B currently has a perfect memory (`exact` 0.999, basin 28.2,
`cos_self` 1.0000) and reach 0.103. If `‖z_goal − z(p+e)‖ − ‖z_goal − z(p)‖` is
reliably negative toward the goal on arm B's codes, then arm B's navigation was
never broken — only its *readout* was, and escape (iii-c) recovers a system with
an exact attractor **and** a direction field.

##### (iii-c) is the descent, constrained to the manifold of realizable codes

Worth stating separately, because it is the reason to expect (iii-c) to work at
all rather than a nice property of it.

The natural question about the memory is why it does not descend an energy
landscape through a sequence of intermediate states, the way a Hopfield network
is supposed to. Three reasons, of increasing importance:

1. **The update is synchronous and `α = 1`.** Classical Hopfield's incremental
   descent is a consequence of updating *one unit at a time*; the monotone-energy
   theorem is proved for asynchronous updates, and synchronous ones can even
   cycle. Updating every coordinate at once, with `α = 1` discarding the current
   state entirely, takes the whole step immediately by construction.
2. **The recall map's image is the span of the memories, not the code
   manifold.** `Wx = s·Zᵀ(Zx) − s·diag·x`, and the diagonal term is smaller than
   the first by `K/D` ≈ 0.5%. So after one step the state lies in the
   `K`-dimensional subspace spanned by the stored goals — it has left the 2D
   surface of position codes altogether. **The intermediate states are mixtures
   of goals, not codes of intermediate positions.** A blend of `z(y₁)` and
   `z(y₂)` is not the code of anywhere between them, so nothing can be decoded
   from it. Lowering `α` makes the descent gradual but does not change this: the
   path still runs through the memory span, transverse to the manifold.
3. **Below the knee, iterating actively destroys information.** `x ←
   normalize(Wx)` is power iteration on a near-degenerate spectrum, so it drifts
   toward a **cue-independent** top eigenvector. That is why more steps makes
   recall worse rather than better (§0), and it is the opposite of pattern
   completion.

> ### ✓ Measured turn 20. Production walks the manifold; arm B cannot.
>
> `alpha_walk_check.py`, 10% `att0.5` s42, K = 5, cues starting a mean 10.46
> cells from the goal, decoding every recall step to its nearest cell **and**
> recording `cos(state, that cell's code)` — the column §6 never had.
>
> **Production (gain 100, β = 100).** Lowering `α` turns the jump into a walk,
> and the walk stays on the manifold:
>
> | α | decoded distance, steps 1→12 | cos to nearest cell |
> |---|---|---|
> | 1.0 | 0.04 · 0.06 · 0.09 · 0.21 · 0.29 · 0.34 | 0.989 → 0.868 |
> | 0.95 | 4.97 · 1.05 · 0.15 · 0.10 · 0.22 · 0.29 | 0.971–0.982 |
> | **0.9** | **7.89 · 3.53 · 1.37 · 0.16 · 0.12 · 0.23** | **0.967–0.982** |
> | 0.8 | 9.39 · 7.72 · 5.43 · 1.91 · 0.42 · 0.13 | 0.970–0.993 |
> | 0.5 | 10.46 · 9.91 · 9.53 · 8.74 · 7.17 · 4.49 | 0.969–0.999 |
>
> At α = 0.9 the state visits 7.89, 3.53 and 1.37 cells from the goal on the way
> in, and **never falls below cos 0.967 of an actual cell's code**. These are
> encoded states, not merely vectors with a nearest cell. `α` is a clean speed
> knob: the walk lengthens smoothly as `α` falls, and at 0.05 it has not left
> the start after 30 steps.
>
> **Arm B (gain 1e6, β = 1e6) cannot do this at any `α`.** Its recall term has
> norm `√D` = 32 against the cue's 1, so it stays recall-dominated down to
> α ≈ 0.03; below that it does not walk, it **stalls and then snaps**:
>
> | α | decoded distance, steps 1→30 | cos |
> |---|---|---|
> | 0.01 | 10.46 · 10.46 · **0.00** · 0.00 · … | 0.979 · 0.934 · **0.919** · 0.973 → 1.000 |
> | 0.003 | 10.46 · 10.46 · 10.46 · 10.46 · 7.45 · **0.00** | 0.998 → **0.906** → 0.998 |
> | 0.001 | 10.46 … 10.46 · **0.00** | 1.000 → **0.925** → 0.930 |
>
> It sits at the *cue's own cell*, then jumps to the goal, and **the cosine dips
> to 0.906–0.93 exactly during the transition** — the state is off the manifold
> while it crosses, and decodes to no position in between.
>
> **That is the chord prediction, confirmed — and the number is exact.**
> `chord_manifold_check.py` walks the chord directly, `x(t) = (1−t)z(here) +
> t·z(goal)`, 23 start cells ~10 from the goal, decoding every point:
>
> | t | 0.0 | 0.2 | 0.4 | **0.5** | 0.6 | 0.8 | 1.0 |
> |---|---|---|---|---|---|---|---|
> | production, decoded cells | 10.03 | 8.50 | 6.44 | **4.94** | 3.77 | 1.57 | 0.00 |
> | production, cos | 1.000 | 0.996 | 0.991 | **0.990** | 0.991 | 0.996 | 1.000 |
> | arm B, decoded cells | 10.03 | 10.03 | 10.03 | **2.69** | 0.00 | 0.00 | 0.00 |
> | arm B, cos | 1.000 | 0.988 | 0.949 | **0.918** | 0.949 | 0.988 | 1.000 |
>
> **Why arm B's chord is off-manifold, exactly.** Blend two hypercube corners.
> On the coordinates where they *agree* the blend keeps magnitude 1; on the
> fraction `f = H/D` where they *disagree* it takes the graded value `|1 − 2t|`.
> So the blend's **sign pattern is that of whichever endpoint dominates** — it
> flips discontinuously at `t = ½`, and there is no intermediate sign pattern,
> hence no intermediate position to decode to. At `t = ½` the disagreeing
> coordinates are exactly zero and
>
> ```
> cos(blend, nearest corner) = √(1 − H/D)
> ```
>
> Measured `H/D` = 0.1568 at this separation, so the prediction is **0.9182**
> against **0.9183** observed. Four decimal places, nothing fitted.
>
> Production's manifold is flat over the range, so its chord is a near-geodesic:
> the decoded position tracks `t` almost linearly (t = 0.5 → 4.94 cells of
> 10.03) and the cosine never drops below **0.990**. **[M]**, one seed.
>
> A third thing fell out, unlooked for: at α = 1 production's cosine **decays
> monotonically 0.989 → 0.729 over 30 steps**. The state drifts *off the
> manifold* as it converges to the cue-independent top eigenvector — §0's
> matched-filter decay, visible as geometry rather than as a reach number.
>
> **Corrected: in decoded position it very nearly *does* stay put.** Calling
> 0.12 → 0.36 cells over 22 steps "moving away" overstated it — that is 2% of a
> 10.46-cell journey, and as a position readout it is arrival. What actually
> degrades is the state's distance **from the manifold**: cos 0.979 → 0.846 at
> α = 0.9, 0.989 → 0.729 at α = 1. At cos 0.846 the state is `√(2·0.154)` =
> 0.555 away from its nearest cell in code space, against a per-cell step
> `‖d₁‖` = 0.086 — **6.5 cells' worth of code distance off the manifold, while
> its projection sits 0.36 cells from the goal.** The drift is transverse, which
> is why the decode barely moves and the readout still suffers: `q` differences
> against `ẑ`, and an off-manifold `ẑ` contributes displacement that corresponds
> to no position at all. That is §0's "the `q` readout decays monotonically with
> recall depth", mechanised.
>
> **So every production row is approach-then-drift, and α only sets the clock.**
> At α = 0.9 the state closes to 0.12 cells by step 8; at α = 1 it closes to
> 0.04 at step 1. The goal is **not a fixed point** of
> production's dynamics — the only fixed point is `W`'s top eigenvector, a fixed
> blend of all `K` stored patterns that does not depend on the cue — so the
> trajectory merely *passes near* its goal on the way there.
>
> And α cannot change that, only the rate. In the linear regime the update is
> `x ← normalize(((1−α)I + αβW)·x)`, and adding a multiple of the identity
> leaves the eigenvectors untouched: **the destination is α-independent, the
> time constant is not.** That is §4.3's "a time constant, not a destroyer",
> derived rather than observed, and it is the same reason α could not rotate `q`
> in §7.1.
>
> The operational consequence is that the best step count moves with α — about
> 1 at α = 1, 5–8 at α = 0.9, ~12 at α = 0.8 — and running past it costs
> accuracy. Production uses `steps = 1` at α = 1, which is the right corner of
> that trade. Note the retreat is a *minority defecting*, not a uniform creep:
> the mean distance rises because a growing fraction of starts decode somewhere
> else, and the `in env` fraction falls with it (α = 1: 1.00 → 0.81 by step 30).
>
> **So the answer to "do we have an attractor whose dynamics pass through
> encoded states" is: we can have one, and it is not arm B.** Production at
> α ≈ 0.8–0.9 traverses the manifold; but its memories are not fixed points, so
> it is not an attractor. Arm B's memories *are* exact fixed points, and it
> cannot traverse.
>
> > **Qualified turn 22 — this is interpolation, not dynamics.** The `(1−α)x`
> > term is not computation; it is retention of the previous state. The network
> > computes `tanh(βWx)`, which is the jump, and α only sets how far toward that
> > answer to move. `chord_manifold_check.py` demonstrates it inadvertently:
> > it contains **no recall at all** — the blend of `z(here)` and `z(goal)` is
> > formed by hand — and it reproduces the same decoded trajectory the α = 0.9
> > recall produces. If straight interpolation gives the same path, the path is
> > not coming from the dynamics. In the linear regime,
> > `x ← normalize(((1−α)I + αβW)·x)` is power iteration on a shifted matrix and
> > the intermediate states are partially-converged iterates toward one fixed
> > point. Nothing moves *along* the manifold; something moves *across* it, and
> > the manifold is flat enough that the shadow looks like a walk.
> >
> > **What would be genuinely different:** the manifold as an *invariant set*,
> > `R(M) ⊆ M` — codes mapping to codes, drifting along the surface toward the
> > goal. That is a continuous attractor network (ring / grid CAN), where the
> > recurrent weights make the manifold itself the attractor set. Ours is an
> > outer-product memory over `K` isolated patterns whose image is the
> > `K`-dimensional span, meeting the manifold only near the memories. **So the
> > honest claim is a *relaxation* that passes through encoded states, not
> > dynamics that traverse them** — and "should the attractor be the code
> > manifold rather than `K` isolated points" is a design fork this campaign has
> > never named.

> **Qualified turn 19 — with `α < 1` on a *continuous* code, it partly does.**
> The claim above is right at `α = 1`, which is production and both saturated
> arms, and PROBE §6 measured it: the first application travels ~10 cells and
> lands on the goal, one step. But §6's `L7 alpha 0.9` row decodes each recall
> step to its nearest cell and gets **8.2 → 6.01 → 3.08 → 1.66 → 0.60** — a
> *walk*, not a jump.
>
> The mechanism to expect, **[G]**: with `α < 1` the state is a blend of the cue
> and the memory span, so it sits on the **chord** from `z(p)` toward `z(y)`.
> A chord lies close to the manifold exactly when the manifold is **flat over
> that range** — which is (J3), the ballistic property. So a continuous code's
> chords are near-geodesics and decode to intermediate positions; a binary
> code's are not, because a blend of two hypercube corners is near no corner at
> all.
>
> **If that holds, the real trade is not the one stated in T1.** It would be:
> *an exact fixed point (binary code, dynamics jump) versus dynamics that
> genuinely traverse encoded states (ballistic code, memories are not fixed
> points)* — with (iii-c) giving navigation in either case. That is a sharper
> and more interesting fork than "attractor versus direction field", and it is
> untested.
>
> **The measurement that would settle it** is one column §6 does not report:
> `cos(state, code of its nearest cell)` at each step. "Decodes to a cell 3.08
> from the goal" is not "is the encoded state of a position 3.08 from the goal"
> — a vector far off the manifold still has a nearest cell. Distinguishing those
> is exactly the kind of gap that has bitten this campaign repeatedly.

But a descent does exist, and it is the same idea with one constraint added.

> **Be careful with the word "energy".** The Hopfield energy is
> `E_H(x) = −½ xᵀWx`, a function of the **state vector** `x ∈ R^D`. What
> (iii-c) descends is `E(p) = −⟨ẑ, z(p)⟩`, a function of **position**. They are
> not the same object, and an earlier draft of this section called them the same
> thing. `E` is the single coupling term of `E_H` between the state and one
> stored pattern, re-read as a field over space. (Restricting `E_H` itself to
> the code manifold gives `−(1/2D)·Σ_k C(p − y_k)²`, which is minimised at
> *whichever* stored goal is nearest — including another environment's. That is
> the wrong target, so `E` is what we want, but it is not `E_H`.)

The right statement is about **where the descent is allowed to go**. Classical
Hopfield dynamics move the state toward the nearest memory through all of `R^D`,
and §7.1's first point is that they immediately leave the 2D surface of position
codes into the `K`-dimensional span of the memories, which is why nothing can be
decoded from their intermediate states.

(iii-c) runs the same "move toward the memory" descent **constrained to the
manifold of realizable codes** — because the agent can only be at real
positions, and `z(p)` is the only code it can occupy. It is a projected descent,
and the projection is physical rather than imposed. That constraint is exactly
what makes the intermediate states meaningful: the trajectory cannot leave the
manifold, so every point on it is a place the agent actually is. **The network
never has to produce intermediate codes, because the agent's own trajectory
supplies them.**

Two things this does not paper over. First, "descent" on a lattice means a
finite difference, the same as the current readout — take central differences
for a 2-vector and follow it (`continuous_flow`), or just step to the best
neighbour (`discrete_flow`); the harness supports both, and they are different
algorithms with different failure modes. Second, for a binary code the potential
is quantised in units of `2/D`, and a one-cell step moves it by about `m` = 18
of those — well resolved, not near the quantisation floor. What is *not*
established is whether the potential is monotone **cell by cell** or only **on
average**: §10.20 shows the mean `H(k)` is near-linear out to `k ≈ 16`, but its
fluctuation was never measured, and if the per-step fluctuation rivals the
per-step drift then a greedy walk is a biased random walk rather than a descent.
That is the same gap as the local-maxima one below, and Stage 0's `acc45`
measures it directly.

Which also says what the test has to look for. The failure mode of a potential
descent is **local minima**, and a local minimum of `E(p)` is precisely a
position whose code is spuriously similar to the goal's — an alias. So the
question "does (iii-c) work" is the question "how many local minima does
`−⟨z_goal, z(·)⟩` have", and `discrete_flow` already records sinks and limit
cycles, which are exactly those.

##### How we would actually do (iii-c)

**The readout, exactly.** At one position `p` (local cell coordinates), in an
env with scaffold `offset`, `e_E = (+1, 0)` and `e_N = (0, +1)`:

1. **Encode the agent's own position.**
   `z_p = field.encoded_state(p, offset)` — unit, `(D,)`. This is the cue, and
   it is the same call the current readout already makes.
2. **Recall.** `ẑ = recall_trajectory(mem, z_p, steps, cfg)[s]`. Unchanged from
   today, and unit because the recall normalises at every step.
3. **Encode the four neighbours.**
   `z_n = field.encoded_state(p + n, offset)` for `n ∈ {+e_E, −e_E, +e_N, −e_N}`
   — four calls where the current readout makes two (`+e_E`, `+e_N`, inside
   `local_basis`). These are *scaffold* neighbours: the code is defined
   everywhere, so a neighbour behind a wall is still encodable, exactly as
   `local_basis` already assumes.
4. **Four dot products.** `s_n = ⟨ẑ, z_n⟩`. Normalise defensively first — the
   probe's `_unit` — since a non-unit `ẑ` leaves the bearing alone but corrupts
   step 6.
5. **Central differences, in the order `project_q` already returns.**

   ```
   q = ( (s_{+E} − s_{−E}) / 2 ,  (s_{+N} − s_{−N}) / 2 )      # (East, North)
   ```

   The `/2` is the central-difference spacing, not a normalisation. Nothing is
   orthonormalised and there is no Gram–Schmidt: `q` is already a 2-vector in
   the (East, North) frame by construction.
6. **Distance, separately.** `s_0 = ⟨ẑ, z_p⟩` from steps 1 and 2, no extra
   encode. On a binary code this is the linear distance readout and `‖q‖` is
   not; see the note above on not mixing them up.

Vectorised over an env, which is how `cell_q_field` would call it:

```
cells = local_cells(size)                              # (n, 2)
cues  = field.encoded_state(cells, offset)             # (n, D)
Zhat  = _unit(recall_trajectory(mem, cues, steps, cfg)[s])
S     = {n: np.einsum("id,id->i", Zhat,
                      _unit(field.encoded_state(cells + n, offset)))
         for n in (+e_E, -e_E, +e_N, -e_N)}
q     = 0.5 * np.stack([S[+e_E] - S[-e_E], S[+e_N] - S[-e_N]], axis=1)
```

**Cost.** Four encodes of `n` cells instead of two, plus `4n` dot products,
minus the Gram–Schmidt and the projection. Recall is untouched and is the
expensive part, so the suite cost barely moves.

**One boundary case to assert, not assume.** `encoded_state` *clips* to
`[0, Npos−1]`, so a neighbour off the scaffold silently returns `z_p` itself and
the central difference degenerates into a one-sided one — which is (∗) without
its correction term, the exact failure this readout exists to avoid. At
`Npos = 1716` with envs placed well inside it should never fire; assert rather
than trust it.

**Two switches to run both ways rather than choose.** Whether `ẑ` is one recall
step or iterated to the fixed point (see below); and whether the flow follows
the 2-vector (`continuous_flow`) or steps to the best of the four neighbours by
`s_n` directly (`discrete_flow`) — the second does not need `q` at all, and on a
lattice it is the more faithful descent.

**What it removes.** No local frame, no Gram–Schmidt, and — the point — **no
local-linearity assumption**. The current readout needs `z` to vary
proportionally with displacement over the whole operating range, which is (J3),
which is what binarisation destroys. (iii-c) needs only that `⟨ẑ, z(·)⟩` be
*monotone* in distance. That is a far weaker requirement and a binary code
satisfies it.

**It gives two signals where the current readout gives one and a half.**
Bearing comes from the gradient; **distance comes from `E(p)` itself**, which is
directly monotone in `‖p − y‖`. That is worth noting because it is not a
consolation prize: today distance is supposed to come from `‖q‖`, and `‖q‖` is
the least trustworthy thing the readout produces. Note the two codes behave
oppositely here — for a binary code `1 − C(k) ∝ k`, so the *slope* is constant
and carries no distance information at all, while `E` itself is a clean linear
distance readout; for the continuous code it is the other way round. So on a
binary code (iii-c) must take distance from the value and bearing from the
gradient, and not mix them up.

##### In what sense is that `q` a gradient? — and the answer explains arm B exactly

Both readouts are finite differences of the **same scalar field**
`s(p) = ⟨ẑ, z(p)⟩`. Three lines show it. With `d_i = z(p+e_i) − z(p)` and unit
codes,

```
⟨d_i, ẑ⟩    = s(p + e_i) − s(p)
⟨d_i, z(p)⟩ = C(1) − 1
```

so the **current** readout's component along axis `i` is

```
q_i^now  =  ⟨d̂_i, ẑ − z(p)⟩  =  [ s(p+e_i) − s(p)  +  (1 − C(1)) ] / ‖d_i‖      (∗)
```

— a **forward** difference of `s`, plus a constant, over `‖d_i‖`. And (iii-c) is

```
q_i^new  =  [ s(p+e_i) − s(p−e_i) ] / 2                                          (∗∗)
```

— a **central** difference of the same `s`. Both estimate `∂s/∂p_i`, and both
vanish at the goal. That is the sense in which either one steps downhill in
`E = −s`: a vector of per-axis finite differences of a scalar *is* the discrete
gradient, and moving along it increases `s`.

**So why does (∗) fail on a binary code when (∗∗) should not?** Because the
constant `(1 − C(1))` in (∗) is a correction calibrated for one particular shape
of `C`, and binarisation changes the shape.

Write the goal at displacement `r`, bearing `θ`, and use `C` as a function of
distance:

| | `1 − C(j)` | `s(p+e_i) − s(p)` | **`q^now ∝`** | **`q^new ∝`** |
|---|---|---|---|---|
| continuous (ballistic) | `a·j²` | `a(2r_i − 1)` | `2a·r_i` ✔ | `2a·r_i` ✔ |
| binary (diffusive) | `(2m/D)·j` | `(2m/D)·cos θ_i` | `(2m/D)(1 + cos θ_i)` ✘ | `(2m/D)·cos θ_i` ✔ |

For a **quadratic** profile the `+(1 − C(1))` exactly cancels the `−1` thrown
off by expanding `|r − e_i|²`, and (∗) returns the true gradient. For a
**linear** profile there is nothing for it to cancel, and it survives as a
constant added to **both** components:

```
q^now  ∝  (1 + cos θ,  1 + sin θ)          instead of   (cos θ, sin θ)
```

which is not a scaling — it is a translation of the 2-vector, and it wrecks the
bearing. Measured on arm B, the constant is 0.0358 against a signal of sd 0.0164:
**2.2× larger than what it is added to.** (The idealised form above omits the
Gram–Schmidt reduction of the East axis, which partly cancels the shared bias —
see the retraction note below.) The **central** difference in (∗∗) never picks up the constant in the
first place, because `s(p+e) − s(p−e)` cancels the self-term identically.

**This predicts arm B's number.** The bearing of `(1 + cos θ, 1 + sin θ)` is
within 45° of `θ` only for `θ ∈ (−30°, 120°)` — a 150° window out of 360°.
`readout_offset_check.py` evaluates (∗) and (∗∗) on the exact profiles, with no
encoder and nothing fitted:

| | current (∗) | (iii-c) (∗∗) | measured |
|---|---|---|---|
| continuous code | **1.000** | 1.000 | 0.995 (production) |
| binary code | **0.423** | **1.000** | 0.392 (arm B) |

and the pure-offset window argument gives 0.417 against the profile evaluation's
0.423. It reproduces the third anchor too: goal due north, binary code, (∗)
equals `√(4m/D)` = **0.2681** independent of distance, against §10.20's measured
`q_north` of **0.267**, flat over k = 1…8. **[D vs M]**

So the answer to "why should (iii-c) work" is no longer an analogy. **Arm B's
direction field fails by a specific, identified term, and the central difference
removes exactly that term.** The prediction is that (iii-c) takes arm B's `acc45`
from 0.392 to near 1 — which is Stage 0, and it is now a falsifiable number
rather than a hope.

> **Caveat on the caveat.** This says the *systematic* part of arm B's failure
> is the offset. It says nothing about the noise, the local maxima, or whether
> `C` is really linear-in-distance and isotropic for this code rather than
> linear along an axis, which is all §10.20 measured. Those remain the Stage 0
> questions below.

**Why expect it to work at all?** Three further reasons, and one gap.

*1. The basin measurement is already a statement about this exact potential.*
`basin_probe` asks whether `argmax_p ⟨ẑ, z(p)⟩` over every cell of a disc is the
goal cell. That argmax **is** the minimum of `E`. So arm B's basin of 28.2 says
precisely: for essentially every cue within 28 cells, the potential (iii-c)
would descend has its **global optimum at the goal**, over a menu of 12,853
candidate positions. We are not hoping the landscape has the right minimum; we
measured that it does, 28 times over, before ever thinking of this readout.
**[M]**

*2. The predicted range and the measured basin agree.* The model says `E` goes
flat once Hamming distance saturates, at `k ≈ D/2m` with `m ≈ 18.4` flips per
cell — **27.8 cells**. Arm B's basin is **28.2**. The basin ends exactly where
the potential stops carrying information, which is what the picture predicts and
a place it could easily have failed. **[M]**

*3. The property (iii-c) needs is not the property that broke.* Binarisation
destroys **proportionality** — `‖Δk‖` going as `√k` rather than `k`. It leaves
**monotonicity** intact: `H(k)` still increases with `k`. (iii-c) asks only for
monotonicity. And the per-cell signal is *larger* after binarisation, by about
9× near the goal — the drop in similarity per cell is 0.036 binarised against
0.0037 continuous (§10.20's `‖d_fwd‖` 0.267 vs 0.086). Whether the
signal-to-noise is also better depends on the roughness of the code, which is
not something we can argue and Stage 0 measures directly. Worth noting the
shapes are opposite: the binary potential has a **constant** slope all the way
in, while the continuous one's slope **vanishes at the goal** — and §3.4 showed
the terminal neighbourhood is where failures are fatal.

*The gap, stated plainly.* A verified global optimum is **not** a verified
landscape. `argmax` being correct says nothing about local maxima between the
cue and the goal, and a greedy walk stalls at one. **We have no evidence either
way** — the campaign has only ever measured the argmax. That is the thing that
could kill this, and it is why Stage 0 below counts local maxima directly rather
than waiting for the flow to reveal them.

**How to test it, in three stages, each against a number we already have.**

> ### ✓ Stage 0 ran, 2026-09-08. It passes.
>
> `potential_readout_check.py`, two training seeds per arm, 15,960 cells each,
> `ẑ` taken from the memory rather than ground truth, and the **current** readout
> computed on the identical memory as a control.
>
> | | current | **(iii-c)** | published |
> |---|---|---|---|
> | production s42 | 0.997 | **1.000** | 0.995 |
> | production s43 | 0.995 | **0.997** | 0.995 |
> | **arm B** s42 | 0.389 | **0.998** | 0.392 |
> | **arm B** s43 | 0.393 | **0.993** | 0.392 |
>
> The control reproduces both published numbers to ±0.003, so the comparison is
> against the real system. **Arm B's `acc45` goes 0.392 → 0.998**, and `|err|`
> goes 66.5° → 11.8°. The predicted failure mode does **not** bind: sinks
> (cells with no better neighbour) are **0.000–0.002**, and the greedy step
> moves closer to the goal in real space **0.998–1.000** of the time. It holds
> across every distance band, including 18–30 cells where the current readout
> is worst (0.281 → 0.986).
>
> And it is not a special case for binary codes: production improves slightly
> too, 0.997 → 1.000 and 0.995 → 0.997. Arm B's 11.8° remains coarser than
> production's 8.3°, which is the discrete field's residual noise.
>
> **Two seeds, not four** — but the effect is 0.39 → 0.99 against a seed spread
> of 0.004, so the ordering is not in question.
>
> ### ✓ Stage 1 ran too. Arm B goes from reach 0.103 to 0.984.
>
> Same `q` fields, pushed through `continuous_flow` and `discrete_flow`
> **unmodified**, on Test D's memory draw so the control has to reproduce the
> published reach. 40 envs per arm.
>
> | | reach (cont) | reach (disc) | sinks/env | published |
> |---|---|---|---|---|
> | arm B, current | 0.100 / 0.103 | 0.230 | 8.62 / 8.97 | **0.103** |
> | **arm B, (iii-c)** | **0.984 / 0.978** | **0.998** | **0.10 / 0.20** | |
> | production, current | 0.993 / 0.987 | 0.997 | 0.00 / 0.17 | **0.987** |
> | production, (iii-c) | 0.967 / 0.997 | 0.999 | 0.00 / 0.07 | |
>
> The control lands on 0.103 and 0.987 exactly. **Arm B's reach goes 0.103 →
> 0.984**, and the sink count collapses from ~9 per env to ~0.15.
>
> > **Retracted: "the readout can only point north-east."** An earlier note here
> > said both components of `q` are non-negative for every bearing, so the
> > heading is confined to [0°, 90°]. True of the idealised algebra, **false of
> > the code**: `gram_schmidt_2d_batch` keeps *North* exactly and *reduces East*
> > against it, which partly cancels the shared bias. Measured on arm B, the
> > identity is exact for the North component (max |diff| 2.5e−05, corr
> > 1.00000000) and approximate for East (corr 0.976); `q_North` ≥ 0 at **98.0%**
> > of cells, `q_East` ≥ 0 at 66.2%, and **64.2%** emit a north-east heading
> > against 25% by chance. The bias is real and large — the constant is 0.0358
> > against a signal of sd 0.0164 — but it is a strong tilt, not a hard
> > restriction.
>
> > **"Sink" here is not a spurious memory.** Two different objects have been
> > called attractors in this document, and only one of them is arm B's. A
> > *spurious memory* is a fixed point of the recall dynamics in `R^D` that is
> > not a stored pattern — and arm B has essentially none, which is the whole
> > point of saturating: `cos_self` = 1.0000, exact retrieval 0.999, basin 28.2.
> > `discrete_flow`'s **sinks** are something else entirely: single-cell
> > terminal states of the *agent's walk on the arena grid*, cells whose `q`
> > points into a wall so the agent stops. They are a property of the readout,
> > not of the memory.
> >
> > And the closed form says why arm B had ~9 of them. With a binary code the
> > old readout gives `q ∝ (1 + cos θ, 1 + sin θ)`, and **both components are
> > non-negative for every θ** — `1 + cos θ ≥ 0`, `1 + sin θ ≥ 0`, never both
> > zero. So the heading always lies in the closed **first quadrant**: the
> > readout is structurally incapable of telling the agent to go south or west.
> > Every agent walks north-east regardless of where its goal is, piles into the
> > north or east wall, and stops. Nine sinks in a 400-cell arena is that
> > pile-up, and it is the same single defect that gives `acc45` = 0.39. The
> > central difference carries no offset, so its `q` spans all four quadrants —
> > which is why one change fixes the bearing and the sinks together.
>
> **What that gives.** Arm B already had `cos_self` = 1.0000, exact retrieval
> 0.999, and the largest basin measured, 28.2. With this readout it also
> navigates. Against production — reach 0.987, basin 27.0, retrieval 0.982, no
> attractor — arm B plus (iii-c) matches the reach, beats the basin and the
> retrieval, and is step-invariant. **It is the first configuration in this
> campaign that is a genuine attractor network *and* navigates**, which §7.1
> said required escaping one of T1's premises. It escapes premise (iii).
>
> **Reported against itself: on the *continuous* code this is a wash, not a
> win.** Production s42 goes 0.993 → 0.967 — a real drop — while s43 goes 0.987
> → 0.997, and s42's per-cell `acc45` was *better* under (iii-c) (1.000 vs
> 0.997). Better bearings with worse reach means the losses are limit cycles
> rather than wrong headings, and the cycle count moved 0.00 → 0.03 per env on
> exactly that seed. So (iii-c) is transformative where the code is binary and
> neutral-to-slightly-worse where it is not; it is a fix for arm B, not an
> upgrade for production.
>
> Two seeds. Stage 2 — the policy path, where `‖q‖` no longer means what the
> magnitude gate was fitted on — is untouched.

*Stage 0 — the field and the landscape, offline.* On arm B's checkpoint
(encoder gain 1e6, β = 1e6), and production as the control, compute the (iii-c)
`q` at every cell of every scored env, with `ẑ` the **retrieved** code rather
than the true goal code so the memory stays in the loop. Report `|err|` and
`acc45` by distance band, which is exactly Test B's output: the comparison is
arm B's **acc45 0.392** and production's **0.995**.

And — because this is the gap above, not an afterthought — **census the local
maxima in the same pass**: count the cells that are not the goal and whose four
neighbours all have lower `⟨ẑ, z(·)⟩`. That is a handful of comparisons per
cell, it needs no flow simulation, and it is the direct measurement of the one
thing we have never looked at. If either number is bad, stop here.

*Stage 1 — the flow.* Feed that `q` field to `continuous_flow` and
`discrete_flow` unchanged. Reach is then directly comparable to arm B's
**0.103** and production's **0.987**, and `discrete_flow`'s sink and limit-cycle
counts are the local-minimum census the mechanism predicts is the binding
failure. This is the whole result: a system with `cos_self` = 1.0000, basin
28.2, and a working direction field would be the first one in the campaign that
is genuinely an attractor network *and* navigates.

*Stage 2 — the policy, only if 0 and 1 pass.* The production `VectorHash` needs
a sibling to `project_displacement`. The interface is unchanged — the policy
still consumes a 2-vector — but `‖q‖` no longer means what it did, so the
magnitude gate (`EXPERIMENTS_NAV_TRI.md`) has to be re-fitted or re-pointed at
`E`. That is the only part of this that touches the production contract.

In the probe the seam is small: `qfield.project_q(basis, current, recalled)`
gains a sibling `potential_q(field, cells, offset, recalled)`, and
`cell_q_field` chooses between them. Everything downstream — Tests B, C, D, the
report — is untouched.

**One design question worth testing both ways.** Does `ẑ` come from a single
recall step, as now, or from iterating to the fixed point? Arm B is a genuine
attractor, so iterating is free and should *help* — cleaning up a corrupted cue
is the thing an attractor is for, and it is the first time in this campaign that
running more steps could be expected to improve anything.

**Three ways it fails, each with the measurement that catches it.**

1. **Local minima (aliases).** The predicted binding failure. Caught by
   `discrete_flow`'s sink census at Stage 1, and visible in Stage 0 as clusters
   of large `|err|` at particular positions rather than a uniform degradation.
2. **A one-cell difference too noisy to resolve.** The signal is `2·C′(k)` and
   the noise is the cell-to-cell roughness of the code. No need to model it —
   Stage 0's `acc45` measures it directly.
3. **Range.** `E` is flat beyond `k ≈ D/2m ≈ 28` cells, so there is no gradient
   at all out there. Fine for a 20×20 arena, fatal for a much larger one, and
   Stage 0's by-distance-band table shows exactly where it dies.

**(iii-d) A learned decoding head.** `(z_here, z_goal) → q` as a trained map,
dropping the local-linearity requirement entirely. Most general and least
attractive: the current readout is parameter-free and environment-agnostic, so
it transfers zero-shot to arenas and goals never seen. A head has to be trained
on `(position, goal)` pairs and inherits that distribution. Only worth it if
(iii-b) and (iii-c) both fail.

---

### T2. A closed form for the exact basin radius — Q2

**Status: derivable now; the sketch below already lands within ~30% of the
measured value.** This is §5.2's (J2) solved for `r`.

At radius `r` the recalled state is `x ∝ C(r)φ(y) + Σ_{k≠y} C(p−y_k)φ(y_k)`, so
the cross-talk residual relative to the signal has norm

```
‖ε‖ ≈ √(K−1) / (√d_eff · C(r))
```

— `K−1` competitors, each entering with a far-field coefficient of order
`1/√d_eff`, against a signal of weight `C(r)`. Localisation fails when `ε`'s
component along `φ(y±e) − φ(y)` exceeds half that vector's squared length,
which is `1 − C(1)`. Taking the angle between `ε` and that direction as generic
in `d_eff` dimensions gives `⟨ε, ·⟩ ≈ ‖ε‖·N(1)/√d_eff`, and the condition
`‖ε‖N(1)/√d_eff < N(1)²/2` collapses to

```
C(r*)  =  2√(K−1) / (d_eff · N(1))                                   (T2)
```

**A closed form for the basin in terms of `d_eff`, `K` and the near field
alone.** Evaluated for the production encoder — `d_eff` = 297, `K` = 5, res90 =
7 so `1 − C(7) = 0.1`, and ballistic `1 − C(k) ∝ k²` giving `1 − C(1) = 0.00204`
and `N(1) = 0.0639`:

```
C(r*) = 4 / (297 × 0.0639) = 0.211   ⇒   1 − C(r*) = 0.789
r*² = 49 × 7.89                      ⇒   r* ≈ 19.7 cells
```

against a **measured 27.0** (§10.18, four seeds). Same order, right ballpark,
from a chain with three order-unity constants dropped. Doing it properly —
keeping the constants, and using the measured `C(·)` rather than a ballistic
extrapolation — is a page of algebra and is the single most checkable item here:
it must reproduce **27.0 / 23.0 / 19.2 / 11.5 / 13.5** across the coverage
ladder, and the `√(K−1)` scaling must hold against the K = 1/3/5/10/20 columns
the probe already stores.

**Falsified by:** the ladder not tracking `d_eff · N(1)`, or the K-dependence
not being `√(K−1)`.

---

### T3. The far-field law `sd(C) = 1/√d_eff` — Q4

**Status: three lines; needs a numerical check of one assumption.** §2.0 states
it; here it is.

Write `φ(p) = Σ_ω a_ω e^{iω·p}` and `P(ω) = ‖a_ω‖²` with `Σ P = C(0) = 1`. Then
`C(a) = Σ_ω P(ω) cos(ω·a)`. For generic `a` the phases `ω·a` are
equidistributed and effectively independent across frequency orbits, so `C(a)`
is a weighted sum of random signs:

```
sd(C_far) = √(Σ_ω P(ω)²) = 1/√PR(P)
```

The remaining step is `PR(P) = d_eff`: the covariance is
`Σ = Σ_ω a_ω a_ω^†`, whose eigenvalues are the `P(ω)` **provided distinct
frequencies map to orthogonal output directions**. That is the one assumption,
and it is checkable directly.

**Why it matters beyond tidiness:** it makes `d_eff` a *spectral* quantity, so
every statement in §2 about "the one variable" becomes a statement about `P`,
which is what T4 and T5 need.

**Falsified by:** `⟨a_ω, a_ω'⟩` not small for ω ≠ ω′, or the measured far-cos sd
departing from `1/√PR(P)` once `PR(P)` is computed spectrally rather than from
the covariance.

---

### T4. When retrieval names the wrong goal — where the 0.25 comes from — Q2, Q4

**Status: tractable, and the highest payoff; also the hardest of the six.**

§10.3: an environment dies when one co-stored competitor exceeds
`cos(φ(y), φ(y_j)) ≈ 0.25`. The competitor enters the readout as an
approximately **constant bias** — for `p` near `y`, `C(p − y_j) ≈ c_j`
independent of `p` — while the signal `B(p)(φ(y) − φ(p))` grows like
`‖r‖·N(1)`. So the crossover is a bias-versus-signal comparison in the same
first-order expansion as T2, and the threshold should come out as a function of
`d_eff`, `K` and `R_op` rather than as a fitted constant.

Doing this gives a **predicted dead-goal rate from `d_eff` and `K` alone**,
which removes the probe from the encoder-selection loop entirely — currently
the most expensive step in the campaign. It is PROBE §10.11's open question 2,
now with (J1)–(J3) to hang it on.

**Falsified by:** the threshold moving with `d_eff` when the derivation says it
should not, or vice versa. The six-arm attract ladder is the test set.

---

### T5. What spectrum maximises `d_eff` at fixed chart length — Q3, Q5

**Status: a clean variational problem; the answer decides whether the grid
input is the right one.** This is the optimality question of §5.5-IV.

With `C(a) ≈ 1 − ¼‖a‖²⟨ω²⟩` for small `a`, res90 is exactly `0.63/ω_rms`. So
"maximise `d_eff` at fixed res90" is

```
maximise PR(P)   subject to   Σ P(ω)ω² fixed,  P ≥ 0,  ΣP = 1
```

Stationarity gives `P(ω) = max(0, ν − μ‖ω‖²)` — an **inverted parabola on a
disc**, not a shell. That is worth stating plainly: a *grid module is a thin
shell*, so under this objective the grid code is **not** the optimum, and the
factor of 4–6 between the measured `d_eff·res90²` (14.6k–25.7k over the attract
ladder) and the moment-bound ceiling (~94k) is a candidate explanation.

The honest version replaces the moment constraint with the real one —
`C(a) ≥ 0.9` for all `‖a‖ ≤ res90` — which makes it an **extremal problem for
positive-definite functions** (the Turán/Beurling–Selberg family), where
technique exists.

**Falsified by:** measuring `P(ω)` for a trained encoder and finding it already
disc-shaped with a parabolic taper, in which case the gap is elsewhere.

---

### T6. The minimum dimension — Q3, Q4

**Status: the topological half is immediate; the monotone half is real work.**

*Injectivity.* A 2-torus does not embed in `S²` at all — the only compact
surface embeddable in `S²` is `S²`. It does embed in `S³`, as the Clifford
torus, which **is** one grid module with continuous phase. So `D ≥ 4` for a
continuous injective code, and the grid module attains it. That is the crisp
answer to "3D cannot do this", and it also says the module is dimension-optimal
for its own period.

*Monotone similarity.* Requiring `C` to depend only on `‖a‖` and to be strictly
decreasing is much stronger. `C` must be positive-definite, so `P ≥ 0`, and
rank is `#{ω : P(ω) > 0}`. Isotropic building blocks `Σ_{‖ω‖=ρ} cos(ω·a)` are
Bessel-like and **oscillate**, so monotonicity requires superposing enough
shells to cancel the oscillations — which is where a real lower bound on `D`
would come from, and it will be far above 4. Schoenberg's characterisation of
positive-definite functions on spheres is the tool.

**Falsified by:** exhibiting a low-rank isotropic monotone `C` on `(Z_L)²`.

---

### What is *not* analytic — Q7

No route, and that is the point. Everything in §5.5-III — which loss, what
coverage, what sampling, whether an equivariant architecture helps — is a
question about what an optimiser *reaches*, not about what exists. T1–T6 say
which codes are admissible and which are optimal; none of them says that
gradient descent on `mse_attract_repel` with `rate_lambda` on finds one. That
gap stays empirical, and it is where §2's heuristics keep their value.

The one lever theory has on it: if T5 says the optimal spectrum is a disc with
a parabolic taper, then "does the loss produce that?" becomes a **measurement
of a predicted object** rather than another sweep.

---

## Conversation log

**Turn 1 — the core heuristics.** Assembled §1 and §2 from the probe log.
Finding while assembling: §3.1 — the basin has never been measured along the
axis the reach heuristics tell us to move.

**Turn 2 — three corrections from Jack.** (i) "`d_eff` is the only training
variable" overstated — it is a sufficient statistic for the far field only, and
the similarity structure is the other half; §2.0 rewritten around the power
spectrum, which makes near and far field two functionals of one object rather
than one number. (ii) R4 was unreadable; split into four statements. (iii) Do
not assume basin and reach share a variable — measured instead, §3.2, and found
the basin metric mixes a cross-talk term with a precision term. Bug found and
fixed on the way (§3.3).

**Turn 17 — Stage 1, run.** Same `q` fields through the unmodified flows, Test
D's memory draw so the control reproduces the published reach — it lands on
0.103 and 0.987 exactly. **Arm B's reach goes 0.103 → 0.984**, discrete 0.998,
and the flow's spurious attractors collapse from ~9 per env to ~0.15. Arm B plus
(iii-c) therefore has `cos_self` 1.0000, exact retrieval 0.999, basin 28.2 *and*
reach 0.98 — the first configuration in the campaign that is a genuine attractor
network and navigates, escaping T1's premise (iii). Reported against itself: on
the **continuous** code it is a wash, s42 going 0.993 → 0.967 with a *better*
per-cell acc45, so those losses are limit cycles rather than headings. A fix for
arm B, not an upgrade for production.

**Turn 16 — Stage 0, run.** `potential_readout_check.py`. The control
reproduces both published numbers to ±0.003 (production 0.995, arm B 0.392), and
**arm B's `acc45` goes 0.392 → 0.998** under the central-difference readout,
`|err|` 66.5° → 11.8°. The predicted binding failure does not bind: sinks
0.000–0.002, greedy step moves closer 0.998–1.000. Holds in every distance band,
including 18–30 cells where the old readout is worst (0.281 → 0.986), and
production improves slightly too, so it is not a binary-code special case. Two
seeds; Stage 1 not yet run.

**Turn 15 — "tell me exactly how you will find `q` at a position `p`."** §7.1's
readout written as a six-step spec pinned to the existing API — `encoded_state`
for the cue and the four neighbours, `recall_trajectory` unchanged, four dot
products, one central difference per axis in `project_q`'s own (East, North)
order, and `s_0 = ⟨ẑ, z_p⟩` for distance at no extra encode. No Gram–Schmidt and
no orthonormalisation: `q` is already in the (East, North) frame. Plus the
vectorised form `cell_q_field` would use, the cost (four encodes where there
were two, recall untouched), the one boundary case that must be *asserted* —
`encoded_state` clips, so an off-scaffold neighbour silently degenerates the
central difference into the one-sided form this readout exists to avoid — and
the two switches to run both ways rather than pick.

**Turn 14 — "I don't get why specifically that equation for `q` is stepping
down in energy."** The best question of the conversation, because working it out
turned (iii-c) from an analogy into a closed-form prediction. Both readouts are
finite differences of the *same* field `s(p) = ⟨ẑ, z(p)⟩`: the current one is a
**forward** difference plus `(1 − C(1))`, over `‖d_i‖`; (iii-c) is a **central**
difference. That constant is a correction calibrated for a **quadratic**
similarity profile — it cancels exactly when `1 − C(j) ∝ j²` and survives as an
additive term in *both* components when `1 − C(j) ∝ j`, which is what
binarisation produces. So on a binary code the current readout returns
`(1 + cos θ, 1 + sin θ)` instead of `(cos θ, sin θ)` — a translation of the
2-vector, which destroys the bearing while leaving the magnitude plausible.
`readout_offset_check.py` reproduces **three** measured anchors with nothing
fitted: arm B's acc45 0.392 (predicts 0.423), production's 0.995 (predicts
1.000), and §10.20's flat `q_north` 0.267 (predicts `√(4m/D)` = 0.2681). And it
predicts (iii-c) gives **1.000** on the binary code. Arm B's direction field
fails by an *identified term*, and the central difference removes exactly it.

**Turn 13 — "but how is this going down the gradient?"** The "energy descent
moved into physical space" framing was loose and is corrected in §7.1. `E(p) =
−⟨ẑ, z(p)⟩` is **not** the Hopfield energy — that is `−½xᵀWx` over the state
vector; `E` is its single state-to-pattern coupling term re-read as a field over
space. (Restricting `E_H` to the code manifold gives `−Σ_k C(p−y_k)²/2D`, which
targets *whichever* goal is nearest, including another environment's — the wrong
target.) The correct statement is about **where the descent may go**: classical
dynamics move toward the memory through all of `R^D` and leave the code manifold
at once; (iii-c) runs the same descent **constrained to the manifold**, because
the agent can only be at real positions. A projected descent whose projection is
physical, which is precisely why its intermediate states mean something. Also
recorded: on a lattice "descent" is a finite difference either way
(`continuous_flow` vs `discrete_flow` are different algorithms), the potential
is quantised in units `2/D` with a one-cell step moving it ~18 quanta, and
whether it is monotone *cell by cell* rather than *on average* is unmeasured.

**Turn 12 — "why do we think this should work?"** Three reasons written into
§7.1, and one gap that had been glossed. The strongest: `basin_probe` *is* a
measurement of this potential — its argmax over the disc is `E`'s minimum — so
arm B's basin of 28.2 already says the landscape's global optimum is the goal
over 12,853 candidates, measured before this readout was conceived. Second: the
model's predicted range `D/2m` = 27.8 cells matches that 28.2. Third:
binarisation destroys proportionality but not monotonicity, and (iii-c) needs
only the latter — with a per-cell signal ~9× larger near the goal, though the
SNR is not something we can argue. **The gap:** a verified global optimum is not
a verified landscape, and nothing in the campaign has ever looked for local
maxima. Stage 0 now censuses them directly instead of waiting for the flow.

**Turn 11 — how we would actually do (iii-c).** The subspace point added to §0
in one sentence. §7.1 gains a concrete design: a central difference of the
scalar `⟨ẑ, z(·)⟩` over the four neighbours, which removes the frame, the
Gram–Schmidt and the local-linearity assumption, and needs only *monotonicity*
of similarity in distance. Bearing from the gradient, **distance from `E` itself**
— and on a binary code those must not be mixed up, since `1 − C(k) ∝ k` makes
the slope constant and the value linear. Three test stages, each against an
existing number (arm B acc45 0.392, reach 0.103; production 0.995, 0.987), with
the probe seam being a sibling to `project_q` and nothing downstream touched.

**Turn 10 — "why is the attractor not producing intermediary codes? I thought
that's how a Hopfield network works — lowering energy iteratively."** Three
reasons, and the third is the useful one: the update is synchronous with α = 1;
the recall map's image is the `K`-dimensional span of the memories rather than
the 2D code manifold, so intermediate states are *mixtures of goals* and not
codes of intermediate positions; and below the knee iterating is power iteration
toward a cue-independent eigenvector, which destroys information. The payoff:
the energy descent does exist, over **position** rather than over the state —
minimising `‖z_goal − z(p)‖` is minimising `−⟨z_goal, z(p)⟩` — so (iii-c) *is*
that descent, performed by moving the agent. Its failure mode is therefore local
minima, i.e. aliases, which `discrete_flow` already records as sinks.

**Turn 9 — the case, for someone outside the project.** §0 added at the top.
Jack drafted it; the revision adds what the system is *for* (the document had
no motivation section at all — the same gap), a "why it matters" line on each
question, and a specific closing (scoring one encoder costs a training run plus
an evaluation suite) in place of a generic one. The hedge on the
attractor/direction incompatibility is **kept deliberately**: what §7.1 proves
is the narrow form, and §0 states the strong one.

**Turn 8 — "what would escape (iii) actually look like?"** §7.1 gains four
candidate forms. First finding: *"read the direction the state moves under
recall"* is already the implementation — `project_q` is literally
`basis @ (recalled − current)` with `recalled = R(current)`. Second: the route
§10.20 parked, a basis from pre-nonlinearity activations, is **dead** by
§10.20's own numbers — the frame was never the problem, the accumulation was.
The live ones are **(iii-b)** heteroassociation, a binary key with a graded
value, which is the key/value split of attention; and **(iii-c)** descending
the *scalar* `‖z_goal − z(p)‖`, which survives binarisation because
binarisation destroys bearing but not distance, has range `D/2m ≈ 28` cells
(the arena), needs no extra storage, and is testable on arm B's existing codes.

**Turn 7 — "if we had an attractor and then reduced α, so the recalled vector
was in the direction of the goal but only a step away from the cue, could that
work?"** No, and the reason exposed that §7.1 was overstated. `α` cannot rotate
`q`: for every `α`, `q` lies in the span of the same two 2-vectors, and the
small-`α` limit differs from `α = 1` only by subtracting `z_here` with weight
`cos_bin·C(r)` instead of 1 — under 4% of the signal at every radius. `α` is a
gain and a time constant, which is what §4.2–4.3 measured. **But the premise
did not need rescuing**: T1's original "an attractor and a direction field are
incompatible" was too strong. Arm A already has both. What is impossible is an
attractor whose fixed point *is* the memory, and stating that premise properly
turns two escapes into three — the third being the readout change, which is
also the real idea underneath the α proposal.

**Turn 6 — pitch the questions at the right level.** Jack: T1 and T2 are right,
the rest are too specific, and "`d_eff` isn't inherent to the problem setup —
that's something we have gotten to as we've worked." Correct. §6 restates seven
questions using only the objects of §5.1 and the two success criteria, each
with the context that makes it worth asking; §7 keeps the calculations as
*routes* under them, which is where campaign-invented quantities belong.

**Turn 5 — "what can and should we answer analytically."** §6: six targets.
T1 (a true attractor and a working direction field are incompatible) is a
theorem, not a measurement — `H` is a metric, so `N(k) ≤ √k·N(1)` for any
binary code, and (J3) wants `N(k) = k·N(1)`. T2 gives a closed form for the
basin, `C(r*) = 2√(K−1)/(d_eff·N(1))`, which already lands at 19.7 against a
measured 27.0 with three constants dropped. T3–T6 are the far-field law, the
0.25 threshold, the optimal spectrum, and the minimum dimension. Named what is
*not* analytic: everything about what training reaches.

**Turn 4 — formalise.** Jack: "there is a lot going on here and many ways to
break the problem apart… it feels like theory can make a lot of headway,
largely because the input and desired behavior are really well defined." §5
written: the objects, the three conditions (J1) addressing / (J2) localisation
/ (J3) differentiation as inequalities, and a four-way taxonomy
(realizability / sufficiency / attainability / mechanism) that Jack's example
questions are instances of. Two claims came out of writing it that were not in
any log: the **grid code already supplies (J3) for free** (a product of
Clifford tori is flat and ballistic by construction, so the encoder's job is
not to build the chart but to avoid breaking it), and **a random MLP can only
reshape the kernel pointwise**, which cannot remove an exact alias — with a
cheap experiment that separates the two.

**Turn 3 — "we need `r_exact` otherwise the policy doesn't find the goal all
the time."** Correct, and §3.2's proposed relaxation is retracted. Measured in
§3.4: a `near` retrieval costs almost nothing at range and ~65% of arrivals in
the terminal neighbourhood, so a tolerance radius certifies the wrong thing and
a guarantee-from-zero certifies the right one. The measurement also turned up
something not being looked for — at production coverage the residual reach loss
is ~80–90% starts that retrieved *exactly*, i.e. the readout, not the memory
(Q4).
