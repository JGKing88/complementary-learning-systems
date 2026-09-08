# Why does grid → encoder → Hopfield work, and is it the best we can do?

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

**B2 — Saturation is a square, not a ladder.** Three corners measured on the 10%
winner: **[M]**

| arm | `cos_self` | basin | exact | acc45 | reach |
|---|---|---|---|---|---|
| production (gain 100, β = 100) | 0.813 | 27.0 | 0.982 | 0.995 | 0.987 |
| β = 1e6 only | 0.957 | 24.5 | — | — | 0.973 |
| gain = 1e6 **and** β = 1e6 | 1.0000 | 28.2 | 0.999 | 0.392 | 0.103 |

Half-saturation *costs* basin (binarised state against a continuous bank); full
saturation gives a perfect fixed point and **destroys the direction field**,
because `q` is a finite difference and `sign(z)` carries no magnitude. The
production corner is the only one where memory and direction both work.

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

1. **The published basin under-reports what navigation gets, and it does so
   more at low coverage.** The retrieved code is consumed as
   `q = basis @ (z_goal − z_here)`; retrieving the cell one north of the goal
   is not a failed retrieval, it is a target one cell off, which at r ≈ 29 is a
   ~2° bearing error. Re-measuring the same discs — paired per map, since the
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
nor no.** The *operational* basin — `r_goal`, purely cross-talk-limited —
plausibly does share reach's variable, and is much flatter across coverage than
the published ladder. The published `r_exact_all` carries an additional
**precision** term which is zero at 10% coverage and grows to ~11 cells at
1.25%. B1's "coverage buys capacity" survives on the dead-goal rate at K = 20,
which is measured independently; part of its *basin* evidence is this precision
term rather than capacity.

Caveat: 2–4 maps per group, one (world, env) pair each, so **[m]**. The
direction is monotone in coverage across all five rungs and the saturation
contrast is 0.02 against 0.90 `near`, both far larger than that noise, but the
per-rung magnitudes are not to be quoted.

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

## 4. Questions for theory

**Q1 — Is `d_eff = PR(P)` and `C = FT(P)` actually right?** §2.0 asserts the
code covariance's eigenvalues are the spatial power spectrum, under a
genericity assumption that distinct spatial frequencies map to orthogonal
output directions. Cheap to check: compute `P(ω)` directly, compare `PR(P)`
against the measured `d_eff` and `FT(P)` against the measured `C(Δ)`, on the
same checkpoints §10.11 used. If it holds, the `1/√d_eff` law stops being an
empirical fit.

**Q2 — What spectra can a 3-module grid input reach?** `d_eff · res90²` runs
14.6k–25.7k against a broadband ceiling of ~234k. Is the missing factor of ~10
a property of the losses, or of the input code?

**Q3 — Where does `r_goal` come from?** §3.2 makes the operational basin a
pure cross-talk quantity. If distant cosines are `N(0, 1/√d_eff)` and the cue
at radius `r` retrieves with weight `C(r)`, the radius at which some cue in a
disc of ~12,853 loses to one of K−1 competitors should be predictable from
`d_eff`, `K` and `C(·)` alone. That would give a *predicted* basin, and it is
the same algebra as PROBE §10.11's open question 2.

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
