# Rectified ideal codes: sparsity by number of units

Started 2026-11-19. Code in `analysis/rect_ideal/`, branch `worktree-rect-ideal-grid`. Results in
`$CLS_RESULTS/rect_ideal_grid/` (`probe/`, `fields/`, `grid_results.json`). Page:
https://claude.ai/artifact/XeicYzsxQq6GrRX5HNJuGD.

## Question

How does a non-negative, sparse version of the ideal encoder do as a Hopfield input, and what do its
single units look like as place cells, as the share of active units f and the number of units D vary?

## Codes

Each code is `ideal_encoder.IdealEncoder` with `relu_mix=1, active=f`:

    y(p) = relu(Q z(p) - t) / |relu(Q z(p) - t)|,    t = Phi^-1(1 - f) / sqrt(D)

* `z` is the ideal code with `n_freq = D / 2` integer frequency vectors (a cos and a sin unit each),
  drawn from a rounded Gaussian with `sigma = 1716 / (2 pi r)`, seed 0. So D is the number of units
  and twice the number of waves, and the unrectified ideal of the same D is each column's baseline.
* `Q` is the fixed D x D rotation `H D3 H D2 H D1 / D^1.5` (Walsh-Hadamard, random signs from
  `default_rng(0)`).
* The threshold leaves a share f of units active at every place, because each coordinate of `Q z`
  is close to N(0, 1/D).

Grid: f in {2, 5, 10, 20, 35, 50}%, D in {256, 512, 1024, 2048, 4096}. 30 codes.

**Kernel width is matched, not fixed.** Rectifying narrows the population kernel, more so at small
f. `codes.matched_r(f)` picks r so that the rectified code's kernel, with the one global inhibition
constant removed, falls to 0.5 at the same distance as the ideal's at r = 16 (18.84 cells for the
expected kernel). In the large-D limit the rectified, centred kernel is
`(h(rho) - h(0)) / (h(1) - h(0))` with `h(rho) = E[(u - tau)+ (v - tau)+]` for unit normals of
correlation rho and `tau = Phi^-1(1 - f)`, and rho is the ideal's own kernel. So r depends on f only:

| f | 2% | 5% | 10% | 20% | 35% | 50% |
|---|---|---|---|---|---|---|
| r (cells) | 28.08 | 25.30 | 23.02 | 20.74 | 18.96 | 17.86 |
| ideal similarity at 18.84 cells | 0.798 | 0.758 | 0.715 | 0.662 | 0.610 | 0.573 |

The measured half-widths of the finite draws are in the field results (`kernel.half_width_inh`,
beside `half_width_ideal_r16`, the signed ideal at the same D).

## Hopfield measures (`analysis/hopfield_probe`, `run.py`)

Every code runs the standard probe, with the same settings as the earlier ideal-encoder sets
(`run_relu_mix.sh`): 8 layouts ("worlds"), each with 20 arenas of 20 x 20 cells at random places on
the 1716 x 1716 scaffold and one goal cell per arena. The memory stores the codes of the first K
goals, K in {1, 3, 5, 10, 20}, as a sum of outer products with the diagonal zeroed (`storage_rule
hebb`). Recall is one step, `x -> normalize(tanh(beta W x))`, and tanh is inert at this scale, so
the recalled vector is the stored goals weighted by their similarity to the current code. The first
5 arenas of each layout are scored. Each rectified code runs twice, plain and with one global
inhibition constant (`center_global=1`: the mean activity over 65,536 random scaffold cells is
subtracted from every unit and the code renormalised). The signed ideal at each D is the baseline.
65 runs.

* **Direction error** (`err`, `acc45`, `acc90`). At each cell c of an arena, recall from c's code.
  q is the recalled code minus the current code, projected on the two directions in which the code
  changes for one step east and one step north (finite differences at c, orthonormalised). The error
  is the angle between q and the true bearing from c to the goal. `err` is the mean absolute angle in
  degrees over the 399 non-goal cells; `acc45` and `acc90` are the shares under 45 and 90 degrees.
  `s15` is `acc45` with q read after 15 recall steps.
* **Exact retrieval** (`exact`). Recall from each of an arena's 400 cells, then decode the result as
  the cell of highest cosine in a bank holding every cell of the K stored arenas. `exact` is the
  share of cells that decode to the goal cell itself.
* **Basin** (`basin`, `basin95`). The cues are every scaffold cell within 64 cells of the goal, not
  only the arena's, and the bank is that whole disc plus the stored goals. `basin` is the largest
  radius r at which every cue within r decodes to the goal, stopping at the first failure; `basin95`
  needs 95% of cues. In cells, 2 arenas per layout.
* **Reach** (`cont`, `disc`). From each of the 400 cell centres, step one cell along the unit vector
  of q at the current (rounded) cell, clipped at the walls, for at most 80 steps. `cont` is the share
  of starts that end within 0.5 cells of the goal. `disc` is the same walk with one up, down, left or
  right move per step, in the direction q points closest to, and the goal absorbing.
* **Dead goals** (`dead`). The share of scored arenas (the first 3 per layout) whose `cont` reach is
  below 0.5.

`summary.py` records every measure at every K.

## Place-field measures (`fields.py`)

Every unit is evaluated at all 1716 x 1716 scaffold cells, with no arenas. The code's frequencies
are integers, so the scaffold is a torus and fields that cross an edge are joined to their other
half. Two field definitions:

* `support`: a connected region (4-neighbour) where the unit is active, `y > 0`. This is the code's
  own threshold and the primary definition.
* `peak20`: a connected region at or above 20% of the unit's own peak over the scaffold.

Every field counts, single cells included; the summaries also give the share under 9 cells and the
statistics without them. Field size is the equivalent diameter `2 sqrt(area / pi)` in cells divided
by 12 cells, the middle grid period (modules 11, 12, 13). Per code: fields per unit (mean, sd,
percentiles, min, max, histogram), field diameter (the same, plus the cell-weighted mean and a
histogram in 0.05-decade bins), each unit's active share of the scaffold, and the largest field's
share of a unit's active area (near 1 when the active regions join into one web).

The script checks its float32 code against `IdealEncoder.closed_form` (float64) at 2000 random
places and stops if they differ by more than 1e-4 (test code f10 D1024: 2.8e-7, no support
mismatch).

## Results

### Kernel match

The measured half-width after inhibition is within 0.3 cells of the signed ideal at the same D
(18.6 to 19.2 cells) for every code but the two with the fewest active units: 2% of 256 units
(5 active per place) gives 17.0 and 2% of 512 gives 17.7.

### Place fields (whole scaffold, 2026-11-19)

**Single-unit fields depend on f and hardly at all on D.** Every statistic below varies by a few
percent across D = 256 to 4096 at fixed f, which is draw noise: at matched kernel width each unit is
close to a Gaussian random field with the code's kernel, thresholded, and D only sets how many such
units there are. The table is D = 1024; the other D are in `grid_results.json` and on the page.

| f | fields per unit, mean [range] | diameter / 12, median | mean | 5-95% | largest | cell-weighted mean | largest field's share of a unit | 20%-of-peak fields per unit, median diameter |
|---|---|---|---|---|---|---|---|---|
| 2% | 58 [35-79] | 2.50 | 2.67 | 0.70-5.25 | 10.8 | 4.0 | 0.09 | 34, 2.20 |
| 5% | 121 [90-154] | 2.65 | 2.88 | 0.73-5.85 | 12.7 | 4.5 | 0.05 | 70, 2.23 |
| 10% | 194 [151-237] | 2.84 | 3.16 | 0.76-6.70 | 16.4 | 5.3 | 0.04 | 124, 2.27 |
| 20% | 251 [192-297] | 3.17 | 3.78 | 0.81-8.81 | 25.4 | 7.3 | 0.04 | 212, 2.36 |
| 35% | 198 [137-270] | 3.46 | 4.95 | 0.81-14.5 | 58.8 | 15.0 | 0.10 | 302, 2.54 |
| 50% | 79 [39-152] | 2.75 | 5.26 | 0.67-13.8 | 118 | 82 | 0.75 | 335, 2.76 |

Diameters are in grid periods of 12 cells (the equivalent diameter of the field's area). About 1% of
fields are under 9 cells at every f.

* **Fields are 2.5 to 3.5 grid periods across** (30 to 42 cells) at the median, about the full width
  at half maximum of the population kernel (2 x 18.8 cells = 3.1 periods). Sparser codes have
  slightly smaller fields, because r grows with sparsity to hold the kernel width.
* **Every unit has many fields over the scaffold**, from 58 at 2% to 251 at 20%: one field per
  roughly 50,000 cells at 2%, so neighbouring fields of a unit are about 19 grid periods apart.
* **The count peaks at 20% and falls after it because the active regions join.** At 50% (threshold
  at the median) one connected web holds 75% of each unit's active area, the percolation of a 2D
  Gaussian field's excursion set at its median. At 35% the regions are still mostly separate (largest
  field 10% of a unit's active area) but the 95th-percentile field is already 14.5 periods across.
* **At 20% of each unit's peak** the fields are smaller (median 2.2 to 2.8 periods) and the count
  rises monotonically with f, since the 20% line cuts the joined regions of dense codes back into
  separate peaks.

**Why the count rises and then falls: the Euler characteristic of a Gaussian field's excursion
set.** Before the threshold, each unit's map `(Q z)_i(p)` is a sum of hundreds of random-phase waves,
so it is close to a smooth stationary Gaussian field whose covariance is the ideal kernel. Its
fields are the excursion set above the threshold, at `u = Phi^-1(1 - f)` standard deviations. As u
falls, more peaks clear it (fields are added) and the saddles between neighbouring peaks clear it
too (fields join). For a 2D isotropic Gaussian field the expected Euler characteristic of the
excursion set (fields minus the holes inside them) per unit area is exact (Adler and Taylor):

    EC density = lambda2 * u * exp(-u^2 / 2) / (2 pi)^(3/2),   lambda2 = 1 / r^2

where lambda2 is the variance of the field's derivative along one axis (the waves' angular
frequencies have variance 1 / r^2 per axis). It peaks at u = 1 (f = 16%) at fixed r, is zero at the
median (u = 0, f = 50%), and contains no D, which is why the field statistics do not depend on D.
Holes are rare until fields start to join, so below 35% it is the field count. Over the
1716 x 1716 scaffold (r from `codes.matched_r`):

| f | u | r | predicted fields per unit | measured, D = 1024 | measured, all D |
|---|---|---|---|---|---|
| 2% | 2.054 | 28.08 | 59 | 58 | 56-59 |
| 5% | 1.645 | 25.30 | 124 | 121 | 119-124 |
| 10% | 1.282 | 23.02 | 199 | 194 | 191-200 |
| 20% | 0.842 | 20.74 | 257 | 251 | 247-259 |
| 35% | 0.385 | 18.96 | 186 | 198 | 194-203 |
| 50% | 0 | 17.86 | 0 | 79 | 72-80 |

The prediction is within 3% up to 20%. At 35% the measured count is 12 above it: those are the holes
inside joined fields (components = EC + holes). At 50% joined fields and holes cancel in the EC, and
the active set sits at its percolation point (the median level of a planar Gaussian field), so one
connected web holds 75% of each unit's active area. The predicted numbers were worked by hand from
the formula; the measured ones are `support.fields_per_unit.mean` in `grid_results.json`.

### Hopfield probe (2026-11-19, all 65 runs)

**With the one inhibition constant, the memory measures depend mainly on the number of active units
per place, f x D.** About 50 active units give exact retrieval near 0.9 and a 6° direction error;
about 100 give exact retrieval near 1 and 4.5°; more keep improving direction. Codes with the same
f x D land close together whatever f is. Direction error at K = 5 with inhibition:

| f x D (active units per place) | 51 | 82 to 102 | 205 | 410 |
|---|---|---|---|---|
| codes | 5% of 1024, 10% of 512, 20% of 256 | 2% of 4096, 5% of 2048, 10% of 1024, 20% of 512, 35% of 256 | 5% of 4096, 10% of 2048, 20% of 1024 | 10% of 4096, 20% of 2048 |
| direction error | 6.4, 6.0, 6.4 | 4.9, 4.9, 4.8, 4.3, 4.4 | 3.7, 3.4, 3.0 | 2.9, 2.5 |
| exact retrieval | 0.90, 0.95, 0.85 | 0.98, 1.00, 1.00, 1.00, 0.99 | 1.00, 1.00, 1.00 | 1.00, 1.00 |

K = 5, with inhibition (the signed ideal of the same D in the first row):

| | err 256 | 512 | 1024 | 2048 | 4096 | exact 256 | 512 | 1024 | 2048 | 4096 | basin 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ideal, signed | 1.8 | 1.6 | 1.0 | 0.9 | 0.9 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 18.8 | 25.8 | 32.1 | 34.8 | 38.3 |
| 2% | 43.5 | 24.2 | 12.8 | 7.6 | 4.9 | 0.06 | 0.20 | 0.41 | 0.82 | 0.98 | -0.8 | -0.2 | 3.4 | 9.3 | 30.8 |
| 5% | 15.6 | 9.8 | 6.4 | 4.9 | 3.7 | 0.22 | 0.50 | 0.90 | 1.00 | 1.00 | -0.2 | 3.3 | 14.2 | 35.2 | 41.4 |
| 10% | 11.1 | 6.0 | 4.8 | 3.4 | 2.9 | 0.48 | 0.95 | 1.00 | 1.00 | 1.00 | 4.4 | 13.2 | 28.7 | 38.2 | 42.2 |
| 20% | 6.4 | 4.3 | 3.0 | 2.5 | 2.2 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 12.6 | 21.6 | 32.1 | 38.5 | 42.1 |
| 35% | 4.4 | 3.5 | 2.6 | 2.1 | 1.8 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 17.4 | 26.2 | 32.7 | 37.9 | 42.1 |
| 50% | 3.4 | 3.0 | 2.1 | 1.8 | 1.6 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 17.9 | 25.3 | 31.4 | 37.2 | 40.2 |

K = 20, with inhibition:

| | err 256 | 512 | 1024 | 2048 | 4096 | exact 256 | 512 | 1024 | 2048 | 4096 | basin 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ideal, signed | 3.2 | 2.0 | 1.3 | 1.1 | 0.9 | 0.67 | 0.98 | 1.00 | 1.00 | 1.00 | 7.1 | 16.4 | 22.1 | 29.3 | 32.8 |
| 2% | 44.3 | 24.3 | 12.9 | 7.6 | 4.9 | 0.06 | 0.20 | 0.43 | 0.82 | 0.98 | -0.8 | -0.2 | 3.3 | 9.6 | 29.8 |
| 5% | 16.3 | 10.0 | 6.4 | 4.9 | 3.7 | 0.17 | 0.47 | 0.88 | 1.00 | 1.00 | -0.3 | 3.2 | 11.6 | 27.9 | 34.0 |
| 10% | 11.2 | 6.1 | 4.9 | 3.5 | 2.9 | 0.43 | 0.90 | 1.00 | 1.00 | 1.00 | 3.9 | 10.0 | 23.4 | 31.8 | 37.1 |
| 20% | 6.7 | 4.4 | 3.1 | 2.6 | 2.2 | 0.72 | 0.96 | 1.00 | 1.00 | 1.00 | 7.4 | 16.1 | 23.5 | 32.9 | 37.1 |
| 35% | 5.1 | 3.9 | 2.8 | 2.2 | 1.9 | 0.73 | 0.99 | 1.00 | 1.00 | 1.00 | 4.9 | 17.6 | 25.8 | 32.1 | 36.6 |
| 50% | 4.3 | 3.3 | 2.3 | 1.9 | 1.6 | 0.78 | 0.95 | 1.00 | 1.00 | 1.00 | 5.9 | 16.4 | 23.9 | 31.1 | 35.4 |

Dead goals with inhibition, at K = 5 and at K = 20 alike, are 0 for every code with 40 or more
active units per place. The others: 2% of 256, 512 and 1024 units have 79%, 50 to 54% and 17%; 5%
of 256 and 512 have 25% and 12%; 10% of 256 has 8%. Reach is 0.98 to 1.00 wherever f x D is 80 or
more.

* **Rectified codes reach the ideal's retrieval and, at 4096 units, pass its basin; they keep a
  direction cost.** With inhibition and f x D of 80 or more, exact retrieval at K = 5 is 0.98 to
  1.00 like the ideal's (at K = 20 the 256-unit codes drop to 0.72 to 0.78, as the 256-unit ideal
  drops to 0.67), and the K = 5 basin is within 20% of the ideal's at the same D. At 2048 and 4096
  units the K = 5 basin is larger than the ideal's for every f from 5% up (4096 units: 40 to 42 cells
  against 38.3); at K = 20 and 4096 units it is 34 to 37 against 32.8. Direction error stays above
  the ideal's: 1.6 to 3.7° at 4096 units for f from 5% up, against 0.9°.
* **Inhibition is what lets dense codes work, and it matters little below 10%.** Without it, 35%
  and 50% codes crowd: at K = 20 their direction error is 12 to 31°, exact retrieval 0.03 to 0.54 and
  dead goals 62 to 96% at every D, while the same codes with inhibition have 1.6 to 5.1°, 0.73 to
  1.00 and none dead. Their similarity floor without inhibition is 0.22 (35%) and 0.32 (50%). At 2%
  and 5% (floor 0.01 and 0.03) inhibition changes little.
* **Without inhibition the best sparsity is 10 to 20%.** It is the trade-off between too few active
  units (2 to 5%) and the floor (35 to 50%): at K = 5 and 1024 units, exact retrieval is 0.41, 0.90,
  1.00, 0.98, 0.84 and 0.57 from 2% to 50%.
* **15 recall steps.** With inhibition, acc45 after 15 steps is 0.97 to 1.00 for f x D of 80 or
  more. Without it, the 35% and 50% codes fall to 0.94 to 0.98 at 1024 units and above (K = 5).

All 11 measures at every K are in `grid_results.json`, on the page, and printed by
`python -m analysis.rect_ideal.summary`.
