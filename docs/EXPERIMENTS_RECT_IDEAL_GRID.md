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

## Other grid modules: 11, 15, 23 (launched 2026-11-20)

The ideal code depends on the grid modules only through the scaffold size `Npos = prod(lambdas)`,
because its units are waves over position (`IdealEncoder` decodes position exactly from the module
phases by the Chinese remainder theorem). Modules 11, 15, 23 (pairwise coprime) give a 3795-cell
scaffold, against 1716. Two variants (`codes.VARIANTS`), each the full 6 x 5 grid, probe and fields:

* `m111523_r16`: the same kernel, r = 16 cells for the unrectified ideal and the same matched r per
  f (28.07 at 2% to 17.86 at 50%, within 0.01 of the original). The Euler-characteristic formula
  predicts the same fields in cells and 4.89 times as many per unit (the area ratio,
  (3795 / 1716)^2), and the Hopfield measures should barely move.
* `m111523_rs`: r scaled by the modules' geometric-mean period, `Npos^(1/3)` = 15.60 cells against
  11.97, to r = 20.85 (half-width 24.5 cells); matched r from 36.6 at 2% to 23.3 at 50%. Fields keep
  their size in grid periods (one 10%, 256-unit test: median 2.88 periods of 15.6 cells, against
  2.87 periods of 12 for the original), and the 20 x 20 probe arenas are smaller relative to the
  kernel.

Sizes for both are in grid periods of 15.60 cells (the geometric mean), and in cells
(`support.diam_cells`). Results: `$CLS_RESULTS/rect_ideal_grid_m111523_r16/` and
`rect_ideal_grid_m111523_rs/`; launch with `VARIANT=<name> bash analysis/rect_ideal/run_probe.sh` and
`run_fields.sh`. One test field run (`m111523_rs`, 10% of 256 units) measured 565 fields per unit
against 574 predicted.

**Results (2026-11-21, all 2 x 95 runs).** Both came out as predicted. At 1024 units:

| | 11, 12, 13, r = 16 | 11, 15, 23, r = 16 | 11, 15, 23, r = 20.85 |
|---|---|---|---|
| fields per unit, 10% / 20% | 194 / 251 | 947 / 1230 | 558 / 723 |
| median field diameter, 10% (cells) | 34.1 | 34.2 | 44.5 |
| median field diameter, 10% (grid periods) | 2.84 of 12 | 2.19 of 15.6 | 2.85 of 15.6 |
| kernel half-width after inhibition, 10% | 19.0 | 19.0 | 24.7 |
| direction error, 10% + inhibition, K = 5 | 4.8° | 4.1° | 3.8° |
| exact retrieval, 10% + inhibition, K = 5 | 1.00 | 0.99 | 1.00 |
| basin, 10% + inhibition, K = 5 (cells) | 28.7 | 31.0 | 29.5 |
| basin, signed ideal, K = 5 (cells) | 32.1 | 31.6 | 39.6 |
| direction error, signed ideal, K = 5 | 1.0° | 1.1° | 0.8° |

* **r = 16 on the 11, 15, 23 scaffold changes only the scaffold's size.** Fields are the same size in
  cells (34.2 against 34.1 at 10%) and 4.88 times as many per unit (947 / 194; the area ratio is 4.89),
  as the Euler-characteristic formula says; every Hopfield measure is within the run-to-run spread of
  the original (direction 4.1 against 4.8°, exact 0.99 against 1.00, basin 31.0 against 28.7). Only
  the conversion to grid periods changes (2.19 periods of 15.6 cells).
* **Scaling r with the periods keeps fields the same size in grid periods** (2.85 against 2.84) and
  2.88 times as many per unit as the original (4.89 / 1.303^2 = 2.88; measured 558 / 194 = 2.88). The
  wider kernel helps direction (3.8° against 4.8° at 10%, 1.6° against 2.2° at 20% of 4096 units) and
  the signed ideal's basin (39.6 against 32.1 cells), because the 20 x 20 arenas are now smaller
  relative to the kernel; exact retrieval is unchanged.

Full tables: `python -m analysis.rect_ideal.summary $CLS_RESULTS/rect_ideal_grid_<variant>`; on the
page, pick the modules in the rectified population's menu.

## Vector-HaSH place cells at the same sparsities (2026-11-20)

`analysis/rect_ideal/hpc.py`, launched with `run_hpc.sh` (one task per module set). The place layer
is `VectorHash.build_scaffold`'s, `p = relu(Wpg g - theta)` on the one-hot grid code: Np = 400 (the
controller's setting), `Wpg` i.i.d. N(0, 1) with about half the entries pruned (c = 0.5), drawn after
`np.random.seed(0)` (the recipe `analysis/tuning_viewer/export.py` checks against `build_scaffold`).
Each module set has its own draw: 434 grid inputs for 11, 12, 13 and 875 for 11, 15, 23. One global
threshold per sparsity f makes a share f of all (cell, place) pairs active over the whole scaffold,
exactly: the periods are pairwise coprime, so along each axis every combination of module phases
occurs once, and a cell's input over the scaffold is its weight sum over every combination of one
phase per module (`matched_thresh`, bisection on `active_fraction`). Cells differ in how much they
fire, as in Vector-HaSH. Fields, sizes and kernels use `fields.py`'s definitions and summaries.
Results: `$CLS_RESULTS/rect_ideal_grid/hpc/` (`vhpc_<variant>_fXX.json`, `hpc_results_<variant>.json`).

The page shows these as the existing "Hippocampus" (one-hot input) and "Hippocampus, smoothed
input" populations, with shared Modules and Active menus. The 11, 12, 13 weights are byte-identical
to the ones those populations already used (`hippocampus_wpg.b64.txt`, `hippocampus_smooth_wpg.b64.txt`);
only the threshold changes (it was 2.0 and 4.63, 26.5% and 6.9% active).

**Results (2026-11-20).** Fields per cell are over the whole scaffold; diameters are equivalent
diameters in cells; the kernel half-width is where the raw population cosine falls to 0.5; "per-cell
active share" is the 5th to 95th percentile over cells (one global threshold, so cells differ).

| input, modules | f | theta | fields per cell | diameter median, p95 (cells) | kernel half-width (cells) | per-cell active share |
|---|---|---|---|---|---|---|
| one-hot, 11 12 13 | 2% | 2.90 | 56,537 | 1.1, 1.1 | 0.51 | 1.0-3.1% |
| | 10% | 1.70 | 235,918 | 1.1, 1.6 | 0.53 | 7.1-12.9% |
| | 20% | 1.07 | 358,143 | 1.1, 2.3 | 0.56 | 15.9-24.3% |
| | 50% (47.3%) | 0 | 228,569 | 1.6, 5.8 | 0.71 | 41-53% |
| one-hot, 11 15 23 | 2% | 2.88 | 276,542 | 1.1, 1.1 | 0.51 | 1.2-2.9% |
| | 20% | 1.05 | 1,747,154 | 1.1, 2.3 | 0.57 | 16.7-23.4% |
| | 50% (47.0%) | 0 | 1,130,226 | 1.6, 5.8 | 0.70 | 43-51% |
| smoothed, 11 12 13 | 2% | 6.41 | 12,846 | 2.0, 4.1 | 1.18 | 0-6.7% |
| | 10% | 4.01 | 31,115 | 2.8, 6.1 | 1.59 | 1.3-24.6% |
| | 20% | 2.66 | 34,269 | 3.4, 7.7 | 1.85 | 4.8-41% |
| | 50% | 0.09 | 12,443 | 4.8, 13.0 | 2.79 | 25-75% |
| smoothed, 11 15 23 | 2% | 8.89 | 34,010 | 2.5, 5.9 | 1.64 | 0-6.9% |
| | 10% | 5.58 | 79,807 | 3.4, 8.8 | 2.26 | 1.6-23.2% |
| | 20% | 3.70 | 87,253 | 4.2, 11.8 | 2.69 | 6.3-40% |
| | 50% | 0.11 | 30,183 | 4.4, 20.5 | 4.14 | 28-74% |

All f are in `hpc_results_<variant>.json` and on the page.

* **With the one-hot input every field is a single scaffold cell** (median equivalent diameter 1.1
  cells, area 1) at every sparsity up to 35%, and the population kernel falls to 0.5 within one cell:
  the layer is spatially white, because neighbouring cells sit at unrelated grid phases. Fields per
  cell are then just active cells, so they scale with the scaffold's area: 4.88 times as many on
  11, 15, 23 at 20% (the area ratio is 4.89).
* **The smoothed input widens fields to 2 to 5 cells** (0.16 to 0.40 grid periods) and the kernel to
  1.2 to 4.1 cells, wider for 11, 15, 23 because its bumps are a fixed fraction (FWHM 0.25) of larger
  periods. Cells differ much more in how much they fire (0 to 7% of the scaffold at f = 2%), and at
  50% the active regions start to join (largest field 42% of a cell's active area).
* **Both are one to two orders of magnitude narrower than the rectified ideal codes**, whose median
  field is 30 to 42 cells across (2.5 to 3.5 periods) at matched kernel width.
* **50% is not reachable with the one-hot input.** About 40% of the weights are pruned to zero, so
  about 6% of (cell, place) pairs have input exactly 0; theta = 0 gives 47.3% (11, 12, 13) and 47.0%
  (11, 15, 23) active (`matched_thresh` snaps to 0; the page shows the fraction actually active).

## Page: one threshold slider (2026-11-20)

The page's sparsity menus are replaced by one slider, share of cells active, 1% to 60% in 0.5% steps,
for every population. The rectified ideal and both hippocampus populations apply it as their own
threshold; the others get relu(y - theta) after the fact, with theta the value their outputs exceed
that share of the time over every unit at 2048 sample places (a checkbox turns this off).

* Rectified ideal: t = Phi^-1(1 - f) / sqrt(D). Its "Kernel r" menu either matches r to the slider
  (`rect_manifest_<variant>.json` `r_table`, `codes.matched_r` at every slider position, rounded to 4
  decimals like the specs) or fixes it at the r matched for one computed share. A code at a new r is
  rebuilt in the browser: `draw_frequencies` is rint(sigma g) with g the seed-0 standard normals
  (`rect_gauss.b64.txt`; legacy `RandomState.normal` is loc + scale * gauss), so the page needs no
  Python per r. The load check confirms the browser rebuilds all six computed codes exactly
  (frequencies, threshold, r) before it checks the maps against Python.
* Hippocampus: theta from `hpc_manifest_<variant>.json` `thresh_table`, `matched_thresh` at every
  slider position (exact by CRT), with the share actually reached (the one-hot layer plateaus at
  theta = 0 from 47% to 53%).
* Whole-scaffold statistics and probe results exist only at the six computed shares; the results
  panel marks the one nearest the slider.

## Independent random bumps through the same probe (2026-11-21)

`analysis/hopfield_probe/bump_encoder.py` (`bump:` spec), `analysis/rect_ideal/bump.py`,
`run_bump_probe.sh`, `bump_summary.py`; results `$CLS_RESULTS/rect_ideal_grid_bump/`. Each unit has a
Poisson number of Gaussian fields (peak 1, uniform centres on the 1716-cell torus, widths log-normal
with the mean FWHM fixed), rates scaled to unit length per place; the probe settings are
`run_probe.sh`'s. Every code has the rectified ideal's kernel: FWHM 27 cells with one width
(kernel half-width = FWHM / sqrt(2)), or mean FWHM 19.37 with spread 0.5. Measured half-widths after
inhibition are 19.08 to 19.51 cells against 19.09 from the formula, and the raw floors match
2 mu / (1 + 2 mu) within 3%. Densities 0.2, 0.66 and 0.85 fields per unit per 100 x 100 cells give
the rectified ideal's 59, 194 and 251 fields per unit at 2%, 10% and 20% active.

1024 units, one recall step, with inhibition (K = 5 / K = 20):

| code | fields per unit | direction error | exact retrieval | basin (cells) | dead goals |
|---|---|---|---|---|---|
| signed ideal | | 1.0 / 1.3 | 1.00 / 1.00 | 32.1 / 22.1 | 0 / 0 |
| rectified 2% | 58 | 12.8 / 12.9 | 0.41 / 0.43 | 3.4 / 3.3 | 0.17 / 0.17 |
| bumps, 0.2 per 100^2 | 59 | 4.8 / 4.9 | 0.86 / 0.85 | 13.9 / 11.4 | 0 / 0 |
| rectified 10% | 194 | 4.8 / 4.9 | 1.00 / 1.00 | 28.7 / 23.4 | 0 / 0 |
| bumps, 0.66 | 194 | 2.8 / 2.9 | 1.00 / 1.00 | 31.6 / 22.9 | 0 / 0 |
| rectified 20% | 251 | 3.0 / 3.1 | 1.00 / 1.00 | 32.1 / 23.5 | 0 / 0 |
| bumps, 0.85 | 250 | 2.7 / 2.7 | 1.00 / 1.00 | 31.4 / 22.6 | 0 / 0 |
| bumps, 3 | 883 | 1.8 / 1.9 | 1.00 / 1.00 | 30.9 / 23.2 | 0 / 0 |
| bumps, 0.05 | 15 | 13.8 / 13.9 | 0.17 / 0.18 | -0.5 / -0.6 | 0.42 / 0.42 |
| bumps, 0.66, spread 0.5 | 194 | 7.0 / 7.1 | 1.00 / 1.00 | 41.2 / 28.4 | 0 / 0 |

At 0.66 fields per 100^2 and K = 5 with inhibition: 256 units 6.1 degrees, exact 0.62, basin 5.1;
4096 units 1.7 degrees, exact 1.00, basin 39.7.

* **At the same kernel and the same number of fields per unit, independent bumps do as well as the
  rectified ideal or better.** The gap is largest when fields are few: with 59 fields per unit the
  bumps have a 4.8 degree direction error and exact retrieval 0.86, against 12.8 degrees and 0.41.
  At 194 fields the bumps' direction is better (2.8 against 4.8 degrees) with the same retrieval;
  at 250 they are level. So the rectified code's structure (fields as the tops of one smooth input)
  gives the memory nothing that random bumps with the same kernel lack.
* **The comparison holds field count fixed, not sparsity.** A bump's rate never reaches exactly zero
  within 5 sigma, so with 59 fields per unit about 12 units are above half their peak at a place, and
  many more carry small graded rates; the 2% rectified code has 20 units active and the rest exactly
  0. Some of the bumps' advantage at low field counts may come from those tails.
* **Without inhibition, dense bump codes crowd.** At 883 fields per unit the raw floor is 0.33; at
  K = 20 that leaves exact retrieval 0.06 and 92% dead goals, against 1.00 and none with inhibition
  (the same pattern as the dense rectified codes).
* **A spread of widths trades direction for basin at a fixed kernel:** 7.0 degrees against 2.8, with
  a basin of 41 cells against 32.

## Field density per grid period, against rats (2026-11-23)

Fields per cell over the whole scaffold grow with the scaffold's area, which is arbitrary. The page
and this table now use **fields per cell per grid period squared**: the mean fields per cell over
the scaffold times (smallest module / scaffold side)^2, i.e. fields per cell in an 11 x 11-cell
square. It is the literature review's "fields per λ²" (fields per m² times the smallest grid spacing
squared, rat module 1 at 39 to 49 cm): rat dorsal CA1 counting every cell, 0.020 to 0.031 (Tanni et
al. 2022, arenas of 1 to 9 m²); long 1D tracks 0.025 to 0.036 per grid period (Rich et al. 2014, Lee
et al. 2020).

| code | 2% | 5% | 10% | 20% | 35% | 50% |
|---|---|---|---|---|---|---|
| rectified ideal, 1024 units (any module set at r = 16) | 0.0024 | 0.0050 | 0.0080 | 0.0103 | 0.0081 | 0.0032 |
| hippocampus, smoothed input, 11 12 13 | 0.53 | 0.93 | 1.28 | 1.41 | 1.02 | 0.51 |
| hippocampus, one-hot input | 2.3 | 5.4 | 9.7 | 14.7 | 14.5 | 9.4 |

Independent bumps: density 0.2, 0.66, 0.85 and 3 fields per 100² cells give 0.0024, 0.0080, 0.0103 and
0.036 per grid period².

* **The rectified ideal codes have a quarter to a third of the rat's field density** at the same grid
  scale, at most 0.010 per grid period² (at 20% active) against 0.020 to 0.031.
* **The hippocampus layer has 20 to 500 times the rat's density**, because its fields are one to five
  cells across.
* **Bumps at rat density (0.020 to 0.031, between the probed 0.85 and 3 per 100²) work well in the
  memory with inhibition:** at 0.85 and 3 per 100² the probe gives 2.7 and 1.8 degrees, exact
  retrieval 1.00 and a 31-cell basin at K = 5.
