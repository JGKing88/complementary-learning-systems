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

Pending: the probe (65 runs) and the field runs (30) were launched 2026-11-19.
