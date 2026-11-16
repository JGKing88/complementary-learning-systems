# Ideal-encoder family: single units and kernels against r, M and sparsity

In the ideal family, r is the SD in cells of a Gaussian population kernel, and a rotated unit cut at threshold t = Phi^-1(1 - p) is a Gaussian random field (GRF) cut at t. GRF theory then predicts almost everything the coding-comparison measures report. Coverage equals p and pooled Treves-Rolls sparsity equals a(t) to within 0.5%. Field widths, field counts and 2D field areas follow Rice's formula and the Euler characteristic to within 2 to 5% once definition A's own level and its 3-cell minimum are included. Kernel widths match to 0.5% (3% at p = 0.02), and the largest far-field alias over the whole scaffold matches sqrt(ln N_eff / M) to about 5%. The structural claim holds for coverage, sparsity and kernel shape, which depend on p alone. Field statistics also depend on the environment's size in units of r, because definition A sets each unit's level from its own peak, and on pixelation once fields are a few cells wide. M sets the kernel's noise floor and alias, and a spread between units that overdisperses counts in long environments; the single-unit marginals are already within 1 to 2% of the GRF at M = 32. A dorsal-like field width, W_A / lambda of 0.3 to 0.5, needs r of about DORSAL_R cells at p = 0.1, with a kernel 0.4 to 0.6 grid periods wide.

## Result

1. **r is the kernel SD in cells, and M is the number of frequencies.** `draw_frequencies` gives `omega` an SD of `1/r` per component, so the expected kernel is `exp(-d^2 / 2 r^2)` with full width 2.355 r at 0.5. Measured along loops over 3 seeds, FW05 / r is 2.33 to 2.41 at M of 128 or more, and FW02 / FW05 is 1.53 to 1.56 against 1.524. Individual draws differ from the expectation by about `1/sqrt(M)`: the three M = 512 seeds are all 1.8% wider (r_eff = 1.019 r), and seed 0's 19.10-cell half-width at r = 16 is that draw. Single-unit statistics below are scaled by each draw's r_eff.
2. **Thresholding sets coverage and sparsity exactly, and narrows the kernel.** For every r and M, coverage is p within 0.002 and pooled Treves-Rolls is a(t) within 0.5% (`a(0) = 1/pi = 0.318`, `a(1.28) = 0.057` at p = 0.1). The rate kernel is `K(rho(d); t)` from the bivariate-normal integral; measured FW05 / r matches it within 0.5% for p of 0.1 and above and within 3% at p = 0.02 (table "Rate kernels"). The far cosine of a thresholded code is not 0 but `a(t)`, its own sparsity (0.0573 measured, 0.0570 predicted at p = 0.1). In units of r the Pearson kernel narrows from FW05 = 2.11 r (rectified, p = 0.5) to 1.64 r (p = 0.1) and 1.34 r (p = 0.02).
3. **The viewer's r = 23.2 matches the Pearson kernel, not the cosine one.** Holding FW05 at the raw r = 16 kernel's 37.7 cells needs r = 23.0 for Pearson and 21.9 for cosine at p = 0.1. The suite measured 38.3 cells (Pearson) and 40.3 (cosine) for the viewer's sparse code. To hold the width as p falls, r must grow: 17.9, 19.5, 20.8, 23.0, 25.3 and 28.1 cells (Pearson) for p = 0.5, 0.3, 0.2, 0.1, 0.05 and 0.02.
4. **Field width follows Rice's formula at definition A's level.** Definition A's level is `u_A = t + 0.2 (M_env - t)`, with `M_env` the unit's own maximum. On a 1716-cell loop at r = 16 the GRF maximum is 2.56 SD, so `u_A = 0.51` for the rectified code and W_A = 35.0 cells (35.2 with the 3-cell minimum), against 35.0 measured by the suite and 35.6 here with fields joined across the loop's seam. A level of 0.7 would give 31. The prediction holds within 2% for every r and p in the GRF references, including r = 3 where the 3-cell minimum lifts W_A by up to 25%.
5. **Fields per field width equals the field coverage.** `nu(u) L(u) = 1 - Phi(u)` for any stationary process, so under definition A it is the share of the environment inside fields: 0.30 rectified, 0.06 at p = 0.1, 0.014 at p = 0.02 on 1716-cell loops at r = 16, each within 3% of Rice.
6. **In 2D the Euler characteristic counts fields for p of 0.3 and below.** With the walls and the 9-cell minimum included, predicted fields per unit and mean field areas in arenas of side 4 r to 16 r are within 3% of the GRF and the encoder for p of 0.3 and below, and within 7% for the rectified code, whose level (0.37 to 0.6) is where the EC starts to fail. Mean field area is 5.8 r^2 (rectified), 3.2 r^2 (p = 0.3), 1.7 r^2 (p = 0.1) and 1.0 r^2 (p = 0.02) in 16 r arenas.
7. **Variance over mean can exceed 1 with identically distributed units, but only through definition A or finite M.** At a fixed level, GRF upcrossings are never overdispersed: the Rice second-moment Fano factor F(u) rises from 0.29 at u = 0 to 1 and never exceeds it by more than 2e-5. Definition A's peak-relative level differs between units and adds variance that grows with the number of fields per unit N; finite M adds `N a_cv2 / (4 (M + 1))`. GRF units reach 1.00 at p = 0.1 on the 6864-cell environment at r = 16 and up to 1.9 at r = 2 to 4. Encoder units reach 1.9 at M = 32. Nothing reaches Rich 2014's 7.1. On a track of Rich's 74 field widths, where N is 22 (rectified) or 4.7 (p = 0.1), the prediction is 0.46 and 0.71, and the suite measured 0.48 and 0.72.
8. **Finite M: a noise floor, an alias and a spread between units.** The far raw kernel has SD `1/sqrt(2M)` (measured times sqrt(2M): 0.97 to 1.04). Its largest value over the whole scaffold beyond 3 kernel widths matches `u* / sqrt(2M) = sqrt(ln N_eff / M)` within about 5% for M from 64 to 2048, with `N_eff = 0.36 to 0.50 A / r^2` and `A = 1716^2 / 2`: 0.12 at r = 16 and 0.15 at r = 3 for M = 512. Rate kernels inherit it through `K(rho; t)`, measured minus predicted 0.001 on average. Each unit's spectral width varies across units with squared CV `a_cv2 / (M + 1)` (measured `a_cv2 (M+1)` 1.9 to 2.2, expectation 2), and a unit's field count correlates with it (r = 0.83 at M = 32, 0.29 at M = 512 for the rectified code on the long environment).
9. **The native code is a set of plane waves, and every statistic follows from |k|.** On loops, coverage is 0.500, Treves-Rolls 0.406 to 0.419 (4/pi^2 = 0.405, lifted by waves with n = 0 along the loop), field coverage 0.437 (arccos(0.2)/pi = 0.436), W_A = 3.43 to 3.53 r (2 arccos(0.2) r sqrt(pi/2) = 3.433 r), and count variance over mean `0.0725 L / r` (40 at r = 3, 3.9 at r = 32 on a 1716-cell loop). Predicted from each draw's own frequencies, W_A and Treves-Rolls match to 0.7% and 0.1%. Direction enters only through one draw's x and y projections, `ln(W_x / W_y)` with SD `1.07 / sqrt(M)`.
10. **Short tracks inflate Treves-Rolls only because silent units are dropped.** On a 4.6 W_A track the sparse code at p = 0.1 measures 0.180 over units that fire somewhere (the suite's 0.18), 0.080 with silent units scored 0, and 0.058 pooled over all units, against a(t) = 0.057. Only 44% of units fire on that track.

## Structural claim

The claim: in the large-M rotated limit r is a pure length scale, so every scale-free statistic depends on p alone, and M sets only the noise floor and how Gaussian the units are.

**It holds for statistics that do not use a peak-relative field definition.** The rotated code at r is the code at r = 1 with distances scaled by r (up to the integer lattice), and three statistics depend on p alone at every r from 2 to 32 and every M from 32:

- coverage is p (spread across r at fixed p below 0.1%);
- pooled Treves-Rolls is a(t) (spread below 0.2%; +1.4% at M = 32);
- the kernel is `K(rho(d/r); t)`, with FW05 / r within 0.5% of theory for p of 0.1 and above.

**Definition A adds the environment's size in units of r.** A unit's level is 20% of its own peak, and the peak grows like `sqrt(2 ln(L / 2 pi r))`. On the fixed 1716-cell loops, fields per field width for the rectified code rise from 0.25 at r = 3 to 0.32 at r = 23.2, and var/mean falls from 0.58 to 0.39. The theory at each r reproduces both (`figures/structural.pdf`, top row). Hold `L / r` fixed and the dependence goes away; that is a statement about the environment, not the code.

**Pixelation adds a floor at small r.** Definition A drops runs under 3 cells (areas under 9 in 2D) and counts 4-connected components on the lattice. In 1D the 2.5-cell ramp cut predicts the effect (field coverage within 1% at r = 3). In 2D it is larger: in arenas of side 8 r, which scale with r, field area per r^2 is 1.62 at r = 8, 1.46 at r = 16 and 1.44 at r = 23.2 for p = 0.1, but 2.51 at r = 3; fields per unit are 2.5, 2.7, 2.8 and 1.4. 2D field statistics settle only once fields cover a few hundred cells (r of 16 and above).

**M does more than set a noise floor and Gaussianity.** It sets three things:

- the kernel's far-field noise, SD `1/sqrt(2M)`, and the alias `sqrt(ln N_eff / M)`;
- how far each draw's realised kernel sits from its expectation (r_eff / r within about `1/sqrt(M)`, and the kernel along x and y can differ by 15% at M = 32);
- a spread between units. Units of one draw share frequencies but not weights, so each unit's spectral width varies with squared CV `a_cv2 / (M + 1)`. That spread does not average out over an environment: field counts on the 6864-cell environment have var/mean 1.86 at M = 32 against 0.56 for the GRF, and a unit's count correlates 0.83 with its spectral width.

The Gaussianity effects are the small ones. Single-unit marginals (coverage, Treves-Rolls, W_A, gap CV) are within 1 to 2% of the GRF from M = 32. The per-place normalisation of the sparse code matters only when few units fire at a place (`D p` small): it raises Treves-Rolls by 0.5% at p = 0.1 and M = 512, and by 46% at p = 0.02 and M = 32.

## Scale: metres per scaffold cell

SCALE_PLACEHOLDER

## Definitions, checked in code

The ideal encoder `ideal:r=16,n_freq=512,seed=0,gain=100` is `IdealEncoder` in `analysis/hopfield_probe/ideal_encoder.py`. `draw_frequencies` draws each wave vector component as `round(Normal(0, sigma_n))` with `sigma_n = Npos / (2 pi r)` and `Npos = 1716`, so the angular frequency `omega = 2 pi n / Npos` has SD `1/r` per component. Unit `i` is `cos(omega_i . p) / sqrt(M)` and unit `M + i` is `sin(omega_i . p) / sqrt(M)`. The code reads the grid code only to recover position, and `IdealEncoder.closed_form` computes the same numbers from `(x, y)`.

- **r** is the SD, in cells, of the Gaussian kernel `E cos(omega . d) = exp(-|d|^2 / 2 r^2)`. Its full width at 0.5 is `2 r sqrt(2 ln 2) = 2.355 r`, so half-width 18.84 cells at r = 16. The suite's 19.10 is the seed-0 draw: with 512 frequencies the realised kernel is 1.4% wider than its expectation. Rounding to integers adds 1/12 to `sigma_n^2`, which is at least 73 for r up to 32, so it changes r by less than 0.1%.
- **M** is `n_freq`, the number of random frequencies. There are `D = 2M` units, a cos and a sin per frequency. Nothing in the code uses M for the unit count, but the viewer and the suite speak of "1024 units". If Jack's M meant units, every M below halves.
- **Variants** (`analysis/ideal_family/codes.py`, built from the closed form; `test_family_reproduces_the_viewer_populations` checks seed 0 against the suite's populations to 1e-6):
  - native: `relu(z)`, the plane-wave units rectified, as in the suite's primary convention;
  - rotated: `relu(Q z)`, with the viewer's `Q = H D3 H D2 H D1 / D^1.5` (`H` Sylvester Walsh-Hadamard, `D_k` random signs from `default_rng(seed)`), applied by a fast transform. Seed s uses frequency seed s and rotation seed s, so seed 0 is the viewer's `ideal_rot`;
  - sparse: `relu(Q z - t / sqrt(D)) / |.|`, with `t = Phi^-1(1 - p)` and unit length at each place. At r = 23.2, M = 512, seed 0, p = 0.1 this is the viewer's `ideal_sparse`;
  - sparse_raw: the same without the per-place normalisation. The rotated variant is sparse_raw at p = 0.5.
- **Measures** reuse `analysis/coding_comparison/measures.py` unchanged: definition A fields (a run at or above 20% of the unit's peak in that environment, at least 3 cells or 9 cells in 2D, 4-connected), Treves-Rolls sparsity `mean(r)^2 / mean(r^2)`, cosine and Pearson lag profiles. `analysis/ideal_family/measure.py` adds per-unit coverage (fraction of positions with rate above 0), the 95% area (the fewest bins that hold 95% of the unit's summed rate, as a fraction of the environment; the user attributed this measure to Ledergerber 2021 and the implementation follows that wording), fields per unit per cell and per field width, single-field share, variance over mean of counts, gap CV between a unit's field centres (units with at least 6 fields, the wrap gap included on loops), and fields on a loop joined across its seam.

## Derivations

Levels are in units of a unit's SD over the scaffold, `1/sqrt(D)`. Lengths are in cells.

### 1. The raw kernel

`z(p) . z(p + d) = (1/M) sum_i cos(omega_i . d) = K_M(d)`, independent of `p`, and the rotation leaves it unchanged. Its expectation is `rho(d) = exp(-|d|^2 / 2 r^2)`: full width 2.355 r at 0.5 and 3.588 r at 0.2, a ratio of 1.524. In expectation it never crosses zero; a measured zero crossing is where the decaying mean meets the finite-M noise.

### 2. Rotated units are Gaussian random fields; thresholds set coverage and sparsity

A rotated unit is `y_u = sum_j Q_uj z_j = sum_i A_ui cos(omega_i . p + phi_ui) / sqrt(M)`, with `A_ui^2 = Q_ui^2 + Q_u,M+i^2`. Rows of Q have unit norm and distinct waves are orthogonal over the torus, so every unit has variance exactly `1/D` over the scaffold, and its spatial correlation is `sum_i w_ui cos(omega_i . d)` with `w_ui = A_ui^2` summing to 1. As M grows each unit tends to a stationary isotropic Gaussian random field (GRF) with correlation `rho(d)`.

For a rate `s = relu(y - t)` with `y` standard normal:

- coverage `P(s > 0) = 1 - Phi(t) = p`;
- `m1 = E s = phi(t) - t (1 - Phi(t))`, `m2 = E s^2 = (1 + t^2)(1 - Phi(t)) - t phi(t)`;
- Treves-Rolls `a(t) = m1^2 / m2`, so `a(0) = (1/2 pi) / (1/2) = 1/pi = 0.318` and `a(1.282) = 0.0570` at p = 0.1.

The per-place normalisation of the sparse code divides each place by `|s(p)|`. It leaves coverage alone and rescales a unit most where it fires most, which changes its peak-relative fields and its sparsity slightly; the tables report both.

### 3. Fields in 1D: Rice, the environment's maximum, and definition A

Along any line a GRF unit has correlation `rho(d)`, so its second spectral moment is `lambda_2 = 1/r^2`. Rice's formula gives:

- upcrossings of level `u` per cell: `nu(u) = exp(-u^2/2) / (2 pi r)`;
- mean field length: `L(u) = (1 - Phi(u)) / nu(u) = 2 pi r (1 - Phi(u)) exp(u^2/2)`;
- fields per field width: `nu(u) L(u) = 1 - Phi(u)`, the fraction of the line inside fields. This identity holds for any stationary process.

Definition A puts a unit's level at 20% of its own peak in the environment. For a rate `relu(y - t)` that is `u_A = t + 0.2 (M_env - t)`, where `M_env` is the unit's maximum. Rice's Poisson-clumping form gives `P(M_env <= u) = Phi(u) exp(-L nu(u))` over L cells. Pooling fields over units,

`W_A = E[1 - Phi(u_A)] / E[nu(u_A)]`,

with the expectation over the maximum's distribution (`theory.def_A_1d`). For the rotated code (t = 0) on a 1716-cell loop at r = 16, `E[M_env] = 2.56`, so `u_A = 0.51`, and `W_A = 35.0` cells. The suite measured 35.0. Plugging `E[M_env]` straight in gives 34.9; a level of 0.7 would give 31, because a 1716-cell loop spans only 17 kernel lengths `2 pi r` and its maximum sits near 2.6, not 3.5.

Definition A also drops runs shorter than 3 cells. With excursion lengths taken as Rayleigh with the Rice mean (the high-level limit, where an excursion is a parabolic cap), the kept fraction is `exp(-m^2 / 2 s^2)` and the kept mean is `m + sqrt(2 pi) s exp(m^2 / 2 s^2) (1 - Phi(m/s))`, with `m = 3` and `s = L(u) / sqrt(pi/2)`. This matters only when W_A is a few cells.

Because `u_A` depends on `M_env`, and `M_env` depends on the environment's length in units of `2 pi r`, definition-A statistics depend on p and on `L / r`, not on p alone. The dependence is logarithmic.

### 4. Fields in 2D: the Euler characteristic

For an isotropic 2D GRF with `lambda_2 = 1/r^2` the expected Euler characteristic (EC) of the excursion set per cell squared is `u exp(-u^2/2) / ((2 pi)^1.5 r^2)`. At levels where every field is a simply connected blob, EC counts fields, so

- fields per cell squared: `u exp(-u^2/2) / ((2 pi)^1.5 r^2)`;
- mean field area: `(2 pi)^1.5 r^2 (1 - Phi(u)) exp(u^2/2) / u`;
- fields in a square of side S, boundary fields included: `1 - Phi(u) + 2 S exp(-u^2/2) / (2 pi r) + S^2 u exp(-u^2/2) / ((2 pi)^1.5 r^2)` (Adler and Taylor 2007).

On GRF samples the count holds within 3% for `u >= 0.5` and the area within 5% for `u >= 0.75`. Below `u = 0.5` fields have holes and merge, and components outnumber the EC: by 18% at `u = 0.25` and 4 times at `u = 0`. The EC density is 0 at `u = 0` and negative below, while the excursion set percolates for `u <= 0`. So the EC formulas fail for the rectified rotated code in arenas, whose definition-A level is 0.4 to 0.6, and hold for sparse codes with `p <= 0.2`. The arena maximum follows from `P(max > u) = 1 - exp(-EC(u))`; on GRF samples it is within 3% for sides of 8 r and more, and 10% low at 4 r.

### 5. The thresholded population kernel

At large D the units at two places are, across units, a bivariate normal pair with correlation `rho = K_M(d)`. With `C(rho; t) = E[relu(X - t) relu(Y - t)]`,

`C = (rho + t^2) P2 - 2 t phi(t) (1 - Phi(q)) + (s / sqrt(2 pi)) phi(t sqrt(2 / (1 + rho)))`,

where `s = sqrt(1 - rho^2)`, `q = t sqrt((1 - rho)/(1 + rho))` and `P2 = P(X > t, Y > t) = 1 - Phi(t) - 2 T(t, sqrt((1 - rho)/(1 + rho)))` (Owen's T). At t = 0 this is the arc-cosine kernel `(sin th + (pi - th) cos th) / 2 pi` with `cos th = rho`. The Mehler series `sum_k c_k^2 rho^k / k!` with `c_0 = m1`, `c_1 = 1 - Phi(t)`, `c_k = phi(t) He_(k-2)(t)` agrees to 1e-7.

- Cosine of rates: `K_cos(rho; t) = C / m2`. Its floor at `rho = 0` is `m1^2 / m2 = a(t)`: the far cosine of a thresholded code equals its Treves-Rolls sparsity.
- Pearson of rates: `K_P(rho; t) = (C - m1^2) / (m2 - m1^2)`, floor 0.
- Widths: solve `K(rho*) = 0.5` for `rho*`; the full width is `2 r sqrt(2 ln(1/rho*))`.
- The per-place normalisation does not change either kernel, because cosine and Pearson ignore a PV's length.

Thresholding sharpens `K` as a function of `rho`, so at fixed r the Pearson kernel narrows as p falls. To hold a width W, use `r(p) = W / (2 sqrt(2 ln(1/rho*(t))))`.

### 6. Finite M

**Kernel noise and the alias.** Far from the origin `K_M(d)` is a sum of M cosines with effectively random phases: mean 0, SD `1/sqrt(2M) = 1/sqrt(D)`. As a function of `d` its covariance is `(1/2M) rho(d - d')`, so the normalised noise is itself a field with `lambda_2 = 1/r^2`. `K(d) = K(-d)`, so half the torus, `A = 1716^2 / 2` cells, is independent. Its expected maximum from the EC heuristic is `u*`, the solution of `A u exp(-u^2/2) / ((2 pi)^1.5 r^2) ~ 1` (`theory.alias_prediction`), so

`alias ~ u* / sqrt(2M) = sqrt(ln N_eff / M)`, with `N_eff = exp(u*^2 / 2)`.

`N_eff` is 0.36 to 0.50 times `A / r^2` for r from 32 down to 4, so `alias ~ sqrt(ln(0.45 A / r^2) / M)`. At M = 512 this is 0.123 for r = 16 and 0.145 for r = 4. A sum of few cosines has lighter tails than a Gaussian, so the formula is an upper estimate at small M. A rate kernel maps the raw alias through `K(rho; t)`. To first order the excess over its floor is `K'(0) alias`, with `K'(0) = (1 - Phi(t))^2 / m2` for cosine and `(1 - Phi(t))^2 / (m2 - m1^2)` for Pearson (0.50 and 0.73 at p = 0.5; 0.25 and 0.27 at p = 0.1).

**Units of one draw differ.** Every unit shares the draw's frequencies and differs only in its weights `w_ui`, which are Dirichlet(1, ..., 1) for a Haar-like rotation. For fixed `a_i = r^2 omega_ix^2`, `Var_u(sum_i w_ui a_i) = s_a^2 / (M + 1)`, so a unit's spectral moment along x has squared CV `a_cv2 / (M + 1)`, where `a_cv2` is the draw's own `var(a) / mean(a)^2` (2 in expectation, chi-square with 1 degree of freedom). A unit's field rate scales as `sqrt(lambda_2)`, so counts gain variance over mean `N a_cv2 / (4 (M + 1))` for N fields per unit. This term grows with the environment and does not average out across rows, because a unit's weights are the same everywhere.

**Units are not Gaussian.** A sum of M random-phase sinusoids with Rayleigh amplitudes has excess kurtosis `-3/2 sum A^4 / (sum A^2)^2 ~ -3/M`. The first Edgeworth term changes coverage at threshold t by `-(t^3 - 3t) phi(t) / (8 M)`: at M = 32 that is +1.2% of p at p = 0.1 and -2.4% at p = 0.02. A unit's own far autocorrelation `sum_i w_ui cos(omega_i . d)` does not decay to 0 but fluctuates with SD near `1/sqrt(M + 1)`, so its fields are quasi-periodic at small M.

### 7. The native code

A plane wave along a line in direction `e` has frequency `k = omega . e`, which is `N(0, 1/r^2)` for every direction because the frequency draw is isotropic. Rectified, its fields are `2 arccos(0.2) / |k|` wide (definition A, peak 1 on any line holding a period) and there are `|k| / 2 pi` per cell. So:

- coverage 1/2 and Treves-Rolls `4/pi^2 = 0.405`, for every r and direction, on whole periods. Waves with `n_x = 0` are constant along an x loop (TR 1 when positive), which lifts the active mean slightly;
- the fraction of each period inside a field is `arccos(0.2) / pi = 0.436`;
- the field-weighted mean width is `2 arccos(0.2) / E|k| = 2 arccos(0.2) r sqrt(pi/2) = 3.433 r`, 54.9 cells at r = 16 (the suite measured 54.6);
- the unit-weighted width `~ 1/|k|` has a Pareto tail of index 1, an infinite mean cut off only by the scaffold at `|n| = 1`, and the field-weighted width has a `w^-3` tail;
- counts are `L |k| / 2 pi`, fixed by the wave, so `var/mean = (L / 2 pi r)(1 - 2/pi) / sqrt(2/pi) = 0.0725 L / r`, growing with track length;
- one draw's x and y projections differ: `ln(W_x / W_y)` has SD `sqrt(2 (1 - 2/pi) / (2/pi) / M) = 1.07 / sqrt(M)`;
- the kernel of rectified rates, averaged over a wave's phase, is `(a_0^2 + sum_k (a_k^2 / 2) rho^(k^2)) / (1/4)` for cosine and `sum_(k>=1) (a_k^2/2) rho^(k^2) / sum_(k>=1) (a_k^2/2)` for Pearson, where `relu(cos th) = sum_k a_k cos(k th)` (`a_0 = 1/pi`, `a_1 = 1/2`, `a_k = (2/pi)(-1)^(k/2+1) / (k^2 - 1)` for even k). The Pearson half-width is 1.052 r.

### 8. Count dispersion

The variance over mean of the number of upcrossings over a long stretch is

`F(u) = 1 + (2 / nu) int_0^inf (f2(tau) - nu^2) dtau`,

where `f2(tau)` is the second-moment Rice density of upcrossings at 0 and `tau` (Cramer and Leadbetter): `p(u, u) E[Y'(0)+ Y'(tau)+ | Y(0) = Y(tau) = u]`. Given the two values the derivatives are bivariate normal with means `-+ rho' u / (1 + rho)` and covariance `[[1/r^2 - rho'^2/(1 - rho^2), -rho'' - rho rho'^2/(1 - rho^2)], [., .]]`; `theory.fano_upcross` does the remaining 1D integral numerically. `F(0) = 0.29`, `F(1) = 0.58`, `F(2) = 0.91`, `F(3) = 0.994`. F rises monotonically to 1 by u = 3 and never exceeds it by more than 2e-5. At a fixed level, GRF counts are never overdispersed.

Two effects add variance on top of F, and both grow with N, the mean number of fields per unit:

- **Definition A's own level.** `u_A = t + 0.2 (M_env - t)` differs between units because their maxima do. At low levels (t = 0) the extra variance over mean is `N 0.04 u_A^2 Var(M_env)`. At high levels a unit with more fields also has a higher maximum, and its top field is always counted; then the dispersion is `F (1 - 1/N) + (0.2 N u_A - E[M_env])^2 Var(M_env) / N` (`theory.fano_def_A`). On GRF loops the first form is within 0.02 at t = 0 and the second within 0.04 at p = 0.1 and 0.02.
- **Finite M.** `N a_cv2 / (4 (M + 1))`, from section 6.

So identically distributed units can be overdispersed under definition A, but only once a unit has of order a hundred fields: a GRF reaches var/mean 1.02 at t = 0 on a 6864-cell loop at r = 4 (213 fields per unit, about 770 field widths). The renewal approximation puts the gap CV near `sqrt(F)`.

RESULTS_TABLES_PLACEHOLDER

## Figures

All in `/orcd/pool/003/jackking/cls_runs/figures/ideal_family/v1/`, as PDF (core Helvetica, no mathtext) and PNG. Lines are theory, markers are measurements (mean over 3 seeds, bars one SD). Filled circles are GRF references, open squares encoder draws.

- `kernels.pdf`: the thresholded kernel against d / r for p = 0.5, 0.1 and 0.02 (cosine and Pearson) over GRF points at r = 16; FW05 / r against p; the r that holds FW05 at 37.7 cells, with the viewer's r = 23.2 on the Pearson curve.
- `single_units.pdf`: coverage, Treves-Rolls (pooled on loops, and over firing units on a 4.6 W_A track), W_A / r and fields per field width against p, at r = 8, 16 and 32, M = 512.
- `counts.pdf`: variance over mean against p on loops and on the 6864-cell environment at r = 16, with F(u_A) and both dispersion forms; variance over mean against fields per unit for every r and p.
- `finite_M.pdf`: the whole-scaffold alias against M for every r with `u* / sqrt(2M)`; long-environment dispersion against M with the between-unit prediction; W_A / r_eff and gap CV against M.
- `arenas_2d.pdf`: measured fields per unit and mean field area in every arena against the EC predictions.
- `native.pdf`: native W_A / r against M; `ln(W_x / W_y)` against M with the `1.07 / sqrt(M)` band; count dispersion against r.
- `scale.pdf`: W_A / lambda, Pearson kernel FW05 / lambda and the Pearson alias against r for each p (M = 512), with the dorsal band.
- `structural.pdf`: scale-free statistics against r, on fixed loops (with the theory at each r) and in arenas of side 8 r.

## What did not hold

1. **A level of 0.7 for definition A.** The GRF maximum over a 1716-cell loop at r = 16 is 2.56 SD, not 3.5, so `u_A = 0.51` and W_A = 35 cells. The structure of the estimate was right; the maximum was too high.
2. **"No setting gives variance over mean above 1."** True at a fixed level, where F(u) never exceeds 1. False under definition A: the peak-relative level differs between units and adds variance in proportion to the number of fields per unit, so identically distributed GRF units exceed 1 on long environments (1.00 at p = 0.1 and r = 16 over 6864 cells, up to 1.9 at small r). Finite M adds more (1.9 at M = 32). It is still far from Rich's 7.1, and on Rich's 74 field widths it stays at 0.46 to 0.71.
3. **"Every scale-free statistic depends on p alone."** Only coverage, pooled Treves-Rolls and the kernel's shape do. See the structural claim section.
4. **My first count-dispersion formula.** Adding `N 0.04 u_A^2 Var(M_env)` to F(u_A) fits the rectified code (within 0.02 on GRF loops) but overshoots by 4 times at p = 0.1, because a unit with more fields also has a higher maximum. The peak-process form fixes p of 0.1 and below. Neither form fits p = 0.2 to 0.3 (u_A near 1), where it is off by up to 0.3; the tables use the GRF reference there.
5. **My first between-unit spread.** I first wrote `CV^2(lambda_2) = 5/M`, then `4/(M+1)` for independent frequencies per unit. Units of one draw share frequencies, so it is `a_cv2/(M+1)`, about `2/(M+1)`, and the per-draw `a_cv2` varies from 1 to 3 at small M. With the draw's own value, the predicted long-environment dispersion still captures only about two thirds of the excess at M = 32 (1.37 predicted against 1.86 measured for the rectified code).
6. **The EC area in arenas.** The interior EC area overstated measured areas by up to 35%, because arena fields are cut by the walls. The in-box form `(1 - Phi(u)) S^2 / EC` with the 9-cell minimum fixes p of 0.1 and above. At p = 0.02 the exponential area law over-corrects the 9-cell cut by about 10% at r = 8.
7. **The renewal gap CV.** Gap CV between a unit's fields is 0.53 (rectified) to 0.75 (p = 0.02), below `sqrt(F(u_A))` (0.61 to 0.97). Part of the gap is the small-sample bias of a CV over 6 to 15 gaps, which Rich's measure shares. I did not derive the gap distribution.
8. **The 2D maximum heuristic in small arenas.** `P(max > u) = 1 - exp(-EC)` is 10% low at a side of 4 r and 17% low at 2 r. The tables use each unit's measured level in 2D instead.

## What was run, pruned and approximated

The grid is `analysis/ideal_family/run.py::grid()`, 156 configurations, each one JSON in `/orcd/pool/003/jackking/cls_runs/results/ideal_family/v1/configs/`.

- **Encoder draws.** r in {4, 8, 16, 23.2, 32} crossed with M in {32, 64, 128, 256, 512, 1024, 2048}, 3 seeds: 105 configurations, all with 1D measures and the whole-scaffold alias. Nothing in this cross was pruned.
- **Dorsal-scale draws.** r in {2, 2.5, 3, 3.5, 5, 6} at M = 512, 3 seeds, with 2D.
- **GRF references.** Direct Gaussian random fields (1024 independent units, `analysis/ideal_family/grf.py`) at all 11 values of r, 3 seeds, with 2D. These are the M = infinity limit of the rotated family on the same integer lattice.
- **Variants.** Every configuration runs native (encoder only), rotated, sparse at p in {0.5, 0.3, 0.2, 0.1, 0.05, 0.02}, and sparse_raw at the five p below 0.5 (sparse_raw at 0.5 is the rotated variant). No p was pruned.
- **2D, pruned.** Arenas only at M = 512 (every r) and at r = 16 (every M). Sides 4 r, 8 r and 16 r, 2 places each. Arenas below 8 cells a side are skipped, which drops the 4 r arena at r = 2.
- **1D environments.** The suite's zigzag tracks at Rich's five lengths were not rerun. In their place: 8 full scaffold loops (4 rows, 4 columns), a 6864-cell environment made of the 4 rows (and one of the 4 columns) with one peak per unit, and straight tracks of 4.6 and 15.4 W_A cut from each loop. The 6864-cell environment is 140 to 690 W_A long, depending on r and p.
- **Rotation.** The family uses the viewer's 3-round Hadamard rotation. The suite's own basis checks used Haar rotations; per-unit weights from the Hadamard rotation have `E sum w^2 (M+1)/2 = 0.96 to 1.02`, the Dirichlet value, so the two should agree.
- **Alias.** The raw kernel is exact at every displacement on the torus (one inverse FFT). The rate kernels' alias is predicted from it through `K(rho; t)` and checked at the worst displacement over 256 anchors, not evaluated at every displacement.
- **Approximations in the theory.** Rice's Poisson clumping for the maximum; the EC as a field count; a Rayleigh (1D) or exponential (2D) field-size law for the 3-cell and 9-cell minimums; additive variance terms for count dispersion; the EC heuristic for the alias. Each is tested against direct GRF samples in `hopfield_nav/tests/test_ideal_family.py`, and where it fails the test states the failure.

## Reproduce

```bash
# synthetic tests (about 30 s): every formula against direct Monte Carlo, and the family against the viewer
~/.conda/envs/cls/bin/python -m pytest hopfield_nav/tests/test_ideal_family.py -q

# the grid: snapshots HEAD to /orcd/pool/003/jackking/cls_runs/code/<tag> and submits 48 array tasks
bash analysis/ideal_family/run.sh ideal_family_v1 48        # about 1 h on ou_bcs_normal, 1 cpu and 20 GB each

# one configuration locally (a few minutes for small r and M)
~/.conda/envs/cls/bin/python -m analysis.ideal_family.run --only enc_r16_M512_s0 --out /tmp/if

# tables and figures
~/.conda/envs/cls/bin/python -m analysis.ideal_family.analyse     # -> results/ideal_family/v1/tables.md, summary.json
~/.conda/envs/cls/bin/python -m analysis.ideal_family.figures     # -> figures/ideal_family/v1/*.pdf, *.png
```

The run that produced these numbers is Slurm array 29377065, from snapshot `ideal_family_v1` (commit `db651f1`), plus array 29380696 from the same snapshot, which ran 27 configurations that were queued behind the slowest 2D tasks (`run.py --only <name>`). Every result JSON records the commit in its `git` field. The analysis and figure code changed after that commit, but the measurements did not.
