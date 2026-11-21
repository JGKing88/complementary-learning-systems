# Ideal-encoder family: single units and kernels against r, M and sparsity

In the ideal family, r is the SD in cells of a Gaussian population kernel, and a rotated unit cut at threshold t = Phi^-1(1 - p) is a Gaussian random field (GRF) cut at t. GRF theory then predicts almost everything the coding-comparison measures report. Coverage equals p and pooled Treves-Rolls sparsity equals a(t) to within 0.5%. Field widths, field counts and 2D field areas follow Rice's formula and the Euler characteristic to within about 5% once definition A's own level, its minimum field size and the field around each unit's maximum are included. Kernel widths match to 0.5% (3% at p = 0.02), and the largest far-field alias over the whole scaffold matches sqrt(ln N_eff / M) with a median ratio of 0.97. The structural claim holds for coverage, sparsity and kernel shape, which depend on p alone. Field statistics also depend on the environment's size in units of r, because definition A sets each unit's level from its own peak, and on pixelation once fields are a few cells wide. M sets the kernel's noise floor and alias, and a spread between units that overdisperses counts in long environments; the single-unit marginals are already within 1 to 2% of the GRF at M = 32. A dorsal-like field width, W_A / lambda of 0.3 to 0.5, needs r of about 2 to 4 cells at p = 0.1 (0.4 at r = 3.0), with a Pearson kernel 0.3 to 0.6 grid periods wide.

## Result

1. **r is the kernel SD in cells, and M is the number of frequencies.** `draw_frequencies` gives `omega` an SD of `1/r` per component, so the expected kernel is `exp(-d^2 / 2 r^2)` with full width 2.355 r at 0.5. Measured along loops over 3 seeds, FW05 / r is 2.33 to 2.42 at M of 128 or more, and FW02 / FW05 is 1.53 to 1.56 against 1.524. Individual draws differ from the expectation by about `1/sqrt(M)`: the three M = 512 seeds are all 1.8% wider (r_eff = 1.019 r), and seed 0's 19.10-cell half-width at r = 16 is that draw. Single-unit statistics below are scaled by each draw's r_eff.
2. **Thresholding sets coverage and sparsity exactly, and narrows the kernel.** For every r and M, coverage is p within 0.002 and pooled Treves-Rolls is a(t) within 0.5% (`a(0) = 1/pi = 0.318`, `a(1.28) = 0.057` at p = 0.1). The rate kernel is `K(rho(d); t)` from the bivariate-normal integral; measured FW05 / r matches it within 0.5% for p of 0.1 and above and within 3% at p = 0.02 (table "Rate kernels"). The far cosine of a thresholded code is not 0 but `a(t)`, its own sparsity (0.0573 measured, 0.0570 predicted at p = 0.1). In units of r the Pearson kernel narrows from FW05 = 2.11 r (rectified, p = 0.5) to 1.64 r (p = 0.1) and 1.34 r (p = 0.02).
3. **The viewer's r = 23.2 matches the Pearson kernel, not the cosine one.** Holding FW05 at the raw r = 16 kernel's 37.7 cells needs r = 23.0 for Pearson and 21.9 for cosine at p = 0.1. The suite measured 38.3 cells (Pearson) and 40.3 (cosine) for the viewer's sparse code. To hold the width as p falls, r must grow: 17.9, 19.5, 20.8, 23.0, 25.3 and 28.1 cells (Pearson) for p = 0.5, 0.3, 0.2, 0.1, 0.05 and 0.02.
4. **Field width follows Rice's formula at definition A's level.** Definition A's level is `u_A = t + 0.2 (M_env - t)`, with `M_env` the unit's own maximum. On a 1716-cell loop at r = 16 the GRF maximum is 2.56 SD, so `u_A = 0.51` for the rectified code and W_A = 35.0 cells (35.2 with the 3-cell minimum). For the three M = 512 draws, whose kernels are 1.9% wider, the prediction is 35.9 against 35.5 measured here with fields joined across the loop's seam; the suite, which splits a field at the seam, measured 35.0. A level of 0.7 would give 31. Across the GRF references the prediction holds within 4% at every r and p, except 5 to 6% at p = 0.02 with r of 16 and above, and it includes the 3-cell minimum, which lifts W_A by 29% (p = 0.1) to 39% (p = 0.02) at r = 3.
5. **Fields per field width equals the field coverage.** `nu(u) L(u) = 1 - Phi(u)` for any stationary process, so under definition A it is the share of the environment inside fields: 0.30 rectified, 0.062 at p = 0.1 and 0.015 at p = 0.02 on 1716-cell loops at r = 16. That is within 2% of Rice for p of 0.1 and above. At p = 0.02, where an active unit has one or two fields, the field around its maximum lifts it 11% above Rice, and fields per active unit match only once that field is counted (1.88 predicted, 1.86 measured).
6. **In 2D the Euler characteristic counts fields.** With the walls and the 9-cell minimum included, predicted fields per unit in arenas of side 4 r to 16 r (r of 8 and above) are within 2% of the GRF for the rectified code and p = 0.3, within 5% at p = 0.1, and within 9% at p = 0.02. Predicted mean areas are within 4% in arenas of 8 r and 16 r for p of 0.1 and above, and 9 to 13% high in 4 r arenas and at p = 0.02, where the exponential area law over-corrects the 9-cell cut. The rectified code's level (0.37 to 0.6) is at the edge of where the EC counts fields, yet it still holds within 3%. Mean field area in 16 r arenas is 5.7 r^2 (rectified), 3.2 r^2 (p = 0.3), 1.6 r^2 (p = 0.1) and 1.0 r^2 (p = 0.02).
7. **Variance over mean can exceed 1 with identically distributed units, but only through definition A or finite M.** At a fixed level, GRF upcrossings are never overdispersed: the Rice second-moment Fano factor F(u) rises from 0.29 at u = 0 to 1 and never exceeds it by more than 2e-5. Definition A's peak-relative level differs between units and adds variance that grows with the number of fields per unit N; finite M adds `N a_cv2 / (4 (M + 1))`. GRF units reach 0.96 at p = 0.1 on the 6864-cell environment at r = 16, and 2.0 to 2.4 at r = 2 (rectified and p = 0.3), where the formulas give only 1.2 to 1.7. Encoder units at r = 16 reach 1.86 at M = 32 against 0.56 for the GRF. Nothing reaches Rich 2014's 7.1. On a track of Rich's 74 field widths, where N is 22 (rectified) or 4.7 (p = 0.1), the prediction is 0.46 and 0.71, and the suite measured 0.48 and 0.72.
8. **Finite M: a noise floor, an alias and a spread between units.** The far raw kernel has SD `1/sqrt(2M)` (measured times sqrt(2M): 0.97 to 1.04). Its largest value over the whole scaffold beyond 3 kernel widths matches `u* / sqrt(2M) = sqrt(ln N_eff / M)` with measured over predicted between 0.89 and 1.10 (median 0.97) over all 46 combinations of r and M, with `N_eff = 0.36 to 0.50 A / r^2` and `A = 1716^2 / 2`: 0.12 at r = 16 and 0.15 at r = 3 for M = 512. Rate kernels inherit it through `K(rho; t)`, measured minus predicted 0.001 on average. Each unit's spectral width varies across units with squared CV `a_cv2 / (M + 1)` (measured `a_cv2 (M+1)` 1.9 to 2.2, expectation 2), and a unit's field count correlates with it (correlation 0.83 at M = 32, 0.29 at M = 512 and 0.16 at M = 2048 for the rectified code on the long environment).
9. **The native code is a set of plane waves, and every statistic follows from |k|.** On loops, coverage is 0.500, Treves-Rolls 0.406 to 0.419 (4/pi^2 = 0.405, lifted by waves with n = 0 along the loop), field coverage 0.437 (arccos(0.2)/pi = 0.436), W_A = 3.43 to 3.53 r (2 arccos(0.2) r sqrt(pi/2) = 3.433 r), and count variance over mean `0.0725 L / r` (40 at r = 3, 3.9 at r = 32 on a 1716-cell loop). Predicted from each draw's own frequencies, W_A and Treves-Rolls match to 0.7% and 0.1%. Direction enters only through one draw's x and y projections, `ln(W_x / W_y)` with SD `1.07 / sqrt(M)`.
10. **Short tracks inflate Treves-Rolls only because silent units are dropped.** On a 4.6 W_A track the viewer's sparse code (r = 23.2, p = 0.1, M = 512) measures 0.179 over units that fire somewhere (the suite's 0.18), 0.080 with silent units scored 0, and 0.058 pooled over all units, against a(t) = 0.057. Only 45% of units fire on that track; GRF units give the same numbers.
11. **Gap CV is the Rice Fano factor seen through a small sample.** The median gap CV between a unit's fields is 0.53 (rectified) to 0.75 (p = 0.02). That equals `sqrt(F(u_A))` times the median CV of n exponential gaps (0.78 at n = 6, 0.89 at n = 15): 0.54 and 0.71 predicted at r = 16 for the rectified code and p = 0.1, against 0.53 and 0.70.

## Structural claim

The claim: in the large-M rotated limit r is a pure length scale, so every scale-free statistic depends on p alone, and M sets only the noise floor and how Gaussian the units are.

**It holds for statistics that do not use a peak-relative field definition.** The rotated code at r is the code at r = 1 with distances scaled by r (up to the integer lattice), and three statistics depend on p alone at every r from 2 to 32 and every M from 32:

- coverage is p (spread across r at fixed p below 0.1%);
- pooled Treves-Rolls is a(t) (spread below 0.2%; +1.4% at M = 32);
- the kernel is `K(rho(d/r); t)`, with FW05 / r within 0.5% of theory for p of 0.1 and above.

**Definition A adds the environment's size in units of r.** A unit's level is 20% of its own peak, and the peak grows like `sqrt(2 ln(L / 2 pi r))`. On the fixed 1716-cell loops, fields per field width for the rectified GRF code rise from 0.27 at r = 4 to 0.33 at r = 32, and var/mean falls from 0.58 to 0.36. The theory at each r reproduces both (`figures/structural.pdf`, top row). Hold `L / r` fixed and the dependence goes away; that is a statement about the environment, not the code.

**Pixelation adds a floor at small r.** Definition A drops runs under 3 cells (areas under 9 in 2D) and counts 4-connected components on the lattice. In 1D the 2.5-cell ramp cut predicts the effect (field coverage within 1% at r = 3). In 2D it is larger: in arenas of side 8 r, which scale with r, the encoder's field area per r^2 at p = 0.1 is 1.62 at r = 8 and 1.46, 1.44 and 1.45 at r = 16, 23.2 and 32, but 2.55 at r = 3; fields per unit are 2.49, then 2.74 to 2.81, but 1.40 at r = 3. 2D field statistics settle only once fields cover a few hundred cells (r of 16 and above).

**M does more than set a noise floor and Gaussianity.** It sets three things:

- the kernel's far-field noise, SD `1/sqrt(2M)`, and the alias `sqrt(ln N_eff / M)`;
- how far each draw's realised kernel sits from its expectation (r_eff / r within about `1/sqrt(M)`, and the kernel along x and y can differ by 15% at M = 32);
- a spread between units. Units of one draw share frequencies but not weights, so each unit's spectral width varies with squared CV `a_cv2 / (M + 1)`. That spread does not average out over an environment: field counts on the 6864-cell environment have var/mean 1.86 at M = 32 against 0.56 for the GRF, and a unit's count correlates 0.83 with its spectral width.

The Gaussianity effects are the small ones. Single-unit marginals (coverage, Treves-Rolls, W_A, gap CV) are within 1 to 2% of the GRF from M = 32. The per-place normalisation of the sparse code matters only when few units fire at a place (`D p` small): it raises Treves-Rolls by 0.5% at p = 0.1 and M = 512, and by 46% at p = 0.02 and M = 32.

## Scale: metres per scaffold cell

The model's grid modules have periods of 11, 12 and 13 cells, so lambda = 12 cells. Mouse dorsal CA1 fields are about 24 cm (Ziv 2013, median) and the smallest dorsal MEC grid spacing is about 49 to 72 cm, so a dorsal-like code has W / lambda of about 0.3 to 0.5, and a scaffold cell is 4.1 to 6.0 cm.

The current ideal-family codes are far wider: at M = 512, W_A / lambda is 4.7 for the native code at r = 16 (56 cells), 3.0 for the rotated code at r = 16 (35.5 cells) and 2.5 for the viewer's sparse code at r = 23.2 (30 cells). At 4.1 to 6.0 cm a cell, those are 1.2 to 3.4 m fields. Their Pearson kernels are 2.9 lambda (rotated, r = 16) and 3.2 lambda (sparse, r = 23.2) wide at 0.5.

W_A scales with r at fixed p (`W_A / r` is 2.2 rectified, 1.3 at p = 0.1, 1.0 at p = 0.02 on loops, plus pixelation at small r), so the dorsal band needs small r. Settings from the M = 512 encoder (sparse code with per-place normalisation; interpolated in log r between measured r of 2, 2.5, 3, 3.5, 4, 5, 6 and 8):

| variant | target W_A / lambda | r (cells) | FW05 cosine / lambda | FW05 Pearson / lambda | alias, cosine | far floor, cosine | alias, Pearson |
|---|---|---|---|---|---|---|---|
| rotated | 0.5 | 2.41 | 0.62 | 0.43 | 0.395 | 0.317 | 0.113 |
| sparse_p0.3 | 0.4 | 2.20 | 0.44 | 0.36 | 0.254 | 0.180 | 0.089 |
| sparse_p0.3 | 0.5 | 3.11 | 0.61 | 0.51 | 0.259 | 0.182 | 0.095 |
| sparse_p0.2 | 0.4 | 2.55 | 0.44 | 0.40 | 0.182 | 0.117 | 0.073 |
| sparse_p0.2 | 0.5 | 3.55 | 0.61 | 0.55 | 0.185 | 0.117 | 0.075 |
| sparse_p0.1 | 0.4 | 3.04 | 0.44 | 0.42 | 0.106 | 0.058 | 0.052 |
| sparse_p0.1 | 0.5 | 4.18 | 0.61 | 0.58 | 0.100 | 0.060 | 0.045 |
| sparse_p0.05 | 0.4 | 3.46 | 0.45 | 0.44 | 0.064 | 0.028 | 0.036 |
| sparse_p0.05 | 0.5 | 4.74 | 0.61 | 0.59 | 0.057 | 0.029 | 0.029 |
| sparse_p0.02 | 0.3 | 2.22 | 0.26 | 0.25 | 0.030 | 0.011 | 0.019 |
| sparse_p0.02 | 0.4 | 3.97 | 0.44 | 0.44 | 0.026 | 0.013 | 0.015 |
| sparse_p0.02 | 0.5 | 5.43 | 0.61 | 0.60 | 0.029 | 0.011 | 0.018 |

- **At p = 0.1, r of about 2 to 4 cells.** W_A / lambda is 0.32, 0.36, 0.40, 0.44, 0.48 and 0.58 at r = 2, 2.5, 3, 3.5, 4 and 5. So 0.4 needs r = 3.0 and 0.5 needs r = 4.2. That confirms the rough Rice estimate of 3 to 4 cells for the upper half of the band. The lower half needs r near 2, where fields are 3.8 cells wide and definition A's 3-cell minimum, not r, sets W_A: at r = 2, W_A / lambda is 0.29 to 0.35 for every p of 0.2 and below.
- **The kernel is narrower than one grid period.** At those settings the Pearson kernel is 0.42 to 0.58 lambda wide at 0.5 (5 to 7 cells), and 0.28 lambda at r = 2. At 4.1 to 6.0 cm a cell, r = 3 is a kernel SD of 12 to 18 cm and a Pearson FW05 of 21 to 30 cm. At these r the Pearson FW05 is 1.0 to 1.2 times W_A for every p.
- **The alias rises a little.** Small r fits more independent kernel areas into the scaffold, so the raw alias at M = 512 is 0.15 at r = 3 against 0.12 at r = 16. In the rate kernels at p = 0.1 that is 0.05 (Pearson) and 0.11 (cosine, on a floor of 0.058). Holding the raw alias at its r = 16 value needs M of about 750, from `alias^2 ~ ln N_eff / M` with `N_eff` 8.4e4 against 2.4e3.
- **The rectified and native codes cannot get there.** The rectified rotated code reaches 0.5 lambda only at r = 2.4, and the native code is still 0.63 lambda at r = 2.
- **Pixelation is the price.** At r = 3 a 1D field is 4 to 6 cells, and in 2D arenas field areas per r^2 are 75% above their large-r value (2.55 against 1.45 r^2 at p = 0.1). Any 2D single-unit statistic at the dorsal scale is shaped by the lattice as much as by the code.

## Definitions, checked in code

The ideal encoder `ideal:r=16,n_freq=512,seed=0,gain=100` is `IdealEncoder` in `analysis/hopfield_probe/ideal_encoder.py`. `draw_frequencies` draws each wave vector component as `round(Normal(0, sigma_n))` with `sigma_n = Npos / (2 pi r)` and `Npos = 1716`, so the angular frequency `omega = 2 pi n / Npos` has SD `1/r` per component. Unit `i` is `cos(omega_i . p) / sqrt(M)` and unit `M + i` is `sin(omega_i . p) / sqrt(M)`. The code reads the grid code only to recover position, and `IdealEncoder.closed_form` computes the same numbers from `(x, y)`.

- **r** is the SD, in cells, of the Gaussian kernel `E cos(omega . d) = exp(-|d|^2 / 2 r^2)`. Its full width at 0.5 is `2 r sqrt(2 ln 2) = 2.355 r`, so half-width 18.84 cells at r = 16. The suite's 19.10 is the seed-0 draw: with 512 frequencies the realised kernel is 1.4% wider than its expectation. Rounding to integers adds 1/12 to `sigma_n^2`, which is at least 73 for r up to 32, so it changes r by less than 0.1%.
- **M** is `n_freq`, the number of random frequencies. There are `D = 2M` units, a cos and a sin per frequency. Nothing in the code uses M for the unit count, but the viewer and the suite speak of "1024 units". If M was meant as the unit count, every M below halves.
- **Variants** (`analysis/ideal_family/codes.py`, built from the closed form; `test_family_reproduces_the_viewer_populations` checks seed 0 against the suite's populations to 1e-6):
  - native: `relu(z)`, the plane-wave units rectified, as in the suite's primary convention;
  - rotated: `relu(Q z)`, with the viewer's `Q = H D3 H D2 H D1 / D^1.5` (`H` Sylvester Walsh-Hadamard, `D_k` random signs from `default_rng(seed)`), applied by a fast transform. Seed s uses frequency seed s and rotation seed s, so seed 0 is the viewer's `ideal_rot`;
  - sparse: `relu(Q z - t / sqrt(D)) / |.|`, with `t = Phi^-1(1 - p)` and unit length at each place. At r = 23.2, M = 512, seed 0, p = 0.1 this is the viewer's `ideal_sparse`;
  - sparse_raw: the same without the per-place normalisation. The rotated variant is sparse_raw at p = 0.5.
- **Measures** reuse `analysis/coding_comparison/measures.py` unchanged: definition A fields (a run at or above 20% of the unit's peak in that environment, at least 3 cells or 9 cells in 2D, 4-connected), Treves-Rolls sparsity `mean(r)^2 / mean(r^2)`, cosine and Pearson lag profiles. `analysis/ideal_family/measure.py` adds per-unit coverage (fraction of positions with rate above 0), the 95% area (the fewest bins that hold 95% of the unit's summed rate, as a fraction of the environment; the brief cites Ledergerber 2021 for this measure, and I implemented that wording without checking the paper's exact definition), fields per unit per cell and per field width, single-field share, variance over mean of counts, gap CV between a unit's field centres (units with at least 6 fields, the wrap gap included on loops), and fields on a loop joined across its seam.

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

with the expectation over the maximum's distribution (`theory.def_A_1d`). For the rotated code (t = 0) on a 1716-cell loop at r = 16, `E[M_env] = 2.56`, so `u_A = 0.51`, and `W_A = 35.0` cells (35.2 with the 3-cell minimum). The suite measured 35.0. Plugging `E[M_env]` straight in gives 34.9; a level of 0.7 would give 31, because a 1716-cell loop spans only 17 kernel lengths `2 pi r` and its maximum sits near 2.6, not 3.5.

Definition A also drops runs shorter than 3 cells. A continuous excursion of length l covers at least 3 lattice cells with probability `clip(l - 2, 0, 1)`, so the cut acts at about `m = 2.5`. With excursion lengths taken as Rayleigh with the Rice mean (the high-level limit, where an excursion is a parabolic cap), the kept fraction is `exp(-m^2 / 2 s^2)` and the kept mean is `m + sqrt(2 pi) s exp(m^2 / 2 s^2) (1 - Phi(m/s))`, with `s = L(u) / sqrt(pi/2)`. This matters only when W_A is a few cells; with it the predicted share of a loop inside fields matches GRF loops within 1% down to r = 3.

An active unit always has the field around its own maximum, which Rice's average count leaves out when units have few fields. Given the maximum M, `E[N | M] = 1 + L (nu(u_A) - nu(M))`, so fields per active unit are `L E[nu(u_A)] + 1 - E[L nu(M) | M > t]`. At p = 0.02 and r = 23.2 this is 1.58 against 1.56 measured, where plain Rice gives 1.13.

Because `u_A` depends on `M_env`, and `M_env` depends on the environment's length in units of `2 pi r`, definition-A statistics depend on p and on `L / r`, not on p alone. The dependence is logarithmic.

### 4. Fields in 2D: the Euler characteristic

For an isotropic 2D GRF with `lambda_2 = 1/r^2` the expected Euler characteristic (EC) of the excursion set per cell squared is `u exp(-u^2/2) / ((2 pi)^1.5 r^2)`. At levels where every field is a simply connected blob, EC counts fields, so

- fields per cell squared: `u exp(-u^2/2) / ((2 pi)^1.5 r^2)`;
- mean field area: `(2 pi)^1.5 r^2 (1 - Phi(u)) exp(u^2/2) / u`;
- fields in a square of side S, boundary fields included: `1 - Phi(u) + 2 S exp(-u^2/2) / (2 pi r) + S^2 u exp(-u^2/2) / ((2 pi)^1.5 r^2)` (Adler and Taylor 2007).

On GRF samples the count holds within 3% for `u >= 0.5` and the area within 5% for `u >= 0.75`. Below `u = 0.5` fields have holes and merge, and components outnumber the EC: by 18% at `u = 0.25` and 4 times at `u = 0`. The EC density is 0 at `u = 0` and negative below, while the excursion set percolates for `u <= 0`. So the EC is marginal for the rectified code in arenas, whose definition-A level is 0.37 to 0.6 (there it still counts fields within 3% in the arenas measured), and reliable for sparse codes with `p <= 0.3`. The arena maximum follows from `P(max > u) = 1 - exp(-EC(u))`; on GRF samples it is within 3% for sides of 8 r and more, and 10% low at 4 r.

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

So identically distributed units can be overdispersed under definition A, but only once a unit has many fields: about 200 for the rectified code (a GRF reaches 1.02 on a 6864-cell loop at r = 4, about 770 field widths), and from about 10 to 30 for p of 0.1 and below (`figures/counts.pdf`, right panel). The renewal approximation puts the gap CV near `sqrt(F)`, seen through the small-sample bias of a CV over a handful of gaps (Result 11).

## Predicted against measured

Selected rows; every table in full is in `/orcd/pool/003/jackking/cls_runs/results/ideal_family/v1/tables.md` (written by `analyse.py`) and `summary.json`. Values are mean ± SD over 3 seeds, theory in brackets or in its own column. `r_eff` is the draw's realised kernel SD (the nominal r for GRFs).

### Raw kernel and alias (M = 512 at every r; every M at r = 16)

| r | M | FW05 / r (2.355) | r_eff / r | FW02 / FW05 (1.524) | zero crossing / r | far SD x sqrt(2M) (1) | alias max | alias predicted |
|---|---|---|---|---|---|---|---|---|
| 2 | 512 | 2.422 ± 0.013 | 1.029 ± 0.005 | 1.557 ± 0.008 | 3.74 ± 0.27 | 1.000 ± 0.002 | 0.144 ± 0.005 | 0.154 |
| 2.5 | 512 | 2.397 ± 0.014 | 1.019 ± 0.006 | 1.567 ± 0.009 | 3.75 ± 0.29 | 0.998 ± 0.002 | 0.148 ± 0.002 | 0.151 |
| 3 | 512 | 2.405 ± 0.013 | 1.022 ± 0.005 | 1.558 ± 0.008 | 3.72 ± 0.25 | 1.001 ± 0.004 | 0.157 ± 0.011 | 0.149 |
| 3.5 | 512 | 2.403 ± 0.014 | 1.021 ± 0.006 | 1.556 ± 0.009 | 3.70 ± 0.22 | 0.999 ± 0.002 | 0.153 ± 0.008 | 0.147 |
| 4 | 512 | 2.400 ± 0.014 | 1.019 ± 0.006 | 1.555 ± 0.009 | 3.69 ± 0.21 | 0.996 ± 0.002 | 0.139 ± 0.008 | 0.145 |
| 5 | 512 | 2.397 ± 0.014 | 1.019 ± 0.006 | 1.554 ± 0.009 | 3.72 ± 0.24 | 1.001 ± 0.004 | 0.138 ± 0.003 | 0.142 |
| 6 | 512 | 2.398 ± 0.013 | 1.019 ± 0.006 | 1.550 ± 0.009 | 3.72 ± 0.26 | 1.002 ± 0.003 | 0.144 ± 0.011 | 0.139 |
| 8 | 512 | 2.398 ± 0.013 | 1.019 ± 0.006 | 1.549 ± 0.007 | 3.71 ± 0.25 | 1.009 ± 0.008 | 0.140 ± 0.014 | 0.135 |
| 16 | 32 | 2.375 ± 0.143 | 1.015 ± 0.061 | 1.621 ± 0.170 | 4.78 ± 1.67 | 1.010 ± 0.015 | 0.480 ± 0.032 | 0.493 |
| 16 | 64 | 2.327 ± 0.160 | 0.991 ± 0.069 | 1.529 ± 0.068 | 3.89 ± 0.53 | 1.015 ± 0.010 | 0.329 ± 0.024 | 0.348 |
| 16 | 128 | 2.348 ± 0.052 | 0.997 ± 0.022 | 1.538 ± 0.051 | 3.29 ± 0.45 | 1.013 ± 0.020 | 0.242 ± 0.004 | 0.246 |
| 16 | 256 | 2.343 ± 0.032 | 0.996 ± 0.014 | 1.534 ± 0.013 | 3.35 ± 0.16 | 1.000 ± 0.011 | 0.181 ± 0.006 | 0.174 |
| 16 | 512 | 2.396 ± 0.013 | 1.018 ± 0.005 | 1.549 ± 0.009 | 3.68 ± 0.17 | 1.007 ± 0.015 | 0.118 ± 0.011 | 0.123 |
| 16 | 1024 | 2.389 ± 0.033 | 1.015 ± 0.014 | 1.539 ± 0.013 | 4.34 ± 0.60 | 1.006 ± 0.010 | 0.083 ± 0.003 | 0.087 |
| 16 | 2048 | 2.362 ± 0.018 | 1.003 ± 0.008 | 1.523 ± 0.009 | 3.97 ± 0.17 | 0.994 ± 0.022 | 0.060 ± 0.005 | 0.062 |
| 23.2 | 512 | 2.397 ± 0.016 | 1.018 ± 0.007 | 1.549 ± 0.007 | 3.61 ± 0.15 | 0.984 ± 0.010 | 0.116 ± 0.006 | 0.116 |
| 32 | 512 | 2.396 ± 0.013 | 1.018 ± 0.006 | 1.550 ± 0.008 | 4.94 ± 2.01 | 0.980 ± 0.032 | 0.113 ± 0.005 | 0.110 |

### Single units on 1716-cell loops, GRF references (prediction in brackets)

| variant | r | coverage (p) | Treves-Rolls, pooled (a(t)) | W_A / r (Rice) | fields per field width (Rice) | u_A (Rice) |
|---|---|---|---|---|---|---|
| rotated | 3 | 0.500 ± 0.000 (0.500) | 0.3186 ± 0.0001 (0.3183) | 2.32 ± 0.00 (2.24) | 0.254 ± 0.000 (0.255) | 0.63 ± 0.00 (0.63) |
| rotated | 8 | 0.500 ± 0.000 (0.500) | 0.3180 ± 0.0001 (0.3183) | 2.16 ± 0.00 (2.15) | 0.286 ± 0.000 (0.286) | 0.57 ± 0.00 (0.57) |
| rotated | 32 | 0.500 ± 0.001 (0.500) | 0.3181 ± 0.0006 (0.3183) | 2.26 ± 0.00 (2.28) | 0.325 ± 0.001 (0.325) | 0.46 ± 0.00 (0.46) |
| sparse_raw_p0.3 | 3 | 0.300 ± 0.000 (0.300) | 0.1812 ± 0.0001 (0.1811) | 1.92 ± 0.00 (1.86) | 0.137 ± 0.000 (0.137) | 1.05 ± 0.00 (1.05) |
| sparse_raw_p0.3 | 8 | 0.300 ± 0.000 (0.300) | 0.1808 ± 0.0001 (0.1811) | 1.71 ± 0.00 (1.71) | 0.162 ± 0.000 (0.162) | 0.99 ± 0.00 (0.99) |
| sparse_raw_p0.3 | 32 | 0.300 ± 0.001 (0.300) | 0.1810 ± 0.0004 (0.1811) | 1.75 ± 0.00 (1.77) | 0.191 ± 0.000 (0.192) | 0.88 ± 0.00 (0.88) |
| sparse_raw_p0.2 | 3 | 0.200 ± 0.000 (0.200) | 0.1176 ± 0.0001 (0.1175) | 1.75 ± 0.00 (1.69) | 0.087 ± 0.000 (0.088) | 1.30 ± 0.00 (1.31) |
| sparse_raw_p0.2 | 8 | 0.200 ± 0.000 (0.200) | 0.1174 ± 0.0001 (0.1175) | 1.51 ± 0.00 (1.51) | 0.107 ± 0.000 (0.108) | 1.24 ± 0.00 (1.24) |
| sparse_raw_p0.2 | 32 | 0.200 ± 0.000 (0.200) | 0.1175 ± 0.0004 (0.1175) | 1.53 ± 0.00 (1.55) | 0.129 ± 0.000 (0.131) | 1.13 ± 0.00 (1.13) |
| sparse_raw_p0.1 | 3 | 0.100 ± 0.000 (0.100) | 0.0570 ± 0.0001 (0.0570) | 1.57 ± 0.00 (1.52) | 0.042 ± 0.000 (0.042) | 1.66 ± 0.00 (1.66) |
| sparse_raw_p0.1 | 8 | 0.100 ± 0.000 (0.100) | 0.0570 ± 0.0000 (0.0570) | 1.30 ± 0.00 (1.30) | 0.056 ± 0.000 (0.056) | 1.59 ± 0.00 (1.59) |
| sparse_raw_p0.1 | 32 | 0.100 ± 0.000 (0.100) | 0.0572 ± 0.0002 (0.0570) | 1.28 ± 0.00 (1.31) | 0.069 ± 0.000 (0.070) | 1.49 ± 0.00 (1.49) |
| sparse_raw_p0.05 | 3 | 0.050 ± 0.000 (0.050) | 0.0279 ± 0.0001 (0.0279) | 1.45 ± 0.00 (1.41) | 0.021 ± 0.000 (0.021) | 1.95 ± 0.00 (1.95) |
| sparse_raw_p0.05 | 8 | 0.050 ± 0.000 (0.050) | 0.0280 ± 0.0000 (0.0279) | 1.16 ± 0.00 (1.17) | 0.030 ± 0.000 (0.030) | 1.88 ± 0.00 (1.88) |
| sparse_raw_p0.05 | 32 | 0.050 ± 0.000 (0.050) | 0.0281 ± 0.0001 (0.0279) | 1.11 ± 0.00 (1.16) | 0.037 ± 0.000 (0.037) | 1.79 ± 0.00 (1.79) |
| sparse_raw_p0.02 | 3 | 0.020 ± 0.000 (0.020) | 0.0110 ± 0.0001 (0.0110) | 1.34 ± 0.00 (1.32) | 0.009 ± 0.000 (0.009) | 2.27 ± 0.00 (2.28) |
| sparse_raw_p0.02 | 8 | 0.020 ± 0.000 (0.020) | 0.0110 ± 0.0001 (0.0110) | 1.02 ± 0.00 (1.05) | 0.013 ± 0.000 (0.013) | 2.21 ± 0.00 (2.21) |
| sparse_raw_p0.02 | 32 | 0.020 ± 0.000 (0.020) | 0.0110 ± 0.0000 (0.0110) | 0.94 ± 0.00 (1.01) | 0.016 ± 0.000 (0.016) | 2.16 ± 0.00 (2.16) |

### Rate kernels, r of 8 and above (theory in brackets)

| variant | code | form | FW05 / r (theory) | FW02 / r (theory) | floor (theory) |
|---|---|---|---|---|---|
| rotated | GRF | cos | 2.983 ± 0.004 (2.984) | - (-) | 0.3186 ± 0.0009 (0.3183) |
| rotated | GRF | pearson | 2.107 ± 0.003 (2.108) | 3.318 ± 0.007 (3.319) | 0.0000 ± 0.0005 (0.0000) |
| rotated | M=512 | cos | 3.002 ± 0.008 (2.984) | - (-) | 0.3204 ± 0.0034 (0.3183) |
| rotated | M=512 | pearson | 2.104 ± 0.003 (2.108) | 3.355 ± 0.014 (3.319) | 0.0027 ± 0.0052 (0.0000) |
| sparse_raw_p0.3 | GRF | cos | 2.294 ± 0.004 (2.295) | 5.059 ± 0.045 (5.055) | 0.1814 ± 0.0008 (0.1811) |
| sparse_raw_p0.3 | GRF | pearson | 1.936 ± 0.003 (1.938) | 3.097 ± 0.008 (3.098) | 0.0001 ± 0.0005 (0.0000) |
| sparse_raw_p0.3 | M=512 | cos | 2.293 ± 0.003 (2.295) | 5.496 ± 0.182 (5.055) | 0.1813 ± 0.0022 (0.1811) |
| sparse_raw_p0.3 | M=512 | pearson | 1.930 ± 0.005 (1.938) | 3.119 ± 0.008 (3.098) | -0.0001 ± 0.0029 (0.0000) |
| sparse_raw_p0.2 | GRF | cos | 2.012 ± 0.004 (2.015) | 3.681 ± 0.015 (3.680) | 0.1179 ± 0.0007 (0.1175) |
| sparse_raw_p0.2 | GRF | pearson | 1.813 ± 0.004 (1.816) | 2.924 ± 0.009 (2.926) | 0.0001 ± 0.0005 (0.0000) |
| sparse_raw_p0.2 | M=512 | cos | 2.009 ± 0.005 (2.015) | 3.746 ± 0.021 (3.680) | 0.1185 ± 0.0010 (0.1175) |
| sparse_raw_p0.2 | M=512 | pearson | 1.807 ± 0.006 (1.816) | 2.939 ± 0.007 (2.926) | 0.0007 ± 0.0012 (0.0000) |
| sparse_raw_p0.1 | GRF | cos | 1.711 ± 0.005 (1.718) | 2.919 ± 0.012 (2.921) | 0.0574 ± 0.0007 (0.0570) |
| sparse_raw_p0.1 | GRF | pearson | 1.629 ± 0.005 (1.637) | 2.650 ± 0.010 (2.655) | 0.0000 ± 0.0006 (0.0000) |
| sparse_raw_p0.1 | M=512 | cos | 1.706 ± 0.007 (1.718) | 2.935 ± 0.013 (2.921) | 0.0576 ± 0.0010 (0.0570) |
| sparse_raw_p0.1 | M=512 | pearson | 1.625 ± 0.007 (1.637) | 2.659 ± 0.010 (2.655) | 0.0002 ± 0.0011 (0.0000) |
| sparse_raw_p0.05 | GRF | cos | 1.510 ± 0.005 (1.526) | 2.518 ± 0.011 (2.530) | 0.0284 ± 0.0006 (0.0279) |
| sparse_raw_p0.05 | GRF | pearson | 1.475 ± 0.005 (1.492) | 2.409 ± 0.010 (2.421) | 0.0001 ± 0.0006 (0.0000) |
| sparse_raw_p0.05 | M=512 | cos | 1.507 ± 0.008 (1.526) | 2.531 ± 0.012 (2.530) | 0.0281 ± 0.0010 (0.0279) |
| sparse_raw_p0.05 | M=512 | pearson | 1.471 ± 0.008 (1.492) | 2.417 ± 0.011 (2.421) | -0.0002 ± 0.0009 (0.0000) |
| sparse_raw_p0.02 | GRF | cos | 1.312 ± 0.006 (1.352) | 2.176 ± 0.010 (2.206) | 0.0115 ± 0.0005 (0.0110) |
| sparse_raw_p0.02 | GRF | pearson | 1.299 ± 0.006 (1.340) | 2.138 ± 0.010 (2.170) | 0.0002 ± 0.0005 (0.0000) |
| sparse_raw_p0.02 | M=512 | cos | 1.312 ± 0.009 (1.352) | 2.190 ± 0.014 (2.206) | 0.0110 ± 0.0009 (0.0110) |
| sparse_raw_p0.02 | M=512 | pearson | 1.300 ± 0.009 (1.340) | 2.151 ± 0.014 (2.170) | -0.0003 ± 0.0009 (0.0000) |
| native | M=512 | cos | 3.706 ± 0.022 (3.650) | - (-) | 0.4079 ± 0.0045 (0.4053) |
| native | M=512 | pearson | 2.108 ± 0.004 (2.104) | 3.435 ± 0.013 (3.392) | 0.0047 ± 0.0078 (0.0000) |

### r that holds FW05 at 37.7 cells

| p | t | rho* (cosine) | r for 37.7-cell FW05, cosine | rho* (Pearson) | r for 37.7-cell FW05, Pearson |
|---|---|---|---|---|---|
| 0.5 | -0.000 | 0.329 | 12.63 | 0.574 | 17.87 |
| 0.3 | 0.524 | 0.518 | 16.42 | 0.625 | 19.45 |
| 0.2 | 0.842 | 0.602 | 18.70 | 0.662 | 20.75 |
| 0.1 | 1.282 | 0.692 | 21.93 | 0.715 | 23.01 |
| 0.05 | 1.645 | 0.747 | 24.69 | 0.757 | 25.26 |
| 0.02 | 2.054 | 0.796 | 27.87 | 0.799 | 28.12 |

### Field counts, GRF references

| variant | r | env | fields per active unit | (Rice + top field) | var / mean | (predicted) | F(u_A) | gap CV (median) | sqrt F (renewal) |
|---|---|---|---|---|---|---|---|---|---|
| rotated | 4 | loop | 51.4 ± 0.0 | 52.4 | 0.584 ± 0.015 | 0.562 | 0.414 | 0.623 ± 0.001 | 0.643 |
| rotated | 4 | long | 192.7 ± 0.1 | 196.9 | 1.065 ± 0.027 | 0.955 | 0.447 | 0.672 ± 0.001 | 0.669 |
| rotated | 16 | loop | 14.9 ± 0.0 | 14.9 | 0.408 ± 0.009 | 0.428 | 0.378 | 0.530 ± 0.000 | 0.615 |
| rotated | 16 | long | 56.0 ± 0.2 | 56.2 | 0.552 ± 0.009 | 0.572 | 0.414 | 0.611 ± 0.002 | 0.643 |
| sparse_raw_p0.3 | 4 | loop | 34.9 ± 0.0 | 35.6 | 0.824 ± 0.007 | 0.876 | 0.592 | 0.744 ± 0.001 | 0.769 |
| sparse_raw_p0.3 | 4 | long | 125.9 ± 0.0 | 128.5 | 1.521 ± 0.024 | 1.478 | 0.629 | 0.798 ± 0.004 | 0.793 |
| sparse_raw_p0.3 | 16 | loop | 11.0 ± 0.0 | 11.0 | 0.572 ± 0.007 | 0.669 | 0.547 | 0.629 ± 0.003 | 0.740 |
| sparse_raw_p0.3 | 16 | long | 39.6 ± 0.1 | 39.7 | 0.799 ± 0.016 | 0.909 | 0.592 | 0.731 ± 0.002 | 0.769 |
| sparse_raw_p0.2 | 4 | loop | 25.2 ± 0.0 | 25.7 | 0.907 ± 0.004 | 1.018 | 0.700 | 0.791 ± 0.002 | 0.837 |
| sparse_raw_p0.2 | 4 | long | 88.8 ± 0.1 | 90.6 | 1.583 ± 0.025 | 1.634 | 0.734 | 0.853 ± 0.002 | 0.857 |
| sparse_raw_p0.2 | 16 | loop | 8.4 ± 0.0 | 8.4 | 0.646 ± 0.012 | 0.809 | 0.659 | 0.664 ± 0.004 | 0.812 |
| sparse_raw_p0.2 | 16 | long | 29.5 ± 0.0 | 29.6 | 0.880 ± 0.015 | 1.066 | 0.700 | 0.783 ± 0.006 | 0.837 |
| sparse_raw_p0.1 | 4 | loop | 14.3 ± 0.0 | 14.6 | 0.908 ± 0.014 | 0.807 | 0.826 | 0.816 ± 0.001 | 0.909 |
| sparse_raw_p0.1 | 4 | long | 48.9 ± 0.1 | 49.7 | 1.491 ± 0.044 | 1.327 | 0.850 | 0.896 ± 0.004 | 0.922 |
| sparse_raw_p0.1 | 16 | loop | 5.2 ± 0.0 | 5.2 | 0.700 ± 0.009 | 0.697 | 0.794 | 0.685 ± 0.006 | 0.891 |
| sparse_raw_p0.1 | 16 | long | 17.6 ± 0.0 | 17.7 | 0.956 ± 0.036 | 0.857 | 0.826 | 0.819 ± 0.005 | 0.909 |
| sparse_raw_p0.05 | 4 | loop | 8.1 ± 0.0 | 8.2 | 0.850 ± 0.008 | 0.791 | 0.900 | 0.797 ± 0.003 | 0.949 |
| sparse_raw_p0.05 | 4 | long | 26.9 ± 0.1 | 27.4 | 1.278 ± 0.030 | 1.156 | 0.916 | 0.898 ± 0.004 | 0.957 |
| sparse_raw_p0.05 | 16 | loop | 3.2 ± 0.0 | 3.2 | 0.707 ± 0.005 | 0.796 | 0.878 | 0.688 ± 0.008 | 0.937 |
| sparse_raw_p0.05 | 16 | long | 10.4 ± 0.0 | 10.5 | 0.912 ± 0.016 | 0.831 | 0.900 | 0.811 ± 0.006 | 0.949 |
| sparse_raw_p0.02 | 4 | loop | 3.8 ± 0.0 | 3.9 | 0.779 ± 0.017 | 0.790 | 0.952 | 0.768 ± 0.005 | 0.976 |
| sparse_raw_p0.02 | 4 | long | 12.3 ± 0.1 | 12.5 | 1.030 ± 0.003 | 0.943 | 0.961 | 0.862 ± 0.005 | 0.980 |
| sparse_raw_p0.02 | 16 | loop | 1.9 ± 0.0 | 1.9 | 0.714 ± 0.010 | 1.047 | 0.942 | 0.762 ± 0.024 | 0.970 |
| sparse_raw_p0.02 | 16 | long | 5.2 ± 0.0 | 5.3 | 0.823 ± 0.007 | 0.788 | 0.952 | 0.787 ± 0.010 | 0.976 |

### Finite M at r = 16

| variant | M | coverage | TR pooled | W_A / r_eff | var/mean, loop | var/mean, 6864 cells | (predicted) | corr(count, sqrt lambda_2) | gap CV |
|---|---|---|---|---|---|---|---|---|---|
| rotated | GRF | 0.500 ± 0.000 | 0.3181 ± 0.0002 | 2.19 ± 0.00 | 0.408 ± 0.009 | 0.560 ± 0.024 | 0.572 | - | 0.530 ± 0.000 |
| rotated | 32 | 0.502 ± 0.002 | 0.3227 ± 0.0001 | 2.21 ± 0.04 | 0.602 ± 0.090 | 1.859 ± 0.682 | 1.371 | 0.83 ± 0.03 | 0.534 ± 0.024 |
| rotated | 64 | 0.501 ± 0.001 | 0.3206 ± 0.0004 | 2.23 ± 0.04 | 0.476 ± 0.047 | 0.969 ± 0.255 | 0.999 | 0.62 ± 0.06 | 0.531 ± 0.011 |
| rotated | 128 | 0.501 ± 0.001 | 0.3193 ± 0.0009 | 2.19 ± 0.06 | 0.448 ± 0.018 | 0.791 ± 0.126 | 0.813 | 0.51 ± 0.09 | 0.532 ± 0.014 |
| rotated | 256 | 0.500 ± 0.001 | 0.3188 ± 0.0009 | 2.19 ± 0.02 | 0.427 ± 0.014 | 0.662 ± 0.073 | 0.693 | 0.37 ± 0.01 | 0.535 ± 0.007 |
| rotated | 512 | 0.500 ± 0.001 | 0.3184 ± 0.0004 | 2.18 ± 0.01 | 0.424 ± 0.009 | 0.611 ± 0.036 | 0.632 | 0.29 ± 0.04 | 0.536 ± 0.003 |
| rotated | 1024 | 0.500 ± 0.000 | 0.3185 ± 0.0002 | 2.19 ± 0.01 | 0.422 ± 0.006 | 0.591 ± 0.018 | 0.599 | 0.23 ± 0.01 | 0.534 ± 0.002 |
| rotated | 2048 | 0.500 ± 0.000 | 0.3183 ± 0.0001 | 2.19 ± 0.01 | 0.408 ± 0.002 | 0.571 ± 0.012 | 0.587 | 0.16 ± 0.01 | 0.530 ± 0.003 |
| sparse_raw_p0.1 | GRF | 0.100 ± 0.000 | 0.0568 ± 0.0000 | 1.27 ± 0.00 | 0.700 ± 0.009 | 0.930 ± 0.046 | 0.857 | - | 0.685 ± 0.006 |
| sparse_raw_p0.1 | 32 | 0.102 ± 0.001 | 0.0589 ± 0.0009 | 1.27 ± 0.03 | 0.748 ± 0.139 | 1.667 ± 0.617 | 1.120 | 0.52 ± 0.05 | 0.687 ± 0.036 |
| sparse_raw_p0.1 | 64 | 0.101 ± 0.000 | 0.0582 ± 0.0002 | 1.28 ± 0.03 | 0.676 ± 0.041 | 1.026 ± 0.107 | 0.995 | 0.24 ± 0.06 | 0.667 ± 0.020 |
| sparse_raw_p0.1 | 128 | 0.101 ± 0.001 | 0.0575 ± 0.0004 | 1.28 ± 0.03 | 0.674 ± 0.030 | 1.170 ± 0.215 | 0.934 | 0.24 ± 0.07 | 0.680 ± 0.007 |
| sparse_raw_p0.1 | 256 | 0.100 ± 0.000 | 0.0573 ± 0.0002 | 1.28 ± 0.00 | 0.665 ± 0.020 | 1.016 ± 0.133 | 0.895 | 0.15 ± 0.03 | 0.689 ± 0.006 |
| sparse_raw_p0.1 | 512 | 0.100 ± 0.000 | 0.0572 ± 0.0001 | 1.27 ± 0.01 | 0.691 ± 0.015 | 1.015 ± 0.028 | 0.876 | 0.12 ± 0.03 | 0.693 ± 0.005 |
| sparse_raw_p0.1 | 1024 | 0.100 ± 0.000 | 0.0572 ± 0.0001 | 1.27 ± 0.01 | 0.697 ± 0.020 | 0.946 ± 0.047 | 0.866 | 0.08 ± 0.02 | 0.690 ± 0.003 |
| sparse_raw_p0.1 | 2048 | 0.100 ± 0.000 | 0.0570 ± 0.0001 | 1.27 ± 0.01 | 0.701 ± 0.009 | 0.953 ± 0.017 | 0.862 | 0.07 ± 0.02 | 0.684 ± 0.004 |

### Per-unit spectral spread

| M | E sum w^2 x (M+1)/2 (Dirichlet: 1) | CV^2(lambda_2) x (M+1) (a_cv2, 2 in expectation) |
|---|---|---|
| 32 | 0.957 ± 0.008 | 1.90 ± 0.76 |
| 64 | 1.027 ± 0.032 | 1.97 ± 0.27 |
| 128 | 0.995 ± 0.013 | 2.12 ± 0.60 |
| 256 | 1.002 ± 0.007 | 2.17 ± 0.48 |
| 512 | 1.000 ± 0.002 | 2.21 ± 0.38 |
| 1024 | 0.999 ± 0.002 | 1.97 ± 0.20 |
| 2048 | 1.000 ± 0.001 | 2.12 ± 0.11 |

### Rate-kernel alias at the worst displacement

| variant | configs | cosine: measured - predicted, mean | max abs | Pearson: mean | max abs | (alias - floor) / raw alias, median | dK/drho at 0 (theory) |
|---|---|---|---|---|---|---|---|
| native | 123 | -0.0004 | 0.0063 | - | - | 2.288 | - |
| rotated | 123 | -0.0002 | 0.0118 | -0.0011 | 0.0179 | 0.520 | 0.500 |
| sparse_p0.5 | 123 | -0.0002 | 0.0118 | -0.0011 | 0.0179 | 0.520 | 0.500 |
| sparse_p0.1 | 123 | -0.0024 | 0.0274 | -0.0039 | 0.0324 | 0.319 | 0.254 |
| sparse_raw_p0.1 | 123 | -0.0024 | 0.0274 | -0.0039 | 0.0324 | 0.319 | 0.254 |
| sparse_p0.02 | 123 | +0.0030 | 0.0774 | +0.0013 | 0.0702 | 0.131 | 0.081 |

### Native code on loops

| r | M | W_A / r (3.433) | W_A / draw's prediction | TR active (0.405) | TR / draw's prediction | coverage (0.5) | field coverage (0.436) | var/mean loop | (predicted) | field size CV |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 512 | 3.77 ± 0.01 | 1.080 ± 0.006 | 0.406 ± 0.001 | 1.000 ± 0.000 | 0.500 ± 0.000 | 0.424 ± 0.001 | 45.44 ± 0.40 | 62.19 | 1.74 ± 0.17 |
| 2.5 | 512 | 3.57 ± 0.02 | 1.024 ± 0.007 | 0.406 ± 0.000 | 1.000 ± 0.001 | 0.500 ± 0.000 | 0.434 ± 0.000 | 44.41 ± 0.61 | 49.75 | 1.78 ± 0.12 |
| 3 | 512 | 3.52 ± 0.02 | 1.007 ± 0.005 | 0.406 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 | 0.436 ± 0.000 | 40.49 ± 0.61 | 41.46 | 1.71 ± 0.11 |
| 3.5 | 512 | 3.50 ± 0.02 | 1.003 ± 0.002 | 0.406 ± 0.000 | 1.000 ± 0.001 | 0.500 ± 0.001 | 0.436 ± 0.001 | 35.56 ± 0.44 | 35.54 | 1.69 ± 0.04 |
| 4 | 512 | 3.50 ± 0.01 | 1.002 ± 0.001 | 0.407 ± 0.001 | 1.000 ± 0.001 | 0.500 ± 0.000 | 0.437 ± 0.000 | 31.37 ± 0.42 | 31.10 | 1.69 ± 0.11 |
| 5 | 512 | 3.49 ± 0.02 | 1.001 ± 0.001 | 0.407 ± 0.001 | 1.000 ± 0.001 | 0.500 ± 0.000 | 0.437 ± 0.000 | 25.35 ± 0.49 | 24.88 | 1.63 ± 0.09 |
| 6 | 512 | 3.50 ± 0.02 | 1.001 ± 0.002 | 0.408 ± 0.001 | 1.000 ± 0.001 | 0.500 ± 0.000 | 0.437 ± 0.000 | 21.12 ± 0.40 | 20.73 | 1.61 ± 0.08 |
| 8 | 512 | 3.50 ± 0.01 | 1.002 ± 0.001 | 0.409 ± 0.001 | 1.000 ± 0.000 | 0.500 ± 0.000 | 0.437 ± 0.000 | 15.82 ± 0.30 | 15.55 | 1.56 ± 0.07 |
| 16 | 32 | 3.53 ± 0.22 | 1.007 ± 0.013 | 0.415 ± 0.007 | 1.004 ± 0.009 | 0.503 ± 0.005 | 0.441 ± 0.005 | 7.79 ± 1.32 | 7.77 | 1.44 ± 0.29 |
| 16 | 64 | 3.44 ± 0.23 | 1.005 ± 0.006 | 0.412 ± 0.002 | 1.002 ± 0.004 | 0.502 ± 0.002 | 0.439 ± 0.002 | 7.54 ± 0.77 | 7.77 | 1.39 ± 0.13 |
| 16 | 128 | 3.43 ± 0.10 | 1.004 ± 0.003 | 0.411 ± 0.000 | 1.002 ± 0.002 | 0.501 ± 0.001 | 0.439 ± 0.001 | 7.81 ± 0.87 | 7.77 | 1.36 ± 0.03 |
| 16 | 256 | 3.42 ± 0.06 | 1.002 ± 0.002 | 0.411 ± 0.001 | 1.000 ± 0.001 | 0.500 ± 0.001 | 0.438 ± 0.001 | 7.80 ± 0.32 | 7.77 | 1.37 ± 0.03 |
| 16 | 512 | 3.50 ± 0.02 | 1.002 ± 0.002 | 0.412 ± 0.001 | 1.000 ± 0.002 | 0.500 ± 0.000 | 0.438 ± 0.000 | 7.90 ± 0.16 | 7.77 | 1.42 ± 0.02 |
| 16 | 1024 | 3.50 ± 0.05 | 1.001 ± 0.002 | 0.412 ± 0.001 | 1.000 ± 0.001 | 0.500 ± 0.001 | 0.438 ± 0.000 | 7.82 ± 0.10 | 7.77 | 1.42 ± 0.03 |
| 16 | 2048 | 3.45 ± 0.03 | 1.001 ± 0.001 | 0.412 ± 0.001 | 1.000 ± 0.001 | 0.500 ± 0.000 | 0.438 ± 0.000 | 7.74 ± 0.18 | 7.77 | 1.42 ± 0.02 |
| 23.2 | 512 | 3.51 ± 0.02 | 1.002 ± 0.004 | 0.416 ± 0.001 | 1.000 ± 0.003 | 0.500 ± 0.001 | 0.439 ± 0.001 | 5.43 ± 0.10 | 5.36 | 1.34 ± 0.02 |
| 32 | 512 | 3.50 ± 0.02 | 0.998 ± 0.003 | 0.420 ± 0.001 | 0.998 ± 0.002 | 0.499 ± 0.001 | 0.438 ± 0.001 | 3.93 ± 0.07 | 3.89 | 1.24 ± 0.02 |

### 2D arenas, GRF references, r of 8 and above

| variant | side | u_A | fields per unit | (EC, 9-cell cut) | field area / r^2 | (EC in the box, 9-cell cut) | single-field share (active) | var/mean | coverage | 95% area (active) | TR (active) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rotated | 4 r | 0.37 ± 0.00 | 1.83 ± 0.08 | 1.87 ± 0.03 | 3.13 ± 0.17 | 3.05 ± 0.05 | 0.389 ± 0.039 | 0.347 ± 0.016 | 0.499 ± 0.005 | 0.385 ± 0.004 | 0.352 ± 0.004 |
| rotated | 8 r | 0.49 ± 0.00 | 4.19 ± 0.17 | 4.29 ± 0.05 | 4.78 ± 0.21 | 4.66 ± 0.06 | 0.022 ± 0.005 | 0.493 ± 0.014 | 0.499 ± 0.003 | 0.377 ± 0.003 | 0.330 ± 0.002 |
| rotated | 16 r | 0.60 ± 0.00 | 12.34 ± 0.32 | 12.54 ± 0.12 | 5.73 ± 0.14 | 5.61 ± 0.06 | 0.000 | 0.559 ± 0.015 | 0.500 ± 0.001 | 0.375 ± 0.001 | 0.322 ± 0.001 |
| sparse_raw_p0.3 | 4 r | 0.79 ± 0.00 | 1.69 ± 0.08 | 1.69 ± 0.05 | 1.99 ± 0.12 | 2.02 ± 0.05 | 0.437 ± 0.041 | 0.355 ± 0.012 | 0.300 ± 0.004 | 0.233 ± 0.005 | 0.210 ± 0.004 |
| sparse_raw_p0.3 | 8 r | 0.91 ± 0.00 | 4.17 ± 0.17 | 4.23 ± 0.08 | 2.78 ± 0.12 | 2.75 ± 0.06 | 0.015 ± 0.005 | 0.441 ± 0.013 | 0.299 ± 0.003 | 0.222 ± 0.002 | 0.192 ± 0.002 |
| sparse_raw_p0.3 | 16 r | 1.02 ± 0.00 | 12.61 ± 0.39 | 12.83 ± 0.22 | 3.15 ± 0.10 | 3.08 ± 0.06 | 0.000 | 0.478 ± 0.015 | 0.300 ± 0.001 | 0.221 ± 0.001 | 0.185 ± 0.001 |
| sparse_raw_p0.1 | 4 r | 1.44 ± 0.00 | 1.05 ± 0.07 | 1.00 ± 0.05 | 1.10 ± 0.09 | 1.20 ± 0.05 | 0.696 ± 0.035 | 0.526 ± 0.017 | 0.099 ± 0.003 | 0.097 ± 0.005 | 0.086 ± 0.002 |
| sparse_raw_p0.1 | 8 r | 1.52 ± 0.00 | 2.74 ± 0.14 | 2.73 ± 0.10 | 1.46 ± 0.07 | 1.52 ± 0.05 | 0.144 ± 0.026 | 0.529 ± 0.015 | 0.099 ± 0.002 | 0.074 ± 0.001 | 0.064 ± 0.001 |
| sparse_raw_p0.1 | 16 r | 1.62 ± 0.00 | 8.19 ± 0.35 | 8.22 ± 0.27 | 1.64 ± 0.07 | 1.63 ± 0.05 | 0.000 ± 0.000 | 0.625 ± 0.028 | 0.100 ± 0.001 | 0.073 ± 0.000 | 0.060 ± 0.000 |
| sparse_raw_p0.02 | 4 r | 2.15 ± 0.00 | 0.36 ± 0.03 | 0.33 ± 0.02 | 0.67 ± 0.07 | 0.75 ± 0.05 | 0.908 ± 0.009 | 0.814 ± 0.024 | 0.020 ± 0.001 | 0.046 ± 0.005 | 0.039 ± 0.001 |
| sparse_raw_p0.02 | 8 r | 2.17 ± 0.00 | 1.08 ± 0.08 | 1.03 ± 0.06 | 0.82 ± 0.06 | 0.93 ± 0.05 | 0.628 ± 0.032 | 0.695 ± 0.023 | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.018 ± 0.001 |
| sparse_raw_p0.02 | 16 r | 2.24 ± 0.00 | 3.24 ± 0.18 | 3.21 ± 0.17 | 0.96 ± 0.06 | 1.00 ± 0.05 | 0.107 ± 0.023 | 0.705 ± 0.014 | 0.020 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 |
| sparse_p0.1 | 4 r | 1.44 ± 0.00 | 1.05 ± 0.07 | 1.00 ± 0.05 | 1.10 ± 0.09 | 1.20 ± 0.05 | 0.694 ± 0.036 | 0.527 ± 0.017 | 0.099 ± 0.003 | 0.097 ± 0.005 | 0.086 ± 0.002 |
| sparse_p0.1 | 8 r | 1.52 ± 0.00 | 2.74 ± 0.14 | 2.73 ± 0.10 | 1.46 ± 0.07 | 1.52 ± 0.05 | 0.145 ± 0.025 | 0.529 ± 0.018 | 0.099 ± 0.002 | 0.074 ± 0.001 | 0.064 ± 0.001 |
| sparse_p0.1 | 16 r | 1.62 ± 0.00 | 8.22 ± 0.35 | 8.22 ± 0.27 | 1.64 ± 0.07 | 1.63 ± 0.05 | 0.000 ± 0.000 | 0.620 ± 0.025 | 0.100 ± 0.001 | 0.073 ± 0.000 | 0.060 ± 0.000 |

### Treves-Rolls on short tracks, r = 16 and 23.2

| variant | code | r | track (W_A) | share firing | TR, firing units | TR, silent = 0 | TR, pooled | a(t) | share with a field | single-field share |
|---|---|---|---|---|---|---|---|---|---|---|
| native | M=512 | 16 | 4.6 | 0.959 ± 0.001 | 0.428 ± 0.001 | 0.411 ± 0.001 | 0.405 ± 0.001 | 0.405 | 0.959 ± 0.001 | 0.306 ± 0.003 |
| native | M=512 | 16 | 15.4 | 0.989 ± 0.001 | 0.413 ± 0.001 | 0.408 ± 0.000 | 0.406 ± 0.000 | 0.405 | 0.989 ± 0.001 | 0.090 ± 0.003 |
| native | M=512 | 23.2 | 4.6 | 0.960 ± 0.003 | 0.430 ± 0.001 | 0.413 ± 0.002 | 0.406 ± 0.001 | 0.405 | 0.960 ± 0.003 | 0.305 ± 0.004 |
| native | M=512 | 23.2 | 15.4 | 0.983 ± 0.001 | 0.417 ± 0.001 | 0.410 ± 0.001 | 0.406 ± 0.001 | 0.405 | 0.983 ± 0.001 | 0.082 ± 0.002 |
| rotated | GRF | 16 | 4.6 | 0.991 ± 0.000 | 0.367 ± 0.001 | 0.364 ± 0.001 | 0.318 ± 0.001 | 0.318 | 0.990 ± 0.001 | 0.286 ± 0.003 |
| rotated | GRF | 16 | 15.4 | 1.000 | 0.334 ± 0.001 | 0.334 ± 0.001 | 0.317 ± 0.001 | 0.318 | 1.000 | 0.002 ± 0.001 |
| rotated | M=512 | 16 | 4.6 | 0.992 ± 0.001 | 0.367 ± 0.001 | 0.365 ± 0.001 | 0.319 ± 0.001 | 0.318 | 0.992 ± 0.001 | 0.288 ± 0.004 |
| rotated | M=512 | 16 | 15.4 | 1.000 | 0.336 ± 0.001 | 0.336 ± 0.001 | 0.320 ± 0.000 | 0.318 | 1.000 | 0.002 ± 0.001 |
| rotated | M=512 | 23.2 | 4.6 | 0.993 ± 0.001 | 0.365 ± 0.003 | 0.363 ± 0.003 | 0.318 ± 0.003 | 0.318 | 0.993 ± 0.001 | 0.270 ± 0.000 |
| rotated | M=512 | 23.2 | 15.4 | 1.000 | 0.335 ± 0.001 | 0.335 ± 0.001 | 0.319 ± 0.001 | 0.318 | 1.000 | 0.001 ± 0.001 |
| sparse_p0.1 | GRF | 16 | 4.6 | 0.445 ± 0.005 | 0.180 ± 0.001 | 0.080 ± 0.001 | 0.058 ± 0.001 | 0.057 | 0.428 ± 0.005 | 0.903 ± 0.005 |
| sparse_p0.1 | GRF | 16 | 15.4 | 0.821 ± 0.003 | 0.088 ± 0.000 | 0.072 ± 0.001 | 0.057 ± 0.001 | 0.057 | 0.812 ± 0.004 | 0.574 ± 0.007 |
| sparse_p0.1 | M=512 | 16 | 4.6 | 0.444 ± 0.003 | 0.180 ± 0.002 | 0.080 ± 0.000 | 0.058 ± 0.001 | 0.057 | 0.426 ± 0.002 | 0.903 ± 0.007 |
| sparse_p0.1 | M=512 | 16 | 15.4 | 0.826 ± 0.002 | 0.089 ± 0.000 | 0.074 ± 0.000 | 0.058 ± 0.000 | 0.057 | 0.816 ± 0.003 | 0.570 ± 0.007 |
| sparse_p0.1 | M=512 | 23.2 | 4.6 | 0.446 ± 0.006 | 0.179 ± 0.001 | 0.080 ± 0.001 | 0.058 ± 0.001 | 0.057 | 0.436 ± 0.006 | 0.896 ± 0.007 |
| sparse_p0.1 | M=512 | 23.2 | 15.4 | 0.824 ± 0.003 | 0.089 ± 0.001 | 0.073 ± 0.001 | 0.057 ± 0.001 | 0.057 | 0.819 ± 0.002 | 0.563 ± 0.003 |
| sparse_raw_p0.1 | GRF | 16 | 4.6 | 0.443 ± 0.005 | 0.181 ± 0.001 | 0.080 ± 0.001 | 0.057 ± 0.001 | 0.057 | 0.426 ± 0.006 | 0.903 ± 0.005 |
| sparse_raw_p0.1 | GRF | 16 | 15.4 | 0.821 ± 0.003 | 0.088 ± 0.000 | 0.072 ± 0.001 | 0.056 ± 0.001 | 0.057 | 0.812 ± 0.004 | 0.573 ± 0.008 |
| sparse_raw_p0.1 | M=512 | 16 | 4.6 | 0.444 ± 0.003 | 0.180 ± 0.002 | 0.080 ± 0.000 | 0.057 ± 0.001 | 0.057 | 0.426 ± 0.002 | 0.905 ± 0.008 |
| sparse_raw_p0.1 | M=512 | 16 | 15.4 | 0.826 ± 0.002 | 0.089 ± 0.000 | 0.073 ± 0.000 | 0.057 ± 0.000 | 0.057 | 0.816 ± 0.003 | 0.569 ± 0.006 |
| sparse_raw_p0.1 | M=512 | 23.2 | 4.6 | 0.446 ± 0.006 | 0.179 ± 0.001 | 0.080 ± 0.001 | 0.057 ± 0.001 | 0.057 | 0.435 ± 0.005 | 0.898 ± 0.007 |
| sparse_raw_p0.1 | M=512 | 23.2 | 15.4 | 0.824 ± 0.003 | 0.089 ± 0.001 | 0.073 ± 0.001 | 0.057 ± 0.001 | 0.057 | 0.819 ± 0.002 | 0.562 ± 0.003 |
| sparse_raw_p0.02 | GRF | 16 | 4.6 | 0.102 ± 0.002 | 0.164 ± 0.005 | 0.017 ± 0.000 | 0.011 ± 0.000 | 0.011 | 0.096 ± 0.001 | 0.985 ± 0.005 |
| sparse_raw_p0.02 | GRF | 16 | 15.4 | 0.267 ± 0.004 | 0.058 ± 0.001 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.011 | 0.258 ± 0.005 | 0.927 ± 0.006 |
| sparse_raw_p0.02 | M=512 | 16 | 4.6 | 0.104 ± 0.002 | 0.161 ± 0.003 | 0.017 ± 0.001 | 0.011 ± 0.001 | 0.011 | 0.097 ± 0.003 | 0.982 ± 0.002 |
| sparse_raw_p0.02 | M=512 | 16 | 15.4 | 0.272 ± 0.002 | 0.059 ± 0.001 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.011 | 0.262 ± 0.003 | 0.917 ± 0.001 |
| sparse_raw_p0.02 | M=512 | 23.2 | 4.6 | 0.102 ± 0.002 | 0.163 ± 0.003 | 0.017 ± 0.000 | 0.011 ± 0.000 | 0.011 | 0.098 ± 0.002 | 0.988 ± 0.001 |
| sparse_raw_p0.02 | M=512 | 23.2 | 15.4 | 0.266 ± 0.005 | 0.059 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.011 | 0.261 ± 0.005 | 0.922 ± 0.005 |


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
2. **"No setting gives variance over mean above 1."** True at a fixed level, where F(u) never exceeds 1. False under definition A: the peak-relative level differs between units and adds variance in proportion to the number of fields per unit, so identically distributed GRF units exceed 1 on long environments (0.96 at p = 0.1 and r = 16 over 6864 cells, 2.0 to 2.4 at r = 2). Finite M adds more (1.86 at M = 32, r = 16). It is still far from Rich's 7.1, and on Rich's 74 field widths it stays at 0.46 to 0.71.
3. **"Every scale-free statistic depends on p alone."** Only coverage, pooled Treves-Rolls and the kernel's shape do. See the structural claim section.
4. **My first count-dispersion formula, and both forms at small r.** Adding `N 0.04 u_A^2 Var(M_env)` to F(u_A) fits the rectified code (within 0.02 on GRF loops) but overshoots by 4 times at p = 0.1, because a unit with more fields also has a higher maximum. The peak-process form fixes p of 0.1 and below. Neither form fits p = 0.2 to 0.3 (u_A near 1), where it is off by up to 0.3, nor r of 2 to 3 on the long environment, where the measured 1.4 to 2.4 exceeds both by up to 0.8. The GRF references are the numbers to use there.
5. **My first between-unit spread.** I first wrote `CV^2(lambda_2) = 5/M`, then `4/(M+1)` for independent frequencies per unit. Units of one draw share frequencies, so it is `a_cv2/(M+1)`, about `2/(M+1)`, and the per-draw `a_cv2` varies from 1 to 3 at small M. With the draw's own value, the predicted long-environment dispersion still captures only about two thirds of the excess at M = 32 (1.37 predicted against 1.86 measured for the rectified code).
6. **The EC area in arenas.** The interior EC area overstated measured areas by up to 35%, because arena fields are cut by the walls. The in-box form `(1 - Phi(u)) S^2 / EC` with the 9-cell minimum brings p of 0.1 and above within 4% in arenas of 8 r and 16 r. At p = 0.02 and in 4 r arenas the exponential area law over-corrects the 9-cell cut by 9 to 13%.
7. **The renewal gap CV, taken at face value.** Gap CV between a unit's fields is 0.53 (rectified) to 0.75 (p = 0.02), below `sqrt(F(u_A))` (0.61 to 0.97). Most of the difference is the estimator: the median CV of n exponential gaps is 0.78 at n = 6, 0.89 at n = 15 and 0.94 at n = 30, and Rich's measure uses units with 6 or more fields. With that factor, `sqrt(F)` predicts 0.54 for the rectified code (about 15 fields per loop at r = 16) and 0.71 at p = 0.1 (6 to 8 fields), against 0.53 and 0.70 measured. I did not derive the gap distribution itself.
8. **The 2D maximum heuristic in small arenas.** `P(max > u) = 1 - exp(-EC)` is 10% low at a side of 4 r and 17% low at 2 r. The tables use each unit's measured level in 2D instead.

## What was run, pruned and approximated

The grid is `analysis/ideal_family/run.py::grid()`, 156 configurations, each one JSON in `/orcd/pool/003/jackking/cls_runs/results/ideal_family/v1/configs/`.

- **Encoder draws.** r in {4, 8, 16, 23.2, 32} crossed with M in {32, 64, 128, 256, 512, 1024, 2048}, 3 seeds: 105 configurations, all with 1D measures and the whole-scaffold alias. Nothing in this cross was pruned.
- **Dorsal-scale draws.** r in {2, 2.5, 3, 3.5, 5, 6} at M = 512, 3 seeds, with 2D.
- **GRF references.** Direct Gaussian random fields (1024 independent units, `analysis/ideal_family/grf.py`) at all 11 values of r, 3 seeds, with 2D. These are the M = infinity limit of the rotated family on the same integer lattice.
- **Variants.** Every configuration runs native (encoder only), rotated, sparse at p in {0.5, 0.3, 0.2, 0.1, 0.05, 0.02}, and sparse_raw at the five p below 0.5 (sparse_raw at 0.5 is the rotated variant). No p was pruned.
- **2D, pruned.** Arenas only at M = 512 (every r) and at r = 16 (every M). Sides 4 r, 8 r and 16 r, 2 places each. Arenas below 8 cells a side are skipped, which drops the 4 r arena at r = 2.
- **1D environments.** The suite's zigzag tracks at Rich's five lengths were not rerun. In their place: 8 full scaffold loops (4 rows, 4 columns), a 6864-cell environment made of the 4 rows with one peak per unit (and a second one from the 4 columns), and straight tracks of 4.6 and 15.4 W_A cut from each loop. The 6864-cell environment is about 100 to 2000 W_A long, depending on r and p.
- **Rotation.** The family uses the viewer's 3-round Hadamard rotation. The suite's own basis checks used Haar rotations; per-unit weights from the Hadamard rotation have `E sum w^2 (M+1)/2` between 0.96 and 1.03 (1 for Dirichlet weights), so the two should agree.
- **Alias.** The raw kernel is exact at every displacement on the torus (one inverse FFT). The rate kernels' alias is predicted from it through `K(rho; t)` and checked at the worst displacement over 256 anchors, not evaluated at every displacement.
- **Approximations in the theory.** Rice's Poisson clumping for the maximum; the EC as a field count; a Rayleigh (1D) or exponential (2D) field-size law for the 3-cell and 9-cell minimums; additive variance terms for count dispersion; the EC heuristic for the alias. Each is tested against direct GRF samples in `hopfield_nav/tests/test_ideal_family.py`, and where it fails the test states the failure.

## Reproduce

```bash
# synthetic tests (about a minute): every formula against direct Monte Carlo, and the family against the viewer
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
