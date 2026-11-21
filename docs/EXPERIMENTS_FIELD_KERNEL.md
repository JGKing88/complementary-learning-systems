# Field width against kernel width: can a code have small place fields and a wide kernel?

Yes. A unit's field width and the population kernel's width are two different moments of one spectrum, so they separate once the spectrum has two scales. Give each unit a slow input (the current ideal code, r = 23.2) plus a fast input (waves of scale 3 cells) before the threshold. Fields then shrink from 30 cells to 6 to 8 cells (2.5 to 0.5 to 0.7 grid periods) and the memory still retrieves exactly, with a basin of 31 to 36 cells. The price is in the direction readout. Its noise grows as fields shrink, so the code needs about 1/W^2 more units, and production's forward one-cell difference adds a bias that no number of units removes. Both scales have to come from cross-module products (the grid wave network). A direct or randomly projected grid input cannot supply either one, because it has no wavelength longer than 13 cells and repeats at the module periods.

Run 2026-11-21 on branch `worktree-field-kernel-tradeoff`. Target: mouse dorsal CA1, field width 0.4 to 0.5 grid periods (see `docs/EXPERIMENTS_IDEAL_FAMILY.md`, "Scale"). All codes are non-negative with 10% of units active, `relu(Q z - t)` at unit length, no inhibition.

## The answer in five statements

1. **The kernel is fixed by the input's correlation and the threshold, exactly.** Let rho(d) be the correlation of the units' input (before the threshold) at separation d. The rate kernel is `K(rho(d); t)`, the map solved in the ideal-family doc. Measured against predicted full width at half height: 34.2 / 34.2, 38.4 / 38.5, 29.6 / 29.6, 32.0 / 32.0 and 6.6 / 6.6 cells over five codes with different spectra.
2. **Field width is set by the spectrum's second moment, not its bulk.** With a slow share `1 - a` at scale R and a fast share `a` at scale r_s, the input's length is `l = [(1 - a) / R^2 + a / r_s^2]^(-1/2)` and the mean field width W_A is 1.3 to 1.7 l at 10% active. The fast part dominates l once `a > (r_s / R)^2`, about 2%, while the kernel's shoulder only falls in proportion to `1 - a`. A Gaussian spectrum has one scale, so there W_A and the kernel are locked: sparsity alone moves W_A / FW05 from 1.07 (50% active) to 0.70 (2% active) and no further.
3. **So small fields and a wide kernel coexist.** Table 1. At a 50% fast share the mean field is 6.2 cells and the similarity is still 0.20 at 16 cells and 0.07 at 32. A single-scale code with the same fields (r = 4.2) has 0.00 at both.
4. **The memory does not care; the direction readout does.** Retrieval and basin stay at the baseline's level (exact 0.99 to 1.000, basin 24 to 36 for fast shares up to 30%; table 2). Direction fails for two reasons, both tied to how fast the code changes per cell, `v = E |z(x+1) - z(x)|^2`, which is what makes fields small:
   - **Bias.** Production projects `recalled - current` on a forward one-cell difference. The current code's projection on that basis is exactly `-|basis| / 2` per axis, a constant push that grows with v. With a 5% fast share the error is 40.9, 41.6 and 42.2 deg at 1024, 4096 and 8192 units. A centred difference removes it (21 deg and 9.6 deg).
   - **Noise.** With a centred readout the error falls as about 1/sqrt(units) and rises as fields shrink. Over 92 two-scale codes from 1024 to 16,384 units in the fast test, `error ~ [1 / (W sqrt(D))]^b` with b = 1.25 (+-2 cell readout, R^2 0.89) and 1.12 (+-4 cells, R^2 0.93), so holding the error needs units in proportion to about `1 / W^2`. A readout over +-4 cells instead of +-1 lowers the noise further (table 2).
5. **The fast input cannot come straight from the grid code.** A linear readout of modules 11, 12 and 13 is a sum of three periodic functions, so its spectrum has no wavelength above 13 cells. Smoothing makes it worse: at fwhm 0.25 of a period, 77% of a module's power is on its first harmonic, and a random projection correlates 0.84 with itself 12 cells away and 0.92 at 143 cells (0.27 and 0.63 for the one-hot code). That is why a random projection of the smoothed grid code gives wider but periodic fields. Used as the fast input, it shrinks fields as well as fast waves do, but its ripples are a systematic slope error, not noise: at a 10% share with the centred readout, 4 times the units lower the error by 0.70 (43.6 to 30.5 deg), against 0.50 for fast waves (31.2 to 15.6 deg), and reach stays at 0.55. Both scales must be made of cross-module products, which the grid wave network already produces for any spectrum.

Where this leaves the mouse target: 0.67 grid periods works at 4096 units (error 14.7 deg, reach 0.976, no dead goals). 0.52 grid periods reaches 0.875 at 8192 units (23 deg) and by the unit law needs about 16k to 32k units to match the baseline's direction error.

## Table 1: fields and kernel (D = 1024, 10% active)

W_A is the mean width of definition-A fields on scaffold loops (run at or above 20% of the unit's peak, at least 3 cells). K(d) is the Pearson similarity of population vectors d cells apart. Arena columns are a 20-cell square (about 1 m at mouse scale).

| code | W_A (cells) | W_A / grid period | median width | K(8) | K(16) | K(32) | far max | active in arena | fields per active unit |
|---|---|---|---|---|---|---|---|---|---|
| one scale, r = 23.2 (baseline) | 30.1 | 2.51 | 27 | 0.87 | 0.61 | 0.19 | 0.013 | 0.22 | 1.00 |
| one scale, r = 16 | 20.7 | 1.73 | 19 | 0.76 | 0.38 | 0.05 | 0.014 | 0.26 | 1.01 |
| one scale, r = 4.2 | 6.0 | 0.50 | 5 | 0.06 | 0.00 | 0.00 | 0.017 | 0.68 | 1.24 |
| + 5% fast waves, scale 2 | 14.0 | 1.17 | 7 | 0.78 | 0.53 | 0.17 | 0.015 | 0.25 | 1.36 |
| + 10% fast waves, scale 2 | 10.3 | 0.86 | 5 | 0.70 | 0.47 | 0.16 | 0.018 | 0.28 | 1.53 |
| + 30% fast waves, scale 3 | 8.0 | 0.67 | 5 | 0.46 | 0.32 | 0.11 | 0.015 | 0.39 | 1.47 |
| + 50% fast waves, scale 3 | 6.2 | 0.52 | 5 | 0.27 | 0.20 | 0.07 | 0.024 | 0.49 | 1.55 |
| + 50% fast waves, scale 2 | 5.0 | 0.42 | 4 | 0.25 | 0.17 | 0.06 | 0.019 | 0.43 | 1.75 |
| + 10% direct grid input | 9.3 | 0.77 | 5 | 0.69 | 0.50 | 0.17 | 0.037 | 0.27 | 1.71 |
| + 30% direct grid input | 5.7 | 0.48 | 4 | 0.42 | 0.33 | 0.14 | 0.112 | 0.33 | 2.07 |
| grid input only | 3.4 | 0.28 | 3 | -0.03 | -0.01 | 0.03 | 0.865 | 0.52 | 1.73 |

- Fields of a two-scale unit come in mixed sizes: a few wide ones where the slow input is high and many small ones around them (median 5 cells, 90th percentile 11 to 16). The kernel is weighted by activity, so the wide fields carry its shoulder.
- More fields per unit is forced: fields per unit times field size equals the share of the arena a unit covers, and the threshold fixes that share. In a 20-cell arena it is 1.5 fields per active unit against 1.0, with half the units active.
- The full width at half height is not a useful summary for these kernels. It falls from 38 to 7.5 cells when the narrow peak crosses 0.5, while the shoulder that the memory uses is still there.

## Table 2: navigation probe

`analysis.hopfield_probe.run`, `run_ideal.sh` settings (whole arena, 8 worlds, 20-cell arenas, Hebbian storage, K = 5, one recall step). Readout: "forward" is production's `z(x+1) - z(x)`; "+-1" and "+-4" are centred differences over that many cells (`--basis_centered`, `--basis_step 4`). Dead = share of goals never reached at K = 5.

| code | W_A / grid period | units | readout | error (deg) | exact | basin | reach | dead |
|---|---|---|---|---|---|---|---|---|
| baseline | 2.51 | 1024 | forward | 4.8 | 1.000 | 28.4 | 0.996 | 0.00 |
| baseline | 2.51 | 1024 | +-1 | 4.4 | 1.000 | 28.4 | 0.995 | 0.00 |
| baseline | 2.51 | 4096 | forward | 2.9 | 1.000 | 42.8 | 0.990 | 0.00 |
| one scale, r = 4.2 | 0.50 | 1024 | forward | 80.5 | 0.536 | 6.5 | 0.372 | 0.83 |
| one scale, r = 4.2 | 0.50 | 1024 | +-4 | 56.3 | 0.536 | 6.5 | 0.532 | 0.42 |
| one scale, r = 4.2 | 0.50 | 4096 | +-1 | 51.3 | 0.551 | 7.4 | 0.569 | 0.38 |
| 5% fast, scale 2 | 1.17 | 1024 | forward | 40.9 | 1.000 | 32.9 | 0.533 | 0.54 |
| 5% fast, scale 2 | 1.17 | 4096 | forward | 41.6 | 1.000 | 42.0 | 0.438 | 0.71 |
| 5% fast, scale 2 | 1.17 | 8192 | forward | 42.2 | 1.000 | 46.3 | 0.420 | 0.75 |
| 5% fast, scale 2 | 1.17 | 1024 | +-1 | 21.1 | 1.000 | 32.9 | 0.923 | 0.04 |
| 5% fast, scale 2 | 1.17 | 4096 | +-1 | 9.6 | 1.000 | 42.0 | 0.993 | 0.00 |
| 5% fast, scale 2 | 1.17 | 1024 | +-4 | 9.5 | 1.000 | 32.9 | 0.999 | 0.00 |
| 10% fast, scale 2 | 0.86 | 1024 | forward | 63.7 | 1.000 | 30.2 | 0.322 | 0.88 |
| 10% fast, scale 2 | 0.86 | 1024 | +-1 | 31.2 | 1.000 | 30.2 | 0.634 | 0.08 |
| 10% fast, scale 2 | 0.86 | 4096 | +-1 | 15.6 | 1.000 | 40.6 | 0.982 | 0.00 |
| 10% fast, scale 2 | 0.86 | 1024 | +-4 | 14.2 | 1.000 | 30.2 | 0.985 | 0.00 |
| 30% fast, scale 3 | 0.67 | 1024 | +-1 | 39.3 | 1.000 | 24.8 | 0.595 | 0.33 |
| 30% fast, scale 3 | 0.67 | 4096 | +-1 | 24.6 | 1.000 | 35.9 | 0.831 | 0.12 |
| 30% fast, scale 3 | 0.67 | 4096 | +-4 | 14.7 | 1.000 | 35.9 | 0.976 | 0.00 |
| 50% fast, scale 3 | 0.52 | 1024 | +-4 | 45.1 | 0.993 | 15.6 | 0.616 | 0.21 |
| 50% fast, scale 3 | 0.52 | 4096 | +-1 | 44.0 | 1.000 | 30.8 | 0.555 | 0.38 |
| 50% fast, scale 3 | 0.52 | 4096 | +-4 | 31.9 | 1.000 | 30.8 | 0.807 | 0.04 |
| 50% fast, scale 3 | 0.52 | 8192 | forward | 81.8 | 1.000 | 35.0 | 0.327 | 0.83 |
| 50% fast, scale 3 | 0.52 | 8192 | +-4 | 23.4 | 1.000 | 35.0 | 0.875 | 0.04 |
| 50% fast, scale 2 | 0.42 | 4096 | +-4 | 39.2 | 1.000 | 30.2 | 0.690 | 0.08 |
| 10% direct grid input | 0.77 | 1024 | forward | 75.9 | 1.000 | 30.3 | 0.230 | 0.92 |
| 10% direct grid input | 0.77 | 1024 | +-1 | 43.6 | 1.000 | 30.3 | 0.359 | 0.83 |
| 10% direct grid input | 0.77 | 4096 | +-1 | 30.5 | 1.000 | 39.3 | 0.554 | 0.42 |
| 10% direct grid input | 0.77 | 1024 | +-4 | 20.8 | 1.000 | 30.3 | 0.869 | 0.04 |
| 30% direct grid input | 0.48 | 1024 | +-4 | 54.9 | 0.998 | 22.6 | 0.348 | 0.71 |
| grid input only | 0.28 | 1024 | forward | 94.9 | 0.214 | 1.7 | 0.060 | 1.00 |

- At equal field width (0.5 grid periods, 1024 units), the two-scale code retrieves (exact 0.993, basin 15.6) where the single-scale code does not (0.536, 6.5). Units do not rescue the single-scale code (0.551 and 7.4 at 4096).
- A +-8 cell readout is too coarse for a 20-cell arena (reach 0.91, 0.75 and 0.43 at 5%, 10% and 30% fast shares).
- Raising the slow scale to 32 at a 50% fast share widens the basin (41.1 against 30.8) but not direction (40 deg, reach 0.70).
- Splitting the fast input across units instead of within them (30% or 50% of units driven by the grid alone) is no better: reach 0.11 and 0.09 with the forward readout.

## Figure

`tradeoff.pdf`: (a) Pearson kernels of the baseline, two two-scale codes and the single-scale code with mouse-sized fields; (b) field width against the fast share for each kind of fast input, with the mouse band; (c) direction error of the fast test (+-2 cell readout, noise only) against units; (d) reach on the real probe against field width for each readout and unit count.

## What each quantity on the list depends on

| quantity | set by |
|---|---|
| r | the slow scale of the input's spectrum; the kernel's tail falls on this scale whatever the threshold |
| kernel half-width | `K(rho(d); t)`: the input's whole correlation, sharpened by the threshold. Not a good summary once the spectrum has two scales |
| field width | 1.3 to 1.7 times `l = 1 / sqrt(second spectral moment)`, at 10% active; smaller at lower coverage |
| fields per unit | coverage times arena size over field size; the threshold fixes coverage |
| sparsity | the threshold alone: coverage p, Treves-Rolls a(t) |
| frequencies in the ideal code | their distribution is the spectrum, so it sets r, l and the kernel together; their number M sets the far noise 1/sqrt(2M) |
| units D | the readout's noise, as 1/sqrt(D); nothing about widths |
| code speed v | `E |z(x+1) - z(x)|^2`, the rate code's own second moment; small fields mean large v, and v is what the direction readout pays for |

## What did not hold

- **"A small fast share costs almost nothing."** My prediction before the probe. It is true for the kernel and the memory (5% fast share: half-width 34.0 against 38.4, exact 1.000, basin 32.9) and false for direction with the production readout (41 deg against 4.8).
- **"More units fix it."** Only after the readout is centred. With the forward difference the error is bias and does not move with units.
- **"A direct grid input can be the fast part, because its ripples enter the kernel at only a/3."** The ripples are small in the kernel (far max 0.037 at 10%) but the readout takes a slope, and a 12-cell ripple has a slope comparable to the shoulder's.

## Not done

- No two-scale code was run through the grid wave network itself. The claim that it can produce the fast waves rests on it already producing any integer wave; a second population of L3 units with a faster depression setting is the obvious construction and is untested.
- One frequency seed and one rotation seed per code. The fast test (`direction.py`) is a stand-in: it matches the probe with the current-code term added (39.5 and 39.1 deg against the probe's 40.9 and 41.6), but its absolute errors are lower than the probe's at large fast shares.
- 16k and 32k units on the real probe, to confirm the extrapolation for 0.5 grid periods. Two 8192-unit probe tasks hit the 3 h limit and have no result (50% fast at scale 2 with the +-4 readout; 50% fast at scale 3 with a +-2 readout). The sweep's 16,384-unit tasks were stopped part way, so that size has fewer codes than the others.
- Whether the RL controller, which acts stochastically and has memory, tolerates a noisier direction signal than the greedy probe walk.
- Dense (50% active) two-scale codes on the centred readout.

## Code and reproduction

- `analysis/hopfield_probe/ideal_encoder.py`: spec keys `fast=a,r_fast=rs` (a share of the waves drawn at a second scale), `grid_fast=a` (direct grid input before the threshold), `grid_units=f`.
- `analysis/hopfield_probe/encode.py`, `run.py`: `--basis_step s` and `--basis_centered` for the direction readout. Defaults reproduce production.
- `analysis/field_kernel/measure.py` (kernel, predicted kernel, fields), `direction.py` (fast direction test), `sweep.py` and `law.py` (unit law), `summary.py` (tables), `figures.py`, `sets/*.txt` (probe sets a to d), `run_probe.sh`, `run_sweep.sh`.
- Results: `/orcd/pool/003/jackking/cls_runs/results/field_kernel/` (`probe_a` to `probe_d`, `measure`, `sweep`). Figure: `/orcd/pool/003/jackking/cls_runs/figures/field_kernel/tradeoff.pdf`.
- Slurm arrays 29980218 (set a), 29982377 (b), 29986158 (c), 29991907 (d), 29983731 (sweep), from code snapshots `field_kernel_a` to `field_kernel_e`.

```bash
~/.conda/envs/cls/bin/python -m pytest hopfield_nav/tests/test_field_kernel.py -q
scripts/snapshot_code.sh <tag>
SET=a CODE=<tag> bash analysis/field_kernel/run_probe.sh
CODE=<tag> bash analysis/field_kernel/run_sweep.sh
~/.conda/envs/cls/bin/python -m analysis.field_kernel.summary a b c d
~/.conda/envs/cls/bin/python -m analysis.field_kernel.law
~/.conda/envs/cls/bin/python -m analysis.field_kernel.figures
```
