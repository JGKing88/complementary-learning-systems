# Coding comparison: does any model code look like a brain region?

No model population reads as one brain region. Each matches a few measurements of some region and fails most of the rest, and the split follows one line. The population similarity kernel belongs to the code, while every single-unit statistic depends on how the code is cut into units. The suite runs the place-cell literature's own measurements on model units, scales every length by the population's own mean field width W, and scores each result against ranges transcribed from `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`. W_A is W under a 20%-of-peak field definition, the one Kjelstrup 2008 used.

## Result

1. **Most codes have a dorsal-width kernel; only the sparse ReLU code is as broad as ventral CA3.** Kjelstrup 2008's PV band at r = 0.2 is 1.44 mean fields wide in dorsal CA3 and 2.32 and 2.46 in intermediate and ventral CA3. Along scaffold loops, with Pearson correlation of rates, the band is 1.02 W_A wide for the ideal code, 1.14 for the trained code, 1.12 for the grid code and 1.55 for the rotated ideal, all inside the dorsal band (1.01 to 1.87). The sparse ReLU code's band is 2.11 W_A, inside the intermediate and ventral bands. The zero crossing is a weaker test. A Gaussian-like kernel reaches zero only at the sampling noise floor, near plus or minus 1/sqrt(units), so the ideal code's zero crossing is 1.15 W_A along the loops and 1.66 W_A in 2D.
2. **The kernel is basis free but its ratio to field width is not.** The rotated ideal has exactly the ideal's population similarity. Its single units, though, are random mixtures with fields of 35 cells instead of 55, so the same kernel measures 1.55 instead of 1.02 field widths at r = 0.2. A kernel-to-field ratio places a code in a region only once the readout into units is fixed. The rotation also changes 2D field counts. Under definition A the rotated ideal has 1.41 fields per active unit in an arena of 3.7 mean fields and 4.41 in one of 13.7, close to Fenton 2008's 1.3 and 3.4 for dorsal CA1 (the second at the edge of its band), while the native ideal and the trained code reach 6.05 and 6.47 in the larger arena.
3. **No code reproduces Rich 2014's field counts.** On a track of 74 mean field widths Rich found overdispersed counts (negative binomial r = 0.57, variance over mean 7.1), randomly placed fields (gap CV near 1) and 35% of cells with no field. Codes whose units are statistically alike give Poisson-like or underdispersed counts with almost every unit active. Variance over mean is 0.66 for the rotated ideal and 0.79 for the sparse ReLU code, and the negative binomial fit runs to its Poisson limit. The native ideal is overdispersed (variance over mean 6.2, inside Rich's band) only because its plane-wave units differ in spatial frequency, and their fields are periodic (gap CV 0.002). The grid code is overdispersed because a grid unit fires on a straight track only if the track crosses its phase, and then periodically (gap CV below 0.001). Matching r = 0.57 needs units that differ in field propensity, which none of these codes has.
4. **Under Rich's 2 Hz rule, dense codes barely fire.** With the population mean rate at 0.5 to 1 Hz, a field needs a peak 2 to 4 times the mean. A rectified cosine peaks at pi times its mean, so the ideal code has no field anywhere at 0.5 Hz. The trained code is close to binary (tanh at gain 100), its rectified peak is about twice its mean, and it goes from 81% of units with no field on the 74-width track at 0.75 Hz to every unit active with 20 fields at 1 Hz. These verdicts carry the `~` flag.
5. **The sparse ReLU code is the closest to dorsal CA1 in a small arena.** In an arena of 8 A0 it has 49% of units active (CA1 20 to 50%), 8.6% of units in a field at a place (CA1 3 to 14%) and 1.10 fields per active unit, and on a short track its Treves-Rolls sparsity is 0.18, at the lower edge of CA1's 0.26 plus or minus 30%. Its counts on long tracks are Poisson-like, not overdispersed.
6. **Every code but the untrained one remaps orthogonally at the population level.** Between arenas at different scaffold locations, active-set overlap is 0.99 to 1.08 times chance and the median PV correlation of rates at matched places is -0.016 to 0.045. That is the CA3 pattern between rooms (overlap at chance, PV correlation 0.03). No code reproduces CA1's partial overlap (1.14 to 1.58 times chance, PV correlation 0.25), the subiculum's preserved mean rates (0.87), or the conserved log rates of CA1 and CA3 (0.65 and 0.59). Two codes keep some single-unit structure across arenas. The hippocampus layer's mean rates correlate at 0.21 (CA1 0.19), and 36% of the ideal code's units have a rate-map correlation above the shuffle null's 95th percentile, because a plane-wave unit in a new arena is the same wave shifted; after a random rotation the share drops to the 5% expected by chance. The untrained encoder does not remap at all (PV correlation 0.999).
7. **Two populations carry almost no place code.** The untrained encoder's output is nearly constant, and its cosine similarity stays above 0.998 at every separation. The Vector-HaSH place layer on a one-hot grid code is spatially white. Neighbouring cells' rates are independent, Rich-rule fields are single cells (W = 1.1 cells), and the kernel falls to zero within one cell, with revivals of about 0.2 at the grid module periods.

### Headline per population

| Population | Nearest region, and on what | What rules it out |
|---|---|---|
| Grid code | none; it is an entorhinal-style periodic code | fields repeat at fixed spacing (gap CV below 0.001); the kernel revives to 0.9 at one module period; in a small arena every unit is active |
| Ideal | dorsal CA3 or CA1 on kernel width (1.02 W_A at r = 0.2) | units are plane waves with periodic fields; under Rich's rule it has no field at 0.5 Hz; counts overdispersed only through spread in wave frequency |
| Ideal, rotated | dorsal CA3 or CA1 on kernel width (1.55 W_A at r = 0.2); dorsal CA1 on 2D field counts (Fenton's two rooms) | counts underdispersed (variance over mean 0.66); nearly every unit has a field from 15 field widths on, where Rich finds 35% silent at 74 |
| Sparse ReLU (10%) | intermediate or ventral CA3 on kernel width (2.11 W_A at r = 0.2); dorsal CA1 on small-arena activity (49% active, 8.6% per place, 1.1 fields per active unit) | counts Poisson-like (variance over mean 0.79), so no Rich-style silent fraction on long tracks; kernel reaches zero only at 3.6 W_A |
| Trained | dorsal CA3 or CA1 on kernel width (1.14 W_A at r = 0.2) | far aliases (cosine 0.69 at 15.2 W_A, 0.3 at 2.8 W_A) and a positive floor; near-binary units, so Rich-rule verdicts flip between 0.75 and 1 Hz |
| Untrained | none | output nearly constant (similarity above 0.998 everywhere) |
| Hippocampus (Vector-HaSH) | none | spatially white at one cell: single-cell fields, kernel at zero within one cell |

Every population except the untrained one remaps orthogonally between scaffold locations, the CA3 pattern, so none matches CA1 or the subiculum on remapping.

## What was measured and how

### Populations

Every population the tuning viewer defines (`analysis/tuning_viewer/export.py`) runs through one adapter, `analysis/coding_comparison/populations.py`. A new encoder is one entry in `POPULATIONS`, or `--pop name=spec` on the command line, where `spec` is a checkpoint path (gain and fwhm are read from the checkpoint) or an `ideal:` spec.

| Name | What it is | Units | Signed |
|---|---|---|---|
| grid | smoothed grid code, modules 11, 12, 13, fwhm 0.25 of the period | 434 | no |
| ideal | `ideal:r=16,n_freq=512,seed=0`: 512 plane waves, a cos and a sin unit each | 1024 | yes |
| ideal_rot | the ideal remixed by the viewer's fixed Walsh-Hadamard rotation | 1024 | yes |
| ideal_sparse | the viewer's sparse ReLU code: r = 23.2, same rotation, threshold for 10% active, unit length | 1024 | no |
| trained | cosyne att0.5 checkpoint (seed 43), tanh(100 z), unit norm | 1024 | yes |
| untrained | same architecture after `torch.manual_seed(0)` | 1024 | yes |
| hippocampus | Vector-HaSH place layer, relu(Wpg g - 2) on the one-hot grid code, Np 400, c 0.5, seed 0 | 400 | no |

The viewer's slow-wave cutoff controls for the ideal codes are not populations of their own and were not run.

### Scale

The physical size of a scaffold cell is unknown, so every length is in units of the population's own mean single-field width W, measured under the same field definition as the quantity it scales. W is measured on 8 full scaffold loops (4 rows, 4 columns, 1716 cells each). Areas are in units of A0 = (pi/4) W_A^2, where W_A is W under definition A in the primary rate convention.

### Rate conventions and bases

Population similarity has two constructions. Cosine similarity of the raw code is unchanged by any orthogonal remixing of the units; `test_cosine_similarity_is_rotation_invariant` checks this. Pearson correlation of population vectors (PVs) is what the papers compute; it centres each PV across units, so it depends on the basis.

Single-unit measures need non-negative rates. Signed codes are run two ways: rectified (keep max(z, 0)) and on/off (each unit becomes an on cell max(z, 0) and an off cell max(-z, 0)). Non-negative codes are used as they are. For signed codes every single-unit measure is also repeated after three random orthogonal rotations of the units (Haar draws, seeds 1 to 3), to show how much each depends on the basis.

### Field definitions

| Name | Rule | Matched to |
|---|---|---|
| A | contiguous run at or above 20% of the unit's peak in that environment, at least 3 cells (9 in 2D), no smoothing, no rate floor | Kjelstrup 2008 |
| Ap | A, plus each field's own peak must reach 1 Hz at the 0.75 Hz rate scale | Eliav 2021 |
| B0.5, B0.75, B1 | one global rate scale that makes the population mean rate 0.5, 0.75 or 1 Hz over 16,384 random scaffold cells; Gaussian smoothing with SD (10/65) W; a 2 Hz threshold; minimum length (15/65) W, minimum area its square | Rich 2014 |

Under B the smoothing and minimum length depend on W, and W depends on them, so W is found by fixed-point iteration from W_A until it changes by less than 1%. Rich's 10 cm smoothing and 15 cm minimum are relative to their 65 cm mean field. B0.75 is the primary B; the scorecard marks a verdict that changes anywhere in the 0.5 to 1 Hz sweep.

Definition A has no rate floor. Every unit with any activity therefore has fields under A, and A is used only for shape and width measures (Kjelstrup's kernel ratios, Fenton's field counts), never for activity.

### Environments

Tracks are zigzags of scaffold cells, with straight runs of 1716 cells (one scaffold period) joined by sideways steps of 4 W_A, capped at 286 cells. Rich's 48 m track was itself a zigzag of about 1 m segments. Each population gets 16 tracks, half running along x and half along y, at random places. Track lengths are Rich's stages in mean field widths: 3, 10, 22 and 48 m over 0.65 m, which is 4.6, 15.4, 33.8 and 73.8 W, plus 147.7 W. A track that would need more cells than the zigzag holds without revisiting a row is skipped.

Arenas are squares with areas of 1, 2, 4, 8, 16, 32, 64 and 128 A0, two locations each, sides capped at 1024 cells. Two arenas of one size are kept at least 2 W_A apart where the torus allows, else they touch, else they are spread along the diagonal; `fields_2d.json` records which. The small arena of the activity rows is 8 A0. Remapping uses 8 arenas of 8 A0 at random scaffold locations at least 2 W_A apart (sides capped at 256 cells), compared at matched relative positions, 28 pairs.

### Measures

All measures are pure functions on rate maps in `analysis/coding_comparison/measures.py`, so they run unchanged on recordings.

1. Field counts on tracks: mean, variance over mean, the share of zero-field units, maximum-likelihood Poisson and negative binomial fits (r and p as Rich reports them, P(0) = p^r), the likelihood ratio against the boundary null, and AIC. For units with six or more fields: the CV of gaps between field centres (1 for exponential gaps, near 0 for a periodic layout) and a KS test of field positions against uniform. Field size CV and the largest-to-smallest field ratio within a unit.
2. Zero-field fraction against track length, against Rich's curve P(0 | L) = (7.75 / (7.75 + 0.65 L))^0.57 with L in field widths.
3. Fields per active unit against arena area (Tanni, Harland and Fenton count place cells): Pearson r, and the fields per active unit where the arena is 3.7 and 13.7 mean fields (Fenton 2008's cylinder and chamber, from his per-field shares of 0.27 and 0.073), interpolated in log area. Fields per unit over all units, slope and log-log exponent are in the JSON.
4. PV similarity against separation along the loops (exact circular lags) and against distance in 2D (96 anchors, 16 directions, up to 20 W_A). Summaries: full widths at r = 0.5 and r = 0.2 (twice the distance where the curve first falls to that level, as Kjelstrup reports the band across the diagonal), the first zero crossing, and the floor between 5 and 15 W_A, each over W_A.
5. Active fraction per environment (units with one or more fields) and per location (share of units in a field at a cell, which equals the mean per-unit coverage).
6. Remapping across arenas: median PV correlation at matched positions with silent units included; active-set overlap over its value under independence; the correlation across units of mean rates (and of log mean rates); the share of units whose rate-map correlation beats a unit-shuffle null's 95th percentile.
7. Treves-Rolls sparsity, mean(r)^2 / mean(r^2), per active unit, on the 4.6 W track.

### Scoring

`literature_targets.json` holds every target with its source, region, species, environment and the review's verification tag. Values marked "derived (this suite)" are arithmetic on numbers the review reports, with the formula stated. A region with no number for a measure has no entry and scores U. The bands follow one stated policy: a reported range exactly, a point within plus or minus 30% (plus or minus 0.1 for correlations), a bound as stated.

A scorecard cell is C (consistent), I (inconsistent), U (no literature number) or U* (the model has no value). Each cell uses the region's primary target. A trailing `~` marks a verdict that changes within the 0.5 to 1 Hz sweep; a trailing `r` one that changes under a random rotation.

## Figures

All in `/orcd/pool/003/jackking/cls_runs/figures/coding_comparison/v1/`, one panel per population, primary convention (rectified for signed codes).

- `kernel_1d.png`: PV similarity against separation in field widths, cosine of the raw code and Pearson of rates, with Kjelstrup's figure-read ventral CA3 points. The sparse ReLU curve follows those points out to 1.5 W; the others fall faster.
- `kernel_far.png`: the same out to 16 W. The grid code revives at its module periods, the trained code has far aliases (cosine 0.69 at 15.2 W), the hippocampus layer revives at about 0.2.
- `zero_curve.png`: share of units with no field against track length under B at each mean rate, against Rich's fitted curve and its plus or minus 0.155 band.
- `counts_74W.png`: fitted Poisson and negative binomial distributions on the 74-width track against Rich's.
- `area_scaling.png`: fields per unit against arena area under A, Ap and B0.75.
- `basis_dependence.png`: single-unit measures of the signed codes in the native basis and after three random rotations.
- `phase2.png`: decoding error against the number of units, and correlation dimension against arena size with Nakai 2024's CA1 and subiculum bands.

## Phase 2

Correlation dimension depends on the arena more than on the code, so the scorecard row is weak. For the smooth codes it rises with arena size, because a 2D sheet looks 2D at scales below the kernel and higher-dimensional above it: 2.2, 2.5 and 3.0 for the ideal code at 4, 16 and 64 A0, and 2.6, 3.1 and 4.4 for the trained code. At 16 A0 the grid code (4.4), the trained code (3.1) and the ideal code (2.52, at the band's edge) fall in Nakai 2024's open-field CA1 band of 2.51 to 4.71; the rotated ideal and the sparse ReLU code (2.2) fall just below it. No code reaches the subiculum band (5.05 to 6.63). The untrained encoder (11.3) and the hippocampus layer (14.1) are far above both, as noise-like codes should be. The fit uses pair fractions between 0.001 and 0.1; Nakai's scaling range is not in the review.

Decoding is reported, not scored, because the review has no decoding number in field widths. A maximum-likelihood decoder on Poisson counts (0.5 s window, rates at the 0.75 Hz scale) reaches a median error of 0.1 W with about 100 units for the ideal (0.096 W), rotated ideal (0.071 W) and sparse ReLU (0.037 W) codes and with about 200 for the trained code (0.061 W), and an exact cell by 500 to 1000 units. The grid code and the hippocampus layer decode exactly with 20 to 50 units, but their decoding arenas are only 18 and 11 cells across. The untrained encoder stays near 0.28 W at every population size, which is chance in its 160-cell arena (its W is 285 cells).

## Caveats

1. The literature's scale enters only through ratios. Every comparison assumes that the brain's mean field width under the paper's definition plays the role of W under the matched model definition. Where the review gives no field size for a study (Tanni's arenas, GoodSmith's platforms, Leutgeb's rooms, Maimon's CA3), the environment's size in field widths is unknown, and the "small arena" of 8 A0 is a stand-in.
2. The models are deterministic and noise-free. There is no pass-to-pass variability, so Rich's "spikes on at least half of passes" and Kjelstrup's "spatially stable" criteria have nothing to remove, and every threshold crossing counts. Real counts carry measurement noise on top of the code.
3. Kjelstrup's PVs pool cells recorded in different rats and sessions. The model PVs are whole simultaneous populations. The far floor of a PV curve depends on how PVs are built; the floor row uses Pearson correlation of rates with silent units included, and Kjelstrup does not state a normalization.
4. A cosine kernel of a non-negative code cannot fall below zero. The cosine zero-crossing row is therefore inconsistent by construction for grid, sparse ReLU and hippocampus, and only the Pearson rows test them.
5. Several matches are approximate. GoodSmith's and Leutgeb's field and activity definitions are not in the review, so their rows use B. The active-set overlap formula of Leutgeb 2004 is not in the review, so the row compares overlap over its own chance value. Kim 2012's "spatial activity fraction" is scored as Treves-Rolls sparsity, an assumption. Fenton's rule is any rate above zero, and the row uses A. Bat lengths in field widths use Eliav's CA1 mean field (5.7 m) for CA3 too. Ap approximates Eliav's rule. It puts the edge at 20% of the unit's peak in the environment, where Eliav uses each field's own peak and merges sub-peaks whose dip stays above 50%. A per-field edge would admit low secondary peaks as fields, so Ap probably undercounts small fields relative to Eliav.
6. Single-unit rows for signed codes depend on the basis and on the rate convention; the `r` flags and the basis figure show how much. Only the cosine kernel rows are basis free.
7. Remapping in the model is a move to another scaffold location. That is how Vector-HaSH assigns a new environment, and it is the assumption these rows test.
8. The tolerance policy (30% or 0.1) is this suite's choice. A different width changes some verdicts near a band edge; the model value is printed beside every verdict.
9. The scaffold is 1716 cells and wraps. Tracks longer than one period are zigzags, arenas are capped at 1024 cells a side, and lengths beyond what fits are skipped and listed in the JSON (`skipped_W`, `skipped_sizes`).
10. The ideal and trained encoders ran their 2D series and remapping with one arena location per size and one random rotation (`--param n_arena_locs=1 --param rot_seeds=[1]`), because the full design spends most of its time on rotated copies of the largest arenas; their 1D rows use three rotations. The untrained encoder ran its 2D series and remapping on a smaller design still (`--param n_arena_locs=1 --param arena_side_cap=540 --param remap_n=4 --param rot_seeds=[1]`). That is one location per size, sides up to 540 cells (so only 1, 2 and 4 A0), four remapping arenas and one rotation. Its field width is about 300 cells under A and a whole scaffold row under Ap and B1, and the full design would have overrun the job limit. Its 1D rows used the full design.
11. `hopfield_nav/tests/test_layering.py` fails on this branch before and after this work, on files outside this package (`encoder_training` importing `analysis`, hardcoded output paths in the tuning viewer and `cosyne2027`). Nothing in `analysis/coding_comparison` trips it.

## Scorecard

Primary-target counts, consistent over scored, per region. The counts are not a ranking. Regions differ in how many measures have a literature number, and several of one region's numbers come from a single study.

| Population | Convention | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid | identity | 8/24 | 7/14 | 2/7 | 1/7 | 0/4 | 2/3 | 0/1 | 1/7 | 0/7 | 1/3 |
| ideal | rectify | 8/24 | 9/14 | 1/7 | 0/7 | 0/4 | 2/3 | 0/1 | 2/7 | 2/7 | 0/3 |
| ideal | onoff | 8/24 | 9/14 | 1/7 | 0/7 | 0/4 | 2/3 | 0/1 | 2/7 | 2/7 | 0/3 |
| ideal_rot | rectify | 6/24 | 6/14 | 5/7 | 4/7 | 1/4 | 3/3 | 0/1 | 1/7 | 2/7 | 0/3 |
| ideal_rot | onoff | 6/24 | 6/14 | 5/7 | 4/7 | 1/4 | 3/3 | 0/1 | 1/7 | 2/7 | 0/3 |
| ideal_sparse | identity | 8/24 | 4/14 | 2/7 | 4/7 | 2/4 | 1/3 | 1/1 | 1/7 | 4/7 | 0/3 |
| trained | rectify | 4/21 | 7/13 | 1/7 | 0/7 | 2/3 | 1/2 | 0/1 | 1/7 | 2/7 | 0/3 |
| trained | onoff | 4/21 | 7/13 | 1/7 | 0/7 | 2/3 | 1/2 | 0/1 | 1/7 | 2/7 | 0/3 |
| untrained | rectify | 0/13 | 1/10 | 0/5 | 0/5 | 1/3 | 0/2 | 0/1 | 0/7 | U | U |
| untrained | onoff | 0/13 | 0/10 | 0/5 | 0/5 | 1/3 | 0/2 | 0/1 | 0/7 | U | U |
| hippocampus | identity | 7/22 | 4/14 | 0/7 | 0/7 | 1/4 | 2/3 | 0/1 | 1/7 | 0/6 | 1/3 |

Cells: C consistent, I inconsistent, U no literature number, U* no model value; ~ the verdict changes within the 0.5 to 1 Hz sweep, r it changes under a random rotation. Brackets give the band and the source tag: ft full text, abs abstract, fig figure-read, der derived, inf inference, rev the review's summary table, asm assumed formula, sec secondary.

### grid, identity

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 9.72 | I [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | 0.118 | I [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 17.1 | I [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | yes | C [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.311 | I [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.364 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.000326 | I [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0 | C [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.216 | I [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.131 | I [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0.0485 | U | U | U | U | U | U | U | U | I [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 1.13 | U | U | U | U | U | U | U | U | I [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 2.74e-12 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 14.1 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 0.415 | U | U | U | U | U | U | U | U | I [0.584 to 1.08; ft] | C [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 1 | C [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 1.67 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1 | C [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 1.58 | I [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 1.67 | C~ [0.91 to 1.69; ft] | I [0.833 to 1.55; ft] | U | U | I [0.728 to 1.35; ft] | C [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 1 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | C [0.616 to 1.14; ft] | I [0.477 to 0.887; ft] | I [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.138 | C~ [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.377 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.766 | U | C [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | I [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.12 | U | C [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 0.739 | C [<= 1.3; ft] | C [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.266 | U | I [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 0.861 | U | C [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.34 | U | C [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.0449 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1 | I [1.14 to 1.58; ft/der] | C [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | -0.041 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | -0.0247 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0 | U | I [0.0217 to 0.0403; ft] | U | U | I [0.0308 to 0.0572; ft] | I [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 4.37 | C [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### ideal, rectify

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 11.8 | I [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | 1.67 | I~ [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 6.24 | C~r [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | yes | C~r [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.518 | I~ [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.95 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.00216 | I~ [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0.00462 | C~ [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 1.79 | I~r [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.159 | I [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 1.32 | U | U | U | U | U | U | U | U | Ir [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 5.31 | U | U | U | U | U | U | U | U | Cr [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.0809 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 14.7 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0.0217 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0.852 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 0.997 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 0.976 | C~ [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 2.03 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.66 | C [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 6.05 | Ir [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 2.16 | I~ [0.91 to 1.69; ft] | I~ [0.833 to 1.55; ft] | U | U | I~ [0.728 to 1.35; ft] | C~r [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0.962 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | C~ [0.616 to 1.14; ft] | I [0.477 to 0.887; ft] | I [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.148 | Ir [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I~ [<= 0.01; inf] | U | U | C~r [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.428 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.628 | U | C [0.581 to 1.08; ft/der] | Ir [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.02 | U | C [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 1.18 | Cr [<= 1.3; ft] | Cr [<= 1.33; ft/fig/der] | Ir [1.32 to 2.44; ft/fig/der] | Ir [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.00125 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 0.7 | U | Cr [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.08 | U | C [1.01 to 1.87; ft/der] | Ir [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | 1.15 | Cr [<= 1.3; ft] | Cr [<= 1.33; ft/fig/der] | Ir [1.32 to 2.44; ft/fig/der] | Ir [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.00665 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1.02 | I~ [1.14 to 1.58; ft/der] | C~ [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0591 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.119 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.358 | U | I [0.0217 to 0.0403; ft] | U | U | Ir [0.0308 to 0.0572; ft] | Ir [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 2.52 | C [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### ideal, onoff

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 11.8 | I [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | 1.68 | I~ [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 6.21 | C~r [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | yes | C~r [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.52 | I~ [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.951 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.00211 | I~ [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0.00474 | C~ [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 1.82 | I~r [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.16 | I [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 1.33 | U | U | U | U | U | U | U | U | Ir [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 5.32 | U | U | U | U | U | U | U | U | Cr [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.0786 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 14.7 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0.0214 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0.851 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 0.996 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 0.975 | C~ [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 2.04 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.68 | C [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 6.04 | Ir [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 2.16 | I~ [0.91 to 1.69; ft] | I~ [0.833 to 1.55; ft] | U | U | I~ [0.728 to 1.35; ft] | C~r [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0.955 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | C~ [0.616 to 1.14; ft] | I [0.477 to 0.887; ft] | I [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.146 | Ir [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I~ [<= 0.01; inf] | U | U | C~r [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.429 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.629 | U | C [0.581 to 1.08; ft/der] | Ir [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.02 | U | C [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 1.16 | Cr [<= 1.3; ft] | Cr [<= 1.33; ft/fig/der] | Ir [1.32 to 2.44; ft/fig/der] | Ir [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.000294 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 0.702 | U | Cr [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.08 | U | C [1.01 to 1.87; ft/der] | Ir [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | 1.15 | Cr [<= 1.3; ft] | Cr [<= 1.33; ft/fig/der] | Ir [1.32 to 2.44; ft/fig/der] | Ir [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.00586 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1.02 | I~ [1.14 to 1.58; ft/der] | C~ [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0507 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.107 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.356 | U | I [0.0217 to 0.0403; ft] | U | U | Ir [0.0308 to 0.0572; ft] | Ir [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 2.56 | C [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### ideal_rot, rectify

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 9.92 | I~ [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | inf | I [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 0.662 | I [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | no | I [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.552 | I [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 1 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.645 | I [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0.0104 | C [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.607 | C [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.135 | I~ [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0.674 | U | U | U | U | U | U | U | U | C [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 9.03 | U | U | U | U | U | U | U | U | I [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.498 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 10.3 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0.999 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 1 | C [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 1.72 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.41 | C [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 4.41 | C [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 1.75 | I~ [0.91 to 1.69; ft] | I~ [0.833 to 1.55; ft] | U | U | I [0.728 to 1.35; ft] | C~ [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0.911 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | C [0.616 to 1.14; ft] | I~ [0.477 to 0.887; ft] | I~ [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.13 | C~ [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.369 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.976 | U | C [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.55 | U | C [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 2.38 | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | C [1.32 to 2.44; ft/fig/der] | C [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.00319 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 1.09 | U | I [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.68 | U | C [1.01 to 1.87; ft/der] | C [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | 1.79 | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | C [1.32 to 2.44; ft/fig/der] | C [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.00501 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1 | I [1.14 to 1.58; ft/der] | C [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0221 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.0291 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0488 | U | I [0.0217 to 0.0403; ft] | U | U | C [0.0308 to 0.0572; ft] | C [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 2.22 | I [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### ideal_rot, onoff

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 9.92 | I~ [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | inf | I [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 0.66 | I [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | no | I [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.552 | I [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 1 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.645 | I [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0.00932 | C [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.608 | C [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.135 | I~ [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0.675 | U | U | U | U | U | U | U | U | C [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 9 | U | U | U | U | U | U | U | U | I [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.497 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 10.3 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0.999 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 1 | C [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 1.72 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.43 | C [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 4.36 | C [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 1.74 | I~ [0.91 to 1.69; ft] | I~ [0.833 to 1.55; ft] | U | U | I [0.728 to 1.35; ft] | C~ [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0.908 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | C [0.616 to 1.14; ft] | I~ [0.477 to 0.887; ft] | I~ [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.129 | C~ [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.368 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.977 | U | C [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.55 | U | C [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 2.31 | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | C [1.32 to 2.44; ft/fig/der] | C [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.00319 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 1.09 | U | I [0.581 to 1.08; ft/der] | C [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.68 | U | C [1.01 to 1.87; ft/der] | C [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | 1.79 | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | C [1.32 to 2.44; ft/fig/der] | C [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.00375 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1 | I [1.14 to 1.58; ft/der] | C [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0235 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.0225 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0513 | U | I [0.0217 to 0.0403; ft] | U | U | C [0.0308 to 0.0572; ft] | C [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 2.23 | I [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### ideal_sparse, identity

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 6.23 | I [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | inf | I [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 0.785 | I [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | no | I [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.425 | I [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.999 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.656 | I [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0.0178 | C [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.538 | C [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.0848 | I [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0.597 | U | U | U | U | U | U | U | U | C [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 3.47 | U | U | U | U | U | U | U | U | C [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.519 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 2.5 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0.219 | U | U | U | U | U | U | U | U | C [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0.0595 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 0.961 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 0.998 | C [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 1.1 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.02 | C [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 1.32 | I [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 1.1 | C [0.91 to 1.69; ft] | C [0.833 to 1.55; ft] | U | U | C [0.728 to 1.35; ft] | I [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0.488 | C~ [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | I [0.616 to 1.14; ft] | C~ [0.477 to 0.887; ft] | C~ [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.0857 | C [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I [<= 0.01; inf] | U | U | I [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.182 | C [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | I [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 1.3 | U | I [0.581 to 1.08; ft/der] | I [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 2.11 | U | I [1.01 to 1.87; ft/der] | C [1.62 to 3.02; ft/der] | C [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 3.58 | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.00223 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 1.36 | U | I [0.581 to 1.08; ft/der] | I [0.693 to 1.29; ft/der] | C [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 2.33 | U | I [1.01 to 1.87; ft/der] | C [1.62 to 3.02; ft/der] | C [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.000328 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 0.995 | I [1.14 to 1.58; ft/der] | C [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0101 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | -0.00258 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0506 | U | I [0.0217 to 0.0403; ft] | U | U | C [0.0308 to 0.0572; ft] | C [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 2.2 | I [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### trained, rectify

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 0.217 | I [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | 6.02 | I~ [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 1.04 | I~ [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | yes | C~ [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.483 | I~ [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.192 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | none | U*r [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | none | U*r [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.614 | C~ [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.00288 | I [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0.767 | U | U | U | U | U | U | U | U | C [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 10.5 | U | U | U | U | U | U | U | U | I [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.382 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 15.4 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 0.996 | C~ [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 2.32 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.76 | I [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 6.47 | Ir [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | none | U*r [0.91 to 1.69; ft] | U*r [0.833 to 1.55; ft] | U | U | U*r [0.728 to 1.35; ft] | U*r [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | I~r [0.616 to 1.14; ft] | I [0.477 to 0.887; ft] | I [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0 | Ir [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | C~r [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.46 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.671 | U | Cr [0.581 to 1.08; ft/der] | Ir [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.14 | U | Cr [1.01 to 1.87; ft/der] | Ir [1.62 to 3.02; ft/der] | Ir [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.0431 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 0.712 | U | Cr [0.581 to 1.08; ft/der] | Cr [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.17 | U | Cr [1.01 to 1.87; ft/der] | Ir [1.62 to 3.02; ft/der] | Ir [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.0014 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1.08 | I~ [1.14 to 1.58; ft/der] | C~ [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0297 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.0334 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0483 | U | I [0.0217 to 0.0403; ft] | U | U | C [0.0308 to 0.0572; ft] | C [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 3.14 | C [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### trained, onoff

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 0.183 | I [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | 3.47 | I~ [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 1.05 | I~ [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | yes | C~ [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.512 | I~ [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.163 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | none | U*r [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | none | U*r [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.609 | C~ [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.00253 | I [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0.761 | U | U | U | U | U | U | U | U | C [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 10.5 | U | U | U | U | U | U | U | U | I [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | 0.382 | U | U | U | U | U | U | U | U | I [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 15.5 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 6.1e-05 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | I [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0.999 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | C [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 0.994 | C~ [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 2.2 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | 1.77 | I [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 6.53 | I [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | none | U*r [0.91 to 1.69; ft] | U*r [0.833 to 1.55; ft] | U | U | U*r [0.728 to 1.35; ft] | U*r [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | I~r [0.616 to 1.14; ft] | I [0.477 to 0.887; ft] | I [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0 | Ir [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | C~r [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.46 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | C [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.669 | U | Cr [0.581 to 1.08; ft/der] | Ir [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 1.14 | U | Cr [1.01 to 1.87; ft/der] | Ir [1.62 to 3.02; ft/der] | Ir [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.0474 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 0.709 | U | Cr [0.581 to 1.08; ft/der] | Cr [0.693 to 1.29; ft/der] | Ir [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 1.17 | U | Cr [1.01 to 1.87; ft/der] | Ir [1.62 to 3.02; ft/der] | Ir [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.00124 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1.03 | I~ [1.14 to 1.58; ft/der] | C~ [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.0321 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.0338 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0464 | U | I [0.0217 to 0.0403; ft] | U | U | C [0.0308 to 0.0572; ft] | C [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 3.19 | C [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### untrained, rectify

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 0 | I~ [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | none | U* [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | none | U* [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | none | U* [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | none | U* [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0 | I~ [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | none | U* [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | none | U* [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | none | U* [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0 | I~ [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.207 to 0.385; ft/der] | U* [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.223 to 0.415; ft] | U* [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.584 to 1.08; ft] | U* [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | none | U* [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | none | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | none | U* [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 13.9 | I [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | none | U* [0.91 to 1.69; ft] | U* [0.833 to 1.55; ft] | U | U | U* [0.728 to 1.35; ft] | U* [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0 | I~ [0.2 to 0.5; ft] | I~ [0.18 to 0.3; ft/abs/rev] | U | U | I~ [0.02 to 0.1; ft/rev] | I~ [0.616 to 1.14; ft] | I~ [0.477 to 0.887; ft] | I~ [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0 | I~ [0.03 to 0.14; der] | I~ [0.021 to 0.039; der] | U | U | C~ [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.965 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | I [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | none | U | U* [0.581 to 1.08; ft/der] | U* [0.693 to 1.29; ft/der] | U* [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | inf | U | I [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | none | U | U* [-0.05 to 0.13; ft/fig] | U* [-0.13 to -0.03; ft/fig] | U* [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | inf | U | I [0.581 to 1.08; ft/der] | I [0.693 to 1.29; ft/der] | I [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | inf | U | I [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.999 | I [0.15 to 0.35; ft] | I [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | none | U* [1.14 to 1.58; ft/der] | U* [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 1 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 1 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0227 | U | Cr [0.0217 to 0.0403; ft] | U | U | I [0.0308 to 0.0572; ft] | I [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 11.3 | I [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### untrained, onoff

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 0 | I~ [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | none | U* [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | none | U* [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | none | U* [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | none | U* [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0 | I~ [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | none | U* [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | none | U* [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | none | U* [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0 | I~ [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.207 to 0.385; ft/der] | U* [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.223 to 0.415; ft] | U* [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.584 to 1.08; ft] | U* [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | none | U* [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | none | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | none | U* [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | 13.9 | I [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | none | U* [0.91 to 1.69; ft] | U* [0.833 to 1.55; ft] | U | U | U* [0.728 to 1.35; ft] | U* [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0 | I~ [0.2 to 0.5; ft] | I~ [0.18 to 0.3; ft/abs/rev] | U | U | I~ [0.02 to 0.1; ft/rev] | I~ [0.616 to 1.14; ft] | I~ [0.477 to 0.887; ft] | I~ [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0 | I~ [0.03 to 0.14; der] | I~ [0.021 to 0.039; der] | U | U | C~ [<= 0.01; inf] | U | U | I~ [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.964 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | I [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | none | U | U* [0.581 to 1.08; ft/der] | U* [0.693 to 1.29; ft/der] | U* [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | inf | U | I [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | none | U | U* [-0.05 to 0.13; ft/fig] | U* [-0.13 to -0.03; ft/fig] | U* [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | inf | U | I [0.581 to 1.08; ft/der] | I [0.693 to 1.29; ft/der] | I [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | inf | U | I [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | 0.999 | I [0.15 to 0.35; ft] | I [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | none | U* [1.14 to 1.58; ft/der] | U* [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 1 | I [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.999 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.0197 | U | I [0.0217 to 0.0403; ft] | U | U | I [0.0308 to 0.0572; ft] | I [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 11.4 | I [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### hippocampus, identity

| Measure | Def. | Model | dCA1 | dCA3 | iCA3 | vCA3 | DG gc | DG mc | CA2 | SUB | bat CA1 | bat CA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fields per unit, track of 74 field widths | B | 4.38 | I~ [2.29 to 4.33; ft] | U | U | U | U | U | U | U | U | U |
| Negative binomial shape r, 74 widths | B | 1.92 | I [0.399 to 0.741; ft] | U | U | U | U | U | U | U | U | U |
| Variance / mean of field counts, 74 widths | B | 2.8 | I [5 to 9.28; ft/der] | U | U | U | U | U | U | U | U | U |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | B | yes | C [yes; ft] | U | U | U | U | U | U | U | U | U |
| Zero-field fraction against length: max distance from Rich's curve | B | 0.222 | I [0 to 0.155; ft/der] | U | U | U | U | U | U | U | U | U |
| Active fraction (one or more fields), 74 widths | B | 0.871 | I [0.46 to 0.77; ft] | U | U | U | U | U | U | U | U | U |
| Gap CV between a unit's fields (units with 6+), 74 widths | B | 0.495 | I [0.7 to 1.3; ft] | U | U | U | U | U | U | U | U | U |
| Share of units whose field positions reject uniform, 74 widths | B | 0 | C [<= 0.1; ft] | U | U | U | U | U | U | U | U | U |
| Field size CV, 74 widths | B | 0.25 | I [0.462 to 0.858; ft/der] | U | U | U | U | U | U | U | U | U |
| Per-location active fraction, 74 widths | B | 0.059 | C [0.035 to 0.065; der] | U | U | U | U | U | U | U | U | U |
| Field size CV, 34 widths | Ap | 0 | U | U | U | U | U | U | U | U | I [0.441 to 0.819; der] | U |
| Largest/smallest field within a unit, 34 widths | Ap | 1 | U | U | U | U | U | U | U | U | I [3.08 to 5.72; ft] | U |
| Gap CV between a unit's fields, 34 widths | Ap | none | U | U | U | U | U | U | U | U | U* [0.7 to 1.3; ft] | U |
| Fields per active unit, 34 widths | Ap | 1.06 | U | U | U | U | U | U | U | U | I [3.43 to 6.37; ft] | U |
| Single-field share of active units, 34 widths | Ap | 0.941 | U | U | U | U | U | U | U | U | I [0.207 to 0.385; ft/der] | C [0.516 to 0.958; ft] |
| Share of active units with 5+ fields, 34 widths | Ap | 0 | U | U | U | U | U | U | U | U | I [0.223 to 0.415; ft] | I [0.0049 to 0.0091; ft] |
| Active fraction, 34 widths | Ap | 0.00531 | U | U | U | U | U | U | U | U | I [0.584 to 1.08; ft] | I [0.41 to 0.762; ft] |
| Fields per active unit against arena area: Pearson r | B | 1 | C [0.883 to 1.08; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, ratio over an 8-fold area step (not scored) | B | 2.85 | [1.78 (not scored)] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 3.7 mean fields | A | none | U* [0.91 to 1.69; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit in an arena of 13.7 mean fields | A | none | U* [2.38 to 4.42; ft] | U | U | U | U | U | U | U | U | U |
| Fields per active unit, small arena (8 A0) | B | 3.39 | I [0.91 to 1.69; ft] | I [0.833 to 1.55; ft] | U | U | I [0.728 to 1.35; ft] | I [1.58 to 2.94; ft] | U | U | U | U |
| Active fraction, small arena (8 A0) | B | 0.975 | I [0.2 to 0.5; ft] | I [0.18 to 0.3; ft/abs/rev] | U | U | I [0.02 to 0.1; ft/rev] | C [0.616 to 1.14; ft] | I [0.477 to 0.887; ft] | I [0.48 to 0.891; ft] | U | U |
| Per-location active fraction, small arena (8 A0) | B | 0.0575 | C [0.03 to 0.14; der] | I [0.021 to 0.039; der] | U | U | I [<= 0.01; inf] | U | U | I [>= 0.14; ft/inf] | U | U |
| Treves-Rolls sparsity, track of 4.6 widths | none | 0.0961 | I [0.182 to 0.338; ft/asm] | U | U | U | U | U | U | I [0.31 to 0.69; ft/asm] | U | U |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | A | 0.333 | U | I [0.581 to 1.08; ft/der] | I [0.693 to 1.29; ft/der] | I [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | A | 0.533 | U | I [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| PV kernel zero crossing / W_A (Pearson, rates) | A | 0.333 | C [<= 1.3; ft] | C [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | C [<= 0.68; ft/inf] | U | U |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | A | 0.0352 | U | C [-0.05 to 0.13; ft/fig] | I [-0.13 to -0.03; ft/fig] | I [-0.13 to -0.03; ft/fig] | U | U | U | U | U | U |
| Cosine kernel full width at 0.5 / W_A (raw code) | A | 0.347 | U | I [0.581 to 1.08; ft/der] | I [0.693 to 1.29; ft/der] | I [0.847 to 1.57; ft/der] | U | U | U | U | U | U |
| Cosine kernel full width at 0.2 / W_A (raw code) | A | 0.556 | U | I [1.01 to 1.87; ft/der] | I [1.62 to 3.02; ft/der] | I [1.72 to 3.2; ft/der] | U | U | U | U | U | U |
| Cosine kernel zero crossing / W_A (raw code) | A | inf | I [<= 1.3; ft] | I [<= 1.33; ft/fig/der] | I [1.32 to 2.44; ft/fig/der] | I [1.36 to 2.52; ft/fig/der] | U | U | U | I [<= 0.68; ft/inf] | U | U |
| PV correlation between arenas at matched places (Pearson, rates) | none | -0.0159 | I [0.15 to 0.35; ft] | C [-0.07 to 0.13; ft] | U | U | U | U | U | U | U | U |
| Active-set overlap / chance between arenas | B | 1 | I [1.14 to 1.58; ft/der] | C [0.79 to 1.17; ft/der] | U | U | U | U | U | U | U | U |
| Mean-rate correlation across arenas | none | 0.212 | C [0.09 to 0.29; ft] | U | U | U | U | U | U | I [0.77 to 0.97; ft] | U | U |
| Log mean-rate correlation across arenas | none | 0.195 | I [0.55 to 0.75; ft] | I [0.49 to 0.69; ft] | U | U | U | U | U | U | U | U |
| Units whose rate maps beat a shuffle (95th pct) across arenas | none | 0.047 | U | I [0.0217 to 0.0403; ft] | U | U | C [0.0308 to 0.0572; ft] | C [0.0294 to 0.0546; ft] | U | U | U | U |
| Correlation dimension of PVs in an arena (phase 2) | none | 14.1 | I [2.51 to 4.71; ft] | U | U | U | U | U | U | I [5.05 to 6.63; ft] | U | U |

### Every target

Non-primary targets are listed beside the primary ones; they do not decide a cell.

| Measure | Region | Band | Source tag | Primary |
|---|---|---|---|---|
| Fields per unit, track of 74 field widths | dorsal CA1 | 2.29 to 4.33 | full text | yes |
| Fields per unit, track of 74 field widths | dorsal CA1 | 2.47 to 4.59 | derived (review: 0.57 / 7.75 fields per metre; x 48 m by this suite) | no |
| Negative binomial shape r, 74 widths | dorsal CA1 | 0.399 to 0.741 | full text | yes |
| Variance / mean of field counts, 74 widths | dorsal CA1 | 5 to 9.28 | full text (p = 0.14); variance/mean = 1/p derived (this suite) | yes |
| Negative binomial beats Poisson (LR p < 0.01), 74 widths | dorsal CA1 | yes | full text | yes |
| Active fraction (one or more fields), 74 widths | dorsal CA1 | 0.46 to 0.77 | full text | yes |
| Zero-field fraction against length: max distance from Rich's curve | dorsal CA1 | 0 to 0.155 | full text; rescaling to field widths derived (this suite) | yes |
| Gap CV between a unit's fields (units with 6+), 74 widths | dorsal CA1 | 0.7 to 1.3 | full text (gaps exponential); CV = 1 is the exponential's | yes |
| Share of units whose field positions reject uniform, 74 widths | dorsal CA1 | <= 0.1 | full text (0 of 61); the 0.1 bound is the suite's allowance for a 5% false-positive rate | yes |
| Field size CV, 74 widths | dorsal CA1 | 0.462 to 0.858 | full text (65 +- 43 cm); CV derived in the review | yes |
| Per-location active fraction, 74 widths | dorsal CA1 | 0.035 to 0.065 | derived in the review (3.5 fields x 0.65 m / 48 m) | yes |
| Field size CV, 34 widths | bat CA1 | 0.441 to 0.819 | derived in the review from the log-normal; approximate (log sizes skewed) | yes |
| Largest/smallest field within a unit, 34 widths | bat CA1 | 3.08 to 5.72 | full text | yes |
| Gap CV between a unit's fields, 34 widths | bat CA1 | 0.7 to 1.3 | full text (exponential gaps); CV = 1 is the exponential's | yes |
| Fields per active unit, 34 widths | bat CA1 | 3.43 to 6.37 | full text | yes |
| Single-field share of active units, 34 widths | bat CA1 | 0.0854 to 0.159 | full text | no |
| Single-field share of active units, 34 widths | bat CA1 | 0.207 to 0.385 | full text; length in field widths derived (this suite) | yes |
| Single-field share of active units, 34 widths | bat CA3 | 0.516 to 0.958 | full text; length in field widths unknown for CA3 | yes |
| Share of active units with 5+ fields, 34 widths | bat CA1 | 0.223 to 0.415 | full text | yes |
| Share of active units with 5+ fields, 34 widths | bat CA3 | 0.0049 to 0.0091 | full text | yes |
| Active fraction, 34 widths | bat CA1 | 0.584 to 1.08 | full text | yes |
| Active fraction, 34 widths | bat CA3 | 0.41 to 0.762 | full text | yes |
| Fields per active unit against arena area: Pearson r | dorsal CA1 | 0.883 to 1.08 | full text | yes |
| Fields per active unit against arena area: Pearson r | dorsal CA1 | 0.889 to 1.09 | full text (R2 = 0.978); r = sqrt(R2) derived (this suite) | no |
| Fields per active unit, ratio over an 8-fold area step (not scored) | dorsal CA1 | 1.78 (not scored) | full text (1.28 to 2.28); ratio derived (this suite) | no |
| Fields per active unit in an arena of 3.7 mean fields | dorsal CA1 | 0.91 to 1.69 | full text; field definition is any rate above 0 over 9 pixels, not A or B | yes |
| Fields per active unit in an arena of 13.7 mean fields | dorsal CA1 | 2.38 to 4.42 | full text; field definition is any rate above 0 over 9 pixels, not A or B | yes |
| Fields per active unit, small arena (8 A0) | dentate granule | 0.728 to 1.35 | full text; field definition not in the review | yes |
| Fields per active unit, small arena (8 A0) | dentate mossy | 1.58 to 2.94 | full text; field definition not in the review | yes |
| Fields per active unit, small arena (8 A0) | dorsal CA3 | 0.833 to 1.55 | full text; field definition not in the review | yes |
| Fields per active unit, small arena (8 A0) | dorsal CA1 | 0.91 to 1.69 | full text; any-rate field definition | yes |
| Active fraction, small arena (8 A0) | dorsal CA1 | 0.2 to 0.5 | full text (Rich 2014's summary of the literature) | yes |
| Active fraction, small arena (8 A0) | dorsal CA1 | 0.231 to 0.429 | abstract; 1 - 2/3 derived (this suite) | no |
| Active fraction, small arena (8 A0) | dorsal CA1 | 0.373 to 0.747 | full text (criterion-dependent) | no |
| Active fraction, small arena (8 A0) | dorsal CA1 | 0.414 to 0.77 | full text | no |
| Active fraction, small arena (8 A0) | dorsal CA3 | 0.18 to 0.3 | review summary table (Vazdarjanova 2004 abstract, 18%; GoodSmith 2017 full text, 29%) | yes |
| Active fraction, small arena (8 A0) | dorsal CA3 | 0.168 to 0.31 | full text (criterion-dependent) | no |
| Active fraction, small arena (8 A0) | dorsal CA3 | 0.337 to 0.625 | full text | no |
| Active fraction, small arena (8 A0) | dentate granule | 0.02 to 0.1 | review summary table (GoodSmith 2017 full text, 9%) | yes |
| Active fraction, small arena (8 A0) | dentate mossy | 0.616 to 1.14 | full text | yes |
| Active fraction, small arena (8 A0) | CA2 | 0.477 to 0.887 | full text | yes |
| Active fraction, small arena (8 A0) | subiculum | 0.48 to 0.891 | full text | yes |
| Per-location active fraction, small arena (8 A0) | dorsal CA1 | 0.03 to 0.14 | derived in the review (place-cell fraction x field coverage) | yes |
| Per-location active fraction, small arena (8 A0) | dorsal CA3 | 0.021 to 0.039 | derived in the review | yes |
| Per-location active fraction, small arena (8 A0) | dentate granule | <= 0.01 | inference in the review | yes |
| Per-location active fraction, small arena (8 A0) | subiculum | >= 0.14 | full text (95% of spikes in 56% of bins against 26% for CA1); the bound, denser than the densest CA1 value, is the suite's inference | yes |
| Treves-Rolls sparsity, track of 4.6 widths | dorsal CA1 | 0.182 to 0.338 | full text; the review calls it a sparseness index, and the suite assumes the Treves-Rolls formula | yes |
| Treves-Rolls sparsity, track of 4.6 widths | subiculum | 0.31 to 0.69 | full text; formula assumed (Treves-Rolls) | yes |
| Treves-Rolls sparsity, track of 4.6 widths | dorsal CA1 | 0.301 to 0.559 | full text (Skaggs sparsity) | no |
| Treves-Rolls sparsity, track of 4.6 widths | subiculum | 0.546 to 1.01 | full text (Skaggs sparsity) | no |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | dorsal CA3 | 0.581 to 1.08 | full text (0.81 m band, 0.98 m mean field); ratio derived (this suite) | yes |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | intermediate CA3 | 0.693 to 1.29 | full text (3.71 / 3.73); ratio derived (this suite) | yes |
| PV kernel full width at r = 0.5 / W_A (Pearson, rates) | ventral CA3 | 0.847 to 1.57 | full text (6.66 / 5.52); ratio derived (this suite) | yes |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | dorsal CA3 | 1.01 to 1.87 | full text (1.41 m band at r = 0.20, 0.98 m mean field); ratio derived (this suite) | yes |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | intermediate CA3 | 1.62 to 3.02 | full text (8.65 / 3.73); ratio derived (this suite) | yes |
| PV kernel full width at r = 0.2 / W_A (Pearson, rates) | ventral CA3 | 1.72 to 3.2 | full text (13.59 / 5.52); ratio derived (this suite) | yes |
| PV kernel zero crossing / W_A (Pearson, rates) | dorsal CA1 | <= 1.3 | full text (Wang 2020: decorrelated beyond the median field); the 1.3 bound is about one width plus the 30% point tolerance | yes |
| PV kernel zero crossing / W_A (Pearson, rates) | dorsal CA3 | <= 1.33 | full text, figure-read (below 0.1 by about 1 m, mean field 0.98 m); 1.02 x 1.3 derived (this suite) | yes |
| PV kernel zero crossing / W_A (Pearson, rates) | intermediate CA3 | 1.32 to 2.44 | full text, figure-read (zero near 7 m / 3.73 m); ratio derived (this suite) | yes |
| PV kernel zero crossing / W_A (Pearson, rates) | ventral CA3 | 1.36 to 2.52 | full text, figure-read (zero near 10.7 m / 5.52 m); ratio derived (this suite) | yes |
| PV kernel zero crossing / W_A (Pearson, rates) | subiculum | <= 0.68 | full text, widths figure only; bound inferred by this suite: CA1's one width times sqrt(26/56), because subicular firing covers 56% of the box against 26% | yes |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | dorsal CA3 | -0.05 to 0.13 | full text, figure-read | yes |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | intermediate CA3 | -0.13 to -0.03 | full text, figure-read | yes |
| PV kernel floor, 5 to 15 W_A (Pearson, rates) | ventral CA3 | -0.13 to -0.03 | full text, figure-read | yes |
| PV correlation between arenas at matched places (Pearson, rates) | dorsal CA3 | -0.07 to 0.13 | full text | yes |
| PV correlation between arenas at matched places (Pearson, rates) | dorsal CA1 | 0.15 to 0.35 | full text | yes |
| PV correlation between arenas at matched places (Pearson, rates) | dorsal CA1 | -0.001 to 0.199 | full text | no |
| Active-set overlap / chance between arenas | dorsal CA3 | 0.79 to 1.17 | full text (0.11 to 0.14 against 0.12 to 0.14); ratio bounds derived (this suite); overlap formula not in the review | yes |
| Active-set overlap / chance between arenas | dorsal CA1 | 1.14 to 1.58 | full text; ratios derived (this suite) | yes |
| Mean-rate correlation across arenas | subiculum | 0.77 to 0.97 | full text | yes |
| Mean-rate correlation across arenas | dorsal CA1 | 0.09 to 0.29 | full text | yes |
| Log mean-rate correlation across arenas | dorsal CA1 | 0.55 to 0.75 | full text | yes |
| Log mean-rate correlation across arenas | dorsal CA3 | 0.49 to 0.69 | full text | yes |
| Units whose rate maps beat a shuffle (95th pct) across arenas | dentate granule | 0.0308 to 0.0572 | full text; the review counts cell pairs, shuffle details not in the review | yes |
| Units whose rate maps beat a shuffle (95th pct) across arenas | dentate mossy | 0.0294 to 0.0546 | full text | yes |
| Units whose rate maps beat a shuffle (95th pct) across arenas | dorsal CA3 | 0.0217 to 0.0403 | full text | yes |
| Correlation dimension of PVs in an arena (phase 2) | dorsal CA1 | 2.51 to 4.71 | full text (3.61 +- 1.10; band is mean +- SD) | yes |
| Correlation dimension of PVs in an arena (phase 2) | subiculum | 5.05 to 6.63 | full text (5.84 +- 0.79; band is mean +- SD) | yes |


## Reproduce

```
bash analysis/coding_comparison/run.sh            # phase 1, one Slurm task per population
bash analysis/coding_comparison/run.sh 0-6 phase2 # phase 2
python -m analysis.coding_comparison.scorecard    # scorecard.md and scorecard.json
python -m analysis.coding_comparison.figures      # PNGs
python -m pytest hopfield_nav/tests/test_coding_comparison.py
```

To add an encoder, run `python -m analysis.coding_comparison.run --pops '' --pop myenc=/path/to/encoder.pt` (or add a line to `POPULATIONS`), then rerun the scorecard. Results go to `<results>/coding_comparison/<tag>/<population>/`, one JSON per measure (`meta`, `fields_1d`, `fields_2d`, `kernel_1d`, `kernel_2d`, `remapping`, `sparsity`, `phase2`), and figures to `<figures>/coding_comparison/<tag>/`.

## Files

- `analysis/coding_comparison/measures.py`: the measures, pure functions on rate maps
- `analysis/coding_comparison/populations.py`: the population adapter
- `analysis/coding_comparison/geometry.py`: tracks, loops, arenas and rings
- `analysis/coding_comparison/run.py`, `run.sh`, `phase2.py`: the driver
- `analysis/coding_comparison/literature_targets.json`: the targets, with sources and tags
- `analysis/coding_comparison/scorecard.py`, `figures.py`: scoring and figures
- `hopfield_nav/tests/test_coding_comparison.py`: the synthetic checks
