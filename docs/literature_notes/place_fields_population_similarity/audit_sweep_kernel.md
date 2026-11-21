# Audit of the kernel sweep

Audit date 2026-11-21. Target: `sweep_kernel.csv` (45 rows, 24 studies) and `sweep_kernel.md`. Corrections are in `audit_sweep_kernel_corrections.csv` (28 entries: 24 on 14 rows and 4 on the note). The sweep files are unchanged.

## Summary

| Verdict | Rows |
|---|---|
| Confirmed | 20 |
| Corrected (a value, label or status changes) | 14 |
| Unverifiable | 0 |
| Not re-checked (note rows outside the spot-check) | 11 |
| Dropped | 0 |

The 20 confirmed rows are K01 to K06, N01 to N11, N14, N15 and N18. For four of them, Gothard N08 to N11, I confirmed the figure values, but PMC and jneurosci.org blocked the text, so their measure, cell-count and level labels rest on the sweep's WebFetch excerpts. The 14 corrected rows are K07 to K11, K19 to K22, N12, N13, N16, N17 and N19. The 11 rows not re-checked are K12 to K18 and K23 to K26. The row total is 20 + 14 + 11 = 45.

Of the note rows, I spot-checked K01, K03, K10 and K11 in full, plus K02, K04 to K06 and K07 in part. All numbers I re-read sit within one pixel or one bin of the sweep's value, except the scopolamine row N17.

## Sources read in this audit

- Ledergerber 2021 supplement `mmc1.pdf` (the sweep's Europe PMC zip). I extracted Fig. S7 as its embedded raster, 1290 x 1397 px. The main text came from Europe PMC, PMC8170370.
- The user's PDFs: Kjelstrup 2008 main text and SOM, and Brun 2008.
- NCBI BioC: Wang 2020 (PMC7259364), Chettih 2024 (PMC11015962).
- Europe PMC XML: Courellis 2019 (PMC6922474), Raveau 2018 (PMC5841929). The sweep's XML files: Farooq 2024, Blair 2023, Gandit 2024, Ottink 2025, Petersen 2020. Green 2022 came from the earlier `txt/PMC9245695.txt` and SI PDF.
- Resnik 2012: the PMC article page through WebFetch. It loaded once and returned verbatim fragments.
- Figures re-read: the sweep's downloads for Resnik, Raveau, Courellis, Gothard, Green, Chettih, Farooq and Blair. The native Wang Fig. 1 raster (1744 x 2500 px) came from `tmp/render/wang_fig1.png`.
- Not reachable: Gothard 2001 text (PMC bot check, jneurosci.org 403). The Wang Fig. 1 figure route also fails now (PMC bot check, not open access on Europe PMC), so I used the earlier raster.

Scripts are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/audit_sweep_kernel/scripts/`. `led_trace.py`, `led_cols.py` and `led_markers.py` read Ledergerber. `row_cross.py` reads linear curves. `band_read.py` and `band_read_h.py` read matrices through vertical or horizontal colour bars. The work images are in `.../audit_sweep_kernel/work/`.

## Ledergerber 2021, Fig. S7H

Verdict: the sweep's reading is right. The p = 0.0173 in the notes is not in the source.

How I read it:

- The panel is a raster, not vector paths, so I traced pixels. The embedded image is itself upsampled about 2x: column pairs repeat. So the effective resolution is about 37 px per decade.
- Axes, open field: 10^-2 m at x = 736, 10^-1 at 810 and 10^1 at 958 (74 px per decade). r = 1.0 at row 1017 and -0.2 at row 1192.
- Axes, task: 10^-2 at x = 1053, 73.5 px per decade. r = 1.0 at row 1015 and -0.2 at row 1192.5.
- The curves are CA1 (191, 21, 63) and SUB (0, 137, 180).

| Curve | r = 0.5 | Zero crossing | Minimum | First point (2.5 cm) |
|---|---|---|---|---|
| CA1 open field | 22.5 cm | 54 to 55 cm | -0.18 at 113 cm | 0.95 |
| SUB open field | 18.6 cm | 58 to 61 cm | -0.145 at 110 cm | 0.90 |
| CA1 task | 22.2 cm | 53 cm | -0.15 at 83 cm | 0.96 |
| SUB task | 19.2 cm | 61 cm | -0.17 at 106 cm | 0.90 |

The sweep has 22, 19, 22 and 19.5 cm at r = 0.5, and zero crossings of 56, 62, 53 and 60 cm. All are within one effective pixel, about 6% of the distance. The CA1 zero crossing is partly hidden under the blue SUB line. I read it from the visible edge on either side.

Points the sweep did not record:

- The markers sit at 2.5, 7.5, 12.5, 17.5, 22.5 and 27.5 cm. So the curve is binned in 5 cm distance bins and plotted at the bin centres.
- The first point is 0.95 in CA1 and 0.90 in SUB, not 1. The binning and zero-distance rule are not described, so the plotted x positions may be offset from the true pair distances by up to half a bin, 2.5 cm.
- The legend gives one statistic: "F-test: p= 0.0048", from an autoregressive model of each bin's rate on its 8 neighbours. It does not say which task. There is no 0.0173 in the legend, the main text or the figure. The test is not a width comparison.
- The open field came from "5 of the 8 rats". The task used "all 8 rats". The main text says only that "the peak in SUB is narrower than that in CA1".

What this means for the review: the subicular curve is lower at short distances (0.90 against 0.95 at 2.5 cm, about 0.1 lower at 12.5 cm). It is narrower at r = 0.5 by 3 cm, about 14% (19 against 22 cm), and it crosses zero later (about 60 against 54 cm). "Narrower peak" is true only near the top of the curve. The sweep's restatement holds: broad subicular tuning did not widen the population kernel, and it did not narrow it much either.

## The kernel-against-field-width claim

The claim: "in rodent CA1 and CA3 with fields at a 20% threshold, the correlation falls to 0.5 at 0.40 to 0.60 field widths (full width 0.8 to 1.2), against 0.46 for identical Gaussian fields."

Verdict: the arithmetic and the 0.46 derivation are right. But the claim describes its base wrongly, and the reference depends on the threshold.

Recomputed ratios. "Audit" uses my re-read where it differs.

| Row | dist_r05 | Field width and rule | dist / field | full / field | Gaussian reference at that threshold |
|---|---|---|---|---|---|
| K01 rat CA3 dorsal | 40.5 | 98, mean of each cell's widest field, 20% | 0.41 | 0.83 | 0.46 |
| K02 rat CA3 intermediate | 185.5 | 373, same rule | 0.50 | 0.99 | 0.46 |
| K03 rat CA3 ventral | 333 | 552, same rule, some truncated | 0.60 | 1.21 | 0.46 |
| N01 mouse CA1 dorsal | 9.0 | 19, median, 20% | 0.47 | 0.95 | 0.46 |
| K10 rat CA1 dorsal | 14 (audit 14.2) | 35, median, **10%** | 0.40 (0.41) | 0.80 (0.82) | 0.39 |
| K11 rat CA3 dorsal | 20 (audit 19.8) | 42, median, **10%** | 0.48 (0.47) | 0.95 (0.94) | 0.39 |

The Gaussian check. Take identical fields exp(-x²/2σ²) tiled densely and evenly. The PV dot product falls as exp(-d²/4σ²). So the cosine similarity reaches 0.5 at d = 2σ√ln2 = 1.665σ. Pearson gives the same in a space much longer than σ. The full field width at 20% of peak is 2σ√(2 ln 5) = 3.588σ. The ratio is 0.464, so the 0.46 is correct.

The same model gives 0.388 at a 10% threshold and 0.707 at 50%.

Problems:

1. Wang 2020 thresholds fields at 10% of peak, not 20%. The STAR Methods say "thresholding the smoothed firing-rate vectors at 10% of the unit's maximum firing rate". So only Kjelstrup and Resnik measure fields at 20%. That is four points from two studies, and three of them come from one track. Wang's two rows should be compared with 0.39. They sit 6 to 22% above it.
2. The field statistics differ. Kjelstrup uses the mean of each cell's widest field. Resnik and Wang use the median over all fields. Kjelstrup's ventral fields are partly cut by the track ends, and the authors say these field measures "underestimate the spatial scale" (SOM Fig. S6 legend). That raises the ventral ratio.
3. The 0.46 holds only when fields are small against the space. Pearson subtracts the mean across cells. For evenly tiled fields in a space of length L, that gives r(d) = (g - a)/(1 - a), where g = exp(-d²/4σ²) and a = 2√π σ/L. Then the r = 0.5 reference is 0.45 for K01, 0.42 for N01, 0.39 for K02 and 0.36 for K03. The same model predicts floors of -0.29 and -0.49 for K02 and K03, but the data floor near -0.1. So for large fields the model fails, and it cannot give a reference for the intermediate and ventral rows.
4. Point 4 of the note says "the kernel scales with the field". The ratio rises from 0.41 to 0.60 dorsal to ventral, so the kernel grows faster than the measured field. Part of that may be field underestimation (problem 2).

Suggested wording: "In two studies with fields measured at 20% of peak, rat CA3 at three long-axis levels on an 18 m track and mouse dorsal CA1 on a 1.4 m track, the correlation falls to 0.5 at 0.41 to 0.60 field widths (full width 0.83 to 1.21). Identical Gaussian fields give 0.46 at that threshold. Rat dorsal CA1 and CA3 on a 2 m loop, with fields at 10% of peak, give 0.41 and 0.47 against 0.39 for Gaussian fields at 10%."

Related ratios:

- GluA1 knockout: 4.6/17 = 0.27, confirmed. Both Resnik quotes in the note are verbatim: "individual place fields had similar widths in the two genotypes" and "much more spatially coherent".
- Marmoset with my reads: posterior 0.60, middle 0.64, anterior 1.33. The sweep has 0.62, 0.66 and 1.39.
- Brun grid cells: 0.29, 0.26 and 0.28. These fields are at 10% of peak, so the matching reference is 0.39, not 0.46.
- Grid-spacing ratios confirmed: 32/171 = 0.19, 47/301 = 0.16, 67/370 = 0.18. CA3 over MEC spacing: 0.47, 1.23, 1.80.

## Row by row

- **K01** Confirmed. Kjelstrup p. 141: "The mean half-width (at r = 0.50) was 0.81 m for dorsal CA3". In Fig. 3, the bars at r = 0.5 span both sides of zero, so 81 cm is a full width. The field is "0.98 ± 0.03 m", fields above 20% of peak (p. 140), each cell's widest field (SOM p. 7). The 15 cm smoothing is stated in Kjelstrup's own SOM p. 6 ("smoothing factor of 15 cm"), so it needs no appeal to Brun.
- **K02** Confirmed. Text 3.71 m. The Fig. 3B bar at 0.5 spans about ±1.9 m. Field 3.73 m. Zero near ±7 m by eye.
- **K03** Confirmed. Text 6.66 m. The Fig. 3C bar spans about ±3.3 m. Field 5.52 m. Zero near ±10.5 m by eye. The "330 population vectors" times 5 cm gives 16.5 m.
- **K04 to K06** Confirmed for the kernel and spacing. Brun pp. 1207 to 1208: half-widths 32, 47 and 67 cm; median spacing "171 ± 13 cm to 301 ± 23 cm and 370 ± 46 cm"; "Repetitive bands were not observed". Fields above 10% of peak with 5 cm smoothing (p. 1203). I did not re-read the field widths 56, 92 and 119.
- **K07** Corrected, label only. My Fig. 3C trace has x = 608 at 0 cm, 0.870 px per cm, r = 0 at row 89.5 and 69 px per unit. It gives r = 0.5 at 17.6 cm (sweep 16.7, one pixel apart). Zero at 32 cm; -0.30 to -0.32 from 70 to 120 cm; -0.14 at 150 cm. The legend says 132 cells for (A), and (B) uses "the same neurons as in A". The field width is not "not given": Fig. 8F gives CA1 field-width distributions, width = "distance in which 95% of spikes occur" (SI p. 3), but no number in the text.
- **K08** Corrected, label only. Fig. 3K (x = 601 at 0, r = 0 at row 488.5, 77 px per unit): 19.2 cm (sweep 19.5); zero 33.7 cm (33); minimum -0.37 near 70 cm; about -0.08 at 150 cm (sweep -0.12, where the three curves overlap). The jump-1 against jump-1 matrix is within one condition, so it is a kernel. Field width as K07.
- **K09** Corrected, label only. Fig. 3O: 17.8 cm (18.4); zero 31.4 cm (31.6); -0.32 at 70 to 80 cm; -0.16 at 150 cm. "Population representation and changes across conditions were remarkably similar between CA1 and CA3 populations" is verbatim. Fig. 8H gives the CA3 field widths.
- **K10** Corrected, label only. Re-read of Wang Fig. 1D CA1-STD on the native raster, with the colour bar 1.0 at row 1185, 0.5 tick at 1300 and 0 at 1416. r = 0.5 at 24.6°, which is 14.2 cm at 0.576 cm per degree (76 cm O.D., 10 cm wide). r = 0.1 near 64°. 669 place cells in CA1-STD. The field width of 60° is at 10% of peak, not 20%.
- **K11** Corrected, label only. CA3-STD: 34.4° = 19.8 cm. r = 0.1 at about 110° here, against 99° in the earlier audit; the colour bottom is hard to read. 293 place cells. Field 73°, at 10% of peak.
- **K12 to K18** Not re-checked. K12: the file `tmp/dl/pub/battaglia.pdf` is a 5 KB error page, so Fig. 7 is still unseen.
- **K19** Corrected: notes, n_animals, pooling. Values confirmed (table above). Open field came from 5 rats. p = 0.0173 is not in the source.
- **K20** Corrected: notes, n_animals, pooling. 18.6 cm (sweep 19). Zero 58 to 61 cm (sweep 62, within one effective pixel). The n values 191 = 109 + 82 match Fig. S7E.
- **K21** Corrected: notes. 22.2 cm. Zero 53 cm. 305 cells in Fig. S7E.
- **K22** Corrected: notes. 19.2 cm (19.5). Zero 61 cm (60). 418 = 277 + 141.
- **K23 to K26** Not re-checked.
- **N01** Confirmed. Fig. 4C trace: x = 534 at 0 cm, 0.888 px per cm; r = 0 at row 171.5, 154 px per unit. The WT curve crosses 0.5 at -8.6 and +9.6 cm, and the inset crosses its line near ±9 cm. So the text's "half-width ... (9.0 cm)" is one-sided, and the full width is about 18 cm. Zero 27 to 29 cm. Mean -0.11 over 50 to 110 cm. The text confirms 86 of 137 cells in 52 sessions, 11 WT mice and 2.5 cm unsmoothed bins. Rates are "normalized by its mean spike rate". Fields are 19 (11 to 29) cm at 20%, measured as the distance between the outermost threshold crossings.
- **N02** Confirmed. The KO curve crosses 0.5 at -4.1 and +5.2 cm; text 4.6 cm, null 1.7 cm. 75 of 88 cells, 33 sessions, 9 mice. Field 17 (7.5 to 30) cm. Long-range 0.024.
- **N03** Confirmed. Fig. 2C (2.37 px per cm, 402 px per unit): both curves cross 0.5 at -13.3 and +13.1 cm. WT floor 0.21 to 0.24. The text confirms Spearman, "all place cells from all mice within each group", left and right separate, the third day, 5 WT mice and 256 pyramidal cells.
- **N04** Confirmed. Same peak. Floor about 0.17. 6 mice, 259 cells.
- **N05** Confirmed, within 4%. Fig. 5C band read on the 3000 px PNG; the colour bar ticks are at 103.2 px per 0.2. The mean crosses 0.5 at 123 cm (sweep 128). Floor 0.29 to 0.36. The field is 92 cm by eye, the widest field per cell at 20% of peak (Methods). The Methods also give 100 bins of about 4.3 cm and "21.5 cm of effective Gaussian smoothing".
- **N06** Confirmed. 55.5 cm (57). Zero 175 cm (178). Minimum -0.02 near 200 cm. 0.18 to 0.23 beyond 330 cm.
- **N07** Confirmed. 40.7 cm (42). The mean is near 0 at 156 cm. It rises to 0.41 near 314 cm (0.42).
- **N08** Confirmed for the figure; the text is unverifiable here. I read Fig. 6 Box 1 with the bar calibrated on its tick marks: x = 144 at 0, 378.5 px per unit, so the bar spans -0.31 to 1.0. The mean crosses 0.5 at 13.7 cm (median 15.0). Zero 77 cm. I could not re-read the "normalized overlap" formula, the cell counts, the 3.8 mm level or the outbound-only rule. The negative colours are real in the matrix (far mean -0.02 to -0.06), so either the measure subtracts a mean or the sweep's formula reading is wrong. That needs the text.
- **N09** Confirmed for the figure. 11.9 cm (median 12.9). Zero 76 cm.
- **N10** Confirmed for the figure. 14.0 cm (15.0). Zero 47 cm.
- **N11** Confirmed for the figure. 8.6 cm (8.4). Zero 42 cm. The block at the fixed-well end is visible.
- **N12** Corrected: normalization, status. Fig. 3a dots (4.14 px per cm, 42.6 px per unit): 1.03, 0.44, 0.39, 0.16, -0.02, -0.18, -0.27, -0.33 and -0.27 at 0, 5.3, 10.8, 16.1, 20.4, 26.1, 31.6, 37.8 and 45.4 cm. r = 0.5 lies between 0 and 5.3 cm; 4.8 cm is an interpolation inside the first gap. The BioC text says: "multiplied by a constant factor (40.95) such that the value of the exponential curve for visit-visit correlations at the same site was exactly 1". The raw mean correlation was 0.024. Each unit's rate had a 30 min running baseline subtracted and was "divided ... by its standard deviation, which was regularized by adding a small number (0.6 Hz)". 5 birds, 54 sessions.
- **N13** Corrected: smoothing. The Fig. 3a cuboid band mean crosses 0.5 at 14.5 cm, and the median at 15.5 cm. It is 0.35 to 0.37 at 24 cm. The Fig. 3d values match (0.41 at 0.24 m, 0.065 to 0.07 at 0.8 to 0.88 m). The Methods say: "non-overlapping 2 cm bins smoothed by a 2 cm Gaussian kernel". 5 cuboid-reared rats, 63 (44 to 99) cells; each direction analysed separately.
- **N14, N15** Confirmed. The Methods give 20 bins of 3 cm, "sigma = 4 bins", even against odd laps, and row normalization "for illustration purposes only".
- **N16** Corrected: field width, ranges. Fig. 2I W front face re-read (85 x 86 px, 2.9 cm per px, colour bar +1 to -1). Diagonal 0.58. r = 0.5 at 15 to 17 cm across 2 to 4 px box shifts. Half of the diagonal at 46 to 51 cm. Floor 0.03 to 0.05. The text confirms R(trn1, trn2), "averaged together across running directions and rats". Width = 10.8 cm times the bins above 50% of peak. Fig. 2F gives 45 cm in the training session the kernel uses and 53 cm in pre-training.
- **N17** Corrected: values, field width. Diagonal 0.56 to 0.60 (sweep 0.55). r = 0.5 at 15 to 17 cm (sweep 19). Half of the diagonal at 43 to 51 cm (sweep 58). Floor 0.11 to 0.13. Fig. 2F: field 47 cm in training against 45 cm drug-free, so it is not narrower than drug-free. It is narrower than the same rats' pre-training value of 52 cm.
- **N18** Confirmed. The abstract sentence is in PMC11868619.
- **N19** Corrected: env_length_m. The quote, 5 rats and 21 sessions are confirmed. "110 cm diameter" is the maze; the path length is not stated. The local `f3a.jpg` is a 48 KB HTML error page.

## Exclusion list

- Cabral 2014: confirmed. Fig. 6B compares clockwise and counter-clockwise runs.
- Blair, pre-training against training: confirmed from the text.
- Gandit, in against out and flipped: confirmed from the text.
- Gothard Boxes 2 to 5: confirmed. Their x axes start at 27, 54, 81 and 108 cm.
- The other exclusions were not re-checked.

## Note-level issues

- Gap 6 says Green gives no field width. Green has field-width distributions in Fig. 8F and 8H, with no number in the text.
- In the synthesis table, K10 and K11 should say 10% threshold. N16's field width for the training session is about 45 cm. With my reads its ratios become 0.33 to 0.38 at r = 0.5 and 1.0 to 1.1 at half-peak. Blair's fields are at 50% of peak, where the Gaussian reference is 0.71.
- In "Measures differ", Blair's diagonal is 0.58 to 0.59, confirmed.
