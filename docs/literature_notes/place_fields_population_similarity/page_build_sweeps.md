# Page build from the November 2026 sweeps: decisions, exclusions and arithmetic

Written 2026-11-21. These notes record how the three audited sweeps were folded into `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md` and its page. Inputs, all in this folder: `sweep_fields_rat.csv`, `sweep_fields_mouse.csv` and `sweep_kernel.csv` with the audit corrections applied, their `sweep_*.md` notes, and `audit_sweep_fields_rat.md`, `audit_sweep_fields_mouse.md` and `audit_sweep_kernel.md`. Neural data only.

## Files changed

- `analysis/literature_page/aggregate_sweeps.py` (new) builds the "Rat" and "Mouse" rows of `ratio_spec.json` from the two field-width tables and adds "Rat kernel" and "Mouse kernel" entries. Every exclusion, value substitution and class override is a named entry in the script with its reason. `python aggregate_sweeps.py ratio_spec.json --report OUT.md` regenerates the spec and prints the two tables in the appendix below; running it twice gives the same file.
- `ratio_spec.json` holds the new rows. Columns are unchanged. Rows carry an optional `studies` list. The kernel entries carry `header` and `columns_from`. The file is gitignored and needs `git add -f`.
- `ratio_tables.py` reads `header`, `columns_from` and `studies`. When rows have studies it appends a final Studies column.
- `heatmap.py` tells the two grid kinds apart by the first header cell, strips a trailing Studies column into the tooltip ("N studies: A; B"), and draws a second figure group headed "Population similarity width against grid spacing" with the same colour scale and legend. The tooltip omits the studies line when the row label already names the study, as in the kernel rows.
- `build_report_page.py` sends any table whose first header cell starts with "Recording (" to the heat maps, links each pointer paragraph to its own group, adds the studies line to the tooltip script, and updates the second answer card and the footer.

## Field-width rows

### Classes

- Rat: real track up to 3 m, real arena up to 4 m², and large real (track over 3 m or arena over 4 m²). Rat virtual tracks have no class.
- Mouse: real environment up to 2 m, virtual track up to 3 m, and large (over 3 m or over 1.2 m²). Every mouse value that falls in the large class comes from a virtual track. No real mouse environment between 2 and 3 m carries a width, so that gap holds no data.
- Linearized mazes count as tracks: Mizuseki 2012's 75 cm radial arms and 1 × 2 m zigzag maze, and Wirtshafter 2020's T maze with 2.2 m paths. The zigzag maze's path length is not stated; its arms are short and the fields are 1D lengths, so it goes with the tracks.
- Burke 2011's 3.35 m circular track is over 3 m, so intermediate CA1 from that study sits in the large class.
- Jarzebowski 2022's 1.2 m cheeseboard (1.13 m²) is under 1.2 m², so it sits in the mouse real class.

### Labels

A row label is level, region, class and N studies. Rat large rows whose only contributor is Kjelstrup 2008 read "large real 18 m track", so `only_rows_containing: "18 m track"` still fills Brun 2008's same-track columns. A large rat row made only of tracks reads "large real track over 3 m". A mouse large row made only of virtual tracks reads "large virtual track over 3 m". Regions kept apart wherever a width exists: CA1, CA3, CA1 and CA3 pooled (Diba 2008 in rats, Quigley 2023 in mice), CA2/3 (Hainmueller 2018, which the sweep labels CA3 but describes as CA2/3), dentate gyrus with cell types not separated (Boublil 2026), dentate granule cells, mossy cells and subiculum. No rat CA2, rat subiculum or mouse CA2 width survives the exclusions, so those regions have no row.

### Deduplication

- One value per study × cell group × environment × statistic. A row's range is the minimum and maximum over its contributing values, so grouping only changes the study count, never the range.
- Per-field over summed, where a study gives both for the same cells: Keinath 2014 (27.1, 22.3 and 29.3 cm per field from the row notes replace the summed 28.0, 33.1 and 41.9 cm), Jarzebowski 2022 (17.0 and 27.2 cm replace 22.1 and 34.4 cm), Kim 2023 (kim2023_1 and _3 used, the summed kim2023_2 and _4 excluded). Keinath's ventral 35 cm cylinder has only a summed value, 33.1 cm, which fills about 89% of the cylinder; it stays, marked as summed.
- Several measures of the same cells form the study's range: Kjelstrup 2008 mean largest field with autocorrelation width (98 to 112, 307 to 373 and 410 to 552 cm); Tanni 2022 environment D near-wall, centre and long-axis widths (rows 4 to 6, 56 to 177 cm), as the rat audit asked for one entry per environment; Hayman 2011 minor and major axes (21 to 38 cm).
- Same cells or mice in more than one row, as flagged by the audits: Mizuseki 2012 and Royer 2010 dorsal CA3 (Royer is excluded as a coverage bound anyway); Chen 2018 real and virtual squares (the virtual one is excluded, see below); Zhao 2014 days 1 to 3 on the novel 1 m track (one environment, 33 to 40 cm) and the familiar 2 m track (24 cm); Dragoi 2013 familiar arm and novel track (22.1 and 23.9 cm); Go 2021 track and arena. Each is one study in its row.
- Henriksen 2010's three transverse bands are separate cells, so all three values enter, as one study.
- Kim 2023 and Abad-Perez 2025 carry ranges for an unresolved unit: Kim's pixel side of 1.3 to 1.4 cm (20.0 to 21.5 and 18.3 to 19.7 cm, from the mouse audit's geometry check), and Abad-Perez's 50 or 70 cm arena (30.4 to 42.5 cm for CA1, 30.9 to 43.3 cm for subiculum).
- Only the `width_cm` column enters, apart from the per-field substitutions and unit ranges above. Values that appear only in notes or `value_reported` (Harvey 2009 whole-cell 43 cm, Hussaini 2011 track widths, Dong 2021 novel-track widths, Gu 2023 CB+ cells, Sumegi 2025 non-BTSP fields, Mallory 2018 largest field) stay out.

### Judgment calls beyond the brief

- Komorowski 2013 (rat CA3 at three levels in 37 cm boxes) is not on the exclusion list, so it stays, although the widths are figure reads that sit near the study's 200 cm² minimum and the walls cap them. It gives the only rat intermediate and ventral CA3 rows outside the 18 m track.
- Sharp & Green 1994 (subiculum, share of visited pixels with any spike) is excluded as a coverage upper bound, the same kind of measure as Royer 2010. With it out, rat subiculum has no row.
- Hayman 2011's vertical pegboard and helical track are excluded because no class fits a climbing surface; the flat arenas stay.
- Rows whose environment size is not stated are excluded: Schlesiger 2015 and Ravassard 2013 real tracks (42.3 and 44.3 cm, both inside the dorsal CA1 track range anyway) and Mallory 2018 (19.7 cm, inside the mouse dorsal CA1 range). Ravassard 2013's virtual track is excluded because the rat classes are real only.
- Chen 2018's virtual 60 cm square (31.3 cm) is excluded because the mouse classes have no virtual-arena class; its real square stays.
- Abad-Perez 2025: the mouse audit gives the verdict "unverifiable (value confirmed, conversion ambiguous)". The CSV status is "FT; derived", and the value itself is confirmed, so the rows stay with both conversions as the range. This gives the only mouse subiculum row. Drop `abadperez2025_1` and `_2` from the script if "unverifiable" should cover this case.
- The mouse peak-only exclusions also remove Chockanathan 2021's ventral CA1 value (11.9 cm), which uses the same 1.5 SD rule as the dorsal one. Mouse ventral CA1 therefore rests on Keinath 2014 alone.
- Quigley 2023 gets its own "CA1 and CA3 pooled" mouse row, by analogy with Diba 2008, because the paper never assigns its units to CA1.

### Ratios quoted in the review text

All come from the regenerated grids, each cell being row low over column high to row high over column low.

- Rat, track up to 3 m and arena up to 4 m², against M1 (39 to 49 cm): lowest 16.2/49 = 0.33 (dorsal CA3 arena), highest 75/39 = 1.9 (dorsal CA1 track). Against all modules: lowest 16.2/171.7 = 0.094 (M5).
- Rat dorsal CA1 large: 56/49 = 1.1 to 177/39 = 4.5. The 4.5 is Tanni 2022's 177 cm long-axis projection in the 8.75 m² arena; 88.1/39 = 2.3 in the 4.38 m² arena.
- Rat intermediate CA1 large (Burke 2011): 55/49 = 1.1 to 80/39 = 2.1.
- 18 m track rows against box M1 and the same-level MEC columns are unchanged from the earlier grid: ventral 410/49 = 8.4 to 552/39 = 14, and against the ventral MEC column 410/370 = 1.1 to 552/269 = 2.1.
- Mouse real up to 2 m against M1 (30 to 36 cm): lowest 14.8/36 = 0.41 (Go 2021 arena), highest 42.5/30 = 1.4 (Abad-Perez 2025, 70 cm conversion). Against all modules: lowest 14.8/76 = 0.19.
- Mouse virtual up to 3 m: 37/36 = 1.0 to 60/30 = 2.0. Mouse large: CA1 45/36 = 1.25 to 46.4/30 = 1.5; CA2/3 40/36 = 1.1 to 40/30 = 1.3; dentate granule cells 65/36 = 1.8 to 65/30 = 2.2. Lowest virtual or large value against any module: 37/76 = 0.49.

## Kernel rows

Full width at r = 0.5 is twice the one-sided distance. Where the sweep (CSV) and the audit read a figure differently, the row spans both readings, as twice the smallest and twice the largest one-sided reading. The audit's mean and median band reads both count as readings. Kjelstrup 2008 prints full widths (0.81, 3.71 and 6.66 m) and Resnik 2012 prints one-sided values (9.0 and 4.6 cm), so those rows use the printed numbers.

| Row id | Grid row | Sweep, one side (cm) | Audit, one side (cm) | Full width used (cm) |
|---|---|---|---|---|
| K07 | Dorsal CA1, Green 2022, control trials | 16.7 | 17.6 | 33.4 to 35.2 |
| K08 | Dorsal CA1, Green 2022, jump trials | 19.5 | 19.2 | 38.4 to 39 |
| K10 | Dorsal CA1, Wang 2020 | 14 | 14.2 | 28 to 28.4 |
| K14 | Dorsal CA1, Terrazas 2005, walking | 27 | not re-checked | 54 |
| K19 | Dorsal CA1, Ledergerber 2021, open field | 22 | 22.5 | 44 to 45 |
| K21 | Dorsal CA1, Ledergerber 2021, memory task | 22 | 22.2 | 44 to 44.4 |
| N08 | Dorsal CA1, Gothard 2001, light | 14 | 13.7 mean, 15.0 median | 27.4 to 30 |
| N09 | Dorsal CA1, Gothard 2001, darkness | 12 | 11.9 mean, 12.9 median | 23.8 to 25.8 |
| N13 | Dorsal CA1, Farooq 2024, juvenile | 15 | 14.5 mean, 15.5 median | 29 to 31 |
| K01 | Dorsal CA3, Kjelstrup 2008, 18 m track | 40.5 | printed full width 81 | 81 |
| K09 | Dorsal CA3, Green 2022, jump trials | 18.4 | 17.8 | 35.6 to 36.8 |
| K11 | Dorsal CA3, Wang 2020 | 20 | 19.8 | 39.6 to 40 |
| N10 | Dorsal dentate gyrus, Gothard 2001, light | 14 | 14.0 mean, 15.0 median | 28 to 30 |
| N11 | Dorsal dentate gyrus, Gothard 2001, darkness | 9 | 8.6 mean, 8.4 median | 16.8 to 18 |
| K20 | Dorsal subiculum, Ledergerber 2021, open field | 19 | 18.6 | 37.2 to 38 |
| K22 | Dorsal subiculum, Ledergerber 2021, memory task | 19.5 | 19.2 | 38.4 to 39 |
| K02 | Intermediate CA3, Kjelstrup 2008 | 186 | printed 371 | 371 |
| K03 | Ventral CA3, Kjelstrup 2008 | 333 | printed 666 | 666 |
| N01 | Mouse dorsal CA1, Resnik 2012 | printed 9.0 | -8.6 and +9.6 | 18 |
| N03 | Mouse dorsal CA1, Raveau 2018 | 13.5 | -13.3 and +13.1 | 26.4 to 27 |

Left out of the grids, with reasons stated in the review: Brun 2008 grid-cell kernels K04 to K06 (MEC, not hippocampus); rows with no width (K12 Battaglia, K13 Maurer, K16 and K17 Mao, K18 Tanni, K23 Maimon, K26 Dasbehera, N14 and N15 Gandit, N19 Petersen); species with no measured grid spacing (N05 to N07 marmoset, N12 chickadee, K24, K25 and N18 human); genetic models (N02 GluA1 knockout, N04 Dp(16)1Yey); a drug session (N17 scopolamine); passive transport (K15, Terrazas car driving); and N16 Blair drug-free, whose cross-half diagonal is 0.58, so its r = 0.5 crossing sits at 86% of the peak (0.5/0.58) instead of half of it. Blair's half-peak distance, 46 to 51 cm, is in the review's dorsal CA1 table.

Kernel ratios quoted in the review: rodent spaces up to about 4 m run from 16.8/49 = 0.34 (Gothard dentate gyrus, darkness) to 54/39 = 1.4 (Terrazas); without Terrazas the top is 45/39 = 1.15, printed 1.2. Mouse: 18/36 = 0.50 to 27/30 = 0.90. Dorsal CA3 on the 18 m track: 81/49 = 1.7 to 81/39 = 2.1 against M1, and 81/171 = 0.47 to 81/91 = 0.89 against the dorsal MEC column. Intermediate CA3: 371/301 = 1.2 to 371/202 = 1.8. Ventral CA3: 666/370 = 1.8 to 666/269 = 2.5. Brun's grid-cell kernels over their own median spacing: 32/171 = 0.19, 47/301 = 0.16, 67/370 = 0.18 [audit].

## Text derivations

- Ledergerber 2021, Fig. S7H, audit reading. Subiculum 18.6 to 19.2 cm against CA1 22.2 to 22.5 cm at r = 0.5. The audit's "about 14%" is (22 - 19)/22 = 0.136 on the rounded values; per condition it is (22.2 - 19.2)/22.2 = 0.135 in the task and (22.5 - 18.6)/22.5 = 0.17 in the open field. Zero crossings about 60 against 54 cm. First point at 2.5 cm, 0.90 against 0.95. As a share of the 150 cm box side, 18.6/150 = 12% to 22.5/150 = 15%.
- Kernel against field width (audit table): K01 40.5/98 = 0.41, K02 185.5/373 = 0.50, K03 333/552 = 0.60, N01 9.0/19 = 0.47, so 0.41 to 0.60 and full widths 0.83 to 1.21; K10 14.2/35 = 0.41 and K11 19.8/42 = 0.47 at a 10% field threshold. Gaussian references from the audit: 0.464 at 20% and 0.388 at 10%. GluA1 knockout 4.6/17 = 0.27. Anterior marmoset 123/92 = 1.34 (audit) to 128/92 = 1.39 (sweep), printed as 1.3 to 1.4.
- Resnik 2012 zero crossing as a share of the 1.4 m track: 26/140 = 19% to 29/140 = 21%.
- Gothard 2001 CA1 zero crossing as a share of the 176 cm analysed: 76/176 = 43% to 79/176 = 45%.
- Blair 2023: r = 0.5 over field width, 15/45 = 0.33 to 17/45 = 0.38; half-peak distance over field width, 46/45 = 1.0 to 51/45 = 1.1. The answer card's "falls to half within about one field width" covers this row.

## Fields per cell per grid period

Model: each cell has a field rate; counts are Poisson given the rate; rates are gamma across cells with mean μ and shape k. μ is given per smallest grid period in 1D (rate per m times λ) and per λ² in 2D (rate per m² times λ²), with λ = rat M1 0.39 to 0.49 m or mouse M1 0.30 to 0.36 m, so λ² = 0.1521 to 0.2401 m² (rat) and 0.09 to 0.1296 m² (mouse). Each lower bound multiplies the lower rate by the smaller spacing, and each upper bound the higher rate by the larger spacing.

Datasets used, with inputs from the CSV columns `fields_per_cell`, `count_distribution`, `env_length_m` and `env_area_m2`: Rich 2014, Lee 2020, Zhao 2014, Tanni 2022, Fenton 2008, Keinath 2014, Grieves 2020, Jung 1994, Kjelstrup 2008, Chen 2026, Sosa 2025, Burke 2011, Mou 2018 (both species), Dombeck 2010, Sloin 2022, Gu 2023, Dragoi 2013, Kim 2020, Henriksen 2010, Park 2011, Jarzebowski 2022, Kim 2023, Donegan 2020, Quigley 2023 and Boublil 2026.

Not used, and why: Park 2011's cylinder (only single-field shares, no mean count, so no slope); Wirtshafter 2020 (total T-maze length not stated); Diamantaki 2016 (cells recorded in different arenas); Neunuebel 2012 (box of 0.34 or 1.82 m², not resolved); Grieves 2020 lattice (3D volume); Hollup 2001 (unverifiable); Mankin 2015 (no number); Senzai 2017 (fractions, and a mean only for a subset of 18 mossy cells); Arbab 2018, GoodSmith 2022 and Tarcsay 2025 (medians); Zaremba 2017, Radvansky 2018, Lenkey 2025 and Bourboulou 2019 (one field per cell by construction); Sheintuch 2023, Rechavi 2022 and Schuette 2022 (peak-only field rules, which can split one field into several peaks); Sumegi 2025 (counts only in a figure).

Rate bases. Rich 2014 counts all 253 putative pyramidal cells, with silent cells found in flanking sleep sessions. Lee 2020 counts all 736 cells imaged in one session, 204 (27.7%) with no field. Grieves 2020 counts all 628 arena cells, 87 with no field. Zhao 2014 counts all putative pyramidal cells. Jung 1994 is given both over all complex-spike cells (44 fields in 80 dorsal cells; 25 fields in 129 intermediate cells, where 25 = 22 place cells + 3 second fields) and over place cells. Every other dataset counts place cells, active cells or cells with a field, or does not say.

Chen 2026 k is a moment estimate from the reported mean and SD over place cells: k = mean² / (SD² - mean). Lee 2020 k comes from the audit's re-extracted Fig. 1G histogram (counts for 0 to 19 fields: 204, 139, 72, 68, 45, 36, 36, 23, 24, 22, 16, 11, 7, 6, 10, 3, 3, 2, 4, 5). The moment estimate uses the population variance; the maximum-likelihood estimate fits a negative binomial with the mean fixed at the sample mean, which is its maximum-likelihood value. Removing the 204 cells with no field and repeating the moment estimate on the remaining 532 cells gives mean 4.682, variance 16.49 and k = 4.682² / (16.49 - 4.682) = 1.86, against 0.886 with them.

Arithmetic, as printed by the scratch script (`tmp/page_build/fields_per_period.py` in the job folder):

```
## Tanni 2022 rat dCA1
  native: slope (2.28 - 1.28) / (8.75 - 1.09) = 1.00 / 7.66 = 0.1305 per m^2 per place cell; intercept 1.28 - 0.1305 x 1.09 = 1.14
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 0.1305 x 0.1521 = 0.01986; 0.1305 x 0.2401 = 0.03134  ->  0.02 to 0.031

## Fenton 2008 rat dCA1
  native: slope (3.4 - 1.3) / (2.1 - 0.36) = 2.1 / 1.74 = 1.207 per m^2; intercept 1.3 - 1.207 x 0.36 = 0.866
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 1.207 x 0.1521 = 0.1836; 1.207 x 0.2401 = 0.2898  ->  0.18 to 0.29

## Keinath 2014 mouse dCA1
  native: slope (2.20 - 1.06) / (0.3848 - 0.0962) = 1.14 / 0.2886 = 3.95 per m^2; intercept 1.06 - 3.95 x 0.0962 = 0.68
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 3.95 x 0.09 = 0.3555; 3.95 x 0.1296 = 0.5119  ->  0.36 to 0.51

## Keinath 2014 mouse vCA1
  native: slope (2.04 - 0.97) / 0.2886 = 1.07 / 0.2886 = 3.708 per m^2; intercept 0.97 - 3.708 x 0.0962 = 0.613
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 3.708 x 0.09 = 0.3337; 3.708 x 0.1296 = 0.4805  ->  0.33 to 0.48

## Rich 2014 rat dCA1
  native: mean of gamma(shape 0.57, rate 7.75 per m) = 0.57 / 7.75 = 0.07355 per m; CV of rates 1 / sqrt(0.57) = 1.32
  per period: 0.07355 x 0.39 = 0.02868; 0.07355 x 0.49 = 0.03604  ->  0.029 to 0.036

Lee 2020 histogram: n = 736, fields = 2491, mean = 3.385, variance (population) = 16.31, var/mean = 4.82, k = mean^2 / (var - mean) = 11.45 / 12.93 = 0.886
  negative-binomial maximum likelihood k (mean fixed at 3.38) = 0.76
## Lee 2020 mouse dCA1
  native: 2491 / 736 = 3.385 per cell; / 41 m = 0.08255; / 40.4 m analysed = 0.08378 per m; active cells 2491 / 532 = 4.68
  per period: 0.08255 x 0.3 = 0.02476; 0.08378 x 0.36 = 0.03016  ->  0.025 to 0.03

## Grieves 2020 rat dCA1
  native: 1.231 / 3.24 = 0.3799 per m^2 (all 628 cells); 1.429 / 3.24 = 0.441 among cells with a field
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 0.3799 x 0.1521 = 0.05779; 0.3799 x 0.2401 = 0.09122  ->  0.058 to 0.091

## Zhao 2014 mouse dCA1 novel 1 m
  native: 0.43 to 0.62 per cell per direction / 1 m
  per period: 0.43 x 0.3 = 0.129; 0.62 x 0.36 = 0.2232  ->  0.13 to 0.22

## Zhao 2014 mouse dCA1 familiar 2 m
  native: 1.29 / 2 = 0.645 per m per direction
  per period: 0.645 x 0.3 = 0.1935; 0.645 x 0.36 = 0.2322  ->  0.19 to 0.23

## Chen 2026 mouse dCA1
  native: 6.7 / 8.88 = 0.7545; 6.1 / 8.08 = 0.755 per m; k(A) = 6.7^2 / (3.7^2 - 6.7) = 44.89 / 6.99 = 6.42; k(B) = 6.1^2 / (3.3^2 - 6.1) = 37.21 / 4.79 = 7.77
  per period: 0.7545 x 0.3 = 0.2264; 0.755 x 0.36 = 0.2718  ->  0.23 to 0.27

## Sosa 2025 mouse dCA1
  native: 1.38 / 4.5 = 0.3067; 1.48 / 4.5 = 0.3289 per m per place cell
  per period: 0.3067 x 0.3 = 0.092; 0.3289 x 0.36 = 0.1184  ->  0.092 to 0.12

## Kjelstrup 2008 rat CA3
  native: 1.5 / 16.5 = 0.09091; 1.7 / 16.5 = 0.103 per m per direction
  per period: 0.09091 x 0.39 = 0.03545; 0.103 x 0.49 = 0.05048  ->  0.035 to 0.05

## Henriksen 2010 rat dCA1
  native: 1.57 / 3.14 = 0.5; 2.13 / 3.14 = 0.6783 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 0.5 x 0.1521 = 0.07605; 0.6783 x 0.2401 = 0.1629  ->  0.076 to 0.16

## Park 2011 rat CA1
  native: 5.89 / 2.52 = 2.337 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 2.337 x 0.1521 = 0.3555; 2.337 x 0.2401 = 0.5612  ->  0.36 to 0.56

## Park 2011 rat CA3
  native: 4.96 / 2.52 = 1.968 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 1.968 x 0.1521 = 0.2994; 1.968 x 0.2401 = 0.4726  ->  0.3 to 0.47

## Park 2011 rat DG
  native: 5.35 / 2.52 = 2.123 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 2.123 x 0.1521 = 0.3229; 2.123 x 0.2401 = 0.5097  ->  0.32 to 0.51

## Jung 1994 rat dCA1
  native: 1.22 / 0.46 = 2.652 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 2.652 x 0.1521 = 0.4034; 2.652 x 0.2401 = 0.6368  ->  0.4 to 0.64

## Jung 1994 rat iCA1
  native: 1.14 / 0.46 = 2.478 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 2.478 x 0.1521 = 0.3769; 2.478 x 0.2401 = 0.595  ->  0.38 to 0.6

## Jung 1994 rat dCA1 over all CS cells
  native: 44 / 80 = 0.55 per cell; / 0.46 = 1.196 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 1.196 x 0.1521 = 0.1819; 1.196 x 0.2401 = 0.2871  ->  0.18 to 0.29

## Jung 1994 rat iCA1 over all CS cells
  native: 25 / 129 = 0.194 per cell; / 0.46 = 0.4213 per m^2
  per period: lambda^2 = 0.39^2 to 0.49^2 = 0.1521 to 0.2401 m^2; 0.4213 x 0.1521 = 0.06408; 0.4213 x 0.2401 = 0.1012  ->  0.064 to 0.1

## Burke 2011 rat iCA1
  native: 1.6 / 3.35 = 0.4776; 2.1 / 3.35 = 0.6269 per m
  per period: 0.4776 x 0.39 = 0.1863; 0.6269 x 0.49 = 0.3072  ->  0.19 to 0.31

## Mou 2018 rat dCA1
  native: 0.90 / 2 = 0.45 per m per trajectory
  per period: 0.45 x 0.39 = 0.1755; 0.45 x 0.49 = 0.2205  ->  0.18 to 0.22

## Mou 2018 mouse dCA1
  native: 1.2 / 2 = 0.6 per m per trajectory
  per period: 0.6 x 0.3 = 0.18; 0.6 x 0.36 = 0.216  ->  0.18 to 0.22

## Dombeck 2010 mouse dCA1
  native: 881 / 808 = 1.09; / 1.8 m = 0.6057 per m
  per period: 0.6057 x 0.3 = 0.1817; 0.6057 x 0.36 = 0.2181  ->  0.18 to 0.22

## Sloin 2022 mouse dCA1
  native: 1.28 / (2 x 1.5) = 0.4267; 1.36 / 3 = 0.4533 per m per direction
  per period: 0.4267 x 0.3 = 0.128; 0.4533 x 0.36 = 0.1632  ->  0.13 to 0.16

## Gu 2023 mouse dCA1
  native: 2.25 / (2 x 1.14) = 0.9868 per m per direction
  per period: 0.9868 x 0.3 = 0.2961; 0.9868 x 0.36 = 0.3553  ->  0.3 to 0.36

## Dragoi 2013 mouse dCA1 L-track
  native: 1.90 / 1.55 = 1.226 per m
  per period: 1.226 x 0.3 = 0.3677; 1.226 x 0.36 = 0.4413  ->  0.37 to 0.44

## Dragoi 2013 mouse dCA1 first arm
  native: 1.52 / 0.9 = 1.689; 1.52 / 0.65 = 2.338 per m
  per period: 1.689 x 0.3 = 0.5067; 2.338 x 0.36 = 0.8418  ->  0.51 to 0.84

## Kim 2020 mouse MC
  native: 3.51 / 2.01 = 1.746; 4.44 / 2.01 = 2.209 per m
  per period: 1.746 x 0.3 = 0.5239; 2.209 x 0.36 = 0.7952  ->  0.52 to 0.8

## Boublil 2026 mouse CA1
  native: 2.26 / 0.17 = 13.29 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 13.29 x 0.09 = 1.196; 13.29 x 0.1296 = 1.723  ->  1.2 to 1.7

## Boublil 2026 mouse CA3
  native: 1.9 / 0.17 = 11.18 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 11.18 x 0.09 = 1.006; 11.18 x 0.1296 = 1.448  ->  1 to 1.4

## Boublil 2026 mouse DG
  native: 1.86 / 0.17 = 10.94 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 10.94 x 0.09 = 0.9847; 10.94 x 0.1296 = 1.418  ->  0.98 to 1.4

## Quigley 2023 mouse CA1/CA3
  native: 1.70 / 0.24 = 7.083 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 7.083 x 0.09 = 0.6375; 7.083 x 0.1296 = 0.918  ->  0.64 to 0.92

## Donegan 2020 mouse CA1
  native: 2.0 / 0.24 = 8.333 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 8.333 x 0.09 = 0.75; 8.333 x 0.1296 = 1.08  ->  0.75 to 1.1

## Donegan 2020 mouse CA2
  native: 2.61 / 0.24 = 10.88 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 10.88 x 0.09 = 0.9788; 10.88 x 0.1296 = 1.409  ->  0.98 to 1.4

## Kim 2023 mouse GC
  native: 1.2 / 0.36 = 3.333 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 3.333 x 0.09 = 0.3; 3.333 x 0.1296 = 0.432  ->  0.3 to 0.43

## Kim 2023 mouse MC
  native: 1.87 / 0.36 = 5.194 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 5.194 x 0.09 = 0.4675; 5.194 x 0.1296 = 0.6732  ->  0.47 to 0.67

## Jarzebowski 2022 mouse dCA1
  native: 1.7 / 1.131 = 1.503 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 1.503 x 0.09 = 0.1353; 1.503 x 0.1296 = 0.1948  ->  0.14 to 0.19

## Jarzebowski 2022 mouse iCA1
  native: 1.6 / 1.131 = 1.415 per m^2
  per period: lambda^2 = 0.3^2 to 0.36^2 = 0.09 to 0.1296 m^2; 1.415 x 0.09 = 0.1273; 1.415 x 0.1296 = 0.1833  ->  0.13 to 0.18
```

## Other changes to the review text

- The second section's heading now reads "Only ventral rat CA3 shows a kernel that declines smoothly across most of an environment; in anterior marmoset hippocampus similarity forms one block per track arm". The anterior marmoset kernel falls to 0.5 at 123 to 128 cm on a 4.26 m L-shaped track and never crosses zero, but the matrix is two blocks bounded by the corner, so it is not a smooth decline along the track. The second answer card says the same.
- The second section's tables gain rows for rat dorsal CA1 (Ledergerber, Gothard, Farooq, Blair), mouse dorsal CA1 (Resnik, Raveau), dorsal CA3 (Green 2022 CA3), a new dentate gyrus table (Gothard), the updated subiculum row (Ledergerber, Fig. S7H), and new marmoset (Courellis 2019) and bird (Chettih 2024) tables. Values and statuses follow the kernel CSV and audit; where they differ the row gives both readings.
- Claims the new kernel rows made false, now corrected: "Every other published curve comes from dorsal CA1, CA3, subiculum or retrosplenial cortex in environments no longer than a 4 m loop" (opening summary); "each one shows a peak that falls to near zero within about one field width, then a flat tail" (second section); "a full-text search of the marmoset paper found no PV-distance analysis" and "no other ventral or intermediate PV-versus-distance curve was found in any species" (similarity section); the similarity section heading "Dorsal similarity curves reach baseline within one field width", because Resnik 2012 crosses zero at 1.4 to 1.5 median field widths and Blair 2023 does not cross zero by 200 cm, now "fall to half within about one field width"; "The table lists every within-environment measurement found".
- Subicular kernel: every statement that the subicular peak is narrower, or that Fig. S7H was not seen, now gives the audited reading and says the subiculum is narrower only near the top of the curve. Places changed: the second section's subiculum row and overview paragraph, the detailed similarity table, the subiculum section heading and paragraph, the summary table's similarity row, Ledergerber's full citation, and the conclusion.
- The field-width section heading changed from "Field widths stay under two grid spacings except on the 18 m track" to "Field widths stay within about two smallest grid spacings except in large rat arenas and on the 18 m track", because Tanni 2022's 4.38 and 8.75 m² arenas give up to 4.5 smallest spacings. "In rats only the 18 m track has all three levels" became false once Komorowski 2013 entered (three CA3 levels in 37 cm boxes) and was rewritten inside "Three limits apply". The mouse paragraph's sentence about Hussaini 2011 falling inside the old tetrode row was replaced by a statement of the peak-only exclusions.
- The summary table's fields-per-cell, similarity and field-width rows were updated to the new numbers.

## Not done or left open

- The review's theory section (normative and network models) predates this task and was left as it is, although the brief says neural data only. Nothing from models was added.
- No browser was available, so the page was checked structurally (script parse, unique ids, anchors, tag balance, heat-map row lengths), not visually.
- The kernel grids cover two mouse rows only, because mouse kernels with a width exist only for Resnik 2012 and Raveau 2018 wild types.

## Appendix: generated row and exclusion tables

Printed by `aggregate_sweeps.py --report`. Count-only rows appear here with the reason "no width in cm"; the review's source tables list only the exclusions that concern a width.

### Rat: rows and contributions

| Row | N | Range | Contributions (row id: value, environment, how) |
|---|---|---|---|
| Dorsal CA1, real track up to 3 m, 5 studies | 5 | 23 to 75 cm | mizuseki2012_1: 75 cm, 250 x 7 cm; mizuseki2012_5: 38.3 cm, Radial maze, seven 75 cm arms; mizuseki2012_7: 48 cm, Zigzag maze 100 x 200 cm; wirtshafter2020_1: 29.1 cm, Double-sided T maze; lee2025_1: 23 cm, 130 x 10 cm track with one rotatable arm; mou2018_1: 40 cm, Rectangular 2 m track run back and forth; wang2020_1: 34.6 cm, Circular track, 76 cm outer diameter, 10 cm wide |
| Dorsal CA1, real arena up to 4 m², 8 studies | 8 | 21 to 74.8 cm | jung1994_1: 24.2 cm, 68 x 68 cm square chamber, 51 cm walls; mizuseki2012_3: 47.5 cm, 180 x 180 cm; park2011_1: 36 cm, 76 cm diameter cylinder; park2011_2: 57.8 cm, 1.8 x 1.4 m box; fenton2008_1: 40.3 cm, 68 cm diameter cylinder; fenton2008_2: 45.9 cm, 150 x 140 cm chamber floor; henriksen2010_1: 53 cm, 200 cm diameter cylinder, 50 cm walls; henriksen2010_2: 58.5 cm, 200 cm diameter cylinder, 50 cm walls; henriksen2010_3: 49.6 cm, 200 cm diameter cylinder, 50 cm walls; tanni2022_1: 50.5 to 58.6 cm, Environment A, 87.5 x 125 cm; tanni2022_2: 67.7 to 74.8 cm, Environment B, 175 x 125 cm; hayman2011_1: 21 to 38 cm, Several square or circular flat arenas, 60 to 200 cm across, minor and major axes of the same main fields; jacob2020_1: 22.9 cm, Grey cylinder 100 cm diameter, 50 cm walls, baseline session with all  |
| Dorsal CA1, large real track or arena, 2 studies | 2 | 56 to 177 cm | rich2014_1: 65 cm, 48 m x 9 cm elevated track with turns, built up in sections of 3, 7, 1; tanni2022_3: 74 to 88.1 cm, Environment C, 175 x 250 cm; tanni2022_4: 76.5 to 110 cm, Environment D, 350 x 250 cm; tanni2022_5: 114.5 to 150.5 cm, Environment D, 350 x 250 cm; tanni2022_6: 56 to 177 cm, Environment D, 350 x 250 cm, projection on the long axis, 56 to 120 cm near the wall, 132 to 177 cm in the centre |
| Dorsal CA1 and CA3 pooled, real track up to 3 m, 1 study | 1 | 42 to 52 cm | diba2008_1: 52 cm, 170 x 6.2 cm; diba2008_2: 42 cm, Same track shortened to 100 cm |
| Dorsal CA3, real track up to 3 m, 2 studies | 2 | 33.8 to 50 cm | mizuseki2012_2: 50 cm, 250 x 7 cm; mizuseki2012_6: 33.8 cm, Radial maze, seven 75 cm arms; mizuseki2012_8: 34 cm, Zigzag maze 100 x 200 cm; wang2020_2: 42 cm, Circular track, 76 cm outer diameter, 10 cm wide |
| Dorsal CA3, real arena up to 4 m², 4 studies | 4 | 16.2 to 42.2 cm | kjelstrup2008_10: 28.1 cm, 1 m x 1 m square box; mizuseki2012_4: 40.3 cm, 180 x 180 cm; park2011_3: 32.1 cm, 76 cm diameter cylinder; park2011_4: 42.2 cm, 1.8 x 1.4 m box; komorowski2013_1: 16.2 cm, Two 37 x 37 cm contexts joined by an alley |
| Dorsal CA3, large real 18 m track, 1 study | 1 | 98 to 112 cm | kjelstrup2008_1: 98 cm, 18 m x 12 cm straight track through three rooms; kjelstrup2008_4: 112 cm, 18 m x 12 cm straight track through three rooms |
| Dorsal dentate granule cells, real arena up to 4 m², 1 study | 1 | 21.9 to 30.6 cm | diamantaki2016_1: 21.9 to 30.6 cm, Cell 103: square 70 x 70 cm arena, cells 102, 103 and 1046, all fields summed; cell 104 left out (O-maze, an equal-area diameter on a track) |
| Intermediate CA1, real track up to 3 m, 1 study | 1 | 37 cm | lee2025_2: 37 cm, 130 x 10 cm track with one rotatable arm |
| Intermediate CA1, real arena up to 4 m², 1 study | 1 | 48.8 cm | jung1994_2: 48.8 cm, 68 x 68 cm square chamber, 51 cm walls |
| Intermediate CA1, large real track over 3 m, 1 study | 1 | 55 to 80 cm | burke2011_1: 80 cm, Circular track, about 335 cm circumference, with a barrier and food si; burke2011_2: 55 cm, Circular track, about 335 cm circumference, with a barrier and food si |
| Intermediate CA3, real arena up to 4 m², 1 study | 1 | 24 to 25.2 cm | komorowski2013_2: 24 to 25.2 cm, Two 37 x 37 cm contexts joined by an alley |
| Intermediate CA3, large real 18 m track, 1 study | 1 | 307 to 373 cm | kjelstrup2008_2: 373 cm, 18 m x 12 cm straight track through three rooms; kjelstrup2008_5: 307 cm, 18 m x 12 cm straight track through three rooms |
| Ventral CA3, real arena up to 4 m², 1 study | 1 | 24 to 25.4 cm | komorowski2013_3: 24 to 25.4 cm, Two 37 x 37 cm contexts joined by an alley |
| Ventral CA3, large real 18 m track, 1 study | 1 | 410 to 552 cm | kjelstrup2008_3: 552 cm, 18 m x 12 cm straight track through three rooms; kjelstrup2008_6: 410 cm, 18 m x 12 cm straight track through three rooms |

### Rat: excluded rows

| Row id | Region, level | Reason | Would belong to |
|---|---|---|---|
| kjelstrup2008_7 | CA3, dorsal | population-vector band at r = 0.5, not a field width; same cells as _1 to _3; in the kernel grid | dorsal CA3, large |
| kjelstrup2008_8 | CA3, intermediate | population-vector band at r = 0.5, not a field width; same cells as _1 to _3; in the kernel grid | intermediate CA3, large |
| kjelstrup2008_9 | CA3, ventral | population-vector band at r = 0.5, not a field width; same cells as _1 to _3; in the kernel grid | ventral CA3, large |
| park2011_5 | DG-unspecified, dorsal | no width (the DG value is only in Fig. 4B, not read) | no row |
| royer2010_1 | CA3, ventral | ratio only (ventral single-arm fields 39% larger), no width | no row |
| royer2010_2 | CA3, dorsal | coverage of all suprathreshold pixels, an upper bound for one field; may share cells with mizuseki2012_4 | dorsal CA3, arena |
| royer2010_3 | CA3, ventral | coverage of all suprathreshold pixels, an upper bound for one field | ventral CA3, arena |
| neunuebel2012_1 | DG-unspecified, dorsal | width cleared by the audit: the area unit cannot be resolved | no row |
| neunuebel2012_2 | DG-unspecified, dorsal | width cleared by the audit: the area unit cannot be resolved | no row |
| neunuebel2012_3 | DG-mossy, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| mankin2015_1 | CA2, dorsal | CA2 size only relative to CA1 (24.5% larger), no width | no row |
| mankin2015_2 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| mankin2015_3 | CA3, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| kim2012_1 | SUB, intermediate | spatial activity fraction, not a width | no row |
| kim2012_2 | SUB, intermediate | spatial activity fraction, not a width | no row |
| kim2012_3 | SUB, intermediate | spatial activity fraction, not a width | no row |
| kim2012_4 | CA1, intermediate | spatial activity fraction, not a width | intermediate CA1, large |
| karlsson2008_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| karlsson2008_2 | CA3, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| grieves2020_1 | CA1, dorsal | diameter of the minimum enclosing sphere, a maximum extent | dorsal CA1, arena |
| grieves2020_2 | CA1, dorsal | diameter of the minimum enclosing sphere, a maximum extent; 3D lattice | no row |
| schlesiger2015_1 | CA1, dorsal | track length not stated, so no environment class | dorsal CA1, track |
| hayman2011_2 | CA1, dorsal | vertical pegboard, a climbing surface that fits no class | no row |
| hayman2011_3 | CA1, dorsal | helical climbing track that fits no class | no row |
| ravassard2013_1 | CA1, dorsal | track length not stated, so no environment class | dorsal CA1, track |
| ravassard2013_2 | CA1, dorsal | virtual track; the rat classes cover real environments only | dorsal CA1, track |
| shen1997_1 | CA1, dorsal | status unverifiable (values from WebFetch fragments only) | dorsal CA1, track |
| shen1997_2 | CA1, dorsal | status unverifiable; also aged rats | dorsal CA1, track |
| hollup2001_1 | CA1, dorsal | status unverifiable | dorsal CA1, arena |
| sharp1994_1 | SUB, dorsal | share of visited pixels with any spike, a coverage upper bound (as Royer) | no row |
| sharp1994_2 | SUB, dorsal | share of visited pixels with any spike, a coverage upper bound (as Royer) | no row |
| leesm2022_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| leesm2022_2 | SUB, dorsal | no width in cm (counts, fractions or ratios only) | no row |

### Mouse: rows and contributions

| Row | N | Range | Contributions (row id: value, environment, how) |
|---|---|---|---|
| Dorsal CA1, real environment up to 2 m, 18 studies | 18 | 14.8 to 42.5 cm | abadperez2025_2: 30.4 to 42.5 cm, 50 x 50 cm and 70 x 70 cm arenas, all fields summed; 50 or 70 cm arena, the paper does not say which; arbab2018_1: 29.3 cm, Transparent circular arena, 64 cm diameter, four wall posters; boublil2026_3: 18.1 cm, Familiar 41 cm square arena; chen2018_1: 26.1 cm, 60 x 60 cm real square; dragoi2013_1: 22.1 cm, One arm of a 90 x 65 cm rectangular walled linear-track maze, 4 cm wid; dragoi2013_2: 23.9 cm, Isolated novel linear track on the same maze; geiller2022_1: 33.2 cm, Real treadmill belt 2 m, cue-rich; go2021_1: 19.7 cm, Real air-floated circular track, outer diameter 32.5 cm, lane width 5 ; go2021_2: 14.8 cm, Real air-floated circular open field, 32.5 cm diameter; gu2023_1: 31.1 cm, U-shaped track, 114 cm total arm length, 5.5 cm wide, back and forth; hussaini2011_1: 25.7 cm, 50 x 50 cm box; indersmitten2019_1: 23.6 cm, Real linear track 76 cm, left half cue-rich, right half plain; jarzebowski2022_1: 17 cm, Cheeseboard maze, 120 cm diameter, per field; the row's 22.1 cm sums a cell's fields; keinath2014_1: 27.1 cm, 35 cm cylinder, per field; the row's 28.0 cm sums a cell's fields; keinath2014_2: 22.3 cm, 70 cm cylinder, per field; the row's 33.1 cm sums a cell's fields; lenkey2025_1: 37.3 cm, Real running wheel 157 cm circumference, 10 cm wide, in darkness, tact; mao2017_1: 37 cm, Real linear treadmill belt 150 cm with tactile cues, sucrose reward on; mou2018_1: 33 cm, Rectangular 2 m track, run back and forth; sloin2022_1: 40 cm, 150 cm linear track, back and forth, freely moving; sloin2022_2: 40 cm, 150 cm linear track; zhao2014_1: 40 cm, Novel linear track about 1 m long in a new room, day 1; zhao2014_2: 35 cm, Novel linear track about 1 m long, day 2; zhao2014_3: 33 cm, Novel linear track about 1 m long, day 3; zhao2014_4: 24 cm, Familiar rectangular track about 2 m long, laps forward and back, trai; ziv2013_1: 24 cm, 84 x 4.5 cm elevated linear track |
| Dorsal CA1, virtual track up to 3 m, 7 studies | 7 | 37 to 60 cm | bourboulou2019_1: 51.5 cm, 2 m virtual linear track, no 3D objects; bourboulou2019_2: 48.7 cm, 2 m virtual linear track with three 3D objects; bourboulou2019_3: 41.5 cm, 2 m virtual linear track with 2D wall patterns, no objects; curreli2022_1: 37 cm, Virtual corridor 180 cm, three 60 cm wall patterns, wheel; dombeck2010_1: 50 cm, 180 cm virtual linear track; dong2021_1: 38 cm, 3 m virtual linear track, one direction with teleport; harvey2009_1: 41 cm, 180 cm virtual linear track; krishnan2022_1: 53.3 cm, Virtual linear track 2 m, treadmill, reward at track end, teleport to ; radvansky2018_1: 60 cm, Olfactory virtual linear track, 2 m, run in darkness with odor gradien |
| Dorsal CA1, large virtual track over 3 m, 3 studies | 3 | 45 to 46.4 cm | hainmueller2018_1: 45 cm, 4 m virtual linear track; lee2020_1: 46 cm, ~41 m winding virtual path, ~14 cm wide; sumegi2025_1: 46.4 cm, Virtual corridor 832 cm, checkerboard walls with six landmarks, reward |
| Dorsal CA1 and CA3 pooled, real environment up to 2 m, 1 study | 1 | 23.1 cm | quigley2023_1: 23.1 cm, 40 x 60 cm box with four interior baffles and wall cues, 20 min |
| Dorsal CA3, real environment up to 2 m, 2 studies | 2 | 17.7 to 27 cm | boublil2026_1: 17.7 cm, Familiar black square arena, 41 cm; hussaini2011_2: 27 cm, 50 x 50 cm box |
| Dorsal CA3, virtual track up to 3 m, 1 study | 1 | 44 cm | dong2021_2: 44 cm, 3 m virtual linear track |
| Dorsal CA2/3, large virtual track over 3 m, 1 study | 1 | 40 cm | hainmueller2018_2: 40 cm, 4 m virtual linear track |
| Dorsal dentate gyrus, real environment up to 2 m, 1 study | 1 | 16.9 cm | boublil2026_2: 16.9 cm, Familiar 41 cm square arena |
| Dorsal dentate granule cells, real environment up to 2 m, 1 study | 1 | 20 to 21.5 cm | kim2023_1: 20 to 21.5 cm, Four enclosures in one room: two 60 cm squares, 64 cm octagon, 69 cm c, per field, pixel side 1.3 to 1.4 cm (the audit's geometry check) |
| Dorsal dentate granule cells, large virtual track over 3 m, 1 study | 1 | 65 cm | hainmueller2018_dg1: 65 cm, 4 m virtual linear track |
| Dorsal mossy cells, real environment up to 2 m, 1 study | 1 | 18.3 to 19.7 cm | kim2023_3: 18.3 to 19.7 cm, As kim2023_1, per field, pixel side 1.3 to 1.4 cm (the audit's geometry check) |
| Dorsal subiculum, real environment up to 2 m, 1 study | 1 | 30.9 to 43.3 cm | abadperez2025_1: 30.9 to 43.3 cm, 50 x 50 cm, all fields summed; 50 or 70 cm arena, the paper does not say which |
| Intermediate CA1, real environment up to 2 m, 1 study | 1 | 27.2 cm | jarzebowski2022_2: 27.2 cm, Cheeseboard maze, 120 cm diameter, per field; the row's 34.4 cm sums a cell's fields |
| Ventral CA1, real environment up to 2 m, 1 study | 1 | 29.3 to 33.1 cm | keinath2014_3: 33.1 cm, 35 cm cylinder, all fields summed, no per-field value given; fills about 89% of the 35 cm cylinder; keinath2014_4: 29.3 cm, 70 cm cylinder, per field; the row's 41.9 cm sums a cell's fields |

### Mouse: excluded rows

| Row id | Region, level | Reason | Would belong to |
|---|---|---|---|
| bittner2017_1 | CA1, dorsal | subthreshold Vm ramp, not a spike field; no width | no row |
| chen2018_2 | CA1, dorsal | virtual 2D square; the mouse classes have no virtual-arena class (same cells as chen2018_1) | dorsal CA1, virtual |
| chen2025_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| chen2026_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| chockanathan2021_1 | CA1, dorsal | peak-only field rule (bins at least 1.5 SD above the mean), which gives 7 to 9 cm in mice | dorsal CA1, virtual |
| chockanathan2024_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| donegan2020_2 | CA1, dorsal | areas in pixels of unstated size, no width | no row |
| grijseels2021_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| lee2020_2 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| mallory2018_1 | CA1, dorsal | arena size not read, so no environment class | dorsal CA1, real |
| rechavi2022_1 | CA1, dorsal | peak-only field rule (4 cm bins with no smoothing), which gives 7 to 9 cm in mice | dorsal CA1, real |
| sartorato2026_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| schuette2022_1 | CA1, dorsal | peak-only field rule (mean + 1.5 SD), which gives 7 to 9 cm in mice | dorsal CA1, real |
| sheintuch2023_2 | CA1, dorsal | peak-only field rule (30% of maximum on 4 cm bins with no smoothing), which gives 7 to 9 cm in mice | dorsal CA1, real |
| sosa2025_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| stefanini2020_2 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| tarcsay2025_1 | CA1, dorsal | peak-only field rule (Gaussian-mixture split), which gives 7 to 9 cm in mice | dorsal CA1, real |
| zaremba2017_1 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| zhang2020_2 | CA1, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| chockanathan2021_2 | CA1, ventral | peak-only field rule (bins at least 1.5 SD above the mean), which gives 7 to 9 cm in mice | no row |
| chockanathan2024_2 | CA1, ventral | no width in cm (counts, fractions or ratios only) | no row |
| mchugh1996_1 | CA1, unspecified | no width in cm (counts, fractions or ratios only) | no row |
| donegan2020_1 | CA2, dorsal | areas in pixels of unstated size, no width | no row |
| kim2020_3 | CA3, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| senzai2017_3 | CA3, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| senzai2017_6 | CA3, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| sheintuch2023_1 | CA3, dorsal | peak-only field rule (30% of maximum on 4 cm bins with no smoothing), which gives 7 to 9 cm in mice | dorsal CA3, real |
| danielson2017_2 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| goodsmith2022_1 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| kim2020_1 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| kim2023_2 | DG-granule, dorsal | all fields summed; the per-field value of the same cells (kim2023_1) is used | dorsal dentate granule cells, real |
| senzai2017_1 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| senzai2017_4 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| stefanini2020_1 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| zhang2020_1 | DG-granule, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| danielson2017_1 | DG-mossy, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| goodsmith2022_2 | DG-mossy, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| kim2020_2 | DG-mossy, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| kim2023_4 | DG-mossy, dorsal | all fields summed; the per-field value of the same cells (kim2023_3) is used | dorsal mossy cells, real |
| senzai2017_2 | DG-mossy, dorsal | no width in cm (counts, fractions or ratios only) | no row |
| senzai2017_5 | DG-mossy, dorsal | no width in cm (counts, fractions or ratios only) | no row |
