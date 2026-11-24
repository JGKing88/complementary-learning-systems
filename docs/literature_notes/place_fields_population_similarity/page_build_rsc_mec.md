# Page build from the retrosplenial and non-grid MEC sweeps: decisions, exclusions and arithmetic

Written 2026-11-23. These notes record how the two audited sweeps of 23 November 2026 were folded into `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md` and its page. Inputs, all in this folder: `sweep_fields_rsc.csv`, `sweep_kernel_rsc.csv`, `sweep_rsc.md` and `audit_sweep_rsc.md`; `sweep_fields_mec_nongrid.csv`, `sweep_kernel_mec.csv`, `sweep_mec_nongrid.md` and `audit_sweep_mec.md`; and `sweep_kernel.csv`. The audit corrections were already applied to the CSVs. Every rule of `page_build_sweeps.md` holds unless a section below says otherwise. Neural data only.

## Files changed

- `sweep_kernel.csv`. K16 (Mao 2017, RSC) and K17 (Mao 2017, CA1) now hold the audited values from `audit_sweep_rsc.md`. Only those two rows changed, and the file keeps its CRLF line ends. Values are in the kernel section below.
- `analysis/literature_page/aggregate_sweeps.py`. It reads `sweep_fields_rsc.csv` and `sweep_fields_mec_nongrid.csv`. Two config dicts, `RSC` and `MEC`, hold every exclusion with its reason, the region labels, and the one class override. `aggregate_extra` builds one row per region and environment class and tags it with a `group`. `passed_ca1` sends the retrosplenial sweep's non-duplicate mouse CA1 widths to the hippocampal mouse rows. The kernel list gains K17 (mouse dorsal CA1) and, under the retrosplenial group, K16. `--report` prints the new sections listed in the appendix. Two runs on the tracked spec gave byte-identical files (sha256 549ac90c6b32e629ec15fa467c6bd3a66e6309156169e9e89ece9ca25c7dfa98) and identical reports.
- `analysis/literature_page/ratio_spec.json`. Six new mouse rows with a `group` field, one changed hippocampal row (dorsal CA1, large virtual track over 3 m) and two new mouse kernel rows. The rat entries are unchanged.
- `analysis/literature_page/ratio_tables.py`. A row with a `group` prints a header row, `| **group** | | ... |`, before the first row of each run of that group.
- `analysis/literature_page/heatmap.py`. `parse` reads those header rows into a general `group` field. Rows before any header keep the old grouping by a leading Dorsal, Intermediate or Ventral, and lose that word from their label as before; rows under a header keep their full label. A rebuild of the unedited review with the new code differed from the committed page only in the corner cell ("Hippocampal recording" became "Recording") and the two captions, which no longer say every row is hippocampal.
- `analysis/literature_page/build_report_page.py`. One footer sentence.
- The review and `docs/run_pages/place_fields_population_similarity.html`.

## Duplicates, one source per curve

- K16 and K17 in `sweep_kernel.csv` are the same Mao 2017 curves as K-mao2017_1 and K-mao2017_2 in `sweep_kernel_rsc.csv`. K16 and K17 now carry the audited values and feed the grids; the K-mao2017 rows do not.
- MK01 to MK03 in `sweep_kernel_mec.csv` are K04 to K06 (Brun 2008 grid cells). Neither set enters a grid, and the review cites the curves once.
- mao2017_ca1_1 and lenkey2025_3 in the retrosplenial sweep repeat mao2017_1 and lenkey2025_1 in `sweep_fields_mouse.csv`, which the dorsal CA1 row already holds. Both are excluded.
- mao2017_rsc_3 (superficial dysgranular, 3 mice) is a subset of the cells in mao2017_rsc_1 and _2 with the same statistic as _2. Excluded.
- Rows by condition from one study share cells and count as one study in their row: navratilova2026_1 and _2 (familiar and novel), mao2017_rsc_1 and _2 (mean and median of the same cells, which form the study's range, as Kjelstrup's two measures did before).

## Field-width rows

### Groups, classes and labels

- The mouse grid ends with two groups, "Retrosplenial cortex" and "MEC, non-grid cells". The rat grid gets no group: its only retrosplenial width is excluded (below) and the non-grid MEC sweep has no rat width.
- Classes follow the mouse rules: real environment up to 2 m, virtual track up to 3 m, large over 3 m or 1.2 m². Navratilova 2026's circular virtual tracks of 3.14, 3.77 and 5.03 m fall in the large class, labelled "large virtual track over 3 m". The large-class label now tests for "virtual" in the environment type as well as "VR track", so the Navratilova rows get the virtual label; the existing hippocampal rows are unaffected.
- A label is region, class and N studies. Retrosplenial rows split by subregion (dysgranular, granular, deep layers), as hippocampal rows split by subfield. They do not split by anteroposterior level: the three dysgranular studies sit at three levels (Mao 2017 posterior, Mao 2018 at AP -2.5 mm, Lenkey 2025 at AP -2.2 mm), no study pairs levels, and splitting would leave one study per row. The source table names each study's level.

### Judgment calls beyond the brief

1. MNG01 and MNG04 (Mallory 2018) stay although the arena size is not in the accessible text. This departs from the rule that excluded mallory2018_1 (the same study's CA1 width) from the hippocampal rows. Reasons: they are the only non-grid MEC widths, the grid spacing that forms the mouse M1 high end (36.26 cm) comes from the same mice, and the label says "real open arena of unstated size". The rows sit in their own group, so the missing class does not mix them with any classed row. Drop `MNG01` and `MNG04` from `MEC["class"]` and add them to `MEC["exclude"]` if the size rule should hold here too; the group then has no rows.
2. navratilova2026_4 (CA1, 30.9 cm, Thy1 mice, 3.14 to 5.03 m virtual tracks) joins the hippocampal row "Dorsal CA1, large virtual track over 3 m". It is an audited mouse CA1 width that no other row holds. The row goes from 3 studies and 45 to 46.4 cm to 4 studies and 30.9 to 46.4 cm. Its width is a first-to-last calcium event span per lap, not a thresholded field, and the source table and text say so. This changed three review statements, listed under "Review text". To undo it, add `navratilova2026_4` to `RSC["exclude"]`.
3. Smith 2012 (rat, plus maze) is excluded from the grid on two grounds: the maze size is not stated, so no class (as Schlesiger 2015 and Ravassard 2013), and an equal-area diameter on narrow arms understates the length along an arm (as Diamantaki 2016 cell 104 on the O-maze). The text gives its numbers.
4. Border width (MNG04) is the field's extent perpendicular to the wall. It has its own labelled row, "Dorsal MEC border cells, width perpendicular to the wall", so it is never pooled with field diameters.
5. Mao 2017's deep-layer (mao2017_rsc_4, 48.0 cm) and superficial granular (mao2017_rsc_5, 49.5 cm) medians come from different cells than the pooled superficial dysgranular rows, so each gets its own row. Both rest on few cells (60 and 30).
6. Exclusions by analogy with earlier rules: mao2018_2 (after hippocampal lesions, a manipulation, as the scopolamine session); MNG03 (HCN1 knockout mice, a genetic model, as N02 and N04 in the kernels); navratilova2026_3 (the pooled cells split by mouse line; the pooled rows _1 and _2 are used and the text gives the split); park2024_1 (vertex-cell fields, not place-like, with an approximate area); MNG05 (grid cells, and an autocorrelogram central peak rather than a rate-map field).
7. Near-width measures get a named reason instead of "no width": subramanian2024_3 (sparsity index), alexander2020_4 (egocentric boundary preferred distance, uncertain axis), vanwijngaarden2020_2 (preferred wall distance), lozano2017_1 and zhang2022_4 (head-direction widths in degrees), MNG02 (box coverage), MNG24 (track share inside cue-cell fields) and MNG29 (wall-to-field-centre distance). The appendix lists every other excluded row with "no width in cm".

### Ratios quoted in the review text

All come from the regenerated mouse grid, each cell being row low over column high to row high over column low, with mouse M1 30 to 36 cm, M2 42 to 60 cm and M3 65 to 76 cm.

- Retrosplenial rows against M1: dysgranular, real up to 2 m, 25/36 = 0.69 to 39.6/30 = 1.3; dysgranular, large virtual, 38.1/36 = 1.1 to 40.2/30 = 1.3; deep layers 48/36 = 1.3 to 48/30 = 1.6; granular 49.5/36 = 1.4 to 49.5/30 = 1.65, which the grid prints as 1.6. So 0.69 to 1.6 against M1. Lowest against any module: 25/76 = 0.33.
- Non-grid MEC against M1: 56.1/36 = 1.56 to 56.1/30 = 1.87, printed 1.6 to 1.9. Same-recording ratio: 56.1/36.26 = 1.55 (`audit_sweep_mec.md` gives 1.547). Border width: 14.4/36 = 0.40 to 14.4/30 = 0.48; same recordings 14.43/36.26 = 0.40.
- Dorsal CA1, large virtual track, with Navratilova: 30.9/36 = 0.86 to 46.4/30 = 1.5. Lowest virtual or large value against any module: 30.9/76 = 0.41 (it was 37/76 = 0.49).

## Kernel rows

Full width at r = 0.5 is twice the one-sided distance; where the sweep and the audit read a figure differently, the row spans both readings.

| Row id | Grid row | Sweep, one side (cm) | Audit, one side (cm) | Full width used (cm) |
|---|---|---|---|---|
| K17 | Mouse dorsal CA1, Mao 2017, 1.5 m treadmill loop | 13.4 | 13.39 | 26.8 (2 x 13.39 = 26.78 rounds to the same value) |
| K16 | Mouse dysgranular RSC, Mao 2017, 1.5 m treadmill loop, under "Retrosplenial cortex" | 13.2 | 13.25 | 26.4 to 26.5 |

K16 also records zero crossing 37.8 cm (the Discussion says "~35 cm"), floor about -0.20 at 75 cm and minimum -0.23 near 93 cm. K17 records zero crossing 33.1 cm, floor about -0.26 and minimum -0.264 at 75.7 cm. Field widths from the same study: RSC 39.6 cm mean and 34.5 cm median, CA1 37.0 and 31.5 cm.

Left out of the grids, with reasons in the review and in the script comment: K-mao2017_1 and _2 (duplicates of K16 and K17); K-mao2017_3 to _5 (layer and subregion subsets of 123, 60 and 30 cells from 3 of the same 4 mice, r = 0.5 at 13.3 to 14.1 cm, within 1 cm of K16; the text gives them); K-mao2017_6 and _7 (Thy1 example matrices read by colour-bar lookup with about ±0.03 in r, cell counts unclear, and _7 removes the belt cues; the text gives the cue effect); fischer2020_k1 and _k2 (split-half correlation peaking at 0.818 and 0.757, so r = 0.5 sits at 61% and 66% of the peak, the reason N16 Blair is out, and the width changes along the track; the text gives the landmark-centred widths, 48.5 to 51.4 cm full width); alexander2017_k1 and miller2019_k1 (no width); MK01 to MK03 and K04 to K06 (grid cells; duplicates of each other); K26 Dasbehera (no width; cosine similarity peaks stay above 0.6 to 8 m).

Kernel ratios quoted in the review: RSC 26.4/36 = 0.73 to 26.5/30 = 0.88; CA1 26.8/36 = 0.74 to 26.8/30 = 0.89. The mouse dorsal CA1 range stays 18 to 27 cm and 0.50 to 0.90 spacings with Mao added.

## Fields per cell per grid period

Same model and convention as `page_build_sweeps.md`: μ per smallest grid period in 1D (rate per m times λ) and per λ² in 2D (rate per m² times λ²), with mouse M1 λ = 0.30 to 0.36 m and λ² = 0.09 to 0.1296 m². Each lower bound multiplies the lower rate by the smaller spacing, and each upper bound the higher rate by the larger spacing. Values are tagged [derived] in the review.

Cells counted:

- Mao 2017: the count distribution (Fig. 3c bars) covers the 297 RSC place cells of 2,256 AAV-labelled neurons. The all-cell value multiplies by 297/2256. CA1 counts come from 611 place cells, imaging and silicon probes together, against a 26% share from imaging alone, so CA1 is given per place cell only.
- Esteves 2023: counts cover spatially selective cells of Thy1 sham mice, by criteria the authors call liberal. The all-cell value multiplies by the panel A share (about 0.46 familiar, 0.41 novel), both means over animals. The lesioned mice (esteves2023_3 and _4) are left out.
- Navratilova 2026: Fig. S2B gives fields per cell per metre over all imaged cells, 0.093 RSC and 0.146 CA1 in the familiar room, 0.059 and 0.080 in the novel one. The place-cell value uses the audit's Fig. S2A reading, about 81% one field and 19% two or more, counted as 2.
- Hardcastle 2017: 307 non-grid cells that encode position in a model and 42 border cells; cells that do not encode position are not counted. 228 of 275 sessions used 100 cm boxes, 29 used 91.4 cm and 18 used 70 cm. The low rate assumes 1 m²; the high rate divides by the session-weighted mean area, 0.949 m². The per-cell box size is not reported, so session shares stand in for cell shares.

Arithmetic, as printed by the scratch script (`tmp/page_build_rsc_mec/derivations.py` in the job folder, with `fix_k16_k17.py`, `splice_grids.py` and `assemble_notes.py` beside it):

```
## Mao 2017 RSC, 1.5 m belt
  fields per place cell: 1 x 0.555 + 2 x 0.336 + 3 x 0.106 = 1.545 (CSV bars); audit 1.546, 1.55 normalised
  per place cell per m: 1.545 / 1.5 = 1.03; 1.55 / 1.5 = 1.033
  per period (lambda = 0.3 to 0.36): 1.03 x 0.3 = 0.309; 1.033 x 0.36 = 0.372  ->  0.31 to 0.37
  share 297 / 2256 = 0.1316; per imaged cell 0.2034 to 0.2041; per m 0.1356 to 0.136
  per period (lambda = 0.3 to 0.36): 0.1356 x 0.3 = 0.04068; 0.136 x 0.36 = 0.04897  ->  0.041 to 0.049
  CA1 per place cell: 0.752 + 2 x 0.223 + 3 x 0.020 = 1.258; 1.265 normalised; per m 0.8387 to 0.8433
  per period (lambda = 0.3 to 0.36): 0.8387 x 0.3 = 0.2516; 0.8433 x 0.36 = 0.3036  ->  0.25 to 0.3

## Esteves 2023 RSC, Thy1 sham, 1.5 m belts
  familiar: 1 x 0.79 + 2 x 0.2 + 3 x 0.006 = 1.208 per selective cell; / 1.5 = 0.8053 per m
  per period (lambda = 0.3 to 0.36): 0.8053 x 0.3 = 0.2416; 0.8053 x 0.36 = 0.2899  ->  0.24 to 0.29
  familiar over all cells: share 0.46 x 1.208 = 0.5557 per cell; / 1.5 = 0.3705 per m
  per period (lambda = 0.3 to 0.36): 0.3705 x 0.3 = 0.1111; 0.3705 x 0.36 = 0.1334  ->  0.11 to 0.13
  novel: 1 x 0.765 + 2 x 0.224 + 3 x 0.01 = 1.243 per selective cell; / 1.5 = 0.8287 per m
  per period (lambda = 0.3 to 0.36): 0.8287 x 0.3 = 0.2486; 0.8287 x 0.36 = 0.2983  ->  0.25 to 0.3
  novel over all cells: share 0.41 x 1.243 = 0.5096 per cell; / 1.5 = 0.3398 per m
  per period (lambda = 0.3 to 0.36): 0.3398 x 0.3 = 0.1019; 0.3398 x 0.36 = 0.1223  ->  0.1 to 0.12

## Navratilova 2026, 3.14 to 5.03 m virtual tracks (per cell per m, all imaged cells, Fig. S2B)
  RSC familiar: 0.093 per cell per m
  per period (lambda = 0.3 to 0.36): 0.093 x 0.3 = 0.0279; 0.093 x 0.36 = 0.03348  ->  0.028 to 0.033
  RSC novel: 0.059 per cell per m
  per period (lambda = 0.3 to 0.36): 0.059 x 0.3 = 0.0177; 0.059 x 0.36 = 0.02124  ->  0.018 to 0.021
  CA1 familiar: 0.146 per cell per m
  per period (lambda = 0.3 to 0.36): 0.146 x 0.3 = 0.0438; 0.146 x 0.36 = 0.05256  ->  0.044 to 0.053
  CA1 novel: 0.08 per cell per m
  per period (lambda = 0.3 to 0.36): 0.08 x 0.3 = 0.024; 0.08 x 0.36 = 0.0288  ->  0.024 to 0.029
  RSC place cells (Fig. S2A, 2+ counted as 2): 0.81 x 1 + 0.19 x 2 = 1.19, about 1.2; 1.2 / 5.03 = 0.2386; 1.2 / 3.14 = 0.3822 per m
  per period (lambda = 0.3 to 0.36): 0.2386 x 0.3 = 0.07157; 0.3822 x 0.36 = 0.1376  ->  0.072 to 0.14
  ratio RSC / CA1 familiar 0.093 / 0.146 = 0.637

## Hardcastle 2017 MEC, boxes of 100 cm (228 sessions), 91.4 cm (29), 70 cm (18)
  session-weighted mean area (228 x 1.0 + 29 x 0.8354 + 18 x 0.49) / 275 = 0.9493 m^2
  non-grid: 1.9 per cell; / 1.0 = 1.9; / 0.9493 = 2.002 per m^2
  per period (lambda^2 = 0.09 to 0.1296): 1.9 x 0.09 = 0.171; 2.002 x 0.1296 = 0.2594  ->  0.17 to 0.26
  border: 1.55 per cell; / 1.0 = 1.55; / 0.9493 = 1.633 per m^2
  per period (lambda^2 = 0.09 to 0.1296): 1.55 x 0.09 = 0.1395; 1.633 x 0.1296 = 0.2116  ->  0.14 to 0.21
  grid (native only): 5.69 per cell; / 1.0 = 5.69; / 0.9493 = 5.994 per m^2
  non-grid / grid fields per cell 1.90 / 5.69 = 0.3339

## Text ratios
  Mallory diameters: 2 sqrt(2473.45/pi) = 56.12; 2 sqrt(2692.29/pi) = 58.55; mean - SD 240.25 -> 17.49; mean + SD 4706.65 -> 77.41
  Mallory ratios: 56.1 / 36.26 = 1.547; 14.43 / 36.26 = 0.398; 58.5 / 42.16 = 1.388; 42.16 / 36.26 = 1.163
  Mallory share 32 / 527 = 0.06072; grid 37 / 527 = 0.07021
  Grid columns: non-grid 56.1/36 = 1.558 to 56.1/30 = 1.87; border 14.4/36 = 0.4 to 14.4/30 = 0.48
  RSC grid: 25/36 = 0.6944; 39.6/30 = 1.32; 38.1/36 = 1.058; 40.2/30 = 1.34; 48/36 = 1.333; 49.5/30 = 1.65; lowest 25/76 = 0.3289
  CA1 large virtual with Navratilova: 30.9/36 = 0.8583; lowest virtual or large 30.9/76 = 0.4066
  Mao field ratios: 39.6/37.0 = 1.07; 34.5/31.5 = 1.095
  Smith: 2 sqrt(1031.44/pi) = 36.24; 2 sqrt(368.16/pi) = 21.65; area ratio 2.802; diameter ratio 1.674
  Navratilova: 40.2/30.9 = 1.301; 38.1/32.1 = 1.187
  Shares: 13/26 = 0.5; 27.0/39.7 = 0.6801; 19.1/28.1 = 0.6797; 23.0/38.3 = 0.6005; 45/85 = 0.5294; Lenkey 20.5/17.4 = 1.178
  Smith learning: 5/56 = 0.08929; 63/225 = 0.28
  Mao fields per m: 1.546/1.5 = 1.031; 1.26/1.5 = 0.84
  Mao kernel: 13.25/13.4 = 0.9888; 13.2/13.4 = 0.9851; zero 37.8/33.1 = 1.142; 37.8/33.0 = 1.145; 37.8/150 = 0.252
  Mao kernel full width over field: RSC 26.4/39.6 = 0.6667, 26.5/39.6 = 0.6692, 26.4/34.5 = 0.7652, 26.5/34.5 = 0.7681; CA1 26.8/37.0 = 0.7243, 26.8/31.5 = 0.8508
  Mao one-sided over field: RSC 13.2/39.6 = 0.3333, 13.25/39.6 = 0.3346, 13.2/34.5 = 0.3826, 13.25/34.5 = 0.3841; CA1 13.39/37.0 = 0.3619, 13.4/37.0 = 0.3622, 13.39/31.5 = 0.4251, 13.4/31.5 = 0.4254
  Kernel grid: RSC 26.4/36 = 0.7333 to 26.5/30 = 0.8833; CA1 26.8/36 = 0.7444 to 26.8/30 = 0.8933
  Fischer peak share at r = 0.5: 0.5/0.818 = 0.6112; 0.5/0.757 = 0.6605
  MEC shares: 307/794 = 0.3866; 101/794 = 0.1272; 42/794 = 0.0529; 81/239 = 0.3389; 75/239 = 0.3138; 226/880 = 0.2568; 139/880 = 0.158; 63/880 = 0.07159; 30/193 = 0.1554; 22/193 = 0.114; 19/69 = 0.2754; 160/1136 = 0.1408; 0.675 x 0.95 = 0.6412; cue cells 0.13 x (1 - 0.28 - 0.11) = 0.0793
  Gardner non-module shares: 1 - 0.078 = 0.922; 1 - 0.196 = 0.804; 1 - 0.212 = 0.788; 1 - 0.256 = 0.744
  RSC HD: van der Goes 319/2777 = 0.1149
```

## Review text

### Audit replacements applied

From `audit_sweep_rsc.md`, "Review sentences that must change":

1. Second section, retrosplenial table: the Mao 2017 row replaced with the audit's text.
2. Similarity paragraph after the region tables: the "~35 cm" sentence replaced with the audit's text, which quotes the Discussion verbatim.
3. Summary table, PV-similarity row: "RSC about 35 cm" became "RSC to 0.5 at 13 cm and to zero at 35 to 38 cm".
4. Regional paragraph ("Structures outside the hippocampus proper"): "place cells with narrow fields" replaced with the field widths, 39.6 ± 1.2 against 37.0 ± 0.8 cm.
5. Same paragraph: "No source gives field sizes in metres for ... retrosplenial cortex" replaced with the audit's text (Lenkey 2025 and Navratilova 2026), plus a pointer to the new sections.
6. Similarity section table (the optional item): the Mao row now gives 13.2 and 37.8 cm (CA1 13.4 and 33 cm); its share column reads "25% at the zero crossing" instead of "about 23%", and its status gains figure-read.

From `audit_sweep_mec.md`, the three Dasbehera sentences, replaced with the audit's suggested text: the "remaining candidate exceptions" paragraph of the second section, the "Other regions" paragraph of the similarity section, and the full citation. The links now point to arXiv v2. Dasbehera 2025 stays as neural data, with the qualifiers (cosine similarity, ad hoc groups of distance-tuned cells, peaks only, 0 to 8 m lags, no r = 0.5 width, v2 only).

### New sections

Both sit after the subiculum section and before "A third to a half of CA1 fires in any one environment".

- "Retrosplenial fields and kernels match CA1 in head-fixed mice, and one rat study finds fields 1.7 times wider", with four subsections: paired field widths (Mao 2017, Mao 2018, Lenkey 2025, Navratilova 2026 with its legend-against-table contradiction and the mouse-line split, Smith 2012 with the area rule, Subramanian 2024 sparsity); shares and fields per cell (four of five paired studies at 0.50 to 0.68, Lenkey's 1.18 exception, the mouse-line effect, Mao 1.55 against 1.26 with P = 0.21, Esteves, Navratilova); kernels (Mao RSC and CA1, subsets, the cue effect, Fischer's landmark-centred curve and why the heat maps leave it out, the rat gap); other cell types with numbers; gaps; full citations.
- "Non-grid MEC fields measure about 1.5 grid spacings in one mouse study and did not grow with grid scale", with subsections: Mallory 2018's same-recording ratio (1.55; border width 0.40; no non-grid over grid field-size ratio, because Mallory's grid field is the autocorrelogram central peak) and its four limits; the HCN1 result (spacing up 16%, non-grid field area, coverage and border width unchanged); fields per cell (Hardcastle 1.90, 1.55, 5.69); the share table with the audit's denominator caveats (Pérez-Escobar's parasubiculum sites, Kinkhabwala's grid and border overlap, Obenhaus's selected animals, Diehl 2019's possible overlap with Diehl 2017, Gardner's double counting); stability and remapping with the audit's corrections (grid patterns moved "1.3 and 2.8 cm more", the layer 3 excess for spatial change only, Campbell 2018's within-session border stability moved out of the less-stable evidence, Butler's field-size sentence not used); no dorsoventral measurement and no non-grid kernel; gaps; full citations.

### Other claims changed or qualified

- Opening summary and the similarity paragraph: rodent and bird curves fall to 0.5 within "5 to 28 cm", not "5 to 27 cm", because Fischer 2020's landmark-centred curve reaches 0.5 at 27.9 cm before the landmark. The similarity paragraph also gains a sentence on Fischer.
- Second section, retrosplenial table: now lists every RSC measurement found (Mao subsets, Mao cue and no-cue matrices, Fischer, Alexander & Nitz, Miller 2019), so the section's "every measurement found" claim still holds.
- Kernel against field width: a sentence adds Mao 2017 (CA1 0.36 of the mean and 0.43 of the median field width, RSC 0.33 and 0.38). The conclusion's "0.41 to 0.60 field widths" became "hippocampal correlation falls to 0.5 at 0.36 to 0.60 field widths", and a sentence says RSC matches CA1 in head-fixed mice.
- Field-width section, from Navratilova CA1 joining the hippocampal row: "On virtual tracks over 3 m, CA1 and CA2/3 fields stay at 1.1 to 1.5 spacings" became "0.86 to 1.5"; "they do not widen further on virtual tracks of 4 to 41 m, where they are 45 to 46.4 cm" now adds the 30.9 cm span on 3.14 to 5.03 m tracks; the summary table's "0.49 to 2.2 on virtual tracks" became "0.41 to 2.2".
- Summary table, field-width row: adds the retrosplenial range (0.33 to 1.6) and the non-grid MEC ratios (1.55; border 0.40), with Navratilova 2026, Mao 2017 and Mallory 2018 among the sources.
- Remapping section: "Non-grid medial entorhinal cells reorganized completely after shape or colour changes" became the audited displacements (non-grid fields moved a median 13.1 and 15.3 cm more than between matching boxes, grid patterns 1.3 and 2.8 cm more).
- Regional paragraph: after "whose field sizes were not reported" (Diehl 2017), a sentence notes that one mouse study reports them.
- Field-width section intro: counts for the two new sweeps. A new paragraph after the mouse module notes explains the two groups, the ratios, the Mallory exception and the dropped duplicates. The mouse source table gains the group rows and the duplicate exclusions; both source tables' "Not in any row" cells gain the width-like exclusions.
- Kernel section: sources for the Mao rows, the RSC ratio, and the new exclusions (Fischer, Mao subsets and matrices, rat curves, duplicate curves). The kernel source table gains the two Mao rows.
- Fields-per-period section: intro names the retrosplenial rows that count all imaged cells; four rows (Mao 2017, Esteves 2023, Navratilova 2026, Hardcastle 2017); a closing paragraph. The heading still starts with "Fields per cell per grid period".
- Heat-map captions and corner cell no longer say every row is hippocampal; the footer dates the new sweeps.

## Not done or left open

- No browser was available, so the page was checked structurally: the inline script parses with esprima, the 60 ids are unique, all 24 anchors resolve, tags balance, every heat-map row has its header's cell count, and the rendered page has no em dashes, stray `**` or unconverted links. Every markdown table in the review has a constant cell count.
- The Alexander & Nitz 2017 numbers rest on a WebFetch summary that the audit could not check; the review tags them "preprint; unverified".
- Mallory 2018's arena size and field rule stay unknown (paywalled Online Methods).
- `sweep_rsc.md` and `sweep_mec_nongrid.md` were not edited. `sweep_rsc.md` still carries some pre-audit figures (1.27 fields per CA1 place cell, 1.17 for superficial granular RSC, 1.22 for Esteves, the "~0.34 per metre" Navratilova estimate), and `sweep_mec_nongrid.md` still calls Dasbehera "simulation only". The review uses the audited values throughout.
- The review's theory section predates these sweeps and was left as it is; nothing from models was added.

## Appendix: generated rows and exclusions

Printed by `aggregate_sweeps.py --report`. Only the sections new in this build are copied here; the rat and mouse sections are unchanged from `page_build_sweeps.md` apart from the dorsal CA1 large-virtual row, which now reads:

| Row | N | Range | Contributions (row id: value, environment, how) |
|---|---|---|---|
| Dorsal CA1, large virtual track over 3 m, 4 studies | 4 | 30.9 to 46.4 cm | hainmueller2018_1: 45 cm, 4 m virtual linear track; lee2020_1: 46 cm, ~41 m winding virtual path, ~14 cm wide; sumegi2025_1: 46.4 cm, Virtual corridor 832 cm, checkerboard walls with six landmarks, reward; navratilova2026_4: 30.9 cm, Head-fixed visual VR on a wheel, circular tracks: small 314 cm or larg |

### Retrosplenial cortex: rows and contributions

| Row | N | Range | Contributions (row id: value, environment, how) |
|---|---|---|---|
| Dysgranular RSC, real environment up to 2 m, 3 studies | 3 | 25 to 39.6 cm | mao2017_rsc_1: 39.6 cm, Real head-fixed linear treadmill: a 150 cm belt loop with tactile cue ; mao2017_rsc_2: 34.5 cm, Real head-fixed linear treadmill: a 150 cm belt loop with tactile cue ; mao2018_1: 25 cm, Real head-fixed linear treadmill, 150 cm belt loop with tactile cues; lenkey2025_2: 37.31 cm, Real head-fixed circular treadmill |
| Dysgranular RSC, large virtual track over 3 m, 1 study | 1 | 38.1 to 40.2 cm | navratilova2026_1: 40.2 cm, Head-fixed visual VR on a wheel, circular tracks: small 314 cm or larg; navratilova2026_2: 38.1 cm, Head-fixed visual VR on a wheel, circular tracks: small 314 cm or larg |
| Granular RSC, real environment up to 2 m, 1 study | 1 | 49.5 cm | mao2017_rsc_5: 49.5 cm, Real head-fixed linear treadmill: a 150 cm belt loop with tactile cue  |
| RSC deep layers, real environment up to 2 m, 1 study | 1 | 48 cm | mao2017_rsc_4: 48 cm, Real head-fixed linear treadmill: a 150 cm belt loop with tactile cue  |

### Retrosplenial cortex: excluded rows

| Row id | Region, level | Reason | Would belong to |
|---|---|---|---|
| mao2017_rsc_3 | RSC-dysgranular, posterior, mouse | superficial dysgranular cells from 3 of the same 4 WT mice as mao2017_rsc_1 and _2, same statistic as _2; the pooled rows are used | Dysgranular RSC, real |
| mao2017_rsc_6 | RSC-dysgranular, posterior, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| mao2017_ca1_1 | CA1, dorsal, mouse | same CA1 cells as mao2017_1 in sweep_fields_mouse.csv, which the CA1 row already uses | dorsal CA1, real |
| navratilova2026_3 | RSC-dysgranular, unspecified, mouse | the cells of navratilova2026_1 split by mouse line (Thy1 29 cm, CaMK2a-tTA 38 cm); the pooled rows are used and the text gives the split | Dysgranular RSC, large |
| campagner2023_1 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| smith2012_1 | RSC-granular, anterior, rat | rat; plus-maze size not stated, so no environment class, and an equal-area diameter on narrow arms understates the length along an arm (as diamantaki2016 cell 104) | no row |
| smith2012_2 | CA1, dorsal, rat | rat CA1 comparison row for smith2012_1; plus-maze size not stated, so no environment class | no row |
| smith2012_3 | RSC-granular, anterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| subramanian2024_1 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| subramanian2024_2 | CA1, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| subramanian2024_3 | RSC-granular, unspecified, rat | sparsity index (the share of the box where a cell fires), not a field width | no row |
| miller2021_1 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| miller2019_1 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| vedder2017_1 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| laurent2026_1 | RSC-dysgranular, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| laurent2026_2 | hippocampus (dorsal; CA1 not named), dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2017_1 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2017_2 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| chen2026_1 | RSC-unspecified, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| chen2026_2 | RSC-dysgranular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| chen2026_3 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| mao2018_2 | RSC-dysgranular, unspecified, mouse | after bilateral hippocampal lesions, a manipulation; the sham row mao2018_1 is used | Dysgranular RSC, real |
| fischer2020_1 | RSC-dysgranular, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| lenkey2025_3 | CA1, dorsal, mouse | same CA1 cells as lenkey2025_1 in sweep_fields_mouse.csv, which the CA1 row already uses | dorsal CA1, real |
| esteves2023_1 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| esteves2023_2 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| esteves2023_3 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| esteves2023_4 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| park2024_1 | RSC-granular, unspecified, mouse | vertex-cell fields at polygon corners, not place-like fields; area approximate (about 80 cm²) in a 25 cm radius chamber | no row |
| park2024_2 | RSC-granular, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| sun2021_1 | RSC-dysgranular, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| chen1994_1 | RSC-dysgranular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| chen1994_2 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| jacob2017_1 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| jacob2017_2 | RSC-dysgranular, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| lozano2017_1 | RSC-unspecified, posterior, rat | head-direction tuning width in degrees, not a spatial width | no row |
| lozano2017_2 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| zhang2022_1 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| zhang2022_2 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| zhang2022_3 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| zhang2022_4 | RSC-unspecified, posterior, rat | head-direction tuning width in degrees, definition not stated | no row |
| alexander2020_1 | RSC-unspecified, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2020_2 | RSC-dysgranular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2020_3 | RSC-granular, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2020_4 | RSC-unspecified, unspecified, rat | egocentric boundary cells' preferred distance to a wall, not a field width; the histogram's distance axis is uncertain | no row |
| alexander2020_5 | RSC-unspecified, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2020_6 | RSC-unspecified, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| vanwijngaarden2020_1 | RSC-unspecified, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| vanwijngaarden2020_2 | RSC-unspecified, unspecified, rat | border cells' preferred distance from the wall (peak at 5 cm), not a field width | no row |
| vanwijngaarden2020_3 | RSC-unspecified, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| lachance2024_1 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| vandergoes2024_1 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| vandergoes2024_2 | RSC-unspecified, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| sit2023_1 | RSC-dysgranular, posterior, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| sit2023_2 | RSC-dysgranular, posterior, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| sato2010_1 | medial parietal region (area 7m, RSC and posterior cingulate, unspecified, macaque | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| sato2006_1 | medial parietal region (area 7m, RSC and posterior cingulate, unspecified, macaque | no width in cm (counts, fractions, ratios, angles or stability only) | no row |

### Retrosplenial cortex: CA1 rows passed to the hippocampal mouse rows

| Row id | Width | Joins |
|---|---|---|
| navratilova2026_4 | 30.9 cm | Dorsal CA1, by the mouse class rules |

### MEC, non-grid cells: rows and contributions

| Row | N | Range | Contributions (row id: value, environment, how) |
|---|---|---|---|
| Dorsal MEC non-grid spatial cells, real open arena of unstated size, 1 study | 1 | 56.1 cm | MNG01: 56.1 cm, 'open arenas' |
| Dorsal MEC border cells, width perpendicular to the wall, real open arena of unstated size, 1 study | 1 | 14.4 cm | MNG04: 14.4 cm, 'open arenas' |

### MEC, non-grid cells: excluded rows

| Row id | Region, level | Reason | Would belong to |
|---|---|---|---|
| MNG02 | MEC, dorsal, mouse | share of the box covered by the cell's fields (43%), not a field width | no row |
| MNG03 | MEC, dorsal, mouse | HCN1 knockout mice, a genetic model; the wild-type littermates (MNG01) are used | Dorsal MEC non-grid spatial cells, unstated |
| MNG05 | MEC, dorsal, mouse | grid cells, and the size is the central peak of the spatial autocorrelogram, not a rate-map field | no row |
| MNG06 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG07 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG08 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG09 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG10 | MEC, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG11 | MEC, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG12 | MEC, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG13 | MEC, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG14 | MEC, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG15 | MEC and surrounding cortex, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG16 | MEC (and parasubiculum in pooled numbers), dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG17 | MEC (and parasubiculum in pooled numbers), dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG18 | MEC (and parasubiculum in pooled numbers), dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG19 | MEC layer 2, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG20 | MEC layer 3, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG21 | MEC and parasubiculum: of 49 recording sites in this 880-cel, dorsal, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG22 | MEC (track cohort includes parasubiculum sites), dorsal, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG23 | MEC (a few parasubiculum cells), dorsal, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG24 | MEC, unspecified, mouse | share of a virtual track inside cue-cell fields under a lenient p < 0.3 rule, not a width | no row |
| MNG25 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG26 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG27 | MEC and parasubiculum, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG28 | MEC, unspecified, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG29 | MEC, unspecified, mouse | distance from the wall to the field centre (8.2 cm), not a width; unclear whether it covers 6 or 20 cells | no row |
| MNG30 | MEC and parasubiculum, layers II and III, dorsal, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG31 | MEC, unspecified, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| MNG32 | MEC layer 2, dorsal, mouse | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
