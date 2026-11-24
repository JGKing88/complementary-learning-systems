# Audit of the non-grid MEC sweep

Audit of 2026-11-23. It covers `sweep_fields_mec_nongrid.csv` (32 rows), `sweep_kernel_mec.csv` (3 rows), row K26 of `sweep_kernel.csv`, and the text of `sweep_mec_nongrid.md`. Corrections are in `audit_sweep_mec_corrections.csv` (29 cell changes in 11 rows, no drops). I did not edit the sweep files or the review.

## Summary

- Rows checked: 36 (32 field rows, 3 kernel rows, K26).
- Confirmed: 25. Corrected: 11 (MNG01, MNG05, MNG13, MNG14, MNG15, MNG21, MNG24, MNG27, MNG29, MNG30, K26). Unverifiable: 0. Dropped: 0.
- Every number the brief listed matches its paper. Every derived number recomputes (see the arithmetic below).
- The largest error is in the sweep's Dasbehera verdict. Version 2 of arXiv 2511.08292 (stamped 4 July 2026) reanalyses real Neuropixels recordings from Campbell et al. 2021. K26 should stay, with corrections. The review's three mentions are accurate in substance but need qualifiers.
- The second error is a method mismatch in Mallory 2018. The grid "field size" of 489 cm2 comes from the autocorrelogram's central peak, so the claim that a non-grid field is "about 5 times" a grid field compares two different measures. The headline ratio (non-grid diameter 56.1 cm over grid spacing 36.26 cm = 1.55) is unaffected.

## How I checked

- I read each row's source text myself from the sweep's downloads (BioC text of the PMC manuscripts, the Mallory and Hoydal Springer ESM PDFs, the Tang 2014 supplement, PubMed abstracts).
- I fetched the PMC open-data XML for Mallory (PMC5823610) to look for the Online Methods. They are not in it.
- I re-extracted Brun 2008 from the PDF with pypdf and rendered Fig. 5 with PyMuPDF to check the width convention.
- I downloaded arXiv 2511.08292 v1 and v2, extracted both with pypdf, and rendered Fig. 2 of v2.
- I recounted the Gardner 2022 source data with my own parser (`audit_sweep_mec/scripts/gardner_counts.py`), which reads cell references.
- Scripts are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/audit_sweep_mec/scripts/`. New downloads are in `.../audit_sweep_mec/dl/`.

## Recomputed numbers

- Mallory equal-area diameters: 2*sqrt(2473.45/pi) = 56.12 cm; 2*sqrt(2692.29/pi) = 58.55 cm; 2*sqrt(489.29/pi) = 24.96 cm. Plus and minus 1 SD in area: 2*sqrt(240.25/pi) = 17.49 cm and 2*sqrt(4706.65/pi) = 77.41 cm.
- Mallory ratios: 56.1/36.26 = 1.547; 58.5/42.16 = 1.388; 14.43/36.26 = 0.398; 14.23/42.16 = 0.338; 25.0/36.26 = 0.689; 2473.45/489.29 = 5.06; 42.16/36.26 = 1.163 (a 16% rise).
- Shares: 32/527 = 6.07%; 37/527 = 7.02%; 307/794 = 38.66%; 101/794 = 12.72%; 42/794 = 5.29%; 0.675 x 0.95 = 0.641; 81/239 = 33.9%; 226/880 = 25.7%; 139/880 = 15.8%; 63/880 = 7.2%; 30/193 = 15.5%; 22/193 = 11.4%; 19/69 = 27.5%; 67/492 = 13.6%; 160/1136 = 14.1%; 151/1136 = 13.3%; 294/925 = 31.8%.
- Hardcastle sample sizes: t(406) for grid against P\{G,B} gives 101 + 307 = 408 cells; t(347) for border against P\{G,B} gives 42 + 307 = 349. So n = 307 is right.
- Gardner recount: Q 163/2094 = 7.8%; R day 1 483/2460 = 19.6%; R day 2 544/2571 = 21.2%; S 140/546 = 25.6%; total 7,671. Module medians Q1 71.3, Q2 102.0, R1a 60.1, R1b 62.6, R1 (day 2) 56.8, R2 89.4 and 84.4, R3 127.0 and 113.2, S1 59.5 cm. Non-grid medians 70.2, 94.5, 57.2 and 56.2 cm. All match the sweep.
- Brun ratios: 32/171 = 0.19; 47/301 = 0.16; 67/370 = 0.18; 56/171 = 0.33; 92/301 = 0.31; 119/370 = 0.32.

## Row verdicts

### sweep_fields_mec_nongrid.csv

- MNG01, Mallory 2018 non-grid field area. CORRECTED (notes only). Table 1: "mean field size 2473.45 ± 2233.2 cm2 2692.29 ± 2768.8 cm2 WRS Z = 0.12, p = 0.90", n = 32 and 38, table note "Mean ± SD". The area is per cell: Supp. Fig. 8e says "The size of the average field is similar between spatially stable cells". "Spatially stable" is a shuffle class: Supp. Fig. 6 sets each class at the 95th percentile of 100 shuffles per cell, and the Results define the row's cells as "spatially stable cells not classified as grid or border cells". Table 1 gives their spatial stability as 0.37 ± 0.15. The exact stability score and the field rule are in the paywalled Online Methods (not in the PMC XML either). Species, dorsal level (grid cells 386 ± 222 um from the dorsal border, Supp. Fig. 7d), 12 iWT mice and 527 iWT cells are right. The note's "about 5 times the grid field area" is corrected; see MNG05.
- MNG02, Mallory coverage. CONFIRMED. Table 1: "% box covered by fields 43.24 ± 30.72 % 45.58 ± 31.62 %", p = 0.88; Supp. Fig. 8f.
- MNG03, Mallory HCN1 knockout. CONFIRMED. Table 1 value and p = 0.90 match. Spacing "iCre-KO = 42.16 ± 10.53 cm, Z = -3.09, p = 0.0020", 62 grid cells, 17 mice. Diameter 58.5 cm and ratio 1.39 recompute.
- MNG04, Mallory border width. CONFIRMED. Table 1: "border width 14.43 ± 6.53 cm 14.23 ± 5.22 cm WRS Z = -0.30, p = 0.76", n = 86 and 71. Supp. Fig. 8b defines it as the mean distance from the wall along each field's inner edge, averaged over the cell's fields, so per cell and perpendicular to the wall. The model text "The boundary width was set to 14 cm, the average width of all recorded border cells" is in the Fig. 2 legend.
- MNG05, Mallory grid field size. CORRECTED (measure, notes). The value "iWT = 489.29 ± 301.62 cm2, iCre-KO = 502.64 ± 207.26 cm2" is right, but the Fig. 1 legend says "Grid field size was determined from the radius of the circle around the center field of the autocorrelation map." The 25.0 cm diameter is the autocorrelogram central peak, not a rate-map field, so it should not be set against the rate-map areas of MNG01 to MNG03.
- MNG06, Hardcastle 2017 non-grid fields per cell. CONFIRMED. Fig. 3D legend: "G = 5.69 ± 0.35, B = 1.55 ± 0.23, P\{G,B} = 1.90 ± 0.10", mean ± SEM; n = 307 from the Fig. 3B legend and the t-test df. Field rule (STAR Methods): "connected regions containing at least 5 bins with a firing rate above 1 Hz and at least 20% of the peak firing rate". The 2 x 2 cm bin is stated for the "spatial map" in the grid-score section, which is the same map, so the 20 cm2 minimum is a fair inference. Boxes "228/275 sessions in 100 × 100 cm boxes, 29/275 ... 91.4 × 91.4 cm ... 18/275 ... 70 × 70 cm" match. Layer II/III, 14 mice.
- MNG07, Hardcastle border fields. CONFIRMED. B = 1.55 ± 0.23, n = 42 (border cells above the 99th shuffle percentile).
- MNG08, Hardcastle grid fields. CONFIRMED. G = 5.69 ± 0.35, n = 101. Ratio 1.90/5.69 = 0.33.
- MNG09, Hardcastle share. CONFIRMED. "307/421 P cells ... were not classified as G or B"; 794 cells. Denominator is all recorded cells.
- MNG10, Diehl 2017 share. CONFIRMED. "Of the spatial cells, 17.7 % had significant grid scores and 9.3 % had significant border scores. The remaining 67.5 % ..."; "~95 %" of 345 cells in 7 rats were spatial. About 64% of all cells is derived and labelled so. Note that 17.7 + 9.3 + 67.5 = 94.5%, so the classes do not quite fill the spatial set.
- MNG11, Diehl 2017 stability. CONFIRMED. Non-grid "median displacement: 4.4 cm and 4.7 cm ... median spatial correlation: 0.62 and 0.62"; grid "0.85 and 0.70". The 236 non-grid and 48 grid cells come from the Fig. 4 legend ("Across all sessions"), not from the stability test itself.
- MNG12, Diehl 2017 remapping. CONFIRMED. Non-grid increases 13.1, 15.3 and 28.9 cm; correlations 0.29 and 0.26. Grid increases 1.3, 2.8 and 20.9 cm (over matching-box displacements of 2.4 and 2.6 cm); correlations 0.72 and 0.42; rate vectors 0.40 and 0.13 against 0.50 and 0.43.
- MNG13, Diehl 2019 spatial share by box size. CORRECTED (n_animals, notes). "about 90 % ... small environment ... ~55 % ... large environment" matches. The Methods give "eight male Long Evans rats", 3 in the small boxes and 5 in the 120 cm box. They also say "Data from three rats (45 out of 313 cells) have been previously reported in a different study". The design of those three rats matches Diehl 2017 rats A to C, so the sweep's statement that Diehl 2019 is a separate cohort is not safe.
- MNG14, Diehl 2019 drift. CORRECTED (n_animals, notes). "16 % of non-grid cells exhibited significant changes ... in their spatial firing pattern and that 16 % exhibited changes ... in their overall firing rate"; 159 of 313 tracked; class percentages match. The layer 3 excess holds for spatial change (Chi-Sq = 5.88, p < 0.05) but not for rate change (Chi-Sq = 0.10).
- MNG15, Butler 2019 remapping. CORRECTED (n_animals, notes). "196/271 non-grid spatial cells significantly remapped ... task-trained ... 0.32 ± 0.22; control: 0.41 ± 0.27; 49/100 control cells re-mapped" matches, and 1.5 x 1.5 m arenas and 102 grid cells match. The 271 cells are task-trained; the 100 control cells are separate; the text does not say whether the 3 control rats are among the "seven rats". The field-size sentence says fields did not extend "in a reward-specific manner", not that they did not change between environments.
- MNG16, Krupic 2012 periodic non-grid share. CONFIRMED. "156/239, 65% in mEC ... (75/156 or 48%)"; 81/239 = 34% derived. Denominator is mEC cells only. The main text calls the box "1.3m2" while the Fig. 1 legend says "1.3×1.3m2"; the row's 1.69 m2 follows the legend.
- MNG17, Krupic stability. CONFIRMED. Fig. 2A legend: "GC:r=0.54±0.02 ... nonGC: r=0.45±0.02, nonSPC: r=0.21±0.03" and across days 0.53, 0.44, 0.18.
- MNG18, Krupic class switching. CONFIRMED. 11% within an environment, 32% between the square and the 1 m circle.
- MNG19, Tang 2014 layer 2 share. CONFIRMED. "33%; n = 63 cells"; grid 19/99 and 3/94; border 10/94 and 1/99; 193 cells; arenas 70 x 70 cm or 1 x 1 m (supplement); dorsalmost MEC. The derived 30/193 = 15.5% assumes grid and border do not overlap and are subsets of the 63, which the classification order supports.
- MNG20, Tang 2015 layer 3 share. CONFIRMED (abstract). "All of the ~28% of neurons that carried significant amounts of spatial information (19 of 69) discharged in irregular spatial patterns." Status "abstract" is honest.
- MNG21, Perez-Escobar 2016 share. CORRECTED (region). Counts 139, 226, 63, 85 of 880 in 8 mice match. But this arena cohort's recording sites were 37/49 MEC, 6/49 at the MEC/parasubiculum transition and 6/49 parasubiculum, so the 25.7% share is not MEC-only.
- MNG22, Perez-Escobar track cohort. CONFIRMED. "We recorded 138 grid cells, 28 border cells and 133 irregular spatially selective cells on the track"; 479 neurons, 5 mice; 80 x 5.4 cm track; site counts 14, 4 and 13.
- MNG23, Miao 2017 aperiodic cells. CONFIRMED. "A total of 214 cells satisfied these criteria, 160 in PV mice (152 in MEC and 8 in parasubiculum) and 54 in SOM mice (52 in MEC and 2 in parasubiculum)"; thresholds 0.43 and 0.48 bits per spike; grid 133 and 115; 1 m box.
- MNG24, Kinkhabwala 2020 cue cells. CORRECTED (cell_type). "~13% of all recorded cells" (789 units, 3 mice), field fractions 0.4 ± 0.1 and 0.2 ± 0.1 (N = 100 cue cells), field rule "p-value was less than 0.3" all match. But "a small percentage of cue cells were conjunctive with border (11%) or grid (28%) cell types", so cue cells are not a purely non-grid class. Non-grid, non-border cue cells are about 13% x (1 - 0.28 - 0.11) = 8% of units if the two overlaps do not coincide (derived).
- MNG25, Campbell 2018. CONFIRMED. "grid cell n = 151/1136, border n = 160/1136"; "grid = 0.40 ± 0.02, border = 0.54 ± 0.02"; 90 x 90 cm box and 400 cm track.
- MNG26, Low 2021. CONFIRMED. Fig. 5B legend "grid cells, 18.26 ± 3.81%; border cells, 13.23 ± 2.07%; spatially stable cells, 24.57 ± 3.12%"; Fig. 5E medians 25.5, 20.6 and 18.5%; 1,379 cells in gain sessions (224 grid and 152 border, so 1,379 is all cells in those sessions); 10,410 in total.
- MNG27, Obenhaus 2022 object-vector share. CORRECTED (statistic). "17.7 ± 7.8% for OV cells (mean ± SD)" in "OV animals"; 932 of 4,399 cells in 34 sessions; 12 of 15 animals. The text does not say whether the SD is over animals or sessions. The OV animals were picked because their OV fraction exceeded their grid fraction, so 17.7% is a selected-animal share, as the sweep's md says.
- MNG28, Andersson 2021. CONFIRMED. "67 out of 492 cells (13.6%)"; per-mouse range 5.7% to 16.7%; 14.7% for the earlier study is secondary, as tagged; 7 mice; 80 x 80 cm arena.
- MNG29, Hoydal 2019 supplement. CORRECTED (value_reported). "Twenty out of 860 recorded cells passed the criterion for boundary-vector cells ... Among these, 14 also passed the criteria for border cells, leaving only 6 cells"; "8.2 cm ± 0.7" (mean ± SEM); criterion of a long axis at least 70% of a wall within 10 degrees; square boxes only. Whether the 8.2 cm covers the 6 cells or all 20 is ambiguous in the text. Aspect ratio 1.6 and map correlation 0.76 ± 0.008 match.
- MNG30, Gardner 2022 source data. CORRECTED (n_cells). Counts and module medians match my recount. Rat R day 1 and day 2 are the same rat with the same modules (R1 to R3) on two days, so the 7,671 units likely count some neurons twice. The "non-grid" cluster holds every unit outside the modules and outside the "noise" cluster, so it includes non-spatial units and parasubiculum, as the row says.
- MNG31, Savelli 2008. CONFIRMED (abstract). "70 out of 93 cells, including 15 of 16 putative interneurons"; 4 rats; dorsal and ventral MEC. The verified_in file `abs1.txt` holds ten abstracts; Savelli is entry 5.
- MNG32, Gu 2018. CONFIRMED. "(294 grid and 631 non-grid cells in 18 FOVs of six mice)"; "7.3±1.3% of cells originally classified as grid cells"; 10 or 18 m tracks; layer 2. The 925 cells are labelled stellate and pyramidal cells (517 + 408).

### sweep_kernel_mec.csv

- MK01, Brun 2008 dorsal. CONFIRMED. PDF p. 1207: "The mean half-width was 32 cm in the dorsal group, 47 cm in the intermediate group, and 67 cm in the ventral group. The mean width at r = 0.20 ... was 54 cm (dorsal), 102 cm (intermediate), and 184 cm (ventral)." 55 cells from 9 rats. Fig. 5 (p. 1208), which I rendered, draws the r = 0.5 and r = 0.2 lines across both sides of the peak, so the widths are full widths and dist_r05 = 16 cm is right. Field width 56 ± 1 cm, largest 90 ± 4 cm, median spacing 171 ± 13 cm, minimum 91 ± 12 cm match. 5 cm bins, 5 cm smoothing, middle 17 m. "Repetitive bands were not observed" matches.
- MK02, Brun intermediate. CONFIRMED. 47 and 102 cm; 59 cells, 6 rats; field 92 ± 6 cm; spacing 301 ± 23 cm.
- MK03, Brun ventral. CONFIRMED. 67 and 184 cm; 29 cells, 4 rats; field 119 ± 7 cm; spacing 370 ± 46 cm; "Four rats contributed cells to more than one group."

### sweep_kernel.csv

- K26, Dasbehera, Dogra & Redman. CORRECTED, not dropped. See the next section.

## Dasbehera 2025 (arXiv 2511.08292)

### Verdict

The sweep is wrong. Version 2 of the preprint (29 pages, stamped 4 July 2026; the arXiv page lists v1 and v2) contains a real-neuron reanalysis. Section 2.1 is titled "Grid cell correlation decays with distance in experimentally recorded MEC data". Section 4.1 says: "We analyze the Neuropixel MEC recordings from Campbell et al. (2021) using the publicly available dataset and code." The analysis uses distance-tuned MEC cells recorded while mice ran in the dark. Fig. 2 shows two recordings (IDs npI5_0417_dark_1 and npI5_0414_dark_1, with 44 and 60 distance-tuned cells). The authors group cells ad hoc by similar autocorrelogram peak spacing, compute cosine similarity between population vectors (Eq. 16), and plot only the peaks. The quote is "This relationship is noisy and not truly monotonic, but there is a general trend towards decreasing population vector correlation with increasing distance."

Version 1 (14 pages, 12 November 2025) has no such analysis; its only real data are the Tennant et al. 2018 behavioural stopping errors. The sweep's abstract file is the v2 abstract page, but the abstract itself does not mention the reanalysis. The sweep's verdict fits v1 and the abstract, not the current full text.

So this is neural data and can stay in a neural-data-only review. It is a small preliminary analysis in a mostly theoretical paper, and the review should say what was measured:
- The distance axis runs 0 to 8 m, not over the hundred-plus metres the mice ran in a session.
- The measure is cosine similarity of non-negative rates, which stays high. Figure-read peaks fall from 1 to about 0.75 and 0.83 (recording 1, two cell groups) and about 0.64 (recording 2) by 7 to 8 m, with a dip to about 0.69 at 2.7 m in recording 2. No r = 0.5 width exists.
- The cells are distance-tuned cells that the preprint treats as grid cells. Campbell 2021 found 8/24 grid cells and 5/134 non-grid cells distance tuned.
- The two recording IDs suggest one mouse.
- Grid-module identity was not available.

K26 should stay in `sweep_kernel.csv` with the corrections in the CSV. The sweep md's "K26 is not a neural kernel and should leave the kernel table" and its table entry "Dasbehera 2025 | decorrelation against distance | simulation only" should not be applied.

### Review sentences that cite it

All three are in `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`. None needs removal. Each needs qualifiers.

1. Line 156. Current: "A reanalysis of mouse medial entorhinal Neuropixels data from a virtual track run over a hundred metres found a general decreasing trend that was "not truly monotonic" [preprint] ([Dasbehera, Dogra & Redman 2025](https://arxiv.org/html/2511.08292))."
   Suggested: "A preprint reanalysed public Neuropixels recordings from Campbell et al. (2021), in which mice ran over a hundred metres in darkness on a virtual-reality treadmill. In two recordings of 44 and 60 distance-tuned cells, grouped ad hoc by peak spacing, the peaks of the population-vector cosine similarity fell over lags of 0 to 8 m, but the trend was "noisy and not truly monotonic" [preprint, v2; values figure-read] ([Dasbehera, Dogra & Redman 2025](https://arxiv.org/abs/2511.08292v2))."
2. Line 557. Current: "A reanalysis of mouse medial entorhinal Neuropixels data on a virtual track run over a hundred metres found a decreasing trend that was "not truly monotonic", without grid-module identity [preprint] ([Dasbehera, Dogra & Redman 2025](https://arxiv.org/html/2511.08292))."
   Suggested: "A preprint reanalysed mouse medial entorhinal Neuropixels data from dark treadmill runs (Campbell et al. 2021). It used cosine similarity, not Pearson correlation, for ad hoc groups of distance-tuned cells, and plotted only the peaks of the curve. The peaks fell from 1 to about 0.64 to 0.83 by 7 to 8 m, a trend the authors call "noisy and not truly monotonic". Grid-module identity was not available, and the curve gives no r = 0.5 width [preprint, v2; values figure-read] ([Dasbehera, Dogra & Redman 2025](https://arxiv.org/abs/2511.08292v2))."
3. Line 565 (full citations). Current: "Dasbehera, Dogra & Redman (2025). arXiv 2511.08292 [preprint]."
   Suggested: "Dasbehera P, Dogra AS, Redman WT (2025). Distance by de-correlation: computing distance with heterogeneous grid cells. arXiv 2511.08292, v2 of 4 July 2026; the reanalysis of Campbell et al. 2021 is in v2 only [preprint, full text]."

## Problems in the sweep's md text

These are sentences in `sweep_mec_nongrid.md`. They have no CSV cell, so they are not in the corrections file.

1. Field size section: "A non-grid field therefore has about 5 times the area of a grid field here." The grid figure is the autocorrelogram central peak (Mallory Fig. 1 legend). Drop the sentence or state the method difference.
2. Field size section, Hoydal: "For the other 6, the field centre sat 8.2 ± 0.7 cm from the wall." The text does not say whether the mean covers the 6 cells or all 20 boundary-vector cells.
3. Method section: "Diehl 2019 (MNG13, MNG14) is a separate cohort from Diehl 2017." Diehl 2019 says 45 of its 313 cells, from 3 rats, were reported before, and the design matches Diehl 2017 rats A to C.
4. Field size section: "Butler 2019 states that non-grid field size, total field area and field number did not change between environments". The sentence is about reward-specific extension of fields; the compared conditions are not stated.
5. Share section: "Cue cells are about 13% of units on cue-rich VR tracks". Add that 28% of cue cells were also grid cells and 11% border cells.
6. Stability section, first bullet: it lists "VR half-session stability 0.54 for border cells against 0.40 for grid cells (Campbell 2018)" as evidence that non-grid cells are less stable than grid cells. That number goes the other way (border cells more stable), and it is within one session, not across sessions. Move it out of that bullet.
7. Stability section, second bullet: "grid patterns moved by 1.3 and 2.8 cm". These are increases over matching boxes, whose displacements were 2.4 and 2.6 cm. Say "moved 1.3 and 2.8 cm more".
8. Stability section, fourth bullet: "Changing cells were more common in layer 3." True for spatial change only, not rate change.
9. Kernels section and the non-kernel table: the Dasbehera paragraph and "simulation only" are wrong for v2 (see above).
10. Share table: the Perez-Escobar 25.7% denominator includes parasubiculum sites (12% of sites) and transition sites (12%).

## Double counting

- Mallory MNG01 to MNG05 come from the same mice. MNG01 and MNG02 are the same 32 cells. MNG03 is the knockout group. MNG04 and MNG05 are other cells from the same recordings. Flagged by the sweep.
- Hardcastle MNG06 to MNG09 are one pool of 794 cells. Flagged.
- Diehl 2017 MNG10 to MNG12 share 345 cells. Flagged.
- Diehl 2019 MNG13 and MNG14 may share 45 cells with Diehl 2017 MNG10 to MNG12. Not flagged by the sweep; flagged now in the corrections.
- Krupic MNG16 to MNG18 share cells. Not stated in the sweep md but obvious from the rows.
- Perez-Escobar MNG21 and MNG22 are separate cohorts (8 mice, then a "new cohort" of 5). No overlap.
- Gardner MNG30: rat R day 1 and day 2 record the same modules; units are probably counted twice. Flagged now.
- MK01 to MK03 are the same data as `sweep_kernel.csv` K04 to K06. If both tables feed the heat maps, use one.
- Kinkhabwala 2020 imaging and Gu 2018 both used 10 and 18 m Tank-lab VR tracks. The Kinkhabwala share row uses tetrode data, so the two rows do not share cells as far as the text shows.

## Share denominators

| Study | Share | Denominator | Verdict |
|---|---|---|---|
| Mallory 2018 | 32/527 = 6.1% | all iWT MEC cells | correct; strict stability class |
| Tang 2014 | 30/193 = 16% | all layer 2 cells, identified plus theta-assigned | correct; derived |
| Perez-Escobar 2016 | 226/880 = 25.7% | all cells in the arena cohort | includes parasubiculum and transition sites |
| Tang 2015 | 19/69 = 28% | all layer 3 cells | correct; abstract |
| Krupic 2012 | 81/239 = 34% | mEC cells (PaS excluded) | correct; "periodic" is a Fourier criterion |
| Hardcastle 2017 | 307/794 = 38.7% | all cells | correct; LN-model position coding |
| Diehl 2017 | 0.675 x 0.95 = 64% | all 345 cells | correct as derived; classes sum to 94.5% of spatial cells |
| Andersson 2021 | 67/492 = 13.6% | all cells, 7 mice | correct |
| Obenhaus 2022 | 17.7 ± 7.8% | all imaged cells in selected "OV animals" | correct value; averaging unit not stated |
| Kinkhabwala 2020 | about 13% | all 789 units | correct value; 39% of cue cells are also grid or border |

## Fields I could not verify

- Mallory 2018 arena size, field rule and the exact stability score: in the Online Methods, which are not in the PMC manuscript, the PMC XML or the ESM. The Nature page redirects to a login.
- Butler 2019 Fig. S10A and Table S1; Diehl 2017 Table S1; Krupic 2012 SOM; Hoydal 2019 main text. The sweep's download attempts for these returned HTML error pages, which I confirmed with `file`.
- These gaps do not change any number in the rows; the rows already mark them as not opened.
