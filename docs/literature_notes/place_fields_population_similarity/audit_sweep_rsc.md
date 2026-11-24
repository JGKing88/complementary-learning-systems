# Audit of the retrosplenial cortex sweep

Audit date: 2026-11-23. Files audited: `sweep_fields_rsc.csv` (66 rows), `sweep_kernel_rsc.csv` (11 rows) and `sweep_rsc.md`. Corrections are in `audit_sweep_rsc_corrections.csv` (28 cell changes in 18 rows). I did not edit the sweep files or the review.

## Result

| File | Rows | Confirmed | Corrected | Unverifiable | Dropped |
|---|---|---|---|---|---|
| sweep_fields_rsc.csv | 66 | 50 | 14 | 2 | 0 |
| sweep_kernel_rsc.csv | 11 | 8 | 2 | 1 | 0 |

The two unverifiable field rows (alexander2017_1 and _2) also get a status change in the corrections file. Both corrected kernel rows change only in their notes. The K-mao2017_1 note made a false symmetry claim, and the K-mao2017_2 note gains a duplicate flag. Their numbers are confirmed.

The most consequential findings:

1. All the key Mao 2017 numbers hold. I re-read Fig. 3b, 3c, 3d, 4c and 4d from the PDF vector paths, Fig. 2c by colour-bar lookup and Fig. 4b by pixel trace. The RSC kernel falls to 0.5 at 13.25 cm and crosses zero at 37.77 cm. The CA1 kernel gives 13.39 and 33.14 cm.
2. The review must change in five places (listed below). It says the "~35 cm" figure was "not confirmed verbatim", but the Discussion says it verbatim. It gives the CA1 zero crossing as "about 34 cm", but the curve reads 33.1 cm. It also says that no source gives RSC field sizes in metres.
3. `sweep_rsc.md` says the RSC/CA1 place-cell share ratio is 0.5 to 0.7 "in every paired study". That is false. Lenkey 2025 found 20.5% in RSC against 17.4% in CA1 (7 mice, 6196 neurons), a ratio of 1.18.
4. `sweep_rsc.md` says RSC place cells have "slightly more fields" (1.55 against 1.27). The CA1 mean is 1.26, and Mao 2017 reports no difference (chi-square P = 0.21).
5. Two sets of cells are counted twice across files. K-mao2017_1 and _2 are the same curves as K16 and K17 in `sweep_kernel.csv`. mao2017_ca1_1 and lenkey2025_3 are the same CA1 cells as mao2017_1 and lenkey2025_1 in `sweep_fields_mouse.csv`. The Mao 2017 subregion rows (_3 to _5) certainly overlap the pooled rows (_1, _2), because the 3 WT subregion mice are among the 4 WT mice.
6. The Alexander & Nitz 2017 numbers (47.2%, 1.9 against 5.2 cm and the others) rest on a WebFetch summary. The sweep saved no text: its bioRxiv files are 17-byte error pages. bioRxiv returned HTTP 429 and Europe PMC HTTP 500 to me, so these numbers stay unverified.

## Method

- Text was read from the sweep's saved files: Europe PMC XML, BioC text, journal PDFs, the Navratilova supplement .docx text and the eLife reviewed-preprint text. New downloads went to `/home/jackking/.claude/jobs/2c8b1c8e/tmp/audit_sweep_rsc/dl/`: PubMed abstracts for Chen 1994 and Sato 2006 and 2010, plus failed attempts for Alexander & Nitz 2017.
- Vector figures were read with PyMuPDF drawing commands and calibrated on the tick segments. Scripts are in `.../audit_sweep_rsc/scripts/` (`vcurve.py`, `vcdf.py`, `vdump.py`, `vlines.py`, `matk.py`, `trace.py`).
- Raster figures were read by pixel, calibrated on detected tick rows. The Mao 2017 Fig. 2c matrices were read by nearest colour-bar lookup, independently of the sweep's script.
- Derived numbers were recomputed by hand: equal-area diameters, per-metre values, ratios and shares.

## Field rows

Verdicts: C = confirmed, X = corrected (see the corrections file), U = unverifiable.

| Row | Verdict | Evidence |
|---|---|---|
| mao2017_rsc_1 | C | Results quote "39.6 ± 1.2 cm; CA1: 37.0 ± 0.8 cm ... P = 0.57" found. 297/2256 = 13.2%. Fig. 3c bars from vector rects: 0.555, 0.337, 0.106, mean 1.546 (1.550 normalised). SD 1.2 x sqrt(297) = 20.7 cm. Level posterior fits the injection 1 mm from the transverse sinus. |
| mao2017_rsc_2 | C | Fig. 3b step path: RSC quartiles 24.70, 34.49, 46.41 cm. CA1 24.70, 31.48, 44.96 cm. Ratio of medians 34.5/31.5 = 1.10. |
| mao2017_rsc_3 | X | Fig. 4c quartiles 30.74, 42.00, 58.49 cm; Fig. 4d bars 0.569, 0.374, 0.057 (mean 1.49); 123/861 and 15 ± 3% quoted. Notes corrected: the overlap with mao2017_rsc_1 is certain, since seven mice were used for RSC imaging (4 WT and 3 Thy1). |
| mao2017_rsc_4 | C | Quartiles 34.49, 47.99, 58.49 cm; bars 0.600, 0.350, 0.050 (mean 1.45); 60/1237 and 6 ± 2%. Region unspecified is right ("deep agranular and granular"). |
| mao2017_rsc_5 | X | Quartiles 38.99, 49.49, 61.49 cm confirmed. Bars 0.759 and 0.207 sum to 0.966, with no third bar. The mean is 1.21 once normalised; 1.17 counts the missing 3.4% as zero-field place cells. |
| mao2017_rsc_6 | C | "56%, 1416/2544, n = 3 mice, eight sessions" found. |
| mao2017_ca1_1 | X | 37.0 ± 0.8 cm, 452/1758 = 26%, n = 611 for counts all found. Bars 0.752, 0.223, 0.020 give 1.26, not 1.27. Duplicate of mao2017_1 in `sweep_fields_mouse.csv` (flagged in its notes). |
| navratilova2026_1 | C | Fig. 5E bars re-read by pixel: 40.4 cm to the top of the outline, about 40.2 at its centre. Ticks every 100.75 px per 20 cm. "19 % bigger" and the legend sentence "There was no significant difference between RSC and CA1 place field sizes" both found. Table 1 row 5E rejects equality, F(20,1) = 6.2, p = 0.022. The contradiction is real. 11,485 cells, 18 sessions, 16 mice (13 CaMK2a-tTA, 3 Thy1); window 1.5 mm anterior to lambda. Fig. S2B read 0.093. Fig. S2A sums read 4970, 2746 and about 640 (sweep 4930, 2730, 600); the 2+ share is 18 to 19%. Lowest dot about 24 cm. |
| navratilova2026_2 | C | Bar 38.4 to the outline top, about 38.2 at its centre (sweep 38.1). Fig. S2B novel 0.059. "63% of familiar in RSC ... p=0.0067 and 55% in CA1" found. 19.1 ± 10.7% quoted. |
| navratilova2026_3 | C | Supplement: "Thy1 mice had somewhat smaller RSC place field sizes (29 cm vs 38 cm, T-test: p=0.038)" and "There was no difference between Thy1 RSC mice and Thy1 CA1 mice in these measures". Only 3 Thy1 RSC mice. |
| navratilova2026_4 | C | Bars 31.1 and 32.3 to the outline top, about 30.9 and 32.1 at the centre. Fig. S2B 0.146 and 0.080. 39.7% and 28.1% quoted. Ratios 40.2/30.9 = 1.30 and 38.1/32.1 = 1.19. |
| campagner2023_1 | C | "836 RSP units, 683 SC units, 4 mice" and "open shelter: 3.5%, closed shelter: 3.5%; P=0.721" found. Arena 92 cm, so 0.665 m². Vale is second author. |
| smith2012_1 | X | Table 2: 1031.44 cm². d = 2 x sqrt(1031.44/pi) = 36.24 cm. Shares 23.0% (84/366) and 38.3% quoted. Method corrected to stereotrode. Level corrected to anterior: AP -3.5 to -4.5 mm lies in front of the -4.5 mm cut that every other rat row uses. |
| smith2012_2 | X | 368.16 cm², d = 21.65 cm. Area ratio 2.80; diameter ratio 1.67. Method corrected to stereotrode and level to dorsal. |
| smith2012_3 | X | Table 1 counts 5/56, 7/45, 9/40, 63/225 and the CA1 counts all match. Method and level corrected as smith2012_1. |
| subramanian2024_1 | C | "85% of HPC neurons, compared to 45% of RSC neurons" found. 143 RSC neurons (97 pyramidal, 46 interneurons) from 4 rats; 51 of 65 position-tuned. Granular b target, with a few dysgranular records. |
| subramanian2024_2 | X | 43/282 = 15% found. Level corrected to dorsal (AP -4, ML 2.5). |
| subramanian2024_3 | X | Rendered Fig. 1f at 8x. The 40% bar is the 0.975 to 1.0 bin (bins are 0.025 wide), and the next bin holds about 20%, so the median is about 0.96. The T-maze CA1 mode is 0.025 to 0.05, not about 0.1. The Markus formula and the "proportion of the area" quote are confirmed. |
| miller2021_1 | C | "82.12%", "RSC mean = 0.32, compared to 0.05 in the hippocampus, Law et al. 2016" and "146 RSC neurons from 4 rats" found. These cells overlap subramanian2024_1 (flagged). |
| miller2019_1 | C | Tetrodes in rats (20-tetrode drives; Long-Evans), not imaging. "21% of neurons at asymptote", 637 neurons in 12 rats, 31 dysgranular neurons. 170 bins of 3 cm give 510 cm. |
| vedder2017_1 | C | "28.29% of RSC neurons at asymptote", "choice point: 58.0%, reward approach: 57.4%", 424 neurons. T-maze 112 x 122 cm (the md wrongly says plus maze). |
| laurent2026_1 | C | 319 and 145 cells, F(3,462) = 11.45 and F(3,630) = 47.99 found. Dysgranular RSC in 14 rats at AP -5.0 to -5.5. |
| laurent2026_2 | X | 91%, 99.5% and 100% found. The 91% pools both two-room environments (57 cells in S1-2id, 31 tracked in S2-2diff). Level corrected to dorsal. |
| alexander2017_1 | U | No text in the sweep folder (17-byte files). bioRxiv returned HTTP 429 and Europe PMC HTTP 500. No PMC copy exists (Europe PMC search: MED 28528904 and PPR32937, neither in PMC). Status corrected. |
| alexander2017_2 | U | As alexander2017_1. The row's own notes already doubt the 70% figure. |
| chen2026_1 | C | "13.4%, 62/463", dRSC 45/62, gRSC 17/62; 316 and 147 cells; 5 rats; 100 cm cylinder. |
| chen2026_2 | C | 45/316 = 14.2%. |
| chen2026_3 | C | 17/147 = 11.6%. |
| mao2018_1 | C | Fig. 3E vector rects: 0 cm at 224.10 pt, 40 cm at 184.60 pt; control bar top at 199.37 pt = 25.04 cm. Dot centres 18.40, 25.11, 25.48, 27.27, 28.95 cm (mean 25.04). "0.50 ± 0.02 ... 0.29 ± 0.04" and "place field width: P = 0.28" found. SI: FOV centred about AP -2.5. 5 sham Thy1 mice. |
| mao2018_2 | C | Bilesion bar top at 196.55 pt = 27.90 cm. Dots 22.24, 32.17, 32.35, 27.75, 25.01 cm. Decoding errors 14.60 and 27.68 cm found. |
| fischer2020_1 | C | "landmark: 22.3 ± 1.6%", 966 active and 491 task-engaged neurons, seven mice, "dysgranular part of RSC (Brodmann Area 30)". The authors are Fischer, Mojica Soto-Albors, Buck and Harnett; Harnett is last author. |
| lenkey2025_2 | C | "20.5% ... 665 out of 3244 cells ... 5 mice, 12 sessions"; "RSC: 37.31 cm (27.49-49.09 cm)"; 50% of (max - min) edge; multi-peak cells excluded; "medial-anterior part of RSC"; window AP -2.2; darkness with tactile cues. |
| lenkey2025_3 | X | "CA1: 37.31 cm (25.53-49.09 cm)" found. Level corrected to dorsal. Duplicate of lenkey2025_1 in `sweep_fields_mouse.csv` (flagged in its notes). The same paragraph gives the CA1 share, 17.4% of 6196 neurons in 7 mice, which the sweep omitted. |
| esteves2023_1 | X | Fig. S3B ring centres: 1 PF 0.792, 2 PF 0.202, 3 PF about 0.006. The 2 PF value is 0.20, not 0.21, and the mean is 1.21, not 1.22 (0.81 per metre either way). Panel A about 0.46. |
| esteves2023_2 | C | Filled markers 0.765, 0.224, 0.010 give a mean of 1.24. |
| esteves2023_3 | C | Rings 0.736, 0.246, 0.018 (sweep 0.73, 0.25, 0.019). |
| esteves2023_4 | C | Filled markers 0.678, 0.278, 0.043 (sweep 0.68, 0.28, 0.044). |
| park2024_1 | C | "vertex field sizes were similarly small (mean: ~80 cm2)", 740 neurons in nine mice, AP -2.5 granular. d = 2 x sqrt(80/pi) = 10.1 cm. "60 % (443 out of 740 neurons)" found. |
| park2024_2 | C | "27%, 200 out of 740" and "16%, 122 out of 740" found. |
| sun2021_1 | C | "~30% of neurons showed a significant contribution" found; Fig. 4 legend "n = 5 mice"; 2 m corridor; eighteen 10 cm bins. |
| chen1994_1 | C | PubMed abstract: "26 of 311 (8.4%) cells in RSA"; radial-arm maze. |
| chen1994_2 | C | Abstract: "8 of 94 (8.5%) cells in RSG". |
| jacob2017_1 | C | "96 classic HD cells (9%)" of 1090; AP -5.5 mm. |
| jacob2017_2 | C | "116 ... (BD) cells (11%)", "exclusively in the dysgranular RSC", 12 BD cells in darkness. |
| lozano2017_1 | C | "43.65° ± 1.13° in PoS and 49.29° ± 1.28° in RSC"; Table 1 columns PoS (n = 74) and RSC (n = 75); width is two circular SD. Ratio 1.13. AP -5.4 mm. |
| lozano2017_2 | C | "coverage for the data was 62 ± 2% and for the control data was 64 ± 2%". The Table 1 coverage row repeats the coherence statistics (1.48 ± 0.04, t = 3.67), as the sweep notes. |
| zhang2022_1 | C | "n = 75/743, 10.1%". |
| zhang2022_2 | C | Table 2 total row: 14 rats, 575 cells, 27 HD, 48 MD. 27/575 = 4.7%. |
| zhang2022_3 | C | "67/743 (9%)"; dark trials 21/28, 16/25, 11/25. |
| zhang2022_4 | C | Table 3 widths 133.97 ± 5.42, 138.66 ± 4.73 and 109.38 ± 2.29. |
| alexander2020_1 | C | "17.3% (n = 96/555)"; 7 male Long-Evans rats; AP 2.9 to 6.8 mm. The arena side is 125 cm ("62.5 cm or half the width of the arena"), so 1.5625 m² is right. |
| alexander2020_2 | C | "38.7% (n = 89/230)". |
| alexander2020_3 | C | "9.9% (n = 24/241) of granular RSC and 25.0% (n = 21/84)". |
| alexander2020_4 | C | Fig. 1K vector bars: first two bins 0.209 + 0.232 = 0.44, and the median falls in bin 3. The x labels (5, 30, 60) do not fit one linear scale, so the 10 to 15 cm reading stays uncertain, as the row says. Walls removed: "-10.0 cm, IQR = -25.75 to -4.5 cm". |
| alexander2020_5 | C | "n = 27/555; 4.9%". |
| alexander2020_6 | C | "45.8% of cells (n = 254/555)". |
| vanwijngaarden2020_1 | C | "485 out of 5415 RSC cells (9.0%)". |
| vanwijngaarden2020_2 | C | "main peak at 5 cm" and "up to 20 cm". |
| vanwijngaarden2020_3 | C | "190 out of 485 neurons (39.2%)". |
| lachance2024_1 | X | "210 RSC cells (40%)" of 519, 6 female rats, 120 x 120 cm. Level corrected to posterior: the bundle sat at AP -6 mm. |
| vandergoes2024_1 | C | "only 12% in RSC" (n = 12 mice) and n = 319 HD of 2458 non-HD units found. 319/2777 = 11.5%. The 50 cm arena gives 0.196 m². |
| vandergoes2024_2 | C | Medians 0.086 and 0.088 (n = 71) and 0.123 against 0.087 (n = 27) found. |
| sit2023_1 | C | "14.0%, 373/2665" and "14.2% (452/3179 ...; n = 16 recordings over 5 mice)"; posterior dysgranular RSC (paper's words); 250 mm chamber. |
| sit2023_2 | C | "Of the total number of tuned cells (12.1 ± 1.1%), landmark cells made up the plurality of cells (33 ± 3.5%) ... alignment cells (28 ± 1.9%) ... heading (22 ± 2.3%)". |
| sato2010_1 | X | Abstract "One-third" found. Region corrected: RSC is pooled with area 7m and posterior cingulate. |
| sato2006_1 | X | Abstract 580, 31% and 77% (139/180) found. Region corrected as sato2010_1. PMC1624861 is not in the open-access subset (BioC: no result). |

## Kernel rows

| Row | Verdict | Evidence |
|---|---|---|
| K-mao2017_1 | X (notes) | Fig. 3d path (drawing 43), ticks 0/75/150 cm at 428.56/461.34/494.12 pt: r = 0.5 at 13.25 cm, zero at 37.77 cm, -0.204 at 75 cm, minimum -0.229 at 93 cm. Samples 0.641 (10 cm), 0.292 (20), 0.102 (30), 0.034 (35), -0.025 (40), -0.110 (50) match the row. Notes corrected: the curve is not symmetric about 75 cm (0.5 again at 147.2 cm, zero again at 135.8 cm), so "values past 75 cm repeat by symmetry" is wrong. Duplicate of K16 in `sweep_kernel.csv`. |
| K-mao2017_2 | X (notes) | Drawing 45: r = 0.5 at 13.39 cm, zero at 33.14 cm, minimum -0.264 at 75.7 cm. Values confirmed (33.0 against 33.1 is within read precision). Notes flag the duplicate of K17. |
| K-mao2017_3 | C | My raster trace of Fig. 4b: 13.78 cm, 36.66 cm, minimum -0.33 at 80 cm. |
| K-mao2017_4 | C | 14.10 cm, 33.14 cm, minimum -0.32 at 83 cm. |
| K-mao2017_5 | C | 13.34 cm, 32.33 cm; -0.215 at 75 cm and a second dip of -0.247 at 123 cm. |
| K-mao2017_6 | C | My colour-bar lookup of Fig. 2c (bar 1 to -0.48, ticks at 568.66/587.37/606.08 pt; 2.5 cm separation bins): r = 0.5 at 9.65 cm, zero at 23.3 cm, minimum -0.37 at 74 cm. Sweep 9.8, 23.8, -0.37. |
| K-mao2017_7 | C | Same method: 12.16 cm, 36.84 cm, minimum -0.33 at 74 cm. Sweep 12.3, 36.8, -0.31, within the stated ±0.03. |
| fischer2020_k1 | C | Red trace of Fig. 3C (ticks at 74.5/164.5/240.5 px and 620.5/766.5 px), kept below the red "Landmark" label: peak 0.818 at -4.7 cm; 0.5 at -27.9 and +23.5 cm (full width 51.4 cm); zero at -93.5 and +70.2 cm. Fig. 3A shows 367 neurons. |
| fischer2020_k2 | C | Fig. 3F: peak 0.757 at -4.9 cm; 0.5 at -26.0 and +22.5 cm (full width 48.5 cm); zero at +82.2 cm; -0.13 at 140 cm. Fig. 3D shows 418 neurons. |
| alexander2017_k1 | U | Its status already says "text via WebFetch excerpts". The text could not be obtained (see alexander2017_1). The 1.9 ± 2.7 and 5.2 ± 6.7 cm errors are unverified. |
| miller2019_k1 | C | Method quotes found: 170 bins of 3 cm, "first 50 neurons", "64% reduction". The figure was not obtained: `A_miller2019_figs/f1.jpg` is not an image. |

## Cells counted twice

- K-mao2017_1 and K-mao2017_2 are the same curves as K16 and K17 in `sweep_kernel.csv`. K16 and K17 carry older values ("about 35" and "about 34", r = 0.5 "not read"). Keep one row per curve, preferably the new ones. K17's zero crossing should become 33 cm.
- mao2017_ca1_1 duplicates mao2017_1 in `sweep_fields_mouse.csv`. lenkey2025_3 duplicates lenkey2025_1 there. The page builder should skip region = CA1 rows from this file, or dedupe them.
- mao2017_rsc_3 to _5 come from 3 of the 4 WT mice in mao2017_rsc_1 and _2, so the cells overlap.
- subramanian2024_1 reuses the Miller 2021 open-field RSC cells (143 against 146 neurons, 4 rats each). subramanian2024_3 includes the Miller 2019 T-maze cells. The rows say so.
- Each study's rows by condition (navratilova2026_1 to _3, esteves2023_1 to _4, alexander2020_1 to _6, zhang2022_1 to _4, chen2026_1 to _3) share cells. The notes say so.

## Fixes to sweep_rsc.md

- Summary, bullet 3. Old: "Fewer RSC cells are spatial than CA1 cells. RSC/CA1 share ratios are 0.5 to 0.7 in every paired study. RSC place cells have slightly more fields: 1.55 against 1.27 fields per place cell on a 1.5 m belt." New: "In four of five paired studies fewer RSC cells are spatial than CA1 cells (RSC/CA1 share ratios 0.50 to 0.68). Lenkey 2025 is the exception, with 20.5% in RSC against 17.4% in CA1 (1.18), both Thy1. RSC place cells had 1.55 fields against 1.26 in CA1 on a 1.5 m belt, a difference Mao 2017 reports as not significant (P = 0.21)."
- "Shares of spatial cells" section, Mao 2017 fields per place cell. Change "a mean of about 1.27" to "a mean of about 1.26", and "1.03 against 0.85 fields per place cell" to "1.03 against 0.84".
- Same section. Change "1.17 superficial granular" to "1.21 superficial granular (bars sum to 0.97)". Change the Esteves line to "0.79, 0.20 and 0.006, a mean of 1.21, or 0.81 per metre".
- Same section, Navratilova. Replace "or 1.1 to 1.7 fields per place cell on its 3.1 to 5.0 m tracks (derived, rough)" with "Fig. S2A gives a more direct figure: about 81% of familiar place cells had one field and 19% two or more, about 1.2 fields per place cell, or 0.24 to 0.38 per metre on the 3.14 to 5.03 m tracks (derived)". The 0.34 figure divides an all-cell rate by a per-session mean share (27%), while the pooled S2A share is 40%.
- "RSC against CA1" table, Mao 2017 row. Change "Fields per place cell 1.55 vs 1.27 (1.22)" to "1.55 vs 1.26 (1.23), not significant". Add that the CA1 imaging set pooled 2 AAV mice and 1 Thy1 mouse, so the 13% against 26% share ratio does not match labels (Thy1 RSC gave 56%).
- Same table, Lenkey 2025 row. Change "RSC 20.5% (CA1 share in the mouse sweep)" to "20.5% vs 17.4% (1.18)".
- Environments table. Change Vedder 2017 from "plus maze | not given" to "T-maze | 112 cm stem x 122 cm".
- Novelty bullet. "Unlike CA1, RSC fields did not expand or shift backwards across laps" is too strong. RSC fields did shift backwards in the familiar room, less than CA1 (Fig. 5C, p < 0.0001 for RSC), and 8% of RSC fields shifted significantly. Suggested text: "RSC fields shifted backwards less than CA1 fields (8% against about 20% of fields significant; no significant mean shift in the novel room), and only CA1 fields expanded when width was measured on dF/F."
- Kernel notes. "On a loop ... values past 75 cm repeat by symmetry" (CSV) is wrong; see K-mao2017_1. The md line "The loop sends the curve back up toward +0.6 at 150 cm" is fine.
- Subramanian. "Sparsity about 0.95" can stay as a rounded figure. The median is about 0.96, and the tallest bin is 0.975 to 1.0.
- Alexander & Nitz 2017 numbers in "Repeated segments" and the "RSC against CA1" table should carry "(WebFetch summary, unverified)".

## Review sentences that must change

File: `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`.

1. Line 116, "Retrosplenial cortex" table.
   - Old: `| Mao et al. 2017 | Mouse, place cells | 150 cm treadmill loop | Pearson, rates scaled 0 to 1, 1.5 cm bins | Crosses zero at about 35 cm (23%; the text says "~35 cm"), then about -0.2 at 70 to 90 cm; CA1 imaged the same way crosses at about 34 cm | full text, figure-read |`
   - New: `| Mao et al. 2017 | Mouse, 297 place cells, 4 mice (AAV) | 150 cm treadmill loop | Pearson, rates scaled 0 to 1, 1.5 cm bins, Gaussian SD 4.5 cm | Falls to 0.5 at 13.2 cm and crosses zero at 37.8 cm (25% of the loop; the Discussion says "~35 cm"), then about -0.2 at 70 to 90 cm, minimum -0.23 near 93 cm; CA1 imaged the same way falls to 0.5 at 13.4 cm and crosses zero at 33 cm, minimum -0.26 near 76 cm | full text, figure-read |`
2. Line 148, PV-similarity paragraph.
   - Old: `Mouse retrosplenial cortex on a 150 cm treadmill belt gave a curve that "steeply falls to zero over a short distance", about 35 cm or 23% of the belt according to a paraphrase of the Discussion, and close to CA1 imaged the same way [full text; 35 cm not confirmed verbatim] ([Mao et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557927)).`
   - New: `Mouse retrosplenial cortex on a 150 cm treadmill belt gave a curve that "steeply falls to zero over a short distance (~35 cm)", in the words of the Discussion. On the Fig. 3d curve it falls to 0.5 at 13.2 cm and crosses zero at 37.8 cm, 25% of the belt. CA1 imaged the same way gives 13.4 and 33 cm [full text, figure-read] ([Mao et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557927)).`
3. Line 172, summary table.
   - Old: `RSC about 35 cm;`
   - New: `RSC to 0.5 at 13 cm and to zero at 35 to 38 cm;`
4. Line 462, regional paragraph, first sentence on RSC.
   - Old: `In mouse retrosplenial cortex on a 150 cm treadmill, 297 of 2,256 imaged neurons, about 13%, were place cells with narrow fields [full text] ([Mao et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557927)).`
   - New: `In mouse retrosplenial cortex on a 150 cm treadmill, 297 of 2,256 AAV-labelled neurons, about 13%, were place cells. Their fields were 39.6 ± 1.2 cm wide (mean ± s.e.m., edge at 20% of peak above baseline), against 37.0 ± 0.8 cm in CA1 imaged the same way [full text] ([Mao et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557927)).`
5. Line 462, last sentence.
   - Old: `No source gives field sizes in metres for nucleus reuniens, retrosplenial cortex or prefrontal cortex.`
   - New: `No source gives field sizes in metres for nucleus reuniens or prefrontal cortex. Mouse retrosplenial place fields on 1.5 to 5 m tracks are 25 to 50 cm wide and close to CA1 fields measured the same way: 37.31 cm in both regions on a 1.57 m wheel ([Lenkey et al. 2025](https://doi.org/10.1038/s41467-025-60679-4)), and about 40 against 31 cm on 3 to 5 m virtual tracks, a gap that disappears when the mouse line is matched ([Navratilova et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092332)) [full text, figure-read].`
6. Line 542, kernel table (optional, for consistency).
   - Old: `"steeply falls to zero over a short distance (~35 cm)", then about -0.2 at 70 to 90 cm (figure-read)`
   - New: `"steeply falls to zero over a short distance (~35 cm)"; the curve reads 0.5 at 13.2 cm and zero at 37.8 cm (CA1 13.4 and 33 cm), then about -0.2 at 70 to 90 cm (figure-read)`

Line 557, "Retrosplenial cortex resembled CA1", is correct and needs no change.

## Follow-ups outside this audit

- Update K16 and K17 in `sweep_kernel.csv`, or drop them in favour of K-mao2017_1 and _2.
- Retry Alexander & Nitz 2017 (bioRxiv 10.1101/100537, JATS at `https://www.biorxiv.org/content/early/2017/01/15/100537.source.xml`) when bioRxiv stops rate-limiting, or get the Curr Biol PDF.
