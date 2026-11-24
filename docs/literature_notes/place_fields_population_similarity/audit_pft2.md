# Audit of the pft2 full-text pass

Date: 2026-11-24. Auditor: independent re-read of the seven PDFs in `/home/jackking/cls/papers/pft2/`, the extracted text, and the PDF figure rasters. The proposals were not edited. Fixes to them are in `audit_pft2_fixes.csv` (54 lines; long cells are fixed by exact phrase, and every phrase was checked by script to occur in the proposal). Scripts and renders: `/home/jackking/.claude/jobs/2c8b1c8e/tmp/audit_pft2/`.

## Summary

| File | Items | Confirmed | Corrected | Rejected |
|---|---|---|---|---|
| pft2_additions_fields_rsc.csv | 6 | 5 | 1 | 0 |
| pft2_additions_kernel_rsc.csv | 6 | 1 | 5 | 0 |
| pft2_additions_fields_mec.csv | 8 | 6 | 2 | 0 |
| pft2_additions_fields_mouse.csv | 1 | 1 | 0 | 0 |
| pft2_corrections.csv | 120 | 110 | 10 | 0 |
| pft2_review_edits.md | 42 | 27 | 15 | 0 |

Every `old_value` in `pft2_corrections.csv` equals the current cell of its target file (120 of 120, script `check_old.py`). Every "old" text in `pft2_review_edits.md` occurs exactly once in the review, and no two old texts overlap (scripts `parse_edits.py`, `overlap.py`). No new text has an em dash, en dash or curly quote, and the new headings are in sentence case.

Main problems:

1. "Rat retrosplenial similarity never falls to 0.5" (edits 1, 2, 3, 4, 18, 21, 23, 24, 28, 29). It holds for the odd/even curves (minimum 0.53 to 0.56 on the W track, 0.71 on the plus lap) but not for the between-session curve of Fig. 6a, which reaches 0.50 at 65 to 75 cm and falls below 0.5 beyond 177 cm. The row alexander2015_k2 reads its minimum as 0.51; the audit reads 0.503.
2. Missing edits. Alexander & Nitz 2017 gives a dorsal CA1 curve on a 6.6 m lap, but three review sentences still say no dorsal CA1 curve exists beyond a 4 m loop (review lines 3, 163 and 177). Edits 1 and 3 would then contradict lines 3 and 163. New edits A1 to A4 in the CSV fix this.
3. alexander2015_k4 misdescribes the RSC alpha/beta matrix: r returns to the diagonal value (0.58 to 0.60) at 90 to 150 cm, so there is no "small remaining peak", and the row contradicts the extractor's own table.
4. alexander2017_k1 (correction [114]) calls a 0.01 rise a local maximum without saying how small it is, and reads the minimum near 490 cm as "start and end of the lap", which it is not.
5. mallory2018_1 notes keep "Same-animal grid spacing"; the CA1 and MEC recordings came from separate mice.
6. The Butler grid field size (1420 cm²) is tied to the 0.8-threshold field method only by inference, and "same recordings" for grid spacing means the same pooled dataset, not the same rats cell by cell.

### Is the 0.40 against 1.55 contrast stated fairly?

Only in part. Both ratios are arithmetically right: 56.1/36.26 = 1.55 (Mallory, iWT mice) and 39.5/99.14 = 0.40 (Butler, rats). Edit 34 names species, recording depth, box size and field rule, and flags the 0.85 threshold as probably giving smaller fields. It leaves out what drives most of the gap. The widths are close, 56.1 cm against 39.5 to 49.9 cm, a factor of 1.1 to 1.4, while the grid spacings differ 2.7-fold (36.26 cm in dorsal mouse MEC against 99.14 cm in rats with modules of 65 to 122 cm). Mallory gives a mean of a skewed distribution (SD 2,233 cm², about equal to the mean) for 32 strictly selected cells; Butler gives a median for 271 model-defined cells, a class closer to Mallory's 355 position-coding non-grid cells, whose fields covered a mean of 35% of the box against a median of 13% in Butler. The heading (edit 30) and the at-a-glance row (edit 5) set the two numbers side by side with no qualifier. The CSV gives replacement text for edits 5, 30 and 34 and for the MNG33 note.

### Matrix re-reads

Own scripts (`kread.py` and per-figure readers) extracted each raster with PyMuPDF, mapped every pixel to the nearest colour-bar entry, masked black and white lines with a 1 px margin, and averaged r by |row - column|. Distances use 1.04 cm per template bin for the 2015 figures (turn markers at 0.258, 0.505 and 0.794 of the template; the same-turn rise of Fig. 3e then falls at 110 cm) and 660 cm per side for 2017 Fig. 1.

| Curve | Proposal | Audit reading |
|---|---|---|
| 2015 Fig. 3e RSC odd/even, mean of 4 | peak 0.86; 0.79, 0.71, 0.63, 0.57 at 10, 20, 30, 50 cm; min 0.56 at 64 cm; half-drop 20 cm; 0.66 at 110 cm | peak 0.82 (masked) to 0.86 (unmasked); 0.80, 0.71, 0.63, 0.57; min 0.557 at 65 cm; half-drop 20 to 23 cm; 0.656 at 109 cm; global min 0.536 |
| 2015 Fig. 6a RSC alpha/alpha-prime | peak 0.77; min 0.51 at 71 cm; 0.58 at 101 cm; never 0.5 before 177 cm | peak 0.75 to 0.77; min 0.503 at 66 to 74 cm; 0.575 at 101 cm; 0.50 to 0.52 at 136 to 176 cm; below 0.5 after 177 cm |
| 2015 Fig. 6a CA1 alpha/alpha-prime | peak 0.72; 0.5 at 26.5 cm; min 0.09 at 93 cm; 0.58 at ±21 cm | peak 0.69 to 0.70; 0.5 at 26.4 cm; min 0.084 at 93 cm; 0.57 and 0.55 at -21 and +21 cm |
| 2015 Fig. 6a RSC alpha/beta | 0.61 at diagonal, 0.55 beyond 40 cm | 0.60 at diagonal; 0.54 at 34 to 72 cm; 0.58 to 0.60 at 93 to 150 cm; 0.47 to 0.52 beyond 175 cm |
| 2015 Fig. 6a CA1 alpha/beta | 0.13 to 0.15 everywhere | 0.10 to 0.15 to 150 cm; 0.19 to 0.21 near 175 cm |
| 2017 Fig. 1C RSC | peak 0.91 (0.93); floor 0.77; min 0.71 to 0.72; half-drop 45 cm; 0.79 at 165 to 182 cm | peak 0.909 (0.933); floor 0.769 (0.778); min 0.714 (0.718) at 486 cm; half-drop 44.9 cm; 0.780 at 122 cm and 0.793 at 170 cm; corners 0.75 to 0.76 |
| 2017 Fig. 1D CA1 | peak 0.81; 0.7 at 35 cm; 0.6 at 48 to 50 cm; median 0.5 at 60 to 63 cm | 0.809; 34.9 cm; 48 to 50 cm; median reaches the floor at 60 to 65 cm (60.6 cm under tick labels) |
| 2017 Fig. S4H ring | peak 0.92; half-drop 38 cm; floor 0.84 | 0.92; 34 to 36 cm; 0.83 to 0.85 |
| 2015 Fig. 6b odd/even error bars | RSC 3.8 to 4.2 cm; CA1 5.2 to 6.6 cm | RSC 3.7 to 4.2 cm; CA1 5.1 to 6.6 cm (scale 0.50 to 0.52 pt per cm fitted to the printed alpha/beta and alpha/alpha-prime means; the axis labels alone are not evenly spaced) |
| Butler Fig. 1K modules | ENV1 64.9 to 121.6 cm | render check: about 64.5, 91, 102, 104, 107, 121.5 cm |

Other checks: Butler Table S1 and Fig. S10A, SM field rule, SM counts and the Fig. S7E contradiction; Mallory Online Methods (arenas, bins, smoothing, speed and coverage cuts, stable-cell rule, place-field rule, border rule) and Table 1; Diehl Table S1 arithmetic and STAR Methods; the 2015 and 2017 shares, geometry and reconstruction errors. Neither Alexander paper describes per-cell normalization of the population vectors; 2015 Fig. 6 names Pearson, 2017 STAR Methods name Spearman. The 2015 Methods give template bins of ~1.2 cm, which conflicts with 200 bins over a 215 cm route; the turn markers and the 110 cm rise support about 1.04 cm per bin, so distances could be up to 15% low only if the Methods value is right.

## Additions

- pft2_additions_fields_rsc.csv alexander2015_1: confirmed. nn.4058 pp. 1145-1146: 45.6% (104/228), 54.3% (124/228), 52.4% (65/124), 21.7% (27/124), chance 4.32 ± 1.7%, 43.3% (45, 19.7% of all), "a separate 26% of all neurons", 22% with turn-bias ratio > 2; p. 1147 "little evidence for single, isolated firing fields"; Suppl. Fig. 5 "100% of RSC neurons"; route 215 cm, turns at 60, 110, 170 cm (Online Methods p. 10); room 13.5' x 13.5' (Suppl. Fig. 1).
- pft2_additions_fields_rsc.csv alexander2015_2: confirmed. p. 1146: µRSC 0.16 (n 374), µHPC 0.07 (n 218); p. 1147: α/α′ 0.34 and HPC 0.40; Fig. 4b labels 137/187 and 134/187; legend "Greater than 70%".
- pft2_additions_fields_rsc.csv alexander2015_3: confirmed. Suppl. Fig. 9: "(6%, n=15/243)", resultant > 0.5 and within 0.36 rad at both locations; Online Methods p. 11: 5° bins, < 20° difference.
- pft2_additions_fields_rsc.csv alexander2015_ca1_1: confirmed. Online Methods p. 10: CA1 rule (≤ 8 Hz, bin > 3 Hz, ≥ 150 spikes, field ≥ 5 Hz over 20 bins, ~24 cm); 4 of 6 rats had a CA1 drive; Suppl. Fig. 4b: n = 39, CV 1.50 ± 0.53 against 1.57 ± 0.42, p = 0.24, RSC 0.52 ± 0.33, "unidirectional place fields that happened to overlap with a turn site".
- pft2_additions_fields_rsc.csv alexander2017_3: confirmed. alex2017 p. 1552: 1.9 ± 2.7 cm RSC, 141 CA1 cells in 3 rats, 5.2 ± 6.7 cm "mean ± SEM"; p. 1556: 2.93 against 12.61 cm, n = 72; Rayleigh HPC p = 0.03; STAR Methods p. e2: CA1 exclusions (> 8 Hz, no 0 Hz bin, no bin > 3 Hz). Fig. 1B shows "55cm" on one of 12 equal edges (render checked).
- pft2_additions_fields_rsc.csv alexander2017_4: corrected. Numbers confirmed (Fig. S2D: n = 110, 0.53, 0.09, 0.07, p = 9.56 x 10^-10; 2 of 5 rats also ran the W track; 25 + 25 trials per day). The measure should name Spearman (STAR Methods p. e4). Fix in CSV.
- pft2_additions_kernel_rsc.csv alexander2015_k1: corrected. Own re-read of Fig. 3e (xref 30, four matrices, bar 0.92 to 0.34): r 0.80, 0.71, 0.63, 0.57 at 10, 20, 30, 50 cm; first minimum 0.557 at 65 cm; rise to 0.656 at 109 cm; 0.54 to 0.58 beyond 140 cm. All match. The near-diagonal peak reads 0.82 with a 1 px mask round the black line and 0.85 to 0.87 without, so the half-drop is 20 to 23 cm, not "about 20". Bin scale: the Methods' ~1.2 cm bins conflict with 1.04; note added. Fixes in CSV.
- pft2_additions_kernel_rsc.csv alexander2015_k2: corrected. Own re-read of Fig. 6a lower left: peak 0.75 to 0.77; 0.73, 0.66, 0.58, 0.52 at 10, 20, 30, 50 cm; minimum 0.503 at 66 to 74 cm; 0.575 at 101 cm; 0.50 to 0.52 at 136 to 176 cm; below 0.5 beyond 177 cm. The minimum is 0.50, not 0.51, so "not reached" depends on the reading error. Fixes in CSV.
- pft2_additions_kernel_rsc.csv alexander2015_k3: corrected. Own re-read of Fig. 6a lower right (bar 0.78 to -0.08): 0.69 at 1 cm, 0.70, 0.59, 0.45, 0.23 at 10, 20, 30, 50 cm; r = 0.5 at 26.4 cm; minimum 0.084 at 93 cm; 0.08 to 0.11 from 80 to 130 cm; 0.57 and 0.55 at -21 and +21 cm. All numbers confirmed; only the bin-scale caveat is added (1.2 cm bins would give about 30 cm). Fix in CSV.
- pft2_additions_kernel_rsc.csv alexander2015_k4: corrected. Own re-read of Fig. 6a upper row: RSC 0.60 at the diagonal, 0.54 at 34 to 72 cm, 0.58 to 0.60 at 93 to 150 cm, 0.47 to 0.52 beyond 175 cm; CA1 0.10 to 0.15 to 150 cm and 0.19 to 0.21 near 175 cm. The row's "0.55 at 40 cm and beyond, a small remaining peak" is wrong (its own table in fulltext_pft2.md shows 0.59 to 0.60 at 110 to 150 cm). Errors 51.32 and 63.45 cm confirmed (p. 1148). Fixes in CSV.
- pft2_additions_kernel_rsc.csv alexander2017_k2: corrected. Own re-read of Fig. 1D (xref 64, bar 1 to 0.5): peak 0.81; 0.7 at 35 cm; 0.6 at 48 to 50 cm; median reaches the clipped floor at 60 to 65 cm (50% of pixels at the floor at 62 cm, 94% by 86 cm); tick-label calibration gives the median 0.5 crossing at 60.6 cm. Confirmed. The note "about twice as wide" understates: 2.2 to 5 times the other rat CA1 kernels. Fix in CSV.
- pft2_additions_kernel_rsc.csv alexander2017_k3: confirmed. Own re-read of Fig. S4H right (xref 672, 147 x 146 px, bar 1 to 0.7, labels aligned with the bar ends): peak 0.92; 0.88 at 34 to 36 cm (row says about 38); 0.85 at 60 cm; 0.83 to 0.85 from 60 to 370 cm. Panel label "RSC neuron (n =229)" against legend n = 90 confirmed. Ring 122 cm (Fig. S4F) and 9.99 ± 9.41 cm error (p. 1555) confirmed.
- pft2_additions_fields_mec.csv MNG33: corrected. SM p. 10 field rule (log-normalized, 0.85 non-grid, 0.8 grid, at least six contiguous bins); SM p. 4 bins 5 cm, 30 x 30; Fig. S10A: 1225 and 1106 cm², p = 0.35, 2 fields, coverage 0.13; Table S1 spacing 99.14 (85.85, 107.37); 2*sqrt(1225/pi) = 39.5 cm; 39.5/99.14 = 0.40; Fig. 1K (render checked) about 65 to 122 cm. Fixes: "same recordings" is pooled, not paired; grid field method link is an inference; "well above" softened; fairness sentence added.
- pft2_additions_fields_mec.csv MNG34: confirmed. Fig. S10A: ENV2 1106 cm², median difference 5 cm², p = 0.35, coverage 0.14; 2*sqrt(1106/pi) = 37.5; 37.5/93.07 = 0.40; spacing ratio 0.95 (Table S1); grid field size p = 0.85.
- pft2_additions_fields_mec.csv MNG35: confirmed. Fig. S10A (SM pp. 32-33): coherence > 2.25, n = 100, 1958 and 1969 cm², 383 cm², p = 0.13; fields 1 and 2, p = 0.37; coverage 0.15 and 0.17, p = 0.70; widths 49.9 and 50.1 cm; 49.9/99.14 = 0.50, 50.1/93.07 = 0.54, 49.9/65 = 0.77.
- pft2_additions_fields_mec.csv MNG36: corrected. Table S1 values confirmed (1420 (925, 2013), 1325 (1027, 1863), p = 0.85; 675 and 625 cm², p = 0.16; spacing p = 0.015; control spacing ratio 1.01, p = 0.19); SM: modules from 193 and 216 cells with grid score > 0.5, four animals (Fig. 1J). Fig. 1K values match the render to about 1 cm. Fix: "detected with the same procedure" is an inference.
- pft2_additions_fields_mec.csv MNG37: confirmed. Main text p. 1447: 778 cells; SM pp. 2-3: 598 MEC, 180 parasubiculum, postsubiculum and other cortex; SM p. 3: experimental n = 8, one excluded; p. 4: control n = 3; rat n = 11 (SM p. 2); border n = 79 (Fig. S4 legend), 24 of 36 remapped (p. 1448); 271/778 = 34.8%, 102/778 = 13.1%.
- pft2_additions_fields_mec.csv MNG38: confirmed. p. 1448: n = 159; Table S3: 62 (39%), 33 (20%), 23 (14%), 41 (26%), baseline 0.58, Group I 59 to 36 cm, p = 2 x 10^-5; Fig. 4J: -4.3 cm, P = 0.028; Fig. S12D: 68.1 and 68.5 cm, 43 sessions; 159/271 = 58.7%.
- pft2_additions_fields_mec.csv MNG39: confirmed. diehl2017 Table S1 (PDF pp. 30-31): shape 188 (83/105), color 227 (120/107), location 70 (35/35); grid 31, 38, 8; border 18, 18, 4; non-grid 129, 161, 57; nonspatial 8, 8, 1; HD 2, 2, 0. Unique: 345, 168/177, 233, 61, 32, 15, 4 (sum 345). 326/345 = 94.5%; 61/345 = 17.7%, 32/345 = 9.3%, 233/345 = 67.5%. STAR Methods p. e3: "over 85% of the data was recorded from more dorsal regions"; no field size or spacing in the paper.
- pft2_additions_fields_mec.csv MNG40: confirmed. Mallory Fig. 1h legend (p. 271): n = 355 iWT, 35.17 ± 32.52%, 38.02 ± 33.53%, P = 0.30, stability 0.18 and 0.19, MI 0.30 and 0.27; arenas 70, 90 or 100 cm (Online Methods p. 14); 355/527 = 67.4%; 0.3517 x 0.49 to 1.0 m² = 0.17 to 0.35 m².
- pft2_additions_fields_mouse.csv mallory2018_2: confirmed. Source found: /home/jackking/.claude/jobs/2c8b1c8e/tmp/sweep_mec/dl/mallory2018_esm/esm1.txt line 357 (Suppl. Fig. 11a: "iWT = 22.98 ± 11.56, iCre-KO = 30.39 ± 14.92 ... p = 0.023; n = 42 fields from 30 cells from 3 iWT mice"), also in tmp/grid_mouse/dl/mallory_nature/a.txt; Fig. 11b: "mean distance from the diagonal ... correlation of 0.2 (iWT = 28.8 ± 8.1 cm"; Online Methods p. 16: track 100 x 7 cm, 23 cm high, 31 cm walls, field rule (≥ 1 Hz, > 20% of cell peak, extended to 20% of field peak), central 70 cm, < 3 cm/s omitted, 15 to 30 min, 2 to 3 weeks, separate CA1 group.

## Corrections (pft2_corrections.csv, index from 0)

- [0] sweep_fields_mouse.csv:mallory2018_1 strain: confirmed. Old value matches. s41593-017-0055-3.pdf Online Methods p. 14: "hybrid 50:50% C57BL/6 J:129SVEV"; "Male (n = 29) and female (n = 25)"; p. 270 "AAV-DJ CMV cre-GFP".
- [1] sweep_fields_mouse.csv:mallory2018_1 level_definition: confirmed. Old value matches. p. 14: "-1.8 mm AP from bregma, 1.8 mm ML and 0.6-0.8 mm below the dura".
- [2] sweep_fields_mouse.csv:mallory2018_1 environment_size: confirmed. Old value matches. p. 14: "50-cm tall square arenas ... for hippocampal recordings: 50 or 62 cm"; 0.5² = 0.25, 0.62² = 0.384 m².
- [3] sweep_fields_mouse.csv:mallory2018_1 env_length_m: confirmed. Old value matches. p. 14: "50-cm tall square arenas ... for hippocampal recordings: 50 or 62 cm"; 0.5² = 0.25, 0.62² = 0.384 m².
- [4] sweep_fields_mouse.csv:mallory2018_1 env_area_m2: confirmed. Old value matches. p. 14: "50-cm tall square arenas ... for hippocampal recordings: 50 or 62 cm"; 0.5² = 0.25, 0.62² = 0.384 m².
- [5] sweep_fields_mouse.csv:mallory2018_1 familiarity: confirmed. Old value matches. p. 14: "20-60 min sessions"; "habituated to the testing arena ≥ 2 weeks".
- [6] sweep_fields_mouse.csv:mallory2018_1 field_rule: confirmed. Old value matches. pp. 14-15: peak-to-trough > 0.20 ms, spatial information > 95th percentile of 100 shuffles, > 100 spikes, peak ≥ 0.1 Hz, 2.5 x 2.5 cm bins, 5 x 5 quasi-Gaussian, speed ≥ 2.5 cm/s, coverage ≥ 70%, ≥ 5 contiguous bins (31.25 cm²) reaching 20% of peak.
- [7] sweep_fields_mouse.csv:mallory2018_1 fields_per_cell: confirmed. Old value matches. Table 1 (p. 274): place cells n 113, number of fields 1.95 ± 1.16.
- [8] sweep_fields_mouse.csv:mallory2018_1 location: confirmed. Old value matches. Pages match the journal PDF.
- [9] sweep_fields_mouse.csv:mallory2018_1 verified_in: confirmed. Old value matches. Pages match the journal PDF.
- [10] sweep_fields_mouse.csv:mallory2018_1 notes: corrected. Old value matches. Arena shares confirmed (19.7/50 = 0.39, 19.7/62 = 0.32; 303.75/2500 = 12%, /3844 = 8%), coverage 12.27 ± 6.1% (Fig. 2d), 1.95 fields. But the kept phrase "Same-animal grid spacing" is wrong: MEC and CA1 drives went to separate mice (Online Methods p. 14; Table 1 EEG n 12 and 11). Fix in CSV.
- [11] sweep_fields_mec_nongrid.csv:MNG01 environment_size: confirmed. Old value matches. p. 14: "for MEC recordings: 70 cm, 90 cm or 100 cm"; sizes chosen per animal.
- [12] sweep_fields_mec_nongrid.csv:MNG01 env_length_m: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [13] sweep_fields_mec_nongrid.csv:MNG01 env_area_m2: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [14] sweep_fields_mec_nongrid.csv:MNG01 familiarity: confirmed. Old value matches. p. 14: ≥ 2 weeks habituation, 20 to 60 min sessions, "not more than twice per day".
- [15] sweep_fields_mec_nongrid.csv:MNG01 strain: confirmed. Old value matches. p. 14 strain and sexes; p. 270 vector.
- [16] sweep_fields_mec_nongrid.csv:MNG01 level_definition: confirmed. Old value matches. p. 14: "0.4-0.5 mm AP from the transverse sinus, 3.25-3.45 mm ML and 0.8-1.0 mm below the dura"; "moved ventral by 25 µm". The 386 ± 222 µm value is in ESM Suppl. Fig. 7d.
- [17] sweep_fields_mec_nongrid.csv:MNG01 verified_in: corrected. Old value matches. Values confirmed in the journal PDF, but the row cites Supp. Figs. 7 and 8, which are only in the ESM; keep that path. Fix in CSV.
- [18] sweep_fields_mec_nongrid.csv:MNG02 environment_size: confirmed. Old value matches. p. 14: "for MEC recordings: 70 cm, 90 cm or 100 cm"; sizes chosen per animal.
- [19] sweep_fields_mec_nongrid.csv:MNG02 env_length_m: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [20] sweep_fields_mec_nongrid.csv:MNG02 env_area_m2: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [21] sweep_fields_mec_nongrid.csv:MNG02 familiarity: confirmed. Old value matches. p. 14: ≥ 2 weeks habituation, 20 to 60 min sessions, "not more than twice per day".
- [22] sweep_fields_mec_nongrid.csv:MNG02 strain: confirmed. Old value matches. p. 14 strain and sexes; p. 270 vector.
- [23] sweep_fields_mec_nongrid.csv:MNG02 level_definition: confirmed. Old value matches. p. 14: "0.4-0.5 mm AP from the transverse sinus, 3.25-3.45 mm ML and 0.8-1.0 mm below the dura"; "moved ventral by 25 µm". The 386 ± 222 µm value is in ESM Suppl. Fig. 7d.
- [24] sweep_fields_mec_nongrid.csv:MNG02 verified_in: corrected. Old value matches. Values confirmed in the journal PDF, but the row cites Supp. Figs. 7 and 8, which are only in the ESM; keep that path. Fix in CSV.
- [25] sweep_fields_mec_nongrid.csv:MNG03 environment_size: confirmed. Old value matches. p. 14: "for MEC recordings: 70 cm, 90 cm or 100 cm"; sizes chosen per animal.
- [26] sweep_fields_mec_nongrid.csv:MNG03 env_length_m: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [27] sweep_fields_mec_nongrid.csv:MNG03 env_area_m2: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [28] sweep_fields_mec_nongrid.csv:MNG03 familiarity: confirmed. Old value matches. p. 14: ≥ 2 weeks habituation, 20 to 60 min sessions, "not more than twice per day".
- [29] sweep_fields_mec_nongrid.csv:MNG03 strain: confirmed. Old value matches. p. 14 strain and sexes; p. 270 vector.
- [30] sweep_fields_mec_nongrid.csv:MNG03 level_definition: confirmed. Old value matches. p. 14: "0.4-0.5 mm AP from the transverse sinus, 3.25-3.45 mm ML and 0.8-1.0 mm below the dura"; "moved ventral by 25 µm". The 386 ± 222 µm value is in ESM Suppl. Fig. 7d.
- [31] sweep_fields_mec_nongrid.csv:MNG03 verified_in: corrected. Old value matches. Values confirmed in the journal PDF, but the row cites Supp. Figs. 7 and 8, which are only in the ESM; keep that path. Fix in CSV.
- [32] sweep_fields_mec_nongrid.csv:MNG04 environment_size: confirmed. Old value matches. p. 14: "for MEC recordings: 70 cm, 90 cm or 100 cm"; sizes chosen per animal.
- [33] sweep_fields_mec_nongrid.csv:MNG04 env_length_m: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [34] sweep_fields_mec_nongrid.csv:MNG04 env_area_m2: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [35] sweep_fields_mec_nongrid.csv:MNG04 familiarity: confirmed. Old value matches. p. 14: ≥ 2 weeks habituation, 20 to 60 min sessions, "not more than twice per day".
- [36] sweep_fields_mec_nongrid.csv:MNG04 strain: confirmed. Old value matches. p. 14 strain and sexes; p. 270 vector.
- [37] sweep_fields_mec_nongrid.csv:MNG04 level_definition: confirmed. Old value matches. p. 14: "0.4-0.5 mm AP from the transverse sinus, 3.25-3.45 mm ML and 0.8-1.0 mm below the dura"; "moved ventral by 25 µm". The 386 ± 222 µm value is in ESM Suppl. Fig. 7d.
- [38] sweep_fields_mec_nongrid.csv:MNG04 verified_in: corrected. Old value matches. Values confirmed in the journal PDF, but the row cites Supp. Figs. 7 and 8, which are only in the ESM; keep that path. Fix in CSV.
- [39] sweep_fields_mec_nongrid.csv:MNG05 environment_size: confirmed. Old value matches. p. 14: "for MEC recordings: 70 cm, 90 cm or 100 cm"; sizes chosen per animal.
- [40] sweep_fields_mec_nongrid.csv:MNG05 env_length_m: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [41] sweep_fields_mec_nongrid.csv:MNG05 env_area_m2: confirmed. Old value matches. 0.7 to 1.0 m; 0.49 to 1.0 m².
- [42] sweep_fields_mec_nongrid.csv:MNG05 familiarity: confirmed. Old value matches. p. 14: ≥ 2 weeks habituation, 20 to 60 min sessions, "not more than twice per day".
- [43] sweep_fields_mec_nongrid.csv:MNG05 strain: confirmed. Old value matches. p. 14 strain and sexes; p. 270 vector.
- [44] sweep_fields_mec_nongrid.csv:MNG05 level_definition: confirmed. Old value matches. p. 14: "0.4-0.5 mm AP from the transverse sinus, 3.25-3.45 mm ML and 0.8-1.0 mm below the dura"; "moved ventral by 25 µm". The 386 ± 222 µm value is in ESM Suppl. Fig. 7d.
- [45] sweep_fields_mec_nongrid.csv:MNG05 verified_in: corrected. Old value matches. Values confirmed in the journal PDF, but the row cites Supp. Figs. 7 and 8, which are only in the ESM; keep that path. Fix in CSV.
- [46] sweep_fields_mec_nongrid.csv:MNG01 field_rule: confirmed. Old value matches. p. 15: "Neurons were defined as stable spatial cells if their stability and coherence scores were higher than the 95th percentile ... and they were not classified as grid or border cells"; within-trial stability from even and odd time bins; no non-grid field rule anywhere in the paper (only the place-field rule).
- [47] sweep_fields_mec_nongrid.csv:MNG04 field_rule: corrected. Old value matches. p. 15: border fields ≥ 10 connected bins in the 75th percentile; width = mean distance of edge bins from the nearest boundary. "Averaged over the cell's fields" is not in the paper. Fix in CSV.
- [48] sweep_fields_mec_nongrid.csv:MNG05 field_rule: confirmed. Old value matches. p. 15: field size from the "inner circle fit around the center autocorrelation peak"; spacing from the six nearest peaks; 3 iWT and 4 iCre-KO cells excluded.
- [49] sweep_fields_mec_nongrid.csv:MNG01 notes: corrected. Old value matches. Arithmetic confirmed (2473/10000 = 25%, 2473/4900 = 50%; 56.1/100 = 0.56, 56.1/70 = 0.80). "Probably caps" needs an inference tag. Fix in CSV.
- [50] sweep_fields_mec_nongrid.csv:MNG10 environment_size: confirmed. Old value matches. diehl2017.pdf STAR Methods p. e4: the 80 x 80 cm area applies to the shape paradigm and to comparisons across locations; the color box is 100 x 100 cm (p. e2).
- [51] sweep_fields_mec_nongrid.csv:MNG10 env_length_m: confirmed. Old value matches. 0.8 to 1.0 m; 0.64 to 1.0 m².
- [52] sweep_fields_mec_nongrid.csv:MNG10 env_area_m2: confirmed. Old value matches. 0.8 to 1.0 m; 0.64 to 1.0 m².
- [53] sweep_fields_mec_nongrid.csv:MNG10 level_definition: confirmed. Old value matches. p. e3: days assigned "to a dorsoventral distance from the parasubiculum border" (none reported); "over 85% of the data was recorded from more dorsal regions of mEC"; layer counts 168 and 177 from Table S1.
- [54] sweep_fields_mec_nongrid.csv:MNG10 verified_in: confirmed. Old value matches. Journal PDF with STAR Methods and Tables S1-S2 read.
- [55] sweep_fields_mec_nongrid.csv:MNG11 environment_size: confirmed. Old value matches. diehl2017.pdf STAR Methods p. e4: the 80 x 80 cm area applies to the shape paradigm and to comparisons across locations; the color box is 100 x 100 cm (p. e2).
- [56] sweep_fields_mec_nongrid.csv:MNG11 env_length_m: confirmed. Old value matches. 0.8 to 1.0 m; 0.64 to 1.0 m².
- [57] sweep_fields_mec_nongrid.csv:MNG11 env_area_m2: confirmed. Old value matches. 0.8 to 1.0 m; 0.64 to 1.0 m².
- [58] sweep_fields_mec_nongrid.csv:MNG11 level_definition: confirmed. Old value matches. p. e3: days assigned "to a dorsoventral distance from the parasubiculum border" (none reported); "over 85% of the data was recorded from more dorsal regions of mEC"; layer counts 168 and 177 from Table S1.
- [59] sweep_fields_mec_nongrid.csv:MNG11 verified_in: confirmed. Old value matches. Journal PDF with STAR Methods and Tables S1-S2 read.
- [60] sweep_fields_mec_nongrid.csv:MNG12 environment_size: confirmed. Old value matches. diehl2017.pdf STAR Methods p. e4: the 80 x 80 cm area applies to the shape paradigm and to comparisons across locations; the color box is 100 x 100 cm (p. e2).
- [61] sweep_fields_mec_nongrid.csv:MNG12 env_length_m: confirmed. Old value matches. 0.8 to 1.0 m; 0.64 to 1.0 m².
- [62] sweep_fields_mec_nongrid.csv:MNG12 env_area_m2: confirmed. Old value matches. 0.8 to 1.0 m; 0.64 to 1.0 m².
- [63] sweep_fields_mec_nongrid.csv:MNG12 level_definition: confirmed. Old value matches. p. e3: days assigned "to a dorsoventral distance from the parasubiculum border" (none reported); "over 85% of the data was recorded from more dorsal regions of mEC"; layer counts 168 and 177 from Table S1.
- [64] sweep_fields_mec_nongrid.csv:MNG12 verified_in: confirmed. Old value matches. Journal PDF with STAR Methods and Tables S1-S2 read.
- [65] sweep_fields_mec_nongrid.csv:MNG10 statistic: confirmed. Old value matches. Table S1 unique counts: 61/345 = 17.7%, 32/345 = 9.3%, 233/345 = 67.5%; text p. 84 says "Of the spatial cells".
- [66] sweep_fields_mec_nongrid.csv:MNG10 value_reported: confirmed. Old value matches. 326/345 = 94.5%; of 326: 71.5%, 18.7%, 9.8%.
- [67] sweep_fields_mec_nongrid.csv:MNG10 n_cells: confirmed. Old value matches. Unique counts as above; Fig. 3D n = 103, 133, 57; Fig. 2D n = 31, 17, 8; p. 90 (Fig. 4 legend): "38 of 48 grid cells ... 109 of 236 NGS cells".
- [68] sweep_fields_mec_nongrid.csv:MNG10 location: confirmed. Old value matches. p. 84 and Table S1 pages match.
- [69] sweep_fields_mec_nongrid.csv:MNG10 notes: confirmed. Old value matches. The printed shares sum to 94.5%, the spatial share, so they are of all cells; no field size or grid spacing in the paper; Discussion quote present.
- [70] sweep_fields_mec_nongrid.csv:MNG10 grid_spacing_same_recordings_cm: confirmed. Old value matches. No grid spacing anywhere in the paper (text search).
- [71] sweep_fields_mec_nongrid.csv:MNG11 n_cells: confirmed. Old value matches. Fig. 2D and 3D counts as above; 103 + 133 = 236, 31 + 17 = 48.
- [72] sweep_fields_mec_nongrid.csv:MNG12 n_cells: confirmed. Old value matches. Fig. 2D and 3D counts as above; 103 + 133 = 236, 31 + 17 = 48.
- [73] sweep_fields_mec_nongrid.csv:MNG15 strain: confirmed. Old value matches. aav5297_butler_sm.pdf SM p. 2: "rat n = 11"; p. 3: Experimental n = 8, one excluded; p. 4: Control n = 3. Strain not stated.
- [74] sweep_fields_mec_nongrid.csv:MNG15 level_definition: confirmed. Old value matches. SM p. 2: 0.5 mm anterior to the transverse sinus (~8.5 mm posterior to bregma), 3.8-4.2 mm lateral, 2.5-3.5 mm at surgery, 0-250 µm per day; SM pp. 2-3: 598 of 778 in MEC. Module inference labelled.
- [75] sweep_fields_mec_nongrid.csv:MNG15 field_rule: corrected. Old value matches. SM p. 4 bins and smoothing, SM p. 10 remapping rule (5th percentile of 100 ENV1 split-half correlations) confirmed; "in either box" for grid and border scores is not stated. Fix in CSV.
- [76] sweep_fields_mec_nongrid.csv:MNG15 n_animals: confirmed. Old value matches. SM pp. 2-4: 8 + 3 = 11 implanted, separate control group; main text p. 1447 "seven rats".
- [77] sweep_fields_mec_nongrid.csv:MNG15 grid_spacing_same_recordings_cm: confirmed. Old value matches. Table S1: 99.14 (85.85, 107.37) and 93.07 (82.22, 104.77), p = 0.015; pooled over 102 paired cells.
- [78] sweep_fields_mec_nongrid.csv:MNG15 status: confirmed. Old value matches. Supplement read.
- [79] sweep_fields_mec_nongrid.csv:MNG15 verified_in: confirmed. Old value matches. Supplement read.
- [80] sweep_fields_mec_nongrid.csv:MNG15 notes: confirmed. Old value matches. Fig. S10A values; Table S1; Fig. S7E legend (SM p. 25, render checked): "a large fraction (49/100) of non-grid spatial cells significantly retain their tuning" against main text p. 1448 "49 of 100 control cells remapped"; 24/36, 49/102, 70/132 confirmed.
- [81] sweep_fields_rsc.csv:alexander2017_1 citation: confirmed. Old value matches. alex2017.pdf p. 1551: journal title spelled "traveled".
- [82] sweep_fields_rsc.csv:alexander2017_1 doi_or_url: confirmed. Old value matches. DOI 10.1016/j.cub.2017.04.036.
- [83] sweep_fields_rsc.csv:alexander2017_1 level_definition: confirmed. Old value matches. STAR Methods p. e2: A/P -5.8, M/L ± 0.7-1.2, D/V -0.5 mm, 10-12°; Fig. 1A sections -5.52, -5.76, -6.36 mm (render checked); Rd and Rg in Fig. S1B legend.
- [84] sweep_fields_rsc.csv:alexander2017_1 status: confirmed. Old value matches. Journal PDF read.
- [85] sweep_fields_rsc.csv:alexander2017_1 verified_in: confirmed. Old value matches. Journal PDF read.
- [86] sweep_fields_rsc.csv:alexander2017_2 citation: confirmed. Old value matches. alex2017.pdf p. 1551: journal title spelled "traveled".
- [87] sweep_fields_rsc.csv:alexander2017_2 doi_or_url: confirmed. Old value matches. DOI 10.1016/j.cub.2017.04.036.
- [88] sweep_fields_rsc.csv:alexander2017_2 level_definition: confirmed. Old value matches. STAR Methods p. e2: A/P -5.8, M/L ± 0.7-1.2, D/V -0.5 mm, 10-12°; Fig. 1A sections -5.52, -5.76, -6.36 mm (render checked); Rd and Rg in Fig. S1B legend.
- [89] sweep_fields_rsc.csv:alexander2017_2 status: confirmed. Old value matches. Journal PDF read.
- [90] sweep_fields_rsc.csv:alexander2017_2 verified_in: confirmed. Old value matches. Journal PDF read.
- [91] sweep_fields_rsc.csv:alexander2017_1 strain: confirmed. Old value matches. p. e1: "Male Long-Evans rats (n = 5)"; "2 of the 5 rats also performed track running on a W-shaped track".
- [92] sweep_fields_rsc.csv:alexander2017_2 strain: confirmed. Old value matches. p. e1: "A subset of rats (n = 3) were also trained to traverse a ring-shaped track".
- [93] sweep_fields_rsc.csv:alexander2017_1 environment_size: confirmed. Old value matches. Fig. 1B "55cm" on one of 12 equal edges, corners cut in the tracking (render checked); 12 x 55 = 660 cm nominal; edges 1 cm (p. e1); reward at the arrow point (legend).
- [94] sweep_fields_rsc.csv:alexander2017_1 env_length_m: confirmed. Old value matches. Fig. 1B "55cm" on one of 12 equal edges, corners cut in the tracking (render checked); 12 x 55 = 660 cm nominal; edges 1 cm (p. e1); reward at the arrow point (legend).
- [95] sweep_fields_rsc.csv:alexander2017_1 location: confirmed. Old value matches. Shares are on p. 1554.
- [96] sweep_fields_rsc.csv:alexander2017_1 notes: confirmed. Old value matches. p. 1554: 43/89 = 48.3%; 21/48 = 43.7%; quarter and half lengths derived.
- [97] sweep_fields_rsc.csv:alexander2017_2 environment_size: confirmed. Old value matches. Fig. S4F "122 cm" (render checked); pi x 1.22 = 3.83 m; p. e1 ring centred on the plus position, 1 cm edges; clockwise analysed (p. e2).
- [98] sweep_fields_rsc.csv:alexander2017_2 env_length_m: confirmed. Old value matches. Fig. S4F "122 cm" (render checked); pi x 1.22 = 3.83 m; p. e1 ring centred on the plus position, 1 cm edges; clockwise analysed (p. e2).
- [99] sweep_fields_rsc.csv:alexander2017_2 familiarity: confirmed. Old value matches. p. e1: "70% of recordings had plus-track session first"; p. e2: "14 and 13".
- [100] sweep_fields_rsc.csv:alexander2017_2 location: confirmed. Old value matches. pp. 1555-1556.
- [101] sweep_fields_rsc.csv:alexander2017_2 notes: confirmed. Old value matches. pp. 1555-1556: 13.3% against 21%, 44.4% against 20.5%, "70% (n = 63/90)", 9.99 ± 9.41 cm.
- [102] sweep_kernel_rsc.csv:alexander2017_k1 citation: confirmed. Old value matches. alex2017.pdf p. 1551: journal title spelled "traveled".
- [103] sweep_kernel_rsc.csv:alexander2017_k1 doi_or_url: confirmed. Old value matches. DOI 10.1016/j.cub.2017.04.036.
- [104] sweep_kernel_rsc.csv:alexander2017_k1 environment_size: confirmed. Old value matches. Fig. 1B "55cm" on one of 12 equal edges, corners cut in the tracking (render checked); 12 x 55 = 660 cm nominal; edges 1 cm (p. e1); reward at the arrow point (legend).
- [105] sweep_kernel_rsc.csv:alexander2017_k1 env_length_m: confirmed. Old value matches. Fig. 1B "55cm" on one of 12 equal edges, corners cut in the tracking (render checked); 12 x 55 = 660 cm nominal; edges 1 cm (p. e1); reward at the arrow point (legend).
- [106] sweep_kernel_rsc.csv:alexander2017_k1 measure: confirmed. Old value matches. p. e4: "Correlation matrices were computed using Spearman's correlations."
- [107] sweep_kernel_rsc.csv:alexander2017_k1 normalization: confirmed. Old value matches. No per-cell normalization stated for the matrices; Fig. 1C left is max-normalized for display.
- [108] sweep_kernel_rsc.csv:alexander2017_k1 simultaneous_or_pooled: confirmed. Old value matches. Fig. 1C legend n = 229; p. 1552: 229 of 315; p. e4: 5 to 140 neuron sampling is for distance reconstruction (Fig. 4F).
- [109] sweep_kernel_rsc.csv:alexander2017_k1 n_cells: confirmed. Old value matches. Fig. 1C legend n = 229; p. 1552: 229 of 315; p. e4: 5 to 140 neuron sampling is for distance reconstruction (Fig. 4F).
- [110] sweep_kernel_rsc.csv:alexander2017_k1 width_r05_full_cm: confirmed. Old value matches. Own re-read of Fig. 1C: mean r ≥ 0.714 at every separation (bar ends) or ≥ 0.718 (tick labels); never near 0.5 or 0.
- [111] sweep_kernel_rsc.csv:alexander2017_k1 dist_r05_cm: confirmed. Old value matches. Own re-read of Fig. 1C: mean r ≥ 0.714 at every separation (bar ends) or ≥ 0.718 (tick labels); never near 0.5 or 0.
- [112] sweep_kernel_rsc.csv:alexander2017_k1 zero_cross_cm: confirmed. Old value matches. Own re-read of Fig. 1C: mean r ≥ 0.714 at every separation (bar ends) or ≥ 0.718 (tick labels); never near 0.5 or 0.
- [113] sweep_kernel_rsc.csv:alexander2017_k1 floor: confirmed. Old value matches. Own re-read: 0.769 (bar ends) to 0.778 (labels) at 100 to 330 cm; minimum 0.714 to 0.718 at about 486 cm.
- [114] sweep_kernel_rsc.csv:alexander2017_k1 far_behaviour: corrected. Old value matches. Own re-read confirms 0.91/0.93 peak, 0.88, 0.82, 0.79 at 27, 55, 82 cm, half-drop 45 cm, dip 0.78 at 122 cm, 0.79 at 166 to 186 cm, 0.74 at 280 cm, 0.78 at 360 to 380 cm. Two problems: the quarter-lap rise is only about 0.01, and "start and end ... (0.72 near 490 cm)" misreads the minimum, which is not the start/end corner (that reads 0.75 to 0.76). Fixes in CSV.
- [115] sweep_kernel_rsc.csv:alexander2017_k1 location: confirmed. Old value matches. pp. 1552, 1557-1558, e3-e4.
- [116] sweep_kernel_rsc.csv:alexander2017_k1 quote_or_reading: confirmed. Old value matches. Journal quotes present (pp. 1552-1553, 1557); the preprint sentence "significant correlation peaks were observed..." is absent from the journal text.
- [117] sweep_kernel_rsc.csv:alexander2017_k1 status: confirmed. Old value matches. Journal and Fig. 1C read.
- [118] sweep_kernel_rsc.csv:alexander2017_k1 verified_in: confirmed. Old value matches. Journal and Fig. 1C read.
- [119] sweep_kernel_rsc.csv:alexander2017_k1 notes: confirmed. Old value matches. p. 1556: Fig. 4E distance reconstruction, 2.93 cm RSC against 12.61 cm HPC, n = 72. The "2.09 cm" and "Fig. 6" of the old note are wrong.

## Review edits (pft2_review_edits.md)

- Edit 1 (Opening summary (first paragraph)): corrected. Old text occurs once. "6.6 m" is right. "Retrosplenial similarity never falls to 0.5" is overstated: the between-session RSC curve (Fig. 6a) reaches 0.50 at 65 to 75 cm. The same paragraph still says no dorsal CA1 curve exists beyond a 4 m loop; see new edit A1.
- Edit 2 (Retrosplenial kernel table in the second section (row for Alexander & Nitz 2017)): corrected. Columns match the table. Peak 0.82 to 0.86 and half-drop 20 to 23 cm (row 1); between-session RSC minimum is 0.50, so "never falls to 0.5" fails for row 2; the plus-lap quarter rise is about 0.01 (row 3). 187/228 cell counts and 2017 values confirmed.
- Edit 3 (Similarity section, opening paragraph): corrected. As edit 1; the 26.5 cm CA1 curve is between sessions. Line 163 of the review ("The test has not been run on dorsal CA1 in environments longer than a 4 m loop") then contradicts it; see new edit A2.
- Edit 4 ("Thirteen measures at a glance" table, PV similarity row): corrected. "Stays above 0.5" as in edit 1. The same row keeps "no dorsal CA1 curve for spaces above a 4 m loop" and a study list without Alexander & Nitz; see new edits A3 and A4.
- Edit 5 ("Thirteen measures at a glance" table, field width over grid spacing row): corrected. Ratios confirmed (56.1/36.26 = 1.55; 39.5/99.14 = 0.40). Setting them side by side without the rule and statistic difference overstates comparability; add a qualifier.
- Edit 6 ("Thirteen measures at a glance" table, field width over grid spacing row (studies)): confirmed. Adds the source.
- Edit 7 (Field-width grid, rat table (after the last row)): confirmed. Ratios checked: 39.5 and 49.9 cm against M1 to M5 give 0.81 to 1.3, 0.66 to 1.0, 0.42 to 0.77, 0.31 to 0.51, 0.23 to 0.37; 10 columns; bold group row matches the mouse table style. Optional. If applied, the mouse-grid closing paragraph (edit 17) should mention the new rat row.
- Edit 8 (Field-width grid, rat sources table (after the last module-free row)): confirmed. Five columns; values and exclusions match MNG33 to MNG35 and Diehl. Optional.
- Edit 9 (Field-width grid, mouse table, dorsal CA1 real-environment row): confirmed. 19.7 cm in 50 or 62 cm squares fits the class and lies inside 14.8 to 42.5 cm, so the ratio cells do not change.
- Edit 10 (Field-width grid, mouse table, dorsal CA1 real-environment row (study list)): confirmed. Alphabetical.
- Edit 11 (Field-width grid, mouse table, MEC non-grid row label): confirmed. Online Methods p. 14.
- Edit 12 (Field-width grid, mouse table, border row label): confirmed. Border cells come from the same MEC mice and arenas.
- Edit 13 (Field-width grid, mouse sources table, dorsal CA1 real-environment row (count)): confirmed. 18 + 1.
- Edit 14 (Field-width grid, mouse sources table, dorsal CA1 real-environment row (contributing studies)): confirmed. 303.75 cm² (Fig. 2b) gives 19.7 cm.
- Edit 15 (Field-width grid, mouse sources table, dorsal CA1 real-environment row (exclusions)): confirmed. The exclusion reason no longer holds.
- Edit 16 (Field-width grid, mouse sources table, MEC non-grid row): confirmed. As edit 11.
- Edit 17 (Mouse field-width grid, closing paragraph): confirmed. Accurate. It drops the old reason sentence, which is fine.
- Edit 18 (Kernel grid, "grids leave out" paragraph): corrected. CA1 widths 53 cm (2 x 26.5) and 120 to 126 cm confirmed by re-read. "Never fall to 0.5" as in edit 1.
- Edit 19 (Kernel grid, rat table (optional rows after Farooq 2024)): confirmed. Ratios checked: 53 cm gives 1.1 to 1.4, 0.88 to 1.1, 0.57 to 0.82, 0.41 to 0.55, 0.31 to 0.39; 120 to 126 cm gives 2.4 to 3.2, 2.0 to 2.6, 1.3 to 1.9, 0.94 to 1.3, 0.70 to 0.93. Optional. The 53 cm assumes 1.04 cm per template bin.
- Edit 20 (Kernel grid, sources table (optional rows after Farooq & Dragoi 2024)): confirmed. Sources match k3 and 2017_k2. Optional.
- Edit 21 (Retrosplenial section heading): corrected. Sentence case. "Rat RSC fields are wider" generalizes one paired study (Smith 2012); "stays above 0.5" as in edit 1.
- Edit 22 (Retrosplenial section, opening paragraph): confirmed. Counts match: 6 RSC field rows, 6 kernel rows, 3 corrected RSC rows (alexander2017_1, _2, _k1).
- Edit 23 (Retrosplenial kernel subsection heading): corrected. "Never fall to 0.5" as in edit 1. "Mouse retrosplenial curves" (Mao subsets) is fine.
- Edit 24 (Retrosplenial kernel subsection, rat paragraph): corrected. Most numbers confirmed by re-read (CA1 26.5 cm and floor 0.09; 2017 CA1 60 to 63 cm; 1.9 ± 2.7 against 5.2 ± 6.7 cm; no per-cell normalization in either paper). Fix "never falls to 0.5", the 0.51 floor, the 20 cm half-drop and the 0.79 quarter-lap rise.
- Edit 25 (Retrosplenial kernel subsection, last sentence of the rat paragraph): confirmed. sweep_kernel_rsc.csv holds no arena kernel; 6.6 m is the longest.
- Edit 26 (Retrosplenial section, gaps paragraph): confirmed. Neither Alexander paper reports RSC field widths.
- Edit 27 (Retrosplenial section, full citations): confirmed. Citations match the PDFs.
- Edit 28 (Section on similarity curves falling to half within a field width, "Other regions" paragraph): corrected. CA1 "27 to 63 cm" confirmed (26.4 and 60 to 63). "Stayed above 0.5" needs "between odd and even trials".
- Edit 29 (Conclusion): corrected. As edit 28; the normalization sentence needs an inference tag.
- Edit 30 (Non-grid MEC section heading): corrected. Sentence case. The side-by-side numbers need "under different rules", and the grid-scale test is mouse-only.
- Edit 31 (Non-grid MEC section, opening paragraph): confirmed. Two studies; file names as proposed.
- Edit 32 (Non-grid MEC subsection heading): confirmed. Sentence case.
- Edit 33 (Non-grid MEC, limits of the 1.55): corrected. Arenas and arithmetic confirmed; no non-grid rule confirmed. Tag "probably caps" as inference; the place-field rule says "reached 20%".
- Edit 34 (Non-grid MEC, new rat paragraph after the knockout paragraph): corrected. Numbers confirmed (778, 598, 0.85/0.8, 6 bins, 1225 cm², 39.5 cm, 2 fields, 13%, 1958 cm², 49.9 cm, 99.14 cm, 102 cells, six modules in four rats, 65 to 122 cm, 0.40, 0.50, 0.61, 1420 cm², 0.86, 1106 cm², p = 0.35). Three fixes: 778 is cells recorded in both boxes, not all recorded; the grid field method link is inferred; the mouse-rat contrast leaves out grid spacing, mean against median and class strictness (see the fairness note above).
- Edit 35 (Non-grid share paragraph): confirmed. Range 6.1% (Mallory) to 67.5% (Diehl, of all cells).
- Edit 36 (Non-grid share table, Diehl row (and a new Butler row)): confirmed. Diehl counts from Table S1; Butler 271/778 = 34.8%, 102/778 = 13.1%; 180 non-MEC cells noted.
- Edit 37 (Lateral septum and other regions paragraph (Diehl sentence)): confirmed. Shares of all 345 cells; both studies cited.
- Edit 38 (Stability and remapping paragraph, Butler sentence): confirmed. 196/271, 0.32 ± 0.22, 0.41 ± 0.27, 3 separate control rats, 49/100, and the Fig. S7E conflict all confirmed.
- Edit 39 (Dorsoventral subsection, Diehl sentence): confirmed. STAR Methods p. e3 wording matches.
- Edit 40 (Further-gaps paragraph): corrected. Diehl Table S1 holds counts only (confirmed). "High threshold" is an inference; name the threshold instead.
- Edit 41 (Non-grid MEC full citations, Mallory): confirmed. Online Methods read.
- Edit 42 (Non-grid MEC full citations, Butler): confirmed. Supplement read.

## Missing review edits (in the CSV as new edits A1 to A4)

- A1, review line 3: "No one has published this curve for dorsal CA1 in an environment longer than a 4 m loop" is false once Alexander & Nitz 2017 Fig. 1D is read.
- A2, review line 163: "The test has not been run on dorsal CA1 in environments longer than a 4 m loop" contradicts edit 3.
- A3, review line 177: "no dorsal CA1 curve for spaces above a 4 m loop" sits in the row that edit 4 changes.
- A4, review line 177: the row's study list needs Alexander & Nitz 2015 and 2017.

