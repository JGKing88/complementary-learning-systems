# Page build from the pft2 full-text pass: steps, counts and decisions

Written 2026-11-24. These notes record how the audited results of the seven-paper full-text pass (pft2) were folded into the sweep tables, `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md` and its page. Inputs, all in this folder: `pft2_additions_fields_rsc.csv`, `pft2_additions_kernel_rsc.csv`, `pft2_additions_fields_mec.csv`, `pft2_additions_fields_mouse.csv`, `pft2_corrections.csv`, `pft2_review_edits.md`, `audit_pft2.md` and `audit_pft2_fixes.csv`. Every rule of `page_build_sweeps.md` and `page_build_rsc_mec.md` holds unless a section below says otherwise. Neural data only. Nothing was committed or published.

Scratch scripts and backups are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/page_build_pft2/`: `fold_fixes.py` (step 1), `append_rows.py` (step 2), `apply_edits.py`, `make_overrides.py` and `overrides_8_20.json` (step 4), `splice_grids.py` (grid splice), `account_changes.py` (replays every audited edit on the backup and lists what is left), and `backup/` with the pre-build copies of every changed file.

## Files changed

- Sweep tables: `sweep_fields_rsc.csv`, `sweep_kernel_rsc.csv`, `sweep_fields_mec_nongrid.csv` and `sweep_fields_mouse.csv` (rows appended, cells corrected). `sweep_kernel.csv` is untouched: no addition or correction targets it, so its CRLF file is byte-identical.
- New proposal copies with the audit's fixes: `pft2_additions_fields_rsc_fixed.csv`, `pft2_additions_kernel_rsc_fixed.csv`, `pft2_additions_fields_mec_fixed.csv`, `pft2_additions_fields_mouse_fixed.csv` (identical to its original, since no fix targets it), `pft2_corrections_fixed.csv` and `pft2_review_edits_fixed.md`. The originals are kept.
- `analysis/literature_page/aggregate_sweeps.py` and the tracked `analysis/literature_page/ratio_spec.json`.
- `analysis/literature_page/heatmap.py`: one caption sentence. `analysis/literature_page/build_report_page.py`: one footer sentence.
- The review and `docs/run_pages/place_fields_population_similarity.html`.

## Step 1: audit fixes folded into the proposals

All 54 lines of `audit_pft2_fixes.csv` were applied by exact phrase. Every phrase occurred exactly once in its cell or edit text, so nothing stopped the script.

| Target | Fix lines | Cells or edits changed |
|---|---|---|
| pft2_additions_fields_rsc.csv | 1 | alexander2017_4 measure (adds "Spearman") |
| pft2_additions_kernel_rsc.csv | 11 | 9 cells: alexander2015_k1 far_behaviour and bin_cm, alexander2015_k2 floor, dist_r05_cm and far_behaviour, alexander2015_k3 bin_cm, alexander2015_k4 floor and far_behaviour, alexander2017_k2 notes |
| pft2_additions_fields_mec.csv | 5 | 2 cells: MNG33 notes (4 fixes), MNG36 notes |
| pft2_additions_fields_mouse.csv | 0 | none |
| pft2_corrections.csv | 11 | new_value of 10 corrections, [10], [17], [24], [31], [38], [45], [47], [49], [75] and [114] (two fixes); each fix's index and row id were checked against the file |
| pft2_review_edits.md | 22 | New text of 15 edits: 1, 2, 3, 4, 5, 18, 21, 23, 24, 28, 29, 30, 33, 34 and 40 |
| New edits A1 to A4 | 4 | appended to `pft2_review_edits_fixed.md` as sections "A1" to "A4" with Old and New blocks |

A cell-by-cell comparison of each original and fixed CSV shows changes only in the cells listed. The fixed CSVs keep CRLF line ends and round-trip through Python's csv module byte for byte, as the originals do.

## Step 2: addition rows appended

| Target | Rows before | Added | Rows after | Ids added |
|---|---|---|---|---|
| sweep_fields_rsc.csv | 66 | 6 | 72 | alexander2015_1, _2, _3, alexander2015_ca1_1, alexander2017_3, _4 |
| sweep_kernel_rsc.csv | 11 | 6 | 17 | alexander2015_k1 to _k4, alexander2017_k2, _k3 |
| sweep_fields_mec_nongrid.csv | 32 | 8 | 40 | MNG33 to MNG40 |
| sweep_fields_mouse.csv | 91 | 1 | 92 | mallory2018_2 |

Each fixed addition file has exactly its target's header (32, 31, 33 and 31 columns), every row has the header's cell count, and no new id occurs in any of the seven sweep tables. The rows were appended as bytes after the target's final CRLF, and each result parses back to the old rows followed by the new ones and round-trips byte for byte. `sweep_fields_mec_nongrid.csv` was pure ASCII before; the Butler and Diehl rows bring non-ASCII characters (², ±), so it is now UTF-8 like the other tables.

## Step 3: corrections

`apply_corrections.py` ran once per target with `pft2_corrections_fixed.csv` and the target's prefix.

| Target | Applied | Conflicts | Unknown ids |
|---|---|---|---|
| sweep_fields_mouse.csv | 11 | 0 | 0 |
| sweep_fields_mec_nongrid.csv | 70 | 0 | 0 |
| sweep_fields_rsc.csv | 21 | 0 | 0 |
| sweep_kernel_rsc.csv | 18 | 0 | 0 |

120 of 120 applied. A comparison with the backups shows that exactly the 120 (row, column) cells named by the corrections changed, in 13 rows: mallory2018_1; MNG01 to MNG05, MNG10 to MNG12 and MNG15; alexander2017_1 and _2; alexander2017_k1. All four files keep CRLF line ends.

## Step 4: review edits

`apply_edits.py` checks before writing that every chosen old text occurs exactly once and that no two overlap, then applies the edits in order and checks each old text again at its turn.

- The 38 non-optional edits (1 to 6, 9 to 18, 21 to 42) were applied in numeric order, then A1 to A4: 42 edits, no failure. Edit 4 and A3 sit in the same table row without overlapping; edit 1 and A1 sit in the same paragraph.
- Optional edits 7 and 19 (heat-map rows) came in through the regenerated grids (step 5). The rat field-width rows that the splice inserted equal edit 7's New text character for character. The two rat kernel rows have edit 19's cells exactly, but their labels read "(52.8 to 53 cm)" and "(1.20 to 1.26 m)" instead of "(53 cm)" and "(120 to 126 cm)": the first because the row spans the sweep and audit readings (see step 5), the second because `ratio_tables.py` prints ranges of 1 m and more in metres.
- Optional edits 8 and 20 (source tables) were applied with changes, so they match the grids. Edit 8 gains a "**MEC, non-grid cells**" group row before the Butler row, as in the mouse source table, and its grid-spacing clause now reads "grid spacing from the same data set has a median of 99.14 cm, a ratio of 0.40, but it pools 102 grid cells and is not paired with these cells by rat" (the brief asked for this; the audit makes the same point for MNG33). Edit 20 gives the one-sided readings as "26.5 cm sweep, 26.4 cm audit" and "about 60 to 63 cm sweep, 60.6 cm audit", and the full widths as the grid's "52.8 to 53 cm" and "1.20 to 1.26 m".
- No edit was left unapplied.

## Step 5: aggregate_sweeps.py

Changes:

1. mallory2018_1 leaves `MOUSE_EXCLUDE` and `MOUSE_ATTACH`, with a comment. Its corrected size, 50 or 62 cm squares (0.5 to 0.62 m, 0.25 to 0.38 m²), puts it in the mouse class "real environment up to 2 m". mallory2018_2 states its size, a 100 × 7 cm linear track of which the central 70 cm was analysed, so it enters the same row; no exclusion was needed.
2. MEC config: MNG01 and MNG04 take the class "70 to 100 cm arena", labelled "real open arena of 70 to 100 cm" (edits 11 and 12); the old "unstated" class is gone. MNG33 and MNG35 form a rat row, region "MEC and nearby cortex, non-grid spatial cells, level unstated", class "1.5 m box", labelled "real 1.5 m arena" (edit 7). Named exclusions: MNG34 (reward-task box, a manipulation; it would belong to the Butler row), MNG36 (grid cells' field size, the comparator), MNG40 (share of the box covered by a field, 35.17%). MNG37 to MNG39 carry no width and get the generic reason. Comments record that Butler's 99.14 cm spacing pools 102 grid cells of the same data set and is not paired by rat, and that the rat grid's columns stay the Stensola modules.
3. `aggregate_extra` now keys rows by species, region and class, and tags each output row with its species; rat rows must carry a class override. `main` sends mouse group rows to the mouse grid and rat group rows to the rat grid, after the hippocampal rows. The RSC sweep's rat rows still give no row: Smith 2012 stays excluded and the Alexander & Nitz field rows carry no width.
4. Kernels: `kernel_rows` accepts ids from `sweep_kernel.csv` and `sweep_kernel_rsc.csv`. Two rows follow Farooq 2024 in the rat list: alexander2015_k3, "between sessions, 2.15 m route", 52.8 to 53 cm (one side 26.5 cm in the sweep row, 26.4 cm in the audit), and alexander2017_k2, "Spearman, 6.6 m lap", 120 to 126 cm (one side about 60 to 63 cm in the sweep; the audit's tick-label reading, 60.6 cm, lies inside). A new dict, `KERNEL_EXCLUDE`, names the five rat RSC curves left out:

| Row id | Reason |
|---|---|
| alexander2015_k1 | rat RSC, odd against even trials on the 2.15 m W-shaped route; never reaches 0.5 (minimum 0.53 to 0.56), so no r = 0.5 width |
| alexander2015_k2 | rat RSC between two sessions at one room location on the 2.15 m route; it only touches 0.50 to 0.503 at 65 to 75 cm and 136 to 176 cm, within reading error, with no clean crossing, and it compares two sessions, not one environment within a session |
| alexander2015_k4 | alpha against beta, the same route after the track moved to another room location, so not one environment; no r = 0.5 width (RSC 0.54 to 0.60 out to 150 cm, CA1 0.10 to 0.21) |
| alexander2017_k1 | rat RSC, odd against even trials on the 6.6 m plus-shaped lap; never below 0.71, so no r = 0.5 width |
| alexander2017_k3 | rat RSC, odd against even trials on the 3.83 m ring; the colour bar spans 0.7 to 1 and r stays at or above 0.83, so no r = 0.5 width |

5. `--report` prints rat group rows with "(rat grid)" after the label and adds a section listing `KERNEL_EXCLUDE`. The docstring names the new sources.

Everything else in the script is unchanged. Reruns: two runs on scratch copies of the spec and two runs on the tracked `ratio_spec.json` gave byte-identical specs (sha256 5077b9b1394cfb2a0d194783fa915427b35621ba888ccb41ceffe7bed3fe97f3), identical reports and identical printed row lists. Outside the rows listed below, every spec entry, column and row order is unchanged.

### Heat-map rows added, changed or removed

Cells are row low over column high to row high over column low. Rat columns: M1 39 to 49 cm, M2 48 to 60, M3 65 to 93, M4 97 to 128, M5 136 to 172; the three 18 m track columns print "not comparable" for these rows. Mouse M1 30 to 36 cm.

| Grid | Row | N | Range | Change |
|---|---|---|---|---|
| Rat field width | "MEC, non-grid cells" group header | | | added |
| Rat field width | MEC and nearby cortex, non-grid spatial cells, level unstated, real 1.5 m arena (Butler 2019: MNG33 39.5 cm, MNG35 49.9 cm) | 1 | 39.5 to 49.9 cm | added: 0.81 to 1.3, 0.66 to 1.0, 0.42 to 0.77, 0.31 to 0.51, 0.23 to 0.37 |
| Rat kernel | Dorsal CA1, Alexander & Nitz 2015, between sessions, 2.15 m route | 1 | 52.8 to 53 cm | added: 1.1 to 1.4, 0.88 to 1.1, 0.57 to 0.82, 0.41 to 0.55, 0.31 to 0.39 |
| Rat kernel | Dorsal CA1, Alexander & Nitz 2017, Spearman, 6.6 m lap | 1 | 1.20 to 1.26 m | added: 2.4 to 3.2, 2.0 to 2.6, 1.3 to 1.9, 0.94 to 1.3, 0.70 to 0.93 |
| Mouse field width | Dorsal CA1, real environment up to 2 m | 18 to 19 | 14.8 to 42.5 cm, unchanged | Mallory 2018 joins (mallory2018_1 19.7 cm, mallory2018_2 23.0 cm, one study); cells unchanged, 0.41 to 1.4, 0.25 to 1.0, 0.19 to 0.65 |
| Mouse field width | Dorsal MEC non-grid spatial cells | 1 | 56.1 cm | label: "real open arena of unstated size" became "real open arena of 70 to 100 cm"; cells unchanged |
| Mouse field width | Dorsal MEC border cells, width perpendicular to the wall | 1 | 14.4 cm | label as above; cells unchanged |

No row was removed. The mouse kernel grid is unchanged. The grid splice replaced the review's four tables with the output of `ratio_tables.py`; the rat grid gained 2 lines and the rat kernel grid 2 lines, and the mouse grids did not change, because edits 9 to 12 had already put the same text there.

Arithmetic: Butler 39.5/49 = 0.81, 49.9/39 = 1.28, 39.5/60 = 0.66, 49.9/48 = 1.04, 39.5/93 = 0.42, 49.9/65 = 0.77, 39.5/128 = 0.31, 49.9/97 = 0.51, 39.5/172 = 0.23, 49.9/136 = 0.37; against Butler's own spacing 39.5/99.14 = 0.40 and 49.9/99.14 = 0.50. Alexander 2015 52.8/49 = 1.08, 53/39 = 1.36, 52.8/60 = 0.88, 53/48 = 1.10, 52.8/93 = 0.57, 53/65 = 0.82, 52.8/128 = 0.41, 53/97 = 0.55, 52.8/172 = 0.31, 53/136 = 0.39. Alexander 2017 120/49 = 2.45, 126/39 = 3.23, 120/60 = 2.0, 126/48 = 2.63, 120/93 = 1.29, 126/65 = 1.94, 120/128 = 0.94, 126/97 = 1.30, 120/172 = 0.70, 126/136 = 0.93. Kernels up to about 4 m still span 16.8 to 54 cm, 0.34 to 1.4 smallest spacings; without Terrazas and Alexander 2015 the top is Ledergerber's 45 cm, 45/39 = 1.15.

## Step 6: review text checked against the grids

The audited edits cover the "4 m loop" statements (A1 to A3) and the mouse labels and counts (edits 9 to 17). `account_changes.py` replays every audited edit and the grid splice on the backup and compares the result with the review. Twelve changes remain, all made by hand for this step:

1. Second section intro: the retrosplenial kernel rows now cite audit_pft2.md as well, and one sentence says the Alexander & Nitz rows also hold the CA1 curves of the same rats, so "each table lists every measurement found" stays true for CA1.
2. Field-width intro: one sentence counts the pft2 rows (1 mouse, 6 retrosplenial and 8 non-grid MEC field rows added, 12 field rows corrected) and one says Mallory's CA1 widths join the mouse grid and the rat grid gains the Butler row.
3. Rat source table, "Not in any row": adds Butler 2019 (MNG36), grid cells' field size.
4. Mouse source table, dorsal CA1 real row: the Mallory entry from edit 14 adds "and 23.0 cm, the mean field length on a 1 m linear track [full text, supplement]" for mallory2018_2.
5. Mouse source table, "Not in any row": adds Mallory 2018 (MNG40), share of the box covered by a field.
6. Mouse closing paragraph (edit 17's sentence): "its CA1 width, 19.7 cm, now sits" became "its CA1 widths, 19.7 cm in those arenas and 23.0 cm on a 1 m linear track, now sit", tagged [full text, supplement].
7. Same paragraph: four sentences on the rat grid's Butler row, 0.81 to 1.3 M1 spacings, and on its own grid spacing, a median of 99.14 cm that pools 102 grid cells of the same data set and is not paired by rat, which gives 0.40 to 0.50. The audit asked for this mention when edit 7 is applied.
8. Kernel intro: one sentence names sweep_kernel_rsc.csv and audit_pft2.md as the source of the two rat CA1 rows.
9. Kernel ranges: "Without that row the range is 17 to 45 cm" was no longer true with the 53 cm row. The paragraph now names the 52.8 to 53 cm Alexander & Nitz 2015 row, says "Without those two rows", and adds the 6.6 m lap row, 1.20 to 1.26 m and 2.4 to 3.2 smallest spacings, with "Spearman correlation and 7 cm smoothing may widen that curve [inference]" from edit 19's audited reason. The section heading ("0.3 to 1.4 smallest grid spacings up to about 4 m") still holds, since the 6.6 m lap is over 4 m.
10. Kernel "grids leave out" paragraph (edit 18's sentence): "The CA1 curves recorded with them, which the rat grid holds, are 53 cm wide", so the paragraph does not read as if those curves were left out.
11. Kernel sources paragraph: three sentences on colour-bar reading and the bin scale, 1.04 cm per template bin against the Methods' ~1.2 cm, which would make the 2015 distances up to 15% larger (audit_pft2_fixes.csv lines 4 and 5, and the audit's note on edit 19).
12. Similarity section: "Among environments longer than a few metres, only the 18 m track has a measured PV curve" became "Among environments longer than the 6.6 m lap of Alexander and Nitz, only the 18 m track has a measured PV curve", the same kind of statement as A1 to A3.

Checked and unchanged: the opening and second-section statements that rodent curves fall to 0.5 within 5 to 28 cm hold for spaces up to about 4 m, which excludes the 6.6 m lap; the summary-table ranges (rat 0.094 to 1.9, mouse 0.19 to 1.4, kernels 0.34 to 1.4 up to about 4 m) still bound every row of their class; the field-width section's 0.33 to 1.9 and 0.41 to 1.4 still hold; the "Fields per cell per grid period" heading is unchanged.

Page code: the field heat-map caption now says the rat grid ends with one non-grid entorhinal row (`heatmap.py`), and the footer gains a sentence dated 24 November 2026 (`build_report_page.py`), as the previous build added one for its sweeps.

## Step 7: rebuild and checks

`build_report_page.py` wrote the page: 514 KB, 22 sections, 1,067 chips, 696 links, 28 tables. Checks on the page (the earlier `check_page.py`, with esprima from the job's pylib):

- The one inline script parses.
- 60 ids, no duplicates; 24 anchors, none dangling.
- Tags balance: nothing unclosed, no mismatched closing tag.
- Heat maps: rat fields 9 columns and 16 rows, mouse fields 4 and 20, rat kernels 9 and 20, mouse kernels 4 and 4. Every row has its header's cell count, and every group header row spans the full width. The rat field map now has the groups Dorsal, Intermediate, Ventral and "MEC, non-grid cells".
- The section title "Fields per cell per grid period are 0.025 to 0.036 on long tracks, counting every cell" is unchanged, and the top link still points to it.
- No em dash, en dash or curly quote in the review, the page or `pft2_review_edits_fixed.md`; no stray `**` or unconverted link on the page. Every markdown table in the review has a constant cell count.

## Judgment calls and open items

- The 2015 CA1 kernel spans 52.8 to 53 cm, following the rule that a row spans the sweep and audit readings (26.5 and 26.4 cm one side), as K16 did. The printed cells are the same as for 53 cm alone. Use 53 alone by changing the tuple in `KERNELS` if one value is preferred.
- The brief's second reason for leaving out alexander2015_k2, that it compares two sessions, applies equally to the CA1 curve alexander2015_k3, which the brief puts in the grid. The script therefore leads with the missing clean crossing for k2 and gives the two-session point second, and the rat kernel label says "between sessions".
- The Butler row's label keeps the audited wording and does not mention pooling; the pooled, unpaired grid spacing is stated in the source table and the closing paragraph. The heat-map columns are the Stensola modules, so the cells do not depend on Butler's spacing.
- Edit 1, as fixed by the audit, gives the sentence "Retrosplenial similarity between odd and even trials stays above 0.5 on routes of 2.15 and 6.6 m, and between two sessions on the 2.15 m route it levels off at about 0.5, and CA1 in the 6.6 m study, read with Spearman correlation, falls to 0.5 only about 60 cm away." It was applied as audited. Ending the sentence after "about 0.5" would read more easily.
- Edit 2's between-session row keeps "CA1 peak 0.72". The audit's re-read gives 0.69 to 0.70 but its fixes file does not change this, so it stays.
- Not added, because no audited edit proposes it: Butler's median of 2 fields per non-grid cell in the 2.25 m² box as a fields-per-period row.
- `page_build_rsc_mec.md` judgment call 1 (Mallory's MEC rows kept despite an unknown arena) is now settled by the Online Methods; that file was not edited. `sweep_rsc.md` and `sweep_mec_nongrid.md` were not edited either.
- No browser was available, so the page was checked structurally only.

## Appendix: new and changed report sections

From `aggregate_sweeps.py --report`. Sections not shown are unchanged from `page_build_rsc_mec.md`, apart from mallory2018_1 leaving the mouse exclusion table.

### Mouse: changed row

| Row | N | Range | New contributions |
|---|---|---|---|
| Dorsal CA1, real environment up to 2 m, 19 studies | 19 | 14.8 to 42.5 cm | mallory2018_1: 19.7 cm, square arenas 50 or 62 cm wide; mallory2018_2: 23 cm, linear track 100 x 7 cm |

### MEC, non-grid cells: rows and contributions

| Row | N | Range | Contributions (row id: value, environment) |
|---|---|---|---|
| Dorsal MEC non-grid spatial cells, real open arena of 70 to 100 cm, 1 study | 1 | 56.1 cm | MNG01: 56.1 cm, square arenas with 50 cm walls, 70, 90 or 100 cm wide |
| Dorsal MEC border cells, width perpendicular to the wall, real open arena of 70 to 100 cm, 1 study | 1 | 14.4 cm | MNG04: 14.4 cm, as above |
| MEC and nearby cortex, non-grid spatial cells, level unstated, real 1.5 m arena, 1 study (rat grid) | 1 | 39.5 to 49.9 cm | MNG33: 39.5 cm; MNG35: 49.9 cm; 150 x 150 cm boxes, foraging box (ENV1) |

### New excluded rows

| Row id | Region, level, species | Reason | Would belong to |
|---|---|---|---|
| alexander2015_1 | RSC-unspecified, posterior, rat | no width in cm (counts, fractions, ratios, angles or stability only) | no row |
| alexander2015_2 | RSC-unspecified, posterior, rat | no width in cm | no row |
| alexander2015_3 | RSC-unspecified, posterior, rat | no width in cm | no row |
| alexander2015_ca1_1 | CA1, dorsal, rat | no width in cm | no row |
| alexander2017_3 | CA1, dorsal, rat | no width in cm | no row |
| alexander2017_4 | RSC-unspecified, posterior, rat | no width in cm | no row |
| MNG34 | MEC and surrounding cortex, unspecified, rat | the same cells in the reward-task box (ENV2), a manipulation; the foraging-box rows MNG33 and MNG35 are used | MEC and nearby cortex, non-grid spatial cells, level unstated, 1.5 m box |
| MNG36 | MEC and surrounding cortex, unspecified, rat | grid cells' field size, the comparator for MNG33 to MNG35, not a non-grid width | no row |
| MNG37 | MEC and surrounding cortex, unspecified, rat | no width in cm | no row |
| MNG38 | MEC and surrounding cortex, unspecified, rat | no width in cm | no row |
| MNG39 | MEC, dorsal, rat | no width in cm | no row |
| MNG40 | MEC, dorsal, mouse | share of the box covered by a field (35.17%), not a field width | no row |
| MNG03 | MEC, dorsal, mouse | unchanged reason (HCN1 knockout); "would belong to" now reads "Dorsal MEC non-grid spatial cells, 70 to 100 cm arena" | |

The kernel exclusions are in the table under step 5.
