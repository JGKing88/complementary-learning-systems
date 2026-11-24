# Proposed review edits from the pft2 full-text pass

Date: 2026-11-24. Target: `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`. Each old text below occurs exactly once in the current review (checked by script). Apply the edits in any order. Edits marked "Optional" add heat-map rows and need a page rebuild; the others fix statements that the new numbers make wrong or incomplete. Numbers and quotes are in `fulltext_pft2.md`; rows are in the `pft2_additions_*.csv` files and `pft2_corrections.csv`.

## 1. Opening summary (first paragraph)

Why: Alexander & Nitz 2017 adds a kernel on a 6.6 m lap, and both rat route studies give curves that break the 5 to 28 cm rule (RSC never reaches 0.5; CA1 with Spearman reaches it at about 60 cm).

Old:

```text
in environments no longer than 4.26 m. The rodent and bird curves fall to 0.5 within 5 to 28 cm of a position during normal locomotion, half a field width or less where the same study reports a field width, and most cross zero 20 to 79 cm away.
```

New:

```text
in environments no longer than 6.6 m. Most rodent and bird curves fall to 0.5 within 5 to 28 cm of a position during normal locomotion, half a field width or less where the same study reports a field width, and most cross zero 20 to 79 cm away. Rats running fixed routes give two exceptions. Retrosplenial similarity never falls to 0.5 on routes of 2.15 and 6.6 m, and CA1 in the 6.6 m study, read with Spearman correlation, falls to 0.5 only about 60 cm away.
```

## 2. Retrosplenial kernel table in the second section (row for Alexander & Nitz 2017)

Why: The row rested on an unverified preprint summary. The journal papers and their matrices now give numbers for both rat studies, with CA1 from the same rats.

Old:

```text
| Alexander & Nitz 2017 | Rat, up to 140 of 229 cells | Plus-shaped track with repeated segments | Odd against even trials | Off-diagonal peaks repeat at quarter-route spacing, which CA1 matrices lack; no width or zero crossing | preprint, unverified |
```

New:

```text
| Alexander & Nitz 2015 | Rat, 228 cells pooled across 6 rats | 215 cm W-shaped route | Odd against even trials, 200-bin template, no per-cell normalization stated | Peak about 0.86; never falls to 0.5 (minimum 0.56 at about 64 cm); half of the drop to that minimum at about 20 cm; rises again to 0.66 at about 110 cm, the spacing of same-direction turns | full text, figure-read |
| Alexander & Nitz 2015 | Rat RSC, 187 cells, and CA1, 109 cells, in the same rats | As above, two sessions at one room location | Pearson between sessions | RSC peak 0.77, never falls to 0.5 (minimum 0.51 at about 71 cm), half of the drop at about 22 cm, rise to 0.58 at about 101 cm; CA1 peak 0.72, 0.5 at 26.5 cm, minimum 0.09, no rise | full text, figure-read |
| Alexander & Nitz 2017 | Rat RSC, 229 cells pooled across 5 rats; CA1, 141 cells in 3 rats | 660 cm closed plus-shaped lap of 12 segments of 55 cm | Spearman, odd against even trials, 3.5 cm bins, Gaussian SD 7 cm | RSC peak 0.91, floor about 0.77, never below 0.71; half of the drop at about 45 cm; local maximum at 165 to 182 cm, one quarter of the lap; CA1 peak 0.81, 0.5 at about 60 to 63 cm, no repeats | full text, figure-read |
```

## 3. Similarity section, opening paragraph

Why: Rat RSC is a further exception to the 5 to 28 cm statement, and the CA1 curves recorded with it belong in the same paragraph.

Old:

```text
Outside intermediate and ventral CA3 and anterior marmoset hippocampus, each rodent or bird curve recorded during normal locomotion falls to 0.5 within 5 to 28 cm of a position in spaces up to about 4 m.
```

New:

```text
Outside intermediate and ventral CA3, anterior marmoset hippocampus and rat retrosplenial cortex, each rodent or bird curve recorded during normal locomotion falls to 0.5 within 5 to 28 cm of a position in spaces up to about 4 m. Rat retrosplenial curves never reach 0.5 on routes of 2.15 and 6.6 m. CA1 recorded in the same rats falls to 0.5 at 26.5 cm on the 2.15 m route, and at about 60 cm on the 6.6 m lap with Spearman correlation and 7 cm smoothing [full text, figure-read] ([Alexander & Nitz 2015](https://doi.org/10.1038/nn.4058); [Alexander & Nitz 2017](https://doi.org/10.1016/j.cub.2017.04.036)).
```

## 4. "Thirteen measures at a glance" table, PV similarity row

Why: Separates mouse from rat RSC.

Old:

```text
RSC to 0.5 at 13 cm and to zero at 35 to 38 cm;
```

New:

```text
mouse RSC to 0.5 at 13 cm and to zero at 35 to 38 cm, while rat RSC stays above 0.5 along routes of 2.15 and 6.6 m;
```

## 5. "Thirteen measures at a glance" table, field width over grid spacing row

Why: Adds the rat non-grid ratio from Butler 2019.

Old:

```text
one dorsal non-grid MEC field 1.55 grid spacings measured in the same mice, and border width 0.40
```

New:

```text
non-grid MEC fields 1.55 grid spacings in one mouse study and 0.40 in one rat study, each against grid cells of the same animals, and border width 0.40
```

## 6. "Thirteen measures at a glance" table, field width over grid spacing row (studies)

Why: Adds the new source.

Old:

```text
Navratilova 2026; Mao 2017; Mallory 2018 | Mixed |
```

New:

```text
Navratilova 2026; Mao 2017; Mallory 2018; Butler 2019 | Mixed |
```

## 7. Field-width grid, rat table (after the last row)

Why: Optional. Butler 2019 gives the first rat non-grid width with grid spacing from the same rats. The row spans all 271 cells (39.5 cm) and the 100 high-coherence cells (49.9 cm). Ratios derived against the M1 to M5 ranges in the header.

Old:

```text
| Ventral CA3, large real 18 m track, 1 study (4.10 to 5.52 m) | 8.4 to 14 | 6.8 to 12 | 4.4 to 8.5 | 3.2 to 5.7 | 2.4 to 4.1 | 2.4 to 6.1 | 1.4 to 2.7 | 1.1 to 2.1 | Kjelstrup 2008 |
```

New:

```text
| Ventral CA3, large real 18 m track, 1 study (4.10 to 5.52 m) | 8.4 to 14 | 6.8 to 12 | 4.4 to 8.5 | 3.2 to 5.7 | 2.4 to 4.1 | 2.4 to 6.1 | 1.4 to 2.7 | 1.1 to 2.1 | Kjelstrup 2008 |
| **MEC, non-grid cells** | | | | | | | | | |
| MEC and nearby cortex, non-grid spatial cells, level unstated, real 1.5 m arena, 1 study (39.5 to 49.9 cm) | 0.81 to 1.3 | 0.66 to 1.0 | 0.42 to 0.77 | 0.31 to 0.51 | 0.23 to 0.37 | not comparable | not comparable | not comparable | Butler 2019 |
```

## 8. Field-width grid, rat sources table (after the last module-free row)

Why: Optional, goes with the previous edit.

Old:

```text
| Not in any row | | | | Park 2011 (park2011_5, DG-unspecified dorsal)
```

New:

```text
| MEC and nearby cortex, non-grid spatial cells, level unstated, real 1.5 m arena | 1 | 39.5 to 49.9 cm | [Butler 2019](https://doi.org/10.1126/science.aav5297) 39.5 cm, the equal-area diameter of the median average field size of 271 non-grid spatial cells in the foraging box, and 49.9 cm for the 100 cells with high spatial coherence; fields from a log-normalized map thresholded at 0.85; grid spacing in the same rats has a median of 99.14 cm, a ratio of 0.40 [full text, supplement, derived] | Butler 2019 (MNG34), the reward-task box, a manipulation; Diehl 2017, no field size |
| Not in any row | | | | Park 2011 (park2011_5, DG-unspecified dorsal)
```

## 9. Field-width grid, mouse table, dorsal CA1 real-environment row

Why: Mallory 2018's CA1 arena is now known (50 or 62 cm squares), so its 19.7 cm width joins this row. The range does not change.

Old:

```text
| Dorsal CA1, real environment up to 2 m, 18 studies (14.8 to 42.5 cm) |
```

New:

```text
| Dorsal CA1, real environment up to 2 m, 19 studies (14.8 to 42.5 cm) |
```

## 10. Field-width grid, mouse table, dorsal CA1 real-environment row (study list)

Why: Goes with the previous edit.

Old:

```text
Lenkey 2025; Mao 2017; Mou 2018
```

New:

```text
Lenkey 2025; Mallory 2018; Mao 2017; Mou 2018
```

## 11. Field-width grid, mouse table, MEC non-grid row label

Why: The arena size is in the Online Methods: 70, 90 or 100 cm squares.

Old:

```text
| Dorsal MEC non-grid spatial cells, real open arena of unstated size, 1 study (56.1 cm) |
```

New:

```text
| Dorsal MEC non-grid spatial cells, real open arena of 70 to 100 cm, 1 study (56.1 cm) |
```

## 12. Field-width grid, mouse table, border row label

Why: As above.

Old:

```text
| Dorsal MEC border cells, width perpendicular to the wall, real open arena of unstated size, 1 study (14.4 cm) |
```

New:

```text
| Dorsal MEC border cells, width perpendicular to the wall, real open arena of 70 to 100 cm, 1 study (14.4 cm) |
```

## 13. Field-width grid, mouse sources table, dorsal CA1 real-environment row (count)

Why: Mallory 2018 CA1 joins the row.

Old:

```text
| Dorsal CA1, real environment up to 2 m | 18 | 14.8 to 42.5 cm |
```

New:

```text
| Dorsal CA1, real environment up to 2 m | 19 | 14.8 to 42.5 cm |
```

## 14. Field-width grid, mouse sources table, dorsal CA1 real-environment row (contributing studies)

Why: Adds the Mallory entry in alphabetical order.

Old:

```text
[Lenkey 2025](https://doi.org/10.1038/s41467-025-60679-4) 37.3 cm on a 1.57 m wheel [full text]; [Mao 2017]
```

New:

```text
[Lenkey 2025](https://doi.org/10.1038/s41467-025-60679-4) 37.3 cm on a 1.57 m wheel [full text]; [Mallory 2018](https://doi.org/10.1038/s41593-017-0055-3) 19.7 cm, the equal-area diameter of a mean field of 303.75 cm², in a 50 or 62 cm square [full text, derived]; [Mao 2017]
```

## 15. Field-width grid, mouse sources table, dorsal CA1 real-environment row (exclusions)

Why: The exclusion reason no longer holds.

Old:

```text
Mallory 2018 (mallory2018_1), arena size not read, so no environment class; Rechavi 2022
```

New:

```text
Rechavi 2022
```

## 16. Field-width grid, mouse sources table, MEC non-grid row

Why: Arena size now known.

Old:

```text
| Dorsal MEC non-grid spatial cells, real open arena of unstated size | 1 | 56.1 cm |
```

New:

```text
| Dorsal MEC non-grid spatial cells, real open arena of 70 to 100 cm | 1 | 56.1 cm |
```

## 17. Mouse field-width grid, closing paragraph

Why: The arena size is known; the CA1 exclusion can be lifted.

Old:

```text
That study's arena size is in its paywalled methods, so both rows lack an environment class. They stay, unlike the same study's CA1 width in the hippocampal rows, because they are the only non-grid widths and their grid spacing comes from the same mice [inference].
```

New:

```text
That study's MEC arenas were 70, 90 or 100 cm squares, chosen per mouse, and its CA1 arenas 50 or 62 cm squares, so its CA1 width, 19.7 cm, now sits in the dorsal CA1 row for real environments up to 2 m [full text].
```

## 18. Kernel grid, "grids leave out" paragraph

Why: The rat RSC curves have no r = 0.5 width because they never reach 0.5, not because the curves are missing. The two CA1 curves of the same studies have widths.

Old:

```text
Mao et al.'s layer subsets and colour-read Thy1 matrices are in the retrosplenial section, and two rat retrosplenial curves have no width.
```

New:

```text
Mao et al.'s layer subsets and colour-read Thy1 matrices are in the retrosplenial section. The rat retrosplenial curves of Alexander and Nitz have no width because they never fall to 0.5. The CA1 curves recorded with them are 53 cm wide at r = 0.5 between two sessions on a 2.15 m route and about 120 to 126 cm wide with Spearman correlation on a 6.6 m lap, both from cells pooled across rats [full text, figure-read; derived] ([Alexander & Nitz 2015](https://doi.org/10.1038/nn.4058); [Alexander & Nitz 2017](https://doi.org/10.1016/j.cub.2017.04.036)).
```

## 19. Kernel grid, rat table (optional rows after Farooq 2024)

Why: Optional. Adds the two rat CA1 kernels of Alexander & Nitz. Both pool cells across rats; the 2015 curve is between sessions and the 2017 curve uses Spearman correlation with 7 cm smoothing, so it may be wide for method reasons. Ratios derived against the header ranges.

Old:

```text
| Dorsal CA1, Farooq 2024, juvenile, first exposure, 1 m track (29 to 31 cm) | 0.59 to 0.79 | 0.48 to 0.65 | 0.31 to 0.48 | 0.23 to 0.32 | 0.17 to 0.23 | not comparable | not comparable | not comparable | Farooq 2024 |
```

New:

```text
| Dorsal CA1, Farooq 2024, juvenile, first exposure, 1 m track (29 to 31 cm) | 0.59 to 0.79 | 0.48 to 0.65 | 0.31 to 0.48 | 0.23 to 0.32 | 0.17 to 0.23 | not comparable | not comparable | not comparable | Farooq 2024 |
| Dorsal CA1, Alexander & Nitz 2015, between sessions, 2.15 m route (53 cm) | 1.1 to 1.4 | 0.88 to 1.1 | 0.57 to 0.82 | 0.41 to 0.55 | 0.31 to 0.39 | not comparable | not comparable | not comparable | Alexander & Nitz 2015 |
| Dorsal CA1, Alexander & Nitz 2017, Spearman, 6.6 m lap (120 to 126 cm) | 2.4 to 3.2 | 2.0 to 2.6 | 1.3 to 1.9 | 0.94 to 1.3 | 0.70 to 0.93 | not comparable | not comparable | not comparable | Alexander & Nitz 2017 |
```

## 20. Kernel grid, sources table (optional rows after Farooq & Dragoi 2024)

Why: Optional, goes with the previous edit.

Old:

```text
| Dorsal CA1, juvenile | [Farooq & Dragoi 2024](https://doi.org/10.1038/s41467-024-52758-9), rats aged 23 to 24 days on their first track, about 63 cells per rat, 5 rats, 1 m track | Group-average PV correlation, normalization not stated | 15 cm; 14.5 and 15.5 cm | 29 to 31 cm | full text, figure-read |
```

New:

```text
| Dorsal CA1, juvenile | [Farooq & Dragoi 2024](https://doi.org/10.1038/s41467-024-52758-9), rats aged 23 to 24 days on their first track, about 63 cells per rat, 5 rats, 1 m track | Group-average PV correlation, normalization not stated | 15 cm; 14.5 and 15.5 cm | 29 to 31 cm | full text, figure-read |
| Dorsal CA1, between sessions | [Alexander & Nitz 2015](https://doi.org/10.1038/nn.4058), 109 CA1 cells with place fields, 4 rats, pooled, 2.15 m W-shaped route | Pearson, first against third session at one room location | 26.5 cm | 53 cm | full text, figure-read |
| Dorsal CA1, plus-shaped lap | [Alexander & Nitz 2017](https://doi.org/10.1016/j.cub.2017.04.036), 141 CA1 cells, 3 rats, pooled, 6.6 m lap | Spearman, odd against even trials, Gaussian SD 7 cm; colour bar clipped at 0.5 | about 60 to 63 cm | about 120 to 126 cm | full text, figure-read |
```

## 21. Retrosplenial section heading

Why: The rat kernels now have numbers and differ from the mouse kernels.

Old:

```text
## Retrosplenial fields and kernels match CA1 in head-fixed mice, and one rat study finds fields 1.7 times wider
```

New:

```text
## Retrosplenial fields and kernels match CA1 in head-fixed mice, while rat RSC fields are wider and rat RSC similarity stays above 0.5 along whole routes
```

## 22. Retrosplenial section, opening paragraph

Why: Records the new rows.

Old:

```text
An audited sweep of retrosplenial cortex (RSC) found 66 field rows from 28 studies and 11 kernel rows from 4 studies, in rats, mice and macaques (sweep_rsc.md and audit_sweep_rsc.md in the literature notes).
```

New:

```text
An audited sweep of retrosplenial cortex (RSC) found 66 field rows from 28 studies and 11 kernel rows from 4 studies, in rats, mice and macaques (sweep_rsc.md and audit_sweep_rsc.md in the literature notes). A later full-text pass on two rat route studies added 6 field rows and 6 kernel rows and corrected 3 rows (pft2_additions_fields_rsc.csv, pft2_additions_kernel_rsc.csv, pft2_corrections.csv and fulltext_pft2.md).
```

## 23. Retrosplenial kernel subsection heading

Why: There is now more than one RSC curve, and the rat curves behave differently.

Old:

```text
### The one retrosplenial distance curve matches CA1 at r = 0.5
```

New:

```text
### Mouse retrosplenial curves match CA1 at r = 0.5, and rat retrosplenial curves never fall to 0.5
```

## 24. Retrosplenial kernel subsection, rat paragraph

Why: Replaces the unverified preprint statement with figure-read numbers from both rat papers and same-rat CA1.

Old:

```text
No rat RSC kernel has numbers. Alexander and Nitz report that rat RSC population matrices from odd and even trials repeat at quarter-route spacing on a plus track with repeated segments, which CA1 matrices do not, but the audit could not check the text behind this [preprint; unverified] ([Alexander & Nitz 2017](https://doi.org/10.1101/100537)).
```

New:

```text
Rat RSC kernels look different. Alexander and Nitz correlated population vectors pooled across rats, from odd and even trials or from two sessions at one room location, along a 215 cm W-shaped route and a 660 cm plus-shaped lap. Read from their matrices, rat RSC similarity peaks at 0.77 to 0.91 and never falls to 0.5 on either route. It levels off at 0.51 to 0.78, and half of the drop to that floor takes about 20 cm on the W track and about 45 cm on the plus lap. Similarity rises again at the spacing of same-shaped segments, to 0.66 at about 110 cm, the distance between same-direction turns on the W track, and to 0.79 at 165 to 182 cm, a quarter of the plus lap [full text, figure-read] ([Alexander & Nitz 2015](https://doi.org/10.1038/nn.4058); [Alexander & Nitz 2017](https://doi.org/10.1016/j.cub.2017.04.036)). CA1 recorded in the same rats falls to 0.5 at 26.5 cm between sessions on the W track, with a floor near 0.09 and no rise at the turn spacing. On the plus lap it falls to 0.5 at about 60 to 63 cm with Spearman correlation and 7 cm smoothing, with no repeats [full text, figure-read]. Neither paper describes a per-cell normalization of the vectors. RSC cells fire throughout the route, so the high floor probably reflects broad firing, and the r = 0.5 width is undefined rather than large [inference]. The same matrices place the rat on the lap to within 1.9 ± 2.7 cm with 229 RSC cells, against 5.2 ± 6.7 cm with 141 CA1 cells [full text] ([Alexander & Nitz 2017](https://doi.org/10.1016/j.cub.2017.04.036)).
```

## 25. Retrosplenial kernel subsection, last sentence of the rat paragraph

Why: The plus lap is longer than 2.8 m.

Old:

```text
The sweep found no RSC kernel for a space longer than 2.8 m.
```

New:

```text
The longest space with an RSC kernel is now the 6.6 m plus-shaped lap; no RSC kernel comes from an open arena [derived].
```

## 26. Retrosplenial section, gaps paragraph

Why: Rat kernels now exist; the gap is the missing link to field widths.

Old:

```text
Three gaps limit these comparisons. No rat RSC kernel has numbers, so the wider rat fields have no population counterpart.
```

New:

```text
Three gaps limit these comparisons. The rat RSC kernels come from route studies that report no field widths, so the wider rat fields and the high rat kernel floor are not measured in the same animals.
```

## 27. Retrosplenial section, full citations

Why: Journal versions read; adds the 2015 paper.

Old:

```text
Alexander AS, Nitz DA (2017). Spatially periodic activation patterns of retrosplenial cortex encode route sub-spaces and distance travelled. bioRxiv 100537, published in Curr Biol 27:1551-1560 [preprint; WebFetch summary, unverified]
```

New:

```text
Alexander AS, Nitz DA (2015). Retrosplenial cortex maps the conjunction of internal and external spaces. Nat Neurosci 18:1143-1151. doi:10.1038/nn.4058 [full text and supplement; matrices figure-read]. Alexander AS, Nitz DA (2017). Spatially periodic activation patterns of retrosplenial cortex encode route sub-spaces and distance traveled. Curr Biol 27:1551-1560. doi:10.1016/j.cub.2017.04.036 [full text and supplement; matrices figure-read]
```

## 28. Section on similarity curves falling to half within a field width, "Other regions" paragraph

Why: Mao 2017 is mouse; the rat RSC result differs.

Old:

```text
Retrosplenial cortex resembled CA1 [full text] ([Mao et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557927)).
```

New:

```text
Mouse retrosplenial cortex resembled CA1 [full text] ([Mao et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557927)). Rat retrosplenial cortex did not. On two routes its similarity stayed above 0.5 and rose again at same-shaped segments, while CA1 in the same rats fell to 0.5 within 27 to 63 cm [full text, figure-read] ([Alexander & Nitz 2015](https://doi.org/10.1038/nn.4058); [Alexander & Nitz 2017](https://doi.org/10.1016/j.cub.2017.04.036)).
```

## 29. Conclusion

Why: Adds the rat result.

Old:

```text
Retrosplenial cortex in head-fixed mice matches CA1 recorded the same way, in field width and in kernel.
```

New:

```text
Retrosplenial cortex in head-fixed mice matches CA1 recorded the same way, in field width and in kernel. In rats running fixed routes, RSC similarity stays above 0.5 along routes of 2.15 and 6.6 m and rises again at same-shaped segments, while CA1 in the same rats falls to 0.5 within 27 to 63 cm. Neither study describes normalizing those vectors per cell, so the high floor probably reflects broad firing in part.
```

## 30. Non-grid MEC section heading

Why: A second same-animal ratio exists, in rats, and it is a quarter of the mouse one.

Old:

```text
## Non-grid MEC fields measure about 1.5 grid spacings in one mouse study and did not grow with grid scale
```

New:

```text
## Non-grid MEC fields measure 1.55 grid spacings in one mouse study and 0.40 in one rat study, and did not grow with grid scale
```

## 31. Non-grid MEC section, opening paragraph

Why: Butler 2019 gives the second study.

Old:

```text
Only one study gives a non-grid field size, and it also gives grid spacing in the same mice.
```

New:

```text
Two studies give a non-grid field size together with grid spacing in the same animals, one in mice and one in rats. The rat values come from a later full-text pass (pft2_additions_fields_mec.csv, pft2_corrections.csv and fulltext_pft2.md).
```

## 32. Non-grid MEC subsection heading

Why: Two studies now.

Old:

```text
### One study gives non-grid field size and grid spacing in the same mice
```

New:

```text
### Two studies give non-grid field size and grid spacing in the same animals
```

## 33. Non-grid MEC, limits of the 1.55

Why: The Online Methods are in the journal PDF: arena sizes are known, and no field rule is given for these cells.

Old:

```text
The arena size and the field rule are in the paywalled Online Methods, and a box caps the fields of broadly firing cells.
```

New:

```text
The arenas were 70, 90 or 100 cm squares chosen per mouse, so the mean field filled 25 to 50% of the box, and its 56.1 cm diameter was 0.56 to 0.80 of the box side; the box probably caps these fields [full text; derived]. The paper gives no field rule for these cells. Its place-field rule is at least five contiguous 2.5 cm bins above 20% of the peak [full text].
```

## 34. Non-grid MEC, new rat paragraph after the knockout paragraph

Why: Adds Butler 2019 (Fig. S10A, Table S1, Fig. 1K), the second same-recording ratio. The reward-task values are kept apart from the foraging baseline.

Old:

```text
In these recordings non-grid field size did not follow a change in grid scale.
```

New:

```text
In these recordings non-grid field size did not follow a change in grid scale.

In rats, Butler et al. recorded 778 cells in two 1.5 m boxes, 598 in MEC and the rest in parasubiculum, postsubiculum and nearby cortex, with no dorsoventral positions given. Fields were found on maps of 5 cm bins matched for speed and occupancy, by thresholding a log-normalized map at 0.85 for non-grid cells and 0.8 for grid cells and keeping regions of at least 6 bins. In the foraging box the 271 non-grid spatial cells had a median average field size of 1,225 cm², an equal-area diameter of 39.5 cm, a median of 2 fields, and fields over 13% of the box. The 100 cells with high spatial coherence had 1,958 cm², 49.9 cm [full text, supplement; diameters derived] ([Butler, Hardcastle & Giocomo 2019](https://doi.org/10.1126/science.aav5297)). Grid spacing in the same rats had a median of 99.14 cm over 102 grid cells, and six modules in four rats ran from about 65 to 122 cm [full text; modules figure-read]. The non-grid diameter is therefore 0.40 grid spacings, 0.50 for the high-coherence cells and 0.61 against the smallest module [derived]. Grid fields found the same way at threshold 0.8 had a median of 1,420 cm², 42.5 cm, so non-grid fields had 0.86 of the grid field area [full text; derived]. The rat ratio is about a quarter of the mouse ratio. The two studies differ in species, recording depth, box size and field rule. Butler et al.'s high threshold probably gives smaller fields than a 20%-of-peak edge, and Mallory et al. give no rule for non-grid fields [inference]. In the reward-task box the same cells kept their field size, 1,106 against 1,225 cm², p = 0.35, as well as their field number and coverage [full text, supplement].
```

## 35. Non-grid share paragraph

Why: Diehl's 67.5% is already a share of all cells (Table S1).

Old:

```text
The share of MEC cells that are spatial but not grid runs from 6% to about 64%, and it depends on the criterion and on what the denominator counts [derived].
```

New:

```text
The share of MEC cells that are spatial but not grid runs from 6% to 67.5%, and it depends on the criterion and on what the denominator counts [derived].
```

## 36. Non-grid share table, Diehl row (and a new Butler row)

Why: Table S1 counts (61, 32 and 233 of 345) show the printed shares are of all cells. Butler 2019 adds a rat row.

Old:

```text
| [Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/) | Rat, dorsal MEC, layers 2 and 3, 345 cells | 67.5% of spatial cells, about 64% of all cells | 17.7% of spatial cells | Shuffles of spatial information and split-half correlation; about 95% of cells were spatial, and the classes add to 94.5% of them | full text, derived |
```

New:

```text
| [Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/) | Rat, dorsal MEC, layers 2 and 3, 345 cells | 233 of 345, 67.5% of all cells (71.5% of the spatial cells) | 61 of 345, 17.7% | Shuffles of spatial information and split-half correlation; the Table S1 counts show the printed shares are of all 345 cells, of which 326, 94.5%, were spatial | full text, supplement, derived |
| [Butler et al. 2019](https://doi.org/10.1126/science.aav5297) | Rat, MEC and nearby cortex, 778 cells recorded in both boxes | 271 of 778, 34.8% | 102 of 778, 13.1% | Position coding in a model, not grid or border; 180 of the cells lay in parasubiculum, postsubiculum and nearby cortex | full text, supplement, derived |
```

## 37. Lateral septum and other regions paragraph (Diehl sentence)

Why: The shares are of all 345 cells; Butler 2019 also reports non-grid field sizes.

Old:

```text
About 95% of medial entorhinal cells passed spatial cutoffs, and of the spatial cells 17.7% were grid cells, 9.3% border cells and 67.5% non-grid spatial cells, whose field sizes were not reported [full text] ([Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/)). One mouse study does report them [full text] ([Mallory et al. 2018](https://doi.org/10.1038/s41593-017-0055-3)).
```

New:

```text
About 95% of medial entorhinal cells passed spatial cutoffs, and of all 345 cells 17.7% were grid cells, 9.3% border cells and 67.5% non-grid spatial cells, whose field sizes were not reported [full text, supplement] ([Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/)). One mouse study and one rat study do report them [full text] ([Mallory et al. 2018](https://doi.org/10.1038/s41593-017-0055-3); [Butler, Hardcastle & Giocomo 2019](https://doi.org/10.1126/science.aav5297)).
```

## 38. Stability and remapping paragraph, Butler sentence

Why: The supplement shows the controls were 3 separate rats, and its Fig. S7E legend contradicts the main text on the 49 control cells.

Old:

```text
Between two 1.5 m arenas, 196 of 271 non-grid spatial cells in rats trained on a reward task remapped, with a mean correlation of 0.32 ± 0.22, against 0.41 ± 0.27 in foraging controls, where 49 of 100 remapped [full text]
```

New:

```text
Between two 1.5 m arenas, 196 of 271 non-grid spatial cells in 7 rats trained on a reward task remapped, with a mean correlation of 0.32 ± 0.22, against 0.41 ± 0.27 in 3 separate foraging control rats, where 49 of 100 remapped; a supplement legend says instead that these 49 retained their tuning [full text, supplement]
```

## 39. Dorsoventral subsection, Diehl sentence

Why: Adds the one dorsoventral statement in the STAR Methods.

Old:

```text
Diehl et al. write that grid and non-grid cells "were highly intermixed along the dorso-ventral axis", but they give no sizes by level [full text] ([Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/)).
```

New:

```text
Diehl et al. write that grid and non-grid cells "were highly intermixed along the dorso-ventral axis", but they give no sizes or distances by level; the methods add only that over 85% of the data lay dorsal to the most ventral grid-cell site in each rat [full text] ([Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/)).
```

## 40. Further-gaps paragraph

Why: Both leads are resolved: Diehl's Table S1 holds only counts, and Butler's Fig. S10A gives medians.

Old:

```text
The sweep lists further gaps: non-grid field size in rats, where Diehl et al.'s Table S1 and Butler et al.'s Fig. S10A may hold it but could not be opened; non-grid fields in spaces larger than about 1.5 m; border field length along the wall; and object-vector field size.
```

New:

```text
The sweep lists further gaps: a non-grid field size measured with the same rule in rats and mice (Diehl et al.'s Table S1 holds only cell counts, and Butler et al.'s Fig. S10A gives medians under a high threshold); non-grid fields in spaces larger than about 1.5 m; border field length along the wall; and object-vector field size.
```

## 41. Non-grid MEC full citations, Mallory

Why: Online Methods read.

Old:

```text
Grid scale drives the scale and long-term stability of place maps. Nat Neurosci 21:270-282. doi:10.1038/s41593-017-0055-3 [full text; Online Methods not read]
```

New:

```text
Grid scale drives the scale and long-term stability of place maps. Nat Neurosci 21:270-282. doi:10.1038/s41593-017-0055-3 [full text with Online Methods]
```

## 42. Non-grid MEC full citations, Butler

Why: Supplement read.

Old:

```text
Remembered reward locations restructure entorhinal spatial maps. Science 363:1447-1452. doi:10.1126/science.aav5297 [full text; supplement not opened]
```

New:

```text
Remembered reward locations restructure entorhinal spatial maps. Science 363:1447-1452. doi:10.1126/science.aav5297 [full text and supplement]
```
