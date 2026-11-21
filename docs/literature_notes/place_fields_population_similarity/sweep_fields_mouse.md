# Mouse place-field widths and field counts: literature sweep

Compiled 2026-11-21. Every number is mouse data. No model results are included. The table is `sweep_fields_mouse.csv` in this folder. It has 91 rows from 52 studies, and 61 rows carry a width in cm. Nineteen rows come from the 11 studies already verified in `grid_ratio_mouse_fields.md`. The other 72 rows come from 41 studies read in this sweep.

The short version:

- No new mouse width exists for intermediate or ventral levels. Ventral CA1 still rests on Keinath 2014 and Chockanathan 2021, intermediate CA1 on Jarzebowski 2022, and intermediate and ventral CA3 have nothing.
- Dorsal CA1 now has 46 studies (35 with a width) across tetrodes, silicon probes, a glass pipette, whole-cell, one-photon and two-photon imaging. No Neuropixels study states a width in its text.
- Setup moves the width more than the recording method does. Real arenas and short tracks give 18 to 40 cm. Virtual tracks from 1.8 to 41 m give 37 to 60 cm, whatever the method. Peak-only rules (mean + 1.5 SD, or fine bins with no smoothing) give 7 to 9 cm.
- Width stops growing with track length in mouse VR. Field count grows instead, to 4.7 fields per active cell on a 41 m track (Lee 2020, preprint).
- Mouse CA3 fields are not narrower than CA1 fields. Three studies test CA3 against CA1 within one study: Dong 2021 finds CA3 wider, and Hainmueller 2018 and Sheintuch 2023 find no difference. Boublil 2026 and Hussaini 2011 report both regions without testing them against each other. Dentate granule-cell fields are wider than CA1 in the one VR study (65 against 45 cm).

## Method

Searches ran on 2026-11-20 and 2026-11-21. I searched Europe PMC (REST search) and the web for mouse studies that report hippocampal place-field width, field area or field counts, and mined the mouse citations of `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`. Full text came from Europe PMC JATS XML, the PMC open-data bucket on S3 (which also holds figures and supplements of open-access papers), the NCBI BioC API (author manuscripts), and open publisher or bioRxiv PDFs. Each number was read in the full text and copied with its sentence, table cell or figure legend into `quote_or_reading`. Search snippets were never used as values. About 150 papers were opened in four parallel passes. One covered dorsal CA1 electrophysiology, one dorsal CA1 imaging, one DG, CA2, CA3 and subiculum, and one ventral levels, large environments and Lee 2020. Most mouse papers give widths only in figures, so many were checked and dropped.

The 11 note studies enter with `verified_in = note` and were not re-read. Keinath, Dong, Hussaini, Hainmueller, Chockanathan and Jarzebowski get one row per region, level or arena, because those splits are what the heat maps need.

Conventions in the CSV:

- `status` is one or more of FT (read in the full text), figure-read (measured from a figure, method in the row), derived (arithmetic in `notes`), abstract, preprint and secondary. The counts are 79 FT, 45 derived, 15 figure-read and 2 preprint.
- `width_cm` holds a length on 1D tracks. For 2D areas it holds the equal-area diameter d = 2 * sqrt(A / pi), tagged derived, with the area kept in `value_reported`. Rows without a usable width leave `width_cm` empty. That covers count-only rows, pixel areas with no stated pixel size, and the subthreshold Vm ramp of Bittner 2017 (125 cm, a depolarization footprint, not a spike field).
- `per_field_or_per_cell` says whether the number is per field, the largest or highest-peak field of each cell, or all fields of a cell summed.
- Imaging rows give indicator, smoothing and threshold in `field_rule`.
- `fields_per_cell` gives the count with its unit. Several track studies count per running direction, and those rows say so.
- In quotes, en dashes and minus signs are written as hyphens.

## Coverage

Each cell gives studies, then (rows with a width / all rows). A study appears in several cells when it reports several regions.

| Region | Level | Tetrode | Silicon | Neuropixels | Whole-cell | Other | 1p | 2p |
|---|---|---|---|---|---|---|---|---|
| CA1 | dorsal | 13 (18/19) | 5 (6/8) | | 2 (0/2) | 1 (1/1) | 10 (7/10) | 15 (13/17) |
| CA1 | intermediate | | | | | | 1 (1/1) | |
| CA1 | ventral | 1 (2/2) | 2 (1/2) | | | | | |
| CA1 | unspecified | 1 (0/1) | | | | | | |
| CA2 | dorsal | 1 (0/1) | | | | | | |
| CA3 | dorsal | 2 (2/2) | 2 (0/3) | | | | 1 (1/1) | 2 (2/2) |
| DG-granule | dorsal | 2 (2/3) | 2 (0/3) | | 1 (0/1) | | 1 (0/1) | 2 (1/2) |
| DG-mossy | dorsal | 2 (2/3) | 2 (0/3) | | | | | 1 (0/1) |
| DG-unspecified | dorsal | 1 (1/1) | | | | | | |
| SUB | dorsal | 1 (1/1) | | | | | | |

"Other" is the glass-pipette single-unit recording of Harvey 2009. Intermediate and ventral CA3, any ventral DG and any Neuropixels width are empty.

## New numbers in the gap areas

Intermediate and ventral levels. Nothing new. Fork searches covered Jimenez-type ventral CA1 imaging, anxiety-arena recordings, dorsoventral comparison papers from 2023 to 2026, and long-axis axon imaging. None reports a ventral or intermediate field size. The one new same-study dorsal and ventral comparison is a null. Aged (14 to 19 month) control mice on a 1.9 m VR track had no place cells in either dorsal (229 units) or ventral (450 units) CA1 (Chockanathan 2024, count rows only).

CA3. Two new cm widths, both in small or short environments. Boublil 2026 (tetrodes, familiar 41 cm square) gives CA3 17.7 cm per field against CA1 18.1 and DG 16.9 cm (GLMM estimates, unit assumed cm²). Sheintuch 2023 (one-photon, 96 cm track) gives CA3 about 9.6 cm against CA1 8.8 cm (figure-read, not different), with fewer fields per cell in CA3 (1.69 against 2.00). Kim 2020 and Senzai 2017 add CA3 counts on a 2 m belt and a T-maze.

CA2. Donegan 2020 (tetrodes, 60 x 40 cm arena) gives areas in pixels of unstated size. CA2 fields are 107 pixels against CA1 63, a linear ratio of 1.31 (derived), with 2.61 against 2.0 fields per cell. No cm width.

Dentate gyrus. Hainmueller & Bartos 2018 Extended Data Fig. 3c, re-read for this sweep, gives a granule-cell median of about 65 cm (quartiles 44 and 100 cm, n = 366) on the 4 m VR track, against 45 cm for CA1 and 40 cm for CA2/3. Kim 2023 (tetrodes, 60 to 69 cm enclosures) gives per-field granule-cell fields of 17.5 cm and mossy-cell fields of 16.1 cm, assuming about 1.3 cm² pixels. Summed over a cell, mossy cells cover more (22.1 against 19.3 cm) because they have more fields (1.87 against 1.20). Boublil 2026 gives DG 16.9 cm without a cell type.

Subiculum. Only Abad-Perez 2025 (tetrodes, mostly proximal subiculum). Summed field area is 30.0% of the arena for subiculum and 29.0% for CA1, not different. That is 30.9 and 30.4 cm if the 50 cm arena was used, or 43.3 and 42.5 cm for the 70 cm arena. The paper does not say which, and its threshold (mean + SE) is low.

Large environments. Lee 2020 (two-photon, about 41 m winding VR path, preprint) gives widths of 40 to 57 cm per mouse, mean of means 46 cm. Sumegi 2025 (two-photon, 8.32 m VR corridor) gives 46.4 ± 24.4 cm for BTSP-like new fields and 50.4 ± 32.5 cm for the rest, measured near the base of the peak. Sosa 2025 (4.5 m VR) and Chen 2026 (one-photon, 8.9 m maze path in a 0.92 m² footprint) give counts only. Hainmueller 2018 (4 m) was already in the note. No mouse width exists for a real track over 3 m or an arena over 1.2 m². The 1.13 m² cheeseboard of Jarzebowski is the largest 2D space with a width.

## Dorsal CA1 widths by method

| Method | Studies with widths | Width range (cm) | Environments | Comment |
|---|---|---|---|---|
| Tetrode | 12 | 18 to 40 | Real arenas 0.1 to 0.4 m², tracks 0.65 to 2 m, one 60 cm VR square | Median of 18 rows 29 cm |
| Silicon probe | 3 | 40 to 52, and 9.4 | 1.5 m real track, 1.9 to 2 m VR | 9.4 cm uses the mean + 1.5 SD rule |
| Glass pipette | 1 | 41 | 1.8 m VR | Harvey 2009 |
| Whole-cell | 0 | | | Bittner 2017 Vm ramp 125 cm, Zhang 2020 counts |
| Neuropixels | 0 | | | Aery Jones 2026 and Masuda 2023 give no CA1 width in text |
| One-photon | 7 | 7 to 24 | Tracks 0.7 to 1 m, 0.11 m² arena, 1.13 m² maze | 22 to 24 with heavy smoothing, 7 to 9 with fine bins or peak-only rules |
| Two-photon | 12 | 15 to 60 | Real belts and tracks 0.86 to 2 m, a 32.5 cm arena, VR 1.8 to 41 m | VR 37 to 60, real belts 33 to 37, small floating track and arena 15 to 20 |

## How method, rule and environment shift widths

### Comparisons within one study

| Factor | Study | Comparison | Result |
|---|---|---|---|
| Method | Dombeck 2010, Harvey 2009 | Same rig and 1.8 m VR track, GCaMP3 imaging against spikes | 50 against 41 cm (imaging 22% wider) |
| Method | Mao 2017 | Same 150 cm belt, imaging against silicon probe | Place cells 26% against 47% of cells (counts only) |
| Rule | Grijseels 2021 | Four place-cell rules on the same two-photon data, 2 m VR | 3.9% to 47.0% of ROIs called place cells, 1.1% by all four |
| Rule | Bourboulou 2019 | Mean-map field against single-trial field, 2 m VR | 48.7 against 34.2 cm |
| Rule | Keinath 2014 | Summed against per field, 70 cm cylinder | 33.1 against 22.3 cm (dorsal), 41.9 against 29.3 cm (ventral) |
| Rule | Jarzebowski 2022 | Summed against per field, 1.13 m² maze | 22.1 against 17.0 cm |
| Rule | Kim 2023 | Summed against per field, 60 cm enclosures | Granule 19.3 against 17.5 cm, mossy 22.1 against 16.1 cm |
| Familiarity | Zhao 2014 | Familiar 2 m track first, then a novel 1 m track on days 1, 2 and 3 | 24 cm on the familiar track; 40, 35 and 33 cm on the novel one |
| Familiarity | Dong 2021 | Novel against familiar 3 m VR track | CA1 46 against 38 cm, CA3 52 against 44 cm |
| Familiarity | Dragoi 2013 | First-ever track, session 1 against 2 | Narrowed by 8 ± 2.3 cm. In experienced mice a new track gave 23.9 cm, the same as the familiar one |
| Familiarity | Rechavi 2022 | One-photon, day 1 against day 22 | 9.2 against 8.2 cm |
| Cues | Bourboulou 2019 | Bare VR track, 3D objects, patterned walls | 51.5, 48.7 and 41.5 cm. Inside object zones 43.2 cm, elsewhere 57.1 cm |
| Reward | Krishnan 2022 | Rewarded, unrewarded, rewarded again (2 m VR) | 53.3, 60.6 and 54.1 cm |
| Size | Keinath 2014 | 35 against 70 cm cylinder | Dorsal summed 28.0 to 33.1 cm (not significant), ventral 33.1 to 41.9 cm (P < 0.001). Fields per cell about doubled |
| Size | Schuette 2022 | One-photon, 70 cm enclosure shortened or lengthened | Width changed by -4.3 cm and +2.0 cm |
| Size | Go 2021 | 86 cm circular track against 32.5 cm arena | 19.7 against 14.8 cm |
| Real against virtual | Chen 2018 | Same cells, 60 cm real square against VR square | 26.1 against 31.3 cm (area factor 1.44 stated by the authors) |
| Species | Mou 2018 | Mouse against rat on the same novel 2 m track | 33 against 40 cm, with more fields per trajectory in mice (1.2 against 0.90) |
| Strain | Sloin 2022 | C57BL/6J against FVB x C57 hybrid | 40 against 40 cm |
| Cell type | Gu 2023 | Calbindin-positive against negative pyramidal cells | 29.8 against 31.1 cm |
| Region | Dong 2021 | CA3 against CA1, 3 m VR | 44 against 38 cm (P = 7.1 x 10^-4) |
| Region | Hainmueller 2018 | DG, CA2/3 and CA1, 4 m VR | 65, 40 and 45 cm |
| Region | Hussaini 2011 | CA3 against CA1, 50 cm box | 27.0 against 25.7 cm |
| Region | Boublil 2026 | CA1, CA3 and DG, 41 cm box | 18.1, 17.7 and 16.9 cm |
| Region | Sheintuch 2023 | CA3 against CA1, one-photon | 9.6 against 8.8 cm |
| Region | Abad-Perez 2025 | Subiculum against CA1 | 30.9 against 30.4 cm |
| Level | Jarzebowski 2022 | Intermediate against dorsal CA1 | Linear ratio 1.56 |
| Level | Chockanathan 2021 | Ventral against dorsal CA1, 1.5 SD rule | 11.9 against 9.4 cm, not significant |

### Patterns across studies

Setup sorts the dorsal CA1 widths better than method does. Tetrode studies in real arenas of 0.1 to 0.4 m² and on tracks up to 2 m give 18 to 40 cm. Virtual tracks give 37 to 60 cm whether the cells are imaged (Dong, Hainmueller, Dombeck, Curreli, Krishnan, Radvansky, Sumegi, Lee) or recorded with silicon probes or pipettes (Bourboulou, Harvey). Real treadmill belts of 1.5 to 2 m sit between, at 33 to 37 cm (Mao, Geiller, Lenkey). The one same-cell test, Chen 2018, makes virtual fields 1.2 times wider in linear size.

The field rule is the next largest factor, and it can move a width by a factor of five. Rules that keep only the top of the field give 7 to 9 cm: mean + 1.5 SD (Chockanathan 2021, Schuette 2022), 30% of peak on 4 cm bins with no smoothing (Sheintuch 2023, Rechavi 2022), and mixture-model splits (Tarcsay 2025, 7.1 cm). Base-to-base or near-base widths are the largest (Krishnan 53 to 61 cm, Sumegi 46 to 50 cm). Heavy smoothing sets a floor. Ziv 2013 (sigma 8.75 cm, floor 20.6 cm) and Indersmitten 2019 (sigma 10.3 cm, floor 24.2 cm) both report about 24 cm. So one-photon widths split into two groups, near 23 cm and near 8 cm, according to the smoothing.

Familiarity narrows fields within a study by 10 to 40%. Zhao 2014 gives the largest contrast: 24 cm on the familiar 2 m track, recorded first, against 40, 35 and 33 cm on days 1 to 3 of a novel 1 m track. The two tracks also differ in length. A first exposure to any track matters more than a later new track (Dragoi 2013).

Track length beyond about 2 m does not widen mouse VR fields. The widths are 41 to 60 cm on 2 m, 38 cm on 3 m, 45 cm on 4 m, 46 to 50 cm on 8.3 m and 46 cm on 41 m. They come from different labs and rules, but no trend shows. In small 2D arenas the walls cap fields. Keinath ventral fields fill 89% of a 35 cm cylinder, and the dorsal per-field width drops from 27 to 22 cm in the 70 cm cylinder while the count doubles.

Mouse CA3 is not narrower than CA1. Of five same-study comparisons, Dong finds CA3 wider (44 against 38 cm). Hussaini (27.0 against 25.7 cm), Hainmueller (40 against 45 cm), Boublil (17.7 against 18.1 cm) and Sheintuch (9.6 against 8.8 cm) find no significant difference. Rat studies usually find CA3 smaller, so rat CA3/CA1 ratios should not be carried over.

## Field counts

### Dorsal CA1, ordered by environment size

| Study | Method | Environment | Fields per cell | Single-field or active fraction |
|---|---|---|---|---|
| Tarcsay 2025 | 1p | 0.11 m² octagon, familiar | median 2 [1-2] (mixture-model split) | |
| Keinath 2014 | tetrode | 35 cm cylinder | 1.06 (figure-read) | |
| Boublil 2026 | tetrode | 41 cm square, familiar | 2.26 (GLMM) | |
| Quigley 2023 | tetrode | 40 x 60 cm, novel | 1.70 ± 0.22 | |
| Donegan 2020 | tetrode | 60 x 40 cm | 2.0 (CA2 2.61) | |
| Arbab 2018 | tetrode | 64 cm circle | median 1 | |
| Keinath 2014 | tetrode | 70 cm cylinder | 2.20 (figure-read) | |
| Jarzebowski 2022 | 1p | 1.13 m² maze | 1.7 (intermediate 1.6) | |
| Schuette 2022 | 1p | 70 cm enclosure | 1.36 | 51% of neurons place cells |
| Dragoi 2013 | tetrode | 0.65 to 0.9 m arm, then about 1.55 m L-track | 1.5, then 1.9 | |
| Rechavi 2022 | 1p | 96 cm track | 1.67 on day 1 to 1.41 on day 22 (figure-read) | Place cells 15% to 28% of active cells |
| Sheintuch 2023 | 1p | 96 cm track | 2.00 per direction (CA3 1.69) | |
| Zhao 2014 | tetrode | Familiar 2 m track, then novel 1 m track | 1.29 on the familiar track, then 0.43 to 0.62 on the novel one, per cell per direction (all pyramidal cells) | 57% to 68% with fields |
| Sloin 2022 | silicon | 1.5 m track | 1.36 over both directions | 57% of active cells |
| Dombeck 2010 | 2p | 1.8 m VR | 1.09 | |
| Zaremba 2017 | 2p | 1.8 to 2 m belt | 1.18 (fields under half the largest dropped) | |
| Mou 2018 | tetrode | Novel 2 m track | 1.2 per direction | 36% multi-field (rats 12%) |
| Sosa 2025 | 2p | 4.5 m VR | 1.38 to 1.48 per place cell | 62.5% to 68.5% single-field |
| Chen 2026 | 1p | 8.9 m maze path, 0.92 m² footprint | 6.7 and 6.1 | Over 90% multi-field, negative binomial fit |
| Lee 2020 | 2p | About 41 m VR path | 4.68 per active cell, 3.38 including silent cells | 27.7% silent, 26% of active cells single-field |

Lee et al. 2020 (rows `lee2020_1`, `lee2020_2`) comes from the bioRxiv v1 preprint, because the Cell PDF returned 403. Four male Thy1-GCaMP6f mice ran winding VR paths about 41 m long and 14 cm wide, in one direction with teleport. Each mouse saw 3 of 4 environments, all familiar (at least 4 days). Each session imaged 700 to 870 dorsal CA1 neurons, and 125 to 430 were tracked for up to 3 months. A field was a region with calcium events on at least 2 of the first 4 laps within 30 cm. The Fig. 1G histogram (one session, 736 cells, read from the vector bars) gives 27.7% of cells with no field, 18.9% with one, and a long tail to 19. A gamma-Poisson (negative binomial) model fits, and an equal-propensity Poisson model is rejected. The ratio of the 90th to the 10th percentile of propensity is 6.1 to 48 across mice (mean 24.4), against 1.7 to 2.2 under Poisson. The distribution did not change across time or environment. Widths did not depend on field count. The preprint gives no gamma parameters and no propensity against environment size.

Count against environment size. Two studies change size within one study. Keinath doubled the cylinder diameter and fields per cell doubled (1.06 to 2.20 dorsal, 0.97 to 2.04 ventral). Dragoi's count rose from 1.5 on one arm to 1.9 on the L-track, but that step also adds novelty and experience, so it does not isolate size. Across studies the total rises with size and the density falls. Sosa gives 0.31 to 0.33 fields per metre per place cell on 4.5 m, and Lee 0.11 per metre per active cell on 41 m (both derived). Chen 2026 is the exception at 0.75 per metre, but its absolute 0.2 Hz threshold favours many fields. The rules differ, so these densities are not strictly comparable.

Several counts are 1 by construction. Radvansky 2018 and Lenkey 2025 keep one field per cell, Zaremba 2017 drops fields under half the largest, and Bourboulou 2019 keeps only the highest-peak field. Rows that enforce this say so in `fields_per_cell`.

### Other regions

- Granule cells have about one field and are mostly silent. Kim 2023 gives 1.20 fields with 16% of cells active in any session, GoodSmith 2022 a median of 1 with 24% active, and Senzai 2017 6.6% of cells with two or more fields on a T-maze. Danielson 2017 calls 1.4% of granule cells place cells, and Zhang 2020 (whole-cell) found 47% of granule cells silent.
- Mossy cells have several fields and are mostly active. Kim 2023 gives 1.87 fields with 95% active, GoodSmith 2022 a median of 2 with 82%, Kim 2020 3.51 to 4.44 on a 2 m belt, and Senzai 2017 57.8% with two or more fields.
- CA3 sits between, at 1.69 to 1.90 fields (Sheintuch, Boublil), 36.1% multi-field on a T-maze (Senzai) and 58% of cells with fields on a belt (Kim 2020).
- Ventral CA1 matches dorsal in the Keinath cylinders (0.97 and 2.04).

Place-cell fractions in dorsal CA1 range from 4% (Stefanini 2020, which the authors attribute to cell detection) and 10% (Radvansky 2018, olfactory VR in darkness) to 94% (Chen 2026 mazes). They depend on the rule as much as on the brain (Grijseels 2021).

## Gaps that remain

- Intermediate and ventral CA3 in mice have no field width or count. Ventral CA1 has two studies, and one uses a peak-only rule. Intermediate CA1 has one. No mouse study has dorsal, intermediate and ventral levels in one environment.
- No Neuropixels study of mouse hippocampus states a field width in its text.
- CA2 has no width in cm. Subiculum has one study with an ambiguous arena.
- Dentate widths in cm rest on Kim 2023 (approximate pixel size), Boublil 2026 (unit assumed, cell type not separated) and a figure read of Hainmueller 2018. Only Hainmueller compares granule cells with CA1 in the same rig.
- No mouse width exists in a 2D arena over 1.2 m², and none on a real track over 3 m. Large-environment widths are all two-photon VR.
- Count against size within one study exists only for 35 against 70 cm cylinders and one against two track arms. Lee 2020's propensity against size is not in the preprint.
- Many dorsal CA1 electrophysiology papers give widths only in figures or closed supplements (Sharif 2021, Jun 2020, McKenzie 2021, Valero 2022, Park 2025, Talbot 2018, Kee 2018, Harvey RE 2018). Some could be figure-read if the images were available.
- The 1p and 2p rules differ so much that no single "imaging width" exists. Use rows with matched rules, or the same-study ratios above.

## Papers to request

Most valuable first. These could not be opened (closed, 403, or values only in closed supplements or figures).

1. Lee JS, Briguglio JJ, Cohen JD, Romani S, Lee AK (2020). Cell 183:620-635. [doi:10.1016/j.cell.2020.09.024](https://doi.org/10.1016/j.cell.2020.09.024). Published numbers, gamma parameters and Fig. S3 (propensity against environment size). The rows use the 2019 preprint.
2. Nikbakht N et al. (2024). Efficient encoding of aversive location by CA3 long-range projections. Cell Rep 43:113957. [doi:10.1016/j.celrep.2024.113957](https://doi.org/10.1016/j.celrep.2024.113957). The only lead for intermediate CA3 place coding in mice.
3. GoodSmith D, Lee H, Neunuebel JP, Song H, Knierim JJ (2019). J Neurosci 39:9570-9584. [doi:10.1523/JNEUROSCI.0940-19.2019](https://doi.org/10.1523/JNEUROSCI.0940-19.2019). Granule, mossy and proximal CA3 field sizes in the same mice.
4. Huang LW, Torelli F, Chen HL, Bartos M (2024). Context and space coding in mossy cell population activity. Cell Rep 43:114386. [doi:10.1016/j.celrep.2024.114386](https://doi.org/10.1016/j.celrep.2024.114386). Mossy-cell widths on the Hainmueller 4 m VR rig.
5. Ding L et al. (2020). Structural correlates of CA2 and CA3 pyramidal cell activity in freely-moving mice. J Neurosci 40:5797-5806. [doi:10.1523/JNEUROSCI.0099-20.2020](https://doi.org/10.1523/JNEUROSCI.0099-20.2020). Identified CA2 and CA3 cells.
6. Jun H et al. (2020). Neuron 107:1095-1112. [doi:10.1016/j.neuron.2020.06.023](https://doi.org/10.1016/j.neuron.2020.06.023). Wild-type CA1 widths in a 1 m² box, figure only (a high-resolution figure would do).
7. Sharif F, Tayebi B, Buzsaki G, Royer S, Fernandez-Ruiz A (2021). Neuron 109:363-376. [doi:10.1016/j.neuron.2020.10.034](https://doi.org/10.1016/j.neuron.2020.10.034). Silicon-probe widths for 445 place cells, cue-rich against cue-poor, Fig. 1B only.
8. Li K, Kyprou M, Ciocchi S (2026). Integration of ethologically defined anxiety-related behaviors in the ventral hippocampus. bioRxiv. [doi:10.64898/2026.01.24.701488](https://doi.org/10.64898/2026.01.24.701488). Mouse ventral hippocampus units.
9. Rowland DC, Yanovich Y, Kentros CG (2011). PNAS 108:14654. [doi:10.1073/pnas.1105445108](https://doi.org/10.1073/pnas.1105445108). Mouse CA1 field sizes with the rule Keinath used.
10. Chen G, King JA, Burgess N, O'Keefe J (2013). PNAS 110:378. [doi:10.1073/pnas.1215834110](https://doi.org/10.1073/pnas.1215834110). Mouse place fields on a VR track against a real one.
11. Jarzebowski P et al. (2021 preprint). bioRxiv. [doi:10.1101/2021.09.07.459245](https://doi.org/10.1101/2021.09.07.459245). An unverified search summary says the preprint found no dorsal against intermediate difference, which would conflict with the published paper.
12. Kentros CG, Agnihotri NT, Streater S, Hawkins RD, Kandel ER (2004). Neuron 42:283-295. [doi:10.1016/S0896-6273(04)00192-8](https://doi.org/10.1016/S0896-6273(04)00192-8). Mouse CA1 field size under attention demands.
13. Grienberger C, Magee JC (2022). Nature 611:554. [doi:10.1038/s41586-022-05378-6](https://doi.org/10.1038/s41586-022-05378-6). Source data would give exact widths on a 180 cm belt.
14. Talbot ZN et al. (2018). Neuron 97:684-697. [doi:10.1016/j.neuron.2017.12.043](https://doi.org/10.1016/j.neuron.2017.12.043). Wild-type CA1 field size and count in Table S1.
15. Nakashiba T et al. (2012). Cell 149:188-201. [doi:10.1016/j.cell.2012.01.046](https://doi.org/10.1016/j.cell.2012.01.046), and McHugh TJ et al. (2007). Science 317:94-99. [doi:10.1126/science.1140263](https://doi.org/10.1126/science.1140263). Control CA3 field properties in supplements.

Other blocked mouse papers with possible widths: Gonzalez 2019 Science (10.1126/science.aav9199), Sheintuch 2020 Curr Biol (10.1016/j.cub.2020.02.018), Geva 2023 Neuron (10.1016/j.neuron.2023.05.005), Rotenberg 2000 J Neurosci (10.1523/JNEUROSCI.20-21-08096.2000), Agnihotri 2004 PNAS (10.1073/pnas.0400385101), Renaudineau 2009 PNAS (10.1073/pnas.0900484106), Mably 2017 Hippocampus (10.1002/hipo.22697) and Ciupek 2015 J Neurosci (10.1523/JNEUROSCI.3130-14.2015).

## Files

- Table: `sweep_fields_mouse.csv` (this folder).
- Working files, downloads and scripts: `/home/jackking/.claude/jobs/2c8b1c8e/tmp/sweep_mouse/`. The per-paper JSON inputs are in `json/` and the unopened lists in `unopened_*.txt`. `scripts/merge.py` checks the schema and vocabularies, `scripts/finalize.py` sorts the rows, and `scripts/coverage_md.py` prints the coverage table.
