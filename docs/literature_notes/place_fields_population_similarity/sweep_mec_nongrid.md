# Sweep for non-grid spatial tuning in medial entorhinal cortex

Sweep of 2026-11-23. Real neural data only. The tables are `sweep_fields_mec_nongrid.csv` (32 rows, 19 studies) and `sweep_kernel_mec.csv` (3 rows, 1 study) in this folder.

The short answer is that the literature measures non-grid MEC cells mostly as shares of the population and as stability or remapping numbers. Field sizes are rare. I found one study with a non-grid field size and grid spacing in the same animals: Mallory et al. 2018 in dorsal mouse MEC. There, the equal-area diameter of a non-grid field is 56 cm and grid spacing is 36 cm, a ratio of 1.55. No study gives non-grid field size at more than one dorsoventral level, and no study gives a within-environment population kernel for non-grid MEC cells.

## Method

### Search and retrieval

1. I started from the 13 leads in the brief and from the review's own notes (`field_size_regions_species.md`, `large_fields_followup.md`, `remapping_orthogonality.md`, `population_similarity_followup.md`, `decoding_dimensionality.md`).
2. I resolved DOIs to PMC ids with the PMC id converter and NCBI E-utilities, then ran web searches for non-grid, irregular, aperiodic, border, band, object-vector and cue cells combined with field size, dorsoventral position and population vector terms.
3. Full text came from the NCBI BioC API, which served 29 PMC articles. Europe PMC `fullTextXML` returned HTTP 500 for every request in this session. Figures and supplements came from the PMC open-data bucket (Tang 2014, Obenhaus 2022) and from Springer's static supplement server (Mallory 2018, Hoydal 2019).
4. Failed routes: science.org supplements (403), jneurosci.org (403), Europe PMC article renders and `/bin/` files (403), eScholarship (403), bioRxiv (429 rate limit), the Nature and Springer article pages (login redirect). BioC returned no JSON for PMC articles outside the open-access subset (Savelli 2008, Tang 2015, Latuske 2015, Sun 2015, Kitamura 2015), and NCBI efetch returned only their front matter.
5. I opened 34 papers or supplements in full and read 4 more as abstracts. The Gardner 2022 Extended Data Fig. 2 source data came from the earlier grid sweep's download.

### How numbers were recorded

- Each row carries an exact quote with its location. Quotes are verbatim except that plus-minus signs are written "+-" and squared units "cm2". The status column says "full text", "abstract", "derived" (with the arithmetic in the notes), "preprint" or "secondary". No search snippet is recorded as verified.
- 2D field areas are converted to the equal-area diameter 2*sqrt(A/pi) and tagged as derived. The original area stays in `value_reported`.
- Border width is the field's extent perpendicular to the wall. No study I could open reports border field length along the wall.
- Shares are given as the paper's counts, with derived percentages marked.
- Rows from one study share cells. Mallory MNG01 to MNG03 and MNG05 come from the same recordings. MNG06 to MNG09 are one pool of 794 cells. MNG10 to MNG12 share 345 cells. Diehl 2019 (MNG13, MNG14) is a separate cohort from Diehl 2017.
- Scripts are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/sweep_mec/scripts/` (`fetch_many.py`, `kgrep.py`, `gardner_nongrid.py`, `build_csv.py`). Downloads are in `.../sweep_mec/dl/`, one directory per source.

## Coverage

### Species, cell type and dorsoventral level

Counts are studies. "Share" means the paper gives the fraction of cells in the class. "Width" means a field size. "Count" means fields per cell. "Stability" covers session-to-session, remapping and drift numbers.

| Species | Cell type | Dorsal | Intermediate | Ventral | Unspecified or mixed |
|---|---|---|---|---|---|
| Rat | Non-grid spatial (irregular, aperiodic) | share: Diehl 2017, Diehl 2019, Tang 2014 (L2); stability: Diehl 2017, Diehl 2019 | none | none | share: Tang 2015 (L3, abstract), Savelli 2008 (abstract); stability: Butler 2019 |
| Rat | Band-like or periodic non-grid | share and stability: Krupic 2012 | none | none | none |
| Rat | Border | share: Tang 2014 | none | none | none |
| Rat | All non-module units | share: Gardner 2022 source data | none | none | none |
| Mouse | Non-grid spatial | width and coverage: Mallory 2018; share: Perez-Escobar 2016, Miao 2017, Gu 2018 | none | none | count and share: Hardcastle 2017; share: Low 2021 |
| Mouse | Border | width: Mallory 2018 | none | none | count: Hardcastle 2017; share and stability: Campbell 2018, Low 2021 |
| Mouse | Object-vector | none | none | none | share: Obenhaus 2022, Andersson 2021, Hoydal 2019 (secondary via Andersson) |
| Mouse | Cue cells (VR) | none | none | none | share and coverage: Kinkhabwala 2020 |
| Mouse | Boundary-vector-like | none | none | none | count and wall distance: Hoydal 2019 supplement |
| Other species | any | none found | | | |

No intermediate or ventral row exists for any non-grid class. Most "unspecified" studies sampled the dorsal 1.5 mm.

### Environments

| Study | Species | Environment | Size | Real or virtual | Familiarity |
|---|---|---|---|---|---|
| Mallory 2018 | mouse | open arena | not in the accessible text | real | not stated |
| Hardcastle 2017 | mouse | open arena | 100 x 100 cm (most), 91.4 and 70 cm | real | placed in it from day 3 after surgery |
| Diehl 2017 | rat | square and 16-gon; second room square | 80 x 80 cm, 50 cm radius; 100 x 100 cm | real | trained |
| Diehl 2019 | rat | squares and circle | 80 x 80 cm, 100 cm diameter; 120 x 120 cm | real | at least 7 days |
| Butler 2019 | rat | two arenas | 1.5 x 1.5 m | real | trained task in ENV2 |
| Krupic 2012 | rat | square; circle | 1.3 x 1.3 m; 1 m diameter | real | familiar |
| Tang 2014 | rat | square | 70 x 70 cm or 1 x 1 m | real | 3 to 7 days |
| Perez-Escobar 2016 | mouse | circular platform; track | 80 cm diameter; 80 cm track | real | trained |
| Miao 2017 | mouse | box | 1 m | real | familiar |
| Kinkhabwala 2020 | mouse | linear tracks | 8 m (10 m, 18 m imaging) | virtual | trained |
| Campbell 2018 | mouse | open field; track | 90 x 90 cm; 4 m | real; virtual | trained |
| Low 2021 | mouse | repeating track | 4 m | virtual | trained |
| Gu 2018 | mouse | linear tracks | 10 m or 18 m | virtual | trained |
| Obenhaus 2022 | mouse | square with objects | mostly 80 x 80 cm | real | not read |
| Andersson 2021 | mouse | square with objects | 80 x 80 cm | real | trained |
| Gardner 2022 | rat | open field | 1.5 x 1.5 m | real | familiar |
| Brun 2008 (kernel) | rat | linear track | 18 m | real | familiar |

No non-grid measurement comes from a large real environment. The largest are Butler's 1.5 m arenas (no field size in the main text) and Gardner's 1.5 m open field (no field analysis of non-grid units).

## Non-grid field widths and counts

### Field size

- Mallory 2018, dorsal mouse MEC, wild-type littermates (iWT). "Spatially stable cells" that are neither grid nor border cells had a mean field size of 2473.45 +- 2233.2 cm2 (mean +- SD, n = 32). The equal-area diameter is 56.1 cm, and +-1 SD in area spans 17.5 to 77.4 cm. Their fields covered 43.24 +- 30.72% of the box. In HCN1 knockouts the values were 2692.29 +- 2768.8 cm2 (58.5 cm) and 45.58 +- 31.62% (n = 38). Table 1. [full text; diameters derived]
- Grid fields in the same iWT mice were 489.29 +- 301.62 cm2, a diameter of 25.0 cm. A non-grid field therefore has about 5 times the area of a grid field here.
- Border width in the same mice was 14.43 +- 6.53 cm perpendicular to the wall (n = 86). Mallory defines it as the mean distance from the wall to the field's inner edge.
- Hoydal 2019 supplement: of 860 cells, 20 met a boundary-vector criterion and 14 of these were border cells. For the other 6, the field centre sat 8.2 +- 0.7 cm from the wall. [full text, supplement]

I found no field size for non-grid cells in rats. Diehl 2017 and 2019 define fields (0.3 x peak, at least 250 cm2) but give no non-grid field sizes in the main text or legends. Butler 2019 states that non-grid field size, total field area and field number did not change between environments, with the numbers in Fig. S10A, which I could not open.

### Fields per cell

- Hardcastle 2017, mouse, mostly 1 m2 boxes, field rule of at least 5 bins of 2 x 2 cm above 1 Hz and 20% of peak: non-grid position-encoding cells 1.90 +- 0.10 fields (mean +- SEM, n = 307), border cells 1.55 +- 0.23 (n = 42), grid cells 5.69 +- 0.35 (n = 101). Non-grid cells have a third of the grid cells' field count in the same boxes. This is about 1.9 fields per m2, a derived rate because box sizes varied. [full text]
- Kinkhabwala 2020, mouse VR, 8 m tracks: a cue cell's fields covered roughly 0.2 to 0.4 of the track (derived from the per-bin field fraction, 0.4 +- 0.1 in cue regions and 0.2 +- 0.1 outside, with a lenient p < 0.3 field rule). Field number follows the number of cues.

### Share of MEC cells that are spatial but not grid

The answer depends on the criterion:

| Study | Species, layer | Non-grid spatial share | Grid share | Criterion |
|---|---|---|---|---|
| Diehl 2017 | rat, L2/3 dorsal | 67.5% of spatial cells; about 64% of all cells (derived) | 17.7% of spatial cells | per-cell shuffle on spatial information and split-half correlation |
| Diehl 2019 | rat, L2/3 dorsal | about 90% of cells spatial in 80 to 100 cm boxes, about 55% in a 120 cm box (all classes) | | same criteria |
| Hardcastle 2017 | mouse, L2/3 | 307/794 = 38.7% (derived) | 101/794 = 12.7% | LN-model position encoding, not grid or border |
| Krupic 2012 | rat, dorsal mEC | 81/239 = 34% periodic non-grid (derived) | 75/239 | 2D Fourier periodicity |
| Perez-Escobar 2016 | mouse, dorsal | 226/880 = 25.7% (derived) | 139/880 = 15.8% | score thresholds |
| Low 2021 | mouse, VR | 24.57 +- 3.12% "spatially stable" other cells; border 13.23% | 18.26% | gain-response classes |
| Tang 2014 | rat, L2 | about 30/193 = 16% (derived) | 22/193 = 11% | spatial information shuffle |
| Tang 2015 | rat, L3 | 19/69 = 28%, all irregular | 0 | spatial information (abstract) |
| Mallory 2018 | mouse, dorsal | 32/527 = 6.1% "spatially stable" (derived) | 37/527 = 7.0% | stability shuffle |
| Miao 2017 | mouse, dorsal | 214 aperiodic cells against 248 grid cells (two cohorts) | | spatial information above 99th percentile |

Specific non-grid classes: object-vector cells 13.6% (67/492, Andersson 2021) and 17.7 +- 7.8% in animals rich in them (Obenhaus 2022). Cue cells are about 13% of units on cue-rich VR tracks (Kinkhabwala 2020). Border cells are 5% (Hardcastle), 7% (Perez-Escobar), 9.3% of spatial cells (Diehl 2017) and 14% (Campbell 2018). In Gardner's Neuropixels recordings, 74 to 92% of all units fell outside grid modules, but that count includes non-spatial units and parasubiculum.

### Stability and remapping, non-grid against grid

- Within a box across sessions, non-grid spatial cells are less stable than grid cells: median spatial correlation 0.62 against 0.85 and 0.70 (Diehl 2017), and r = 0.45 against 0.54 (Krupic 2012). Campbell 2018 points the other way within one session: VR half-session stability was 0.54 for border cells against 0.40 for grid cells.
- Across box shape or colour in one room, non-grid fields moved by a median 13.1 and 15.3 cm more than between matching boxes, while grid patterns moved by 1.3 and 2.8 cm more than between matching boxes (Diehl 2017). Across rooms the increases were 28.9 cm (non-grid) and 20.9 cm (grid).
- Between two 1.5 m arenas, 196/271 non-grid spatial cells remapped. Their mean correlation was 0.32 +- 0.22 in task-trained rats and 0.41 +- 0.27 in controls (Butler 2019).
- Over six hours, 16% of non-grid cells changed their spatial pattern and 16% changed their rate. Grid cells did not change (Diehl 2019). Cells whose spatial pattern changed were more common in layer 3; rate changes showed no such layer excess.
- Of spatially periodic cells, 11% switched between grid and non-grid patterns within an environment and 32% between environments (Krupic 2012).

## Width over grid spacing in the same recordings

| Row | Recording | Non-grid or border width | Grid spacing, same animals | Ratio |
|---|---|---|---|---|
| MNG01 | Mallory 2018, iWT mice, dorsal MEC (grid cells 386 +- 222 um from the dorsal border) | 56.1 cm, equal-area diameter of the mean non-grid field (2473 cm2) | 36.26 +- 9.27 cm (37 cells) | 1.55 |
| MNG03 | Mallory 2018, HCN1 knockout | 58.5 cm (2692 cm2) | 42.16 +- 10.53 cm (62 cells) | 1.39 |
| MNG04 | Mallory 2018, iWT, border cells | 14.4 cm perpendicular to the wall | 36.26 cm | 0.40 |
| MNG05 | Mallory 2018, iWT, grid fields (comparator) | 25.0 cm (489 cm2) | 36.26 cm | 0.69 |
| MK01 to MK03 | Brun 2008, rat grid cells on an 18 m track (reference, grid only) | 56, 92, 119 cm mean field width | median 171, 301, 370 cm | 0.33, 0.31, 0.32 |

Caveats for the Mallory ratio:
- The ratio uses the mean of per-cell mean field areas. The SD is about as large as the mean, so the typical cell may be smaller.
- The arena size and field rule are in the paywalled Online Methods. Box size bounds field size for broad cells.
- Grid spacing is per cell from the autocorrelogram, and the module structure is not given. The smallest spacing per mouse had a median of 26.6 cm (Supp. Fig. 7e).
- The spacings (mean 36 cm, smallest per mouse 26.6 cm) point to the one or two most dorsal modules, so the ratio is against dorsal grid scale only.

Butler 2019 has both quantities (non-grid field size in Fig. S10A, grid spacing per rat in Table S1) and is the next candidate. Gardner 2022 has both in its raw data but reports no non-grid field analysis.

## Does non-grid scale follow the dorsoventral axis?

No study measures it. The evidence is indirect:

1. Diehl 2017: "recordings of grid and non-grid spatial cells were highly intermixed along the dorso-ventral axis." No sizes by level.
2. Hardcastle 2017: "Unlike the increase in the spatial distance between grid firing fields along the dorsalventral MEC axis, the degree of mixed selectivity did not change from dorsal to ventral MEC." This covers coding type, not field size, over 0 to 1.54 mm.
3. Mallory 2018 gives a causal test within dorsal MEC. HCN1 knockout raised grid spacing by 16% (36.26 to 42.16 cm, p = 0.002), but non-grid field area did not change (2473 to 2692 cm2, p = 0.90), nor did border width (14.43 to 14.23 cm, p = 0.76). In these recordings non-grid field size did not follow a change in grid scale.
4. Fyhn 2004 (abstract only): "Precise positional modulation was not observed more ventromedially in the entorhinal cortex." This predates grid classification and gives no field sizes.
5. Savelli 2008 recorded "from the dorsal and ventral MEC", but only the abstract was retrievable.

So whether non-grid field size scales along the axis is open. The one manipulation suggests that non-grid field size is set independently of grid scale. A dorsoventral comparison is the clearest gap in this sweep.

## Kernels

- Brun 2008, re-verified against the PDF text (pp. 1207 to 1208): "The mean half-width was 32 cm in the dorsal group, 47 cm in the intermediate group, and 67 cm in the ventral group." Widths at r = 0.20 were 54, 102 and 184 cm. The earlier audit read these half-widths as full widths from the Fig. 5 raster, so dist_r05 is 16, 23.5 and 33.5 cm. All 143 track cells were grid cells (55, 59 and 29 per group, from 9, 6 and 4 rats). Rows MK01 to MK03 match `sweep_kernel.csv` K04 to K06.
- I found no within-environment kernel for non-grid MEC cells or for all MEC cells.
- Superseded by the audit (audit_sweep_mec.md): this sweep proposed dropping K26 (Dasbehera, Dogra and Redman 2025, arXiv 2511.08292) as simulation only. Its v2 reanalyses public Neuropixels recordings from Campbell et al. 2021, so K26 stays, with the audit's corrections.

### Between-session and other comparisons that are not kernels

| Study | What it measures | Why it is not a kernel |
|---|---|---|
| Diehl 2017 | PV and spatial cross-correlation between sessions as a function of x-y shift; displacement of the central peak | between sessions; reports displacement, not decay with distance |
| Neunuebel 2013 | MEC PV correlation matrices on a 76 cm circular track, standard 1 vs standard 2 and cue rotations | between sessions; MEC cells not split into grid and non-grid |
| Low 2021 | trial-by-trial similarity of MEC maps on a VR track | across trials, not across positions |
| Campbell 2021 | trial similarity matrices under gain changes | across trials |
| Heys 2014, Gu 2018 | correlation of track tuning against anatomical distance between cells | anatomical distance, not spatial separation |
| Obenhaus 2022 | nearest-neighbour anatomical distances between cell classes | anatomical |
| Mallory 2018 Supp. Fig. 11b | field width from PV cross-correlation | CA1 place cells, not MEC |
| Gardner 2022 | toroidal manifold of grid modules | no position kernel |
| Hardcastle 2017, Butler 2019 | position decoding error | decoding, not similarity against distance |
| Dasbehera 2025 | decorrelation against distance | v2 reanalyses real Neuropixels data (Campbell 2021); see audit_sweep_mec.md |

## Excluded items

- Gardner 2022 source data, non-grid "grid_spacing_cm". The column is the autocorrelogram peak distance computed for every unit. For non-grid units the medians are 70 cm (rat Q), 94.5 cm (R day 1), 57 cm (R day 2) and 56 cm (S), against module spacings of 57 to 127 cm. It is not a field width, it includes non-spatial units, and the 1.5 m box limits it. Not used.
- Krupic 2012 band wavelengths: in Fig. 1G and the SOM only, neither retrievable.
- Brun 2008 track fields: grid cells only, already in the grid tables.
- Heys 2014 VR field widths (44 +- 24 cm, 0.5 fields per metre) and Domnisoru 2013: grid cells only. Domnisoru shows non-grid fields only in Supp. Fig. 26.
- Rowland 2018: grid shares among optogenetically tagged stellate cells (23.2%) and untagged cells (33.8%); no non-grid spatial class.
- Campbell 2021 distance-tuned cells: putative grid cells; only 5/134 non-grid cells were distance tuned.
- Ismakov 2017, Jacob 2019, Gerlei 2020, Munn 2020, Jun 2020, Ying 2022, Kraus 2015, Chen 2018: grid, place, speed or time cells; no non-grid field numbers.
- Perez-Escobar 2016 track: class counts only; no widths.
- Monkey entorhinal studies (Killian 2012 and later) map gaze position on a screen, not navigation. Not included.
- Kropff 2015 speed cells: not opened; the brief's condition (spatial numbers) is unlikely to be met.

## Gaps

1. Non-grid field size in rats. None found. Diehl's Table S1 and Butler's Fig. S10A probably hold it.
2. Non-grid field size at intermediate and ventral MEC levels. None in any species.
3. Non-grid fields in large or long environments. The largest real box with a non-grid field number is about 1 m2. No 1D track study gives non-grid field widths in real space.
4. Border field length along the wall. Only the perpendicular width (Mallory) and wall-to-centre distances (Hoydal supplement) exist in what I could open.
5. Object-vector field size and distance tuning. The numbers are in the paywalled Hoydal 2019 main text and Extended Data.
6. Any population kernel for non-grid or all MEC cells within one environment.
7. Non-grid MEC cells in species other than rats and mice. Bat MEC (Yartsev 2011) was not opened.
8. Field counts against environment size for non-grid cells. Diehl 2019 shows the spatial share falls from 90% to 55% from small boxes to a 120 cm box, but gives no field counts.

Public data that could fill gaps 1, 3 and 6 directly: Gardner 2022 (figshare; Neuropixels with every unit in a 1.5 m box), Perez-Escobar 2016 (Dryad), and the Giocomo lab data used by Hardcastle and Mallory.

## Papers to request

Ordered by value for this review.

| Rank | Paper | DOI | What it would add |
|---|---|---|---|
| 1 | Mallory et al. 2018, Online Methods | 10.1038/s41593-017-0055-3 | arena size and field rule behind the only same-recording ratio |
| 2 | Butler, Hardcastle & Giocomo 2019, Supplementary Materials | 10.1126/science.aav5297 | non-grid field size (Fig. S10A) and grid spacing per rat (Table S1), a second ratio in 1.5 m boxes |
| 3 | Diehl et al. 2017, Supplemental Information | 10.1016/j.neuron.2017.03.004 | Table S1 and Figs. S1 and S2: per-rat classes, layer and DV positions, maybe field metrics |
| 4 | Hoydal et al. 2019, main text and Extended Data | 10.1038/s41586-019-1077-7 | OV share, field size, distance distribution, DV distribution |
| 5 | Solstad et al. 2008, with SOM | 10.1126/science.1166466 | border field length along the wall, share by layer and DV level |
| 6 | Krupic, Burgess & O'Keefe 2012, SOM | 10.1126/science.1222403 | band wavelengths against grid wavelengths in the same animals |
| 7 | Quirk et al. 1992 | 10.1523/JNEUROSCI.12-05-01945.1992 | pre-grid MEC field sizes in cylinders, with recording sites |
| 8 | Fyhn et al. 2004 | 10.1126/science.1099901 | dorsocaudal against ventromedial MEC spatial firing |
| 9 | Savelli, Yoganarasimha & Knierim 2008 | 10.1002/hipo.20511 | boundary cells and spatial cells in dorsal and ventral MEC |
| 10 | Tang et al. 2015 | 10.1523/JNEUROSCI.0696-15.2015 | layer 3 irregular spatial cells, possibly field metrics |
| 11 | Kanter et al. 2017 | 10.1016/j.neuron.2017.03.001 | non-grid MEC cells under layer II depolarization |
| 12 | Bjerknes, Moser & Moser 2014 | 10.1016/j.neuron.2014.02.014 | border cell fields in young rats |
| 13 | Yartsev, Witter & Ulanovsky 2011 | 10.1038/nature10583 | bat MEC border and non-grid cells |
| 14 | Navratilova, Godfrey & McNaughton 2016 | 10.1152/jn.00699.2015 | whether band cells survive unit isolation checks |
| 15 | Sun et al. 2015; Kitamura et al. 2015 | 10.1073/pnas.1511668112; 10.1016/j.neuron.2015.08.036 | island and ocean non-grid cells in mice |
| 16 | Latuske, Toader & Allen 2015 | 10.1523/JNEUROSCI.0276-15.2015 | irregular spatial cells by burst class |
| 17 | Zhang et al. 2013 | 10.1126/science.1232627 | share of irregular cells among hippocampus-projecting MEC cells |
