# Retrosplenial cortex sweep: spatial fields and population kernels

Sweep date: 2026-11-23. Files: `sweep_fields_rsc.csv` (66 rows, 28 studies) and `sweep_kernel_rsc.csv` (11 rows, 4 studies; 9 rows have numbers).

## Summary

- RSC place-like fields on 1.5 to 5 m tracks are about as wide as CA1 fields recorded the same way. Mouse 2p widths are 25 to 50 cm, depending on the field rule, the mouse line and the subregion. Same-method RSC/CA1 width ratios are 1.07 (Mao 2017), 1.00 (Lenkey 2025) and 1.19 to 1.30 (Navratilova 2026). In Navratilova 2026 the difference disappears when the mouse line is matched.
- In rats the picture differs. On a plus maze, granular RSC fields covered 2.8 times the area of simultaneously recorded CA1 fields: equal-area diameters are 36.2 against 21.7 cm, a ratio of 1.67. In a 1 m box most RSC cells fire over nearly the whole box (sparsity about 0.95).
- In most paired studies fewer RSC cells are spatial than CA1 cells, with RSC/CA1 share ratios of 0.5 to 0.7. Lenkey 2025 is the exception: RSC 20.5% against CA1 17.4% (6196 neurons, 7 mice), a ratio of 1.18. Fields per place cell on a 1.5 m belt were 1.55 in RSC against 1.26 in CA1, which Mao 2017 report as not different (P = 0.21).
- Only one study gives a within-environment RSC kernel as a distance curve: Mao 2017, on a 150 cm treadmill loop. I re-read it from the PDF vector paths. r = 0.5 is 13.2 cm from a position, the curve crosses zero at 37.8 cm (the text says "~35 cm") and the floor is about -0.2. CA1 imaged the same way gives 13.4 cm and 33.0 cm. The granular, superficial dysgranular and deep subregions all reach r = 0.5 at 13 to 14 cm.
- Removing tactile cues widened the RSC kernel, from a zero crossing at 23.8 cm to 36.8 cm (matrix read). Fischer 2020 gives a split-half, landmark-centred VR kernel: r = 0.5 is 22 to 28 cm from the landmark.
- No rat RSC kernel with numbers was found. No RSC kernel exists for any environment longer than 2.8 m.

## Method

- Scope. Rat, mouse and other-species RSC single-unit or cellular-imaging studies that report spatial tuning numbers. Human fMRI was excluded.
- Search. Europe PMC REST queries on title and abstract terms (retrosplenial with place, spatial, position, route, landmark, head direction, boundary, egocentric, sequence), author queries for each starting lead, and WebSearch. References in the papers read were followed up.
- Reading was split four ways: Mao 2017 and Navratilova 2026; rat place, route and context papers; mouse papers; and head-direction, boundary and other-species papers. The rows were then merged. Spot checks against the source text confirmed Smith 2012 Table 2 (1031.44 and 368.16 cm²), Lenkey 2025 (20.5%, 665/3244; medians 37.31 cm) and Lozano 2017 (49.29 ± 1.28° against 43.65 ± 1.13°).
- Full-text sources were Europe PMC fullTextXML, the NCBI BioC API (for NIH manuscripts that Europe PMC refused), PDFs, figures and supplements from the PMC open-data bucket, the eLife reviewed-preprint API, and open PDFs on lab sites. bioRxiv returned HTTP 429. Publisher sites (cell.com, jneurosci.org, APA, Springer) refused downloads.
- Figure reading.
  - Vector paths: bars, step CDFs and curves read from PDF drawing commands and calibrated on the tick marks. Precision is about 0.5% of the axis.
  - Pixel traces: embedded rasters calibrated on vector ticks or on detected tick rows.
  - Colour-bar lookup for correlation matrices: each pixel was mapped to the nearest colour-bar colour, then averaged by circular separation. This is the least precise method, about ±0.03 in r near the diagonal.
- Status tags. "FT" means an exact quote from the full text. "FT; derived" shows the arithmetic in notes. "FT; figure-read (...)" names the method. "preprint" and "abstract" mean what they say. No search snippet was recorded as verified.
- Region labels. Rat Rga, Rgb and area 29 are RSC-granular, and Rdg and area 30 are RSC-dysgranular. Mouse RSPd and RSPv are granular. Mouse RSPagl ("agranular", area 30) is dysgranular. A window spanning both, or no stated subregion, is RSC-unspecified.
- Level labels. RSC spans about AP -1 to -8 mm in rats and -1 to -4 mm in mice. A row is called anterior or posterior only when the paper says so or the site is clearly in one half. The coordinates are in level_definition.
- Widths on tracks are lengths. 2D areas are kept and converted to equal-area diameter 2*sqrt(A/pi), tagged derived.
- Comparison rows. CA1 rows from the same study are kept as comparison rows (region CA1). Some duplicate rows in `sweep_fields_mouse.csv`: mao2017_1 and lenkey2025_1 there.

## Coverage

Studies by species, subregion and cell type (study ids as in the CSV):

| Species | Subregion | place-like | landmark | route | distance | HD | boundary | egocentric-boundary | conjunctive | other |
|---|---|---|---|---|---|---|---|---|---|---|
| rat | RSC-granular | smith2012, subramanian2024 |  | miller2019, vedder2017 | chen2026 | chen1994 |  | alexander2020 |  | miller2021 |
| rat | RSC-dysgranular | laurent2026 | jacob2017 |  | chen2026 | chen1994 |  | alexander2020 |  |  |
| rat | RSC-unspecified |  |  | alexander2017 | chen2026 | alexander2020, jacob2017, lozano2017, zhang2022 | vanwijngaarden2020 | alexander2020, lachance2024, vanwijngaarden2020 |  | alexander2020, zhang2022 |
| mouse | RSC-granular | mao2017 |  |  |  |  | park2024 |  |  | park2024 |
| mouse | RSC-dysgranular | lenkey2025, mao2017, mao2018, navratilova2026, sun2021 | fischer2020, sit2023 |  |  | sit2023 |  |  |  |  |
| mouse | RSC-unspecified | esteves2023, mao2017 |  |  |  | vandergoes2024 |  |  |  | campagner2023 |
| macaque | RSC-unspecified | sato2010 |  |  |  |  |  |  | sato2006 |  |

Environments:

| Study | Species, method | Environment | Size | Real or virtual | Familiarity |
|---|---|---|---|---|---|
| Mao 2017 | mouse, 2p | treadmill belt loop, tactile cues | 1.5 m | real, head-fixed | familiar |
| Mao 2018 | mouse, 2p | treadmill belt | 1.5 m | real, head-fixed | familiar |
| Esteves 2023 | mouse, 2p | treadmill belts | 1.5 m | real, head-fixed | familiar and novel |
| Lenkey 2025 | mouse, 2p | wheel in darkness, tactile cues | 1.57 m | real, head-fixed | familiar |
| Sun 2021 | mouse, 2p | three VR corridors | 2.0 m | virtual | learning |
| Fischer 2020 | mouse, 2p | VR corridor, landmarks | 2.2 m (short), 2.8 m (long) | virtual | trained |
| Navratilova 2026 | mouse, 2p | circular VR tracks | 3.14, 3.77, 5.03 m | virtual | familiar and novel |
| Park 2024 | mouse, 1p miniscope | polygon and circle chambers | 25 cm radius (0.196 m²) | real, freely moving | not stated |
| van der Goes 2024 | mouse, tetrodes | circular arena | 0.196 m² | real | familiar |
| Sit 2023 | mouse, 2p | rotating chamber | 0.049 m² | real, head-fixed | habituated |
| Campagner 2023 | mouse, Neuropixels | circular arena with shelter | 92 cm diameter (0.665 m²) | real | familiar |
| Smith 2012 | rat, stereotrodes | plus maze | arm size not given | real | across learning |
| Miller 2019 | rat, tetrodes | continuous T-maze | 5.1 m lap | real | across learning |
| Vedder 2017 | rat, tetrodes | T-maze | not given | real | across learning |
| Alexander & Nitz 2017 | rat, tetrodes | plus track with repeated segments; ring | not read | real | trained |
| Subramanian 2024, Miller 2021 | rat, tetrodes | open boxes | 1 m² | real | familiar |
| Chen 2026 | rat, tetrodes | cylinder | 0.785 m² | real | trained |
| van Wijngaarden 2020 | rat, tetrodes and silicon | square arenas | 0.25 to 1.44 m² | real | familiar and novel |
| Jacob 2017, Zhang 2022 | rat, tetrodes | two- and four-compartment boxes | 1.44 m² | real | repeated |
| LaChance 2024 | rat, tetrodes | square | 1.44 m² | real | habituated |
| Alexander 2020 | rat, tetrodes | square | 1.5625 m² | real | familiar |
| Lozano 2017 | rat, tetrodes | cylinder | 0.43 m² | real | familiar |
| Laurent 2026 | rat, tetrodes | 2 or 4 connected rooms | 2.25 m² | real | two sessions |
| Sato 2006, 2010 | macaque, single units | virtual town | not given | virtual | trained |

## Place-like field widths

| Row | Species, subregion, level | Environment | Rule | Width | RSC/CA1 |
|---|---|---|---|---|---|
| mao2017_rsc_1, _2 | mouse, superficial dysgranular, posterior | 1.5 m belt | above 20% of peak minus baseline | mean 39.6 ± 1.2 cm SEM (n = 297); median 34.5 [24.7, 46.4] cm | 1.07 (mean), 1.10 (median) |
| mao2017_rsc_3 | mouse, superficial dysgranular | 1.5 m belt | same | median 42.0 [30.7, 58.5] cm | |
| mao2017_rsc_4 | mouse, deep layers | 1.5 m belt | same | median 48.0 [34.5, 58.5] cm | |
| mao2017_rsc_5 | mouse, superficial granular | 1.5 m belt | same | median 49.5 [39.0, 61.5] cm (n = 30) | |
| mao2018_1, _2 | mouse, dysgranular, mid (AP -2.5), Thy1 | 1.5 m belt | same | about 25.0 cm control, 27.9 cm after hippocampal lesion | |
| lenkey2025_2 | mouse, dysgranular L2/3, anterior (AP -2.2), Thy1 | 1.57 m wheel, darkness | above 50% of range | median 37.31 [27.49, 49.09] cm | 1.00 |
| navratilova2026_1, _2 | mouse, dysgranular L2/3, mid | 3.1 to 5.0 m VR | first-to-last calcium event per lap | about 40 cm familiar, 38 cm novel | 1.30 familiar, 1.19 novel; text 19% |
| navratilova2026_3 | same, by mouse line | same | same | Thy1 29 cm; CaMK2a-tTA 38 cm | Thy1 RSC = Thy1 CA1 |
| smith2012_1 | rat, granular | plus maze | at least 4 adjacent 2.8 cm pixels, less than half the maze, in/out >= 2 | area 1031.44 cm², d = 36.2 cm (derived) | area 2.80, diameter 1.67 |
| park2024_1 | mouse, granular, mid | 25 cm radius chambers | vertex score | vertex fields about 80 cm², d = 10.1 cm (derived) | |

The rules differ: a 20% edge, a 50% edge, and a first-to-last event span. Compare widths within a study, not across studies.

## Shares of spatial cells and fields per cell

- Place-cell shares in mice depend strongly on labelling. With AAV, 13% of superficial dysgranular cells were place cells (297/2256). With Thy1 GP4.3 the share was 56% (1416/2544) in Mao 2017 and 0.50 ± 0.02 in Mao 2018. Lenkey 2025, also Thy1, found 20.5% (665/3244) with a stricter rule. Navratilova 2026 found 27.0 ± 11.3% familiar and 19.1 ± 10.7% novel. Esteves 2023 found 0.46 on a familiar belt and 0.41 on a novel one.
- Mao 2017 subregion shares: superficial dysgranular 15 ± 3%, superficial granular 15 ± 1%, deep layers 6 ± 2%.
- Rat shares: 23.0% granular place cells on a plus maze, against 38.3% in CA1 (Smith 2012). RSC rose from 8.9% before training to 28.0% at asymptote. 45% of RSC cells were position-tuned by an L-N model, against 85% in CA1 (Subramanian 2024).
- Fields per place cell from count distributions:
  - Mao 2017, 1.5 m belt: RSC 0.555, 0.336 and 0.106 of place cells had 1, 2 and 3 fields, a mean of about 1.55. CA1 had 0.752, 0.223 and 0.020, a mean of about 1.26 (not different from RSC, P = 0.21). Per metre that is about 1.03 against 0.85 fields per place cell (derived).
  - Mao 2017 subregion means: 1.49 superficial dysgranular, 1.45 deep, 1.21 superficial granular (the bars sum to 0.966; corrected by the audit).
  - Esteves 2023, sham mice on a familiar 1.5 m belt: 0.79, 0.20 and 0.006, a mean of 1.21, or 0.81 per metre (derived; corrected by the audit).
  - Navratilova 2026: 0.093 fields per cell per metre over all imaged cells (CA1 0.146). Divided by the 27% share, that is about 0.34 fields per place cell per metre, or 1.1 to 1.7 fields per place cell on its 3.1 to 5.0 m tracks (derived, rough). In Fig. S2A about 18% of familiar place cells had two or more fields. Fields per cell per metre fell to 63% of the familiar value in a novel environment, against 55% in CA1.
- Field counts against environment size: no study varied size and counted fields. Across studies, real 1.5 m belts give about 0.8 to 1.0 fields per place cell per metre. The 3 to 5 m VR tracks give about 0.34. That gap mixes track length with VR and with the rule. Navratilova 2026 had 3.14 m and 5.03 m versions but did not compare them. Sun et al. 2024 (J Physiol) imaged RSC in two open fields of different sizes, but it could not be opened.

## Population similarity against distance (kernels)

| Row | Data | r = 0.5 distance (full width) | Zero crossing | Floor | Status |
|---|---|---|---|---|---|
| K-mao2017_1 | mouse RSC place cells, 297 cells, 4 mice, 150 cm loop | 13.2 cm (26.4) | 37.8 cm (text "~35 cm") | about -0.20 at 75 cm, -0.23 near 90 cm | vector paths |
| K-mao2017_2 | CA1 on the same treadmill | 13.4 cm (26.8) | 33.0 cm | -0.26 at 76 cm | vector paths |
| K-mao2017_3 | superficial dysgranular, 123 cells | 13.8 cm (27.6) | 36.9 cm | -0.33 | pixel trace |
| K-mao2017_4 | deep layers, 60 cells | 14.1 cm (28.2) | 33.3 cm | -0.32 | pixel trace |
| K-mao2017_5 | superficial granular, 30 cells | 13.3 cm (26.6) | 32.6 cm | -0.25 | pixel trace |
| K-mao2017_6 | Thy1, cue belt | 9.8 cm (19.6) | 23.8 cm | -0.37 | colour-bar lookup |
| K-mao2017_7 | Thy1, blank belt | 12.3 cm (24.6) | 36.8 cm | -0.31 | colour-bar lookup |
| fischer2020_k1 | mouse dysgranular, VR, split-half, at the landmark, short trials | 28 cm before, 23 cm after (51) | 94 before, 70 after | about -0.1 | pixel trace |
| fischer2020_k2 | same, long trials | 27 before, 22 after (49) | 82 after | -0.13 to -0.18 | pixel trace |

Two more rows have no numbers. alexander2017_k1 records that rat RSC odd/even population matrices repeat at quarter-route spacing on the plus track, while CA1's do not (preprint text only). miller2019_k1 is a rat cross-half matrix on a 5.1 m T-maze lap; its figure was not obtained.

Notes on the kernels:
- Mao 2017 builds vectors from place cells only, scales each cell to 0 to 1, uses 1.5 cm bins and smooths with a 4.5 cm Gaussian. The loop sends the curve back up toward +0.6 at 150 cm.
- The RSC kernel full width at r = 0.5 is 0.67 of the mean field width (26.4/39.6) or 0.77 of the median (derived). CA1 gives 0.72 (26.8/37.0).
- Fischer's peak is 0.76 to 0.82 because the vectors come from split halves, and its 5 cm bins widen the crossing. It also changes along the track: it is broad at the start and sharpest at the landmark. So it is not directly comparable with Mao.
- Re-verification of the review's Mao row: zero crossing about 35 cm confirmed in the text, 37.8 cm on the vector path. The review's CA1 crossing "about 34 cm" reads 33.0 cm on the vector path. The floor "about -0.2" is confirmed. The r = 0.5 distance (13.2 cm) was not in the review.

## Excluded items and why

Not kernels:
- Mao 2017 Fig. 5b and 5d and Suppl. Figs. 4 and 5: light against dark and original against shifted reward. Between conditions.
- Mao 2018 SI Fig. S5C and Fig. 4E: between days.
- Navratilova 2026 Fig. 3: correlations between lap sets at matched bins, plus single-session example matrices with the colour scale clipped at 0. No distance curve.
- Fischer 2020 Fig. 3 supplement 1: short-track against long-track populations. Sun 2021: between VR contexts.
- Smith 2012 and Miller 2021: block-to-block and context-to-context map correlations. Laurent 2026: room to room within a multi-room box.
- Alexander & Nitz 2017 Fig. 6 "distance reconstruction": vectors mirrored about a symmetry point, so it measures distance from that point. Carstensen 2021: correlation over 400 ms time bins.
- van Wijngaarden 2020 Fig. 6 supplement 1: correlation against distance from a wall, not separation between two positions.
- Jacob 2017 and Zhang 2022: between-compartment correlations.
- Esteves 2023 Fig. 3 matrices: lesioned mice only, colour scale clipped at 0, 749 px figure. Not read.

No usable RSC spatial numbers:
- Mouse: Voigts & Harnett 2020 (L5 information measure only), Voigts 2025, Franco & Goard 2021 and 2024 (joystick T-maze without locomotion), Powell 2020 (visual receptive fields; visual responses "limited to the caudal RSC"), Minderer 2019, Madruga 2025, Bech 2026, Esteves 2026 (imaged M2), Hou 2024 (no widths or totals), Lian 2026 (model).
- Rat: Lin 2025, Hayashi & Sato 2024, Alexander 2018, Carstensen 2021 (rate changes only), Subramanian 2024 NLM (trial-start cells).
- Mouse Neuropixels (Campbell 2021, PMC8437084): RSC area A29 on a 400 cm VR track with towers every 80 cm. RSC firing peaked about 20 cm before each landmark. Maps stayed stable under gain change; the map shift was -1.4 ± 0.4 cm against -4.1 ± 0.5 cm in MEC. Coding was retrospective, 0.06 ± 0.02 s. No RSC field sizes or shares appear in the text, so it is notes only.
- Other cell types: Keshavarzi 2022 and Hennestad 2021 (angular head velocity), Brennan 2020 (slice), Lomi 2021 (theta and tracing).
- Primate: Liu, Tian & Gu 2021 (macaque RSC tuned to self-motion, "lacked significant spatial tuning"). Yu 2022 recorded posterior cingulate, not RSC.

## How cues, darkness and environment shift RSC fields

- Tactile cues. Removing the belt cues kept RSC place cells but added lap-to-lap jitter. SI fell from 0.86 to 0.61, and the kernel's zero crossing moved from about 24 to 37 cm. CA1 kept "the same rapid decrease in correlation with distance" without cues, with SI 0.93 against 0.86 (Mao 2017). So RSC depends more on local cues than CA1 here.
- Darkness.
  - Mao 2017: RSC place fields were "robust to changes in illumination", with "a small fraction" rearranging. CA1 behaved the same.
  - Lenkey 2025: in total darkness with tactile cues, 20.5% of cells were place cells and widths equalled CA1.
  - Darkness lowered mouse RSC HD tuning (median resultant 0.123 to 0.087) but not ADn tuning (van der Goes 2024).
  - In rats: bidirectional cells kept their pattern in darkness in the compartment box. 75% of tetradirectional cells kept the fourfold pattern in the first dark trial and 44% in both. Border scores did not change (Jacob 2017, Zhang 2022, van Wijngaarden 2020).
- Visual landmarks and optic flow.
  - In VR, 22% of active dysgranular neurons were landmark-aligned, and the population code was sharpest at the landmark (Fischer 2020).
  - RSC maps followed the visual landmarks through gain changes (Campbell 2021, notes only). Mao 2020 (abstract) reports sequences driven by visual flow combined with movement.
  - Landmark cells are the largest tuned class in posterior dysgranular RSC under head fixation (Sit 2023).
- Repeated segments and symmetric spaces.
  - On a plus track with repeated sub-routes, 47.2% of rat RSC cells repeated their firing at sub-route scale (quarters 21%, halves 20.5%), and 38.9% cycled once over the full route.
  - On a ring track the half-route share rose to 44.4% and the quarter-route share fell to 13.3%. The population matrices aliased at quarter-route spacing, which CA1 did not (Alexander & Nitz 2017, preprint).
  - In connected rooms, RSC maps repeated across rooms after rotation (180° for 2 rooms, 90° for 4), while hippocampal cells remapped in 91% to 100% of cases (Laurent 2026, preprint).
  - Multidirectional HD and egocentric bearing cells take the rotational symmetry of the enclosure (Jacob 2017, Zhang 2022, LaChance 2024).
- Geometry. Granular RSC cells fire near polygon vertices (fields about 80 cm²). Vertex selectivity fades in 12-gon and 24-gon chambers and disappears without physical walls (Park 2024). Replacing walls with drop-offs moved egocentric boundary cell preferred distance about 10 cm outward (Alexander 2020).
- Novelty and learning.
  - Rat RSC place-field share rose with training, from 8.9% to 28.0% (Smith 2012).
  - Mouse RSC remapped globally in a novel VR room, like CA1. Its place-cell share dropped from 27% to 19%, and field size did not change. Unlike CA1, RSC fields did not expand or shift backwards across laps (Navratilova 2026).
  - Bilateral hippocampal lesions cut the RSC place-cell fraction from 0.50 to 0.29 without changing width (Mao 2018). Lesioned mice formed little spatial information on a novel belt (Esteves 2023).
- Environment size. No usable data. See the gaps section.

## RSC against CA1 recorded the same way

| Study | Match | Width | Share | Other |
|---|---|---|---|---|
| Mao 2017 | mouse 2p, same treadmill, same rule | 39.6 vs 37.0 cm mean (1.07); 34.5 vs 31.5 cm median (1.10) | 13% vs 26% imaging (0.50) | Fields per place cell 1.55 vs 1.26 (1.23; not different, P = 0.21). Kernel r = 0.5 at 13.2 vs 13.4 cm (0.99); zero at 37.8 vs 33.0 cm (1.15) |
| Lenkey 2025 | mouse 2p, same wheel in darkness, same rule, Thy1 | 37.31 vs 37.31 cm median (1.00) | RSC 20.5% (CA1 share in the mouse sweep) | CA1 amplitude slightly higher |
| Navratilova 2026 | mouse 2p, same VR rooms, same rule; RSC mostly CaMK2a-tTA, CA1 Thy1 | about 40 vs 31 cm familiar (1.30), 38 vs 32 cm novel (1.19); none with the line matched | 27.0% vs 39.7% familiar (0.68); 19.1% vs 28.1% novel | Fields per cell per metre 0.093 vs 0.146 (0.64). SI 0.71 vs 0.83. Decoder error 18.9 vs 14.5 cm. Backward shift in 8% vs about 20% of fields |
| Smith 2012 | rat, same rats, simultaneous, same rule, plus maze | area 1031 vs 368 cm² (2.80); diameter 36.2 vs 21.7 cm (1.67) | 23.0% vs 38.3% (0.60) | In-field/out-of-field rate 8.9 vs 50.3 |
| Subramanian 2024 | rat, same lab, boxes and analysis, different rats | not defined | 45% vs 85% position-tuned (0.53) | Sparsity about 0.95 vs about 0.2 (eye read) |
| Alexander & Nitz 2017 (preprint) | rat, same track | not read | | Odd/even reconstruction error 1.9 vs 5.2 cm; RSC aliases at repeats |
| Laurent 2026 (preprint) | rat, same apparatus | none | | RSC repeats across rooms; hippocampus remaps |
| Miller 2021 vs Law 2016 | rat, same lab and boxes, different studies | | | Between-context map r = 0.32 vs 0.05 |

Pattern across studies: head-fixed mouse imaging gives RSC fields close to CA1 width. Freely moving rat recordings give broader, less selective RSC firing. The Navratilova discussion makes the same point and calls freely moving RSC fields "at least an order of magnitude larger" than CA1 fields. That statement is secondary and has no numbers behind it in the papers read here. The rat plus-maze ratio is 2.8 in area.

## Other RSC cell types with tuning numbers

- HD tuning width (two circular SDs) is 49.29 ± 1.28° in rat RSC against 43.65 ± 1.13° in PoS, a ratio of 1.13 (Lozano 2017). Zhang 2022 gives 109° for HD cells and 134 to 139° for multidirectional cells, with the width definition unstated.
- HD shares are 8 to 10% in rats (Chen 1994, Jacob 2017, Zhang 2022), 4.9% in Alexander 2020, 12% in mouse tetrodes (van der Goes 2024) and 14% in mouse 2p (Sit 2023).
- Egocentric boundary cells are 17.3% of rat RSC cells: 38.7% in dysgranular, 25.0% in the border zone and 9.9% in granular RSC. Their preferred distance is skewed toward the animal, about 10 to 15 cm (figure-read, uncertain) (Alexander 2020). Egocentric bearing cells are 40% (LaChance 2024).
- Border cells are 9.0% (485/5415), with preferred wall distance peaking at 5 cm and reaching up to 20 cm. 39.2% of them are also egocentrically tuned (van Wijngaarden 2020). In mice, granular border cells are 27% (Park 2024).
- Route cells: 28.3% stem splitters at asymptote (Vedder 2017) and 21% (Miller 2019). Distance-to-goal cells: 13.4%, 14.2% in dysgranular against 11.6% in granular (Chen 2026, preprint).
- Shelter-direction (egocentric) cells: 3.5% of mouse RSP units for an open or a closed shelter (Campagner 2023, the "Vale" lead).

## Corrections to the starting leads and to the review

- Miller, Mau & Smith 2019 used tetrodes in rats, not calcium imaging.
- Fischer, Mojica Soto-Albors, Buck & Harnett 2020: the last author is Harnett, not Harvey.
- "Mao, Neumann, Sutherland, McNaughton" matches Mao, Neumann, Sun, Bonin, Mohajerani & McNaughton 2018 (PNAS). I found no Sutherland co-authored RSC paper. I found no joint Smith and Mao paper; Mao 2018 and Mao 2020 cover RSC sequences.
- "Vale" matches Campagner, Vale et al. 2023 (Nature), on shelter-direction coding rather than place fields.
- The review (`LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`, regional section) says "No source gives field sizes in metres for nucleus reuniens, retrosplenial cortex or prefrontal cortex." That no longer holds for RSC. Mao 2017 reports 39.6 ± 1.2 cm, and Lenkey 2025, Mao 2018 and Navratilova 2026 add more widths. `field_size_regions_species.md` lists Mao 2017 as abstract only with widths "Not obtained"; the full text has them.

## Gaps

- No rat RSC kernel with numbers. The two candidates are Miller 2019 Fig. 1F and Alexander & Nitz 2017 Fig. 1C. The second would show how repeated segments shape the kernel.
- No RSC kernel beyond 2.8 m. Navratilova 2026 recorded on 3 to 5 m VR loops but published no distance curve. Its data could give one.
- No place-like field width for RSC in a freely moving 2D open field, apart from a plus-maze area and polygon vertex fields. The 1 m box data suggest no compact fields.
- No study varies environment size and counts RSC fields. Sun 2024 (J Physiol, two open fields) may do this and is unopened.
- Anterior against posterior: no paired measurement of place-like fields. Lenkey 2025 (AP -2.2) and Mao 2017 (caudal) differ in rule and mouse line. Yang 2026 (PNAS) compares anterior and posterior RSC but is embargoed in PMC until 2027-01-28. Powell 2020 shows only that visual responses are caudal.
- Granular against dysgranular: only Mao 2017 for place cells (no difference in width or count). Elsewhere it is only shares of egocentric boundary, bidirectional, HD and distance cells.
- Mouse line confound. Thy1 GP4.3 mice give higher place-cell shares (50 to 56% against 13% with AAV) and smaller fields (29 against 38 cm in Navratilova 2026; about 25 cm in Mao 2018 against 39.6 cm with AAV in Mao 2017 under the same rule). Any RSC/CA1 ratio needs matched lines. This pattern is my reading across studies.
- No HD tuning width in mice. No border field length along the wall. No egocentric boundary receptive-field size in text.
- No human single-unit or bat RSC data. Macaque data pool RSC with area 7m and posterior cingulate.

## Papers to request (could not open), most valuable first

1. Alexander AS, Nitz DA (2017). Spatially periodic activation patterns of retrosplenial cortex encode route sub-spaces and distance traveled. Curr Biol 27:1551-1560. doi:10.1016/j.cub.2017.04.036. Needed for track dimensions, final percentages and the odd/even population matrices (a repeated-segment kernel, with CA1).
2. Alexander AS, Nitz DA (2015). Retrosplenial cortex maps the conjunction of internal and external spaces. Nat Neurosci 18:1143-1151. doi:10.1038/nn.4058. Rat route-position tuning on a W-shaped track at two room locations.
3. Esteves IM et al. (2021). Spatial information encoding across multiple neocortical regions depends on an intact hippocampus. J Neurosci 41:307-319. doi:10.1523/JNEUROSCI.1788-20.2020 (PMC7810652, not in the OA subset). Shares and possibly widths per cortical area, including RSC.
4. Sun H, Cai R, Li R, Li M, Gao L, Li X (2024). Conjunctive processing of spatial border and locomotion in retrosplenial cortex during spatial navigation. J Physiol. doi:10.1113/jp286434. Mouse RSC in two open fields of different sizes.
5. Mao D, Molina LA, Bonin V, McNaughton BL (2020). Vision and locomotion combine to drive path integration sequences in mouse retrosplenial cortex. Curr Biol 30:1680. doi:10.1016/j.cub.2020.02.070. VR gain and visual-only conditions; possibly widths and kernels.
6. Miller AMP, Mau W, Smith DM (2019), figures. Curr Biol 29:2083. doi:10.1016/j.cub.2019.05.034, or bioRxiv doi:10.1101/315200. Fig. 1F would give a rat RSC kernel.
7. Cho J, Sharp PE (2001). Head direction, place, and movement correlates for cells in the rat retrosplenial cortex. Behav Neurosci 115:3-25. doi:10.1037/0735-7044.115.1.3. Rat open-field place and HD shares.
8. Demchuk AM, Esteves IM, Chang H, Sun J, McNaughton BL (2024). J Neurosci. doi:10.1523/JNEUROSCI.1619-23.2024 (PMC11293447, not in the OA subset). RSC place-cell fractions, laminar gradients and lesion effects.
9. Yang Y et al. (2026). Anterior and posterior retrosplenial cortex employ distinct strategies for egocentric-allocentric transformation. PNAS. doi:10.1073/pnas.2600565123. Embargoed in PMC to 2027-01-28.
10. Gianatti M et al. (2023). Multiple long-range projections convey position information to the agranular retrosplenial cortex. Cell Rep 42:113109. doi:10.1016/j.celrep.2023.113109.
11. Chen LL, Lin LH, Green EJ, Barnes CA, McNaughton BL (1994). Head-direction cells in the rat posterior cortex. I. Exp Brain Res 101:8-23. doi:10.1007/BF00243212. Part II: doi:10.1007/BF00243213.
12. Chinzorig C et al. (2020). Cereb Cortex 30:1985-2004. doi:10.1093/cercor/bhz183. Rat RSC with darkness and treadmill-off conditions.
13. Lande AS et al. (2023). Representations of tactile object location in the retrosplenial cortex. Curr Biol. doi:10.1016/j.cub.2023.09.019. Not read; found late.
14. Cheng N et al. (2024). Neuron. doi:10.1016/j.neuron.2023.11.018. Egocentric cells in RSC somata, dendrites and spines.
15. Zhang X et al. (2026). Cell Rep. doi:10.1016/j.celrep.2026.118296.
16. Sato N, Sakata H, Tanaka YL, Taira M (2010). Cereb Cortex 20:846-858. doi:10.1093/cercor/bhp147, and (2006) PNAS 103:17001. doi:10.1073/pnas.0604277103. Check whether RSC neurons are reported apart from 7m and posterior cingulate.
17. Lower priority: Ivan et al. 2024 (doi:10.1111/ejn.16558), Chang et al. 2020 (doi:10.1098/rstb.2019.0228), Koay et al. 2022 (doi:10.1016/j.neuron.2021.10.020), Subramanian & Smith 2024 (doi:10.1002/hipo.23635, title not checked).

## Working files

Downloads, scripts and part files are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/sweep_rsc/` (dl/, scripts/, parts/). `scripts/write_M.py`, `write_A.py`, `B_write.py` and `write_C.py` rebuild the part CSVs, and `scripts/merge.py` merges them into the two CSVs here.
