# Rat MEC grid spacing per module, for the field-width over grid-spacing table

Checked on 2026-11-20. Real neural data only. This note supplies the columns of a table whose rows are hippocampal recordings and whose cells are field width divided by grid spacing.

Status tags used below:

- [full text]: quoted from the paper or its supplement, which I read.
- [abstract]: only the abstract was reachable.
- [figure-read]: read from a figure. Each row says how.
- [derived]: my arithmetic, shown.
- [source data]: computed by me from the authors' published per-cell data file.
- [secondary]: reported by a later paper. Not checked against the original.

How figures were read. Stensola Fig. 5b, Supp. Fig. 18b and Supp. Fig. 2c are vector graphics. I extracted the marker and line coordinates with PyMuPDF and calibrated them against the axis frame and tick labels. Two checks support the calibration. Rat 14257's four module means come out at 39.1, 48.0, 65.2 and 96.6 cm, against 39.1, 47.9, 65.1 and 96.6 cm printed in Fig. 4a. The mean successive ratios come out at 1.415, 1.429 and 1.434, against 1.401, 1.429 and 1.434 printed in Fig. 5d. Stensola Fig. 4e, Supp. Fig. 2f, Hafting Fig. 2h and Barry thesis Fig. 4.14 were partly or wholly raster. I rendered those at 3 to 8 times zoom and read them by eye against the tick marks, so expect errors of about 1 to 2 cm or 0.03 mm.

Text-layer note. In the Stensola PDF text layer, "6" encodes ±, "5" encodes =, "," encodes <, "." encodes >, a leading "2" encodes a minus sign, and "mm" often encodes μm. The quotes below use the printed symbols.

## Proposed columns

Familiar-environment values, modules ranked by spacing within each animal. Module rank is assigned per animal. Stensola et al. say module means "were widely distributed between rats, with no apparent peaks in the distribution" (p. 75), so neighbouring ranges overlap. The core ranges use Stensola's five data sets with four detected modules, which also span the most dorsoventral distance. DV positions are distance from the dorsal MEC border along layer II.

| Column | Spacing range | Lower end | Upper end | DV position | Cross-checks |
|---|---|---|---|---|---|
| M1 | 39 to 49 cm | 39.1 cm, rat 14257, Stensola Fig. 4a [full text, number printed in figure] | 49.2 cm, Stensola Supp. Fig. 18b data set 4 [figure-read] | Rat 14147: median about 0.62 mm, box 0.40 to 0.82 mm, whiskers 0.02 to 1.12 mm [figure-read, Fig. 4e] | Hafting 2005: 39.1 to 43.0 cm next to the postrhinal border in a 2 m cylinder [full text]. Widening to every Stensola data set whose smallest module is under 56 cm gives 32.6 to 55.0 cm [figure-read]. Hafting's 1 m box shows about 29 to 35 cm at the border [figure-read] |
| M2 | 48 to 60 cm | 47.9 cm, rat 14257, Fig. 4a [full text] | 59.5 cm, Supp. Fig. 18b data set 4 [figure-read] | Rat 14147: median about 0.75 mm, box 0.60 to 0.85 mm, whiskers 0.40 to 1.00 mm [figure-read] | Gardner 2022 smallest modules R1 57.8 to 60.4 cm and S1 59.7 cm [source data] sit at the top of this range |
| M3 | 65 to 93 cm | 65.1 cm, rat 14257, Fig. 4a [full text] | 93.1 cm, data set 4 [figure-read] | Rat 14147: median about 1.27 mm, box 1.18 to 1.33 mm, whiskers 0.97 to 1.50 mm [figure-read] | Gardner Q1 71.5 cm, R2 83.9 to 89.1 cm [source data] |
| M4 | 97 to 128 cm | 96.6 cm, rat 14257, Fig. 4a [full text] | 127.6 cm, data set 7 [figure-read] | Rat 14147: median about 0.95 mm, box 0.71 to 1.28 mm, whiskers 0.64 to 1.76 mm [figure-read]. In this rat M4 sits partly dorsal to M3 | Gardner Q2 102.4 cm, R3 107.6 to 125.7 cm [source data] |
| M5 | about 136 to 172 cm, sparse | 136.2 cm, a pooled KSD peak, Supp. Fig. 2c [figure-read] | 171.7 cm, largest single cell, tangential rats in a 220 cm box, p. 72 [full text] | Not measured numerically. Ventral to M4 where sampled. Supp. Fig. 2f shows 95 to 135 cm grids at 0.75 to 1.9 mm in two multisite rats [figure-read] | Module means of 155.9 cm (Supp. Fig. 18b) and peaks of 142.4 and 162.9 cm (Supp. Fig. 2c) [figure-read]. 112.7 cm, the mean of the five M4 values, times 1.42 is 160 cm [derived]. A putative fifth module in rat 14147 was "too large" to measure in a 150 cm box [full text] |
| M6 and beyond | No familiar-box value | | | Ventral MEC, beyond about 50% of the DV axis | Only linear-track data, 18 m track. Brun 2008: "approximately 3 m at the most ventral recording positions" [abstract]. Solstad 2009 thesis: inter-peak distance "at least 8m" [thesis abstract]. Stensola extrapolation: "~10 m" with 8 to 10 modules [full text, Supp. Fig. 18 legend] |

Ratio between successive modules. Mean 1.42 [full text]. The printed "s.d. = 0.02" is the spread of the three mean ratios, not the spread across animals. Individual successive ratios run from 1.14 to 1.78, with s.d. 0.175 [figure-read and derived, n = 24 or 25 pairs]. Gardner's ratios are 1.28 to 1.48 [source data]. Barry's thesis gives about 1.7 [full text].

Where the modules sit. All four Stensola modules lie within about 1.9 mm of the dorsal border, about the dorsal half of MEC. The ratio statistic is stated for "the dorsal 50% of the MEC" (Supp. Fig. 18 legend). MEC is about 3.8 to 4 mm long dorsoventrally [derived, rows S21 and B2]. Modules overlap. At 37.0 ± 7.3% of recording locations, cells came from more than one module (p. 74).

Anatomy statement for pairing columns with rows (Canto, Wouterlood & Witter 2008, Neural Plast 2008:381243, full text):

> "Interconnected portions of the LEC and MEC close to the rhinal fissure, in rats referred to as the dorsolateral band of entorhinal cortex, are connected to the dorsal (nonprimate) or posterior (primate) part of the hippocampal formation (see Figures 1 (c), 1 (d)). Interconnected cells in the intermediate band, encompassing again parts of both LEC and MEC, connect to the intermediate hippocampal formation, whereas the most medially interconnected band of entorhinal cortex is mainly connected to the ventral (nonprimate) or anterior (primate) hippocampal formation."

The paper does not say "dorsal MEC". Linking the dorsolateral band to dorsal MEC, where M1 lies, takes one inference. The rat dorsal MEC border is the postrhinal border along the rhinal fissure, and Dolorfo & Amaral name the septal-projecting MEC part "caudomedial", the region the grid papers call dorsocaudal MEC (see section 5).

## 1. Stensola et al. 2012

Stensola H, Stensola T, Solstad T, Frøland K, Moser MB, Moser EI (2012). The entorhinal grid map is discretized. Nature 492:72-78. doi:10.1038/nature11649. I read the main text from the University of Tartu ownCloud copy, 7 pages, journal pp. 72 to 78. I also read the Supplementary Information from Nature's servers: MOESM246, the Supplementary Methods, 17 pages, and MOESM247, the Supplementary Figures, 26 pages. "SI-M" and "SI-F" give page numbers in those two PDFs.

Sample. "we recorded sequentially or simultaneously from 968 widespread grid cells in 15 animals, with up to 186 grid cells per animal" (p. 72). Five rats had tangential 4-tetrode bundles, recorded in a 220 × 220 cm box. Ten had 12-tetrode multisite drives, recorded in 100, 150 or 190 cm boxes (SI-M p. 3). Rats trained in the boxes for 2 to 3 weeks before testing (SI-M p. 3), so baseline sessions are familiar. Spacing method: "grid scale was expressed as the arithmetic mean distance from the field centers associated with Axes 1-3 to the autocorrelogram midpoint" (SI-M p. 7), using the six inner-ring fields of the spatial autocorrelogram (SI-M p. 6).

| ID | Value | Quote | Location | Status | N | Arena | Method | DV position |
|---|---|---|---|---|---|---|---|---|
| S1 | 35.2 to 171.7 cm, cell range | "Grid spacing spanned from 35.2 cm to 171.7 cm (74.4 ± 35.6 cm, mean range ± s.d.)." | p. 72 | full text | 5 tangential rats, up to 62 cells each (45.6 ± 19.0); n = 228 cells in the depth correlation | 220 × 220 cm | Autocorrelogram inner ring | Tracks covering "up to 1,800 μm of layers II and III, or more than one-third of the dorsoventral length of the MEC" (p. 72). Fig. 1b depth axis 1.0 to 3.5 mm from the brain surface |
| S2 | 74.4 ± 35.6 cm | Same sentence, "mean range ± s.d." | p. 72 | full text | 5 rats | 220 cm | | Read as the per-animal spread of spacing, averaged over rats. The wording is ambiguous |
| S3 | Arena sizes 100 to 220 cm | "foraged in 100–220-cm-wide square boxes" (p. 72). "The box size was 220 cm × 220 cm for all tangentially implanted animals and either 100 cm x 100 cm, 150 cm x 150 cm or 190 cm x 190 cm for multisite implanted animals. A small set of experiments were performed in a 180 cm circular enclosure" | p. 72; SI-M p. 3 | full text | 15 rats | | | |
| S4 | Max 5 modules per animal by k-means | "The maximum k value was 5 (rat 14147)." | p. 74 | full text | | | k from KSD peaks of spacing and orientation | |
| S5 | Max 4 or 5 modules detected | "The maximum number of modules detected in any animal was four or five, with some uncertainty about the fifth module because the putative grids were too large for periodicity to be detected in this study" | p. 75 | full text | | | | |
| S6 | Modules per data set: 1 to 4 | Supp. Fig. 18b: 4 data sets with 1 module, 7 with 2, 1 with 3, 5 with 4 | SI-F p. 24 | figure-read, vector markers | 17 data set positions, 41 modules | | k-means | The text says 16 data sets (p. 73). The figure has 17 positions |
| S7 | Rat 14257 module means 39.1, 47.9, 65.1, 96.6 cm | Fig. 4a legend: "Mean grid spacings in cm are indicated" | p. 75, Fig. 4a | full text, printed numbers | 62 cells, 1 rat | 220 cm | k-means module means | Fig. 4a depth from the brain surface along a tangential track: M1 from 2.0 to 3.2 mm, M2 from about 2.55 mm, M3 from about 2.95 mm, M4 at about 3.2 mm [figure-read] |
| S8 | Rat 14257 KSD peaks 38.8, 48.4, 65.0, 98.4 cm | Fig. 1d legend: "Red text, spacing in cm for the estimated peaks" | p. 73, Fig. 1d; SI-F p. 24 | full text, printed numbers | 62 cells | 220 cm | KSD peaks | as S7 |
| S9 | Rat 14257 ratios 1.23, 1.34, 1.51 | Printed in Supp. Fig. 18a: "M2/1=1.23", "M3/2=1.34", "M4/3=1.51", "M4/1=2.54" | SI-F p. 24 | full text | 1 rat | 220 cm | From KSD peaks | 48.4 / 38.8 = 1.247, not 1.23 [derived]. The other three printed ratios match |
| S10 | Rat 14147 KSD peaks 46.6, 63.9, 93.4, 118.9 cm, ratios 1.37, 1.46, 1.27 | Printed in Supp. Fig. 18a | SI-F p. 24 | full text | 176 cells; 133 cells "after experience-dependent stabilization" in Fig. 4d | 150 × 150 cm (SI-F p. 23) | KSD peaks | See S19 |
| S11 | Mean successive ratio 1.42 | "The ratio between successive module averages (M2/M1, M3/M2, M4/M3) fluctuated around a constant value of 1.42 (s.d. = 0.02; Fig. 5d)" | p. 75 | full text | 16 data sets per Fig. 5 | mixed | | Supp. Fig. 18 legend: "an average scale ratio of 1.42 and an S.D. of only 0.02 in the dorsal 50% of the MEC" (SI-F p. 24) |
| S12 | Mean ratios by pair 1.401, 1.429, 1.434; average 1.421 | Printed in Fig. 5d | p. 75 | full text | | | | The s.d. of the three means is 0.018 [derived], which is the printed 0.02 |
| S13 | Individual successive ratios 1.14 to 1.78, mean 1.42, s.d. 0.175 | Fig. 5d shows "Individual module pairs in grey" | p. 75 | figure-read from Fig. 5b lines (25 pairs) and Supp. Fig. 18b markers (24 pairs) [derived s.d.] | 13 to 17 data sets | mixed | | M2/M1 1.14 to 1.78, M3/M2 1.29 to 1.57, M4/M3 1.19 to 1.60 (Fig. 5b) |
| S14 | Increment between modules 24.9 ± 13.3 cm | "mean ± SD increment size 24.9 ± 13.3 cm; Cv = 50.8%" | SI-F p. 24 | full text | all data sets | | | |
| S15 | 41 module means, 32.6 to 155.9 cm | Supp. Fig. 18c legend: "all module means for grid scale (n=41) pooled across animals" | SI-F p. 24 | figure-read, Supp. Fig. 18b vector markers | 41 modules | mixed, box per module not given | | |
| S16 | Pooled by rank: M1 32.6 to 121.3 (median 46.5, n = 17); M2 48.0 to 155.9 (median 57.1, n = 13); M3 65.2 to 113.6 (median 80.3, n = 6); M4 96.6 to 127.6 cm (median 111.0, n = 5) | | SI-F p. 24 | figure-read | 41 modules | mixed | Rank within data set | Data sets that missed dorsal MEC call a large module "M1". One data set's modules are 121.3 and 155.9 cm |
| S17 | Four-module data sets: M1 39.1 to 49.2, M2 48.0 to 59.5, M3 65.2 to 93.1, M4 96.6 to 127.6 cm | Values per set: 49.2/59.5/93.1/111.0; 39.4/53.0/79.7/105.3; 39.1/48.0/65.2/96.6; 47.3/57.1/81.0/127.6; 46.5/52.8/77.2/122.8 | SI-F p. 24; same five lines in Fig. 5b, p. 75 | figure-read | 5 data sets | mixed | k-means means | Basis of the proposed columns |
| S18 | Pooled KSD peaks of tangential rats, 39.6 to 162.9 cm | Supp. Fig. 2c, "All peaks detected in KSDs of rats with tangential implants" | SI-F p. 6 to 7 | figure-read, vector lines; about 1 cm low against rat 14257's printed peaks | 21 peaks, 5 rats | 220 cm | KSD peaks | Values: 39.6, 44.2, 45.4, 47.8, 52.0, 52.6, 56.1, 56.4, 56.8, 64.0, 79.5, 80.4, 81.4, 97.3, 102.0, 104.0, 114.3, 121.3, 136.2, 142.4, 162.9 |
| S19 | DV position per module, rat 14147 | Fig. 4e legend: "Boxplot showing dorsoventral (DV) distribution of recording locations for each module" | p. 75, Fig. 4e | figure-read; boxes and two whisker sets from vectors, medians and the rest from a render | 176 cells, 9 tetrodes | 150 cm | | Medians about 0.62, 0.75, 1.27 and 0.95 mm for M1 to M4. Boxes as in the proposed columns. Zero is the dorsal border: flat maps "were prepared by mapping the dorsal border of the entorhinal cortex onto a straight line" (SI-M p. 16) |
| S20 | Spacing against distance from the dorsal border, two multisite rats | Supp. Fig. 2f, "recording depth is measured from the dorsal border of MEC" | SI-F p. 6 to 7 | figure-read, raster render | rats 15708 and 15314, 5 tetrodes | 100 to 190 cm | | About 40 to 55 cm at 0.1 to 1.2 mm; about 75 to 85 cm at 0.45 to 1.2 mm; about 95 to 135 cm at 0.75 to 1.9 mm. The depth of each step differs between tetrodes and rats |
| S21 | Sampled DV extent about 1.9 mm, "nearly 50%" | "The largest area covered by tetrodes in any animal spanned nearly 50% of the dorsoventral axis" (p. 75). Fig. 4f: "approximately 50% of the dorsoventral axis ... (~1,900 μm ...)" | p. 75 | full text | rat 14147 | | | MEC DV length about 1.9 / 0.5 = 3.8 mm [derived] |
| S22 | Arena caps the largest measurable spacing | Supp. Fig. 17: "the fields and the putative spacing were too large to allow quantification of spatial periodicity, despite the large scale of the recording environment (150 cm x 150 cm). It is possible that these cells represent a 5th grid module that could have been detected in larger environments." | SI-F p. 23 | full text | 8 cells, rat 14147 | 150 cm | | |
| S23 | Grid-score rings stop 10 cm short of the box width | "to a maximum of 10 cm less than the width of the box" | SI-M p. 5 | full text | | | Grid score | A 100 cm box allows rings out to 90 cm |
| S24 | Extrapolated ventral spacing about 10 m | "With 8-10 modules and a constant distance between the dorsal borders of the successive modules, an average scale factor of 1.42 would yield grid-field distances of ~10 m in the ventral part of the MEC, which is consistent with data from this region in large linear environments" | SI-F p. 24 | full text, extrapolation by the authors | | | | Ventral MEC |
| S25 | Module overlap along DV | "The proportion of recording locations containing cells from more than one grid module was 37.0 ± 7.3%. A total of 10.3 ± 3.8% locations had more than two modules." | p. 74 | full text | tangential rats | 220 cm | | "the modules may be organized as horizontal bands parallel to the dorsal border of the MEC" (p. 75) |

Discrepancy. Fig. 5b has a three-module line at 76.9, 121.3 and 156.1 cm [figure-read]. Supp. Fig. 18b has that data set with only 121.3 and 155.9 cm, plus four single-module data sets: 51.8, 32.6, 34.5 and 85.6 cm. Both figures total 41 modules if Fig. 5b's three missing modules are single-module sets. I could not resolve which version is right.

## 2. Brun et al. 2008

Brun VH, Solstad T, Kjelstrup KB, Fyhn M, Witter MP, Moser EI, Moser MB (2008). Progressive increase in grid scale from dorsal to ventral medial entorhinal cortex. Hippocampus 18:1200-1212. doi:10.1002/hipo.20504. PMID 19021257. Only the abstract was reachable. The paper is free on Wiley ("bronze" open access per Unpaywall and Semantic Scholar), but Wiley returned HTTP 403 to curl and to WebFetch. It is not in PMC. The thesis copy on NVA also returned 403. Box values from this paper were not obtained.

| ID | Value | Quote | Location | Status | N | Arena | Method | DV position |
|---|---|---|---|---|---|---|---|---|
| B1 | About 50 cm dorsal to about 3 m ventral | "The average interpeak distance between the fields increased from approximately 50 cm in the most dorsal part to approximately 3 m at the most ventral recording positions." | Abstract, Europe PMC record | abstract | not in abstract | 18 m linear track | Interpeak distance along the track | Most dorsal to most ventral recording positions |
| B2 | Recordings spanned more than 3 mm | "The recordings spanned a range of more than 3 mm, covering approximately three quarters of the dorsoventral extent of the medial entorhinal cortex." | Abstract | abstract | | 18 m track | | MEC DV length about 3 / 0.75 = 4 mm or more [derived] |
| B3 | Spacing and field size roughly linear with DV position, more variable ventrally | "The increase in average spacing and field size was approximately linear but this relationship coincided with a substantial increase in the variability of each measure." | Abstract | abstract | | 18 m track | | |
| B4 | Inter-peak distance under 1 m to at least 8 m | Solstad thesis English abstract: "In the MEC field length ranged from less than 50cm to approximately 3m, and inter-peak distance ranged from less than 1m to at least 8m." | Solstad T (2009), NTNU thesis 2009:167, NVA record 0198cc6087c8-637e01b6-4364-4aaa-a65b-0d9195b70f67, field alternativeAbstracts | abstract of thesis; paper III of the thesis is the MEC track study | | 18 m track | | "dorsal-most 75% of the MEC (Paper III)" |
| B5 | 143 grid cells, 15 rats; DV groups dorsal 0 to 25% (55 cells), intermediate 25 to 50% (59), ventral 50 to 75% (29) | "Brun et al. [5] recorded 143 grid cells in 15 rats that shuttled back and forth on a 18m long, linear track" | Kerdels & Peters 2018, arXiv:1810.07429, pp. 7 to 8 | secondary | 143 cells, 15 rats | 18 m track | | Locations "distributed between 1% and 75% along the dorsoventral axis" |
| B6 | Median spacing 171 ± 13, 301 ± 23, 370 ± 46 cm; minimum spacing 91 ± 12, 202 ± 24, 269 ± 47 cm (dorsal, intermediate, ventral) | Kerdels & Peters Table 1 | arXiv:1810.07429, p. 8 | secondary | 55, 59, 29 cells | 18 m track | Interpeak distances of detected fields. The survey says the field detector "missed a substantial number of visually discernable fields due to low firing rates. Thus, the median values may overestimate the true grid spacing." | |
| B7 | Mean field width 56 ± 1, 92 ± 6, 119 ± 7 cm; fields per 18 m 8.4, 5.9, 4.6 | Kerdels & Peters Table 1 | arXiv:1810.07429, p. 8 | secondary | | 18 m track | | |

Contradiction to flag. B6 gives a dorsal median of 171 cm and a dorsal minimum of 91 cm, but the abstract (B1) says "approximately 50 cm in the most dorsal part". The survey's explanation is missed fields, which inflates the interpeak distance. "Most dorsal" in B1 may also be a narrower group than the 0 to 25% bin. Track spacing is a 1-D section through a 2-D lattice, so it is not directly comparable with box spacing. I recommend that box columns use box values only.

## 3. Hafting et al. 2005

Hafting T, Fyhn M, Molden S, Moser MB, Moser EI (2005). Microstructure of a spatial map in the entorhinal cortex. Nature 436:801-806. doi:10.1038/nature03721. Read in full from a course copy at its.caltech.edu. PDF pages 1 to 6 are journal pp. 801 to 806. Method: "Spacing was expressed for each grid as the distance from the central peak to the vertices of the inner hexagon in the autocorrelogram (the median of the six distances)" (p. 802). The paper does not state how familiar the rats were with each enclosure.

| ID | Value | Quote | Location | Status | N | Arena | Method | DV position |
|---|---|---|---|---|---|---|---|---|
| H1 | 39 to 73 cm | "Although the average of this distance varied from 39 to 73 cm across different cells of different rats, the standard deviation (s.d.) within a single grid was only 3.2 cm (mean s.d. across 45 cells)." | p. 801 | full text | 45 cells, 6 rats | Circular, 2 m diameter: "a circular enclosure with a diameter of 2 m (45 neurons from 6 rats)" | Inner hexagon distances | Layer II of dorsocaudal MEC |
| H2 | 39.1 to 43.0 cm dorsal; 48.1 to 52.2 cm 560 μm deeper | "In the dorsal cells, the spacing ranged from 39.1 to 43.0 cm (14 cells recorded over 3 days). In the more ventrally located cells, the spacing ranged from 48.1 to 52.2 cm (5 cells recorded over 2 days)." | p. 802 | full text | 14 + 5 cells, 1 rat | Large circular enclosure | | "one near the postrhinal border and one 560 μm deeper" |
| H3 | 41.9 to 45.9 cm rising to 44.9 to 49.7 cm over 100 μm | "Over a distance of 100 μm, the spacing increased from between 41.9 and 45.9 cm (n = 3) to between 44.9 and 49.7 cm (n = 4)." | p. 802 | full text | 3 + 4 cells | Large circular enclosure | | Tangential track along the layer II/III border |
| H4 | Population spread over 30 cm | "While spacing varied by more than 30 cm in the entire cell population, the s.d. among cells recorded on the same tetrode was only 2.1 cm" | p. 802 | full text | 42 cells in 12 simultaneously recorded sets | | | |
| H5 | About 29 to 53 cm in a 1 m box | Fig. 2h: "Grid spacing as a function of distance from the postrhinal cortex for all animals tested in the square enclosure" | p. 803, Fig. 2h | figure-read, raster render: about 31 to 35 cm at 0 μm, low outliers about 29 cm, top about 52 to 53 cm at about 900 μm; printed R² = 0.67 | 57 cells, 11 rats | Square, 100 × 100 cm | | 0 to about 0.9 mm from the postrhinal border. Text: r = 0.82 for spacing against distance (p. 802) |
| H6 | DV fraction "upper 25%" | Not found in the main text | | search-snippet claim not verified | | | | |

## 4. Other rat studies

### Gardner et al. 2022

Gardner RJ, Hermansen E, Pachitariu M, Burak Y, Baas NA, Dunn BA, Moser MB, Moser EI (2022). Toroidal topology of population activity in grid cells. Nature 602:123-128. doi:10.1038/s41586-021-04268-7. PMC8810387. Main text and methods read from Europe PMC. Per-cell spacing taken from Source Data Extended Data Fig. 2 (MOESM10, sheets grid_params_ratQ, _ratR_day1, _ratR_day2, _ratS, column grid_spacing_cm). Arena: "a square open-field (OF) arena of size 1.5 × 1.5 m". Familiarity: "At the time of the surgery, each rat was highly familiar with the environment and task (10–20 training sessions lasting at least 20 min each". Neuropixels probes in MEC and parasubiculum. The text gives no DV position in mm. Extended Data Fig. 1 shows the probe sites on histology only. I did not read how spacing was computed for these values, which is in the Supplementary Methods.

| ID | Module | Mean spacing | s.d. | Median | Min to max | N cells with spacing | Status |
|---|---|---|---|---|---|---|---|
| G1 | R1, day 2 | 57.8 cm | 4.9 | 56.8 | 48.8 to 87.9 | 188 | source data |
| G2 | R1a / R1b, day 1 | 60.4 / 67.2 cm | 1.6 / 11.7 | 60.1 / 62.6 | 57.4 to 66.8 / 54.0 to 106.1 | 132 / 32 | source data |
| G3 | S1 | 59.7 cm | 2.3 | 59.5 | 50.5 to 70.0 | 139 | source data |
| G4 | Q1 | 71.5 cm | 4.2 | 71.3 | 61.8 to 92.7 | 97 | source data |
| G5 | R2, day 1 / day 2 | 89.1 / 83.9 cm | 5.1 / 6.3 | 89.4 / 84.4 | 73.8 to 105.7 / 57.1 to 98.5 | 168 / 172 | source data |
| G6 | Q2 | 102.4 cm | 6.7 | 102.0 | 88.2 to 140.5 | 65 | source data |
| G7 | R3, day 1 / day 2 | 125.7 / 107.6 cm | 12.0 / 19.6 | 127.0 / 113.2 | 69.8 to 149.3 / 32.6 to 141.0 | 147 / 177 | source data |

Ratios [derived from G means]: R2/R1 = 89.1/60.4 = 1.48 on day 1 and 83.9/57.8 = 1.45 on day 2. R3/R2 = 1.41 on day 1 and 1.28 on day 2, or 1.34 from medians. Q2/Q1 = 1.43. The session totals match the text: R day 2 has 188 + 172 + 177 cells with spacing plus a few without, 544 in all, and S has 140 grid cells. The text says "from 140 to 544 grid cells per session". Gardner's smallest modules, about 58 to 72 cm, are larger than Stensola's M1. The probes may not have reached the most dorsal MEC. I did not check this.

Side note on an existing note. decoding_dimensionality.md cites Redman et al. 2025 for "one recorded grid module (N = 168 putative grid cells)" with spacing 65 to 90 cm. Gardner's R2 on day 1 also has 168 cells in this source data, with spacing 73.8 to 105.7 cm and mean 89.1 cm. This suggests Redman reanalysed Gardner's R2. I did not check Redman.

### Barry et al. 2007, through Barry's PhD thesis

Barry C, Hayman R, Burgess N, Jeffery KJ (2007). Experience-dependent rescaling of entorhinal grids. Nat Neurosci 10:682-684. doi:10.1038/nn1905. Only the abstract was reachable, since the Nature PDF is paywalled. The same experiment is chapter 4 of Barry's University of London thesis, which I read from UCL Discovery (eprint 1444512, file U591818.pdf, OCR text). The thesis carries a copyright notice restricting quotation, so I quote it sparingly.

| ID | Value | Quote | Location | Status | N | Arena | Method | DV position |
|---|---|---|---|---|---|---|---|---|
| BT1 | About 13.5 to 43 cm, familiar enclosure | Fig. 4.14 caption: "Grid scale, measured from familiar enclosures, by rat." | Thesis p. 120 (PDF p. 123) | figure-read, raster render. Two clusters per rat, about 15 to 22 cm and 25 to 37 cm, one cell at about 43 cm | 6 rats, 38 to 40 grid cells, mostly layer III, some layers II, V and VI | Familiar 100 × 100 cm (3 rats) or 70 × 100 cm (3 rats) | "Grid scale, or grid spacing, was the median distance from the central peak to the six surrounding peaks." | Not given numerically |
| BT2 | Ratio about 1.7, third scale about 2.5 | "a tendency for the second-shortest group of grids from each rat to be approximately 1.7 times the length of the shortest. One rat had grids of three different scales and shows an additional peak close to 2.5 times the length of the shortest group." | Thesis p. 120 | full text, thesis | 5 rats with at least two scales | as BT1 | Groups of cells within 20% of each other | |

These spacings are about half of those from Stensola, Hafting and Gardner, which no other source here shows. Possible reasons are layer III and deep-layer sampling, small boxes that only resolve small grids, or a calibration difference. None of these is confirmed. I recommend not using BT1 for the columns.

## 5. Anatomy: MEC to hippocampus along the long axis

| ID | Statement | Source | Location | Status |
|---|---|---|---|---|
| A1 | "Interconnected portions of the LEC and MEC close to the rhinal fissure, in rats referred to as the dorsolateral band of entorhinal cortex, are connected to the dorsal (nonprimate) or posterior (primate) part of the hippocampal formation ... Interconnected cells in the intermediate band, encompassing again parts of both LEC and MEC, connect to the intermediate hippocampal formation, whereas the most medially interconnected band of entorhinal cortex is mainly connected to the ventral (nonprimate) or anterior (primate) hippocampal formation." | Canto CB, Wouterlood FG, Witter MP (2008). What does the anatomical organization of the entorhinal cortex tell us? Neural Plast 2008:381243. doi:10.1155/2008/381243. PMC2526269 | Section on intrinsic connections | full text |
| A2 | Fig. 1 legend: "A dorsolateral band of entorhinal cortex (magenta) is preferentially connected to the dorsal hippocampus. Increasingly, more ventral and medial bands of entorhinal cortex (purple to blue) are connected to increasingly more ventral levels of the hippocampus." | Canto et al. 2008 | Fig. 1c, d legend | full text |
| A3 | "Injections confined to the septal half of the dentate gyrus resulted in a band that occupied the most lateral and caudomedial portions of the entorhinal cortex." Mid-septotemporal injections, 50 to 75% from the septal pole, labelled "a more medial portion of the lateral entorhinal cortex and a more rostral portion of the medial entorhinal area". Temporal-quarter injections labelled "the most medial portion of the lateral entorhinal area and the most rostral portion of the medial entorhinal area." Also: "there was relatively little overlap of the three populations of neurons." | Dolorfo CL, Amaral DG (1998). Entorhinal cortex of the rat: topographic organization of the cells of origin of the perforant path projection to the dentate gyrus. J Comp Neurol 398:25-48. PMID 9703026 | Abstract | abstract. Layer II to dentate gyrus only |
| A4 | "Dorsal and intermediate regions are preferentially connected, via the dorsolateral and intermediate bands of the entorhinal cortex, to visual and somatosensory cortices" and "the ventral one-third of the hippocampus has strong connections with ... the lateral septum, hypothalamus, and amygdala, both directly and through the ventromedial portions of the entorhinal cortex". Also: the dorsal-to-intermediate place-field scale increase is "matching the progressive dorsoventral increase in the spacing of grid cells in medial entorhinal cortex, one synapse upstream". | Kjelstrup et al. 2008, Science 321:140-143 | p. 140 | full text, local PDF science.1157086.pdf |

Linking A1 to grid modules [derived, terminology only]. The grid papers record from "dorsocaudal medial entorhinal cortex (dMEC)" next to the postrhinal border (Hafting 2005, p. 801), and measure DV position from the dorsal border (Stensola). In the rat, this part of MEC lies along the rhinal fissure. Dolorfo & Amaral call the septal-projecting MEC part "caudomedial" and the temporal-projecting part "most rostral". So small modules in dorsal MEC fall in the dorsolateral band, which projects to dorsal hippocampus. Ventral and rostral MEC, with large spacings, projects to ventral hippocampus. No source here maps band borders onto mm from the dorsal border or onto module numbers.

## 6. Caveats: novelty, rescaling, arena size

| ID | Effect | Quote | Location | Status |
|---|---|---|---|---|
| C1 | Grid scale follows changes to a familiar box | "the spatial scales of the grids (which are aligned and have fixed relative sizes within each animal) vary parametrically with changes to a familiar environment's size and shape" | Barry et al. 2007, abstract | abstract |
| C2 | Novelty expands grids | "environmental novelty causes the spatial firing patterns of grid cells to expand in scale and reduce in regularity, reverting to their familiar scale as the environment becomes familiar" | Barry C, Ginzberg LL, O'Keefe J, Burgess N (2012). PNAS 109:17687-17692. doi:10.1073/pnas.1209918109. PMID 23045662, abstract | abstract. PNAS returned 403 and Europe PMC full text returned HTTP 500 three times |
| C3 | Modules rescale independently in a compressed box | "The degree of rescaling varied substantially (from −18 to 110%; mean ± s.e.m. = 35.1 ± 17.0%; 4 animals, 51 grid cells)." M1 "showed only minimal rescaling, with peak correlations appearing between −18.4% and −1.4% of the stretch distance", while larger modules "rescaled completely" | Stensola 2012, p. 77; familiar box 150 × 150 cm, novel 150 × 100 cm (p. 76) | full text |
| C4 | Arena caps measurable spacing | See S22 and S23. Boxes were 100 to 220 cm wide, so the largest modules are under-sampled or missed | Stensola SI | full text |
| C5 | Box size and spacing | Hafting's 1 m box values, about 29 to 53 cm (H5), run lower than his 2 m cylinder values, 39 to 73 cm (H1). The two samples also differ in rats and DV positions, so this does not isolate a box-size effect | Hafting 2005 | figure-read and full text |

## 7. Verdicts on the claims in the brief

- "Stensola spacing range 35 to 172 cm". Verified as 35.2 to 171.7 cm (p. 72) but only for the five tangentially implanted rats in the 220 cm box. It is not the range of the whole 15-rat sample. The main text gives no cell range for all 968 cells. Two single-module data sets have module means of 32.6 and 34.5 cm (Supp. Fig. 18b, figure-read), so the full sample goes below 35.2 cm.
- "Hafting dorsal range 39 to 73 cm". Verified (p. 801): 45 cells, 6 rats, 2 m cylinder. The added phrase "upper 25%" from a search summary is not in the main text. The 1 m box sample reaches about 29 cm (figure-read).
- "Stensola ratio 1.42 with s.d. 0.02". Verified, but the 0.02 describes the three mean ratios. Across individual animals the successive ratios run from 1.14 to 1.78 (s.d. about 0.175, figure-read and derived). Stensola say so too: "Because step sizes and step ratios were unique for each animal" (p. 77).
- "Brun about 50 cm to 3 m". Abstract only. The Solstad thesis abstract gives inter-peak distances up to "at least 8m". A secondary table gives a dorsal median of 171 cm, which conflicts with "approximately 50 cm".
- Existing note large_fields_followup.md, line 30, called the Solstad thesis English abstract unverified. It is now verified from the NVA API record, field alternativeAbstracts: "In hippocampus, field length ranged from less than 1m to more than 10m. In the MEC field length ranged from less than 50cm to approximately 3m, and inter-peak distance ranged from less than 1m to at least 8m." I did not edit that file.

## 8. Not verified

- Brun 2008 full text, including box spacings, the track spacing method, per-level means and the DV position of each cell. Only the abstract, the Solstad thesis abstract and a secondary table were reachable.
- Barry 2007 and Barry 2012 full texts. The Barry 2007 baseline spacings come from the thesis figure only.
- Which box size each Stensola multisite module was measured in. The box per data set is not given, except 150 cm for rat 14147.
- DV positions of modules in Stensola animals other than 14147 (Fig. 4e) and 14257 (Fig. 4a, depth from the brain surface rather than from the dorsal border).
- The Fig. 5b against Supp. Fig. 18b discrepancy in one data set (76.9 cm module).
- Hafting supplementary Fig. S4 and the Hafting supplement in general.
- Gardner's spacing computation and probe DV positions in mm.
- A source that maps entorhinal band borders onto mm from the dorsal MEC border.

## Files

Downloads are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/grid_rat/dl/`, one directory per source. Scripts are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/grid_rat/scripts/`: `fig5b_values.py` and `supp18b_values.py` (Stensola module means), `supp2c_values.py` (KSD peaks), `gardner_modules.py` (Gardner per-module statistics), `polylines.py`, `markers.py` and `render.py` (figure extraction). Renders are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/grid_rat/render/`.

Sources:
- Stensola 2012 main text: https://owncloud.ut.ee/owncloud/s/rEJMz3SagzLBPcm/download/The%20entorhinal%20grid%20map%20is%20discretized.pdf
- Stensola 2012 supplement: https://static-content.springer.com/esm/art%3A10.1038%2Fnature11649/MediaObjects/41586_2012_BFnature11649_MOESM246_ESM.pdf and ..._MOESM247_ESM.pdf
- Hafting 2005: https://www.its.caltech.edu/~jkenny/nb250c/papers/hafting_05.pdf
- Brun 2008 abstract: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:19021257%20AND%20SRC:MED&resultType=core&format=json
- Solstad 2009 thesis record: https://api.nva.unit.no/publication/0198cc6087c8-637e01b6-4364-4aaa-a65b-0d9195b70f67
- Kerdels & Peters 2018 survey: https://arxiv.org/pdf/1810.07429
- Gardner 2022: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8810387/fullTextXML and source data https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-04268-7/MediaObjects/41586_2021_4268_MOESM10_ESM.xlsx
- Barry thesis: https://discovery.ucl.ac.uk/id/eprint/1444512/1/U591818.pdf
- Canto, Wouterlood & Witter 2008: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2526269/fullTextXML
- Dolorfo & Amaral 1998 abstract: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:9703026%20AND%20SRC:MED&resultType=core&format=json
