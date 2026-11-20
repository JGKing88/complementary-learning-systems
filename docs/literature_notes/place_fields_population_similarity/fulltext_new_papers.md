# Brun 2008, Kjelstrup 2008 supplement, Jung 1994 and Terrazas 2005: full-text notes

Four PDFs supplied on 2026-11-20 in `papers/new/`. I read each one in full from a page-tagged text layer and checked figures against the PDF.

- Brun VH, Solstad T, Kjelstrup KB, Fyhn M, Witter MP, Moser EI, Moser MB (2008). Progressive increase in grid scale from dorsal to ventral medial entorhinal cortex. Hippocampus 18:1200-1212. doi:10.1002/hipo.20504. PDF page n is journal page 1199 + n.
- Kjelstrup KB, Solstad T, Brun VH, Hafting T, Leutgeb S, Witter MP, Moser EI, Moser MB (2008). Finite scales of spatial representation in the hippocampus. Science 321:140-143, Supporting Online Material (SOM). The SOM is paginated from PDF page 2 (SOM p. 1). Figures S1 to S8 are on PDF pages 17 to 24.
- Jung MW, Wiener SI, McNaughton BL (1994). Comparison of spatial firing characteristics of units in dorsal and ventral hippocampus of the rat. J Neurosci 14:7347-7356. A scanned PDF. PDF page n is journal page 7346 + n.
- Terrazas A, Krause M, Lipa P, Gothard KM, Barnes CA, McNaughton BL (2005). Self-motion and the hippocampal spatial metric. J Neurosci 25:8085-8096. PDF page n is journal page 8084 + n.

Tags. [full text] means the number is printed in the text, a table or a figure legend. [figure-read] means I read it from a figure, and the tag says how. [derived] means I computed it, and the arithmetic is shown. [inference] marks a judgement that goes beyond what the paper states.

Quotes are copied from the PDF with the text layer's glyph errors fixed. Brun's PDF prints "~" where the Europe PMC abstract has "approximately". I checked the "±" signs and the "~" signs against page renders.

## Summary of changes to the tables

| Table cell | Old | New | Status |
|---|---|---|---|
| Rat MEC column, 18 m track | "Ventral MEC, 18 m track: about 3 m" (abstract) | Per-level track spacing. Dorsal MEC (0-25% of the DV axis): minimum 0.91 ± 0.12 m, median 1.71 ± 0.13 m. Intermediate (25-50%): 2.02 ± 0.24 and 3.01 ± 0.23 m. Ventral (50-75%): 2.69 ± 0.47 and 3.70 ± 0.46 m. "About 3 m" lies between the ventral minimum and median | [full text] Brun p. 1207 |
| Survey values 171/301/370 and 91/202/269 cm | Called medians and minimums; seemed to conflict with the abstract's 50 cm | Copied correctly. Each is a group mean (± SEM) of a per-cell statistic, the median or the minimum distance between adjacent detected fields. The abstract's 50 cm is the smallest single-cell estimate near the dorsal border, a different quantity | [full text] Brun pp. 1207, 1211 |
| CA3 / MEC ratio, 18 m track | Each CA3 level over one ventral MEC value | Divide each CA3 level by the same-named MEC level. Dorsal 0.57 to 1.23, intermediate 1.02 to 1.85, ventral 1.11 to 2.05. With minimum spacing only, 1.08 to 1.23, 1.52 to 1.85 and 1.52 to 2.05. Pairing the levels by name is an assumption that neither paper tests | [derived] |
| Dorsal / intermediate / ventral CA3, 18 m | 0.98 to 1.12, 3.07 to 3.73, 4.10 to 5.52 m; strain, method, layout and familiarity unknown | Values confirmed. The largest-field means are 0.98, 3.73 and 5.52 m. The autocorrelation widths are 1.12, 3.07 and 4.10 m. Male Long-Evans rats, tetrodes, a straight 18 m track crossing three rooms. Familiarity is not stated. Track cells sit at 7 to 19%, 55 to 70% and 71 to 87% of the CA3 axis, not across the full 40 to 70% and 70 to 100% bands | [full text] SOM; [figure-read] Fig. S1, S6 vector paths |
| Ventral CA1 rat row | None verified | Still none. Jung 1994 "ventral" CA1 is at about 65% by the paper's own estimate, which falls in Kjelstrup's intermediate band (40 to 70%), and at 30 to 45% by Kjelstrup's note 29. Proposed as an intermediate CA1 row: 49 cm equal-area diameter against 24 cm dorsal, in a 68 × 68 cm box | [full text]; diameters [derived] |
| Kernel row, Terrazas 2005 | "About 3 times longer when the rat drives a car ... longer again when the room turns around a still rat; no distances in the text" | Overlap falls to 0.5 at about 24° (27 cm) walking and 86° (97 cm) driving, a ratio of 3.6. The text says "approximately three times". In WORLD the curve never reaches 0.5 (lowest about 0.54). The apparatus rotated, not the room, and WORLD came from different rats on a different apparatus | [full text]; [figure-read] Fig. 9 trace |
| New candidate rat rows | | Dorsal CA3, 1 m box: 619 ± 116 cm², equal-area diameter 28 cm (Kjelstrup SOM). Dorsal CA1, 68 cm box: 24 cm (Jung). Intermediate CA1, 68 cm box: 49 cm (Jung) | [full text]; diameters [derived] |

## 1. Brun et al. 2008

### Subjects, apparatus and analysis

| Item | Value | Location | Quote |
|---|---|---|---|
| Rats | 15 male Long-Evans, 300 to 350 g at surgery | p. 1201 | "Fifteen male Long-Evans rats were used for the experiment." "Body weight at the time of surgery was 300–350 g." |
| Implant | One microdrive per hemisphere, four tetrodes each, aimed at MEC | p. 1201 | "implanted with a microdrive in each hemisphere, each connected to four tetrodes" |
| Boxes | 100 × 100 cm and 220 × 220 cm, 50 cm walls, used to screen cells while lowering tetrodes | p. 1201 | "a small or medium-sized square recording enclosure (100 cm × 100 cm × 50 cm height or 220 cm × 220 cm × 50 cm height)". "The recordings in the boxes were used to identify cells with grid-like firing patterns as the electrodes were lowered through the entorhinal cortex (Fig. 2A)." |
| Track | 18 m × 12 cm, six 3 m modules, black linoleum, 50 cm high, through three rooms joined by open doors | p. 1201 | "an 18 m long and 12 cm wide linear track consisting of six identical interconnected modules of 3 m each". "The track was placed centrally and 50 cm above the floor in three test rooms connected by open doors (3 × 19 m). The doors were located 6 m and 8.5 m from the starting position at the east end of the track." |
| Position tracking | Infrared photocells every 5 cm along the track rims | pp. 1201, 1203 | "infrared photocells at intervals of 5 cm"; "The position signals were read from the photo-cell sensors on the track by the Axona recording system." |
| Training and familiarity | Track training began at least 2 days before the first recording. Recordings continued over weeks as tetrodes were lowered. About 10 laps each way per trial | p. 1201 | "Training on the track began at least two days before the first recording session." "Over the course of several weeks, the tetrodes were lowered in steps of 50 µm or less". "On recording trials, the rats ran approximately 10 consecutive laps in each direction." |
| Cues | Cue-rich rooms | p. 1207 | "The 18 m track was placed in a cue-rich environment" |
| Rate maps | 50 cm cut at each end, 17 m analysed, 5 cm bins, Gaussian smoothing factor 5 cm, directions separate | p. 1203 | "spikes that were sampled less than 50 cm from the ends of the 18-m track were excluded ... The remaining 17 m of the track was divided into bins of 5 cm." "smoothing factor of 5 cm" |
| Field definition | At least 3 bins (15 cm) above 10% of peak, peak at least 1 Hz, spikes on at least 1/3 of runs, at least 2 spikes per run | p. 1203 | "Firing fields were estimated as contiguous regions of at least three consecutive bins (15 cm) where the firing rate was above 10% of the peak rate and the maximum rate was at least 1 Hz." |
| Grid spacing on the track | Minimum and median distance between adjacent detected fields, per cell and direction | p. 1203; Fig. 4 legend p. 1206 | "The scale of the grid was estimated by calculating both the minimum and the median distance between adjacent firing fields (for median distances as in Hafting et al., 2005)." "(B) Grid spacing expressed as the minimum and median distances between adjacent peaks of the grid fields on the track. Median distance is probably overestimated because the algorithm for detecting firing fields may have missed a substantial fraction of the firing fields" |
| Autocorrelation width | Mean-subtracted (xcov), width at 10% of peak | p. 1203 | "The width of the spatial autocorrelation function at 10% of the peak was used as a separate measure of spatial scale" |
| Population vectors | Normalized rate maps stacked per level, 5 cm bins, 340 vectors per direction, Pearson correlation | pp. 1203, 1207 | "stacking the normalized linear firing rate maps of all cells at a given septotemporal level"; "giving a total of 340 population vectors per running direction" |

### Dorsoventral levels and N

The paper normalizes every site to the MEC dorsoventral axis, measured as 3.5 mm.

- "estimated to be 3.5 mm = 100% (range 3.40–3.56 mm)" (p. 1204). [full text]
- "the recordings were subdivided into a dorsal group (0–25% of the dorsoventral extent of medial entorhinal cortex), an intermediate group (25–50%), and a ventral group (50–75%), based on the estimated position of the electrodes on the recording day. The dorsal group consisted of 55 cells (9 rats), whereas the intermediate group had 59 cells (6 rats) and the ventral group contained 29 cells (4 rats). Four rats contributed cells to more than one group." (p. 1205). [full text]
- In millimetres the bands are 0 to 0.875, 0.875 to 1.75 and 1.75 to 2.625 mm from the postrhinal border (0.25 × 3.5 = 0.875 mm). [derived]
- "the two running directions were treated independently, so that a total of 286 grid patterns were analyzed." (p. 1205). The text does not say whether the ± values are over cells or over the 286 direction patterns. [full text]
- Group means sit at about 10%, 35% and 59% of the axis. [figure-read: column of each mean's error bar in Fig. 4B, native 600 × 963 raster, x axis calibrated from tick marks at 31.45 px per 10%]
- Scatter positions in Fig. 4C to F cluster at about 3 to 5, 10 to 12, 19 to 25, 27 to 35, 41 to 44, 48 to 59, 68 to 70 and 75%. So the ventral group mixes cells at about 50 to 59% with a few at 68 to 75%. [figure-read: same raster]
- Internal inconsistency. The text says the sites "were spread between the 1% and 75% positions along the dorsoventral axis, covering more than 3 mm of length" (p. 1204). On a 3.5 mm axis, 1 to 75% is 0.74 × 3.5 = 2.6 mm. [derived] The abstract repeats "more than 3 mm". An earlier note (grid_ratio_rat_grid.md, B2) inferred an MEC axis of about 4 mm from that phrase. The paper's own value is 3.5 mm.

### Results per level, 18 m track

All values are mean ± SEM. [full text], pp. 1205 and 1207 unless noted.

| Measure | Dorsal (0-25%) | Intermediate (25-50%) | Ventral (50-75%) | Statistics |
|---|---|---|---|---|
| Cells (rats) | 55 (9) | 59 (6) | 29 (4) | |
| Fields per running direction | 8.4 ± 0.3 | 5.9 ± 0.4 | 4.6 ± 0.4 | F = 35.8 |
| Mean width of all fields | 56 ± 1 cm | 92 ± 6 cm | 119 ± 7 cm | F = 39.8, all pairs P < 0.001 |
| Width of largest field | 90 ± 4 cm | 129 ± 8 cm | 190 ± 13 cm | Largest single value 525 cm |
| Largest field with Kjelstrup's settings (15 cm smoothing, 5 bins, 20% threshold) | not given | not given | 224 ± 13 cm | |
| Autocorrelation central peak width (10% of peak) | 65 ± 2 cm | 122 ± 9 cm | 200 ± 18 cm | F = 46.1 |
| Minimum spacing | 91 ± 12 cm | 202 ± 24 cm | 269 ± 47 cm | F = 12.9; intermediate vs ventral P = 0.20, n.s. |
| Median spacing | 171 ± 13 cm | 301 ± 23 cm | 370 ± 46 cm | F = 15.7; intermediate vs ventral P = 0.16, n.s. |
| PV correlation band, full width at r = 0.5 ("half-width") | 32 cm | 47 cm | 67 cm | No N or SEM |
| PV correlation band, full width at r = 0.20 | 54 cm | 102 cm | 184 cm | |
| Mean rate; peak rate | 3.6 ± 0.2 Hz; 21.3 ± 0.9 Hz | 4.5 ± 0.4; 17.4 ± 1.1 | 2.0 ± 0.2; 11.4 ± 0.9 | p. 1207 |

Quotes for the spacing rows: "The minimum spacing between the grid fields increased from 91 ± 12 cm in the dorsal group to 202 ± 24 cm in the intermediate group and 269 ± 47 cm in the ventral group (F = 12.9, P < 0.001, dorsal vs. intermediate post hoc test P ≤ 0.01, intermediate vs. ventral P = 0.20 n.s.; Figs. 4B,E). The median spacing increased from 171 ± 13 cm to 301 ± 23 cm and 370 ± 46 cm, respectively (F = 15.7, P < 0.001, dorsal vs. intermediate P < 0.001, intermediate vs. ventra P = 0.16 n.s.; Figs. 4B,F)." (p. 1207)

Fig. 4A and 4B plot the same values. I read 0.88, 1.98 and 2.65 m (minimum) and 1.68, 2.97 and 3.67 m (median), which match the text. [figure-read: enlarged raster, by eye]

What the PV "half-width" means. In Fig. 5 (p. 1208) the correlation is plotted against signed distance from -17 to +17 m. The black bars at r = 0.5 and r = 0.2 span both sides of zero. In panel C the r = 0.2 bar is 67 px long at 36.5 px per m, which is 1.84 m, and the r = 0.5 bar is 24 px, which is 0.66 m. Both match the stated 184 and 67 cm. So the stated widths are full widths. [figure-read: native raster cropped and enlarged 4 ×, axis 0 to 17 m] Correlation falls to 0.5 about 16, 23.5 and 33.5 cm from the diagonal and to 0.2 about 27, 51 and 92 cm away (each width / 2). [derived] Kjelstrup's main text uses the same wording, and an earlier note found the same full-width convention there.

Spread. Per-cell spacings in Fig. 4E and F run from about 0.5 m to about 15 m. Every level includes a few cells at 11 to 15 m, probably cells with only two detected fields. At 68 to 75%, minimum spacings are about 0.7 to 13 m and median spacings about 1.1 to 13 m. [figure-read: raster at 15.7 px per m, markers about 5 px, many overlapping, so counts per position are not recoverable] Left and right runs did not differ: "minimum grid spacing: left 162 ± 21 cm and right 173 ± 20; median grid spacing: 252 ± 21 cm and 266 ± 20" (p. 1207). [full text]

Typo in the paper. The theta section says "at all levels along the dorsoventral axis of CA3 (69 grid cells, 21 trials)" (p. 1207). The recordings are from MEC.

### Do the abstract's 50 cm and 3 m hold?

The abstract reads: "The average interpeak distance between the fields increased from ~50 cm in the most dorsal part to ~3 m at the most ventral recording positions." (p. 1200)

The two ends are different statistics.

- 50 cm is the smallest single-cell estimate near the dorsal border, not a group mean. "On the 18-m track, the smallest estimates for grid scale, obtained near the dorsal border of medial entorhinal cortex, were ~30 cm for field width and ~50 cm for grid spacing (Figs. 4C–F)." (p. 1211) [full text] The lowest dorsal points in Fig. 4E are near 0.5 m. [figure-read]
- 3 m matches the ventral group means, 2.69 m (minimum) and 3.70 m (median). The Discussion gives a range: "a spacing of 0.5–1.5 m in the dorsal region, and reaching ... a spacing of 2.5–5 m at the ventral end of the map" (p. 1209). [full text]
- The ventral group's mean position is about 59%, not 75%. [figure-read] "Most ventral recording positions" therefore describes the group's upper limit, not where its mean sits.
- The abstract's pair implies a 6-fold increase (300 / 50). The group means give 2.96-fold for minimum spacing (269 / 91) and 2.16-fold for median spacing (370 / 171). [derived]

### Do the survey's values appear in the paper?

Yes. Kerdels & Peters 2018 reproduce Brun's text values exactly: 171 ± 13, 301 ± 23 and 370 ± 46 cm (median) and 91 ± 12, 202 ± 24 and 269 ± 47 cm (minimum). Brun has no table, so the survey built its Table 1 from the Results text. Each value is the mean across cells (or direction patterns) of each cell's median, or minimum, distance between adjacent detected fields on the 18 m track. "Medians" and "minimums" in the survey refer to the per-cell statistic, not to a population median. The dorsal 91 and 171 cm do not conflict with the abstract's 50 cm, because 50 cm is the smallest single-cell value. [full text]

The authors expect both spacing measures to be too large. "the current values for field size and grid spacing on the 18-m track are likely to be somewhat overestimated as a result of nonperfect algorithms for detection of firing fields." "because more than half of the fields appeared to be lost on some trials, the minimum distance between the firing fields may be closer to the actual grid spacing than the median distance." (p. 1211) [full text]

### Box data and the relation to 2D spacing

- Box sizes are 100 × 100 cm and 220 × 220 cm. No box spacing or box field size is reported for any cell. [full text]
- At intermediate and ventral levels the boxes showed only one or a few fields. "At intermediate-to-ventral locations, only one or a few firing fields could be observed (Fig. 2). There were clear differences between active and silent regions in the box but the relatively small size of the recording enclosure prevented further assessment" (p. 1205). [full text]
- The only comparison with 2D spacing is qualitative and uses other papers' box values. "In the smaller environments, the corresponding values were ~20 cm (diameter) and ~30 cm (Hafting et al., 2005, 2008)." (p. 1211) The authors attribute the difference partly to missed fields and partly to the environment (Barry 2007 rescaling, Fyhn 2007). [full text] That puts the dorsal-border track spacing at about 50 / 30 = 1.7 times box spacing. [derived from the paper's approximate values]
- The paper does not treat track firing as a 1D slice through a 2D lattice and gives no conversion from track spacing to 2D spacing. [full text, by absence]

### Same track and same rats as Kjelstrup 2008?

- Same track. The apparatus text matches the Kjelstrup SOM almost word for word: 18 m × 12 cm, six 3 m modules, black linoleum, 50 cm high, three rooms joined by open doors (3 × 19 m), doors at 6 m and 8.5 m from the east end, chocolate at the turning points, about 10 laps each way, a 10 m cable on ceiling wires, an experimenter 1 m behind. Brun also states that it re-ran field detection "as used for CA3 place cells in a previous study (Kjelstrup et al., 2008)" (p. 1203). [full text]
- Differences in method. Brun tracked position with photocells every 5 cm; Kjelstrup used "an eye-safe invisible laser beam placed at one end of the track" (SOM p. 4). Brun cut 50 cm at each end, Kjelstrup 75 cm. Smoothing was 5 cm against 15 cm, threshold 10% against 20%, minimum field 15 cm against 25 cm. Brun normalized rate maps before the PV analysis; the Kjelstrup SOM does not mention normalization. [full text]
- Different rats. Brun's Fig. 1 lists 15 rats: 11760, 11587, 11750, 11661, 11647, 11553, 11265, 11570, 11569, 11659, 11468, 11769, 11645, 11646 and 11946. Kjelstrup's Fig. S1 lists 21 others, and no number appears in both lists. Brun's rats carried two MEC drives and Kjelstrup's one CA3 drive, and Brun's weighed 300 to 350 g at surgery against 350 to 400 g. [figure-read: rat labels in Brun Fig. 1 and the text layer of Kjelstrup Fig. S1]

### Can each CA3 level be divided by the matching MEC level?

What the paper supports [full text]:

- Same apparatus, same lab, three named levels and two spacing statistics per level, with N.
- The authors themselves compare MEC and hippocampal scale at "corresponding dorsoventral levels": "The most extended fields in the entorhinal cortex were generally smaller than at corresponding dorsoventral levels of the hippocampus even when difference in procedures for defining firing fields were taken into account." (p. 1211)
- One like-for-like field-width value. Ventral MEC largest field with Kjelstrup's settings is 224 ± 13 cm, against 5.52 m for ventral CA3, a ratio of 2.46. [derived]

What it does not support:

- No mapping from MEC position to CA3 position. Fig. 1 shows the topography only as a colour gradient ("red areas being connected preferentially with the dorsal hippocampus and blue with more ventral parts", p. 1202). The bands differ. MEC levels are 0 to 25, 25 to 50 and 50 to 75%, and the CA3 levels are 7 to 22, 40 to 70 and 70 to 100%. MEC above 75% was not recorded. Pairing by level name is an assumption. [inference]
- Different rats, not simultaneous recordings.
- Intermediate and ventral MEC spacings do not differ significantly (P = 0.20 and 0.16), so the ventral ratio rests mostly on the CA3 value.
- Track spacing is not 2D grid spacing, and the authors say both track measures are inflated by missed fields.

Ratios of CA3 width (Kjelstrup main text) to MEC spacing on the same track. [derived]

| Level | CA3 largest field / MEC minimum | CA3 autocorrelation / MEC minimum | CA3 largest field / MEC median | CA3 autocorrelation / MEC median |
|---|---|---|---|---|
| Dorsal | 0.98 / 0.91 = 1.08 | 1.12 / 0.91 = 1.23 | 0.98 / 1.71 = 0.57 | 1.12 / 1.71 = 0.65 |
| Intermediate | 3.73 / 2.02 = 1.85 | 3.07 / 2.02 = 1.52 | 3.73 / 3.01 = 1.24 | 3.07 / 3.01 = 1.02 |
| Ventral | 5.52 / 2.69 = 2.05 | 4.10 / 2.69 = 1.52 | 5.52 / 3.70 = 1.49 | 4.10 / 3.70 = 1.11 |

The same methods produce a population comparison, CA3 PV width over MEC PV width (Kjelstrup 0.81, 3.71 and 6.66 m against Brun 0.32, 0.47 and 0.67 m at r = 0.5). That gives 2.5, 7.9 and 9.9 at r = 0.5 and 2.6, 8.5 and 7.4 at r = 0.2. [derived] This is a field-to-field ratio, not field over spacing, and the PV inputs differ (normalized maps in Brun, normalization not stated in Kjelstrup).

## 2. Kjelstrup et al. 2008, Supporting Online Material

### Subjects, recording and track

| Item | Value | Location | Quote |
|---|---|---|---|
| Strain | 21 male Long-Evans rats, 350 to 400 g at surgery | SOM p. 1 | "Twenty-one male Long-Evans rats were used for the experiment." "Body weight at the time of surgery was 350-400 g." |
| Recording method | One microdrive with four tetrodes (17 µm Pt-Ir), vertical | SOM pp. 1-2 | "implanted with a microdrive, each connected to four tetrodes cut flat at the same level." "All implants were oriented vertically." |
| Targets | Dorsal CA3 in 7 rats; intermediate-to-ventral CA3 in 14 rats | SOM p. 1 | "Tetrodes aimed for the dorsal CA3 were inserted at AP 3.5 and ML 2.5-3.5 relative to bregma, 1.5 mm below the dura mater (7 animals). Tetrodes aimed for the intermediate-to-ventral CA3 were implanted at AP 4.1, ML 4.8, and DV 4.0 ... (14 animals)." |
| Why no CA1 | CA3 is easier to reach near the ventral pole | SOM p. 1 | "The tetrodes were targeted at CA3 because near the ventral pole of the hippocampus, the curvature of this subfield makes it more accessible than the almost vertically oriented CA1 region." |
| CA1 data | None. CA1 appears only in the 3D brain model of Fig. S1 | SOM; Fig. S1 | "Final recording positions in CA3 for all 21 animals." |
| Track | 18 m × 12 cm, six 3 m modules, black linoleum, 50 cm high, straight, through three rooms joined by open doors | SOM p. 2 | "an 18 m long and 12 cm wide linear track consisting of 6 identical interconnected modules of 3 m each." "The track was placed centrally and 50 cm above the floor in three test rooms connected by open doors (3 × 19 m). The doors were located 6 m and 8.5 m from the east end of the track." |
| Segments and turns | No turns along the track. The rat reverses at the two ends. Rooms span about 0 to 6, 6 to 8.5 and 8.5 to 18 m along the track | SOM p. 2 | [derived] from the door positions, assuming the track starts at the east wall |
| Reward and laps | Chocolate at each end; about 10 laps each way per trial | SOM p. 3 | "Running was maintained by chocolate crumbs delivered at each of the two turning points. On recording trials, the rats ran approximately 10 consecutive laps in each direction" |
| Position tracking | Laser range finder at one end, 5 to 20 Hz, interpolated to 20 Hz | SOM pp. 4-5 | "On the 18 m track, the rat's position was tracked by an eye-safe invisible laser beam placed at one end of the track. The sampling frequency of the laser was 5-20 Hz." |
| Training and familiarity | Not stated for the track. Training was post-surgical. Tetrodes were lowered over 10 to 30 days, and the most ventral sites were reached 30 to 90 days after turning began | SOM pp. 2-3 | "All training was performed post-surgically." "The most ventral recording locations were encountered 30-90 days after the start of tetrode turning." The track was probably familiar by the ventral recordings, but the SOM gives no session count [inference] |
| Cell inclusion | Pyramidal cells with mean rate above 0.10 Hz in at least one environment | SOM p. 4 | "Pyramidal cells were accepted for further analysis if the average rate was >0.10 Hz in at least one environment." |

### Analysis on the track

| Item | Value | Location | Quote |
|---|---|---|---|
| End cut and binning | 75 cm cut at each end; 16.5 m in 5 cm bins | SOM p. 5 | "spikes that were sampled less than 75 cm from the ends of the 18-m track were excluded from further analysis. The remaining 16.5 m of the track was divided into bins of 5 cm." |
| Smoothing | Gaussian kernel, smoothing factor 15 cm | SOM p. 5 | "Linear rate maps were calculated using spatial smoothing with a Gaussian kernel and smoothing factor of 15 cm." |
| Field detection | Above 20% of peak, peak at least 1 Hz, at least 5 bins (25 cm), spikes on at least 1/3 of runs, at least 2 spikes per run | SOM p. 5 | "A similar threshold was used for the linear track data, except that firing was required in at least 5 consecutive bins (25 cm) and that fields with spikes on less than one third of the runs or with less than an average of 2 spikes per run were discarded." The "similar threshold" is the 2D one: "above 20% of the peak rate and the maximum rate was at least 1 Hz" |
| Autocorrelation | Mean-subtracted (xcov), width at 20% of peak | SOM p. 6 | "The width of the spatial autocorrelation function at 20% of the peak was used as a separate measure of spatial scale" |
| Per-cell field measure | Widest field per cell | SOM p. 6 | "The width of each cell's widest place field on the track and the width of each cell's spatial autocorrelation function were correlated with the position of recording ... based on the data for recordings between 7% and 87% along this axis" |
| Population vectors | Stacked rate maps per level, 5 cm bins, 0.75 m cut at each end, directions separate, Pearson correlation, widths at half-height and at r = 0.20 | SOM pp. 6-7 | "stacking the linear firing rate maps of all cells at a given septotemporal level". No normalization is mentioned, unlike Brun |
| Caveat on single-cell measures | Both underestimate scale | Fig. S6 legend | "these measures are likely to underestimate the spatial scale due to (i) occasional cleavage of individual place fields ... and (ii) truncation of firing fields at the ends of the track. Low values for field width in the above distributions are likely to reflect such instances." |
| Fields per direction | 1.5, 1.7 and 1.6, overestimates because of cleaving | SOM p. 11 | "the mean number of place fields per running direction was 1.5 ± 0.1 (dorsal), 1.7 ± 0.1 (intermediate), and 1.6 ± 0.1 (ventral) (F(2,327) = 1.36, n.s.). These numbers are overestimates" |
| Field size compared with older work | Larger than in most dorsal studies | SOM p. 12 | "The difference may reflect expansion of fields in large environments (S12, S13) or drift of firing locations across laps when animals traverse long segments with few proximal cues (S14)." |

### Long-axis mapping (Fig. S1)

Method. "the border between hippocampal fields CA3 and CA2 was reconstructed. The length of the line represents an estimate of position along the total longitudinal length of field CA3 (0-100%, 0% represents the dorsal tip and 100% represents the ventral tip). All recording locations in CA3 were projected onto that line" (SOM p. 10). Histology used coronal sections and an assumed 20% shrinkage. [full text]

Per-rat levels, read from the coloured bars and squares in the right panel of Fig. S1. The panel is vector graphics. I matched each mark to its rat by colour and label position and converted y to percent with the drawn ticks at 10 to 90% and the axis ends at 0 and 100%. [figure-read: vector paths]

| Group | Rat and level |
|---|---|
| Dorsal (7 rats) | 11278 6.5%, 11275 9.2%, 11342 9.9%, 10440 13.5%, 11183 14.7%, 11086 18.2%, 10584 21.5% |
| Intermediate to ventral (14 rats) | 10414 41.8 to 73.3%, 10479 50.1%, 10662 52.3%, 11364 55.0 to 59.9%, 10918 59.2 to 88.5%, 11480 60.2 to 85.8%, 10416 60.7 to 63.8%, 11285 66.4 to 84.8%, 10882 74.3%, 11581 81.2 to 83.6%, 11365 83.2 to 86.4%, 10417 83.8 to 89.1%, 10961 90.0%, 10779 94.5% |

The dorsal rats span 6.5 to 21.5%, which matches the main text's "7 to 22%". The main text's 40 to 70% band starts at the top of rat 10414's bar.

### Per-level values from Fig. S6 (vector extraction)

Fig. S6 is vector graphics. I took every diamond marker (211 in panel A, 328 in panel B) and converted it with the drawn tick marks. Refitting a line to the extracted points gives 6.415x + 10.2 (panel A) and 4.265x + 56.7 (panel B), against the printed 6.4159x + 11.046 and 4.2661x + 56.323. So the extraction is complete. The printed equations give width in cm for position in %, although the axes are in metres (at 87% the panel A line reads 5.69 m, and 6.4159 × 87 + 11.046 = 569). [figure-read: vector paths; derived]

| Level (main-text band) | Panel A, largest field: n, mean ± SEM | Main text | Panel B, autocorrelation: n, mean ± SEM | Main text |
|---|---|---|---|---|
| Dorsal (0-25%) | 111, 0.97 ± 0.03 m; median 0.87; max 2.40 | 0.98 ± 0.03 m, 111 cells | 155, 1.12 ± 0.03 m; median 1.06 | 1.12 ± 0.03 m |
| Intermediate (40-70%) | 38, 3.68 ± 0.42 m; median 2.93; max 15.6 | 3.73 ± 0.43 m, 37 cells | 63, 3.08 ± 0.19 m; median 2.87 | 3.07 ± 0.19 m |
| Ventral (70-100%) | 62, 5.48 ± 0.47 m; median 4.48; max 15.9; 7 cells above 10 m | 5.52 ± 0.54 m, 61 cells | 110, 4.10 ± 0.18 m; median 4.44 | 4.10 ± 0.18 m |

What this establishes.

- The three table rows are right. Within each range, the largest-field mean is 0.98, 3.73 and 5.52 m, and the autocorrelation width is 1.12, 3.07 and 4.10 m. In the dorsal row the field mean is the low end; in the other two rows it is the high end. [full text, main text; confirmed by figure-read]
- The level split that reproduces the main-text autocorrelation means puts the points at 70.9% into the ventral group, which supports 70% as the boundary. [figure-read]
- Track cells sit at 6.9 to 18.5%, 54.9 to 69.5% and 70.9 to 86.9%. No track cell lies between 18.5% and 54.9%, and none beyond 87%, although Fig. S1 shows rats at 90.0 and 94.5%. [figure-read] Suggested labels are "Intermediate CA3 (track cells at 55 to 70%)" and "Ventral CA3 (track cells at 71 to 87%)".
- The main text's regression "r = 0.67, n = 111" (an earlier note quotes it) cannot be panel A, which has 211 points and prints R² = 0.45 (0.67² = 0.45). The "n = 111" is probably a misprint for 211. [inference]
- The main text's F(2,517) implies 520 observations. The SOM lists "63 out of 236 dorsal fields, 15 out of 109 intermediate fields, 13 out of 175 ventral fields" (SOM p. 7), and 236 + 109 + 175 = 520. So the ANOVA was probably run over all detected fields, while the means are per-cell largest fields. [derived; inference]
- Panel B's 328 points are close to the 330 cells implied by F(2,327). Panel A's 211 are close to the main text's 209 cells (111 + 37 + 61). Panel A probably holds only cells with a detected field. [inference] The SOM does not say whether a point is a cell or a cell-direction pair.
- The analysed track is 16.5 m long, and panel A reaches 15.6 to 15.9 m. Some ventral fields fill nearly the whole track, so those values are capped. [figure-read]

### Two-dimensional data in the SOM

- 1 m box. "Initially, neural activity was recorded while the rats ran in a small square enclosure (1 m × 1 m; 25 dorsal, 63 intermediate, and 118 ventral pyramidal cells; Fig. S3). As expected (S9, S10), dorsal cells had sharply confined place fields (mean ± S.E.M.: 619 ± 116 cm², or ~10% of the box)" (SOM p. 10). "In contrast, pyramidal cells in intermediate and ventral CA3 showed less spatial modulation and place fields could generally not be defined." [full text]
- 619 cm² is 6.2% of a 1 m² box (619 / 10,000), not about 10%. [derived] Equal-area diameter is 2 × √(619 / π) = 28.1 cm, or 25.3 to 30.6 cm at ± 1 SEM. [derived]
- The 2D field threshold is 20% of peak, at least 200 cm², peak at least 1 Hz (SOM p. 5). [full text]
- 198 cm circle. "7 dorsal cells, 27 intermediate cells, 29 ventral cells" (SOM p. 11). Only spatial information is reported, no field sizes. [full text]
- Rectangular box 100 × 70 cm was also used (SOM p. 2). No values are reported for it. [full text]

### Other SOM results that bear on the rows

- Fields in both directions: "dorsal: 41.4%; intermediate: 70.3%; ventral: 82.0%". Left-right map correlation r = 0.142, 0.394 and 0.505 (SOM p. 13). [full text]
- Spatial information on the track: 3.78, 1.97 and 1.77 bits per spike. Mean rate 0.33, 1.45 and 1.70 Hz. Peak rate 9.3, 10.9 and 7.3 Hz (SOM p. 12). [full text]
- Remapping between rooms in boxes: rate overlap 0.12 (dorsal), 0.22 (intermediate) and 0.27 (ventral), near the shuffled 0.18 to 0.21 (SOM p. 14). [full text]

## 3. Jung, Wiener & McNaughton 1994

### Methods

| Item | Value | Location | Quote |
|---|---|---|---|
| Rats | 7 male Fischer-344, 9 months old | p. 7348 | "Seven male Fischer-344 rats aged 9 months" |
| Environment | Square chamber 68 × 68 cm, 51 cm walls, dark grey, one white card | p. 7348 | "The rats were trained in a square chamber (68 × 68 cm, 51 cm height) placed on an elevated table covered with fresh paper. The chamber walls were dark gray in color and one was covered with a white card" |
| Task and familiarity | Random foraging for pellets, trained over several days; 30 min sessions | p. 7348 | "The rats learned to retrieve the food and visit all parts of the floor typically within several days of training." "Following baseline recording, the rat was allowed to run in the arena for 30 min." |
| Implants | Dorsal drive at 3.3 to 3.8 mm posterior, 1.7 to 2.0 mm lateral. Ventral drive in the opposite hemisphere at 5.8 to 6.3 mm posterior, 5.5 to 5.8 mm lateral | p. 7348 | "one aimed at the dorsal hippocampus (3.3–3.8 mm posterior and 1.7–2.0 lateral to bregma) and the other at the ventral hippocampus (5.8–6.3 mm posterior and 5.5–5.8 lateral to bregma)" |
| Subfield | CA1 cell layer at both sites | p. 7349 | "All the recording sites were verified to be in the CA1 cell body layer for both dorsal and ventral hippocampal penetrations (Fig. 2)." |
| Binning | 168 × 168 cm camera field in 64 × 64 bins, so 2.625 cm bins of 6.89 cm² | p. 7348 | "The 168 × 168 cm square containing the apparatus was divided into 64 × 64 bins" (168 / 64 = 2.625 cm) [derived] |
| Smoothing | Adaptive binning. The sampling radius grows until d ≥ α / (n √s), with α = 1000 | p. 7348 | "Sampling radius of a bin was increased starting from 0 until the following condition was met" |
| Field definition | At least 4 adjoining bins, each above a fixed 1 Hz | p. 7348 | "For this study the minimal number of adjacent bins was set at 4, and the threshold was set at a constant rate of 1 Hz." "The size of a place field was defined as the number of bins in a place field" |
| N | Dorsal: 36 of 80 complex-spike cells had fields, 8 with two fields. Ventral: 22 of 129, 3 with two fields. Both fields of dual-field cells were counted | pp. 7350-7351 | "Only 18% (22 of 129) of ventral hippocampal CS cells had place fields, whereas 45% (36 of 80) of the dorsal CS cells had place fields". "8 of 36 dorsal and 3 of 22 ventral units had two place fields". "both place fields were entered into the analysis for cells with dual place fields" |

So the field counts are 36 + 8 = 44 dorsal and 22 + 3 = 25 ventral. [derived]

### Field sizes

"place field size was much smaller for dorsal CS cells compared to ventral CS cells (67.0 ± 7.5 bins for dorsal and 272.0 ± 44.3 bins for ventral, Z = 3.2, p = 0.001" (p. 7351). Table 1 (p. 7350) prints the same values as mean ± SEM. [full text]

| | Dorsal CA1 | "Ventral" CA1 |
|---|---|---|
| Bins, mean ± SEM | 67.0 ± 7.5 | 272.0 ± 44.3 |
| Area | 67.0 × 6.890625 = 461.7 cm² (SEM 51.7) | 272.0 × 6.890625 = 1874.2 cm² (SEM 305.3) |
| Equal-area diameter, 2√(A/π) | 2 × √(461.7 / 3.1416) = 24.2 cm (22.8 to 25.6 at ± 1 SEM) | 2 × √(1874.2 / 3.1416) = 48.9 cm (44.7 to 52.7) |
| Share of the 4624 cm² box | 10.0% | 40.5% |

All rows in this table are [derived]. The diameter of the mean area is not the mean of the diameters. The area ratio is 4.06 and the diameter ratio 2.0.

The box limits the ventral value. "some ventral place fields were as large as 80% of the entire recording apparatus (i.e., sparsity is about 0.8) whereas others were as small as the typical dorsal place field" (p. 7351). [full text] 80% of the box is about 3700 cm², or 537 bins. [derived] Because the threshold is a fixed 1 Hz rather than a fraction of peak, field size also depends on rate, and ventral cells fired more overall (0.35 against 0.29 Hz). [full text; inference] Table 1 prints the ventral running rate as 0.35 ± 0.09 Hz; the text prints 0.35 ± 0.05.

### Long-axis position of the "ventral" sites

- The paper's own estimate: "The recording sites for the dorsal and ventral hippocampus corresponded to about 20% and 65%, respectively, of the distance along the septotemporal axis." (p. 7349) [full text] The text does not say how the percentages were measured.
- Fig. 2 (p. 7350) draws the sites on sagittal outlines. The ventral sites (panels B to D, three mediolateral levels, 7 circles) all sit on the upper edge of the CA1 ring in lateral sagittal sections. None is near the lower part of the ring or the temporal pole. The figure has no long-axis scale. [figure-read: page render at 4 ×]
- Kjelstrup 2008 note 29 (main text p. 143, quoted in fulltext_large_fields.md) says "Previous studies have demonstrated place fields in intermediate parts of the hippocampus, at approximately 30 to 45% of the dorsal-to-ventral axis (13, 14)", citing Jung 1994 and Maurer 2005. I cannot settle 65% against 30 to 45% from Jung's schematic. [inference]
- Either way the sites are not ventral by Kjelstrup's bands. 65% falls in the intermediate band (40 to 70%), and 30 to 45% falls between the dorsal and intermediate bands. [derived]

### Proposed rat rows

| Row | Region and level | Environment | Field definition | Value | N | Status |
|---|---|---|---|---|---|---|
| Jung 1994, dorsal | Dorsal CA1, about 20% of the septotemporal axis | 68 × 68 cm square box, familiar (several days of training), Fischer-344, 9 months | At least 4 adjoining 2.625 cm bins above 1 Hz, adaptive binning | 462 ± 52 cm²; equal-area diameter 24 cm | 36 cells, 44 fields, 6 hemispheres | [full text]; cm [derived] |
| Jung 1994, intermediate | CA1, "ventral" in the paper; about 65% by the paper, 30 to 45% by Kjelstrup | Same box, same rats | Same | 1874 ± 305 cm²; equal-area diameter 49 cm; box-limited | 22 cells, 25 fields, 6 hemispheres | [full text]; cm [derived] |

Neither row counts as ventral CA1. Label the second "intermediate CA1 (Jung's 'ventral' sites)". There is still no verified rat ventral CA1 field width.

## 4. Terrazas et al. 2005

### Check of the kernel-table row

| Part of the row | Verdict | Evidence |
|---|---|---|
| "Ring 130 cm across" | Correct | "a 130-cm-diameter G-scale circular track" (RTA, p. 8086); the RCV track was "also 130 cm diameter" (p. 8086) [full text] |
| "about 4.1 m around" | Correct | π × 130 = 408.4 cm. The paper's "1° = 1.13 cm" gives 360 × 1.13 = 406.8 cm, and 128 bins × 3.2 cm = 409.6 cm [derived] |
| "Overlap without mean subtraction" | Correct | "The overlap matrix for population vectors was computed (normalized dot products)." Rate maps formed "an N cells × 128 bin matrix (3.2 cm/bin)" (p. 8087) [full text] |
| "distance where it falls to 0.5" | Correct as the paper's proposed measure, with the authors' caveat | "In principal, the point at which the average overlap falls to 0.5 provides a convenient, unbiased definition of the scale ... however, this function can be distorted by sampling noise and by the periodic boundary conditions imposed by the track configuration" (p. 8088) [full text] |
| "About 3 times longer when the rat drives a car than when it walks" | Correct in the text; the figure gives 3.6 | "the scale at which the CA1 population firing rate vector changes with position was approximately three times larger during CAR than during WALK" (p. 8091) [full text]. Figure-read ratio 85.5 / 23.9 = 3.6 [derived] |
| "longer again when the room turns around a still rat" | Wording wrong; the claim needs qualifying | The rotating part was the apparatus, not the room: "the entire 'world,' including floor, track, and curtains, could be rotated while the rat was on the stationary vehicle" (p. 8086). It was a 140 cm platform with black curtains and a fabric ceiling. WORLD data came from different rats on that curtained apparatus (RCV). WALK and CAR came from the open toy-train apparatus (RTA) in the same sessions: "For the CAR and WALK conditions, within-session data acquired on the RTA were used" (p. 8087); "Data from active cells (n = 217) in the WORLD condition were obtained from different subjects" (p. 8091). In WORLD the overlap never falls to 0.5, so there is no 0.5 distance to call "longer" [full text; figure-read] |
| "no distances in the text" | Correct | No crossing distance appears in the text or legends [full text, by absence] |
| "full text, figure only" | Update | Now full text plus figure-read |

### Distances where the overlap falls to 0.5 (Fig. 9, third column)

I traced the thick curve in each panel of the native 688 × 643 raster. The y axis line runs from overlap 0 at the x axis to 1.0 at its top; the "0.5" label sits at the midpoint (row 164 between 89 and 239), which confirms the scale. The x axis line runs from 0° to 180°, and the "90" label sits at the midpoint. At each column I took the middle of the largest dark run. Resolution is about 0.007 overlap per pixel and 1° per pixel. [figure-read: raster trace, script below]

| Condition | Cells | Overlap at 10°, 30°, 90°, 180° | Falls to 0.5 at | Lowest value |
|---|---|---|---|---|
| WALK (RTA) | 152 active cells | 0.76, 0.43, 0.26, 0.29 | 23.9° = 27 cm | 0.25 near 76° |
| CAR (RTA, same sessions) | 96 active cells | 0.92, 0.80, 0.48, 0.30 | 85.5° = 97 cm | 0.30 near 175° |
| WORLD (RCV, other rats) | 217 active cells | 0.67, 0.62, 0.57, 0.54 | Never; the curve drops to about 0.7 within the first few degrees and then stays above 0.5 | 0.54 at 170 to 180° |

Distances use 1° = 1.134 cm (408.4 / 360). [derived] I estimate the crossing angles to be within about ±3° for WALK and ±5° for CAR, from ±0.02 in reading overlap. The half-ring is 180° = 204 cm, so the CAR crossing is near half of the largest possible separation. [derived]

The floor is well above zero (0.25 to 0.30 walking or driving, about 0.55 in WORLD), as expected for a dot product of non-negative rates. The 0.5 crossing therefore sits closer to the peak than it would for a mean-subtracted correlation. [inference] The authors explain the WORLD shape: "First, many of the active cells fired over nearly the whole track. Second, there were many fewer laps for the WORLD condition than for the CAR condition ... Third, the overall firing rates were substantially lower" (Fig. 9 legend, p. 8093). [full text]

Suggested replacement row. "Ring 130 cm across, 408 cm around | Overlap (normalized dot product, no mean subtraction); distance where it falls to 0.5 | About 27 cm walking and 97 cm driving (3.6 times; text says about 3 times). With the curtained apparatus rotating around a stationary rat (WORLD, different rats and apparatus) it stays above 0.5 to 180° (204 cm) | full text; distances figure-read"

### Other details

- Rats: "Six male Fischer 344 rats (age range 10–18 months)" (p. 8086). These are middle-aged rats. [full text]
- Recording: dorsal CA1, 12-tetrode hyperdrive (pp. 8085-8086). [full text]
- Population vectors pool all recorded cells per condition: "Composite population vectors were constructed from the entire sample of recorded cells for each condition and position" (Fig. 9 legend). The number of rats behind each pool is not stated. [full text]
- Experience. Rats were shaped to drive over many sessions. On the RTA, "The subject was placed on the track for the WALK condition only after completing 60–100 laps of the CAR condition. This typically resulted in 10–20 laps of WALK" (p. 8086). CAR fields were "remarkably stable over many laps (>100 in many cases)" (p. 8090). [full text]
- Speeds: "WALK, 10.72 ± 0.22; CAR, 12.56 ± 0.24; WORLD, 8.43 ± 0.18" cm/s (p. 8088). [full text]
- Fig. 10's legend says "circle diameter is reduced progressively (0.45, 0.15, and 0.05" but the Methods, Results and Table 1 call these the radius ("Rad 0.45 0.15 0.05"). The red circle in Fig. 10A spans about 0.05 to 0.95 of the unit square, so radius is right and the legend's "diameter" is a misprint. [full text; figure-read]

### Place-field size numbers usable as a rat row

None. The text gives no field width or area in cm for any condition. [full text, by absence]

The simulation fitted to WALK sets the Gaussian field SD to 0.10 with a track radius of 0.45 in a unit square: "The place field size was defined as the SD of the Gaussian and was set to 0.1 to match the empirically derived value obtained for the WALK condition" (p. 8088). Scaling the 0.45 radius to 65 cm gives SD = 0.10 / 0.45 × 65 = 14.4 cm, and a full width at half maximum of 2.355 × 14.4 = 34 cm. [derived] This is a model parameter matched to the population curve, not a measured field size. A cross-check is that Gaussian fields with this SD put the 0.5 crossing at 2√(ln 2) × 14.4 = 24 cm with no floor, close to the 27 cm read from Fig. 9. [derived] I do not recommend it as a table row. At most it can be a footnote: "dorsal CA1, 130 cm ring, walking, familiar; model-matched Gaussian SD about 14 cm (Terrazas 2005, derived)".

Fig. 6 shows only the five most informative cells per condition, so field widths read from it would not represent the population. I did not read them.

## Scripts and renders

Everything is in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/newpapers/`.

- `kj_s6_values.py` extracts the Fig. S6 markers and computes the per-level statistics. Its output is `kj_s6_points.json`. `kj_s6_extract.py` and `dump_draw.py` were used to find the axis ticks.
- `kj_s1_extract.py` and `kj_s1_levels.py` read the Fig. S1 bars and convert them to percent.
- `terr_fig9_trace2.py` traces the three Terrazas Fig. 9 overlap curves and finds the 0.5 crossings.
- `brun_fig4_axes.py`, `brun_fig4_ticks.py` and `brun_fig4_means_x.py` calibrate Brun Fig. 4 and locate the group means. `brun_fig4_points.py` is a rough marker count for Fig. 4E and F. Its group means are biased upward by merged markers, so I used it only for ranges.
- `derived.py` holds all the arithmetic in this note.
- `render.py`, `extract_img.py`, `find_text.py` and `info.py` are helpers. PNG renders of every figure I read are in the same folder.
