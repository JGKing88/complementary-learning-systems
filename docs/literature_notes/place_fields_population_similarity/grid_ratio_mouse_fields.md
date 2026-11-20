# Mouse place-field widths by region and dorsoventral level (rows for the grid-ratio table)

Compiled 2026-11-20. These are the rows of the mouse table whose cells are field width divided by grid spacing. Every number below is mouse data. No model results are included.

## Tags and conventions

- FT. I read the number in the full text and quote it, with its location.
- figure-read. I measured the number from a figure. The method is under "How figure values were read".
- derived. I computed the number from reported values. The arithmetic is under "Derivations".
- abstract. I saw only the abstract.
- Equivalent diameter. For an open-field area A, the width is d = 2 * sqrt(A / pi). It is always tagged derived, and the original area is kept next to it.
- "Summed" means the paper adds the areas of all fields of a cell into one field size. "Per field" divides that sum by the mean number of fields per cell. That ratio of means is approximate.
- Calcium-imaging widths depend on indicator, smoothing and threshold. They sit in their own tables and their own proposed rows.
- In quotes, minus signs and range dashes are written as plain hyphens. Everything else in a quote is verbatim, with "..." marking cuts.

## Corrections to the starting points

1. Ziv et al. 2013. Both numbers are correct, and the imaging was in dorsal CA1. The virus went into CA1 at -1.9 mm AP, 1.4 mm ML, -1.65 mm DV from bregma. Three caveats were missing. First, the mean is given only as "∼27%" of 84 cm, which is about 22.7 cm (derived), while the median is 24 cm. Second, width is the extent at 50% of the peak after Gaussian smoothing with sigma 8.75 cm. That kernel alone has a full width at half maximum of 20.6 cm (derived), so the 24 cm median sits close to the smoothing floor. Third, the analysis dropped the last 7 cm at each end of the track.
2. Dombeck et al. 2010. 50 ± 19 cm is correct. It is dorsal CA1, GCaMP3, two-photon, on a 180 cm virtual track. The paper itself says this is about 20% wider than the electrophysiology value from the same rig, 41 ± 14 cm (Harvey et al. 2009, verified below). Fields touching a track end were excluded from the mean.
3. Keinath et al. 2014. The quote is correct (Discussion, p. 1545). Absolute sizes exist only in Figure 1J and Figure 3H, so I read them from the figures. Dorsal is about 0.061 m² and ventral about 0.086 m² in the 35 cm cylinder. In the 70 cm cylinder dorsal is about 0.086 m² and ventral about 0.138 m². One correction matters for the table. Only ventral fields grew significantly in the 70 cm cylinder (P < 0.001). Dorsal fields "did not significantly expand". Field size is summed over all fields of a cell, and cells had about 2 fields in the 70 cm cylinder against about 1 in the 35 cm cylinder. The paper calls the environments "novel".
4. Ciocchi et al. 2015 (Science) recorded rats, not mice. The Europe PMC abstract says "By recording from ventral hippocampal CA1 neurons in rats", and the MeSH terms list "Rats, Inbred LEC". It is out of the mouse table. [abstract]
5. Jimenez et al. 2018 imaged mouse ventral CA1, but the main text reports only spatial information and stability for place cells (Fig. S4). It gives no field sizes. [FT]
6. "Kim & Frank". I found no mouse study by these authors that compares field size along the long axis.
7. Contradicting number for the ventral row. In head-fixed mice on a 1.9 m virtual track, Chockanathan & Padmanabhan 2021 found no difference between dorsal and ventral CA1 field size, 9.4 ± 2.0 cm vs 11.9 ± 6.3 cm (P = 0.84). Their width criterion is unusual and gives narrow fields (rate at least 1.5 SD above the mean).
8. Contradicting pattern for CA3. Three mouse studies found CA3 fields no smaller than CA1 fields. Hussaini et al. 2011 found control CA3 slightly larger than CA1 in both a box and a track. Dong et al. 2021 found CA3 wider than CA1 in a familiar virtual track (P = 7.1 × 10^-4). Hainmueller & Bartos 2018 found no difference. Rat studies usually report smaller CA3 fields, so a rat CA3/CA1 ratio should not be carried over to mice.
9. Jarzebowski et al. 2022 compared dorsal with intermediate CA1, not ventral CA1. The bioRxiv preprint title said "dorsal and ventral". The accepted manuscript says "dorsal and intermediate", and the iCA1 lens sat at -3.16 mm AP, 3.6 to 3.8 mm ML, 3.40 to 3.70 mm DV.
10. Mallory et al. 2018 reports CA1 field areas in open arenas. The arena size is in the Online Methods, which I could not read.

## Values, one row per number

### Table 1. Dorsal CA1, electrophysiology

| ID | Study | Strain | Region and level, how defined | Method | Environment | Field definition | Value as reported, statistic | N | Width for the table | Tag |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Harvey et al. 2009 | Male C57BL/6J, 8-12 weeks | Dorsal CA1. Craniotomy "centered over dorsal hippocampus (2.2 mm caudal, 1.7 lateral" | Acute extracellular, one unit per recording, head-fixed | Virtual linear track, 180 cm long, effective width about 1 cm. Familiar, trained at least 5 days | Adjacent bins above 0.25 × peak. At least 8 bins (18 cm). In-field rate above 1.5 Hz and above 3 × out-of-field rate | 41 ± 14 cm, mean. The ± is not defined for this value; nearby values in the paper are mean ± sd | 23 place cells, 8 mice | 41 cm (length) | FT |
| D2 | Harvey et al. 2009 | As D1 | As D1 | Whole-cell patch, same rig | As D1 | As D1 | 43 ± 13 cm, mean | 8 place cells, 8 mice | 43 cm (length) | FT |
| D3 | Hussaini et al. 2011 | Control littermates (HCN1 f/f, no cre), hybrid 50:50 C57BL/6J:129SVEV, 3-6 months | Dorsal, proximal CA1. Tetrodes at 1.8 mm posterior, 1.8 mm lateral to bregma. Recording sites 1.8 ± 0.06 mm lateral | Tetrodes, freely moving | Square box 50 × 50 cm, real. Familiarity not stated | Pixels at or above 20% of peak ("top 80 % of the firing peak"), at least 8 contiguous pixels. Size = field pixels / visited pixels | 20.8 ± 1.55% of area, mean ± SEM | Control CA1: 77 place cells from 11 mice across box and track. Box and track split not given | 25.7 cm equivalent diameter (A = 520 cm²) | FT, derived |
| D4 | Hussaini et al. 2011 | As D3 | As D3 | As D3 | Track enclosure 100 × 15 cm with 30 cm walls, real. Familiarity not stated | As D3 | 24.8 ± 1.8% of area, mean ± SEM | As D3 | 24.8 cm as a fraction of the 100 cm length (assumes the field spans the 15 cm width). 21.8 cm equivalent diameter (A = 372 cm²) | FT, derived |
| D5 | Mallory et al. 2018 (Giocomo lab) | iWT, wild-type littermates of floxed HCN1 mice, injected with AAV-cre in MEC | Dorsal hippocampus, CA1 ("how grid scale influenced spatial coding in dorsal hippocampus") | Tetrodes, freely moving | Open arenas, real. Size in Online Methods, not read. Familiarity not stated | Not in the text read | Mean place field size 303.75 ± 204.07 cm², mean ± SD across cells. Mean of each cell's largest field 373.34 ± 205.71 cm² | 113 iWT place cells, 11 iWT mice | 19.7 cm equivalent diameter (largest field: 21.8 cm) | FT, derived |
| D6 | Keinath et al. 2014 | Male C57BL/6, 2-5 months | Dorsal CA1. Tetrodes at AP -1.8, ML -1.5, DV -1.0 mm, chosen with Fanselow & Dong 2010 markers, checked by histology | Tetrodes, freely moving | 35 cm cylinder with 7 cues (SEA), real, described as novel | Contiguous pixels at least 80 cm² with rate at least 20% of peak. 70 × 70 grid, 3 cm SD Gaussian. Fields of a cell summed | About 0.061 m² summed, mean (error bars are SE). About 64% of the 962 cm² arena | 89 cells, 6 mice (visuospatial set) | 28.0 cm equivalent diameter summed (A = 614 cm²). 27.1 cm per field (1.06 fields per cell) | figure-read, derived |
| D7 | Keinath et al. 2014 | As D6 | As D6 | As D6 | 35 cm cylinder with different cues (SEB) | As D6 | About 0.063 m² summed | As D6 | 28.2 cm equivalent diameter (A = 627 cm²) | figure-read, derived |
| D8 | Keinath et al. 2014 | As D6 | As D6 | As D6 | 70 cm cylinder, scaled-up SEA (LE). Cells mostly remapped | As D6 | About 0.086 m² summed. About 22% of the 3,848 cm² arena | As D6 | 33.1 cm equivalent diameter summed (A = 860 cm²). 22.3 cm per field (2.20 fields per cell) | figure-read, derived |
| D9 | Keinath et al. 2014 | As D6 | As D6 | As D6 | 35 cm cylinder without visual cues, no odor | As D6 | About 0.058 m² summed (neutral odor gave the same reading) | 89 cells, 5 mice (olfactory set) | 27.2 cm equivalent diameter (A = 581 cm²) | figure-read, derived |
| D10 | McHugh et al. 1996 | Controls: 3 Cre (T29-1), 1 floxed NR1, 1 wild-type. Hybrid CBA-C57BL/6J-129/Sv, 4-6 months | CA1 (level not stated in the text read) | Tetrodes, freely moving | Linear track 1.2 m × 3 cm and an L-shaped version, real | Count of 2 × 2 cm pixels with mean rate above 1 Hz, summed over the whole map | 106.0 ± 7.40 pixels, "≈420 cm²". The ± is not defined; its size fits SEM | 56 cells, 12 days, 5 mice (Table 1); 55 cells in the Fig. 4B legend | Not converted. A 2D area on a 3 cm wide track with an absolute 1 Hz threshold has no clean width. Not used in the proposed rows | FT |
| D11 | Chockanathan & Padmanabhan 2021 | Male C57BL/6J, 9-10 weeks | Dorsal CA1. Craniotomy 2.5 mm caudal, 1.5 mm right of bregma | Acute 128-channel silicon probe (4 shanks), head-fixed on a wheel | Virtual 1D track, 1.9 m. Habituated 7 days | Bins around the peak with rate at least 1.5 SD above the mean. 111 bins of 1.7 cm, 5-bin box smoothing. All units, not only place cells | 9.4 ± 2.0 cm, mean ± std across sessions | 7 sessions, 4 mice, 42-80 units per recording, 329 units in all | 9.4 cm (length). Definition-specific, see proposed rows | FT |

Related number from the same animals as D5. Grid spacing in iWT mice was 36.26 ± 9.27 cm (mean ± SD, 37 grid cells, 12 iWT mice, MEC tetrodes, same kind of open arenas). The same-study ratio is 19.7 / 36.3 = 0.54 (derived). The grid cells were not assigned to modules or dorsoventral levels in the text I read.

### Table 2. Dorsal CA1, calcium imaging

| ID | Study | Strain | Region and level, how defined | Method | Environment | Field definition | Value as reported, statistic | N | Width for the table | Tag |
|---|---|---|---|---|---|---|---|---|---|---|
| I1 | Ziv et al. 2013 | Male C57BL/6, 8-12 weeks | Dorsal CA1. AAV injected at -1.9 mm AP, 1.4 mm ML, -1.65 mm DV; guide tube just dorsal to CA1 | One-photon miniscope, GCaMP3 (AAV2/5-CaMKIIα), freely moving | Elevated linear track 84 × 4.5 cm, real, familiar (trained, imaged on 10 sessions over 45 days) | Transient rate in 3.5 cm bins, last 7 cm at each end excluded, Gaussian sigma 8.75 cm. Width = bins at or above 50% of the field maximum. Place field required significant mutual information | "∼27%" of the 84 cm track, mean | 1,656 place fields, 4 mice | 22.7 cm (length) | FT, derived |
| I2 | Ziv et al. 2013 | As I1 | As I1 | As I1 | As I1 | As I1 | 24 cm median; "± 3.5 cm" is a 33% confidence interval and equals one bin | 4 mice, days 5-35 | 24 cm (length) | FT |
| I3 | Dombeck et al. 2010 | Male C57BL/6, about P50 | Dorsal CA1. Window over dorsal hippocampus; Fig. 2 "Imaging CA1 place cells in the dorsal hippocampus" | Two-photon, GCaMP3 (AAV2/1-synapsin-1), head-fixed on a ball | Virtual linear track, 180 cm, familiar (7-10 days of training before imaging) | Mean ΔF/F in 80 bins, 3-point smoothing. Contiguous bins above 25% of (peak minus baseline). Width above 18 cm, plus rate and transient criteria and a bootstrap test. Fields touching a track end excluded from the mean | 50 ± 19 cm, mean. The ± is not defined; its size fits SD | 881 fields in 808 place cells, 47 time-series, 4 mice | 50 cm (length) | FT |
| I4 | Jarzebowski et al. 2022 | 13 male Thy1-GCaMP6f mice (C57BL/6J-Tg(Thy1-GCaMP6f)GP5.5Dkim/J) | Dorsal CA1. GRIN lens at -1.75 AP, 1.75 ML, 1.35-1.40 DV mm from bregma | One-photon miniscope (v3 and v4), freely moving | Cheeseboard maze, 120 cm diameter, real. Random foraging on the first 3 days on the maze | 6 × 6 cm bins, Gaussian sigma 12 cm. Size = fraction of the place map above half the maximum, so all fields of a cell are summed | 3.4 ± 0.1% of the maze, mean ± SEM. Fields per place cell 1.7 ± 0.04 | 720 cells from 13 mice (dCA1 and iCA1 together); dCA1 7 mice, 170 ± 19 cells per day. Field count from 39 sessions | 22.1 cm equivalent diameter summed (A = 385 cm²). 17.0 cm per field | FT, derived |
| I5 | Dong et al. 2021 (Sheffield lab) | Male C57BL/6J, 10-12 weeks | Dorsal CA1 | Two-photon, GCaMP6f, head-fixed on a treadmill | Virtual linear track, 3 m, one direction with teleport. Familiar (F) | Mean ΔF/F in 50 bins of 6 cm. Contiguous bins above 15% of (peak minus baseline). Width must be 20-150 cm. Clipped fields at the track ends excluded from width analyses | Median about 38 cm (quartiles about 26 and 56 cm), from the familiar CDF | 1,643 F fields before exclusions, 7 sessions, 4 mice | 38 cm (length) | figure-read |
| I6 | Dong et al. 2021 | As I5 | As I5 | As I5 | Same rig, novel track (N), first day | As I5 | Median about 46 cm (quartiles about 28 and 70 cm) | 1,712 N fields before exclusions | 46 cm, novel | figure-read |
| I7 | Hainmueller & Bartos 2018 | Not in the text read | CA1, window "above CA1" | Two-photon, GCaMP6f, head-fixed on a ball | Virtual linear track, 4 m, familiar (at least 10 days) and novel contexts. Which context the CDF uses is not stated in the legend I read | Not in the text read (Methods missing from the Europe PMC version) | Median about 0.45 m (quartiles about 0.30 and 0.66 m) | 1,015 CA1 place cells | 45 cm (length) | figure-read |

### Table 3. Dorsal CA3

| ID | Study | Strain | Region and level, how defined | Method | Environment | Field definition | Value as reported, statistic | N | Width for the table | Tag |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Hussaini et al. 2011 | As D3 | Dorsal, proximal CA3. Same AP and ML as CA1, lowered 1.4 mm below the brain surface or advanced from CA1 | Tetrodes, freely moving | Square box 50 × 50 cm | As D3 | 22.9 ± 1.3% of area, mean ± SEM | Control CA3: 57 place cells from 8 mice across box and track | 27.0 cm equivalent diameter (A = 573 cm²) | FT, derived |
| C2 | Hussaini et al. 2011 | As D3 | As C1 | As C1 | Track enclosure 100 × 15 cm | As D3 | 27.4 ± 1.55% of area, mean ± SEM | As C1 | 27.4 cm as a length fraction. 22.9 cm equivalent diameter (A = 411 cm²) | FT, derived |
| C3 | Dong et al. 2021 | Grik4-cre (C57BL/6-Tg(Grik4-cre)G32-4Stl/J), male | Dorsal CA3a; cre restricts GCaMP6f to CA3 pyramidal cells | As I5 | As I5, familiar | As I5 | Median about 44 cm (quartiles about 32 and 62 cm). CA3 wider than CA1, P = 7.1 × 10^-4 | 336 F fields before exclusions, 13 sessions, 7 mice | 44 cm (length) | figure-read, FT for the P value |
| C4 | Dong et al. 2021 | As C3 | As C3 | As I5 | As I6, novel | As I5 | Median about 52 cm (quartiles about 34 and 76 cm) | 532 N fields before exclusions | 52 cm, novel | figure-read |
| C5 | Hainmueller & Bartos 2018 | Not in the text read | "CA2/3", imaged through a more lateral window. "some CA2 cells may also have been included" | As I7 | As I7 | As I7 | Median about 0.40 m (quartile 0.75 at about 0.61 m). CA3 vs CA1 not significant | 243 CA2/3 place cells | 40 cm (length) | figure-read |

### Table 4. Intermediate CA1

| ID | Study | Strain | Region and level, how defined | Method | Environment | Field definition | Value as reported, statistic | N | Width for the table | Tag |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | Jarzebowski et al. 2022 | As I4 | Intermediate CA1. GRIN lens at -3.16 AP, 3.6-3.8 ML, 3.40-3.70 DV mm | As I4 | As I4 | As I4 | 8.2 ± 0.4% of the maze, mean ± SEM. iCA1 larger than dCA1, F(1,10) = 12.5, p = 0.006. Fields per place cell 1.6 ± 0.06 | iCA1 6 mice, 70 ± 11 cells per day (720 cells in total with dCA1) | 34.4 cm equivalent diameter summed (A = 927 cm²). 27.2 cm per field | FT, derived |

### Table 5. Ventral CA1

| ID | Study | Strain | Region and level, how defined | Method | Environment | Field definition | Value as reported, statistic | N | Width for the table | Tag |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | Keinath et al. 2014 | As D6 | Ventral CA1. Tetrodes at AP -3.0, ML -3.9, DV -2.8 mm, Fanselow & Dong 2010 markers, histology | As D6 | 35 cm cylinder (SEA) | As D6 | About 0.086 m² summed, about 89% of the arena | 65 cells, 5 mice (visuospatial set) | 33.1 cm equivalent diameter summed (A = 860 cm²). 33.7 cm per field (0.97 fields per cell). Capped by the 35 cm arena | figure-read, derived |
| V2 | Keinath et al. 2014 | As D6 | As V1 | As D6 | 35 cm cylinder (SEB) | As D6 | About 0.085 m² summed | As V1 | 32.8 cm equivalent diameter (A = 847 cm²) | figure-read, derived |
| V3 | Keinath et al. 2014 | As D6 | As V1 | As D6 | 70 cm cylinder (LE) | As D6 | About 0.138 m² summed, about 36% of the arena. Increase from the 35 cm cylinder P < 0.001 | As V1 | 41.9 cm equivalent diameter summed (A = 1,380 cm²). 29.3 cm per field (2.04 fields per cell) | figure-read, derived, FT for the P value |
| V4 | Keinath et al. 2014 | As D6 | As V1 | As D6 | 35 cm cylinder without visual cues, no odor | As D6 | About 0.082 m² summed (neutral odor gave the same reading). Region effect in the olfactory set F(1,164) = 12.3, P < 0.006 | 42 cells, 4 mice (olfactory set) | 32.3 cm equivalent diameter (A = 821 cm²) | figure-read, derived |
| V5 | Chockanathan & Padmanabhan 2021 | As D11 | Ventral CA1. Craniotomy 3.15 mm caudal, 3.15 mm right of bregma; depth not given in the text read | As D11 | As D11 | As D11 | 11.9 ± 6.3 cm, mean ± std across sessions. Not different from dorsal, P = 0.84 | 6 sessions, 4 mice, 43-62 units per recording, 196 units in all | 11.9 cm (length). Definition-specific | FT |

Keinath et al. 2014, visuospatial set, region effect on field size F(1,197) = 5.9, P < 0.02 (Table 1, F). No mouse study I found reports intermediate or ventral CA3 field sizes.

## Quotes and locations

Harvey CD, Collman F, Dombeck DA, Tank DW (2009), Nature 461:941-946. Europe PMC author manuscript PMC2771429.
- D1, Results, "Place cells in a virtual environment": "In all mice tested, we identified units with place cell characteristics (23 cells from 8 mice, Methods). ... Place field size was, on average, 41 ± 14 cm."
- D2, Results, "Intracellular dynamics of place cells": "field size = 43 ± 13 cm; n = 8 cells from 8 mice".
- Methods: "To identify place fields, we found groups of adjacent bins with firing rates greater than 0.25 times the rate in the peak bin. We selected only those fields that were larger than 8 bins (18 cm) in length, had mean in-field firing rates of greater than 1.5 Hz, and had mean in-field firing rates more than 3 times larger than the mean out-of-field firing rate". "The virtual track was 180 cm long and 9 cm wide". "The effective width of the track (i.e. the distance the mouse could actually move) was ~1 cm".

Hussaini SA, Kempadoo KA, Thuault SJ, Siegelbaum SA, Kandel ER (2011), Neuron 72:643-653. PMC4435580.
- D3, D4, C1, C2, Results: "In the box, CA1 place fields ... were on average 55.3 % larger ... in knockout mice (percent area = 32.3 ± 2.75 %) compared to control mice (percent area = 20.8 ± 1.55 %). In the track ... (percent area = 37.9 ± 2.55 %) compared to littermate controls (percent area = 24.8 ± 1.8 %)." "In the box ... CA3 ... in knockout mice (percent area = 29 ± 2.5 %) compared to control mice (22.9 ± 1.3 %). Similarly in the track, CA3 place fields were about 22.9 % larger ... in knockout mice (percent area = 33.7 ±2.35 %) compared to control mice (27.4 ± 1.55 %)".
- Results: "Place cell recordings were obtained in the dorsal hippocampus from proximal regions of CA1 and CA3 (1.8 ± 0.06 mm lateral from midline)".
- Methods, Place field size: "we calculated the number of pixels inside the enclosure where place cells fired normalized with the number of pixels the mice visited. Only the top 80 % of the firing peak with at least 8 contiguous pixels was used and defined as the place field." Bins were 3 cm × 3 cm.
- Methods: "One enclosure was a box (50 × 50 × 50 cm) ... The other enclosure was a track (100 × 15 × 30 cm)". "In CA1, we recorded 80 place cells ... from 10 knockout mice and 77 place cells ... from 11 control mice. In CA3, we recorded 63 place cells ... from 7 knockout mice and 57 place cells ... from 8 control mice."
- Figure 3 legend: "Data show means ±SEM."

Mallory CS, Hardcastle K, Bant JS, Giocomo LM (2018), Nat Neurosci 21:270-282. PMC5823610.
- D5, Figure 2 legend (b, c): "Cumulative distributions of the mean place field sizes (mean ± SD; iWT = 303.75 ± 204.07 cm2, iCre-KO = 438.71 ± 337.04 cm2 ...) (b), maximum place field sizes (mean ± SD; iWT = 373.34 ± 205.71 cm2 ...)". "(n = 113 iWT place cells, 126 iCre-KO place cells)".
- Results: "We recorded CA1 place cells in iWT (n = 11) and iCre-KO mice (n = 8) as they explored open arenas".
- Grid spacing, Results: "(mean ± SD; spacing: iWT = 36.26 ± 9.27 cm, iCre-KO = 42.16 ± 10.53 cm ...)" with "37 iWT and 62 iCre-KO grid cells".

Keinath AT, Wang ME, Wann EG, Yuan RK, Dudman JT, Muzzio IA (2014), Hippocampus 24:1533-1548. PDF from the UIC Indigo repository (figshare file 40117177). Page numbers are journal pages.
- Coordinates, p. 1534: "dorsal: AP = -1.8; ML = -1.5; DV = -1.0; ventral: AP = -3.0, ML = -3.9; DV = -2.8" (minus and equals signs restored from the PDF text layer). "Coordinates were selected according to the neuroanatomical markers recently characterized in C57Bl6 mice along the longitudinal axis (Fanselow and Dong, 2010)."
- Environments, p. 1534: "(a) A 35 cm cylinder (standard environment, SEA) containing a distinct configuration of seven black visual cues". "(d) A scaled-up large environment identical in shape, color, and cue configuration to the SEA (LE, 70 cm in diameter)". "we recorded from the DH of six animals (89 cells) and the VH of five animals (65 cells)". Olfactory set, p. 1536: "we recorded from the DH of five animals (89 cells) and the VH of four animals (42 cells)".
- Field definition, p. 1536: "Place fields were defined as any set of contiguous pixels totaling at least 80 cm2 in which the average firing rate was at least 20% of the peak firing rate (Rowland et al., 2011). In the event that a cell yielded multiple place fields, the sum of all fields was taken as the place field size."
- Figure 1 legend, p. 1535: "J) Consistent with prior work, ventral cells displayed larger fields than dorsal cells, an effect that was significant in the LE (Table 1; F)." "Error bars indicate ±1 standard error."
- Results, p. 1538: "Place field size exhibited by VH cells increased significantly in the LE compared to the SE (P < 0.001), while those of dorsal cells on average did not significantly expand (Fig. 1J, Table 1; F)." "we typically observed remapping in the LE rather than expansion of the place fields".
- Table 1, F, p. 1539: Region F(1,197) = 5.9, P < 0.02 (visuospatial); F(1,164) = 12.3, P < 0.006 (olfactory).
- Discussion, p. 1545: "because ventral cells are broadly tuned and frequently span the entirety of the environment". Discussion also says "in novel, non-rewarded environments".

McHugh TJ, Blum KI, Tsien JZ, Tonegawa S, Wilson MA (1996), Cell 87:1339-1349. PDF from the Tonegawa lab site via the Internet Archive.
- D10, Fig. 4B legend, p. 1344: "Pixels in which the average rate of firing exceeded 1 Hz were included when calculating field size. ... The mean field size in NMDAR1 CA1-KO animals was 140.3 pixels (≈560 cm2), while in control animals the mean size was 106.0 pixels (≈420 cm2)". The PDF text layer renders ≈ as "z".
- Results, p. 1342: "We measured place field size by counting the number of pixels (pixels cover a 2 cm × 2 cm area) where average firing rates exceeded 1 spike per second. ... We therefore restricted our analysis to data from the linear and L-shaped tracks". Methods, p. 1348: "The linear track was 1.2 meters in length and 3 cm in width."
- Table 1, p. 1341: RUN control "(n = 56, D = 12, N = 5)", field size "106.0 ± 7.40".

Chockanathan U, Padmanabhan K (2021), eNeuro 8(5):ENEURO.0211-21.2021. PMC8425966.
- D11, V5, Figure 3H legend: "Place field sizes were not significantly different for dorsal and ventral CA1 neurons (mean ± std: dCA1 = 9.4 ± 2.0 cm, vCA1 = 11.9 ± 6.3 cm, p = 0.84, two-sided Wilcoxon rank-sum test, ndCA1 = 7 recording sessions, nvCA1 = 6 recording sessions)."
- Methods: "The width of a place field was defined as the number of spatial bins around the place field center for which the neuron's firing rate was at least 1.5 SDs above the mean." "A virtual 1D 1.9-m track". Craniotomies "dorsal CA1 (coordinates relative to bregma: 2.5 mm caudal, 1.5 mm right) and ventral CA1 (coordinates relative to bregma: 3.15 mm caudal, 3.15 mm right)".
- Robustness checks in the Extended Data legends (place-cell subsets, other smoothing kernels, centre-of-mass centres) gave dCA1 means of 7.0 to 14.5 cm and vCA1 means of 6.8 to 25.1 cm. None reached P < 0.05. The lowest was P = 0.05, for all units pooled ("ndCA1 = 329 neurons, nvCA1 = 196 neurons", dCA1 9.942 ± 10.935 cm, vCA1 10.114 ± 12.910 cm).

Ziv Y, Burns LD, Cocker ED, Hamel EO, Ghosh KK, Kitch LJ, El Gamal A, Schnitzer MJ (2013), Nat Neurosci 16:264-266. PMC3784308.
- I1, Results: "The mean place field size was ∼27% of the 84-cm track, within the range published for mice". "For each place field we detected Ca2+ activity in 17 ± 14% of running passes (n = 1656 place fields, mean ± s.d.; range: 2-87%)."
- I2, Figure 2h legend: "Place fields had a median (± 33% confidence interval) width of 24 ± 3.5 cm."
- Online Methods, Place fields: "On the linear track we considered 3.5-cm spatial bins and excluded the last 7 cm at each end where water rewards were given. ... applied a Gaussian smoothing filter (linear track: σ = 8.75 cm; open field: σ = 3.5 cm) ... The number of bins in which a place field had a value ≥50% of its maximum determined the place field's width."
- Online Methods, Mice: "Male C57/BL6 mice ... We first injected AAV2/5-CaMKIIα-GCaMP3 (250 nL) into CA1 (-1.9 mm from Bregma, 1.4 mm mediolateral, -1.65 mm dorsoventral)."

Dombeck DA, Harvey CD, Tian L, Looger LL, Tank DW (2010), Nat Neurosci 13:1433-1440. PMC2967725.
- I3, Results: "Across our 47 time-series datasets, 808 place cells and 881 place fields in those place cells were identified." "The mean width of the optically identified place fields was 50±19 cm (Fig. 4d), which is ∼20% larger than the mean width of place fields defined by firing rate found in extracellular recordings under similar conditions (41±14cm)".
- Methods, Data analysis: "Potential place fields were first identified as contiguous regions of this plot in which all of the points were greater than 25% of the difference between the peak ΔF/F value (for all 80 bins) and the baseline value". "The field must be >18cm in width". "For the mean place field width calculation, only fields in which neither field edge was at the end of the track were included."

Jarzebowski P, Hay YA, Grewe BF, Paulsen O (2022), Curr Biol 32:834-841.e5. Accepted manuscript from UCL Discovery (eprint 10142268). Page numbers are manuscript pages.
- I4, M1, Table S1, p. 29 (column header "mean ± SEM"; I checked the rendered table, since the text layer scrambles it): "Place field size (% of maze)", "dCA1: 3.4 ± 0.1%", "iCA1: 8.2 ± 0.4%", "log-linear mixed-effects F(1,10) = 12.5, p = 0.006", BF10 11.5, "720 cells from 13 mice". Next row: "Mean count of fields", "dCA1: 1.7 ± 0.04", "iCA1: 1.6 ± 0.06", "39 sessions from 13 mice".
- Results, p. 3: "We imaged daily 170 ± 19 dCA1 cells from seven animals and 70 ± 11 iCA1 cells from six animals". "Mice foraged for liquid rewards baited in randomly selected wells of the 120-cm-diameter maze." "the place fields were larger in iCA1 (strong evidence: BF10 = 12, CI = [24%, 204%]; Figure 1F)".
- Methods, p. 12-13: dCA1 lens "at coordinates -1.75 AP, 1.75 ML, 1.35-1.40 DV (in mm from Bregma)"; iCA1 lens "at coordinates -3.16 AP, 3.6-3.8 ML, 3.40-3.70 DV".
- Methods, p. 16: "The total activity inside 6 x 6 cm bins was summed ... smoothed across the space using a 2D Gaussian kernel with σ = 12 cm." "we report field size as percentage of the maze area." "We defined the field size as the fraction of a place map with values exceeding half the maximum value."

Dong C, Madar AD, Sheffield MEJ (2021), Nat Commun 12:2977. PMC8137926 and Supplementary Information PDF.
- I5, I6, C3, C4, Supplementary Fig. 2 legend: "e, Histogram of each PFs width in CA1 (green) and CA3 (orange) in the familiar environment. Right, corresponding cumulative fraction plot. Wilcoxon rank sum test, two-sided, ***, P < 0.001, CA1-CA3 F, P = 7.1 × 10−4, CA1-CA3 N, P = 4.9 × 10−5, CA1 F-N, P = 3.2 × 10−15, CA3F-N, P = 2.6 × 10−6. f, Same plots as (e) but in N."
- Methods, Defining PFs: "The 3 m track was divided into 50 bins (6 cm per bin)." "points were >15% of the difference between the peak ΔF/F value (from all 50 bins) and the baseline value". "The field width must be >20 cm and <150 cm." Methods, PF width: "PF width on each lap is calculated as the difference between the first and last bin with in-field activity."
- Fig. 2 legend, panel d: "F (left, CA1 n = 1643 PFs, CA3 n = 336 PFs) and N (middle, CA1 n = 1712 PFs, CA3 n = 532 PFs)". Same legend, panel a: "n = 7 sessions in four mice for CA1, n = 13 sessions in seven mice for CA3".

Hainmueller T, Bartos M (2018), Nature 558:292-296. Europe PMC author manuscript PMC7115829; figure from Nature's Extended Data Fig. 3 image.
- I7, C5, author-manuscript Extended Data Figure 2 legend: "(a) Cumulative distribution of place field widths for GCs (red), CA2/3 PYR (blue) and CA1 PYR (black). *** p<0.001, n.s. not significant". In the published figure this is panel c, with n = 366 DG, 243 CA3 and 1,015 CA1.
- Main text: "a 4-m-long virtual linear track". "After at least 10 days of familiarization to this track (familiar context)". "Data were obtained predominantly from CA3 ... but some CA2 cells may also have been included".

## Derivations

Equivalent diameter d = 2 * sqrt(A / pi).
- D3. A = 0.208 × 2,500 cm² = 520 cm². d = 2 * sqrt(520 / 3.1416) = 25.7 cm.
- D4. A = 0.248 × 1,500 cm² = 372 cm², d = 21.8 cm. Length fraction 0.248 × 100 cm = 24.8 cm.
- C1. A = 0.229 × 2,500 = 572.5 cm², d = 27.0 cm.
- C2. A = 0.274 × 1,500 = 411 cm², d = 22.9 cm. Length fraction 27.4 cm.
- Hussaini divides by visited pixels, not the whole floor. The conversions assume the mouse visited nearly all of the enclosure.
- D5. A = 303.75 cm², d = 19.7 cm. Largest field A = 373.34 cm², d = 21.8 cm.
- I1. 0.27 × 84 cm = 22.7 cm. Ziv smoothing floor: FWHM = 2.3548 × 8.75 cm = 20.6 cm.
- I4. Maze area = pi × 60² = 11,310 cm². A = 0.034 × 11,310 = 385 cm², d = 22.1 cm. Per field A = 385 / 1.7 = 226 cm², d = 17.0 cm.
- M1. A = 0.082 × 11,310 = 927 cm², d = 34.4 cm. Per field A = 927 / 1.6 = 580 cm², d = 27.2 cm.
- Keinath arenas. 35 cm cylinder = pi × 17.5² = 962 cm². 70 cm cylinder = pi × 35² = 3,848 cm².
- D6. A = 614 cm², d = 28.0 cm, 614 / 962 = 64% of the arena. Per field 614 / 1.06 = 578 cm², d = 27.1 cm.
- D8. A = 860 cm², d = 33.1 cm, 22% of the arena. Per field 860 / 2.20 = 392 cm², d = 22.3 cm.
- V1. A = 860 cm², d = 33.1 cm, 89% of the arena. Per field 860 / 0.97 = 890 cm², d = 33.7 cm.
- V3. A = 1,380 cm², d = 41.9 cm, 36% of the arena. Per field 1,380 / 2.04 = 675 cm², d = 29.3 cm.
- D7, D9, V2, V4 follow the same formula. The A values are in the tables.
- Within-study ventral or intermediate to dorsal ratios of linear size. Keinath, 70 cm cylinder: 41.9 / 33.1 = 1.27 summed, 29.3 / 22.3 = 1.31 per field. Keinath, 35 cm cylinder: 33.1 / 28.0 = 1.18, held down by the arena. Jarzebowski: 34.4 / 22.1 = 1.56 summed, 27.2 / 17.0 = 1.60 per field. Chockanathan: 11.9 / 9.4 = 1.27, not significant.

A check that weakens I4 and M1. A 2D Gaussian kernel with sigma 12 cm has a half-maximum disc of radius 1.1774 × 12 = 14.1 cm, area 627 cm², which is 5.5% of the maze. That is larger than the reported dCA1 mean of 3.4%. A field smoothed by that kernel cannot be smaller than the kernel, so the size was probably computed on a less smoothed map, or the kernel is described differently from how it was applied. The iCA1 to dCA1 ratio is more reliable than either absolute value.

## How figure values were read

- Keinath Figs. 1I, 1J and 3H. I rendered the PDF page at 400 dpi with PyMuPDF and located the tick-label centres by pixel rows. In Fig. 1J the labels 0, 0.1 and 0.2 m² sit 154 pixels apart. I took the top row of each coloured bar fill as the bar height. The figure is a raster image, so the reading error is about ±0.002 m² (about ±20 cm²). Field counts in Fig. 1I were read the same way, ±0.03.
- Dong Supplementary Fig. 2e, f. I rendered the CDF panels at 400 dpi, calibrated both axes from the tick marks and labels, and found where each coloured step curve crosses 0.25, 0.5 and 0.75. Widths come in 6 cm steps, so medians are about ±3 cm.
- Hainmueller Extended Data Fig. 3c (published numbering). I used Nature's JPEG (1,521 × 2,296 pixels), calibrated the axes from the ticks (0.5 m = 70 pixels) and found the 0.5 crossing of the black (CA1) and blue (CA2/3) curves. The error is about ±0.02 m. The two curves overlap, so the CA3 quartiles are less certain.

## Proposed rows (dorsal to ventral)

Widths in cm. Each end of a range names its source. Rows with different methods stay separate.

| Row | Region and level | Method | Width range | Low end | High end | Environment | Status |
|---|---|---|---|---|---|---|---|
| 1 | Dorsal CA1 | Tetrodes, real open arenas and track enclosures | 20 to 28 | Mallory 2018, 19.7 (D5, equivalent diameter of the mean field, arena size not read) | Keinath 2014, 28.0 (D6, summed fields, 35 cm cylinder) | 35 to 100 cm. Hussaini sits inside the range (25.7 box, 24.8 track). Keinath per field in the 70 cm cylinder is 22.3 | FT for Mallory and Hussaini, figure-read for Keinath. All widths derived from areas |
| 2 | Dorsal CA1 | Extracellular and whole-cell, virtual track | 41 to 43 | Harvey 2009, 41 (D1) | Harvey 2009, 43 (D2) | 180 cm virtual track. One lab, one rig | FT |
| 2b | Dorsal CA1 | Silicon probe, virtual track, 1.5 SD criterion | 9.4 | Chockanathan 2021 (D11) | same | 1.9 m virtual track. The criterion keeps only the top of the field. Do not pool with rows 1 and 2 | FT |
| 3 | Dorsal CA1 | One-photon miniscope, real environments | 22 to 24 | Jarzebowski 2022, 22.1 (I4, summed, derived; see the smoothing check) or Ziv 2013 mean 22.7 (I1, derived) | Ziv 2013 median 24 (I2) | 84 cm track; 120 cm maze. Ziv's smoothing floor is 20.6 cm | FT and derived |
| 4 | Dorsal CA1 | Two-photon, virtual tracks | 38 to 50 | Dong 2021 median 38 (I5, familiar) | Dombeck 2010 mean 50 (I3) | 1.8 to 4 m virtual tracks. Hainmueller median 45 (I7) sits inside | FT for Dombeck, figure-read for Dong and Hainmueller |
| 5 | Dorsal CA3 | Tetrodes, real enclosures | 23 to 27 | Hussaini 2011, 22.9 (C2, track, equivalent diameter) | Hussaini 2011, 27.4 (C2 length fraction) or 27.0 (C1 box) | 50 × 50 cm box, 100 × 15 cm track. Single study | FT, derived |
| 6 | Dorsal CA3 | Two-photon, virtual tracks | 40 to 44 | Hainmueller 2018 CA2/3 median 40 (C5) | Dong 2021 CA3a median 44 (C3, familiar) | 3 to 4 m virtual tracks | figure-read |
| 7 | Intermediate CA1 | One-photon miniscope, real maze | 27 to 34 | Jarzebowski 2022, 27.2 per field (M1) | Jarzebowski 2022, 34.4 summed (M1) | 120 cm maze. Single study. Ratio to dorsal CA1 in the same study is 1.56 | FT, derived |
| 8 | Ventral CA1 | Tetrodes, real cylinders | 29 to 42 | Keinath 2014, 29.3 per field in the 70 cm cylinder (V3) | Keinath 2014, 41.9 summed in the 70 cm cylinder (V3) | 35 and 70 cm cylinders. The 35 cm values (32 to 33, V1, V2, V4) fill about 89% of the arena, so they are capped by the walls | figure-read, derived |
| 8b | Ventral CA1 | Silicon probe, virtual track, 1.5 SD criterion | 11.9 | Chockanathan 2021 (V5) | same | 1.9 m virtual track. Not different from dorsal | FT |
| 9 | Intermediate and ventral CA3 | Any | No data | | | I found no mouse study | |

Notes on environment size.
- The mouse electrophysiology arenas are small, 35 to 120 cm across. Fields there are 20 to 34 cm wide, and in a 35 cm cylinder ventral fields cover most of the floor. These widths are probably lower bounds on the field size in a large space.
- In Keinath et al. the summed dorsal field area rose from about 614 to 860 cm² when the cylinder doubled in diameter, though not significantly. The ventral area rose from about 860 to 1,380 cm² (P < 0.001). Most cells also gained a second field and remapped, so the per-field width stayed near 22 cm (dorsal) and 29 cm (ventral).
- Virtual tracks of 1.8 to 4 m give wider dorsal CA1 fields, 38 to 50 cm by imaging and 41 cm by electrophysiology. Within one rig, imaging widened fields by about 20% over spikes (Dombeck 2010).
- For the ratio table, use row 1 or row 3 with grid spacings measured in similar-sized arenas. Use rows 2, 4 and 6 only with grid spacings measured in virtual reality or on long tracks.
- If a single ventral-to-dorsal multiplier is needed, the within-study values are 1.18 to 1.31 (Keinath, ventral CA1) and 1.56 (Jarzebowski, intermediate CA1).

## Not verified or not obtained

- Kentros et al. 2004 (Neuron). Not in PMC, and I found no open copy. No numbers.
- Cacucci et al. 2008 (PNAS, PMC2396558). The PMC record has no open full-text XML, and the PNAS site blocked downloads. The UCL repository copy (eprint 14898) is the 2009 PhD thesis of the second author, Ming Yi. It defines field size as "the percentage of sampled environment in which the cell's firing rate was greater than the overall mean firing rate during that trial" and shows values only in figures. Not extracted.
- Chen, King, Burgess & O'Keefe 2013 (PNAS, mouse real vs virtual square). Full text blocked. No numbers.
- Resnik et al. 2012 (J Neurosci, Mehta lab, mouse GluA1 knockout, PMC3752061). Full text blocked, so the track size and the wild-type widths are unverified. I found no other Mehta-lab mouse field-size paper.
- Buzsáki-lab mouse papers. Royer et al. 2012 (PMC4919905) gives no field widths in the text. Senzai & Buzsáki 2017 does not report field size per cell type in the text (existing note).
- GoodSmith et al. 2019 (J Neurosci, mouse proximal CA3). Full text blocked. No numbers.
- Suh et al. 2013 (Neuron, PMC3871857). Field size is only in Fig. 2B, as the number of 2 cm bins above 1 Hz on a 76 × 10 cm track. Not read from the figure.
- Danielson et al. 2016 and Sheffield, Adoff & Dombeck 2017. Neither gives a field-width number in the text. Danielson reports no deep vs superficial difference (Fig. S2b).
- Cheng & Ji 2013 (eLife). No field-size measure in the text.
- Mallory 2018. Arena size and field definition are in the Online Methods, which I could not read.
- Hainmueller 2018. Strain, field-width definition and whether the CDF pools familiar and novel contexts are in the Methods, which the Europe PMC version lacks.
- Neuropixels studies of mouse CA1 field width. I found none.
- Intermediate and ventral CA3 in mice. I found none.

## Sources

- Chockanathan U, Padmanabhan K (2021). Divergence in population coding for space between dorsal and ventral CA1. eNeuro 8(5). [doi:10.1523/ENEURO.0211-21.2021](https://doi.org/10.1523/ENEURO.0211-21.2021), [PMC8425966](https://pmc.ncbi.nlm.nih.gov/articles/PMC8425966/)
- Ciocchi S, Passecker J, Malagon-Vina H, Mikus N, Klausberger T (2015). Selective information routing by ventral hippocampal CA1 projection neurons. Science 348:560-563. Rats. [doi:10.1126/science.aaa3245](https://doi.org/10.1126/science.aaa3245)
- Dombeck DA, Harvey CD, Tian L, Looger LL, Tank DW (2010). Functional imaging of hippocampal place cells at cellular resolution during virtual navigation. Nat Neurosci 13:1433-1440. [doi:10.1038/nn.2648](https://doi.org/10.1038/nn.2648), [PMC2967725](https://pmc.ncbi.nlm.nih.gov/articles/PMC2967725/)
- Dong C, Madar AD, Sheffield MEJ (2021). Distinct place cell dynamics in CA1 and CA3 encode experience in new environments. Nat Commun 12:2977. [doi:10.1038/s41467-021-23260-3](https://doi.org/10.1038/s41467-021-23260-3), [PMC8137926](https://pmc.ncbi.nlm.nih.gov/articles/PMC8137926/)
- Hainmueller T, Bartos M (2018). Parallel emergence of stable and dynamic memory engrams in the hippocampus. Nature 558:292-296. [doi:10.1038/s41586-018-0191-2](https://doi.org/10.1038/s41586-018-0191-2), [PMC7115829](https://pmc.ncbi.nlm.nih.gov/articles/PMC7115829/)
- Harvey CD, Collman F, Dombeck DA, Tank DW (2009). Intracellular dynamics of hippocampal place cells during virtual navigation. Nature 461:941-946. [doi:10.1038/nature08499](https://doi.org/10.1038/nature08499), [PMC2771429](https://pmc.ncbi.nlm.nih.gov/articles/PMC2771429/)
- Hussaini SA, Kempadoo KA, Thuault SJ, Siegelbaum SA, Kandel ER (2011). Increased size and stability of CA1 and CA3 place fields in HCN1 knockout mice. Neuron 72:643-653. [doi:10.1016/j.neuron.2011.09.007](https://doi.org/10.1016/j.neuron.2011.09.007), [PMC4435580](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435580/)
- Jarzebowski P, Hay YA, Grewe BF, Paulsen O (2022). Different encoding of reward location in dorsal and intermediate hippocampus. Curr Biol 32:834-841.e5. [doi:10.1016/j.cub.2021.12.024](https://doi.org/10.1016/j.cub.2021.12.024), [accepted manuscript](https://discovery.ucl.ac.uk/10142268/1/manuscript-deposited-merged.pdf)
- Jimenez JC et al. (2018). Anxiety cells in a hippocampal-hypothalamic circuit. Neuron 97:670-683. [doi:10.1016/j.neuron.2018.01.016](https://doi.org/10.1016/j.neuron.2018.01.016), [PMC5877404](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877404/)
- Keinath AT, Wang ME, Wann EG, Yuan RK, Dudman JT, Muzzio IA (2014). Precise spatial coding is preserved along the longitudinal hippocampal axis. Hippocampus 24:1533-1548. [doi:10.1002/hipo.22333](https://doi.org/10.1002/hipo.22333), [PMC4447627](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/), [UIC repository PDF](https://indigo.uic.edu/articles/journal_contribution/Precise_spatial_coding_is_preserved_along_the_longitudinal_hippocampal_axis/22596763)
- Mallory CS, Hardcastle K, Bant JS, Giocomo LM (2018). Grid scale drives the scale and long-term stability of place maps. Nat Neurosci 21:270-282. [doi:10.1038/s41593-017-0055-3](https://doi.org/10.1038/s41593-017-0055-3), [PMC5823610](https://pmc.ncbi.nlm.nih.gov/articles/PMC5823610/)
- McHugh TJ, Blum KI, Tsien JZ, Tonegawa S, Wilson MA (1996). Impaired hippocampal representation of space in CA1-specific NMDAR1 knockout mice. Cell 87:1339-1349. [doi:10.1016/S0092-8674(00)81828-0](https://doi.org/10.1016/S0092-8674(00)81828-0)
- Ziv Y, Burns LD, Cocker ED, Hamel EO, Ghosh KK, Kitch LJ, El Gamal A, Schnitzer MJ (2013). Long-term dynamics of CA1 hippocampal place codes. Nat Neurosci 16:264-266. [doi:10.1038/nn.3329](https://doi.org/10.1038/nn.3329), [PMC3784308](https://pmc.ncbi.nlm.nih.gov/articles/PMC3784308/)
