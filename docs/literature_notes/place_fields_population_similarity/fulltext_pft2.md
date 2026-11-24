# Full-text pass on seven supplied PDFs (pft2)

Date: 2026-11-24. Sources: `/home/jackking/cls/papers/pft2/` (seven PDFs) and their page-tagged text in `jobs/2c8b1c8e/tmp/pdftext_pft2/`. Scripts and renders are in `jobs/2c8b1c8e/tmp/pft2_extract/`. Every number below is a quote with its page, or is tagged figure-read (with the method) or derived (with the arithmetic).

Outputs from this pass, all in this folder:

- `pft2_additions_fields_rsc.csv` (6 rows), `pft2_additions_kernel_rsc.csv` (6 rows), `pft2_additions_fields_mec.csv` (8 rows), `pft2_additions_fields_mouse.csv` (1 row).
- `pft2_corrections.csv` (120 cell changes to 13 existing rows).
- `pft2_review_edits.md` (42 proposed edits to the review, each old text checked to occur once).

Page conventions. Journal page numbers where the PDF prints them; otherwise "PDF p." counts pages of the supplied file. "SM p." is the printed page of the Butler supplement.

Figure reading.

- Vector paths. Marker and error-bar centres taken from PDF drawing commands with PyMuPDF, calibrated on the vector tick marks (`markers.py`, `vec.py`). Precision about 0.5% of the axis.
- Colour-bar lookup on rasters. The figure raster was extracted at its native resolution. Each pixel took the value of the nearest colour in the colour bar of the same raster (RGB distance). Black and white overlay pixels (reconstruction lines, turn markers, dashed diagonals) were masked. Values were averaged over all pixels at the same |row - column| separation (`kernel_matrix.py`, `kernel_an15_final.py`, `kernel_an17.py`, `kernel_ring.py`). Precision about ±0.03 in r; distance scales are stated per figure.

## 1. Mallory, Hardcastle, Bant & Giocomo 2018, Nat Neurosci 21:270-282

File `s41593-017-0055-3.pdf`, 21 pages: article pp. 270-282 (PDF pp. 1-13), Online Methods and reporting summary (PDF pp. 14-21). The supplement (ESM) is not among the seven PDFs; two legends were read from the earlier sweep's text file `tmp/sweep_mec/dl/mallory2018_esm/esm1.txt` and are marked as such.

### Animals, arenas and maps (Online Methods)

- Strain. "Mice were generated from a hybrid 50:50% C57BL/6 J:129SVEV background" (PDF p. 14). "Male (n = 29) and female (n = 25) mice were used for in vivo tetrode recordings" (PDF p. 14).
- Arena. "Data was collected during 20–60 min sessions, while mice foraged for scattered food (crumbled chocolate cereal) in one of several 50-cm tall square arenas (box width and length, for MEC recordings: 70 cm, 90 cm or 100 cm; for hippocampal recordings: 50 or 62 cm). Box sizes were selected on a per-animal basis to maximize coverage." (PDF p. 14). "The animal was habituated to the testing arena ≥ 2 weeks before data collection began." (PDF p. 14). "Mice were exposed to the open field not more than twice per day, with sessions separated by at least 3 h" (PDF p. 14).
- Implants. "For MEC recordings, tetrodes were implanted at 0.4–0.5 mm AP from the transverse sinus, 3.25–3.45 mm ML and 0.8–1.0 mm below the dura, angled 0–4° in the posterior direction in the sagittal plane. For hippocampal recordings, tetrodes were implanted at –1.8 mm AP from bregma, 1.8 mm ML and 0.6–0.8 mm below the dura." (PDF p. 14). "After all data collection for a given set of cells was complete, tetrodes were moved ventral by 25 µm." (PDF p. 14).
- Maps. "Only cells with ≥ 100 spikes and sessions where the animal covered ≥ 70% of the environment were used in analyses. Position data was sorted into 2.5 × 2.5 cm bins and smoothed with a 21-sample boxcar window filter (400 ms, ten samples on each side). Time and spiking maps were smoothed separately using a quasi-Gaussian kernel over the adjacent 5 × 5 bins" (PDF p. 15). Speed filter: "only epochs of instantaneous running speeds ≥ 2.5 cm/s included" (PDF p. 15).

### Cell classes and field rules

- Stable spatial cells (the non-grid class). "Neurons were defined as stable spatial cells if their stability and coherence scores were higher than the 95th percentile of the shuffled data and they were not classified as grid or border cells (Supplementary Fig. 6)." (PDF p. 15). Stability: "within-trial stability was calculated by generating smoothed rate maps from even and odd time bins and quantifying the correlation in rates of firing between the two rate maps" (PDF p. 15). Shuffles: spikes shifted by more than 20 s, "This procedure was repeated 100 times" (PDF p. 15).
- No field rule is stated for stable spatial cells. The only field rule in the paper is for place cells: "Place field size was determined from binned firing rate maps, with place fields quantified as portions of the map with at least five contiguous bins (31.25 cm2) that reached 20% of the peak firing rate." (PDF p. 15). That the same rule was applied to the non-grid cells is an inference.
- Border cells. "Border cell firing fields were detected from the smoothed rate map as ≥ 10 connected bins with firing rates in 75th percentile of firing rates for that rate map. The edge of the firing field was determined by finding the bins that neighbored nonfield bins, and the mean distance of these edge bins from the nearest boundary defined as the width of the border cell." (PDF p. 15).
- Grid cells. "Grid field size was calculated from the radium of an inner circle fit around the center autocorrelation peak. Grid spacing was defined as the mean spacing between the central autocorrelation peak and the nearest six peaks. ... spacing measurements from cells in which the s.d. of the three grid spacing measurements fell above 2 s.d. of the mean were excluded (WT n = 3 cells; iCre-KO n = 4 cells)." (PDF p. 15).
- Place cells. "putative excitatory pyramidal neurons based on a waveform analysis (longest waveform peak-to-trough > 0.20 ms). Place cells were then defined as neurons with a spatial information value exceeding the 95th percentile of the shuffled distribution, with > 100 spikes and a peak firing rate of ≥ 0.1 Hz." (PDF p. 15).

### Numbers confirmed

- "we implanted tetrodes into 29 mice (12 iWT; 17 iCre-KO) and recorded cells in MEC (iWT n = 527; iCre-KO n = 600)" (p. 270).
- "spacing: iWT = 36.26 ± 9.27 cm, iCre-KO = 42.16 ± 10.53 cm, Z = –3.09, P = 0.0020; field size: iWT = 489.29 ± 301.62 cm2, iCre-KO = 502.64 ± 207.26 cm2" (p. 272). Fig. 1e-f legend: "n = 37 iWT cells, 62 iCre-KO cells ... Grid field size was determined from the radius of the circle around the center field of the autocorrelation map." (p. 271).
- Table 1 (p. 273), "Spatially stable cells n 32 38 ... Spatial stability 0.37 ± 0.15 0.37 ± 0.17 ... Mean field size 2,473.45 ± 2,233.2 cm2 2,692.29 ± 2,768.8 cm2 WRS Z = 0.12, P = 0.90 Percent of box covered by fields 43.24 ± 30.72% 45.58 ± 31.62% WRS Z = –0.15, P = 0.88". Table note: "Mean ± s.d. N values indicate cells" (p. 274). Confirms 2473 ± 2233 cm² (n = 32) and 2692 cm².
- Table 1, "Border cells n 86 71 ... Border width 14.43 ± 6.53 cm 14.23 ± 5.22 cm WRS Z = –0.30, P = 0.76" (p. 273). Confirms 14.43 ± 6.53 cm.
- Fig. 2 legend (p. 272): "The boundary width was set to 14 cm, the average width of all recorded border cells."; "mean place field sizes (mean ± s.d.; iWT = 303.75 ± 204.07 cm2"; "maximum place field sizes (mean ± s.d.; iWT = 373.34 ± 205.71 cm2"; "percentage of environment covered by fields (mean ± s.d.; iWT = 12.27 ± 6.1%"; "n = 113 iWT place cells, 126 iCre-KO place cells". Main text: "We recorded CA1 place cells in iWT (n = 11) and iCre-KO mice (n = 8)" (p. 274).
- Table 1 continued (p. 274): "Place cells n 113 126 Number of fields 1.95 ± 1.16 1.71 ± 1.09 WRS Z = 1.66, P = 0.10".
- Fig. 1h legend (p. 271), a broad model-defined non-grid class: "P-cells not classified as border or grid cells ... n = 355 iWT cells, 419 iCre-KO cells), percent of the environment covered by a field, or spatial stability (mean ± s.d.; % environment covered: iWT = 35.17 ± 32.52%, iCre-KO = 38.02 ± 33.53%, Z = –1.04, P = 0.30; stability: iWT = 0.18 ± 0.16". New row MNG40.
- ESM Supp. Fig. 7 (earlier sweep's text): grid cells "dorsal–ventral: iWT = 386 ± 222 µm"; smallest spacing per mouse "median: iWT = 26.6 cm".

### Arena size and the CA1 exclusion

The CA1 width row `mallory2018_1` (19.7 cm, the equal-area diameter of 303.75 cm²) was left out of the heat maps because its arena size was unknown. The CA1 arenas were 50 or 62 cm squares. The 19.7 cm diameter is 19.67/62 = 0.32 to 19.67/50 = 0.39 of the box side, and the mean field is 303.75/3844 = 7.9% to 303.75/2500 = 12.2% of the box area (derived). The row fits the class "real environment up to 2 m" and does not change that row's 14.8 to 42.5 cm range. The non-grid MEC arenas were 70, 90 or 100 cm squares. The mean non-grid field area is 2473.45/10000 = 25% to 2473.45/4900 = 50% of the box, and its 56.1 cm diameter is 0.56 to 0.80 of the box side (derived), so the box probably caps these fields.

### Linear-track CA1 (new mouse row `mallory2018_2`)

- Online Methods (PDF p. 16): "The linear track, constructed of black-painted wood, was 100 × 7 cm (L × W), elevated 23 cm above the floor and surrounded by black walls (31 cm)."; "Fields on the linear track were defined as follows: first, the peak bin rate in a field was required to be at least 1 Hz and exceed 20% of the peak rate for that cell (bins = 2.5 cm). The field was then extended in either direction as long as the firing rate remained above 20% of the field's peak rate. Field size was defined as the distance between the two field boundaries."; "Linear track analyses were restricted to fields whose boundaries fell within the central 70 cm of the track".
- ESM Supp. Fig. 11 (earlier sweep's text): "Field size on the linear track was larger in iCre-KO mice (mean ± SD; iWT = 22.98 ± 11.56, iCre-KO = 30.39 ± 14.92, two-tailed WRS Z = -2.26, p = 0.023; n = 42 fields from 30 cells from 3 iWT mice". The same legend: "Field width was estimated as the mean distance from the diagonal matrix to the population vectors with a mean correlation of 0.2 (iWT = 28.8 ± 8.1 cm, iCre-KO = 34.2 ± 8.5 cm". That is a population width at r = 0.2, not at 0.5, so it is not added as a kernel row.

### Corrections made

Arena size, familiarity, strain background, implant coordinates and field rules for MNG01 to MNG05 and `mallory2018_1`; `mallory2018_1` gains 1.95 fields per cell and a note that it can now enter the heat map.

## 2. Butler, Hardcastle & Giocomo 2019, Science 363:1447-1452

Files `science.aav5297.pdf` (6 pages, "1 of 6" to "6 of 6" = pp. 1447-1452) and `aav5297_butler_sm.pdf` (44 pages; Methods SM pp. 2-13, figures S1-S12, Tables S1-S4).

### Animals, sites, arenas

- "this study used male rats (rat n = 11, aged 4 - 8 months old at the start of experiments" (SM p. 2). "rats in the Experimental group (n = 8)" (SM p. 3); "One animal did not reach this performance criterion and therefore was excluded from all analyses." (SM p. 3); "We also recorded cells in a separate group of Control rats (n = 3)." (SM p. 4). Strain is not stated. So the 7 task-trained rats and the 3 controls are separate animals.
- "tetrodes were implanted 0.5 mm anterior to the transverse sinus (~8.5 mm posterior to bregma), and 3.8-4.2 mm lateral from midline"; "tetrodes were driven 2.5-3.5 mm ventrally into the brain"; "individual tetrodes were each driven ventrally by 0-250 µm" per day (SM p. 2). "Approximate recording sites were determined by calculating the distance corresponding to the amount of microdrive screw turns and following the tetrode track dorsally from the most ventral point (Fig. S1). Of the 778 cells recorded in both environments, 598 were recorded from MEC; the remaining 180 cells were recorded from surrounding cortical regions, including parasubiculum and postsubiculum." (SM pp. 2-3). No dorsoventral distances are given.
- "Neural data was collected as rats freely foraged in a 150 cm x 150 cm box (ENV1)"; ENV2 "a second, distinct 150 cm x 150 cm open field arena"; "a unique reward zone was defined as an unmarked 20 cm x 20 cm area whose borders were all at least 15 cm from the walls of the arena" (SM p. 3). Main text: "two arenas (1.5 m by 1.5 m)" (p. 1447); "mean number of sessions to reach criterion = 15; range = 8 to 24" (p. 1447).
- Maps: "Position data was sorted into 5 cm x 5 cm bins (each 1.5 m box was 30 bins x 30 bins). Occupancy and spiking maps were smoothed separately using a quasi-Gaussian kernel." (SM p. 4).

### Field rule

"Fields were identified in each speed- and position-matched rate map in order to compute field properties as a function of distance from the reward zone. First, we converted the rate map to a log-normalized scale. This log-normalized rate map was then thresholded at a value of 0.8 (0.85 for non-grid, non-border position cells) to identify fields. Firing fields were defined as regions of at least six contiguous spatial bins, and field size was computed as the total area of the spatial bins comprising that field." (SM p. 10). Matching: "we first binned position into a 10 x 10 grid (15 cm bin width). Within each position bin, we then binned speed into 10 bins (15 cm/s bin width) ... this procedure was repeated 50 times for each cell, and the down-sampled rate maps were then averaged" (SM p. 8). The log-normalization is not defined further.

### Non-grid field size (Fig. S10A; new rows MNG33 to MNG35)

Fig. S10A legend (SM pp. 32-33): "In non-grid spatial cells, the number of fields (left), average field size (middle), and the total field coverage (right; fraction of environment covered by fields) did not significantly change between ENV1 and ENV2 (field number: median field number for ENV1 = 2, ENV2 = 2, median difference in fields = 0, signed-rank test p = 0.81; field size: median field size for ENV1 = 1225 cm2, ENV2 = 1106 cm2, median difference in size = 5 cm2, signed-rank test p = 0.35, total field coverage: median field coverage for ENV1 = 0.13, ENV2 = 0.14, median difference in coverage = 0.001, signed-rank test p = 0.84). Further, we did not observe any difference in field properties when considering only cells with high spatial coherence in both environments (spatial coherence > 2.25, n = 100, field number: median field number for ENV1 = 1, ENV2 = 2, median difference in fields = 0, signed-rank test p = 0.37, field size: median field size for ENV1 = 1958 cm2, ENV2 = 1969 cm2, median difference in size = 383 cm2, signed-rank test p = 0.13; total field coverage: median field coverage for ENV1 = 0.15, ENV2 = 0.17".

Main text: "We then investigated how nongrid spatial cells (n = 271 cells) remapped to support reward-localized changes in firing rates. Nongrid spatial cells did not extend their firing fields in a reward-specific manner, as average field size, total field area, and number of fields did not change (fig. S10A)." (p. 1448). So the compared conditions are ENV1 (foraging) and ENV2 (reward task) in the same cells.

Derived widths: 2*sqrt(1225/pi) = 39.49 cm; 2*sqrt(1106/pi) = 37.53 cm; 2*sqrt(1958/pi) = 49.93 cm; 2*sqrt(1969/pi) = 50.07 cm.

### Grid spacing in the same rats (Table S1, Fig. 1)

Table S1 (SM p. 38), "All values are shown as median (25th, 75th quantiles) Paired grid cells (n=102) ... spacing 99.14 (85.85,107.37) 93.07 (82.22,104.77) signed-rank 0.015 ellipticity 1.24 (1.18, 1.44) 1.17 (1.08, 1.25) signed-rank 0.006 mean field size (cm2) 1420 (925, 2013) 1325 (1027, 1863) signed-rank 0.85 autocorr. peak size (cm2) 675 (525, 975) 625 (525, 875) signed-rank p=0.16"; "Experimental grid cells (n=102) vs Control grid cells (n=27) ... spacing ratio 0.95 (0.88, 1.05) 1.01 (0.96, 1.96) rank-sum 0.19". The unit is cm (Fig. 1G axis "grid spacing, ENV1 (cm)"). Table S1 pools cells; it is not per animal. Spacing method: "Grid spacing was defined as the mean of the distances of the six closest fields from the center peak in the autocorrelation." (SM p. 8).

Fig. 1K modules, figure-read from the PDF vector paths (error-bar centres; x ticks at 496.07, 513.02, 529.97 and 546.92 pt for 60 to 120 cm; y ticks at 550.00 to 499.15 pt for 60 to 120 cm; 0.8475 pt per cm). ENV1 module spacings 64.9, 91.4, 102.1, 103.9, 107.6 and 121.6 cm; ENV2 59.4, 84.0, 96.3, 96.4, 99.6 and 127.6 cm. The modules come from "all unique units with a grid score >0.5 (n = 193 cells in ENV1, n = 216 cells in ENV2)" clustered per rat (SM p. 8), six modules in four rats (Fig. 1J,K legend, p. 1448).

Ratios (derived): non-grid width over median spacing 39.49/99.14 = 0.40 (ENV1), 37.53/93.07 = 0.40 (ENV2), 49.93/99.14 = 0.50 (high coherence); against the smallest module 39.49/64.9 = 0.61 and 49.93/64.9 = 0.77; against the largest 39.49/121.6 = 0.32. Grid field width 2*sqrt(1420/pi) = 42.52 cm, 0.43 of the spacing; non-grid over grid area 1225/1420 = 0.86. Mallory 2018 gives 1.55 in mice; 0.40/1.55 = 0.26.

Reading the rat ratio. The 0.85 threshold on a log-normalized map probably cuts fields well inside a 20%-of-peak edge (inference; the scale is not defined). Grid fields found the same way are 0.43 of the spacing, so non-grid fields are about as large as grid fields of the same rats. The grid spacing (median 99 cm, smallest module about 65 cm) is larger than the most dorsal rat modules, which suggests recordings below the dorsal pole (inference).

### Cell counts and remapping (MNG15 confirmed; MNG36 to MNG38)

- "We considered the coding features of 778 cells recorded in both environments (fig. S3)." (p. 1447). Classes: position coding in an LN model, then "Cells were assigned to a specific class (e.g., grid cell) if their score exceeded the shuffled population score threshold in either environment, and if the cell also encoded the relevant variable" (SM pp. 6-7).
- Grid: "Grid cells' (n = 102 cells) firing patterns reorganized"; "there was a small decrease in grid spacing (P = 0.015) (Fig. 1G), but not in field size (P = 0.85), in ENV2"; "Overall, 49 of 102 grid cells showed a statistically significant change on at least one measure" (p. 1447).
- Head direction: "70 of 132 cells exhibiting significant changes in tuning" (p. 1447). Border: "A majority (24 of 36) of border cells remapped between environments, primarily through rotations" (p. 1447); "border cell n = 79" in Fig. S4B.
- Non-grid: "196 of 271 nongrid spatial cells significantly remapped between ENV1 and ENV2, with task-trained animals showing more remapping than free-foraging controls (task-trained mean correlation coefficient ± SD: 0.32 ± 0.22; control: 0.41 ± 0.27; 49 of 100 control cells remapped, proportions test P = 3×10−5)" (p. 1447). Confirms MNG15. The Fig. S7E legend says the opposite about the controls: "In control animals, a large fraction (49/100) of non-grid spatial cells significantly retain their tuning between ENV1 and ENV2" (SM p. 25). Remapping test: "If the across-session correlation coefficients were less than 5th percentile of the distribution, the cell was considered to have significantly re-mapped." (SM p. 10).
- Reward groups: "many cells (n = 159 cells) heterogeneously remapped to preferentially encode the reward location" (p. 1448). Table S3 (SM p. 40): "I 62 39% II 33 20% III 23 14% IV 41 26%"; "distance to closest field, Group I (ENV1, ENV2) 59, 36 signed-rank 2x10-5".
- Decoding: "the decoding error within 30 cm of the reward zone is lower in ENV2 than ENV1 (median difference in error = −4.3 cm, signed-rank P = 0.028)" (Fig. 4J, p. 1451); "median decoding error in ENV1 = 68.1 cm, median decoding error in ENV2 = 68.5 cm" (Fig. S12D, n = 43 sessions).

### Corrections made

MNG15: strain, level_definition, field_rule, n_animals (controls are 3 separate rats), grid spacing (Table S1 values), status, notes (Fig. S10A numbers, Table S1 pooled, S7E conflict).

## 3. Diehl, Hon, Leutgeb & Leutgeb 2017, Neuron 94:83-92

File `diehl2017.pdf`, 34 pages: article pp. 83-92 (PDF pp. 2-11), STAR Methods pp. e1-e6 (PDF pp. 12-17), supplement (PDF pp. 18-34: Figs. S1-S4, Tables S1-S2).

### Animals, arenas, maps, field rule

- "Data from the dorsal medial entorhinal cortex (mEC) were collected from seven adult male, experimentally naive, Long Evans rats (Rats A-G)"; three more rats (H-J) gave CA1 only (p. e1).
- "A 'hyperdrive' consisting of 14 independently movable tetrodes was implanted 0.5-1.0 mm anterior to the transverse sinus and 4.6-5.2 mm lateral to the midline." (p. e1).
- Arenas: "an open field enclosure with flexible black walls that could be shaped either as a square (80 × 80 cm) or as a 16-sided polygon (50 cm radius, referred to as a 'circle')"; "a larger square enclosure (100 × 100 cm), located in a second room" (p. e2). "Prior to any electrophysiological recordings, all enclosures were made highly familiar over at least six training days" (p. e2). Analysis areas: "For the shape-change paradigm, only the area common to both shapes (80 × 80 cm) was used for calculations. ... Since the color-change box was larger, only the central 80 × 80 cm was used for calculations." (p. e4; the second cut applies to the room comparisons).
- Maps: "location bin (5 × 5 cm) ... smoothing with a 5 × 5 bin Gaussian filter with a standard deviation of approximately 1 bin" (p. e3).
- Field rule, stated only for grid cells and place cells: "The minimum peak rate to identify a field was 2 Hz for grid cells and 1 Hz for place cells, with a minimum field size of 250 cm2. ... field bounds were defined on the reference map by building contours iteratively outward until a threshold value of 0.3 times that peak rate was reached." (p. e3).
- Non-grid spatial class: "a two-fold criterion was used in which both the cell's spatial information and its within-session correlation had to pass their respective cutoff scores" (p. e5), with cutoffs from "the distribution of 1000 scores that were obtained from the same cell's shuffled data" (p. e5).

### Dorsoventral position

"tetrode locations on each recording day were assigned to either the deep or superficial layers of mEC and to a dorsoventral distance from the parasubiculum border." (p. e3). No distances are reported. "Along the dorsoventral axis, recording locations of mEC grid and non-grid cells were intermixed with numerous instances of simultaneous recordings of grid and non-grid spatial cells on the same tetrode (69 grid and 71 non-grid spatial cells across 55 recordings). In each rat we also identified the most ventral recording location with an identified grid cell and found that over 85% of the data was recorded from more dorsal regions of mEC." (p. e3). Fig. S1A shows only the most ventral tetrode positions. There are no per-level field sizes.

### Table S1: counts only

Table S1 (PDF p. 30) is "Distribution of mEC cell types across animals" with, per box, "Total number of cells in the superficial layers" and "Number of layer II/ layer III cells" (legend, PDF p. 31). It holds no field sizes; the sweep's pointer was wrong. Totals: shape paradigm 188 cells (83/105), color 227 (120/107), recorded in both rooms 70 (35/35); grid 31, 38, 8; border 18, 18, 4; nongrid spatial 129, 161, 57; nonspatial 8, 8, 1; head direction 2, 2, 0. Per rat (shape): A 32, B 14, C 8, D 31, E 15, F 10, G 78; (color): D 74, E 16, F 61, G 76.

Unique counts (derived, shape + color - both): 345 cells (168 layer II, 177 layer III), grid 61, border 32, non-grid spatial 233 (120 and 113), nonspatial 15, pure HD 4. Then 61/345 = 17.7%, 32/345 = 9.3%, 233/345 = 67.5%, and spatial = 326/345 = 94.5%. The printed percentages, "Of the spatial cells, 17.7 % had significant grid scores and 9.3 % had significant border scores. The remaining 67.5 %" (p. 84), are therefore shares of all 345 cells. As shares of the 326 spatial cells they are 18.7%, 9.8% and 71.5% (derived). Fig. 1A's pie (18%, 68%, 9%, 4%, 1%) is of all 345 cells. The earlier derived "about 64% of all cells" double-counts the 95%.

The manipulation analyses count cells per paradigm. Fig. 3D: "n = 103 cells" (shape), "n = 133 cells" (color), "n = 57 cells" (location); Fig. 2D: "n = 31 cells", "n = 17 cells", "n = 8 cells". 103 + 133 = 236 and 31 + 17 = 48, the numbers of the Fig. 4 legend. Exclusions: "Rat D in shape-change: 0 grid cells and 31 non-grid cells; Rat F in color-change: 21 grid cells and 40 non-grid cells" (p. e6).

### Grid spacing

Not reported anywhere in the main text, STAR Methods or supplement.

### Stability and remapping numbers confirmed

- Grid, matching boxes: "median displacement, 2.4 and 2.6 cm for matching box shape and color, respectively"; different shapes or colours: "median increase, 1.3 and 2.8 cm, p < 0.001 and p < 0.01"; rooms: "median increase, 20.9 cm" (p. 86). Correlations: "median spatial correlation, 0.85 and 0.70, respectively ... but lower across different box shapes or colors (median correlation, 0.72 and 0.42" (p. 88). Rate vectors: "(median Spearman's rank correlation, 0.50 and 0.43" and "(median correlation, 0.40 and 0.13" (pp. 86-88).
- Non-grid, matching boxes: "(median displacement, 4.4 and 4.7 cm, respectively) and that spatial correlations were high (median spatial correlation, 0.62 and 0.62, respectively)" (p. 88). Changes: "a median increase in displacement of 13.1 and 15.3 cm for shape and color change, respectively (both p < 0.001 versus matching)"; rooms "(median increase, 28.9 cm"; correlations "(median spatial correlation, 0.29 and 0.26, respectively)" (p. 88).
- So 0.62 against 0.85 and 0.70, 13.1 and 15.3 cm against 1.3 and 2.8 cm, and 28.9 against 20.9 cm across rooms are all confirmed. The MNG11 quote uses the PMC manuscript's punctuation ("median displacement: 4.4 cm and 4.7 cm respectively"); the numbers are the same.
- Simultaneous ensembles (Table S2, PDF pp. 32-33). The shape-paradigm mean row reads "Mean 1.4 3.5 2.5 5.5 4.9 26.0 0.83 0.64 0.70 0.22", grid then non-grid for each pair of columns: cells per ensemble 1.4 and 3.5; matching displacement 2.5 and 5.5 cm; different-shape displacement 4.9 and 26.0 cm; matching correlation 0.83 and 0.64; different-shape correlation 0.70 and 0.22. Colour: "Mean 1.9 6.3 2.6 7.6 5.5 28.6 0.68 0.61 0.45 0.16". Rooms: "Mean 1.0 3.8 2.8 6.9 24.8 39.2 0.73 0.59 -0.08 -0.02".
- CA1 (Fig. S4 legend, PDF p. 28): place-cell displacement increases of 6.3 cm (shape), 4.9 cm (color) and 36.7 cm (location).
- Diehl 2019 overlap (MNG13, MNG14): rats A to C ran "two training blocks of the box shape-change paradigm ... each day, separated by six hours" (p. e2), which matches the Diehl 2019 design and supports the audit's overlap flag.

### Corrections made

MNG10 to MNG12: environment_size and its numeric columns (the 80 cm cut is not universal), level_definition (layer counts, the 85% statement), n_cells (cell-paradigm counts), MNG10 statistic, value and notes (shares of all cells), grid spacing (not reported). New row MNG39 holds the Table S1 counts.

## 4. Alexander & Nitz 2015, Nat Neurosci 18:1143-1151

Files `nn.4058.pdf` (12 pages: pp. 1143-1151, then Online Methods on PDF pp. 10-12) and `41593_2015_BFnn4058_MOESM28_ESM.pdf` (Supplementary Figs. 1-11, Supplementary Table 1).

### Animals, sites, track

- "Male Long-Evans rats (n = 6) served as behavioral subjects." (Online Methods, PDF p. 10). "microdrives were implanted relative to bregma (A/P −5.8 mm, M/L ±1.3 mm, D/V −0.5 mm, 10–12° medial/lateral angle). Four of the six animals received a unilateral HPC microdrive targeted to the CA1 subregion (target coordinates relative to bregma, A/P −3.8 mm, M/L ±2.3 mm" (PDF p. 10). Fig. 1c sections at bregma -5.28 to -5.88 mm. "Neurons were primarily recorded in the dysgranular, granular c, and granular b subregions of RSC." (p. 1144).
- Track: "Animals were trained to navigate along a 'W'-shaped track for reward ... The track edges were only 2 cm in height ... Total route lengths were 215 cm with turns located at 60 cm, 110 cm and 170 cm." (PDF p. 10). Track locations: "rats were required to traverse the track when set in two distinct positions (here called α and β) in the recording room before the track was returned to the initial position (α′) for a third and final session." (PDF p. 10). Room: "The recording room is 13.5' x 13.5'." (Supplementary Fig. 1). "Recordings lasted approximately 45 min" (PDF p. 10); "the mean number of clean ballistic runs in each was 16" (PDF p. 10).
- Template: "a series of template bins was generated with a spacing of three pixels (~1.2 cm) ... positional data was smoothed using a narrow Gaussian filter with a s.d. of two pixels." (PDF p. 10); "The final template length (200 bins), across all animals, recordings, and routes, was determined by averaging the length of templates across all recordings." (PDF p. 10).
- Inclusion: "We recorded a total of 243 RSC neurons and 109 HPC neurons across the six rats. ... RSC units that did not exhibit peak activation of at least 3 Hz for a single track bin on either route at any track location were excluded from analyses. ... Defined place fields had peak activation greater than or equal to 5 Hz and a minimum length of 20 adjacent template bins (approximately 24 cm total)." (PDF p. 10).

### Field widths, field counts, spatial shares

- No RSC field widths or field counts: "there was little evidence for single, isolated firing fields of the type typically observed for HPC neurons (also known as place cells)" (p. 1147); "RSC firing fields rarely manifested as single, isolated 'place-specific' firing fields against a background of near-zero out-of-field firing" (p. 1149). No CA1 widths either; the CA1 minimum is about 24 cm by rule.
- Shares (row alexander2015_1): 22% turn-bias ratio above 2 and 45.6% (104/228) turn-sensitive (p. 1145); 54.3% (124/228) not turn-sensitive, of which 52.4% (65/124) odd vs even r > 0.4 and 21.7% (27/124) alpha vs beta r > 0.4, chance "4.32% (±1.7% s.d.)" (p. 1145); "43.3% (n = 45 neurons, or 19.7% of all neurons" route-modulated turn cells and "a separate 26% of all neurons" pure turn cells (p. 1146). Supplementary Fig. 5: "100% of RSC neurons had significantly greater actual correlation distributions than both of the 'chance' correlation distributions".
- Head direction (alexander2015_3): "allocentrically-referenced RSC head-direction neurons (6%, n=15/243" (Supplementary Fig. 9).
- Allocentric sensitivity (alexander2015_2): "(µRSC = 0.16, nRSC = 374 (both routes); µHPC = 0.07, nHPC = 218" (p. 1146); "(µ = 0.34" for alpha vs alpha-prime and "(µ = 0.40" for HPC (p. 1147); Fig. 4b panels "α′ > β = 137/187" and "134/187".
- CA1 turn-biased cells: "(n = 39)", CV "(µ±SD = 1.50 ± 0.53)" against RSC "0.52 ± 0.33" (Supplementary Fig. 4).

### Position reconstruction error

- Text: "(n = 200 spatial positions, µ error RSC α/β = 51.32 cm; µ error HPC α/β = 63.45 cm)"; "(n = 200 spatial positions, µ error RSC α/α′ = 4.51 cm; µ error HPC α/α′ = 7.82 cm)" (p. 1148). Fig. 6 legend: "(αα′ error group, bars 7 and 8), which did not differ from the within position error group (bars 7 and 8, µ error LRL = 4.32 cm, µ error RLR = 4.70 cm)".
- Odd vs even (within-position) errors, figure-read from the Fig. 6b vector bars (bar baseline at y = 349.82 pt taken as 0; tick labels 20, 80 and 140 cm give about 0.50 pt per cm): RSC 4.2, 3.8, 3.8 and 4.0 cm (alpha LRL, alpha RLR, beta LRL, beta RLR); CA1 6.0, 6.6, 5.2 and 5.4 cm. Check against the text: the CA1 alpha/alpha-prime bars read 6.8 and 8.8 cm, mean 7.8 (text 7.82); the RSC alpha/beta bars 76.4 and 29.4 cm, mean 52.9 (text 51.32).

### Kernels (rows alexander2015_k1 to _k4)

Method. Fig. 3e is a 409 x 343 px raster (xref 30) with four 192 x 152 px matrices and its colour bar inside the raster; the vector labels "0.92" and "0.34" sit at its ends (label centres about 2 pt inside). Fig. 6a is a 629 x 501 px raster (xref 90) with RSC (bar 0.84 to 0.27) and CA1 (bar 0.78 to -0.08) matrices; the labels sit at the bar ends. Distance scale: the white turn markers sit at template fractions 0.26, 0.50 and 0.79 in every matrix; turns 1 and 3 are 110 cm apart, so 1.04 cm per bin (215 cm over 200 bins would give 1.075). The Fig. 6 legend names Pearson correlation; Fig. 3e does not name the measure. No per-cell normalization is described.

Text: "As indicated by the high correlations along these diagonals, the even-trial ensemble activity at all route positions was most strongly correlated with odd-trial ensemble activity for the same or nearby positions. Although track positions sharing the same behaviors (for example, right or left turning) yielded high correlations as well (appearing as off-diagonal red patches), the patterns of activation were distinct enough to enable a very accurate prediction of the animal's location" (p. 1146). Cells: Fig. 3e legend "(n = 228)"; Online Methods "Only RSC neurons that were recorded when the animal ran all sessions (α, β, α′) were included (n = 187 of 228)" (PDF p. 11).

Mean r by separation (figure-read, colour-bar lookup):

| cm | 0 | 5 | 10 | 15 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RSC odd/even, mean of four (Fig. 3e) | 0.84 | 0.84 | 0.79 | 0.75 | 0.71 | 0.63 | 0.59 | 0.57 | 0.56 | 0.56 | 0.58 | 0.62 | 0.65 | 0.66 | 0.64 | 0.58 | 0.58 | 0.57 | 0.54 |
| RSC alpha vs alpha-prime (Fig. 6a) | 0.77 | 0.76 | 0.73 | 0.70 | 0.66 | 0.58 | 0.55 | 0.53 | 0.51 | 0.51 | 0.52 | 0.55 | 0.58 | 0.57 | 0.55 | 0.51 | 0.52 | 0.49 | 0.43 |
| CA1 alpha vs alpha-prime (Fig. 6a) | 0.72 | 0.72 | 0.70 | 0.67 | 0.60 | 0.45 | 0.34 | 0.24 | 0.17 | 0.14 | 0.11 | 0.09 | 0.09 | 0.10 | 0.10 | 0.11 | 0.14 | 0.12 | 0.11 |
| RSC alpha vs beta (Fig. 6a) | 0.59 | 0.60 | 0.60 | 0.59 | 0.57 | 0.56 | 0.55 | 0.55 | 0.54 | 0.54 | 0.55 | 0.58 | 0.59 | 0.60 | 0.60 | 0.60 | 0.57 | 0.51 | 0.50 |
| CA1 alpha vs beta (Fig. 6a) | 0.13 | 0.14 | 0.13 | 0.14 | 0.14 | 0.15 | 0.14 | 0.14 | 0.14 | 0.14 | 0.13 | 0.12 | 0.12 | 0.11 | 0.11 | 0.12 | 0.16 | 0.19 | 0.12 |

The 0 cm entries are low where the dashed diagonal and the reconstruction line mask most diagonal pixels; the peaks are 0.86 (Fig. 3e), 0.77 and 0.72. Summary:

- RSC odd vs even: never falls to 0.5 (minimum 0.535 at 206 cm); first minimum 0.56 at 64 cm; half of the drop to it (r 0.71) at 19.7 cm (19.1 to 20.4 cm per matrix); a second peak of 0.66 at 110 cm, the spacing between same-direction turns (60 and 170 cm). Under the tick-centre reading of the bar labels, values become 0.955 - 1.122 x (0.92 - r) and the curve still stays above 0.52; distances do not change.
- RSC alpha vs alpha-prime: peak 0.77; minimum 0.51 at 71 cm; half-drop (0.64) at 22.3 cm; second peak 0.58 at 101 cm; r = 0.5 only at the far ends (177 cm).
- CA1 alpha vs alpha-prime: peak 0.72; r = 0.5 at 26.5 cm, full width 53 cm (derived); minimum 0.09 at 93 cm; half-drop (0.40) at 33.6 cm; no rise at 110 cm.
- Across track locations (alpha vs beta) the RSC diagonal is 0.61 against a floor of 0.55, and CA1 is flat near 0.13.
- The RSC floor probably reflects broad continuous firing in vectors with no stated per-cell normalization (inference); the CA1 floor is near 0.1.

## 5. Alexander & Nitz 2017, Curr Biol 27:1551-1560

File `alex2017.pdf`, 20 pages: highlights page (PDF p. 1), article pp. 1551-1560 (PDF pp. 2-11), STAR Methods pp. e1-e4 (PDF pp. 12-15), supplement (PDF pp. 16-20: Figs. S1-S4).

### Animals, sites, tracks

- "Male Long-Evans rats (n = 5) served as behavioral subjects"; "2 of the 5 rats also performed track running on a W-shaped track that has previously been published [5]" (p. e1).
- "microdrives were implanted relative to bregma (A/P −5.8 mm, M/L ± 0.7-1.2 mm, D/V −0.5 mm, 10-12° medial/lateral angle). 3 animals received a unilateral HPC microdrive targeted to the CA1 sub-region" (p. e2). Fig. 1A: "Placement of electrode wires across all rats (n = 5) in caudal RSC"; section levels -5.52, -5.76 and -6.36 mm. "recorded in all RSC sub-regions" (p. 1552).
- Plus track: "The track edges were 1 cm in height, which allowed the animal an unobstructed view of the full environment which had fixed distal cues across recording days. Animals ran the plus track in the clockwise direction." (p. e1). Fig. 1B schematic labels one segment "55cm"; 12 segments (p. e3: "there were 12 total segments on the plus"), so a lap is 12 x 55 = 660 cm (derived). A quarter is 165 cm and a half 330 cm (derived). The tracking in Fig. 1B shows rounded corners, so the path is somewhat shorter.
- Ring: Fig. S4F schematic labels the diameter "122 cm"; lap pi x 122 = 383 cm (derived). "The ring track was centered on the same allocentric position as the plus-shaped track ... Track edges on the ring were 1 cm in height." (p. e1). "70% of recordings had plus-track session first" (p. e1).
- Template and smoothing: "a series of template bins was generated with a spacing of 10 pixels (approx. 3.5 cm)"; "Activity patterns were smoothed with a narrow Gaussian filter (7 cm s.d.)." (pp. e2-e3). "The average number of traversals on the plus and ring tracks were 14 and 13, respectively." (p. e2). "Recordings lasted approximately 20-45 min" (p. e2).
- Inclusion: "Three hundred fifteen neurons from five adult male rats were recorded ... Of these, 229 exhibited greater than 3 Hz firing for at least one track position and were not head-direction (HD) neurons" (p. 1552). CA1: "HPC neurons were excluded if they had mean activity across all track sessions greater than 8 Hz ..., no bins in which the firing rate dropped to 0 Hz ..., and no bins exhibiting a peak firing rate amplitude above 3 Hz." (p. e2).
- Correlation type: "Correlation matrices were computed using Spearman's correlations." (p. e4).

### Rows alexander2017_1 and _2 verified (were unverifiable)

- "63.7% of RSC neurons (n = 146/229) exhibited periodicity in firing according to at least one full-route or sub-route space. Specifically, 47.2% (n = 108/229) were modulated by sub-routes ... the most prominent encoding fell at quarters (21%; n = 48/229) and halves (20.5%; n = 47/229). 38.9% (n = 89/229) of RSC cells had an activation pattern that cycled a single time across the full space of the trajectory." (p. 1554); "48.3% of full neurons (n = 43/89) were conjunctively sensitive to one or more sub-routes. Of all neurons, 26.6% (n = 61/229) were conjunctively sensitive to multiple spatial scales simultaneously" (p. 1554).
- Ring: "86.7% (n = 78/90) of RSC neurons were significantly modulated by at least one oscillating spatial predictor" (p. 1555); "the frequency of quarter sub-route encoding (13.3%) was diminished compared to the plus track (21%" and "the scale of half of the route was higher for the ring (44.4%) than the plus (20.5%" and "70% (n = 63/90) of RSC neurons exhibited periodic patterns on the track that oscillated at the scale of the full ring traversal" (p. 1556). The doubted 70% is confirmed. Ring reconstruction: "reconstruction error = 9.99 ± 9.41cm; µ ± SEM" (p. 1555).
- Movement: "Only 5.2% (n = 12/229) and 8.7% (n = 20/229) of RSC neurons were significantly impacted by linear acceleration and velocity, respectively. 13.5% (n = 31/229) ... angular velocity"; "just 43.7% (n = 21/48) of neurons that exhibited quarter-scale periodicity also exhibited significant sensitivity to any of the movement variables" (p. 1554).
- No field widths or field counts are reported. A periodic cell has 1, 2 or 4 activity cycles per lap by the definition of its predictor.

### Position and distance reconstruction (verified)

- "Such 'reconstruction' of the animal's progression along the route was accurate to 1.9 ± 2.7 cm based on RSC populations, a value comparing favorably to the accuracy seen for 141 hippocampal CA1 'place cells' recorded in three animals (Figures 1C and 1D; CA1 accuracy = 5.2 ± 6.7 cm mean ± SEM)." (p. 1552). The earlier sweep's 1.9 against 5.2 cm is confirmed.
- "Mean distance reconstruction error for RSC was 2.93 cm, significantly lower than that observed for HPC (µHPC = 12.61 cm; Wilcoxon rank-sum test; n = 72" (p. 1556), from Fig. 4 (not Fig. 6, and not 2.09 cm as the earlier kernel note said, which came from the preprint summary).
- The kernel quote "significant correlation peaks were observed for analogous route positions with spacing at the scale of route quarter sections" is not in the journal text. The journal says: "Unlike matrices for the CA1 ensemble, off-diagonal correlations for RSC ensembles exhibited repeating patterns across the track, indicating the presence of a sub-population of RSC neurons having repeating patterns over quarter and/or half sections of the full route." (pp. 1552-1553) and "although strong population correlations were observed between analogous positions of route sub-spaces, the strongest ensemble correlation values across odd versus even trials occurred at similar full-route positions" (pp. 1557-1558).
- Half-route test (Fig. S2, row alexander2017_4): "110 RSC neurons were recorded under these conditions"; "µOverlap = 0.53, µFirstHalf = 0.09, µSecondHalf = 0.07, p = 9.56 x 10-10".

### Kernels (rows alexander2017_k1 corrected, _k2 and _k3 added)

Method. Fig. 1 is one 1339 x 1617 px raster (xref 64). The RSC (Fig. 1C) and CA1 (Fig. 1D) matrices are 512 x 512 px, rows 470-982 and 1067-1579. Their Spearman colour bars sit in the raster (rows 469-794 and 1065-1391), with tick labels "1" and "0.5" centred 20 rows inside the top and 12 rows inside the bottom. Two calibrations were run: bar ends = 1 and 0.5, or tick-label centres = 1 and 0.5 (bar then spans 0.48 to 1.03). Black turn lines and the white peak line were masked (15% of pixels). 660 cm per side. The CA1 matrix saturates at the bottom of the bar beyond about 85 cm, so medians and the share of clipped pixels were used near r = 0.5.

Mean r by linear separation (figure-read):

| cm | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 165 | 180 | 200 | 250 | 300 | 330 | 365 | 400 | 450 | 495 | 550 | 600 | 640 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RSC (bar ends) | 0.91 | 0.89 | 0.86 | 0.82 | 0.80 | 0.78 | 0.78 | 0.78 | 0.79 | 0.79 | 0.79 | 0.75 | 0.74 | 0.76 | 0.78 | 0.77 | 0.73 | 0.72 | 0.76 | 0.77 | 0.75 |
| RSC (tick centres) | 0.93 | 0.92 | 0.87 | 0.83 | 0.81 | 0.80 | 0.79 | 0.79 | 0.80 | 0.80 | 0.80 | 0.76 | 0.75 | 0.77 | 0.79 | 0.77 | 0.73 | 0.72 | 0.76 | 0.78 | 0.76 |
| CA1 (bar ends) | 0.81 | 0.79 | 0.68 | 0.55 | 0.51 | 0.50 | 0.50 | clipped from here | | | | | | | | | | | | | |
| CA1 (tick centres) | 0.82 | 0.79 | 0.67 | 0.54 | 0.49 | 0.48 | 0.48 | clipped from here | | | | | | | | | | | | | |

Summary:

- RSC: peak 0.91 to 0.93; never below 0.71; floor about 0.77 (100 to 300 cm); half of the drop to the floor (about 0.84) at 45 cm; a dip to 0.78 at about 120 cm and a local maximum of 0.79 at 165 to 182 cm (one quarter lap); a dip to 0.74 near 280 to 290 cm and a second rise to 0.78 at 360 to 385 cm. The start and end of the lap are not similar (0.72 near 490 cm), so the matrix does not wrap around the closed loop.
- CA1: peak 0.81 to 0.82; r = 0.7 at 34.6 cm and 0.6 at 47.8 to 49.6 cm; median r = 0.5 at 59.9 cm (tick centres) or half the pixels clipped at 63.0 cm (bar ends); full width about 120 to 126 cm (derived). No off-diagonal repeats.
- Ring (Fig. S4H, 146 x 146 px, bar 1 to 0.7, n = 90 per the legend though the panel says "n =229"): peak about 0.92; half of the drop to the floor (0.84) at about 38 cm; 0.85 by 60 cm and 0.84 out to 200 cm; blocks by quarter, with the first two quarters correlated with each other.

Caveats: the vectors pool cells across rats; Spearman correlation on vectors with many silent CA1 cells and the 7 cm smoothing may widen the CA1 curve (inference); distances assume 55 cm segments and may be up to about 10% high because rats cut corners.

### Corrections made

alexander2017_1 and _2: citation, DOI, strain, level_definition, environment_size and length (from Figs. 1B and S4F), familiarity, location, status (now FT), verified_in, notes. alexander2017_k1: citation, DOI, environment, measure (Spearman), normalization, pooling, n_cells (229), widths (not reached), floor, far_behaviour, location, quote (journal text), status, notes (Fig. 4 and 2.93 cm).

## What is still missing

- Rat RSC field widths. Neither Alexander & Nitz paper defines RSC fields. The rat RSC width remains Smith, Barredo & Mizumori 2012 only.
- Rat RSC kernels with per-cell normalization. Both rat curves use vectors that, as far as stated, are not normalized per cell, so their floors cannot be compared with Mao 2017's 0-to-1 scaled curve.
- A non-grid MEC field size measured with the same rule in rats and mice. Butler's threshold (0.85 on an undefined log-normalized scale) and Mallory's unstated rule make the 0.40 against 1.55 contrast partly a method contrast.
- Dorsoventral positions. Butler gives none; Diehl gives only the 85% statement; Mallory gives grid-cell positions only.
- Diehl 2017 non-grid field size: not in the paper or its supplement.
- Mallory's Supplementary Fig. 11 gives a mouse CA1 population width at r = 0.2 on a 1 m track (28.8 ± 8.1 cm); a kernel row would need the curve itself (Supp. Fig. 11b), which was not read.
