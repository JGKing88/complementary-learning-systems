# Audit of the PV similarity versus distance table

This file checks every cell of the 16-row table that summarizes population-vector (PV) similarity against spatial distance in real recordings. Each check gives a short verbatim quote and its location. The report itself is unchanged.

Verdicts use three words. "Confirmed" means the source supports the cell as written. "Corrected" means the cell is wrong or imprecise, and replacement text follows. "Not verifiable" means the source text available to this audit does not settle the point, and the reason is given.

## Sources and access

| Row | Source read | Access |
|---|---|---|
| 1 Green 2022 | Europe PMC XML (PMC9245695), SI Appendix PDF and Fig. 3 image from the Europe PMC supplementary files | full text and SI |
| 2 Wang 2020 | mmc2.pdf (33 pages: article, STAR Methods, supplemental figures) | full text |
| 3 Battaglia 2004 | PMC6729394 article page, read with WebFetch | full text, quotes as fragments, figures not seen |
| 4 Mao 2017 | Europe PMC XML (PMC5557927), Fig. 3 image from the supplementary files | full text |
| 5 Rubin 2015 | Europe PMC XML (PMC4749549), with the eLife decision letter and author response | full text |
| 6 Ledergerber 2021 | ledergerber_2021.txt (article only); Document S1 with Fig. S7 not available | full text, supplement missing |
| 7 Tanni 2022 | Europe PMC XML (PMC9616721) | full text |
| 8-10 Kjelstrup 2008 | science.1157086.pdf; SOM not read | full text |
| 11 Maurer 2005 | PubMed abstract and MeSH terms (PMID 16145692); Wiley full text returned HTTP 403 | abstract only |
| 12 Terrazas 2005 | PMC6725460 article page, read with WebFetch | full text, quotes as fragments, figures not seen |
| 13 Nielson 2015 | PMC4568259 article page, read with WebFetch | full text, quotes as fragments |
| 14 Deuker 2016 | deuker_2016.txt | full text |
| 15 Maimon 2026 | Europe PMC XML (PMC13421320), with Extended Data legends | full text |
| 16 Rich 2014 | Main text (its.caltech.edu/~jkenny/nb250c/papers/Rich-2014.pdf) and the supplement PDF | full text |
| 16 Eliav 2021 | Ulanovsky lab PDF, main text plus supplementary materials (83 pages) | full text |
| 16 Harland 2021 | Accepted manuscript, par.nsf.gov/servlets/purl/10294326 | full text (accepted manuscript) |

Europe PMC returned HTTP 500 twice for PMC6729394, PMC6725460 and PMC4568259. NCBI efetch returned only front matter for these three ("The publisher of this article does not allow downloading of the full text in XML form"). The J Neurosci and PNAS PDF links returned HTTP 403. The PMC article pages did load through WebFetch, which returns quotes of under 125 characters chosen by a summarizing model. Those rows are marked "(WebFetch)". Their quotes are less certain than quotes copied from a PDF or XML file.

## How figure values were read

Four rows depend on numbers read from figures for this audit. A row that adopts them should tag them "figure-read".

- Kjelstrup 2008, Fig. 3 line graphs. The curves are vector paths in the PDF. I extracted their coordinates and calibrated the axes on the tick marks (r = 1, 0.5, 0; distance -16 to 16 m). The horizontal bars in the figure match the text widths to within 3 to 5%, which sets the distance precision.
- Green 2022, Fig. 3C. Pixel trace of the red curve in the published 764 x 791 px JPEG, calibrated on ticks at 0 to 150 cm and r = 1 to -0.5. One pixel is about 1.2 cm.
- Mao 2017, Fig. 3d. Pixel trace of the 675 x 182 px JPEG, calibrated on ticks at 0, 75 and 150 cm and r = 1, 0.5, 0. One pixel is about 1.4 cm. The black RSC curve lies under the red CA1 curve near the peak, so only the CA1 crossing is read.
- Wang 2020, Fig. 1D. The figure is one embedded raster (1744 x 2500 px). Each matrix spans 226 px for 360°, so one pixel is 1.6°. I mapped each pixel to the nearest colour in the colour bar (0 to 1) and averaged by circular lag in 5° bins. The colour bar has no colour below 0, so negative values read as about 0, and the far floor of 0.03 to 0.05 is an upper bound.

## Row-by-row audit

### Row 1. Green et al. 2022

- **Region, "Rat dorsal CA1".** Confirmed. "Each animal was implanted with silicon probes above the dorsal hippocampal CA1 pyramidal layer." (Results, first section). "Four adult Long-Evans rats, 2 males and 2 females, were used in this study." (SI Appendix p. 2). Fig. 3C is CA1. The CA3 cells (n = 275) appear only in Fig. 3L-O.
- **Space, "1.8 m track".** Confirmed for the apparatus. "We trained rats to walk back and forth on a linear track (1.8 m) for water reward, available at a platform at each end of the track" (Results). The PV matrix covers less. Its position axis runs from 0 to about 160 cm (Fig. 3B, measured on the tick marks), and the SI averages "values at distances of 0-150 cm" (SI Appendix p. 3).
- **PV construction.** Within one condition (control trials), pooled across sessions. "The population vector correlations were computed by first z-scoring the trial-averaged firing rate for a given position of each neuron ... and then calculating the correlation across neurons between their activities at each combination of position bins (Fig. 3B)." (Results). "The firing rate of each neuron was computed in 2 cm bins and normalized by time occupancy." (SI Appendix p. 3). No smoothing is mentioned. The cell threshold is inconsistent. The Fig. 3 legend includes "only neurons with a mean firing rate >0.1 Hz in at least one condition ... (n = 209)", while the SI says "Neurons with an average firing rate of less than 0.01 Hz were excluded." Neither source says whether the two running directions were pooled.
- **Similarity near zero, "30 to 40 cm, 17 to 22%".** Corrected. The cell merges an upper bound with an undefined scale. The text says "Average population vector correlation decreased to near zero within <40 cm of separation (Fig. 3C)" (Results), and the legend says "Neural activity shows a spatial scale of ∼30 cm." (Fig. 3C legend). Read from Fig. 3C, r is 0.17 at 25 cm and 0.04 at 30 cm, and it crosses zero at about 32 cm. It does not stay near zero. It reaches -0.12 at 40 cm, about -0.30 from 70 to 120 cm (minimum -0.32 near 100 cm) and -0.14 at 150 cm. This is a measured result.
- **Arithmetic.** Confirmed as stated (30/180 = 16.7%, 40/180 = 22.2%). The zero crossing gives 32/180 = 18%. Against the 160 cm the matrix covers, the shares would be 19 to 25%. Rows 8 to 10 divide by the analysed length, so the table mixes denominators.
- **Pattern, "Local".** Confirmed, with an omission. The decline is local, but the tail is negative. Each neuron is z-scored across positions, so its rates sum to zero over the track, which pulls the mean off-diagonal correlation below zero. That explanation is my reading of the construction, not the authors'.
- **Status, "full text".** Confirmed. This audit also read the SI Appendix.

| Cell | Corrected text |
|---|---|
| Similarity near zero at | Crosses zero at about 32 cm, 18% (figure-read; text "near zero within <40 cm", legend "spatial scale of ∼30 cm"), then negative, about -0.3 from 70 to 120 cm |
| Pattern | Local, with a negative tail (z-scored rates) |
| Status | full text and SI, figure-read |

### Row 2. Wang, Monaco & Knierim 2020

- **Region, "Rat CA1 and CA3".** Corrected (imprecise). The recordings are dorsal. "The edges of place fields recorded from dorsal CA1 and CA3 concentrated near the boundaries" (Introduction, p. 1397). "the current work recorded only from the dorsal hippocampus" (Discussion, p. 1406).
- **Space, "76 cm circular track".** Confirmed. "Rats were trained to run clockwise on a circular track (76 cm O.D., 10 cm wide)" (STAR Methods, Double rotation task, p. e2). The centerline diameter is 66 cm, so the centerline is about 207 cm long and 1° is about 0.58 cm.
- **PV construction.** Within one session type. The matrices do not compare STD sessions with MIS sessions. "Vertically stacking the transposed firing-rate vectors formed an N x 360 population firing rate matrix" and "The (ith,jth) element of the cross-correlogram was the correlation between the ith and jth column of the population firing rate matrix" (STAR Methods, Cross-correlograms, p. e3). Each of the four matrices (CA1-STD, CA1-MIS, CA3-STD, CA3-MIS) comes from its own session type. Rates use 1° bins, "circularly smoothed with a Gaussian kernel with standard deviation 4.3°" (Place field detection, p. e3), and each unit's vector was "divided by their maximum firing rates before being stacked together" (Cross-correlograms, p. e3). Only place cells enter, pooled across rats, one cell-session per unit per session type. "When the same types of sessions were presented in the same day, we only included the cell-session with the highest within-field mean firing rate of the day" (Single-unit isolation, p. e2).
- **Similarity near zero, "Not measured; authors expect one field, median fields 17% (CA1) and 20% (CA3)".** Corrected (imprecise). The decay is measured as a matrix (Fig. 1D), but the text never states its width. The cell correctly labels the authors' expectation as reasoning. "if two locations are not farther than the average size of place fields (i.e., the spatial scale factor), the distance between them would be negatively correlated with the similarities between their neural representations" and "When two locations are sufficiently far apart, such that there is no longer any overlap in the population of place cells encoding the locations, the representations are maximally decorrelated with no further relationship to longer distances." (Discussion, p. 1406). The field sizes are as stated. "The median place field size was 60° with interquartile range (IQR) 38° for CA1 fields, and was 73° with IQR 62° for CA3 fields" (STAR Methods, Place field detection, p. e3). Read through the colour bar, Fig. 1D falls to r = 0.5 at about 24° (CA1-STD) and 34° (CA3-STD). It falls to r = 0.1 at about 64° (CA1-STD), 69° (CA1-MIS), 99° (CA3-STD) and 98° (CA3-MIS). Beyond about 130° it reads 0.03 to 0.05, the bottom of the colour scale.
- **Arithmetic.** Confirmed. 60/360 = 16.7% and 73/360 = 20.3%. For the colour read, 64/360 = 18% (about 37 cm on the centerline) and 99/360 = 28% (about 57 cm).
- **Pattern, "Local, narrower at texture boundaries, especially CA3".** Confirmed. "The diagonal bands became narrower near the locations of the local-cue boundaries (especially in CA3), indicating a more rapid change in the population activity near the boundaries." (Results, p. 1399). Averaged over the loop, the CA3 band is wider than the CA1 band (colour read above), as the larger CA3 fields predict.
- **Status, "full text, derived".** Confirmed. mmc2.pdf holds the article, STAR Methods and supplemental figures.

| Cell | Corrected text |
|---|---|
| Region | Rat dorsal CA1 and CA3 |
| Similarity near zero at | Not stated in text. Fig. 1D within-session matrices fall to r = 0.1 at about 64° (CA1) and 99° (CA3), 18% and 28% of the loop (figure-read). Authors expect decorrelation beyond one field; median fields 60° (17%, CA1) and 73° (20%, CA3) |
| Status | full text; field shares derived; Fig. 1D figure-read |

### Row 3. Battaglia, Sutherland & McNaughton 2004 (WebFetch)

- **Region, "Rat dorsal CA1".** Confirmed. "over the right dorsal hippocampus" and "stabilize above the CA1 hippocampal subregion" (Methods).
- **Space, "1.8 m tracks".** Confirmed, but the cell hides the design. Experiment B used one track with two halves. "For the recordings, a blue-painted wooden linear track (180 × 8 cm) was used." and "The other half was left unadorned" (Methods). Experiment C used two tracks "placed in two adjacent recording rooms, each equipped with a recording setup." (Methods). Experiment B had four rats ("four rats were pretrained to shuttle back and forth in an enclosed alley for food reward") and experiment C had two ("two rats were food-deprived and trained to shuttle back and forth").
- **PV construction.** Within one session, place-field profiles normalized to their peak. "for the spatially selective cells, normalizing them so that the profile peak value was 1" and "segmented into bins corresponding to 4.5 cm on the tracks" (Methods). "recordings were performed on both the cue-rich and the cue-poor track during a single session" (Methods). The fetched text describes no smoothing. The Fig. 6 matrices hold same-direction and between-direction quadrants. "The top left quadrants represent the correlations between population vectors in the rightbound journey" (Fig. 6 legend).
- **Similarity near zero, "Narrower on a cue-rich track".** Confirmed for experiment C, with qualifications. The measure is "the distance at which the population vectors representing two locations become uncorrelated" (Results). "this correlation distance is significantly smaller on the cue-rich track" (Results), and "The correlation on the cue-poor track decayed significantly more slowly than on the rich track" (Fig. 7 legend). In experiment B, "Population vector correlation tended to decay more slowly with distance on the cue-poor track" "but the effect did not reach significance." (Fig. 7 legend). The authors attribute part of the difference to sparseness. "This is partly a consequence of the sparser population vectors." (Results). The text gives no distance in cm.
- **Pattern, "Local".** Not verifiable. The text gives no width, and the PMC figure files sit behind the bot check, so I could not view Fig. 7.
- **Status, "full text, figure only".** Confirmed. The value exists only in Fig. 7.

| Cell | Corrected text |
|---|---|
| Space | 1.8 m tracks; one track with cue-rich and cue-poor halves (Exp. B), two tracks in adjacent rooms (Exp. C) |
| Similarity near zero at | No distance in text; decorrelation distance significantly smaller on the cue-rich track in Exp. C (2 rats), non-significant trend in Exp. B (4 rats); partly attributed to sparser PVs |

### Row 4. Mao et al. 2017

- **Region, "Mouse retrosplenial cortex".** Confirmed. "Combining cellular Ca2+ imaging in RSC of mice with a head-fixed locomotion assay, we identified a population of RSC neurons, located predominantly in superficial layers" (Abstract). Fig. 3d uses "Data from WT mice with AAV1-hSyn-GCaMP6m injections." (Fig. 3 legend).
- **Space, "1.5 m belt".** Confirmed. "Mice moved a 150-cm linear treadmill (top) with tactile cues on its surface" (Fig. 1a legend). The belt is a loop. "Note that the periodicity occurred because of the periodicity of the track." (Fig. 3d legend).
- **PV construction.** Within one session, place cells only, each cell scaled 0 to 1. "Activity in the position maps was normalized so that the activity of each cell ranged between 0 and 1. For each session, a population vector was constructed using the occupancy-normalized trial-averaged activity of all place cells at each position bin." (Methods, Population vector analysis). "The 150-cm track was divided into 100 position intervals (bins)" and "Data were filtered with a Gaussian smoothing window with 4.5 cm s.d. width" (Methods, place-field section). The Methods wording "between each pair of columns ... in two population vector matrices" also serves the light-versus-dark comparison of Fig. 5b. For Fig. 1h the matrix is "for data shown in g", which is one session.
- **Similarity near zero, "About 35 cm, 23%".** Confirmed. "the correlation between population vectors steeply falls to zero over a short distance (~35 cm)" (Discussion). The authors summarize measured curves (Figs. 3d and 4b). The CA1 curve crosses zero at about 34 cm in my pixel read. Both curves then go negative, CA1 to about -0.26 between 70 and 90 cm and RSC to about -0.2, before the loop brings them back up near 150 cm.
- **Arithmetic.** Confirmed. 35/150 = 23%. On a 150 cm loop the largest separation is 75 cm, and 35/75 = 47%.
- **Pattern, "Local".** Confirmed.
- **Status, "full text".** Confirmed.

| Cell | Corrected text (optional addition) |
|---|---|
| Similarity near zero at | About 35 cm, 23% of the loop; then negative, about -0.2 to -0.26 at 70 to 90 cm (figure-read) |

### Row 5. Rubin et al. 2015

- **Region, "Mouse CA1".** Confirmed. "We performed time-lapse imaging of thousands of neurons over weeks in the hippocampal CA1 of mice" (Abstract).
- **Space, "96 cm track".** Corrected. The value compares two tracks. "Environment A was a straight 96 cm long track and Environment B was an L-shaped track consisting of two 48 cm long arms." (Materials and methods).
- **PV construction.** Between environments at matched positions. This is the main error in the row. "To determine the level of similarity between representations of the different environments, we calculated the mean population vector correlation between them (Leutgeb et al., 2005)." and "We computed the correlation between the population vector in one environment with that of the matching location in the other environment, and averaged the scores over all positions." (Materials and methods, Population vector correlation). Bins are 4 cm, "excluding the last 2 bins at both ends of the tracks". The Results describe Fig. 1G the same way, as "the correlations between the place field maps of the two environments (Figure 1G)".
- **Similarity near zero, '"Low" between locations, no curve'.** Corrected. The word "low" refers to the two tracks, not to separated locations on one track. "(G) Average correlations of place cell firing patterns (i.e. the 'population vector', PV) during running epochs at different locations along the two linear tracks were low and did not change over time (One-way repeated-measures ANOVA; F<1, p = 0.44)." (Fig. 1 legend). The paper has no within-environment PV-versus-distance measure.
- **Pattern, "Local".** Corrected. There is no within-environment data to classify.
- **Status, "full text".** Confirmed.

This row does not belong in a within-environment kernel table. If it stays, it should be marked as a between-environment contrast.

| Cell | Corrected text |
|---|---|
| Space | Two tracks, straight 96 cm and L-shaped 2 x 48 cm, compared with each other |
| Similarity near zero at | Not measured within an environment. Fig. 1G correlates PVs at matching positions between the two tracks; these were low and stable over days (F<1, p = 0.44) |
| Pattern | No within-environment data (remapping measure) |

### Row 6. Ledergerber et al. 2021

- **Region, "Rat subiculum and CA1".** Confirmed, and "dorsal" can be added. "All tetrodes were placed in the dorsal one-third of each region" and "We isolated 760 putative principal neurons from recordings in 8 rats ... 421 were located in SUB and 325 in CA1" (Results, p. 2).
- **Space, "1.5 m box".** Confirmed. "The arena was a 150 × 150 cm square box" (STAR Methods, p. e2). Two tasks used this box, open-field foraging (5 of 8 rats) and a spatial memory task (all 8 rats). The text does not say which one Fig. S7H uses.
- **PV construction.** Not verifiable. The main text and STAR Methods do not describe the PV autocorrelation. Rate maps in general used "a 5 × 5 cm square grid" and "a Gaussian kernel with σ = 1.5 bins" (STAR Methods, Tracking, rate maps and tuning curves, p. e3). "Data from multiple sessions of the same type was concatenated for analysis purposes." (p. e2).
- **Similarity near zero, "Subiculum narrower than CA1".** Confirmed, qualitative only. "Plotting the average population vector autocorrelation as a function of distance between spatial bins (Figure S7H) shows that the peak in SUB is narrower than that in CA1." (Results, p. 11). The text gives no width and no statistic.
- **Pattern, "Local, narrower".** "Narrower" is confirmed. "Local" is not verifiable without the figure.
- **Status, "full text, figure only".** Corrected. The article was read, but Fig. S7H is in Document S1, which was not available. Neither the earlier notes nor this audit saw the figure.

| Cell | Corrected text |
|---|---|
| Region | Rat dorsal subiculum and CA1 |
| Pattern | Narrower in subiculum (width not given) |
| Status | full text; Fig. S7H not seen (supplement unavailable) |

### Row 7. Tanni, de Cothi & Barry 2022

- **Region, "Rat CA1".** Confirmed. "we recorded 629 CA1 place cells (89–172 cells/rat) from five rats" (Results).
- **Space, "Arenas up to 3.5 × 2.5 m".** Confirmed. "The environment sizes were 87.5 x 125 cm (environment A), 175 x 125 cm (environment B), 175 x 250 cm (environment C) and 350 x 250 cm (environment D)." (STAR Methods). The wall result comes from environment D.
- **PV construction.** Not a PV correlation and not built from rate maps. The measure is the Euclidean distance between instantaneous population vectors 1 cm apart along the path. "The population activity change was then computed as the Euclidean distance between consecutive samples of place cell firing rates - population activity vectors measured at 1 cm intervals. All place cells detected in an animal were included and periods where the animal was moving slower than 10 cm/s were excluded." (STAR Methods, population activity change). Rates were "computed at 33 ms bins and smoothed with a Gaussian (166 ms sigma)".
- **Similarity near zero, "Changes fastest near walls".** Corrected (imprecise). The effect holds only for runs orthogonal to a wall, and the paper has no similarity-versus-distance curve. "analyzing only trajectories running orthogonal to the short wall of the largest environment (D), we found that the Euclidean distance between activity vectors (1-cm intervals) was the greatest when animals were close to the wall (Kruskal-Wallis: H = 12.2, p = 0.007, n = 5; Figure 5A), an effect that was not found for parallel runs (Kruskal-Wallis: H = 3.78, p = 0.29, n = 5" (Results).
- **Pattern, "Width varies with position".** Confirmed as an inference. The paper measures a local rate of change, not a kernel width. Its field-size result supports the inference. "the average size of place fields increased with distance from the wall" (Results).
- **Status, "full text".** Confirmed.

| Cell | Corrected text |
|---|---|
| Similarity near zero at | No PV-versus-distance curve. The activity vector changes most per cm near the wall on runs orthogonal to it (env. D, H = 12.2, p = 0.007), not on parallel runs (H = 3.78, p = 0.29) |
| Pattern | Local rate of change varies with position and heading (inference) |

### Rows 8 to 10. Kjelstrup et al. 2008

- **Region, "Rat dorsal / intermediate / ventral CA3".** Confirmed. "we compared neural activity at multiple longitudinal levels of CA3 in 21 rats" and "The data were subdivided into three groups based on recording location (dorsal, 7 to 22%; intermediate, 40 to 70%; and ventral, 70 to 100%)." (p. 140).
- **Space, "18 m track".** Confirmed. "we tested the animals on an 18-m-long linear track" (p. 140). The analysed length is 16.5 m. "the length of the track was segmented into 5-cm bins, giving a total of 330 population vectors per running direction after exclusion of the turning points" (p. 141). 330 x 5 cm = 16.5 m.
- **PV construction.** Within one composite map per level and direction, Pearson correlation, pooled across rats. "Linear rate maps of all pyramidal cells from each location were stacked, and a population vector was defined for each 5-cm bin of the composite map. Pearson's product moment correlation coefficient was calculated for each pair of population vectors" (Fig. 3 legend, p. 142). "Rate maps were stacked separately for left and right running directions" (p. 141). The main text does not state smoothing, and the SOM was not read.
- **Similarity near zero.** The text gives widths only at r = 0.5 and r = 0.2. "The mean half-width (at r = 0.50) was 0.81 m for dorsal CA3, 3.71 m for intermediate CA3, and 6.66 m for ventral CA3 (Fig. 3). The mean width at r = 0.20 was 1.41 m (dorsal), 8.65 m (intermediate), and 13.59 m (ventral)." (p. 141). The near-zero distances are figure reads, made here from the vector paths of Fig. 3.
  - Dorsal "about 1 m". Confirmed. The two curves fall below r = 0.1 at 0.9 and 1.1 m and read 0.08 and 0.12 at 1 m. They first cross zero only at 5.5 to 6.7 m, because the tail hovers between 0 and 0.1.
  - Intermediate "about 7 m". Confirmed. The curves cross zero at 6.9 and 7.1 m and then fall to about -0.1.
  - Ventral "about 10.7 m". Corrected (minor). The curves cross zero at 10.45 and 10.51 m and then fall to between -0.05 and -0.13. Calibrating the axis on the text widths instead of the ticks gives 10.2 to 10.3 m, so 10.7 m lies above every estimate.
  - Ventral "0.5 at 3.3 m". Confirmed. The line graphs are symmetric about zero ("Correlation as a function of spatial distance between population vectors. Horizontal lines indicate half-width (14) and 20% width.", Fig. 3 legend, p. 142), and the half-width bar spans both sides. The 6.66 m "half-width" is therefore a full width, and r = 0.5 falls 3.33 m from the diagonal. The vector curves give 3.30 and 3.47 m.
- **Arithmetic, "6%, 42%, 65%" of 16.5 m.** 1/16.5 = 6.1% and 7/16.5 = 42.4% are confirmed. 10.7/16.5 = 64.8% is correct for the stated distance. With 10.5 m the share is 63.6%, so 64%.
- **Pattern, "Local / Broad / Broad".** Confirmed as a classification. The intermediate matrices in Fig. 3B show large blocks rather than a uniform band.
- **Status, "full text, figure-read".** Confirmed. The SOM, which has fig. S8 with both directions, was not read.

| Cell | Corrected text |
|---|---|
| Similarity near zero at | dorsal about 1 m, 6%; intermediate about 7 m, 42%; ventral about 10.5 m, 64%, and 0.5 at 3.3 m (text full width at r = 0.5 is 6.66 m); shares are of the 16.5 m analysed |

### Row 11. Maurer et al. 2005

- **Region, "Rat CA1, dorsal and middle".** Confirmed. "ensembles of CA1 pyramidal cells and interneurons were recorded from the extreme dorsal and middle portions of the hippocampus" (Abstract). The abstract does not name the species. PubMed indexes the paper under the MeSH term "Rats, Inbred F344".
- **Space, "Track".** Confirmed as far as the abstract goes ("during track running"). Track type and length are not verifiable.
- **PV construction.** Not verifiable. The Wiley full text returned HTTP 403.
- **Similarity near zero, "Faster dorsally, no distances given".** Confirmed. "Pyramidal cell population vectors representing successive locations became decorrelated over substantially shorter distances in the dorsal than in the middle hippocampus." (Abstract).
- **Pattern, "Gradient".** Corrected (minor). Two levels were compared, so "gradient" overstates the sampling.
- **Status, "abstract".** Confirmed.

| Cell | Corrected text |
|---|---|
| Pattern | Narrower dorsally than in middle CA1 (two levels) |

### Row 12. Terrazas et al. 2005 (WebFetch)

- **Region, "Rat dorsal CA1".** Confirmed. "Multiple single units were recorded from the CA1 region of the dorsal hippocampus" and "Six male Fischer 344 rats" (Methods).
- **Space, "4 m ring".** Confirmed, approximate. "a 130-cm-diameter G-scale circular track" (Methods). π x 130 cm = 408 cm, consistent with the "128 bin matrix (3.2 cm/bin)" (409.6 cm). The largest separation on the ring is about 2 m.
- **PV construction.** Within one condition, a normalized dot product with no mean subtraction. "The overlap matrix for population vectors was computed (normalized dot products)." (Methods). Cells were "active (firing >40 spikes) pyramidal cells" (Fig. 9 legend): 152 in WALK, 96 in CAR and 217 in WORLD. For CAR and WALK, "within-session data acquired on the RTA were used"; the WORLD cells "were obtained from different subjects" (Methods).
- **Similarity near zero, "About 3 times wider when the rat drives a car".** Confirmed, but the measure needs stating. "approximately three times larger during CAR than during WALK" (Results). In CAR the rat drives itself "by pressing and holding down a GO button on a small car (CAR)" (Introduction). The scale is not a near-zero distance. "the point at which the average overlap falls to 0.5 provides a convenient, unbiased definition" (Methods). The text gives no distance for any condition. On a ring the curve also turns back, "the minimal value of the function is reached at 180°, after which the function increases again" (Numerical simulations).
- **Pattern, "Broad only with altered self-motion".** Partly verifiable. WORLD is supported, since the scale "substantially increased again during WORLD". Whether CAR is broad in absolute terms needs Fig. 9, which I could not view.
- **Status, "full text, figure only".** Confirmed.

| Cell | Corrected text |
|---|---|
| Space | 130 cm diameter ring (about 4.1 m around) |
| Similarity near zero at | The distance where non-mean-subtracted overlap falls to 0.5 is about 3 times larger when the rat drives a car than when it walks, and larger again when the room rotates around a still rat; no distances in text |

### Row 13. Nielson et al. 2015 (WebFetch)

- **Region, "Human left anterior hippocampus, fMRI, 9 people".** Confirmed. "We recruited 10 participants (aged 19–26 y, mean age = 21.4 y; nine female)", and the data of one participant were dropped ("we did not include the data for that participant in any of our analyses"). The ROI is the anterior third of the Harvard-Oxford hippocampus ("dividing these regions into thirds based on their anteroposterior extent").
- **Space, "Real world, 100 m to 30 km".** Confirmed. "we required that image pairs have a spatial distance between 100 m and 30 km". The items are recalled events cued by lifelogging photos, not current positions. Participants were told to "try to remember the event depicted in the picture, and try to mentally relive your experience".
- **PV construction.** "neural distance for each image pair was calculated as 1 minus the correlation" and "Neural distances were z-scored within subjects." A per-subject GLM ("In each ROI in each subject, we then ran a GLM") regressed neural distance on log space, log time, their interaction and scanner time. Group inference used a permutation test, with Bonferroni correction across ROIs.
- **Similarity near zero, "No zero or plateau in range; pattern distance rises linearly with log distance (slope 0.71, t = 5.55)".** Corrected. Linearity in log distance is the model's assumption, not a tested result, and the paper reports no test for a plateau. The log was chosen "based on previous literature that has shown a power-law relationship for neural representations". The numbers are right. Table S1 lists space in left anterior hippocampus as beta = 0.7064, t = 5.5539, P = 0.005, and the main text gives "spatial distance (corrected P = 0.021)". The beta is in within-subject z-scored neural distance per log10 unit of distance.
- **Pattern, "Broad, remembered places".** Confirmed, with a wording note. "Recalled events" describes the task more closely.
- **Status, "full text".** Confirmed.

| Cell | Corrected text |
|---|---|
| Similarity near zero at | No length scale. The GLM assumes neural distance is linear in log10 distance; no plateau test is reported. Space beta = 0.71 (z-scored neural distance per log10 unit), t = 5.55, corrected P = 0.021 |
| Pattern | Broad, recalled real-world events |

### Row 14. Deuker et al. 2016

- **Region, "Human hippocampus, fMRI, 26 people".** Confirmed. "26 participants signed up for the study" (Materials and methods, p. 18 of 26). The ROI is bilateral hippocampal grey matter.
- **Space, "Virtual city, about 390 m".** Confirmed. "Relating the eye-level height of the first-person player (assumed to be at 1.60 m) to these arbitrary units, one side length of the square city roughly translates to 390 m." (p. 18).
- **PV construction.** The measure is the change in pattern similarity for object cues from before to after learning (PS'), with objects "shown out of context and in random order" (p. 8). For each participant, PS' was correlated with distance and converted to a z score against shuffles. The group test used sign flips, "the observed mean z-statistic was compared to a distribution derived by performing random sign-flips" (p. 23).
- **Similarity near zero, "No length scale reported; similarity gain correlates with distance (group Z = -3.7 remembered, -2.85 true)".** Confirmed. "pattern similarity changes in bilateral hippocampus reflected participants' spatial distance judgments (Z = -3.719, pFDR=0.0005" (p. 8). "objective spatial distances between pairs of items were associated with an increase in pattern similarity across all gray-matter voxels in bilateral hippocampus (Z = -2.85, pFDR = 0.02" (p. 14). "True" means Euclidean distance. "We defined objective spatial distances between items as the Euclidean distance" (p. 14). The minus signs are PDF glyphs restored here.
- **Pattern, "Broad, remembered places".** Corrected (imprecise). The paper tests a correlation and illustrates it with binned bars, so "broad" is not a measured shape, and the cues are objects placed in the city.
- **Status, "full text".** Confirmed.

| Cell | Corrected text |
|---|---|
| Pattern | Graded with distance across the city (shape not tested); object cues after learning |

### Row 15. Maimon et al. 2026

- **Region, "Bat CA1".** Confirmed. CA3 is also compared. Recordings came "from dorsal CA1 and dorsal CA3 in five bats flying in the long tunnel, and an additional nine bats with only dorsal CA1" (main text). The curve is CA1. "Population-vector correlations as a function of position for CA1 place cells" (Fig. 5e legend). The CA3 retrospective code "was less robust and exhibited greater variability" (Fig. 5g legend).
- **Space, "174 to 194 m tunnels".** Corrected. The route-history result comes from the multicompartment tunnel. "the 180-m tunnel was connected at one end to two additional 10-m compartments" with "(n = 4 bats)" (Results), and the bats "flew in the large multicompartment tunnel ... 193 m length" (Methods). 194 m belongs to a tunnel this analysis did not use. "The 200-m tunnel had an effective length of 194 m" (Methods). The figure 174 m does not appear anywhere in the text.
- **PV construction.** Between flight types at the same position, pooled over days, smoothed over 10 m. "we constructed population vectors from the firing-rate maps of the entire place cells population (separately for each bat, pooled over all of the recording days of the bat)", "smoothed by averaging over a sliding window of 10 m (boxcar smoothing), with 0.2 m steps", and "for each spatial bin (each position), we calculated the Pearson correlation between pairs of population vectors" (Methods).
- **Similarity near zero, "Route history lasts over 90 m".** Confirmed. "The retrospective population vector correlation was lower for around 90 m after the turn" (Results). "In bat 2, this convergence occurred at x = 88.5 m, namely 91.5 m from the junction; thus, the retrospective signal lasted >90 m after the bat has left the junction." (Extended Data Fig. 13 legend).
- **Pattern, "Not a distance measure".** Confirmed.
- **Status, "full text".** Confirmed.

| Cell | Corrected text |
|---|---|
| Space | Multicompartment tunnel, 193 m (a 180 m tunnel ending in two 10 m compartments), 4 bats |

### Row 16. Rich 2014, Eliav 2021, Harland 2021

- **Region, "Rat and bat CA1".** Confirmed, all dorsal. Rich: "A craniotomy was performed over the CA1 field of the right dorsal hippocampus" in "five adult male Long-Evans rats" (supplement p. 2). Eliav: "recorded from hippocampal dorsal CA1 neurons of wild-born bats" (PDF p. 2). Harland: "We recorded 539 place cells from dorsal CA1 over 54 sessions in five rats" (manuscript p. 3).
- **Space, "48 m, 200 m, 18.6 m²".** Confirmed. Rich: "A 48-meter-long elevated linear track was laid out around the room." (supplement p. 2). Eliav: "flying in a 200-m-long tunnel" (PDF p. 1). Harland: "Rats were recorded in a megaspace (5.3 x 3.5 m; 18.6 m2" (manuscript p. 2).
- **Similarity near zero, "Not measured".** Confirmed. "Population vector" occurs zero times in the Rich main text and supplement. In Eliav it appears only as a decoder run on simulated data, "we generated synthetic data for each coding scheme and then used maximum-likelihood and population-vector decoders to test their decoding performance" (Discussion, PDF p. 13). Harland has no PV analysis. Its one decoding sentence is a Discussion suggestion that "subfields use an ensemble pattern decoding scheme" (manuscript p. 7).
- **Pattern, "No data".** Confirmed.
- **Status, "full text".** Confirmed. Harland was read as the accepted manuscript, and the Eliav PDF includes the supplementary materials.

## Summary of corrections

1. Row 5 (Rubin 2015) is a between-environment measure. Fig. 1G correlates PVs at matching positions on two different tracks, so it says nothing about similarity between separated locations, and both '"Low" between locations' and "Local" are wrong. Drop the row or move it to a remapping contrast. Space should read "Two tracks, straight 96 cm and L-shaped 2 x 48 cm, compared with each other".
2. Row 15 (Maimon 2026) Space. The route-history result comes from the 193 m multicompartment tunnel (a 180 m tunnel plus two 10 m compartments, 4 bats). The paper never mentions 174 m, and 194 m is the effective length of a different tunnel.
3. Row 6 (Ledergerber 2021) Status. Neither the earlier notes nor this audit saw Fig. S7H, because the supplement is unavailable, so "figure only" overstates the access. Use "full text; Fig. S7H not seen (supplement unavailable)". Region should add "dorsal", and Pattern should drop the unverifiable "Local".
4. Row 13 (Nielson 2015). Linearity in log distance is the model's assumption, and the paper reports no plateau test. Restate the result as a GLM coefficient (beta = 0.71 per log10 unit, t = 5.55, corrected P = 0.021).
5. Row 7 (Tanni 2022). The near-wall effect holds only on runs orthogonal to the wall, in environment D. The measure is a Euclidean rate of change along the path, not a similarity curve.
6. Row 1 (Green 2022). "30 to 40 cm" merges the legend's undefined "spatial scale of ∼30 cm" with the text's bound "<40 cm". The curve crosses zero at about 32 cm and continues to about -0.3, so it does not settle at zero. The PV matrix covers about 160 cm of the 1.8 m track.
7. Row 2 (Wang 2020). Region should say dorsal. "Not measured" should become "not stated in text", because Fig. 1D is a measured within-session matrix. Its colour read gives r = 0.1 at about 64° (CA1) and 99° (CA3), 18% and 28% of the loop. The matrices are within one session type, not STD against MIS.
8. Rows 8 to 10 (Kjelstrup 2008). The ventral zero crossing is about 10.5 m (64%), not 10.7 m (65%). The dorsal and intermediate values and the r = 0.5 point at 3.3 m are confirmed.
9. Row 12 (Terrazas 2005). The factor of 3 refers to the distance where a non-mean-subtracted overlap falls to 0.5, not to a near-zero distance. The "4 m ring" is a 130 cm diameter ring.
10. Row 3 (Battaglia 2004). The significant difference comes from experiment C (two tracks, 2 rats). Experiment B (one track with two halves, 4 rats) shows only a trend. "Local" cannot be checked without Fig. 7.
11. Row 14 (Deuker 2016) Pattern. The paper does not test the shape, so "Broad" is unsupported. Use "Graded with distance across the city (shape not tested); object cues after learning".
12. Row 11 (Maurer 2005). The species comes from MeSH indexing, not the abstract. "Gradient" rests on two levels.
13. Row 4 (Mao 2017) and row 16 are confirmed. Mao's cell could add that the curve goes negative beyond 35 cm.

One issue cuts across rows. In Green, Mao and Kjelstrup, "near zero" is a crossing point, and the curves continue below zero by an amount that depends on the PV construction. Green z-scores each neuron and reaches about -0.3. Mao scales each cell from 0 to 1 and reaches about -0.26. Kjelstrup correlates raw rate maps and reaches about -0.1. Terrazas uses an overlap that cannot fall below zero, and Wang's colour scale cannot display negative values. A "near zero at" column compares like with like only if each entry names its construction.
