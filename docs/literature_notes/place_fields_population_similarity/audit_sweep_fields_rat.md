# Audit of the rat field-width sweep

Audit date 2026-11-21. Target: `sweep_fields_rat.csv` (82 rows, 30 studies) and `sweep_fields_rat.md`. Corrections are in `audit_sweep_fields_rat_corrections.csv` (79 entries on 39 rows). The sweep files are unchanged.

## Summary

| Verdict | Rows |
|---|---|
| Confirmed | 39 |
| Corrected (a value, label or status changes) | 28 |
| Unverifiable | 5 |
| Not re-checked (note rows outside the spot-check) | 10 |
| Dropped | 0 |

Seven confirmed rows also get a flag in `notes` for double counting (kjelstrup2008_4 to _9, neunuebel2012_3). One unverifiable row also gets a flag (royer2010_2). The 10 rows not re-checked are park2011_1 to _4, rich2014_1, lee2025_1 and _2, and komorowski2013_1 to _3. Park and Komorowski count here because their widths are figure readings I did not redo, although I confirmed their field rules, bin sizes and counts in the text. The row total is 39 + 28 + 5 + 10 = 82.

## Sources read in this audit

- Full text through the NCBI BioC API, downloaded into `tmp/audit_sweep_rat/dl/bioc/`: Mizuseki 2012 (PMC3718552), Diba & Buzsaki 2008 (PMC2656938), Royer 2010 (PMC2825159), Komorowski 2013 (PMC3667351), Park 2011 (PMC3137630), Wirtshafter 2020 (PMC7274787), Wang 2020 (PMC7259364).
- The sweep's BioC and XML downloads, re-read here: Fenton 2008, Henriksen 2010, Tanni 2022, Neunuebel 2012, Diamantaki 2016, Mankin 2015, Kim 2012, Burke 2011, Karlsson 2008, Grieves 2020, Schlesiger 2015, Hayman 2011, Ravassard 2013, Mou 2018, Jacob 2020, Lee SM 2022.
- Supplements and source data, parsed here: the Grieves source data (recomputed with `scripts/grieves_check.py`), the Diamantaki Figure 4 source data, Schlesiger Supplementary Table 1, the Hayman and Grieves supplements.
- PDF text of the user's PDFs: Kjelstrup 2008 main text and SOM, Jung 1994, Sharp & Green 1994, Wang 2020.
- Not reachable: Shen 1997 and Hollup 2001. Both are page scans in PMC. NCBI BioC has no text, and the jneurosci.org, Europe PMC render and PMC PDF routes returned Cloudflare or proof-of-work challenges. Rich 2014 is closed access. Lee 2025 (preprint) and Lyttle 2013 were not fetched.

## The eight WebFetch rows

All eight (mizuseki2012_1, _2, _5, _6, _7, _8 and diba2008_1, _2) match the full text, number for number. Their status changes to "full text". Mizuseki and Diba still need label corrections, listed below. Mizuseki rows 3 and 4 and royer2010_1, also WebFetch-sourced, match the full text as well.

## Row by row

### Kjelstrup 2008 (user PDF and SOM; note rows, spot-checked)

- kjelstrup2008_1: confirmed. Main text: "0.98 ± 0.03 m in dorsal hippocampus (mean ± SEM; 111 cells, five rats)". The Fig. S6 legend says size is "the width of the largest place field", so "largest field per cell" is right.
- kjelstrup2008_2: confirmed. "3.73 ± 0.43 m in intermediate hippocampus (37 cells, three rats)".
- kjelstrup2008_3: confirmed. "5.52 ± 0.54 m in ventral hippocampus (61 cells, four rats). Few fields were longer than 10 m."
- kjelstrup2008_4, _5, _6: confirmed. "width of the central peak of the spatial autocorrelation function, measured at 20% of the peak value (dorsal, 1.12 ± 0.03 m; intermediate, 3.07 ± 0.19 m; and ventral, 4.10 ± 0.18 m". Flag added: same cells as rows 1 to 3.
- kjelstrup2008_7, _8, _9: confirmed. "mean half-width (at r = 0.50) was 0.81 m for dorsal CA3, 3.71 m for intermediate CA3, and 6.66 m for ventral CA3". Flag added: same cells as rows 1 to 3.
- kjelstrup2008_10: confirmed. SOM: "619 ± 116 cm2, or ~10% of the box"; 25 dorsal cells in a 1 m × 1 m box; field rule "at least 200 cm2 ... above 20% of the peak rate ... at least 1 Hz". 2*sqrt(619/π) = 28.1 cm.

### Jung 1994 (user PDF; note rows, spot-checked)

- jung1994_1: confirmed. Table 1 "Size of place fields (bins) 67.0 ± 7.5"; 168 cm square divided into 64 × 64 bins, so 2.625 cm bins of 6.89 cm2; rule "minimal number of adjacent bins was set at 4, and the threshold ... 1 Hz". 67.0 × 6.89 = 462 cm2, so 24.2 cm. "8 of 36 dorsal ... units had two place fields". Rats "trained in a square chamber (68 × 68 cm)".
- jung1994_2: confirmed. 272.0 ± 44.3 bins, 1874 cm2, 48.8 cm; "3 of 22 ventral units had two place fields"; "as large as 80% of the entire recording apparatus".

### Mizuseki 2012 (PMC3718552)

- mizuseki2012_1: corrected (status only). "The mean place field size was significantly larger for CA1 (75 cm) than in CA3 (50 cm; P < 0.0001; Fig. 7E)". Track 250 × 7 cm; rule "> 10% of the peak", "> 2 Hz", coherence "> 0.7"; Gaussian SD 5 cm; 13 rats; 48.0% of CA1 cells met the criteria.
- mizuseki2012_2: corrected (status only). Same sentence; CA3 from "six of the rats"; 16.8%.
- mizuseki2012_3: corrected. Value confirmed: "CA1 (1,775 cm2, median)". The field size comes from the 180 × 180 cm field only. Fig. 5 legend: "Size of the open field was 180 × 180 cm2, except for E, where 200 × 100 cm2 field was used." Environment set to 180 × 180 cm and 3.24 m2. 2*sqrt(1775/π) = 47.5 cm.
- mizuseki2012_4: corrected. "CA3 neurons (1,275 cm2 ...)", 40.3 cm; same environment fix. Flag added for overlap with Royer 2010 dorsal CA3 (see "Double counting").
- mizuseki2012_5: corrected. Value confirmed: "the median place field size of CA1 neurons was significantly larger (38.3 cm) than that of CA3 cells (33.8 cm". Arms "seven 75-cm arms". The field rule copied the track's 5 cm bins, 15 cm minimum and 5 cm SD. The paper gives the mazes their own binning: "Each 'linearized' arm ... divided into 100 equal pixels ... smoothed ... (5 pixels half-width)". Familiarity changed to "trained" ("animals were trained to seek out water rewards").
- mizuseki2012_6: corrected, as mizuseki2012_5.
- mizuseki2012_7: corrected. Value confirmed: "CA1 = 48 cm, CA3 = 34 cm". The statistic is a median. Fig. 11B legend: "White arrows indicate median field size". Field rule as for the radial maze. Familiarity changed to "trained" ("the animals learned to run back and forth"). Zigzag maze "100 × 200 cm2" confirmed.
- mizuseki2012_8: corrected, as mizuseki2012_7.

### Diba & Buzsaki 2008 (PMC2656938)

- diba2008_1: corrected. Value confirmed: "decreased significantly from a mean of 52 cm to 42 cm (19% decrease; rank-sum p < 10−12; Figure 1d)". Edge rule "portion of the track with > 2 Hz firing". 170 cm × 6.2 cm track, 1.44 cm2 bins, eight-point smoothing, three rats, "trained" before implant: all confirmed. Region changed to "CA1+CA3 (pooled)", because "all data is hereby pooled" and 688 of 1750 fields are CA3. n_fields changed: the sentence sits in a paragraph about the 1147 persisting fields ("the persisting subset", "this population"), so the mean is most likely over those 1147, not 1750.
- diba2008_2: corrected (status, region). 100 cm track, 1147 persisting fields.

### Wirtshafter 2020 (PMC7274787; note row, spot-checked)

- wirtshafter2020_1: confirmed. "The average length of a CA1 place field was 29.1+−14.7 cm"; "104 units had at least one place field, with 63 ... one field, 33 with two fields, seven with three fields, and a single cell with four fields", so 154 fields and 1.48 per cell. The CA1 vs LS test has t(320), which fits 154 + 168 fields. Edge at mean + 1 SD, peak at mean + 2 SD, 15 cm minimum, 2 cm bins with a 10 cm SD kernel, "trained for 2–4 weeks".

### Park 2011 (PMC3137630; note rows)

- park2011_1 to _4: not re-checked (widths are readings of Fig. 4B). The text parts are confirmed: "2.8 cm square pixel"; field "any contiguous set of 9 or more pixels with greater that 0 AP/s"; "we compared the size of the largest place field for each cell (Fig. 4B)"; pooled fields "F1,1316 = 0.01 p = 0.93"; counts "5.89±0.40", "17.9% (14/78)", "60.7% (34/56)", "4.96±0.56", "81.8% (18/22)", "29% (12/41)"; "both enclosures were equally familiar". Pixel area 7.84 cm2 and the arithmetic are right.
- park2011_5: confirmed. "5.35±0.38 fields/cell"; "63.9% (23/36) had a single place field in the cylinder just 8.2% (4/49) ... in the box"; "CA1>CA3 = DG" for the largest field.

### Rich 2014

- rich2014_1: not re-checked. Closed access; carried from the earlier note audit.

### Lee 2025 (preprint)

- lee2025_1, lee2025_2: not re-checked. Figure readings from a preprint; carried from the earlier note audit.

### Royer 2010 (PMC2825159)

- royer2010_1: corrected (status only). "The size of single-arm firing fields, quantified as the length of activity between the 20% boundaries of the peak firing rate in a given arm was 39% larger in vCA3 cells"; 329 dCA3 and 81 vCA3 pyramidal cells.
- royer2010_2: unverifiable. The value comes from Lyttle 2013 Table 1, which I did not read. The primary paper shows coverage only as a distribution in Fig. 2B. The coverage rule is confirmed ("summed up all pixels where the firing rate exceeded 20% of the peak firing rate and calculated the fraction of 'active pixels' to all pixels"), as are the 1 m × 2 m floor with 30 cm walls and 569 dorsal cells. Flag added for overlap with Mizuseki 2012.
- royer2010_3: unverifiable, as royer2010_2. 95 ventral cells and "rarely showed continuous two-dimensional place fields" (abstract) are confirmed.

### Komorowski 2013 (PMC3667351; note rows)

- komorowski2013_1 to _3: not re-checked (Fig. 5B readings). The text parts are confirmed: two "37 × 37 cm environments"; field "any contiguous set of at least 50 pixels with a firing rate greater than 20% of the peak"; 2 cm bins; size "number of bins within the field divided by all bins visited"; levels "Dorsal CA3 ... between 2 and 4 mm away from the dorsal CA2/3 border, intermediate CA3 ... between 4 and 7mm"; Kruskal-Wallis H = 25.81 and 23.19; cell counts 163/35/31 and 96/40/31. The figure is Fig. 5 ("Place Field Sizes in Learning and with Overtraining"), although the manuscript text cites "Figure 4B".

### Fenton 2008 (PMC2695947)

- fenton2008_1: confirmed. Table 1 "place field size (pixels) | 57.8±2.4 | 66.2±3.5 | 75.0±3.9"; Fig. 8 legend "A pixel is 22 cm2"; field "at least nine contiguous pixels had a rate greater than 0 AP/s"; "1.3±0.03"; "Most (72%) place cells had single place fields in the cylinder"; "64% (307/477)"; 68 cm cylinder; rats kept familiar by weekly screening sessions; AP 3.8, L 2.5 mm. With 22 cm2 the width is 40.2 cm; the sweep's 40.3 uses 4.7 × 4.7 = 22.09 cm2. The difference is rounding.
- fenton2008_2: confirmed. Floor 75.0 ± 3.9 pixels, 45.8 to 45.9 cm; "3.4 ± 0.11" on the floor, "4.05±0.12" with stairs; "72% (322/448)"; Table 2 52.4 ± 3.5 vs 94.6 ± 8.9 pixels; 150 × 140 cm floor.

### Henriksen 2010 (PMC3093538)

- henriksen2010_1: confirmed. "proximal: 2208 ± 146 cm2 ... all fields of all cells on Trial 1"; field "9 or more 5 cm × 5 cm bins ... exceeded 20% of the cell's peak rate ... higher than 0.25 Hz"; "Trial 1: 1.57 ± 0.08"; "57.0%" single-field; 79 active cells; 7 rats; 2 m cylinder with 50 cm walls; 2 to 3 weeks of training in both cylinders; speed ≥ 5 cm/s. 2*sqrt(2208/π) = 53.0 cm.
- henriksen2010_2: confirmed. 2688 ± 195 cm2, 58.5 cm; 1.57 ± 0.08; 56.7%; 90 active cells.
- henriksen2010_3: confirmed. 1931 ± 137 cm2, 49.6 cm; 2.13 ± 0.14; 33.8%; 77 active cells.

### Tanni 2022 (PMC9616721 and figures)

- tanni2022_1: confirmed. I re-read Fig. 3B (left panel, "< 25 cm from wall") on the log axis, calibrated from the 10^4 tick and the 2×10^4 to 5×10^4 minor ticks. Environment A sits at about 2000 to 2700 cm2, so 50.5 to 58.6 cm. "1.28 fields/cell in A", "four familiar, differently sized environments", 629 cells, 5 rats, the iterative 1 Hz + 0.05 Hz rule and the bin-weighted averaging ("averaging the values of all fields overlapping a particular spatial bin") are confirmed.
- tanni2022_2: confirmed. B about 3500 to 4400 cm2.
- tanni2022_3: confirmed. C about 4200 to 6100 cm2.
- tanni2022_4: corrected. Areas confirmed (about 4600 to 9500 cm2), as are "2.28 fields/cell in D", log-likelihoods "−1,538 versus −1,679" and "84.8%". The "active cells 15%" in count_distribution is a misreading. The paper's 15% is the share of place cells co-active (rate > 1 Hz) at a location, constant with wall distance, not a share of cells active in the box. The sweep md repeats the misreading in its "Silent or inactive share" column.
- tanni2022_5: confirmed. On the enlarged Fig. 3B panel for D, the 75 to 100 cm bin runs from about 10,300 to 17,900 cm2, so 114.5 to 150.5 cm.
- tanni2022_6: confirmed. Fig. 3C: about 57 to 120 cm in the first bin and about 133 to 177 cm in the last.

### Neunuebel 2012 (PMC3321836)

- neunuebel2012_1: corrected. The quotes are right: "single fields: median 42.1 cm2, IQR 29.0–72.8; multiple fields: median 13.3 cm2, IQR 7.1–21.0". But the area unit cannot be resolved. The paper says "a 64 × 48 grid. Each square pixel was ~2.64 cm2 (small box) and ~3.38 cm2 (large box)". With 1.84 cm pixels, 48 pixels span only 88 cm, less than the 135 cm box. So the pixel figures must be side lengths in cm, and the 10-pixel minimum field is 70 to 114 cm2. The reported areas fall below that minimum in cm2, and the IQR also falls below 10 if read as pixels. I cleared width_cm (7.3 cm). Second error: the 25 cells and their count distribution come from all DG recording sites, not from granule-layer or transition tetrodes. The paper: "We thus decided to segregate all of our recording sites ... From this sample, there were 25 spatially active cells". Of the 26 granule-layer or transition tetrodes, "only 1 tetrode recorded a cell with multiple place fields, 5 tetrodes recorded cells with single fields". Large box familiar, small box not stated.
- neunuebel2012_2: corrected. Width cleared for the same unit reason. The ratio inside the study still holds: subfields of multi-field cells are about a third the area of single fields, and the summed areas are similar.
- neunuebel2012_3: confirmed. "the majority of the 8 spatially selective cells fired in multiple locations (quadruple fields: 1; triple fields: 3; double fields: 3; single fields: 1)"; one silent cell, two diffuse cells, one sub-criterion cell. Flag added: the 8 cells come from the last foraging day of 5 tetrodes, and the 25-cell sample takes each tetrode on its day with the most sleep clusters, so the same tetrodes may feed both groups.

### Diamantaki 2016 (PMC5077296 and source data)

- diamantaki2016_1: corrected. Source data "Field Size (pixels)": cells 102, 103, 104 and 1046 = 118, 71, 80 and 60. Pixels "2.5 cm x 2.5 cm". Widths recomputed as 30.6, 23.8, 25.2 and 21.9 cm. Silent count "163 were from silent neurons" of 190 and the counts "In eight recordings, a single firing field ... in three cases, an additional field" are confirmed. But the cells were not all in the square arena. Fig. 3 says cell 103 was "recorded in a square arena" and cell 104 "on an O-shaped linear arena" (14 cm path). For cell 104 an equal-area diameter is meaningless; 500 cm2 on a 14 cm path is about 36 cm of track. The arena for cells 102 and 1046 is not stated.

### Mankin 2015 (PMC4392894)

- mankin2015_1: confirmed. "the fields of CA2 cells were 24.5 % larger than those of CA1 cells (Mann-Whitney U: CA1 vs. CA2, z = −2.54, P = 0.034)"; CA2 vs CA3 "P = 0.053"; "highly familiar environments"; 80 × 80 cm square and 16-sided polygon of 50 cm radius; 62 CA2 cells in 5 rats, 54 active; "place fields smaller than 25 % of the recording enclosure in at least one session".
- mankin2015_2: confirmed. "CA1, 59.2 %"; "less than one field per cell in CA1 and CA3"; 43 + 46 cells.
- mankin2015_3: confirmed. "CA3, 48.1 %"; 42 + 29 cells.

### Kim 2012 (PMC3458125)

- kim2012_1 to _3: confirmed. "0.31 (0.19–0.43), 0.54 (0.32–0.68), and 0.69 (0.59–0.74)"; "subiculum at intermediate locations along the septal-temporal axis"; Table 1 coordinates from 6.6 mm posterior; U track 6 cm wide and 350 cm long; Table 2 unit totals 21, 35 and 32; 5 rats.
- kim2012_4: confirmed. "distal CA1 was 0.26 (0.21–0.37)"; 24 units in CA1 and the transition zone.

### Burke 2011 (PMC3314262)

- burke2011_1: confirmed. "the mean size of place fields for behavioral conditions with no objects on the track was ~80 cm ... reduced the mean size of place fields to ~55 cm (T[3] = 5.68, p < 0.01"; phase-precession field rule; rate rule "exceeded 10% of the maximum for at least 5 consecutive 3.84 cm bins, which yielded similar results"; "~335 cm in circumference"; "distal (closest to the subiculum) intermediate CA1"; 6.0 mm posterior; 1036 pyramidal cells; 4 Fischer-344 rats aged 8 to 12 months; track familiar; "an average of 1.6 place fields". The Discussion gives "~2.3 versus ~1.6"; the row uses the Results value.
- burke2011_2: confirmed. "~55 cm", 2.1 fields per active neuron.

### Karlsson & Frank 2008 (PMC2632980)

- karlsson2008_1: confirmed. "151/275 (55%) ... 203/406 (50%) ... 239/600 (40%)"; 0.1 Hz threshold; CA1 total 708; 7 rats; AP −3.7, ML ±3.7.
- karlsson2008_2: confirmed. "108/373 (29%), 185/591 (31%), and 234/776 (30%)"; CA3 total 984.

### Grieves 2020 (PMC7005894 and source data)

- grieves2020_1: corrected. My recomputation from the source data matches the sweep: 628 arena cells, counts 0:87, 1:351, 2:150, 3:38, 4:2, 1.231 fields per cell (1.429 among cells with a field), variance/mean 0.48; 693 diameters with median 0.646 m, IQR 0.476 to 0.872 and mean 0.717 ± 0.318 m. Three label changes. (1) The measure is defined: "diameter (diameter of minimum enclosing sphere)". It is a maximum extent, not comparable to equal-area diameters. (2) Familiarity is not stated in the paper or supplement. The rats lived for 8 weeks or more in a climbing enclosure with a miniature lattice. (3) The source-data columns disagree. The count column sums to 773 fields but the diameter column holds 693. The paper says 82.5% of arena cells had a field, the count column 86.1%.
- grieves2020_2: corrected (familiarity, measure). Recomputed: 432 cells, 1.762 per cell, 604 diameters, median 0.668 m, IQR 0.481 to 0.940, mean 0.732 ± 0.304 m. Tilted lattice: median 0.677 m, 1.904 per cell.

### Schlesiger 2015 (PMC4711275 and Supplementary Table 1)

- schlesiger2015_1: corrected. The value is confirmed: "Field size (cm)b 42.30 ± 2.36 | 63.48 ± 2.10 | 40.79 | 65.84"; "b51 fields, 4 rats"; 50 cells. The rule was misread. Size is measured at the 20% field boundary, not at the 50% contour. Methods: "A field boundary and a contour corresponding to 50 % of the peak bin rate were drawn ... field sizes were calculated as the length between the beginning and end of the field and the area within the field boundary". The pass definition then uses the two as separate regions: "within the field boundary ... and for which the animal spent at least 200 ms within the 50 % contour". This moves Schlesiger from the narrowest rule to the common 20% rule in the sweep md's rule table.

### Hayman 2011 (PMC3166852 and supplement)

- hayman2011_1: corrected. Values confirmed: Table 1 "Major axis/height (cm) | Flat arenas | 53 | 38 | 2" and "Minor axis/ width (cm) | Flat arenas | 53 | 21 | 1"; "square or circular arenas ranging from 60-200 cm across". Strain is "22 male Lister-hooded rats". The field rule omitted the 15% threshold, which applies to all maps ("patches of at least 5 consecutive bins with firing greater than 15% of the peak rate"). Familiarity of the flat arenas is not stated.
- hayman2011_2: corrected (strain, familiarity, rule). "major axes=78+/−3cm on the pegboard ... minor axes=39+/−3cm"; 40 cells, 7 rats; pegboard 121 × 121 cm.
- hayman2011_3: corrected (strain, n_animals). "The minor axis averaged 40+/−2cm"; "Vertical field extent averaged 69+/−2cm"; 5-coil 35 ± 3 and 6-coil 45 ± 4 cm; 103 cells; helix pre-trained "until they were familiar with the apparatus"; 6 hippocampal rats on the helix.

### Ravassard 2013 (PMC4049564)

- ravassard2013_1: corrected (familiarity). "Place fields were 26% wider (p<10-10) in VR (55.8 ± 1.2 cm, n=482) compared to RW (44.3 ± 1.4 cm, n=365)"; 227 stable track-active cells; "45.5% were track active in RW"; "90% (75%) of track active cells had at least one place field"; six rats; dorsal CA1. The main text never says the rats were trained or that the track was familiar.
- ravassard2013_2: corrected (familiarity). 55.8 ± 1.2 cm, n = 482; 392 cells; 20.4%.

### Mou 2018 (PMC6160568 XML)

- mou2018_1: confirmed. "Place fields in mice had significantly shorter length (33 [25, 40] cm, N = 105 fields) that those in rats (40 [33, 48] cm, N = 60 fields"; rule "peak rate ≥3 Hz and boundaries were determined by a threshold of 10% of peak rate. Only place fields with a minimum of 5 cm"; 2.5 cm bins; "0.90 ± 0.07, N = 67 cell × active trajectory"; "12%"; novel 2 m track; 4 male Long-Evans rats; 39 cells.

### Shen 1997

- shen1997_1: unverifiable. No full text reachable (see Sources). The value rests on WebFetch fragments only. The status is changed to say so.
- shen1997_2: unverifiable, as shen1997_1. The row is the aged group (25 to 31 months), which a heat map of adult rats should leave out.

### Hollup 2001

- hollup2001_1: unverifiable. No full text reachable. The arithmetic 2*sqrt(800/π) = 31.9 cm is right, but the 0.08 m2 itself is unchecked.

### Wang 2020 (PMC7259364 and user PDF text)

- wang2020_1: corrected. Value confirmed: "The median place field size was 60° with interquartile range (IQR) 38° for CA1 fields, and was 73° with IQR 62° for CA3 fields". The track size is stated in the paper, both in the NIH manuscript and in the user's PDF text (line 1129): "circular track (76 cm O.D., 10 cm wide)". The cm conversion is therefore derived from the paper, not assumed. Centreline 66 cm, circumference 207 cm, so 60° = 34.6 cm. Rule "10% of the unit's maximum", "> 1.5 Hz", "≤ 330°", 1° bins and SD 4.3° confirmed; "~1–2 weeks of pretraining".
- wang2020_2: corrected. 73° × π × 66 / 360 = 42.05 cm, which rounds to 42.0 cm, not 42.1.

### Sharp & Green 1994 (user PDF; note rows, spot-checked)

- sharp1994_1: corrected (level). Table 1 field size "0.88 (±0.02) 0.78 (±0.04) 0.71 (±0.09)" with N 52, 34, 13; "percentage of visited pixels in which the rate was above zero"; 2.9 cm pixels; 76 cm cylinder; 21 female Long-Evans rats trained in the cylinder. The level is not "unspecified": the authors recorded "the dorsal subiculum" at "6.04-6.30 mm posterior, and 2.90-3.20 mm lateral". Kim 2012 calls 6.6 to 7.0 mm posterior "intermediate", so the label follows each paper's own usage.
- sharp1994_2: corrected (level). 0.78 ± 0.04, 34 cells.

### Jacob 2020 (PMC7545758)

- jacob2020_1: confirmed. Table 2 "Field size (cm2) | 413.4 ± 37.6 | 750.6 ± 53.7***", SHAM n = 78; rule "at least nine contiguous pixels of the non-smoothed rate map had a firing rate above the mean firing rate"; "dark grey cylinder ... 100-cm diameter"; "explored a familiar environment"; 2.5 cm bins. 2*sqrt(413.4/π) = 22.9 cm.

### Lee SM 2022 (PMC8830791)

- leesm2022_1: confirmed. "approximately 90% were single-field cells in the CA1"; boundaries "below 33% of the peak firing rate"; peak > 1 Hz and > 0.5 bits/spike; T-maze "stem, 73 × 8 cm; arms, 38 × 8 cm"; CA1 n = 270. Fig. 2F field widths may be in the paper's S1 Data file, which nobody has opened yet.
- leesm2022_2: corrected (count wording). "only about half of cells exhibited either single- or multi-fields in the subiculum". The paper does not say the rest "fired too broadly"; they simply had no field by the criteria. SUB n = 151.

## Double counting

- Kjelstrup 2008 rows 4 to 9 measure the same cells as rows 1 to 3 (autocorrelation width and population band). Only rows 1 to 3 and 10 are field widths. Flags added.
- Royer 2010 and Mizuseki 2012 very likely share dorsal CA3 cells. Both papers describe four rats with "8 independently movable wire tetrodes aimed to record from the dorsal (1 tetrode) and ventral parts", the same 1 × 2 m open field, the same 7-arm radial maze with the same water schedule and the same zigzag maze. Flags added to mizuseki2012_4 and royer2010_2. The CA3 maze rows (mizuseki2012_2, _6, _8 and royer2010_1) carry the same risk.
- Neunuebel 2012: the 8 hilar cells (row 3) may overlap the 25-cell sample (rows 1 and 2). Flag added.
- Tanni 2022 rows 4, 5 and 6 all describe environment D fields of the same cells (near wall, centre, axis widths). They are distinct measures, but a heat map should take one per environment.
- Henriksen 2010 gives three rows for one 2 m cylinder, separate cells by transverse band. That is not double counting, but it gives Henriksen three times the weight of other studies in one heat-map cell.
- Diba 2008 pools CA1 and CA3 fields under one value. Region changed to "CA1+CA3 (pooled)".

## Systematic problems

1. Familiarity was filled in as "familiar" or "trained" without support from the paper in 6 rows: Grieves (2), Hayman flat arena and pegboard (2), Ravassard (2). Three more rows understated it: Mizuseki radial and zigzag rows said "not stated" where the paper says the rats were trained.
2. Derived widths assumed a geometry or unit the paper does not support. Neunuebel's area unit cannot be resolved. One Diamantaki cell was on a linear maze. The Grieves diameter is a minimum enclosing sphere, not an equal-area diameter. In the opposite direction, Wang 2020 was flagged as an assumption although the track size is printed in the paper and in the user's own PDF text.
3. Field rules were misread or copied across. Schlesiger's size is at the 20% boundary, not the 50% contour. Mizuseki's maze rows carry the track rule. Hayman's 15% threshold was listed for the helix only.
4. Group and count labels were wrong in two places. Neunuebel's 25-cell count distribution is for all DG sites, not granule-layer tetrodes. Tanni's 15% is instantaneous co-activity, not an active-cell share.
5. The WebFetch values themselves held up. All 8 fragment rows and the 3 other WebFetch rows match the full text. The errors in those rows were in labels the summariser could not carry: the statistic in the zigzag legend, which environment a figure used, and the pooled region.

## Errors in sweep_fields_rat.md (not in the CSV)

- "Dentate gyrus, putative types": "on granule-layer and transition tetrodes, 12 of 25 active cells had one field" is wrong. The 25 cells come from all sites; granule-layer and transition tetrodes gave 5 single-field and 1 multi-field tetrode.
- Field-counts table, Tanni row: "constant proportion of active cells (15%)" under "Silent or inactive share" is a misreading (see tanni2022_4).
- Field-rule table and "Between studies": Schlesiger belongs with the 20% rules, not at "50% of peak (size taken at the 50% contour)".
- "CA3, CA2, dentate gyrus and subiculum" and "Caveats": the Wang 2020 track size is stated (76 cm O.D., 10 cm wide), so the cm values are not an assumption.
- "Large environments": the Grieves "enclosing diameter" is defined as the diameter of the minimum enclosing sphere, so "probably a maximum extent" can drop the "probably".
- "Dentate gyrus, identified granule cells": the four field sizes were not all measured in the 70 × 70 cm arena; cell 104 was on the O-maze.

## Files

- Corrections: `audit_sweep_fields_rat_corrections.csv` in this folder.
- Scripts: `/home/jackking/.claude/jobs/2c8b1c8e/tmp/audit_sweep_rat/scripts/` (`make_corrections.py`, `grieves_check.py`, `xlsx_dump.py`, `pdf2txt.py`, `bioc2txt.py`).
- Downloads: `/home/jackking/.claude/jobs/2c8b1c8e/tmp/audit_sweep_rat/dl/`.
