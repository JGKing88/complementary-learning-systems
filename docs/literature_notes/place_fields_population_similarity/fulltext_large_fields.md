# Full-text check of four large-field and long-axis sources

Checked on 2026-11-15 against the PDFs in `~/cls/place_field_tuning/`. Real neural data only. Page numbers are journal pages, except the Maurer and Nadel review, which uses the NIH manuscript's own page numbers.

How figures were read. PDF page rendering was unavailable on the cluster, so I read vector figures by extracting their drawing paths and calibrating them against the axis tick marks, and I read scanned tables by decoding the page images. Every value marked "read from figure" is approximate. For Kjelstrup Fig. 3, the marker lines the authors drew at r = 0.5 and 0.2 come out within about 0.3 m of the widths stated in the text, so distances read from that figure carry roughly ±0.3 m of error, and correlations about ±0.02.

The most important result is a correction. Kjelstrup et al. 2008 Fig. 3 is a population-vector correlation against distance for dorsal, intermediate and ventral CA3 on an 18 m track. In ventral CA3 the correlation falls gradually and reaches zero only at about 10.7 m. The report says no ventral or intermediate PV-against-distance curve exists and that no real-data kernel declines gradually across most of an environment. Both statements are wrong.

## Part 1. Verified facts

### Kjelstrup et al. 2008

Kjelstrup KB, Solstad T, Brun VH, Hafting T, Leutgeb S, Witter MP, Moser EI, Moser MB (2008). Finite scale of spatial representation in the hippocampus. Science 321:140-143. doi:10.1126/science.1157086. I read the main text only. The supplement, with figs. S1 to S8 and the SOM text, was not provided. PDF pages 1 to 4 are journal pages 140 to 143, and PDF page 1 also carries the end of the previous article.

The four strings the brief asked about all appear verbatim on p. 140: "0.98 ± 0.03 m", "3.73 ± 0.43 m", "5.52 ± 0.54 m" and "21 r a t s" (spaced out by the text layer, "in 21 rats" on the page).

| Item | Value | Page | Quote |
|---|---|---|---|
| Task and track | Rats ran back and forth on an 18 m linear track | 140 | "rats ran back and forth on an 18-meter-long linear track" |
| Why a long track | Small arenas may hide the largest representations | 140 | "Conventional recording environments may be too small to visualize the most extended hippocampal representations (figs. S2 to S4 and supporting online text)" |
| Track layout | Not described in the main text. Rate-map axes run 0 to 18 m (Figs. 1 and 2). Nothing in the main text says whether the track ran through one room or several | 140-142 | none |
| Region and rats | CA3, 21 rats | 140 | "we compared neural activity at multiple longitudinal levels of CA3 in 21 rats" |
| Recording levels | Dorsal 7 to 22%, intermediate 40 to 70%, ventral 70 to 100% of the long axis of CA3 | 140 | "(dorsal, 7 to 22%; intermediate, 40 to 70%; and ventral, 70 to 100%)" |
| Field definition | Stable contiguous region above 20% of peak rate | 140 | "spatially stable contiguous regions with firing above 20% of the peak rate" |
| Field size, dorsal | 0.98 ± 0.03 m, mean ± SEM, 111 cells, 5 rats | 140 | "from an average place field size of 0.98 ± 0.03 m in dorsal hippocampus (mean ± SEM; 111 cells, five rats)" |
| Field size, intermediate | 3.73 ± 0.43 m, 37 cells, 3 rats | 140 | "to 3.73 ± 0.43 m in intermediate hippocampus (37 cells, three rats)" |
| Field size, ventral | 5.52 ± 0.54 m, 61 cells, 4 rats | 140 | "and 5.52 ± 0.54 m in ventral hippocampus (61 cells, four rats)" |
| Fraction of the 18 m track | 5.4, 20.7 and 30.7% [derived] | 140 | none |
| Rats behind the means | 5 + 3 + 4 = 12 of the 21 rats, 209 cells [derived] | 140 | none |
| Statistics | F(2,517) = 86.5, all pairs P < 0.001. The 517 error df imply about 520 observations, more than the 209 cells, and the main text does not explain the difference [derived]. The regression has r = 0.67 and n = 111 and explains 45% of the variance | 140 | "Forty-five per cent of the variance was explained by a linear regression model [r = 0.67, n = 111, P < 0.001; (fig. S6)]" |
| What the per-level mean measures | Fig. 4 calls the single-cell measure the largest field per cell, so the means may be of each cell's largest field | 143 | "the average width of the largest place field of individual cells (mean ± SEM)" |
| Largest fields | Some fields exceeded 10 m, and the text gives no maximum | 140 | "Few fields were longer than 10 m." |
| Ventral example fields | 5 to 10 m fields in ventral CA3 | 141 | "Note 5- to 10-m-long place fields in ventral CA3" (Fig. 1 legend) |
| Largest drawn field | Cell 11581, at 83% of the long axis. In one running direction its single field ran from about 1.2 to 16.9 m, about 15.7 m or 87% of the track. In the other direction the detector split its field into 4.9 to 7.3 m and 7.5 to 17.1 m [read from figure, approximate] | 141 (Fig. 1B) | none |
| Other ventral examples | Cell 11285 at 87%, one field of about 8.4 m (1.6 to 10.0 m). Cell 11285 at 79%, about 9.3 m (5.6 to 14.9 m) in one direction [read from figure, approximate] | 141 (Fig. 1B) | none |
| Fields at the track ends | Ventral fields reached to within about 1 m of either end in the examples [read from figure]. The authors state that truncation is possible | 141 | "because fields at the end of the track may be truncated" |
| Field cleaving | The detector sometimes split one field into several | 141 | "the algorithm sometimes cleaves fields as a result of local rate variations (see examples in Fig. 1 and fig. S5)" |
| One field or several per ventral cell | Long-distance phase precession implies one field | 141 | "Precession over such long distances also implies that the active firing regions in the ventral hippocampus are part of one place field rather than multiple overlapping fields." |
| Phase precession | Present at all levels in 81 place cells. In ventral CA3 one cycle often spanned up to 10 m | 140 | "In the ventral CA3, the phase advance often occurred over distances as large as 10 m (Fig. 2, right)." |
| Precession slope | -102° ± 33° per m dorsal and -18° ± 9° per m for intermediate and ventral combined, "because of low numbers" | 140-141 | "The mean regression slope changed from –102° ± 33° per meter in dorsal hippocampus ... to –18° ± 9° per meter in intermediate and ventral hippocampus" |
| Single-cell autocorrelation | Width of the central peak at 20% of peak was 1.12 ± 0.03, 3.07 ± 0.19 and 4.10 ± 0.18 m. F(2,327) = 179.8, and a linear fit explains 53% of the variance, r = 0.73 | 140 | "(dorsal, 1.12 ± 0.03 m; intermediate, 3.07 ± 0.19 m; and ventral, 4.10 ± 0.18 m" |
| PV construction | Rate maps from multiple recordings were stacked per running direction in 5 cm bins, giving 330 PVs per direction, or 16.5 m, after the turning points were removed [16.5 m derived] | 141 | "Rate maps were stacked separately for left and right running directions, and the length of the track was segmented into 5-cm bins, giving a total of 330 population vectors per running direction after exclusion of the turning points." |
| PV cells and similarity | All pyramidal cells per level, Pearson correlation. The main text gives neither the cell count per PV set nor any rate normalization | 142 | "Linear rate maps of all pyramidal cells from each location were stacked ... Pearson's product moment correlation coefficient was calculated for each pair of population vectors" (Fig. 3 legend) |
| PV band width at r = 0.5 | 0.81 m dorsal, 3.71 m intermediate, 6.66 m ventral | 141 | "The mean half-width (at r = 0.50) was 0.81 m for dorsal CA3, 3.71 m for intermediate CA3, and 6.66 m for ventral CA3" |
| PV band width at r = 0.2 | 1.41 m dorsal, 8.65 m intermediate, 13.59 m ventral | 141 | "The mean width at r = 0.20 was 1.41 m (dorsal), 8.65 m (intermediate), and 13.59 m (ventral)." |
| What "width" means | The Fig. 3 line graphs plot correlation against signed distance from -16 to +16 m, and the marker lines at r = 0.5 and 0.2 span both sides of zero. So the stated widths are full widths across the diagonal, and the one-sided distance is half of each value. At r = 0.2 that is about 0.7, 4.3 and 6.8 m [read from figure; derived] | 142 | "Correlation as a function of spatial distance between population vectors. Horizontal lines indicate half-width (14) and 20% width." |
| Fig. 3 curves | Two curves per level, which appear to match the two running directions in the left and middle matrices. The legend does not label them | 142 | "Arrows indicate running direction; see fig. S8 for both directions." |
| Linearity | Band width rises about linearly with dorsoventral position | 141 | "the relation between position on the dorsoventral axis and width of the correlated band was approximately linear (Fig. 4)" |
| Preferred estimate | The authors trust the population width over single-field sizes | 143 | "The population estimate is more likely to reflect the true spatial scale because the algorithm for detecting individual place fields occasionally divided fields as a consequence of local rate variations." |
| Scale statement, abstract | Under 1 m dorsal to about 10 m ventral. The PDF prints "~10 meters", not "approximately 10 meters" | 140 | "The scale of representation increased almost linearly from <1 meter at the dorsal pole to ~10 meters at the ventral pole." |
| Scale statement, discussion | Same range, called a finite upper limit | 143 | "The spatial map expands approximately linearly from a scale of <1 m near the dorsal pole to ~10 m near the ventral pole (33). Within this range, the hippocampus can represent environments at multiple spatial scales and levels of detail, up to a given limit." |
| Home range | Compared with a home-range radius of 30 to 50 m | 143 | "The upper limit of ~10 m may be sufficiently large to enable the hippocampus to form detailed representations of all environments within the rat's home range, which has an average radius of about 30 to 50 m" |
| Running directions and discrete environments | Left and right runs were more correlated ventrally, but separate environments stayed orthogonal in ventral CA3. The details are in the unread supplement | 143 (note 33) | "Place fields on left and right runs on the 18-m track were more correlated in ventral CA3 than in dorsal CA3 ... For discrete environments, ventral CA3 still had strongly orthogonalized place maps (supporting online text)." |
| Earlier intermediate and ventral work | Jung 1994 and Maurer 2005 recorded at about 30 to 45% of the axis. Poucet 1994 had no histology and found dorsal-sized fields | 143 (note 29) | "Previous studies have demonstrated place fields in intermediate parts of the hippocampus, at approximately 30 to 45% of the dorsal-to-ventral axis (13, 14). Ventral place cells were also reported in another study (36) but because this study did not include histological assessment of the recording locations and because the size of the fields was similar to that of dorsal place cells, it is hard to compare the results." |

PV correlation against one-sided distance from the diagonal in Kjelstrup Fig. 3, read from the figure. Each cell gives the range over the two curves at that level. All values are approximate.

| One-sided distance | Dorsal CA3 r | Intermediate CA3 r | Ventral CA3 r |
|---|---|---|---|
| 1 m | 0.09 to 0.13 | 0.71 to 0.75 | 0.84 |
| 2 m | 0.02 to 0.07 | 0.45 to 0.51 | 0.69 to 0.70 |
| 3 m | 0.01 to 0.10 | 0.36 to 0.37 | 0.55 to 0.57 |
| 4 m | 0.05 to 0.06 | 0.24 to 0.26 | 0.43 to 0.45 |
| 5 m | 0.00 to 0.03 | 0.14 to 0.15 | 0.32 to 0.36 |
| 6 m | -0.02 to 0.02 | 0.03 to 0.06 | 0.25 to 0.28 |
| 8 m | 0.02 to 0.03 | -0.05 to -0.04 | 0.14 to 0.17 |
| 10 m | 0.02 to 0.03 | -0.08 to -0.05 | 0.04 to 0.06 |
| 12 m | -0.02 to 0.01 | -0.10 to -0.09 | -0.05 to -0.03 |
| 14 m | -0.02 to 0.03 | -0.11 to -0.09 | -0.10 to -0.04 |
| 16 m | -0.03 to 0.07 | -0.10 to -0.04 | -0.10 to -0.06 |
| First distance below r = 0.5 | 0.4 to 0.45 m | 1.8 to 2.1 m | 3.4 to 3.6 m |
| First distance below r = 0.2 | 0.7 to 0.8 m | 4.4 to 4.6 m | 7.1 to 7.2 m |
| First distance below r = 0 | Below 0.1 by 0.9 to 1.1 m, then between -0.05 and 0.13 out to 16 m, with no clean crossing | 6.8 to 7.1 m | 10.7 to 10.8 m |
| Most negative value | -0.05 | -0.11 to -0.12 | -0.07 to -0.13 |

Derived from this table, the ventral zero crossing at about 10.7 m is 65% of the 16.5 m analysed and 59% of the 18 m track. The intermediate crossing at about 7 m is 42% of 16.5 m. At both levels the crossing sits at about 1.9 times the mean field size of 5.52 and 3.73 m. The dorsal curve falls below 0.1 at about one mean field width of 0.98 m. Between 1 and 8 m the ventral curve declines smoothly, with no plateau.

Kjelstrup Fig. 4 (p. 143), read from the figure, approximate. The figure plots four dorsoventral positions, while the text pools the data into three groups. The 13% points match the text's dorsal values, but the 56, 70 and 84% points do not equal the text's intermediate or ventral means, so the figure must regroup the non-dorsal data. The legend names the third series "Spatial Autocorrelation", while the caption describes the third measure as a field width from the precession slope. The dorsal value of 1.11 m matches the text's autocorrelation width of 1.12 m, so the legend is probably right.

| Position on the long axis | Largest field width, mean ± SEM | PV half-width | Spatial autocorrelation, mean ± SEM |
|---|---|---|---|
| 13% | 0.97 ± 0.04 m | 0.80 m | 1.11 ± 0.04 m |
| 56% | 3.04 ± 0.36 m | 4.58 m | 3.09 ± 0.26 m |
| 70% | 4.18 ± 0.49 m | 6.02 m | 3.21 ± 0.19 m |
| 84% | 5.92 ± 0.57 m | 6.77 m | 4.34 ± 0.21 m |

### Muller and Kubie 1987

Muller RU, Kubie JL (1987). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. J Neurosci 7:1951-1968. doi:10.1523/JNEUROSCI.07-07-01951.1987. The PDF is a scan with an OCR layer, and journal page = 1950 + PDF page. The OCR misreads some page headers, such as 1960 as "1990" and 1968 as "1966". I checked Tables 1 and 2 against the scanned image.

| Item | Value | Page | Quote |
|---|---|---|---|
| Rats and recording site | 42 young female Long-Evans rats, with electrodes implanted above dorsal CA1. The legends label cells CA1 or CA3/4 | 1953 | "Units in this study were recorded from 42 young Long-Evans female rats." |
| Cells | 160 units, selected and not a random sample | 1953 | "The data presented here are based on observations of 160 units selected in this way. The cells investigated in detail were, thus, not a random sample" |
| Selection rule | A clear field in the small cylinder or small rectangle, with a rate above about 5 AP/s at the field centre | 1953 | "a cell's firing rate in the center of the apparent field had to be greater than about 5 action potentials (AP)/sec" |
| Place-cell share | About 65%, an impression only | 1953 | "It is our impression that about 65% of well-isolated complex-spike units behaved as place cells, but a proper estimate will require additional work." |
| Small cylinder | 76 cm diameter, 51 cm high, with a cue card over 100° of arc | 1952 | "a gray cylinder, 76 cm in diameter and 51 cm high" |
| Large cylinder | 152 cm diameter, 102 cm high, a scaled-by-2 copy | 1953 | "the large cylinder was 152 cm in diameter and 102 cm high. Thus, the large cylinder was a scaled-by-2 version of the small one." |
| Rectangles | Small 40 × 56 cm and 51 cm high. The large one is scaled by 2, so 80 × 112 cm and 102 cm high [derived] | 1953 | "The small rectangle was 40 x 56 cm and 51 cm high; the large rectangle was again a scaled-by-2 version of the small." |
| Session length | 16 min standard. Large-apparatus sessions were also 16 min "to minimize the fraction of the area unvisited" | 1953 | quoted in the value |
| Field definition | At least 9 contiguous pixels above 1.0 AP/s | 1957 | "A cell is considered to have a field in an apparatus if there is a region at least 9 pixels in area such that the firing rate in each pixel is greater than 1.0 AP/sec" |
| Cell classes | N/N means no field in either apparatus, F/N a field in one, and F/F fields in both. F/F splits into RF (related fields) and UF (unrelated fields) | 1957 | "A cell is UF if its field in one apparatus appears to be unrelated to its field in the other. By contrast, the fields of RF cells in the 2 apparatuses must bear a simple relationship to each other." |
| The 36 / 52 / 12% figures | Defined in the abstract for cells recorded in both cylinders | 1951 | "the firing fields of 36% of the cells observed in both cylinders also scaled, in the sense that the field stayed at the same angular position and at the same relative radial position. Of the cells recorded in both cylinders, 52% showed very different firing patterns in one cylinder than in the other. The remaining 12% of the cells were virtually silent in both cylinders." |
| Counts behind them | Of 25 cells, 9 were RF (36%), 9 F/N plus 4 UF made 13 (52%), and 3 were N/N (12%) [percentages derived from Table 2]. So the 52% mixes cells with a field in only one cylinder and cells with unrelated fields | 1957 | "We first consider cells that were recorded in both the small and large cylinders (n = 25 ...) ... (3 N/N units) ... (9 F/N units) ... The remaining 13 (F/F) cells had firing fields in each cylinder. Nine of the F/F cells were considered RF cells" |
| Scaling test | 9 of 13 F/F cells, all RF, had angular peaks within ±30°, binomial P < 0.001. At ±15° and ±7.5°, 5 of 13 met the criterion, P = 0.0029 and P < 0.001 | 1957 | "Of 13 F/F cells, 9 cells, all of which were RF, satisfied this criterion." |
| Rectangle scaling | 7 of 19 cells were RF | 1957 | "Figure 7, A, B, contains colored maps for 2 of the 7 RF cells; Figure 7C contains maps for 1 of the 2 UF cells." Table 2 lists 1 UF, not 2, which is an inconsistency in the paper |
| Table 2 (scan) | N/N, N/F, RF, UF, total. Large/small cylinder 3, 9, 9, 4, 25. Large/small rectangle 6, 5, 7, 1, 19. Small cylinder/small rectangle 0, 12, 1, 9, 22. Large cylinder/large rectangle 5, 8, 0, 5, 18. Small cylinder/large rectangle 1, 12, 0, 5, 18. Large cylinder/small rectangle 3, 10, 0, 7, 20. 18 cells ran in all four | 1957 | "A total of 18 cells were run in all 4 apparatuses" |
| Field size against apparatus size | Fields did not quadruple. For RF cells the mean large-to-small field size ratio was 0.87, ranging from 0.24 to 2.06 | 1957 | "If scaling were ideal, fields in large apparatuses should have 4 times the area of those in the small versions. In reality, the maximum ratio of field size in the cylinders for an individual RF cell was 2.06, and the minimum 0.24; the mean field size ratio was 0.87." |
| Sampling corrections | Equalizing time per unit area gave 1.73, and merging 2 × 2 pixels gave 2.49 | 1957 | "This raised the mean ratio of field size of the large to the small cylinder from 0.87 to 1.73 ... In this case, the mean size ratio was 2.49. We conclude that sampling problems may be obscuring an interesting relationship between field sizes in the large and small cylinders" |
| Implied area exponent | Field area grows as floor area to the power -0.10, 0.40 or 0.66 for the three ratios over a 4-fold area [derived] | 1957 | none |
| Fields per cell | No count. Figure legends show cells with more than one firing region | 1954, 1955, 1960 | "Both fields in the small cylinder have scaled counterparts in the large cylinder." (Fig. 6C, p. 1960). "a secondary area of relatively high firing diametrically opposite the main firing field" (Fig. 3, p. 1955) |
| Cylinder against rectangle | Unpredictable, but Table 2 lists 1 RF cell of 22 for the two small enclosures, against the text's "missing" | 1951, 1959 | "the RF cells are missing in the 4 cylinder-rectangle pairs ... it is impossible to predict the nature of a firing field in a cylinder from a knowledge of the cell's field in a rectangle, and vice versa." |
| Same animal, both classes | RF and UF cells occurred in the same rat and in both CA1 and CA3/4 | 1960, 1962 | "Note that 1 of the RF cells and the UF cell were recorded from the same rat." (Fig. 6). "RF and UF cells are each seen in the CA1 and CA3/4 pyramidal cell layers" (Fig. 7) |
| Card rotation | Fields followed the wall and card in 14 of 15 rotations and the card alone in 8 of 8. Mean error 3.8° ± 14.1° SD over 15 rotations, against 0° ± 4.35° SD for 20 repeated standard sessions | 1953-1955 | "The same result was obtained 14 out of the 15 times this experiment was repeated" |
| Card removal | Fields kept their size and shape but rotated to unpredictable angles. Peak rotational correlation fell to 0.680 (n = 11), against 0.884 for repeated sessions (n = 20), P < 0.01 (Table 1, scan) | 1955-1957 | "the spatial firing distribution most often appeared unchanged with regard to radial position, shape, size, and intensity, and differed from the standard session only in angular coordinate" and "cue deletion produces a measurable loss of spatial coherence, but that fields are fundamentally unchanged" |
| Table 1 against text | The table gives 0.884 for repeated sessions and 0.886 for the half-width card (N = 6). The text gives 0.876 and 0.888 and says "all 10 cells" were recorded with the half-width card | 1955, 1957 | none |
| Reversibility | Changes lasted only while the manipulation did, with 3 exceptions | 1952, 1960 | "the spatial firing patterns of single cells in the standard apparatus are generally the same before and after the rat is run in a variant" |
| Barriers | A barrier bisecting a field cut firing in 9 of 10 cells, and its effect was local. In one cell a field expanded around the barrier | 1959-1960 | "the control exerted by the barrier is local in nature" |
| Ensemble or population comparisons | None. The paper classifies single cells only and computes no population vector or ensemble similarity | whole paper | none |

### O'Keefe and Burgess 1996

O'Keefe J, Burgess N (1996). Geometric determinants of the place fields of hippocampal neurons. Nature 381:425-428. doi:10.1038/381425a0. The PDF is a scan with an OCR layer, PDF pages 1 to 4 are journal pages 425 to 428, and I checked Table 1 and the "81 of 112" sentence against the scanned image. I left out everything that concerns the paper's model, Fig. 3 except panel f, which shows recorded data.

| Item | Value | Page | Quote |
|---|---|---|---|
| Rats | 6, implanted with tetrodes. 4 were first trained in the vertical rectangle and 2 in the horizontal one | 425 | "Briefly, 6 rats were implanted with moveable 'tetrodes'" |
| Cells and fields | 28 fields from 27 complex-spike cells in CA1 and CA3, one cell with two fields in every box | 425 | "We analysed 28 place fields from 27 well-isolated hippocampal complex spike (pyramidal) cells (one cell had two separate fields in all boxes)." |
| Boxes | 61 × 61, 122 × 122 and two 61 × 122 cm boxes, with walls 61 cm high made of 122 cm planks. Trials lasted 8 min | 425 | "The SS measured 61 x 61 cm, the LS 122 x 122 cm and the two rectangles 61 x 122 cm." |
| Bins | Each bin was 1/24 of the box area, so bin sides were about 12.5, 17.6 and 24.9 cm in the small, rectangular and large boxes [derived] | 425 | "Square bins 1/24 of the area of the current box were placed around grid points" |
| Map display | Autoscaled, 5 colours of 20% of peak each. Fields with a peak below 1.0 Hz were not shown | 425 | "Fields with a peak rate of less than 1.0 Hz are not shown." |
| Single peaks | The text says 81 of 112 maps. Table 1a gives 80, of which 11 were stretched, an inconsistency in the paper | 425, 428 | "Most (81 of 112) of the firing rate maps (Fig. 1) showed one contiguous region of firing with a single peak" |
| Table 1a by box (scan) | No field in 3, 8, 2 and 3 maps for the small square, horizontal rectangle, vertical rectangle and large square. One peak in 24, 14, 23 and 19, with 1, 2, 4 and 4 of those stretched. Two peaks in 0, 5, 3 and 5. More than two in 1, 1, 0 and 1. Totals 16 none, 80 one-peak (11 stretched), 13 two-peak and 3 multi-peak, out of 112 | 428 | none |
| Stretch and split definitions | "Stretched" was judged by eye. "2 peaks" meant two regions separated by at least half the field width | 428 | "'stretched' indicates singly peaked fields judged by eye to be at least as stretched as that in the LS in Fig. 1d; '2 peaks' indicates fields with two regions of elevated firing separated by at least half of the total width of the field" |
| Second peaks | 9 of 28 fields | 426 | "Overall, 9 of 28 fields showed a second peak in one or more of the larger boxes." |
| More than two components | One cell recorded in the 4 standard boxes and 8 intermediate boxes split into more than 2 components as the box grew (Fig. 3f) | 427 | "a place field is pulled into more than 2 components as the box expands (the place cell in f was recorded in the 4 canonical boxes and 8 boxes of intermediate shape)" |
| Field size against box size | Not quantified. No field areas or lengths appear in the paper | whole paper | none |
| Peak location | 38 of 56 field-by-axis determinants were a fixed distance to a wall or a fixed proportion between walls | 426 | "The locations of the peak firing rate in most of these place fields are determined by fixed distances or proportions of distances to a box wall in the directions defined by the axes of the box (38 of 56 determinants; see Table 1b)." |
| Distances against proportions | Table 1b gives, along x, west wall 8, east wall 4, proportion 7 and unknown 9. Along y it gives south wall 5, north wall 11, proportion 3 and unknown 9. So 28 of the 38 determinants were distances to a wall and 10 were proportions [derived] | 428 | none |
| Near-wall fields | 17 of 56 cases qualified both as a distance and as a proportion, all for fields near a wall | 428 | "If more than one measure qualified as a determinant (17 of 56 cases, all a distance and a proportion for a field near to a box wall)" |
| Illustrated distances | 3.5 ± 0.5 and 2.7 ± 2.1 cm from the west and south walls (Fig. 1b). 24.0 ± 6.4 and 11.7 ± 1.4 cm from west and north (Fig. 1c). 9.4 ± 3.3 cm from west with a north-south proportion of 0.64 ± 0.066 (Fig. 1d). 6.7 ± 1.8 cm from north (Fig. 2a). 24.9 ± 3.9 and 35.5 ± 6.2 cm from west and north (Fig. 2b). 15.1 ± 0.8 and 36.5 ± 4.0 cm from west and north (Fig. 2e) | 425-426 | "Fields B and C occur at fixed distances from the closest two walls, while field D occurs at a fixed distance from the west wall and at a fixed ratio of the distance between the north and south walls." |
| Room frame | 3 of 28 cells had one determinant fixed to the room | 426 | "3 of the 28 cells clearly had a determinant in one direction that was fixed to some aspect of the laboratory frame" |
| Field presence | 14 of 28 cells fired in all 4 boxes, 9 in 3 and 5 in 2. The pairs of boxes always differed in one dimension only | 426 | "Of 28 cells, 5 had fields in 2 of the 4 boxes, 9 had fields in 3, and 14 had fields in all 4." |
| Rate and box width | In one cell, the peak rate fell steadily as the box widened until the field vanished (Fig. 2e) | 426 | "the cell's peak firing rate decreases monotonically with the X dimension of the box (while the Y dimension is held constant), until the rate is so low that the field disappears" |
| Direction of travel | In 22 of 24 stretched or double-peaked maps, from 14 of 15 cells, the peak location depended on direction. In 13, from 9 cells, it shifted toward the wall the rat was leaving | 426 | "Many of the stretched and doubly peaked fields (22 of 24, from 14 of 15 cells) indicated a dramatic difference in the location of the peak firing rate according to the rat's direction of travel." |
| Training box | There was no clear evidence that fields were most compact in the first-trained box | 426 | "We examined the hypothesis that fields should be most compact in the box shape in which the animal was first trained, but found no clear evidence for this." |
| Local cues | Swapping the wall planks or rotating the floor left fields unchanged | 428 | "Interchanging the planks that form the walls of a box, or rotating the box floor, did not alter the place field" |

### Maurer and Nadel review

Maurer AP, Nadel L (2021). The continuity of context: a role for the hippocampus. Trends Cogn Sci 25(3):187-199. doi:10.1016/j.tics.2020.12.007. I read the NIH author manuscript (nihms-1842769). Its page numbers match the PDF pages, and the journal page range is 187 to 199.

The review gives no measured rate of population-vector decorrelation along the dorsoventral axis, no ventral field size and no data-backed claim that population similarity spans large distances. It does not cite Kjelstrup 2008. Its statements on these points are listed below.

| Statement | Page | Quote | What it cites |
|---|---|---|---|
| Active fraction | 2 | "Approximately 5% of neurons are active at any single location in space [23]." | [23] Wilson & McNaughton 1993 |
| Population roll-over with movement, no rate given | 2 | "As the rat moves, the population of active cells 'rolls over', with some cells becoming silent and others active [24,25]." | [24] Gothard et al. 2001; [25] Maurer et al. 2005. This is the review's only citation of Maurer 2005 |
| Field size grows toward ventral, no numbers | 3 | "the size of the place fields systematically increases as one moves from dorsal to ventral in the rat hippocampus [30]" | [30] Jung et al. 1994 only |
| Morph between contexts | 3 | "when environmental contexts are slowly morphed from one to another, hippocampal ensemble activity smoothly changes between these contexts [17]" | [17] Leutgeb JK et al. 2005, Neuron |
| Large anterior fields | 4 | "the anterior end of the hippocampus has large 'fields'" | No citation |
| Slower ventral change | 19-20 (Fig. 2 legend) | "place cell activity will change more slowly in the ventral region" and "ventral hippocampal population activity shows little change" | No data. The legend presents this as the consequence of the authors' own proposal ("Operationalizing this model as a function of spatial movement") |
| Percentages on p. 4 | 4 | "a threshold of 5% change", "will 'smooth over' 50% (let's say)", "only 2.5% changed", "a 10% change" | Illustrative values, not measurements. Do not cite them as data |

## Part 2. Report changes

The report is `docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md`. I checked that each old text below occurs exactly once in it. Line numbers are those of the file on 2026-11-15. The "New" texts follow the report's tag conventions and add "read from figure" for figure-derived values. I suggest replacing the Kjelstrup links with the DOI, https://doi.org/10.1126/science.1157086, and removing the Bohrium aggregator link. Keeping the Europe PMC abstract link is also fine if an open link is preferred.

### Section "Hippocampal fields stretch far while population similarity stays local" (opening summary, line 3)

1. Correction and upgrade. The ventral CA3 numbers are now full text.

Old:
```
Fields covering a third or more of an environment appear in four places: ventral rat CA3, with a scale of about 10 m on an 18 m track, roughly 56%, known only from the abstract;
```
New:
```
Fields covering a third or more of an environment appear in four places: ventral rat CA3 on an 18 m track, where fields averaged 5.52 m, or 31%, few exceeded 10 m, and one drawn field ran about 15.7 m, or 87%;
```

2. Correction. A ventral and intermediate PV-against-distance curve exists (Kjelstrup Fig. 3).

Old:
```
**On the second question the answer is no.** Every published curve of population-vector similarity against distance comes from dorsal CA1, CA3, subiculum or retrosplenial cortex in environments no longer than a 4 m loop. Where widths are reported, the curve falls to near zero within about one field width, 30 to 40 cm or 15 to 22% of a 1 to 2 m environment, and then stays flat. Hints of broader kernels come only from altered self-motion, an abstract-only comparison of dorsal and middle CA1, ventral single-cell scales that no one has turned into a population curve, and human fMRI of remembered distances. No one has published this curve for a larger environment, although two public datasets, the 48 m track and a 3.5 × 2.5 m arena, would allow it.
```
New:
```
**On the second question the answer is yes for ventral and intermediate CA3 on one 18 m track, and no wherever else it has been measured.** Kjelstrup et al. plotted PV correlation against distance at three levels of rat CA3. Dorsal correlation fell below 0.1 within about 1 m, but intermediate correlation reached zero only at about 7 m and ventral correlation at about 10.7 m, about 65% of the 16.5 m analysed, passing 0.44 at 4 m and 0.15 at 8 m. Every other published curve comes from dorsal CA1, CA3, subiculum or retrosplenial cortex in environments no longer than a 4 m loop, and where widths are reported it falls to near zero within about one field width, 30 to 40 cm or 15 to 22% of a 1 to 2 m environment, and then stays flat. Other hints of broader kernels come only from altered self-motion, an abstract-only comparison of dorsal and middle CA1, and human fMRI of remembered distances. No one has published this curve for dorsal hippocampus in an environment longer than a 4 m loop, or for any region in an environment longer than 18 m, although two public datasets, the 48 m track and a 3.5 × 2.5 m arena, would allow it.
```

### Section "Very large fields exist, but few cover much of their environment" (lines 9 to 37)

3. Upgrade from [abstract] to [full text] (line 9).

Old:
```
The ventral pole of rat CA3 is the one rodent site with a reported scale near 10 m, but only the abstract confirms it [abstract] ([Kjelstrup et al. 2008](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:18599792&resultType=core&format=json)).
```
New:
```
Ventral rat CA3 is the one rodent site with fields near 10 m. On an 18 m track, ventral fields averaged 5.52 ± 0.54 m, the authors write that "few fields were longer than 10 m", and one ventral field drawn in the paper ran about 15.7 m [full text; 15.7 m read from figure] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086)).
```

4. Correction (table row, line 15). The "about 10 m scale" is the authors' summary, not a field measure. Replace the row with the largest drawn field.

Old:
```
| Rat CA3, ventral pole | 18 m linear track | "Scale" about 10 m | about 56% | abstract |
```
New:
```
| Rat CA3, one ventral cell at 83% of the long axis | 18 m linear track | Single field about 15.7 m in one running direction | about 87% | full text, read from figure |
```

5. Confirmed and upgraded from [unverified] (table row, line 16).

Old:
```
| Rat CA3, ventral group | 18 m linear track | Mean 5.52 ± 0.54 m | 31% | unverified, aggregator only |
```
New:
```
| Rat CA3, ventral group, 70 to 100% of the long axis | 18 m linear track | Mean 5.52 ± 0.54 m, 61 cells, 4 rats | 31% | full text, derived |
```
Optional extra row, placed by share next to the 22% bat row: `| Rat CA3, intermediate group, 40 to 70% of the long axis | 18 m linear track | Mean 3.73 ± 0.43 m, 37 cells, 3 rats | 21% | full text, derived |`

6. Source list under the table (line 33). Drop the aggregator.

Old:
```
[Kjelstrup et al. 2008](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:18599792&resultType=core&format=json); [Bohrium aggregator page](https://www.bohrium.com/paper-details/finite-scale-of-spatial-representation-in-the-hippocampus/811915281508073473-7984); [Courellis
```
New:
```
[Kjelstrup et al. 2008, Fig. 1B](https://doi.org/10.1126/science.1157086); [Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086); [Courellis
```

7. Correction (line 37). The per-level means are in the paper itself, and the authors raise truncation themselves.

Old:
```
The ventral 10 m figure is bounded by an 18 m track, so the ventral scale in a longer space is unknown, and the per-level means quoted for that study, 0.98, 3.73 and 5.52 m, appear only on an aggregator page, though a commentary independently reports the dorsal mean of 98 cm [secondary] ([Hasselmo 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2590634)).
```
New:
```
The ventral fields are bounded by an 18 m track, and the authors note that "fields at the end of the track may be truncated", so the ventral scale in a longer space is unknown [full text] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086)).
```
The bold sentence earlier in line 37, "No source reports a single dorsal field covering a third or more of an environment larger than about 6 m or 2 m²", is confirmed. Kjelstrup's dorsal mean is 0.98 m, 5% of 18 m. The sentence "Fields that large relative to the space are reported only in small arenas, in ventral rodent hippocampus, ..." is also confirmed.

### Section "No measured population kernel declines gradually across a whole environment" (lines 39 to 45)

8. Correction (heading, line 39).

Old:
```
## No measured population kernel declines gradually across a whole environment
```
New:
```
## Only ventral CA3 shows a population kernel that declines across most of an environment
```

9. Correction (line 41, first sentence).

Old:
```
Population-vector (PV) similarity has been plotted against the distance between two positions in only a few real datasets, and each one shows a peak that falls to near zero within about one field width, then a flat tail.
```
New:
```
Population-vector (PV) similarity has been plotted against the distance between two positions in only a few real datasets. In dorsal hippocampus each one shows a peak that falls to near zero within about one field width, then a flat tail.
```

10. Correction (line 41, last sentence). It holds for the subiculum but not for CA3 along the long axis.

Old:
```
Broad single-cell tuning does not, on this evidence, produce a broad population kernel.
```
New:
```
In the subiculum, then, broad single-cell tuning did not produce a broad population kernel, but in ventral CA3 broader fields did come with a broader one, as the next paragraph shows.
```

11. Correction and new finding (line 43, first sentence). Replace it with the Kjelstrup PV result and keep the original sentence at the end.

Old:
```
The candidate exceptions are weak or indirect.
```
New:
```
One exception is a direct measurement. On an 18 m track, Kjelstrup et al. stacked the rate maps of all CA3 pyramidal cells recorded at each longitudinal level, pooled across recordings, in 5 cm bins, giving 330 PVs per running direction after the turning points were removed. They then plotted the Pearson correlation between PVs against their separation. The correlated band was 0.81, 3.71 and 6.66 m wide at r = 0.5 and 1.41, 8.65 and 13.59 m wide at r = 0.2 in dorsal, intermediate and ventral CA3 [full text] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086)). The figure's marker lines span both sides of zero distance, so these are full widths across the diagonal. Read from the figure, ventral correlation was about 0.70 at 2 m, 0.44 at 4 m, 0.27 at 6 m and 0.15 at 8 m, crossed zero at about 10.7 m and stayed between -0.03 and -0.13 out to 16 m. Intermediate correlation crossed zero at about 7 m, and dorsal correlation fell below 0.1 by about 1 m and then fluctuated near zero [full text, read from figure, approximate] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086)). The ventral fall to zero spans about 65% of the 16.5 m analysed, and at both the intermediate and the ventral level the zero crossing sits at about twice the mean field size [derived]. The main text does not say whether rates were normalized or how many cells entered each level's vectors, and the authors call this population width their most reliable estimate of scale. The remaining candidate exceptions are weak or indirect.
```

12. Correction (line 43). One of the four cited papers did measure ventral decorrelation.

Old:
```
The claim that ventral PVs "remain correlated over much larger distances" traces to an introduction citing four papers, of which only Maurer 2005 measured population decorrelation, and that comparison was dorsal against middle, not ventral [full text] ([Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)).
```
New:
```
The claim that ventral PVs "remain correlated over much larger distances" comes from an introduction citing four papers [full text] ([Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)). Of these, Kjelstrup et al. measured it directly in ventral CA3, as above, and Maurer et al. compared the dorsal pole with the middle hippocampus.
```

13. Correction (line 45).

Old:
```
The answer to the second question is that **no real-data evidence shows a kernel declining gradually across a full environment or most of one under normal locomotion**, and the test has not been run where it would matter most.
```
New:
```
The answer to the second question is that **one real-data curve declines gradually across most of an environment under normal locomotion, in intermediate and ventral rat CA3 on an 18 m track, and no dorsal curve does**. The test has not been run on dorsal recordings from environments longer than a 4 m loop, or in any environment longer than 18 m.
```

### Section "Twelve measures at a glance" (lines 52, 54, 59)

14. Upgrade (line 52, end of the "Field size, fraction of environment" row).

Old:
```
ventral CA3 about 56% of 18 m; marmoset 58% of 2.44 m | Fenton 2008; Diba 2008; Eliav 2021; Harland 2021; Kjelstrup 2008; Courellis 2019 | Full text, except Kjelstrup (abstract) and its per-level means (unverified) |
```
New:
```
ventral CA3 mean 31% of 18 m, one drawn field about 87%; marmoset 58% of 2.44 m | Fenton 2008; Diba 2008; Eliav 2021; Harland 2021; Kjelstrup 2008; Courellis 2019 | Full text; Kjelstrup example field read from figure |
```

15. Upgrade (line 54, "Dorsoventral gradient" row). Two separate replacements.

Old:
```
| Dorsoventral gradient | Field scale and place-cell share along the long axis | Rat CA3 scale under 1 m dorsal to about 10 m ventral;
```
New:
```
| Dorsoventral gradient | Field scale and place-cell share along the long axis | Rat CA3 mean field 0.98, 3.73 and 5.52 m and PV band width at r = 0.5 0.81, 3.71 and 6.66 m from dorsal to ventral on an 18 m track;
```
Old:
```
| Kjelstrup 2008; Royer 2010; Keinath 2014; Jung 1994; Chockanathan 2021 | Mixed: Kjelstrup, Jung, Maurer abstract; Royer, Keinath, Chockanathan full text |
```
New:
```
| Kjelstrup 2008; Royer 2010; Keinath 2014; Jung 1994; Chockanathan 2021 | Mixed: Jung, Maurer abstract; Kjelstrup, Royer, Keinath, Chockanathan full text |
```

16. Correction (line 59, "PV similarity against distance" row).

Old:
```
| PV similarity against distance | Correlation of PVs at two positions in one environment | Dorsal CA1 and CA3 near zero by 30 to 40 cm, 15 to 22% of 1 to 2 m spaces; RSC about 35 cm; subiculum narrower than CA1; no curve for spaces above a 4 m loop or for ventral hippocampus | Green 2022; Wang 2020; Battaglia 2004; Mao 2017; Ledergerber 2021 | Full text, widths often figure only; Maurer abstract |
```
New:
```
| PV similarity against distance | Correlation of PVs at two positions in one environment | Dorsal CA1 and CA3 near zero by 30 to 40 cm, 15 to 22% of 1 to 2 m spaces, and dorsal CA3 below 0.1 by about 1 m on an 18 m track; intermediate and ventral CA3 on that track reach zero at about 7 and 10.7 m, 42 and 65% of the 16.5 m analysed; RSC about 35 cm; subiculum narrower than CA1; no dorsal curve for spaces above a 4 m loop | Green 2022; Wang 2020; Battaglia 2004; Mao 2017; Ledergerber 2021; Kjelstrup 2008 | Full text, widths often figure only; Kjelstrup zero crossings read from figure; Maurer abstract |
```

### Section "Larger spaces add fields faster than they enlarge them" (lines 100 to 110)

17. Upgrade from [abstract] to [full text] (line 100, Muller and Kubie). The 36, 52 and 12% figures are confirmed. "Changed their firing completely" is looser than the paper. The 52% combines 9 cells with a field in only one cylinder and 4 with unrelated fields.

Old:
```
When a 76 cm cylinder was doubled in diameter and height, which quadruples floor area, **36% of cells recorded in both kept a field at the same angular and relative radial position, 52% changed their firing completely and 12% were nearly silent in both**. The abstract does not say whether field area quadrupled [abstract] ([Muller & Kubie 1987](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)).
```
New:
```
When a 76 cm cylinder was doubled in diameter and height, which quadruples floor area, **36% of 25 cells recorded in both kept a field at the same angular and relative radial position, 52% had a field in only one cylinder or unrelated fields, and 12% were nearly silent in both**, counts of 9, 13 and 3. Cells had been chosen for a clear field in the small cylinder or small rectangle. Field area did not quadruple. For the 9 scaling cells the large-to-small ratio of field size averaged 0.87 and ranged from 0.24 to 2.06. It rose to 1.73 when dwell time per unit area was equalized and to 2.49 when 2 × 2 pixels were merged, and the authors concluded that "sampling problems may be obscuring an interesting relationship between field sizes". Doubling a 40 × 56 cm rectangle gave scaled fields in 7 of 19 cells [full text] ([Muller & Kubie 1987](https://doi.org/10.1523/JNEUROSCI.07-07-01951.1987)).
```

18. Upgrade from [abstract] and [secondary] to [full text] (line 100, O'Keefe and Burgess). The box sizes and the Barry et al. summary are confirmed, so the secondary citations can go.

Old:
```
O'Keefe and Burgess recorded the same cells in four rectangular boxes differing only in the length of one or both sides [abstract] ([O'Keefe & Burgess 1996](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:8632799&resultType=core&format=json)). A later paper gives the boxes as 61 to 122 cm on a side [secondary] ([Keinath, Epstein & Balasubramanian 2018](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6203432/fullTextXML)), and a review summarises the result as peaks keeping "a constant position to the nearest walls" while "several fields were stretched along the axes of the environment, with some becoming bimodal in the larger environments" [secondary] ([Barry et al. 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC2677716/)).
```
New:
```
O'Keefe and Burgess recorded 28 fields from 27 CA1 and CA3 complex-spike cells in 6 rats in four boxes, 61 × 61, 61 × 122, 122 × 61 and 122 × 122 cm, with 61 cm walls. In 38 of 56 field-by-axis cases the peak sat at a fixed distance from one wall or a fixed proportion of the distance between two, and 28 of those 38 were distances [full text; split derived from Table 1b]. Peaks in the illustrated cells sat 2.7 to 36.5 cm from the walls that fixed them. Of 112 rate maps, 16 had no field, 69 had one compact peak, 11 had one stretched peak, 13 had two peaks and 3 had more, and 9 of 28 fields showed a second peak in at least one larger box. In 22 of 24 stretched or double-peaked maps, from 14 of 15 cells, the peak moved with running direction. The paper reports no field areas, and its bins were 1/24 of each box's area, so bin size grew with the box [full text] ([O'Keefe & Burgess 1996](https://doi.org/10.1038/381425a0)).
```

19. New derived value (line 106). Append after the sentence ending "...from Diba and Buzsáki [derived]."

Old:
```
Implied growth exponents are 0.36 for bat field length against tunnel length,
```
Keep that sentence unchanged and add after it:
```
Muller and Kubie's fourfold larger cylinder gives field-area exponents of -0.10, 0.40 or 0.66, depending on which of their three sampling corrections is used [derived].
```

20. Citation tags (line 110).

Old:
```
J Neurosci 7:1951-1968. doi:10.1523/JNEUROSCI.07-07-01951.1987 [abstract].
```
New:
```
J Neurosci 7:1951-1968. doi:10.1523/JNEUROSCI.07-07-01951.1987 [full text].
```
Old:
```
O'Keefe J, Burgess N (1996). Geometric determinants of the place fields of hippocampal neurons. Nature 381:425-428. doi:10.1038/381425a0 [abstract; results via secondary sources]. Keinath AT, Epstein RA, Balasubramanian V (2018). eLife 7:e38169. doi:10.7554/eLife.38169 [cited only for box sizes, secondary]. Barry C et al. (2006). Rev Neurosci 17:71-97. PMID 16703944 [secondary].
```
New, if change 18 is applied, since Keinath 2018 and Barry 2006 are cited only in the sentence it replaces:
```
O'Keefe J, Burgess N (1996). Geometric determinants of the place fields of hippocampal neurons. Nature 381:425-428. doi:10.1038/381425a0 [full text].
```
Line 108's statement that the Kjelstrup track "passed through several small rooms" stays as Eliav et al.'s claim. The Kjelstrup main text does not describe the layout.

### Section "Field scale rises about tenfold toward the ventral pole, on thin evidence" (lines 112 to 124)

21. Heading (line 112). Mean field size rises 5.6-fold and PV band width 8.2-fold [derived], while the authors' summary is "<1 m" to "~10 m".

Old:
```
## Field scale rises about tenfold toward the ventral pole, on thin evidence
```
New:
```
## Field scale rises five- to eightfold toward the ventral pole, in one study
```

22. Upgrade and rewrite (line 114, whole paragraph). It runs from "The dorsoventral gradient rests on one study whose numbers are mostly unverifiable." to "...so the ventral scale in a longer open space is unknown [inference]." The report's quotation of the abstract, "approximately 10 meters", should read "~10 meters" in the PDF. If this rewrite is used, the Solstad 2009 and Saxena et al. 2018 entries in line 124 are no longer cited and can go. The main text supports them, with one long field per ventral cell up to about 10 m, but they are no longer needed.

New:
```
The dorsoventral gradient rests on one study. Kjelstrup et al. recorded CA3 in 21 rats running back and forth on an 18 m linear track and grouped sites by position along the long axis, dorsal at 7 to 22%, intermediate at 40 to 70% and ventral at 70 to 100%. Fields were "spatially stable contiguous regions with firing above 20% of the peak rate". Mean field size, ± SEM, was 0.98 ± 0.03 m for 111 dorsal cells from 5 rats, 3.73 ± 0.43 m for 37 intermediate cells from 3 rats and 5.52 ± 0.54 m for 61 ventral cells from 4 rats, or 5, 21 and 31% of the track [full text, fractions derived]. "Few fields were longer than 10 m", and one ventral cell drawn in the paper had a single field from about 1.2 to 16.9 m in one running direction, 87% of the track [full text; length read from figure, approximate]. The central peak of the single-cell spatial autocorrelation, measured at 20% of its height, was 1.12, 3.07 and 4.10 m wide at the three levels. Theta phase precession appeared at every level, and in ventral CA3 "the phase advance often occurred over distances as large as 10 m", which the authors read as showing that ventral firing regions "are part of one place field rather than multiple overlapping fields" [full text] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086)). Their summary is that the map "expands approximately linearly from a scale of <1 m near the dorsal pole to ~10 m near the ventral pole". They trust the population estimate, given in the similarity section below, more than single-field sizes, because their detector "occasionally divided fields as a consequence of local rate variations" and because "fields at the end of the track may be truncated" [full text]. Left-run and right-run fields were more correlated in ventral than in dorsal CA3, yet for discrete environments ventral CA3 "still had strongly orthogonalized place maps", a result the paper leaves to its supplement, which was not read [full text, note 33]. A same-year commentary describes ventral neurons that "fire almost everywhere in an environment in one room, and nowhere in an identical environment in another room" [secondary] ([Hasselmo 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2590634)). The main text does not describe the track's layout, which Eliav et al. say passed through several small rooms [full text] ([Eliav et al. 2021](https://www.weizmann.ac.il/brain-sciences/labs/ulanovsky/sites/brain-sciences.labs.ulanovsky/files/2024-11/Eliav2021.pdf)). A later paper reports that fewer than 10% of ventral CA3 cells in that study showed reliable phase precession [secondary] ([Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)), a share the main text does not give. Because the track was only 18 m long, a 10 m ventral field cannot be independent of the track's ends, so the ventral scale in a longer open space is unknown [inference].
```

23. New finding (line 116, Poucet sentence). Append Kjelstrup's comment.

Old:
```
In a cylinder, ventral cells had positional firing "very similar" to dorsal place cells [abstract] ([Poucet, Thinus-Blanc & Muller 1994](https://doi.org/10.1097/00001756-199410270-00014)).
```
New:
```
In a cylinder, ventral cells had positional firing "very similar" to dorsal place cells [abstract] ([Poucet, Thinus-Blanc & Muller 1994](https://doi.org/10.1097/00001756-199410270-00014)). Kjelstrup et al. point out that this study did not verify recording sites histologically and found fields of dorsal size, so "it is hard to compare the results" [full text] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086), note 29).
```

24. New finding (line 116, Maurer sentence). The recording level comes via Kjelstrup.

Old:
```
Recording CA1 at the extreme dorsal pole and in the middle hippocampus, Maurer et al. found smaller fields and higher rates dorsally and a falling probability of a field toward the temporal pole [abstract] ([Maurer et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16145692/)).
```
New:
```
Recording CA1 at the extreme dorsal pole and in the middle hippocampus, Maurer et al. found smaller fields and higher rates dorsally and a falling probability of a field toward the temporal pole [abstract] ([Maurer et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16145692/)). Kjelstrup et al. place the intermediate recordings of Maurer et al. and of Jung et al. at about 30 to 45% of the dorsal-to-ventral axis [secondary] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086), note 29).
```

25. New finding (line 122).

Old:
```
Ventral field counts are lower per environment in Maurer, equal in Keinath and spread over many arms in Royer.
```
New:
```
Ventral field counts are lower per environment in Maurer, equal in Keinath and spread over many arms in Royer, while on the 18 m track ventral phase precession implies one long field per cell in Kjelstrup.
```

26. Citation tag (line 124).

Old:
```
Finite scale of spatial representation in the hippocampus. Science 321:140-143. doi:10.1126/science.1157086 [abstract; per-level means unverified].
```
New:
```
Finite scale of spatial representation in the hippocampus. Science 321:140-143. doi:10.1126/science.1157086 [full text, main text; supplement not read].
```

### Section "Every measured similarity curve reaches baseline within one field width" (lines 205 to 237)

27. Correction (heading, line 205). Ventral and intermediate CA3 reach zero at about 1.9 mean field widths.

Old:
```
## Every measured similarity curve reaches baseline within one field width
```
New:
```
## Dorsal similarity curves reach baseline within one field width, ventral CA3 within about two
```

28. New table row (line 217). Insert after the Maurer et al. 2005 row:

Old (anchor row, unchanged):
```
| Maurer et al. 2005 | Rat CA1, dorsal pole and middle | Track | Not available | Decorrelation over shorter distances dorsally | unknown | abstract |
```
Add below it:
```
| Kjelstrup et al. 2008 | Rat CA3 at dorsal, intermediate and ventral levels, 21 rats | 18 m linear track, 16.5 m analysed | Rate maps of all pyramidal cells per level stacked across recordings, 5 cm bins, 330 PVs per running direction, Pearson correlation | Full band width 0.81, 3.71, 6.66 m at r = 0.5 and 1.41, 8.65, 13.59 m at r = 0.2; ventral r about 0.44 at 4 m and 0.15 at 8 m, zero at about 10.7 m; intermediate zero at about 7 m; dorsal below 0.1 by about 1 m | ventral zero at about 65%, intermediate about 42% | full text; curve values read from figure |
```
In the source list on line 221, add `[Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086);` after the Maurer et al. 2005 link. Its anchor is `[Terrazas et al. 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC6725460); [Maurer et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16145692/); [Rubin`.

29. Correction (line 225), four separate replacements.

Old:
```
The case for broader kernels in ventral hippocampus is an extrapolation.
```
New:
```
The case for broader kernels in ventral hippocampus rests on one measurement.
```
Old:
```
Only Maurer measured population decorrelation, and it compared the dorsal pole with the middle hippocampus [abstract] ([Maurer et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16145692/)).
```
New:
```
Two of them measured population decorrelation. Kjelstrup et al. measured it in intermediate and ventral CA3 on an 18 m track, as in the table above [full text] ([Kjelstrup et al. 2008](https://doi.org/10.1126/science.1157086)), and Maurer et al. compared the dorsal pole with the middle hippocampus [abstract] ([Maurer et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16145692/)).
```
Old:
```
returned no such study, and no ventral or intermediate PV-versus-distance curve was found in any species [full text search]
```
New:
```
returned no other such study, and no other ventral or intermediate PV-versus-distance curve was found in any species [full text search]
```
Old:
```
A broad ventral kernel, if measured, could reflect category membership as much as metric distance [inference].
```
New:
```
The broad ventral kernel on the 18 m track could reflect category membership as much as metric distance, since the whole track was one connected place [inference].
```
Optional addition after the Keinath sentence. A 2021 review says ventral place-cell activity "will change more slowly", but it presents this as a consequence of the authors' own proposal and cites no measurement. For field size it cites only Jung et al. 1994 [full text] ([Maurer & Nadel 2021](https://doi.org/10.1016/j.tics.2020.12.007)).

### Section "CA3 maps for different rooms overlap no more than random draws" (lines 241, 255)

30. Upgrade from [abstract] to [full text] (line 241). Quotes and claims confirmed (pp. 1951, 1955-1956, 1959), with numbers added.

Old:
```
Firing in a circular enclosure "could not be predicted from a knowledge of the firing pattern" in a rectangular one, and removing the cue card rotated fields to unpredictable angles without changing their size or shape [abstract] ([Muller & Kubie 1987](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)).
```
New:
```
Firing in a circular enclosure "could not be predicted from a knowledge of the firing pattern" in a rectangular one. Of 18 to 22 cells recorded in each cylinder-rectangle pair, only 1, in the pair of small enclosures, kept a related field, although the text calls such cells "missing". Removing the cue card rotated fields to unpredictable angles while their size and shape "most often appeared unchanged", but peak rotational correlation with the standard session fell from 0.884 for repeated sessions to 0.680, P < 0.01 [full text] ([Muller & Kubie 1987](https://doi.org/10.1523/JNEUROSCI.07-07-01951.1987)).
```

31. Citation tag (line 255).

Old:
```
**Full citations.** Muller & Kubie 1987 as above [abstract].
```
New:
```
**Full citations.** Muller & Kubie 1987 as above [full text].
```
Optional addition to this section. Kjelstrup et al. report that ventral CA3 "still had strongly orthogonalized place maps" for discrete environments, though left and right runs on one track were more correlated ventrally [full text, note 33; details in the unread supplement].

### Section "Conclusion" (lines 344, 346)

32. Correction (line 344), two replacements.

Old:
```
a ventral "scale", a sparseness index
```
New:
```
a ventral field length on a track barely longer than the field, a sparseness index
```
Old:
```
Broad single-unit tuning does not imply a broad population kernel in these data, so single-cell field size is a poor proxy for how fast the population code changes with distance.
```
New:
```
Broad single-unit tuning does not imply a broad population kernel in the subiculum, while along the CA3 long axis field size and kernel width grew together, so single-cell field size tracks the population kernel in one region and fails in another.
```

33. Correction (line 346), two replacements.

Old:
```
No direct evidence supports a population kernel that declines gradually across a whole environment in rodent or bat hippocampus under normal movement.
```
New:
```
One study supports a population kernel that declines gradually across most of an environment under normal movement, in intermediate and ventral rat CA3 on an 18 m track, where PV correlation reached zero only after about 7 and 10.7 m. No dorsal or bat recording shows one.
```
Old:
```
Long-track recordings in ventral hippocampus and large-environment recordings in subiculum would test the two regions where broad kernels are most often assumed. Until those exist, a statement that hippocampal population similarity falls off gradually across metres is an extrapolation from single-cell scale, not an observation.
```
New:
```
Ventral recordings on tracks longer than 18 m, analysed away from the ends, would show whether the ventral kernel is set by the region or by the track, and large-environment recordings in subiculum would test the other region where broad kernels are assumed. Until those exist, gradual fall-off across metres is an observation for intermediate and ventral CA3 on one track and an extrapolation everywhere else.
```

### Confirmed with no change needed

- Line 37, Hasselmo 2008 gives the dorsal mean as 98 cm. This matches 0.98 m on p. 140. Change 7 drops the sentence, and change 22 keeps Hasselmo for the room statement.
- Line 100, the companion-paper quote "a single cell may have more than one field" is from Muller, Kubie & Ranck 1987, not checked here. Muller & Kubie 1987 shows two-field cells in its figure legends but gives no count.
- Line 108, Eliav et al.'s description of the Kjelstrup track as several small rooms cannot be checked against the Kjelstrup main text.
- Line 116, the Jung et al. 1994 abstract statements are not covered by these four papers. Kjelstrup note 29 places that study's intermediate recordings at about 30 to 45% of the axis.

## Part 3. New findings worth adding, by report section

- "No measured population kernel declines gradually across a whole environment" (renamed in change 8). Kjelstrup Fig. 3 PV correlation against distance for three CA3 levels, with the figure-read curve table in Part 1. This is the main addition, covered by changes 2, 11 and 13.
- "Every measured similarity curve reaches baseline within one field width" (renamed in change 27). The Kjelstrup table row (change 28), and the fact that the far tail of the intermediate and ventral curves is negative, about -0.05 to -0.13 beyond 12 m [read from figure]. This bears on the report's caution that PV construction sets the far-distance floor.
- "Very large fields exist, but few cover much of their environment". The largest drawn Kjelstrup field, about 15.7 m on the 18 m track, and the detector's tendency to split ventral fields (change 4).
- "Field scale rises about tenfold toward the ventral pole, on thin evidence" (renamed in change 21). The single-cell autocorrelation widths of 1.12, 3.07 and 4.10 m. The precession slopes of -102° ± 33° per m dorsal against -18° ± 9° per m intermediate and ventral, with ventral precession over up to 10 m. The Fig. 4 four-level values, approximate. Left and right runs more correlated ventrally. Note 29 on earlier studies (changes 22 to 24).
- "Larger spaces add fields faster than they enlarge them". Muller and Kubie's field-size ratios and the derived area exponents (changes 17 and 19). O'Keefe and Burgess's stretch and split counts, direction dependence and bin-size caveat (change 18).
- "Field definitions decide the largest field a study can report". Three rows could be added to the definitions table:
  - `| Kjelstrup 2008 | 20% of peak, spatially stable, contiguous | None stated | None; fields at track ends may be truncated | Not given in the main text |`
  - `| Muller & Kubie 1987 | Every pixel above 1.0 AP/s | 9 contiguous pixels | None | Pixel size not given in this paper |`
  - `| O'Keefe & Burgess 1996 | Peak above 1.0 Hz; display in 20% steps of peak | None stated | None | Bins 1/24 of box area, so they grow with the box |`
- "A third to a half of CA1 fires in any one environment". Muller and Kubie's impression that about 65% of well-isolated complex-spike units were place cells, which the authors say "will require additional work" to estimate properly (p. 1953). It is a selected sample, so it belongs at most as a caveat.
- "CA3 maps for different rooms overlap no more than random draws". The Kjelstrup note 33 statement that ventral CA3 stays orthogonal across discrete environments (optional addition under change 31).

## Inconsistencies inside the papers

- Kjelstrup 2008. The F(2,517) error df imply about 520 observations against 209 listed cells. The regression reports n = 111 although it pools three levels. Fig. 4 shows four dorsoventral bins whose non-dorsal values differ from the text's three-group means. The Fig. 4 legend names "Spatial Autocorrelation" where the caption names a precession-based width.
- Muller and Kubie 1987. Table 2 lists 1 RF cell for the small cylinder against the small rectangle, while the text says RF cells are "missing" in all four cylinder-rectangle pairs. Table 2 lists 1 UF cell for the rectangles, while the text mentions "the 2 UF cells". Table 1 gives replication r_max as 0.884 and half-card r_max as 0.886 with N = 6, while the text gives 0.876 and 0.888 and says "all 10 cells".
- O'Keefe and Burgess 1996. The text gives 81 of 112 single-peak maps, while Table 1a gives 80.
