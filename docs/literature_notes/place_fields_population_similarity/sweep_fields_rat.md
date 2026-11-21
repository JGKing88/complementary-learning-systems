# Rat place-field widths and field counts: literature sweep

Sweep date 2026-11-21. Rat only, recorded neural data only. The table is `sweep_fields_rat.csv` in this folder: 82 rows from 30 studies. 68 rows carry a width in cm. The other 14 carry only a ratio, a coverage measure or field counts.

## Method

I re-used the ten studies already verified in this folder (`grid_ratio_rat_fields.md`, `fulltext_new_papers.md`, `audit_grid_ratio.md`, `audit_new_papers.md`) and Sharp & Green 1994 from `fulltext_subiculum.md`. Their rows say `verified_in=note`. Every other row comes from full text I read in this pass.

Sources of full text, in the order I tried them:

1. NCBI BioC API (`pmcoa.cgi/BioC_json/PMC.../unicode`). It serves NIH author manuscripts as well as the open-access subset, so it opened Neuron, Nat Neurosci and J Neurosci papers that the Europe PMC XML service refuses with HTTP 500.
2. Europe PMC `fullTextXML` for open-access papers.
3. The PMC open-data bucket on S3 (`pmc-oa-opendata.s3.amazonaws.com/PMC<id>.1/`). It holds figures, supplements and source-data files for open-access papers and for many author manuscripts. Supplementary Table 1 of Schlesiger 2015, the supplement of Hayman 2011, the source data of Grieves 2020 and Diamantaki 2016, and the figures of Tanni 2022 came from there.
4. WebFetch on PMC article pages. This worked for two papers (Shen 1997, Hollup 2001) before PMC switched to a reCAPTCHA. WebFetch relays text through a summariser in fragments of up to 125 characters, so those rows are tagged "full text (WebFetch)" and their table values should be checked against a PDF.
5. User-supplied PDF text in `pdftext/` (Wang 2020).

PMC pages, jneurosci.org, cell.com, bioRxiv (rate limit 1015) and Europe PMC preprint XML all refused at some point. Pre-2000 J Neurosci papers in PMC are page scans with no text layer reachable by these routes.

Rules for each number. A number comes from the printed text, a table or a source-data file where possible, with the quote in `quote_or_reading`. Figure readings say how I read them (Tanni 2022: log axis calibrated from tick marks on a 3x enlargement). For 2D areas, `width_cm` is the equal-area diameter 2*sqrt(A/pi), with the arithmetic in `notes` and the area kept in `value_reported`. Scripts are in the job folder `tmp/sweep_rat/scripts/` (`rows_*.py` hold every row; `make_csv.py` writes the CSV; `grieves_stats.py` computes the Grieves counts and diameters from the source data).

## Coverage

Studies by region, long-axis level and environment size. The first number counts studies with a width in cm. "(+n)" counts further studies with only counts, a ratio or a coverage measure. Size bins use track length in metres or floor area in square metres. One study can appear in several cells.

| Region | Level | under 1 | 1 to 4 | over 4 | size not stated |
|---|---|---|---|---|---|
| CA1 | dorsal | 7 (+1) | 13 (+1) | 2 | 3 (+1) |
| CA1 | intermediate | 1 | 2 (+1) | 0 | 0 |
| CA1 | ventral | 0 | 0 | 0 | 0 |
| CA2 | dorsal | 0 (+1) | 0 | 0 | 0 |
| CA3 | dorsal | 3 (+1) | 4 | 1 | 1 (+1) |
| CA3 | intermediate | 1 | 0 | 1 | 0 |
| CA3 | ventral | 1 | 1 | 1 | 0 (+1) |
| DG-granule | dorsal | 1 | 0 | 0 | 0 |
| DG-mossy | dorsal | 0 | 0 (+1) | 0 | 0 |
| DG-unspecified | dorsal | 0 | 1 (+1) | 0 | 0 |
| SUB | dorsal | 0 | 0 (+1) | 0 | 0 |
| SUB | intermediate | 0 | 0 (+1) | 0 | 0 |
| SUB | level not given | 1 | 0 | 0 | 0 |

Dorsal CA1 with a width: 19 studies (Jung 1994, Mizuseki 2012, Diba 2008 pooled with CA3, Wirtshafter 2020, Park 2011, Rich 2014, Lee 2025, Fenton 2008, Henriksen 2010, Tanni 2022, Grieves 2020, Hayman 2011, Hollup 2001, Mou 2018, Shen 1997, Jacob 2020, Schlesiger 2015, Ravassard 2013, Wang 2020). That is short of the 30 asked for. Most rat CA1 papers I opened (about 25 more, listed under "Checked, no usable number") put field size only in a figure or a supplement I could not reach.

## New numbers in the gap areas

### Intermediate and ventral levels

- Ventral CA1: still no rat field width. Two rat ventral CA1 studies exist. Malagon-Vina et al. 2023 (eLife, PMC10089657) recorded vCA1 on an elevated linear maze and reports spatial information, sparsity and coherence, but no field size, and says "most of the neurons do not have a spatially restricted activity similar to a typical place cell". Ciocchi et al. 2015 (Science) is paywalled. Sosa et al. 2020 recorded vCA1 on a W-track but reports no spatial firing of vCA1 cells.
- Intermediate CA1, circular track. Burke et al. 2011 recorded "distal ... intermediate CA1" at 6.0 mm posterior on a 335 cm circular track in Fischer-344 rats. Mean field size was "~80 cm" on the empty track and "~55 cm" with eight objects on it, with fields defined as one cycle of theta phase precession. Active cells had 1.6 fields per epoch (empty) and 2.1 (objects). This is the first intermediate CA1 width in cm on a track longer than 1.3 m. It is not paired with a dorsal value from the same rats, and the rule differs from all other rows.
- Intermediate subiculum. Kim, Ganguli & Frank 2012 recorded the subiculum at "intermediate locations along the septal-temporal axis" (6.6 to 7.0 mm posterior) on a 350 cm U-track. They give a spatial activity fraction, not a width: median 0.31 (proximal third), 0.54 (middle), 0.69 (distal), and 0.26 for distal CA1 at the same level.
- Intermediate CA1 in a 68 cm box (Jung 1994, 49 cm against 24 cm dorsal) and on a 130 cm track (Lee 2025 preprint, 37 cm against 23 cm) are unchanged from the notes.

### CA3, CA2, dentate gyrus and subiculum

- CA3: Wang, Monaco & Knierim 2020 give median field length 73 deg (IQR 62 deg) for CA3 and 60 deg (IQR 38 deg) for CA1 on the same circular track. The paper states the track as 76 cm in outer diameter and 10 cm wide, a centreline of about 207 cm, so that is about 42 and 35 cm [derived]. CA3 is 1.2 times CA1 in the same rats either way.
- CA2: Mankin et al. 2015 report CA2 fields "24.5 % larger than those of CA1 cells" in 0.6 to 0.8 m2 boxes. The absolute sizes are in Table S2, which I could not reach. Active cells per 10-min session: CA1 59.2%, CA2 68.2%, CA3 48.1%.
- Dentate gyrus, identified granule cells: Diamantaki et al. 2016 juxtacellularly labelled granule cells in freely moving Wistar rats. The source data give total field sizes of 118, 71, 80 and 60 pixels of 6.25 cm2 for the four identified cells with fields, equal-area diameters 30.6, 23.8, 25.2 and 21.9 cm (70 x 70 cm arena). 163 of 190 recordings in the granule layer were silent (86%). Of spatially selective granule cells, 8 had one field and 3 had a second field.
- Dentate gyrus, putative types: Neunuebel & Knierim 2012 give a median single-field area of 42.1 cm2 for cells on granule-layer tetrodes (7.3 cm equal-area diameter) and 13.3 cm2 per subfield for multi-field cells. These areas are smaller than the paper's own 10-pixel minimum however the unit is read, and the pixel grid cannot cover the box, so the audit cleared these widths. Field counts are usable, but the 25-cell distribution covers all dentate recording sites, not only the granule-layer tetrodes: 12 of 25 active cells had one field and 5 had five or more; on hilar tetrodes (putative mossy cells), 7 of 8 had two to four fields.
- Subiculum: no single-field width in cm. Sharp & Green 1994 (from the note) gives firing over 0.88 (distal) and 0.78 (proximal) of the visited pixels of a 76 cm cylinder, equal-area diameters 71 and 67 cm as upper bounds capped by the wall. Lee, Seol & Lee 2022 report wider fields in subiculum than CA1 on a T-maze (figure only), with about 90% of active CA1 cells single-field against about half of subicular cells having any detectable field.

### Large environments

- Tanni, de Cothi & Barry 2022: four familiar boxes of 1.09, 2.19, 4.38 and 8.75 m2, same 629 place cells in 5 rats. Mean field area within 25 cm of the wall rose from about 2,000 to 2,700 cm2 (A) to about 4,600 to 9,500 cm2 (D), and in D from the wall to 75 to 100 cm from it, to about 10,300 to 17,800 cm2. Equal-area diameters: 50 to 59 cm (A, near wall), 77 to 110 cm (D, near wall), 115 to 151 cm (D, centre). Field width projected on the long axis in D was 56 to 120 cm near the short wall and 132 to 177 cm at about 88 cm from it. These are per-rat values read from Fig. 3. The paper averages, for each bin, all fields overlapping it, so large fields weigh more than in a per-field mean.
- Grieves et al. 2020: 1.8 x 1.8 m arena, median "enclosing diameter" 64.6 cm (IQR 47.6 to 87.2), mean 71.7 +- 31.8 cm over 693 fields, computed from the source data. Enclosing diameter is probably a maximum extent and reads larger than an equal-area diameter.
- Harland et al. 2021 (18.6 m2) and Lee JS et al. 2020 (long tracks) could not be opened.

## Field rules and how they move widths

Rules seen in the table, roughly from widest to narrowest edge:

| Edge rule | Studies |
|---|---|
| Any firing (rate > 0) | Park 2011, Fenton 2008, Sharp & Green 1994 |
| Above the cell's mean rate | Jacob 2020 |
| 10% of peak | Mizuseki 2012, Mou 2018, Wang 2020, Burke 2011 (rate check) |
| 15% of peak | Hayman 2011 (all its maps) |
| 20% of peak | Kjelstrup 2008, Henriksen 2010, Hollup 2001, Komorowski 2013, Lee 2025, Neunuebel 2012, Diamantaki 2016, Grieves 2020, Royer 2010, Schlesiger 2015 |
| 33% of peak | Lee SM 2022 |
| Fixed rate: 1 Hz (Jung 1994, Shen 1997), 2 Hz (Diba 2008, Rich 2014) | |
| Mean + 1 SD of the cell's rate | Wirtshafter 2020 |
| Iterative threshold from 1 Hz up, lowest valid region | Tanni 2022 |
| One cycle of theta phase precession | Burke 2011 |

Other choices that change the number: largest field per cell (Kjelstrup, Park, Lee 2025, Diba) against all fields (most others) against all fields summed (Diamantaki, Royer coverage); minimum field size (from 5 cm on a track in Mou to 200 cm2 in Komorowski, which sits at the floor of that study's values); smoothing (Kjelstrup 15 cm against Brun 5 cm); axis (Hayman's major and minor axes differ by 1.8 times for the same fields).

Within-study comparisons:

- Same data, two rules. Brun et al. 2008 (in `fulltext_new_papers.md`) re-ran ventral MEC fields with Kjelstrup's settings (20% edge, 15 cm smoothing, 25 cm minimum) instead of their own (10%, 5 cm, 15 cm). The largest field went from 190 +- 13 to 224 +- 13 cm, 18% longer. The note does not separate the effect of smoothing from that of the edge. MEC, not hippocampus, but the only same-data comparison found.
- Largest field against all fields. Park 2011: the largest CA1 field per cell grew from 36 to 58 cm equal-area diameter between a 76 cm cylinder and a 1.8 x 1.4 m box, while all fields pooled did not change ("F1,1316 = 0.01 p = 0.93").
- Per-cell field against autocorrelation against population. Kjelstrup 2008 in dorsal, intermediate and ventral CA3: largest field 0.98, 3.73, 5.52 m; autocorrelation width 1.12, 3.07, 4.10 m; population-vector full width at r = 0.5 0.81, 3.71, 6.66 m. The three measures differ by up to 38% in dorsal CA3 (0.81 against 1.12 m), 21% in intermediate CA3 and 62% in ventral CA3.
- Phase-precession rule against rate rule. Burke 2011: a rate rule (10% of max over 5 bins of 3.84 cm) gave "similar results" to the phase-precession rule. No numbers.
- Fixed-rate thresholds. Shen 1997: old rats had smaller fields than young at 0.5, 1 and 2 Hz thresholds. Values only at 1 Hz (31.0 against 23.8 cm).
- Major against minor axis. Hayman 2011: flat-arena fields 38 +- 2 cm long and 21 +- 1 cm wide (n = 53).

Between studies, dorsal CA1 on familiar 1 to 3 m tracks spans 29 cm (Wirtshafter, mean + 1 SD edge) to 75 cm (Mizuseki, 10% edge). In between: 31 cm (Shen, 1 Hz, 2.7 m loop), 40 cm (Mou, 10%, novel 2 m track), 42 cm (Schlesiger, 20% field boundary), 44 cm (Ravassard, rule in the supplement), 52 cm (Diba, 2 Hz, CA1 and CA3 pooled). In familiar 2D arenas under 1 m2, equal-area diameters run from 23 cm (Jacob, edge at the mean rate, 1 m cylinder) and 24 cm (Jung, 1 Hz, 68 cm box) to 40 cm (Fenton, rate > 0, 68 cm cylinder). The rule and the environment are confounded across studies, so these spreads cannot be split into the two causes.

## Environment size

Within-study changes of field width with environment size:

| Study | Small | Large | Width change |
|---|---|---|---|
| Diba 2008 (CA1+CA3) | 1.0 m track | 1.7 m track | 42 to 52 cm (+24% for +70% length) |
| Fenton 2008 (CA1, per field) | 68 cm cylinder, 0.36 m2 | 150 x 140 cm chamber, 2.1 m2 | 40.3 to 45.9 cm; same cells 38.4 to 51.6 cm |
| Park 2011 (CA1, largest field) | 76 cm cylinder | 1.8 x 1.4 m box | 36 to 58 cm |
| Tanni 2022 (CA1, near wall) | 1.09 m2 | 8.75 m2 | 50 to 59 cm to 77 to 110 cm |
| Grieves 2020 (CA1) | 1.8 m arena | 0.97 m climbing lattice | 64.6 to 66.8 cm (3D lattice; not a size series) |
| Burke 2011 (intermediate CA1) | empty 3.35 m track | same track with objects | 80 to 55 cm (objects, not size) |

Fields widen with the environment but less than in proportion, as the earlier notes found. Tanni adds that width depends on distance to the nearest wall: fields near walls stay narrow in every box, and the centre of a large box holds the widest fields.

## Field counts

| Study | Region | Environment | Fields per cell | Single-field share | Silent or inactive share | Fit |
|---|---|---|---|---|---|---|
| Rich 2014 | dCA1 | 48 m novel track | per-rat means 2.3 to 4.3 | about 5 to 10% predicted | 35% of cells never fired a field at 48 m | Negative binomial r = 0.57, p = 0.14 (gamma-Poisson) |
| Tanni 2022 | dCA1 | 1.09 to 8.75 m2 | 1.28 (A) to 2.28 (D) | not given per box | 15% is the share of place cells firing together at any one spot, not a silent or recruited share | Gamma-Poisson beats equal-Poisson (log-lik -1538 vs -1679) |
| Fenton 2008 | dCA1 | 0.36 vs 2.1 m2 | 1.3 vs 3.4 (4.05 with stairs) | 72% vs 11% | place cells 64% vs 72% of active cells | none |
| Park 2011 | dCA1, CA3, DG | 0.45 vs 2.52 m2 | box: 5.89, 4.96, 5.35 | cylinder 61%, 82%, 64%; box 18%, 29%, 8% | not given | none |
| Henriksen 2010 | dCA1 by transverse band | 2 m cylinder | 1.57, 1.57, 2.13 | 57%, 57%, 34% | 1 to 10% of active cells not spatial | none |
| Grieves 2020 | dCA1 | 1.8 m arena; 0.97 m lattice | 1.23 (1.43 if >= 1); 1.76 | 56% of all cells; 42% | 14% no field; 9% | Underdispersed in the arena (variance/mean 0.48), computed here |
| Wirtshafter 2020 | dCA1 | T maze | 1.48 | 61% (63/104) | not given | none |
| Jung 1994 | dCA1, iCA1 | 68 cm box | 1.22, 1.14 | 78%, 86% | 55% and 82% of CS cells had no field | none |
| Mou 2018 | dCA1 | 2 m novel track | 0.90 per active trajectory | 88% | not given | none |
| Burke 2011 | iCA1 | 3.35 m track | 1.6 (empty), 2.1 (objects) | Fig. 3A only | objects add 10 to 15% active cells | none |
| Kjelstrup 2008 | CA3 d, i, v | 18 m track | 1.5, 1.7, 1.6 per direction (overestimates) | not given | not given | none |
| Karlsson & Frank 2008 | dCA1, dCA3 | W-tracks, novel to familiar | not given | not given | CA1 active 55% to 40%; CA3 about 30% | none |
| Mankin 2015 | CA1, CA2, CA3 | 0.6 to 0.8 m2 boxes | CA2 > CA1, CA3; CA1 and CA3 < 1 over all cells | not given | active per session 59%, 68%, 48% | none |
| Ravassard 2013 | dCA1 | real vs virtual track | not given | not given | active on track 45.5% vs 20.4% | none |
| Diamantaki 2016 | DG granule | 0.49 m2 arena | 8 one-field, 3 two-field | 73% of spatial GCs | 86% of GC-layer recordings silent | none |
| Neunuebel 2012 | DG (granule layer; hilus) | 0.34 or 1.82 m2 | distributions in the CSV | 48%; 13% | not given | none |
| Lee SM 2022 | dCA1, dSUB | T-maze | not given | about 90% CA1 | about half of SUB cells had no detectable field | none |
| Hollup 2001 | dCA1 | annular water maze | not given | 70% single-peaked | not given | none |

Count against environment size, within study: Fenton (1.3 to 3.4 for 5.8 times the area), Park (about 3 times more fields in 5.6 times the area), Tanni (1.28 to 2.28 for 8 times the area), Rich (recruitment reaches about 65% at 48 m; the fitted model predicts 90% at about 500 m).

## Gaps that remain

- Ventral CA1 in rats: no field width anywhere I could read. The two rat vCA1 recording papers give no widths.
- Intermediate CA1 with a dorsal comparison in the same environment exists only in Jung 1994 (68 cm box) and the Lee 2025 preprint (130 cm track). Burke 2011 adds an intermediate value on a 3.35 m track without a dorsal partner.
- Intermediate and ventral CA3 in standard environments: only Komorowski 2013 (capped by a 37 cm box) and Royer 2010 (ratio and coverage only).
- CA2: no absolute width.
- Dentate gyrus: four identified granule cells and one study with doubtful units. Mossy cells: counts only.
- Subiculum: coverage-type measures only (Sharp & Green, Kim), no per-field widths.
- Over 4 m or 4 m2: Rich 2014, Tanni 2022 and Kjelstrup 2008 only.
- Dorsal CA1 sample: 19 studies with widths, short of 30. The field rule and environment size are confounded across studies.

## Papers to request

Most valuable first. These were paywalled, figure-only in what I could reach, or blocked by reCAPTCHA.

1. Maurer AP, VanRhoads SR, Sutherland GR, Lipa P, McNaughton BL (2005). Self-motion and the origin of differential spatial scaling along the septo-temporal axis of the hippocampus. Hippocampus 15:841-852. doi:10.1002/hipo.20114. Dorsal against middle CA1 field size on a track.
2. Maurer AP, Cowen SL, Burke SN, Barnes CA, McNaughton BL (2006). Organization of hippocampal cell assemblies based on theta phase precession. Hippocampus 16:785-794. doi:10.1002/hipo.20202. Dorsal against intermediate CA1.
3. Jin SW, Lee I (2021). Differential encoding of place value between the dorsal and intermediate hippocampus. Curr Biol 31:3053-3072. doi:10.1016/j.cub.2021.04.073. dCA1 against iCA1 in the same rats.
4. Ciocchi S, Passecker J, Malagon-Vina H, Mikus N, Klausberger T (2015). Selective information routing by ventral hippocampal CA1 projection neurons. Science 348:560-563. doi:10.1126/science.aaa3245. Rat vCA1 spatial firing.
5. Royer S, Sirota A, Patel J, Buzsaki G (2010). J Neurosci 30:1777-1787. doi:10.1523/JNEUROSCI.4681-09.2010. A PDF with readable Figs. 2B, 4B and 5 would give absolute dorsal and ventral CA3 widths.
6. Harland B, Contreras M, Souder M, Fellous JM (2021). Dorsal CA1 hippocampal place cells form a multi-scale representation of megaspace. Curr Biol 31:2178-2190. doi:10.1016/j.cub.2021.03.003 (preprint 10.1101/2021.02.15.431172). 18.6 m2 arena, widths and counts.
7. Lee JS, Briguglio JJ, Cohen JD, Romani S, Lee AK (2020). The statistical structure of the hippocampal code for space as a function of time, context, and value. Cell 183:620-635. doi:10.1016/j.cell.2020.09.024. Field counts on long tracks.
8. Lu L, Igarashi KM, Witter MP, Moser EI, Moser MB (2015). Topography of place maps along the CA3-to-CA2 axis of the hippocampus. Neuron 87:1078-1092. doi:10.1016/j.neuron.2015.07.007. CA3 and CA2 field sizes.
9. Mankin EA et al. (2015) supplement (Table S2). doi:10.1016/j.neuron.2014.12.001. Absolute CA2, CA1, CA3 field sizes and counts.
10. Leutgeb JK, Leutgeb S, Moser MB, Moser EI (2007). Pattern separation in the dentate gyrus and CA3 of the hippocampus. Science 315:961-966. doi:10.1126/science.1135801. DG and CA3 fields and counts.
11. Jung MW, McNaughton BL (1993). Spatial selectivity of unit activity in the hippocampal granular layer. Hippocampus 3:165-182. doi:10.1002/hipo.450030209.
12. Skaggs WE, McNaughton BL, Wilson MA, Barnes CA (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. Hippocampus 6:149-172. doi:10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K. DG and CA1 fields.
13. Barnes CA, McNaughton BL, Mizumori SJ, Leonard BW, Lin LH (1990). Comparison of spatial and temporal characteristics of neuronal activity in sequential stages of hippocampal processing. Prog Brain Res 83:287-300. doi:10.1016/S0079-6123(08)61257-1. CA1, CA3, DG and subiculum in one study.
14. Sharp PE (1999). Subicular place cells expand or contract their spatial firing pattern to fit the size of the environment in an open field but not in the presence of barriers. Behav Neurosci 113:643-662. doi:10.1037/0735-7044.113.4.643.
15. Anderson MI, O'Mara SM (2003). Analysis of recordings of single-unit firing and population activity in the dorsal subiculum of unrestrained, freely moving rats. J Neurophysiol 90:655-665. doi:10.1152/jn.00723.2002. Subicular field sizes.
16. Poucet B, Thinus-Blanc C, Muller RU (1994). Place cells in the ventral hippocampus of rats. NeuroReport 5:2045-2048. doi:10.1097/00001756-199410270-00014.
17. Thompson LT, Best PJ (1989). Place cells and silent cells in the hippocampus of freely-behaving rats. J Neurosci 9:2382-2390. doi:10.1523/JNEUROSCI.09-07-02382.1989. Silent fraction (abstract: nearly two thirds silent).
18. Markus EJ, Qin YL, Leonard B, Skaggs WE, McNaughton BL, Barnes CA (1995). J Neurosci 15:7079-7094. doi:10.1523/JNEUROSCI.15-11-07079.1995. Field size and fields per cell under two foraging tasks.
19. Muller RU, Kubie JL, Ranck JB (1987). J Neurosci 7:1935-1950. doi:10.1523/JNEUROSCI.07-07-01935.1987. Field size in the 76 cm cylinder.
20. Epsztein J, Brecht M, Lee AK (2011). Neuron 70:109-120. doi:10.1016/j.neuron.2011.03.006. Silent and place cells by whole-cell recording.
21. Aghajan ZM et al. (2015). Nat Neurosci 18:121-128. doi:10.1038/nn.3884. Rat 2D virtual reality fields.
22. Davidson TJ, Kloosterman F, Wilson MA (2009). Neuron 63:497-507. doi:10.1016/j.neuron.2009.07.027. 10 m track.
23. Reachable but figure-only, worth a figure read from a PDF: Hales 2014 (doi:10.1016/j.celrep.2014.10.009), Lee H et al. 2015 CA3 transverse axis (doi:10.1016/j.neuron.2015.07.012), Roth et al. 2012 CA1 against CA3 (PMC3338615), Schlesiger 2015 open-field sizes (Fig. 1).

## Checked, no usable number

Read in full here but no field width or count in the text or reachable supplement: Sosa 2020 (vCA1, no spatial data), Malagon-Vina 2023 (vCA1, sparsity only), Jin 2024 (iHP, figure only), Michon Linde 2026 (iHP, inclusion rule only), Kitanishi 2021 and Nakai 2024 (SUB, information measures), Simonnet & Brecht 2019 (SUB), Stewart 2014 and Poulter 2021 (SUB), Lee H 2015 (CA3 field size in Fig. 2D only), Oliva 2016 and Kay 2016 (CA2), Neunuebel & Knierim 2014 and Hwaun & Colgin 2019 (CA3), Hales 2014, Brandon 2014, Muessig 2015, Spiers 2015, Kraus 2013 (normalised treadmill widths), Komorowski 2009, Maurer 2012, Monaco 2014, Navratilova 2012 (lap-wise widths in a figure), Mamad 2015 and 2017, Bieri 2014, Duvelle 2019 (CA3 larger than CA1, no numbers), Roth 2012 (figure), Haugland 2025 (figure, dm2), Speers 2023, Porter 2018, Mizuseki 2011 (supplement), Sharif 2021 (rat maze sizes in a figure), Gothard 2001 (no sizes). Scanned by keyword only, no number found: Ewell 2019, Kaefer 2019, Zitricky 2019, Tirole 2022, Miranda 2024, Wang N 2025, Donahue 2025, Zhang & Manahan-Vaughan 2014. Abstract only: Thompson & Best 1989, Markus 1995, Muller, Kubie & Ranck 1987.

## Caveats

- Rows tagged "WebFetch" (Shen 1997, Hollup 2001) carry values relayed by a summarising tool. Check against the PDFs before relying on them. The Hollup corridor area looks garbled and is not used.
- Wang 2020 widths in cm are derived from the stated 76 cm outer diameter and 10 cm width.
- Neunuebel 2012 widths were cleared by the audit; only its field counts are used.
- Diba 2008 pools CA1 and CA3; the region column says CA1 because the schema allows one value.
- Royer 2010 coverage rows are upper bounds for one field.
- Several 2D widths are capped by walls: Jung 1994 ventral, Komorowski 2013, Sharp & Green 1994, the Kjelstrup ventral track fields (track end).
