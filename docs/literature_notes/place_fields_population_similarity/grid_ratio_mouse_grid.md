# Mouse MEC grid spacing per module, checked against the papers

These notes supply the columns of the MOUSE table. Each column is a mouse MEC grid module, from smallest to largest spacing. The cells of that table are field width divided by grid spacing, so here I only need spacing per module. All numbers are neural data. Rat values are covered elsewhere and appear here only where a mouse paper compares itself to them.

Checked 2026-11-20. Full texts came from NCBI efetch (PMC XML), the eLife article XML, the Nature Neuroscience PDF (Gil et al.) and the nature.com article page (Mallory et al. supplement caption). PMC author manuscripts have no page numbers, so I cite section and figure instead.

Status tags:

- full text. The quote is in the paper's text, a table or a figure caption.
- abstract. Only the abstract could be read.
- figure-read. Read off a figure image. I say how.
- derived. My arithmetic, shown.
- inferred. A fact not stated in the paper, filled in from a closely related paper. I say from where.
- not found. The claimed number is not in the source I could read.

## 1. The three claims in the brief

1. "Discrete grid spacing with peaks at 49, 71 cm, for a ratio of 1.45." Not found. I searched the full text of Gu et al. 2018 (Cell 175:736), including the STAR Methods and every figure caption with Figure S2, for "49", "71" and "1.45". None occur as spacings or ratios. Gu et al. report an average ratio of 1.51 between successive mouse modules (row G1 below). I also checked Yoon et al. 2016 (Neuron 89:1086, the 1D-slice method paper from the same groups) and found neither number. 71 / 49 = 1.449 [derived], so the pair is internally consistent, but I could not trace it to any paper. Gu et al. also recorded mouse grid cells with tetrodes in a 0.5 m arena and split them into two modules by 2D spacing (Figs. S2O to S2T). Those panels might show module spacings, but the supplement PDFs returned HTTP 403 from Europe PMC and the Caltech and Princeton copies were restricted. Treat the claim as unverified.
2. "Grid spacing of 72.5 ± 13.7 cm" in dorsomedial MEC. Verified, full text. The paper is Clark H & Nolan MF, "Task-anchored grid cell firing is selectively associated with successful path integration-dependent behaviour", eLife 12:RP89356 (reviewed preprint July 2023, version of record March 2024), University of Edinburgh. Nine mice with tetrodes ran a virtual location memory task. Spacing was measured in a real open arena recorded on the same days. See rows C1 and C2.
3. Several candidate papers are not mouse spacing sources. Waaga et al. 2022 (Neuron 110:1843) recorded Long-Evans rats ("Rat: Long-Evans", key resources table). Carpenter et al. 2015 (Curr Biol 25:1176) recorded rats (abstract, "we recorded mEC grid cells in rats"). Obenhaus et al. 2022 is in PNAS 119:e2121655119, not Nature, and gives no spacing values in its text. Campbell et al. 2018 (Nat Neurosci 21:1096) and 2021 (Cell Rep 36:109669) are mouse VR papers with no spacing values in their text.

## 2. One row per number

| Row | Source | Value | Quote | Location | Status | N | Arena | Method | DV position |
|---|---|---|---|---|---|---|---|---|---|
| Z1 | Zong et al. 2022, Cell 185:1240 | 30 cm dorsal to about 60 cm ventral, within one imaged area | "grid spacing within the FOV ranged from 30 cm (dorsal) to ∼60 cm (ventral) (Figure 5H)" | Results, "Using MINI2P to analyze spatial tuning in MEC"; Fig. 5H | full text | 310 grid cells, 300 in MEC; 5 stitched FOVs in "a freely moving mouse" (1 mouse) | 80 × 80 cm, familiar ("pre-trained in the recording environment until they were familiar with the environment") | Two-photon miniscope (MINI2P), calcium | "near its dorsomedial tip"; 0.9 × 0.9 mm imaged through a prism; 100 and 140 µm below the MEC surface (layer 2 and superficial layer 3) |
| M1 | Mallory et al. 2018, Nat Neurosci 21:270 | Control (iWT) mean 36.26 ± 9.27 cm, mean ± SD | "spacing: iWT = 36.26 ± 9.27 cm, iCre-KO = 42.16 ± 10.53 cm" | Results, "Loss of MEC HCN1 channels selectively impacts grid scale"; Fig. 1d to f | full text (PMC author manuscript) | 37 iWT grid cells; 12 iWT mice (29 mice implanted in all) | "open arenas"; size not found because the Online Methods were behind the paywall | Tetrodes | MEC; per-cell DV position plotted in Supp. Fig. 7b, range not stated in text |
| M2 | Mallory et al. 2018 | Control intercept 31.97 cm; spacing rises along the DV axis | "grid spacing in both groups increased along the dorsal-ventral axis ( F (1,96) = 4.94, p = 0.029)" and "Y intercept iWT = 31.97 cm, iCre-KO = 37.48 cm" | Supplementary Fig. 7b,c caption, as served on the nature.com article page | full text (caption) | as M1 | as M1 | Tetrodes | Slope not given in the caption. The intercept is presumably at the dorsal border, but the caption does not define the DV origin |
| G0 | Gil et al. 2018, Nat Neurosci 21:81 | Control mean 59.00 ± 1.91 cm, mean ± SEM with mice as units | Table 1 row "Grid spacing (cm) 59.00 ± 1.91" and footnote "Mean ± s.e.m. using mice (MEC) or hemispheres (HP) as experimental units." | Table 1, journal p. 86 (PDF p. 6) | full text (publisher PDF) | 7 control mice, 207 grid cells | 70 × 70 × 30 cm, "trained during 2–4 weeks to forage inside the arena" (Methods, PDF p. 13) | Tetrodes, 5 or 6 per drive | Right MEC. Craniotomy "+3.1 mm lateral from bregma, 0.2 mm anterior from the transverse sinus", tetrodes started 0.8 mm below the surface and moved 25 to 50 µm per session. Distance from the dorsal border not given |
| C1 | Clark & Nolan 2024, eLife 12:RP89356 | 72.5 ± 13.7 cm (the paper does not say SD or SEM in this sentence) | "we identified 103/1881 neurons as grid cells (11.4±17.4 grid cells/mouse, range 0.4–9.5%, Supplementary file 1) ... These grid cells had field sizes of 7.5±3.3 cm and grid spacing of 72.5±13.7 cm and were found in dorsomedial parts of the MEC." | Results, first section | full text (eLife VOR XML) | 103 grid cells, 9 male C57BL/6J mice | "a metal box with a square floor area"; size not stated. Gerlei et al. 2020 (same lab, same frame part numbers) states "a floor area of 1 m2" | inferred (arena) | Tetrodes, 4 per drive | "dorsomedial parts of the MEC". Tetrodes lowered 1.2 to 1.4 mm at 3.4 mm lateral along the lambdoid suture, angled 15° posterior, then 50 to 100 µm per session; positions by microCT |
| C2 | Clark & Nolan 2024 | No grid cell below 40 cm | "Spatial frequencies >5 were discarded from further analysis as no grid cells were found with grid spacings <40 cm." | Materials and methods, "Analysis of neural activity during the location memory task" | full text | as C1 | as C1 | as C1 | as C1 |
| W1 | Wen et al. 2024, Nature 635:943 | Scale ratio between modules 1.4 to 1.7 | "Grid cells clustered into discrete modules by their power spectra (Fig. 1c) and, in sessions with multiple modules, the inferred distance between grid firing fields (that is, grid scale) increased with a scale ratio between 1.4 and 1.7 (ref. 19; Fig. 1d)." Caption: "Histogram of inferred 2D grid scale ratios between modules (n = 4 animals, n = 19 sessions, n = 51 modules and n = 1,178 cells). The dashed lines denote powers of √2." | Results, "The real-time grid cell attractor state"; Fig. 1d | full text | 4 mice, 19 sessions, 51 modules, 1,178 cells (15,342 grid cells in the whole study) | 1D virtual track, 320 cm lap, run in darkness | Neuropixels, head-fixed. 2D scale inferred from the 1D power spectrum | Probe inserted 10° from vertical "as close to the sinus as possible" to 3,200 to 3,400 µm angled depth; Fig. 1c sorts cells dorsal to ventral |
| W2 | Wen et al. 2024 | Ratio histogram shape | Most counts lie between about 1.3 and 1.5. A second cluster sits at about 1.9 to 2.1 and a smaller bar at about 2.8 to 2.9. A few counts lie near 1.0 to 1.1. Dashed lines sit at about 1.41, 2.0 and 2.83 | Fig. 1d | figure-read (published Fig. 1 PNG from media.springernature.com, bar positions read against the x axis) | as W1 | as W1 | as W1 | as W1 |
| W3 | Wen et al. 2024 | Largest modules 76 cm and 65 cm (VR-inferred) | "the largest modules in (a) (Module 2) and (b) (Module 3) did not show clear grid-like tuning in 2D because their inferred 2D grid scales, 76 cm and 65 cm, respectively, were too large to exhibit clear periodic firing in the 75 cm wide environment" | Extended Data Fig. 3 caption | full text | 3 mice, 504 cells in this figure | 75 cm square open field with 30 cm walls (Methods) | Neuropixels, chronic | Largest module seen in each of two mice |
| W4 | Wen et al. 2024 | One module at 25 cm (VR-inferred) | "for this module of inferred grid scale 25 cm, corresponds to a spatial error of ~5 cm" | Extended Data Fig. 6c caption | full text | 1 module, 1 session | 1D virtual track | Neuropixels | Not stated |
| G1 | Gu et al. 2018, Cell 175:736 | Successive-module ratio 1.51 on average | "The ratio of 2D spacing between successive modules, which was the spacing of their best fit triangular lattice, was 1.51 on average (Figure S2V), close to the ratio in 2D real environments (1.42) (Stensola et al., 2012)." Caption S2V: "Red dashed line marks the average ratio (1.51) of all pairs of modules." | Results, "Grid modules cluster"; Fig. S2V caption | full text (PMC author manuscript) | Number of module pairs not stated in the text | 1D virtual tracks of 1000 or 1800 cm | Two-photon calcium imaging; 2D lattice fitted to 1D responses | Layer 2 of dorsal MEC, imaged through a microprism in the transverse fissure |
| G2 | Gu et al. 2018 | Three modules in one imaged region | "Anatomical distribution of grid modules in two mice. Grid spacing increases from modules 1 to 3." Also "20 modules in ten FOVs of six mice" (Fig. 3F) and "33 identified modules with at least 14 cells" | Fig. 3F,G captions; Results | full text | 2 mice for Fig. 3G; 6 mice in all | 1D VR | Calcium imaging | Dorsal MEC surface |
| G3 | Gu et al. 2018 | Two modules by 2D spacing in a real arena; values not in text | "The real 2D arena was a 0.5m square environment"; "Only grid spacings were used for the analysis because they were sufficient to generate the clustering of grid cells in the two animals we analyzed." | STAR Methods, "Tetrode recordings" and "Grid modules based on 2D responses (Figures S2O-S2T)" | full text; values not found | 2 mice | 0.5 × 0.5 m | Tetrodes | Craniotomy 3.2 mm lateral, rostral to the lambdoid suture |
| F1 | Fyhn et al. 2008, Hippocampus 18:1230 | Gradient only, no number | "The spacing of the grid increased with distance from the dorsal border of the medial EC. The lowest values for grid spacing, recorded at the dorsal end, were comparable to those of the rat, suggesting that grid fields do not scale up proportionally with body size." | Abstract | abstract | Not in abstract | Not in abstract | Tetrodes | Superficial layers of medial EC, dorsal to more ventral |
| H1 | Giocomo et al. 2011, Cell 147:1159 | Gradient only, no number | "although the dorsal-ventral gradient of the grid pattern was preserved in HCN1 knockout mice, the size and spacing of the grid fields, as well as the period of the accompanying theta modulation, was expanded at all dorsal-ventral levels" | Abstract | abstract | Not in abstract | Not in abstract | Tetrodes | Several DV levels |
| O1 | Obenhaus et al. 2022, PNAS 119:e2121655119 | Gradient only, no number | "grid modules appeared clustered according to gradients described in the literature (38, 40, 41), with smaller-spaced grid cells located more dorsally than larger-spaced ones (see SI Appendix, Fig. S8 for an example)" | Results; SI Fig. S8 not read | full text | Not stated for this point | 80 × 80 cm | Two-photon miniscope | MEC and parasubiculum, dorsal MEC |

## 3. Modules and ratio in mice

Mice have discrete modules. Gu et al. found anatomically clustered modules with up to three in one imaged region (G2), and an average ratio of 1.51 between successive modules (G1). Wen et al. found "discrete modules" with ratios of 1.4 to 1.7 (W1), from 51 modules in 19 sessions, which is 2.7 modules per session [derived, 51 / 19]. Their example session in Fig. 1c shows four modules [figure-read, four colour bars beside the spectra].

Two points limit this. First, both ratios come from 2D lattices inferred from 1D virtual-track firing, not from 2D open-field spacing. Second, the Wen histogram (W2) has clear mass near 2.0 and 2.8 as well as near 1.4. Those may be pairs of non-adjacent modules (√2 × √2 = 2), but the caption says only "between modules". The text range of 1.4 to 1.7 therefore describes the main peak, not every pair.

No mouse paper I could read gives absolute 2D open-field spacings module by module. The rat value of 1.42 that Gu et al. cite is from Stensola et al. 2012.

## 4. How arena size caps the measurable spacing

- Wen et al. state the cap directly. Modules of 65 and 76 cm "were too large to exhibit clear periodic firing in the 75 cm wide environment" (W3). So in a 75 cm box the practical limit is somewhere below 65 cm.
- Rowland et al. 2018 (eLife 7:e36664, mouse) state "grid cells with inter-field spacing larger than our 0.8–1.0 meter recording box would be missed in our analysis (Stensola et al., 2012)" (Discussion) [full text].
- Gil et al. measured a mean of 59 cm in a 70 cm box (G0). This is close to the cap, so larger modules were probably not counted and the mean leans small [my reading].
- Zong et al. measured up to about 60 cm in an 80 cm box (Z1). The ventral end of their range is near the cap.
- Clark & Nolan measured a mean of 72.5 cm in a box that is probably 1 m across (C1, inferred). Cells much above 1 m would be missed.
- Gu et al. sorted modules in a 0.5 m box (G3). That box cannot show a full hexagon for spacings near or above 50 cm [my reading].

No mouse open field I found exceeds about 1 m. Ventral mouse MEC spacing has not been measured in a familiar 2D arena large enough to show it.

## 5. Conflicts between the dorsal values

The labs disagree on what dorsal mouse MEC looks like. Mallory et al. report a control mean of 36.26 cm with an intercept of 31.97 cm (M1, M2), and Zong et al. report 30 cm at the dorsal edge (Z1). Clark & Nolan report a mean of 72.5 cm in "dorsomedial" MEC and no cell below 40 cm (C1, C2). Gil et al. sit between them at 59.00 cm (G0). The papers do not explain the difference. Plausible causes are deeper tetrode sampling in Clark & Nolan, which started 1.2 to 1.4 mm into the brain, and box sizes [my reading]. For the table, the Clark & Nolan and Gil means are population means over unknown module mixes. They should not be read as single-module spacings.

## 6. Caveats: novelty, virtual reality and rescaling

These are not familiar-environment open-field values, so they stay out of the columns.

| Source | Value | Quote | Location | Status | N | Arena | Method |
|---|---|---|---|---|---|---|---|
| Chen et al. 2018, eLife 7:e34789 | VR spacing / real spacing = 1.42 | "The spatial scale of the grids was larger in VR than in R, with an average increase of 1.42 (grid scale in VR/grid scale in R, n = 6 mice)"; "the grid scale expansion factor being 1.43 in 60 cm, 1.36 in 90 cm" | Results, electrophysiology section; Fig. 4E (n = 61 cells) | full text | 6 mice, 61 grid cells | Real 60 × 60 cm; VR 60 × 60 or 90 × 90 cm | Tetrodes, head-restrained VR |
| Campbell et al. 2018, Nat Neurosci 21:1096 | Gain changes rescale grids in 1D VR | "gain increases causing phase shifts, and gain decreases causing grid scale changes" | Discussion | full text | 21 mice; 151 open-field grid cells, 96 matched in VR | 1D VR track, 400 cm | Head-fixed mice; electrode type not re-checked |
| Mallory et al. 2018 | HCN1 knockout raises spacing to 42.16 ± 10.53 cm | see M1 | Results | full text | 62 grid cells, 17 knockout mice | as M1 | Tetrodes |

Two consequences follow. The Wen and Gu scales are VR-inferred, and Chen et al. found VR grids 1.42 times larger than real ones in open-field VR. So W3 and W4 may overstate real-world spacing by a similar factor, though Chen et al. tested open-arena VR, not a 1D track [my reading]. Ratios between modules recorded together should not depend on this gain.

I found no mouse-specific measurement of grid expansion in a novel environment. Mallory et al. mention transient expansion of grid and place scale in novel environments in their Discussion, citing other work, which I did not trace.

## 7. Mouse MEC to hippocampus topography (optional item)

Tsoi et al. 2022 (eLife 11:e73162) is mouse-specific but covers layer 5a back-projections, not the layer 2/3 perforant path. With MAPseq barcodes in 3 mice they found that "dorsal MEC neurons preferentially project to dorsal aspects of the hippocampus and ventral MEC neurons to the ventral aspects, indicating that the back-projections to the hippocampus are topographically organized (Figure 4C)" (Results) [full text]. Their Discussion says this "is similar to the dorso-ventral topography of perforant-path projections of the superficial EC targeting the dentate gyrus", but it cites Ruth et al. 1982 and Dolorfo & Amaral 1998, which are not mouse studies [full text]. Mallory et al. also cite rat anatomy (Witter et al. 2000) for the MEC to CA1 topography. I did not find a mouse tracing paper that states the perforant-path DV topography in its own data.

## 8. What I could not verify

- Fyhn et al. 2008. The Wiley full text and PDF returned HTTP 403 by curl and by WebFetch. Only the abstract was read, and it gives no numbers.
- Giocomo et al. 2011. Cell and ScienceDirect returned HTTP 403. Only the abstract was read. The full paper likely has wild-type spacing against DV depth, which would be the best mouse source for ventral columns.
- Gu et al. 2018 supplement PDFs (Figs. S2O to S2V). These might hold the source of "49 and 71 cm".
- Mallory et al. 2018 arena size and the DV range in Supp. Fig. 7b (Online Methods paywalled; figure not read).
- The Clark & Nolan arena size is inferred from Gerlei et al. 2020, not stated in the eLife paper.
- Whether "±" in Clark & Nolan is SD or SEM.

## 9. Proposed columns

Familiar-environment values where possible. Each range end names its source. "VR-inferred" means a 2D scale inferred from 1D virtual-track firing.

| Column | Spacing range | Low end | High end | Status |
|---|---|---|---|---|
| M1 | 30 to 36 cm | 30 cm, Zong et al. 2022, dorsal edge of the imaged area (Z1) | 36.26 cm, Mallory et al. 2018 control mean (M1); their DV intercept is 31.97 cm (M2) | full text at both ends. The 36 cm end is a population mean that may include some M2 cells |
| M2 | 42 to 60 cm | 42 cm [derived, 30 cm × 1.4, Z1 low end times the W1 low ratio] | about 60 cm, Zong et al. 2022 ventral edge (Z1). Gil et al. 2018 control mean 59.00 cm (G0) agrees, in a 70 cm box | Low end derived; high end full text |
| M3 | 65 to 76 cm | 65 cm, Wen et al. 2024, VR-inferred (W3) | 76 cm, Wen et al. 2024, VR-inferred (W3) | full text, but VR-inferred and above the cap of the 75 cm box they used. The open-field mean of 72.5 ± 13.7 cm from Clark & Nolan (C1) falls in this range |
| M4 | about 91 to 129 cm | 91 cm [derived, 65 × 1.4] | 129 cm [derived, 76 × 1.7] | derived only. No mouse measurement found |

A cross-check using Gu's average ratio of 1.51 from an M1 of 30 to 36 cm gives M2 = 45 to 54 cm, M3 = 68 to 82 cm and M4 = 103 to 124 cm [derived, successive multiplication by 1.51]. This agrees with the measured anchors above to within about 10%.

How many columns. Mouse data verified here support three modules with measured values, M1 to M3, all in dorsal or dorsomedial MEC. Gu et al. saw three in one region and Wen et al. averaged 2.7 per probe session. A fourth module is plausible from the ratios but is not measured. Anything more ventral rests on the Fyhn 2008 and Giocomo 2011 abstracts, which give the gradient without numbers.

## Sources

- Zong W, Obenhaus HA, Skytøen ER, et al. (2022). Large-scale two-photon calcium imaging in freely moving mice. Cell 185:1240-1256. doi:10.1016/j.cell.2022.02.017. PMC8970296 [full text].
- Mallory CS, Hardcastle K, Bant JS, Giocomo LM (2018). Grid scale drives the scale and long-term stability of place maps. Nat Neurosci 21:270-282. doi:10.1038/s41593-017-0055-3. PMC5823610 [full text]; Supp. Fig. 7 caption from https://www.nature.com/articles/s41593-017-0055-3.
- Gil M, Ancau M, Schlesiger MI, Neitz A, Allen K, De Marco RJ, Monyer H (2018). Impaired path integration in mice with disrupted grid cell firing. Nat Neurosci 21:81-91. doi:10.1038/s41593-017-0039-3 [full text, publisher PDF].
- Clark H, Nolan MF (2024). Task-anchored grid cell firing is selectively associated with successful path integration-dependent behaviour. eLife 12:RP89356. doi:10.7554/eLife.89356. https://elifesciences.org/reviewed-preprints/89356 [full text, VOR XML].
- Gerlei K, Passlack J, Hawes I, et al. (2020). Grid cells are modulated by local head direction. Nat Commun 11:4228. doi:10.1038/s41467-020-17500-1. PMC7445272 [full text, arena only].
- Wen JH, Sorscher B, Aery Jones EA, Ganguli S, Giocomo LM (2024). One-shot entorhinal maps enable flexible navigation in novel environments. Nature 635:943-950. doi:10.1038/s41586-024-08034-3. PMC11602719 [full text; Fig. 1d figure-read].
- Gu Y, Lewallen S, Kinkhabwala AA, Domnisoru C, Yoon K, Gauthier JL, Fiete IR, Tank DW (2018). A map-like micro-organization of grid cells in the medial entorhinal cortex. Cell 175:736-750. doi:10.1016/j.cell.2018.08.066. PMC6591153 [full text, author manuscript; supplement PDFs not read].
- Yoon K, Lewallen S, Kinkhabwala AA, Tank DW, Fiete IR (2016). Grid cell responses in 1D environments assessed as slices through a 2D lattice. Neuron 89:1086-1099. PMC5507689 [full text, searched only].
- Fyhn M, Hafting T, Witter MP, Moser EI, Moser MB (2008). Grid cells in mice. Hippocampus 18:1230-1238. doi:10.1002/hipo.20472 [abstract].
- Giocomo LM, Hussaini SA, Zheng F, Kandel ER, Moser MB, Moser EI (2011). Grid cells use HCN1 channels for spatial scaling. Cell 147:1159-1170. doi:10.1016/j.cell.2011.08.051 [abstract].
- Obenhaus HA, Zong W, Jacobsen RI, et al. (2022). Functional network topography of the medial entorhinal cortex. PNAS 119:e2121655119. PMC8851479 [full text].
- Chen G, King JA, Lu Y, Cacucci F, Burgess N (2018). Spatial cell firing during virtual navigation of open arenas by head-restrained mice. eLife 7:e34789. PMC6029848 [full text].
- Rowland DC, Obenhaus HA, Skytøen ER, et al. (2018). Functional properties of stellate cells in medial entorhinal cortex layer II. eLife 7:e36664. PMC6140717 [full text].
- Campbell MG, Ocko SA, Mallory CS, Low IIC, Ganguli S, Giocomo LM (2018). Principles governing the integration of landmark and self-motion cues in entorhinal cortical codes for navigation. Nat Neurosci 21:1096-1106. PMC6205817 [full text].
- Campbell MG, Attinger A, Ocko SA, Ganguli S, Giocomo LM (2021). Distance-tuned neurons drive specialized path integration calculations in medial entorhinal cortex. Cell Rep 36:109669. PMC8437084 [full text, searched only].
- Waaga T, Agmon H, Normand VA, et al. (2022). Grid-cell modules remain coordinated when neural activity is dissociated from external sensory cues. Neuron 110:1843-1856. PMC9235855 [full text, species check only: rats].
- Carpenter F, Manson D, Jeffery K, Burgess N, Barry C (2015). Grid cells form a global representation of connected environments. Curr Biol 25:1176-1182 [abstract, species check only: rats].
- Tsoi SY, Öncül M, et al. (2022). Telencephalic outputs from the medial entorhinal cortex are copied directly to the hippocampus. eLife 11:e73162. PMC8940174 [full text].
