# Sweep for population kernels: similarity against distance within one environment

Sweep of 2026-11-20 and 21. The table is `sweep_kernel.csv` in this folder, 45 rows from 24 studies. Of these, 14 studies were already in the review's kernel tables or notes and 10 are new. Rows K01 to K26 re-use the covered studies. Rows N01 to N19 are new.

The new data add eight kernels with a number: mouse dorsal CA1 in two labs, rat dorsal CA3 from a second study, rat dentate gyrus, rat subiculum, marmoset hippocampus at three long-axis levels, a juvenile rat track and a bird. Every new rodent kernel is local. The only broad one is the anterior marmoset hippocampus on a 4.26 m track, where the correlation falls to 0.5 about 1.3 m from a position.

## Method

### Search

1. Europe PMC full-text queries for "population vector correlation", "PV correlation", "population vector overlap", "ensemble correlation" and "pattern similarity", combined with distance phrases and region terms. Eight queries returned about 500 records, and I screened the titles.
2. Targeted web searches for dorsal against ventral comparisons, dentate gyrus, CA3, bats, long tracks and human fMRI.
3. Full text came from four routes. Europe PMC `fullTextXML` served open-access papers. NCBI BioC JSON and the PMC open-data bucket served NIH author manuscripts, as the coordinator suggested. WebFetch on PMC pages worked until PMC began returning a bot check. Publisher figure servers were the fourth route: PLoS, Springer, the eLife IIIF server and the PMC CDN `cdn.ncbi.nlm.nih.gov/pmc/blobs/...`.
4. Supplements came from the Europe PMC `supplementaryFiles` endpoint. It returns a zip with every figure and the supplement PDF of an open-access paper. Ledergerber 2021 Fig. S7H and Green 2022 Fig. 3 came from it.

I opened about 75 papers and grepped each for sentences that mention population similarity together with distance, separation, diagonal or width. Ten new studies contained a kernel, a related within-environment measure or a citable claim. About 15 had only between-environment, between-session or between-direction matrices, and they are listed under "Excluded". About 40 had nothing relevant, and they are listed under "Checked, no kernel".

### How numbers were read

- `dist_r05_cm` is the distance from the diagonal at which the mean similarity first falls to 0.5. `width_r05_full_cm` is twice that, both sides of the diagonal. When a paper states a "half-width", I checked against its figure whether it is one-sided or full. In Kjelstrup 2008 and Brun 2008 it is full. In Resnik 2012 it is one-sided: the figure reaches 0.5 about 7 to 11 cm from zero, and the text gives 9.0 cm.
- Curves were traced in pixels. For each column I took the mean row of pixels matching the curve colour and converted with two-point axis calibrations read from tick labels. `scripts/trace.py` does this, with a log axis for Ledergerber.
- Matrices were read by colour-bar inversion. Each pixel maps to the nearest colour-bar colour, and the values are averaged along lines parallel to the diagonal over both signs of offset. `scripts/matrix_band.py` does this. These reads are coarse. Where the colour bar is clipped, as in Farooq 2024 below 0.2, values past the clip are not readable.
- Status tags in the table are "full text", "figure-read" with the method, "derived", "abstract" and "preprint". A WebFetch excerpt counts as full text only for the short quote it returned. No search snippet is recorded as verified.
- Scripts are in `/home/jackking/.claude/jobs/2c8b1c8e/tmp/sweep_kernel/scripts/` and downloads in `.../sweep_kernel/dl/`. Working notes with every read value are in `.../sweep_kernel/rows_notes.md`.

### Measures differ

The far floor depends on the construction:

- Per-cell z-scoring, as in Green 2022, gives a negative tail near -0.3.
- Mean normalization with Pearson correlation, as in Resnik 2012, gives about -0.1.
- Unnormalized rates of place cells give a positive floor: +0.2 in Raveau 2018 with Spearman correlation, +0.25 to +0.3 in Terrazas 2005 with a dot product.
- Cross-validated designs, which correlate the first half of a session with the second or odd laps with even ones, put the diagonal below 1. Blair 2023's diagonal is 0.59, so its r = 0.5 distance understates the width. The table also gives its half-peak distance.

Compare widths only between rows that share a measure, or use the half-peak distance.

## New kernels

| Row | Study | Animal and region | Space | dist_r05 | Zero crossing and floor | Field width, same study |
|---|---|---|---|---|---|---|
| N01 | Resnik 2012 | Mouse dorsal CA1, 86 cells, 11 mice | 1.4 m track | 9.0 cm, text | 26 to 29 cm; -0.1 | 19 cm median |
| N02 | Resnik 2012 | Mouse dorsal CA1, GluA1 knockout | 1.4 m track | 4.6 cm, text | near 0 tail | 17 cm median |
| N03 | Raveau 2018 | Mouse dorsal CA1, 5 mice, pooled place cells | 1.7 m track | 13.5 cm | none; +0.22 | area only |
| N05 | Courellis 2019 | Marmoset, anterior hippocampus | 4.26 m L track | about 128 cm | none; +0.3 | about 92 cm |
| N06 | Courellis 2019 | Marmoset, middle | 4.26 m L track | about 57 cm | about 178 cm; rises to +0.3 far | about 87 cm |
| N07 | Courellis 2019 | Marmoset, posterior | 4.26 m L track | about 42 cm | near 160 cm; rises to +0.42 near 316 cm | about 68 cm |
| N08 | Gothard 2001 | Rat CA1, 144 cells, light | 1.76 m track | about 14 cm | about 79 cm; near 0 | none |
| N10 | Gothard 2001 | Rat dentate gyrus, 125 putative granule cells, light | 1.76 m track | about 14 cm | about 48 cm; near 0 | none |
| N11 | Gothard 2001 | Rat dentate gyrus, darkness | 1.76 m track | about 9 cm | about 42 cm; near 0 | none |
| K09 | Green 2022 | Rat dorsal CA3, 275 cells | 1.8 m track | 18.4 cm | 31.6 cm; -0.32 | none |
| K19, K21 | Ledergerber 2021 | Rat dorsal CA1, 227 to 305 cells | 1.5 m box | 22 cm | 53 to 56 cm; -0.14 to -0.17 | none |
| K20, K22 | Ledergerber 2021 | Rat dorsal subiculum, 191 to 418 cells | 1.5 m box | 19 to 19.5 cm | 60 to 62 cm; -0.14 to -0.17 | none |
| N12 | Chettih 2024 | Chickadee hippocampus, 5 birds | 76 cm arena | about 4.7 cm | about 20 cm; -0.3 | none |
| N13 | Farooq 2024 | Juvenile rat dorsal CA1, first exposure | 1 m track | about 15 cm | clipped; about 0.07 at 80 to 88 cm | none |
| N16 | Blair 2023 | Rat dorsal CA1, 9 rats, miniscope | 2.5 m path | 17 cm; half-peak 51 cm | none by 200 cm; +0.05 | 44 to 50 cm |

Ledergerber 2021 now has numbers. The review says the subicular peak is narrower than CA1's, and the supplement shows it is only slightly narrower: 19 against 22 cm at r = 0.5 in both tasks. The subicular curve also crosses zero later. So broad subicular tuning does not widen the population kernel, and it does not narrow it much either.

Green 2022 gives a second dorsal CA3 kernel on a short track. Within one trial type it is 18.4 cm against 19.5 cm for CA1. That matches the text's statement that CA1 and CA3 were "remarkably similar".

The marmoset result follows the rodent long-axis gradient. The anterior matrix is two blocks, one per arm, and the elbow of the L bounds it. The authors read this as "seemingly distinct subcomponents of the place-cell population being sensitive to the long and short arms". All three levels use 21.5 cm of effective smoothing, which widens them. The posterior and middle matrices rise again far from the diagonal, up to +0.42 at 316 cm. Off-diagonal blocks cause this rise, which is aliasing within one environment.

## Coverage

Each cell counts rows with a usable width. In brackets is the count of all rows, including qualitative or non-kernel rows.

| Species | CA1 dorsal | CA1 intermediate or ventral | CA3 dorsal | CA3 intermediate | CA3 ventral | Dentate gyrus | CA2 | Subiculum | MEC | Other |
|---|---|---|---|---|---|---|---|---|---|---|
| Rat | 12 [15] | 0 [1, Maurer abstract] | 3 | 1 | 1 | 2 | 0 | 2 | 3, Brun grid cells | 0 |
| Mouse | 4 [5] | 0 | 0 [1, Gandit qualitative] | 0 | 0 | 0 [1, Gandit qualitative] | 0 | 0 | 0 [1, Dasbehera preprint] | RSC 0 [1, zero crossing only] |
| Marmoset | hippocampus, posterior 1 | hippocampus, middle 1 and anterior 1 | not separated | | | | | | | |
| Bat | 0 [1, Maimon route history] | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | |
| Bird | | | | | | | | | | hippocampus 1 |
| Human | | | | | | | | | | hippocampus fMRI 0 [3, no length scale] |

The 12 usable rat dorsal CA1 rows count each condition separately: Terrazas walking and driving, Blair drug-free and scopolamine, Gothard light and dark, Green control and jump trials, Ledergerber open field and task, Wang and Farooq. The three without a usable width are Battaglia 2004 and Petersen 2020, whose numbers sit only in figures, and Tanni 2022, which is not a kernel.

## Kernel width against field width

Field widths below come from the same study. The definitions differ, so the ratios are rough.

| Row | Region, level | dist_r05 | Full width | Field width and definition | dist / field | full / field |
|---|---|---|---|---|---|---|
| K01 | Rat CA3, dorsal | 40.5 | 81 | 98, mean largest field, 20% threshold | 0.41 | 0.83 |
| K02 | Rat CA3, intermediate | 186 | 371 | 373 | 0.50 | 0.99 |
| K03 | Rat CA3, ventral | 333 | 666 | 552, some truncated | 0.60 | 1.21 |
| K10 | Rat CA1, dorsal | 14 | 28 | 35, median, fields at 10% of peak (reference 0.39) | 0.40 | 0.80 |
| K11 | Rat CA3, dorsal | 20 | 40 | 42, median, fields at 10% of peak (reference 0.39) | 0.48 | 0.95 |
| N01 | Mouse CA1, dorsal | 9.0 | 18 | 19, median, 20% threshold | 0.47 | 0.95 |
| N02 | Mouse CA1, GluA1 knockout | 4.6 | 9 | 17 | 0.27 | 0.54 |
| N07 | Marmoset, posterior | 42 | 84 | 68, median largest field, 20% | 0.62 | 1.24 |
| N06 | Marmoset, middle | 57 | 114 | 87 | 0.66 | 1.31 |
| N05 | Marmoset, anterior | 128 | 256 | 92 | 1.39 | 2.78 |
| N16 | Rat CA1, dorsal, cross-half | 17; half-peak 51 | 34; 102 | about 47, bins above 50% of peak | 0.36; half-peak 1.09 | 0.72; 2.2 |
| K04 to K06 | Rat MEC grid cells, three levels | 16 to 33.5 | 32 to 67 | 56 to 119, all fields, 10% threshold (compare with the 10% reference 0.39, not 0.46) | 0.26 to 0.29 | 0.51 to 0.57 |

What the new data say:

1. In two studies with fields measured at 20% of peak, rat CA3 at three long-axis levels on an 18 m track and mouse dorsal CA1 on a 1.4 m track, the correlation falls to 0.5 at 0.41 to 0.60 field widths (full width 0.83 to 1.21), against 0.46 for identical Gaussian fields. Rat dorsal CA1 and CA3 on a 2 m loop, with fields at 10% of peak, give 0.41 and 0.47, against 0.39 for Gaussian fields at 10%. This holds from mouse dorsal CA1 at 19 cm fields to rat ventral CA3 at 5.5 m fields. My derivation gives a reference value. For identical Gaussian fields tiled evenly, the PV correlation falls to 0.5 at 0.46 of the width at 20% of peak. The rodent rows sit close to that.
2. Two rows depart from this. The GluA1 knockout has fields as wide as wild type but half the kernel width, 0.27 field widths. Resnik et al. write that "individual place fields had similar widths in the two genotypes" and that the wild-type population code was "much more spatially coherent". So kernel width is not set by field width alone. The anterior marmoset kernel is 1.4 field widths, wider than its fields, because cells active over a whole arm make blocks.
3. Grid cells give a narrower kernel than their fields: 0.26 to 0.29 field widths at r = 0.5.
4. Where a long-axis gradient was measured, the kernel grows with the field, and faster than it. The ratio rises from 0.41 to 0.60 dorsal to ventral. Kjelstrup warn that their field measures underestimate ventral scale, which raises the ventral ratio. Kjelstrup's ratio runs 0.41, 0.50 and 0.60 from dorsal to ventral, and the marmoset's 0.62, 0.66 and 1.39 from posterior to anterior.

### Kernel width against grid spacing

Only Brun 2008 measured both in the same animals. For grid cells, the full width at r = 0.5 is 0.16 to 0.19 of the median grid spacing on the track at all three levels.

No study gives a hippocampal kernel and grid spacing in the same animals. A derived comparison uses Kjelstrup's CA3 levels against Brun's MEC levels on the same track in different rats, with the review's level matching. The CA3 full width over the MEC median spacing is 81/171 = 0.47 dorsal, 371/301 = 1.23 intermediate and 666/370 = 1.80 ventral. Using dist_r05 instead halves these.

## Excluded: between environments, sessions or directions

| Study | What the matrix compares | Reason |
|---|---|---|
| Rubin et al. 2015, eLife | Matching positions on a 96 cm straight track and an L-shaped track | Between environments |
| Cabral et al. 2014, J Neurosci | Clockwise against counter-clockwise runs, normalized distance from departure | Between running directions |
| Chen et al. 2025, PNAS | One running direction against the other, with an antidiagonal for distance coding | Between running directions |
| Gandit et al. 2024, iScience | In against out and flipped runs, Figs. 2 and 3 | Between running directions. Their odd-even matrices are rows N14 and N15 |
| Ravassard et al. 2013, Science | PV overlap of bidirectional cells along the -45 degree diagonal | Between running directions |
| Lee, Yoganarasimha, Rao & Knierim 2004, Nature | Standard 1 against standard 2, and standard against mismatch sessions | Between sessions and conditions |
| Blair et al. 2023, eLife | Pre-training against training sessions, 48 h apart | Between sessions. The within-session matrix is rows N16 and N17 |
| Kong et al. 2026, Neuron; Sheintuch et al. 2023, Cell Rep; Hainmueller & Bartos 2018, Nature | Same position across days, or across contexts | Between sessions or environments |
| Navratilova et al. 2026, Prog Neurobiol | Late against early laps, familiar against novel environment, single example sessions | Mostly between environments. The familiar late-against-early matrix is within session but only one example session, so it was not read |
| Gothard et al. 2001, Boxes 2 to 5 | Shortened journeys against the full journey | Between conditions. Box 1 is rows N08 to N11 |
| Farooq & Dragoi 2024, Figs. 5 and 6 | Three tracks against each other | Between environments. The within-track matrix is row N13 |
| Jung et al. 2019, Sci Rep | PV correlation against distance from an added cue, before against after | Between conditions |
| Qian, Li & Magee 2025, Nat Neurosci | Familiar against novel belt, pre against post | Between conditions |
| Maimon et al. 2026, Nature | Flights reaching one place by different routes | Not excluded but not a kernel. It measures route history and is kept as row K23 |
| Priestley et al. 2022, Neuron | Lag-shifted PV correlation between two trials in a novel environment | Between single trials during learning. Not read, figure needed |

Decoding error only, with no kernel, in Bourboulou 2019, Aronov & Tank 2014 and Stefanini 2020. Bourboulou gives Bayesian errors of 27 cm with virtual objects against 46 cm without, but no PV curve.

## Checked, no kernel

The full text was searched. None had a within-environment similarity curve or matrix:

- Gonzalez et al. 2019, main text and supplement; Keinath et al. 2014; Royer et al. 2010.
- Maurer et al. 2012, whose theta-time against space correlation is not position against position.
- Hazon et al. 2022; Najafian Jazi 2023; Spiers et al. 2015, which has single-cell map correlations between compartments only.
- Varga 2024; Kapl 2022; Duan 2025; Sosa 2025; Inayat 2026, a locomotion-in-place task with no spatial positions.
- Cholvin 2021, 2022 and 2026; Allegra 2020; Hainmueller 2024 and 2026; Nyilas 2025; Krishnan 2022; Grienberger & Magee 2022.
- Golipour 2026; Yagishita 2026; Vaidya 2025; Miranda 2024; Zhang 2024; Huelin Gorriz 2023; Zhu 2023; Rechavi 2022; Tuncdemir 2022; Keinath 2020; Arriaga & Han 2019; Sun 2021.
- Zutshi 2022; Widloski & Foster 2022; Senzai & Buzsaki 2017; Danielson 2016; Fenton et al. 2008; McKenzie 2021. These were read through BioC.

## Gaps

1. No dorsal CA1 kernel exists for any rodent environment larger than a 4 m ring or a 2.5 m path. The 48 m track, the 18.6 m² arena and the bat tunnels still have no kernel. The 4.26 m marmoset track is the largest new space.
2. Ventral and intermediate rodent hippocampus has only Kjelstrup's CA3. Maurer 2005 compared dorsal and middle CA1, but only the abstract is available. No mouse ventral CA1 kernel was found.
3. The dentate gyrus has one numeric kernel, Gothard 2001, with "putative" granule cells on a 1.76 m track. CA2 has none.
4. The subiculum has one study in one 1.5 m box. No large-space subiculum data exist.
5. No study measured a hippocampal kernel and grid spacing in the same animals. Brun 2008 is grid cells only.
6. Few studies give kernel and field width together: Kjelstrup, Wang, Resnik, Courellis, Blair and Brun. Gothard, Ledergerber, Chettih and Farooq give no field width in cm. Green gives field-width distributions in Fig. 8F and 8H (width holding 95% of spikes), with no number in the text.
7. Bats have no spatial kernel. Maimon's analysis measures route history over a 10 m boxcar.
8. Human data have no length scale. The fMRI studies regress similarity on distance or on log distance.
9. Most kernels pool cells across animals and sessions into pseudo-populations. Only Chettih, Gandit, Mao and Petersen work within one simultaneous recording. Farooq and Blair compute per animal and then average.

## Papers to request

In order of value:

1. Maurer, VanRhoads, Sutherland, Lipa & McNaughton 2005, Hippocampus 15:841. doi:10.1002/hipo.20114. Dorsal pole against middle CA1 PV decorrelation distances. The only other long-axis kernel in rats, and only the abstract is available.
2. Battaglia, Sutherland & McNaughton 2004, J Neurosci 24:4541. doi:10.1523/JNEUROSCI.4896-03.2004. Fig. 7 decorrelation distances on cue-rich and cue-poor tracks. PMC6729394 is not in the open-access set and the PMC page shows a bot check.
3. Petersen & Buzsaki 2020, Neuron 107:731. doi:10.1016/j.neuron.2020.05.023. Fig. 3h and 3i give the per-session distance at which PV correlation drops to 0.5 for 21 sessions on a figure-8 maze. The text is available, but no figure route worked.
4. Gothard, Skaggs & McNaughton 1996, J Neurosci 16:8027. doi:10.1523/JNEUROSCI.16-24-08027.1996. The original PV matrix method on a linear track. It may contain within-condition matrices.
5. GoodSmith et al. 2017, Neuron 93:677. doi:10.1016/j.neuron.2016.12.026. Dentate granule and mossy cells. Not checked; it is a possible second DG kernel.
6. Nordlund et al. 2025, Curr Biol. doi:10.1016/j.cub.2025.07.050. The preprint matched "population vector correlation" with "as a function of distance". Neither the preprint nor the paper could be opened.
7. Guardamagna, Stella & Battaglia 2021, bioRxiv. doi:10.1101/2021.12.22.473863. It matched PV correlation with distance in CA1 and was not opened.
8. Herweg et al. 2023, J Neurosci. doi:10.1523/JNEUROSCI.0181-22.2023. Human medial temporal lobe recordings with a learned map of places. It may give a length scale in a virtual environment.
9. Courellis et al. 2019, S3 Table. Per-site place-cell counts behind the three marmoset matrices. It is open access, but the table was not fetched.

## Possible follow-ups with files already in hand

- Wang 2020's 2D cross-correlograms on 1 by 1 m boards, perpendicular to texture boundaries. The PDF text is in `pdftext/wang_2020.txt`. Only the 1D ring is in the table.
- Navratilova 2026, Fig. 3A, familiar late against early laps for CA1 and RSC. One example session.
- Farooq 2024 has an unclipped colour scale in its supplement, if one exists. That would give the floor of the juvenile kernel.
