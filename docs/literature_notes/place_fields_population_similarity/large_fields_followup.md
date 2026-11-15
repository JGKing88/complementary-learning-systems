# Very large spatial firing fields: targeted follow-up

Scope note for the report writer. This file follows up on the first-pass notes in the same folder (`field_size_regions_species.md`, `large_environments.md`, `subiculum.md`). It tries to verify numbers that the first pass had only from abstracts or secondary pages. Verification tags used below:

- [Abs] means I read the primary abstract in full through the Europe PMC API.
- [FT] means I read the primary full text through a fetch-tool reading of the PMC page. Quotes marked [FT] are as the fetch tool returned them.
- [Sec] means the number comes from a later paper or review citing the primary source.
- [1st] means the first pass verified the value in the primary full text and I did not re-read it.
- [Unverified] means no primary or peer-reviewed source confirmed it.

Access failures, so the writer knows what was tried. The Science full text and supplement for Kjelstrup 2008 returned HTTP 403. jneurosci.org returned 403 for Muller & Kubie 1987, Sharp & Green 1994 and Jung 1994. Their PMC copies (PMC6568940, PMC6577112, PMC6576902) are scanned PDFs behind a browser challenge that the fetch tool cannot pass. UCL pages hosting Burgess-lab PDFs (Hartley 2000, the publications list) returned 403. The NTNU doctoral thesis that contains the Kjelstrup data is listed as an open file, but its download endpoint returned 403. I did not try to get around any of these blocks.

## Kjelstrup et al. 2008 and the ventral hippocampus: field sizes by dorsoventral level

### Takeaway
The primary abstract confirms that the recordings were from CA3 on an 18 m linear track. It also confirms that the "scale of representation" rose almost linearly from under 1 m at the dorsal pole to about 10 m at the ventral pole, which is under 5.6% and about 56% of the track. The per-level means (0.98, 3.73 and 5.52 m), the "21 rats" count, the field definition and the cell counts are still unverified. No primary text, supplement or open-access citing paper I could reach reproduces them. Follow-up studies in rats and mice confirm that ventral fields are larger, more diffuse and less often present. None gives ventral field lengths in metres, but Keinath et al. 2014 state that ventral mouse CA1 cells "frequently span the entirety of the environment".

### Cited findings

Kjelstrup et al. 2008, primary record
- Citation: Kjelstrup KB, Solstad T, Brun VH, Hafting T, Leutgeb S, Witter MP, Moser EI, Moser MB (2008). Finite scale of spatial representation in the hippocampus. Science 321(5885):140-143. doi:10.1126/science.1157086. PMID 18599792. Europe PMC lists it as not open access, not in PMC, with no supplement deposited. [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:18599792&resultType=core&format=json)
- Method and result from the abstract: rats ran back and forth on an 18 m linear track while the authors recorded at several longitudinal levels. "CA3 cells had well-defined place fields at all levels. The scale of representation increased almost linearly from <1 meter at the dorsal pole to approximately 10 meters at the ventral pole." The authors conclude that environments are represented at "a topographically graded but finite continuum of scales". [Abs] [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:18599792&resultType=core&format=json)
  - Arithmetic: dorsal under 1 / 18 m = under 5.6% of the track. Ventral 10 / 18 m = 55.6% of the track.
  - The abstract says "scale". It does not say mean field length, maximum field length or how scale was measured.
- The region is CA3, not CA1. The abstract names CA3 only. [Abs] [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:18599792&resultType=core&format=json)

Solstad 2009 doctoral thesis (co-author's thesis containing the Kjelstrup data)
- Citation: Solstad T (2009). Neural representations of Euclidean space. Doctoral thesis, NTNU (thesis series 2009:167). Supervisors listed as E. I. Moser and M.-B. Moser. [NVA repository API record](https://api.nva.unit.no/publication/0198cc6087c8-637e01b6-4364-4aaa-a65b-0d9195b70f67) (landing page [hdl.handle.net/11250/264026](https://hdl.handle.net/11250/264026))
- The Norwegian summary says rats ran back and forth on a custom-built 18 m track. It says place cells and grid cells were found along the full length of both hippocampus and entorhinal cortex (thesis papers II and III). It says the scale of both cell types rose from places of less than 1 m at the dorsal end to "opp mot 10 meter" (up to about 10 m) at the ventral end. [NVA repository API record](https://api.nva.unit.no/publication/0198cc6087c8-637e01b6-4364-4aaa-a65b-0d9195b70f67)
- A search-engine snippet of the thesis's English abstract read "field length ranged from less than 1m to more than 10m" in hippocampus, and "less than 50cm to approximately 3m" for MEC grid field length, with grid spacing from under 1 m to at least 8 m. I could not open the English abstract or the PDF to check this wording. The Norwegian summary says "up to about 10 m", not "more than 10 m". [Unverified] [Search snippet source: NTNU Open handle](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/264026)

Per-level means, still unverified
- The values 0.98 ± 0.03 m (dorsal), 3.73 ± 0.43 m (intermediate) and 5.52 ± 0.54 m (ventral), and "21 rats", appear only on an aggregator page and in search-engine summaries. [Unverified] [Bohrium aggregator page](https://www.bohrium.com/paper-details/finite-scale-of-spatial-representation-in-the-hippocampus/811915281508073473-7984)
- I ran a Europe PMC full-text search of open-access articles for "Kjelstrup" together with "5.52", "3.73" or "0.98 m". The only "5.52" hit was a false positive, a value in an ion-channel table in Mishra & Narayanan 2021. The "3.73" hits were unrelated, and "0.98 m" returned none. No open-access peer-reviewed paper reproduces these means. [Europe PMC search, "Kjelstrup" AND "5.52"](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Kjelstrup%22%20AND%20%225.52%22&resultType=lite&format=json)
  - Arithmetic, if the means are correct: 0.98 / 18 = 5.4%, 3.73 / 18 = 20.7%, 5.52 / 18 = 30.7% of the track.

What citing papers say about Kjelstrup 2008 [Sec]
- Royer et al. 2010 state that in Kjelstrup et al. 2008, "<10%" of ventral CA3 neurons showed reliable phase precession. They say their own results "support and extend" Kjelstrup's. [Royer et al. 2010, PMC2825159](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)
- Saxena, Barde & Deshmukh 2018 (Inexpensive, scalable camera system for tracking rats in large spaces. J Neurophysiol 120:2383-2395, doi:10.1152/jn.00215.2018): "single place field seems to be the norm for place cells recorded on an 18-m-long track (Kjelstrup et al. 2008)". They contrast this with multiple fields per cell in large 2D arenas. [Saxena et al. 2018, PMC6295524](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6295524/fullTextXML)
- Fanselow & Dong 2010 describe ventral-pole place fields as "so large (e.g., 10 m" and say field size "increases dramatically" from dorsal to ventral. [Fanselow & Dong 2010, Neuron 65:7-19, doi:10.1016/j.neuron.2009.11.031, PMC2822727](https://pmc.ncbi.nlm.nih.gov/articles/PMC2822727/)
- Eliav et al. 2021 state that the 18 m Kjelstrup track passed through several small rooms, so it was not a single open compartment. [1st] [Eliav 2021 PDF](https://www.weizmann.ac.il/brain-sciences/labs/ulanovsky/sites/brain-sciences.labs.ulanovsky/files/2024-11/Eliav2021.pdf)

Royer et al. 2010, rat dorsal vs ventral CA3 [FT]
- Citation: Royer S, Sirota A, Patel J, Buzsáki G (2010). Distinct representations and theta dynamics in dorsal and ventral hippocampus. J Neurosci 30:1777-1787. doi:10.1523/JNEUROSCI.4681-09.2010. PMC2825159. [Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)
- Subjects and method: 6 Long-Evans rats, tetrodes (2 rats also had a 32-site silicon probe dorsally). Dorsal target AP −4.0, ML 2.6 mm. Ventral target AP −4.0, ML 2.0 mm at a 170° angle, reaching ML 4.5 mm at 8 mm depth. "Most recordings in the ventral part were performed in the CA3 region." [Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)
- Environments: an open maze with a 1 × 2 m floor and 30 cm walls, a 7-arm radial maze (arm length not given), and a zigzag maze made by adding 5 partitions to the open maze, giving 11 corridors. [Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)
- Field definition: the arena was divided into 200 × 200 pixels, and pixels with under 200 ms occupancy were dropped. Smoothing used a Gaussian with a 5-pixel half-width, or 15 pixels for ventral cells in the zigzag maze. Field size was measured two ways, as the fraction of pixels above 20% of peak rate and as the area inside the convex contour around all active pixels divided by total area. No minimum or maximum field size was stated. [Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)
- Cells: dorsal CA3 1,546 pyramidal and 107 interneurons in total, ventral CA3 351 pyramidal and 35 interneurons. Open field: 569 dorsal and 95 ventral pyramidal cells. [Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)
- Results: ventral CA3 had "significantly larger spatial coverage" in the open field by both measures (p < 0.01, Fig. 2B), and single-arm fields in the radial maze were "39% larger in vCA3 cells" (p < 0.01). Continuous 2D fields were "only exceptionally observed in vCA3". Most ventral cells "lacked the defining criteria of place cells" and fired "at multiple locations with interruption of activity between them". The text gives no field sizes in cm or as fractions. The values are only in Fig. 2B. [Royer et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2825159/)

Keinath et al. 2014, mouse dorsal vs ventral CA1 [FT]
- Citation: Keinath AT, Wang ME, Wann EG, Yuan RK, Dudman JT, Muzzio IA (2014). Precise spatial coding is preserved along the longitudinal hippocampal axis. Hippocampus 24:1533-1548. doi:10.1002/hipo.22333. PMC4447627. [Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)
- Subjects and sites: male C57BL/6 mice. Dorsal CA1 at AP −1.8, ML −1.5, DV −1.0 mm. Ventral CA1 at AP −3.0, ML −3.9, DV −2.8 mm. Visuospatial condition: dorsal 6 mice and 89 cells, ventral 5 mice and 65 cells. Olfactory condition: dorsal 5 mice and 89 cells, ventral 4 mice and 42 cells. [Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)
- Environments: 35 cm diameter cylinders (small) and a 70 cm diameter cylinder (large). [Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)
- Field definition: a 70 × 70 pixel grid, 3 cm SD Gaussian smoothing, and a field defined as contiguous pixels of at least 80 cm² with rate at least 20% of peak. Field size per cell is the sum over its fields. No upper cap was stated. [Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)
  - Arithmetic: the small cylinder is π × 17.5² = 962 cm², so the 80 cm² minimum is 8.3% of it. The large cylinder is π × 35² = 3,848 cm², so the minimum is 2.1%.
- Results: ventral fields were larger than dorsal fields, with the difference significant in the large environment, and ventral field size grew from small to large environment (P < 0.001). The Discussion describes ventral cells as "broadly tuned and frequently span the entirety of the environment" and says they "fired much more diffusely than dorsal cells". The paper gives no numeric field sizes (Table 1 has only F and P values). [Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)

Jung, Wiener & McNaughton 1994, rat dorsal vs ventral CA1 [Abs]
- Citation: Jung MW, Wiener SI, McNaughton BL (1994). Comparison of spatial firing characteristics of units in dorsal and ventral hippocampus of the rat. J Neurosci 14:7347-7356. doi:10.1523/JNEUROSCI.14-12-07347.1994. PMC6576902. [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:7996180&resultType=core&format=json)
- Task: food-reinforced random search in "a square chamber containing simple visual landmarks". Dorsal and ventral CA1 were recorded in opposite hemispheres of the same rats, often simultaneously. Among ventral complex-spike cells "a significantly smaller number had 'place fields'" and "the average spatial selectivity was of significantly lower resolution". Theta cells in both regions fired at high rates with low spatial selectivity. [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:7996180&resultType=core&format=json)

Mouse dorsal vs ventral CA1 on a short virtual track [FT]
- Citation: Chockanathan U, Padmanabhan K (2021). Divergence in population coding for space between dorsal and ventral CA1. eNeuro 8(5):ENEURO.0211-21.2021. doi:10.1523/ENEURO.0211-21.2021. PMC8425966. A bioRxiv preprint (doi:10.1101/2021.03.10.434839) preceded it. [eNeuro article](https://www.eneuro.org/content/8/5/ENEURO.0211-21.2021)
- Method: 4 head-fixed C57BL/6J mice, 128-channel silicon probes. dCA1 at 2.5 mm caudal, 1.5 mm lateral, 1 mm deep. vCA1 at 3.15 mm caudal, 3.15 mm lateral, 4.25 mm deep. A 1.9 m virtual track in 111 bins of about 1.7 cm. Field width was the number of bins around the peak with rate at least 1.5 SD above the mean. No size cap was stated. [eNeuro article](https://www.eneuro.org/content/8/5/ENEURO.0211-21.2021)
- Result: per-session mean field width was 9.4 ± 2.0 cm in dCA1 and 11.9 ± 6.3 cm in vCA1 (p = 0.84). Per neuron, it was 9.9 ± 10.9 cm (335 dCA1 neurons) and 10.1 ± 12.9 cm (228 vCA1 neurons). [eNeuro article](https://www.eneuro.org/content/8/5/ENEURO.0211-21.2021)
  - Arithmetic: 11.9 / 190 = 6.3% and 9.4 / 190 = 4.9% of the track.

### Inferences
- The only verified ventral number in metres is the abstract's "approximately 10 meters" scale, about 56% of the 18 m track. A field of 10 m on an 18 m track cannot be independent of the track boundary, so the true ventral scale in a longer environment is unknown.
- "Scale" in the Kjelstrup abstract may not mean mean field length. If the secondary means are right, the ventral mean (5.52 m, 31% of the track) is about half the abstract's 10 m scale. Report the two as different quantities.
- Ventral hippocampus in rats (Jung 1994, Royer 2010) and mice (Keinath 2014) shows broader and more diffuse firing, but also fewer cells meeting place-field criteria. Field-detection rules that require a compact, contiguous field may undercount the broadest ventral cells. Royer's statement that most vCA3 cells "lacked the defining criteria of place cells" points to this.
- The mouse virtual-track result (vCA1 about 12 cm vs dCA1 about 9 cm) shows that a ventral site does not by itself guarantee large fields. The vCA1 coordinates there (4.25 mm deep) may be intermediate rather than the ventral pole, and the track is short. This is my reading, not the authors' claim.

### Gaps
- Kjelstrup 2008 full text and supplement (kjelstrup.som.pdf on science.org) were not reachable (HTTP 403). Still unverified: field definition, bin size, smoothing, any size cap, dorsoventral coordinates per level, N rats, N cells per level, per-level means, the single largest field, track layout beyond "18 m linear", and whether the largest ventral fields covered most of the track.
- The Solstad 2009 thesis PDF (file "236839_FULLTEXT02.pdf") is listed as open but returned 403 from the repository download API.
- Jung 1994 chamber size, N, percentage of cells with fields and field areas: unreachable (scanned PMC PDF).
- Maurer et al. 2005 (Hippocampus 15:841-852): not in PMC. No new numbers.
- Royer 2010 field-size distributions exist only in Fig. 2B and Fig. 4B, which I could not read numerically.

## Classic scaling studies: Muller & Kubie 1987 and O'Keefe & Burgess 1996

### Takeaway
Muller & Kubie doubled a 76 cm cylinder to about 152 cm (4× the floor area). Of cells recorded in both, 36% kept a field at the same angular and relative radial position, 52% changed their firing pattern completely, and 12% were nearly silent in both. O'Keefe & Burgess recorded the same cells in four boxes that, per a later paper, ran from 61 × 61 to 122 × 122 cm. Peak locations held a fixed distance to the nearest walls, and several fields stretched along the lengthened axis, some becoming bimodal in the larger boxes. Neither abstract gives field areas, and I could not reach either full text, so no field size as a fraction of the box is verified.

### Cited findings

Muller & Kubie 1987 [Abs]
- Citation: Muller RU, Kubie JL (1987). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. J Neurosci 7:1951-1968. doi:10.1523/JNEUROSCI.07-07-01951.1987. PMID 3612226. PMC6568940 (scanned PDF only). [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)
- Apparatus: "a cylinder, 76 cm in diameter, with walls 51 cm high", uniformly gray with a white cue card covering 100° of arc. [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)
- Enlargement: "When the standard (small) cylinder was scaled up in diameter and height by a factor of 2, the firing fields of 36% of the cells observed in both cylinders also scaled, in the sense that the field stayed at the same angular position and at the same relative radial position. Of the cells recorded in both cylinders, 52% showed very different firing patterns in one cylinder than in the other. The remaining 12% of the cells were virtually silent in both cylinders." Similar results held for a small vs a large rectangular enclosure. [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)
  - Arithmetic: the large cylinder is about 152 cm across with 102 cm walls. Floor areas are π × 38² = 4,536 cm² (0.45 m²) and π × 76² = 18,146 cm² (1.81 m²), a 4× increase.
  - The abstract defines "scaled" by angular and relative radial position only. It does not state that field area grew 4×, so constant fractional coverage is not established from the abstract.
- Barriers: an opaque or transparent barrier bisecting a field "nearly abolished" it, "even though the barrier occupied only a small fraction of the firing field area". Changing the floor plan from circle to rectangle made the firing pattern unpredictable from one shape to the other. [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)
- The PubMed abstract is marked "(ABSTRACT TRUNCATED AT 400 WORDS)". [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612226&resultType=core&format=json)

Muller, Kubie & Ranck 1987, companion paper [Abs]
- Citation: Muller RU, Kubie JL, Ranck JB Jr (1987). Spatial firing patterns of hippocampal complex-spike cells in a fixed environment. J Neurosci 7:1935-1950. doi:10.1523/JNEUROSCI.07-07-01935.1987. PMID 3612225. PMC6568929. [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612225&resultType=core&format=json)
- The abstract reports "a sample of 40 place cells in a fixed environment", used to describe "the average size of fields" and their spatial distribution. It also says "a single cell may have more than one field in a given apparatus". The abstract does not give the average size. [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3612225&resultType=core&format=json)

O'Keefe & Burgess 1996
- Citation: O'Keefe J, Burgess N (1996). Geometric determinants of the place fields of hippocampal neurons. Nature 381:425-428. doi:10.1038/381425a0. PMID 8632799. [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:8632799&resultType=core&format=json)
- Abstract: the location and shape of place fields were studied "by recording from the same cell in four rectangular boxes that differed solely in the length of one or both sides". "Most of our results are explained by a model in which the place field is formed by the summation of gaussian tuning curves, each oriented perpendicular to a box wall and peaked at a fixed distance from it." [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:8632799&resultType=core&format=json)
- Box sizes [Sec]: Keinath, Epstein & Balasubramanian 2018 (a modelling paper) simulated "chamber lengths between 61 cm and 122 cm, chamber widths 61 cm or 122 cm; chamber sizes chosen to match experiment (O'Keefe and Burgess, 1996)". That implies boxes of 61 × 61, 61 × 122, 122 × 61 and 122 × 122 cm. [Keinath et al. 2018, eLife 7:e38169, doi:10.7554/eLife.38169, PMC6203432](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6203432/fullTextXML)
  - Arithmetic: floor areas run from 61 × 61 = 3,721 cm² to 122 × 122 = 14,884 cm², a 4× range. The rectangles are 7,442 cm².
- Field changes [Sec]: Barry et al. 2006 summarise the result as "the location of peak firing typically maintained a constant position to the nearest walls" and "several fields were stretched along the axes of the environment, with some becoming bimodal in the larger environments". [Barry et al. 2006, Rev Neurosci 17:71-97, PMID 16703944, PMC2677716](https://pmc.ncbi.nlm.nih.gov/articles/PMC2677716/)
- Barry et al. 2006 also state that in the boundary vector cell model the radial tuning width "increases linearly with the preferred tuning distance", so model fields far from walls are broader. The numeric parameters were not in the part of the review I read. [Barry et al. 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC2677716/)

### Inferences
- Both studies show that field size and shape follow the walls of a small enclosure. In Muller & Kubie only about a third of cells kept the same relative layout when the cylinder doubled, so "fields scale with the box" describes a minority of cells in that experiment.
- In O'Keefe & Burgess, stretching and splitting in the larger boxes mean a field can become longer along one axis or break into two peaks rather than stay a compact blob. The number of cells doing each is unverified.

### Gaps
- Muller & Kubie 1987: N rats and N cells per manipulation, pixel size, field-area criteria, and field areas in the small vs large cylinder. The PMC scan (13.2 MB) and jneurosci.org were unreachable.
- Muller, Kubie & Ranck 1987: the average field size in the 76 cm cylinder, which would give a baseline fraction of the apparatus.
- O'Keefe & Burgess 1996: N cells and rats, bin size, field definition, the counts that stretched, split or stayed fixed, and field size relative to the box. Box sizes rest on a later paper's statement.
- Hartley et al. 2000 (Hippocampus), which fits the boundary vector cell model to these data, was hosted only on a UCL page that returned 403.

## Subiculum: field areas and extents

### Takeaway
None of Sharp & Green 1994, Barnes et al. 1990 or Brotons-Mas et al. 2017 gives a field area I could verify. Their abstracts agree that subicular cells fire over much of the environment with graded modulation rather than compact fields. The verified quantitative extents come from later papers. Distal subiculum has a spatial activity fraction of 0.69 on a 350 cm track (Kim et al. 2012). Subiculum puts 95% of its spikes in 56% of a 150 × 150 cm box, against 26% for CA1 (Ledergerber et al. 2021). The boundary vector cell papers (Lever 2009, Stewart 2014) give no field areas.

### Cited findings

Sharp & Green 1994 [Abs]
- Citation: Sharp PE, Green C (1994). Spatial correlates of firing patterns of single cells in the subiculum of the freely moving rat. J Neurosci 14:2339-2356. doi:10.1523/JNEUROSCI.14-04-02339.1994. PMID 8158272. PMC6577112 (scanned PDF only). [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:8158272&resultType=core&format=json)
- Dorsal subiculum, open-field pellet searching. "Most subicular cells showed a robust locational signal." Cells "tended to fire throughout much of the environment, but showed graded, location-related rate modulation, such that there were some localized regions of high firing and other regions with relatively low firing." Distal cells showed "slightly higher average firing rates, spatial signaling, and firing field size" than proximal cells. The authors conclude that the subiculum sends "a robust, highly distributed spatial signal". [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:8158272&resultType=core&format=json)

Barnes et al. 1990 [Abs]
- Citation: Barnes CA, McNaughton BL, Mizumori SJ, Leonard BW, Lin LH (1990). Comparison of spatial and temporal characteristics of neuronal activity in sequential stages of hippocampal processing. Prog Brain Res 83:287-300. doi:10.1016/S0079-6123(08)61257-1. PMID 2392566. [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:2392566&resultType=core&format=json)
- "Neither entorhinal cortex nor subiculum exhibited the highly localized patterns of spatial firing observed in the CA fields", and their rates were "substantially higher". In subiculum there was "spatially consistent, but dispersed, firing in some cells, suggestive of the convergence of a number of CA1 place cells". Hippocampal output "appears to be expressed as a much more highly distributed spatial code". The abstract also reports that CA3 complex-spike cells were "significantly more spatially specific than their CA1 counterparts". [Abs] [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:2392566&resultType=core&format=json)

Kim, Ganguli & Frank 2012 [1st]
- Citation: Kim SM, Ganguli S, Frank LM (2012). Spatial information outflow from the hippocampal circuit: distributed spatial coding and phase precession in the subiculum. J Neurosci 32:11539-11558. doi:10.1523/JNEUROSCI.5942-11.2012. PMC3458125. [Kim et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458125)
- 5 rats, 91 subicular units, two U-shaped tracks 350 cm long. Median "spatial activity fraction" (a sparseness index the authors say "can be interpreted as the proportion of the environment over which a neuron fires") was 0.31 proximal, 0.54 middle and 0.69 distal subiculum vs 0.26 in distal CA1. Maps used 2 cm bins and 20 cm kernel bandwidth. Most subicular neurons had "multiple, irregularly spaced unitary place fields". [1st] [Kim et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458125)
  - Arithmetic: 0.69 × 350 = 242 cm for distal subiculum and 0.26 × 350 = 91 cm for CA1. This is an index of how spread out firing is, not a thresholded field length.

Ledergerber et al. 2021 [1st]
- Citation: Ledergerber D, Battistin C, Blackstad JS, Gardner RJ, Witter MP, Moser MB, Roudi Y, Moser EI (2021). Task-dependent mixed selectivity in the subiculum. Cell Rep 35:109175. doi:10.1016/j.celrep.2021.109175. PMC8170370. [Ledergerber et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8170370)
- 8 rats, 421 subicular and 325 CA1 principal cells, a 150 × 150 cm box, 5 cm bins. 95% of subicular spikes fell in 56.3% ± 0.7% of bins, against 25.7% ± 1.1% for CA1. [1st] [Ledergerber et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8170370)
  - Arithmetic: 0.563 × 22,500 cm² = 12,668 cm², about a 113 cm square.

Boundary vector cells [1st]
- Lever C, Burton S, Jeewajee A, O'Keefe J, Burgess N (2009). Boundary vector cells in the subiculum of the hippocampal formation (J Neurosci 29:9771-9777, doi:10.1523/JNEUROSCI.1319-09.2009, PMC2736390): 6 rats, 62 × 62 cm box. Field sizes are not quantified. Far-tuned boundary vector cells had broader fields. [Lever et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2736390)
- Stewart et al. 2014 (Phil Trans R Soc B 369:20120514, doi:10.1098/rstb.2012.0514, PMC3866438): in a 150 cm circle, boundary vector cell peaks sat 70.74 ± 1.26 cm from the centre, so about 4.3 cm from the wall. No field areas are given. "Boundary-off cells" fired everywhere except where a matching boundary vector cell would fire, so their firing covers most of the arena, but only a few were recorded. [Stewart et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3866438)

Subiculum in a large primate arena [FT]
- Mao et al. 2021: 3 rhesus macaques, tetrodes and single electrodes, a 3.30 m diameter, 2.12 m high circular arena. Of 110 subicular units, 5% passed a traditional spatial test (spatial information above 1 bit/spike, 6 × 6 cm bins, 9 cm Gaussian smoothing, 99th percentile of 500 shuffles), and 12% were spatially tuned in a model-based analysis. Hippocampus was 26% and 7%, entorhinal cortex 10% and 14%. The paper reports no field sizes. [Mao, Avila, Caziot, Laurens, Dickman & Angelaki 2021, Neuron, PMID 34644546, PMC8571030](https://pmc.ncbi.nlm.nih.gov/articles/PMC8571030/)

### Inferences
- Sharp & Green and Barnes describe the same thing that Kim and Ledergerber quantify: subicular firing spread over half or more of the environment. The broadest verified figure, 0.69 of a 350 cm track, comes from distal subiculum.
- Mao 2021's 1 bit/spike threshold would tend to exclude broadly tuned cells, so its low subicular percentage may reflect the criterion more than an absence of spatial signal. Kim 2012 says subicular neurons "sacrifice spatial selectivity for higher information capacity", which fits this.

### Gaps
- Sharp & Green 1994: cylinder size, pixel size, field definition, N, and field areas. The PMC scan was unreachable.
- Barnes 1990: methods and numbers. Only the abstract exists in Europe PMC.
- Brotons-Mas et al. 2017 (Neuroscience 343:174-189): abstract only in the first pass. No field areas found.
- Sharp 1997 and 1999, which recorded subiculum in more than one environment: not reached.
- Boundary vector cell field extents in cm, and Hartley 2000 model parameters: not reached.
- No subicular recording in an environment larger than the 3.3 m macaque arena turned up.

## Other regions and species with fields or tuning covering a third or more of the environment, and studies that cap field size

### Takeaway
Three new verified examples cover a third or more of an environment. Marmoset hippocampal fields on a 2.44 m track averaged 141.85 ± 45.96 cm, 58% of the track (N cells not stated). Lateral septum "phase fields" could cover "virtually the entire length" of a ~3.2 m maze, though lateral septum rate maps rarely showed place fields. Ventral mouse CA1 cells "frequently span the entirety" of 35 to 70 cm cylinders. Tingley & Buzsáki 2018 capped place-field width at 120 cm, which joins the first pass's caps (Tanni 2022 at half the map, Mankin at a quarter of the enclosure). In macaques in a 3.3 m arena and in rat MEC non-grid cells, no field sizes were reported.

### Cited findings

Marmoset hippocampus [FT]
- Citation: Courellis HS, Nummela SU, Metke M, Diehl GW, Bussell R, Cauwenberghs G, Miller CT (2019). Spatial encoding in primate hippocampus during free navigation. PLoS Biol 17:e3000546. doi:10.1371/journal.pbio.3000546. PMID 31815940. PMC6922474. [Courellis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6922474/)
- Method: 2 common marmosets (3 hemispheres), 64-channel microwire brush arrays. Units from DG, CA3 and CA1 were pooled. Main task: an L-track with arms of 1.83 m and 2.43 m (4.26 m total). Earlier sessions used a 2.44 m linear track. Only spikes during movement above 20 cm/s were used. Rate maps used 100 bins (about 4.3 cm per bin on the L-track) and a 5-point Gaussian filter, about 21.5 cm effective smoothing. Field width was the distance between the first drops below 20% of peak rate. Fields within 5 cm were merged, and otherwise the largest width was used. Place cells were defined by circular-shift shuffles (1,000 shuffles, p < 5 × 10⁻³). No maximum field size was stated. [Courellis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6922474/)
- Counts: 289 place cells of 2,054 single and multi-units (14.1%), from 37 sessions. The text gives 269 in some places. [Courellis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6922474/)
- Field size: the S1 Fig legend, covering the first spatially selective neurons recorded on the 2.44 m linear track over 2 sessions, gives "field size = 141.85 ± 45.96 cm". The legend does not state N cells. L-track widths appear only in Fig. 5A. The authors refer to "the relatively large place fields observed in many of the track recordings". Field width "significantly decreased the further posterior the recording was conducted", and anterior recordings are "expected to carry a more spatially diffuse representation of space". [Courellis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6922474/)
  - Arithmetic: 141.85 / 244 = 58.1% of the linear track. Mean plus 1 SD is 187.81 / 244 = 77.0%.

Rat lateral septum [FT]
- Citation: Tingley D, Buzsáki G (2018). Transformation of a spatial map across the hippocampal-lateral septal circuit. Neuron 98:1229-1242.e5. doi:10.1016/j.neuron.2018.04.028. PMID 29779942. PMC6605060. [Tingley & Buzsáki 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6605060/)
- Method: 5 Long-Evans rats (3 male, 2 female), silicon probes in lateral septum (dorsal and intermediate), CA1 and CA3. Three linear tracks and a circular maze. The text gives the full traversal of the maze as "~3.2 m". [Tingley & Buzsáki 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6605060/)
- Field definition: peak rate at least 2 Hz, field width between 8 and 120 cm, at least 1.5 spikes per trial on average, at least 10 trials. Field edges were where rate fell below 20% of peak. Bins were about 1.5 cm. [Tingley & Buzsáki 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6605060/)
- Results: of 4,393 lateral septum rate maps, 232 (5.3%) had a place field, against 1,820 of 5,735 (31.7%) hippocampal rate maps. Peak information was "0.4 bits/cm versus 1.7 bits/cm" for hippocampal fields. Lateral septum firing rates "lacked reliable tuning to spatial features of the task", but lateral septum phase fields "could cover virtually the entire length of the maze (~3.2 m)". The strength of phase coding rose with recording depth in lateral septum. Median field width was about 57 cm (38 bins); the fetch did not make clear which region this median refers to. [Tingley & Buzsáki 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6605060/)
  - Arithmetic: the 120 cm cap is 120 / 320 = 37.5% of a 3.2 m traversal. A field broader than that could not count as a place field in this paper. The individual linear-track lengths were not in the text I read.
- The first pass found that lateral septum rate fields on a T maze are no longer than CA1 fields (28.8 vs 29.1 cm, Wirtshafter & Wilson 2020). [1st] [Wirtshafter & Wilson 2020, eLife 9:e55252](https://doi.org/10.7554/eLife.55252)

Mouse ventral CA1 [FT]
- Keinath et al. 2014: ventral cells are "broadly tuned and frequently span the entirety of the environment" in 35 cm and 70 cm cylinders. Details are in the first section. [Keinath et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447627/)

Rat MEC non-grid spatial cells [FT]
- Citation: Diehl GW, Hon OJ, Leutgeb S, Leutgeb JK (2017). Grid and nongrid cells in medial entorhinal cortex represent spatial location and environmental features with complementary coding schemes. Neuron. PMID 28343867. PMC5444540. [Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/)
- 7 rats with tetrodes in superficial dorsal MEC. An 80 × 80 cm square or a 16-sided polygon with 50 cm radius, plus a 100 × 100 cm square analysed over its central 80 × 80 cm. 5 × 5 cm bins. Fields needed at least 250 cm² with edges at 0.3 × peak, and no maximum was stated. Counts were 48 grid cells and 236 non-grid spatial cells. Non-grid cells were classified by spatial information and within-session stability, not by fields, and are described as having "less specialized, yet consistent spatial firing patterns". The paper reports no field size or coverage for them. [Diehl et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444540/)
  - Arithmetic: the 250 cm² minimum is 250 / 6,400 = 3.9% of the 80 × 80 cm box.

Rhesus macaque hippocampus [FT]
- Mao et al. 2021, 3.30 m arena: 273 hippocampal units, 26% spatial by the 1 bit/spike criterion. No field sizes. The authors say preferred locations across cells "covered a broad space", which describes the population, not field size. [Mao et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8571030/)

Studies that cap or filter field size
- Tingley & Buzsáki 2018: place-field width limited to 8 to 120 cm. [Tingley & Buzsáki 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6605060/)
- Tanni et al. 2022 rejected candidate fields larger than half the rate-map bins, so a field covering more than 50% of an arena could not be counted. [1st] [Tanni et al. 2022, Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9616721/fullTextXML)
- Mankin et al. excluded fields of 25% of the enclosure or more, per the first pass. [1st] See `field_size_regions_species.md`, line 233, for the source link.
- Studies here with no stated upper cap: Royer 2010, Keinath 2014, Diehl 2017, Courellis 2019, the 2021 eNeuro mouse study. Spatial-information thresholds work like a cap on broad cells: Mao 2021 required 1 bit/spike, and Diehl 2017 classified non-grid cells by spatial information.

### Inferences
- Relative to the environment, the broadest verified spatial tuning is in hippocampal output and ventral or primate hippocampus, not dorsal rodent CA1. Approximate fractions: marmoset fields 58% of a 2.44 m track, distal subiculum activity fraction 69% of a 350 cm track, subiculum 56% of box bins, the ventral CA3 scale of about 56% of an 18 m track (abstract level), lateral septum phase fields near 100% of a 3.2 m maze, and ventral mouse CA1 "entirety" of a cylinder (qualitative).
- These measures differ in kind: thresholded field width (marmoset), a sparseness index (Kim), a spike-percentile area (Ledergerber), a "scale" (Kjelstrup), and theta-phase tuning (lateral septum). They should not be ranked against one another as if they were the same quantity.
- In absolute terms, the first pass's bat result (dorsal CA1 fields up to 32 m in a 200 m tunnel, about 17% of the 185 m flight path) is still the largest. [1st] [Eliav et al. 2021 PDF](https://www.weizmann.ac.il/brain-sciences/labs/ulanovsky/sites/brain-sciences.labs.ulanovsky/files/2024-11/Eliav2021.pdf)
- Field caps (120 cm, half the map, a quarter of the enclosure) and spatial-information thresholds remove exactly the cells this question asks about. Any comparison of "largest field" across studies needs each study's cap stated.

### Gaps
- Marmoset: N cells behind the 141.85 cm figure, and L-track field widths (Fig. 5A only).
- Lateral septum: lengths of the individual linear tracks in Tingley & Buzsáki, and which region the 57 cm median refers to.
- Nucleus reuniens, retrosplenial cortex and prefrontal cortex: I found no field sizes in metres or as a fraction of the environment, here or in the first pass.
- Humans: I did not find human single-unit studies that report field size as a fraction of a virtual or real environment.
- Ventral subiculum: no spatial field-size data found.
- Poucet, Thinus-Blanc & Muller 1994 ("Place cells in the ventral hippocampus of rats") may report ventral field sizes in a cylinder. Not reached.
