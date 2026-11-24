"""Build the heat-map rows of ratio_spec.json from the audited sweep tables.

    python aggregate_sweeps.py [SPEC.json] [--report OUT.md]

Field-width rows ("Rat", "Mouse") come from sweep_fields_rat.csv and sweep_fields_mouse.csv:
one row per region x long-axis level x environment class. A row's range runs from the smallest
to the largest contributing value, and its "studies" field lists the contributing studies.
The mouse grid also takes rows from sweep_fields_rsc.csv and sweep_fields_mec_nongrid.csv, one
per region x environment class, each carrying a "group" ("Retrosplenial cortex" or "MEC, non-grid
cells") that the heat map prints as a header; the rat grid takes the rat rows of those groups
(Butler 2019's non-grid MEC widths). Kernel rows ("Rat kernel", "Mouse kernel") come from
sweep_kernel.csv and its audits, plus the two Alexander & Nitz rat CA1 curves in
sweep_kernel_rsc.csv: one row per study x region x level x condition, as the full width at which
population-vector correlation falls to 0.5. Both kinds keep the existing columns.
Every per-row decision (exclusion, value substitution, environment class) is written out below
with its reason, and --report prints the exclusion and contribution tables used in the notes.
"""
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES = HERE.parents[1] / "docs" / "literature_notes" / "place_fields_population_similarity"
LEVELS = ("dorsal", "intermediate", "ventral")

# ---------------------------------------------------------------- shared helpers

SHORT = {"leesm2022": "Lee SM 2022", "abadperez2025": "Abad-Perez 2025",
         "goodsmith2022": "GoodSmith 2022", "hainmueller2018": "Hainmueller 2018"}


def study_of(row_id):
    return row_id.rsplit("_", 1)[0]


def study_key(r):
    """'mao2017_rsc_1' -> 'mao2017'; ids without a study prefix (MNG01) use the citation."""
    m = re.match(r"([a-z]+\d{4})_", r["id"])
    if m:
        return m.group(1)
    m = re.match(r"([A-Za-z]+)\b.*?\((\d{4})\)", r["citation"])
    return f"{m.group(1).lower()}{m.group(2)}"


def is_virtual(env_type):
    return env_type == "VR track" or "virtual" in env_type.lower()


def short(study):
    if study in SHORT:
        return SHORT[study]
    m = re.fullmatch(r"([a-z]+)(\d{4})", study)
    return f"{m.group(1).capitalize()} {m.group(2)}"


def first_num(text):
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def num_range(text):
    """'98' -> (98, 98); '50.5 to 58.6' -> (50.5, 58.6); anything else -> None."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(?:\s+to\s+(\d+(?:\.\d+)?))?\s*", text or "")
    if not m:
        return None
    lo = float(m.group(1))
    return lo, float(m.group(2)) if m.group(2) else lo


def tag(status):
    """CSV status -> review tag text, e.g. 'FT; derived' -> 'full text, derived'."""
    s = status.lower()
    kinds = []
    for key, name in (("preprint", "preprint"), ("full text", "full text"), ("ft", "full text"),
                      ("secondary", "secondary"), ("figure-read", "figure-read"),
                      ("derived", "derived"), ("computed", "derived")):
        if re.search(r"\b" + re.escape(key) + r"\b", s) and name not in kinds:
            kinds.append(name)
    order = ["preprint", "full text", "secondary", "figure-read", "derived"]
    return ", ".join(k for k in order if k in kinds)


def load(name):
    with open(NOTES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def doi_url(text):
    m = re.search(r"10\.\d{4,9}/[^\s;()]+", text or "")
    if m:
        return "https://doi.org/" + m.group().rstrip(".")
    m = re.search(r"https?://\S+", text or "")
    return m.group() if m else ""


# ---------------------------------------------------------------- rat field widths

RAT_CLASS_LABEL = {"track": "real track up to 3 m", "arena": "real arena up to 4 m²",
                   "large": "large real track or arena"}

RAT_EXCLUDE = {
    "kjelstrup2008_7": "population-vector band at r = 0.5, not a field width; same cells as _1 to _3; in the kernel grid",
    "kjelstrup2008_8": "population-vector band at r = 0.5, not a field width; same cells as _1 to _3; in the kernel grid",
    "kjelstrup2008_9": "population-vector band at r = 0.5, not a field width; same cells as _1 to _3; in the kernel grid",
    "royer2010_1": "ratio only (ventral single-arm fields 39% larger), no width",
    "royer2010_2": "coverage of all suprathreshold pixels, an upper bound for one field; may share cells with mizuseki2012_4",
    "royer2010_3": "coverage of all suprathreshold pixels, an upper bound for one field",
    "neunuebel2012_1": "width cleared by the audit: the area unit cannot be resolved",
    "neunuebel2012_2": "width cleared by the audit: the area unit cannot be resolved",
    "grieves2020_1": "diameter of the minimum enclosing sphere, a maximum extent",
    "grieves2020_2": "diameter of the minimum enclosing sphere, a maximum extent; 3D lattice",
    "schlesiger2015_1": "track length not stated, so no environment class",
    "ravassard2013_1": "track length not stated, so no environment class",
    "ravassard2013_2": "virtual track; the rat classes cover real environments only",
    "hayman2011_2": "vertical pegboard, a climbing surface that fits no class",
    "hayman2011_3": "helical climbing track that fits no class",
    "shen1997_1": "status unverifiable (values from WebFetch fragments only)",
    "shen1997_2": "status unverifiable; also aged rats",
    "hollup2001_1": "status unverifiable",
    "sharp1994_1": "share of visited pixels with any spike, a coverage upper bound (as Royer)",
    "sharp1994_2": "share of visited pixels with any spike, a coverage upper bound (as Royer)",
    "kim2012_1": "spatial activity fraction, not a width",
    "kim2012_2": "spatial activity fraction, not a width",
    "kim2012_3": "spatial activity fraction, not a width",
    "kim2012_4": "spatial activity fraction, not a width",
    "mankin2015_1": "CA2 size only relative to CA1 (24.5% larger), no width",
    "park2011_5": "no width (the DG value is only in Fig. 4B, not read)",
}
# Values that need more than the width_cm column, with the reason.
RAT_VALUE = {
    "tanni2022_6": (56, 177, "projection on the long axis, 56 to 120 cm near the wall, 132 to 177 cm in the centre"),
    "hayman2011_1": (21, 38, "minor and major axes of the same main fields"),
    "diamantaki2016_1": (21.9, 30.6, "cells 102, 103 and 1046, all fields summed; cell 104 left out (O-maze, an equal-area diameter on a track)"),
}
# Environment class where the size columns do not settle it.
RAT_CLASS = {"mizuseki2012_5": "track", "mizuseki2012_6": "track", "mizuseki2012_7": "track",
             "mizuseki2012_8": "track", "wirtshafter2020_1": "track", "komorowski2013_1": "arena",
             "komorowski2013_2": "arena", "komorowski2013_3": "arena", "hayman2011_1": "arena",
             "diamantaki2016_1": "arena"}
RAT_REGION = {"CA1": "CA1", "CA3": "CA3", "CA1+CA3 (pooled)": "CA1 and CA3 pooled",
              "DG-granule": "dentate granule cells", "DG-mossy": "mossy cells",
              "DG-unspecified": "dentate gyrus", "SUB": "subiculum", "CA2": "CA2"}
# Where an excluded row with a width (or a near-width) would have gone. None: no row exists.
RAT_ATTACH = {
    "kjelstrup2008_7": ("CA3", "dorsal", "large"), "kjelstrup2008_8": ("CA3", "intermediate", "large"),
    "kjelstrup2008_9": ("CA3", "ventral", "large"), "royer2010_2": ("CA3", "dorsal", "arena"),
    "royer2010_3": ("CA3", "ventral", "arena"), "grieves2020_1": ("CA1", "dorsal", "arena"),
    "schlesiger2015_1": ("CA1", "dorsal", "track"), "ravassard2013_1": ("CA1", "dorsal", "track"),
    "ravassard2013_2": ("CA1", "dorsal", "track"), "shen1997_1": ("CA1", "dorsal", "track"),
    "shen1997_2": ("CA1", "dorsal", "track"), "hollup2001_1": ("CA1", "dorsal", "arena"),
    "kim2012_4": ("CA1", "intermediate", "large"),
    "royer2010_1": None, "neunuebel2012_1": None, "neunuebel2012_2": None, "grieves2020_2": None,
    "hayman2011_2": None, "hayman2011_3": None, "sharp1994_1": None, "sharp1994_2": None,
    "kim2012_1": None, "kim2012_2": None, "kim2012_3": None, "mankin2015_1": None, "park2011_5": None,
}


def rat_class(r):
    if r["id"] in RAT_CLASS:
        return RAT_CLASS[r["id"]]
    length, area = first_num(r["env_length_m"]), first_num(r["env_area_m2"])
    if r["environment_type"] in ("linear track", "maze") and length is not None:
        return "large" if length > 3 else "track"
    if r["environment_type"] == "open field" and area is not None:
        return "large" if area > 4 else "arena"
    raise ValueError(f"no class for {r['id']}")


# ---------------------------------------------------------------- mouse field widths

MOUSE_CLASS_LABEL = {"real": "real environment up to 2 m", "virtual": "virtual track up to 3 m",
                     "large": "large environment over 3 m or 1.2 m²"}

PEAK_ONLY = "peak-only field rule ({}), which gives 7 to 9 cm in mice"
MOUSE_EXCLUDE = {
    "chockanathan2021_1": PEAK_ONLY.format("bins at least 1.5 SD above the mean"),
    "chockanathan2021_2": PEAK_ONLY.format("bins at least 1.5 SD above the mean"),
    "schuette2022_1": PEAK_ONLY.format("mean + 1.5 SD"),
    "sheintuch2023_1": PEAK_ONLY.format("30% of maximum on 4 cm bins with no smoothing"),
    "sheintuch2023_2": PEAK_ONLY.format("30% of maximum on 4 cm bins with no smoothing"),
    "rechavi2022_1": PEAK_ONLY.format("4 cm bins with no smoothing"),
    "tarcsay2025_1": PEAK_ONLY.format("Gaussian-mixture split"),
    # mallory2018_1 was excluded here while its arena size was unread. The Online Methods give
    # 50 or 62 cm squares, a real environment up to 2 m, so it enters (pft2_corrections.csv), as
    # does mallory2018_2 (1 m linear track, size stated; pft2_additions_fields_mouse.csv).
    "chen2018_2": "virtual 2D square; the mouse classes have no virtual-arena class (same cells as chen2018_1)",
    "kim2023_2": "all fields summed; the per-field value of the same cells (kim2023_1) is used",
    "kim2023_4": "all fields summed; the per-field value of the same cells (kim2023_3) is used",
    "bittner2017_1": "subthreshold Vm ramp, not a spike field; no width",
    "donegan2020_1": "areas in pixels of unstated size, no width",
    "donegan2020_2": "areas in pixels of unstated size, no width",
}
MOUSE_VALUE = {
    "keinath2014_1": (27.1, 27.1, "per field; the row's 28.0 cm sums a cell's fields"),
    "keinath2014_2": (22.3, 22.3, "per field; the row's 33.1 cm sums a cell's fields"),
    "keinath2014_3": (33.1, 33.1, "all fields summed, no per-field value given; fills about 89% of the 35 cm cylinder"),
    "keinath2014_4": (29.3, 29.3, "per field; the row's 41.9 cm sums a cell's fields"),
    "jarzebowski2022_1": (17.0, 17.0, "per field; the row's 22.1 cm sums a cell's fields"),
    "jarzebowski2022_2": (27.2, 27.2, "per field; the row's 34.4 cm sums a cell's fields"),
    "kim2023_1": (20.0, 21.5, "per field, pixel side 1.3 to 1.4 cm (the audit's geometry check)"),
    "kim2023_3": (18.3, 19.7, "per field, pixel side 1.3 to 1.4 cm (the audit's geometry check)"),
    "abadperez2025_1": (30.9, 43.3, "all fields summed; 50 or 70 cm arena, the paper does not say which"),
    "abadperez2025_2": (30.4, 42.5, "all fields summed; 50 or 70 cm arena, the paper does not say which"),
}
MOUSE_REGION = {"CA1": "CA1", "CA3": "CA3", "CA2": "CA2", "DG-granule": "dentate granule cells",
                "DG-mossy": "mossy cells", "DG-unspecified": "dentate gyrus", "SUB": "subiculum",
                "CA1/CA3 pooled (not separated by the paper)": "CA1 and CA3 pooled"}
MOUSE_REGION_ID = {"hainmueller2018_2": "CA2/3"}
MOUSE_ATTACH = {
    "chockanathan2021_1": ("CA1", "dorsal", "virtual"), "chockanathan2021_2": None,
    "schuette2022_1": ("CA1", "dorsal", "real"), "sheintuch2023_2": ("CA1", "dorsal", "real"),
    "sheintuch2023_1": ("CA3", "dorsal", "real"), "rechavi2022_1": ("CA1", "dorsal", "real"),
    "tarcsay2025_1": ("CA1", "dorsal", "real"),
    "chen2018_2": ("CA1", "dorsal", "virtual"), "kim2023_2": ("dentate granule cells", "dorsal", "real"),
    "kim2023_4": ("mossy cells", "dorsal", "real"), "bittner2017_1": None,
    "donegan2020_1": None, "donegan2020_2": None,
}


def mouse_class(r):
    length, area = first_num(r["env_length_m"]), first_num(r["env_area_m2"])
    if r["environment_type"] == "VR track":
        return "large" if length > 3 else "virtual"
    if (length or 0) > 3 or (area or 0) > 1.2:
        return "large"
    if length is None and area is None:
        raise ValueError(f"no size for {r['id']}")
    if (length or 0) <= 2:
        return "real"
    raise ValueError(f"{r['id']} is a real environment between 2 and 3 m, which has no class")


# ---------------------------------------------------------------- aggregation


def aggregate(rows, species):
    if species == "rat":
        exclude, value, attach, labels = RAT_EXCLUDE, RAT_VALUE, RAT_ATTACH, RAT_CLASS_LABEL
    else:
        exclude, value, attach, labels = MOUSE_EXCLUDE, MOUSE_VALUE, MOUSE_ATTACH, MOUSE_CLASS_LABEL
    groups, excluded = OrderedDict(), []
    for r in rows:
        rid = r["id"]
        width = num_range(r["width_cm"])
        if rid in exclude:
            excluded.append((rid, exclude[rid], attach.get(rid)))
            continue
        if rid not in value and width is None:
            excluded.append((rid, "no width in cm (counts, fractions or ratios only)", None))
            continue
        lo, hi, how = value[rid] if rid in value else (*width, "")
        if species == "rat":
            region, cls = RAT_REGION[r["region"]], rat_class(r)
        else:
            region = MOUSE_REGION_ID.get(rid, MOUSE_REGION[r["region"]])
            cls = mouse_class(r)
        key = (r["level"], region, cls)
        groups.setdefault(key, []).append({
            "id": rid, "study": study_of(rid), "lo": lo, "hi": hi, "how": how,
            "env": r["environment_size"], "env_type": r["environment_type"],
            "length": first_num(r["env_length_m"]), "area": first_num(r["env_area_m2"]),
            "tag": tag(r["status"]), "doi": doi_url(r["doi_or_url"])})
    out = []
    region_order = ["CA1", "CA1 and CA3 pooled", "CA2", "CA3", "CA2/3", "dentate gyrus",
                    "dentate granule cells", "mossy cells", "subiculum"]
    class_order = list(labels)
    for key in sorted(groups, key=lambda k: (LEVELS.index(k[0]), region_order.index(k[1]),
                                             class_order.index(k[2]))):
        level, region, cls = key
        contrib = groups[key]
        studies = list(OrderedDict.fromkeys(c["study"] for c in contrib))
        n = len(studies)
        env = labels[cls]
        if cls == "large" and species == "rat" and all(c["study"] == "kjelstrup2008" for c in contrib):
            env = "large real 18 m track"
        elif cls == "large" and species == "rat" and all(c["env_type"] == "linear track" for c in contrib):
            env = "large real track over 3 m"
        elif cls == "large" and species == "mouse" and all(is_virtual(c["env_type"]) for c in contrib):
            env = "large virtual track over 3 m"
        label = f"{level.capitalize()} {region}, {env}, {n} {'study' if n == 1 else 'studies'}"
        out.append({"key": key, "label": label, "lo_cm": min(c["lo"] for c in contrib),
                    "hi_cm": max(c["hi"] for c in contrib),
                    "studies": [short(s) for s in studies], "contrib": contrib})
    return out, excluded


# ---------------------------------------------------------------- retrosplenial and non-grid MEC widths

# Both sweeps add rows under their own group headers, one row per species x region x environment
# class, with the deduplication rules above; mouse rows go to the mouse grid and rat rows to the
# rat grid. The retrosplenial sweep also holds CA1 comparison rows: two repeat rows of
# sweep_fields_mouse.csv and are dropped; a new mouse CA1 width (navratilova2026_4) joins the
# hippocampal mouse rows. No rat retrosplenial row survives: the only rat RSC width (Smith 2012)
# has no environment size. The one rat non-grid MEC row is Butler 2019 (MNG33 and MNG35).
GROUP_RSC, GROUP_MEC = "Retrosplenial cortex", "MEC, non-grid cells"
EXTRA_CLASS_LABEL = dict(MOUSE_CLASS_LABEL, **{"70 to 100 cm arena": "real open arena of 70 to 100 cm",
                                               "1.5 m box": "real 1.5 m arena"})
NO_WIDTH = "no width in cm (counts, fractions, ratios, angles or stability only)"

RSC = {
    "group": GROUP_RSC,
    "exclude": {
        "mao2017_ca1_1": "same CA1 cells as mao2017_1 in sweep_fields_mouse.csv, which the CA1 row already uses",
        "lenkey2025_3": "same CA1 cells as lenkey2025_1 in sweep_fields_mouse.csv, which the CA1 row already uses",
        "mao2017_rsc_3": "superficial dysgranular cells from 3 of the same 4 WT mice as mao2017_rsc_1 and _2, "
                         "same statistic as _2; the pooled rows are used",
        "mao2018_2": "after bilateral hippocampal lesions, a manipulation; the sham row mao2018_1 is used",
        "navratilova2026_3": "the cells of navratilova2026_1 split by mouse line (Thy1 29 cm, CaMK2a-tTA 38 cm); "
                             "the pooled rows are used and the text gives the split",
        "park2024_1": "vertex-cell fields at polygon corners, not place-like fields; area approximate "
                      "(about 80 cm²) in a 25 cm radius chamber",
        "smith2012_1": "rat; plus-maze size not stated, so no environment class, and an equal-area diameter "
                       "on narrow arms understates the length along an arm (as diamantaki2016 cell 104)",
        "smith2012_2": "rat CA1 comparison row for smith2012_1; plus-maze size not stated, so no environment class",
        "subramanian2024_3": "sparsity index (the share of the box where a cell fires), not a field width",
        "alexander2020_4": "egocentric boundary cells' preferred distance to a wall, not a field width; the "
                           "histogram's distance axis is uncertain",
        "vanwijngaarden2020_2": "border cells' preferred distance from the wall (peak at 5 cm), not a field width",
        "lozano2017_1": "head-direction tuning width in degrees, not a spatial width",
        "zhang2022_4": "head-direction tuning width in degrees, definition not stated",
    },
    # Where an excluded width would have gone: (region, level, class); None means no row exists.
    "attach": {
        "mao2017_ca1_1": ("CA1", "dorsal", "real"), "lenkey2025_3": ("CA1", "dorsal", "real"),
        "mao2017_rsc_3": ("Dysgranular RSC", "", "real"), "mao2018_2": ("Dysgranular RSC", "", "real"),
        "navratilova2026_3": ("Dysgranular RSC", "", "large"), "park2024_1": None,
        "smith2012_1": None, "smith2012_2": None, "subramanian2024_3": None, "alexander2020_4": None,
        "vanwijngaarden2020_2": None, "lozano2017_1": None, "zhang2022_4": None,
    },
    "region": {"RSC-dysgranular": "Dysgranular RSC", "RSC-granular": "Granular RSC",
               "RSC-unspecified": "RSC, subregion not stated"},
    "region_id": {"mao2017_rsc_4": "RSC deep layers"},
    "region_order": ["Dysgranular RSC", "Granular RSC", "RSC deep layers", "RSC, subregion not stated"],
    "class": {},
}
MEC = {
    "group": GROUP_MEC,
    "exclude": {
        "MNG03": "HCN1 knockout mice, a genetic model; the wild-type littermates (MNG01) are used",
        "MNG05": "grid cells, and the size is the central peak of the spatial autocorrelogram, not a rate-map field",
        "MNG02": "share of the box covered by the cell's fields (43%), not a field width",
        "MNG24": "share of a virtual track inside cue-cell fields under a lenient p < 0.3 rule, not a width",
        "MNG29": "distance from the wall to the field centre (8.2 cm), not a width; unclear whether it covers "
                 "6 or 20 cells",
        "MNG34": "the same cells in the reward-task box (ENV2), a manipulation; the foraging-box rows MNG33 "
                 "and MNG35 are used",
        "MNG36": "grid cells' field size, the comparator for MNG33 to MNG35, not a non-grid width",
        "MNG40": "share of the box covered by a field (35.17%), not a field width",
    },
    "attach": {"MNG03": ("Dorsal MEC non-grid spatial cells", "", "70 to 100 cm arena"), "MNG05": None,
               "MNG02": None, "MNG24": None, "MNG29": None,
               "MNG34": ("MEC and nearby cortex, non-grid spatial cells, level unstated", "", "1.5 m box"),
               "MNG36": None, "MNG40": None},
    "region": {},
    # Border width is the field's extent perpendicular to the wall, so it gets its own labelled row.
    # Butler 2019's cells lie in MEC (598 of 778) and nearby parasubiculum, postsubiculum and
    # cortex, at depths not given; MNG35 is the high-coherence subset of MNG33's cells, so the two
    # form that study's range, as Mao 2017's mean and median do.
    "region_id": {"MNG01": "Dorsal MEC non-grid spatial cells",
                  "MNG04": "Dorsal MEC border cells, width perpendicular to the wall",
                  "MNG33": "MEC and nearby cortex, non-grid spatial cells, level unstated",
                  "MNG35": "MEC and nearby cortex, non-grid spatial cells, level unstated"},
    "region_order": ["Dorsal MEC non-grid spatial cells", "Dorsal MEC border cells, width perpendicular to the wall",
                     "MEC and nearby cortex, non-grid spatial cells, level unstated"],
    # Mallory 2018's Online Methods give the MEC arenas, 70, 90 or 100 cm squares chosen per mouse
    # (pft2_corrections.csv). Butler 2019's boxes are 1.5 m squares. Butler's grid spacing, a median
    # of 99.14 cm, pools 102 grid cells of the same data set and is not paired with the non-grid
    # cells by rat; the rat grid's columns stay the Stensola modules.
    "class": {"MNG01": "70 to 100 cm arena", "MNG04": "70 to 100 cm arena",
              "MNG33": "1.5 m box", "MNG35": "1.5 m box"},
}


def passed_ca1(rows, cfg):
    """Mouse CA1 widths in a group sweep that are not duplicates: they join the hippocampal rows."""
    return [r for r in rows if r["species"] == "mouse" and r["region"] == "CA1"
            and r["id"] not in cfg["exclude"] and num_range(r["width_cm"]) is not None]


def aggregate_extra(rows, cfg):
    groups, excluded, passed = OrderedDict(), [], []
    for r in rows:
        rid = r["id"]
        if rid in cfg["exclude"]:
            excluded.append((rid, cfg["exclude"][rid], cfg["attach"].get(rid)))
            continue
        width = num_range(r["width_cm"])
        if width is None:
            excluded.append((rid, NO_WIDTH, None))
            continue
        species = r["species"]
        if species not in ("mouse", "rat"):
            raise ValueError(f"{rid}: only mouse and rat rows have a grid")
        if r["region"] == "CA1":
            if species != "mouse":
                raise ValueError(f"{rid}: a rat CA1 width in a group sweep needs a rule")
            passed.append(rid)
            continue
        region = cfg["region_id"].get(rid) or cfg["region"][r["region"]]
        if species == "rat" and rid not in cfg["class"]:
            raise ValueError(f"{rid}: rat rows need an environment class override")
        cls = cfg["class"].get(rid) or mouse_class(r)
        groups.setdefault((species, region, cls), []).append({
            "id": rid, "study": study_key(r), "lo": width[0], "hi": width[1], "how": "",
            "env": r["environment_size"], "env_type": r["environment_type"],
            "length": first_num(r["env_length_m"]), "area": first_num(r["env_area_m2"]),
            "tag": tag(r["status"]), "doi": doi_url(r["doi_or_url"])})
    out, class_order = [], list(EXTRA_CLASS_LABEL)
    for key in sorted(groups, key=lambda k: (k[0] != "mouse", cfg["region_order"].index(k[1]),
                                             class_order.index(k[2]))):
        species, region, cls = key
        contrib = groups[key]
        studies = list(OrderedDict.fromkeys(c["study"] for c in contrib))
        n = len(studies)
        env = EXTRA_CLASS_LABEL[cls]
        if cls == "large" and all(is_virtual(c["env_type"]) for c in contrib):
            env = "large virtual track over 3 m"
        out.append({"key": key, "label": f"{region}, {env}, {n} {'study' if n == 1 else 'studies'}",
                    "lo_cm": min(c["lo"] for c in contrib), "hi_cm": max(c["hi"] for c in contrib),
                    "studies": [short(s) for s in studies], "contrib": contrib, "group": cfg["group"],
                    "species": species})
    return out, excluded, passed


# ---------------------------------------------------------------- kernels

# One row per study x region x level x condition. Full width at r = 0.5 is twice the one-sided
# distance. Where the sweep and the audit read a figure differently, the row spans both readings
# (one-sided readings in the comment, sweep first). Kjelstrup and Resnik widths are printed values.
# Left out: MEC grid-cell kernels (Brun, K04 to K06; grid cells, not hippocampal or non-grid; the
# same curves are MK01 to MK03 in sweep_kernel_mec.csv), rows without a width (K12, K13, K18,
# K23 to K26, N14, N15, N18, N19), species without measured grid spacing (marmoset N05 to N07,
# bird N12, human K24, K25, N18), genetic models (N02 GluA1 knockout, N04 Dp(16)1Yey), a drug
# session (N17 scopolamine), passive transport (K15 car driving) and the cross-half design whose
# diagonal is 0.58 (N16 Blair), where r = 0.5 is not half the peak.
# Mao 2017 (K16 RSC, K17 CA1) holds the audited retrosplenial values; K-mao2017_1 and _2 in
# sweep_kernel_rsc.csv are the same curves and do not enter. Also left out from that file:
# K-mao2017_3 to _5 (layer and subregion subsets, 123, 60 and 30 cells from 3 of the same 4 mice;
# r = 0.5 at 13.3 to 14.1 cm, within 1 cm of K16), K-mao2017_6 and _7 (Thy1 example matrices read
# by colour-bar lookup, cell counts unclear; _7 removes the belt cues), fischer2020_k1 and _k2
# (split-half correlation peaking at 0.76 to 0.82, so r = 0.5 is not half the peak, as N16; the
# width also changes along the track and is narrowest at the landmark) and miller2019_k1 (no
# width). The Alexander & Nitz retrosplenial curves are left out with the reasons in
# KERNEL_EXCLUDE; their two CA1 curves (alexander2015_k3, alexander2017_k2) enter the rat grid.
KERNEL_EXCLUDE = {
    "alexander2015_k1": "rat RSC, odd against even trials on the 2.15 m W-shaped route; never reaches 0.5 "
                        "(minimum 0.53 to 0.56), so no r = 0.5 width",
    "alexander2015_k2": "rat RSC between two sessions at one room location on the 2.15 m route; it only touches "
                        "0.50 to 0.503 at 65 to 75 cm and 136 to 176 cm, within reading error, with no clean "
                        "crossing, and it compares two sessions, not one environment within a session",
    "alexander2015_k4": "alpha against beta, the same route after the track moved to another room location, so "
                        "not one environment; no r = 0.5 width (RSC 0.54 to 0.60 out to 150 cm, CA1 0.10 to 0.21)",
    "alexander2017_k1": "rat RSC, odd against even trials on the 6.6 m plus-shaped lap; never below 0.71, so no "
                        "r = 0.5 width",
    "alexander2017_k3": "rat RSC, odd against even trials on the 3.83 m ring; the colour bar spans 0.7 to 1 and r "
                        "stays at or above 0.83, so no r = 0.5 width",
}
KERNELS = {
    "Rat kernel": [
        # id, level, region, study, condition, lo, hi, status
        ("K07", "dorsal", "CA1", "green2022", "control trials, 1.8 m track", 33.4, 35.2, "full text, figure-read"),  # 16.7; 17.6
        ("K08", "dorsal", "CA1", "green2022", "jump trials, 1.8 m track", 38.4, 39.0, "full text, figure-read"),  # 19.5; 19.2
        ("K10", "dorsal", "CA1", "wang2020", "2.07 m circular track", 28.0, 28.4, "full text, figure-read"),  # 14; 14.2
        ("K14", "dorsal", "CA1", "terrazas2005", "walking, 4.08 m ring, dot product", 54, 54, "full text, figure-read"),  # 27
        ("K19", "dorsal", "CA1", "ledergerber2021", "open field, 1.5 m box", 44, 45, "full text, figure-read"),  # 22; 22.5
        ("K21", "dorsal", "CA1", "ledergerber2021", "memory task, 1.5 m box", 44, 44.4, "full text, figure-read"),  # 22; 22.2
        ("N08", "dorsal", "CA1", "gothard2001", "light, 1.76 m track", 27.4, 30, "figure-read"),  # 14; 13.7 mean, 15.0 median
        ("N09", "dorsal", "CA1", "gothard2001", "darkness, 1.76 m track", 23.8, 25.8, "figure-read"),  # 12; 11.9, 12.9
        ("N13", "dorsal", "CA1", "farooq2024", "juvenile, first exposure, 1 m track", 29, 31, "full text, figure-read"),  # 15; 14.5, 15.5
        # From sweep_kernel_rsc.csv and audit_pft2.md. Both pool cells across rats; 2015 compares two
        # sessions, 2017 uses Spearman with 7 cm smoothing and a colour bar clipped at 0.5.
        ("alexander2015_k3", "dorsal", "CA1", "alexander2015", "between sessions, 2.15 m route", 52.8, 53, "full text, figure-read"),  # 26.5; 26.4
        ("alexander2017_k2", "dorsal", "CA1", "alexander2017", "Spearman, 6.6 m lap", 120, 126, "full text, figure-read"),  # 60 to 63; 60.6 by tick labels
        ("K01", "dorsal", "CA3", "kjelstrup2008", "18 m track", 81, 81, "full text"),
        ("K09", "dorsal", "CA3", "green2022", "jump trials, 1.8 m track", 35.6, 36.8, "full text, figure-read"),  # 18.4; 17.8
        ("K11", "dorsal", "CA3", "wang2020", "2.07 m circular track", 39.6, 40, "full text, figure-read"),  # 20; 19.8
        ("N10", "dorsal", "dentate gyrus", "gothard2001", "light, 1.76 m track", 28, 30, "figure-read"),  # 14; 14.0, 15.0
        ("N11", "dorsal", "dentate gyrus", "gothard2001", "darkness, 1.76 m track", 16.8, 18, "figure-read"),  # 9; 8.6, 8.4
        ("K20", "dorsal", "subiculum", "ledergerber2021", "open field, 1.5 m box", 37.2, 38, "full text, figure-read"),  # 19; 18.6
        ("K22", "dorsal", "subiculum", "ledergerber2021", "memory task, 1.5 m box", 38.4, 39, "full text, figure-read"),  # 19.5; 19.2
        ("K02", "intermediate", "CA3", "kjelstrup2008", "18 m track", 371, 371, "full text"),
        ("K03", "ventral", "CA3", "kjelstrup2008", "18 m track", 666, 666, "full text"),
    ],
    "Mouse kernel": [
        ("N01", "dorsal", "CA1", "resnik2012", "1.4 m track", 18, 18, "full text, figure-read"),  # text 9.0
        ("N03", "dorsal", "CA1", "raveau2018", "1.7 m track, Spearman", 26.4, 27, "full text, figure-read"),  # 13.5; 13.3 and 13.1
        ("K17", "dorsal", "CA1", "mao2017", "1.5 m treadmill loop", 26.8, 26.8, "full text, figure-read"),  # 13.4; 13.39
    ],
}
# Rows under a group header, after the hippocampal rows: group, id, region, study, condition, lo, hi, status.
KERNEL_GROUPS = {
    "Rat kernel": [],
    "Mouse kernel": [
        (GROUP_RSC, "K16", "Dysgranular RSC", "mao2017", "1.5 m treadmill loop", 26.4, 26.5, "full text, figure-read"),  # 13.2; 13.25
    ],
}
KERNEL_SHORT = {"green2022": "Green 2022", "wang2020": "Wang 2020", "terrazas2005": "Terrazas 2005",
                "ledergerber2021": "Ledergerber 2021", "gothard2001": "Gothard 2001",
                "farooq2024": "Farooq 2024", "kjelstrup2008": "Kjelstrup 2008",
                "resnik2012": "Resnik 2012", "raveau2018": "Raveau 2018", "mao2017": "Mao 2017",
                "alexander2015": "Alexander & Nitz 2015", "alexander2017": "Alexander & Nitz 2017"}


def kernel_rows(name):
    known = {r["id"] for r in load("sweep_kernel.csv") + load("sweep_kernel_rsc.csv")}
    assert all(rid in known for rid in KERNEL_EXCLUDE)
    assert not any(row[0] in KERNEL_EXCLUDE for row in KERNELS[name])
    rows = []
    for rid, level, region, study, cond, lo, hi, status in KERNELS[name]:
        assert rid in known, rid
        rows.append({"label": f"{level.capitalize()} {region}, {KERNEL_SHORT[study]}, {cond}",
                     "lo_cm": lo, "hi_cm": hi, "studies": [KERNEL_SHORT[study]],
                     "source": rid, "status": status})
    for group, rid, region, study, cond, lo, hi, status in KERNEL_GROUPS[name]:
        assert rid in known, rid
        rows.append({"label": f"{region}, {KERNEL_SHORT[study]}, {cond}", "lo_cm": lo, "hi_cm": hi,
                     "studies": [KERNEL_SHORT[study]], "group": group, "source": rid, "status": status})
    return rows


# ---------------------------------------------------------------- report


def fmt_cm(lo, hi):
    f = lambda v: f"{v:g}"
    return f"{f(lo)} cm" if lo == hi else f"{f(lo)} to {f(hi)} cm"


def report(species, rows, excluded, data, passed=(), show_species=False):
    by_id = {r["id"]: r for r in data}
    lines = [f"### {species}: rows and contributions", "",
             "| Row | N | Range | Contributions (row id: value, environment, how) |", "|---|---|---|---|"]
    if not rows:
        lines.append("| No row | 0 | | |")
    for row in rows:
        parts = []
        for c in row["contrib"]:
            how = f", {c['how']}" if c["how"] else ""
            env = c["env"].split(";")[0].split(" (")[0][:70]
            parts.append(f"{c['id']}: {fmt_cm(c['lo'], c['hi'])}, {env}{how}")
        grid = " (rat grid)" if row.get("species") == "rat" else ""
        lines.append(f"| {row['label']}{grid} | {len(row['studies'])} | {fmt_cm(row['lo_cm'], row['hi_cm'])} | "
                     + "; ".join(parts).replace("|", "/") + " |")
    lines += ["", f"### {species}: excluded rows", "", "| Row id | Region, level | Reason | Would belong to |",
              "|---|---|---|---|"]
    for rid, why, att in excluded:
        r = by_id[rid]
        where = "no row" if att is None else " ".join(x for x in (att[1], att[0]) if x) + f", {att[2]}"
        where_from = f"{r['region'][:60]}, {r['level']}" + (f", {r['species']}" if show_species else "")
        lines.append(f"| {rid} | {where_from} | {why} | {where} |")
    if passed:
        lines += ["", f"### {species}: CA1 rows passed to the hippocampal mouse rows", "",
                  "| Row id | Width | Joins |", "|---|---|---|"]
        for rid in passed:
            lines.append(f"| {rid} | {by_id[rid]['width_cm']} cm | Dorsal CA1, by the mouse class rules |")
    return "\n".join(lines) + "\n"


def main():
    argv = sys.argv[1:]
    dest = None
    if "--report" in argv:
        i = argv.index("--report")
        dest = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    spec_path = Path(argv[0]) if argv else HERE / "ratio_spec.json"
    spec = json.load(open(spec_path, encoding="utf-8"))
    out_report = []
    keep = ("label", "lo_cm", "hi_cm", "studies", "group")
    extra = [(RSC, load("sweep_fields_rsc.csv")), (MEC, load("sweep_fields_mec_nongrid.csv"))]
    extra_out = [(cfg, rows_x, *aggregate_extra(rows_x, cfg)) for cfg, rows_x in extra]
    for name, csv_name, species in (("Rat", "sweep_fields_rat.csv", "rat"),
                                    ("Mouse", "sweep_fields_mouse.csv", "mouse")):
        data = load(csv_name)
        if species == "mouse":
            data = data + [r for cfg, rows_x in extra for r in passed_ca1(rows_x, cfg)]
        rows, excluded = aggregate(data, species)
        out_report.append(report(name, rows, excluded, data))
        for cfg, rows_x, grows, gexcluded, gpassed in extra_out:
            if species == "mouse":
                out_report.append(report(cfg["group"], grows, gexcluded, rows_x, gpassed, show_species=True))
            rows = rows + [g for g in grows if g["species"] == species]
        spec[name]["rows"] = [{k: r[k] for k in keep if k in r} for r in rows]
    kx = ["### Kernels: curves of sweep_kernel_rsc.csv left out by name", "", "| Row id | Reason |", "|---|---|"]
    out_report.append("\n".join(kx + [f"| {rid} | {why} |" for rid, why in KERNEL_EXCLUDE.items()]) + "\n")
    for name, base in (("Rat kernel", "Rat"), ("Mouse kernel", "Mouse")):
        spec[name] = {"header": "Recording (kernel full width at 0.5)", "columns_from": base,
                      "rows": [{k: v for k, v in r.items() if k in keep} for r in kernel_rows(name)]}
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for name in ("Rat", "Mouse", "Rat kernel", "Mouse kernel"):
        print(f"{name}: {len(spec[name]['rows'])} rows")
        for r in spec[name]["rows"]:
            print(f"  {r['label']} ({fmt_cm(r['lo_cm'], r['hi_cm'])})")
    if dest:
        open(dest, "w", encoding="utf-8").write("\n".join(out_report))


if __name__ == "__main__":
    main()
