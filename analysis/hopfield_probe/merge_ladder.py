"""Assemble the coverage-ladder result set, then build the page from it.

Two things live in one report and they arrive by different routes:

* the five coverage rungs, four training seeds each, from ``splice_basin`` --
  the suite was measured once and only the basin is recomputed;
* the two saturated arms of the 10% encoder, four seeds each, from a full
  ``run.py`` sweep, because saturation changes the recall map and therefore
  every test, not just the basin.

``report.build`` renders one directory, in manifest order, so this puts them in
one directory in the order the tabs should read: the ladder from most coverage
to least, then the saturated arms. Nothing is recomputed here -- files are
copied and an ordering is written.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

T = "/home/jackking/.claude/jobs/d05f5770/tmp"
LADDER = f"{T}/probe_spliced"
SAT = f"{T}/probe_sat10"

# Group order on the page. The ladder groups are the first segment of the
# labels `splice_basin` writes; the saturated ones the first segment of the
# labels `run_sat10.sh` writes.
ORDER = ["10%", "5%", "2.5%", "1.25%", "0.75%",
         "10% β=1e6", "10% gain=1e6, β=1e6"]


def _slug(label: str) -> str:
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in label]
    return "".join(keep).strip("_")


def main() -> None:
    # 1. The saturated arms, one result JSON per task dir, copied in.
    copied = 0
    for path in sorted(glob.glob(f"{SAT}/t*/*.json")):
        if path.endswith("manifest.json"):
            continue
        res = json.load(open(path))
        label = res["header"]["label"]
        shutil.copy2(path, os.path.join(LADDER, _slug(label) + ".json"))
        copied += 1

    # 2. One manifest, ordered. Anything whose group is not in ORDER still
    # gets a page -- appended at the end rather than silently dropped, so a
    # new arm shows up even before this list learns about it.
    rows = []
    for path in sorted(glob.glob(f"{LADDER}/*.json")):
        if path.endswith("manifest.json"):
            continue
        res = json.load(open(path))
        label = res["header"].get("label", pathlib.Path(path).stem)
        group = label.split(" · ")[0].strip()
        rank = ORDER.index(group) if group in ORDER else len(ORDER)
        rows.append((rank, group, label, os.path.basename(path),
                     res["header"]))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    manifest = {
        "encoders": [{"label": lab, "file": f, "header": h}
                     for _rank, _g, lab, f, h in rows],
        "created": "merge_ladder",
    }
    with open(os.path.join(LADDER, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)

    seen: dict[str, int] = {}
    for _rank, g, *_ in rows:
        seen[g] = seen.get(g, 0) + 1
    print(f"copied {copied} saturated results into {LADDER}")
    for g in [g for g in ORDER if g in seen] + [g for g in seen
                                                if g not in ORDER]:
        print(f"  {g:24s} {seen[g]} seeds")
    print(f"{len(rows)} results total")


if __name__ == "__main__":
    main()
