"""Pull selected rows of ``tables.md`` into markdown for the write-up.

    python -m analysis.ideal_family.doc_tables [--root RESULTS] > /tmp/doc_tables.md

Each entry names a section of ``tables.md`` and a row filter on its first
columns, so the write-up's tables are copies of the computed ones, not
retyped numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .run import OUT

PICKS = [
    ("Raw kernel and alias against r and M", "Raw kernel and alias (M = 512 at every r; every M at r = 16)",
     lambda c: c[1] == "512" or c[0] == "16"),
    ("Single units, GRF limit, every r (Rice predictions in brackets)",
     "Single units on 1716-cell loops, GRF references (prediction in brackets)",
     lambda c: c[1] in ("3", "8", "32")),
    ("Rate kernels in units of r (r >= 8)", "Rate kernels, r of 8 and above (theory in brackets)", lambda c: True),
    ("r that holds FW05 at the raw r = 16 kernel's 37.7 cells", "r that holds FW05 at 37.7 cells",
     lambda c: True),
    ("Field counts, GRF limit", "Field counts, GRF references", lambda c: c[1] in ("4", "16")),
    ("Finite M at r = 16", "Finite M at r = 16", lambda c: c[0] in ("rotated", "sparse_raw_p0.1")),
    ("Per-unit spectral spread", "Per-unit spectral spread", lambda c: True),
    ("Alias in the rate kernels: measured at the worst displacement against g(K)",
     "Rate-kernel alias at the worst displacement", lambda c: True),
    ("Native plane-wave code", "Native code on loops", lambda c: c[1] == "512" or c[0] == "16"),
    ("2D arenas, GRF limit (r >= 8)", "2D arenas, GRF references, r of 8 and above", lambda c: True),
    ("Treves-Rolls on short tracks", "Treves-Rolls on short tracks, r = 16 and 23.2", lambda c: True),
]


def sections(text: str) -> dict[str, list[str]]:
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=OUT)
    a = ap.parse_args()
    secs = sections((a.root / "tables.md").read_text())
    for name, title, keep in PICKS:
        lines = [l for l in secs.get(name, []) if l.startswith("|")]
        if len(lines) < 2:
            continue
        rows = [l for l in lines[2:] if keep([c.strip() for c in l.strip("|").split("|")])]
        print(f"### {title}\n")
        print("\n".join(lines[:2] + rows).replace("nan ± nan", "-"))
        print()


if __name__ == "__main__":
    main()
