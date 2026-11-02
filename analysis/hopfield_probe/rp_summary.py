"""Tables for Sec 19 (rp_readout_check / rp_sfa_check JSON).

    python -m analysis.hopfield_probe.rp_summary step1 DIR
    python -m analysis.hopfield_probe.rp_summary step2 DIR
"""
from __future__ import annotations

import glob
import json
import os
import sys

ORDER = ["sig1", "sig4", "sig16", "hard50", "hard10"]


def _load(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def _f(v, fmt="{:.3f}"):
    return "-" if v is None else fmt.format(v)


def _key(r):
    v = r["variant"]
    return (-r["fwhm"], ORDER.index(v) if v in ORDER else 99, v, r["H"], r["coverage"])


def step1(d):
    rows = sorted(_load(d), key=_key)
    ref = rows[0]["ideal"]["unvisited"] if rows else None
    if ref:
        print(f"ideal r=16 (unvisited): C1 {ref['C1']:.3f}  r_half {ref['r_half']:.1f}  "
              f"far_sd {ref['far_sd']:.3f}  alias_max {ref['alias_max']:.3f}\n")
    print("| bump | layer | H | cov | R² visited | R² unvisited | R² far | cos to ideal (far) "
          "| C1 (far) | r_half (far) | far sd (far) | alias max (far) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["sets"]
        f = s.get("far", s["unvisited"])
        print(f"| {r['fwhm']:g} | {r['variant']} | {r['H']} | {r['coverage']:g} "
              f"| {s['visited']['r2']:.3f} | {s['unvisited']['r2']:.3f} "
              f"| {_f(s.get('far', {}).get('r2'))} | {f['cos_to_ideal']:.3f} | {f['C1']:.3f} "
              f"| {_f(f['r_half'], '{:.1f}')} | {f['far_sd']:.3f} | {f['alias_max']:.3f} |")


def step2(d):
    rows = sorted(_load(d), key=_key)
    print("| bump | layer | H | cov | readout | slow features r=32 / waves | r=16 / waves "
          "| C1 vis / far | r_half vis / far | far sd vis / far | alias max vis / far |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["slow_counts"]
        for name, ro in r["readouts"].items():
            v, f = ro["visited"], ro.get("far", ro["unvisited"])
            print(f"| {r['fwhm']:g} | {r['variant']} | {r['H']} | {r['coverage']:g} | {name} "
                  f"| {c['32']['sfa']} / {c['32']['waves']:.0f} | {c['16']['sfa']} / {c['16']['waves']:.0f} "
                  f"| {v['C1']:.3f} / {f['C1']:.3f} | {_f(v['r_half'], '{:.1f}')} / {_f(f['r_half'], '{:.1f}')} "
                  f"| {v['far_sd']:.3f} / {f['far_sd']:.3f} | {v['alias_max']:.3f} / {f['alias_max']:.3f} |")


if __name__ == "__main__":
    {"step1": step1, "step2": step2}[sys.argv[1]](sys.argv[2])
