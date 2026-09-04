"""Recompute only the basin, and splice it into existing per-seed results.

Every archived run predates the basin fixes: they measured `r_exact_*` over the
evaluation env's cells, and the later ones through a retrieval bank that held a
duplicate of the goal. Reach, direction and flow in those runs are unaffected
and were already measured at four training seeds per rung.

So re-running the whole suite to correct one column is the expensive way round.
This reads the five existing per-seed directories, recomputes `basin_probe` for
each checkpoint, writes the corrected `r_exact_all` / `r_exact_95` into the
scalars (moving the env-bounded values aside as `*_envcues`, matching what
`run_test_a` now emits), and writes the merged set to one output directory the
report can be built from.

The five sources are internally consistent -- same probe seed, same settings,
same gains as the ladder page uses -- which is what makes the merge legitimate
rather than hand-assembly.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.hopfield_probe.attractor import basin_probe
from analysis.hopfield_probe.encode import Field
from analysis.hopfield_probe.harness import (ProbeConfig, build_memory,
                                             load_probe_encoder, sample_worlds,
                                             scored_envs)

ARCH = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
S = "/orcd/pool/003/jackking/cls_runs/sweeps"
OUT = "/home/jackking/.claude/jobs/d05f5770/tmp/probe_spliced"
# With a beta override the recall regime changes, so the results go elsewhere:
# one ladder must not mix beta = gain rows with saturated ones.
OUT_BETA = "/home/jackking/.claude/jobs/d05f5770/tmp/probe_spliced_b{tag}"

# coverage label, source archive dir, arm label prefix, checkpoint glob, gain
RUNGS = [
    ("10%",   "attlow_g100",       "att0.5",
     f"{S}/w52_attract_fwhm/*_att0.5_seed=",    100.0),
    ("5%",    "w57_half_a0.5_ps0", "half_a0.5",
     f"{S}/w57_cov5/*_half_a0.5_seed=",          75.0),
    ("2.5%",  "w58_q_a1_g100_ps0", "q_a1",
     f"{S}/w58_cov2.5/*_q_a1_seed=",            100.0),
    ("1.25%", "w60_ps0",           "sm35x_a2",
     f"{S}/w60_cov1.25/*_sm35x_a2_seed=",       100.0),
    ("0.75%", "w61_ps0",           "y50_a2",
     f"{S}/w61_cov0.75/*_y50_a2_seed=",         200.0),
]
SEEDS = (42, 43, 44, 45)


def main(only: int | None = None, beta: float | None = None) -> None:
    """Recompute every checkpoint, or just index ``only`` of the 20.

    The work is per-checkpoint and independent, so splitting it across jobs is
    the difference between one long serial run and twenty short parallel ones.
    Each writes its own file into ``OUT``; nothing is shared.
    """
    out_dir = (OUT if beta is None
               else OUT_BETA.format(tag=f"{beta:g}".replace("+", "")))
    os.makedirs(out_dir, exist_ok=True)
    cfg = ProbeConfig(n_worlds=8, n_envs_per_world=20, env_size=20, Npos=1716,
                      k_values=(1, 3, 5, 10, 20),
                      steps=(1, 2, 3, 5, 10, 15), seed=0,
                      beta_override=beta)
    worlds = sample_worlds(cfg)
    written = 0

    idx = -1
    for cov, src, arm, pat, gain in RUNGS:
        for seed in SEEDS:
            idx += 1
            if only is not None and idx != only:
                continue
            hits = [f for f in glob.glob(f"{ARCH}/{src}/*.json")
                    if "manifest" not in f
                    and json.load(open(f)).get("header", {}).get("label", "")
                    == f"{arm}-s{seed}"]
            if not hits:
                print(f"  no source for {cov} {arm}-s{seed}", file=sys.stderr)
                continue
            res = json.load(open(hits[0]))

            ck = sorted(glob.glob(f"{pat}{seed}"))[0] + "/encoder_final.pt"
            enc, ecfg, own, fwhm, header = load_probe_encoder(
                ck, fwhm_fallback=0.25)
            enc.gain = gain
            field = Field(enc, list(ecfg.lambdas), fwhm, gain, 1716)

            # The header is rebuilt so the spliced file carries `coverage`,
            # which the source predates and the coverage charts need.
            header["label"] = f"{cov} · {arm} · s{seed}"
            header["gain"] = gain
            header["gain_was_overridden"] = True
            header["beta"] = gain if beta is None else float(beta)
            res["header"] = header

            for k in cfg.k_values:
                node = res.get("test_a", {}).get("k", {}).get(str(k))
                if not node:
                    continue
                per = node["per_step"]
                acc: dict[str, dict[str, list[float]]] = {}
                for w in worlds:
                    mem = build_memory(
                        field, w, k, cfg,
                        np.random.RandomState(w.seed * 31 + k))
                    for e in scored_envs(cfg, k)[:cfg.basin_envs]:
                        bp = basin_probe(field, w, e, mem, cfg)
                        for s, vals in bp.items():
                            a = acc.setdefault(s, {"all": [], "p95": []})
                            a["all"].append(vals["r_exact_all"])
                            a["p95"].append(vals["r_exact_95"])
                for s, a in acc.items():
                    if s not in per:
                        continue
                    sc = per[s]["scalars"]
                    # Move the env-bounded originals aside under the names
                    # `run_test_a` now uses, so the two are never confused.
                    for old, new in (("r_exact_all", "r_exact_all_envcues"),
                                     ("r_exact_95", "r_exact_95_envcues")):
                        if old in sc and new not in sc:
                            sc[new] = sc[old]
                    for name, v in (("r_exact_all", a["all"]),
                                    ("r_exact_95", a["p95"])):
                        arr = np.asarray(v, float)
                        sc[name] = {
                            "values": [float(x) for x in arr],
                            "mean": float(arr.mean()),
                            "std": float(arr.std()),
                            "p25": float(np.percentile(arr, 25)),
                            "p50": float(np.percentile(arr, 50)),
                            "p75": float(np.percentile(arr, 75)),
                            "n": int(arr.size),
                        }

            name = f"{cov}_{arm}_s{seed}".replace("%", "pct").replace(".", "_")
            with open(os.path.join(out_dir, name + ".json"), "w") as fh:
                json.dump(res, fh)
            written += 1
            print(f"  {cov:6s} {arm:10s} s{seed}  ok", flush=True)

    src_manifest = glob.glob(f"{ARCH}/attlow_g100/manifest.json")
    if src_manifest and only in (None, 0):
        shutil.copy2(src_manifest[0],
                     os.path.join(out_dir, "manifest.json"))
    print(f"\n{written} files -> {out_dir}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None,
         float(sys.argv[2]) if len(sys.argv) > 2 else None)
