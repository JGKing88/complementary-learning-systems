"""Abstraction metrics on collected runs.

    python -m analysis.phase_decoding.run_analysis --config CONFIG.json --out OUT

For every run, condition and row selection, and for each feature space (the
RNN state ``h`` and the policy input ``x``):

- parallelism score with a trial-shuffle null;
- CCGP against the number of training arenas, with the geometric null;
- within-arena decoding with whole trials held out (the ceiling);
- a behaviour-only decoder (time in segment, distance to goal, speed, wall
  contact, distractor count), trained on 80% of arenas: how much phase is
  predictable from behaviour alone in that selection.

Selections:
- ``balanced``: equal rows per phase per arena, capped.
- ``matched``: per arena, the same joint histogram of (time in segment,
  distance to goal) in both phases.
- ``matched_dist_ge1``: ``matched`` restricted to trials with at least one
  distractor, where explore ``q`` is not identically zero.

Rows whose input is at the goal (definition B arrival steps) are excluded: the
goal reward enters the input on exactly those steps.

Definition B also gets switch-aligned decoder traces: a decoder fit on matched
``store`` rows of half the arenas, applied to every step of the other arenas'
``store`` and ``sham`` trials, aligned on the first arrival.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import geometry as g

BEHAVIOUR = ("t_seg", "dist", "speed", "wall", "n_dist")


def load_run(path: Path) -> tuple[dict, dict]:
    meta = json.loads((path / "meta.json").read_text())
    cols: dict[str, list] = {}
    for f in sorted((path / "per_arena").glob("*.npz")):
        z = np.load(f)
        n = len(z["t"])
        for k in z.files:
            if k in ("goal", "offset"):
                continue
            v = z[k]
            if k == "arena":
                v = np.full(n, int(v))
            cols.setdefault(k, []).append(v)
    data = {k: np.concatenate(v) for k, v in cols.items()}
    assert data["h"].shape[1] == meta["hidden_size"]
    data["xq"] = with_q_norms(data["x"], meta["channels"])
    data["qn"] = data["xq"][:, data["x"].shape[1]:]          # the norms alone
    return data, meta


def with_q_norms(x: np.ndarray, channel_layout: list) -> np.ndarray:
    """The policy input plus the length of every Hopfield readout channel.

    A linear decoder on ``x`` cannot compute ||q||, and the exploit policy is
    known to gate on it. Appending the norms gives the input baseline that
    nonlinearity for free, so ``h`` has to beat it to claim more than ||q||.
    """
    off, norms = 0, []
    for name, width in channel_layout:
        if name == "hopfield_signal" or name.startswith("hopfield_multistep_"):
            assert width == 2, (name, width)
            norms.append(np.linalg.norm(x[:, off:off + width], axis=1))
        off += width
    assert off == x.shape[1], (off, x.shape)
    assert norms, "no Hopfield readout channel in the policy input"
    return np.concatenate([x, np.stack(norms, 1).astype(np.float32)], 1)


def select(data: dict, mask: np.ndarray, how: str, cfg: dict,
           rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if how in ("matched_dist_ge1", "matched_qnorm"):
        idx = idx[data["n_dist"][idx] >= 1]
    # Definition B arenas where no search found the goal have no phase-1 rows.
    idx = idx[_arenas_with_both_phases(data, idx)]
    ph, ar = data["phase"][idx], data["arena"][idx]
    if how == "balanced":
        keep = g.balance_rows(ph, ar, cap=cfg["cap_rows_per_phase"], rng=rng)
    else:
        nuis = [data["t_seg"][idx], data["dist"][idx]]
        edges = [np.asarray(cfg["match"]["t_seg_edges"], float),
                 np.asarray(cfg["match"]["dist_edges"], float)]
        if how == "matched_abs_t":
            # Definition B: post-arrival rows are always later in the trial,
            # so an RNN state that drifts with time since the trial began
            # would decode phase. Match time since the trial began instead
            # of time since the last teleport.
            nuis[0] = data["t"][idx]
            edges[0] = np.asarray(cfg["match"]["abs_t_edges"], float)
        if how == "matched_qnorm":
            # Match time and the length of the current Hopfield readout, so
            # the input's ||q|| cannot tell the phases apart. Distance is
            # dropped: the ||q|| distributions overlap only partly (explore
            # 95th percentile ~ exploit 5th), and a third matched variable
            # leaves too few rows per arena. The behaviour-only decoder
            # reports the distance confound that remains. Edges are quantiles
            # of the pooled (phase-blind) distribution, open at both ends.
            qn = data["qnorm"][idx].astype(float)
            e = np.quantile(qn, np.linspace(0, 1, cfg["match"]["qnorm_bins"] + 1))
            e[0], e[-1] = -np.inf, np.inf
            nuis = [nuis[0], qn]
            edges = [edges[0], e]
        keep = g.match_rows(ph, ar, np.stack(nuis, 1).astype(float), edges, rng=rng)
        keep = keep[g.balance_rows(ph[keep], ar[keep], cap=cfg["cap_rows_per_phase"], rng=rng)]
    return idx[keep]


def _arenas_with_both_phases(data: dict, idx: np.ndarray) -> np.ndarray:
    ok = []
    for a in np.unique(data["arena"][idx]):
        p = data["phase"][idx][data["arena"][idx] == a]
        if (p == 0).any() and (p == 1).any():
            ok.append(a)
    return np.isin(data["arena"][idx], ok)


def metrics_for(data: dict, idx: np.ndarray, cfg: dict, rng: np.random.Generator) -> dict:
    out = {"n_rows": int(len(idx)),
           "n_arenas": int(len(np.unique(data["arena"][idx]))),
           "rows_per_phase": [int((data["phase"][idx] == p).sum()) for p in (0, 1)]}
    # The resampling unit is a segment: (trial, phase). Under definition A a
    # trial is one segment. Under B it has two, before and after the switch;
    # adjacent segments carry opposite labels, so holding one out while
    # training on its sibling leaks no label information.
    unit = data["trial"][idx] * 2 + data["phase"][idx]
    for feat in cfg["features"]:
        zs, keep = g.zscore_features(data[feat][idx], min_std=cfg["min_std"])
        r = g.Rows(zs, data["phase"][idx], data["arena"][idx], unit)
        out[feat] = {
            "n_features_kept": int(keep.sum()),
            "parallelism": g.parallelism(r, n_shuffle=cfg["parallelism"]["n_shuffle"], rng=rng),
            "ccgp": g.ccgp(r, rng=rng, n_jobs=cfg["n_jobs"], **cfg["ccgp"]),
            "within_arena": g.within_arena_decoding(r, rng=rng, **cfg["within"]),
        }
    beh = np.stack([data[k][idx].astype(float) for k in BEHAVIOUR], 1)
    rb = g.Rows(beh.astype(np.float32), data["phase"][idx], data["arena"][idx], unit)
    n_ar = out["n_arenas"]
    out["behaviour_only"] = g.ccgp(rb, k_train=[int(0.8 * n_ar)], n_rep=cfg["ccgp"]["n_rep"],
                                   n_test_arenas=n_ar, n_null=0, C=cfg["ccgp"]["C"],
                                   rng=rng, n_jobs=cfg["n_jobs"])
    return out


def switch_traces(data: dict, cfg: dict, rng: np.random.Generator) -> dict:
    """Decoder output around the first arrival, store vs sham, held-out arenas."""
    base = (~data["at_goal_in"].astype(bool))
    store = base & (data["condition"] == "store")
    idx = select(data, store, "matched", cfg, rng)
    arenas = rng.permutation(np.unique(data["arena"][idx]))
    tr_ar, te_ar = arenas[: len(arenas) // 2], arenas[len(arenas) // 2:]
    lo, hi = cfg["switch_window"]
    rel = np.arange(lo, hi + 1)
    out = {"rel": rel.tolist(), "n_train_arenas": int(len(tr_ar))}
    for feat in cfg["features"]:
        itr = idx[np.isin(data["arena"][idx], tr_ar)]
        sc = StandardScaler().fit(data[feat][itr])
        clf = LogisticRegression(C=cfg["ccgp"]["C"], max_iter=2000,
                                 class_weight="balanced").fit(sc.transform(data[feat][itr]),
                                                              data["phase"][itr])
        for cond in ("store", "sham"):
            m = np.isin(data["arena"], te_ar) & (data["condition"] == cond)
            trials = np.unique(data["trial"][m])
            mat = np.full((len(trials), len(rel)), np.nan)
            for i, tid in enumerate(trials):
                mt = np.flatnonzero(data["trial"] == tid)
                arr = np.flatnonzero(data["at_goal_in"][mt])
                if len(arr) == 0:
                    continue                      # search never found the goal
                t_sw = data["t"][mt][arr[0]]
                logit = clf.decision_function(sc.transform(data[feat][mt]))
                rr = data["t"][mt] - t_sw
                inw = (rr >= lo) & (rr <= hi)
                mat[i, rr[inw] - lo] = logit[inw]
            n = np.sum(~np.isnan(mat), 0)
            out[f"{feat}_{cond}_mean"] = np.nanmean(mat, 0).tolist()
            out[f"{feat}_{cond}_sem"] = (np.nanstd(mat, 0, ddof=1) / np.sqrt(n)).tolist()
            out[f"{feat}_{cond}_n"] = n.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated run names")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.json")
    names = a.only.split(",") if a.only else list(cfg["runs"])
    for name in names:
        run = cfg["runs"][name]
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(name.encode())])
        data, meta = load_run(Path(run["path"]))
        base = ~data["at_goal_in"].astype(bool)
        res = {"meta": meta, "definition": run["definition"], "by_condition": {}}
        conds = ["all"] if run["definition"] == "A" else ["store", "sham"]
        for cond in conds:
            m = base if cond == "all" else base & (data["condition"] == cond)
            res["by_condition"][cond] = {}
            for how in cfg["selections"]:
                idx = select(data, m, how, cfg, rng)
                print(f"[analysis] {name} {cond} {how}: {len(idx)} rows", flush=True)
                res["by_condition"][cond][how] = metrics_for(data, idx, cfg, rng)
                res["by_condition"][cond][how]["n_arenas_total"] = int(
                    len(np.unique(data["arena"])))
        if run["definition"] == "B":
            res["switch_traces"] = switch_traces(data, cfg, rng)
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"[analysis] wrote {out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
