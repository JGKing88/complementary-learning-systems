"""Is the RNN's phase signal beyond ||q|| just recent input history?

    python -m analysis.phase_decoding.history --config CONFIG.json --out OUT --only d0_A

Two complementary tests on the same rows (``matched``: time since teleport and
distance to goal equalised per arena), scored by the same cross-arena decoding
(train a linear decoder on k arenas, test on held-out arenas):

1. Input-history baselines. A linear decoder on the input over a window of
   the last W steps of the same trial: the readout lengths ``||q||`` alone, or
   the whole policy input ``x``, each with a lag-validity mask, optionally plus
   running summaries of ``||q||`` since the trial began. If a window reaches
   ``h``, what ``h`` adds over the current input is recent input; W is its
   timescale.
2. ``h`` given ``||q||``, two ways, with a spline basis of ``||q||``
   (current, a lag window, or the window plus since-start summaries) fit on
   the training arenas only within every fold:
   - residual: subtract the ridge prediction of ``h`` from the basis, decode
     the remainder. This measures phase information in ``h`` beyond what the
     basis predicts, like matching on ``||q||``. Rows where ``||q||`` already
     settles the phase carry no residual phase signal by construction, so
     they sit at chance and dilute the accuracy toward 0.5.
   - incremental: decode the basis alone, then the basis and ``h`` together.
     The gain is what ``h`` adds beyond ``||q||`` on every row, undiluted.

``h`` is a function of the input history alone, so test 1 can never exceed it
in information. Both tests ask about format: whether a linear readout of a
short input history already has the arena-invariant phase direction ``h`` has.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .run_analysis import load_run, select


# ------------------------------------------------------------------ features

def lag_index(trial: np.ndarray, t: np.ndarray, lag: int) -> np.ndarray:
    """Row index of the same trial's step t - lag, or -1 before the trial began.

    Relies on each trial's rows being contiguous and in step order, which the
    collector guarantees; asserted rather than assumed.
    """
    n = len(t)
    idx = np.arange(n) - lag
    ok = (t >= lag) & (idx >= 0)
    ok_idx = idx[ok]
    assert (trial[ok_idx] == trial[ok]).all() and (t[ok_idx] == t[ok] - lag).all(), \
        "trial rows are not contiguous and step-ordered"
    out = np.full(n, -1)
    out[ok] = ok_idx
    return out


def window(feat: np.ndarray, trial: np.ndarray, t: np.ndarray, width: int,
           rows: np.ndarray) -> np.ndarray:
    """For ``rows``: features at lags 0..width-1 (zero where the trial had not
    begun), then a validity mask per lag. (len(rows), width * (F + 1))."""
    parts, masks = [], []
    for lag in range(width):
        j = lag_index(trial, t, lag)[rows]
        v = np.zeros((len(rows), feat.shape[1]), feat.dtype)
        v[j >= 0] = feat[j[j >= 0]]
        parts.append(v)
        masks.append((j >= 0).astype(feat.dtype)[:, None])
    return np.concatenate(parts + masks, 1)


def since_start(qn: np.ndarray, trial: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Running mean, max and min of each ||q|| channel from the trial's first
    step to the current one. (n, 3 * channels)."""
    out = np.empty((len(qn), 3 * qn.shape[1]), qn.dtype)
    starts = np.flatnonzero(t == 0)
    ends = np.r_[starts[1:], len(t)]
    for s, e in zip(starts, ends):
        assert (trial[s:e] == trial[s]).all() and (t[s:e] == np.arange(e - s)).all()
        seg = qn[s:e]
        cnt = np.arange(1, e - s + 1)[:, None]
        out[s:e, :qn.shape[1]] = np.cumsum(seg, 0) / cnt
        out[s:e, qn.shape[1]:2 * qn.shape[1]] = np.maximum.accumulate(seg, 0)
        out[s:e, 2 * qn.shape[1]:] = np.minimum.accumulate(seg, 0)
    return out


def build_features(data: dict, windows: list[int], rows: np.ndarray):
    """Every baseline feature set and every regressor set, for ``rows``.
    History is looked up over all rows of the trial, selected or not."""
    tr, t = data["trial"], data["t"]
    qn_all = data["qn"].astype(np.float32)
    summ = since_start(qn_all, tr, t)[rows]
    qn = qn_all[rows]
    feats = {"x_now": data["x"][rows], "q_now": qn}
    reg = {"q_now": qn}
    for w in windows:
        qw = window(qn_all, tr, t, w, rows)
        feats[f"q_win{w}"] = qw
        feats[f"x_win{w}"] = window(data["x"], tr, t, w, rows)
        reg[f"q_win{w}"] = qw
    wmax = max(windows)
    feats["q_hist"] = np.concatenate([feats[f"q_win{wmax}"], summ], 1)
    feats["x_hist"] = np.concatenate([feats[f"x_win{wmax}"], summ], 1)
    reg["q_hist"] = feats["q_hist"]
    return feats, reg


# ------------------------------------------------------------------ decoding

def _fit_score(ztr, ytr, zte, yte, C):
    sc = StandardScaler().fit(ztr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(ztr), ytr)
    return float(balanced_accuracy_score(yte, clf.predict(sc.transform(zte))))


MODES = ("plain", "residual", "basis_only", "basis_plus")


def _basis(rtr, rte, spline_cfg):
    """Spline basis on the continuous regressor columns, fit on training
    arenas only; 0/1 columns (lag masks) enter linearly, and columns constant
    on the training rows are dropped."""
    live = rtr.std(0) > 0
    binary = live & np.all((rtr == 0) | (rtr == 1), 0)
    cont = live & ~binary
    sp = SplineTransformer(**spline_cfg).fit(rtr[:, cont])
    btr = np.concatenate([sp.transform(rtr[:, cont]), rtr[:, binary]], 1)
    bte = np.concatenate([sp.transform(rte[:, cont]), rte[:, binary]], 1)
    return btr, bte


def _job(z, y, arena, tr_ar, te_ar, reg, mode, spline_cfg, ridge_alpha, C):
    """One fold. ``mode``:
    plain       decode z
    residual    decode z minus its ridge prediction from the basis of reg
    basis_only  decode the basis of reg (z unused)
    basis_plus  decode [basis of reg, z]
    """
    mtr, mte = np.isin(arena, tr_ar), np.isin(arena, te_ar)
    if mode == "plain":
        return _fit_score(z[mtr], y[mtr], z[mte], y[mte], C)
    btr, bte = _basis(reg[mtr], reg[mte], spline_cfg)
    if mode == "basis_only":
        return _fit_score(btr, y[mtr], bte, y[mte], C)
    ztr, zte = z[mtr], z[mte]
    if mode == "basis_plus":
        return _fit_score(np.concatenate([btr, ztr], 1), y[mtr],
                          np.concatenate([bte, zte], 1), y[mte], C)
    assert mode == "residual", mode
    mu = ztr.mean(0)
    rg = Ridge(alpha=ridge_alpha).fit(btr, ztr - mu)
    return _fit_score(ztr - mu - rg.predict(btr), y[mtr],
                      zte - mu - rg.predict(bte), y[mte], C)


def cross_arena(z, y, arena, *, reg, mode, k_train, n_rep, n_test, rng, n_jobs,
                spline_cfg, ridge_alpha, C) -> dict:
    assert mode in MODES and (reg is None) == (mode == "plain"), (mode, reg is None)
    arenas = np.unique(arena)
    jobs, keys = [], []
    for k in k_train:
        assert 1 <= k < len(arenas)
        n_te = min(n_test, len(arenas) - k)
        for _ in range(n_rep):
            p = rng.permutation(arenas)
            jobs.append(delayed(_job)(z, y, arena, p[:k], p[k:k + n_te], reg,
                                      mode, spline_cfg, ridge_alpha, C))
            keys.append(k)
    scores = np.asarray(Parallel(n_jobs=n_jobs)(jobs))
    keys = np.asarray(keys)
    return {str(k): {"mean": float(scores[keys == k].mean()),
                     "sd": float(scores[keys == k].std(ddof=1)),
                     "values": scores[keys == k].tolist()} for k in k_train}


# ------------------------------------------------- definition B: time since arrival

def steps_since_arrival(data: dict) -> np.ndarray:
    """Per row, steps since the trial's first arrival (the at-goal input step),
    or -1 for rows at or before it."""
    trial, t, arr = data["trial"], data["t"], data["at_goal_in"].astype(bool)
    first = {}
    for tid, tt in zip(trial[arr], t[arr]):
        first.setdefault(int(tid), int(tt))
    t_arr = np.array([first.get(int(k), -1) for k in trial])
    lag = np.where((t_arr >= 0) & (t > t_arr), t - t_arr, -1)
    # phase 1 is exactly "after the first arrival"
    assert ((lag > 0) == (data["phase"] == 1)).all()
    return lag


def _lag_job(z, y, arena, lag, tr_ar, te_ar, bins, C):
    mtr, mte = np.isin(arena, tr_ar), np.isin(arena, te_ar)
    sc = StandardScaler().fit(z[mtr])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(z[mtr]), y[mtr])
    pred = clf.predict(sc.transform(z[mte]))
    yt, lt = y[mte], lag[mte]
    tnr = float((pred[yt == 0] == 0).mean())
    out = []
    for lo, hi in bins:
        m = (yt == 1) & (lt >= lo) & (lt <= hi)
        out.append((float((pred[m] == 1).mean()) if m.any() else np.nan, int(m.sum())))
    return tnr, out


def lag_resolved(z, y, arena, lag, *, bins, k, n_rep, n_test, rng, n_jobs, C) -> dict:
    """Train on k arenas, test on held-out arenas. Balanced accuracy per bin of
    steps since arrival: (recall of post-arrival rows in the bin + recall of
    pre-arrival rows) / 2."""
    arenas = np.unique(arena)
    n_te = min(n_test, len(arenas) - k)
    jobs = []
    for _ in range(n_rep):
        p = rng.permutation(arenas)
        jobs.append(delayed(_lag_job)(z, y, arena, lag, p[:k], p[k:k + n_te], bins, C))
    res = Parallel(n_jobs=n_jobs)(jobs)
    tnr = np.array([r[0] for r in res])
    tpr = np.array([[b[0] for b in r[1]] for r in res])
    n = np.array([[b[1] for b in r[1]] for r in res])
    bal = (tpr + tnr[:, None]) / 2
    return {"bins": [list(b) for b in bins], "pre_recall": float(tnr.mean()),
            "balanced_mean": np.nanmean(bal, 0).tolist(),
            "balanced_sd": np.nanstd(bal, 0, ddof=1).tolist(),
            "n_post_rows_per_rep": n.mean(0).tolist(), "k_train": k, "n_rep": n_rep}


def run_definition_b(name, data, meta, cfg, rng) -> dict:
    hb = cfg["history_b"]
    base = ~data["at_goal_in"].astype(bool)
    lag_all = steps_since_arrival(data)
    kw = dict(k_train=hb["k_train"], n_rep=hb["n_rep"], n_test=hb["n_test_arenas"],
              n_jobs=cfg["n_jobs"], spline_cfg=hb["spline"], ridge_alpha=hb["ridge_alpha"],
              C=hb["C"])
    res = {"meta": meta, "definition": "B", "by_condition": {}}
    for cond in hb["conditions"]:
        res["by_condition"][cond] = {}
        for sel in hb["selections"]:
            idx = select(data, base & (data["condition"] == cond), sel, cfg, rng)
            feats, _ = build_features(data, hb["windows"], idx)
            feats["h"] = data["h"][idx]
            y, ar, lag = data["phase"][idx], data["arena"][idx], lag_all[idx]
            r = {"n_rows": int(len(idx)), "baselines": {}, "by_lag": {}}
            for fname in hb["baseline_features"]:
                r["baselines"][fname] = cross_arena(feats[fname], y, ar, reg=None,
                                                    mode="plain", rng=rng, **kw)
                print(f"[history-B] {name} {cond} {sel} {fname}: "
                      f"{ {k: round(v['mean'], 3) for k, v in r['baselines'][fname].items()} }",
                      flush=True)
            for fname in hb["lag_features"]:
                r["by_lag"][fname] = lag_resolved(
                    feats[fname], y, ar, lag, bins=hb["lag_bins"], k=hb["lag_k"],
                    n_rep=hb["n_rep"], n_test=hb["n_test_arenas"], rng=rng,
                    n_jobs=cfg["n_jobs"], C=hb["C"])
                print(f"[history-B] {name} {cond} {sel} by-lag {fname}: "
                      f"{[round(v, 3) for v in r['by_lag'][fname]['balanced_mean']]}", flush=True)
            res["by_condition"][cond][sel] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.json")
    for name in a.only.split(","):
        run = cfg["runs"][name]
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(name.encode())])
        data, meta = load_run(Path(run["path"]))
        if run["definition"] == "B":
            res = run_definition_b(name, data, meta, cfg, rng)
            (out / f"{name}.json").write_text(json.dumps(res, indent=1))
            print(f"[history-B] wrote {out / (name + '.json')}", flush=True)
            continue
        hc = cfg["history"]
        mask = ~data["at_goal_in"].astype(bool)
        idx = select(data, mask, hc["selection"], cfg, rng)
        feats, reg = build_features(data, hc["windows"], idx)
        y, ar = data["phase"][idx], data["arena"][idx]
        print(f"[history] {name}: {len(idx)} rows", flush=True)
        kw = dict(k_train=hc["k_train"], n_rep=hc["n_rep"], n_test=hc["n_test_arenas"],
                  n_jobs=cfg["n_jobs"], spline_cfg=hc["spline"],
                  ridge_alpha=hc["ridge_alpha"], C=hc["C"])
        h = data["h"][idx]
        res = {"meta": meta, "selection": hc["selection"], "n_rows": int(len(idx)),
               "baselines": {}, "h_residual": {}, "q_basis": {}, "q_basis_plus_h": {}}

        def log(kind, key, r):
            print(f"[history] {name} {kind} {key}: "
                  f"{ {k: round(v['mean'], 3) for k, v in r.items()} }", flush=True)

        res["baselines"]["h"] = cross_arena(h, y, ar, reg=None, mode="plain", rng=rng, **kw)
        log("baseline", "h", res["baselines"]["h"])
        for fname, f in feats.items():
            res["baselines"][fname] = cross_arena(f, y, ar, reg=None, mode="plain", rng=rng, **kw)
            log("baseline", fname, res["baselines"][fname])
        for rname, r in reg.items():
            for kind, mode in (("h_residual", "residual"), ("q_basis", "basis_only"),
                               ("q_basis_plus_h", "basis_plus")):
                res[kind][rname] = cross_arena(h, y, ar, reg=r, mode=mode, rng=rng, **kw)
                log(kind, rname, res[kind][rname])
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"[history] wrote {out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
