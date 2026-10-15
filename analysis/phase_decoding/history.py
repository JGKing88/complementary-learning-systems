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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    hc = cfg["history"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.json")
    for name in a.only.split(","):
        run = cfg["runs"][name]
        assert run["definition"] == "A", "history windows are defined for definition A"
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(name.encode())])
        data, meta = load_run(Path(run["path"]))
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
