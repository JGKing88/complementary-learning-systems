"""P3 stage 2 -- fit the ideal observer and read off the bound.

`docs/EXPERIMENTS_NAV_P2.md` §7.3. Consumes what `ideal_observer.py` wrote and
answers the four questions the spec asks of it:

  1. how separable each cue is on its own, as a **distribution over envs**
     rather than a mean (finding 19, and §5.2.1: the draw count is the larger
     variance term);
  2. the ideal-observer AUC on the full policy-visible feature vector,
     cross-validated across **held-out envs** -- the number that bounds how
     much of phase 1's mode B is even detectable;
  3. which cue group carries it -- by leave-one-out, **leave-one-in, and the
     A u C joint drop**, because §7.2 predicts A and C are largely redundant
     and a leave-one-out alone would report both as worthless;
  4. what probing behaviour buys information fastest, per step and per unit of
     distance travelled.

**Controls, on every number.** A result here is three numbers, never one:

  * **chance** -- the same fit with the labels permuted within (env, level).
    Anything that does not land at 0.500 means the CV is leaking across the
    fold boundary, which is the failure mode this protocol exists to prevent.
  * **ceiling** -- the same fit with the oracle channels (`o_c1D`, `o_c2D`)
    added. Those are the group-C statistics *before* the 2-D tangent
    projection, i.e. an input channel that could be fed but is not; the gap
    says how much is being thrown away by projecting.
  * **degenerate condition, separated** -- at `n_dist = 0` the goal-absent
    memory is empty and `q = 0` exactly. That is a perfect cue for a reason
    that has nothing to do with the signal, so it is printed on its own line
    and excluded from every pooled headline.

**Why held-out envs and not held-out rows.** Every row of an env shares its
wall code, its goal, its scaffold patch and its distractor exclusion region.
Splitting rows at random lets the classifier memorize the env and would report
an AUC that no policy in a new arena could realize.

    python -m analysis.nav_p2.ideal_observer_fit --npz results/nav_p2/io_probe.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.nav_p2.io_features import GROUPS, POLICY_GROUPS

_ANALYSES = ["anchor", "single", "main", "ablation", "probes", "walls"]


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based ROC AUC. NaN when one class is absent -- never 0.5.

    Returning 0.5 for a one-class problem is how a silently empty negative set
    reads as "chance" instead of as "no measurement", which is exactly the
    fault this phase caught twice.
    """
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _avg_rank(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _avg_rank(s: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged.

    Tie handling is not a nicety here. Without it a constant score gets an
    arbitrary strict ordering, and because rows arrive grouped by regime that
    ordering correlates with the label -- the constant-feature control then
    reads 0.518 instead of 0.500 and the leakage detector is itself leaking.
    """
    order = np.argsort(s, kind="mergesort")
    sv = np.asarray(s)[order]
    r = np.empty(len(sv), dtype=np.float64)
    r[order] = np.arange(1, len(sv) + 1, dtype=np.float64)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def _fmt(x, w=6, p=3):
    return f"{'--':>{w}}" if not np.isfinite(x) else f"{x:>{w}.{p}f}"


# ---------------------------------------------------------------------------
# Cross-validated fit
# ---------------------------------------------------------------------------

def _fit_predict(Xtr, ytr, Xte, model):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ztr = np.clip(Ztr, -8, 8); Zte = np.clip(Zte, -8, 8)
    if model == "gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.12,
                                           max_leaf_nodes=15,
                                           early_stopping=False,
                                           l2_regularization=1.0,
                                           random_state=0)
        m.fit(Ztr, ytr)
        return m.predict_proba(Zte)[:, 1]
    m = LogisticRegression(max_iter=3000, C=1.0)
    m.fit(Ztr, ytr)
    return m.decision_function(Zte)


def cv_auc(X, y, env, cols, n_folds=6, model="lr", rng=None, permute=False):
    """Out-of-fold AUC, folds split by ENV. Returns (pooled, per_env array).

    `permute` shuffles labels within each env, which must send the pooled AUC
    to 0.500. It is the leakage detector, not a nicety.
    """
    uenv = np.unique(env)
    if len(uenv) < n_folds:
        n_folds = max(2, len(uenv))
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), np.array([])
    yy = y.copy()
    if permute:
        rng = rng or np.random.RandomState(0)
        for e in uenv:
            m = env == e
            yy[m] = rng.permutation(yy[m])
    Xc = X[:, cols]
    scores = np.full(len(y), np.nan)
    fold_of = {e: i % n_folds for i, e in enumerate(uenv)}
    fid = np.array([fold_of[e] for e in env])
    for f in range(n_folds):
        te = fid == f
        tr = ~te
        if not te.any() or yy[tr].sum() in (0, int(tr.sum())):
            continue
        s = _fit_predict(Xc[tr], yy[tr], Xc[te], model)
        # Rank-normalize within the fold before pooling. Each fold's model has
        # its own intercept, so raw scores are not comparable across folds and
        # pooling them lets a difference in fold class-balance masquerade as
        # signal -- visible as a constant-feature control that reads 0.496
        # instead of exactly 0.500.
        scores[te] = _avg_rank(s) / (len(s) + 1.0) if len(s) > 1 else 0.5
    ok = np.isfinite(scores)
    pooled = auc(yy[ok], scores[ok])
    per = []
    for e in uenv:
        m = (env == e) & ok
        if m.sum() > 4:
            per.append(auc(yy[m], scores[m]))
    return pooled, np.array([v for v in per if np.isfinite(v)])


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

class Data:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.X = z["X"]                        # (T, N, F)
        self.Xo = z["X_oracle"]                # (T, N, Fo)
        self.feat = [str(s) for s in z["features"]]
        self.ofeat = [str(s) for s in z["oracle_features"]]
        self.probes = [str(s) for s in z["probes"]]
        self.ck = z["checkpoints"]
        self.levels = z["levels"]
        self.y = {"ep": z["y_ep"], "step": z["y_step"], "trust": z["y_trust"]}
        self.valid = {"ep": z["valid"], "step": z["valid"],
                      "trust": z["valid_trust"]}
        self.env = z["meta_env"]
        self.level = z["meta_level"]
        self.regime = z["meta_regime"]
        self.probe = z["meta_probe"]
        self.path_len = z["path_len"]
        self.net_disp = z["net_disp"]
        self.wall = z["wall_dist"]
        self.clip = z["clip_any"]
        self.trust_thresh = float(z["trust_thresh"])
        # X with the oracle channels appended, so a column list can address
        # either. Built once; the arrays are already in memory.
        self.Xall = np.concatenate([self.X, self.Xo], axis=2)
        self.allfeat = self.feat + self.ofeat
        self.idx = {n: i for i, n in enumerate(self.allfeat)}

    def cols(self, groups=None, names=None, drop=None):
        if names is None:
            names = []
            for g in (groups or POLICY_GROUPS):
                names += GROUPS[g] if g in GROUPS else [g]
        if drop:
            names = [n for n in names if n not in drop]
        return [self.idx[n] for n in names]

    def slice(self, target, ti, level=None, probe=None, regime=None,
              exclude_zero=True):
        # `trust_present` is Q_trust restricted to episodes whose memory DOES
        # contain the goal -- the exploit regime, and therefore the version of
        # the headline that speaks directly to phase 1's mode B. Kept as an
        # alias rather than a separate label so both read the same array.
        if target in TARGET_ALIAS:
            target, regime = TARGET_ALIAS[target]
        m = self.valid[target][ti].copy()
        if level is not None:
            m &= self.level == level
        elif exclude_zero:
            m &= self.level != 0
        if probe is not None:
            m &= self.probe == self.probes.index(probe)
        if regime is not None:
            m &= self.regime == regime
        return (self.Xall[ti][m], self.y[target][ti][m].astype(int),
                self.env[m], m)


TARGET_ALIAS = {
    "trust_present": ("trust", 1),
    "trust_absent": ("trust", 0),
    "step_present": ("step", 1),
}

POLICY = POLICY_GROUPS
SETS = {
    "A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"],
    "ABC (policy)": POLICY,
    "ABCD": POLICY + ["D"],
    "AB (drop C)": ["A", "B"],
    "AC (drop B)": ["A", "C"],
    "BC (drop A)": ["B", "C"],
    "B (drop AuC)": ["B"],
}


def _hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# ---------------------------------------------------------------------------


def run_anchor(d: argparse.Namespace, D: Data, args):
    _hdr("A. SANITY -- controls that must pass before any number is read")
    ti = len(D.ck) - 1
    probe = args.probe
    for target in ("ep", "trust"):
        X, y, env, _ = D.slice(target, ti, probe=probe)
        cols = D.cols(POLICY)
        pooled, per = cv_auc(X, y, env, cols, args.folds, "lr")
        chance, _ = cv_auc(X, y, env, cols, args.folds, "lr",
                           rng=np.random.RandomState(1), permute=True)
        const, _ = cv_auc(np.zeros_like(X), y, env, cols, args.folds, "lr")
        print(f"  Q_{target:<6} t={D.ck[ti]:<3} n={len(y):>7}  "
              f"AUC {_fmt(pooled)}   label-permuted {_fmt(chance)}   "
              f"constant-features {_fmt(const)}")
    print("  Both controls must read 0.500 +/- sampling noise. A permuted AUC "
          "above chance means the fold split leaks.")


def run_single(D: Data, args):
    _hdr("B. PER-STATISTIC SEPARABILITY (§7.3 item 1) -- single-feature AUC")
    print(f"  probe = {args.probe}; per-env median [p10, p90] over "
          f"{len(np.unique(D.env))} envs; sign-corrected (a cue that predicts "
          "the negative class is still a cue)")
    for target in args.targets:
        for lvl in args.levels:
            ti_list = [i for i, c in enumerate(D.ck) if c in args.t_show]
            print(f"\n  --- Q_{target}, n_dist = {lvl} ---")
            print(f"  {'feature':<16} " + " ".join(
                f"t={D.ck[i]:<7}" for i in ti_list))
            rows = []
            for name in D.allfeat:
                cells = []
                for ti in ti_list:
                    X, y, env, _ = D.slice(target, ti, level=lvl,
                                           probe=args.probe)
                    if len(y) == 0 or y.sum() in (0, len(y)):
                        cells.append(float("nan")); continue
                    per = []
                    for e in np.unique(env):
                        m = env == e
                        if m.sum() > 4:
                            a = auc(y[m], X[m, D.idx[name]])
                            if np.isfinite(a):
                                per.append(max(a, 1 - a))
                    cells.append(np.median(per) if per else float("nan"))
                rows.append((name, cells))
            rows.sort(key=lambda r: -np.nanmax(r[1]) if np.any(
                np.isfinite(r[1])) else 0)
            for name, cells in rows[:args.top]:
                print(f"  {name:<16} " + " ".join(_fmt(c, 9) for c in cells))


def run_main(D: Data, args):
    _hdr("C. THE IDEAL-OBSERVER BOUND (§7.3 item 2)")
    print(f"  probe = {args.probe}; logistic regression and gradient-boosted "
          f"trees on the {len(D.cols(POLICY))} policy-visible features; "
          f"{args.folds}-fold CV over held-out envs.")
    print("  'ceiling' adds the two D-dimensional recall statistics -- an "
          "input channel the policy could be fed but is not.")
    out = {}
    for target in args.targets:
        print(f"\n  --- Q_{target} " + "-" * 50)
        print(f"  {'n_d':>4} {'t':>4} {'n':>7} {'P(y)':>6} "
              f"{'LR':>7} {'GBT':>7} {'ceiling':>8} {'chance':>7}  "
              f"per-env median [p10,p90]")
        for lvl in args.levels:
            for ti, c in enumerate(D.ck):
                if c not in args.t_show:
                    continue
                X, y, env, _ = D.slice(target, ti, level=lvl,
                                       probe=args.probe)
                if len(y) < 20 or y.sum() in (0, len(y)):
                    print(f"  {lvl:>4} {c:>4} {len(y):>7}  "
                          "(one class -- no measurement)")
                    continue
                cp = D.cols(POLICY)
                lr, per = cv_auc(X, y, env, cp, args.folds, "lr")
                gbt, _ = cv_auc(X, y, env, cp, args.folds, "gbt")
                ceil, _ = cv_auc(X, y, env, cp + D.cols(names=["o_c1D",
                                                               "o_c2D"]),
                                 args.folds, "gbt")
                ch, _ = cv_auc(X, y, env, cp, args.folds, "lr",
                               rng=np.random.RandomState(2), permute=True)
                band = (f"{np.median(per):.3f} [{np.percentile(per, 10):.3f}, "
                        f"{np.percentile(per, 90):.3f}]" if len(per) else "--")
                print(f"  {lvl:>4} {c:>4} {len(y):>7} {y.mean():>6.3f} "
                      f"{_fmt(lr, 7)} {_fmt(gbt, 7)} {_fmt(ceil, 8)} "
                      f"{_fmt(ch, 7)}  {band}")
                out[f"{target}|{lvl}|{c}"] = dict(
                    lr=lr, gbt=gbt, ceiling=ceil, chance=ch, n=int(len(y)),
                    pos=float(y.mean()),
                    per_env=[float(v) for v in per])
    return out


def run_ablation(D: Data, args):
    _hdr("D. CUE ABLATION (§7.3 item 3) -- leave-one-out AND leave-one-in")
    print(f"  probe = {args.probe}, GBT, {args.folds}-fold CV over envs. "
          "§7.2 predicts A and C are largely redundant, so the leave-one-out "
          "column alone would report both as contributing nothing.")
    print("  'AB (drop C)' is also the §7.3 item 3b test: recall depths "
          "{1,2,3} against depth {1} alone.")
    names = list(SETS.keys())
    for target in args.targets:
        for lvl in args.ablation_levels:
            for c in args.ablation_t:
                if c not in list(D.ck):
                    continue
                ti = list(D.ck).index(c)
                X, y, env, _ = D.slice(target, ti, level=lvl, probe=args.probe)
                if len(y) < 20 or y.sum() in (0, len(y)):
                    continue
                print(f"\n  --- Q_{target}, n_dist={lvl}, t={c}, n={len(y)}")
                full, _ = cv_auc(X, y, env, D.cols(POLICY), args.folds, "gbt")
                print(f"  {'feature set':<16} {'AUC':>7} {'d vs ABC':>9}")
                for nm in names:
                    cols = D.cols(SETS[nm])
                    a, _ = cv_auc(X, y, env, cols, args.folds, "gbt")
                    print(f"  {nm:<16} {_fmt(a, 7)} {_fmt(a - full, 9)}")


def run_probes(D: Data, args):
    _hdr("E. PROBING BEHAVIOUR (§7.3 item 4)")
    print(f"  GBT on the policy-visible set, {args.folds}-fold CV over envs. "
          "'dist' is mean path length travelled by t; the per-unit-distance "
          "column is (AUC - 0.5) / dist, since probing costs coverage.")
    for target in args.targets:
        for lvl in args.probe_levels:
            print(f"\n  --- Q_{target}, n_dist = {lvl} " + "-" * 40)
            head = " ".join(f"t={c:<5}" for c in D.ck if c in args.t_show)
            print(f"  {'probe':<10} {head}   {'steps->0.95':>12} "
                  f"{'AUC/dist@t=8':>13}")
            for pr in D.probes:
                aucs, dists = [], []
                for ti, c in enumerate(D.ck):
                    X, y, env, m = D.slice(target, ti, level=lvl, probe=pr)
                    if len(y) < 20 or y.sum() in (0, len(y)):
                        aucs.append(float("nan")); dists.append(float("nan"))
                        continue
                    a, _ = cv_auc(X, y, env, D.cols(POLICY), args.folds, "gbt")
                    aucs.append(a)
                    dists.append(float(D.path_len[ti][m].mean()))
                aucs = np.array(aucs); dists = np.array(dists)
                # first t at which the curve crosses 0.95, linearly interpolated
                cross = float("nan")
                for i in range(len(aucs)):
                    if np.isfinite(aucs[i]) and aucs[i] >= 0.95:
                        if i == 0:
                            cross = float(D.ck[0])
                        else:
                            x0, x1 = D.ck[i - 1], D.ck[i]
                            y0, y1 = aucs[i - 1], aucs[i]
                            cross = float(x0 + (0.95 - y0) * (x1 - x0)
                                          / max(y1 - y0, 1e-9))
                        break
                i8 = [j for j, c in enumerate(D.ck) if c == 8]
                per_d = ((aucs[i8[0]] - 0.5) / max(dists[i8[0]], 1e-9)
                         if i8 and np.isfinite(aucs[i8[0]]) else float("nan"))
                show = " ".join(_fmt(aucs[j], 7)
                                for j, c in enumerate(D.ck) if c in args.t_show)
                print(f"  {pr:<10} {show}   {_fmt(cross, 12, 1)} "
                      f"{_fmt(per_d, 13, 4)}")
            # The distance each probe actually covers, so the column above can
            # be read: a probe can win per-step and lose per-unit-motion.
            row = []
            for pr in D.probes:
                ti = len(D.ck) - 1
                _, _, _, m = D.slice(target, ti, level=lvl, probe=pr)
                row.append(f"{pr}={D.path_len[ti][m].mean():.1f}/"
                           f"{D.net_disp[ti][m].mean():.1f}")
            print(f"  path length / net displacement at t={D.ck[-1]}: "
                  + "  ".join(row))


def run_walls(D: Data, args):
    _hdr("F. CHANNEL ABLATION AT WALLS (§7.3 item 6, H-wall)")
    print("  a3 and b2 computed from the REALIZED displacement (the policy "
          "input added in §4 B4) against the same two from the COMMANDED "
          "action. Single-feature AUC, per-env median, split on whether the "
          "episode has hit a clip by t.")
    pairs = [("a3_resid", "x_a3_cmd"), ("b2_spread", "x_b2_cmd")]
    for target in args.targets:
        for lvl in args.probe_levels:
            for c in args.ablation_t:
                if c not in list(D.ck):
                    continue
                ti = list(D.ck).index(c)
                print(f"\n  --- Q_{target}, n_dist={lvl}, t={c}")
                print(f"  {'condition':<22} " + " ".join(
                    f"{a:>12}/{b:<12}" for a, b in pairs))
                for cond, sel in (("no clip yet", False), ("has clipped", True)):
                    X, y, env, m = D.slice(target, ti, level=lvl,
                                           probe=args.probe)
                    sub = D.clip[ti][m] == sel
                    if sub.sum() < 40 or y[sub].sum() in (0, int(sub.sum())):
                        print(f"  {cond:<22} (too few rows: {int(sub.sum())})")
                        continue
                    out = []
                    for a, b in pairs:
                        for nm in (a, b):
                            per = []
                            for e in np.unique(env[sub]):
                                mm = env[sub] == e
                                if mm.sum() > 4:
                                    v = auc(y[sub][mm], X[sub][mm, D.idx[nm]])
                                    if np.isfinite(v):
                                        per.append(max(v, 1 - v))
                            out.append(np.median(per) if per else float("nan"))
                    print(f"  {cond:<22} " + " ".join(_fmt(v, 12) for v in out)
                          + f"   (n={int(sub.sum())})")
                # And the near-wall split the original H-wall asked for.
                X, y, env, m = D.slice(target, ti, level=lvl, probe=args.probe)
                w = D.wall[ti][m]
                for cond, sub in (("wall_dist <= 1", w <= 1),
                                  ("wall_dist >= 4", w >= 4)):
                    if sub.sum() < 40 or y[sub].sum() in (0, int(sub.sum())):
                        continue
                    out = []
                    for a, b in pairs:
                        for nm in (a, b):
                            per = []
                            for e in np.unique(env[sub]):
                                mm = env[sub] == e
                                if mm.sum() > 4:
                                    v = auc(y[sub][mm], X[sub][mm, D.idx[nm]])
                                    if np.isfinite(v):
                                        per.append(max(v, 1 - v))
                            out.append(np.median(per) if per else float("nan"))
                    print(f"  {cond:<22} " + " ".join(_fmt(v, 12) for v in out)
                          + f"   (n={int(sub.sum())})")


def run_degenerate(D: Data, args):
    _hdr("G. THE DEGENERATE CONDITION, REPORTED SEPARATELY (§7.3 item 1)")
    print("  n_dist = 0: the goal-absent memory is EMPTY, so q = 0 exactly. "
          "Q_ep is perfect for a reason that is not about the signal, and "
          "Q_trust has no negative rows at all because dir_cos is undefined.")
    for target in args.targets:
        for c in args.ablation_t:
            if c not in list(D.ck):
                continue
            ti = list(D.ck).index(c)
            X, y, env, _ = D.slice(target, ti, level=0, probe=args.probe)
            if len(y) == 0:
                print(f"  Q_{target} t={c}: no valid rows")
                continue
            if y.sum() in (0, len(y)):
                print(f"  Q_{target} t={c}: n={len(y)}, all one class "
                      f"(P(y)={y.mean():.3f}) -- no AUC exists")
                continue
            a, _ = cv_auc(X, y, env, D.cols(POLICY), args.folds, "gbt")
            print(f"  Q_{target} t={c}: n={len(y)}  P(y)={y.mean():.3f}  "
                  f"AUC {_fmt(a)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--do", nargs="+",
                   default=["anchor", "degenerate", "single", "main",
                            "ablation", "probes", "walls"])
    p.add_argument("--probe", default="billiard",
                   help="probe whose trajectories the main tables are fitted "
                        "on. billiard is the reactive explore baseline the "
                        "trained policies most resemble.")
    p.add_argument("--targets", nargs="+", default=["trust", "ep", "step"])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 3, 10])
    p.add_argument("--ablation_levels", type=int, nargs="+", default=[10])
    p.add_argument("--probe_levels", type=int, nargs="+", default=[10])
    p.add_argument("--t_show", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--ablation_t", type=int, nargs="+", default=[1, 8, 64])
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--top", type=int, default=14)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    D = Data(args.npz)
    print(f"loaded {args.npz}")
    print(f"  {D.X.shape[1]:,} rows x {len(D.ck)} checkpoints, "
          f"{len(np.unique(D.env))} envs, probes {D.probes}, "
          f"levels {list(D.levels)}, trust threshold cos >= {D.trust_thresh}")

    main_out = None
    if "anchor" in args.do:
        run_anchor(None, D, args)
    if "degenerate" in args.do:
        run_degenerate(D, args)
    if "single" in args.do:
        run_single(D, args)
    if "main" in args.do:
        main_out = run_main(D, args)
    if "ablation" in args.do:
        run_ablation(D, args)
    if "probes" in args.do:
        run_probes(D, args)
    if "walls" in args.do:
        run_walls(D, args)

    if args.json_out and main_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(main_out, f, indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
