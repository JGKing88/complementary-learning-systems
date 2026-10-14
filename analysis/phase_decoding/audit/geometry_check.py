"""Audit check 2: offline checks on an existing exp1 trials/ directory.

A. Parallelism: per-arena coding vectors d_a = mean(h|exploit,a) - mean(h|explore,a).
   Bernardi-style PS = mean pairwise cos(d_a, d_b) versus the pipeline's pooled
   cos(group-A diff, group-B diff) as a function of group size k, with a
   trial-label shuffle null (labels permuted across whole trials within arena).
B. Decoding: within-arena timestep split (pipeline) vs trial-held-out split;
   train-on-ONE-arena -> test on other arenas (CCGP-like); time-in-trial-only decoder.
"""
import argparse, json, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--trials", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_shuffle", type=int, default=20)
ap.add_argument("--n_dec_arenas", type=int, default=20)
ap.add_argument("--rows_per_arena", type=int, default=4000)
a = ap.parse_args()
rng = np.random.RandomState(0)
files = sorted(glob.glob(f"{a.trials}/per_arena/*.npz"))

def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))

def t_in_trial(trial):
    t = np.zeros(len(trial), int)
    for k in np.unique(trial):
        m = trial == k
        t[m] = np.arange(m.sum())
    return t

# ---- per-trial sums (exactly enough to recompute every centroid) ----
S, N, LAB = [], [], []          # per arena: (n_trials, H) sums, counts, labels
for f in files:
    z = np.load(f)
    s_list, n_list, l_list = [], [], []
    for lab, key in ((0, "explore"), (1, "exploit")):
        h = z[f"h_{key}"].astype(np.float64); tr = z[f"trial_{key}"]
        for k in np.unique(tr):
            m = tr == k
            s_list.append(h[m].sum(0)); n_list.append(m.sum()); l_list.append(lab)
    S.append(np.stack(s_list)); N.append(np.array(n_list)); LAB.append(np.array(l_list))
A = len(S); H = S[0].shape[1]

def arena_parts(labs):
    """Row-weighted per-arena sums/counts for each phase under labelling labs."""
    s0 = np.stack([S[i][labs[i] == 0].sum(0) for i in range(A)])
    s1 = np.stack([S[i][labs[i] == 1].sum(0) for i in range(A)])
    n0 = np.array([N[i][labs[i] == 0].sum() for i in range(A)])
    n1 = np.array([N[i][labs[i] == 1].sum() for i in range(A)])
    return s0, s1, n0, n1

def geometry(labs, ks=(1, 2, 5, 10, 20, 50), n_draw=50):
    s0, s1, n0, n1 = arena_parts(labs)
    d = s1 / n1[:, None] - s0 / n0[:, None]
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    C = dn @ dn.T
    iu = np.triu_indices(A, 1)
    out = {"ps_pairwise_mean": float(C[iu].mean()),
           "ps_pairwise_median": float(np.median(C[iu])),
           "shared_fraction": float(np.linalg.norm(d.mean(0))**2 /
                                    np.mean(np.sum(d**2, 1)))}
    r = np.random.RandomState(1)
    for k in ks:
        if 2 * k > A: continue
        cs = []
        for _ in range(n_draw):
            p = r.permutation(A); ga, gb = p[:k], p[k:2 * k]
            da = s1[ga].sum(0) / n1[ga].sum() - s0[ga].sum(0) / n0[ga].sum()
            db = s1[gb].sum(0) / n1[gb].sum() - s0[gb].sum(0) / n0[gb].sum()
            cs.append(cos(da, db))
        out[f"pooled_cos_k{k}"] = float(np.mean(cs))
    # the pipeline's own 80/20 number
    cs = []
    for _ in range(n_draw):
        p = r.permutation(A); te = p[:A // 5]; trn = p[A // 5:]
        da = s1[trn].sum(0) / n1[trn].sum() - s0[trn].sum(0) / n0[trn].sum()
        db = s1[te].sum(0) / n1[te].sum() - s0[te].sum(0) / n0[te].sum()
        cs.append(cos(da, db))
    out["pooled_cos_80_20"] = float(np.mean(cs))
    return out

res = {"trials": a.trials, "n_arenas": A, "H": H}
res["geometry"] = geometry(LAB)
nulls = [geometry([rng.permutation(l) for l in LAB]) for _ in range(a.n_shuffle)]
res["geometry_null_mean"] = {k: float(np.mean([n[k] for n in nulls])) for k in nulls[0]}
res["geometry_null_max"] = {k: float(np.max([n[k] for n in nulls])) for k in nulls[0]}
print(json.dumps(res, indent=1), flush=True)

# ---- decoding ----
def load_rows(f, cap=None, r=None):
    z = np.load(f)
    X = np.concatenate([z["h_explore"], z["h_exploit"]]).astype(np.float32)
    y = np.r_[np.zeros(len(z["h_explore"]), int), np.ones(len(z["h_exploit"]), int)]
    g = np.r_[z["trial_explore"], z["trial_exploit"] + 10_000]
    t = np.r_[t_in_trial(z["trial_explore"]), t_in_trial(z["trial_exploit"])]
    if cap and len(y) > cap:
        i = r.choice(len(y), cap, replace=False); X, y, g, t = X[i], y[i], g[i], t[i]
    return X, y, g, t

def fit_score(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr), ytr)
    return float(balanced_accuracy_score(yte, clf.predict(sc.transform(Xte))))

dec_ids = rng.choice(A, a.n_dec_arenas, replace=False)
w_step, w_trial = [], []
for i in dec_ids:
    X, y, g, t = load_rows(files[i])
    p = rng.permutation(len(y)); nt = len(y) // 5
    w_step.append(fit_score(X[p[nt:]], y[p[nt:]], X[p[:nt]], y[p[:nt]]))
    gs = np.unique(g); gte = set(rng.choice(gs, len(gs) // 5, replace=False))
    mte = np.array([k in gte for k in g])
    w_trial.append(fit_score(X[~mte], y[~mte], X[mte], y[mte]))
res["within_arena_timestep_split"] = float(np.mean(w_step))
res["within_arena_trial_split"] = float(np.mean(w_trial))
print("within", res["within_arena_timestep_split"], res["within_arena_trial_split"], flush=True)

# train on ONE arena, test on the pooled others (subsampled)
cache = {i: load_rows(files[i], a.rows_per_arena, rng) for i in dec_ids}
one = []
for i in dec_ids:
    Xtr, ytr, _, _ = cache[i]
    others = [j for j in dec_ids if j != i]
    Xte = np.concatenate([cache[j][0] for j in others]); yte = np.concatenate([cache[j][1] for j in others])
    one.append(fit_score(Xtr, ytr, Xte, yte))
res["train_one_arena_test_others"] = float(np.mean(one))
# time-in-trial only, cross-arena
half = len(dec_ids) // 2
tr_ids, te_ids = dec_ids[:half], dec_ids[half:]
def tfeat(t): return np.stack([t, t**2], 1).astype(np.float64)
Ttr = np.concatenate([tfeat(cache[j][3]) for j in tr_ids]); ytr = np.concatenate([cache[j][1] for j in tr_ids])
Tte = np.concatenate([tfeat(cache[j][3]) for j in te_ids]); yte = np.concatenate([cache[j][1] for j in te_ids])
res["time_only_cross_arena"] = fit_score(Ttr, ytr, Tte, yte)
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, "w"), indent=1)
