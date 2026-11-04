"""Is the goal in idea 3's hidden state? Linear probes (GRID_MLP_NAV_PLAN §7).

    python -m analysis.idea3.goal_probe --ckpt <navigate_uN.pt> --out <json>
        [--mode task|lifetime] [--trials 32] [--n_envs 6] [--iters 2]

If idea 3 fails to return to a goal, this says which part failed. The GRU
state is recorded at every step on the held-out arenas and ridge readouts are
fit on half the trajectories and scored on the other half:

- ``goal_phase``: the goal cell's grid phases, (cos, sin) of 2 pi x / lambda
  and 2 pi y / lambda for each module, 12 numbers. Readable after the goal
  was found but not before means the goal is held.
- ``goal_disp``: the displacement from the agent's cell to the goal, in
  cells, the quantity acting on it needs. Its decoded direction is also
  scored as an angular error.
- ``pos_phase``: the agent's own cell, the same 12 numbers -- a control,
  since it is an input (the grid code) every step.

Each is scored separately on steps after the row first touched this arena's
goal (``known``) and before (``unknown``, the chance level). ``--mode task``
runs evaluate_task's protocol (two visits per arena, state carried) for
``n_groups`` goals per arena; ``--mode lifetime`` runs ``n_groups`` lifetimes
over ``n_envs`` arenas (rollout/lifetime.py), each with fresh goals, and also
probes the goals of the arenas found earlier that the agent is NOT in
(``other_goal_phase_r2``): is more than one goal held at once?

Goals are redrawn for every group and the readouts are trained on even groups
and scored on odd ones, so a readout cannot get the goal from which arena the
agent is in. With one fixed goal per arena it could (the arena is readable
from the grid code), and the "unknown" steps then decode the goal too.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint, load_agent
from hopfield_nav.memory import backend as memory_backend
from hopfield_nav.policy import channels
from hopfield_nav.rollout import lifetime
from hopfield_nav.rollout.collector import RolloutCollector
from hopfield_nav.world.generate import build_envs
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import WorldSpec


class Recorder(torch.nn.Module):
    """The agent, with the hidden state of every rollout step kept."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.hs: list[torch.Tensor] = []

    @property
    def rnn(self):
        return self.agent.rnn

    def forward(self, *a, **k):
        return self.agent(*a, **k)

    def get_action_and_value(self, *a, **k):
        r = self.agent.get_action_and_value(*a, **k)
        self.hs.append(r["h_next"][-1].detach().cpu())
        return r


def phases(gx, gy, lambdas):
    """(N, 4 * len(lambdas)) cos/sin of each module's x and y phase."""
    out = []
    for l in lambdas:
        for g in (gx, gy):
            a = 2 * np.pi * (np.asarray(g) % l) / l
            out += [np.cos(a), np.sin(a)]
    return np.stack(out, -1)


def cells_from_code(code, lambdas, npos):
    """Global cells from grid codes: each module's peak gives its phase, and
    the phases pin the cell modulo prod(lambdas) (= npos on this scaffold)."""
    x_ph, y_ph, o = [], [], 0
    for l in lambdas:
        blk = code[:, o:o + l * l].reshape(-1, l, l)       # [x mod l, y mod l]
        flat = blk.reshape(len(code), -1).argmax(1)
        x_ph.append(flat // l)
        y_ph.append(flat % l)
        o += l * l
    xs = np.arange(npos)
    out = np.zeros((len(code), 2), dtype=np.int64)
    for axis, ph in ((0, x_ph), (1, y_ph)):
        ok = np.ones((len(code), npos), dtype=bool)
        for l, p in zip(lambdas, ph):
            ok &= (xs[None, :] % l) == p[:, None]
        out[:, axis] = ok.argmax(1)
    return out


def ridge_fit_score(Xtr, Ytr, Xte, Yte, alphas=(1e-2, 1e-1, 1, 10, 100)):
    """Ridge with the alpha picked on a split of the training rows; returns
    (test R^2 pooled over target dims, prediction on Xte)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    n = len(A)
    cut = int(0.8 * n)

    def fit(X, Y, a):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        reg = a * np.eye(X1.shape[1])
        reg[-1, -1] = 0
        return np.linalg.solve(X1.T @ X1 + reg, X1.T @ Y)

    def pred(W, X):
        return np.hstack([X, np.ones((len(X), 1))]) @ W

    best = min(alphas, key=lambda a: ((pred(fit(A[:cut], Ytr[:cut], a), A[cut:])
                                       - Ytr[cut:]) ** 2).sum())
    W = fit(A, Ytr, best)
    P = pred(W, B)
    r2 = 1 - ((P - Yte) ** 2).sum() / ((Yte - ym) ** 2).sum()
    return float(r2), P


def collect(agent, cfg, vh, envs, offsets, mode, trials, max_steps, n_envs,
            iters, seed, device, n_groups):
    """Per-step records over ``n_groups`` goal draws: hidden state, own cell,
    goal cell, arena, known flag, group, and the goal cells of the arenas this
    row found earlier."""
    rng = np.random.RandomState(seed + 7919)
    parts = []
    for grp in range(n_groups):
        for e in envs:
            e.set_goal(tuple(int(v) for v in rng.randint(0, e.size, 2)))
        d = _collect_one(agent, cfg, vh, envs, offsets, mode, trials, max_steps,
                         n_envs, iters, seed + grp, device)
        d["group"] = np.full(len(d["H"]), grp)
        d["goal"] = d["goal_global"][d["arena"]]
        d["found_goal"] = np.where(d["found_before"][..., None],
                                   d["goal_global"][None], -1)
        parts.append(d)
    keys = ("H", "cell", "arena", "known", "row", "found_before", "act", "group",
            "goal", "found_goal")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["lambdas"] = parts[0]["lambdas"]
    return out


def _collect_one(agent, cfg, vh, envs, offsets, mode, trials, max_steps, n_envs,
                 iters, seed, device):
    ecfg = copy.deepcopy(cfg)
    ecfg.batch_envs = int(trials)
    ecfg.steps_per_rollout = int(max_steps)
    ecfg.obs_dropout = 0.0
    ecfg.hopfield.heading_dropout = 0.0
    embed_dim = vh.encoded_Phi.shape[2]
    rec = Recorder(agent).to(device)
    rec.eval()
    collector = RolloutCollector(vh, ecfg, embed_dim, device)
    specs = channels.channel_specs(cfg.agent, embed_dim, cfg.env.observation_size)
    off = np.cumsum([0] + [s.width for s in specs])
    gsl = slice(off[[s.name for s in specs].index("grid_code")],
                off[[s.name for s in specs].index("grid_code") + 1])
    lambdas = list(cfg.vectorhash.lambdas)
    npos = cfg.vectorhash.Npos or int(np.prod(lambdas))
    torch.manual_seed(seed)
    np.random.seed(seed)

    runs = []   # (rollout, arena index, hs (B, T, H))
    if mode == "task":
        for j, env in enumerate(envs):
            h, fired = None, None
            env.goals_active = True
            for _v in range(2):
                rec.hs.clear()
                r = collector.collect_rollout(
                    env, rec, [None] * trials, allow_store=True, h_rnn=h,
                    env_offset=offsets[j], task_mode=True, store_fired_init=fired)
                runs.append((r, j, torch.stack(rec.hs, 1).numpy(), fired))
                h, fired = r.h_final, r.store_fired_final
    else:
        rec.hs.clear()
        rs, order = [], lifetime.protocol_order(n_envs, iters)
        fired = [None] * n_envs
        h = None
        for j in order:
            envs[j].goals_active = True
            rec.hs.clear()
            r = collector.collect_rollout(
                envs[j], rec, [None] * trials, allow_store=True, h_rnn=h,
                env_offset=offsets[j], ends_on_goal=True, task_mode=True,
                store_fired_init=fired[j], freeze_done_state=True)
            runs.append((r, j, torch.stack(rec.hs, 1).numpy(), fired[j]))
            h, fired[j] = r.h_final, r.store_fired_final

    goal_global = [np.array(e.goal_location) + np.array(o) for e, o in zip(envs, offsets)]
    H, cell, arena, known, row, found_before, act = [], [], [], [], [], [], []
    found_so_far = np.zeros((trials, len(envs)), dtype=bool)
    for r, j, hs, fired0 in runs:
        g = r.obs[..., gsl].cpu().numpy()                     # (B, T, Ng)
        reached = r.goal_reached.cpu().numpy() > 0
        alive = (r.alive_mask.cpu().numpy() > 0 if r.alive_mask is not None
                 else np.ones(reached.shape, bool))
        B, T = reached.shape
        k0 = np.zeros(B, bool) if fired0 is None else np.asarray(fired0, bool)
        # known at step t: touched before t (this visit) or on an earlier one
        touched_before = np.concatenate(
            [np.zeros((B, 1), bool), np.cumsum(reached, 1)[:, :-1] > 0], 1)
        kn = k0[:, None] | touched_before
        for b in range(B):
            m = alive[b]
            n = int(m.sum())
            if n == 0:
                continue
            H.append(hs[b, :T][m])
            act.append(r.move_actions[b].cpu().numpy()[m])
            cell.append(cells_from_code(g[b][m], lambdas, npos))
            arena.append(np.full(n, j))
            known.append(kn[b][m])
            row.append(np.full(n, b))
            fb = np.repeat(found_so_far[b][None], n, 0)
            fb[:, j] = False
            found_before.append(fb)
        if mode == "lifetime":
            # Task mode resets the state between arenas, so only a lifetime
            # can hold another arena's goal.
            found_so_far[:, j] |= reached.any(1) | k0
    return dict(H=np.concatenate(H), cell=np.concatenate(cell),
                arena=np.concatenate(arena), known=np.concatenate(known),
                row=np.concatenate(row), found_before=np.concatenate(found_before),
                act=np.concatenate(act),
                goal_global=np.stack(goal_global), lambdas=lambdas)


def score(d):
    lam = d["lambdas"]
    goal = d["goal"]
    disp = goal - d["cell"]
    targets = {
        "goal_phase": phases(goal[:, 0], goal[:, 1], lam),
        "goal_disp": disp.astype(np.float64),
        "pos_phase": phases(d["cell"][:, 0], d["cell"][:, 1], lam),
    }
    # Held-out goals: train on even goal draws, score on odd ones.
    train = d["group"] % 2 == 0
    test = ~train
    out = {}
    for split, m in (("known", d["known"]), ("unknown", ~d["known"])):
        tr, te = train & m, test & m
        if tr.sum() < 50 or te.sum() < 50:
            continue
        res = {"n_test": int(te.sum())}
        for name, Y in targets.items():
            r2, P = ridge_fit_score(d["H"][tr], Y[tr], d["H"][te], Y[te])
            res[name + "_r2"] = r2
            if name == "goal_disp":
                far = np.linalg.norm(Y[te], axis=1) >= 2
                ang = np.arctan2(P[far, 1], P[far, 0]) - np.arctan2(Y[te][far, 1], Y[te][far, 0])
                res["goal_dir_err_deg"] = float(np.degrees(
                    np.abs(np.angle(np.exp(1j * ang)))).mean())
                # Does the agent move the way the goal lies, and the way its
                # own state says it lies? Mean cosine of the action with the
                # true and with the decoded direction, goals 2+ cells away.
                a = d["act"][te][far]
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
                for tag, v in (("true", Y[te][far]), ("decoded", P[far])):
                    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
                    res[f"action_cos_{tag}_dir"] = float((an * vn).sum(1).mean())
        out[split] = res
    # Other arenas' goals while in this one (lifetime mode): one readout per
    # arena k, on steps where the row found k's goal earlier and is now in
    # another arena; R^2 pooled over arenas, on held-out lifetimes.
    fb = d["found_before"]
    if fb.any():
        res = {}
        for k in range(fb.shape[1]):
            m = fb[:, k]
            tr, te = train & m, test & m
            if tr.sum() < 50 or te.sum() < 50:
                continue
            gk = d["found_goal"][:, k]
            Y = phases(gk[:, 0], gk[:, 1], lam)
            r2, _ = ridge_fit_score(d["H"][tr], Y[tr], d["H"][te], Y[te])
            res[f"arena{k}"] = r2
        if res:
            out["other_goal_phase_r2"] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--world", default=None, help="world.json (default: the ckpt's run dir)")
    p.add_argument("--mode", choices=["task", "lifetime"], default="task")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=6)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--n_groups", type=int, default=8,
                   help="goal draws (task: per arena; lifetime: lifetimes); "
                        "readouts train on even draws and score on odd ones")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if not memory_backend.is_none(cfg):
        raise SystemExit("goal_probe is for idea 3 (memory_backend none)")
    world = a.world or os.path.join(os.path.dirname(a.ckpt), "world.json")
    specs = WorldSpec.read(world).split.base_val
    offsets = [tuple(s.offset) for s in specs]
    encoder, enc_cfg, enc_gain = load_encoder(cfg.encoder_checkpoint, str(device))
    torch.manual_seed(0)
    np.random.seed(0)
    vh = VectorHash(cfg.vectorhash)
    vh.build_scaffold()
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(enc_gain)
    agent = load_agent(cfg, ck["agent_state_dict"], enc_cfg.out_dim, device)
    envs = build_envs(specs, cfg.env, cfg.agent.movement_mode)
    d = collect(agent, cfg, vh, envs, offsets, a.mode, a.trials, a.max_steps,
                min(a.n_envs, len(envs)), a.iters, a.seed, device, a.n_groups)
    res = {"ckpt": a.ckpt, "mode": a.mode, "n_steps": int(len(d["H"])),
           "known_frac": float(d["known"].mean()), **score(d)}
    head = getattr(agent, "goal_disp_head", None)
    if head is not None and getattr(agent, "head_mlp", None) is None:
        # The trained displacement head itself, on these held-out arenas (it
        # reads the top recurrent layer's output, which is what H records).
        # Unlike the ridge readouts it is not refit on 6 arenas.
        with torch.no_grad():
            P = head(torch.from_numpy(d["H"]).float().to(device)).cpu().numpy() * 10.0
        Y = (d["goal"] - d["cell"]).astype(np.float64)
        m = d["known"]
        far = m & (np.linalg.norm(Y, axis=1) >= 2)
        r2 = 1 - ((P[m] - Y[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum()
        ang = np.arctan2(P[far, 1], P[far, 0]) - np.arctan2(Y[far, 1], Y[far, 0])
        res["trained_disp_head"] = {
            "r2_known": float(r2),
            "dir_err_deg_known": float(np.degrees(np.abs(np.angle(np.exp(1j * ang)))).mean())}
    print(json.dumps(res, indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
