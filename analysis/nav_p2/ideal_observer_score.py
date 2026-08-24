"""P3 §7.3 item 5 -- score a trained policy's OWN trajectories.

The probe study (`ideal_observer.py`) answers "how much regime information is
available if you move like *this*". This answers the question P6 actually needs:
**is the agent collecting it?** A classifier is fitted on the scripted-probe
data, frozen, and applied to trajectories the trained policies generate
themselves. The gap between a policy's AUC and the best probe's AUC is the
information its behaviour is leaving on the table.

The features are computed by `io_features.StepFeatures`, the same object the
probe pass drives -- if the two used separate implementations the comparison
would be between two statistics rather than between two behaviours.

**Protocol, and why it is the way it is.** Both regimes are run with
`goals_active = False`: no arrival, no teleport, a fixed step budget. The
alternative -- letting the exploit regime end on arrival -- makes episode
*length* a function of the label, so "the episode is still running" becomes a
cue and every AUC below would be partly measuring that. Steps standing on the
goal are masked out (their `dir_cos` is undefined) exactly as in the probe
pass, and the valid-row count is printed per checkpoint: an exploit policy that
parks on the goal thins those rows out at late `t`, which is a result about the
policy and not a defect in the measurement.

**The classifier is frozen across envs, not across rows.** It is fitted on the
probe tensor's envs and applied to a *fresh* world drawn with a different seed,
so no env is shared between fitting and scoring.

    python -m analysis.nav_p2.ideal_observer_score \
        --npz results/nav_p2/io_probe.npz \
        --policy_ckpts <ckpt> [<ckpt> ...] --envs 24
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from analysis.nav_p2.ideal_observer_fit import Data, auc, cv_auc
from analysis.nav_p2.io_features import (
    FEATURES, POLICY_GROUPS, StepFeatures, all_cells, cell_tables,
    chart_basis, fit_obs_decoder, frame_self_test,
)
from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, load_agent,
)
from hopfield_nav.policy import channels
from hopfield_nav.rollout import signal as signal_ops
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world.env import raycast_codes
from hopfield_nav.world.vec_env import make_vec


@torch.no_grad()
def rollout(agent, cfg, env, offset, vh, hopfields, starts, steps, device,
            deterministic=False):
    """Run the policy for `steps` steps, recording the trajectory.

    Deliberately a close paraphrase of `evaluation/batched.batched_navigation_
    trials` -- same channel assembly through `channels.build_policy_input`,
    same `signal_ops` calls -- with two differences, both stated in the module
    docstring: no arrival contract, and the caller supplies the per-trial
    Hopfields (which may be goal-absent).

    Returns (positions (T, B, 2) int, actions (T, B, 2), displacements
    (T, B, 2)).
    """
    B = len(hopfields)
    vec = make_vec(env, B, cfg.agent.movement_mode, cfg.env.continuous_scale,
                   continuous_normalize=cfg.env.continuous_normalize,
                   reset=False)
    vec.goals_active = False          # symmetric across regimes -- see docstring
    vec.set_positions(starts)

    specs = channels.channel_specs(cfg.agent, vh.encoded_Phi.shape[2],
                                   cfg.env.observation_size)
    pa_dim = channels.prev_action_width(cfg.agent)
    h_rnn = None
    prev_reward_t = torch.zeros(B, 1, device=device)
    prev_action_t = torch.zeros(B, pa_dim, device=device)
    prev_disp_t = torch.zeros(B, 2, device=device)
    P = np.zeros((steps, B, 2), dtype=np.int32)
    A = np.zeros((steps, B, 2), dtype=np.float64)
    Dsp = np.zeros((steps, B, 2), dtype=np.float64)

    for t in range(steps):
        pos = vec.positions()
        P[t] = pos
        cur_r = np.full(B, -cfg.env.time_penalty, dtype=np.float32)
        emb_np = vh.get_encoded_state(pos, offset)
        emb = torch.from_numpy(emb_np).float().to(device)

        if cfg.agent.input_hopfield_signal:
            sig_t, q, _m, _W = signal_ops.hopfield_signal_at(
                vh, cfg, emb_np, emb, pos, offset, hopfields, False, device,
                emb.shape[1])
            hop_signal = (torch.from_numpy(q.astype(np.float32)).to(device)
                          if (cfg.agent.input_hopfield_raw
                              and cfg.agent.hopfield_mode != "discrete")
                          else sig_t)
        else:
            hop_signal = torch.zeros(B, channels.signal_width(cfg.agent),
                                     device=device)

        values = {
            "current_reward": torch.from_numpy(cur_r).to(device).unsqueeze(1),
            "prev_reward": prev_reward_t,
            "encoded_state": emb,
            "hopfield_signal": hop_signal,
            "prev_action": prev_action_t,
            "prev_displacement": prev_disp_t,
            "goal_in_memory": torch.ones(B, 1, device=device),
        }
        if cfg.agent.input_sensory:
            values["sensory"] = torch.from_numpy(
                vec.obs_batch()).float().to(device)
        if (cfg.agent.input_hopfield_multistep
                and cfg.agent.hopfield_mode == "continuous"):
            Wg = vh.gram_schmidt_projection(pos, offset)
            for s, q_s in signal_ops.multistep_q(
                    vh, cfg, emb_np, emb, hopfields, False, Wg,
                    cfg.agent.input_hopfield_multistep, emb.shape[1],
                    device).items():
                values[channels.multistep_name(s)] = torch.from_numpy(
                    q_s.astype(np.float32)).to(device)

        rnn_in = channels.build_policy_input(specs, values,
                                             batch_size=B).unsqueeze(1)
        res = agent.get_action_and_value(rnn_in, h_rnn,
                                         deterministic=deterministic)
        h_rnn = res["h_next"]
        if cfg.agent.movement_mode == "discrete":
            act = res["move_action"].cpu().numpy().astype(int)
            prev_action_t = torch.nn.functional.one_hot(
                res["move_action"].long().view(-1), num_classes=4).float()
            A[t] = 0.0
        else:
            act = res["move_action"].cpu().numpy()
            prev_action_t = res["move_action"].float().view(B, -1)
            A[t] = act
        prev_reward_t = torch.from_numpy(cur_r).to(device).unsqueeze(1)
        vec.step_batch(act)
        Dsp[t] = vec.last_displacement()
        prev_disp_t = torch.from_numpy(Dsp[t]).float().to(device)
    return P, A, Dsp


def _fit_frozen(D: Data, probe: str, level, target, ti, cols):
    """Fit one classifier on the probe tensor and return a scorer."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y, env, _ = D.slice(target, ti, level=level, probe=probe)
    if len(y) < 40 or y.sum() in (0, len(y)):
        return None
    Xc = X[:, cols]
    mu, sd = Xc.mean(0), Xc.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    m = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.12,
                                       max_leaf_nodes=15, early_stopping=False,
                                       l2_regularization=1.0, random_state=0)
    m.fit(np.clip((Xc - mu) / sd, -8, 8), y)
    return lambda Z: m.predict_proba(
        np.clip((Z[:, cols] - mu) / sd, -8, 8))[:, 1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="probe tensor to fit on")
    p.add_argument("--policy_ckpts", nargs="+", required=True)
    p.add_argument("--fit_probe", default="billiard")
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--n_distractors", type=int, nargs="+", default=[0, 1, 3, 10])
    p.add_argument("--targets", nargs="+", default=["ep", "trust"])
    p.add_argument("--t_show", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--obs_ridge", type=float, default=1e-3)
    p.add_argument("--chart_k", type=int, default=64)
    p.add_argument("--seed", type=int, default=101,
                   help="differs from the probe pass so the scored world is "
                        "drawn fresh -- no env is shared with the fit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    D = Data(args.npz)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cols = D.cols(POLICY_GROUPS)
    ckpts = [c for c in D.ck if c in args.t_show and c <= args.steps]
    print(f"probe tensor: {args.npz}   fitting on probe={args.fit_probe}")
    print(f"scoring {len(args.policy_ckpts)} policies on a fresh world "
          f"(seed {args.seed}), {args.envs} envs x {args.draws} draws x "
          f"{args.starts} starts x {len(args.n_distractors)} levels x 2 regimes")

    rows = {}
    for ck_path in args.policy_ckpts:
        tag = os.path.basename(os.path.dirname(ck_path)) + "/" + \
            os.path.basename(ck_path)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = cfg_from_checkpoint(ck["config"])
        cfg.num_val_envs = args.envs
        encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint,
                                              str(device))
        if cfg.hopfield.beta is None:
            cfg.hopfield.beta = float(gain)
        Dd = enc_cfg.out_dim
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        envs, vh, offsets = build_eval_world(cfg, encoder, str(device),
                                             ckpt_path=None)
        agent = load_agent(cfg, ck["agent_state_dict"], Dd, device)
        size = envs[0].size
        cells = all_cells(size)
        frame_self_test(vh, envs[0], offsets[0], cells)

        rng = np.random.RandomState(args.seed + 1)
        srng = np.random.RandomState(args.seed + 13)
        # (checkpoint, level) -> lists
        acc = {}
        t0 = time.time()
        for ei, env in enumerate(envs):
            off, goal = offsets[ei], env.goal_location
            emb_np = vh.get_encoded_state(cells, off)
            emb_t = torch.from_numpy(emb_np).float().to(device)
            Wb = vh.gram_schmidt_projection(cells, off)
            goal_vec = (np.asarray(goal, float)[None, :]
                        - cells.astype(float))
            goal_dist = np.linalg.norm(goal_vec, axis=1)
            obs_N = raycast_codes(env._wall_code, size, cells[:, 0],
                                  cells[:, 1], np.zeros(len(cells)),
                                  cfg.env.observation_size,
                                  env.wall_resolution)
            A_dec, b_dec, _ = fit_obs_decoder(emb_np, obs_N, args.obs_ridge,
                                              rng)
            ops = emb_np @ A_dec + b_dec
            basis = chart_basis(emb_np, args.chart_k)
            g_pat = goal_encoding(vh, off, goal)

            for n_d in args.n_distractors:
                hops, tabs, reg_of = [], [], []
                for di in range(args.draws):
                    d_pats = (sample_distractors(vh, off, size, n_d, rng)
                              if n_d > 0 else [])
                    for reg in (0, 1):
                        hop = Hopfield(Dd, beta=cfg.hopfield.beta,
                                       device=str(device))
                        pats = ([g_pat] + list(d_pats)) if reg else list(d_pats)
                        for j in rng.permutation(len(pats)):
                            hop.input_memory(torch.from_numpy(pats[j]).float())
                        tb = cell_tables(vh, hop, emb_np, emb_t, Wb, g_pat,
                                         d_pats, goal_vec, A_dec, b_dec, ops,
                                         basis)
                        for _ in range(args.starts):
                            hops.append(hop)
                            tabs.append(tb)
                            reg_of.append(reg)
                B = len(hops)
                okc = np.where(goal_dist > 0)[0]
                c0 = okc[srng.randint(0, len(okc), size=B)]
                starts = [tuple(cells[c]) for c in c0]
                P, Aa, Dsp = rollout(agent, cfg, env, off, vh, hops, starts,
                                     args.steps, device, args.deterministic)
                idx = P[..., 0] * size + P[..., 1]        # (T, B)
                sf = StepFeatures(B)
                b_ar = np.arange(B)
                reg_ar = np.asarray(reg_of)
                for t in range(args.steps):
                    c = idx[t]
                    q1 = np.stack([tabs[b]["q1"][c[b]] for b in b_ar])
                    q2 = np.stack([tabs[b]["q2"][c[b]] for b in b_ar])
                    q3 = np.stack([tabs[b]["q3"][c[b]] for b in b_ar])
                    dv1 = np.array([tabs[b]["d1_valid1"][c[b]] for b in b_ar])
                    dv3 = np.array([tabs[b]["d1_valid3"][c[b]] for b in b_ar])
                    dsc = np.array([tabs[b]["d1_selfcos"][c[b]] for b in b_ar])
                    dch = np.array([tabs[b]["d1_chart"][c[b]] for b in b_ar])
                    f = sf.observe(q1, q2, q3, dv1, dv3, dsc, dch)
                    if (t + 1) in ckpts:
                        dcv = np.array([tabs[b]["dir_cos"][c[b]]
                                        for b in b_ar])
                        cg = np.array([tabs[b]["cos_goal"][c[b]] for b in b_ar])
                        cd = np.array([tabs[b]["cos_dmax"][c[b]]
                                       for b in b_ar])
                        v = goal_dist[c] >= cfg.env.goal_radius
                        key = (t + 1, n_d)
                        a = acc.setdefault(key, dict(X=[], ep=[], trust=[],
                                                     step=[], v=[], vt=[],
                                                     env=[], plen=[], gd=[]))
                        a["X"].append(f.copy())
                        a["ep"].append(reg_ar.copy())
                        a["trust"].append((dcv >= D.trust_thresh).astype(int))
                        a["step"].append(((cg >= 0.9) & (cg >= cd)).astype(int))
                        a["v"].append(v)
                        a["vt"].append(v & np.isfinite(dcv))
                        a["env"].append(np.full(B, ei))
                        a["plen"].append(sf.path_len.copy())
                        a["gd"].append(goal_dist[c].copy())
                    sf.act(Aa[t], Dsp[t])
            if (ei + 1) % 4 == 0 or ei + 1 == len(envs):
                el = time.time() - t0
                print(f"    env {ei + 1}/{len(envs)}  {el / (ei + 1):.1f} s/env",
                      flush=True)

        print(f"\n  === {tag}")
        print("  Three numbers per cell, and all three are needed:")
        print("    frozen -- the probe-fitted classifier applied as-is. Low "
              "here means EITHER the behaviour destroys the signal OR the "
              "trajectory is off the distribution the classifier was fitted "
              "on; frozen alone cannot tell those apart.")
        print("    refit  -- the same feature set refitted on the agent's own "
              "rows, cross-validated over held-out envs. This is whether the "
              "information is THERE in the agent's trajectory.")
        print("    probe  -- billiard at the same (level, t), for reference.")
        for target in args.targets:
            print(f"\n  --- Q_{target}")
            print(f"  {'n_d':>4} {'arm':>7} " + " ".join(f"t={c:<6}" for c in ckpts))
            for n_d in args.n_distractors:
                froz, refit, ref, ns, gds = [], [], [], [], []
                for c in ckpts:
                    key = (c, n_d)
                    ti = list(D.ck).index(c)
                    if key not in acc:
                        for L in (froz, refit, ref):
                            L.append(float("nan"))
                        ns.append(0); gds.append(float("nan")); continue
                    a = acc[key]
                    X = np.concatenate(a["X"])
                    y = np.concatenate(a[target])
                    v = np.concatenate(a["vt" if target == "trust" else "v"])
                    env_a = np.concatenate(a["env"])
                    gd = np.concatenate(a["gd"])
                    ns.append(int(v.sum()))
                    gds.append(float(gd[v].mean()) if v.any() else float("nan"))
                    if v.sum() < 40 or y[v].sum() in (0, int(v.sum())):
                        for L in (froz, refit, ref):
                            L.append(float("nan"))
                        continue
                    sc = _fit_frozen(D, args.fit_probe, n_d, target, ti, cols)
                    froz.append(auc(y[v], sc(X[v])) if sc else float("nan"))
                    r, _ = cv_auc(X[v], y[v], env_a[v], cols,
                                  n_folds=args.folds, model="gbt")
                    refit.append(r)
                    Xp, yp, envp, _ = D.slice(target, ti, level=n_d,
                                              probe=args.fit_probe)
                    rp, _ = cv_auc(Xp, yp, envp, cols, n_folds=args.folds,
                                   model="gbt")
                    ref.append(rp)

                def _row(name, vals):
                    return (f"  {n_d:>4} {name:>7} " + " ".join(
                        ("   --   " if not np.isfinite(x) else f"{x:>7.3f} ")
                        for x in vals))
                print(_row("frozen", froz))
                print(_row("refit", refit))
                print(_row("probe", ref))
                print(f"  {n_d:>4} {'rows':>7} " + " ".join(
                    f"{n:>7d} " for n in ns))
                print(f"  {n_d:>4} {'meanD':>7} " + " ".join(
                    f"{g:>7.2f} " for g in gds)
                    + "   <- mean distance to goal on the scored steps")
                rows[(tag, target, n_d)] = froz
                rows[(tag + "|refit", target, n_d)] = refit
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        np.savez_compressed(
            args.out,
            keys=np.array([f"{a}|{b}|{c}" for a, b, c in rows]),
            vals=np.array(list(rows.values()), dtype=np.float32),
            checkpoints=np.asarray(ckpts))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
