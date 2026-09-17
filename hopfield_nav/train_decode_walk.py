"""Phase 1: the self-taught displacement decode (plan sec 6.4).

    python -m hopfield_nav.train_decode_walk --mode grid --n_envs 64 ...

A1's network and loss, trained with no teacher, no goal and no position:
random walkers in many envs, and for two steps of one walk the input is the
pair of codes and the target is the unit vector of the walker's own
recorded displacement -- odometry. Compared with A1 (the same decode from
teacher-labelled i.i.d. pairs) and with the encoder (proximity
supervision) on one axis: env-steps of experience.

  data     N scattered envs, `walkers` per env, `steps_per_update` discrete
           unit steps each per update, uniform random actions, walls block,
           goals inert (no reward, no teleport). Segments go into a replay
           buffer of the last `buffer_updates` updates.
  pairs    (t, t + k) of one walk, k ~ U[1, k_max], kept if the displacement
           is non-zero and within Chebyshev `max_abs` (A1's 19); with
           `--balance_range`, uniform over that Chebyshev size.
  target   unit(p_{t+k} - p_t), or the nearest of 8 headings (`--target
           heading8`, the label-richness ablation).
  eval     every `eval_every` updates the static quadrant table on the
           training, held-out and `same` envs (A1's readout 1); enumerated
           at the end. Artifacts match `train_goal_pairs` so
           `analysis/decode_probe.py` and `goal_pairs_results.py` read them.

Resumable: `pairs_latest.pt` in the save dir holds model, optimiser,
schedule, update and env-step counters; a requeued job with the same
`--save_dir` continues (the buffer refills from fresh walks).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from .config import EnvConfig, RNNTrainConfig, RNNBCConfig
from .evaluation.goal_pairs import format_table, pair_inputs
from .policy.agent_rnn import compute_rnn_input_dim
from .policy.pair_regressor import PairRegressor
from .training.goal_pairs_setup import (
    MODES, agent_cfg_for_mode, build_env_sets, eval_all, jsonable)
from .training.rnn_setup import write_rnn_world_spec
from .world.vec_env import make_vec

HEADINGS8 = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4], dtype=np.float32)


# ---------------------------------------------------------------------------
# Walks and the replay buffer
# ---------------------------------------------------------------------------

class Walkers:
    """`W` random walkers per training env, stepped `T` at a time. Goals are
    inert (`goals_active` off): no reward, no teleport, a walk is a walk."""

    def __init__(self, envs, walkers: int, seed: int):
        self.vecs = []
        for i, env in enumerate(envs):
            vec = make_vec(env, walkers, "discrete")
            vec.goals_active = False
            vec._rng = np.random.RandomState(seed * 1000 + i)
            vec.reset_all()
            self.vecs.append(vec)
        self.W = walkers
        self.rng = np.random.RandomState(seed + 7)

    def segment(self, T: int) -> np.ndarray:
        """`(n_envs, W, T + 1, 2)` positions: every walker's next `T` steps."""
        out = np.zeros((len(self.vecs), self.W, T + 1, 2), dtype=np.int64)
        for i, vec in enumerate(self.vecs):
            out[i, :, 0] = vec.positions()
            for t in range(T):
                vec.step_batch(self.rng.randint(0, 4, size=self.W))
                out[i, :, t + 1] = vec.positions()
        return out


class Buffer:
    """Each walker's last `history` positions, as one continuous walk.

    Segments are appended in order and every walker's segment continues its
    previous one, so a pair `(t, t + k)` may span segment boundaries and `k`
    may be as long as the history. `sample` draws `k ~ U[1, k_max]`; with
    `balance` it then keeps pairs so that the Chebyshev size of the
    displacement is uniform over `1..max_abs` -- the walker choosing which
    of its own experiences to learn from, since a random walk's own
    displacement distribution is concentrated at a few cells."""

    def __init__(self, n_envs: int, walkers: int, T: int, n_updates: int):
        self.L = n_updates * T + 1
        self.pos = np.zeros((n_envs, walkers, self.L, 2), dtype=np.int64)
        self.n, self.head = 0, 0                    # valid length, next write slot (ring)
        self.T, self.W, self.n_envs = T, walkers, n_envs

    def add(self, seg: np.ndarray) -> None:
        """`seg` is `(n_envs, W, T + 1, 2)`; its first position repeats the last stored one."""
        block = seg[:, :, 1:] if self.n > 0 else seg
        for j in range(block.shape[2]):
            self.pos[:, :, self.head] = block[:, :, j]
            self.head = (self.head + 1) % self.L
            self.n = min(self.n + 1, self.L)

    def _at(self, e, w, i):
        """Position at logical index `i` (0 = oldest valid)."""
        return self.pos[e, w, (self.head - self.n + i) % self.L]

    def sample(self, rng, n: int, k_max: int, max_abs: int, balance: bool = False):
        """`n` pairs: env id, start cell, end cell, displacement. Zero and
        out-of-range displacements are rejected; with `balance`, at most
        `n / max_abs` per Chebyshev size, topped up from the leftovers."""
        k_max = min(k_max, self.n - 1)
        envs, ps, gs = [], [], []
        got = 0
        per_bin = int(np.ceil(n / max_abs))
        counts = np.zeros(max_abs + 1, dtype=np.int64)
        spare_e, spare_p, spare_g = [], [], []
        tries = 0
        while got < n:
            tries += 1
            m = 4 * (n - got) if balance else 2 * (n - got)
            e = rng.randint(0, self.n_envs, size=m)
            w = rng.randint(0, self.W, size=m)
            k = rng.randint(1, k_max + 1, size=m)
            t = np.floor(rng.rand(m) * (self.n - k)).astype(np.int64)
            p = self._at(e, w, t)
            g = self._at(e, w, t + k)
            d = g - p
            r = np.abs(d).max(1)
            ok = (r >= 1) & (r <= max_abs)
            if balance:
                keep = np.zeros(m, dtype=bool)
                for rr in range(1, max_abs + 1):
                    idx = np.where(ok & (r == rr))[0]
                    room = per_bin - counts[rr]
                    if room > 0 and len(idx) > 0:
                        idx = idx[:room]
                        keep[idx] = True
                        counts[rr] += len(idx)
                spare = ok & ~keep
                spare_e.append(e[spare]); spare_p.append(p[spare]); spare_g.append(g[spare])
                ok = keep
                if tries > 50:                      # the walk cannot supply some sizes: top up
                    fill = np.concatenate(spare_e), np.concatenate(spare_p), np.concatenate(spare_g)
                    envs.append(fill[0]); ps.append(fill[1]); gs.append(fill[2])
                    got += len(fill[0])
            envs.append(e[ok]); ps.append(p[ok]); gs.append(g[ok])
            got += int(ok.sum())
        envs = np.concatenate(envs)[:n]
        p = np.concatenate(ps)[:n]
        g = np.concatenate(gs)[:n]
        return envs, p, g, (g - p).astype(np.float32)


def targets_for(d: np.ndarray, kind: str) -> np.ndarray:
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    if kind == "direction":
        return u.astype(np.float32)
    if kind == "heading8":
        return HEADINGS8[np.argmax(u @ HEADINGS8.T, axis=1)]
    raise ValueError(kind)


def batch_inputs(train, acfg, size: int, envs: np.ndarray, p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Model inputs for cell pairs, gathered per env from its precomputed tensors."""
    x = None
    order = np.argsort(envs, kind="stable")
    envs, p, g = envs[order], p[order], g[order]
    pid = p[:, 0] * size + p[:, 1]
    gid = g[:, 0] * size + g[:, 1]
    parts = []
    for e in np.unique(envs):
        m = envs == e
        parts.append(pair_inputs(train.tensors[e], acfg, pid[m], gid[m]))
    x = np.concatenate(parts)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return x[inv]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="grid")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--nonlinearity", choices=["tanh", "relu"], default="relu")
    p.add_argument("--dropout", type=float, default=0.0)
    # Walks
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_val_envs", type=int, default=16)
    p.add_argument("--n_same_envs", type=int, default=4)
    p.add_argument("--walkers", type=int, default=8)
    p.add_argument("--steps_per_update", type=int, default=64)
    p.add_argument("--buffer_updates", type=int, default=20)
    p.add_argument("--k_max", type=int, default=30, help="steps between the two ends of a pair")
    p.add_argument("--max_abs", type=int, default=19, help="Chebyshev range kept (A1's 19)")
    p.add_argument("--pairs_per_update", type=int, default=32768)
    p.add_argument("--target", choices=["direction", "heading8"], default="direction")
    p.add_argument("--range_warmup_updates", type=int, default=0,
                   help="grow the kept Chebyshev range from 19 to max_abs over this many updates "
                        "(size-50 arenas: a 1..49-balanced batch from the start stalls on the 1-cos "
                        "plateau in two of three seeds; short pairs first is the walker's own curriculum)")
    p.add_argument("--balance_range", action=argparse.BooleanOptionalAction, default=False,
                   help="keep training pairs uniform over Chebyshev |Delta| = 1..max_abs (a random "
                        "walk's own displacements are concentrated at a few cells)")
    # World
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=120)
    p.add_argument("--wall_resolution", type=int, default=1)
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--place_margin", type=int, default=20)
    p.add_argument("--goal_val_frac", type=float, default=0.2)
    p.add_argument("--region_val_frac", type=float, default=0.1)
    p.add_argument("--wall_seeds", type=str, default="0,10000000")
    p.add_argument("--place_region", type=str, default="anywhere")
    # Optimisation
    p.add_argument("--n_updates", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_step_at", type=float, default=0.7)
    p.add_argument("--lr_step_gamma", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_pairs", type=int, default=2048)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--final_enumerate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", type=str, default="train_decode_walk")
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    # Kept so `decode_probe` and the results collector can read the argv like a pairs run.
    p.add_argument("--movement_mode", default="continuous", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.movement_mode = "continuous"          # the decode's output is a direction
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    acfg = agent_cfg_for_mode(args.mode, "continuous", hidden_size=args.hidden_size,
                              num_rnn_layers=args.num_layers, rnn_cell="mlp",
                              rnn_nonlinearity=args.nonlinearity, dropout=args.dropout)
    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode="discrete", wall_resolution=args.wall_resolution),
        agent=acfg, bc=RNNBCConfig(lr=args.lr),
        mode="pairs", n_envs=args.n_envs, n_val_envs=args.n_val_envs,
        n_updates=args.n_updates, eval_every=args.eval_every, seed=args.seed,
        device=args.device, use_wandb=args.use_wandb, wandb_project=args.wandb_project,
        fwhm_ratio=args.fwhm_ratio, lambdas=list(args.lambdas),
        env_generator=True, place_margin=args.place_margin,
        goal_val_frac=args.goal_val_frac, region_val_frac=args.region_val_frac,
        wall_seeds=args.wall_seeds, pairs_per_env=0, place_region=args.place_region)
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    train, heldout, same, split, vh, sgb = build_env_sets(cfg, rng, n_same=args.n_same_envs)[:6]
    cells = split.cell_sets()
    print(f"world: {len(train)} train / {len(heldout)} {heldout.name} / {len(same)} same envs; "
          f"place={args.place_region}; Npos={vh.Npos} Ng={vh.Ng}; {time.time()-t0:.1f}s")
    D = compute_rnn_input_dim(acfg, args.observation_size, vh.Ng)
    model = PairRegressor(D, args.hidden_size, args.num_layers, "continuous",
                          nonlinearity=args.nonlinearity, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: mode={args.mode} D={D} hidden={args.hidden_size} layers={args.num_layers} params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.lr_step_at * args.n_updates)], gamma=args.lr_step_gamma)

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={**asdict(cfg), "tag": args.tag,
                                                                    "input_dim": D, "params": n_params})
    if args.save_dir is None:
        sub = run_name(wandb_run.name if wandb_run is not None else None)
        if args.tag:
            sub = f"{args.tag}_{sub}"
        args.save_dir = str(run_dir("goal_pairs", sub))
    else:
        sub = os.path.basename(args.save_dir.rstrip("/"))
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"save_dir={args.save_dir}")

    # Resume: the same save_dir with a `pairs_latest.pt` continues the run.
    u0, env_steps, history = 0, 0, []
    latest = os.path.join(args.save_dir, "pairs_latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        sched.load_state_dict(ck["sched_state_dict"])
        u0, env_steps, history = int(ck["update"]), int(ck["env_steps"]), ck.get("history", [])
        print(f"resumed from update {u0} ({env_steps:,} env-steps)")
    else:
        write_rnn_world_spec(cfg, split, vh, generator="declared", save_dir=args.save_dir)
        run_manifest.begin(args.save_dir, kind="goal_pairs", name=sub,
                           config={**asdict(cfg), "argv": vars(args)}, parent=None, wandb_run=wandb_run)

    sets = [train, heldout, same]
    walkers = Walkers(train.envs, args.walkers, args.seed + 1000 * (u0 + 1))
    buf = Buffer(len(train), args.walkers, args.steps_per_update, args.buffer_updates)
    data_rng = np.random.RandomState(args.seed + 1 + u0)
    steps_per_update = len(train) * args.walkers * args.steps_per_update
    t_train = time.time()

    def save(name: str, u: int) -> str:
        path = os.path.join(args.save_dir, name)
        torch.save({"model_state_dict": model.state_dict(), "opt_state_dict": opt.state_dict(),
                    "sched_state_dict": sched.state_dict(), "cfg": asdict(cfg),
                    "argv": vars(args), "input_dim": D, "update": u, "env_steps": env_steps,
                    "history": history}, path)
        return path

    for u in range(u0 + 1, args.n_updates + 1):
        buf.add(walkers.segment(args.steps_per_update))
        env_steps += steps_per_update
        max_abs_u = args.max_abs
        if args.range_warmup_updates > 0 and u <= args.range_warmup_updates:
            lo = min(19, args.max_abs)
            max_abs_u = int(round(lo + (args.max_abs - lo) * u / args.range_warmup_updates))
        envs, p, g, d = buf.sample(data_rng, args.pairs_per_update, args.k_max, max_abs_u,
                                   balance=args.balance_range)
        x = torch.from_numpy(batch_inputs(train, acfg, args.size, envs, p, g)).to(device)
        y = torch.from_numpy(targets_for(d, args.target)).to(device)
        loss = model.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()

        if u == 1 or u % args.eval_every == 0 or u == args.n_updates:
            tables = eval_all(model, sets, acfg, cells, "continuous", device,
                              n_per_quadrant=args.eval_pairs, seed=u)
            tt = tables["train"][("train", "train")]["model"]["metric"]
            ht = tables[heldout.name][("train", "train")]["model"]["metric"]
            hr = tables[heldout.name][("region", "region")]["model"]["metric"]
            nn_hr = tables[heldout.name][("region", "region")]["nn"]["metric"]
            history.append({"update": u, "env_steps": env_steps, "loss": float(loss.item()),
                            "tables": jsonable(tables)})
            print(f"u={u:5d} steps={env_steps:>11,d} loss={loss.item():.4f} gn={float(gn):.2f} | "
                  f"train tt={tt:.2f} | {heldout.name} tt={ht:.2f} rr={hr:.2f} (nn {nn_hr:.2f}) | "
                  f"{time.time()-t_train:.0f}s", flush=True)
            if wandb_run is not None:
                wandb_run.log({"update": u, "env_steps": env_steps, "train/loss": float(loss.item()),
                               "eval/train_tt": tt, "eval/heldout_tt": ht, "eval/heldout_rr": hr})
        if u % args.ckpt_every == 0:
            save("pairs_latest.pt", u)
            path = save(f"pairs_u{u}.pt", u)
            run_manifest.record_checkpoint(args.save_dir, os.path.basename(path), update=u)

    final = eval_all(model, sets, acfg, cells, "continuous", device,
                     n_per_quadrant=None if args.final_enumerate else args.eval_pairs, seed=args.n_updates)
    print("\n=== FINAL (enumerated)" if args.final_enumerate else "\n=== FINAL (sampled)")
    for es in sets:
        print(format_table(final[es.name], "continuous", title=es.name))
        print()
    # Env-steps to thresholds on the held-out envs, from the eval history.
    reached = {}
    for thr in (10.0, 5.0, 2.0, 1.0):
        hit = [h for h in history if h["tables"][heldout.name]["trainxtrain"]["model"]["metric"] <= thr]
        reached[str(thr)] = (hit[0]["env_steps"], hit[0]["update"]) if hit else None
    print("held-out env-steps to threshold:", reached)
    with open(os.path.join(args.save_dir, "final_tables.json"), "w") as f:
        json.dump({"final": jsonable(final), "history": history, "argv": vars(args),
                   "cells": cells.summary(), "input_dim": D, "params": n_params,
                   "env_steps": env_steps, "steps_to_threshold": reached}, f, indent=1)
    path = save("pairs_final.pt", args.n_updates)
    run_manifest.record_checkpoint(args.save_dir, "pairs_final.pt", update=args.n_updates)
    run_manifest.finish(args.save_dir)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
