"""Enumerate the quadrant table for a saved goal-pairs or goal-lifetimes checkpoint.

    python -m hopfield_nav.eval_goal_pairs --ckpt <run_dir>/pairs_u6000.pt
    python -m hopfield_nav.eval_goal_pairs --ckpt <run_dir>/life_final.pt --by_distance

Rebuilds the run's world from the checkpoint's argv (same seed, same split,
same corner if any), loads the model -- a `PairRegressor` from
`train_goal_pairs` or an `RNNAgent` from `train_goal_lifetimes`, detected by
which state-dict key is present -- and writes `<ckpt stem>_tables.json`
beside it with every pair in every quadrant enumerated.

`--by_distance` adds, per env set, the mean score binned by Chebyshev
|g - p| over the enumerated train x train quadrant (plan D1): where a
model's error lives, not just how much of it there is.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import EnvConfig, RNNTrainConfig
from .evaluation.goal_pairs import (
    RNNAgentAsPairModel, aggregate_by_distance, evaluate_pairs_by_distance, format_table)
from .policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from .policy.pair_regressor import PairRegressor
from .training.goal_pairs_setup import ARMS, agent_cfg_for_mode, build_env_sets, eval_all, jsonable


def load_model(ck: dict, acfg, D: int, device):
    """Either trainer's checkpoint -> something with predict_direction/logits."""
    a = ck["argv"]
    if "model_state_dict" in ck:
        m = PairRegressor(D, a["hidden_size"], a["num_layers"], a["movement_mode"],
                          nonlinearity=a["nonlinearity"], dropout=a.get("dropout", 0.0)).to(device)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        return m
    agent = RNNAgent(acfg, D).to(device)
    agent.load_state_dict(ck["agent_state_dict"]); agent.eval()
    return RNNAgentAsPairModel(agent)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--by_distance", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["argv"]
    is_lifetimes = "agent_state_dict" in ck
    extra = {}
    if is_lifetimes:
        extra = dict(ARMS[a["arm"]], init_log_std=a.get("init_log_std", 0.0))
        nonlin = a["nonlinearity"] if extra["rnn_cell"] == "mlp" else "tanh"
    else:
        extra = dict(rnn_cell="mlp", dropout=a.get("dropout", 0.0))
        nonlin = a["nonlinearity"]
    acfg = agent_cfg_for_mode(a["mode"], a["movement_mode"],
                              hidden_size=a["hidden_size"], num_rnn_layers=a["num_layers"],
                              rnn_nonlinearity=nonlin, **extra)
    cfg = RNNTrainConfig(
        env=EnvConfig(size=a["size"], observation_size=a["observation_size"],
                      movement_mode=a["movement_mode"], wall_resolution=a["wall_resolution"]),
        agent=acfg, n_envs=a["n_envs"], n_val_envs=a["n_val_envs"], seed=a["seed"],
        fwhm_ratio=a["fwhm_ratio"], lambdas=list(a["lambdas"]), env_generator=True,
        place_margin=a["place_margin"], goal_val_frac=a["goal_val_frac"],
        region_val_frac=a["region_val_frac"], wall_seeds=a["wall_seeds"],
        place_region=a.get("place_region", "anywhere"))
    built = build_env_sets(cfg, np.random.RandomState(a["seed"]), n_same=a["n_same_envs"],
                           n_ood_place=a.get("n_ood_place", 0))
    train, heldout, same, split, vh, _ = built[:6]
    sets = [train, heldout, same] + ([built[6]] if len(built) > 6 else [])
    cells = split.cell_sets()
    D = compute_rnn_input_dim(acfg, a["observation_size"], vh.Ng)
    assert D == ck["input_dim"], (D, ck["input_dim"])
    model = load_model(ck, acfg, D, torch.device(args.device))

    tables = eval_all(model, sets, acfg, cells, a["movement_mode"],
                      torch.device(args.device), n_per_quadrant=None, seed=ck["update"])
    kind = "lifetimes" if is_lifetimes else "pairs"
    print(f"=== ENUMERATED  {os.path.basename(args.ckpt)}  ({kind}, update {ck['update']})")
    for es in [x.name for x in sets]:
        print(format_table(tables[es], a["movement_mode"], title=es))
        print()
    out = {"update": ck["update"], "argv": a, "kind": kind, "tables": jsonable(tables)}

    if args.by_distance:
        out["by_distance"] = {}
        print("=== BY DISTANCE (train x train quadrant, Chebyshev |g - p|)")
        hdr = "d:      " + " ".join(f"{k:5d}" for k in range(1, 20))
        print(hdr)
        for es in sets:
            res = [evaluate_pairs_by_distance(model, t, acfg, cells, movement_mode=a["movement_mode"],
                                              device=torch.device(args.device))
                   for t in es.tensors]
            agg = aggregate_by_distance(res)
            out["by_distance"][es.name] = agg
            print(f"{es.name:12s}" + " ".join(f"{v:5.1f}" for v in agg["mean"]))

    path = os.path.splitext(args.ckpt)[0] + "_tables.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
