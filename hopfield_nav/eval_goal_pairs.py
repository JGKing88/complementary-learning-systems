"""Enumerate the quadrant table for a saved goal-pairs checkpoint.

    python -m hopfield_nav.eval_goal_pairs --ckpt <run_dir>/pairs_u6000.pt

Rebuilds the run's world from its `world.json` config (same seed, same
split), loads the checkpoint, and writes `<ckpt stem>_tables.json` beside it
with every pair in every quadrant enumerated. For picking a checkpoint by
the plan's rule (heldout train x train) and reporting it enumerated rather
than sampled.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import EnvConfig, RNNTrainConfig
from .evaluation.goal_pairs import format_table
from .policy.agent_rnn import compute_rnn_input_dim
from .policy.pair_regressor import PairRegressor
from .training.goal_pairs_setup import agent_cfg_for_mode, build_env_sets, eval_all, jsonable


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["argv"]
    acfg = agent_cfg_for_mode(a["mode"], a["movement_mode"],
                              hidden_size=a["hidden_size"], num_rnn_layers=a["num_layers"],
                              rnn_cell="mlp", rnn_nonlinearity=a["nonlinearity"],
                              dropout=a["dropout"])
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
    model = PairRegressor(D, a["hidden_size"], a["num_layers"], a["movement_mode"],
                          nonlinearity=a["nonlinearity"], dropout=a["dropout"]).to(args.device)
    model.load_state_dict(ck["model_state_dict"])

    tables = eval_all(model, sets, acfg, cells, a["movement_mode"],
                      torch.device(args.device), n_per_quadrant=None, seed=ck["update"])
    print(f"=== ENUMERATED  {os.path.basename(args.ckpt)}  (update {ck['update']})")
    for es in [x.name for x in sets]:
        print(format_table(tables[es], a["movement_mode"], title=es))
        print()
    out = os.path.splitext(args.ckpt)[0] + "_tables.json"
    with open(out, "w") as f:
        json.dump({"update": ck["update"], "argv": a, "tables": jsonable(tables)}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
