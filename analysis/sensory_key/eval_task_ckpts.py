"""Score several checkpoints with ``evaluate_task`` on ONE set of val envs.

    python -m analysis.sensory_key.eval_task_ckpts \
        --world <run>/world.json --seeds 10 --out <json> \
        d0_base=<ckpt> orig=<ckpt> distal=<ckpt> gmlp=<ckpt>

Like-for-like with the in-training ``task=`` eval of the task3r runs: the same
protocol (``evaluate_task``: search, oracle store, keep going; visits = 2,
sampled), 16 trials per env, 200 steps, distractors 0 and 10 -- on the
``base_val`` envs of the given world, rebuilt with each checkpoint's own env
config (so a panorama run sees its panorama and a run trained without one does
not). Several eval seeds per checkpoint, because one eval swings a lot.

The scaffold is built once and shared: every checkpoint must use the same
encoder and scaffold config, which is checked.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint, load_agent
from hopfield_nav.evaluation.metrics import evaluate_task
from hopfield_nav.world.generate import build_envs
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import WorldSpec

KEYS = ["found_rate", "steps_first", "reaches_post", "steps_per_reach",
        "revisit_found", "revisit_steps_first", "revisit_steps_per_reach",
        "cos_aq_pre", "cos_aq_post"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpts", nargs="+", help="name=path")
    p.add_argument("--world", required=True, help="world.json whose base_val envs to use")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--dist", type=int, nargs="+", default=[0, 10])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    specs = WorldSpec.read(a.world).split.base_val
    offsets = [tuple(s.offset) for s in specs]
    loaded = {}
    for item in a.ckpts:
        name, path = item.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        loaded[name] = (path, ck, cfg_from_checkpoint(ck["config"]))

    first = next(iter(loaded.values()))[2]
    for name, (_, _, cfg) in loaded.items():
        for f in ("encoder_checkpoint", "fwhm_ratio"):
            assert getattr(cfg, f) == getattr(first, f), (name, f)
        assert list(cfg.vectorhash.lambdas) == list(first.vectorhash.lambdas), name
        assert cfg.vectorhash.Npos == first.vectorhash.Npos, name

    encoder, enc_cfg, enc_gain = load_encoder(first.encoder_checkpoint, str(device))
    torch.manual_seed(0)
    np.random.seed(0)
    vh = VectorHash(first.vectorhash)
    vh.build_scaffold()
    vh.precompute_encoded_phi(encoder, first.fwhm_ratio, device=str(device))
    print(f"scaffold built: encoded_Phi {vh.encoded_Phi.shape}", flush=True)

    results = {}
    for name, (path, ck, cfg) in loaded.items():
        if cfg.hopfield.beta is None:
            cfg.hopfield.beta = float(enc_gain)
        agent = load_agent(cfg, ck["agent_state_dict"], enc_cfg.out_dim, device)
        envs = build_envs(specs, cfg.env, cfg.agent.movement_mode)
        per_seed = []
        for s in range(a.seeds):
            per_seed.append(evaluate_task(
                agent, envs, vh, offsets, cfg, device, num_trials=a.trials,
                max_steps=a.max_steps, n_distractors_list=list(a.dist), seed=s))
        summary = {}
        for nd in a.dist:
            summary[nd] = {k: (float(np.nanmean([r[nd].get(k, np.nan) for r in per_seed])),
                               float(np.nanstd([r[nd].get(k, np.nan) for r in per_seed])))
                           for k in KEYS}
        results[name] = dict(path=path, per_seed=per_seed, summary=summary,
                             distal_amp=float(getattr(cfg.env, "distal_amp", 0.0)),
                             backend=getattr(cfg.agent, "memory_backend", "hopfield"))
        print(f"\n{name}  ({results[name]['backend']}, distal_amp "
              f"{results[name]['distal_amp']:g}, {a.seeds} eval seeds)", flush=True)
        for nd in a.dist:
            print(f"  d={nd:2d}  " + "  ".join(
                f"{k}={summary[nd][k][0]:.3f}±{summary[nd][k][1]:.3f}" for k in KEYS), flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
