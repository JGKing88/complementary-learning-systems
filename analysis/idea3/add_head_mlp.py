"""Put a trainable MLP between a pretrained frozen trunk and the heads.

    python -m analysis.idea3.add_head_mlp --ckpt <pretrained.pt> --layers 2 --out <pt>

With ``--freeze_trunk`` the heads read the trunk directly and are linear, so
the policy cannot compute "follow the goal direction if this arena's goal is
known, else search" (a product of two signals), and a frozen-trunk fork's
revisits got slower as it trained (§7.6). This copies the pretrained trunk and
heads into an agent with ``head_mlp_layers`` set and ``freeze_trunk`` on, the
MLP freshly initialised, and saves it with the matching config, so a fork
(``LOAD_CKPT``) builds the same architecture and loads strictly. The heads now
read the MLP, so their pretrained weights are only a starting point.
"""
from __future__ import annotations

import argparse
import copy

import torch

from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint, load_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.agent.head_mlp_layers = int(a.layers)
    cfg.agent.freeze_trunk = True
    torch.manual_seed(a.seed)
    agent = load_agent(cfg, None, 1024, torch.device("cpu"), eval_mode=False)
    missing, unexpected = agent.load_state_dict(ck["agent_state_dict"], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("head_mlp.") for k in missing), missing
    out = copy.deepcopy(ck)
    out["agent_state_dict"] = agent.state_dict()
    out["config"]["agent"]["head_mlp_layers"] = int(a.layers)
    out["config"]["agent"]["freeze_trunk"] = True
    out["add_head_mlp"] = {"from": a.ckpt, "layers": int(a.layers)}
    torch.save(out, a.out)
    print(f"saved {a.out}: new {sorted(missing)}")


if __name__ == "__main__":
    main()
