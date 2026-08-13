"""How much drive does each input channel actually deliver to the trunk?

The gate this project needs is a function of ‖q‖, and `q` arrives as 8 of 70
input dimensions at magnitudes around 0.05-0.45, against 60 sensory dimensions
at ±1. At initialization that is a ~10x difference in contribution to the RNN's
pre-activation, which is a concrete reason the gate could be slow to form and
is not addressable by any training knob.

This reads it off a checkpoint. For each channel group it reports:

    w_rms        RMS of that group's input weights, against the init scale
    w_vs_init    the same, as a ratio -- >1 means training GREW this channel
    drive        w_rms * (typical magnitude of the channel) * sqrt(width),
                 i.e. the group's share of the pre-activation
    drive_frac   that share, normalized across groups

`drive_frac` is the number to read. If the Hopfield channels sit at a percent
or two after training, the policy is barely reading the thing it has to gate
on, and a gain on those channels is worth trying. If training has already grown
them, the dilution argument is answered and the problem is elsewhere.

No scaffold and no GPU: this needs the checkpoint and nothing else.

    python -m hopfield_nav.probes.input_gain --ckpt <navigate_uN.pt> ...
"""
from __future__ import annotations

import argparse
import json

import torch

from ..evaluation.checkpoint_io import cfg_from_checkpoint
from ..policy import channels

# Typical |value| per dimension of each channel, for turning a weight norm into
# an actual contribution. The Hopfield figures are the medians measured by
# `hopfield_separability` (goal ~0.28, distractors ~0.14, so ~0.2 either way);
# sensory is a +-1 barcode; the reward channels are constant under
# explore_goals_off and carry a time penalty otherwise.
TYPICAL = {
    "current_reward": 0.05,
    "prev_reward": 0.05,
    "hopfield_signal": 0.20,
    "prev_action": 1.0,
    "sensory": 1.0,
    "goal_in_memory": 1.0,
    "encoded_state": 0.03,
}


def _typical(name: str) -> float:
    if name.startswith("hopfield_multistep"):
        return TYPICAL["hopfield_signal"]
    return TYPICAL.get(name, 1.0)


def report(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    sd = ck["agent_state_dict"]

    w = sd.get("rnn.weight_ih_l0")
    if w is None:                       # softplus cell subclasses nn.RNN too
        raise KeyError(f"no rnn.weight_ih_l0 in {ckpt_path}; "
                       f"have {sorted(k for k in sd if k.startswith('rnn'))}")
    hidden = cfg.agent.hidden_size
    # A GRU stacks three gates into weight_ih, so its row count is 3*hidden.
    # Only the column blocks matter here, so this is reported, not corrected
    # for -- an RMS over 3x as many rows of the same distribution is the same
    # number.
    gates = w.shape[0] // hidden

    # `embed_dim` only enters when input_encoded_state is on, which no run in
    # this line uses. Solve for it from the width the checkpoint was actually
    # built with rather than rebuilding the scaffold to ask.
    specs = channels.channel_specs(cfg.agent, 0, cfg.env.observation_size)
    missing = int(w.shape[1]) - sum(s.width for s in specs)
    if missing:
        specs = channels.channel_specs(cfg.agent, missing,
                                       cfg.env.observation_size)
    if sum(s.width for s in specs) != w.shape[1]:
        raise ValueError(
            f"channel layout sums to {sum(s.width for s in specs)} but the "
            f"checkpoint's input is {w.shape[1]} wide; the config and the "
            f"weights disagree about what the policy reads")

    init_scale = hidden ** -0.5
    rows, offset = [], 0
    for spec in specs:
        block = w[:, offset:offset + spec.width]
        w_rms = float(block.pow(2).mean().sqrt())
        typ = _typical(spec.name)
        rows.append({
            "channel": spec.name,
            "width": spec.width,
            "w_rms": w_rms,
            "w_vs_init": w_rms / init_scale,
            # Contribution to a pre-activation: independent-ish terms add in
            # quadrature, so width enters as sqrt.
            "drive": w_rms * typ * (spec.width ** 0.5),
            "typical_value": typ,
        })
        offset += spec.width

    tot = sum(r["drive"] for r in rows) or 1.0
    for r in rows:
        r["drive_frac"] = r["drive"] / tot

    hop = [r for r in rows if r["channel"].startswith("hopfield")]
    return {
        "ckpt": ckpt_path,
        "hidden_size": hidden,
        "gates_in_weight_ih": gates,
        "input_dim": int(w.shape[1]),
        "channels": rows,
        "hopfield_drive_frac": sum(r["drive_frac"] for r in hop),
        "hopfield_w_vs_init": (sum(r["w_vs_init"] for r in hop) / len(hop)
                               if hop else float("nan")),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    out = []
    for path in args.ckpt:
        r = report(path)
        out.append(r)
        print(f"\n=== {path}  (hidden={r['hidden_size']}, "
              f"input_dim={r['input_dim']}) ===")
        print(f"{'channel':<22} {'width':>5} {'w/init':>8} {'drive':>8} "
              f"{'share':>7}")
        for c in r["channels"]:
            print(f"{c['channel']:<22} {c['width']:>5} {c['w_vs_init']:>8.3f} "
                  f"{c['drive']:>8.4f} {c['drive_frac']:>6.1%}")
        print(f"{'HOPFIELD TOTAL':<22} {'':>5} "
              f"{r['hopfield_w_vs_init']:>8.3f} {'':>8} "
              f"{r['hopfield_drive_frac']:>6.1%}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
