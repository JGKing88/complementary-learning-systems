"""Initialize an encoder from a given config and save it untrained.

The saved checkpoint is compatible with `encoder_training.train.load_encoder`.

Examples:
    python -m encoder_training.save_untrained_encoder \
        --encoder-type mlp --out-dim 256 --hidden-dim 1024 \
        --num-hidden-layers 4 --gain 5.0 \
        --out /home/jackking/cls/encoders/untrained_mlp.pt

    # Or load the model config from a JSON file:
    python -m encoder_training.save_untrained_encoder \
        --config my_model_cfg.json --out untrained.pt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from encoder_training.config import EncoderModelConfig
from encoder_training.models import create_encoder


def build_model_config(args: argparse.Namespace) -> EncoderModelConfig:
    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        valid = set(EncoderModelConfig.__dataclass_fields__.keys())
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
        return EncoderModelConfig(**cfg_dict)

    kwargs = {}
    for name in EncoderModelConfig.__dataclass_fields__:
        val = getattr(args, name, None)
        if val is not None:
            kwargs[name] = val
    return EncoderModelConfig(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output .pt path")
    p.add_argument("--config", default=None,
                   help="Optional JSON file with EncoderModelConfig fields")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    # EncoderModelConfig overrides (all optional; unset → dataclass default)
    p.add_argument("--encoder-type", dest="encoder_type", choices=["mlp", "cnn"])
    p.add_argument("--lambdas", type=int, nargs="+")
    p.add_argument("--out-dim", dest="out_dim", type=int)
    p.add_argument("--nonlinearity")
    p.add_argument("--output-nonlinearity", dest="output_nonlinearity")
    p.add_argument("--gain", type=float)
    p.add_argument("--hidden-dim", dest="hidden_dim", type=int)
    p.add_argument("--num-hidden-layers", dest="num_hidden_layers", type=int)
    p.add_argument("--hidden-channels", dest="hidden_channels", type=int)
    p.add_argument("--num-conv-layers", dest="num_conv_layers", type=int)
    p.add_argument("--kernel-size", dest="kernel_size", type=int)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    mcfg = build_model_config(args)
    encoder = create_encoder(mcfg, device=args.device)

    torch.save({
        "state_dict": encoder.state_dict(),
        "model_config": asdict(mcfg),
        "gain": float(mcfg.gain),
        "untrained": True,
        "seed": args.seed,
    }, args.out)

    print(f"Saved untrained encoder → {args.out}")
    print(f"  config: {asdict(mcfg)}")


if __name__ == "__main__":
    main()
