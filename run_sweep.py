"""Wandb sweep agent entry point for distance encoder training."""

import os
import sys
import numpy as np
import torch
import wandb

# Ensure project root on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
NOTEBOOKS_DIR = os.path.join(HERE, "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from train_dist_encoder import train

lambdas = [11, 12, 13]

def sweep_train():
    """Single sweep run: read params from wandb.config, build config, train."""
    wandb.init()
    sc = wandb.config

    config = {
        "model_params": {
            "lambdas": lambdas,
            "hidden_dim": sc.get("hidden_dim", 512),
            "hidden_channels": sc.get("hidden_channels", 128),
            "num_layers": sc.get("num_layers", 3),
            "out_dim": sc.get("out_dim", 128),
            "num_hidden_layers": sc.get("num_hidden_layers", 2),
            "kernel_size": sc.get("kernel_size", 5),
            "nonlinearity": "gelu",
            "output_nonlinearity": "tanh",
            "gain": sc.get("gain_end", 5),
            "Npos": sc.get("Npos", 50),
            "Nenv": sc.get("Nenv", 50),
            "encoder_type": sc.get("encoder_type", "cnn"),
            "input_type": sc.get("input_type", "smoothed"),
            "rhc_D": sc.get("rhc_D", 256),
        },
        "training_params": {
            "lr": round(sc.get("lr", 0.0001), 6),
            "batch_size": sc.get("batch_size", 8192),
            "epochs": 300,
            "gain_start": 1,
            "gain_end": sc.get("gain_end", 5),
            "gain_up_epochs": 50,
            "uniformity_lambda_start": 0,
            "uniformity_lambda_end": round(sc.get("uniformity_lambda_end", 0.1), 4),
            "uniformity_lambda_scale_up_epochs": 25,
            "cka_alpha": 1,
            "cka_topk": sc.get("cka_topk", 20),
            "mod_loss_lambda": round(sc.get("mod_loss_lambda", 0.75), 4),
        },
        "wandb_params": {
            "use_wandb": True,
            "wandb_project": "dist-encoder",
        },
        "nav_eval_params": {
            "eval_every": 10,
            "eval_env_size": 20,
            "n_train_envs": 5,
            "n_val_envs": 5,
            "num_hopfields": 10,
            "n_starts_per_env": 100,
            "max_steps_mult": 3,
            "scale": 1.0,
            "normalize": True,
            "platform_radius": 1.0,
            "recompute_interval": 1,
            "hopfield_alpha": 0.8,
            "save_heatmaps_final": False,
        },
    }

    # train() detects existing wandb.run and won't re-init
    train(config, save_every=False)


if __name__ == "__main__":
    sweep_train()
