"""Binary-method encoder training.

Usage:
    python -m encoder_training.train [flags]

Options (selected):
    --nenv / --npos                fixed-size patches
    --npos_list 40,60,80,...       variable-size patches (overrides --nenv/--npos)
    --per_env_radius_frac 0.1      per-env radius = frac * env_size
    --single_env_batch             batches drawn from one env at a time
    --loss_mode mse_contrastive    also "cka"
    --attract_lambda 2.0
    --repel_weight 5.0
    --uniformity_lambda 0.0        set >0 to add spread regularizer

Produces:
    {save_dir}/{run_name}/encoder_best.pt   — peak val_nav checkpoint
    {save_dir}/{run_name}/encoder_final.pt  — final-epoch checkpoint
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from .config import (
    TrainConfig, EncoderModelConfig, LossConfig, PatchConfig, NavEvalConfig,
)
from .data import (
    build_full_grid, sample_nonoverlapping_patches, extract_patches,
    mixed_batch_iterator, single_env_batch_iterator,
)
from .losses import cka_loss, uniformity_loss, mse_attract_repel
from .models import create_encoder
from .evaluate import encode_grid, run_nav_eval


def _build_near_mask(
    idx: torch.Tensor,
    env_ids: torch.Tensor,
    coords: torch.Tensor,
    env_radius: torch.Tensor | None,
    local_radius: float,
) -> torch.Tensor:
    """Return [B, B] boolean mask of "near" pairs (excluding diagonal).

    If `env_radius` is given, each point's threshold is its own env's radius
    (works for both single-env and mixed batches). Otherwise uses the scalar
    `local_radius`. Radius <= 0 → "same env" used as the mask.
    """
    env_b = env_ids[idx]                                 # [B]
    same_env = env_b[:, None] == env_b[None, :]          # [B, B]
    B = idx.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=idx.device)

    if env_radius is not None:
        r_b = env_radius[env_b]                          # [B]
        if (r_b <= 0).all():
            return same_env & ~eye
        # Pairs are only candidates when same_env, so r_b[i] == r_b[j];
        # broadcasting along rows is sufficient.
        r_thresh = r_b[:, None]                          # [B, 1]
    else:
        if local_radius <= 0:
            return same_env & ~eye
        r_thresh = float(local_radius)

    Xb = coords[idx]
    diff = Xb[:, None, :] - Xb[None, :, :]
    dist = diff.square().sum(-1).sqrt()
    return (dist < r_thresh) & same_env & ~eye


def train(cfg: TrainConfig) -> str:
    """Run training. Returns the run directory."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # --- Seed ---
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    # --- Run dir ---
    run_name = cfg.run_name or f"enc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # --- Build full grid and sample patches ---
    Phi_full, full_Npos = build_full_grid(cfg.model.lambdas, cfg.fwhm_ratio)
    patch_cfg = cfg.patches
    npos_arg = patch_cfg.npos_list if patch_cfg.npos_list else patch_cfg.npos
    nenv_arg = None if patch_cfg.npos_list else patch_cfg.nenv
    y0s, x0s, sizes = sample_nonoverlapping_patches(
        full_Npos, full_Npos, npos_arg, nenv_arg)
    print(f"Patches: {len(sizes)} envs, sizes {sorted(set(sizes))}")

    Phi_flat, coords, env_ids = extract_patches(Phi_full, y0s, x0s, sizes, device)
    N = Phi_flat.shape[0]
    print(f"Total points N={N} ({N / (full_Npos**2) * 100:.2f}% of grid)")

    # --- Shuffled-input ablation ---
    if cfg.shuffle_inputs:
        perm = torch.randperm(N, device=device)
        Phi_flat = Phi_flat[perm]
        print("*** SHUFFLED INPUTS: grid codes permuted across positions ***")

    # --- Per-env radius ---
    env_radius: Optional[torch.Tensor] = None
    if patch_cfg.per_env_radius_frac > 0:
        env_radius = torch.tensor(
            [s * patch_cfg.per_env_radius_frac for s in sizes],
            device=device, dtype=torch.float32)
        print(f"Per-env radius (frac={patch_cfg.per_env_radius_frac}): "
              f"{env_radius.min().item():.1f} .. {env_radius.max().item():.1f}")
    else:
        print(f"Fixed local_radius={patch_cfg.local_radius}")

    # --- Encoder & optimizer ---
    encoder = create_encoder(cfg.model, str(device))
    optimizer = torch.optim.AdamW(encoder.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params")

    # --- Schedules ---
    gains = np.linspace(cfg.gain_start, cfg.gain_end, cfg.epochs)
    unif_anneal = max(1, cfg.loss.uniformity_anneal_epochs)
    unif_ramp = np.linspace(0.0, cfg.loss.uniformity_lambda, unif_anneal)
    unif_schedule = np.concatenate([
        unif_ramp,
        np.full(max(0, cfg.epochs - unif_anneal), cfg.loss.uniformity_lambda),
    ])[:cfg.epochs]

    # --- Env index lookups for single-env batching ---
    env_indices_list: list[torch.Tensor] | None = None
    if patch_cfg.single_env_batch:
        env_indices_list = [torch.where(env_ids == e)[0]
                            for e in range(len(sizes))]

    best_val_nav = -1.0
    rng_nav = np.random.RandomState(cfg.seed)

    # --- Training loop ---
    for ep in range(1, cfg.epochs + 1):
        gain = float(gains[ep - 1])
        unif_lam = float(unif_schedule[ep - 1])

        if env_indices_list is not None:
            batch_iter = single_env_batch_iterator(env_indices_list, cfg.batch_size)
        else:
            batch_iter = mixed_batch_iterator(N, cfg.batch_size)

        encoder.train()
        running = 0.0
        n_batches = 0

        for idx in batch_iter:
            idx = idx.to(device).long()
            zb = encoder(Phi_flat[idx], gain)
            K_pred = (zb @ zb.T).clamp(-1.0, 1.0)

            near = _build_near_mask(idx, env_ids, coords, env_radius,
                                    patch_cfg.local_radius)

            if cfg.loss.mode == "mse_contrastive":
                loss = mse_attract_repel(
                    K_pred, near,
                    attract_lambda=cfg.loss.attract_lambda,
                    repel_weight=cfg.loss.repel_weight,
                )
            elif cfg.loss.mode == "cka":
                B = K_pred.size(0)
                eye = torch.eye(B, dtype=torch.bool, device=device)
                K_binary = (near | eye).float()
                loss = cfg.loss.attract_lambda * cka_loss(
                    K_pred, K_binary, centered=cfg.loss.centered)
            else:
                raise ValueError(f"Unknown loss mode: {cfg.loss.mode}")

            if unif_lam > 0:
                loss = loss + unif_lam * uniformity_loss(zb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        print(f"Epoch {ep:3d} | loss={avg_loss:.4f} | gain={gain:.2f} "
              f"| unif_lam={unif_lam:.4f}")

        # --- Nav eval ---
        if cfg.eval_every > 0 and ep % cfg.eval_every == 0:
            encoded = encode_grid(encoder, Phi_full, gain, device)
            val_nav = run_nav_eval(
                encoded, y0s, x0s, sizes, full_Npos, gain, cfg.nav_eval,
                rng=rng_nav, split="val")
            print(f"  Val nav: acc={val_nav['accuracy']:.3f} | "
                  f"steps={val_nav['mean_steps']:.1f} | "
                  f"speed={val_nav['mean_speed']:.3f}")
            if val_nav['accuracy'] > best_val_nav:
                best_val_nav = float(val_nav['accuracy'])
                _save_ckpt(encoder, cfg, y0s, x0s, sizes, gain,
                           os.path.join(run_dir, "encoder_best.pt"),
                           val_nav_acc=best_val_nav, epoch=ep)

    # --- Final save ---
    _save_ckpt(encoder, cfg, y0s, x0s, sizes, float(gains[-1]),
               os.path.join(run_dir, "encoder_final.pt"))
    print(f"Done. Best val_nav: {best_val_nav:.3f}")
    return run_dir


def _save_ckpt(encoder, cfg: TrainConfig, y0s, x0s, sizes, gain: float,
               path: str, **extras) -> None:
    torch.save({
        "state_dict": encoder.state_dict(),
        "model_config": asdict(cfg.model),
        "train_config": asdict(cfg),
        "y0s": y0s, "x0s": x0s, "sizes": sizes,
        "gain": float(gain),
        **extras,
    }, path)


def load_encoder(path: str, device: str | torch.device = "cuda"):
    """Load a saved encoder + metadata. Returns (encoder, ckpt_dict).

    Backward-compatible: handles both the new format ({"model_config", ...})
    and the old `train_binary_encoder.py` format ({"config": {"model_params"}}).
    The returned ckpt dict is normalized — it always has "model_config" as a
    top-level key.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "model_config" in ckpt:
        model_cfg_dict = ckpt["model_config"]
    elif "config" in ckpt and "model_params" in ckpt["config"]:
        model_cfg_dict = ckpt["config"]["model_params"]
        ckpt["model_config"] = model_cfg_dict
    else:
        raise KeyError(
            f"Could not find model config in checkpoint {path}. "
            f"Expected 'model_config' or 'config.model_params'. "
            f"Top-level keys: {list(ckpt.keys())}")

    # Keep only fields that EncoderModelConfig knows about
    valid_fields = set(EncoderModelConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in model_cfg_dict.items() if k in valid_fields}
    mcfg = EncoderModelConfig(**filtered)

    encoder = create_encoder(mcfg, str(device))
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()

    # Backfill common fields for viz scripts that assume them
    ckpt.setdefault("gain", mcfg.gain)
    return encoder, ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cfg_from_args(args) -> TrainConfig:
    model = EncoderModelConfig(
        encoder_type=args.encoder_type, lambdas=args.lambdas,
        out_dim=args.out_dim, hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        hidden_channels=args.hidden_channels,
        num_conv_layers=args.num_conv_layers, kernel_size=args.kernel_size,
        gain=args.gain_end,
    )
    loss = LossConfig(
        mode=args.loss_mode,
        attract_lambda=args.attract_lambda,
        repel_weight=args.repel_weight,
        uniformity_lambda=args.uniformity_lambda,
        uniformity_anneal_epochs=args.uniformity_anneal_epochs,
    )
    npos_list = None
    if args.npos_list:
        npos_list = [int(x) for x in args.npos_list.split(",")]
    patches = PatchConfig(
        nenv=args.nenv, npos=args.npos, npos_list=npos_list,
        per_env_radius_frac=args.per_env_radius_frac,
        local_radius=args.radius,
        single_env_batch=args.single_env_batch,
    )
    nav = NavEvalConfig(
        env_size=args.nav_env_size,
        n_train_envs=args.nav_n_train,
        n_val_envs=args.nav_n_val,
        num_hopfields=args.nav_num_hopfields,
        n_starts_per_env=args.nav_n_starts,
    )
    n_total = (sum(s * s for s in npos_list) if npos_list
               else args.nenv * args.npos * args.npos)
    batch_size = min(args.batch_size, n_total)
    return TrainConfig(
        model=model, loss=loss, patches=patches, nav_eval=nav,
        fwhm_ratio=args.fwhm_ratio, lr=args.lr, epochs=args.epochs,
        batch_size=batch_size, seed=args.seed, device=args.device,
        gain_start=args.gain_start, gain_end=args.gain_end,
        shuffle_inputs=args.shuffle,
        save_dir=args.save_dir, run_name=args.run_name,
        eval_every=args.eval_every,
    )


def main():
    p = argparse.ArgumentParser(description="Binary-method encoder training")
    # Model
    p.add_argument("--encoder_type", default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--out_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_conv_layers", type=int, default=3)
    p.add_argument("--kernel_size", type=int, default=5)
    # Patches
    p.add_argument("--nenv", type=int, default=25)
    p.add_argument("--npos", type=int, default=100)
    p.add_argument("--npos_list", type=str, default=None)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--per_env_radius_frac", type=float, default=0.0)
    p.add_argument("--single_env_batch", action="store_true")
    # Loss
    p.add_argument("--loss_mode", default="mse_contrastive",
                   choices=["mse_contrastive", "cka"])
    p.add_argument("--attract_lambda", type=float, default=2.0)
    p.add_argument("--repel_weight", type=float, default=5.0)
    p.add_argument("--uniformity_lambda", type=float, default=0.0)
    p.add_argument("--uniformity_anneal_epochs", type=int, default=25)
    # Training
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=2.48e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--gain_start", type=float, default=1.0)
    p.add_argument("--gain_end", type=float, default=5.0)
    p.add_argument("--shuffle", action="store_true")
    # Nav eval
    p.add_argument("--nav_env_size", type=int, default=20)
    p.add_argument("--nav_n_train", type=int, default=5)
    p.add_argument("--nav_n_val", type=int, default=5)
    p.add_argument("--nav_num_hopfields", type=int, default=20)
    p.add_argument("--nav_n_starts", type=int, default=100)
    # Checkpointing
    p.add_argument("--save_dir", default="/home/jackking/cls/encoders")
    p.add_argument("--run_name", default="")
    p.add_argument("--eval_every", type=int, default=50)

    args = p.parse_args()
    cfg = _build_cfg_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
