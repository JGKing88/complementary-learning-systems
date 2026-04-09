"""Encoder training script.

Usage:
    python -m encoder_training.train [--epochs 300] [--device cuda] ...
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainConfig, EncoderModelConfig, LossConfig
from .models import create_encoder
from .losses import (
    kernel_alignment_loss, local_attract_far_repel_loss,
    uniformity_loss, coplanarity_loss_sphere,
)
from .data import (
    build_grid_data, IndexDataset, rbf_kernel_batch,
    estimate_tau_median, build_grid_triples,
)
from .evaluate import eval_encoder


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    encoder,
    Phi: torch.Tensor,
    Xcoords: torch.Tensor,
    tau: float,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gain: float,
    loss_cfg: LossConfig,
    uniformity_lambda: float,
    triples_all: torch.Tensor | None = None,
    plane_lambda: float = 0.0,
    T_triple_batch: int = 4096,
) -> float:
    """Run one training epoch.  Returns mean loss."""
    device = next(encoder.parameters()).device
    dl = DataLoader(IndexDataset(Xcoords.size(0)), batch_size=batch_size,
                    shuffle=True, drop_last=True)
    encoder.train()
    running = 0.0

    for idx in dl:
        idx = idx.to(device).long()
        zb = encoder(Phi[idx], gain)
        K_pred = (zb @ zb.T).clamp(-1.0, 1.0)
        K_tgt = rbf_kernel_batch(Xcoords, idx, tau)

        # --- Compute loss based on mode ---
        if loss_cfg.mode == "local_far":
            loss = local_attract_far_repel_loss(
                K_pred, K_tgt,
                topk=loss_cfg.cka_topk or 5,
                centered=loss_cfg.centered,
                far_lambda=loss_cfg.far_lambda,
            )
        elif loss_cfg.mode == "mod_only":
            K_tgt_mod = _modulate_target(K_tgt, loss_cfg)
            loss = loss_cfg.mod_loss_lambda * kernel_alignment_loss(
                K_pred, K_tgt_mod, centered=loss_cfg.centered)
        else:  # "cka" (default)
            loss = kernel_alignment_loss(K_pred, K_tgt, centered=loss_cfg.centered)
            K_tgt_mod = _modulate_target(K_tgt, loss_cfg)
            if not torch.equal(K_tgt_mod, K_tgt):
                loss = loss + loss_cfg.mod_loss_lambda * kernel_alignment_loss(
                    K_pred, K_tgt_mod, centered=loss_cfg.centered)

        # Uniformity regularizer
        if uniformity_lambda > 0:
            loss = loss + uniformity_lambda * uniformity_loss(zb)

        # Coplanarity loss (optional)
        if plane_lambda > 0 and triples_all is not None:
            T_all = triples_all.size(0)
            sel = torch.randint(0, T_all, (min(T_triple_batch, T_all),))
            triples = triples_all[sel].to(device)
            flat = triples.reshape(-1)
            uniq, inv = torch.unique(flat, return_inverse=True)
            Zuniq = encoder(Phi.index_select(0, uniq), gain)
            z_triples = Zuniq.index_select(0, inv).view(-1, 3, Zuniq.size(-1))
            loss = loss + plane_lambda * coplanarity_loss_sphere(z_triples)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        running += loss.item()

    return running / max(len(dl), 1)


def _modulate_target(K_tgt: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """Apply power and top-K masking to the target kernel."""
    if (cfg.cka_alpha is None or cfg.cka_alpha == 1.0) and cfg.cka_topk is None:
        return K_tgt
    B = K_tgt.size(0)
    eye = torch.eye(B, device=K_tgt.device, dtype=K_tgt.dtype)
    K = K_tgt.clamp(0, 1)
    if cfg.cka_alpha is not None and cfg.cka_alpha != 1.0:
        K = K.pow(cfg.cka_alpha)
    if cfg.cka_topk is not None and cfg.cka_topk < B - 1:
        _, idx_top = torch.topk(K_tgt, k=min(cfg.cka_topk, B - 1), dim=1)
        mask = torch.zeros_like(K_tgt, dtype=torch.bool)
        mask.scatter_(1, idx_top, True)
        mask = (mask | mask.T) & ~eye.bool()
        K = K * mask.float() + eye
    return K


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_checkpoint(encoder, cfg: TrainConfig, epoch: int, path: str) -> None:
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "config": asdict(cfg.model),
        "epoch": epoch,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Data
    print(f"Building grid data: lambdas={cfg.model.lambdas}, fwhm_ratio={cfg.fwhm_ratio}")
    Phi, Xcoords, Npos = build_grid_data(cfg.model.lambdas, cfg.fwhm_ratio, str(device))
    N = Phi.size(0)

    tau = cfg.rbf_tau or estimate_tau_median(Xcoords)
    print(f"N={N}, code_dim={Phi.size(1)}, Npos={Npos}, tau={tau:.4f}, device={device}")

    triples_all = build_grid_triples(Npos, Npos, stride=3)

    # Model
    encoder = create_encoder(cfg.model, str(device))
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # Gain / uniformity annealing schedules
    gains = np.linspace(cfg.gain_start, cfg.gain_end, min(cfg.gain_up_epochs, cfg.epochs))
    if len(gains) < cfg.epochs:
        gains = np.concatenate([gains, np.full(cfg.epochs - len(gains), gains[-1])])
    uni_lambdas = np.linspace(cfg.uniformity_start, cfg.uniformity_end, cfg.epochs)

    # Output directory
    run_name = cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}")

    # Wandb
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        gain = float(gains[epoch - 1])
        uni_lam = float(uni_lambdas[epoch - 1])

        loss = train_epoch(
            encoder, Phi, Xcoords, tau, optimizer,
            cfg.batch_size, gain, cfg.loss, uni_lam,
            triples_all=triples_all, plane_lambda=cfg.loss.plane_lambda,
        )

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:04d} | loss {loss:.6f} | gain {gain:.2f} | uni_lam {uni_lam:.4f}")

        # Eval
        if cfg.eval_every > 0 and epoch % cfg.eval_every == 0:
            encoder.eval()
            metrics = eval_encoder(
                encoder, Phi, Xcoords, tau, gain=gain,
                cka_alpha=cfg.loss.cka_alpha, cka_topk=cfg.loss.cka_topk,
            )
            encoder.train()
            print(f"  eval | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if cfg.use_wandb:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=epoch)

        # Save
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(encoder, cfg, epoch,
                            os.path.join(run_dir, f"encoder_ep{epoch}.pt"))

        if cfg.use_wandb:
            import wandb
            wandb.log({"train/loss": loss, "train/gain": gain,
                        "train/uniformity_lambda": uni_lam}, step=epoch)

    # Final save
    save_checkpoint(encoder, cfg, cfg.epochs,
                    os.path.join(run_dir, "encoder_final.pt"))
    print(f"Training complete. Final checkpoint: {run_dir}/encoder_final.pt")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train grid cell encoder")
    # Model
    parser.add_argument("--encoder_type", type=str, default="cnn")
    parser.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_conv_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)
    # Loss
    parser.add_argument("--loss_mode", type=str, default="cka", choices=["cka", "mod_only", "local_far"])
    parser.add_argument("--cka_alpha", type=float, default=1.0)
    parser.add_argument("--cka_topk", type=int, default=None)
    parser.add_argument("--mod_loss_lambda", type=float, default=2.0)
    parser.add_argument("--far_lambda", type=float, default=1.0)
    # Training
    parser.add_argument("--lr", type=float, default=2.48e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fwhm_ratio", type=float, default=0.25)
    parser.add_argument("--gain_start", type=float, default=1.0)
    parser.add_argument("--gain_end", type=float, default=5.0)
    parser.add_argument("--gain_up_epochs", type=int, default=50)
    parser.add_argument("--uniformity_start", type=float, default=0.0)
    parser.add_argument("--uniformity_end", type=float, default=0.1)
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="encoders")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=50)
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="encoder-training")

    args = parser.parse_args()

    model_cfg = EncoderModelConfig(
        encoder_type=args.encoder_type, lambdas=args.lambdas,
        out_dim=args.out_dim, hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        hidden_channels=args.hidden_channels,
        num_conv_layers=args.num_conv_layers, kernel_size=args.kernel_size,
        gain=args.gain_end,
    )
    loss_cfg = LossConfig(
        mode=args.loss_mode, cka_alpha=args.cka_alpha,
        cka_topk=args.cka_topk, mod_loss_lambda=args.mod_loss_lambda,
        far_lambda=args.far_lambda,
    )
    cfg = TrainConfig(
        model=model_cfg, loss=loss_cfg,
        fwhm_ratio=args.fwhm_ratio, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, device=args.device,
        gain_start=args.gain_start, gain_end=args.gain_end,
        gain_up_epochs=args.gain_up_epochs,
        uniformity_start=args.uniformity_start,
        uniformity_end=args.uniformity_end,
        save_dir=args.save_dir, run_name=args.run_name,
        save_every=args.save_every, eval_every=args.eval_every,
        use_wandb=args.use_wandb, wandb_project=args.wandb_project,
    )
    train(cfg)


if __name__ == "__main__":
    main()
