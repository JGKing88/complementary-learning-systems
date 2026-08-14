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

from cls_paths import encoders_dir
from .config import (
    TrainConfig, EncoderModelConfig, LossConfig, PatchConfig, NavEvalConfig,
    UniqueRadiusConfig,
)
from .data import (
    build_full_grid, build_patch_codes, sample_nonoverlapping_patches,
    extract_patches, mixed_batch_iterator, single_env_batch_iterator,
)
from .losses import (
    cka_loss, coding_rate_loss, mse_attract_repel, participation_ratio,
    uniformity_loss, vicreg_terms,
)
from .models import create_encoder
from .evaluate import encode_grid, run_nav_eval
from .eval_unique_radius import grid_code_batch


def _build_near_mask(
    idx: torch.Tensor,
    env_ids: torch.Tensor,
    coords: torch.Tensor,
    env_radius: torch.Tensor | None,
    local_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return ([B, B] "near" mask excluding the diagonal, same-env mask, dist).

    If `env_radius` is given, each point's threshold is its own env's radius
    (works for both single-env and mixed batches). Otherwise uses the scalar
    `local_radius`. Radius <= 0 → "same env" used as the mask, and no distance
    matrix is needed, so ``dist`` comes back None.

    ``same_env`` is returned as well because the caller may need to exclude
    cross-environment pairs from the *far* set; it is computed here anyway, as
    is ``dist``, which a distance-graded target needs.
    """
    env_b = env_ids[idx]                                 # [B]
    same_env = env_b[:, None] == env_b[None, :]          # [B, B]
    B = idx.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=idx.device)

    if env_radius is not None:
        r_b = env_radius[env_b]                          # [B]
        if (r_b <= 0).all():
            return same_env & ~eye, same_env, None
        # Pairs are only candidates when same_env, so r_b[i] == r_b[j];
        # broadcasting along rows is sufficient.
        r_thresh = r_b[:, None]                          # [B, 1]
    else:
        if local_radius <= 0:
            return same_env & ~eye, same_env, None
        r_thresh = float(local_radius)

    Xb = coords[idx]
    diff = Xb[:, None, :] - Xb[None, :, :]
    dist = diff.square().sum(-1).sqrt()
    return (dist < r_thresh) & same_env & ~eye, same_env, dist


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
    # The lazy path skips the 10.2 GB codebook and builds each patch directly.
    # It needs the full grid only for the Hopfield nav eval, so it is available
    # exactly when that eval is off. Neither builder touches the torch RNG, so
    # the patch placement is the same either way.
    lazy = cfg.lazy_codes and cfg.eval_every <= 0
    full_Npos = int(np.prod(cfg.model.lambdas))
    Phi_full = None
    if not lazy:
        Phi_full, full_Npos = build_full_grid(cfg.model.lambdas, cfg.fwhm_ratio)
    patch_cfg = cfg.patches
    npos_arg = patch_cfg.npos_list if patch_cfg.npos_list else patch_cfg.npos
    nenv_arg = None if patch_cfg.npos_list else patch_cfg.nenv
    y0s, x0s, sizes = sample_nonoverlapping_patches(
        full_Npos, full_Npos, npos_arg, nenv_arg,
        placement=patch_cfg.patch_placement)
    print(f"Patches: {len(sizes)} envs, sizes {sorted(set(sizes))}"
          + f"  [{patch_cfg.patch_placement} placement]"
          + ("  [lazy codes]" if lazy else ""))

    if lazy:
        Phi_flat, coords, env_ids = build_patch_codes(
            cfg.model.lambdas, y0s, x0s, sizes, device, cfg.fwhm_ratio)
    else:
        Phi_flat, coords, env_ids = extract_patches(
            Phi_full, y0s, x0s, sizes, device)
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
    best_r_min = -1.0
    last_ur: dict = {}          # most recent unique-radius summary, for ckpts
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
        running_extra = 0.0        # the spread terms alone, to see if they dominate
        epoch_pr = float("nan")
        n_batches = 0

        for idx in batch_iter:
            idx = idx.to(device).long()
            zb = encoder(Phi_flat[idx], gain)
            K_pred = (zb @ zb.T).clamp(-1.0, 1.0)

            near, same_env, dist = _build_near_mask(
                idx, env_ids, coords, env_radius, patch_cfg.local_radius)

            if cfg.loss.mode == "mse_contrastive":
                # Withholding cross-environment pairs from the repel term
                # reproduces what single_env_batch=True does to the *loss*,
                # while leaving each gradient step drawn from many envs. The
                # two differ, and only this separates them.
                far = None
                if cfg.loss.exclude_cross_env_pairs:
                    far = ~near & same_env
                if cfg.loss.input_far_tau >= 0:
                    # Loophole arm: env-blind, but restores long-range
                    # repulsion. Recorded as such -- see LossConfig.
                    phi = torch.nn.functional.normalize(Phi_flat[idx], dim=-1)
                    k_in = phi @ phi.T
                    far = (k_in < cfg.loss.input_far_tau) & ~near
                target = None
                if cfg.loss.graded_sigma > 0 and dist is not None:
                    s = cfg.loss.graded_sigma
                    target = torch.exp(-dist.square() / (2.0 * s * s))
                loss = mse_attract_repel(
                    K_pred, near,
                    attract_lambda=cfg.loss.attract_lambda,
                    repel_weight=cfg.loss.repel_weight,
                    far_mask=far,
                    target=target,
                )
            elif cfg.loss.mode == "cka":
                B = K_pred.size(0)
                eye = torch.eye(B, dtype=torch.bool, device=device)
                K_binary = (near | eye).float()
                loss = cfg.loss.attract_lambda * cka_loss(
                    K_pred, K_binary, centered=cfg.loss.centered)
            else:
                raise ValueError(f"Unknown loss mode: {cfg.loss.mode}")

            # OUT OF BRIEF, diagnostic only (§5.6k, PatchConfig.spread_arena_frac).
            # The spread terms see these extra whole-arena positions; no pair
            # term ever does, so `near`/`same_env` stay the batch's own.
            z_spread = zb
            if patch_cfg.spread_arena_frac > 0:
                n_extra = int(round(patch_cfg.spread_arena_frac * len(idx)))
                # Argument order mirrors build_patch_codes: the first argument
                # indexes the first spatial axis, whatever the parameter is
                # called. Both are uniform here, but the two paths must agree.
                r_ys = np.random.randint(0, full_Npos, size=n_extra)
                r_xs = np.random.randint(0, full_Npos, size=n_extra)
                phi_extra = torch.as_tensor(
                    grid_code_batch(cfg.model.lambdas, r_ys, r_xs,
                                    cfg.fwhm_ratio),
                    device=device, dtype=Phi_flat.dtype)
                z_spread = torch.cat([zb, encoder(phi_extra, gain)], dim=0)

            extra = 0.0
            if unif_lam > 0:
                pair_mask = ~near if cfg.loss.uniformity_scope == "nonnear" \
                    else None
                if pair_mask is not None and z_spread is not zb:
                    pair_mask = None    # the mask is batch-shaped; scope drops
                u = unif_lam * uniformity_loss(
                    z_spread, t=cfg.loss.uniformity_t, pair_mask=pair_mask)
                loss, extra = loss + u, extra + float(u.item())
            if cfg.loss.var_lambda > 0 or cfg.loss.cov_lambda > 0:
                var_l, cov_l = vicreg_terms(z_spread, gamma=cfg.loss.var_gamma)
                v = cfg.loss.var_lambda * var_l + cfg.loss.cov_lambda * cov_l
                loss, extra = loss + v, extra + float(v.item())
            if cfg.loss.rate_lambda > 0:
                r = cfg.loss.rate_lambda * coding_rate_loss(
                    z_spread, eps=cfg.loss.rate_eps)
                loss, extra = loss + r, extra + float(r.item())
            running_extra += extra
            # The quantity that separates the regimes; see losses.
            if n_batches == 0:
                epoch_pr = float(participation_ratio(zb.detach()).item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        avg_extra = running_extra / max(n_batches, 1)
        print(f"Epoch {ep:3d} | loss={avg_loss:.4f} | spread={avg_extra:+.4f} "
              f"| pr={epoch_pr:6.1f} | gain={gain:.2f} | unif_lam={unif_lam:.4f}")

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
                           val_nav_acc=best_val_nav, epoch=ep,
                           unique_radius=last_ur)

        # --- Unique-radius eval ---
        # Scored on the whole arena, not on patches, so it is the one signal
        # here that reports what happens outside the training envs.
        ur = cfg.unique_radius
        if ur.enabled and ur.every > 0 and ep % ur.every == 0:
            last_ur = _unique_radius_eval(encoder, cfg, gain)
            if last_ur:
                print(f"  Unique radius: r_min={last_ur['r_min']:.1f} | "
                      f"median={last_ur['r_median']:.1f} | "
                      f"alias={last_ur['alias_ceiling_max']:.3f}")
                # With the nav eval off there is no other selection signal, so
                # the radius picks encoder_best.pt. When both run, nav keeps
                # that job and the radius is recorded but does not select --
                # otherwise the two would fight over the same file.
                if cfg.eval_every <= 0 and last_ur["r_min"] > best_r_min:
                    best_r_min = float(last_ur["r_min"])
                    _save_ckpt(encoder, cfg, y0s, x0s, sizes, gain,
                               os.path.join(run_dir, "encoder_best.pt"),
                               epoch=ep, unique_radius=last_ur)

    # --- Final save ---
    if cfg.unique_radius.enabled and not last_ur:
        last_ur = _unique_radius_eval(encoder, cfg, float(gains[-1]))
    _save_ckpt(encoder, cfg, y0s, x0s, sizes, float(gains[-1]),
               os.path.join(run_dir, "encoder_final.pt"),
               unique_radius=last_ur)
    print(f"Done. Best val_nav: {best_val_nav:.3f}")
    if last_ur:
        print(f"      Unique radius r_min: {last_ur['r_min']:.1f}")
    return run_dir


def _unique_radius_eval(encoder, cfg: TrainConfig, gain: float) -> dict:
    """Summary row from ``evaluate_unique_radius``, or {} if it fails.

    Imported lazily and guarded: this is a diagnostic, and a long training run
    must not die at epoch 900 because a metric raised.
    """
    try:
        from encoder_training.eval_unique_radius import evaluate_unique_radius
        ur = cfg.unique_radius
        _, summary = evaluate_unique_radius(
            encoder, lambdas=cfg.model.lambdas, gain=gain,
            n_refs=ur.n_refs, border=ur.border, seed=ur.seed,
            device=cfg.device, batch_size=ur.batch_size,
            fwhm_ratio=cfg.fwhm_ratio,
        )
        return summary
    except Exception as exc:                       # noqa: BLE001 - diagnostic
        print(f"  [unique-radius eval failed: {type(exc).__name__}: {exc}]")
        return {}


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
        uniformity_t=args.uniformity_t,
        uniformity_scope=args.uniformity_scope,
        exclude_cross_env_pairs=args.exclude_cross_env_pairs,
        var_lambda=args.var_lambda,
        cov_lambda=args.cov_lambda,
        var_gamma=args.var_gamma,
        rate_lambda=args.rate_lambda,
        rate_eps=args.rate_eps,
        graded_sigma=args.graded_sigma,
        input_far_tau=args.input_far_tau,
    )
    npos_list = None
    if args.npos_list:
        npos_list = [int(x) for x in args.npos_list.split(",")]
    patches = PatchConfig(
        nenv=args.nenv, npos=args.npos, npos_list=npos_list,
        per_env_radius_frac=args.per_env_radius_frac,
        local_radius=args.radius,
        single_env_batch=args.single_env_batch,
        patch_placement=args.patch_placement,
        spread_arena_frac=args.spread_arena_frac,
    )
    nav = NavEvalConfig(
        env_size=args.nav_env_size,
        n_train_envs=args.nav_n_train,
        n_val_envs=args.nav_n_val,
        num_hopfields=args.nav_num_hopfields,
        n_starts_per_env=args.nav_n_starts,
    )
    ur = UniqueRadiusConfig(
        enabled=not args.no_unique_radius,
        every=args.ur_every, n_refs=args.ur_n_refs,
        border=args.ur_border, seed=args.ur_seed,
    )
    n_total = (sum(s * s for s in npos_list) if npos_list
               else args.nenv * args.npos * args.npos)
    batch_size = min(args.batch_size, n_total)
    return TrainConfig(
        model=model, loss=loss, patches=patches, nav_eval=nav,
        unique_radius=ur,
        fwhm_ratio=args.fwhm_ratio, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=batch_size, seed=args.seed, device=args.device,
        gain_start=args.gain_start, gain_end=args.gain_end,
        shuffle_inputs=args.shuffle, lazy_codes=args.lazy_codes,
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
    p.add_argument("--patch_placement", default="random",
                   choices=["random", "stratified"],
                   help="where patches sit: uniform rejection sampling, or a "
                        "jittered lattice (one per coarse-grid cell)")
    p.add_argument("--spread_arena_frac", type=float, default=0.0,
                   help="OUT OF BRIEF, DIAGNOSTIC ONLY (§5.6k): extra positions "
                        "from the whole arena, as a fraction of batch_size, fed "
                        "to the spread terms only. Breaks the coverage "
                        "constraint by construction -- never a headline number")
    # Loss
    p.add_argument("--loss_mode", default="mse_contrastive",
                   choices=["mse_contrastive", "cka"])
    p.add_argument("--attract_lambda", type=float, default=2.0)
    p.add_argument("--repel_weight", type=float, default=5.0)
    p.add_argument("--uniformity_lambda", type=float, default=0.0)
    p.add_argument("--exclude_cross_env_pairs", action="store_true",
                   help="withhold cross-env pairs from the repel term; "
                        "isolates that from single_env_batch's effect on "
                        "which envs a gradient step sees")
    p.add_argument("--uniformity_anneal_epochs", type=int, default=25)
    p.add_argument("--uniformity_t", type=float, default=2.0)
    p.add_argument("--uniformity_scope", default="all",
                   choices=["all", "nonnear"],
                   help="'nonnear' drops the near pairs -- but 'not near' "
                        "includes every cross-env pair, so it restores the "
                        "supervision exclude_cross_env_pairs removes")
    p.add_argument("--var_lambda", type=float, default=0.0,
                   help="VICReg variance hinge; pair-free spread")
    p.add_argument("--cov_lambda", type=float, default=0.0,
                   help="VICReg off-diagonal covariance penalty")
    p.add_argument("--var_gamma", type=float, default=1.0)
    p.add_argument("--rate_lambda", type=float, default=0.0,
                   help="MCR^2 log-det coding rate; rewards an even covariance "
                        "spectrum, which is the deficit the collapsed codes have")
    p.add_argument("--rate_eps", type=float, default=0.5)
    p.add_argument("--graded_sigma", type=float, default=0.0,
                   help="distance-graded pair target exp(-d^2/2s^2); "
                        "0 keeps the binary near=1/far=0 targets")
    p.add_argument("--input_far_tau", type=float, default=-1.0,
                   help="LOOPHOLE: repel pairs whose input grid codes have "
                        "cosine below this, ignoring env labels")
    # Training
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=2.48e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--gain_start", type=float, default=1.0)
    p.add_argument("--gain_end", type=float, default=5.0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--lazy_codes", action="store_true",
                   help="build patch codes directly (~1 GB instead of ~20 GB); "
                        "ignored unless --eval_every 0")
    # Nav eval
    p.add_argument("--nav_env_size", type=int, default=20)
    p.add_argument("--nav_n_train", type=int, default=5)
    p.add_argument("--nav_n_val", type=int, default=5)
    p.add_argument("--nav_num_hopfields", type=int, default=20)
    p.add_argument("--nav_n_starts", type=int, default=100)
    # Checkpointing
    p.add_argument("--save_dir", default=str(encoders_dir()))
    p.add_argument("--run_name", default="")
    p.add_argument("--eval_every", type=int, default=50,
                   help="epochs between Hopfield nav evals; 0 disables them")
    # Unique radius. Independent of the nav eval: it is scored on the whole
    # arena rather than on patches, and needs no Hopfield.
    p.add_argument("--no_unique_radius", action="store_true")
    p.add_argument("--ur_every", type=int, default=100)
    p.add_argument("--ur_n_refs", type=int, default=20)
    p.add_argument("--ur_border", type=int, default=100)
    p.add_argument("--ur_seed", type=int, default=0,
                   help="reference positions; keep fixed across a sweep")

    args = p.parse_args()
    cfg = _build_cfg_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
