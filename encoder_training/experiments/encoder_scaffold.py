"""Experiment: use a trained encoder as the hippocampal layer of VectorHash.

In standard VectorHash (``hopfield_nav.vectorhash.VectorHash``):

    p = nonlin(W_pg @ g - theta)            # random sparse projection + ReLU

Here we replace that with:

    p = encoder(g)                          # learned embedding

The rest of the scaffold is unchanged in *shape* — we heteroassociatively
train:

    W_gp  (p -> g)   via pseudo-inverse over all gridbook positions
    W_sp  (p -> s)   via pseudo-inverse over explored (s, p) pairs
    W_ps  (s -> p)   via pseudo-inverse over explored (s, p) pairs

Recall chain:

    obs --Wps--> p_in --Wgp--> g_in --WTA_per_module--> g_out
         --encoder--> p_clean --Wsp--> s_out

We compare to the standard random-projection VectorHash on the same envs
and seed, so we can read grid-recovery and observation-recall accuracy
side-by-side.

Nothing in this file modifies existing code.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

# --- Reused, unmodified utilities ------------------------------------------
from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d
from cls.vectorhash.assoc_utils_np import (
    nonlin,
    train_pbook,
    train_gcpc,
    pseudotrain_Wsp,
    pseudotrain_Wps,
    pseudotrain_Wgp,
)
from cls_paths import encoders_dir
from encoder_training.train import load_encoder
from encoder_training.utils import smooth_gbook
from hopfield_nav.env import GridEnv


DEFAULT_CKPT = str(encoders_dir() / "binary_20260409_083227" / "encoder_final.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlaps(x1, y1, x2, y2, size, gap=0) -> bool:
    return not (x1 + size + gap <= x2 or x2 + size + gap <= x1 or
                y1 + size + gap <= y2 or y2 + size + gap <= y1)


def _place_envs(n_envs: int, env_size: int, Npos: int,
                rng: np.random.RandomState) -> list[tuple[int, int]]:
    """Place non-overlapping ``env_size`` patches inside an Npos x Npos world."""
    placed: list[tuple[int, int]] = []
    for _ in range(200_000):
        if len(placed) == n_envs:
            return placed
        x = rng.randint(0, Npos - env_size + 1)
        y = rng.randint(0, Npos - env_size + 1)
        if all(not _overlaps(x, y, px, py, env_size) for (px, py) in placed):
            placed.append((x, y))
    raise RuntimeError(f"Could only place {len(placed)}/{n_envs} envs in Npos={Npos}.")


def _explore_envs(envs: list[GridEnv], offsets: list[tuple[int, int]]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Visit every cell in each env, returning global (locs, obs).

    Heading-invariant: keep one heading per position.
    """
    all_locs, all_obs = [], []
    for env, (cx, cy) in zip(envs, offsets):
        pos_obs_head = [p for p in env.fully_explore_random() if p[2] == (1, 0)]
        locs = np.array([p[0] for p in pos_obs_head])
        obs = np.array([p[1] for p in pos_obs_head])
        locs = locs.copy()
        locs[:, 0] += cx
        locs[:, 1] += cy
        all_locs.append(locs)
        all_obs.append(obs)
    return np.concatenate(all_locs), np.concatenate(all_obs)


# ---------------------------------------------------------------------------
# Encoder taps (B-series: use different layers as the hippocampal representation)
# ---------------------------------------------------------------------------

NAMED_TAPS = ("final", "pre_tanh", "last_hidden")


def _num_hidden_in_head(head: torch.nn.Sequential) -> int:
    """How many (Linear, act) hidden pairs the head has.

    Head structure (both ``GridEncoder.net`` and ``GridEncoderCNN.mlp``):

        [Linear, act,  Linear, act,  ..., Linear, act,  Linear]
         <--- k hidden pairs (2k slots) --->    <- final Linear (1 slot) ->

    so ``len(head) = 2k + 1`` with ``k = num_hidden_layers``.
    """
    n = len(head)
    if n < 1 or n % 2 != 1:
        raise RuntimeError(
            f"Unexpected encoder head length {n}; expected 2k+1 (k hidden layers).")
    return (n - 1) // 2


def _parse_tap(tap: str, head: torch.nn.Sequential) -> tuple[str, int]:
    """Normalize a tap string and validate it against the head's depth.

    Returns ``(kind, k)`` where ``kind`` is one of
    ``{final, pre_tanh, hidden}`` and ``k`` is the hidden-layer index
    (1..num_hidden_layers), or 0 when unused.
    """
    num_hidden = _num_hidden_in_head(head)
    if tap in ("final", "pre_tanh"):
        return tap, 0
    if tap == "last_hidden":
        return "hidden", num_hidden
    if tap.startswith("hidden_"):
        try:
            k = int(tap.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Could not parse tap {tap!r}; use hidden_<k>.") from exc
        if not (1 <= k <= num_hidden):
            raise ValueError(
                f"tap {tap!r}: k={k} out of range 1..{num_hidden} "
                f"(encoder has {num_hidden} hidden layers).")
        return "hidden", k
    raise ValueError(
        f"Unknown tap {tap!r}.  Named taps: {NAMED_TAPS}; or hidden_1..hidden_{num_hidden}.")


def _encoder_forward_tap(encoder: torch.nn.Module, x: torch.Tensor,
                         tap: str) -> torch.Tensor:
    """Run `encoder` forward up to `tap` and return the layer activation.

    Taps:

    - ``final``      : current default, L2-normalized tanh output.
    - ``pre_tanh``   : output of the last Linear, BEFORE tanh/sigmoid and L2-norm.
    - ``last_hidden``: output of the final nonlinearity before the last Linear
                      (alias for ``hidden_<num_hidden_layers>``).
    - ``hidden_k``   : output of the k-th nonlinearity (1..num_hidden_layers),
                      i.e. ``head[:2k](features)``.  Earlier k = deeper in
                      the network's feature hierarchy (less training-loss
                      pressure to collapse representations).

    Note: ``rp:<subtap>`` wrapping is handled one level up (in
    :class:`EncoderScaffold`), not here — this function only knows about the
    raw encoder layers.

    Works for both ``GridEncoder`` (MLP) and ``GridEncoderCNN``.
    """
    if tap == "final":
        return encoder(x)

    # Feed-forward to the MLP head (CNN or pure-MLP).
    if hasattr(encoder, "reshape_to_2d"):  # CNN
        features = encoder.pool(encoder.convs(encoder.reshape_to_2d(x))).flatten(1)
        head = encoder.mlp
    else:  # MLP
        features = x
        head = encoder.net

    kind, k = _parse_tap(tap, head)
    if kind == "pre_tanh":
        return head(features)
    if kind == "hidden":
        # Output of k-th activation = head[:2k](features).
        return head[:2 * k](features)
    raise AssertionError(f"unreachable tap kind {kind!r}")


# ---------------------------------------------------------------------------
# Random-projection wrapper: p = nonlin(Wrp @ tap(g) - thresh)
# ---------------------------------------------------------------------------

@dataclass
class RPConfig:
    """Config for the ``rp:<subtap>`` wrapper.

    Applies a VectorHash-style random projection *on top of* an encoder tap:

        p = nonlin(Wrp @ tap(g) - thresh)      with Wrp sparsity = c

    This is an analogue of the baseline ``p = nonlin(Wpg @ g - thresh)`` but
    with ``g`` replaced by a learned representation.
    """
    Np: int
    thresh: float
    c: float
    seed: int


class _RPWrapper:
    """Fixed random sparse projection + thresholded ReLU on a (in_dim, B) batch."""

    def __init__(self, cfg: RPConfig, in_dim: int) -> None:
        self.cfg = cfg
        self.in_dim = in_dim
        rng = np.random.RandomState(cfg.seed)
        Wrp = rng.randn(cfg.Np, in_dim)
        prune = int((1 - cfg.c) * cfg.Np * in_dim)
        mask = np.ones((cfg.Np, in_dim))
        if prune > 0:
            mask[rng.randint(0, cfg.Np, prune),
                 rng.randint(0, in_dim, prune)] = 0.0
        self.Wrp = (mask * Wrp).astype(np.float32)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x: (in_dim, B) -> (Np, B), post-thresholded."""
        return nonlin(self.Wrp @ x, thresh=self.cfg.thresh)


def _split_rp_tap(tap: str) -> tuple[str, bool]:
    """``rp:<subtap>`` -> (subtap, True); otherwise (tap, False)."""
    if tap.startswith("rp:"):
        sub = tap[3:]
        if not sub:
            raise ValueError("Empty subtap after 'rp:'; use e.g. rp:final.")
        return sub, True
    return tap, False


def _encode_gbook(encoder: torch.nn.Module, gbook: np.ndarray,
                  lambdas: list[int], fwhm_ratio: float, device: str,
                  tap: str = "final", chunk: int = 1000) -> np.ndarray:
    """Encode every (x, y) in a (Ng, Npos, Npos) gbook -> (Nt, Npos, Npos).

    ``Nt`` is the tap dim (``out_dim`` or ``hidden_dim`` depending on tap).
    Uses the same FWHM smoothing the encoder was trained with.

    This returns the *raw* tap output; any ``rp:`` wrapping is applied
    afterwards by the caller (``EncoderScaffold``).
    """
    Ng, Npos, _ = gbook.shape
    if fwhm_ratio > 0:
        sgb = smooth_gbook(gbook, lambdas, fwhm_ratio)
    else:
        sgb = gbook.astype(np.float32, copy=False)

    flat = sgb.reshape(Ng, Npos * Npos).T.astype(np.float32)   # (Npos*Npos, Ng)
    outs: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for s in range(0, flat.shape[0], chunk):
            x = torch.from_numpy(flat[s:s + chunk]).to(device)
            outs.append(_encoder_forward_tap(encoder, x, tap).cpu().numpy())
    enc = np.concatenate(outs, axis=0)                          # (Npos*Npos, Nt)
    Nt = enc.shape[1]
    return enc.T.reshape(Nt, Npos, Npos)                        # (Nt, Npos, Npos)


def _module_wise_wta(gin: np.ndarray, module_sizes: list[int]) -> np.ndarray:
    """Column-wise WTA within each module.  gin: (Ng, B) -> one-hot (Ng, B)."""
    g_out = np.zeros_like(gin)
    B = gin.shape[1]
    idx = 0
    for j in module_sizes:
        block = gin[idx:idx + j]
        winners = block.argmax(axis=0)
        g_out[idx + winners, np.arange(B)] = 1.0
        idx += j
    return g_out


# ---------------------------------------------------------------------------
# Scaffolds
# ---------------------------------------------------------------------------

_WGP_RULES = ("hebbian", "pseudo")


@dataclass
class ScaffoldCommon:
    """Things both scaffolds share."""
    lambdas: list[int]
    Npos: int
    Ng: int
    module_sizes: list[int]
    gbook: np.ndarray          # (Ng, Npos, Npos)
    offsets: list[tuple[int, int]]
    locs: np.ndarray           # (Npatts, 2) global
    obs: np.ndarray            # (Npatts, Ns)
    sbook: np.ndarray          # (Ns, Npatts)


class EncoderScaffold:
    """VectorHash scaffold with ``p = encoder_tap(g)`` (optionally RP-wrapped).

    The ``tap`` argument can be either a plain tap name (``final``,
    ``pre_tanh``, ``last_hidden``, ``hidden_k``) or ``rp:<subtap>``.  In the
    latter case a fixed sparse random projection + thresholded ReLU is applied
    on top of the subtap output, producing

        p = nonlin(Wrp @ subtap(g) - rp.thresh)

    with Wrp sparsity ``rp.c`` and output dim ``rp.Np``.  This is the direct
    VectorHash-style random-projection construction, but built on top of a
    learned feature instead of the raw grid code.
    """

    def __init__(self, com: ScaffoldCommon, encoder: torch.nn.Module,
                 fwhm_ratio: float, device: str, tap: str = "final",
                 rp: RPConfig | None = None,
                 wgp_rule: str = "pseudo") -> None:
        if wgp_rule not in _WGP_RULES:
            raise ValueError(f"wgp_rule={wgp_rule!r}; must be one of {_WGP_RULES}.")
        self.com = com
        self.encoder = encoder
        self.device = device
        self.fwhm_ratio = fwhm_ratio
        self.tap = tap
        self.wgp_rule = wgp_rule

        self._inner_tap, wants_rp = _split_rp_tap(tap)
        if wants_rp and rp is None:
            raise ValueError(
                f"tap={tap!r} requests a random projection but no RPConfig was given.")
        if rp is not None and not wants_rp:
            raise ValueError(
                f"RPConfig provided but tap={tap!r} does not start with 'rp:'.")

        print(f"  [enc] encoding full gbook: (Ng={com.Ng}, Npos={com.Npos}, "
              f"tap={tap!r}, inner={self._inner_tap!r})")
        t0 = time.time()
        tap_book = _encode_gbook(
            encoder, com.gbook, com.lambdas, fwhm_ratio, device,
            tap=self._inner_tap)
        Nt, Npos, _ = tap_book.shape
        print(f"  [enc] raw tap shape={tap_book.shape}  "
              f"(Nt={Nt})  took {time.time()-t0:.1f}s")

        # Optionally wrap with a VectorHash-style random projection.
        self.rp: _RPWrapper | None = None
        if rp is not None:
            self.rp = _RPWrapper(rp, in_dim=Nt)
            flat = tap_book.reshape(Nt, -1)                           # (Nt, Npos^2)
            p_flat = self.rp.apply(flat)                              # (Np, Npos^2)
            self.pbook = p_flat.reshape(rp.Np, Npos, Npos)
            active = (p_flat > 0).mean()
            print(f"  [rp ] applied Wrp({rp.Np}x{Nt}) c={rp.c} thresh={rp.thresh} "
                  f"-> pbook shape={self.pbook.shape}  "
                  f"mean-active={active:.3f}")
        else:
            self.pbook = tap_book
        self.Np = self.pbook.shape[0]

        # Wgp mapping p -> g over ALL gbook positions. Same rule choice as
        # the random scaffold so encoder-vs-random comparisons can be made
        # with the rule held fixed.
        pflat = self.pbook.reshape(self.Np, -1)          # (Np, Npos^2)
        gflat = com.gbook.reshape(com.Ng, -1)            # (Ng, Npos^2)
        Npatts_full = pflat.shape[1]
        if wgp_rule == "hebbian":
            print("  [enc] Wgp rule = 'hebbian' (G P^T / Npatts)")
            self.Wgp = train_gcpc(pflat, gflat, Npatts=Npatts_full)       # (Ng, Np)
        else:  # pseudo
            print("  [enc] Wgp rule = 'pseudo' (G @ pinv(P))")
            self.Wgp = pseudotrain_Wgp(pflat, gflat, Npatts_full)         # (Ng, Np)

        # Wsp / Wps via pseudo-inverse over explored (p, s) pairs.
        Npatts = com.sbook.shape[1]
        path_pbook = np.stack(
            [self.pbook[:, x, y] for (x, y) in com.locs], axis=1)  # (Np, Npatts)
        print(f"  [enc] pseudotrain_Wsp/Wps  (Npatts={Npatts})")
        self.Wsp = pseudotrain_Wsp(com.sbook, path_pbook, Npatts)   # (Ns, Np)
        self.Wps = pseudotrain_Wps(path_pbook, com.sbook, Npatts)   # (Np, Ns)
        self._path_pbook = path_pbook

    # ------------------------------------------------------------------
    def recall_batch(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """obs_batch: (B, Ns) -> (s_out, p_out, g_out) each (B, dim)."""
        S = obs_batch.T                                        # (Ns, B)
        pin = self.Wps @ S                                     # (Np, B) -- no 2nd nonlin
        gin = self.Wgp @ pin                                   # (Ng, B)
        gout = _module_wise_wta(gin, self.com.module_sizes)    # (Ng, B)

        # Clean-up pass through the encoder at the configured tap.
        g_flat = gout.T.astype(np.float32)                     # (B, Ng)
        if self.fwhm_ratio > 0:
            # Smooth per-example using the same per-module Gaussian bumps.
            g_flat = _smooth_gflat(g_flat, self.com.lambdas, self.fwhm_ratio)
        with torch.no_grad():
            x = torch.from_numpy(g_flat).to(self.device)
            tout = _encoder_forward_tap(self.encoder, x, self._inner_tap).cpu().numpy()
        tout = tout.T                                          # (Nt, B)

        if self.rp is not None:
            pout = self.rp.apply(tout)                         # (Np, B)
        else:
            pout = tout                                        # (Nt=Np, B)

        sout = (self.Wsp @ pout > 0.5).astype(np.float32)      # (Ns, B)
        return sout.T, pout.T, gout.T


def _smooth_gflat(g_flat: np.ndarray, lambdas: list[int], fwhm_ratio: float) -> np.ndarray:
    """Batched version of smooth_g — keeps the path differentiable-ish by just
    reusing smooth_gbook-style vectorized convolution on a (B, Ng) batch."""
    # Treat (B, Ng) as a (Ng, B, 1) gbook so we can reuse smooth_gbook.
    Ng = g_flat.shape[1]
    fake = g_flat.T[:, :, None]                                # (Ng, B, 1)
    smoothed = smooth_gbook(fake, lambdas, fwhm_ratio)          # (Ng, B, 1)
    return smoothed[:, :, 0].T.astype(np.float32)              # (B, Ng)


class RandomProjectionScaffold:
    """Standard VectorHash scaffold: ``p = nonlin(Wpg @ g - theta)``.

    Set ``smooth_fwhm_ratio > 0`` to apply the same per-module Gaussian
    smoothing the encoder uses, so the random projection sees
    ``nonlin(Wpg @ smooth_g - theta)`` at both build time and recall time.
    Defaults to 0 (no smoothing) to preserve the canonical VectorHash baseline.

    ``wgp_rule`` selects the training rule for ``Wgp``:

    - ``"hebbian"``: ``(1/Npatts) * G @ P^T`` (canonical; relies on pbook
      columns being near-orthogonal, i.e. sparse-binary & uncorrelated
      across positions).
    - ``"pseudo"``: ``G @ pinv(P)`` — same rule the encoder scaffold uses.
      Tolerates correlated pbook columns (e.g. when smoothing is on, or
      when we want an apples-to-apples comparison with the encoder path).
    """

    def __init__(self, com: ScaffoldCommon, Np: int, thresh: float, c: float,
                 rng: np.random.RandomState,
                 smooth_fwhm_ratio: float = 0.0,
                 wgp_rule: str = "hebbian") -> None:
        if wgp_rule not in _WGP_RULES:
            raise ValueError(f"wgp_rule={wgp_rule!r}; must be one of {_WGP_RULES}.")
        self.com = com
        self.Np = Np
        self.thresh = thresh
        self.c = c
        self.smooth_fwhm_ratio = smooth_fwhm_ratio
        self.wgp_rule = wgp_rule

        Wpg = rng.randn(Np, com.Ng)
        prune = int((1 - c) * Np * com.Ng)
        mask = np.ones((Np, com.Ng))
        if prune > 0:
            mask[rng.randint(0, Np, prune), rng.randint(0, com.Ng, prune)] = 0.0
        self.Wpg = mask * Wpg

        if smooth_fwhm_ratio > 0:
            gbook_in = smooth_gbook(com.gbook, com.lambdas, smooth_fwhm_ratio)
            print(f"  [rnd] using smoothed gbook (fwhm_ratio={smooth_fwhm_ratio})")
        else:
            gbook_in = com.gbook
        self.pbook = nonlin(train_pbook(self.Wpg, gbook_in), thresh=thresh)   # (Np, Npos, Npos)

        gflat = com.gbook.reshape(com.Ng, -1)
        pflat = self.pbook.reshape(Np, -1)
        Npatts_full = pflat.shape[1]
        if wgp_rule == "hebbian":
            self.Wgp = train_gcpc(pflat, gflat, Npatts=Npatts_full)           # (Ng, Np)
        else:  # pseudo
            self.Wgp = pseudotrain_Wgp(pflat, gflat, Npatts_full)             # (Ng, Np)
        print(f"  [rnd] Wgp rule = {wgp_rule!r}")

        Npatts = com.sbook.shape[1]
        path_pbook = np.stack(
            [self.pbook[:, x, y] for (x, y) in com.locs], axis=1)             # (Np, Npatts)
        self.Wsp = pseudotrain_Wsp(com.sbook, path_pbook, Npatts)
        self.Wps = pseudotrain_Wps(path_pbook, com.sbook, Npatts)
        self._path_pbook = path_pbook

    def recall_batch(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = obs_batch.T
        pin = self.Wps @ S
        gin = self.Wgp @ pin
        gout = _module_wise_wta(gin, self.com.module_sizes)
        # Clean-up: re-encode via Wpg, matching the build-time preprocessing.
        if self.smooth_fwhm_ratio > 0:
            g_clean = _smooth_gflat(
                gout.T.astype(np.float32),
                self.com.lambdas, self.smooth_fwhm_ratio,
            ).T                                                    # (Ng, B)
        else:
            g_clean = gout
        pout = nonlin(self.Wpg @ g_clean, thresh=self.thresh)
        sout = (self.Wsp @ pout > 0.5).astype(np.float32)
        return sout.T, pout.T, gout.T


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _path_gbook(com: ScaffoldCommon) -> np.ndarray:
    """True grid vectors at every explored global position, shape (Ng, Npatts)."""
    out = np.zeros((com.Ng, com.locs.shape[0]), dtype=com.gbook.dtype)
    for k, (x, y) in enumerate(com.locs):
        out[:, k] = com.gbook[:, x, y]
    return out


def evaluate(scaffold, com: ScaffoldCommon, pflip: float = 0.0,
             rng: np.random.RandomState | None = None) -> dict:
    """Report (possibly-corrupted) recall accuracy at every explored position."""
    rng = rng or np.random.RandomState(0)
    obs = com.obs.copy()                                        # (Npatts, Ns)
    if pflip > 0:
        mask = rng.rand(*obs.shape) < pflip
        obs = np.where(mask, 1.0 - obs, obs)

    s_out, p_out, g_out = scaffold.recall_batch(obs)            # each (B, *)
    truth_g = _path_gbook(com).T                                # (Npatts, Ng)
    truth_s = com.obs                                           # (Npatts, Ns)
    truth_p = scaffold._path_pbook.T                            # (Npatts, Np)

    g_correct = np.all(g_out == truth_g, axis=1).mean()
    s_err = np.mean(np.abs(s_out - truth_s))                    # bit error rate
    # p recall quality: cosine sim
    norms = (np.linalg.norm(p_out, axis=1) * np.linalg.norm(truth_p, axis=1) + 1e-12)
    p_cos = (p_out * truth_p).sum(axis=1) / norms

    return {
        "g_accuracy": float(g_correct),
        "s_bit_err": float(s_err),
        "p_cos_mean": float(p_cos.mean()),
        "p_cos_std": float(p_cos.std()),
        "n_patterns": int(obs.shape[0]),
        "pflip": float(pflip),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_common(args, encoder_lambdas: list[int]) -> ScaffoldCommon:
    lambdas = list(encoder_lambdas)
    Ng = int(np.sum(np.square(lambdas)))
    Npos = args.Npos
    module_sizes = [l * l for l in lambdas]

    print(f"  building gbook: lambdas={lambdas}  Ng={Ng}  Npos={Npos}")
    gbook = gen_gbook_2d(lambdas, Ng, Npos)                     # (Ng, Npos, Npos)

    # Place envs.
    place_rng = np.random.RandomState(args.seed)
    offsets = _place_envs(args.n_envs, args.env_size, Npos, place_rng)
    print(f"  placed {len(offsets)} envs of size {args.env_size} at {offsets}")

    # Build envs with DISTINCT codebooks (seed per-env).
    envs = [
        GridEnv(size=args.env_size, observation_size=args.Ns,
                seed=args.seed + 1 + i)
        for i in range(args.n_envs)
    ]
    locs, obs = _explore_envs(envs, offsets)
    sbook = obs.T                                               # (Ns, Npatts)
    print(f"  explored Npatts={locs.shape[0]}  Ns={obs.shape[1]}")

    return ScaffoldCommon(
        lambdas=lambdas, Npos=Npos, Ng=Ng, module_sizes=module_sizes,
        gbook=gbook, offsets=offsets, locs=locs, obs=obs, sbook=sbook,
    )


def _print_result(name: str, res: dict) -> None:
    print(f"    {name}: "
          f"g_acc={res['g_accuracy']*100:6.2f}%  "
          f"s_bit_err={res['s_bit_err']*100:6.3f}%  "
          f"p_cos={res['p_cos_mean']:.3f}±{res['p_cos_std']:.3f}  "
          f"(pflip={res['pflip']:.2f}, N={res['n_patterns']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", type=str, default="both",
                   choices=["encoder", "random", "both"],
                   help="Which scaffold to build/evaluate. "
                        "'random' is the standard VectorHash "
                        "(p = nonlin(Wpg @ g - theta)). "
                        "'encoder' uses a trained encoder as the pc layer. "
                        "'both' runs both side-by-side (default).")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                   help="Encoder checkpoint (only read in 'encoder' / 'both' modes).")
    p.add_argument("--lambdas", type=int, nargs="+", default=None,
                   help="Grid module periods.  In encoder/both modes this "
                        "defaults to the encoder's lambdas; in 'random' mode "
                        "it defaults to [11,12,13].")
    p.add_argument("--Npos", type=int, default=100,
                   help="World size (Npos x Npos).  Must be >= env_size.")
    p.add_argument("--Ns", type=int, default=1600,
                   help="Observation dimension per env.")
    p.add_argument("--Np", type=int, default=None,
                   help="Np for the random-projection scaffold. "
                        "Defaults to 1600 in 'random' mode and to encoder "
                        "out_dim in 'both' mode (for an equal-Np comparison).")
    p.add_argument("--thresh", type=float, default=0.5,
                   help="Random-projection scaffold threshold.")
    p.add_argument("--c", type=float, default=0.5,
                   help="Random-projection scaffold Wpg sparsity.")
    p.add_argument("--smooth_random", action="store_true",
                   help="Apply per-module Gaussian grid-smoothing to the "
                        "random-projection scaffold too (input to Wpg at both "
                        "build-time and recall clean-up). Uses the same "
                        "--fwhm_ratio as the encoder path. Default: off "
                        "(canonical VectorHash).")
    p.add_argument("--fwhm_ratio_random", type=float, default=None,
                   help="Override the fwhm_ratio used by --smooth_random. "
                        "Defaults to the encoder's fwhm_ratio (or 0.25 if "
                        "no encoder is loaded).")
    p.add_argument("--wgp_rule", "--random_wgp", dest="wgp_rule",
                   type=str, default="pseudo",
                   choices=list(_WGP_RULES),
                   help="Wgp training rule used by *both* scaffolds so they "
                        "can be compared apples-to-apples. 'hebbian' is the "
                        "canonical VectorHash random-projection rule "
                        "(1/Npatts * G P^T); 'pseudo' is G @ pinv(P). "
                        "Default: 'pseudo' (handles correlated pbook "
                        "columns, required for encoder scaffolds and for "
                        "random scaffolds with --smooth_random). "
                        "'--random_wgp' is a backwards-compatible alias.")
    p.add_argument("--n_envs", type=int, default=5)
    p.add_argument("--env_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--fwhm_ratio", type=float, default=None,
                   help="Override fwhm_ratio; default: read from ckpt (else 0.25). "
                        "Only relevant for the encoder scaffold.")
    p.add_argument("--encoder_tap", type=str, default="final",
                   help="Which layer of the encoder to use as the pc layer. "
                        "Options: 'final' (L2-normalized tanh, Np=out_dim), "
                        "'pre_tanh' (last Linear, Np=out_dim), "
                        "'last_hidden' (output of final activation, alias "
                        "for 'hidden_<num_hidden_layers>'), or "
                        "'hidden_<k>' for k=1..num_hidden_layers "
                        "(smaller k = deeper in the feature hierarchy). "
                        "For the default MLP encoder, Np=hidden_dim for any "
                        "hidden_k tap. "
                        "Prefix with 'rp:' (e.g. 'rp:final', 'rp:hidden_4') "
                        "to apply a VectorHash-style random projection "
                        "p = nonlin(Wrp @ tap(g) - rp_thresh) on top of the "
                        "tap output; see --rp_np / --rp_thresh / --rp_c.")
    p.add_argument("--rp_np", type=int, default=1024,
                   help="Output dim of the rp:<subtap> wrapper.")
    p.add_argument("--rp_thresh", type=float, default=0.0,
                   help="Threshold for the rp:<subtap> wrapper "
                        "(nonlin(x - rp_thresh)). Default 0.0 = plain ReLU; "
                        "raise to make the representation sparser.")
    p.add_argument("--rp_c", type=float, default=0.5,
                   help="Wrp sparsity for the rp:<subtap> wrapper "
                        "(fraction of non-pruned entries).")
    p.add_argument("--rp_seed", type=int, default=None,
                   help="Seed for Wrp in the rp:<subtap> wrapper. "
                        "Defaults to --seed + 7777 so it's independent of "
                        "the env placement seed.")
    p.add_argument("--pflips", type=str, default="0.0,0.05,0.1,0.2",
                   help="Comma-separated obs flip probabilities for stress tests.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    need_encoder = args.mode in ("encoder", "both")

    # ---- Resolve encoder / lambdas / fwhm_ratio -----------------------------
    encoder = None
    fwhm_ratio = args.fwhm_ratio if args.fwhm_ratio is not None else 0.25
    out_dim: int | None = None

    if need_encoder:
        print(f"[1/4] Loading encoder: {args.ckpt}")
        encoder, ckpt = load_encoder(args.ckpt, device=device)
        encoder.eval()
        enc_lambdas = list(ckpt["model_config"]["lambdas"])
        train_cfg = ckpt.get("train_config") or {}
        fwhm_ratio = (args.fwhm_ratio if args.fwhm_ratio is not None
                      else float(train_cfg.get("fwhm_ratio", 0.25)))
        out_dim = int(ckpt["model_config"]["out_dim"])
        lambdas = args.lambdas if args.lambdas is not None else enc_lambdas
        print(f"  encoder lambdas={enc_lambdas}  out_dim={out_dim}  "
              f"fwhm_ratio={fwhm_ratio}")
        if args.lambdas is not None and args.lambdas != enc_lambdas:
            print(f"  WARNING: overriding lambdas to {lambdas} "
                  f"(does not match encoder's {enc_lambdas})")
    else:
        lambdas = args.lambdas if args.lambdas is not None else [11, 12, 13]
        print("[1/4] Encoder not loaded (mode=random).  "
              f"Using lambdas={lambdas}")

    # Stash the resolved lambdas so _build_common can read them via args.
    args.lambdas = lambdas

    # ---- Build shared scaffold data -----------------------------------------
    print(f"[2/4] Building gbook + envs (Npos={args.Npos}, "
          f"{args.n_envs} x {args.env_size}, Ns={args.Ns})")
    com = _build_common(args, lambdas)

    # ---- Build scaffolds ----------------------------------------------------
    enc_sc = None
    rand_sc = None

    if args.mode in ("encoder", "both"):
        rp_cfg: RPConfig | None = None
        if args.encoder_tap.startswith("rp:"):
            rp_seed = args.rp_seed if args.rp_seed is not None else args.seed + 7777
            rp_cfg = RPConfig(
                Np=args.rp_np, thresh=args.rp_thresh,
                c=args.rp_c, seed=rp_seed)
            print(f"[3/4] Building ENCODER scaffold (p = nonlin(Wrp @ tap(g) - {args.rp_thresh}), "
                  f"tap={args.encoder_tap!r}, rp_np={args.rp_np}, rp_c={args.rp_c})")
        else:
            print(f"[3/4] Building ENCODER scaffold (p = encoder(g), tap={args.encoder_tap!r})")
        enc_sc = EncoderScaffold(
            com, encoder, fwhm_ratio, device,
            tap=args.encoder_tap, rp=rp_cfg,
            wgp_rule=args.wgp_rule)

    if args.mode in ("random", "both"):
        if args.Np is not None:
            Np_r = args.Np
        elif args.mode == "both":
            Np_r = out_dim   # equal-Np comparison
        else:
            Np_r = 1600      # standard VectorHashConfig default
        if args.smooth_random:
            rand_fwhm = (args.fwhm_ratio_random
                         if args.fwhm_ratio_random is not None else fwhm_ratio)
        else:
            rand_fwhm = 0.0
        tag = "3b" if args.mode == "both" else "3"
        smooth_note = f", smooth_fwhm={rand_fwhm}" if rand_fwhm > 0 else ""
        print(f"[{tag}/4] Building RANDOM-PROJECTION scaffold "
              f"(Np={Np_r}, thresh={args.thresh}, c={args.c}{smooth_note}, "
              f"wgp={args.wgp_rule})")
        rand_sc = RandomProjectionScaffold(
            com, Np=Np_r, thresh=args.thresh, c=args.c,
            rng=np.random.RandomState(args.seed),
            smooth_fwhm_ratio=rand_fwhm,
            wgp_rule=args.wgp_rule,
        )

    # ---- Evaluate -----------------------------------------------------------
    print("[4/4] Evaluating recall on explored positions")
    pflips = [float(x) for x in args.pflips.split(",") if x.strip()]
    for pf in pflips:
        print(f"  -- pflip = {pf:.2f} --")
        if enc_sc is not None:
            rng = np.random.RandomState(args.seed + 1000 + int(pf * 1000))
            _print_result("encoder", evaluate(enc_sc, com, pflip=pf, rng=rng))
        if rand_sc is not None:
            rng = np.random.RandomState(args.seed + 1000 + int(pf * 1000))
            _print_result("random ", evaluate(rand_sc, com, pflip=pf, rng=rng))

    print("done.")


if __name__ == "__main__":
    sys.exit(main())
