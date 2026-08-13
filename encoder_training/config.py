"""Configuration dataclasses for binary-method encoder training."""
from __future__ import annotations

from dataclasses import dataclass, field

from cls_paths import encoders_dir


@dataclass
class EncoderModelConfig:
    """Architecture config for the grid cell encoder."""
    encoder_type: str = "mlp"           # "mlp" or "cnn"
    lambdas: list[int] = field(default_factory=lambda: [11, 12, 13])
    out_dim: int = 256
    nonlinearity: str = "gelu"
    output_nonlinearity: str = "tanh"
    gain: float = 5.0
    # MLP-specific
    hidden_dim: int = 1024
    num_hidden_layers: int = 4
    # CNN-specific
    hidden_channels: int = 128
    num_conv_layers: int = 3
    kernel_size: int = 5

    @property
    def in_dim(self) -> int:
        return sum(l * l for l in self.lambdas)


@dataclass
class LossConfig:
    """Binary-method loss configuration.

    Primary loss: `mse_contrastive` — attracts near pairs (cos sim → 1),
    repels far pairs (cos sim → 0). Within-env target = 1 if dist < radius,
    else 0. With single-env batches there are no cross-env pairs by
    construction.

    `uniformity_lambda` is usually 0 for the MSE method — included for
    ablation. `cka` mode retained for baseline comparison.
    """
    mode: str = "mse_contrastive"       # "mse_contrastive" or "cka"
    attract_lambda: float = 2.0         # weight on near-pair MSE (target 1)
    repel_weight: float = 5.0           # weight on far-pair MSE (target 0)
    uniformity_lambda: float = 0.0      # uniformity regularizer weight (end value)
    uniformity_anneal_epochs: int = 25  # epochs to ramp uniformity from 0 to end
    uniformity_t: float = 2.0           # temperature in logsumexp(-t * d^2)
    # Which pairs the uniformity term acts on.
    #   "all"     — every off-diagonal pair. Asks nothing about environments,
    #               so it is the only spread term still available when the
    #               cross-environment pairs are withheld. Also the reason it
    #               fights `attract`: logsumexp is dominated by the closest
    #               pairs, which are the near pairs.
    #   "nonnear" — drops the near pairs. Cheap and effective, but "not near"
    #               includes every cross-environment pair, so this quietly
    #               restores the supervision `exclude_cross_env_pairs` removed.
    #               Report it as a loophole, not as a result under the flag.
    uniformity_scope: str = "all"       # "all" | "nonnear"
    centered: bool = True               # centered CKA (only used if mode="cka")

    # VICReg-style spread. Both are batch statistics rather than pair terms, so
    # they cannot concentrate on the closest pairs the way uniformity does.
    var_lambda: float = 0.0             # hinge on per-coordinate std
    cov_lambda: float = 0.0             # off-diagonal covariance penalty
    var_gamma: float = 1.0              # std target, in units of 1/sqrt(out_dim)
    rate_lambda: float = 0.0            # MCR^2 log-det coding rate
    rate_eps: float = 0.5

    # Distance-graded pair target: target = exp(-d^2 / (2 sigma^2)) instead of
    # the binary 1-on-near / 0-on-far. 0 keeps the binary targets.
    #
    # The binary target asks for a *plateau* at cosine 1 inside the radius, and
    # the unique radius is a strictly-decreasing test, so a perfectly satisfied
    # binary target scores zero. What is actually being measured is the residual
    # slope the network fails to flatten. Naming the slope directly is the
    # obvious thing nobody has tried.
    graded_sigma: float = 0.0

    # LOOPHOLE, label it as one. Repel pairs whose *input* grid codes are
    # dissimilar, regardless of environment. Env-blind and available to an agent
    # that only ever sees observations — but the smoothed code decorrelates
    # within ~5 cells, so "input-dissimilar" is very nearly "far apart
    # anywhere", and that puts the cross-environment repulsion back in.
    input_far_tau: float = -1.0         # <0 → off

    # Withhold cross-environment pairs from the repel term. There is no
    # dedicated cross-env term: any pair that is not "near" is repelled, so in
    # a mixed batch every cross-env pair is pushed toward cosine 0. Setting
    # this reproduces what `single_env_batch=True` does to the *loss* while
    # still drawing each gradient step from many environments — the two are
    # confounded otherwise, and this is what separates them.
    exclude_cross_env_pairs: bool = False


@dataclass
class PatchConfig:
    """Spatial-patch setup for training."""
    # Fixed-size patches: sample `nenv` patches of `npos × npos`.
    nenv: int = 25
    npos: int = 100
    # Alternative: explicit list of patch sizes (overrides nenv & npos).
    npos_list: list[int] | None = None

    # Radius defining "near" within each env:
    #   - If `per_env_radius_frac > 0`: radius = frac * env_size (per env).
    #   - Else: `local_radius` is used (fixed across envs). 0 → full same-env.
    per_env_radius_frac: float = 0.1
    local_radius: float = 10.0

    # Batching:
    #   - single_env_batch=True: each batch is from one env only.
    #   - single_env_batch=False: mixed batches from all envs.
    single_env_batch: bool = True


@dataclass
class NavEvalConfig:
    """Navigation-eval settings (run on val envs placed outside training patches)."""
    env_size: int = 20
    n_train_envs: int = 5
    n_val_envs: int = 5
    num_hopfields: int = 20
    n_starts_per_env: int = 100
    max_steps_mult: int = 3
    scale: float = 1.0
    normalize: bool = True
    platform_radius: float = 1.0
    recompute_interval: int = 1
    hopfield_alpha: float = 0.8


@dataclass
class UniqueRadiusConfig:
    """Unique-coding-radius eval (encoder_training.eval_unique_radius).

    Scored on the full Npos x Npos arena rather than on patches, so unlike the
    nav eval it says how far a position stays identifiable *globally*. Costs
    roughly 15 s at lambdas (11,12,13), which is why ``every`` defaults to a
    multiple of the nav eval rather than every epoch.
    """
    enabled: bool = True
    every: int = 100                    # epochs between evals (0 = off)
    n_refs: int = 20
    border: int = 100                   # keep refs this far from any edge
    seed: int = 0                       # shared across runs: paired comparison
    batch_size: int = 16384


@dataclass
class TrainConfig:
    """Full training configuration."""
    model: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    nav_eval: NavEvalConfig = field(default_factory=NavEvalConfig)
    unique_radius: UniqueRadiusConfig = field(
        default_factory=UniqueRadiusConfig)

    fwhm_ratio: float = 0.25            # Gaussian smoothing of grid codes

    # Training
    lr: float = 2.48e-4
    weight_decay: float = 1e-4
    epochs: int = 600
    batch_size: int = 4096
    seed: int = 42
    device: str = "cuda"
    grad_clip: float = 1.0

    # Gain annealing (linear over ALL epochs)
    gain_start: float = 1.0
    gain_end: float = 5.0

    # Input perturbation (ablation): randomly permute grid codes across positions
    shuffle_inputs: bool = False

    # Build each patch's codes directly instead of slicing the full codebook.
    # Same values (tests/test_lazy_patch_codes.py) at ~1 GB instead of ~20 GB of
    # host memory, which is what lets several runs share a node. Only possible
    # when the Hopfield nav eval is off, since that eval needs the full grid.
    # The two builders group the Gaussian factors differently, so the codes
    # agree to float32 rounding rather than bit-for-bit: within a wave every run
    # takes the same path, but a seed-for-seed replay of an older run needs this
    # off.
    lazy_codes: bool = False

    # Checkpointing
    save_dir: str = str(encoders_dir())
    run_name: str = ""                  # empty → auto timestamp
    eval_every: int = 50                # epochs between nav evals (0 = off)
