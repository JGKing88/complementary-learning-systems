"""Configuration dataclasses for binary-method encoder training."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    centered: bool = True               # centered CKA (only used if mode="cka")


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
class TrainConfig:
    """Full training configuration."""
    model: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    nav_eval: NavEvalConfig = field(default_factory=NavEvalConfig)

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

    # Checkpointing
    save_dir: str = "/home/jackking/cls/encoders"
    run_name: str = ""                  # empty → auto timestamp
    eval_every: int = 50                # epochs between nav evals (0 = off)
