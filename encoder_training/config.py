from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderModelConfig:
    """Architecture config for the grid cell encoder."""
    encoder_type: str = "cnn"           # "mlp" or "cnn"
    lambdas: list[int] = field(default_factory=lambda: [11, 12])
    out_dim: int = 128
    nonlinearity: str = "gelu"
    output_nonlinearity: str = "tanh"
    gain: float = 1.0
    # MLP-specific
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    # CNN-specific
    hidden_channels: int = 128
    num_conv_layers: int = 3
    kernel_size: int = 5

    @property
    def in_dim(self) -> int:
        return sum(l * l for l in self.lambdas)


@dataclass
class LossConfig:
    """Loss function configuration."""
    mode: str = "cka"                   # "cka", "mod_only", "local_far"
    cka_alpha: float = 1.0              # power for modulated target kernel
    cka_topk: int | None = None         # top-K mask for modulated CKA
    mod_loss_lambda: float = 1.0        # weight of modulated CKA term
    uniformity_lambda: float = 0.0      # uniformity regularizer weight
    far_lambda: float = 1.0             # far-repel weight (for local_far mode)
    plane_lambda: float = 0.0           # coplanarity loss weight
    centered: bool = True               # use centered CKA


@dataclass
class TrainConfig:
    """Full training configuration."""
    model: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Data
    fwhm_ratio: float = 0.25           # Gaussian smoothing of grid codes
    rbf_tau: float | None = None        # RBF scale; None = estimate from median distance

    # Training
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 4096
    seed: int = 0
    device: str = "cuda"

    # Gain annealing
    gain_start: float = 1.0
    gain_end: float = 5.0
    gain_up_epochs: int = 50

    # Uniformity annealing
    uniformity_start: float = 0.0
    uniformity_end: float = 0.1

    # Checkpointing
    save_dir: str = "encoders"
    run_name: str = ""
    save_every: int = 50

    # Eval
    eval_every: int = 50

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "encoder-training"
