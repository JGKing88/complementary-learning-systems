"""Thin loader for pretrained encoder checkpoints."""
from __future__ import annotations

import torch

from encoder_training.models import GridEncoder, GridEncoderCNN, create_encoder
from encoder_training.config import EncoderModelConfig


def load_encoder(
    checkpoint_path: str,
    device: str = "cpu",
    gain_override: float | None = None,
) -> tuple[GridEncoder | GridEncoderCNN, EncoderModelConfig, float]:
    """Load a pretrained encoder from an encoder_training checkpoint.

    Returns:
        encoder: Frozen model in eval mode.
        config: The EncoderModelConfig from the checkpoint.
        effective_gain: The gain to use (override or checkpoint's).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_cfg = ckpt["config"]

    # Support both old format (nested model_params) and new format (flat)
    cfg_dict = raw_cfg.get("model_params", raw_cfg)

    cfg = EncoderModelConfig(
        encoder_type=cfg_dict.get("encoder_type", "cnn"),
        lambdas=cfg_dict["lambdas"],
        out_dim=cfg_dict["out_dim"],
        hidden_dim=cfg_dict.get("hidden_dim", 512),
        num_hidden_layers=cfg_dict.get("num_hidden_layers", 2),
        hidden_channels=cfg_dict.get("hidden_channels", 128),
        num_conv_layers=cfg_dict.get("num_conv_layers", 3),
        kernel_size=cfg_dict.get("kernel_size", 5),
        nonlinearity=cfg_dict.get("nonlinearity", "gelu"),
        output_nonlinearity=cfg_dict.get("output_nonlinearity", "tanh"),
        gain=cfg_dict.get("gain", 1.0),
    )

    encoder = create_encoder(cfg, device)
    # Support both key names
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.requires_grad_(False)

    effective_gain = gain_override if gain_override is not None else cfg.gain
    return encoder, cfg, effective_gain


def validate_config(
    encoder_cfg: EncoderModelConfig,
    vectorhash_lambdas: list[int],
    encoder_gain: float,
    fwhm_ratio: float,
) -> None:
    """Validate that encoder config is compatible with vectorhash/hopfield config.

    Raises ValueError with a clear message if anything is mismatched.
    """
    if list(encoder_cfg.lambdas) != list(vectorhash_lambdas):
        raise ValueError(
            f"Encoder lambdas {encoder_cfg.lambdas} != VectorHash lambdas {vectorhash_lambdas}. "
            "These must match for grid codes to be compatible."
        )
