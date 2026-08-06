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

    # Three historical save formats:
    #   1. Newest (encoder_training/_save_ckpt):   top-level "model_config" + "gain"
    #   2. Middle (old train_binary_encoder.py):   ckpt["config"]["model_params"]
    #   3. Oldest:                                 ckpt["config"] is already the model dict
    if "model_config" in ckpt:
        cfg_dict = ckpt["model_config"]
    elif "config" in ckpt:
        raw_cfg = ckpt["config"]
        cfg_dict = raw_cfg.get("model_params", raw_cfg)
    else:
        raise KeyError(
            f"Could not find encoder model config in checkpoint {checkpoint_path}. "
            f"Expected 'model_config' or 'config'/'config.model_params'. "
            f"Top-level keys: {list(ckpt.keys())}"
        )

    # Only keep fields EncoderModelConfig knows about (new save format dumps
    # extra training-time fields alongside model fields).
    valid_fields = set(EncoderModelConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    cfg = EncoderModelConfig(**filtered)

    encoder = create_encoder(cfg, device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.requires_grad_(False)

    # Gain resolution order: explicit override → top-level "gain" (new format)
    # → model-config "gain" (old format) → 1.0 fallback.
    if gain_override is not None:
        effective_gain = gain_override
    elif "gain" in ckpt:
        effective_gain = float(ckpt["gain"])
    else:
        effective_gain = float(cfg.gain)
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
