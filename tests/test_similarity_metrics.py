import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F


def _load_testing_dist_encoder_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "testing_dist_encoder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "testing_dist_encoder_module", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dist = _load_testing_dist_encoder_module()


class IdentityEncoder(nn.Module):
    def forward(self, x: torch.Tensor, sigmoid_scale: float = 1.0) -> torch.Tensor:
        return x


def test_kernel_alignment_loss_identical_kernels_are_zero():
    K = torch.tensor(
        [
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.05],
            [-0.1, 0.05, 1.0],
        ],
        dtype=torch.float32,
    )
    loss = dist.kernel_alignment_loss(K, K, centered=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_kernel_alignment_loss_handles_constant_shift():
    base = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float32)
    shifted = base + 0.2  # global offset should be removed when centered

    centered_loss = dist.kernel_alignment_loss(shifted, base, centered=True).item()
    uncentered_loss = dist.kernel_alignment_loss(shifted, base, centered=False).item()

    assert centered_loss == pytest.approx(0.0, abs=1e-7)
    assert uncentered_loss == pytest.approx(centered_loss, abs=1e-7)


def test_weighted_local_mse_matches_manual_two_point_case():
    # Two points => single off-diagonal entry after masking.
    K_pred = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    K_tgt = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float32)

    loss = dist._weighted_local_mse(K_pred, K_tgt, alpha=2.0, topk=None).item()

    # Manually computed:
    # Off-diagonal similarity mismatch = 0.75 - 0.8 = -0.05.
    # After weighting/normalization, MSE contribution is (0.05^2) averaged over 2 entries.
    assert loss == pytest.approx(0.0025, rel=1e-5)


def test_weighted_local_mse_topk_path_preserves_zero_when_kernels_match():
    K_tgt = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.0, 0.1, 0.8, 1.0],
        ],
        dtype=torch.float32,
    )
    # Convert target kernel in [0,1] back to a cosine-style [-1,1] kernel so
    # the (K+1)/2 remapping inside _weighted_local_mse perfectly matches.
    K_pred = (K_tgt * 2.0) - 1.0

    loss = dist._weighted_local_mse(K_pred, K_tgt, alpha=1.0, topk=1)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_local_weighted_alignment_loss_matches_weighted_mse(monkeypatch):
    patch = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    monkeypatch.setattr(dist, "sample_random_patches", lambda *a, **k: [patch])

    Phi = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    encoder = IdentityEncoder()
    alpha = 1.5

    Z_all = F.normalize(Phi.index_select(0, patch), dim=-1)
    K_pred = (Z_all @ Z_all.T).clamp(-1.0, 1.0)
    K_tgt, _ = dist.rbf_from_coords_local(coords, patch)
    expected = dist._weighted_local_mse(K_pred, K_tgt, alpha=alpha, topk=None).item()

    loss = dist.local_weighted_alignment_loss(
        encoder=encoder,
        Phi=Phi,
        Xcoords=coords,
        H=2,
        W=2,
        num_patches=1,
        max_ph=2,
        max_pw=2,
        min_ph=2,
        min_pw=2,
        alpha=alpha,
        topk=None,
    )
    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_local_weighted_alignment_loss_zero_when_no_valid_patches(monkeypatch):
    monkeypatch.setattr(
        dist, "sample_random_patches", lambda *a, **k: [torch.tensor([0, 1, 2])]
    )
    Phi = torch.randn(3, 4)
    coords = torch.zeros(3, 2)
    encoder = IdentityEncoder()

    loss = dist.local_weighted_alignment_loss(
        encoder=encoder,
        Phi=Phi,
        Xcoords=coords,
        H=1,
        W=3,
        num_patches=1,
        max_ph=1,
        max_pw=3,
        min_ph=1,
        min_pw=3,
        alpha=1.0,
        topk=None,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)

