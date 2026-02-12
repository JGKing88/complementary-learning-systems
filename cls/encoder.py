from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class GridEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, nonlinearity: str = "gelu", output_nonlinearity: str = "tanh", gain: float = 1.0):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(nonlinearity.lower(), nn.GELU)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act(),
            nn.Linear(hidden, out_dim),
        )
        self.output_nonlinearity = output_nonlinearity
        self.gain = gain

    def forward(self, x: torch.Tensor, gain: float | None = None) -> torch.Tensor:
        z = self.net(x)                         # [B, out_dim]
        g = gain if gain is not None else self.gain
        if self.output_nonlinearity == "tanh":
            z = torch.tanh(g * z)
        elif self.output_nonlinearity == "sigmoid":
            z = torch.sigmoid(g * z)
        z = F.normalize(z, p=2, dim=-1)          # project to unit sphere
        return z


class GridEncoderCNN(nn.Module):
    def __init__(
        self,
        lambdas: list[int],
        hidden_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int = 3,
        num_hidden_layers: int = 1,
        hidden_dim = 128,
        out_dim: int = 128,
        nonlinearity: str = "gelu",
        output_nonlinearity: str = "tanh",
        gain: float = 1.0,
    ):
        super().__init__()

        self.lambdas = lambdas
        self.out_dim = out_dim
        self.output_nonlinearity = output_nonlinearity
        self.gain = gain
        self.hidden_dim = hidden_dim
        self.num_modules = len(lambdas)
        self.max_lambda = max(lambdas)

        # module ranges
        self.module_ranges = []
        start = 0
        for l in lambdas:
            end = start + l * l
            self.module_ranges.append((start, end, l))
            start = end

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(
            nonlinearity.lower(), nn.GELU
        )

        conv_layers = []
        in_ch = self.num_modules
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_ch, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(act_fn())
            in_ch = hidden_channels

        self.convs = nn.Sequential(*conv_layers)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        hidden_layers = []
        if num_hidden_layers == 0:
            hidden_layers.append(nn.Linear(hidden_channels * 4 * 4, out_dim))
        else:
            hidden_layers.append(nn.Linear(hidden_channels * 4 * 4, hidden_dim))
            hidden_layers.append(act_fn())
            for _ in range(num_hidden_layers - 1):
                hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
                hidden_layers.append(act_fn())
            hidden_layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*hidden_layers)

    def _reshape_single_g(self, g: torch.Tensor, channel_offset: int, out: torch.Tensor) -> None:
        batch_size = g.shape[0]

        for m_idx, (start, end, l) in enumerate(self.module_ranges):
            module_flat = g[:, start:end]
            module_2d = module_flat.view(batch_size, l, l)

            # compute top-left corner for centering
            pad = self.max_lambda - l
            top = pad // 2
            left = pad // 2

            out[
                :,
                channel_offset + m_idx,
                top : top + l,
                left : left + l,
            ] = module_2d

    def reshape_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = torch.zeros(
            batch_size,
            self.num_modules,
            self.max_lambda,
            self.max_lambda,
            device=x.device,
            dtype=x.dtype,
        )
        self._reshape_single_g(x, 0, out)
        return out

    def forward(self, x: torch.Tensor, gain: Optional[float] = None) -> torch.Tensor:
        squeeze_time = False
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            squeeze_time = True

        x_2d = self.reshape_to_2d(x)
        features = self.convs(x_2d)                  # [B, C, H, W]
        features = self.pool(features)              # [B, C, 4, 4]
        features = features.flatten(1)              # [B, C*16]

        z = self.mlp(features)

        g = gain if gain is not None else self.gain
        if self.output_nonlinearity == "tanh":
            z = torch.tanh(g * z)
        elif self.output_nonlinearity == "sigmoid":
            z = torch.sigmoid(g * z)

        z = F.normalize(z, p=2, dim=-1)

        if squeeze_time:
            z = z.reshape(B, T, -1)

        return z
    
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Remap legacy fc1/fc2 keys → mlp.0/mlp.2
        key_map = {
            "fc1.weight": "mlp.0.weight",
            "fc1.bias":   "mlp.0.bias",
            "fc2.weight": "mlp.2.weight",
            "fc2.bias":   "mlp.2.bias",
        }
        remapped = {}
        for k, v in state_dict.items():
            remapped[key_map.get(k, k)] = v
        return nn.Module.load_state_dict(self, remapped, strict=strict, **kwargs)