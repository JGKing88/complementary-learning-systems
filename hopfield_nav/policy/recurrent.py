"""The policy trunk, built from config in one place.

`NavAgent` and `RNNAgent` each used to construct their own `nn.GRU` from an
identical five-line block. That duplication was harmless while there was one
cell type; it stops being harmless the moment there are three, because the two
agents can then disagree about what `--rnn_cell` means. One factory, two
callers.

Three cores are reachable:

  ``gru``                  -- `nn.GRU`, the historical default, unchanged.
  ``rnn`` + tanh/relu      -- `nn.RNN`, which is cuDNN-backed like the GRU.
  ``rnn`` + softplus       -- `SoftplusRNN` below, a Python recurrence.

Everything downstream reaches the trunk through four contracts -- `input_size`,
`parameters()`, an `(num_layers, B, hidden)` hidden state, and equivalence
between a T-step call and T single-step calls carrying `h`. `SoftplusRNN`
inherits from `nn.RNN` precisely so the first three come for free rather than
being re-derived and drifting; the fourth is pinned by a test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CELLS = ("gru", "rnn")
NONLINEARITIES = ("tanh", "relu", "softplus")


def validate_recurrent_core(cell: str, nonlinearity: str) -> None:
    """Reject the combinations that would otherwise fail silently or late.

    Called from `build_recurrent_core`, so an invalid core is unconstructible
    whichever entry point asked for it -- `train_rnn` builds an
    `RNNTrainConfig`, which no `validate_train_config` ever sees.
    """
    if cell not in CELLS:
        raise ValueError(
            f"rnn_cell={cell!r} is not one of {CELLS}.")
    if nonlinearity not in NONLINEARITIES:
        raise ValueError(
            f"rnn_nonlinearity={nonlinearity!r} is not one of {NONLINEARITIES}.")
    if cell == "gru" and nonlinearity != "tanh":
        raise ValueError(
            f"rnn_cell='gru' has no selectable nonlinearity -- a GRU's gates "
            f"are sigmoid and its candidate is tanh by construction, so "
            f"rnn_nonlinearity={nonlinearity!r} would be silently ignored. "
            f"Pass --rnn_cell rnn to choose a nonlinearity.")


class SoftplusRNN(nn.RNN):
    """`nn.RNN`'s parameters and shapes, with softplus in place of tanh.

    Subclasses rather than wraps. `nn.RNN.__init__` registers exactly the four
    parameters per layer this recurrence needs, under exactly the names a
    checkpoint should carry, initialized from exactly the right distribution --
    so a tanh checkpoint loads into a softplus model and back, which is what
    makes the nonlinearity an ablation axis rather than a fork. It also means
    `input_size`, `hidden_size`, `num_layers` and `parameters()` are inherited,
    and every existing reader of `agent.rnn` keeps working untouched.

    Only `forward` is overridden. cuDNN cannot fuse a softplus recurrence, so
    this is a Python loop over T -- the input projection is hoisted out of it
    (one matmul over the flattened sequence per layer), leaving a single
    (B, H) x (H, H) matmul per step.

    Note that softplus is unbounded above and strictly positive: unlike tanh
    there is no contraction toward zero, and h sits at a positive DC offset of
    roughly softplus(0) = 0.693 per unit. Measured over 400 steps at the
    default init the norm is stable (it settles near 0.693*sqrt(H) and stays
    there), but the feature distribution the heads see is shifted well off
    zero and scales with sqrt(hidden_size) -- which is a real difference from
    tanh, not a bug.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        # `nonlinearity` is what the base class would apply; we never call its
        # forward, so the value only has to be one it accepts.
        super().__init__(
            input_size, hidden_size, num_layers=num_layers,
            nonlinearity="tanh", batch_first=batch_first, dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, input_size), h: (num_layers, B, hidden) or None.

        Returns (output, h_next) with output (B, T, hidden) and h_next
        (num_layers, B, hidden) -- `nn.RNN`'s batch_first contract.
        """
        if x.dim() != 3:
            raise ValueError(
                f"SoftplusRNN expects a batched (B, T, input_size) sequence, "
                f"got shape {tuple(x.shape)}.")
        B, T, _ = x.shape
        if h is None:
            h = x.new_zeros(self.num_layers, B, self.hidden_size)

        layer_in = x
        h_next: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            w_ih = getattr(self, f"weight_ih_l{layer}")
            w_hh = getattr(self, f"weight_hh_l{layer}")
            b_ih = getattr(self, f"bias_ih_l{layer}")
            b_hh = getattr(self, f"bias_hh_l{layer}")

            # Input projection for every timestep at once: the loop below then
            # only carries the recurrent term.
            gi = torch.addmm(
                b_ih, layer_in.reshape(-1, layer_in.shape[-1]), w_ih.t(),
            ).view(B, T, self.hidden_size)

            h_l = h[layer]
            steps = []
            for t in range(T):
                h_l = F.softplus(gi[:, t] + torch.addmm(b_hh, h_l, w_hh.t()))
                steps.append(h_l)
            layer_in = torch.stack(steps, dim=1)

            # nn.RNN drops between layers, never on the final output.
            if self.dropout > 0.0 and layer < self.num_layers - 1:
                layer_in = F.dropout(layer_in, self.dropout, self.training)
            h_next.append(h_l)

        return layer_in, torch.stack(h_next, dim=0)


def build_recurrent_core(cfg, input_dim: int) -> nn.Module:
    """The trunk `cfg` asks for.

    Duck-typed on ``hidden_size`` / ``num_rnn_layers`` / ``dropout`` /
    ``rnn_cell`` / ``rnn_nonlinearity`` so that `AgentConfig` and
    `RNNAgentConfig` -- separate dataclasses that share these field names --
    both work without this module importing either.

    `getattr` defaults cover configs reconstructed from checkpoints written
    before these fields existed; `cfg_from_checkpoint` already fills them, so
    this is belt-and-braces for direct constructions in analysis code.
    """
    cell = getattr(cfg, "rnn_cell", "gru")
    nonlinearity = getattr(cfg, "rnn_nonlinearity", "tanh")
    validate_recurrent_core(cell, nonlinearity)

    layers = cfg.num_rnn_layers
    kwargs = dict(
        num_layers=layers,
        batch_first=True,
        dropout=cfg.dropout if layers > 1 else 0.0,
    )
    if cell == "gru":
        return nn.GRU(input_dim, cfg.hidden_size, **kwargs)
    if nonlinearity == "softplus":
        return SoftplusRNN(input_dim, cfg.hidden_size, **kwargs)
    return nn.RNN(input_dim, cfg.hidden_size, nonlinearity=nonlinearity, **kwargs)
