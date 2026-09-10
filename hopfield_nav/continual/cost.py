"""How much work an update actually does, counted rather than derived.

The frontier table says every method spends "200 gradient steps and 200
episodes per environment". Gradient *steps* are constant; the work inside one
is not, and the spread is large enough to matter -- Experience Replay at
``replay_batches=4`` concatenates its replayed trajectories into the training
batch, so one of its gradient steps pushes five sequences through the trunk
where naive SGD pushes one.

Deriving that from the configs is how it goes wrong. Reading
``replay_batches`` suggests CLEAR and DER++ cost the same as ER at the same
setting; they do not, because their replayed batches go through the main loop
*and* get further forwards in ``aux_loss``. EWC looks free per update and is
not, because it runs one backward per stored trajectory at every task
boundary. So this counts.

The unit is **trunk-steps**: sequences x timesteps pushed through the recurrent
core. It is hardware-independent, exactly countable, and the thing that
actually scales -- unlike wall-clock, which at ``batch_envs=1`` is dominated by
environment simulation and by whatever else shares the node.

Forward and backward are counted separately. A grad-enabled forward is charged
to both, because the backward traverses the same graph; a ``no_grad`` teacher
forward is charged only to the forward.

The counter is module-level and **never reset**. Callers record the running
total, so per-update cost is a difference and a task-boundary spike appears as
a jump between two updates rather than needing its own bookkeeping. A global is
the wrong default, and it is the right call here: the alternative is threading
an accumulator through ``bc_rnn_update`` and six method hooks, which is a lot
of signature churn for an instrument that is read once per update.
"""
from __future__ import annotations


class TrunkCounter:
    """Running totals of trunk-steps, forward and backward."""

    def __init__(self) -> None:
        self.fwd = 0
        self.bwd = 0

    def reset(self) -> None:
        """Only for tests, and for a driver that wants per-run totals."""
        self.fwd = 0
        self.bwd = 0

    def add(self, obs, *, backward: bool) -> None:
        """Charge one pass over ``obs``, shaped (batch, time, ...).

        Silently ignores anything without at least two dimensions rather than
        raising: this is an instrument, and an instrument that can halt a
        14-hour wave because it met an unexpected shape is worse than one that
        under-reports.
        """
        shape = getattr(obs, "shape", None)
        if shape is None or len(shape) < 2:
            return
        n = int(shape[0]) * int(shape[1])
        self.fwd += n
        if backward:
            self.bwd += n

    def snapshot(self) -> dict:
        return {"trunk_fwd_steps": self.fwd, "trunk_bwd_steps": self.bwd}


#: The one counter. See the module docstring for why it is module-level.
COUNTER = TrunkCounter()


__all__ = ["TrunkCounter", "COUNTER"]
