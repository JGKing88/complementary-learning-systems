# Retired config schedules

Phase-schedule dataclasses deleted from `hopfield_nav/config.py` because nothing
imported them. They are recorded here rather than only in git history: the
docstrings are the *only* description of what these schedules were, and the
`EXPERIMENTS_PHASE2_V2.md` that `PhasedConfigV2` cites was never written.

None of this is loadable. A checkpoint stores `asdict(TrainConfig)`, and none of
these classes was ever a field of `TrainConfig`, so no saved run depends on them.

## `PhasedConfigV2`

Deleted 2026-08-06 (phase 6). Superseded by `PhasedConfig`, which
`train_phased.py` uses, and by the standalone `train_phase_a_only` /
`train_phase_b_only` entry points that replaced the V2 A/B/C split.

```python
@dataclass
class PhasedConfigV2:
    """V2: follow+explore interleaved → store pretrain → compose (frozen trunk).

    Rationale in EXPERIMENTS_PHASE2_V2.md. Three phases:

      A. Interleaved trunk training: half the envs run with pre_stored goal
         (follow gradient), half with empty hopfield (explore gradient).
         Trunk + move + value unfrozen; store head frozen at init.
         ``init_log_std=-0.5`` should be set on AgentConfig throughout.
      B. Store head pretrain on frozen Phase-A trunk. Only store_head
         trainable; BCE with features.detach(). Short (~30 updates).
      C. Compose. Soft-frozen trunk (tiny LR) + trainable heads. Agent-driven
         store; hopfield accumulates within each rollout. Permanent detached
         BCE anchors store head.
    """
    # Phase A — interleaved follow+explore
    phase_a_updates: int = 500
    phase_a_interleave_ratio: float = 0.5   # fraction of envs_per_world that run pre_stored
    phase_a_lr: float = 3e-4

    # Phase B — store pretrain on frozen trunk
    phase_b_updates: int = 30
    phase_b_lr: float = 3e-4
    phase_b_bce_weight: float = 1.0

    # Phase C — compose with soft-frozen trunk
    phase_c_updates: int = 300
    phase_c_lr_trunk: float = 1e-5          # Set to 0.0 for hard freeze.
    phase_c_lr_move: float = 1e-5           # Move head shares trunk's caution.
    phase_c_lr_heads: float = 1e-4          # Store + value adapt normally.
    phase_c_bce_weight: float = 0.1
    phase_c_bce_detach_trunk: bool = True
```
