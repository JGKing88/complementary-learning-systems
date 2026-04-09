"""Configuration dataclasses for hopfield_nav."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    size: int = 8
    speed: int = 1
    observation_size: int = 512
    time_penalty: float = 0.01
    movement_mode: str = "discrete"         # "discrete" | "continuous"
    continuous_scale: float = 1.0
    continuous_normalize: bool = True


@dataclass
class VectorHashConfig:
    lambdas: list[int] = field(default_factory=lambda: [11, 12])
    Np: int = 1600
    Npos: int | None = None                 # None = prod(lambdas)
    thresh: float = 2.0
    c: float = 0.5


@dataclass
class HopfieldConfig:
    beta: float = 2.0
    alpha: float = 1.0
    steps: int = 1
    init_mode: str = "empty"                # "empty" | "pre_stored"
    agent_can_store: bool = True
    store_cost: float = 0.0               # reward penalty per store action
    store_bonus: float = 0.0              # reward bonus for storing at goal
    # embed_dim derived from encoder checkpoint at startup


@dataclass
class AgentConfig:
    hidden_size: int = 128
    num_rnn_layers: int = 1
    dropout: float = 0.0
    # What the RNN sees (configurable)
    input_encoded_state: bool = True
    input_hopfield_signal: bool = True
    input_prev_action: bool = False
    # Linked to movement_mode
    hopfield_mode: str = "discrete"         # "discrete" (4-d) | "continuous" (2-d)
    movement_mode: str = "discrete"         # "discrete" (Categorical 4) | "continuous" (Gaussian 2)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    store_ent_coef: float = 0.05            # separate entropy bonus for store action
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    vectorhash: VectorHashConfig = field(default_factory=VectorHashConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # Encoder
    encoder_checkpoint: str = ""
    encoder_gain: float | None = None
    fwhm_ratio: float = 0.25
    # World structure
    num_worlds: int = 1
    envs_per_world: int = 4
    val_envs_per_world: int = 2
    # Rollout
    batch_envs: int = 16
    steps_per_rollout: int = 64
    explore_steps: int | None = None        # None = single-phase; set to enable two-phase rollout
    # Checkpoint loading
    load_checkpoint: str | None = None
    # Training
    n_updates: int = 1000
    eval_every: int = 50
    save_every: int = 100
    save_dir: str = "checkpoints"
    seed: int = 0
    device: str = "cuda"
    recompute_interval: int = 1
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hopfield-nav"
