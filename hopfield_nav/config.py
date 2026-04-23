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
    # If True: only build gbook (+ encoded_Phi after precompute); skip pbook, Wgp, Wps, self-test.
    gbook_only: bool = False


@dataclass
class HopfieldConfig:
    beta: float | None = None               # None = use encoder_gain at train startup
    alpha: float = 1.0
    steps: int = 1
    init_mode: str = "empty"                # "empty" | "pre_stored"
    agent_can_store: bool = True
    store_cost: float = 0.0               # reward penalty per store action
    store_bonus: float = 0.0              # reward bonus for storing at goal
    auto_store_warmup: int = 0            # updates during which at-goal forces a store regardless of agent action
    auto_nav_warmup: int = 0              # updates during which movement is force-copied from the Hopfield suggestion whenever that env has any stored memory (teacher forcing for navigation)
    aux_anneal_updates: int = 0           # linearly decay store_bonus + store_bc_weight from full→0 over these many updates (0 = no decay)
    novelty_reward: float = 0.0           # +reward for first-visit to a snapped cell during explore phase (per-rollout visit count)
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
    init_log_std: float = 0.0               # continuous policy: initial log std (default std=1.0)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    store_ent_coef: float = 0.05            # separate entropy bonus for store action
    store_bc_weight: float = 0.0            # auxiliary BCE loss weight: BCE(store_logits, at_goal)
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    n_minibatches: int = 4                  # minibatches per epoch over the pooled rollout buffer


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
    # Eval: a single eval world with its own VectorHash scaffold, decoupled from
    # num_worlds. Built once at startup and reused for every eval pass.
    num_val_envs: int = 2
    n_val_trials: int = 32
    val_n_distractors_list: list[int] = field(default_factory=lambda: [0])
    # Realistic end-of-training eval: one Hopfield accumulates across envs
    # sequentially. Set 0 to skip. Only runs at the end of training.
    realistic_steps_per_env: int = 1000
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
    save_dir: str | None = None  # default resolved in train() from wandb name or timestamp
    seed: int = 0
    device: str = "cuda"
    recompute_interval: int = 1
    # Eval: replace Hopfield recall with the oracle direction in embedding
    # tangent space (same projection as the real signal) when the goal pattern
    # is in memory, to separate policy error from readout error. See eval._agent_step.
    hopfield_oracle: bool = False
    # Eval: when the goal is in memory, replace the policy's movement with a
    # greedy best step toward the goal (discrete) or a unit step in that
    # direction (continuous). Isolates policy error from other components.
    action_oracle: bool = False
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hopfield-nav"
