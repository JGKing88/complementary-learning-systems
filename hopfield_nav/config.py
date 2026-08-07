"""Configuration dataclasses for hopfield_nav."""
from __future__ import annotations

from dataclasses import dataclass, field


def validate_train_config(cfg: "TrainConfig") -> None:
    """Cross-field checks on a TrainConfig that the dataclass alone can't catch.

    Raises ValueError on any silent-no-op combination so the user gets a clear
    failure instead of a quiet misconfiguration.
    """
    h = cfg.hopfield
    if not h.agent_can_store and h.auto_store_warmup > 0:
        raise ValueError(
            "auto_store_warmup > 0 has no effect when agent_can_store is False: "
            "the rollout uses a single shared Hopfield in that mode and skips "
            "the auto-store path entirely. Set agent_can_store=True or "
            "auto_store_warmup=0."
        )


@dataclass
class EnvConfig:
    size: int = 8
    speed: int = 1
    observation_size: int = 60
    time_penalty: float = 0.01
    movement_mode: str = "discrete"         # "discrete" | "continuous"
    continuous_scale: float = 1.0
    # When True, each (dx, dy) action is L2-normalized to a unit vector before
    # * continuous_scale, so step magnitude is fixed at continuous_scale. Both
    # ContinuousVecEnv (training) and ContinuousGridEnv.step (single-env eval)
    # honor this. Default False so the policy mean directly controls step size.
    continuous_normalize: bool = False
    # If set, env clips L2(action) to this max value before applying scale.
    # Soft cap on action magnitude. Only applies when continuous_normalize=False.
    max_action_norm: float | None = None
    # If set, env scales action UP to this min L2 value (direction preserved)
    # when ‖action‖ < min. Forces a minimum step size. Combined with
    # max_action_norm, action L2 is clamped to [min, max]. Only applies
    # when continuous_normalize=False.
    min_action_norm: float | None = None
    goals_active: bool = True               # When False: no +1 goal reward, no teleport on goal-reach. For pure-explore Phase A.
    goal_reward: float = 1.0                # +reward at goal cell when goals_active. Bumping >1 strengthens follow PPO updates vs explore reward signals (novelty + revisit).
    goal_radius: float = 0.5                # Euclidean radius around goal that counts as "at goal". Default 0.5 reproduces snap-equality on integer-snapped positions. Larger values fuzz the goal region; e.g. 1.0 includes 4-connected neighbor cells.
    # What a store writes when the agent is at goal but standing on a different
    # cell. Only reachable when goal_radius > 0.5 in continuous mode: at_goal
    # tests the float position while embeddings are read at the *snapped* cell,
    # so at radius 1.0 the agent can be at goal on a 4-connected neighbour and a
    # store there would write the NEIGHBOUR's embedding as the goal memory.
    # False (default) substitutes the goal cell's embedding instead, so the
    # stored pattern is the one navigation will later recall. True restores the
    # pre-2026-08 behavior, in which such a store wrote the neighbour; setup
    # warns when that is combined with goal_radius > 0.5. At goal_radius <= 0.5
    # the two are identical -- every at-goal position snaps to the goal cell.
    allow_offcell_store: bool = False


@dataclass
class VectorHashConfig:
    lambdas: list[int] = field(default_factory=lambda: [11, 12])
    Np: int = 1600
    Npos: int | None = None                 # None = prod(lambdas)
    thresh: float = 2.0
    c: float = 0.5
    # If True: build gbook + sbook (+ encoded_Phi after precompute); skip pbook, Wgp, Wps, self-test.
    static_vectorhash: bool = False


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
    revisit_penalty: float = 0.0          # -reward applied each step the agent occupies an already-visited cell (per-rollout visit count). Densifies the coverage gradient: novelty alone goes silent on revisits, this keeps signal alive late in a rollout.
    wall_penalty: float = 0.0             # -reward each step the agent occupies a grid edge cell (x or y in {0, size-1}). Counters the "perimeter-walk basin" learned when novelty rewards walking along edges (high coverage from wall-clip). Applied during the explore phase, alongside novelty.
    persistence_bonus: float = 0.0        # +reward × cos(action_t, action_{t-1}) per step, encouraging straight-line motion. Stateless dense alternative to revisit_penalty for explore-phase shaping. Applied during the explore phase only.
    novelty_scale_remaining: bool = False # If True, scale novelty by total_cells / n_remaining_unvisited. Late-game cells (rare) pay more than early-game cells, keeping gradient alive as coverage saturates.
    novelty_scale_cap: float = 10.0       # Upper bound on the remaining-scale multiplier, to avoid value-head instability from rare-cell jackpots.
    n_train_distractors: int = 0          # if >0, pre-populate each training env's per-env Hopfield with this many distractor patterns (from outside that env's region) at rollout start. Matches eval-time distractor setup so the training/eval distributions align.
    n_train_distractors_min: int = 0      # If n_train_distractors_max > 0, distractor count per rollout is sampled uniformly from [min, max]. Overrides the fixed n_train_distractors knob when max>0.
    n_train_distractors_max: int = 0      # See n_train_distractors_min. Set max>0 to enable variable-count distractors during training (count is uniform[min, max] per parallel rollout).
    epsilon_explore: float = 0.0          # per-step probability of replacing the sampled movement action with a uniform-random direction. Injected as an override; the agent re-scores log_prob under the current policy so PPO's importance ratio stays well-defined.
    epsilon_anneal_updates: int = 0       # linearly decay epsilon_explore from full→0 over this many updates (0 = constant)
    refresh_envs_each_update: bool = False  # if True, re-sample env_offsets at the start of every PPO update so each rollout buffer covers a different patch of the global scaffold. Variance-reduction trick: damps the seed-lottery on which scaffold positions the policy ever sees.
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
    input_prev_reward: bool = False         # Phase-2 enrichment: add prev step's reward as input channel
    input_sensory: bool = False
    input_hopfield_raw: bool = False        # Phase-2 enrichment: feed raw (unnormalized) q in continuous mode
    input_hopfield_multistep: list[int] = field(default_factory=list)  # If non-empty, project recall at these Hopfield iteration counts and pass each as 2-D extra input. Lets the policy read recall-convergence dynamics. Continuous mode only.
    input_goal_in_memory: bool = False      # Add 1-bit input indicating that the agent has stored at goal during this rollout (i.e., Hopfield content is trustworthy goal-direction). Lets policy distinguish explore (bit=0) from nav (bit=1) cleanly.
    # Linked to movement_mode
    hopfield_mode: str = "discrete"         # "discrete" (4-d) | "continuous" (2-d)
    movement_mode: str = "discrete"         # "discrete" (Categorical 4) | "continuous" (Gaussian 2)
    init_log_std: float = 0.0               # continuous policy: initial log std (default std=1.0)
    freeze_log_std: bool = False            # When True, movement_log_std is held at its initial value (no gradient). For Phase A: pin variance low so PPO loss directly pressures the policy mean instead of letting samples "hide" the mean.


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
    bce_detach_trunk: bool = False          # When True, BCE routes through store_head(features.detach())
                                            # so its gradient does NOT flow through the RNN trunk.
                                            # Runs 2/8/10 in Phase 1 suffered because BCE on shared
                                            # logits distorted the trunk and killed exploration.
    bce_pos_weight_cap: float = 0.0         # If > 0, cap BCE pos_weight at this value. Prevents the
                                            # fire@off pathology in Phase B (raw n_neg/n_pos ≈ 19
                                            # drove false-fire to 0.31-0.68 in V3 Phase B). Reasonable
                                            # cap is 5. 0 disables the cap (use raw imbalance ratio).
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    n_minibatches: int = 4                  # minibatches per epoch over the pooled rollout buffer


@dataclass
class BCConfig:
    """DAgger-style behavior cloning against a state-dependent oracle.

    Only active when TrainConfig.training_mode == "bc". Oracle:
      - Pre-memory (no stored pattern): novelty action (neighbor to unvisited cell).
      - At goal: store = 1.
      - Post-memory (off-goal): movement = Hopfield-derived direction (q_full).
    Student samples its own action (DAgger); loss is computed against the oracle.
    """
    lr: float = 3e-4
    store_weight: float = 1.0        # weight on store-head BCE vs. movement CE
    move_ent_coef: float = 0.0       # optional entropy bonus on movement logits
    epochs: int = 1                  # gradient epochs per rollout buffer
    n_minibatches: int = 4
    max_grad_norm: float = 1.0
    bce_pos_weight_cap: float = 0.0  # If > 0, cap pos_weight on the store BCE.
                                     # Mirrors PPOConfig.bce_pos_weight_cap so
                                     # at-goal events being rare doesn't blow up
                                     # pos_weight early in BC training.
    supervise_explore: bool = True   # False = mask out pre-memory nav labels
    novelty_fallback: str = "random" # when all neighbors visited: "random" | "stay"
    nav_weight: float = 1.0          # Per-step weight on trust_hop (post-store-at-goal Hopfield-direction) move labels. >1 upweights nav-following relative to abundant novelty labels — fights the dilution that lets the policy ignore the small q_full input channel.


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    vectorhash: VectorHashConfig = field(default_factory=VectorHashConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    # Training mode: "ppo" (default, unchanged behavior) | "bc" (DAgger supervised).
    training_mode: str = "ppo"
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
    union_cov_trials: int = 0           # DEPRECATED 2026-08-06, no longer read. evaluate_union_coverage was absorbed into evaluate_exploration, which now computes the union over its own rollouts -- so the union is taken over n_val_trials, not over a separate budget. Kept because checkpoints are keyed by field name and 309 run dirs carry it.
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
    # Which evaluators an in-training eval runs. "all" is the three-evaluator
    # pass every phased trainer has always done. "expl" runs exploration only,
    # for runs where navigation and goal-discovery are not merely uninteresting
    # but undefined: under a pure-explore schedule with explore_goals_off the
    # policy is never trained to reach or store a goal, so nav/disc measure
    # nothing -- and they are two thirds of the eval cost, which on a short run
    # is a large fraction of the whole run.
    eval_scope: str = "all"                 # "all" | "expl"
    # Step budget for in-training evals. None keeps the historical behavior of
    # following steps_per_rollout. They need to come apart whenever rollout
    # length is itself the variable: mean_coverage is cells / grid-cells, so a
    # run trained on 100-step rollouts would report it over 100 steps and a
    # 400-step run over 400, and the two numbers are not the same measurement.
    # Pinning this makes coverage comparable across a rollout-length sweep --
    # at the cost of an eval whose price no longer scales down with the run.
    eval_max_steps: int | None = None
    # Checkpoint cadence for the phased trainers, independent of eval_every.
    # None = follow eval_every, which is what they did unconditionally until
    # 2026-08: the per-update save sat inside the eval branch, so raising
    # eval_every to make a long run affordable also thinned the checkpoints
    # that analysis.trajectories draws its rows from. `hopfield_nav.train` has
    # always had its own save_every below and is unaffected.
    ckpt_every: int | None = None
    save_every: int = 100
    save_dir: str | None = None  # default resolved in train() from wandb name or timestamp
    seed: int = 0
    device: str = "cuda"
    recompute_interval: int = 1
    # Eval: replace Hopfield recall with the oracle direction in embedding
    # tangent space (same projection as the real signal) when the goal pattern
    # is in memory, to separate policy error from readout error. See eval.agent_step.
    hopfield_oracle: bool = False
    # Eval: when the goal is in memory, replace the policy's movement with a
    # greedy best step toward the goal (discrete) or a unit step in that
    # direction (continuous). Isolates policy error from other components.
    action_oracle: bool = False
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hopfield-nav"

    # --- train_navigate's schedule -----------------------------------------
    # Read only by `hopfield_nav.train_navigate`; harmless defaults everywhere
    # else. They live on TrainConfig rather than as bare function arguments so
    # that `asdict(cfg)` -- which is what the checkpoint and `run.json` record --
    # says what regime the run actually trained under. Before this they reached
    # the manifest only as raw `argv`, which meant a run could not describe
    # itself and a child run could not inherit its parent's recipe.
    #
    # `schedule` is the stage list; see hopfield_nav/training/stages.py for the
    # grammar. Everything below it is a run-wide default that a stage may
    # override.
    # --- env generator (phase 3) ------------------------------------------
    # Domains reach the config as compact strings so `asdict(cfg)` stays
    # JSON-native and the checkpoint carries them intact -- same shape as
    # `schedule` above. Parsed at startup by world/domains.py.
    env_generator: bool = False             # draw envs from declared domains instead of the historical placement path
    place_region: str = "anywhere"          # 'anywhere' | 'rect:X0,Y0,W,H'
    goal_region: str = "any"                # 'any' | 'ring:W' | 'interior:W' | 'quadrant:Q'
    wall_seeds: str = "0,10000000"          # 'LO,HI' -- the range training draws wall seeds from
    place_margin: int | None = None         # edge-to-edge train/val clearance; None derives it from the scaffold's own cosine curve
    goal_val_frac: float = 0.2              # share of goal cells reserved for validation when goals refresh
    schedule: str | None = None
    novelty_anneal: bool = False            # linearly scale novelty_reward -> 0 across the whole run
    epsilon_explore: float = 0.0            # per-step chance of a uniform-random move, explore regime only
    epsilon_anneal_updates: int = 0         # linearly scale epsilon_explore -> 0 over this many updates; 0 = constant
    explore_goals_off: bool = False         # explore-regime envs emit no goal reward and never teleport
    randomize_goal_per_rollout: bool = False  # re-draw the goal each explore rollout, breaking the memorization shortcut
    n_train_distractors_min: int = 0        # non-goal patterns preloaded per exploit rollout
    n_train_distractors_max: int = 0
    n_train_emp_distractors_min: int = 0    # ditto per explore rollout (no goal among them)
    n_train_emp_distractors_max: int = 0
    n_train_distractors_max_end: int | None = None      # curriculum target for the two maxima
    n_train_emp_distractors_max_end: int | None = None
    distractor_curriculum_updates: int = 0  # updates over which the maxima ramp start -> end; 0 = no ramp
    log_std_anneal_start_update: int = 0    # window over which movement_log_std is driven to its target
    log_std_anneal_end_update: int = 0
    log_std_anneal_target: float | None = None


@dataclass
class RNNAgentConfig:
    """Vanilla RNN policy: GRU + single move head. No store, no value, no Hopfield.

    Used by ``train_rnn.py`` as the no-memory control baseline.
    """
    hidden_size: int = 128
    num_rnn_layers: int = 1
    dropout: float = 0.0
    movement_mode: str = "discrete"         # "discrete" (Categorical 4) | "continuous" (Normal 2)
    init_log_std: float = 0.0               # continuous policy: initial log std
    freeze_log_std: bool = False
    # Optional auxiliary input channels (sensory codebook vector is always on).
    input_prev_action: bool = False
    input_prev_reward: bool = False
    input_grid_state: bool = False          # current (x, y) cell normalized to [0, 1]^2


@dataclass
class RNNBCConfig:
    """BC update for the vanilla RNN: CE on move action only (no store BCE)."""
    lr: float = 1e-3
    move_ent_coef: float = 0.0
    epochs: int = 4
    n_minibatches: int = 4
    max_grad_norm: float = 1.0
    only_train_on_reached: bool = False     # drop trajectories whose rollout never reached goal


@dataclass
class RNNTrainConfig:
    """Top-level config for the vanilla-RNN BC continual-learning baseline.

    Independent from TrainConfig: no encoder, no VectorHash, no Hopfield, no
    world abstraction. The continual unit is a single GridEnv (each instance
    has its own codebook + goal).
    """
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: RNNAgentConfig = field(default_factory=RNNAgentConfig)
    bc: RNNBCConfig = field(default_factory=RNNBCConfig)
    # Mode: "sequential" trains envs one-by-one (continual). "mixed" pools
    # rollouts from all envs each update (pretraining). "finetune" loads a
    # checkpoint and runs sequential.
    mode: str = "sequential"
    n_envs: int = 4
    updates_per_env: int = 100              # sequential / finetune mode
    n_updates: int = 1000                   # mixed mode only
    batch_envs: int = 16                    # parallel rollouts per env
    steps_per_rollout: int = 64
    eval_every: int = 25                    # within-env training log cadence
    n_eval_trials: int = 32                 # parallel eval trials per env
    eval_max_steps: int = 64
    seed: int = 0
    device: str = "cuda"
    save_dir: str | None = None
    load_checkpoint: str | None = None
    use_wandb: bool = False
    wandb_project: str = "hopfield-nav-rnn"
    plot_smooth_window: int = 1             # rolling-mean window for forgetting/steps_to_goal plots; 1 = no smoothing
    fwhm_ratio: float = 0.25                # spatial smoothing for gbook lookup (only used when input_grid_state)
    lambdas: list[int] = field(default_factory=lambda: [11, 12])  # VectorHash module periods (only used when input_grid_state)


@dataclass
class PhasedConfig:
    """Per-phase overrides for train_phased.py.

    Each phase gets its own update budget and a narrow set of overrides that
    switch: (a) which heads are frozen, (b) the Hopfield init_mode (empty vs
    pre_stored), (c) aux schedule flags (auto_store_warmup, auto_nav_warmup).
    Everything else inherits from TrainConfig.
    """
    # Phase 1: store head pretraining (BCE only, trunk detached, move frozen).
    phase1_updates: int = 20
    phase1_lr: float = 3e-4
    phase1_bce_weight: float = 1.0          # Weight on BCE loss (only loss in phase 1)
    phase1_force_store_at_goal: bool = True # Always force-store at goal during phase 1

    # Phase 2: follow pretraining (pre_stored goal, auto_nav teacher forces move head).
    phase2_updates: int = 100
    phase2_auto_nav_warmup: int = 30        # First N updates of phase 2 use teacher forcing
    phase2_freeze_store: bool = True        # Don't update store head during phase 2

    # Phase 3: explore pretraining (empty Hopfield, store frozen).
    phase3_updates: int = 200
    phase3_init_log_std: float = -0.5       # Tight Gaussian for directional walks
    phase3_freeze_store: bool = True

    # Phase 4: compose (all heads unfrozen, small permanent BCE).
    phase4_updates: int = 300
    phase4_lr: float = 1e-4                 # Smaller LR to preserve what phases 1-3 taught
    phase4_bce_weight: float = 0.1          # Small permanent BCE as stability anchor
    phase4_bce_detach_trunk: bool = True    # BCE gradient isolated to store head
