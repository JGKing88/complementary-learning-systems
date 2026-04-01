# cls - Complementary Learning Systems

A codebase for training navigation agents on discrete grid environments using **vector hash** memory systems inspired by grid cells and place cells in the hippocampal formation.

## Overview

This project implements:
1. **Grid-based navigation environments** (`WMEnv`, `GridWMEnv`) with codebook-based observations
2. **VectorHash memory system** - a biologically-inspired encoding that maps sensory observations to modular grid cell representations
3. **Policy training** using supervised imitation learning or PPO reinforcement learning (`train.py`)
4. **Action classifier training** from (start, end) state pairs (`train_action_classifier.py`)
5. **Grid encoders** (MLP and CNN) for learning normalized embeddings of grid cell activations
6. **Vectorized environments** for efficient batched rollouts

---

## Installation

```bash
# Install as editable package
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With visualization
pip install -e ".[viz]"
```

---

## Architecture Overview

```
cls/
├── __init__.py          # Public API: WMEnv, Position, Vector2, FovFunction
├── types.py             # Type definitions
├── models.py            # Neural network agents (GRU, MLP, GridCNN, Agent)
├── hopfield.py          # Hopfield associative memory
├── encoder.py           # GridEncoder (MLP) and GridEncoderCNN
├── envs/
│   └── environments.py  # WMEnv, GridWMEnv, WMVecEnv, GridWMVecEnv
├── utils/
│   └── GridUtils.py     # VectorHash class for memory encoding
└── vectorhash/          # Core vector hash utilities (see below)

train.py                 # Policy training (supervised / PPO)
train_action_classifier.py  # Action classifier training
notebooks/
└── train_dist_encoder.py   # Encoder training script
encoders/                # Saved encoder checkpoints
action_classifiers/      # Saved action classifier checkpoints
```

---

## Core Components

### 1. Types (`cls/types.py`)

Basic type definitions used throughout:

```python
Position = Tuple[int, int]        # (x, y) grid coordinate
Vector2 = Tuple[int, int]         # (dx, dy) action/heading vector
FovFunction = Callable[[int, int], int]  # FOV calculation function
```

### 2. Environments (`cls/envs/environments.py`)

#### `WMEnv` - Base Navigation Environment

A discrete square grid environment with codebook-based observations.

```python
from cls import WMEnv

env = WMEnv(
    size=8,              # Grid size (8x8)
    speed=1,             # Steps per action
    seed=42,             # Random seed
    observation_size=64, # Binary observation code length
    time_penalty=0.01,   # Penalty per timestep
    use_headings=False,  # If False, observation is heading-invariant
)

# Reset and get initial state
pos, goal, obs, reward = env.reset()

# Take an action (cardinal direction)
pos, goal, obs, reward = env.step((1, 0))  # Move East

# Get optimal action toward goal
best_action = env.best_action_to_goal()
```

**Key features:**
- **Coordinate system**: Origin at bottom-left `(0,0)`, x increases right, y increases up
- **Observations**: Pre-generated binary codes indexed by `(position, heading)` - stored in `_codebook` array of shape `(size, size, 4, observation_size)`. When `use_headings=False` (default), the same code is shared across all headings at each position
- **Heading**: One of 4 cardinal directions: `(0,1)` North, `(1,0)` East, `(0,-1)` South, `(-1,0)` West
- **Actions**: Vector `(dx, dy)` with components in `{-1, 0, 1}`

#### `GridWMEnv` - Environment with VectorHash Integration

Extends `WMEnv` to transform raw observations through a VectorHash memory system:

```python
from cls.envs.environments import GridWMEnv
from cls.utils.GridUtils import VectorHash

env = GridWMEnv(
    size=8,
    speed=1,
    input_type="g_idx",  # Output format: "g_hot", "g_idx", "s", or "p"
)

# Setup VectorHash (see below)
vh = VectorHash(Np=1600, lambdas=[11, 12], size=8)
vh.initiate_vectorhash([env])

# Now env.obs() returns transformed features
obs = env.obs()  # Shape depends on input_type
```

**Input types:**
- `"g_hot"`: One-hot grid cell activations (size: `sum(λ²)` for each λ in lambdas)
- `"g_idx"`: Grid phase indices per module (size: `2 * len(lambdas)`)
- `"s"`: Sensory representation
- `"p"`: Place cell representation
- `"encoded_g"`: Grid activations projected through a pretrained `GridEncoder` or `GridEncoderCNN` (requires `--encoder_weights`)

#### Vectorized Environments

`WMVecEnv` and `GridWMVecEnv` provide batched operations for efficient parallel rollouts:

```python
from cls.envs.environments import GridWMVecEnv

vec_env = GridWMVecEnv(base_env=env, batch_size=64)
vec_env.reset_all()

# Batched operations
obs_batch = vec_env.obs_batch(indices=[0, 1, 2, 3])
actions = vec_env.best_action_to_goal_batch(indices=[0, 1, 2, 3])
curs, goals, rewards, dones = vec_env.step_batch(indices, action_vectors)
```

---

### 3. VectorHash Memory System (`cls/utils/GridUtils.py`)

The `VectorHash` class implements a memory encoding system inspired by the hippocampal-entorhinal circuit:

```python
from cls.utils.GridUtils import VectorHash

vh = VectorHash(
    Np=1600,           # Number of place cells
    lambdas=[11, 12],  # Grid cell module periods
    size=8,            # Environment grid size
)

# Initialize with list of environments
vh.initiate_vectorhash(envs)

# Recall: sensory → place → grid transformation
s_out, p_out, g_out = vh.recall(observation)
```

#### How VectorHash Works

The system models three interacting populations:

1. **Sensory cells (s)**: Raw binary observations from the environment
2. **Place cells (p)**: Sparse, distributed representations of locations  
3. **Grid cells (g)**: Modular periodic representations

**Initialization flow:**

```
1. setup_scaffold():
   - Generate grid codebook (gbook): modular grid patterns for each position
   - Create random weights Wpg (place ← grid)
   - Compute place codebook: pbook = nonlin(Wpg @ gbook)
   - Train Wgp (grid ← place) via Hebbian learning

2. setup_envs():
   - Map environment observations to positions in abstract grid space
   - Train Wsp, Wps (sensory ↔ place) via pseudoinverse learning
```

**Recall flow:**

```
observation → Wps → p_in → Wgp → g_in → [module-wise winner-take-all] → g_out
                                                      ↓
                         s_out ← Wsp ← p_out ← Wpg ← g_out
```

#### Key Weight Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| `Wpg` | `(Np, Ng)` | Grid → Place projection (random, sparse) |
| `Wgp` | `(Ng, Np)` | Place → Grid (Hebbian trained) |
| `Wsp` | `(Ns, Np)` | Place → Sensory (pseudoinverse trained) |
| `Wps` | `(Np, Ns)` | Sensory → Place (pseudoinverse trained) |

#### Grid Cell Modules

Grid cells are organized into **modules** with different spatial periods (lambdas). Each module:
- Has `λ²` cells for period `λ`
- Forms a 2D toroidal representation
- Uses winner-take-all within the module during recall

```python
lambdas = [11, 12]  # Two modules with periods 11 and 12
Ng = 11² + 12² = 121 + 144 = 265  # Total grid cells
Npos = 11 * 12 = 132  # Total unique positions representable
```

---

### 4. Neural Network Models (`cls/models.py`)

#### `Agent` - Policy Network

```python
from cls.models import Agent

agent = Agent(
    input_size=64,           # Observation dimension
    hidden_size=128,         # Hidden layer size
    num_model_layers=1,      # Backbone depth
    num_actions=4,           # Output actions (N, E, S, W)
    model_class="GRU",       # "GRU", "MLP", or "CNN"
    encoder_dim=64,          # Optional encoder output dim
    num_encoder_layers=2,    # Number of encoder layers
    lambdas=[11, 12],        # Grid module periods (required for CNN backbone)
)

logits, values, h_next = agent(obs_tensor, hidden_state)
```

#### Architecture

```
Input (B, T, input_size)
    │
    ▼
┌─────────────────────────────────────────────┐
│ ENCODER (optional, if num_encoder_layers>0) │
│   [Linear → ReLU] × num_encoder_layers      │
└─────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ BACKBONE (GRU, MLP, or CNN)                      │
│                                                  │
│ GRU: nn.GRU (internal sigmoid/tanh gates)        │
│      ├─ input: (B, T, dim)                       │
│      └─ output: features + hidden state          │
│                                                  │
│ MLP: [Linear → Dropout → ReLU] × num_layers      │
│      └─ output: features (no hidden state)       │
│                                                  │
│ CNN (GridCNNBackbone): reshapes g_hot → 2D        │
│      ├─ [Conv2d → ReLU] × num_layers             │
│      ├─ AdaptiveAvgPool2d(1) → flatten            │
│      └─ output: features (no hidden state)       │
│      Note: disables encoder layers automatically  │
└──────────────────────────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌──────────┐    ┌──────────┐
│ Policy   │    │ Value    │
│ Linear   │    │ Linear   │
│ (no act) │    │ (no act) │
└────┬─────┘    └────┬─────┘
     ▼               ▼
 logits (B,T,4)  values (B,T)
```

**Nonlinearities:**
- Encoder: ReLU after each layer
- GRU: sigmoid (gates) + tanh (candidate) — standard PyTorch GRU
- MLP: ReLU after each layer
- CNN: ReLU after each conv layer
- Output heads: None (raw logits/values)

---

### 5. Training Script (`train.py`)

The training script trains navigation policies via imitation learning or reinforcement learning.

#### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Initialize environment pools (train, pos_val, goal_val)     │
│  2. Optionally setup VectorHash encoding                        │
│  3. Create Agent model                                          │
│  4. For each epoch:                                             │
│     a. Collect episodes via generate_episodes_vectorized()      │
│     b. Collate into padded batches                              │
│     c. Compute loss (CrossEntropy or PPO)                       │
│     d. Update model                                             │
│     e. Validate periodically                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Functions

| Function | Description |
|----------|-------------|
| `generate_episode()` | Roll out single episode, label with `best_action_to_goal` |
| `generate_episodes_vectorized()` | Batched rollout using `WMVecEnv`/`GridWMVecEnv` |
| `collate_supervised()` | Pad episodes into `(B, T, F)` tensors for supervised learning |
| `collate_rollouts()` | Pad episodes with actions/rewards/values for PPO |
| `compute_gae()` | Generalized Advantage Estimation for PPO |
| `ppo_loss()` | Clipped PPO objective with value and entropy terms |
| `validate()` | Evaluate accuracy and success rate on environment pool |
| `train()` | Main training loop |

#### Data Flow

```
Environment Pool
       │
       ▼
generate_episodes_vectorized()
  - Creates WMVecEnv/GridWMVecEnv (batched)
  - Rolls out batch_episodes in parallel
  - Labels each step with best_action_to_goal
  - Returns list of episode dicts:
      {obs, labels, actions, rewards, dones, values, log_probs}
       │
       ▼
collate_supervised() or collate_rollouts()
  - Pads variable-length episodes
  - Returns (B, T, F) tensors
       │
       ▼
Model Forward Pass
  - logits, values, h_next = model(obs_batch)
       │
       ▼
Loss Computation
  - Supervised: CrossEntropyLoss(logits, labels)
  - PPO: clipped surrogate + value loss + entropy bonus
       │
       ▼
Optimizer Step
```

#### Training Methods

**Supervised Imitation Learning** (`--train_method supervised`)
- Labels: `env.best_action_to_goal()` (greedy optimal)
- Loss: `CrossEntropyLoss(ignore_index=-100)` for padded sequences
- Action selection during rollout: greedy (argmax) with optional ε-greedy

**PPO Reinforcement Learning** (`--train_method ppo`)
- Reward: +1 at goal, `-time_penalty` per step
- Uses GAE (γ=0.99, λ=0.95) for advantage estimation
- Clipped surrogate objective with value and entropy bonuses
- Action selection during rollout: sampled from policy

#### Environment Pools

Three separate pools for different evaluation purposes:

| Pool | Purpose | Goal | Start |
|------|---------|------|-------|
| `env_pool` | Training | Fixed per env | Random |
| `pos_env_pool` | Position validation | Same as training | Different distribution |
| `new_env_pool` | Goal validation | Different (new) | Random |

#### Example Usage

```bash
# Supervised with VectorHash
python train.py \
    --size 8 \
    --num_envs 4 \
    --train_method supervised \
    --vectorhash \
    --input_type g_idx \
    --lambdas 11 12

# PPO without VectorHash
python train.py \
    --size 8 \
    --train_method ppo \
    --ppo_clip 0.2 \
    --ppo_ent_coef 0.01
```

---

## VectorHash Utility Functions

The `vectorhash/` directory contains supporting functions. Key ones used by `VectorHash`:

### From `assoc_utils_np.py`:
- `nonlin(x, thresh)`: ReLU activation shifted by threshold
- `randn`, `randint`: NumPy random generators  
- `train_gcpc(pbook, gbook, Npatts)`: Hebbian learning for Wgp
- `pseudotrain_Wsp/Wps`: Pseudoinverse-based weight learning

### From `assoc_utils_np_2D.py`:
- `gen_gbook_2d(lambdas, Ng, Npos)`: Generate 2D grid codebook
- `module_wise_NN_2d(gin, module_gbooks, module_sizes)`: Module-wise winner-take-all
- `path_integration_Wgg_2d(lambdas, Ng, axis, direction)`: Path integration matrices

### From `senstranspose_utils.py`:
- Additional dynamics and capacity analysis functions

---

## Example Usage

### Basic Navigation

```python
from cls import WMEnv

env = WMEnv(size=8, speed=1, seed=42)
pos, goal, obs, reward = env.reset()

while pos != goal:
    action = env.best_action_to_goal()
    pos, goal, obs, reward = env.step(action)
    print(f"Position: {pos}, Goal: {goal}")
```

### With VectorHash Encoding

```python
from cls.envs.environments import GridWMEnv
from cls.utils.GridUtils import VectorHash

# Create environments
envs = [GridWMEnv(size=8, speed=1, input_type="g_idx") for _ in range(4)]

# Initialize VectorHash
vh = VectorHash(Np=1600, lambdas=[11, 12], size=8)
vh.initiate_vectorhash(envs)

# Use environment
env = envs[0]
pos, goal, obs, reward = env.reset()
print(f"Observation shape: {obs.shape}")  # Grid cell indices
```

### Training a Policy

```python
from cls.models import Agent
import torch

# Create agent
agent = Agent(
    input_size=4,  # For g_idx with 2 modules: 2*2=4
    hidden_size=128,
    model_class="GRU",
)

# Forward pass
obs = torch.randn(1, 1, 4)  # (batch, seq, features)
logits, values, hidden = agent(obs)
action = torch.argmax(logits[0, 0])
```

---

## Validation

The training script validates on three environment pools:

1. **Training environments**: Same goals, same start distributions
2. **Position validation**: Same goals, different start distributions  
3. **Goal validation**: Different goals (generalization test)

Metrics tracked:
- **Accuracy**: Match rate between policy action and `best_action_to_goal`
- **Success rate**: Episodes reaching goal before max steps

---

## References

The VectorHash system is inspired by theoretical models of hippocampal-entorhinal interactions:
- Grid cells provide a compressed, modular representation of space
- Place cells act as a distributed, high-dimensional representation
- Associative memories enable bidirectional recall between representations

---

### 6. Hopfield Network (`cls/hopfield.py`)

Continuous Hopfield network for associative memory with sequential storage.

```python
from cls.hopfield import Hopfield

net = Hopfield(num_units=256, beta=2.0, zero_diag=True)

# Store memories one at a time (sequential Hebbian learning)
for pattern in patterns:
    net.input_memory(pattern)  # W += scale * z ⊗ zᵀ

# Recall from noisy cue
x_recalled, cos_sims = net.recall(
    x0=noisy_cue,
    steps=15,
    use_tanh=True,
    target=original,  # optional, for tracking similarity
)
```

| Method | Description |
|--------|-------------|
| `input_memory(z)` | Store single pattern via Hebbian update |
| `recall(x0, ...)` | Iterative recall with sync/async dynamics |
| `reset()` | Clear all memories |
| `energy(x)` | Compute E = -0.5 xᵀWx |

#### Where Hopfield is Used

**Initialization flow:**
```
train.py
  └─► VectorHash(..., use_hopfield=True, hopfield_gain=..., hopfield_alpha=...)
        └─► initiate_vectorhash(envs)
              └─► _init_hopfield(envs)
                    ├─► Hopfield(num_units=pattern_dim, beta=hopfield_gain)
                    └─► For each env: hopfield.input_memory(env.obs_at_goal())
```

**Runtime flow (when `--input_addendum hopfield`):**
```
generate_episodes_vectorized()
  └─► GridWMVecEnv.obs_batch(indices, input_addendum="hopfield")
        └─► vectorhash.hopfield_recall_batch(obs)
              └─► For each obs: hopfield.recall(obs, steps=hopfield_steps, alpha=hopfield_alpha)
        └─► return concat([obs, recalled])  # doubles observation size
```

**Key parameters:**
| Parameter | CLI Argument | Description |
|-----------|--------------|-------------|
| `beta` / `gain` | `--hopfield_gain` | Temperature scaling: `tanh(gain * h)`. Default: uses encoder's gain if available, else 2.0 |
| `alpha` | `--hopfield_alpha` | Mixing coefficient: `x = (1-α)*x + α*update`. Default: 1.0 (full update) |
| `steps` | `--hopfield_steps` | Number of recall iterations. Default: 1 |

---

### 7. Grid Encoder (`cls/encoder.py`)

Encoders for projecting `g_hot` observations to a learned normalized embedding space. Two architectures are available:

#### `GridEncoder` - MLP Encoder

```python
from cls.encoder import GridEncoder

enc = GridEncoder(
    in_dim=265,          # sum(λ² for λ in lambdas), e.g. 11²+12² = 265
    hidden=1024,         # hidden layer size
    out_dim=512,         # embedding dimension
    nonlinearity="gelu", # hidden activation
    output_nonlinearity="tanh",  # output activation before normalization
    gain=3.0,            # scales output: tanh(gain * z)
)
z = enc(x)  # Output: L2-normalized embedding
```

#### `GridEncoderCNN` - CNN Encoder

Reshapes flattened `g_hot` into 2D module grids and applies convolutions:

```python
from cls.encoder import GridEncoderCNN

enc = GridEncoderCNN(
    lambdas=[11, 12],          # Grid module periods (determines 2D reshape)
    hidden_channels=32,        # Conv channels
    num_conv_layers=3,         # Number of conv layers
    hidden_dim=128,            # MLP hidden dimension after pooling
    num_hidden_layers=1,       # Number of MLP hidden layers
    out_dim=128,               # Embedding dimension
    nonlinearity="gelu",       # Hidden activation
    output_nonlinearity="tanh",# Output activation before normalization
    gain=5.0,                  # Scales output
)
z = enc(x)  # Output: L2-normalized embedding, supports (B, D) or (B, T, D) input
```

The CNN encoder reshapes each grid module into a 2D grid, zero-pads to `max(λ) × max(λ)`, stacks as channels, then applies conv layers followed by adaptive average pooling and an MLP head.

#### Where GridEncoder is Used

**Initialization flow:**
```
train.py (when --input_type encoded_g --vectorhash --encoder_weights my_encoder.pt)
  └─► Load checkpoint from encoders/my_encoder.pt
        ├─► Extract config: hidden, out_dim, gain, nonlinearity
        └─► Override gain if --encoder_gain provided
  └─► GridEncoder(in_dim=g_hot_dim, **config)
  └─► encoder.load_state_dict(checkpoint["state_dict"])
  └─► encoder.eval()
  └─► Pass encoder to _make_env() for each environment
        └─► GridWMEnv(..., encoder=encoder)
```

**Runtime flow:**
```
GridWMEnv.obs() / GridWMEnv.convert_obs()
  └─► If input_type == "encoded_g":
        └─► g_hot = vectorhash.recall(raw_obs)[2]  # get grid state
        └─► (optionally) g_hot = smooth_g(g_hot, lambdas, fwhm_ratio)
        └─► encoded = encoder(g_hot)  # uses encoder.gain
        └─► return encoded

GridWMVecEnv._convert_obs_batch() / GridWMVecEnv._build_preconv_codebook()
  └─► Same logic, batched
```

**Key parameters:**
| Parameter | CLI Argument | Description |
|-----------|--------------|-------------|
| (loaded from checkpoint) | `--encoder_weights` | Filename in `encoders/` directory |
| `gain` | `--encoder_gain` | Override checkpoint's gain (default: use saved gain) |

**Note:** When both encoder and Hopfield are used, `hopfield_gain` defaults to the encoder's gain unless explicitly overridden with `--hopfield_gain`.

#### Training Encoders

Encoders are trained separately via `notebooks/train_dist_encoder.py` (run with `python notebooks/train_dist_encoder.py` or via `train.sh`) using:
- **CKA loss**: Kernel alignment between encoded distances and target distances
- **Uniformity loss**: Encourages spread across the embedding sphere
- **Sweep parameters**: `cka_alpha`, `sigmoid_scale_end` (gain), `uniformity_lambda_end`
- Supports both `GridEncoder` (MLP) and `GridEncoderCNN` architectures

Trained encoders are saved to `encoders/` with descriptive names like:
```
encoder_alpha3.0_sig4_uni0.01.pt
```

Loading a saved encoder:
```python
checkpoint = torch.load("encoders/encoder_alpha3.0_sig4_uni0.01.pt", weights_only=False)
config = checkpoint["config"]
encoder = GridEncoder(in_dim=..., **config)
encoder.load_state_dict(checkpoint["state_dict"])
gain = checkpoint["sweep_params"]["sigmoid_scale_end"]  # use this for forward pass
```

---

### 8. Action Classifier (`train_action_classifier.py`)

Trains an MLP to classify the action direction given (start_state, end_state) observation pairs. This is a standalone training script separate from `train.py`.

#### How It Works

1. **Sample generation**: For each training environment, enumerate all valid (start_pos, end_pos) pairs within a configurable displacement shell (`--max_steps min max`)
2. **Label**: Each pair is labeled with the direction of the first step (N/E/S/W) based on the displacement vector
3. **Input**: Concatenation of start and end observations (or their difference with `--use_displacement`), using the chosen `--input_type` representation
4. **Model**: Agent with MLP backbone trained via cross-entropy

#### Key Features

- **Multiple training environments**: `--num_train_envs` with independent or `--shared_vectorhash`
- **Global space mode**: `--global_space` places environments at random offsets in VectorHash's abstract position space, enabling cross-environment generalization
- **Displacement shells**: `--max_steps 1 4` trains on displacements from 1 to 3 cells
- **Validation on new environments**: `--num_val_new_envs` creates held-out environments with separate VectorHash instances
- **Learning rate scheduling**: `--scheduler cosine|step|plateau` with `--warmup_epochs`
- **Model checkpointing**: `--save_every N --save_dir action_classifiers`

#### Example Usage

```bash
python train_action_classifier.py \
    --model_type mlp \
    --vectorhash \
    --input_type encoded_g \
    --max_steps 1 4 \
    --fwhm_ratio 0.25 \
    --size 8 \
    --num_train_envs 20 \
    --Np 4000 \
    --lambdas 11 12 13 \
    --Npos 60 \
    --hidden_size 512 \
    --num_model_layers 2 \
    --batch_size 256 \
    --n_epochs 20000 \
    --lr 1e-6 \
    --encoder_weights goated_cnn_encoder.pt \
    --use_wandb \
    --wandb_project cls_action_classifier
```

---

## File Structure Summary

```
cls/
├── __init__.py              # Package exports
├── types.py                 # Type aliases
├── models.py                # GRU, MLP, GridCNNBackbone, Agent
├── hopfield.py              # Hopfield associative memory
├── encoder.py               # GridEncoder (MLP) and GridEncoderCNN
├── envs/
│   ├── __init__.py
│   └── environments.py      # WMEnv, GridWMEnv, WMVecEnv, GridWMVecEnv
├── utils/
│   ├── __init__.py
│   └── GridUtils.py         # VectorHash, smooth_g, smooth_gbook
└── vectorhash/
    ├── assoc_utils_np.py    # Core associative memory functions
    ├── assoc_utils_np_2D.py # 2D grid cell functions
    ├── senstranspose_utils.py # Sensory-transpose dynamics
    ├── seq_utils.py         # Sequence/action utilities
    └── ...                  # Additional utilities

train.py                     # Policy training (supervised / PPO)
train_action_classifier.py   # Action classifier from (start, end) state pairs
notebooks/
└── train_dist_encoder.py    # Encoder training script
encoders/                    # Saved encoder checkpoints
action_classifiers/          # Saved action classifier checkpoints
```

---

## Argument Reference

### Environment

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size` | int | 8 | Grid world size (size × size) |
| `--speed` | int | 1 | Steps per action |
| `--seed` | int | 0 | Random seed |
| `--num_envs` | int | 1 | Number of training environments per world |
| `--num_val_envs` | int | 4 | Number of validation environments |
| `--num_train_worlds` | int | 1 | Number of independent training worlds (each with its own VectorHash/Hopfield) |
| `--time_penalty` | float | 0.01 | Penalty per timestep (reward = -time_penalty) |
| `--observation_size` | int | 512 | Binary observation code length |
| `--use_headings` | flag | False | Use heading-dependent observations (default: heading-invariant) |

### Model

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hidden_size` | int | 128 | Hidden layer size |
| `--num_model_layers` | int | 1 | Number of backbone layers |
| `--model_class` | str | "GRU" | Backbone type: `"GRU"`, `"MLP"`, or `"CNN"` |
| `--encoder_dim` | int | None | Encoder output dimension |
| `--num_encoder_layers` | int | 0 | Number of encoder layers (0 = no encoder) |
| `--num_actions` | int | 4 | Action space size |
| `--dropout` | float | 0.0 | Dropout probability |

### Training

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_method` | str | "supervised" | `"supervised"` or `"ppo"` |
| `--n_epochs` | int | 200 | Total training epochs |
| `--lr` | float | 1e-3 | Learning rate |
| `--batch_episodes` | int | 16 | Episodes per batch per environment |
| `--steps_per_episode` | int | 20 | Max steps per episode |
| `--max_envs_per_epoch` | int | 8 | Max environments sampled per epoch |
| `--val_epochs` | int | 1 | Validate every N epochs |
| `--val_batch_episodes` | int | 4 | Episodes per validation batch |
| `--plot_every` | int | 100 | Save plots every N epochs |
| `--input_addendum` | str | "none" | Extra input: `"goal"`, `"diff"`, `"next_best"`, `"hopfield"`, `"none"` |
| `--expert_actions` | flag | False | Use best action for stepping during rollouts (behavioral cloning from expert) |

### PPO-Specific

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ppo_clip` | float | 0.2 | PPO clipping parameter ε |
| `--ppo_vf_coef` | float | 0.5 | Value function loss coefficient |
| `--ppo_ent_coef` | float | 0.0 | Entropy bonus coefficient |
| `--ppo_epochs` | int | 4 | PPO optimization epochs per batch |
| `--ppo_input_reward` | flag | True | Append previous reward to observation |

### VectorHash

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--vectorhash` | flag | False | Enable VectorHash encoding |
| `--Np` | int | 1600 | Number of place cells |
| `--Npos` | int | None | Number of unique positions (default: product of lambdas) |
| `--lambdas` | int[] | [11, 12] | Grid module periods |
| `--input_type` | str | "g_idx" | Output type: `"g_idx"`, `"g_hot"`, `"s"`, `"p"`, `"encoded_g"` |
| `--use_preconv_codebook` | flag | False | Precompute converted codebook for speed |
| `--fwhm_ratio` | float | 0.0 | FWHM ratio for smoothing g_hot (0 = no smoothing, e.g. 0.25 = FWHM is 1/4 of λ) |

### GridEncoder

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--encoder_weights` | str | None | Filename of encoder weights in `encoders/` directory |
| `--encoder_gain` | float | None | Override gain from loaded encoder (default: use saved gain) |

### Hopfield

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hopfield_gain` | float | None | Hopfield gain/beta (default: use encoder's gain if available, else 2.0) |
| `--hopfield_alpha` | float | 1.0 | Hopfield recall mixing coefficient (1.0 = full update) |
| `--hopfield_steps` | int | 1 | Number of Hopfield recall iterations |

### Logging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb` | flag | False | Enable Weights & Biases logging |
| `--wandb_project` | str | "cls" | W&B project name |
