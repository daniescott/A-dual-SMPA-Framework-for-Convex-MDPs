# CMPT409 Project: Convex MDP Solver with SPMA

A Python implementation of a **Constrained Markov Decision Process (CMDP)** solver using **Softmax Policy Mirror Ascent (SPMA)**. This project provides a theoretically-grounded framework for solving reinforcement learning problems with constraints through convex optimization and saddle-point formulations.

## Table of Contents

- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Algorithm Details](#algorithm-details)
- [Contributing](#contributing)
- [References](#references)

---

## Overview

This repository implements a policy optimization framework for solving CMDPs, where an agent must maximize cumulative reward while satisfying constraints. The approach:

1. **Reformulates CMDPs** as saddle-point problems using Lagrangian duality
2. **Alternates between**:
   - **Primal updates**: Policy improvement via SPMA (given dual variables)
   - **Dual updates**: Constraint enforcement via gradient ascent
3. **Supports both**:
   - Tabular settings (discrete state-action spaces)
   - Function approximation (continuous/large spaces with features)

### Key Innovation: SPMA (Softmax Policy Mirror Ascent)

SPMA is a mirror descent algorithm on the policy space that uses KL divergence as the Bregman divergence. It provides:
- Theoretical convergence guarantees
- Natural policy parameterization
- Efficient exploration-exploitation tradeoff

---

## Theoretical Background

### CMDP Formulation

A CMDP extends standard MDPs with constraints:

```
maximize   E[Σ_t γ^t r(s_t, a_t)]
subject to E[Σ_t γ^t c_i(s_t, a_t)] ≤ b_i  for i=1,...,m
           π ∈ Π (policy space)
```

### Saddle-Point Reformulation

Using Lagrangian duality:

```
L(π, y) = E_π[Σ_t γ^t (r(s_t, a_t) - y^T c(s_t, a_t))]
        = ⟨d_π, r_y⟩
```

where:
- `d_π(s,a)` is the discounted state-action occupancy
- `r_y(s,a) = r(s,a) - y^T c(s,a)` is the shaped reward
- `y` are dual variables (Lagrange multipliers)

The solution is found by solving: `min_y max_π L(π, y)`

### SPMA Update Rule

Given dual variables `y`, SPMA updates the policy via:

```
π_{k+1} = argmax_π [ ⟨π, ∇_π L⟩ - (1/η) D_KL(π || π_k) ]
```

where `D_KL` is the KL divergence and `η` is the step size.

This translates to the practical loss function:

```
L_SPMA = E[ -Δlog π · A + (1/η) · ((exp(Δlog π) - 1) - Δlog π) ]
```

where `Δlog π = log π_new - log π_old` and `A` is the advantage function.

---

## Features

### Core Capabilities

-  **SPMA Policy Oracle**: Theoretically-grounded policy improvement
-  **Dual Variable Updates**: Constraint enforcement through gradient ascent
-  **Armijo Line Search**: Robust step size selection with backtracking
-  **GAE (Generalized Advantage Estimation)**: Low-variance advantage estimates
-  **Shaped Reward Environments**: Automatic reward transformation for constrained optimization

### Supported Settings

| Feature | Tabular | Function Approximation |
|---------|---------|------------------------|
| State Space | Discrete | Continuous/Large |
| Action Space | Discrete | Discrete or Continuous |
| Dual Variables | Table `y[s,a]` | Feature weights `w` |
| Estimator | Occupancy `d(s,a)` | Feature expectations `E[φ]` |
| Example Env | FrozenLake | Pendulum |

### Neural Network Architectures

- **Discrete Actor**: Softmax policy over action logits
- **Gaussian Actor**: State-dependent mean with learned log-std
- **Critic**: Value function approximation (MLP)

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Gymnasium (OpenAI Gym)
- NumPy

### Install Dependencies

```bash
pip install torch gymnasium numpy
```

### Install Package

```bash
cd implementations/cmdp_spma_package
pip install -e .
```

---

## Quick Start

### Tabular Example (FrozenLake)

```python
import gymnasium as gym
import numpy as np
from cmdp_spma import PolicyOracleSPMA, OracleConfig, TabularEstimator

# Setup environment
def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False)

env = make_env()
nS, nA = env.observation_space.n, env.action_space.n
gamma = 0.99

# Initialize dual variables (constraint multipliers)
y = np.zeros((nS, nA), dtype=np.float32)

# Create occupancy estimator
estimator = TabularEstimator(nS, nA, gamma)

# Configure SPMA oracle
config = OracleConfig(
    discrete=True,
    steps_per_rollout=512,
    K_inner=3,
    gamma=gamma
)

# Run policy improvement
oracle = PolicyOracleSPMA(make_env, estimator, config)
policy, d_hat, logs = oracle.improve(y, K=3, rollout_steps=1024, seed=0)

print(f"Sum of occupancy: {d_hat.sum():.4f}")
print(f"Lagrangian value: {logs['y_dot_d']:.4f}")
```

### Function Approximation Example (Pendulum)

```python
import gymnasium as gym
import numpy as np
from cmdp_spma import PolicyOracleSPMA, OracleConfig, FeatureEstimator

# Define feature function
def phi_fn(s, a):
    """Feature vector: [s, a, s^2, a^2]"""
    s = np.asarray(s, dtype=np.float32)
    a = np.atleast_1d(a).astype(np.float32)
    return np.concatenate([s, a, s**2, a**2])

def make_env():
    return gym.make("Pendulum-v1")

gamma = 0.99
d = 3 + 1 + 3 + 1  # feature dimension
w = np.zeros(d, dtype=np.float32)  # dual weights
y = (phi_fn, w)  # dual variable tuple

# Create feature expectation estimator
estimator = FeatureEstimator(phi_fn, d, gamma)

# Configure for continuous actions
config = OracleConfig(
    discrete=False,
    steps_per_rollout=2048,
    K_inner=3,
    gamma=gamma
)

# Run policy improvement
oracle = PolicyOracleSPMA(make_env, estimator, config)
policy, ephi_hat, logs = oracle.improve(y, K=3, rollout_steps=4096, seed=0)

print(f"Feature expectation norm: {np.linalg.norm(ephi_hat):.4f}")
```

---

## Project Structure

```
CMPT409-Project/
├── README.md                          # This file
├── .gitmodules                        # Git submodules configuration
│
├── implementations/
│   ├── clean_spma/                    # External SPMA reference (submodule)
│   │
│   └── cmdp_spma_package/             # Main implementation
│       ├── README.md                  # Package-specific docs
│       ├── cmdp_spma/                 # Core library
│       │   ├── __init__.py           # Package exports
│       │   ├── policy_oracle.py      # SPMA policy oracle (main class)
│       │   ├── spma_losses.py        # SPMA loss functions
│       │   ├── line_search.py        # Armijo backtracking
│       │   ├── nets.py               # Neural network architectures
│       │   ├── rollout.py            # Trajectory collection & GAE
│       │   ├── occupancy.py          # Occupancy/feature estimators
│       │   └── shaped_reward_env.py  # Environment wrappers
│       │
│       ├── scripts/                   # Example scripts
│       │   ├── run_tabular_example.py
│       │   ├── run_feature_example.py
│       │   └── outer_loop_demo.py    # Full CMDP solver demo
│       │
│       └── tests/                     # Unit tests
│           └── test_estimator.py
```

---

## Usage Guide

### 1. Define Your Environment

```python
def make_env():
    return gym.make("YourEnvironment-v1")
```

### 2. Choose Tabular or Function Approximation

#### Tabular (Discrete S × A)

```python
env = make_env()
nS = env.observation_space.n
nA = env.action_space.n
y = np.zeros((nS, nA), dtype=np.float32)
estimator = TabularEstimator(nS, nA, gamma)
config = OracleConfig(discrete=True, ...)
```

#### Function Approximation (Continuous/Large)

```python
def phi_fn(s, a):
    # Return feature vector
    return np.array([...])

d = 10  # feature dimension
w = np.zeros(d, dtype=np.float32)
y = (phi_fn, w)
estimator = FeatureEstimator(phi_fn, d, gamma)
config = OracleConfig(discrete=False, ...)
```

### 3. Configure the Oracle

```python
config = OracleConfig(
    device="cpu",              # "cpu" or "cuda"
    gamma=0.99,                # Discount factor
    lam=0.95,                  # GAE lambda
    inv_eta=5.0,               # 1/η in SPMA (regularization strength)
    max_grad_norm=1.0,         # Gradient clipping
    critic_lr=3e-4,            # Critic learning rate
    K_inner=5,                 # SPMA iterations per call
    steps_per_rollout=2048,    # Steps per rollout
    discrete=True,             # Discrete or continuous actions
    hidden=(64, 64),           # Hidden layer sizes
    # Armijo line search parameters
    armijo_c=1e-4,
    armijo_beta=0.5,
    armijo_init=1.0,
    armijo_max_steps=15
)
```

### 4. Run the Policy Oracle

```python
oracle = PolicyOracleSPMA(make_env, estimator, config)
policy, d_hat, logs = oracle.improve(
    y,                  # Dual variables
    K=5,                # Number of SPMA iterations
    rollout_steps=4096, # Total environment steps
    seed=42             # Random seed
)
```

### 5. Update Dual Variables (Outer Loop)

For a simple quadratic regularization `f*(y) = (λ/2)||y||²`:

```python
lam = 0.1       # Regularization strength
alpha = 0.5     # Dual step size

# Gradient of Lagrangian w.r.t. y
grad_y = d_hat - lam * y

# Dual ascent step
y = y + alpha * grad_y
```

---

## Examples

### Example 1: Single Policy Improvement

Run a single policy improvement step on FrozenLake:

```bash
cd implementations/cmdp_spma_package
python scripts/run_tabular_example.py
```

**Expected Output:**
```
sum d_hat: 0.995
y·d: 0.0000
```

### Example 2: Continuous Control

Test SPMA on Pendulum with feature approximation:

```bash
python scripts/run_feature_example.py
```

**Expected Output:**
```
||E[phi]||: 1.234
y·d: 0.0000
```

### Example 3: Full CMDP Solver

Run the outer-loop demo with dual variable updates:

```bash
python scripts/outer_loop_demo.py
```

**Expected Output:**
```
[iter 0] sum d=0.998, y·d=0.0012, L=0.0011
[iter 1] sum d=0.995, y·d=0.0034, L=0.0029
[iter 2] sum d=0.997, y·d=0.0052, L=0.0045
[iter 3] sum d=0.996, y·d=0.0068, L=0.0058
[iter 4] sum d=0.998, y·d=0.0079, L=0.0067
Done.
```

---

## API Reference

### PolicyOracleSPMA

**Main class for policy improvement under shaped rewards.**

```python
class PolicyOracleSPMA:
    def __init__(
        self,
        env_maker: Callable[[], gym.Env],
        estimator: Union[TabularEstimator, FeatureEstimator],
        cfg: OracleConfig
    )
```

**Methods:**

```python
def improve(
    self,
    y: Union[np.ndarray, Tuple[Callable, np.ndarray]],
    K: Optional[int] = None,
    rollout_steps: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[Dict, np.ndarray, Dict]:
    """
    Perform K iterations of SPMA under shaped reward r_y.
    
    Args:
        y: Dual variables (table or (phi_fn, w) tuple)
        K: Number of SPMA iterations (default: cfg.K_inner)
        rollout_steps: Total environment steps (default: cfg.steps_per_rollout)
        seed: Random seed for reproducibility
    
    Returns:
        policy_snapshot: Dict of policy network parameters
        d_hat: Occupancy estimate or feature expectations
        logs: Dict with training metrics
    """
```

### OracleConfig

**Configuration dataclass for the policy oracle.**

```python
@dataclass
class OracleConfig:
    device: str = "cpu"
    gamma: float = 0.99              # Discount factor
    lam: float = 0.95                # GAE lambda
    inv_eta: float = 5.0             # 1/η (SPMA regularization)
    max_grad_norm: float = 1.0       # Gradient clipping
    armijo_c: float = 1e-4           # Armijo sufficient decrease
    armijo_beta: float = 0.5         # Armijo backtrack factor
    armijo_init: float = 1.0         # Initial step size
    armijo_max_steps: int = 15       # Max backtracking steps
    critic_lr: float = 3e-4          # Critic learning rate
    K_inner: int = 5                 # SPMA iterations
    steps_per_rollout: int = 2048    # Steps per rollout
    discrete: bool = True            # Discrete vs continuous actions
    hidden: Tuple[int,int] = (64,64) # MLP hidden sizes
```

### Estimators

**TabularEstimator**

```python
class TabularEstimator:
    def __init__(self, nS: int, nA: int, gamma: float)
    
    def update_from_batch(self, obs, acts, dones):
        """Accumulate occupancy from trajectory batch."""
    
    def value(self) -> np.ndarray:
        """Return current occupancy estimate d[s,a]."""
    
    def y_dot_d(self, y_table: np.ndarray) -> float:
        """Compute y^T d (Lagrangian value)."""
```

**FeatureEstimator**

```python
class FeatureEstimator:
    def __init__(self, phi_fn: Callable, d: int, gamma: float)
    
    def update_from_batch(self, obs, acts, dones):
        """Accumulate feature expectations from batch."""
    
    def value(self) -> np.ndarray:
        """Return E[phi(s,a)]."""
    
    def y_dot_d(self, w: np.ndarray) -> float:
        """Compute w^T E[phi]."""
```

---

## Algorithm Details

### SPMA Loss Function

The SPMA loss at each iteration is:

```
L(θ) = E_batch[ -Δlog π(a|s;θ) · A(s,a) + (1/η) · Bregman(π_new || π_old) ]
```

where:
- `Δlog π = log π_new - log π_old` is the log-ratio
- `A(s,a)` is the advantage function (from GAE)
- `Bregman(p||q) = (exp(Δlog π) - 1) - Δlog π` is the KL Bregman divergence
- `η` is the step size (inverse of `inv_eta`)

### Armijo Backtracking

To ensure descent, we use Armijo line search:

1. Compute gradient `g` and descent direction `d = -g`
2. Try step sizes `α = α₀, α₀β, α₀β², ...`
3. Accept first `α` satisfying: `L(θ + αd) ≤ L(θ) + c·α·g^T d`
4. If no step accepted after max iterations, take tiny gradient step

### GAE (Generalized Advantage Estimation)

Advantages are computed using GAE with parameters `γ` (discount) and `λ` (bias-variance):

```
A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error.

### Occupancy Estimation

**Unbiased discounted occupancy estimator:**

```
d̂(s,a) = (1-γ)/N · Σ_i Σ_t γ^t · 1{(s_t, a_t) = (s,a)}
```

where `N` is the number of episodes.

**Properties:**
- Unbiased: `E[d̂(s,a)] = d_π(s,a)`
- Normalizes approximately to 1: `Σ_{s,a} d̂(s,a) ≈ 1`

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow the code style**: Use type hints, docstrings, and consistent formatting
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Development Setup

```bash
git clone https://github.com/yourusername/CMPT409-Project.git
cd CMPT409-Project/implementations/cmdp_spma_package
pip install -e .
python -m pytest tests/
```

---

## References

### Key Papers
<!-- 
1. **Mirror Descent in Saddle-Point Problems**  
   Nemirovski, A. (2004). *Prox-method with rate of convergence O(1/t) for variational inequalities with Lipschitz continuous monotone operators and smooth convex-concave saddle point problems.*

2. **Policy Mirror Descent**  
   Tomar, M., et al. (2020). *Mirror descent policy optimization.*

3. **Constrained MDPs**  
   Altman, E. (1999). *Constrained Markov decision processes.*

4. **Convex MDP Duality**  
   Puterman, M. L. (1994). *Markov decision processes: Discrete stochastic dynamic programming.*

### Related Work

- **PPO (Proximal Policy Optimization)**: Trust-region-free policy optimization
- **TRPO (Trust Region Policy Optimization)**: Natural gradient with KL constraint
- **CPO (Constrained Policy Optimization)**: Safe RL with hard constraints
- **Lagrangian Methods**: Primal-dual approaches for constrained optimization -->

---

## License

This project is part of CMPT409 coursework. Please check with the course instructor regarding licensing and usage rights.

---

## Acknowledgments

---

