# Optimal Functional Flow Matching (OFFM) - Implementation Plan

## Executive Summary

This document outlines the implementation plan for **Optimal Functional Flow Matching (OFFM)**, which extends Optimal Flow Matching (OFM) from finite-dimensional spaces to infinite-dimensional Hilbert spaces for generating spatio-temporal PDE trajectories.

### Key Innovation
- **OFM** (finite-dim): Uses ICNNs to parameterize convex potentials Ψ:ℝᵈ→ℝ, learns OT map in one iteration
- **OFFM** (infinite-dim): Uses projected ICNNs to parameterize convex functionals Φ:H_traj→ℝ on Hilbert spaces of trajectories

### Target Dataset: Stochastic KdV
- Data shape: `(1200, 128, 101)` = (batch_size, N_spatial, T_time)
- Trajectory space: L²(0,T; H^s(D)) where D=[0,L) is periodic

---

## 0. Existing Codebase to Leverage

We build on top of the existing functional flow matching infrastructure:

### Data Loading (REUSE)
- **`util/util.py`**: `load_stochastic_kdv(path, mode='trajectory')` already supports trajectory mode
- **`scripts/stochastic_kdv.py`**: Reference for experiment structure

### Reference Measure Sampling (EXTEND)
- **`util/gaussian_process.py`**: `GPPrior` class for GP sampling
  - Currently samples spatial functions: `(n_samples, n_channels, *dims)`
  - Need to extend for trajectory sampling: `(n_samples, n_t, n_x)`

### Model Architecture (REUSE patterns)
- **`functional_fm.py`**: `FFMModel` class structure
  - Training loop pattern
  - `simulate()` for interpolation
  - `sample()` for generation
- **`functional_fm_ot.py`**: `FFMModelOT` for OT-enhanced version
- **`models/fno.py`**: FNO architecture (may be useful for comparison)

### Utilities (REUSE)
- **`util/util.py`**: `make_grid()`, `reshape_for_batchwise()`, `plot_loss_curve()`
- **`util/eval.py`**: `GenerationQualityMetrics` for evaluation
- **`util/ot_monitoring.py`**: `TrainingMonitor` for tracking

### Experiment Scripts (FOLLOW patterns)
- **`scripts/kdv_ot.py`**: Comprehensive experiment structure with configs
- **`scripts/stochastic_kdv_ot.py`**: OT experiments on stochastic KdV

---

## 0.1 Comparison: FFMModel vs OFFMModel

| Aspect | FFMModel (existing) | OFFMModel (new) |
|--------|---------------------|-----------------|
| **What it learns** | Vector field `u(t, x)` | Convex potential `Φ(u)` |
| **Model architecture** | FNO (time-conditioned) | ProjectedICNN (convex) |
| **Loss function** | `‖u(t,x) - v_t(x)‖²` | Gap + KKT (stop-grad) |
| **Reference measure** | GP via `GPPrior` | Gaussian on trajectories |
| **Data representation** | Snapshots `(B, 1, N_x)` | Trajectories `(B, T, N_x)` |
| **Sampling** | ODE integration | Direct OT map |
| **Training loop** | Same pattern | Same pattern |
| **Utilities used** | `make_grid`, `plot_loss_curve` | Same utilities |

---

## 1. Mathematical Formulation Recap

### 1.1 Hilbert Trajectory Space
```
H_traj = L²(0,T; H^s(D))
```

With discrete approximation:
```
||u||²_traj ≈ Δt Σₙ ||uₙ||²_{H^s(D)}
```

where the Sobolev norm uses FFT:
```
||uₙ||²_{H^s} = Δx Σₖ (1+|k|²)^s |û_n(k)|²
```

### 1.2 Convex Functional Parameterization
```
Φ(u) = φ(Pu) + (λ/2)||u||²_traj
```
- P: H_traj → ℝᵐ is a linear projection (truncated Fourier coefficients)
- φ: ℝᵐ → ℝ is an ICNN (Input Convex Neural Network)
- λ ≥ 0 is a regularization parameter

### 1.3 Training Objective (Stop-Gradient)

**Inner Convex Problem:**
```
z* = argmin_z { (1/2)||z||²_traj - ⟨U_t, z⟩_traj + t·Φ(z) }
```

**Gap Loss:**
```
L_gap(θ) = E[g_θ(U₀; U_t, t) - g_θ(z*; U_t, t)]
```

**KKT Residual Loss:**
```
L_KKT(θ) = E[||U_t - z* - t·∇Φ_θ(z*)||²_traj]
```

**Combined Loss:**
```
L(θ) = L_gap(θ) + γ·L_KKT(θ)
```

---

## 2. Implementation Components

### 2.1 Hilbert Space Utilities (`util/hilbert.py`) [NEW FILE]

**Purpose:** Implement discrete Sobolev norms and inner products via FFT

**Relation to existing code:** Complements `util/gaussian_process.py` by providing the Hilbert space structure

```python
# Key functions to implement:

def get_sobolev_weights_1d(n_x: int, s: float, L: float = 1.0) -> torch.Tensor:
    """Get (1+|k|²)^s weights for 1D FFT."""
    
def sobolev_norm_1d(u: torch.Tensor, s: float = 1.0, L: float = 1.0) -> torch.Tensor:
    """Compute ||u||_{H^s(D)} for 1D periodic functions via FFT.
    
    Args:
        u: (batch, n_x) or (batch, T, n_x) tensor
        s: Sobolev exponent
        L: domain length
    Returns:
        Norm values
    """

def trajectory_norm(u: torch.Tensor, s: float = 1.0, dt: float = 1.0, L: float = 1.0) -> torch.Tensor:
    """Compute ||u||_{L²(0,T; H^s(D))} for trajectories.
    
    Args:
        u: (batch, T, n_x) tensor - trajectories
        s: spatial Sobolev exponent
        dt: time step
        L: spatial domain length
    Returns:
        (batch,) norm values
    """

def trajectory_inner_product(u: torch.Tensor, v: torch.Tensor, s: float = 1.0, 
                             dt: float = 1.0, L: float = 1.0) -> torch.Tensor:
    """Compute ⟨u, v⟩_{L²(0,T; H^s(D))} for trajectories."""

# Optional: time-derivative regularized norm (H¹ in time)
def trajectory_norm_with_time_reg(u: torch.Tensor, s: float = 1.0, 
                                   beta: float = 0.1, dt: float = 1.0) -> torch.Tensor:
    """Compute ||u||²_traj + β||∂_t u||²_{H^{s-1}}"""
```

### 2.2 ICNN Model (`models/icnn.py`) [NEW FILE]

**Purpose:** Input Convex Neural Network for parameterizing convex functions

```python
class ICNN(nn.Module):
    """Input Convex Neural Network.
    
    Architecture ensures convexity by:
    1. Non-negative weights on z-path (enforced via softplus or ReLU)
    2. Non-decreasing convex activations (LeakyReLU, ReLU, ELU, softplus)
    3. Skip connections from input preserve convexity
    
    Forward: φ(z) where φ is convex in z
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "leaky_relu",
        init_scale: float = 0.1,
    ):
        ...
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute φ(z) - scalar output, convex in z."""
        ...

class ProjectedICNN(nn.Module):
    """Convex functional Φ(u) = φ(Pu) + (λ/2)||u||²_traj
    
    Args:
        projection_type: 'fourier' (truncated FFT), 'pca', or 'identity'
        projection_dim: dimension m of projection output
        icnn_hidden_dims: hidden dimensions for ICNN
        lambda_reg: quadratic regularization coefficient
        sobolev_s: Sobolev exponent for norm
    """
    
    def __init__(
        self,
        n_t: int,              # number of time steps
        n_x: int,              # number of spatial points
        projection_dim: int = 256,
        projection_type: str = "fourier",
        icnn_hidden_dims: List[int] = [256, 256],
        lambda_reg: float = 0.0,
        sobolev_s: float = 1.0,
    ):
        ...
    
    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Project trajectory u to finite-dimensional z = Pu."""
        ...
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Compute Φ(u) = φ(Pu) + (λ/2)||u||²_traj."""
        ...
    
    def gradient(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ∇Φ(u) = P^T ∇φ(Pu) + λu (in Hilbert metric)."""
        ...
```

### 2.3 Inner Convex Solver (`util/convex_solver.py`) [NEW FILE]

**Purpose:** Solve the inner optimization problem for z*

```python
class ConvexSolver:
    """Solve: z* = argmin_z { (1/2)||z||²_traj - ⟨U_t, z⟩_traj + t·Φ(z) }
    
    Methods:
    - Gradient descent with momentum
    - L-BFGS (for small problems)
    - Warm-starting from previous iteration
    """
    
    def __init__(
        self,
        method: str = "gd",  # 'gd', 'lbfgs', 'nesterov'
        n_steps: int = 10,
        lr: float = 0.1,
        momentum: float = 0.9,
        tol: float = 1e-6,
    ):
        ...
    
    def solve(
        self,
        U_t: torch.Tensor,          # interpolated point
        t: torch.Tensor,            # time in [0,1]
        Phi: ProjectedICNN,         # convex functional
        z_init: torch.Tensor = None,  # warm start
        sobolev_s: float = 1.0,
    ) -> torch.Tensor:
        """Solve inner convex problem.
        
        IMPORTANT: This runs with torch.no_grad() or detaches intermediate
        computations to avoid building huge autograd graphs.
        
        Returns:
            z_star: (batch, T, n_x) solution tensor (detached!)
        """
        ...
```

### 2.4 OFFM Model (`optimal_ffm.py`) [NEW FILE]

**Purpose:** Main Optimal Functional Flow Matching model class

**Relation to existing code:** 
- Follows `FFMModel` class pattern from `functional_fm.py`
- Similar training loop structure with `train()` and `sample()` methods
- Uses same utility functions: `plot_loss_curve`, etc.

**Key difference from FFMModel:**
- FFMModel learns a vector field `u(t, x)` and uses ODE integration for sampling
- OFFMModel learns a convex potential `Φ(u)` and uses direct OT map for sampling

```python
class OFFMModel:
    """Optimal Functional Flow Matching for trajectory generation.
    
    Learns a convex functional Φ such that the induced OT map
    transports Gaussian reference measure to the data distribution.
    
    Analogous to FFMModel but:
    - Learns convex potential instead of vector field
    - Uses stop-gradient training instead of FM loss
    - Samples via OT map instead of ODE integration
    """
    
    def __init__(
        self,
        n_t: int,                    # time steps in trajectory
        n_x: int,                    # spatial points
        projection_dim: int = 256,   # dimension of Fourier projection
        icnn_hidden_dims: List[int] = [256, 256, 256],
        lambda_reg: float = 0.01,    # quadratic regularization
        sobolev_s: float = 1.0,      # Sobolev exponent
        gamma_kkt: float = 0.1,      # KKT loss weight
        solver_steps: int = 10,      # inner solver iterations
        solver_lr: float = 0.1,      # inner solver learning rate
        eps_t: float = 1e-3,         # minimum t to avoid singularity
        # GP params (same as FFMModel)
        kernel_length: float = 0.01,
        kernel_variance: float = 0.1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        
        # Convex potential (new for OFFM)
        self.Phi = ProjectedICNN(...)
        self.solver = ConvexSolver(...)
        
        # Reference measure (extends GPPrior pattern)
        self.gaussian_sampler = GaussianTrajectorySampler(
            n_t=n_t, n_x=n_x,
            kernel_length=kernel_length,
            kernel_variance=kernel_variance,
            device=device
        )
        ...
    
    def simulate(self, t, U_0, U_1):
        """Compute linear interpolation U_t = (1-t)*U_0 + t*U_1.
        
        Analogous to FFMModel.simulate() but simpler (just linear interp).
        """
        ...
    
    def compute_losses(
        self,
        U_0: torch.Tensor,   # reference samples (batch, T, n_x)
        U_1: torch.Tensor,   # data samples (batch, T, n_x)
    ) -> Dict[str, torch.Tensor]:
        """Compute gap and KKT losses with stop-gradient.
        
        Returns:
            {'total': L, 'gap': L_gap, 'kkt': L_kkt}
        """
        # 1. Sample t ~ Uniform(eps_t, 1)
        # 2. Compute U_t = (1-t)*U_0 + t*U_1
        # 3. Solve z* = argmin g_θ(z; U_t, t) [DETACH z*]
        # 4. Compute gap loss: g_θ(U_0) - g_θ(z*)
        # 5. Compute KKT loss: ||U_t - z* - t∇Φ(z*)||²
        # 6. Return weighted sum
        ...
    
    def train(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        epochs: int,
        scheduler: Optional = None,
        eval_int: int = 0,
        save_int: int = 0,
        save_path: Path = None,
        **kwargs
    ):
        """Training loop - follows FFMModel.train() pattern."""
        # Same structure as FFMModel.train() but calls compute_losses()
        ...
    
    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate trajectory samples.
        
        Unlike FFMModel which uses ODE integration,
        OFFM applies the learned OT map directly.
        """
        # 1. Sample U_0 ~ μ_0 (Gaussian)
        # 2. Apply OT map: U_1 = T(U_0) = U_0 - ∇Φ(U_0) or via prox
        ...
```

### 2.5 Gaussian Trajectory Sampler (`util/gaussian_trajectory.py`) [NEW FILE]

**Purpose:** Sample from Gaussian measure on trajectory Hilbert space

**Relation to existing code:**
- Extends/wraps `GPPrior` from `util/gaussian_process.py`
- Uses same kernel parameters (`kernel_length`, `kernel_variance`)
- GPPrior samples spatial functions; this samples space-time trajectories

```python
from util.gaussian_process import GPPrior

class GaussianTrajectorySampler:
    """Sample from Gaussian measure on L²(0,T; H^s(D)).
    
    Builds on GPPrior but samples trajectories instead of single functions.
    
    Options:
    1. Independent: Sample each time slice independently using GPPrior
    2. Spectral: E[|û(k_t, k_x)|²] ∝ (1+|k_t|²+|k_x|²)^{-α}
    3. Separable: C = C_t ⊗ C_x (use GPPrior for C_x)
    """
    
    def __init__(
        self,
        n_t: int,
        n_x: int,
        mode: str = "independent",  # 'independent', 'spectral', 'separable'
        # Same params as GPPrior for compatibility
        kernel_length: float = 0.01,
        kernel_variance: float = 0.1,
        smoothness_alpha: float = 2.0,  # for spectral mode
        device: str = "cuda",
    ):
        self.n_t = n_t
        self.n_x = n_x
        self.mode = mode
        self.device = device
        
        if mode == "independent":
            # Reuse existing GPPrior for each time slice
            self.gp = GPPrior(
                lengthscale=kernel_length,
                var=kernel_variance,
                device=device
            )
        ...
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample trajectories from Gaussian prior.
        
        Returns:
            (batch_size, n_t, n_x) tensor
        """
        if self.mode == "independent":
            # Sample n_t independent spatial functions using GPPrior
            grid = make_grid([self.n_x])
            # Sample (batch_size, n_t, n_x) by treating n_t as channels
            samples = self.gp.sample(grid, [self.n_x], 
                                     n_samples=batch_size, 
                                     n_channels=self.n_t)
            # Reshape from (batch, n_t, n_x) 
            return samples
        ...
```

---

## 3. KdV Experiment Script (`scripts/stochastic_kdv_offm.py`) [NEW FILE]

**Relation to existing code:**
- Follows `scripts/stochastic_kdv_ot.py` pattern for experiment structure
- Uses `load_stochastic_kdv()` from `util/util.py` (existing!)
- Uses `GenerationQualityMetrics` from `util/eval.py` for evaluation
- Uses `TrainingMonitor` from `util/ot_monitoring.py` for tracking

```python
"""
OFFM Experiments on Stochastic KdV.

This script runs Optimal Functional Flow Matching experiments on
stochastic KdV trajectories, treating each trajectory as an element
of L²(0,T; H^s(D)).

Data shape: (1200, 128, 101) -> trajectories of shape (1200, 101, 128)

Usage:
    python stochastic_kdv_offm.py
    python stochastic_kdv_offm.py --projection_dim 512 --epochs 200
"""

import sys
sys.path.append('../')

# REUSE existing utilities
from util.util import load_stochastic_kdv, plot_loss_curve
from util.eval import GenerationQualityMetrics
from util.ot_monitoring import TrainingMonitor

# NEW OFFM imports
from optimal_ffm import OFFMModel

# Configuration (similar structure to kdv_ot.py)
DEFAULT_CONFIG = {
    # Data - uses existing load_stochastic_kdv()
    'data_path': '../data/stochastic_kdv.mat',
    'mode': 'trajectory',  # NOT 'snapshot'! Key difference.
    'subsample_time': 1,
    'n_train': 1000,
    'n_test': 200,
    
    # Model (OFFM-specific)
    'projection_dim': 256,
    'projection_type': 'fourier',
    'icnn_hidden_dims': [256, 256, 256],
    'lambda_reg': 0.01,
    'sobolev_s': 1.0,
    
    # Training
    'gamma_kkt': 0.1,
    'solver_steps': 10,
    'solver_lr': 0.1,
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-3,
    
    # Reference measure (same params as FFMModel for comparison)
    'kernel_length': 0.01,
    'kernel_variance': 0.1,
}

def main():
    # Load data - REUSE existing function
    data = load_stochastic_kdv(args.data_path, mode='trajectory')
    # Shape: (1200, 101, 128) = (batch, T, N_x)
    
    # ... rest follows kdv_ot.py pattern ...
```

---

## 4. Implementation Order (Phases)

### Phase 1: Core Infrastructure (Priority: HIGH)
1. **`util/hilbert.py`** [NEW] - Sobolev norms via FFT
   - `get_sobolev_weights_1d()`
   - `sobolev_norm_1d()`
   - `trajectory_norm()`
   - `trajectory_inner_product()`
   - Unit tests

2. **`models/icnn.py`** [NEW] - ICNN architecture
   - `ICNN` class
   - Test convexity verification
   - Gradient computation

### Phase 2: Convex Optimization
3. **`util/gaussian_trajectory.py`** [NEW, extends GPPrior]
   - Wrapper around existing `GPPrior` for trajectory sampling
   - Spectral mode for smoothness control
   - Verify smoothness of samples

4. **`util/convex_solver.py`** [NEW] - Inner optimization
   - Gradient descent solver
   - L-BFGS solver (optional)
   - Warm-starting logic

### Phase 3: OFFM Model
5. **`models/icnn.py`** - Add `ProjectedICNN`
   - Fourier projection implementation
   - Gradient computation in Hilbert metric

6. **`optimal_ffm.py`** [NEW, follows FFMModel pattern]
   - Loss computation with stop-gradient
   - Training loop (reuse `plot_loss_curve` etc.)
   - Sampling (no ODE, direct map)

### Phase 4: Experiments
7. **`scripts/stochastic_kdv_offm.py`** [NEW, follows kdv_ot.py pattern]
   - Data loading via `load_stochastic_kdv(mode='trajectory')` [EXISTING]
   - Training with `TrainingMonitor` [EXISTING]
   - Evaluation with `GenerationQualityMetrics` [EXISTING]
   - Visualization

### What We DON'T Need to Implement (Already Exists)
- ❌ Data loading → use `load_stochastic_kdv()` from `util/util.py`
- ❌ Quality metrics → use `GenerationQualityMetrics` from `util/eval.py`
- ❌ Training monitoring → use `TrainingMonitor` from `util/ot_monitoring.py`
- ❌ Plot utilities → use `plot_loss_curve()` from `util/util.py`
- ❌ Grid utilities → use `make_grid()` from `util/util.py`

---

## 5. Key Implementation Details

### 5.1 Stop-Gradient Pattern
```python
# During training - critical to avoid expensive unrolled backprop

# Solve inner problem (NO autograd graph!)
with torch.no_grad():
    z_star = solver.solve(U_t, t, Phi)

# Detach for loss computation
z_det = z_star.detach()  # CRUCIAL

# Compute losses with proper gradients
Phi_U0 = Phi(U_0)         # gradients flow through Phi
Phi_z = Phi(z_det)        # gradients flow through Phi, not z_det
grad_Phi_z = Phi.gradient(z_det)  # gradients flow through Phi
```

### 5.2 FFT-based Sobolev Norm (1D Periodic)
```python
def sobolev_norm_squared_1d(u, s, L):
    """u: (batch, n_x) tensor on periodic domain [0, L)."""
    n_x = u.shape[-1]
    dx = L / n_x
    
    # FFT
    u_hat = torch.fft.rfft(u, dim=-1)
    
    # Frequencies: k = 0, 1, 2, ..., n_x//2
    k = torch.fft.rfftfreq(n_x, d=dx/(2*np.pi))  # scaled frequencies
    
    # Sobolev weights: (1 + |k|²)^s
    weights = (1 + k**2)**s
    
    # Parseval: ||u||²_{H^s} = sum_k (1+|k|²)^s |û(k)|²
    # Factor of 2 for negative frequencies (except k=0, k=n_x/2)
    norm_sq = dx * torch.sum(weights * torch.abs(u_hat)**2, dim=-1)
    
    return norm_sq
```

### 5.3 Projection to Finite Dimensions
```python
def fourier_projection(u, n_modes_t, n_modes_x):
    """Project trajectory to truncated Fourier coefficients.
    
    u: (batch, n_t, n_x) trajectory tensor
    
    Returns: (batch, 2 * n_modes_t * n_modes_x) real vector
             (real and imaginary parts concatenated)
    """
    # 2D FFT over (t, x)
    u_hat = torch.fft.rfft2(u, dim=(-2, -1))
    
    # Keep only low frequencies
    u_hat_trunc = u_hat[:, :n_modes_t, :n_modes_x]
    
    # Flatten to real vector
    real_part = u_hat_trunc.real.reshape(u.shape[0], -1)
    imag_part = u_hat_trunc.imag.reshape(u.shape[0], -1)
    
    return torch.cat([real_part, imag_part], dim=-1)
```

### 5.4 ICNN Architecture
```python
class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        # Standard path (can have any weights)
        self.fc_x = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.fc_x.append(nn.Linear(input_dim, hidden_dims[i+1]))
        
        # Convex path (non-negative weights)
        self.fc_z = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fc_z.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=False))
        
        # Output layer (non-negative weights)
        self.fc_out = nn.Linear(hidden_dims[-1], 1, bias=False)
    
    def forward(self, x):
        z = F.leaky_relu(self.fc_x[0](x), 0.2)
        
        for i in range(len(self.fc_z)):
            # Use softplus to ensure non-negative weights
            W_z = F.softplus(self.fc_z[i].weight)
            z = F.leaky_relu(
                F.linear(z, W_z) + self.fc_x[i+1](x),
                0.2
            )
        
        W_out = F.softplus(self.fc_out.weight)
        return F.linear(z, W_out)
```

---

## 6. Evaluation Metrics

For trajectory generation quality:

1. **Marginal Statistics**
   - Mean trajectory profile: E[u(t,x)]
   - Variance profile: Var[u(t,x)]
   - Higher moments at each (t,x)

2. **Spectral Properties**
   - Power spectrum in space: E[|û(t,k)|²]
   - Power spectrum in time: E[|û(ω,x)|²]
   - Energy cascade comparison

3. **Temporal Coherence**
   - Autocorrelation in time
   - Cross-correlation between spatial points
   - Trajectory smoothness metrics

4. **Distribution Distances**
   - Wasserstein distance (empirical)
   - Maximum Mean Discrepancy (MMD)
   - Sliced Wasserstein

---

## 7. Potential Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Inner solver convergence | Warm-start from previous z*, adaptive step size |
| Memory for long trajectories | Gradient checkpointing, smaller batch sizes |
| ICNN expressivity | Deeper networks, skip connections, wider layers |
| Projection information loss | Adaptive projection dim, multi-scale projections |
| Numerical stability | Proper normalization, gradient clipping |

---

## 8. File Structure After Implementation

```
function_fm_OT/
├── optimal_ffm.py              # Main OFFM model class [NEW]
├── functional_fm.py            # [EXISTING] FFMModel - reference for structure
├── functional_fm_ot.py         # [EXISTING] FFMModelOT - OT-enhanced version
├── models/
│   ├── icnn.py                 # ICNN and ProjectedICNN [NEW]
│   ├── fno.py                  # [EXISTING] FNO architecture
│   └── ...
├── util/
│   ├── hilbert.py              # Sobolev norms, inner products [NEW]
│   ├── convex_solver.py        # Inner optimization [NEW]
│   ├── gaussian_trajectory.py  # Reference measure [NEW] - extends GPPrior
│   ├── gaussian_process.py     # [EXISTING] GPPrior class
│   ├── util.py                 # [EXISTING] load_stochastic_kdv(), make_grid()
│   ├── eval.py                 # [EXISTING] GenerationQualityMetrics
│   └── ot_monitoring.py        # [EXISTING] TrainingMonitor
├── scripts/
│   ├── stochastic_kdv_offm.py  # KdV OFFM experiments [NEW]
│   ├── stochastic_kdv.py       # [EXISTING] KdV FFM experiments - reference
│   ├── stochastic_kdv_ot.py    # [EXISTING] KdV OT-FFM experiments - reference
│   └── ...
└── optimal_ffm/
    ├── offm.md                 # Research notes
    ├── ofm.md                  # OFM paper reference
    └── implementation.md       # This document
```

### Summary of Changes
| Type | Files | Description |
|------|-------|-------------|
| **NEW** | `optimal_ffm.py`, `models/icnn.py`, `util/hilbert.py`, `util/convex_solver.py`, `util/gaussian_trajectory.py`, `scripts/stochastic_kdv_offm.py` | Core OFFM implementation |
| **REUSE** | `util/gaussian_process.py`, `util/util.py`, `util/eval.py` | Data loading, GP sampling base, metrics |
| **PATTERN** | `functional_fm.py`, `scripts/stochastic_kdv_ot.py` | Follow same class/script structure |

---

## 9. Testing Strategy

1. **Unit Tests**
   - Sobolev norm correctness (compare with analytical for simple functions)
   - ICNN convexity verification (check Hessian positive semi-definite)
   - Projection/unprojection round-trip

2. **Integration Tests**
   - Inner solver convergence on synthetic problems
   - Loss computation gradient flow
   - Sample quality on toy distributions

3. **Ablation Studies**
   - Projection dimension: 64, 128, 256, 512
   - Sobolev exponent: s = 0, 1, 2
   - ICNN depth: 2, 3, 4 layers
   - Solver steps: 5, 10, 20
   - Loss weighting: γ = 0.01, 0.1, 1.0

---

## 10. Next Steps

1. [ ] Implement `util/hilbert.py` with unit tests
2. [ ] Implement `models/icnn.py` with convexity tests
3. [ ] Implement `util/gaussian_trajectory.py`
4. [ ] Implement `util/convex_solver.py`
5. [ ] Implement `optimal_ffm.py` (main model)
6. [ ] Create `scripts/stochastic_kdv_offm.py`
7. [ ] Run baseline experiments
8. [ ] Iterate on hyperparameters
9. [ ] Compare with FFM, DDPM baselines

---

## Summary

OFFM extends OFM to function spaces by:
1. Working in trajectory Hilbert spaces with FFT-based Sobolev structure
2. Projecting to finite dimensions + ICNN for convex functional
3. Using stop-gradient training with gap + KKT losses

The implementation follows a modular design with clear separation between:
- Hilbert space utilities (norms, inner products)
- Neural network architectures (ICNN)
- Optimization (convex solver)
- Model class (OFFM)
- Experiments (KdV scripts)
