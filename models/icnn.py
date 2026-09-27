"""
Input Convex Neural Network (ICNN) for Optimal Functional Flow Matching.

Implements convex-by-construction neural networks for parameterizing
convex potentials Φ: H_traj → ℝ.

Architecture ensures convexity via:
1. Non-negative weights on the "z-path" (enforced via softplus)
2. Convex non-decreasing activations (LeakyReLU, ReLU, softplus)
3. Skip connections from input preserve convexity

References:
    - Amos et al. (2017): Input Convex Neural Networks
    - OFM paper: Optimal Flow Matching
    - OFFM formulation in optimal_ffm/offm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Literal
import numpy as np


class ICNN(nn.Module):
    """Input Convex Neural Network.
    
    A neural network φ(z) that is convex in its input z by construction.
    
    Architecture:
        Layer 0: z₀ = σ(W₀ᵡ x + b₀)
        Layer k: zₖ = σ(Wₖᶻ zₖ₋₁ + Wₖᵡ x + bₖ)  for k = 1, ..., L-1
        Output:  y  = Wₗᶻ zₗ₋₁ + Wₗᵡ x + bₗ
    
    Convexity is ensured by:
        - Wₖᶻ ≥ 0 (non-negative weights on z-path)
        - σ is convex and non-decreasing (e.g., ReLU, LeakyReLU, softplus)
    
    Args:
        input_dim: Dimension of input z
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'leaky_relu', 'softplus', 'elu')
        final_activation: Whether to apply activation on final layer
        positive_weights: Method for ensuring non-negative weights ('softplus', 'relu', 'exp')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: Literal["relu", "leaky_relu", "softplus", "elu"] = "leaky_relu",
        final_activation: bool = False,
        positive_weights: Literal["softplus", "relu", "exp", "abs"] = "softplus",
        negative_slope: float = 0.2,  # For leaky_relu
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.final_activation = final_activation
        self.positive_weights = positive_weights
        self.negative_slope = negative_slope
        
        # Build layers
        dims = [input_dim] + hidden_dims + [1]
        
        # First layer: only x-path (no z yet)
        self.fc_x_first = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers: both z-path and x-path
        self.fc_z = nn.ModuleList()  # z-path (non-negative weights)
        self.fc_x = nn.ModuleList()  # x-path (skip connections)
        
        for i in range(len(hidden_dims) - 1):
            # z-path: no bias (or bias is fine, only weights need to be non-negative)
            self.fc_z.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False))
            # x-path: skip connection from input
            self.fc_x.append(nn.Linear(input_dim, hidden_dims[i + 1], bias=True))
        
        # Output layer
        self.fc_z_out = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.fc_x_out = nn.Linear(input_dim, 1, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize z-path weights to be positive after softplus
        for fc in self.fc_z:
            # Initialize so that softplus(w) starts small positive
            nn.init.uniform_(fc.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_z_out.weight, -0.1, 0.1)
        
        # Initialize x-path weights normally
        nn.init.xavier_uniform_(self.fc_x_first.weight)
        for fc in self.fc_x:
            nn.init.xavier_uniform_(fc.weight)
        nn.init.xavier_uniform_(self.fc_x_out.weight)
    
    def _get_positive_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Ensure weight is non-negative."""
        if self.positive_weights == "softplus":
            return F.softplus(weight)
        elif self.positive_weights == "relu":
            return F.relu(weight)
        elif self.positive_weights == "exp":
            return torch.exp(weight.clamp(max=10))  # Clamp to avoid overflow
        elif self.positive_weights == "abs":
            return torch.abs(weight)
        else:
            raise ValueError(f"Unknown positive_weights: {self.positive_weights}")
    
    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "leaky_relu":
            return F.leaky_relu(x, self.negative_slope)
        elif self.activation_name == "softplus":
            return F.softplus(x)
        elif self.activation_name == "elu":
            return F.elu(x) + 1  # Shift ELU to be non-negative
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, input_dim) input tensor
            
        Returns:
            y: (batch,) scalar output (convex in x)
        """
        # First layer
        z = self._activation(self.fc_x_first(x))
        
        # Hidden layers
        for fc_z, fc_x in zip(self.fc_z, self.fc_x):
            # Get non-negative weights for z-path
            W_z = self._get_positive_weight(fc_z.weight)
            # z_new = σ(W_z @ z + W_x @ x + b)
            z = self._activation(F.linear(z, W_z) + fc_x(x))
        
        # Output layer
        W_z_out = self._get_positive_weight(self.fc_z_out.weight)
        y = F.linear(z, W_z_out) + self.fc_x_out(x)
        
        if self.final_activation:
            y = self._activation(y)
        
        return y.squeeze(-1)  # (batch,)


class ProjectedICNN(nn.Module):
    """Convex functional on trajectory space via projection + ICNN.
    
    Implements Φ(u) = φ(Pu) + (λ/2)||u||²_traj
    
    where:
        - u ∈ H_traj is a trajectory (batch, n_t, n_x)
        - P: H_traj → ℝᵐ is a linear projection (e.g., truncated Fourier)
        - φ: ℝᵐ → ℝ is an ICNN
        - λ ≥ 0 is a regularization coefficient
    
    The functional is convex because:
        - P is linear
        - φ is convex (ICNN)
        - ||·||² is convex
        - Sum of convex functions is convex
    
    Args:
        n_t: Number of time steps
        n_x: Number of spatial points
        projection_dim: Output dimension of projection (m)
        projection_type: Type of projection ('fourier', 'random', 'identity')
        icnn_hidden_dims: Hidden dimensions for ICNN
        lambda_reg: Quadratic regularization coefficient
        sobolev_s: Sobolev exponent for norm (default 0 = L²)
    """
    
    def __init__(
        self,
        n_t: int,
        n_x: int,
        projection_dim: int = 256,
        projection_type: Literal["fourier", "random", "identity"] = "fourier",
        icnn_hidden_dims: List[int] = [256, 256],
        lambda_reg: float = 0.0,
        sobolev_s: float = 0.0,
        icnn_activation: str = "leaky_relu",
        dt: float = 1.0,
        L: float = 1.0,
    ):
        super().__init__()
        
        self.n_t = n_t
        self.n_x = n_x
        self.projection_dim = projection_dim
        self.projection_type = projection_type
        self.lambda_reg = lambda_reg
        self.sobolev_s = sobolev_s
        self.dt = dt
        self.L = L
        
        # Initialize projection
        self._init_projection()
        
        # ICNN takes projected vector as input
        self.icnn = ICNN(
            input_dim=projection_dim,
            hidden_dims=icnn_hidden_dims,
            activation=icnn_activation,
        )
    
    def _init_projection(self):
        """Initialize the projection operator P."""
        if self.projection_type == "fourier":
            self._init_fourier_projection()
        elif self.projection_type == "random":
            self._init_random_projection()
        elif self.projection_type == "identity":
            self._init_identity_projection()
        else:
            raise ValueError(f"Unknown projection_type: {self.projection_type}")
    
    def _init_fourier_projection(self):
        """Initialize Fourier projection (truncated FFT coefficients)."""
        # Determine how many Fourier modes to keep
        # We want projection_dim real numbers from complex FFT coefficients
        # For rfft2 on (n_t, n_x), output is (n_t, n_x//2+1) complex
        # Each complex number gives 2 real numbers
        
        n_freq_t = self.n_t // 2 + 1
        n_freq_x = self.n_x // 2 + 1
        max_coeffs = n_freq_t * n_freq_x
        
        # Number of complex coefficients to keep
        n_complex = min(self.projection_dim // 2, max_coeffs)
        
        # Compute which frequency indices to keep (low frequencies first)
        # Create a grid of (freq_t, freq_x) and sort by |freq|²
        freq_t = torch.arange(n_freq_t).float()
        freq_x = torch.arange(n_freq_x).float()
        freq_grid_t, freq_grid_x = torch.meshgrid(freq_t, freq_x, indexing='ij')
        freq_magnitude = freq_grid_t**2 + freq_grid_x**2
        
        # Flatten and get indices of lowest frequencies
        flat_magnitude = freq_magnitude.flatten()
        _, sorted_indices = flat_magnitude.sort()
        keep_indices = sorted_indices[:n_complex]
        
        # Convert flat indices to 2D indices
        keep_t = keep_indices // n_freq_x
        keep_x = keep_indices % n_freq_x
        
        # Register as buffers (not parameters)
        self.register_buffer('fourier_keep_t', keep_t)
        self.register_buffer('fourier_keep_x', keep_x)
        self.n_fourier_coeffs = n_complex
        
        # Actual projection dimension (real + imag)
        self.actual_proj_dim = 2 * n_complex
    
    def _init_random_projection(self):
        """Initialize random projection matrix."""
        total_dim = self.n_t * self.n_x
        # Random Gaussian projection (scaled for stability)
        proj_matrix = torch.randn(self.projection_dim, total_dim) / np.sqrt(total_dim)
        self.register_buffer('proj_matrix', proj_matrix)
        self.actual_proj_dim = self.projection_dim
    
    def _init_identity_projection(self):
        """Identity projection (flatten only, must match dims)."""
        total_dim = self.n_t * self.n_x
        if self.projection_dim != total_dim:
            raise ValueError(
                f"For identity projection, projection_dim ({self.projection_dim}) "
                f"must equal n_t * n_x ({total_dim})"
            )
        self.actual_proj_dim = total_dim
    
    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Project trajectory to finite-dimensional vector.
        
        Args:
            u: (batch, n_t, n_x) trajectory tensor
            
        Returns:
            z: (batch, projection_dim) projected vector
        """
        if self.projection_type == "fourier":
            return self._fourier_project(u)
        elif self.projection_type == "random":
            return self._random_project(u)
        elif self.projection_type == "identity":
            return u.reshape(u.shape[0], -1)
        else:
            raise ValueError(f"Unknown projection_type: {self.projection_type}")
    
    def _fourier_project(self, u: torch.Tensor) -> torch.Tensor:
        """Project via truncated 2D FFT."""
        batch = u.shape[0]
        
        # 2D FFT
        u_hat = torch.fft.rfft2(u, dim=(-2, -1))  # (batch, n_t, n_x//2+1) complex
        
        # Select kept frequencies
        u_hat_selected = u_hat[:, self.fourier_keep_t, self.fourier_keep_x]  # (batch, n_coeffs)
        
        # Concatenate real and imaginary parts
        z_real = u_hat_selected.real
        z_imag = u_hat_selected.imag
        z = torch.cat([z_real, z_imag], dim=-1)  # (batch, 2*n_coeffs)
        
        # Normalize for numerical stability
        z = z / np.sqrt(self.n_t * self.n_x)
        
        return z
    
    def _random_project(self, u: torch.Tensor) -> torch.Tensor:
        """Project via random matrix."""
        u_flat = u.reshape(u.shape[0], -1)  # (batch, n_t * n_x)
        z = F.linear(u_flat, self.proj_matrix)  # (batch, projection_dim)
        return z
    
    def unproject_gradient(self, grad_z: torch.Tensor) -> torch.Tensor:
        """Unproject gradient from projected space to trajectory space.
        
        Given ∇_z φ(z) where z = Pu, computes P^T ∇_z φ(z).
        
        Args:
            grad_z: (batch, projection_dim) gradient in projected space
            
        Returns:
            grad_u: (batch, n_t, n_x) gradient in trajectory space
        """
        if self.projection_type == "fourier":
            return self._fourier_unproject(grad_z)
        elif self.projection_type == "random":
            return self._random_unproject(grad_z)
        elif self.projection_type == "identity":
            return grad_z.reshape(-1, self.n_t, self.n_x)
        else:
            raise ValueError(f"Unknown projection_type: {self.projection_type}")
    
    def _fourier_unproject(self, grad_z: torch.Tensor) -> torch.Tensor:
        """Unproject Fourier gradient."""
        batch = grad_z.shape[0]
        n_freq_t = self.n_t // 2 + 1
        n_freq_x = self.n_x // 2 + 1
        
        # Split real and imaginary parts
        grad_real = grad_z[:, :self.n_fourier_coeffs]
        grad_imag = grad_z[:, self.n_fourier_coeffs:]
        grad_complex = torch.complex(grad_real, grad_imag)
        
        # Unnormalize
        grad_complex = grad_complex / np.sqrt(self.n_t * self.n_x)
        
        # Place back into full frequency grid
        grad_u_hat = torch.zeros(batch, n_freq_t, n_freq_x, dtype=grad_complex.dtype, device=grad_z.device)
        grad_u_hat[:, self.fourier_keep_t, self.fourier_keep_x] = grad_complex
        
        # Inverse FFT
        grad_u = torch.fft.irfft2(grad_u_hat, s=(self.n_t, self.n_x))
        
        return grad_u
    
    def _random_unproject(self, grad_z: torch.Tensor) -> torch.Tensor:
        """Unproject random projection gradient."""
        # P^T @ grad_z
        grad_u_flat = F.linear(grad_z, self.proj_matrix.T)  # (batch, n_t * n_x)
        return grad_u_flat.reshape(-1, self.n_t, self.n_x)
    
    def trajectory_norm_squared(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ||u||²_traj (L² or Sobolev norm)."""
        if self.sobolev_s == 0.0:
            # Simple L² norm
            return self.dt * self.L / self.n_x * (u ** 2).sum(dim=(-2, -1))
        else:
            # Import Sobolev norm
            from util.hilbert import trajectory_norm_squared
            return trajectory_norm_squared(u, s=self.sobolev_s, dt=self.dt, L=self.L)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Compute Φ(u) = φ(Pu) + (λ/2)||u||²_traj.
        
        Args:
            u: (batch, n_t, n_x) trajectory tensor
            
        Returns:
            Phi: (batch,) functional values
        """
        # Project
        z = self.project(u)
        
        # Apply ICNN
        phi_z = self.icnn(z)  # (batch,)
        
        # Add quadratic regularization
        if self.lambda_reg > 0:
            norm_sq = self.trajectory_norm_squared(u)
            Phi = phi_z + 0.5 * self.lambda_reg * norm_sq
        else:
            Phi = phi_z
        
        return Phi
    
    def gradient(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ∇Φ(u) = P^T ∇φ(Pu) + λu.
        
        Args:
            u: (batch, n_t, n_x) trajectory tensor
            
        Returns:
            grad: (batch, n_t, n_x) gradient
        """
        # Enable gradient computation even if we're in no_grad context
        with torch.enable_grad():
            u_for_grad = u.detach().clone().requires_grad_(True)
            
            # Forward pass through ICNN
            z = self.project(u_for_grad)
            phi_z = self.icnn(z)
            
            # Backprop to get gradient w.r.t. u
            grad_phi = torch.autograd.grad(
                phi_z.sum(), u_for_grad, create_graph=False
            )[0]
        
        # Add regularization gradient (outside enable_grad since u doesn't need grad)
        if self.lambda_reg > 0:
            grad = grad_phi + self.lambda_reg * u
        else:
            grad = grad_phi
        
        return grad.detach()


def test_icnn():
    """Quick test for ICNN."""
    print("Testing ICNN...")
    
    icnn = ICNN(input_dim=32, hidden_dims=[64, 64])
    x = torch.randn(5, 32)
    y = icnn(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output: {y}")
    
    # Test convexity numerically
    print("\n  Testing convexity (should be convex along any line)...")
    x1 = torch.randn(1, 32)
    x2 = torch.randn(1, 32)
    
    t_vals = torch.linspace(0, 1, 11)
    y_interp = []
    y_line = []
    
    for t in t_vals:
        x_t = (1 - t) * x1 + t * x2
        y_t = icnn(x_t)
        y_interp.append(y_t.item())
        y_line.append((1 - t.item()) * icnn(x1).item() + t.item() * icnn(x2).item())
    
    # For convex function, y_interp <= y_line (Jensen's inequality)
    y_interp = torch.tensor(y_interp)
    y_line = torch.tensor(y_line)
    
    is_convex = (y_interp <= y_line + 1e-5).all()
    print(f"  Convexity check passed: {is_convex}")
    
    return is_convex


def test_projected_icnn():
    """Quick test for ProjectedICNN."""
    print("\nTesting ProjectedICNN...")
    
    n_t, n_x = 20, 64
    
    for proj_type in ["fourier", "random"]:
        print(f"\n  Projection type: {proj_type}")
        
        model = ProjectedICNN(
            n_t=n_t, n_x=n_x,
            projection_dim=128,
            projection_type=proj_type,
            icnn_hidden_dims=[64, 64],
            lambda_reg=0.1,
        )
        
        u = torch.randn(5, n_t, n_x)
        Phi = model(u)
        grad = model.gradient(u)
        
        print(f"    Input shape: {u.shape}")
        print(f"    Φ(u) shape: {Phi.shape}")
        print(f"    ∇Φ(u) shape: {grad.shape}")
        print(f"    Φ(u): {Phi}")


if __name__ == "__main__":
    test_icnn()
    test_projected_icnn()
