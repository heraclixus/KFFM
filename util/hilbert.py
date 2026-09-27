"""
Hilbert Space Utilities for Optimal Functional Flow Matching.

Implements discrete Sobolev norms and inner products via FFT for periodic domains.
Used for defining the trajectory Hilbert space L²(0,T; H^s(D)).

References:
    - FFT-based Sobolev norms on periodic domains
    - OFFM formulation in optimal_ffm/offm.md
"""

import torch
import numpy as np
from typing import Optional, Tuple


def get_sobolev_weights_1d(
    n_x: int, 
    s: float = 1.0, 
    L: float = 1.0,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Get Sobolev weights (1 + |k|²)^s for 1D FFT on periodic domain.
    
    For periodic domain [0, L), the Fourier frequencies are k = 2πn/L
    where n = 0, 1, ..., N/2, -N/2+1, ..., -1.
    
    Using rfft, we only get non-negative frequencies: n = 0, 1, ..., N//2.
    
    Args:
        n_x: Number of spatial grid points
        s: Sobolev exponent (s=0 gives L², s=1 gives H¹, etc.)
        L: Domain length (default 1.0)
        device: Torch device
        dtype: Torch dtype
        
    Returns:
        weights: (n_x//2 + 1,) tensor of Sobolev weights
    """
    # Frequencies for rfft: 0, 1, 2, ..., n_x//2
    # Physical frequencies: k_n = 2πn/L
    n_freqs = n_x // 2 + 1
    freq_indices = torch.arange(n_freqs, device=device, dtype=dtype)
    
    # Physical wavenumbers
    k = (2 * np.pi / L) * freq_indices
    
    # Sobolev weights: (1 + |k|²)^s
    weights = (1.0 + k**2) ** s
    
    return weights


def sobolev_norm_squared_1d(
    u: torch.Tensor,
    s: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute squared H^s Sobolev norm for 1D periodic functions via FFT.
    
    ||u||²_{H^s} = Δx Σ_k (1 + |k|²)^s |û(k)|²
    
    Uses Parseval's theorem. The factor accounts for:
    - rfft normalization
    - Proper discretization scaling
    
    Args:
        u: (..., n_x) tensor - last dim is spatial
        s: Sobolev exponent
        L: Domain length
        
    Returns:
        norm_sq: (...) tensor of squared norms
    """
    n_x = u.shape[-1]
    dx = L / n_x
    
    # Get Sobolev weights
    weights = get_sobolev_weights_1d(n_x, s, L, device=u.device, dtype=u.dtype)
    
    # FFT (real-to-complex)
    u_hat = torch.fft.rfft(u, dim=-1)
    
    # Power spectrum |û(k)|²
    power = torch.abs(u_hat) ** 2
    
    # Weight by Sobolev weights
    # For rfft, we need to account for negative frequencies (except DC and Nyquist)
    # DC (k=0) appears once, Nyquist (k=N/2, if N even) appears once
    # All other frequencies have conjugate pairs, so multiply by 2
    n_freqs = n_x // 2 + 1
    multiplier = 2.0 * torch.ones(n_freqs, device=u.device, dtype=u.dtype)
    multiplier[0] = 1.0  # DC component
    if n_x % 2 == 0:
        multiplier[-1] = 1.0  # Nyquist component (if n_x is even)
    
    # Squared norm: Σ_k (1+|k|²)^s |û(k)|² × multiplier × dx/n_x
    # The dx/n_x factor comes from Parseval: ||u||² = (1/n_x) Σ_k |û(k)|²
    # and we want Δx Σ_k ... in physical units
    weighted_power = weights * multiplier * power
    norm_sq = (dx / n_x) * weighted_power.sum(dim=-1)
    
    return norm_sq


def sobolev_norm_1d(
    u: torch.Tensor,
    s: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute H^s Sobolev norm for 1D periodic functions.
    
    Args:
        u: (..., n_x) tensor
        s: Sobolev exponent
        L: Domain length
        
    Returns:
        norm: (...) tensor of norms
    """
    return torch.sqrt(sobolev_norm_squared_1d(u, s, L))


def sobolev_inner_product_1d(
    u: torch.Tensor,
    v: torch.Tensor,
    s: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute H^s inner product for 1D periodic functions via FFT.
    
    ⟨u, v⟩_{H^s} = Δx Σ_k (1 + |k|²)^s û(k) v̂(k)*
    
    Args:
        u: (..., n_x) tensor
        v: (..., n_x) tensor (same shape as u)
        s: Sobolev exponent
        L: Domain length
        
    Returns:
        inner: (...) tensor of inner products (real-valued)
    """
    assert u.shape == v.shape, f"Shape mismatch: {u.shape} vs {v.shape}"
    n_x = u.shape[-1]
    dx = L / n_x
    
    # Get Sobolev weights
    weights = get_sobolev_weights_1d(n_x, s, L, device=u.device, dtype=u.dtype)
    
    # FFT
    u_hat = torch.fft.rfft(u, dim=-1)
    v_hat = torch.fft.rfft(v, dim=-1)
    
    # Cross-spectrum: û(k) × v̂(k)*
    cross = u_hat * torch.conj(v_hat)
    
    # Multiplier for negative frequencies
    n_freqs = n_x // 2 + 1
    multiplier = 2.0 * torch.ones(n_freqs, device=u.device, dtype=u.dtype)
    multiplier[0] = 1.0
    if n_x % 2 == 0:
        multiplier[-1] = 1.0
    
    # Inner product (take real part)
    weighted_cross = weights * multiplier * cross
    inner = (dx / n_x) * weighted_cross.sum(dim=-1).real
    
    return inner


# =============================================================================
# Trajectory Space Norms and Inner Products
# =============================================================================

def trajectory_norm_squared(
    u: torch.Tensor,
    s: float = 1.0,
    dt: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute squared trajectory norm ||u||²_{L²(0,T; H^s(D))}.
    
    ||u||²_traj = Δt Σ_n ||u_n||²_{H^s(D)}
    
    Args:
        u: (batch, n_t, n_x) tensor - trajectories
        s: Spatial Sobolev exponent
        dt: Time step
        L: Spatial domain length
        
    Returns:
        norm_sq: (batch,) tensor of squared norms
    """
    # Compute H^s norm for each time slice: (batch, n_t)
    spatial_norms_sq = sobolev_norm_squared_1d(u, s, L)
    
    # Sum over time with quadrature weight
    norm_sq = dt * spatial_norms_sq.sum(dim=-1)
    
    return norm_sq


def trajectory_norm(
    u: torch.Tensor,
    s: float = 1.0,
    dt: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute trajectory norm ||u||_{L²(0,T; H^s(D))}.
    
    Args:
        u: (batch, n_t, n_x) tensor - trajectories
        s: Spatial Sobolev exponent
        dt: Time step
        L: Spatial domain length
        
    Returns:
        norm: (batch,) tensor of norms
    """
    return torch.sqrt(trajectory_norm_squared(u, s, dt, L))


def trajectory_inner_product(
    u: torch.Tensor,
    v: torch.Tensor,
    s: float = 1.0,
    dt: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute trajectory inner product ⟨u, v⟩_{L²(0,T; H^s(D))}.
    
    ⟨u, v⟩_traj = Δt Σ_n ⟨u_n, v_n⟩_{H^s(D)}
    
    Args:
        u: (batch, n_t, n_x) tensor
        v: (batch, n_t, n_x) tensor
        s: Spatial Sobolev exponent
        dt: Time step
        L: Spatial domain length
        
    Returns:
        inner: (batch,) tensor of inner products
    """
    assert u.shape == v.shape, f"Shape mismatch: {u.shape} vs {v.shape}"
    
    # Compute H^s inner product for each time slice: (batch, n_t)
    spatial_inners = sobolev_inner_product_1d(u, v, s, L)
    
    # Sum over time with quadrature weight
    inner = dt * spatial_inners.sum(dim=-1)
    
    return inner


# =============================================================================
# Optional: Time-Regularized Trajectory Norm (H¹ in time)
# =============================================================================

def trajectory_norm_squared_with_time_reg(
    u: torch.Tensor,
    s: float = 1.0,
    beta: float = 0.1,
    dt: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute trajectory norm with time derivative regularization.
    
    ||u||²_traj = Δt Σ_n ||u_n||²_{H^s} + β Δt Σ_n ||(u_{n+1}-u_n)/Δt||²_{H^{s-1}}
    
    This encourages temporal smoothness.
    
    Args:
        u: (batch, n_t, n_x) tensor - trajectories
        s: Spatial Sobolev exponent
        beta: Time regularization weight
        dt: Time step
        L: Spatial domain length
        
    Returns:
        norm_sq: (batch,) tensor of squared norms
    """
    # Spatial norm term
    spatial_norm_sq = trajectory_norm_squared(u, s, dt, L)
    
    if beta == 0.0:
        return spatial_norm_sq
    
    # Time derivative term: (u_{n+1} - u_n) / dt
    # Shape: (batch, n_t - 1, n_x)
    du_dt = (u[:, 1:, :] - u[:, :-1, :]) / dt
    
    # H^{s-1} norm of time derivatives
    time_deriv_norm_sq = sobolev_norm_squared_1d(du_dt, s - 1, L)  # (batch, n_t-1)
    time_reg = beta * dt * time_deriv_norm_sq.sum(dim=-1)
    
    return spatial_norm_sq + time_reg


# =============================================================================
# Utility: Compute Sobolev gradient (for optimization in Hilbert space)
# =============================================================================

def apply_sobolev_inverse(
    u: torch.Tensor,
    s: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Apply inverse Sobolev operator (1 + |k|²)^{-s} in Fourier domain.
    
    This converts an L² gradient to an H^s gradient (Riesz representation).
    
    Args:
        u: (..., n_x) tensor
        s: Sobolev exponent
        L: Domain length
        
    Returns:
        result: (..., n_x) tensor
    """
    n_x = u.shape[-1]
    
    # Inverse weights: (1 + |k|²)^{-s}
    weights = get_sobolev_weights_1d(n_x, -s, L, device=u.device, dtype=u.dtype)
    
    # Apply in Fourier domain
    u_hat = torch.fft.rfft(u, dim=-1)
    u_hat_scaled = u_hat * weights
    result = torch.fft.irfft(u_hat_scaled, n=n_x, dim=-1)
    
    return result


def trajectory_sobolev_inverse(
    u: torch.Tensor,
    s: float = 1.0,
    L: float = 1.0,
) -> torch.Tensor:
    """Apply inverse Sobolev operator to trajectory (each time slice).
    
    Args:
        u: (batch, n_t, n_x) tensor
        s: Sobolev exponent
        L: Domain length
        
    Returns:
        result: (batch, n_t, n_x) tensor
    """
    batch, n_t, n_x = u.shape
    
    # Reshape to apply per time slice
    u_flat = u.reshape(-1, n_x)  # (batch * n_t, n_x)
    result_flat = apply_sobolev_inverse(u_flat, s, L)
    result = result_flat.reshape(batch, n_t, n_x)
    
    return result


# =============================================================================
# Testing / Verification
# =============================================================================

def verify_parseval(n_x: int = 128, L: float = 1.0, atol: float = 1e-5) -> bool:
    """Verify that s=0 norm equals L² norm (Parseval's theorem).
    
    Returns True if verification passes.
    """
    # Random function
    u = torch.randn(10, n_x)
    dx = L / n_x
    
    # L² norm via direct integration
    l2_norm_sq = dx * (u ** 2).sum(dim=-1)
    
    # L² norm via FFT (s=0)
    fft_norm_sq = sobolev_norm_squared_1d(u, s=0.0, L=L)
    
    # Check closeness
    close = torch.allclose(l2_norm_sq, fft_norm_sq, atol=atol)
    
    if not close:
        print(f"Parseval verification failed!")
        print(f"  Direct L² norm²: {l2_norm_sq[:3]}")
        print(f"  FFT L² norm²:    {fft_norm_sq[:3]}")
    
    return close


if __name__ == "__main__":
    # Quick tests
    print("Testing Hilbert utilities...")
    
    # Test Parseval
    print("\n1. Parseval verification (s=0 should equal L²):")
    passed = verify_parseval()
    print(f"   Passed: {passed}")
    
    # Test trajectory norm
    print("\n2. Trajectory norm test:")
    u = torch.randn(5, 20, 64)  # 5 trajectories, 20 time steps, 64 spatial points
    norm = trajectory_norm(u, s=1.0, dt=0.05, L=1.0)
    print(f"   u shape: {u.shape}")
    print(f"   ||u||_traj: {norm}")
    
    # Test inner product
    print("\n3. Inner product test (should equal norm² for u=v):")
    inner = trajectory_inner_product(u, u, s=1.0, dt=0.05, L=1.0)
    norm_sq = trajectory_norm_squared(u, s=1.0, dt=0.05, L=1.0)
    print(f"   ⟨u,u⟩: {inner[:3]}")
    print(f"   ||u||²: {norm_sq[:3]}")
    print(f"   Close: {torch.allclose(inner, norm_sq)}")
    
    print("\nAll tests completed!")
