"""
Gaussian Trajectory Sampler for Optimal Functional Flow Matching.

Samples from Gaussian measures on trajectory Hilbert spaces L²(0,T; H^s(D)).
Extends/wraps GPPrior from util/gaussian_process.py for trajectory sampling.

References:
    - OFFM formulation in optimal_ffm/offm.md
    - GPPrior in util/gaussian_process.py
"""

import torch
import numpy as np
from typing import Optional, Tuple, Literal

from util.gaussian_process import GPPrior
from util.util import make_grid


class GaussianTrajectorySampler:
    """Sample from Gaussian measure on trajectory Hilbert space L²(0,T; H^s(D)).
    
    Provides several modes for trajectory sampling:
    
    1. 'independent': Sample each time slice independently using GPPrior
       - Simple, but no temporal correlation
       - Good baseline
       
    2. 'spectral': Use spectral decay (1 + |k_t|² + |k_x|²)^{-α} in 2D FFT
       - Smooth in both space and time
       - Physically motivated for PDE trajectories
       
    3. 'separable': C = C_t ⊗ C_x (tensor product covariance)
       - Use GPPrior for spatial covariance
       - Separate temporal correlation kernel
    """
    
    def __init__(
        self,
        n_t: int,
        n_x: int,
        mode: Literal["independent", "spectral", "separable"] = "independent",
        # Spatial GP params (same as GPPrior for compatibility)
        kernel_length: float = 0.01,
        kernel_variance: float = 0.1,
        # Spectral mode params
        smoothness_alpha: float = 2.0,
        # Separable mode params
        time_kernel_length: float = 0.1,
        # Domain params
        L_x: float = 1.0,
        T: float = 1.0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize Gaussian trajectory sampler.
        
        Args:
            n_t: Number of time steps
            n_x: Number of spatial grid points
            mode: Sampling mode ('independent', 'spectral', 'separable')
            kernel_length: Spatial GP lengthscale (for independent/separable)
            kernel_variance: Overall variance scale
            smoothness_alpha: Spectral decay exponent (for spectral mode)
            time_kernel_length: Temporal lengthscale (for separable mode)
            L_x: Spatial domain length
            T: Time domain length
            device: Torch device
            dtype: Torch dtype
        """
        self.n_t = n_t
        self.n_x = n_x
        self.mode = mode
        self.kernel_length = kernel_length
        self.kernel_variance = kernel_variance
        self.smoothness_alpha = smoothness_alpha
        self.time_kernel_length = time_kernel_length
        self.L_x = L_x
        self.T = T
        self.device = device
        self.dtype = dtype
        
        # Initialize based on mode
        if mode == "independent":
            self._init_independent()
        elif mode == "spectral":
            self._init_spectral()
        elif mode == "separable":
            self._init_separable()
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'independent', 'spectral', 'separable'.")
    
    def _init_independent(self):
        """Initialize for independent sampling using GPPrior."""
        self.gp = GPPrior(
            lengthscale=self.kernel_length,
            var=self.kernel_variance,
            device=self.device,
        )
        self.spatial_grid = make_grid([self.n_x]).to(self.device)
    
    def _init_spectral(self):
        """Initialize spectral covariance weights."""
        # 2D spectral weights for (time, space) FFT
        # Frequencies in time: 0, 1, ..., n_t//2 (for rfft)
        # Frequencies in space: 0, 1, ..., n_x//2 (for rfft)
        
        dt = self.T / self.n_t
        dx = self.L_x / self.n_x
        
        # Time frequencies
        freq_t = torch.fft.rfftfreq(self.n_t, d=dt, device=self.device)  # (n_t//2 + 1,)
        k_t = 2 * np.pi * freq_t  # Convert to angular frequency
        
        # Space frequencies  
        freq_x = torch.fft.rfftfreq(self.n_x, d=dx, device=self.device)  # (n_x//2 + 1,)
        k_x = 2 * np.pi * freq_x
        
        # 2D grid of (|k_t|², |k_x|²)
        k_t_sq = k_t.unsqueeze(1) ** 2  # (n_t//2+1, 1)
        k_x_sq = k_x.unsqueeze(0) ** 2  # (1, n_x//2+1)
        
        # Spectral weights: (1 + |k_t|² + |k_x|²)^{-α/2} for std, ^{-α} for variance
        # We sample by: û ~ N(0, 1) * sqrt(spectral_variance)
        spectral_variance = (1.0 + k_t_sq + k_x_sq) ** (-self.smoothness_alpha)
        spectral_std = torch.sqrt(spectral_variance * self.kernel_variance)
        
        self.spectral_std = spectral_std.to(self.dtype)  # (n_t//2+1, n_x//2+1)
    
    def _init_separable(self):
        """Initialize separable covariance C = C_t ⊗ C_x."""
        # Spatial covariance via GPPrior
        self.gp = GPPrior(
            lengthscale=self.kernel_length,
            var=1.0,  # Scale by kernel_variance at the end
            device=self.device,
        )
        self.spatial_grid = make_grid([self.n_x]).to(self.device)
        
        # Temporal covariance: use Matern or RBF correlation
        # Build correlation matrix directly
        time_grid = torch.linspace(0, self.T, self.n_t, device=self.device, dtype=self.dtype)
        time_dists = torch.cdist(time_grid.unsqueeze(1), time_grid.unsqueeze(1)).squeeze()
        
        # RBF correlation: exp(-d² / (2 * l²))
        time_corr = torch.exp(-time_dists**2 / (2 * self.time_kernel_length**2))
        
        # Add small diagonal for numerical stability
        time_corr = time_corr + 1e-6 * torch.eye(self.n_t, device=self.device, dtype=self.dtype)
        
        # Cholesky for sampling
        self.time_chol = torch.linalg.cholesky(time_corr)
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample trajectories from Gaussian prior.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            samples: (batch_size, n_t, n_x) tensor of trajectories
        """
        if self.mode == "independent":
            return self._sample_independent(batch_size)
        elif self.mode == "spectral":
            return self._sample_spectral(batch_size)
        elif self.mode == "separable":
            return self._sample_separable(batch_size)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _sample_independent(self, batch_size: int) -> torch.Tensor:
        """Sample with independent GP for each time slice."""
        # Use GPPrior.sample() - it returns (n_samples, n_channels, *dims)
        # We want (batch_size, n_t, n_x)
        # Trick: treat n_t as channels
        samples = self.gp.sample(
            self.spatial_grid, 
            dims=[self.n_x],
            n_samples=batch_size,
            n_channels=self.n_t,
        )
        # samples shape: (batch_size, n_t, n_x) - already correct!
        return samples.to(self.dtype)
    
    def _sample_spectral(self, batch_size: int) -> torch.Tensor:
        """Sample using 2D spectral covariance."""
        # Sample white noise in Fourier domain
        # For rfft2, output shape is (n_t, n_x//2 + 1) complex
        n_freq_t = self.n_t // 2 + 1
        n_freq_x = self.n_x // 2 + 1
        
        # Complex Gaussian noise
        noise_real = torch.randn(batch_size, n_freq_t, n_freq_x, device=self.device, dtype=self.dtype)
        noise_imag = torch.randn(batch_size, n_freq_t, n_freq_x, device=self.device, dtype=self.dtype)
        noise = torch.complex(noise_real, noise_imag) / np.sqrt(2)
        
        # Scale by spectral std
        scaled_noise = noise * self.spectral_std.unsqueeze(0)
        
        # Inverse 2D FFT
        # irfft2 expects input of shape (..., n_freq_t, n_freq_x) and produces (..., n_t, n_x)
        samples = torch.fft.irfft2(scaled_noise, s=(self.n_t, self.n_x))
        
        # Normalize by grid size for proper scaling
        samples = samples * np.sqrt(self.n_t * self.n_x)
        
        return samples
    
    def _sample_separable(self, batch_size: int) -> torch.Tensor:
        """Sample with separable covariance C = C_t ⊗ C_x."""
        # First, sample spatial functions: (batch_size, n_x)
        # Then, correlate in time using time_chol
        
        # Sample n_t independent spatial functions per trajectory
        spatial_samples = self.gp.sample(
            self.spatial_grid,
            dims=[self.n_x],
            n_samples=batch_size * self.n_t,
            n_channels=1,
        ).squeeze(1)  # (batch_size * n_t, n_x)
        
        # Reshape to (batch_size, n_t, n_x)
        spatial_samples = spatial_samples.reshape(batch_size, self.n_t, self.n_x)
        
        # Apply temporal correlation via Cholesky
        # For each spatial point, correlate the time series
        # samples[b, :, x] = time_chol @ spatial_samples[b, :, x]
        # This is: (n_t, n_t) @ (batch_size, n_t, n_x) along dim 1
        
        # Transpose for matmul: (batch_size, n_x, n_t)
        spatial_T = spatial_samples.transpose(1, 2)
        # Apply chol: result is (batch_size, n_x, n_t)
        correlated_T = torch.matmul(spatial_T, self.time_chol.T)
        # Transpose back: (batch_size, n_t, n_x)
        samples = correlated_T.transpose(1, 2)
        
        # Scale by variance
        samples = samples * np.sqrt(self.kernel_variance)
        
        return samples.to(self.dtype)
    
    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """Compute log probability (up to normalization constant).
        
        For spectral mode only (others require full covariance matrix).
        
        Args:
            u: (batch_size, n_t, n_x) tensor
            
        Returns:
            log_p: (batch_size,) tensor of log probabilities (up to constant)
        """
        if self.mode != "spectral":
            raise NotImplementedError("log_prob only implemented for spectral mode")
        
        # FFT and compute weighted norm in Fourier domain
        u_hat = torch.fft.rfft2(u, dim=(-2, -1))  # (batch, n_freq_t, n_freq_x)
        
        # Precision (inverse variance)
        precision = 1.0 / (self.spectral_std ** 2 + 1e-10)
        
        # Weighted squared magnitude
        weighted_sq = precision.unsqueeze(0) * torch.abs(u_hat) ** 2
        
        # Sum (negative quadratic form)
        log_p = -0.5 * weighted_sq.sum(dim=(-2, -1))
        
        return log_p


def test_gaussian_trajectory_sampler():
    """Test the Gaussian trajectory sampler."""
    print("Testing GaussianTrajectorySampler...")
    
    n_t, n_x = 20, 64
    batch_size = 10
    
    for mode in ["independent", "spectral", "separable"]:
        print(f"\n  Mode: {mode}")
        sampler = GaussianTrajectorySampler(
            n_t=n_t, n_x=n_x, mode=mode,
            kernel_length=0.05, kernel_variance=0.5,
            smoothness_alpha=2.0, time_kernel_length=0.2,
            device='cpu',
        )
        
        samples = sampler.sample(batch_size)
        print(f"    Shape: {samples.shape}")
        print(f"    Mean: {samples.mean():.4f}")
        print(f"    Std:  {samples.std():.4f}")
        print(f"    Min:  {samples.min():.4f}")
        print(f"    Max:  {samples.max():.4f}")


if __name__ == "__main__":
    test_gaussian_trajectory_sampler()
