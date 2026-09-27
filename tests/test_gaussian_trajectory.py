"""
Unit tests for util/gaussian_trajectory.py - Gaussian trajectory sampler.

Tests:
1. Output shape correctness
2. Zero mean (approximately)
3. Variance scaling
4. Different sampling modes
5. Smoothness properties
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from util.gaussian_trajectory import GaussianTrajectorySampler


class TestSamplerBasics:
    """Basic tests for all sampling modes."""
    
    @pytest.fixture(params=["independent", "spectral", "separable"])
    def sampler(self, request):
        """Fixture providing sampler for each mode."""
        return GaussianTrajectorySampler(
            n_t=20, n_x=64,
            mode=request.param,
            kernel_length=0.05,
            kernel_variance=1.0,
            smoothness_alpha=2.0,
            time_kernel_length=0.2,
            device='cpu',
        )
    
    def test_output_shape(self, sampler):
        """Samples should have correct shape (batch, n_t, n_x)."""
        samples = sampler.sample(batch_size=10)
        assert samples.shape == (10, sampler.n_t, sampler.n_x)
    
    def test_batch_size_one(self, sampler):
        """Should work with batch_size=1."""
        samples = sampler.sample(batch_size=1)
        assert samples.shape == (1, sampler.n_t, sampler.n_x)
    
    def test_different_batch_sizes(self, sampler):
        """Should work with various batch sizes."""
        for batch_size in [1, 5, 10, 32]:
            samples = sampler.sample(batch_size=batch_size)
            assert samples.shape[0] == batch_size
    
    def test_samples_finite(self, sampler):
        """All samples should be finite (no NaN or Inf)."""
        samples = sampler.sample(batch_size=20)
        assert torch.isfinite(samples).all()


class TestIndependentMode:
    """Tests specific to independent sampling mode."""
    
    def test_zero_mean_approx(self):
        """Mean should be approximately zero (GP with zero mean)."""
        sampler = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="independent",
            kernel_variance=1.0,
        )
        samples = sampler.sample(batch_size=500)
        
        # Mean over all samples
        mean = samples.mean()
        assert torch.abs(mean) < 0.1  # Should be close to 0
    
    def test_variance_scaling(self):
        """Variance should scale with kernel_variance."""
        sampler_v1 = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="independent",
            kernel_variance=1.0,
        )
        sampler_v4 = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="independent",
            kernel_variance=4.0,
        )
        
        samples_v1 = sampler_v1.sample(batch_size=200)
        samples_v4 = sampler_v4.sample(batch_size=200)
        
        var_v1 = samples_v1.var()
        var_v4 = samples_v4.var()
        
        # Variance should scale approximately linearly
        ratio = var_v4 / var_v1
        assert 2.0 < ratio < 6.0  # Should be around 4
    
    def test_time_slices_independent(self):
        """Different time slices should be (approximately) independent."""
        sampler = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="independent",
        )
        samples = sampler.sample(batch_size=500)
        
        # Correlation between time slice 0 and time slice 10
        # Should be low for independent mode
        t0 = samples[:, 0, :].flatten()
        t10 = samples[:, 10, :].flatten()
        
        # Compute correlation coefficient
        corr = torch.corrcoef(torch.stack([t0, t10]))[0, 1]
        
        # Should be close to 0 for independent sampling
        assert torch.abs(corr) < 0.2


class TestSpectralMode:
    """Tests specific to spectral sampling mode."""
    
    def test_spectral_smoothness(self):
        """Higher smoothness_alpha should give smoother samples."""
        sampler_rough = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="spectral",
            smoothness_alpha=1.0,
        )
        sampler_smooth = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="spectral",
            smoothness_alpha=3.0,
        )
        
        samples_rough = sampler_rough.sample(batch_size=50)
        samples_smooth = sampler_smooth.sample(batch_size=50)
        
        # Measure "roughness" via finite differences
        diff_rough = torch.abs(samples_rough[:, :, 1:] - samples_rough[:, :, :-1]).mean()
        diff_smooth = torch.abs(samples_smooth[:, :, 1:] - samples_smooth[:, :, :-1]).mean()
        
        # Smoother samples should have smaller differences
        assert diff_smooth < diff_rough
    
    def test_spectral_decay(self):
        """High frequencies should have less power for spectral mode."""
        sampler = GaussianTrajectorySampler(
            n_t=20, n_x=128, mode="spectral",
            smoothness_alpha=2.0,
        )
        samples = sampler.sample(batch_size=100)
        
        # Compute average power spectrum
        fft = torch.fft.rfft(samples, dim=-1)
        power = (torch.abs(fft) ** 2).mean(dim=(0, 1))
        
        # Low frequencies should have more power than high frequencies
        low_freq_power = power[:10].mean()
        high_freq_power = power[-10:].mean()
        
        assert low_freq_power > high_freq_power


class TestSeparableMode:
    """Tests specific to separable sampling mode."""
    
    def test_temporal_correlation(self):
        """Nearby time slices should be correlated in separable mode."""
        sampler = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="separable",
            time_kernel_length=0.3,  # Longer correlation
        )
        samples = sampler.sample(batch_size=500)
        
        # Correlation between adjacent time slices
        t0 = samples[:, 0, :].flatten()
        t1 = samples[:, 1, :].flatten()
        t10 = samples[:, 10, :].flatten()
        
        # Adjacent should be more correlated than distant
        corr_adjacent = torch.corrcoef(torch.stack([t0, t1]))[0, 1]
        corr_distant = torch.corrcoef(torch.stack([t0, t10]))[0, 1]
        
        assert corr_adjacent > corr_distant
    
    def test_time_kernel_length_effect(self):
        """Longer time kernel should give more temporal correlation."""
        sampler_short = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="separable",
            time_kernel_length=0.05,
        )
        sampler_long = GaussianTrajectorySampler(
            n_t=20, n_x=64, mode="separable",
            time_kernel_length=0.5,
        )
        
        samples_short = sampler_short.sample(batch_size=300)
        samples_long = sampler_long.sample(batch_size=300)
        
        # Compute temporal autocorrelation at lag 5
        def temporal_autocorr(samples, lag=5):
            t0 = samples[:, :-lag, :].flatten()
            t_lag = samples[:, lag:, :].flatten()
            return torch.corrcoef(torch.stack([t0, t_lag]))[0, 1]
        
        corr_short = temporal_autocorr(samples_short)
        corr_long = temporal_autocorr(samples_long)
        
        # Longer kernel should give higher correlation
        assert corr_long > corr_short


class TestDeviceAndDtype:
    """Tests for device and dtype handling."""
    
    def test_cpu_device(self):
        """Should work on CPU."""
        sampler = GaussianTrajectorySampler(
            n_t=10, n_x=32, mode="independent", device='cpu'
        )
        samples = sampler.sample(batch_size=5)
        assert samples.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Should work on CUDA if available."""
        sampler = GaussianTrajectorySampler(
            n_t=10, n_x=32, mode="independent", device='cuda'
        )
        samples = sampler.sample(batch_size=5)
        assert samples.device.type == 'cuda'
    
    def test_float32_dtype(self):
        """Samples should be float32 by default."""
        sampler = GaussianTrajectorySampler(
            n_t=10, n_x=32, mode="independent",
            dtype=torch.float32,
        )
        samples = sampler.sample(batch_size=5)
        assert samples.dtype == torch.float32


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_small_grid(self):
        """Should work with small grids."""
        sampler = GaussianTrajectorySampler(
            n_t=5, n_x=8, mode="spectral"
        )
        samples = sampler.sample(batch_size=3)
        assert samples.shape == (3, 5, 8)
        assert torch.isfinite(samples).all()
    
    def test_single_time_step(self):
        """Should work with n_t=1."""
        sampler = GaussianTrajectorySampler(
            n_t=1, n_x=32, mode="independent"
        )
        samples = sampler.sample(batch_size=5)
        assert samples.shape == (5, 1, 32)
    
    def test_large_batch(self):
        """Should handle large batch sizes."""
        sampler = GaussianTrajectorySampler(
            n_t=10, n_x=32, mode="spectral"
        )
        samples = sampler.sample(batch_size=500)
        assert samples.shape == (500, 10, 32)
        assert torch.isfinite(samples).all()


class TestReproducibility:
    """Tests for reproducibility with random seeds."""
    
    def test_different_samples(self):
        """Different calls should give different samples."""
        sampler = GaussianTrajectorySampler(
            n_t=10, n_x=32, mode="spectral"
        )
        
        samples1 = sampler.sample(batch_size=5)
        samples2 = sampler.sample(batch_size=5)
        
        # Should not be identical
        assert not torch.allclose(samples1, samples2)
    
    def test_seed_reproducibility(self):
        """Same seed should give same samples."""
        sampler = GaussianTrajectorySampler(
            n_t=10, n_x=32, mode="spectral"
        )
        
        torch.manual_seed(42)
        samples1 = sampler.sample(batch_size=5)
        
        torch.manual_seed(42)
        samples2 = sampler.sample(batch_size=5)
        
        # Note: This may not always work perfectly due to GPU non-determinism
        # but should work on CPU
        assert torch.allclose(samples1, samples2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
