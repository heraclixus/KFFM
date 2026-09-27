"""
Unit tests for util/hilbert.py - Sobolev norms and inner products.

Tests:
1. Parseval's theorem (s=0 should equal L² norm)
2. Inner product equals norm² when u=v
3. Inner product symmetry
4. Trajectory norm integration
5. Sobolev inverse is inverse of forward operator
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from util.hilbert import (
    get_sobolev_weights_1d,
    sobolev_norm_squared_1d,
    sobolev_norm_1d,
    sobolev_inner_product_1d,
    trajectory_norm_squared,
    trajectory_norm,
    trajectory_inner_product,
    trajectory_norm_squared_with_time_reg,
    apply_sobolev_inverse,
    trajectory_sobolev_inverse,
)


class TestSobolevWeights:
    """Tests for Sobolev weight computation."""
    
    def test_weights_shape(self):
        """Weights should have correct shape for rfft."""
        n_x = 64
        weights = get_sobolev_weights_1d(n_x, s=1.0)
        assert weights.shape == (n_x // 2 + 1,)
    
    def test_weights_positive(self):
        """All weights should be positive."""
        weights = get_sobolev_weights_1d(128, s=2.0)
        assert (weights > 0).all()
    
    def test_weights_s0_is_one(self):
        """For s=0, weights should be all 1."""
        weights = get_sobolev_weights_1d(64, s=0.0)
        assert torch.allclose(weights, torch.ones_like(weights))
    
    def test_weights_monotonic_for_positive_s(self):
        """For s>0, weights should increase with frequency."""
        weights = get_sobolev_weights_1d(64, s=1.0)
        # Weights should be non-decreasing (increasing for s>0)
        assert (weights[1:] >= weights[:-1]).all()


class TestSobolevNorm1D:
    """Tests for 1D Sobolev norm."""
    
    def test_parseval_s0(self):
        """s=0 should give L² norm (Parseval's theorem)."""
        n_x = 128
        L = 1.0
        dx = L / n_x
        
        u = torch.randn(10, n_x)
        
        # Direct L² norm
        l2_norm_sq = dx * (u ** 2).sum(dim=-1)
        
        # FFT-based with s=0
        fft_norm_sq = sobolev_norm_squared_1d(u, s=0.0, L=L)
        
        assert torch.allclose(l2_norm_sq, fft_norm_sq, rtol=1e-4, atol=1e-6)
    
    def test_norm_positive(self):
        """Norm should be positive for non-zero input."""
        u = torch.randn(5, 64)
        norm = sobolev_norm_1d(u, s=1.0)
        assert (norm > 0).all()
    
    def test_norm_zero_for_zero_input(self):
        """Norm should be zero for zero input."""
        u = torch.zeros(5, 64)
        norm = sobolev_norm_1d(u, s=1.0)
        assert torch.allclose(norm, torch.zeros(5), atol=1e-10)
    
    def test_norm_scaling(self):
        """Norm should scale linearly with input."""
        u = torch.randn(5, 64)
        c = 3.0
        
        norm_u = sobolev_norm_1d(u, s=1.0)
        norm_cu = sobolev_norm_1d(c * u, s=1.0)
        
        assert torch.allclose(norm_cu, c * norm_u, rtol=1e-5)
    
    def test_higher_s_penalizes_oscillations(self):
        """Higher s should penalize high-frequency content more."""
        n_x = 128
        x = torch.linspace(0, 2 * np.pi, n_x)
        
        # Low frequency signal
        u_low = torch.sin(x).unsqueeze(0)
        # High frequency signal (same amplitude)
        u_high = torch.sin(10 * x).unsqueeze(0)
        
        # For s=0 (L²), norms should be similar
        norm_low_s0 = sobolev_norm_1d(u_low, s=0.0)
        norm_high_s0 = sobolev_norm_1d(u_high, s=0.0)
        assert torch.allclose(norm_low_s0, norm_high_s0, rtol=0.1)
        
        # For s=2 (H²), high frequency should have much larger norm
        norm_low_s2 = sobolev_norm_1d(u_low, s=2.0)
        norm_high_s2 = sobolev_norm_1d(u_high, s=2.0)
        assert norm_high_s2 > 10 * norm_low_s2  # Much larger


class TestSobolevInnerProduct1D:
    """Tests for 1D Sobolev inner product."""
    
    def test_inner_equals_norm_squared(self):
        """⟨u,u⟩ should equal ||u||²."""
        u = torch.randn(5, 64)
        
        inner = sobolev_inner_product_1d(u, u, s=1.0)
        norm_sq = sobolev_norm_squared_1d(u, s=1.0)
        
        assert torch.allclose(inner, norm_sq, rtol=1e-5)
    
    def test_inner_symmetry(self):
        """Inner product should be symmetric: ⟨u,v⟩ = ⟨v,u⟩."""
        u = torch.randn(5, 64)
        v = torch.randn(5, 64)
        
        inner_uv = sobolev_inner_product_1d(u, v, s=1.0)
        inner_vu = sobolev_inner_product_1d(v, u, s=1.0)
        
        assert torch.allclose(inner_uv, inner_vu, rtol=1e-5)
    
    def test_inner_bilinearity(self):
        """Inner product should be bilinear."""
        u = torch.randn(5, 64)
        v = torch.randn(5, 64)
        w = torch.randn(5, 64)
        c = 2.5
        
        # ⟨cu, v⟩ = c⟨u,v⟩
        inner_cuv = sobolev_inner_product_1d(c * u, v, s=1.0)
        c_inner_uv = c * sobolev_inner_product_1d(u, v, s=1.0)
        assert torch.allclose(inner_cuv, c_inner_uv, rtol=1e-5)
        
        # ⟨u+w, v⟩ = ⟨u,v⟩ + ⟨w,v⟩
        inner_uwv = sobolev_inner_product_1d(u + w, v, s=1.0)
        inner_uv_wv = sobolev_inner_product_1d(u, v, s=1.0) + sobolev_inner_product_1d(w, v, s=1.0)
        assert torch.allclose(inner_uwv, inner_uv_wv, rtol=1e-5)
    
    def test_cauchy_schwarz(self):
        """Cauchy-Schwarz: |⟨u,v⟩| ≤ ||u|| ||v||."""
        u = torch.randn(5, 64)
        v = torch.randn(5, 64)
        
        inner = sobolev_inner_product_1d(u, v, s=1.0)
        norm_u = sobolev_norm_1d(u, s=1.0)
        norm_v = sobolev_norm_1d(v, s=1.0)
        
        assert (torch.abs(inner) <= norm_u * norm_v + 1e-6).all()


class TestTrajectoryNorm:
    """Tests for trajectory space norms."""
    
    def test_trajectory_norm_shape(self):
        """Output should have batch dimension only."""
        u = torch.randn(5, 20, 64)  # (batch, time, space)
        norm = trajectory_norm(u, s=1.0, dt=0.05)
        assert norm.shape == (5,)
    
    def test_trajectory_norm_positive(self):
        """Trajectory norm should be positive for non-zero input."""
        u = torch.randn(5, 20, 64)
        norm = trajectory_norm(u, s=1.0, dt=0.05)
        assert (norm > 0).all()
    
    def test_trajectory_inner_equals_norm_squared(self):
        """⟨u,u⟩_traj should equal ||u||²_traj."""
        u = torch.randn(5, 20, 64)
        
        inner = trajectory_inner_product(u, u, s=1.0, dt=0.05)
        norm_sq = trajectory_norm_squared(u, s=1.0, dt=0.05)
        
        assert torch.allclose(inner, norm_sq, rtol=1e-5)
    
    def test_trajectory_inner_symmetry(self):
        """Trajectory inner product should be symmetric."""
        u = torch.randn(5, 20, 64)
        v = torch.randn(5, 20, 64)
        
        inner_uv = trajectory_inner_product(u, v, s=1.0, dt=0.05)
        inner_vu = trajectory_inner_product(v, u, s=1.0, dt=0.05)
        
        assert torch.allclose(inner_uv, inner_vu, rtol=1e-5)
    
    def test_time_integration_correct(self):
        """Verify time integration is correct (dt scaling)."""
        u = torch.randn(5, 20, 64)
        
        # With dt=0.1, norm should be sqrt(2) times larger than dt=0.05
        norm_dt1 = trajectory_norm_squared(u, s=0.0, dt=0.1)
        norm_dt2 = trajectory_norm_squared(u, s=0.0, dt=0.05)
        
        assert torch.allclose(norm_dt1, 2 * norm_dt2, rtol=1e-5)


class TestTrajectoryNormWithTimeReg:
    """Tests for time-regularized trajectory norm."""
    
    def test_beta_zero_equals_standard(self):
        """With β=0, should equal standard trajectory norm."""
        u = torch.randn(5, 20, 64)
        
        norm_reg = trajectory_norm_squared_with_time_reg(u, s=1.0, beta=0.0, dt=0.05)
        norm_std = trajectory_norm_squared(u, s=1.0, dt=0.05)
        
        assert torch.allclose(norm_reg, norm_std, rtol=1e-5)
    
    def test_time_reg_penalizes_jitter(self):
        """Time regularization should penalize temporal variation."""
        n_t, n_x = 20, 64
        dt = 0.05
        
        # Smooth trajectory (truly constant in time)
        base_slice = torch.randn(1, n_x)
        u_smooth = base_slice.unsqueeze(1).expand(1, n_t, n_x).clone().contiguous()
        
        # Jittery trajectory (independent random at each time)
        torch.manual_seed(123)  # Different seed for jittery
        u_jitter = torch.randn(1, n_t, n_x)
        
        # With time reg, the time derivative term should differ significantly
        # For constant trajectory, time derivative is 0
        # For jittery trajectory, time derivative is large
        norm_smooth_reg = trajectory_norm_squared_with_time_reg(u_smooth, s=0.0, beta=1.0, dt=dt)
        norm_jitter_reg = trajectory_norm_squared_with_time_reg(u_jitter, s=0.0, beta=1.0, dt=dt)
        
        # The smooth trajectory's time-reg norm should just be spatial norm (no time deriv contribution)
        # The jittery trajectory should have additional time derivative penalty
        # So norm_jitter_reg should be larger
        assert norm_smooth_reg.item() < norm_jitter_reg.item(), \
            f"Smooth norm {norm_smooth_reg.item():.4f} should be < jittery norm {norm_jitter_reg.item():.4f}"


class TestSobolevInverse:
    """Tests for Sobolev inverse operator."""
    
    def test_inverse_shape(self):
        """Output shape should match input shape."""
        u = torch.randn(5, 64)
        result = apply_sobolev_inverse(u, s=1.0)
        assert result.shape == u.shape
    
    def test_inverse_smooths_signal(self):
        """Inverse Sobolev (s>0) should smooth the signal."""
        n_x = 128
        x = torch.linspace(0, 2 * np.pi, n_x)
        
        # High frequency signal
        u = torch.sin(10 * x).unsqueeze(0)
        
        # Apply inverse (should reduce high frequencies)
        u_smooth = apply_sobolev_inverse(u, s=2.0)
        
        # High frequency content should be reduced
        # Check via FFT
        u_hat = torch.fft.rfft(u, dim=-1)
        u_smooth_hat = torch.fft.rfft(u_smooth, dim=-1)
        
        # Ratio of high freq to low freq should be smaller after inverse
        high_freq_orig = torch.abs(u_hat[0, 10:]).mean()
        high_freq_smooth = torch.abs(u_smooth_hat[0, 10:]).mean()
        
        assert high_freq_smooth < high_freq_orig
    
    def test_trajectory_inverse_shape(self):
        """Trajectory inverse should preserve shape."""
        u = torch.randn(5, 20, 64)
        result = trajectory_sobolev_inverse(u, s=1.0)
        assert result.shape == u.shape


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_values(self):
        """Should handle large input values."""
        u = torch.randn(5, 64) * 1000
        norm = sobolev_norm_1d(u, s=1.0)
        assert torch.isfinite(norm).all()
    
    def test_small_values(self):
        """Should handle small input values."""
        u = torch.randn(5, 64) * 1e-6
        norm = sobolev_norm_1d(u, s=1.0)
        assert torch.isfinite(norm).all()
    
    def test_different_grid_sizes(self):
        """Should work with different grid sizes."""
        for n_x in [32, 64, 128, 256]:
            u = torch.randn(3, n_x)
            norm = sobolev_norm_1d(u, s=1.0)
            assert torch.isfinite(norm).all()
            assert (norm > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
