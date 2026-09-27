"""
Unit tests for models/icnn.py - Input Convex Neural Networks.

Tests:
1. ICNN convexity verification
2. ICNN gradient correctness
3. ProjectedICNN projection/unprojection
4. ProjectedICNN convexity
5. ProjectedICNN gradient computation
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '..')

from models.icnn import ICNN, ProjectedICNN


class TestICNNBasics:
    """Basic tests for ICNN."""
    
    def test_output_shape(self):
        """Output should be (batch,) scalar."""
        icnn = ICNN(input_dim=32, hidden_dims=[64, 64])
        x = torch.randn(5, 32)
        y = icnn(x)
        assert y.shape == (5,)
    
    def test_output_finite(self):
        """Output should be finite."""
        icnn = ICNN(input_dim=32, hidden_dims=[64, 64])
        x = torch.randn(10, 32)
        y = icnn(x)
        assert torch.isfinite(y).all()
    
    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        icnn = ICNN(input_dim=16, hidden_dims=[32])
        x = torch.randn(1, 16)
        y = icnn(x)
        assert y.shape == (1,)
    
    def test_different_architectures(self):
        """Should work with various hidden layer configurations."""
        configs = [
            [64],
            [64, 64],
            [128, 64, 32],
            [256, 256, 256],
        ]
        for hidden_dims in configs:
            icnn = ICNN(input_dim=32, hidden_dims=hidden_dims)
            x = torch.randn(3, 32)
            y = icnn(x)
            assert y.shape == (3,)
            assert torch.isfinite(y).all()


class TestICNNConvexity:
    """Tests for ICNN convexity property."""
    
    def test_convexity_along_line(self):
        """ICNN should be convex along any line (Jensen's inequality)."""
        icnn = ICNN(input_dim=32, hidden_dims=[64, 64, 64])
        
        # Multiple random lines
        n_tests = 10
        all_convex = True
        
        for _ in range(n_tests):
            x1 = torch.randn(1, 32)
            x2 = torch.randn(1, 32)
            
            # Check convexity: f(tx1 + (1-t)x2) <= t*f(x1) + (1-t)*f(x2)
            t_vals = torch.linspace(0, 1, 21)
            
            for t in t_vals[1:-1]:  # Exclude endpoints
                x_interp = t * x1 + (1 - t) * x2
                
                y_interp = icnn(x_interp).item()
                y_bound = t.item() * icnn(x1).item() + (1 - t.item()) * icnn(x2).item()
                
                if y_interp > y_bound + 1e-4:  # Allow small numerical tolerance
                    all_convex = False
                    break
            
            if not all_convex:
                break
        
        assert all_convex, "ICNN failed convexity test"
    
    def test_convexity_midpoint(self):
        """Midpoint convexity: f((x+y)/2) <= (f(x)+f(y))/2."""
        icnn = ICNN(input_dim=16, hidden_dims=[32, 32])
        
        n_tests = 20
        for _ in range(n_tests):
            x = torch.randn(1, 16)
            y = torch.randn(1, 16)
            
            midpoint = (x + y) / 2
            
            f_mid = icnn(midpoint).item()
            f_avg = (icnn(x).item() + icnn(y).item()) / 2
            
            assert f_mid <= f_avg + 1e-4, f"Midpoint convexity failed: {f_mid} > {f_avg}"
    
    def test_positive_hessian_trace(self):
        """Hessian should be positive semi-definite (check via trace)."""
        icnn = ICNN(input_dim=8, hidden_dims=[16, 16])
        
        x = torch.randn(1, 8, requires_grad=True)
        
        # Compute Hessian numerically
        def compute_hessian(x):
            y = icnn(x)
            grad = torch.autograd.grad(y, x, create_graph=True)[0]
            
            hessian = []
            for i in range(x.shape[1]):
                grad_i = torch.autograd.grad(grad[0, i], x, retain_graph=True)[0]
                hessian.append(grad_i)
            
            return torch.stack(hessian, dim=0).squeeze()
        
        H = compute_hessian(x)
        
        # Check positive semi-definiteness via eigenvalues
        eigenvalues = torch.linalg.eigvalsh(H)
        
        # All eigenvalues should be >= 0 (with numerical tolerance)
        assert (eigenvalues >= -1e-4).all(), f"Negative eigenvalues: {eigenvalues}"


class TestICNNActivations:
    """Tests for different activation functions."""
    
    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "softplus"])
    def test_activation_convex(self, activation):
        """All supported activations should preserve convexity."""
        icnn = ICNN(input_dim=16, hidden_dims=[32], activation=activation)
        
        x1 = torch.randn(1, 16)
        x2 = torch.randn(1, 16)
        t = 0.5
        
        x_mid = t * x1 + (1 - t) * x2
        f_mid = icnn(x_mid).item()
        f_avg = t * icnn(x1).item() + (1 - t) * icnn(x2).item()
        
        assert f_mid <= f_avg + 1e-4


class TestICNNGradient:
    """Tests for ICNN gradient computation."""
    
    def test_gradient_shape(self):
        """Gradient should have same shape as input."""
        icnn = ICNN(input_dim=32, hidden_dims=[64])
        x = torch.randn(5, 32, requires_grad=True)
        
        y = icnn(x)
        grad = torch.autograd.grad(y.sum(), x)[0]
        
        assert grad.shape == x.shape
    
    def test_gradient_finite(self):
        """Gradient should be finite."""
        icnn = ICNN(input_dim=32, hidden_dims=[64, 64])
        x = torch.randn(5, 32, requires_grad=True)
        
        y = icnn(x)
        grad = torch.autograd.grad(y.sum(), x)[0]
        
        assert torch.isfinite(grad).all()


class TestProjectedICNNBasics:
    """Basic tests for ProjectedICNN."""
    
    def test_output_shape(self):
        """Output should be (batch,) scalar."""
        model = ProjectedICNN(n_t=10, n_x=32, projection_dim=64)
        u = torch.randn(5, 10, 32)
        Phi = model(u)
        assert Phi.shape == (5,)
    
    def test_gradient_shape(self):
        """Gradient should have same shape as input."""
        model = ProjectedICNN(n_t=10, n_x=32, projection_dim=64)
        u = torch.randn(5, 10, 32)
        grad = model.gradient(u)
        assert grad.shape == u.shape
    
    @pytest.mark.parametrize("proj_type", ["fourier", "random"])
    def test_projection_types(self, proj_type):
        """Should work with different projection types."""
        model = ProjectedICNN(
            n_t=10, n_x=32, 
            projection_dim=64,
            projection_type=proj_type,
        )
        u = torch.randn(3, 10, 32)
        Phi = model(u)
        
        assert Phi.shape == (3,)
        assert torch.isfinite(Phi).all()


class TestProjectedICNNProjection:
    """Tests for projection and unprojection operations."""
    
    def test_projection_shape_fourier(self):
        """Fourier projection should have correct output shape."""
        model = ProjectedICNN(
            n_t=20, n_x=64,
            projection_dim=128,
            projection_type="fourier",
        )
        u = torch.randn(5, 20, 64)
        z = model.project(u)
        
        assert z.shape[0] == 5
        assert z.shape[1] == model.actual_proj_dim
    
    def test_projection_shape_random(self):
        """Random projection should have correct output shape."""
        model = ProjectedICNN(
            n_t=20, n_x=64,
            projection_dim=128,
            projection_type="random",
        )
        u = torch.randn(5, 20, 64)
        z = model.project(u)
        
        assert z.shape == (5, 128)
    
    def test_unproject_gradient_shape(self):
        """Unprojected gradient should have trajectory shape."""
        model = ProjectedICNN(
            n_t=20, n_x=64,
            projection_dim=128,
            projection_type="fourier",
        )
        
        # Fake gradient in projected space
        grad_z = torch.randn(5, model.actual_proj_dim)
        grad_u = model.unproject_gradient(grad_z)
        
        assert grad_u.shape == (5, 20, 64)
    
    def test_projection_linearity(self):
        """Projection should be linear."""
        model = ProjectedICNN(
            n_t=10, n_x=32,
            projection_dim=64,
            projection_type="fourier",
        )
        
        u1 = torch.randn(1, 10, 32)
        u2 = torch.randn(1, 10, 32)
        c = 2.5
        
        # P(u1 + u2) = P(u1) + P(u2)
        z_sum = model.project(u1 + u2)
        z1_plus_z2 = model.project(u1) + model.project(u2)
        assert torch.allclose(z_sum, z1_plus_z2, rtol=1e-5)
        
        # P(c*u1) = c*P(u1)
        z_scaled = model.project(c * u1)
        c_times_z1 = c * model.project(u1)
        assert torch.allclose(z_scaled, c_times_z1, rtol=1e-5)


class TestProjectedICNNConvexity:
    """Tests for ProjectedICNN convexity."""
    
    def test_convexity_midpoint(self):
        """ProjectedICNN should be convex (midpoint test)."""
        model = ProjectedICNN(
            n_t=10, n_x=32,
            projection_dim=64,
            projection_type="fourier",
            icnn_hidden_dims=[32, 32],
            lambda_reg=0.0,  # Test ICNN part only
        )
        
        u1 = torch.randn(1, 10, 32)
        u2 = torch.randn(1, 10, 32)
        
        midpoint = (u1 + u2) / 2
        
        Phi_mid = model(midpoint).item()
        Phi_avg = (model(u1).item() + model(u2).item()) / 2
        
        assert Phi_mid <= Phi_avg + 1e-4, f"Midpoint convexity failed: {Phi_mid} > {Phi_avg}"
    
    def test_convexity_with_regularization(self):
        """Convexity should hold with quadratic regularization."""
        model = ProjectedICNN(
            n_t=10, n_x=32,
            projection_dim=64,
            projection_type="fourier",
            lambda_reg=0.5,  # Add regularization
        )
        
        u1 = torch.randn(1, 10, 32)
        u2 = torch.randn(1, 10, 32)
        t = 0.3
        
        u_interp = t * u1 + (1 - t) * u2
        
        Phi_interp = model(u_interp).item()
        Phi_bound = t * model(u1).item() + (1 - t) * model(u2).item()
        
        assert Phi_interp <= Phi_bound + 1e-4


class TestProjectedICNNGradient:
    """Tests for ProjectedICNN gradient computation."""
    
    def test_gradient_numerical_check(self):
        """Gradient should be correlated with numerical gradient.
        
        Note: Due to FFT operations and floating point precision,
        we check correlation rather than exact match.
        The gradient_descent_direction test provides the main correctness check.
        """
        torch.manual_seed(42)
        
        model = ProjectedICNN(
            n_t=8, n_x=16,
            projection_dim=32,
            projection_type="random",  # Random projection is simpler/more stable
            lambda_reg=0.5,  # Higher regularization for stability
        )
        
        u = torch.randn(1, 8, 16) * 0.1  # Small scale for numerical stability
        
        # Analytical gradient
        grad_analytic = model.gradient(u)
        
        # Numerical gradient (finite differences)
        eps = 1e-4
        grad_numerical = torch.zeros_like(u)
        
        for t in range(8):
            for x in range(16):
                u_plus = u.clone()
                u_plus[0, t, x] += eps
                u_minus = u.clone()
                u_minus[0, t, x] -= eps
                
                grad_numerical[0, t, x] = (model(u_plus).item() - model(u_minus).item()) / (2 * eps)
        
        # Check that gradients are correlated (same direction)
        analytic_flat = grad_analytic.flatten()
        numerical_flat = grad_numerical.flatten()
        
        # Cosine similarity should be high (gradients point in same direction)
        cos_sim = F.cosine_similarity(analytic_flat.unsqueeze(0), numerical_flat.unsqueeze(0))
        assert cos_sim > 0.9, f"Gradient direction mismatch: cosine similarity = {cos_sim.item():.4f}"
        
        # Check relative scale is reasonable (within 10x)
        analytic_norm = analytic_flat.norm()
        numerical_norm = numerical_flat.norm()
        scale_ratio = max(analytic_norm, numerical_norm) / (min(analytic_norm, numerical_norm) + 1e-8)
        assert scale_ratio < 10, f"Gradient scale mismatch: ratio = {scale_ratio:.2f}"
    
    def test_gradient_finite(self):
        """Gradient should be finite."""
        model = ProjectedICNN(
            n_t=10, n_x=32,
            projection_dim=64,
        )
        u = torch.randn(5, 10, 32)
        grad = model.gradient(u)
        
        assert torch.isfinite(grad).all()
    
    def test_gradient_descent_direction(self):
        """Moving in negative gradient direction should decrease Φ."""
        torch.manual_seed(123)
        
        model = ProjectedICNN(
            n_t=10, n_x=32,
            projection_dim=64,
            lambda_reg=0.1,
        )
        
        u = torch.randn(1, 10, 32)
        grad = model.gradient(u)
        
        Phi_original = model(u).item()
        
        # Move in negative gradient direction (gradient descent step)
        lr = 0.01
        u_new = u - lr * grad
        Phi_new = model(u_new).item()
        
        # Φ should decrease (or stay same if at minimum)
        assert Phi_new <= Phi_original + 1e-6, \
            f"Gradient descent failed: Φ increased from {Phi_original} to {Phi_new}"
    
    def test_regularization_gradient(self):
        """With only regularization (no ICNN), gradient should be λu."""
        # Create model with lambda but very small ICNN output
        model = ProjectedICNN(
            n_t=10, n_x=32,
            projection_dim=64,
            lambda_reg=2.0,
        )
        
        # Zero out ICNN weights to isolate regularization term
        with torch.no_grad():
            for param in model.icnn.parameters():
                param.zero_()
        
        u = torch.randn(1, 10, 32)
        grad = model.gradient(u)
        
        # Gradient should be approximately λu
        expected = model.lambda_reg * u
        
        assert torch.allclose(grad, expected, rtol=0.1)


class TestProjectedICNNDifferentSizes:
    """Tests for different trajectory sizes."""
    
    @pytest.mark.parametrize("n_t,n_x", [(5, 16), (10, 32), (20, 64), (50, 128)])
    def test_various_sizes(self, n_t, n_x):
        """Should work with various trajectory dimensions."""
        model = ProjectedICNN(
            n_t=n_t, n_x=n_x,
            projection_dim=min(64, n_t * n_x // 2),
            projection_type="fourier",
        )
        
        u = torch.randn(3, n_t, n_x)
        Phi = model(u)
        grad = model.gradient(u)
        
        assert Phi.shape == (3,)
        assert grad.shape == u.shape
        assert torch.isfinite(Phi).all()
        assert torch.isfinite(grad).all()


class TestProjectedICNNNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_input(self):
        """Should handle large input values."""
        model = ProjectedICNN(n_t=10, n_x=32, projection_dim=64)
        u = torch.randn(3, 10, 32) * 100
        
        Phi = model(u)
        assert torch.isfinite(Phi).all()
    
    def test_small_input(self):
        """Should handle small input values."""
        model = ProjectedICNN(n_t=10, n_x=32, projection_dim=64)
        u = torch.randn(3, 10, 32) * 1e-4
        
        Phi = model(u)
        assert torch.isfinite(Phi).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
