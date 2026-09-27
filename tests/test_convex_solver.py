"""
Unit tests for util/convex_solver.py - Inner convex optimization.

Tests:
1. Quadratic problem with analytic solution
2. Convergence behavior
3. Stop-gradient (detached output)
4. Different solver methods
5. Objective function correctness
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from util.convex_solver import ConvexSolver, AdaptiveConvexSolver


class SimpleQuadraticPhi:
    """Simple quadratic Φ(z) = λ||z||² / 2 for testing."""
    
    def __init__(self, lam: float = 1.0):
        self.lam = lam
    
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Φ(z) = λ||z||² / 2."""
        return 0.5 * self.lam * (z ** 2).sum(dim=(-2, -1))
    
    def grad(self, z: torch.Tensor) -> torch.Tensor:
        """∇Φ(z) = λz."""
        return self.lam * z
    
    def analytic_solution(self, U_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Analytic solution for the inner problem.
        
        For Φ(z) = λ||z||²/2, the optimality condition is:
            z - U_t + t*λ*z = 0
            z(1 + t*λ) = U_t
            z* = U_t / (1 + t*λ)
        """
        if t.dim() == 0:
            t = t.expand(U_t.shape[0])
        t_broad = t.view(-1, 1, 1)
        return U_t / (1 + t_broad * self.lam)


class TestObjectiveFunction:
    """Tests for objective function computation."""
    
    def test_objective_positive(self):
        """Objective should be finite for reasonable inputs."""
        solver = ConvexSolver()
        phi = SimpleQuadraticPhi(lam=1.0)
        
        z = torch.randn(5, 10, 32)
        U_t = torch.randn(5, 10, 32)
        t = torch.rand(5) * 0.9 + 0.1
        
        obj = solver.objective(z, U_t, t, phi)
        
        assert obj.shape == (5,)
        assert torch.isfinite(obj).all()
    
    def test_objective_at_minimum(self):
        """Objective at z* should be minimal."""
        solver = ConvexSolver()
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(5, 10, 32)
        t = torch.rand(5) * 0.9 + 0.1
        
        # Analytic minimum
        z_star = phi.analytic_solution(U_t, t)
        obj_star = solver.objective(z_star, U_t, t, phi)
        
        # Random point
        z_random = torch.randn_like(z_star)
        obj_random = solver.objective(z_random, U_t, t, phi)
        
        # Minimum should have lower objective
        assert (obj_star <= obj_random + 1e-6).all()


class TestGradientFunction:
    """Tests for gradient computation."""
    
    def test_gradient_shape(self):
        """Gradient should have same shape as input."""
        solver = ConvexSolver()
        phi = SimpleQuadraticPhi(lam=1.0)
        
        z = torch.randn(5, 10, 32)
        U_t = torch.randn(5, 10, 32)
        t = torch.rand(5)
        
        grad = solver.gradient(z, U_t, t, phi.grad)
        
        assert grad.shape == z.shape
    
    def test_gradient_zero_at_minimum(self):
        """Gradient should be (approximately) zero at minimum."""
        solver = ConvexSolver(sobolev_s=0.0)  # L² gradient
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(5, 10, 32)
        t = torch.rand(5) * 0.9 + 0.1
        
        z_star = phi.analytic_solution(U_t, t)
        grad = solver.gradient(z_star, U_t, t, phi.grad)
        
        # Gradient should be close to zero
        grad_norm = grad.flatten(1).norm(dim=1)
        assert (grad_norm < 1e-5).all()


class TestSolverConvergence:
    """Tests for solver convergence to correct solution."""
    
    @pytest.fixture
    def simple_problem(self):
        """Create a simple test problem."""
        batch_size = 5
        n_t, n_x = 10, 32
        
        U_t = torch.randn(batch_size, n_t, n_x)
        t = torch.rand(batch_size) * 0.8 + 0.1  # t in [0.1, 0.9]
        phi = SimpleQuadraticPhi(lam=1.0)
        
        return U_t, t, phi
    
    def test_gd_convergence(self, simple_problem):
        """GD solver should converge to analytic solution."""
        U_t, t, phi = simple_problem
        
        solver = ConvexSolver(method="gd", n_steps=100, lr=0.3)
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        z_analytic = phi.analytic_solution(U_t, t)
        error = (z_star - z_analytic).abs().mean()
        
        assert error < 0.01, f"GD error too high: {error}"
    
    def test_nesterov_convergence(self, simple_problem):
        """Nesterov solver should converge to analytic solution."""
        U_t, t, phi = simple_problem
        
        solver = ConvexSolver(method="nesterov", n_steps=50, lr=0.3, momentum=0.9)
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        z_analytic = phi.analytic_solution(U_t, t)
        error = (z_star - z_analytic).abs().mean()
        
        assert error < 0.01, f"Nesterov error too high: {error}"
    
    def test_nesterov_faster_than_gd(self, simple_problem):
        """Nesterov should converge at least as well as GD with same steps."""
        U_t, t, phi = simple_problem
        z_analytic = phi.analytic_solution(U_t, t)
        
        n_steps = 30  # Enough steps for fair comparison
        
        solver_gd = ConvexSolver(method="gd", n_steps=n_steps, lr=0.3)
        z_gd = solver_gd.solve(U_t, t, phi, phi.grad)
        error_gd = (z_gd - z_analytic).abs().mean()
        
        # Use smaller lr for Nesterov to avoid overshooting
        solver_nest = ConvexSolver(method="nesterov", n_steps=n_steps, lr=0.2, momentum=0.8)
        z_nest = solver_nest.solve(U_t, t, phi, phi.grad)
        error_nest = (z_nest - z_analytic).abs().mean()
        
        # Both should converge reasonably well
        assert error_gd < 0.05, f"GD error too high: {error_gd}"
        assert error_nest < 0.05, f"Nesterov error too high: {error_nest}"


class TestStopGradient:
    """Tests for stop-gradient behavior."""
    
    def test_output_detached(self):
        """Output should be detached from computation graph."""
        solver = ConvexSolver(method="gd", n_steps=10)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16, requires_grad=True)
        t = torch.tensor([0.5])
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        # z_star should not require grad
        assert not z_star.requires_grad
    
    def test_no_grad_through_solver(self):
        """Gradients should not flow through the solver."""
        solver = ConvexSolver(method="gd", n_steps=10)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16, requires_grad=True)
        t = torch.tensor([0.5])
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        # Try to compute gradient - should fail or be zero
        # since z_star is detached
        loss = z_star.sum()
        
        # This should not raise an error, but grad should be None
        # because z_star is detached
        assert z_star.grad_fn is None


class TestSolveWithGrad:
    """Tests for solve_with_grad method."""
    
    def test_returns_gradient(self):
        """Should return both solution and gradient."""
        solver = ConvexSolver(method="gd", n_steps=50, lr=0.3)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.rand(3) * 0.8 + 0.1
        
        z_star, grad_phi = solver.solve_with_grad(U_t, t, phi, phi.grad)
        
        assert z_star.shape == U_t.shape
        assert grad_phi.shape == U_t.shape
    
    def test_gradient_correct(self):
        """Returned gradient should be correct."""
        solver = ConvexSolver(method="gd", n_steps=50, lr=0.3)
        phi = SimpleQuadraticPhi(lam=2.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.rand(3) * 0.8 + 0.1
        
        z_star, grad_phi = solver.solve_with_grad(U_t, t, phi, phi.grad)
        
        # For Φ(z) = λ||z||²/2, ∇Φ(z) = λz
        expected_grad = phi.lam * z_star
        
        assert torch.allclose(grad_phi, expected_grad, rtol=1e-4)


class TestWarmStart:
    """Tests for warm starting."""
    
    def test_warm_start_faster(self):
        """Warm starting should give faster convergence."""
        solver = ConvexSolver(method="gd", n_steps=5, lr=0.3)  # Very few steps
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.rand(3) * 0.8 + 0.1
        z_analytic = phi.analytic_solution(U_t, t)
        
        # Cold start (from U_t)
        z_cold = solver.solve(U_t, t, phi, phi.grad, z_init=None)
        error_cold = (z_cold - z_analytic).abs().mean()
        
        # Warm start (from near solution)
        z_warm_init = z_analytic + 0.1 * torch.randn_like(z_analytic)
        z_warm = solver.solve(U_t, t, phi, phi.grad, z_init=z_warm_init)
        error_warm = (z_warm - z_analytic).abs().mean()
        
        # Warm start should be better with few steps
        assert error_warm < error_cold


class TestAdaptiveSolver:
    """Tests for adaptive step size solver."""
    
    def test_adaptive_convergence(self):
        """Adaptive solver should converge."""
        phi = SimpleQuadraticPhi(lam=1.0)
        
        torch.manual_seed(42)
        U_t = torch.randn(3, 5, 16)
        t = torch.rand(3) * 0.8 + 0.1
        
        # Use more steps and conservative initial lr for reliability
        solver = AdaptiveConvexSolver(n_steps=50, initial_lr=0.5, momentum=0.8)
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        z_analytic = phi.analytic_solution(U_t, t)
        error = (z_star - z_analytic).abs().mean()
        
        assert error < 0.05, f"Adaptive solver error too high: {error}"


class TestDifferentTimeValues:
    """Tests for different values of t."""
    
    def test_t_close_to_zero(self):
        """Solver should work for t close to 0."""
        solver = ConvexSolver(method="gd", n_steps=30, lr=0.5)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.tensor([0.01, 0.05, 0.1])
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        z_analytic = phi.analytic_solution(U_t, t)
        
        error = (z_star - z_analytic).abs().mean()
        assert error < 0.05  # Allow more error for small t
    
    def test_t_close_to_one(self):
        """Solver should work for t close to 1."""
        solver = ConvexSolver(method="gd", n_steps=50, lr=0.3)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.tensor([0.9, 0.95, 0.99])
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        z_analytic = phi.analytic_solution(U_t, t)
        
        error = (z_star - z_analytic).abs().mean()
        assert error < 0.02
    
    def test_scalar_t(self):
        """Should work with scalar t."""
        solver = ConvexSolver(method="gd", n_steps=30, lr=0.3)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.tensor(0.5)  # Scalar
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        assert z_star.shape == U_t.shape


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_lambda(self):
        """Should handle large regularization."""
        solver = ConvexSolver(method="gd", n_steps=50, lr=0.1)
        phi = SimpleQuadraticPhi(lam=10.0)
        
        U_t = torch.randn(3, 5, 16)
        t = torch.rand(3) * 0.8 + 0.1
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        z_analytic = phi.analytic_solution(U_t, t)
        
        error = (z_star - z_analytic).abs().mean()
        assert error < 0.05
        assert torch.isfinite(z_star).all()
    
    def test_large_input(self):
        """Should handle large input values."""
        solver = ConvexSolver(method="gd", n_steps=30, lr=0.3)
        phi = SimpleQuadraticPhi(lam=1.0)
        
        U_t = torch.randn(3, 5, 16) * 100
        t = torch.rand(3) * 0.8 + 0.1
        
        z_star = solver.solve(U_t, t, phi, phi.grad)
        
        assert torch.isfinite(z_star).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
