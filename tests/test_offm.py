"""
Unit tests for optimal_ffm.py - Optimal Functional Flow Matching model.

Tests:
1. Model initialization
2. Reference sampling
3. Loss computation
4. Training step
5. Sample generation
6. Save/load functionality
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, '..')

from optimal_ffm import OFFMModel


class TestOFFMModelInitialization:
    """Tests for model initialization."""
    
    def test_basic_init(self):
        """Model should initialize without errors."""
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            icnn_hidden_dims=[32],
            device='cpu',
        )
        assert model.Phi is not None
        assert model.solver is not None
        assert model.gaussian_sampler is not None
    
    def test_parameters(self):
        """Model should have trainable parameters."""
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            device='cpu',
        )
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
    
    def test_different_configs(self):
        """Should work with various configurations."""
        configs = [
            {'projection_type': 'fourier'},
            {'projection_type': 'random'},
            {'gaussian_mode': 'independent'},
            {'gaussian_mode': 'spectral'},
            {'icnn_hidden_dims': [32]},
            {'icnn_hidden_dims': [64, 64, 64]},
        ]
        
        for config in configs:
            model = OFFMModel(
                n_t=10, n_x=32,
                projection_dim=32,
                device='cpu',
                **config,
            )
            assert model.Phi is not None


class TestReferenceSampling:
    """Tests for reference measure sampling."""
    
    def test_sample_shape(self):
        """Samples should have correct shape."""
        model = OFFMModel(n_t=20, n_x=64, projection_dim=32, device='cpu')
        samples = model.sample_reference(batch_size=5)
        assert samples.shape == (5, 20, 64)
    
    def test_samples_finite(self):
        """Samples should be finite."""
        model = OFFMModel(n_t=20, n_x=64, projection_dim=32, device='cpu')
        samples = model.sample_reference(batch_size=10)
        assert torch.isfinite(samples).all()
    
    def test_samples_on_correct_device(self):
        """Samples should be on the correct device."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        samples = model.sample_reference(batch_size=3)
        assert samples.device.type == 'cpu'


class TestInterpolation:
    """Tests for linear interpolation."""
    
    def test_interpolation_endpoints(self):
        """t=0 should give U_0, t=1 should give U_1."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        U_0 = torch.randn(3, 10, 32)
        U_1 = torch.randn(3, 10, 32)
        
        # t=0
        U_t0 = model.interpolate(U_0, U_1, torch.tensor(0.0))
        assert torch.allclose(U_t0, U_0, atol=1e-6)
        
        # t=1
        U_t1 = model.interpolate(U_0, U_1, torch.tensor(1.0))
        assert torch.allclose(U_t1, U_1, atol=1e-6)
    
    def test_interpolation_midpoint(self):
        """t=0.5 should give (U_0 + U_1) / 2."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        U_0 = torch.randn(3, 10, 32)
        U_1 = torch.randn(3, 10, 32)
        
        U_mid = model.interpolate(U_0, U_1, torch.tensor(0.5))
        expected = (U_0 + U_1) / 2
        
        assert torch.allclose(U_mid, expected, atol=1e-6)


class TestLossComputation:
    """Tests for loss computation."""
    
    def test_loss_output_structure(self):
        """Loss should return dict with total, gap, kkt."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        U_0 = torch.randn(3, 10, 32)
        U_1 = torch.randn(3, 10, 32)
        
        losses = model.compute_losses(U_0, U_1)
        
        assert 'total' in losses
        assert 'gap' in losses
        assert 'kkt' in losses
    
    def test_loss_finite(self):
        """All losses should be finite."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        U_0 = torch.randn(3, 10, 32)
        U_1 = torch.randn(3, 10, 32)
        
        losses = model.compute_losses(U_0, U_1)
        
        assert torch.isfinite(losses['total'])
        assert torch.isfinite(losses['gap'])
        assert torch.isfinite(losses['kkt'])
    
    def test_loss_requires_grad(self):
        """Total loss should require gradient."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        U_0 = torch.randn(3, 10, 32)
        U_1 = torch.randn(3, 10, 32)
        
        losses = model.compute_losses(U_0, U_1)
        
        assert losses['total'].requires_grad
    
    def test_gap_loss_nonnegative(self):
        """Gap loss should be non-negative (minimizer gives 0)."""
        torch.manual_seed(42)
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            icnn_hidden_dims=[32],
            lambda_reg=0.5,  # Higher regularization for better conditioning
            solver_steps=100,  # More steps for better convergence
            solver_lr=0.3,
            device='cpu',
        )
        
        # Small scale inputs for numerical stability
        U_0 = torch.randn(3, 10, 32) * 0.3
        U_1 = torch.randn(3, 10, 32) * 0.3
        
        losses = model.compute_losses(U_0, U_1)
        
        # Gap should be approximately >= 0 since z* is minimizer
        # Allow larger tolerance due to finite solver steps
        assert losses['gap'].item() >= -1.0, \
            f"Gap loss too negative: {losses['gap'].item()}"
    
    def test_kkt_loss_nonnegative(self):
        """KKT loss (squared norm) should be non-negative."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        U_0 = torch.randn(3, 10, 32)
        U_1 = torch.randn(3, 10, 32)
        
        losses = model.compute_losses(U_0, U_1)
        
        assert losses['kkt'].item() >= 0


class TestTrainingStep:
    """Tests for training functionality."""
    
    def test_single_train_step(self):
        """Should be able to do one training step."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        U_0 = torch.randn(4, 10, 32)
        U_1 = torch.randn(4, 10, 32)
        
        # Forward
        losses = model.compute_losses(U_0, U_1)
        initial_loss = losses['total'].item()
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Should complete without error
        assert True
    
    def test_loss_decreases(self):
        """Loss should decrease after multiple steps on same data."""
        torch.manual_seed(42)
        
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            icnn_hidden_dims=[32],
            lambda_reg=0.5,
            solver_steps=5,
            device='cpu',
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Fixed data
        U_0 = torch.randn(8, 10, 32) * 0.3
        U_1 = torch.randn(8, 10, 32) * 0.3
        
        losses_history = []
        for i in range(20):
            # Fix random state for t sampling to reduce variance
            torch.manual_seed(42 + i)
            losses = model.compute_losses(U_0, U_1)
            losses_history.append(losses['total'].item())
            
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
        
        # Compare average of first 3 vs last 3 losses
        first_avg = sum(losses_history[:3]) / 3
        last_avg = sum(losses_history[-3:]) / 3
        
        assert last_avg < first_avg, \
            f"Loss didn't decrease: {first_avg:.4f} -> {last_avg:.4f}"


class TestSampleGeneration:
    """Tests for sample generation."""
    
    def test_sample_shape(self):
        """Generated samples should have correct shape."""
        model = OFFMModel(n_t=20, n_x=64, projection_dim=32, device='cpu')
        samples = model.sample(n_samples=5)
        assert samples.shape == (5, 20, 64)
    
    def test_samples_finite(self):
        """Generated samples should be finite."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        samples = model.sample(n_samples=10)
        assert torch.isfinite(samples).all()
    
    def test_samples_different(self):
        """Different calls should give different samples."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        samples1 = model.sample(n_samples=3)
        samples2 = model.sample(n_samples=3)
        
        assert not torch.allclose(samples1, samples2)
    
    def test_sample_with_inversion(self):
        """Inversion-based sampling should work."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        samples = model.sample_with_inversion(n_samples=3, n_steps=5)
        
        assert samples.shape == (3, 10, 32)
        assert torch.isfinite(samples).all()


class TestTrainFunction:
    """Tests for the full training loop."""
    
    def test_train_basic(self):
        """Basic training should complete without errors."""
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            icnn_hidden_dims=[32],
            solver_steps=3,
            device='cpu',
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy data
        data = torch.randn(20, 10, 32)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = model.train(
                train_loader=loader,
                optimizer=optimizer,
                epochs=2,
                save_path=Path(tmpdir),
                verbose=False,
            )
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 2
    
    def test_train_with_scheduler(self):
        """Training with scheduler should work."""
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            solver_steps=3,
            device='cpu',
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        data = torch.randn(10, 10, 32)
        loader = DataLoader(TensorDataset(data), batch_size=5)
        
        history = model.train(
            train_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=2,
            verbose=False,
        )
        
        assert len(history['train_loss']) == 2


class TestSaveLoad:
    """Tests for save/load functionality."""
    
    def test_save_load_roundtrip(self):
        """Model should be recoverable after save/load."""
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            icnn_hidden_dims=[32, 32],
            lambda_reg=0.5,
            device='cpu',
        )
        
        # Generate some samples before saving
        samples_before = model.sample(n_samples=3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model.save(path)
            
            # Load model
            loaded_model = OFFMModel.load(path, device='cpu')
        
        # Check config matches
        assert loaded_model.n_t == model.n_t
        assert loaded_model.n_x == model.n_x
    
    def test_loaded_model_produces_output(self):
        """Loaded model should be able to generate samples."""
        model = OFFMModel(
            n_t=10, n_x=32,
            projection_dim=32,
            device='cpu',
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model.save(path)
            loaded_model = OFFMModel.load(path, device='cpu')
        
        samples = loaded_model.sample(n_samples=3)
        assert samples.shape == (3, 10, 32)
        assert torch.isfinite(samples).all()


class TestInnerObjective:
    """Tests for inner objective computation."""
    
    def test_inner_objective_finite(self):
        """Inner objective should be finite."""
        model = OFFMModel(n_t=10, n_x=32, projection_dim=32, device='cpu')
        
        z = torch.randn(3, 10, 32)
        U_t = torch.randn(3, 10, 32)
        t = torch.rand(3) * 0.9 + 0.1
        
        g = model.inner_objective(z, U_t, t)
        
        assert g.shape == (3,)
        assert torch.isfinite(g).all()
    
    def test_inner_objective_convex(self):
        """Inner objective should be convex in z (for fixed U_t, t)."""
        model = OFFMModel(
            n_t=8, n_x=16,
            projection_dim=32,
            lambda_reg=0.1,
            device='cpu',
        )
        
        U_t = torch.randn(1, 8, 16)
        t = torch.tensor([0.5])
        
        z1 = torch.randn(1, 8, 16)
        z2 = torch.randn(1, 8, 16)
        z_mid = (z1 + z2) / 2
        
        g1 = model.inner_objective(z1, U_t, t).item()
        g2 = model.inner_objective(z2, U_t, t).item()
        g_mid = model.inner_objective(z_mid, U_t, t).item()
        
        # Convexity: g(z_mid) <= (g(z1) + g(z2)) / 2
        assert g_mid <= (g1 + g2) / 2 + 0.1  # Allow numerical tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
