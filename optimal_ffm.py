"""
Optimal Functional Flow Matching (OFFM) for trajectory generation.

Extends Optimal Flow Matching (OFM) to infinite-dimensional Hilbert spaces
for generating spatio-temporal PDE trajectories.

Key differences from FFMModel:
- Learns convex potential Φ(u) instead of vector field u(t, x)
- Uses stop-gradient training with gap + KKT losses
- Samples via direct OT map instead of ODE integration

References:
    - OFM paper: Optimal Flow Matching
    - OFFM formulation in optimal_ffm/offm.md
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, List, Literal
from pathlib import Path
import time

from models.icnn import ProjectedICNN
from util.gaussian_trajectory import GaussianTrajectorySampler
from util.convex_solver import ConvexSolver
from util.hilbert import trajectory_norm_squared, trajectory_inner_product
from util.util import plot_loss_curve


class OFFMModel:
    """Optimal Functional Flow Matching for trajectory generation.
    
    Learns a convex functional Φ such that the induced OT map
    transports Gaussian reference measure to the data distribution.
    
    The training objective uses stop-gradient with:
    - Gap loss: g_θ(U_0) - g_θ(z*)
    - KKT loss: ||U_t - z* - t∇Φ(z*)||²_traj
    
    where z* is the solution to the inner convex problem.
    
    Attributes:
        Phi: ProjectedICNN - the convex functional
        solver: ConvexSolver - for inner optimization
        gaussian_sampler: GaussianTrajectorySampler - reference measure
    """
    
    def __init__(
        self,
        n_t: int,
        n_x: int,
        # Convex functional params
        projection_dim: int = 256,
        projection_type: Literal["fourier", "random"] = "fourier",
        icnn_hidden_dims: List[int] = [256, 256, 256],
        lambda_reg: float = 0.01,
        # Hilbert space params
        sobolev_s: float = 0.0,
        dt: float = 1.0,
        L: float = 1.0,
        # Loss params
        gamma_kkt: float = 0.1,
        eps_t: float = 1e-3,
        # Inner solver params
        solver_method: str = "gd",
        solver_steps: int = 10,
        solver_lr: float = 0.5,
        # Reference measure params
        gaussian_mode: str = "independent",
        kernel_length: float = 0.01,
        kernel_variance: float = 0.1,
        smoothness_alpha: float = 2.0,
        # General params
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize OFFM model.
        
        Args:
            n_t: Number of time steps in trajectory
            n_x: Number of spatial grid points
            projection_dim: Dimension of Fourier/random projection
            projection_type: Type of projection ('fourier' or 'random')
            icnn_hidden_dims: Hidden dimensions for ICNN
            lambda_reg: Quadratic regularization in Φ
            sobolev_s: Sobolev exponent for Hilbert norm
            dt: Time step for discretization
            L: Spatial domain length
            gamma_kkt: Weight for KKT loss
            eps_t: Minimum t value (avoid t=0 singularity)
            solver_method: Inner solver method ('gd', 'nesterov')
            solver_steps: Number of inner solver steps
            solver_lr: Inner solver learning rate
            gaussian_mode: Reference measure mode ('independent', 'spectral', 'separable')
            kernel_length: GP kernel lengthscale
            kernel_variance: GP kernel variance
            smoothness_alpha: Spectral smoothness for reference measure
            device: Torch device
            dtype: Torch dtype
        """
        self.n_t = n_t
        self.n_x = n_x
        self.sobolev_s = sobolev_s
        self.dt = dt
        self.L = L
        self.gamma_kkt = gamma_kkt
        self.eps_t = eps_t
        self.device = device
        self.dtype = dtype
        
        # Create convex functional
        self.Phi = ProjectedICNN(
            n_t=n_t,
            n_x=n_x,
            projection_dim=projection_dim,
            projection_type=projection_type,
            icnn_hidden_dims=icnn_hidden_dims,
            lambda_reg=lambda_reg,
            sobolev_s=sobolev_s,
            dt=dt,
            L=L,
        ).to(device)
        
        # Create inner convex solver
        self.solver = ConvexSolver(
            method=solver_method,
            n_steps=solver_steps,
            lr=solver_lr,
            sobolev_s=sobolev_s,
            dt=dt,
            L=L,
        )
        
        # Create reference measure sampler
        self.gaussian_sampler = GaussianTrajectorySampler(
            n_t=n_t,
            n_x=n_x,
            mode=gaussian_mode,
            kernel_length=kernel_length,
            kernel_variance=kernel_variance,
            smoothness_alpha=smoothness_alpha,
            L_x=L,
            T=dt * n_t,
            device=device,
            dtype=dtype,
        )
        
        # Store config for saving
        self.config = {
            'n_t': n_t, 'n_x': n_x,
            'projection_dim': projection_dim,
            'projection_type': projection_type,
            'icnn_hidden_dims': icnn_hidden_dims,
            'lambda_reg': lambda_reg,
            'sobolev_s': sobolev_s,
            'gamma_kkt': gamma_kkt,
        }
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.Phi.parameters()
    
    def sample_reference(self, batch_size: int) -> torch.Tensor:
        """Sample U_0 ~ μ_0 (Gaussian reference measure).
        
        Args:
            batch_size: Number of samples
            
        Returns:
            U_0: (batch_size, n_t, n_x) reference samples
        """
        return self.gaussian_sampler.sample(batch_size).to(self.device)
    
    def interpolate(self, U_0: torch.Tensor, U_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute linear interpolation U_t = (1-t)*U_0 + t*U_1.
        
        Args:
            U_0: (batch, n_t, n_x) reference samples
            U_1: (batch, n_t, n_x) data samples
            t: (batch,) or scalar, flow time in [0, 1]
            
        Returns:
            U_t: (batch, n_t, n_x) interpolated samples
        """
        if t.dim() == 0:
            t = t.expand(U_0.shape[0])
        t = t.view(-1, 1, 1)  # (batch, 1, 1) for broadcasting
        return (1 - t) * U_0 + t * U_1
    
    def inner_objective(
        self,
        z: torch.Tensor,
        U_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute inner objective g(z; U_t, t).
        
        g(z) = (1/2)||z||²_traj - ⟨U_t, z⟩_traj + t·Φ(z)
        
        Args:
            z: (batch, n_t, n_x) variable
            U_t: (batch, n_t, n_x) interpolated point
            t: (batch,) flow time
            
        Returns:
            g: (batch,) objective values
        """
        # ||z||²_traj / 2
        norm_sq = trajectory_norm_squared(z, s=self.sobolev_s, dt=self.dt, L=self.L)
        term1 = 0.5 * norm_sq
        
        # -⟨U_t, z⟩_traj
        inner = trajectory_inner_product(U_t, z, s=self.sobolev_s, dt=self.dt, L=self.L)
        term2 = -inner
        
        # t · Φ(z)
        Phi_z = self.Phi(z)
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        term3 = t * Phi_z
        
        return term1 + term2 + term3
    
    def compute_losses(
        self,
        U_0: torch.Tensor,
        U_1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute gap and KKT losses with stop-gradient.
        
        Args:
            U_0: (batch, n_t, n_x) reference samples
            U_1: (batch, n_t, n_x) data samples
            
        Returns:
            Dictionary with 'total', 'gap', 'kkt' losses
        """
        batch_size = U_0.shape[0]
        
        # Sample t ~ Uniform(eps_t, 1)
        t = torch.rand(batch_size, device=self.device) * (1 - self.eps_t) + self.eps_t
        
        # Compute interpolation U_t = (1-t)*U_0 + t*U_1
        U_t = self.interpolate(U_0, U_1, t)
        
        # Solve inner convex problem: z* = argmin g(z; U_t, t)
        # This uses stop-gradient (no backprop through solver)
        z_star = self.solver.solve(
            U_t=U_t,
            t=t,
            Phi=self.Phi,
            Phi_grad=self.Phi.gradient,
            z_init=U_t.clone(),  # Warm start from U_t
        )
        
        # z_star is detached - this is the key stop-gradient trick
        z_det = z_star.detach()
        
        # ===== Gap Loss =====
        # L_gap = E[g(U_0; U_t, t) - g(z*; U_t, t)]
        # Note: gradient flows through Φ(U_0) and Φ(z_det), not through z_det
        
        g_U0 = self.inner_objective(U_0, U_t, t)
        g_z = self.inner_objective(z_det, U_t, t)
        
        L_gap = (g_U0 - g_z).mean()
        
        # ===== KKT Loss =====
        # L_kkt = E[||U_t - z* - t·∇Φ(z*)||²_traj]
        # The KKT condition states: at optimum, U_t = z* + t·∇Φ(z*)
        
        grad_Phi_z = self.Phi.gradient(z_det)  # Gradient flows through Φ
        t_broad = t.view(-1, 1, 1)
        
        kkt_residual = U_t - z_det - t_broad * grad_Phi_z
        kkt_residual_norm_sq = trajectory_norm_squared(
            kkt_residual, s=self.sobolev_s, dt=self.dt, L=self.L
        )
        L_kkt = kkt_residual_norm_sq.mean()
        
        # ===== Total Loss =====
        L_total = L_gap + self.gamma_kkt * L_kkt
        
        return {
            'total': L_total,
            'gap': L_gap,
            'kkt': L_kkt,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        test_loader: Optional[DataLoader] = None,
        eval_int: int = 0,
        save_int: int = 0,
        save_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Training loop.
        
        Args:
            train_loader: DataLoader for training trajectories
            optimizer: Optimizer for Φ parameters
            epochs: Number of training epochs
            scheduler: Optional learning rate scheduler
            test_loader: Optional DataLoader for evaluation
            eval_int: Evaluate every eval_int epochs (0 = no eval)
            save_int: Save model every save_int epochs (0 = no save)
            save_path: Path to save models and plots
            verbose: Print training progress
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_gap': [],
            'train_kkt': [],
            'test_loss': [],
            'epoch_time': [],
        }
        
        evaluate = (eval_int > 0) and (test_loader is not None)
        
        for ep in range(1, epochs + 1):
            t0 = time.time()
            
            # ===== Training =====
            self.Phi.train()
            epoch_loss = 0.0
            epoch_gap = 0.0
            epoch_kkt = 0.0
            n_batches = 0
            
            for batch in train_loader:
                # batch is U_1 (data trajectories): (batch, n_t, n_x)
                # Handle both tensor and tuple/list (from TensorDataset)
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                U_1 = batch.to(self.device).to(self.dtype)
                batch_size = U_1.shape[0]
                
                # Sample reference U_0
                U_0 = self.sample_reference(batch_size)
                
                # Compute losses
                losses = self.compute_losses(U_0, U_1)
                
                # Gradient step
                optimizer.zero_grad()
                losses['total'].backward()
                optimizer.step()
                
                epoch_loss += losses['total'].item()
                epoch_gap += losses['gap'].item()
                epoch_kkt += losses['kkt'].item()
                n_batches += 1
            
            # Average losses
            epoch_loss /= n_batches
            epoch_gap /= n_batches
            epoch_kkt /= n_batches
            
            history['train_loss'].append(epoch_loss)
            history['train_gap'].append(epoch_gap)
            history['train_kkt'].append(epoch_kkt)
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_time = time.time() - t0
            history['epoch_time'].append(epoch_time)
            
            if verbose:
                print(f"Epoch {ep}/{epochs} | "
                      f"Loss: {epoch_loss:.6f} (gap: {epoch_gap:.6f}, kkt: {epoch_kkt:.6f}) | "
                      f"Time: {epoch_time:.2f}s")
            
            # ===== Evaluation =====
            if evaluate and (ep % eval_int == 0):
                self.Phi.eval()
                test_loss = 0.0
                n_test = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        if isinstance(batch, (tuple, list)):
                            batch = batch[0]
                        U_1 = batch.to(self.device).to(self.dtype)
                        U_0 = self.sample_reference(U_1.shape[0])
                        losses = self.compute_losses(U_0, U_1)
                        test_loss += losses['total'].item()
                        n_test += 1
                
                test_loss /= n_test
                history['test_loss'].append(test_loss)
                
                if verbose:
                    print(f"  Test Loss: {test_loss:.6f}")
            
            # ===== Saving =====
            if save_int > 0 and (ep % save_int == 0) and save_path is not None:
                torch.save(self.Phi.state_dict(), save_path / f'phi_epoch_{ep}.pt')
            
            # Plot loss curve
            if save_path is not None:
                plot_loss_curve(history['train_loss'], save_path / 'loss.pdf')
        
        return history
    
    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate trajectory samples via learned OT map.
        
        Samples U_0 ~ μ_0 and applies the transport map T(U_0).
        
        The OT map is derived from Φ via:
            T(u) = u - ∇Φ(u)  (for the dual potential convention)
        or via the proximal operator.
        
        For OFFM, we use the relation from the KKT condition:
        At t=1: U_1 = z* + ∇Φ(z*) where z* = U_0 at the optimum.
        So: T(U_0) ≈ U_0 + ∇Φ(U_0) (approximate, since Φ is learned)
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            samples: (n_samples, n_t, n_x) generated trajectories
        """
        self.Phi.eval()
        
        # Sample from reference measure
        U_0 = self.sample_reference(n_samples)
        
        # Apply transport map: T(U_0) = U_0 + ∇Φ(U_0)
        # This comes from the OFM formulation where ∇Φ gives the displacement
        grad_Phi = self.Phi.gradient(U_0)
        U_1 = U_0 + grad_Phi
        
        return U_1
    
    @torch.no_grad()
    def sample_with_inversion(self, n_samples: int, n_steps: int = 20) -> torch.Tensor:
        """Generate samples by inverting the flow at t=1.
        
        More accurate but slower than direct sampling.
        Solves: z* = argmin { (1/2)||z||² - ⟨U_1, z⟩ + Φ(z) }
        where U_1 = sample we want to generate.
        
        This is an alternative sampling strategy that's more consistent
        with the training objective.
        
        Args:
            n_samples: Number of samples
            n_steps: Solver steps for inversion
            
        Returns:
            samples: (n_samples, n_t, n_x) generated trajectories
        """
        self.Phi.eval()
        
        # Sample from reference
        U_0 = self.sample_reference(n_samples)
        
        # We want to find U_1 such that U_0 is the solution to the
        # inner problem at t=1. This is complex, so we use an iterative approach.
        
        # Simple approach: use gradient ascent on Φ
        # U_1 = U_0 + ∇Φ(U_0) gives approximate sample
        # For better samples, iterate
        
        U = U_0.clone()
        for _ in range(n_steps):
            grad = self.Phi.gradient(U)
            U = U + 0.1 * grad  # Small steps
        
        return U
    
    def save(self, path: Path):
        """Save model state and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'phi_state_dict': self.Phi.state_dict(),
            'config': self.config,
        }, path / 'offm_model.pt')
    
    @classmethod
    def load(cls, path: Path, device: str = 'cuda') -> 'OFFMModel':
        """Load model from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path / 'offm_model.pt', map_location=device)
        
        config = checkpoint['config']
        model = cls(
            n_t=config['n_t'],
            n_x=config['n_x'],
            projection_dim=config['projection_dim'],
            projection_type=config['projection_type'],
            icnn_hidden_dims=config['icnn_hidden_dims'],
            lambda_reg=config['lambda_reg'],
            sobolev_s=config['sobolev_s'],
            gamma_kkt=config['gamma_kkt'],
            device=device,
        )
        
        model.Phi.load_state_dict(checkpoint['phi_state_dict'])
        return model


def test_offm_model():
    """Quick test for OFFMModel."""
    print("Testing OFFMModel...")
    
    # Create model
    model = OFFMModel(
        n_t=20,
        n_x=64,
        projection_dim=64,
        icnn_hidden_dims=[64, 64],
        lambda_reg=0.1,
        gamma_kkt=0.1,
        solver_steps=5,
        device='cpu',
    )
    
    # Test sampling reference
    U_0 = model.sample_reference(5)
    print(f"  Reference samples shape: {U_0.shape}")
    
    # Test loss computation
    U_1 = torch.randn(5, 20, 64)
    losses = model.compute_losses(U_0, U_1)
    print(f"  Losses: total={losses['total']:.4f}, gap={losses['gap']:.4f}, kkt={losses['kkt']:.4f}")
    
    # Test sampling
    samples = model.sample(3)
    print(f"  Generated samples shape: {samples.shape}")
    
    print("\nOFFMModel test passed!")


if __name__ == "__main__":
    test_offm_model()
