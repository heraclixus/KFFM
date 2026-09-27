"""
Optimal Functional Flow Matching (OFFM) Experiments on KdV Data.

This script runs OFFM experiments on the deterministic KdV equation.

**Dataset Note**: The KdV dataset is a SINGLE trajectory with shape (512, 201)
representing 512 spatial points over 201 time steps. We use a sliding window
approach to create multiple trajectory samples for training.

Data shape: (N=512, T=201) - a single trajectory

Usage:
    python kdv_offm.py
    python kdv_offm.py --epochs 200 --n_seeds 3
    python kdv_offm.py --config offm_small --seed 1
"""

import sys
sys.path.append('../')

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from util.util import load_kdv
from optimal_ffm import OFFMModel
from functional_fm import FFMModel
from models.fno import FNO

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser('kdv_offm_experiment')
    
    # Data params
    parser.add_argument('--dpath', help='Path to dataset', type=str,
                        default='../data/KdV.mat')
    parser.add_argument('--spath', help='Path to save outputs', type=str,
                        default='../outputs/kdv_offm/')
    parser.add_argument('--window_size', help='Sliding window size (time steps)', type=int, default=20)
    parser.add_argument('--stride', help='Sliding window stride', type=int, default=5)
    
    # Training params
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Training epochs', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    
    # OFFM model params
    parser.add_argument('--projection_dim', help='Projection dimension', type=int, default=128)
    parser.add_argument('--projection_type', help='fourier or random', type=str, default='fourier')
    parser.add_argument('--icnn_hidden', help='ICNN hidden dims (comma-sep)', type=str, default='128,128')
    parser.add_argument('--lambda_reg', help='Quadratic regularization', type=float, default=0.2)
    parser.add_argument('--gamma_kkt', help='KKT loss weight', type=float, default=0.1)
    
    # Solver params
    parser.add_argument('--solver_method', help='gd or nesterov', type=str, default='gd')
    parser.add_argument('--solver_steps', help='Inner solver steps', type=int, default=10)
    parser.add_argument('--solver_lr', help='Inner solver learning rate', type=float, default=0.5)
    
    # Reference measure params
    parser.add_argument('--gaussian_mode', help='independent, spectral, or separable', 
                        type=str, default='spectral')
    parser.add_argument('--kernel_length', help='GP kernel lengthscale', type=float, default=0.02)
    parser.add_argument('--kernel_var', help='GP kernel variance', type=float, default=0.1)
    
    # Experiment params
    parser.add_argument('--n_seeds', help='Number of random seeds', type=int, default=3)
    parser.add_argument('--n_gen', help='Number of samples to generate', type=int, default=100)
    
    # Config selection
    parser.add_argument('--config', help='Run specific config', type=str, default=None)
    parser.add_argument('--seed', help='Run specific seed', type=int, default=None)
    parser.add_argument('--list-configs', action='store_true', help='List configs and exit')
    
    return parser.parse_args()


def create_sliding_window_dataset(data: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """Create trajectory samples using sliding window.
    
    Args:
        data: (T, 1, N) single trajectory data
        window_size: Number of time steps per trajectory
        stride: Step between windows
        
    Returns:
        (n_windows, window_size, N) tensor of trajectory samples
    """
    T, _, N = data.shape
    data = data.squeeze(1)  # (T, N)
    
    windows = []
    for start in range(0, T - window_size + 1, stride):
        window = data[start:start + window_size]  # (window_size, N)
        windows.append(window)
    
    return torch.stack(windows)  # (n_windows, window_size, N)


# =============================================================================
# OFFM Configurations (smaller for limited data)
# =============================================================================

OFFM_CONFIGS = {
    # =========================================================================
    # BASELINE - Standard FFM (this SHOULD work!)
    # =========================================================================
    
    # FFM baseline - uses standard FFM with ODE integration
    # This verifies that the data loading and setup are correct
    "baseline_ffm": {
        "model_type": "ffm",  # Special flag for FFM baseline
        "modes": 64,
        "hidden_channels": 128,
        "proj_channels": 64,
        "kernel_length": 0.02,
        "kernel_variance": 0.1,
    },
    
    # =========================================================================
    # SANITY CHECKS - Verify basic functionality
    # =========================================================================
    
    # Sanity check 1: Reference only (no training, just output reference samples)
    # This shows what the Gaussian reference measure looks like
    "sanity_reference_only": {
        "projection_type": "fourier",
        "projection_dim": 32,
        "icnn_hidden_dims": [32],
        "lambda_reg": 1.0,  # High reg makes Phi nearly quadratic
        "gamma_kkt": 0.0,
        "solver_steps": 1,
        "gaussian_mode": "spectral",
        "sanity_mode": "reference_only",  # Special flag
    },
    
    # Sanity check 2: No inner optimization (solver_steps=1)
    # z* ≈ U_t, so gap ≈ g(U_0) - g(U_t), training mostly via gap loss
    "sanity_no_inner_opt": {
        "projection_type": "fourier",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.5,
        "gamma_kkt": 0.0,  # No KKT loss since z* is not optimized
        "solver_steps": 1,
        "gaussian_mode": "spectral",
    },
    
    # Sanity check 3: Pure quadratic (no ICNN, just λ||u||²)
    # With high lambda_reg and tiny ICNN, Phi ≈ (λ/2)||u||²
    # Gradient: ∇Φ ≈ λu, so T(u) = u + λu = (1+λ)u (scaling)
    "sanity_pure_quadratic": {
        "projection_type": "fourier",
        "projection_dim": 16,
        "icnn_hidden_dims": [8],  # Tiny ICNN
        "lambda_reg": 1.0,  # Dominated by quadratic
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # Sanity check 4: Identity-like (very small lambda, minimal transformation)
    "sanity_identity_like": {
        "projection_type": "fourier",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.01,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # =========================================================================
    # MAIN CONFIGS
    # =========================================================================
    
    # Small model (best for limited data)
    "offm_small": {
        "projection_type": "fourier",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.3,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # Medium model
    "offm_medium": {
        "projection_type": "fourier",
        "projection_dim": 128,
        "icnn_hidden_dims": [128, 128],
        "lambda_reg": 0.2,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # Random projection
    "offm_random": {
        "projection_type": "random",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.3,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "independent",
    },
    
    # Higher regularization (more robust with limited data)
    "offm_high_reg": {
        "projection_type": "fourier",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.5,
        "gamma_kkt": 0.2,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # More solver steps
    "offm_more_solver": {
        "projection_type": "fourier",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.3,
        "gamma_kkt": 0.1,
        "solver_steps": 20,
        "gaussian_mode": "spectral",
    },
}


# =============================================================================
# Training Functions
# =============================================================================

def create_offm_model(
    n_t: int,
    n_x: int,
    config: Dict[str, Any],
    args,
    device: str,
) -> OFFMModel:
    """Create OFFM model from config."""
    
    return OFFMModel(
        n_t=n_t,
        n_x=n_x,
        projection_dim=config.get("projection_dim", args.projection_dim),
        projection_type=config.get("projection_type", args.projection_type),
        icnn_hidden_dims=config.get("icnn_hidden_dims", [int(x) for x in args.icnn_hidden.split(',')]),
        lambda_reg=config.get("lambda_reg", args.lambda_reg),
        gamma_kkt=config.get("gamma_kkt", args.gamma_kkt),
        solver_method=config.get("solver_method", args.solver_method),
        solver_steps=config.get("solver_steps", args.solver_steps),
        solver_lr=config.get("solver_lr", args.solver_lr),
        gaussian_mode=config.get("gaussian_mode", args.gaussian_mode),
        kernel_length=config.get("kernel_length", args.kernel_length),
        kernel_variance=config.get("kernel_var", args.kernel_var),
        device=device,
        dtype=torch.float32,
    )


def train_ffm_baseline(
    config_name: str,
    config: Dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    ground_truth: torch.Tensor,
    n_t: int,
    n_x: int,
    args,
    save_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """Train FFM baseline for comparison."""
    
    print(f"\n  Training FFM baseline {config_name} (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create FNO model for vector field
    # Note: FFM expects data shape (batch, 1, n_x) for 1D, but we have (batch, n_t, n_x)
    # We'll treat each time slice independently for fair comparison
    model = FNO(
        config.get("modes", 64),
        vis_channels=1,
        hidden_channels=config.get("hidden_channels", 128),
        proj_channels=config.get("proj_channels", 64),
        x_dim=1,
        t_scaling=1000
    ).to(device)
    
    # Create FFM
    ffm = FFMModel(
        model,
        kernel_length=config.get("kernel_length", 0.02),
        kernel_variance=config.get("kernel_variance", 0.1),
        sigma_min=1e-4,
        device=device,
    )
    
    # Reshape data for FFM: (batch, n_t, n_x) -> (batch*n_t, 1, n_x)
    reshaped_data = ground_truth.reshape(-1, 1, n_x)
    reshaped_loader = DataLoader(reshaped_data, batch_size=args.bs * 4, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    t0 = time.time()
    ffm.train(
        reshaped_loader, 
        optimizer, 
        epochs=args.epochs,
        scheduler=scheduler,
        eval_int=0,
        save_int=args.epochs,
        generate=False,
        save_path=save_dir,
    )
    train_time = time.time() - t0
    
    # Generate samples: (n_gen * n_t) 1D samples, then reshape to trajectories
    print(f"    Generating {args.n_gen} trajectory samples...")
    n_1d_samples = args.n_gen * n_t
    samples_1d = ffm.sample([n_x], n_samples=n_1d_samples, n_channels=1).cpu().squeeze(1)
    # Reshape back to trajectories: (n_gen * n_t, n_x) -> (n_gen, n_t, n_x)
    samples = samples_1d.reshape(args.n_gen, n_t, n_x)
    
    # Generate reference samples for comparison
    from util.util import make_grid
    from util.gaussian_process import GPPrior
    gp = GPPrior(lengthscale=config.get("kernel_length", 0.02), 
                 var=config.get("kernel_variance", 0.1), device='cpu')
    grid = make_grid([n_x])
    ref_1d = gp.sample(grid, [n_x], n_samples=50 * n_t, n_channels=1).squeeze(1)
    ref_samples = ref_1d.reshape(50, n_t, n_x)
    
    # Compute quality metrics
    print(f"    Computing quality metrics...")
    quality_metrics = compute_trajectory_metrics(
        ground_truth=ground_truth,
        generated=samples,
        config_name=config_name,
    )
    quality_metrics['train_time'] = train_time
    quality_metrics['final_train_loss'] = None
    quality_metrics['final_gap_loss'] = None  
    quality_metrics['final_kkt_loss'] = None
    
    ref_metrics = compute_trajectory_metrics(
        ground_truth=ground_truth,
        generated=ref_samples,
        config_name=f"{config_name}_reference",
    )
    quality_metrics['ref_mean_mse'] = ref_metrics['mean_mse']
    quality_metrics['ref_spectrum_mse'] = ref_metrics['spectrum_mse']
    
    # Save results
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(ref_samples, save_dir / 'reference_samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(save_dir / 'quality_metrics.json', 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    history = {'train_loss': [], 'train_gap': [], 'train_kkt': [], 'epoch_time': []}
    
    return {
        'samples': samples,
        'history': history,
        'quality_metrics': quality_metrics,
    }


def train_single_config(
    config_name: str,
    config: Dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    ground_truth: torch.Tensor,
    n_t: int,
    n_x: int,
    args,
    save_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """Train a single OFFM configuration."""
    
    print(f"\n  Training {config_name} (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create model
    model = create_offm_model(n_t, n_x, config, args, device)
    
    # Check for sanity mode
    sanity_mode = config.get("sanity_mode", None)
    
    if sanity_mode == "reference_only":
        # Sanity check: Just output reference samples (no training)
        print(f"    SANITY MODE: reference_only - skipping training")
        print(f"    Generating {args.n_gen} reference samples...")
        samples = model.sample_reference(args.n_gen).cpu()
        
        history = {
            'train_loss': [],
            'train_gap': [],
            'train_kkt': [],
            'epoch_time': [],
        }
        train_time = 0.0
    else:
        # Normal training
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        t0 = time.time()
        history = model.train(
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=args.epochs,
            scheduler=scheduler,
            eval_int=20,
            save_int=args.epochs,
            save_path=save_dir,
            verbose=True,
        )
        train_time = time.time() - t0
        
        # Generate samples via learned OT map
        print(f"    Generating {args.n_gen} samples via OT map...")
        samples = model.sample(n_samples=args.n_gen).cpu()
    
    # Also generate reference samples for comparison
    print(f"    Generating {min(args.n_gen, 50)} reference samples for comparison...")
    ref_samples = model.sample_reference(min(args.n_gen, 50)).cpu()
    
    # Compute quality metrics
    print(f"    Computing quality metrics...")
    quality_metrics = compute_trajectory_metrics(
        ground_truth=ground_truth,
        generated=samples,
        config_name=config_name,
    )
    quality_metrics['train_time'] = train_time
    quality_metrics['final_train_loss'] = history['train_loss'][-1] if history['train_loss'] else None
    quality_metrics['final_gap_loss'] = history['train_gap'][-1] if history['train_gap'] else None
    quality_metrics['final_kkt_loss'] = history['train_kkt'][-1] if history['train_kkt'] else None
    
    # Also compute metrics for reference samples
    ref_metrics = compute_trajectory_metrics(
        ground_truth=ground_truth,
        generated=ref_samples,
        config_name=f"{config_name}_reference",
    )
    quality_metrics['ref_mean_mse'] = ref_metrics['mean_mse']
    quality_metrics['ref_spectrum_mse'] = ref_metrics['spectrum_mse']
    
    # Save results
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(ref_samples, save_dir / 'reference_samples.pt')
    torch.save(history, save_dir / 'training_history.pt')
    model.save(save_dir)
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(save_dir / 'quality_metrics.json', 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    return {
        'samples': samples,
        'history': history,
        'quality_metrics': quality_metrics,
    }


def compute_trajectory_metrics(
    ground_truth: torch.Tensor,
    generated: torch.Tensor,
    config_name: str,
) -> Dict[str, Any]:
    """Compute quality metrics for trajectory generation."""
    metrics = {'config_name': config_name}
    
    # Basic statistics comparison
    gt_mean = ground_truth.mean(dim=0)
    gen_mean = generated.mean(dim=0)
    metrics['mean_mse'] = ((gt_mean - gen_mean) ** 2).mean().item()
    
    gt_std = ground_truth.std(dim=0)
    gen_std = generated.std(dim=0)
    metrics['std_mse'] = ((gt_std - gen_std) ** 2).mean().item()
    
    # Spatial spectrum comparison
    gt_spectrum = torch.fft.rfft(ground_truth, dim=-1).abs().mean(dim=(0, 1))
    gen_spectrum = torch.fft.rfft(generated, dim=-1).abs().mean(dim=(0, 1))
    metrics['spectrum_mse'] = ((gt_spectrum - gen_spectrum) ** 2).mean().item()
    
    # Temporal correlation
    gt_temp_corr = compute_temporal_correlation(ground_truth)
    gen_temp_corr = compute_temporal_correlation(generated)
    metrics['temporal_corr_mse'] = ((gt_temp_corr - gen_temp_corr) ** 2).mean().item()
    
    # Energy
    gt_energy = (ground_truth ** 2).mean(dim=-1)
    gen_energy = (generated ** 2).mean(dim=-1)
    metrics['energy_mean_mse'] = ((gt_energy.mean(dim=0) - gen_energy.mean(dim=0)) ** 2).mean().item()
    
    # Smoothness
    gt_dt = torch.diff(ground_truth, dim=1)
    gen_dt = torch.diff(generated, dim=1)
    metrics['smoothness_mse'] = ((gt_dt.std() - gen_dt.std()) ** 2).item()
    
    return metrics


def compute_temporal_correlation(x: torch.Tensor, max_lag: int = 10) -> torch.Tensor:
    """Compute autocorrelation in time dimension."""
    correlations = []
    x_mean = x.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    var = (x_centered ** 2).mean()
    
    for lag in range(1, max_lag + 1):
        if lag >= x.shape[1]:
            correlations.append(0.0)
        else:
            corr = (x_centered[:, :-lag] * x_centered[:, lag:]).mean() / (var + 1e-8)
            correlations.append(corr.item())
    
    return torch.tensor(correlations)


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectory_samples(
    ground_truth: torch.Tensor,
    generated: torch.Tensor,
    config_name: str,
    save_path: Path,
    n_plot: int = 5,
):
    """Plot sample trajectories."""
    
    n_plot = min(n_plot, ground_truth.shape[0], generated.shape[0])
    fig, axes = plt.subplots(2, n_plot, figsize=(3*n_plot, 6))
    
    n_t, n_x = ground_truth.shape[1], ground_truth.shape[2]
    
    for i in range(n_plot):
        # Ground truth
        ax = axes[0, i]
        im = ax.imshow(ground_truth[i].numpy().T, aspect='auto', cmap='RdBu_r',
                       extent=[0, n_t, 0, n_x])
        ax.set_xlabel('Time')
        if i == 0:
            ax.set_ylabel('Ground Truth\nSpace')
        ax.set_title(f'Sample {i+1}')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Generated
        ax = axes[1, i]
        im = ax.imshow(generated[i].numpy().T, aspect='auto', cmap='RdBu_r',
                       extent=[0, n_t, 0, n_x])
        ax.set_xlabel('Time')
        if i == 0:
            ax.set_ylabel(f'{config_name}\nSpace')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'KdV Trajectories: {config_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_spectrum_comparison(
    ground_truth: torch.Tensor,
    generated_dict: Dict[str, torch.Tensor],
    save_path: Path,
):
    """Plot spatial spectrum comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spatial spectrum
    ax = axes[0]
    gt_spectrum = torch.fft.rfft(ground_truth, dim=-1).abs().mean(dim=(0, 1))
    freqs = torch.arange(len(gt_spectrum))
    
    ax.semilogy(freqs, gt_spectrum, 'k-', linewidth=2, label='Ground Truth')
    
    for name, samples in generated_dict.items():
        spectrum = torch.fft.rfft(samples, dim=-1).abs().mean(dim=(0, 1))
        ax.semilogy(freqs, spectrum, '--', linewidth=1.5, label=name, alpha=0.8)
    
    ax.set_xlabel('Frequency Mode')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spatial Power Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temporal correlation
    ax = axes[1]
    gt_corr = compute_temporal_correlation(ground_truth, max_lag=15)
    lags = torch.arange(1, len(gt_corr) + 1)
    
    ax.plot(lags, gt_corr, 'k-', linewidth=2, label='Ground Truth')
    
    for name, samples in generated_dict.items():
        corr = compute_temporal_correlation(samples, max_lag=15)
        ax.plot(lags, corr, '--', linewidth=1.5, label=name, alpha=0.8)
    
    ax.set_xlabel('Time Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Temporal Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    histories: Dict[str, Dict],
    save_path: Path,
):
    """Plot training loss curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], label=name)
        axes[1].plot(epochs, history['train_gap'], label=name)
        axes[2].plot(epochs, history['train_kkt'], label=name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Gap Loss')
    axes[1].set_title('Gap Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KKT Loss')
    axes[2].set_title('KKT Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # List configs mode
    if args.list_configs:
        print("\nAvailable OFFM configurations:")
        print("=" * 60)
        for name, config in OFFM_CONFIGS.items():
            print(f"  {name}")
            print(f"    projection: {config.get('projection_type')}, dim={config.get('projection_dim')}")
            print(f"    solver_steps: {config.get('solver_steps')}, lambda_reg: {config.get('lambda_reg')}")
        sys.exit(0)
    
    # Load data
    print("\nLoading KdV data...")
    print("NOTE: KdV is a SINGLE trajectory with 512 spatial points × 201 time steps")
    print(f"      Using sliding window (size={args.window_size}, stride={args.stride})")
    
    raw_data = load_kdv(args.dpath, mode='snapshot')  # (201, 1, 512)
    print(f"Raw data shape: {raw_data.shape}")
    
    # Create sliding window trajectory samples
    train_data = create_sliding_window_dataset(raw_data, args.window_size, args.stride)
    print(f"Trajectory samples shape: {train_data.shape}")  # (n_windows, window_size, 512)
    
    n_samples, n_t, n_x = train_data.shape
    print(f"Created {n_samples} trajectory samples")
    print(f"Each trajectory: {n_t} time steps × {n_x} spatial points")
    
    ground_truth = train_data.clone()
    
    # Create data loader
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    
    # Output directory
    spath = Path(args.spath)
    spath.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = vars(args)
    config_dict['n_t'] = n_t
    config_dict['n_x'] = n_x
    config_dict['n_samples'] = n_samples
    config_dict['device'] = device
    torch.save(config_dict, spath / 'experiment_config.pt')
    torch.save(ground_truth, spath / 'ground_truth.pt')
    
    # Random seeds
    random_seeds = [2**i for i in range(args.n_seeds)]
    
    # Determine which configs to run
    if args.config:
        if args.config not in OFFM_CONFIGS:
            print(f"ERROR: Config '{args.config}' not found.")
            print(f"Available: {list(OFFM_CONFIGS.keys())}")
            sys.exit(1)
        configs_to_run = {args.config: OFFM_CONFIGS[args.config]}
        seeds_to_run = [args.seed] if args.seed else random_seeds
    else:
        configs_to_run = OFFM_CONFIGS
        seeds_to_run = random_seeds
    
    print("\n" + "=" * 60)
    print(f"OFFM Experiments on KdV Trajectories")
    print(f"Configs: {len(configs_to_run)}, Seeds: {len(seeds_to_run)}")
    print(f"Output: {spath}")
    print("=" * 60)
    
    # Run experiments
    all_results = {}
    all_histories = {}
    all_samples = {}
    
    for config_name, config in configs_to_run.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        
        config_results = {}
        
        # Check if this is an FFM baseline
        is_ffm_baseline = config.get("model_type") == "ffm"
        
        for seed in seeds_to_run:
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            if is_ffm_baseline:
                # Use FFM baseline training
                result = train_ffm_baseline(
                    config_name=config_name,
                    config=config,
                    seed=seed,
                    train_loader=train_loader,
                    ground_truth=ground_truth,
                    n_t=n_t,
                    n_x=n_x,
                    args=args,
                    save_dir=config_dir,
                    device=device,
                )
            else:
                # Use OFFM training
                result = train_single_config(
                    config_name=config_name,
                    config=config,
                    seed=seed,
                    train_loader=train_loader,
                    ground_truth=ground_truth,
                    n_t=n_t,
                    n_x=n_x,
                    args=args,
                    save_dir=config_dir,
                    device=device,
                )
            
            config_results[seed] = result
        
        all_results[config_name] = config_results
        
        # Use first seed for visualization
        first_seed = seeds_to_run[0]
        all_histories[config_name] = config_results[first_seed]['history']
        all_samples[config_name] = config_results[first_seed]['samples']
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # Plot sample trajectories
    for config_name, samples in all_samples.items():
        plot_trajectory_samples(
            ground_truth=ground_truth,
            generated=samples,
            config_name=config_name,
            save_path=spath / f'samples_{config_name}.pdf',
        )
    
    # Plot spectrum comparison
    plot_spectrum_comparison(
        ground_truth=ground_truth,
        generated_dict=all_samples,
        save_path=spath / 'spectrum_comparison.pdf',
    )
    
    # Plot training curves
    plot_training_curves(
        histories=all_histories,
        save_path=spath / 'training_curves.pdf',
    )
    
    # Aggregate and save summary
    summary = {
        'dataset': 'KdV (Deterministic, Sliding Window)',
        'n_samples': n_samples,
        'window_size': args.window_size,
        'stride': args.stride,
        'n_t': n_t,
        'n_x': n_x,
        'epochs': args.epochs,
        'batch_size': args.bs,
        'n_seeds': len(seeds_to_run),
        'configs': {},
    }
    
    for config_name, config_results in all_results.items():
        metrics_list = [r['quality_metrics'] for r in config_results.values()]
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key == 'config_name':
                continue
            values = [m[key] for m in metrics_list if m[key] is not None]
            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values))
        
        summary['configs'][config_name] = avg_metrics
    
    with open(spath / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("Results Summary")
    print("=" * 90)
    print(f"{'Config':<25} {'Mean MSE':>12} {'Ref MSE':>12} {'Spectrum MSE':>14} {'Train Time':>12}")
    print("-" * 90)
    
    for config_name, metrics in summary['configs'].items():
        mean_mse = metrics.get('mean_mse', float('nan'))
        ref_mse = metrics.get('ref_mean_mse', float('nan'))
        spec_mse = metrics.get('spectrum_mse', float('nan'))
        train_time = metrics.get('train_time', float('nan'))
        
        # Highlight if generated is better than reference
        better = "✓" if mean_mse < ref_mse else ""
        print(f"{config_name:<25} {mean_mse:>12.2f} {ref_mse:>12.2f} {spec_mse:>14.2f} {train_time:>10.1f}s {better}")
    
    print("-" * 90)
    print("Note: 'Ref MSE' shows the MSE of raw reference (Gaussian) samples.")
    print("      If 'Mean MSE' < 'Ref MSE', the model learned something useful!")
    print("\n" + "=" * 90)
    print(f"Complete! Results saved to: {spath}")
    print("=" * 90)


if __name__ == '__main__':
    main()
