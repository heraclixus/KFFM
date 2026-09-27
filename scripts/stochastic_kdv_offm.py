"""
Optimal Functional Flow Matching (OFFM) Experiments on Stochastic KdV.

This script runs OFFM experiments on stochastic Korteweg-de Vries 
equation trajectory data. OFFM learns optimal transport maps in function
space via convex potentials.

Key features:
- Works with full spatio-temporal trajectories (not snapshots)
- No ODE integration needed for sampling (direct OT map)
- Stop-gradient training with gap + KKT losses

Data shape: (1200, 101, 128) = (batch_size, T_time, N_spatial)

Usage:
    python stochastic_kdv_offm.py
    python stochastic_kdv_offm.py --epochs 100 --n_seeds 3
    python stochastic_kdv_offm.py --config offm_fourier --seed 1
"""

import sys
sys.path.append('../')

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from util.util import load_stochastic_kdv
from util.eval import GenerationQualityMetrics
from optimal_ffm import OFFMModel

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser('stochastic_kdv_offm_experiment')
    
    # Data params
    parser.add_argument('--dpath', help='Path to dataset', type=str, 
                        default='../data/stochastic_kdv.mat')
    parser.add_argument('--spath', help='Path to save outputs', type=str,
                        default='../outputs/stochastic_kdv_offm/')
    parser.add_argument('--subsample_time', help='Subsample time steps', type=int, default=1)
    
    # Training params
    parser.add_argument('--ntr', help='Number of training samples', type=int, default=1000)
    parser.add_argument('--nte', help='Number of test samples', type=int, default=200)
    parser.add_argument('--bs', help='Batch size', type=int, default=32)
    parser.add_argument('--epochs', help='Training epochs', type=int, default=100)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    
    # OFFM model params
    parser.add_argument('--projection_dim', help='Projection dimension', type=int, default=256)
    parser.add_argument('--projection_type', help='fourier or random', type=str, default='fourier')
    parser.add_argument('--icnn_hidden', help='ICNN hidden dims (comma-sep)', type=str, default='256,256,256')
    parser.add_argument('--lambda_reg', help='Quadratic regularization', type=float, default=0.1)
    parser.add_argument('--gamma_kkt', help='KKT loss weight', type=float, default=0.1)
    
    # Solver params
    parser.add_argument('--solver_method', help='gd or nesterov', type=str, default='gd')
    parser.add_argument('--solver_steps', help='Inner solver steps', type=int, default=10)
    parser.add_argument('--solver_lr', help='Inner solver learning rate', type=float, default=0.5)
    
    # Reference measure params
    parser.add_argument('--gaussian_mode', help='independent, spectral, or separable', 
                        type=str, default='spectral')
    parser.add_argument('--kernel_length', help='GP kernel lengthscale', type=float, default=0.05)
    parser.add_argument('--kernel_var', help='GP kernel variance', type=float, default=0.1)
    
    # Experiment params
    parser.add_argument('--n_seeds', help='Number of random seeds', type=int, default=3)
    parser.add_argument('--n_gen', help='Number of samples to generate', type=int, default=200)
    
    # Config selection (for parallel runs)
    parser.add_argument('--config', help='Run specific config', type=str, default=None)
    parser.add_argument('--seed', help='Run specific seed', type=int, default=None)
    parser.add_argument('--list-configs', action='store_true', help='List configs and exit')
    
    return parser.parse_args()


# =============================================================================
# OFFM Configurations
# =============================================================================

OFFM_CONFIGS = {
    # Fourier projection (default, captures frequency structure)
    "offm_fourier": {
        "projection_type": "fourier",
        "projection_dim": 256,
        "icnn_hidden_dims": [256, 256, 256],
        "lambda_reg": 0.1,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # Fourier with higher projection dim
    "offm_fourier_large": {
        "projection_type": "fourier",
        "projection_dim": 512,
        "icnn_hidden_dims": [512, 512, 512],
        "lambda_reg": 0.05,
        "gamma_kkt": 0.1,
        "solver_steps": 15,
        "gaussian_mode": "spectral",
    },
    
    # Random projection (simpler, faster)
    "offm_random": {
        "projection_type": "random",
        "projection_dim": 256,
        "icnn_hidden_dims": [256, 256, 256],
        "lambda_reg": 0.1,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "independent",
    },
    
    # More regularization (smoother)
    "offm_high_reg": {
        "projection_type": "fourier",
        "projection_dim": 256,
        "icnn_hidden_dims": [256, 256, 256],
        "lambda_reg": 0.5,
        "gamma_kkt": 0.2,
        "solver_steps": 10,
        "gaussian_mode": "spectral",
    },
    
    # More solver steps (better inner opt)
    "offm_more_solver": {
        "projection_type": "fourier",
        "projection_dim": 256,
        "icnn_hidden_dims": [256, 256, 256],
        "lambda_reg": 0.1,
        "gamma_kkt": 0.1,
        "solver_steps": 30,
        "gaussian_mode": "spectral",
    },
    
    # Nesterov acceleration
    "offm_nesterov": {
        "projection_type": "fourier",
        "projection_dim": 256,
        "icnn_hidden_dims": [256, 256, 256],
        "lambda_reg": 0.1,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "solver_method": "nesterov",
        "gaussian_mode": "spectral",
    },
    
    # Separable reference (time-correlated)
    "offm_separable_ref": {
        "projection_type": "fourier",
        "projection_dim": 256,
        "icnn_hidden_dims": [256, 256, 256],
        "lambda_reg": 0.1,
        "gamma_kkt": 0.1,
        "solver_steps": 10,
        "gaussian_mode": "separable",
    },
    
    # Small model (faster training)
    "offm_small": {
        "projection_type": "fourier",
        "projection_dim": 64,
        "icnn_hidden_dims": [64, 64],
        "lambda_reg": 0.2,
        "gamma_kkt": 0.1,
        "solver_steps": 5,
        "gaussian_mode": "independent",
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


def train_single_config(
    config_name: str,
    config: Dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
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
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Train
    t0 = time.time()
    history = model.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        scheduler=scheduler,
        test_loader=test_loader,
        eval_int=10,
        save_int=args.epochs,
        save_path=save_dir,
        verbose=True,
    )
    train_time = time.time() - t0
    
    # Generate samples
    print(f"    Generating {args.n_gen} samples...")
    samples = model.sample(n_samples=args.n_gen).cpu()
    
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
    
    # Save results
    torch.save(samples, save_dir / 'samples.pt')
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
    """Compute quality metrics for trajectory generation.
    
    Args:
        ground_truth: (n_samples, n_t, n_x) ground truth trajectories
        generated: (n_gen, n_t, n_x) generated trajectories
        config_name: Name of configuration
        
    Returns:
        Dictionary of metrics
    """
    metrics = {'config_name': config_name}
    
    # Basic statistics comparison
    gt_mean = ground_truth.mean(dim=0)  # (n_t, n_x)
    gen_mean = generated.mean(dim=0)
    metrics['mean_mse'] = ((gt_mean - gen_mean) ** 2).mean().item()
    
    gt_std = ground_truth.std(dim=0)
    gen_std = generated.std(dim=0)
    metrics['std_mse'] = ((gt_std - gen_std) ** 2).mean().item()
    
    # Spatial spectrum comparison (averaged over time)
    gt_spectrum = torch.fft.rfft(ground_truth, dim=-1).abs().mean(dim=(0, 1))
    gen_spectrum = torch.fft.rfft(generated, dim=-1).abs().mean(dim=(0, 1))
    metrics['spectrum_mse'] = ((gt_spectrum - gen_spectrum) ** 2).mean().item()
    
    # Temporal correlation
    gt_temp_corr = compute_temporal_correlation(ground_truth)
    gen_temp_corr = compute_temporal_correlation(generated)
    metrics['temporal_corr_mse'] = ((gt_temp_corr - gen_temp_corr) ** 2).mean().item()
    
    # Energy conservation (PDE-specific)
    gt_energy = (ground_truth ** 2).mean(dim=-1)  # (n_samples, n_t)
    gen_energy = (generated ** 2).mean(dim=-1)
    metrics['energy_mean_mse'] = ((gt_energy.mean(dim=0) - gen_energy.mean(dim=0)) ** 2).mean().item()
    
    # Trajectory smoothness (time derivative magnitude)
    gt_dt = torch.diff(ground_truth, dim=1)
    gen_dt = torch.diff(generated, dim=1)
    metrics['smoothness_mse'] = ((gt_dt.std() - gen_dt.std()) ** 2).item()
    
    return metrics


def compute_temporal_correlation(x: torch.Tensor, max_lag: int = 10) -> torch.Tensor:
    """Compute autocorrelation in time dimension.
    
    Args:
        x: (batch, n_t, n_x) tensor
        max_lag: Maximum lag to compute
        
    Returns:
        (max_lag,) tensor of autocorrelation values
    """
    # Compute correlation at each lag
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
    
    plt.suptitle(f'Stochastic KdV Trajectories: {config_name}', fontsize=14)
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
    
    # Spatial spectrum (averaged over time and batch)
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
    gt_corr = compute_temporal_correlation(ground_truth, max_lag=20)
    lags = torch.arange(1, len(gt_corr) + 1)
    
    ax.plot(lags, gt_corr, 'k-', linewidth=2, label='Ground Truth')
    
    for name, samples in generated_dict.items():
        corr = compute_temporal_correlation(samples, max_lag=20)
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
    
    # Load data (trajectory mode for OFFM)
    print("\nLoading stochastic KdV trajectory data...")
    data = load_stochastic_kdv(args.dpath, shuffle=True, mode='trajectory',
                                subsample_time=args.subsample_time)
    print(f"Data shape: {data.shape}")  # Should be (1200, 101, 128)
    
    # Get dimensions
    n_total, n_t, n_x = data.shape
    print(f"Trajectories: {n_total}, Time steps: {n_t}, Spatial points: {n_x}")
    
    # Split into train/test
    args.ntr = min(args.ntr, int(0.8 * n_total))
    args.nte = min(args.nte, n_total - args.ntr)
    
    train_data = data[:args.ntr]
    test_data = data[args.ntr:args.ntr + args.nte]
    ground_truth = train_data.clone()
    
    print(f"Training trajectories: {train_data.shape[0]}")
    print(f"Test trajectories: {test_data.shape[0]}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False)
    
    # Output directory
    spath = Path(args.spath)
    spath.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = vars(args)
    config_dict['n_t'] = n_t
    config_dict['n_x'] = n_x
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
    print(f"OFFM Experiments on Stochastic KdV Trajectories")
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
        
        for seed in seeds_to_run:
            config_dir = spath / config_name / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            result = train_single_config(
                config_name=config_name,
                config=config,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
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
    
    # Plot sample trajectories for each config
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
        'dataset': 'Stochastic KdV (Trajectory)',
        'n_train': args.ntr,
        'n_t': n_t,
        'n_x': n_x,
        'epochs': args.epochs,
        'batch_size': args.bs,
        'n_seeds': len(seeds_to_run),
        'configs': {},
    }
    
    for config_name, config_results in all_results.items():
        # Average metrics across seeds
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
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Config':<25} {'Mean MSE':>12} {'Spectrum MSE':>12} {'Temp Corr':>12} {'Train Time':>12}")
    print("-" * 75)
    
    for config_name, metrics in summary['configs'].items():
        mean_mse = metrics.get('mean_mse', float('nan'))
        spec_mse = metrics.get('spectrum_mse', float('nan'))
        temp_corr = metrics.get('temporal_corr_mse', float('nan'))
        train_time = metrics.get('train_time', float('nan'))
        print(f"{config_name:<25} {mean_mse:>12.6f} {spec_mse:>12.6f} {temp_corr:>12.6f} {train_time:>10.1f}s")
    
    print("\n" + "=" * 60)
    print(f"Complete! Results saved to: {spath}")
    print("=" * 60)


if __name__ == '__main__':
    main()
