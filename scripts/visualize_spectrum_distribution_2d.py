#!/usr/bin/env python
"""
Visualization of Power Spectrum Distributions for 2D PDE Datasets.

This script visualizes and compares the distribution of power spectra between
generated samples and ground truth for 2D PDE data (Navier-Stokes, etc.).

Key visualizations:
1. Mean spectrum with quantile bands (showing spread across samples)
2. Violin plots at selected wavenumbers (showing full distribution shape)
3. Distribution divergence metrics (KL divergence, Wasserstein distance)
4. Per-wavenumber statistical tests

Usage:
    # Single config visualization
    python visualize_spectrum_distribution_2d.py --data-dir ../outputs/stochastic_ns_ot/

    # Compare multiple configs
    python visualize_spectrum_distribution_2d.py --data-dir ../outputs/stochastic_ns_ot/ \
        --configs euclidean_sinkhorn rbf_sinkhorn_reg0.1

    # Custom output
    python visualize_spectrum_distribution_2d.py --data-dir ../outputs/navier_stokes_ot/ \
        --output spectrum_analysis.pdf
"""

import sys
sys.path.append('../')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from scipy.ndimage import gaussian_filter1d
import argparse
import json


# =============================================================================
# Spectrum Computation (per-sample)
# =============================================================================

def compute_energy_spectrum_per_sample(
    samples: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D energy spectrum for each sample individually.
    
    Parameters
    ----------
    samples : Tensor, shape (n_samples, H, W) or (n_samples, C, H, W)
        2D field samples. If 4D, uses first channel.
        
    Returns
    -------
    wavenumbers : ndarray, shape (n_wavenumbers,)
        Wavenumber values
    spectra : ndarray, shape (n_samples, n_wavenumbers)
        Energy spectrum for each sample
    """
    if samples.ndim == 4:
        samples = samples[:, 0]  # Take first channel
    
    n_samples, s, _ = samples.shape
    assert samples.shape[1] == samples.shape[2], "Expected square fields"
    
    # FFT
    u_fft = torch.fft.fft2(samples)
    
    # Wavenumbers
    k_max = s // 2
    wavenumbers_1d = torch.cat([
        torch.arange(0, k_max),
        torch.arange(-k_max, 0)
    ])
    k_x = wavenumbers_1d.unsqueeze(0).repeat(s, 1).T
    k_y = wavenumbers_1d.unsqueeze(0).repeat(s, 1)
    
    # Sum of absolute wavenumbers
    sum_k = (torch.abs(k_x) + torch.abs(k_y)).numpy()
    
    # Remove symmetric components
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]
    
    # Bin by wavenumber
    spectrum_arr = np.zeros((n_samples, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        if len(ind[0]) > 0:
            spectrum_arr[:, j - 1] = np.sqrt(
                np.abs(u_fft[:, ind[0], ind[1]].sum(axis=1).numpy()) ** 2
            )
    
    # Only keep up to Nyquist
    spectrum_arr = spectrum_arr[:, :s // 2]
    wavenumbers = np.arange(1, s // 2 + 1)
    
    return wavenumbers, spectrum_arr


# =============================================================================
# Distribution Metrics
# =============================================================================

def compute_spectrum_distribution_metrics(
    spec_real: np.ndarray,
    spec_gen: np.ndarray,
    log_scale: bool = True,
) -> Dict[str, float]:
    """
    Compute distribution metrics comparing real and generated spectra.
    
    Parameters
    ----------
    spec_real : ndarray, shape (n_real, n_wavenumbers)
        Per-sample spectra for real data
    spec_gen : ndarray, shape (n_gen, n_wavenumbers)
        Per-sample spectra for generated data
    log_scale : bool
        Whether to compute metrics in log scale
        
    Returns
    -------
    metrics : dict
        Dictionary containing various distribution metrics
    """
    eps = 1e-10
    
    if log_scale:
        spec_real = np.log10(spec_real + eps)
        spec_gen = np.log10(spec_gen + eps)
    
    n_wavenumbers = spec_real.shape[1]
    
    # Per-wavenumber Wasserstein distances
    wasserstein_dists = []
    ks_stats = []
    ks_pvals = []
    
    for k in range(n_wavenumbers):
        w_dist = wasserstein_distance(spec_real[:, k], spec_gen[:, k])
        wasserstein_dists.append(w_dist)
        
        ks_stat, ks_pval = ks_2samp(spec_real[:, k], spec_gen[:, k])
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_pval)
    
    # Summary statistics
    mean_real = spec_real.mean(axis=0)
    mean_gen = spec_gen.mean(axis=0)
    std_real = spec_real.std(axis=0)
    std_gen = spec_gen.std(axis=0)
    
    # Mean-level MSE
    mean_mse = np.mean((mean_real - mean_gen) ** 2)
    
    # Variance-level MSE  
    var_mse = np.mean((std_real**2 - std_gen**2) ** 2)
    
    # Average Wasserstein distance
    avg_wasserstein = np.mean(wasserstein_dists)
    
    # Fraction of wavenumbers where distributions significantly differ (p < 0.05)
    frac_significant = np.mean(np.array(ks_pvals) < 0.05)
    
    return {
        'mean_mse': mean_mse,
        'var_mse': var_mse,
        'avg_wasserstein': avg_wasserstein,
        'wasserstein_per_k': np.array(wasserstein_dists),
        'ks_stats': np.array(ks_stats),
        'ks_pvals': np.array(ks_pvals),
        'frac_significantly_different': frac_significant,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_spectrum_with_quantile_bands(
    wavenumbers: np.ndarray,
    spec_real: np.ndarray,
    spec_gen: np.ndarray,
    config_name: str = "Generated",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 7),
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    log_scale: bool = True,
    smooth_sigma: float = 0.0,
) -> plt.Figure:
    """
    Plot spectrum comparison with quantile bands showing distribution spread.
    
    Parameters
    ----------
    wavenumbers : ndarray
        Wavenumber values
    spec_real : ndarray, shape (n_real, n_wavenumbers)
        Per-sample spectra for real data
    spec_gen : ndarray, shape (n_gen, n_wavenumbers)
        Per-sample spectra for generated data
    config_name : str
        Name of the generated configuration
    save_path : Path, optional
        Where to save the figure
    figsize : tuple
        Figure size
    quantiles : list
        Quantiles to show as bands
    log_scale : bool
        Whether to plot in log scale
    smooth_sigma : float
        Gaussian smoothing sigma for bands (0 = no smoothing)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    eps = 1e-10
    
    # Compute quantiles for real data
    real_quantiles = np.percentile(spec_real, [q * 100 for q in quantiles], axis=0)
    gen_quantiles = np.percentile(spec_gen, [q * 100 for q in quantiles], axis=0)
    
    if smooth_sigma > 0:
        for i in range(len(quantiles)):
            real_quantiles[i] = gaussian_filter1d(real_quantiles[i], smooth_sigma)
            gen_quantiles[i] = gaussian_filter1d(gen_quantiles[i], smooth_sigma)
    
    # Convert to log scale if needed
    if log_scale:
        real_quantiles = np.log10(real_quantiles + eps)
        gen_quantiles = np.log10(gen_quantiles + eps)
        ylabel = r'$\log_{10}$ Energy E(k)'
    else:
        ylabel = 'Energy E(k)'
    
    # Color schemes
    real_color = '#2E86AB'  # Blue
    gen_color = '#E94F37'   # Red
    
    # Plot real data bands (outer to inner)
    alpha_values = [0.15, 0.25, 0.35]
    for i, (q_low, q_high) in enumerate([(0, 4), (1, 3)]):  # 10-90, 25-75
        ax.fill_between(
            wavenumbers,
            real_quantiles[q_low],
            real_quantiles[q_high],
            alpha=alpha_values[i],
            color=real_color,
            linewidth=0,
        )
    
    # Plot generated data bands
    for i, (q_low, q_high) in enumerate([(0, 4), (1, 3)]):
        ax.fill_between(
            wavenumbers,
            gen_quantiles[q_low],
            gen_quantiles[q_high],
            alpha=alpha_values[i],
            color=gen_color,
            linewidth=0,
        )
    
    # Plot medians
    ax.plot(wavenumbers, real_quantiles[2], '-', color=real_color, 
            linewidth=2.5, label='Ground Truth (median)', zorder=5)
    ax.plot(wavenumbers, gen_quantiles[2], '--', color=gen_color, 
            linewidth=2.5, label=f'{config_name} (median)', zorder=5)
    
    # Plot means for reference
    real_mean = np.log10(spec_real.mean(axis=0) + eps) if log_scale else spec_real.mean(axis=0)
    gen_mean = np.log10(spec_gen.mean(axis=0) + eps) if log_scale else spec_gen.mean(axis=0)
    ax.plot(wavenumbers, real_mean, ':', color=real_color, linewidth=1.5, 
            label='Ground Truth (mean)', alpha=0.7)
    ax.plot(wavenumbers, gen_mean, ':', color=gen_color, linewidth=1.5, 
            label=f'{config_name} (mean)', alpha=0.7)
    
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Energy Spectrum Distribution Comparison', fontsize=14, fontweight='bold')
    
    # Custom legend with band explanation
    from matplotlib.patches import Patch
    handles = [
        plt.Line2D([0], [0], color=real_color, linewidth=2.5, linestyle='-'),
        plt.Line2D([0], [0], color=gen_color, linewidth=2.5, linestyle='--'),
        Patch(facecolor=real_color, alpha=0.25, label='GT 25-75%'),
        Patch(facecolor=gen_color, alpha=0.25, label='Gen 25-75%'),
        Patch(facecolor=real_color, alpha=0.15, label='GT 10-90%'),
        Patch(facecolor=gen_color, alpha=0.15, label='Gen 10-90%'),
    ]
    labels = [
        'Ground Truth (median)',
        f'{config_name} (median)',
        'Ground Truth 25-75%',
        f'{config_name} 25-75%',
        'Ground Truth 10-90%',
        f'{config_name} 10-90%',
    ]
    ax.legend(handles, labels, loc='upper right', fontsize=9, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(wavenumbers[0], wavenumbers[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved quantile band plot to {save_path}")
    
    return fig


def plot_spectrum_violin_comparison(
    wavenumbers: np.ndarray,
    spec_real: np.ndarray,
    spec_gen: np.ndarray,
    config_name: str = "Generated",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
    n_wavenumbers: int = 8,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot violin plot comparison at selected wavenumbers.
    
    Shows the full distribution shape at each selected wavenumber.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    eps = 1e-10
    
    # Select wavenumbers to plot (evenly spaced)
    n_total = len(wavenumbers)
    indices = np.linspace(0, n_total - 1, n_wavenumbers, dtype=int)
    selected_k = wavenumbers[indices]
    
    # Prepare data
    if log_scale:
        spec_real = np.log10(spec_real + eps)
        spec_gen = np.log10(spec_gen + eps)
    
    # Create violin data
    positions_real = np.arange(n_wavenumbers) * 2 - 0.35
    positions_gen = np.arange(n_wavenumbers) * 2 + 0.35
    
    real_color = '#2E86AB'
    gen_color = '#E94F37'
    
    # Plot violins for real data
    parts_real = ax.violinplot(
        [spec_real[:, i] for i in indices],
        positions=positions_real,
        widths=0.6,
        showmeans=True,
        showmedians=True,
    )
    for pc in parts_real['bodies']:
        pc.set_facecolor(real_color)
        pc.set_alpha(0.6)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts_real:
            parts_real[partname].set_color(real_color)
    
    # Plot violins for generated data
    parts_gen = ax.violinplot(
        [spec_gen[:, i] for i in indices],
        positions=positions_gen,
        widths=0.6,
        showmeans=True,
        showmedians=True,
    )
    for pc in parts_gen['bodies']:
        pc.set_facecolor(gen_color)
        pc.set_alpha(0.6)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts_gen:
            parts_gen[partname].set_color(gen_color)
    
    # X-axis labels
    ax.set_xticks(np.arange(n_wavenumbers) * 2)
    ax.set_xticklabels([f'k={k}' for k in selected_k], fontsize=10)
    
    ylabel = r'$\log_{10}$ Energy E(k)' if log_scale else 'Energy E(k)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Wavenumber', fontsize=12)
    ax.set_title('Spectrum Distribution at Selected Wavenumbers', fontsize=14, fontweight='bold')
    
    # Legend
    handles = [
        Patch(facecolor=real_color, alpha=0.6, label='Ground Truth'),
        Patch(facecolor=gen_color, alpha=0.6, label=config_name),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=11)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved violin plot to {save_path}")
    
    return fig


def plot_wasserstein_profile(
    wavenumbers: np.ndarray,
    metrics: Dict[str, Union[float, np.ndarray]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot Wasserstein distance profile across wavenumbers.
    
    Shows where the generated distribution deviates most from ground truth.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    w_dist = metrics['wasserstein_per_k']
    ks_pvals = metrics['ks_pvals']
    
    # Wasserstein distance profile
    ax1.fill_between(wavenumbers, 0, w_dist, alpha=0.4, color='#9B59B6')
    ax1.plot(wavenumbers, w_dist, '-', color='#9B59B6', linewidth=2)
    ax1.axhline(metrics['avg_wasserstein'], color='red', linestyle='--', 
                linewidth=1.5, label=f"Mean: {metrics['avg_wasserstein']:.4f}")
    ax1.set_xlabel('Wavenumber k', fontsize=11)
    ax1.set_ylabel('Wasserstein Distance', fontsize=11)
    ax1.set_title('Distribution Divergence by Wavenumber', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(wavenumbers[0], wavenumbers[-1])
    
    # KS test p-value profile
    significance_threshold = 0.05
    ax2.semilogy(wavenumbers, ks_pvals, '-', color='#3498DB', linewidth=2)
    ax2.axhline(significance_threshold, color='red', linestyle='--', 
                linewidth=1.5, label=f'α = {significance_threshold}')
    ax2.fill_between(
        wavenumbers, 0, ks_pvals, 
        where=(ks_pvals < significance_threshold),
        alpha=0.3, color='#E74C3C', label='Significant difference'
    )
    ax2.set_xlabel('Wavenumber k', fontsize=11)
    ax2.set_ylabel('KS Test p-value', fontsize=11)
    ax2.set_title('Statistical Significance of Differences', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(wavenumbers[0], wavenumbers[-1])
    ax2.set_ylim(1e-10, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved Wasserstein profile to {save_path}")
    
    return fig


def plot_multi_config_spectrum_comparison(
    wavenumbers: np.ndarray,
    spec_real: np.ndarray,
    generated_specs: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    log_scale: bool = True,
    top_k: int = 5,
) -> plt.Figure:
    """
    Compare spectrum distributions across multiple configurations.
    
    Shows mean + std bands for each config, ranked by Wasserstein distance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    eps = 1e-10
    
    # Compute metrics for each config
    config_metrics = {}
    for name, spec_gen in generated_specs.items():
        metrics = compute_spectrum_distribution_metrics(spec_real, spec_gen, log_scale=log_scale)
        config_metrics[name] = metrics
    
    # Sort by average Wasserstein distance
    sorted_configs = sorted(
        config_metrics.keys(),
        key=lambda x: config_metrics[x]['avg_wasserstein']
    )
    
    # Select top-k
    top_configs = sorted_configs[:top_k]
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_configs)))
    
    # Convert real spectrum
    if log_scale:
        spec_real_plot = np.log10(spec_real + eps)
        ylabel = r'$\log_{10}$ Energy E(k)'
    else:
        spec_real_plot = spec_real
        ylabel = 'Energy E(k)'
    
    real_mean = spec_real_plot.mean(axis=0)
    real_std = spec_real_plot.std(axis=0)
    
    # Plot real data with band
    ax1.fill_between(wavenumbers, real_mean - real_std, real_mean + real_std,
                     alpha=0.3, color='black')
    ax1.plot(wavenumbers, real_mean, 'k-', linewidth=2.5, label='Ground Truth')
    
    # Plot each config
    for i, name in enumerate(top_configs):
        spec_gen = generated_specs[name]
        if log_scale:
            spec_gen_plot = np.log10(spec_gen + eps)
        else:
            spec_gen_plot = spec_gen
        
        gen_mean = spec_gen_plot.mean(axis=0)
        gen_std = spec_gen_plot.std(axis=0)
        
        w_dist = config_metrics[name]['avg_wasserstein']
        
        ax1.fill_between(wavenumbers, gen_mean - gen_std, gen_mean + gen_std,
                         alpha=0.15, color=colors[i])
        ax1.plot(wavenumbers, gen_mean, '--', color=colors[i], linewidth=2,
                 label=f'{name} (W={w_dist:.4f})')
    
    ax1.set_xlabel('Wavenumber k', fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title('Spectrum Comparison (Mean ± Std)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(wavenumbers[0], wavenumbers[-1])
    
    # Bar chart of metrics
    config_names_short = [n[:20] + '...' if len(n) > 20 else n for n in top_configs]
    wasserstein_vals = [config_metrics[n]['avg_wasserstein'] for n in top_configs]
    
    bars = ax2.barh(range(len(top_configs)), wasserstein_vals, color=colors)
    ax2.set_yticks(range(len(top_configs)))
    ax2.set_yticklabels(config_names_short, fontsize=9)
    ax2.set_xlabel('Average Wasserstein Distance', fontsize=11)
    ax2.set_title(f'Top {len(top_configs)} Configurations', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()  # Best at top
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, wasserstein_vals)):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Saved multi-config comparison to {save_path}")
    
    return fig


def create_comprehensive_spectrum_report(
    ground_truth: torch.Tensor,
    generated_samples: Dict[str, torch.Tensor],
    output_dir: Path,
    dataset_name: str = "2D_PDE",
) -> Dict[str, Dict[str, float]]:
    """
    Create a comprehensive spectrum analysis report.
    
    Parameters
    ----------
    ground_truth : Tensor
        Ground truth samples
    generated_samples : dict
        Dictionary mapping config names to generated sample tensors
    output_dir : Path
        Directory to save plots and metrics
    dataset_name : str
        Name of the dataset for titles
        
    Returns
    -------
    all_metrics : dict
        Dictionary of metrics for each config
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute ground truth spectrum
    print("Computing ground truth spectrum...")
    wavenumbers, spec_real = compute_energy_spectrum_per_sample(ground_truth)
    
    # Compute spectra for all configs
    all_specs = {}
    all_metrics = {}
    
    for name, samples in generated_samples.items():
        print(f"Computing spectrum for {name}...")
        _, spec_gen = compute_energy_spectrum_per_sample(samples)
        all_specs[name] = spec_gen
        
        # Compute metrics
        metrics = compute_spectrum_distribution_metrics(spec_real, spec_gen)
        all_metrics[name] = {
            'mean_mse': float(metrics['mean_mse']),
            'var_mse': float(metrics['var_mse']),
            'avg_wasserstein': float(metrics['avg_wasserstein']),
            'frac_significantly_different': float(metrics['frac_significantly_different']),
        }
    
    # Save metrics to JSON
    metrics_path = output_dir / 'spectrum_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Generate individual plots for each config
    for name, spec_gen in all_specs.items():
        safe_name = name.replace('/', '_').replace(' ', '_')
        
        # Quantile band plot
        plot_spectrum_with_quantile_bands(
            wavenumbers, spec_real, spec_gen,
            config_name=name,
            save_path=output_dir / f'{safe_name}_quantile_bands.pdf',
        )
        plt.close()
        
        # Violin plot
        plot_spectrum_violin_comparison(
            wavenumbers, spec_real, spec_gen,
            config_name=name,
            save_path=output_dir / f'{safe_name}_violin.pdf',
        )
        plt.close()
        
        # Wasserstein profile
        metrics = compute_spectrum_distribution_metrics(spec_real, spec_gen)
        plot_wasserstein_profile(
            wavenumbers, metrics,
            save_path=output_dir / f'{safe_name}_wasserstein.pdf',
        )
        plt.close()
    
    # Multi-config comparison
    if len(all_specs) > 1:
        plot_multi_config_spectrum_comparison(
            wavenumbers, spec_real, all_specs,
            save_path=output_dir / 'multi_config_comparison.pdf',
        )
        plt.close()
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"SPECTRUM DISTRIBUTION ANALYSIS - {dataset_name}")
    print("=" * 80)
    print(f"{'Config':<40} {'W-Dist':>12} {'Mean MSE':>12} {'% Sig Diff':>12}")
    print("-" * 80)
    
    # Sort by Wasserstein distance
    sorted_configs = sorted(all_metrics.keys(), key=lambda x: all_metrics[x]['avg_wasserstein'])
    for name in sorted_configs:
        m = all_metrics[name]
        print(f"{name[:40]:<40} {m['avg_wasserstein']:>12.6f} {m['mean_mse']:>12.6f} {m['frac_significantly_different']*100:>11.1f}%")
    
    print("=" * 80)
    
    return all_metrics


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_samples_from_directory(
    data_dir: Path,
    config_names: Optional[List[str]] = None,
    ground_truth_filename: str = 'ground_truth.pt',
    regenerate_from_model: bool = False,
    n_samples: int = 100,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load ground truth and generated samples from a directory structure.
    
    Expected structure:
    data_dir/
        ground_truth.pt
        config_name_1/
            samples.pt or seed_1/samples.pt
        config_name_2/
            ...
            
    Parameters
    ----------
    data_dir : Path
        Root directory containing experiment outputs
    config_names : list, optional
        Specific configs to load. If None, discovers all config directories.
    ground_truth_filename : str
        Name of ground truth file
    regenerate_from_model : bool
        If True and samples not found, try to generate from saved model
    n_samples : int
        Number of samples to generate if regenerating
    device : str
        Device to use for generation
    """
    data_dir = Path(data_dir)
    
    # Load ground truth
    gt_path = data_dir / ground_truth_filename
    if not gt_path.exists():
        # Try alternative names
        for alt_name in ['ground_truth.pt', 'ground_truth_rescaled.pt', 'gt.pt']:
            alt_path = data_dir / alt_name
            if alt_path.exists():
                gt_path = alt_path
                break
    
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found in {data_dir}")
    
    ground_truth = torch.load(gt_path, map_location='cpu')
    print(f"Loaded ground truth from {gt_path}: shape {ground_truth.shape}")
    
    # Find config directories (excluding known non-config directories)
    exclude_dirs = {'sample_comparisons', '__pycache__', '.git'}
    
    if config_names is None:
        config_names = []
        for item in data_dir.iterdir():
            if item.is_dir() and item.name not in exclude_dirs and not item.name.startswith('.'):
                # Check if it looks like a config directory (has seed subdirs or sample files)
                has_seeds = any((item / f'seed_{i}').exists() for i in [1, 2, 4, 8])
                has_samples = any(item.glob('*.pt')) or any(item.glob('**/samples.pt'))
                has_model = any(item.glob('**/model.pt'))
                if has_seeds or has_samples or has_model:
                    config_names.append(item.name)
    
    if not config_names:
        print(f"No config directories found in {data_dir}")
        print("Looking for directories with seed_* subdirs or .pt files...")
    
    # Load generated samples
    generated_samples = {}
    
    for config_name in sorted(config_names):
        config_dir = data_dir / config_name
        if not config_dir.exists():
            print(f"Warning: Config directory not found: {config_dir}")
            continue
        
        # Try different sample file patterns
        sample_paths = []
        
        # Pattern 1: Direct samples.pt in config dir
        if (config_dir / 'samples.pt').exists():
            sample_paths.append(config_dir / 'samples.pt')
        
        # Pattern 2: samples.pt in seed directories
        for seed_dir in sorted(config_dir.glob('seed_*')):
            if (seed_dir / 'samples.pt').exists():
                sample_paths.append(seed_dir / 'samples.pt')
        
        # Pattern 3: generated_samples*.pt anywhere
        sample_paths.extend(config_dir.glob('**/generated_samples*.pt'))
        
        if sample_paths:
            # Load and concatenate all sample files
            all_samples = []
            for sp in sample_paths:
                try:
                    samples = torch.load(sp, map_location='cpu')
                    # Handle different tensor formats
                    if samples.ndim == 4 and samples.shape[1] == 1:
                        samples = samples.squeeze(1)  # Remove channel dim
                    all_samples.append(samples)
                except Exception as e:
                    print(f"  Warning: Could not load {sp}: {e}")
            
            if all_samples:
                generated_samples[config_name] = torch.cat(all_samples, dim=0)
                print(f"  Loaded {config_name}: shape {generated_samples[config_name].shape}")
        elif regenerate_from_model:
            # Try to regenerate from saved model
            samples = regenerate_samples_from_model(config_dir, n_samples, device)
            if samples is not None:
                generated_samples[config_name] = samples
                print(f"  Generated {config_name}: shape {samples.shape}")
        else:
            print(f"  Warning: No samples found for {config_name}")
    
    return ground_truth, generated_samples


def regenerate_samples_from_model(
    config_dir: Path,
    n_samples: int = 100,
    device: str = 'cuda',
) -> Optional[torch.Tensor]:
    """
    Regenerate samples from a saved model checkpoint.
    
    Looks for model.pt files in seed directories and generates samples.
    """
    try:
        # Import required modules
        from functional_fm_ot import FFMModelOT
        from models.fno import FNO
        
        # Find a model checkpoint
        model_paths = list(config_dir.glob('**/model.pt'))
        if not model_paths:
            return None
        
        model_path = model_paths[0]  # Use first available
        seed_dir = model_path.parent
        
        # Load config to get model parameters
        config_path = seed_dir / 'config.json'
        if not config_path.exists():
            print(f"  No config.json found for {config_dir.name}")
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Infer model architecture from config or use defaults
        # This is a simplified version - may need adjustment for specific experiments
        print(f"  Regenerating samples from {model_path}...")
        
        # Load state dict and infer dimensions
        state_dict = torch.load(model_path, map_location=device)
        
        # For 2D PDE data, we typically use FNO
        # Try to infer dimensions from state dict
        # This is a heuristic - may need adjustment
        
        # For now, just return None and warn the user
        print(f"  Model regeneration requires manual configuration - skipping {config_dir.name}")
        print(f"  To generate samples, run the experiment script with --generate-only flag")
        return None
        
    except Exception as e:
        print(f"  Could not regenerate samples: {e}")
        return None


def generate_samples_script(
    experiment_type: str = 'stochastic_ns',
    data_dir: str = '../outputs/stochastic_ns_ot/',
    configs: Optional[List[str]] = None,
    n_samples: int = 100,
):
    """
    Generate a shell script to regenerate samples for all configs.
    
    This creates commands to run the experiment scripts with --generate-only mode.
    """
    script_map = {
        'navier_stokes': 'navier_stokes_ot.py',
        'stochastic_ns': 'stochastic_ns_ot.py',
        'ginzburg_landau': 'ginzburg_landau_ot.py',
    }
    
    if experiment_type not in script_map:
        print(f"Unknown experiment type: {experiment_type}")
        print(f"Available: {list(script_map.keys())}")
        return
    
    data_path = Path(data_dir)
    
    if configs is None:
        # Discover configs
        configs = []
        for item in data_path.iterdir():
            if item.is_dir() and any(item.glob('**/model.pt')):
                configs.append(item.name)
    
    print(f"# Script to regenerate samples for {experiment_type}")
    print(f"# Configs: {len(configs)}")
    print()
    
    script_name = script_map[experiment_type]
    for config in sorted(configs):
        print(f"python {script_name} --config {config} --generate-only --n-gen {n_samples}")


# =============================================================================
# Quick Analysis Mode (Direct File Input)
# =============================================================================

def quick_spectrum_analysis(
    ground_truth_path: str,
    generated_path: str,
    output_path: Optional[str] = None,
    config_name: str = "Generated",
    show_plots: bool = True,
) -> Dict[str, float]:
    """
    Quick spectrum distribution analysis between two .pt files.
    
    This is a simple interface for comparing a single generated sample
    against ground truth without the full directory structure.
    
    Parameters
    ----------
    ground_truth_path : str
        Path to ground truth .pt file
    generated_path : str
        Path to generated samples .pt file
    output_path : str, optional
        Base path for saving plots (without extension)
    config_name : str
        Name for the generated samples in plots
    show_plots : bool
        Whether to display plots interactively
        
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics
    """
    # Load data
    ground_truth = torch.load(ground_truth_path, map_location='cpu')
    generated = torch.load(generated_path, map_location='cpu')
    
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Generated shape: {generated.shape}")
    
    # Handle channel dimension
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[:, 0]
    if generated.ndim == 4:
        generated = generated[:, 0]
    
    # Compute spectra
    wavenumbers, spec_real = compute_energy_spectrum_per_sample(ground_truth)
    _, spec_gen = compute_energy_spectrum_per_sample(generated)
    
    # Compute metrics
    metrics = compute_spectrum_distribution_metrics(spec_real, spec_gen)
    
    print("\n" + "=" * 50)
    print("SPECTRUM DISTRIBUTION METRICS")
    print("=" * 50)
    print(f"Mean MSE (log):               {metrics['mean_mse']:.6f}")
    print(f"Variance MSE:                 {metrics['var_mse']:.6f}")
    print(f"Avg Wasserstein Distance:     {metrics['avg_wasserstein']:.6f}")
    print(f"Frac Significantly Different: {metrics['frac_significantly_different']*100:.1f}%")
    print("=" * 50)
    
    # Create plots
    save_base = Path(output_path) if output_path else None
    
    # 1. Quantile band plot
    fig1 = plot_spectrum_with_quantile_bands(
        wavenumbers, spec_real, spec_gen,
        config_name=config_name,
        save_path=Path(f"{save_base}_quantile_bands.pdf") if save_base else None,
    )
    
    # 2. Violin plot
    fig2 = plot_spectrum_violin_comparison(
        wavenumbers, spec_real, spec_gen,
        config_name=config_name,
        save_path=Path(f"{save_base}_violin.pdf") if save_base else None,
    )
    
    # 3. Wasserstein profile
    fig3 = plot_wasserstein_profile(
        wavenumbers, metrics,
        save_path=Path(f"{save_base}_wasserstein.pdf") if save_base else None,
    )
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    return {
        'mean_mse': float(metrics['mean_mse']),
        'var_mse': float(metrics['var_mse']),
        'avg_wasserstein': float(metrics['avg_wasserstein']),
        'frac_significantly_different': float(metrics['frac_significantly_different']),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize power spectrum distributions for 2D PDE datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis with two files:
  python visualize_spectrum_distribution_2d.py --quick \\
      --gt ground_truth.pt --gen generated_samples.pt

  # Full analysis from experiment directory:
  python visualize_spectrum_distribution_2d.py \\
      --data-dir ../outputs/stochastic_ns_ot/

  # Compare specific configs:
  python visualize_spectrum_distribution_2d.py \\
      --data-dir ../outputs/navier_stokes_ot/ \\
      --configs euclidean_sinkhorn_reg0.1 rbf_sinkhorn_reg0.5
"""
    )
    
    # Mode selection
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: compare two .pt files directly')
    
    # Quick mode arguments
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth .pt file (for --quick mode)')
    parser.add_argument('--gen', type=str, default=None,
                        help='Generated samples .pt file (for --quick mode)')
    
    # Full mode arguments
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing ground truth and generated samples')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Specific config names to analyze (default: all)')
    parser.add_argument('--gt-file', type=str, default='ground_truth.pt',
                        help='Ground truth filename within data-dir')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output base path for quick mode (without extension)')
    parser.add_argument('--dataset-name', type=str, default='2D_PDE',
                        help='Dataset name for plot titles')
    parser.add_argument('--name', type=str, default='Generated',
                        help='Name for generated samples in plots (quick mode)')
    parser.add_argument('--no-show', action='store_true',
                        help="Don't display plots interactively")
    parser.add_argument('--regenerate', action='store_true',
                        help='Try to regenerate samples from saved models if not found')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        if not args.gt or not args.gen:
            parser.error("--quick mode requires --gt and --gen arguments")
        
        quick_spectrum_analysis(
            args.gt,
            args.gen,
            output_path=args.output,
            config_name=args.name,
            show_plots=not args.no_show,
        )
        return
    
    # Full mode
    if not args.data_dir:
        parser.error("Either --quick mode or --data-dir is required")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'spectrum_analysis'
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading samples...")
    print("=" * 60)
    
    ground_truth, generated_samples = load_samples_from_directory(
        data_dir,
        config_names=args.configs,
        ground_truth_filename=args.gt_file,
        regenerate_from_model=args.regenerate,
    )
    
    if not generated_samples:
        print("\n" + "=" * 60)
        print("NO GENERATED SAMPLES FOUND!")
        print("=" * 60)
        print("\nPossible solutions:")
        print("1. Run experiments with sample saving enabled")
        print("2. Use --quick mode with direct .pt file paths:")
        print(f"   python {sys.argv[0]} --quick --gt ground_truth.pt --gen samples.pt")
        print("3. Manually generate samples from saved models")
        print("\nAvailable directories in", data_dir)
        for item in sorted(data_dir.iterdir()):
            if item.is_dir():
                pt_files = list(item.glob('**/*.pt'))
                print(f"  {item.name}/  ({len(pt_files)} .pt files)")
        return
    
    # Create comprehensive report
    create_comprehensive_spectrum_report(
        ground_truth,
        generated_samples,
        output_dir,
        dataset_name=args.dataset_name,
    )
    
    print(f"\nAll plots saved to: {output_dir}")
    
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
