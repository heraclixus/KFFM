"""
Plot hyperparameter sensitivity using violin plots.

For each dataset, creates:
1. Violin plot showing distribution of mean_mse across kernel types
2. Scatter plots showing OT regularization (epsilon) vs performance
3. Violin plots for categorical OT parameters (coupling, method)
4. Kernel-specific parameter sensitivity (RBF sigma, signature dyadic_order, etc.)

Usage:
    python plot_hyperparam_sensitivity.py
    python plot_hyperparam_sensitivity.py --dataset aemet
    python plot_hyperparam_sensitivity.py --ot-params  # Plot OT hyperparameter sensitivity
    python plot_hyperparam_sensitivity.py --output_dir ../outputs/sensitivity_plots
"""

import sys
sys.path.append('../')

import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings

# =============================================================================
# Configuration
# =============================================================================

# Dataset categories
PDE_DATASETS = ["kdv", "navier_stokes", "stochastic_kdv", "stochastic_ns"]
SEQUENCE_DATASETS = ["aemet", "expr_genes", "economy", "heston", "rbergomi", "heston-long", "rbergomi-long"]
ALL_DATASETS = PDE_DATASETS + SEQUENCE_DATASETS

# Dataset to output directory mapping
DATASET_OUTPUT_DIRS = {
    "kdv": "kdv_ot",
    "navier_stokes": "navier_stokes_ot",
    "stochastic_kdv": "stochastic_kdv_ot",
    "stochastic_ns": "stochastic_ns_ot",
    "aemet": "AEMET_ot_comprehensive",
    "expr_genes": "expr_genes_ot_comprehensive",
    "economy": "econ_ot_comprehensive",
    "heston": "Heston_ot_kappa1.0",
    "rbergomi": "rBergomi_ot_H0p10",
    "heston-long": "Heston_ot_long",
    "rbergomi-long": "rBergomi_ot_long",
}

# Pretty names for display
DATASET_DISPLAY_NAMES = {
    "kdv": "KdV",
    "navier_stokes": "Navier-Stokes",
    "stochastic_kdv": "Stochastic KdV",
    "stochastic_ns": "Stochastic NS",
    "aemet": "AEMET",
    "expr_genes": "Gene Expression",
    "economy": "Economy",
    "heston": "Heston",
    "rbergomi": "rBergomi",
    "heston-long": "Heston (Long)",
    "rbergomi-long": "rBergomi (Long)",
}

KERNEL_DISPLAY_NAMES = {
    "none": "None\n(Independent)",
    "euclidean": "Euclidean",
    "rbf": "RBF",
    "signature": "Signature",
}

KERNEL_COLORS = {
    "none": "#7f7f7f",      # gray
    "euclidean": "#1f77b4",  # blue
    "rbf": "#ff7f0e",        # orange
    "signature": "#2ca02c",  # green
}

# OT hyperparameter display names
OT_PARAM_DISPLAY_NAMES = {
    "ot_reg": "Regularization (ε)",
    "ot_coupling": "Coupling Method",
    "ot_method": "OT Method",
    "sigma": "RBF σ",
    "dyadic_order": "Dyadic Order",
    "lead_lag": "Lead-Lag",
    "static_kernel_sigma": "Static Kernel σ",
    "max_seq_len": "Max Seq Length",
    "normalize": "Normalize",
    "time_aug": "Time Aug",
}

# =============================================================================
# Data Loading Functions
# =============================================================================

def categorize_config(config_name: str, config_data: Dict) -> Optional[str]:
    """
    Determine which kernel category a config belongs to.
    Returns: 'none', 'signature', 'rbf', 'euclidean', or None if not categorizable.
    """
    # Skip baselines
    if config_name in ['DDPM', 'NCSN']:
        return None
    
    # Independent = no OT
    if config_name == 'independent' or not config_data.get('use_ot', False):
        return 'none'
    
    # Skip gaussian OT (different method)
    if config_data.get('ot_method') == 'gaussian':
        return None
    
    # Use ot_kernel field (most reliable)
    ot_kernel = config_data.get('ot_kernel', '').lower()
    if ot_kernel == 'signature':
        return 'signature'
    elif ot_kernel == 'rbf':
        return 'rbf'
    elif ot_kernel == 'euclidean':
        return 'euclidean'
    
    # Fallback: infer from config name
    name_lower = config_name.lower()
    if 'signature' in name_lower or name_lower.startswith('sig_'):
        return 'signature'
    elif 'rbf' in name_lower:
        return 'rbf'
    elif 'euclidean' in name_lower:
        return 'euclidean'
    elif 'independent' in name_lower:
        return 'none'
    
    return None


def load_dataset_results(dataset: str, outputs_dir: Path) -> Dict[str, List[float]]:
    """
    Load all experiment results for a dataset and organize by kernel category.
    
    Returns: Dict[kernel_category, List[mean_mse_values]]
    """
    dataset_dir = outputs_dir / DATASET_OUTPUT_DIRS.get(dataset, dataset)
    
    if not dataset_dir.exists():
        print(f"  Warning: Directory not found: {dataset_dir}")
        return {}
    
    # Collect all configs
    all_configs = {}
    
    # Search patterns for result files
    search_dirs = [dataset_dir]
    
    # For economy, also search subdirectories
    if dataset == "economy":
        for subdir in ["econ1_population", "econ2_gdp", "econ3_labor"]:
            subpath = dataset_dir / subdir
            if subpath.exists():
                search_dirs.append(subpath)
    
    for search_dir in search_dirs:
        # Try aggregated_results files
        for agg_file in search_dir.glob("aggregated_results*.json"):
            try:
                with open(agg_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    for name, cfg in data['configs'].items():
                        if name not in all_configs:
                            if 'metrics' in cfg:
                                all_configs[name] = cfg['metrics']
                            else:
                                all_configs[name] = cfg
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not load {agg_file}: {e}")
                continue
        
        # Try comprehensive_metrics.json
        metrics_file = search_dir / "comprehensive_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    cfg_list = data['configs']
                    if isinstance(cfg_list, list):
                        for cfg in cfg_list:
                            name = cfg.get('config_name', '')
                            # Remove dataset prefix if present
                            for prefix in ['econ1_population_', 'econ2_gdp_', 'econ3_labor_']:
                                if name.startswith(prefix):
                                    name = name[len(prefix):]
                                    break
                            if name and name not in all_configs:
                                all_configs[name] = cfg
                    elif isinstance(cfg_list, dict):
                        for name, cfg in cfg_list.items():
                            if name not in all_configs:
                                all_configs[name] = cfg
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not load {metrics_file}: {e}")
        
        # Also search for individual quality_metrics.json files in subdirectories
        for qm_file in search_dir.glob("*/quality_metrics.json"):
            try:
                with open(qm_file, 'r') as f:
                    cfg = json.load(f)
                name = cfg.get('config_name', qm_file.parent.name)
                if name not in all_configs:
                    all_configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
    
    # Organize by kernel category
    kernel_results = defaultdict(list)
    
    for config_name, config_data in all_configs.items():
        category = categorize_config(config_name, config_data)
        if category is None:
            continue
        
        # Get mean_mse value
        mean_mse = config_data.get('mean_mse')
        if mean_mse is None or not isinstance(mean_mse, (int, float)):
            continue
        
        # Skip extreme outliers (likely failed runs)
        if mean_mse > 1e10 or mean_mse < 0:
            continue
        
        kernel_results[category].append(mean_mse)
    
    return dict(kernel_results)


def parse_param_from_name(config_name: str, param_prefix: str) -> Optional[float]:
    """
    Parse a numeric parameter from config name.
    
    Examples:
        'euclidean_sinkhorn_reg0.1' -> param_prefix='reg' -> 0.1
        'rbf_sigma0.5_reg0.1' -> param_prefix='sigma' -> 0.5
        'sig_leadlag_order1_sigma0.5_reg0.05' -> param_prefix='order' -> 1.0
    """
    # Pattern: param_prefix followed by a number (with optional decimal)
    pattern = rf'{param_prefix}(\d+\.?\d*)'
    match = re.search(pattern, config_name.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_bool_from_name(config_name: str, param_name: str) -> Optional[bool]:
    """
    Parse a boolean parameter from config name.
    
    Examples:
        'sig_leadlag_order1' -> param_name='leadlag' -> True
        'sig_order1' (no leadlag) -> param_name='leadlag' -> False (if 'no_leadlag' not present)
    """
    name_lower = config_name.lower()
    # Check for explicit presence
    if param_name in name_lower:
        # Check if it's negated (no_leadlag, no_normalize, etc.)
        if f'no_{param_name}' in name_lower or f'no{param_name}' in name_lower:
            return False
        return True
    return None


def extract_ot_params(config_data: Dict, config_name: str = None) -> Dict[str, Any]:
    """
    Extract OT hyperparameters from a config dictionary AND config name.
    
    Many hyperparameters are encoded in config names like:
    - 'euclidean_sinkhorn_reg0.1' -> ot_reg=0.1
    - 'rbf_sigma0.5_reg0.1' -> sigma=0.5, ot_reg=0.1
    - 'sig_leadlag_order1_sigma0.5_reg0.05' -> lead_lag=True, dyadic_order=1, static_kernel_sigma=0.5, ot_reg=0.05
    
    Returns dict with keys: ot_reg, ot_coupling, ot_method, ot_kernel,
    and kernel-specific params (sigma for RBF, dyadic_order/lead_lag/etc for signature)
    """
    params = {}
    
    # Get config name from data if not provided
    if config_name is None:
        config_name = config_data.get('config_name', '')
    
    # Basic OT params from data
    params['use_ot'] = config_data.get('use_ot', False)
    params['ot_reg'] = config_data.get('ot_reg')
    params['ot_coupling'] = config_data.get('ot_coupling', 'sample')
    params['ot_method'] = config_data.get('ot_method', 'sinkhorn')
    params['ot_kernel'] = config_data.get('ot_kernel', '').lower()
    
    # Kernel-specific params from ot_kernel_params
    kernel_params = config_data.get('ot_kernel_params', {})
    
    # =========================================================================
    # Parse params from config NAME if not found in data
    # =========================================================================
    
    # OT regularization (epsilon) - common patterns: reg0.1, reg1.0, epsilon0.05
    if params['ot_reg'] is None:
        params['ot_reg'] = parse_param_from_name(config_name, 'reg')
        if params['ot_reg'] is None:
            params['ot_reg'] = parse_param_from_name(config_name, 'epsilon')
    
    # Coupling method from name
    name_lower = config_name.lower()
    if 'barycentric' in name_lower:
        params['ot_coupling'] = 'barycentric'
    
    # RBF params
    if params['ot_kernel'] == 'rbf' or 'rbf' in name_lower:
        params['sigma'] = kernel_params.get('sigma', config_data.get('sigma'))
        # Parse from name if not in data
        if params['sigma'] is None:
            params['sigma'] = parse_param_from_name(config_name, 'sigma')
    
    # Signature params
    if params['ot_kernel'] == 'signature' or 'sig' in name_lower:
        # From kernel_params first
        params['dyadic_order'] = kernel_params.get('dyadic_order')
        params['lead_lag'] = kernel_params.get('lead_lag')
        params['static_kernel_sigma'] = kernel_params.get('static_kernel_sigma')
        params['static_kernel_type'] = kernel_params.get('static_kernel_type')
        params['max_seq_len'] = kernel_params.get('max_seq_len')
        params['normalize'] = kernel_params.get('normalize')
        params['time_aug'] = kernel_params.get('time_aug')
        params['add_basepoint'] = kernel_params.get('add_basepoint')
        
        # Parse from name if not in data
        if params['dyadic_order'] is None:
            params['dyadic_order'] = parse_param_from_name(config_name, 'order')
        
        if params['lead_lag'] is None:
            params['lead_lag'] = parse_bool_from_name(config_name, 'leadlag')
            # Also check for 'lead_lag' with underscore
            if params['lead_lag'] is None:
                params['lead_lag'] = parse_bool_from_name(config_name, 'lead_lag')
        
        if params['static_kernel_sigma'] is None:
            # Try 'sigma' pattern in signature config names
            params['static_kernel_sigma'] = parse_param_from_name(config_name, 'sigma')
        
        if params['max_seq_len'] is None:
            params['max_seq_len'] = parse_param_from_name(config_name, 'maxlen')
            if params['max_seq_len'] is None:
                params['max_seq_len'] = parse_param_from_name(config_name, 'maxseqlen')
    
    return params


def load_detailed_results(dataset: str, outputs_dir: Path) -> List[Dict]:
    """
    Load all experiment results with full config details.
    
    Returns: List of dicts, each containing:
        - config_name: str
        - kernel_category: str ('none', 'euclidean', 'rbf', 'signature')
        - mean_mse: float
        - ot_params: Dict of OT hyperparameters
        - full_config: original config data
    """
    dataset_dir = outputs_dir / DATASET_OUTPUT_DIRS.get(dataset, dataset)
    
    if not dataset_dir.exists():
        print(f"  Warning: Directory not found: {dataset_dir}")
        return []
    
    # Collect all configs (same logic as load_dataset_results)
    all_configs = {}
    search_dirs = [dataset_dir]
    
    if dataset == "economy":
        for subdir in ["econ1_population", "econ2_gdp", "econ3_labor"]:
            subpath = dataset_dir / subdir
            if subpath.exists():
                search_dirs.append(subpath)
    
    for search_dir in search_dirs:
        # Try aggregated_results files
        for agg_file in search_dir.glob("aggregated_results*.json"):
            try:
                with open(agg_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    for name, cfg in data['configs'].items():
                        if name not in all_configs:
                            if 'metrics' in cfg:
                                # Merge metrics into config
                                merged = {**cfg, **cfg['metrics']}
                                all_configs[name] = merged
                            else:
                                all_configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
        
        # Try comprehensive_metrics.json
        metrics_file = search_dir / "comprehensive_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    cfg_list = data['configs']
                    if isinstance(cfg_list, list):
                        for cfg in cfg_list:
                            name = cfg.get('config_name', '')
                            for prefix in ['econ1_population_', 'econ2_gdp_', 'econ3_labor_']:
                                if name.startswith(prefix):
                                    name = name[len(prefix):]
                                    break
                            if name and name not in all_configs:
                                all_configs[name] = cfg
                    elif isinstance(cfg_list, dict):
                        for name, cfg in cfg_list.items():
                            if name not in all_configs:
                                all_configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                pass
        
        # Individual quality_metrics.json files
        for qm_file in search_dir.glob("*/quality_metrics.json"):
            try:
                with open(qm_file, 'r') as f:
                    cfg = json.load(f)
                name = cfg.get('config_name', qm_file.parent.name)
                if name not in all_configs:
                    all_configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
    
    # Process into detailed results
    results = []
    for config_name, config_data in all_configs.items():
        category = categorize_config(config_name, config_data)
        if category is None:
            continue
        
        mean_mse = config_data.get('mean_mse')
        if mean_mse is None or not isinstance(mean_mse, (int, float)):
            continue
        if mean_mse > 1e10 or mean_mse < 0:
            continue
        
        # Pass config_name to extract params from both data and name
        ot_params = extract_ot_params(config_data, config_name)
        
        results.append({
            'config_name': config_name,
            'kernel_category': category,
            'mean_mse': mean_mse,
            'ot_params': ot_params,
            'full_config': config_data,
        })
    
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def create_violin_plot(
    kernel_data: Dict[str, List[float]],
    dataset: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Create a violin plot for a single dataset.
    """
    # Order kernels consistently
    kernel_order = ['none', 'euclidean', 'rbf', 'signature']
    
    # Filter to kernels that have data
    kernels = [k for k in kernel_order if k in kernel_data and len(kernel_data[k]) > 0]
    
    if not kernels:
        print(f"  No data found for {dataset}")
        return False
    
    # Prepare data
    data = [kernel_data[k] for k in kernels]
    positions = list(range(len(kernels)))
    colors = [KERNEL_COLORS[k] for k in kernels]
    labels = [KERNEL_DISPLAY_NAMES[k] for k in kernels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')
    
    # Add scatter points for individual values
    for i, (k, vals) in enumerate(zip(kernels, data)):
        # Add jitter
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(
            [i + j for j in jitter], 
            vals, 
            c=colors[i], 
            alpha=0.4, 
            s=20, 
            edgecolors='none'
        )
    
    # Customize axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean MSE', fontsize=12)
    ax.set_xlabel('Kernel Type', fontsize=12)
    
    # Title
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_title(f'Hyperparameter Sensitivity: {display_name}', fontsize=14, fontweight='bold')
    
    # Add count annotations
    for i, (k, vals) in enumerate(zip(kernels, data)):
        ax.annotate(
            f'n={len(vals)}',
            xy=(i, ax.get_ylim()[1]),
            ha='center',
            va='bottom',
            fontsize=9,
            color='gray'
        )
    
    # Log scale if values span multiple orders of magnitude
    all_vals = [v for vals in data for v in vals]
    if len(all_vals) > 0:
        val_range = max(all_vals) / (min(all_vals) + 1e-10)
        if val_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Mean MSE (log scale)', fontsize=12)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend for mean/median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_combined_plot(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    dataset_type: str = 'all',
):
    """
    Create a combined plot with subplots for multiple datasets.
    """
    if dataset_type == 'pde':
        datasets = [d for d in PDE_DATASETS if d in all_data]
        title = 'Hyperparameter Sensitivity: PDE Datasets'
    elif dataset_type == 'sequence':
        datasets = [d for d in SEQUENCE_DATASETS if d in all_data]
        title = 'Hyperparameter Sensitivity: Sequence Datasets'
    else:
        datasets = [d for d in ALL_DATASETS if d in all_data]
        title = 'Hyperparameter Sensitivity: All Datasets'
    
    if not datasets:
        print(f"  No data for {dataset_type} datasets")
        return False
    
    # Determine grid size
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    kernel_order = ['none', 'euclidean', 'rbf', 'signature']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        kernel_data = all_data[dataset]
        
        # Filter to kernels that have data
        kernels = [k for k in kernel_order if k in kernel_data and len(kernel_data[k]) > 0]
        
        if not kernels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset))
            continue
        
        data = [kernel_data[k] for k in kernels]
        positions = list(range(len(kernels)))
        colors = [KERNEL_COLORS[k] for k in kernels]
        labels = [KERNEL_DISPLAY_NAMES[k].replace('\n', ' ') for k in kernels]
        
        # Create violin plot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        
        # Scatter points
        for i, vals in enumerate(data):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter([i + j for j in jitter], vals, c=colors[i], alpha=0.3, s=15, edgecolors='none')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Mean MSE', fontsize=10)
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11, fontweight='bold')
        
        # Log scale if needed
        all_vals = [v for vals in data for v in vals]
        if len(all_vals) > 0:
            val_range = max(all_vals) / (min(all_vals) + 1e-10)
            if val_range > 100:
                ax.set_yscale('log')
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


# =============================================================================
# OT Hyperparameter Sensitivity Plotting Functions
# =============================================================================

def plot_ot_reg_sensitivity(
    detailed_results: List[Dict],
    dataset: str,
    output_path: Path,
    metric: str = 'mean_mse',
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot OT regularization (epsilon) vs metric for each kernel type.
    Creates a scatter plot with log-scale x-axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    kernels = ['euclidean', 'rbf', 'signature']
    
    for ax, kernel in zip(axes, kernels):
        # Filter results for this kernel
        kernel_results = [r for r in detailed_results 
                          if r['kernel_category'] == kernel]
        
        if not kernel_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(KERNEL_DISPLAY_NAMES.get(kernel, kernel).replace('\n', ' '))
            continue
        
        # Extract ot_reg and metric values
        ot_regs = []
        metrics = []
        for r in kernel_results:
            reg = r['ot_params'].get('ot_reg')
            mse = r.get(metric)
            if reg is not None and mse is not None and reg > 0:
                ot_regs.append(reg)
                metrics.append(mse)
        
        if not ot_regs:
            ax.text(0.5, 0.5, 'No ot_reg data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(KERNEL_DISPLAY_NAMES.get(kernel, kernel).replace('\n', ' '))
            continue
        
        # Scatter plot
        ax.scatter(ot_regs, metrics, c=KERNEL_COLORS[kernel], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        
        # Log scale for x-axis
        ax.set_xscale('log')
        
        # Log scale for y if needed
        if len(metrics) > 0 and max(metrics) / (min(metrics) + 1e-10) > 100:
            ax.set_yscale('log')
        
        ax.set_xlabel('OT Regularization (ε)', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=10)
        ax.set_title(KERNEL_DISPLAY_NAMES.get(kernel, kernel).replace('\n', ' '), 
                     fontsize=11, fontweight='bold', color=KERNEL_COLORS[kernel])
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Annotate count
        ax.annotate(f'n={len(ot_regs)}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9, color='gray')
    
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig.suptitle(f'OT Regularization Sensitivity: {display_name}', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def plot_rbf_sigma_sensitivity(
    detailed_results: List[Dict],
    dataset: str,
    output_path: Path,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot RBF sigma parameter vs mean_mse.
    """
    # Filter to RBF kernel results
    rbf_results = [r for r in detailed_results if r['kernel_category'] == 'rbf']
    
    if not rbf_results:
        print(f"  No RBF results for {dataset}")
        return False
    
    # Extract sigma and mean_mse
    sigmas = []
    mses = []
    for r in rbf_results:
        sigma = r['ot_params'].get('sigma')
        mse = r.get('mean_mse')
        if sigma is not None and mse is not None:
            sigmas.append(sigma)
            mses.append(mse)
    
    if not sigmas:
        print(f"  No sigma values found for RBF in {dataset}")
        return False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(sigmas, mses, c=KERNEL_COLORS['rbf'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Log scale if wide range
    if max(sigmas) / (min(sigmas) + 1e-10) > 10:
        ax.set_xscale('log')
    if max(mses) / (min(mses) + 1e-10) > 100:
        ax.set_yscale('log')
    
    ax.set_xlabel('RBF σ (Bandwidth)', fontsize=11)
    ax.set_ylabel('Mean MSE', fontsize=11)
    
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_title(f'RBF Kernel Bandwidth Sensitivity: {display_name}', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.annotate(f'n={len(sigmas)}', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def plot_signature_params_sensitivity(
    detailed_results: List[Dict],
    dataset: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot signature kernel parameter sensitivity (multiple subplots).
    Includes: dyadic_order, lead_lag, static_kernel_sigma, ot_reg.
    """
    # Filter to signature kernel results
    sig_results = [r for r in detailed_results if r['kernel_category'] == 'signature']
    
    if not sig_results:
        print(f"  No signature results for {dataset}")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    
    # 1. Dyadic Order (categorical)
    ax = axes[0]
    orders = defaultdict(list)
    for r in sig_results:
        order = r['ot_params'].get('dyadic_order')
        if order is not None:
            orders[order].append(r['mean_mse'])
    
    if orders:
        order_keys = sorted(orders.keys())
        data = [orders[k] for k in order_keys]
        positions = list(range(len(order_keys)))
        
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(KERNEL_COLORS['signature'])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')
        
        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in order_keys])
        ax.set_xlabel('Dyadic Order')
        ax.set_ylabel('Mean MSE')
        ax.set_title('Dyadic Order', fontsize=11, fontweight='bold')
        
        if max([max(d) for d in data]) / (min([min(d) for d in data]) + 1e-10) > 100:
            ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Dyadic Order')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 2. Lead-Lag (binary categorical)
    ax = axes[1]
    lead_lag_groups = defaultdict(list)
    for r in sig_results:
        ll = r['ot_params'].get('lead_lag')
        if ll is not None:
            lead_lag_groups[ll].append(r['mean_mse'])
    
    if lead_lag_groups:
        labels = ['False', 'True']
        data = [lead_lag_groups.get(False, []), lead_lag_groups.get(True, [])]
        data = [d for d in data if d]  # Remove empty
        positions = list(range(len(data)))
        
        if data:
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(KERNEL_COLORS['signature'])
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('black')
            
            actual_labels = []
            if lead_lag_groups.get(False):
                actual_labels.append('False')
            if lead_lag_groups.get(True):
                actual_labels.append('True')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(actual_labels)
            ax.set_xlabel('Lead-Lag Augmentation')
            ax.set_ylabel('Mean MSE')
            ax.set_title('Lead-Lag', fontsize=11, fontweight='bold')
            
            all_vals = [v for d in data for v in d]
            if max(all_vals) / (min(all_vals) + 1e-10) > 100:
                ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lead-Lag')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 3. Static Kernel Sigma (continuous)
    ax = axes[2]
    sigmas = []
    mses = []
    for r in sig_results:
        sigma = r['ot_params'].get('static_kernel_sigma')
        mse = r.get('mean_mse')
        if sigma is not None and mse is not None:
            sigmas.append(sigma)
            mses.append(mse)
    
    if sigmas:
        ax.scatter(sigmas, mses, c=KERNEL_COLORS['signature'], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        if max(sigmas) / (min(sigmas) + 1e-10) > 10:
            ax.set_xscale('log')
        if max(mses) / (min(mses) + 1e-10) > 100:
            ax.set_yscale('log')
        ax.set_xlabel('Static Kernel σ')
        ax.set_ylabel('Mean MSE')
        ax.set_title('Static Kernel Bandwidth', fontsize=11, fontweight='bold')
        ax.annotate(f'n={len(sigmas)}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9, color='gray')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Static Kernel σ')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 4. OT Regularization (continuous)
    ax = axes[3]
    regs = []
    mses = []
    for r in sig_results:
        reg = r['ot_params'].get('ot_reg')
        mse = r.get('mean_mse')
        if reg is not None and mse is not None and reg > 0:
            regs.append(reg)
            mses.append(mse)
    
    if regs:
        ax.scatter(regs, mses, c=KERNEL_COLORS['signature'], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        ax.set_xscale('log')
        if max(mses) / (min(mses) + 1e-10) > 100:
            ax.set_yscale('log')
        ax.set_xlabel('OT Regularization (ε)')
        ax.set_ylabel('Mean MSE')
        ax.set_title('OT Regularization', fontsize=11, fontweight='bold')
        ax.annotate(f'n={len(regs)}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9, color='gray')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('OT Regularization')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f'Signature Kernel Parameter Sensitivity: {display_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f"signature_params_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def plot_coupling_method_sensitivity(
    detailed_results: List[Dict],
    dataset: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot coupling method (sample vs barycentric) sensitivity across kernels.
    """
    # Filter to OT-based kernels
    ot_results = [r for r in detailed_results if r['kernel_category'] in ['euclidean', 'rbf', 'signature']]
    
    if not ot_results:
        print(f"  No OT results for {dataset}")
        return False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by kernel and coupling method
    kernel_order = ['euclidean', 'rbf', 'signature']
    coupling_methods = ['sample', 'barycentric']
    
    data_by_group = {}
    for kernel in kernel_order:
        for coupling in coupling_methods:
            key = (kernel, coupling)
            values = []
            for r in ot_results:
                if r['kernel_category'] == kernel:
                    c = r['ot_params'].get('ot_coupling', 'sample')
                    if c == coupling:
                        values.append(r['mean_mse'])
            if values:
                data_by_group[key] = values
    
    if not data_by_group:
        print(f"  No coupling data for {dataset}")
        return False
    
    # Create grouped violin plot
    positions = []
    data = []
    colors = []
    labels = []
    
    pos = 0
    for kernel in kernel_order:
        kernel_has_data = False
        for coupling in coupling_methods:
            key = (kernel, coupling)
            if key in data_by_group:
                positions.append(pos)
                data.append(data_by_group[key])
                colors.append(KERNEL_COLORS[kernel])
                labels.append(f"{coupling[:4]}")
                kernel_has_data = True
                pos += 1
        if kernel_has_data:
            pos += 0.5  # Gap between kernels
    
    if not data:
        print(f"  No data after grouping for {dataset}")
        return False
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean MSE', fontsize=11)
    ax.set_xlabel('Coupling Method', fontsize=11)
    
    # Add kernel labels
    pos = 0
    for kernel in kernel_order:
        count = sum(1 for k, c in data_by_group.keys() if k == kernel)
        if count > 0:
            mid_pos = pos + (count - 1) / 2
            ax.annotate(KERNEL_DISPLAY_NAMES.get(kernel, kernel).replace('\n', ' '),
                        xy=(mid_pos, ax.get_ylim()[1]), ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=KERNEL_COLORS[kernel])
            pos += count + 0.5
    
    # Log scale if needed
    all_vals = [v for d in data for v in d]
    if max(all_vals) / (min(all_vals) + 1e-10) > 100:
        ax.set_yscale('log')
    
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_title(f'Coupling Method Sensitivity: {display_name}', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def plot_combined_ot_sensitivity(
    all_detailed_data: Dict[str, List[Dict]],
    output_path: Path,
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Create a combined plot showing OT regularization sensitivity across all datasets.
    Layout: datasets as rows, kernels as columns.
    """
    datasets_with_data = [d for d in ALL_DATASETS if d in all_detailed_data and all_detailed_data[d]]
    
    if not datasets_with_data:
        print("  No data for combined OT sensitivity plot")
        return False
    
    n_datasets = len(datasets_with_data)
    n_kernels = 3  # euclidean, rbf, signature
    
    fig, axes = plt.subplots(n_datasets, n_kernels, figsize=(figsize[0], 3 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    kernels = ['euclidean', 'rbf', 'signature']
    
    for row, dataset in enumerate(datasets_with_data):
        results = all_detailed_data[dataset]
        
        for col, kernel in enumerate(kernels):
            ax = axes[row, col]
            
            # Filter results
            kernel_results = [r for r in results if r['kernel_category'] == kernel]
            
            # Extract ot_reg and mean_mse
            regs = []
            mses = []
            for r in kernel_results:
                reg = r['ot_params'].get('ot_reg')
                mse = r.get('mean_mse')
                if reg is not None and mse is not None and reg > 0:
                    regs.append(reg)
                    mses.append(mse)
            
            if regs:
                ax.scatter(regs, mses, c=KERNEL_COLORS[kernel], alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
                ax.set_xscale('log')
                if max(mses) / (min(mses) + 1e-10) > 100:
                    ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=9, color='gray')
            
            # Labels
            if row == n_datasets - 1:
                ax.set_xlabel('ε', fontsize=10)
            if col == 0:
                ax.set_ylabel(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=10)
            if row == 0:
                ax.set_title(KERNEL_DISPLAY_NAMES.get(kernel, kernel).replace('\n', ' '),
                             fontsize=11, fontweight='bold', color=KERNEL_COLORS[kernel])
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=8)
    
    fig.suptitle('OT Regularization Sensitivity Across Datasets and Kernels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_5x2_combined_plot(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    dpi: int = 300,
):
    """
    Create a 4x2 combined violin plot for all datasets (excluding moGP).
    High resolution for full-page figures.
    """
    # Order datasets: sequence first, then PDE (8 datasets for 4x2)
    # Note: expr_genes excluded (missing data)
    datasets_order = [
        # Row 1: Sequence datasets
        'aemet', 'economy',
        # Row 2: Sequence datasets
        'heston', 'rbergomi',
        # Row 3: PDE datasets
        'kdv', 'navier_stokes',
        # Row 4: PDE datasets
        'stochastic_kdv', 'stochastic_ns',
    ]
    
    # Filter to datasets with data
    datasets_with_data = [d for d in datasets_order if d and d in all_data]
    
    if not datasets_with_data:
        print("  No data for 4x2 combined plot")
        return False
    
    # Create 4x2 figure
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()
    
    kernel_order = ['none', 'euclidean', 'rbf', 'signature']
    
    for idx, dataset in enumerate(datasets_order):
        ax = axes[idx]
        
        if dataset is None or dataset not in all_data:
            ax.set_visible(False)
            continue
        
        kernel_data = all_data[dataset]
        
        # Filter to kernels that have data
        kernels = [k for k in kernel_order if k in kernel_data and len(kernel_data[k]) > 0]
        
        if not kernels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=12, fontweight='bold')
            continue
        
        data = [kernel_data[k] for k in kernels]
        positions = list(range(len(kernels)))
        colors = [KERNEL_COLORS[k] for k in kernels]
        labels = [KERNEL_DISPLAY_NAMES[k].replace('\n', ' ') for k in kernels]
        
        # Create violin plot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        
        # Scatter points
        for i, vals in enumerate(data):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter([i + j for j in jitter], vals, c=colors[i], alpha=0.3, s=12, edgecolors='none')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Mean MSE', fontsize=10)
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=12, fontweight='bold')
        
        # Log scale if needed
        all_vals = [v for vals in data for v in vals]
        if len(all_vals) > 0:
            val_range = max(all_vals) / (min(all_vals) + 1e-10)
            if val_range > 100:
                ax.set_yscale('log')
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add count annotations
        for i, (k, vals) in enumerate(zip(kernels, data)):
            ymax = ax.get_ylim()[1]
            ax.annotate(
                f'n={len(vals)}',
                xy=(i, ymax),
                ha='center',
                va='bottom',
                fontsize=8,
                color='gray'
            )
    
    # Hide unused subplots
    for idx in range(len(datasets_order), len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend at bottom
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor=KERNEL_COLORS['none'], edgecolor='black', alpha=0.7, label='None (Independent)'),
        Patch(facecolor=KERNEL_COLORS['euclidean'], edgecolor='black', alpha=0.7, label='Euclidean'),
        Patch(facecolor=KERNEL_COLORS['rbf'], edgecolor='black', alpha=0.7, label='RBF'),
        Patch(facecolor=KERNEL_COLORS['signature'], edgecolor='black', alpha=0.7, label='Signature'),
        Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Leave room for legend
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot hyperparameter sensitivity using violin plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic kernel type sensitivity plots
    python plot_hyperparam_sensitivity.py
    
    # OT hyperparameter sensitivity plots (epsilon, coupling, kernel-specific)
    python plot_hyperparam_sensitivity.py --ot-params
    
    # Specific dataset
    python plot_hyperparam_sensitivity.py --dataset aemet --ot-params
    
    # Full-page combined plot
    python plot_hyperparam_sensitivity.py --full-page
"""
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to plot (default: all)')
    parser.add_argument('--outputs_dir', type=str, default='../outputs',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='../outputs/sensitivity_plots',
                        help='Directory to save plots')
    parser.add_argument('--combined-only', action='store_true',
                        help='Only generate combined plots')
    parser.add_argument('--full-page', action='store_true',
                        help='Generate 5x2 full-page combined plot (high resolution)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for high resolution plots (default: 300)')
    parser.add_argument('--ot-params', action='store_true',
                        help='Generate OT hyperparameter sensitivity plots (epsilon, coupling, kernel-specific params)')
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Hyperparameter Sensitivity Plots")
    print("=" * 60)
    
    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ALL_DATASETS
    
    # Load all data (basic kernel categorization)
    all_data = {}
    for dataset in datasets:
        print(f"\nLoading {dataset}...")
        kernel_data = load_dataset_results(dataset, outputs_dir)
        
        if kernel_data:
            all_data[dataset] = kernel_data
            for kernel, values in kernel_data.items():
                print(f"  {kernel}: {len(values)} configs, "
                      f"mean_mse range: [{min(values):.2e}, {max(values):.2e}]")
        else:
            print(f"  No data found")
    
    if not all_data:
        print("\nNo data found for any dataset!")
        return
    
    # =========================================================================
    # OT Hyperparameter Sensitivity Plots
    # =========================================================================
    if args.ot_params:
        print("\n" + "=" * 60)
        print("OT Hyperparameter Sensitivity Analysis")
        print("=" * 60)
        
        # Load detailed results with full config info
        all_detailed = {}
        for dataset in datasets:
            detailed = load_detailed_results(dataset, outputs_dir)
            if detailed:
                all_detailed[dataset] = detailed
                print(f"\n{dataset}: {len(detailed)} configs with OT params")
        
        if not all_detailed:
            print("\nNo detailed config data found!")
        else:
            # Create OT params output subdirectory
            ot_output_dir = output_dir / "ot_params"
            ot_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Per-dataset plots
            for dataset in all_detailed:
                print(f"\nGenerating OT param plots for {dataset}...")
                detailed_results = all_detailed[dataset]
                
                # 1. OT Regularization (epsilon) sensitivity across kernels
                reg_path = ot_output_dir / f"ot_reg_sensitivity_{dataset}.png"
                if plot_ot_reg_sensitivity(detailed_results, dataset, reg_path):
                    print(f"  ✓ OT regularization plot: {reg_path}")
                
                # 2. RBF sigma sensitivity
                rbf_path = ot_output_dir / f"rbf_sigma_{dataset}.png"
                if plot_rbf_sigma_sensitivity(detailed_results, dataset, rbf_path):
                    print(f"  ✓ RBF sigma plot: {rbf_path}")
                
                # 3. Signature kernel parameters (multi-panel)
                if plot_signature_params_sensitivity(detailed_results, dataset, ot_output_dir):
                    print(f"  ✓ Signature params plot: {ot_output_dir / f'signature_params_{dataset}.png'}")
                
                # 4. Coupling method sensitivity
                coupling_path = ot_output_dir / f"coupling_method_{dataset}.png"
                if plot_coupling_method_sensitivity(detailed_results, dataset, coupling_path):
                    print(f"  ✓ Coupling method plot: {coupling_path}")
            
            # Combined OT sensitivity plot (all datasets)
            print("\nGenerating combined OT sensitivity plot...")
            combined_ot_path = ot_output_dir / "ot_reg_combined.png"
            if plot_combined_ot_sensitivity(all_detailed, combined_ot_path):
                print(f"  ✓ Combined OT sensitivity: {combined_ot_path}")
            
            print(f"\nOT param plots saved to: {ot_output_dir}")
        
        return  # Exit after OT param plots
    
    # =========================================================================
    # Standard Kernel Type Sensitivity Plots
    # =========================================================================
    
    # Generate 4x2 full-page plot if requested
    if args.full_page:
        print("\nGenerating 4x2 full-page combined plot...")
        fullpage_path = output_dir / "sensitivity_4x2_combined.png"
        if create_5x2_combined_plot(all_data, fullpage_path, dpi=args.dpi):
            print(f"  ✓ Saved: {fullpage_path} (dpi={args.dpi})")
            print(f"  ✓ Saved: {fullpage_path.with_suffix('.pdf')}")
        print("\nDone!")
        return
    
    # Generate individual plots
    if not args.combined_only:
        print("\nGenerating individual plots...")
        for dataset in all_data:
            output_path = output_dir / f"sensitivity_{dataset}.png"
            success = create_violin_plot(all_data[dataset], dataset, output_path)
            if success:
                print(f"  Saved: {output_path}")
    
    # Generate combined plots
    print("\nGenerating combined plots...")
    
    # PDE datasets
    pde_path = output_dir / "sensitivity_pde_combined.png"
    if create_combined_plot(all_data, pde_path, 'pde'):
        print(f"  Saved: {pde_path}")
    
    # Sequence datasets
    seq_path = output_dir / "sensitivity_sequence_combined.png"
    if create_combined_plot(all_data, seq_path, 'sequence'):
        print(f"  Saved: {seq_path}")
    
    # All datasets
    all_path = output_dir / "sensitivity_all_combined.png"
    if create_combined_plot(all_data, all_path, 'all'):
        print(f"  Saved: {all_path}")
    
    print("\nDone!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
