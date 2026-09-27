"""
Plot Gaussian Process Prior hyperparameter sensitivity.

Analyzes sensitivity of FFM model performance to GP prior hyperparameters:
- kernel_length (lengthscale): controls smoothness of GP samples

Creates a 2x3 violin plot for Navier-Stokes and Stochastic NS datasets.

Usage:
    python gaussian_prior_sensitivity_plot.py
    python gaussian_prior_sensitivity_plot.py --output_dir ../outputs/gp_sensitivity_plots
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

# Focus datasets
FOCUS_DATASETS = ["navier_stokes", "stochastic_ns"]

# Dataset to output directory mapping
DATASET_OUTPUT_DIRS = {
    "navier_stokes": "navier_stokes_ot",
    "stochastic_ns": "stochastic_ns_ot",
}

# Pretty names for display
DATASET_DISPLAY_NAMES = {
    "navier_stokes": "Navier-Stokes",
    "stochastic_ns": "Stochastic NS",
}

# Metrics to analyze (columns)
METRICS = ["mean_mse", "variance_mse", "spectrum_mse_log"]
METRIC_DISPLAY_NAMES = {
    "mean_mse": "Mean MSE",
    "variance_mse": "Variance MSE",
    "spectrum_mse_log": "Spectrum MSE (log)",
}

# Color scheme
COLORS = {
    'violin': '#4C72B0',
    'median': '#C44E52',
    'mean': '#55A868',
    'best': '#d62728',
}

# =============================================================================
# Data Loading Functions
# =============================================================================

def parse_gp_params_from_name(config_name: str) -> Dict[str, Optional[float]]:
    """
    Parse GP parameters from config name.
    
    Examples:
        'gp_kl0.05' -> kernel_length=0.05
        'gp_kl0.05_euc' -> kernel_length=0.05
        'gp_kl0.1_rbf_sigma5' -> kernel_length=0.1
        'gp0.05_euc_reg0.2' -> kernel_length=0.05
    """
    params = {'gp_kernel_length': None, 'gp_kernel_variance': None}
    
    name_lower = config_name.lower()
    
    # Pattern: gp_kl followed by number
    kl_match = re.search(r'gp_kl(\d+\.?\d*)', name_lower)
    if kl_match:
        params['gp_kernel_length'] = float(kl_match.group(1))
    
    # Alternative pattern: gp followed by number (e.g., gp0.05)
    if params['gp_kernel_length'] is None:
        gp_match = re.search(r'gp(\d+\.?\d*)_', name_lower)
        if gp_match:
            params['gp_kernel_length'] = float(gp_match.group(1))
    
    # Pattern: kv followed by number for variance
    kv_match = re.search(r'kv(\d+\.?\d*)', name_lower)
    if kv_match:
        params['gp_kernel_variance'] = float(kv_match.group(1))
    
    return params


def extract_gp_params(config_data: Dict, config_name: str) -> Dict[str, Any]:
    """
    Extract GP prior parameters from config data AND config name.
    """
    params = {}
    
    # From config data
    params['gp_kernel_length'] = config_data.get('gp_kernel_length')
    params['gp_kernel_variance'] = config_data.get('gp_kernel_variance')
    params['use_ot'] = config_data.get('use_ot', False)
    params['ot_kernel'] = config_data.get('ot_kernel', '').lower()
    
    # Parse from name if not in data
    name_params = parse_gp_params_from_name(config_name)
    
    if params['gp_kernel_length'] is None:
        params['gp_kernel_length'] = name_params['gp_kernel_length']
    
    if params['gp_kernel_variance'] is None:
        params['gp_kernel_variance'] = name_params['gp_kernel_variance']
    
    return params


def load_gp_sensitivity_results(dataset: str, outputs_dir: Path) -> List[Dict]:
    """
    Load all experiment results with GP parameter information.
    """
    dataset_dir = outputs_dir / DATASET_OUTPUT_DIRS.get(dataset, dataset)
    
    if not dataset_dir.exists():
        print(f"  Warning: Directory not found: {dataset_dir}")
        return []
    
    # Collect all configs
    all_configs = {}
    search_dirs = [dataset_dir]
    
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
    
    # Process into results with GP params
    results = []
    for config_name, config_data in all_configs.items():
        # Skip baseline models
        if config_name in ['DDPM', 'NCSN', 'GANO']:
            continue
        
        mean_mse = config_data.get('mean_mse')
        if mean_mse is None or not isinstance(mean_mse, (int, float)):
            continue
        if mean_mse > 1e10 or mean_mse < 0:
            continue
        
        gp_params = extract_gp_params(config_data, config_name)
        
        result = {
            'config_name': config_name,
            'mean_mse': mean_mse,
            'variance_mse': config_data.get('variance_mse'),
            'spectrum_mse': config_data.get('spectrum_mse'),
            'spectrum_mse_log': config_data.get('spectrum_mse_log'),
            **gp_params,
        }
        results.append(result)
    
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_gp_sensitivity_2x3_violin(
    all_results: Dict[str, List[Dict]],
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Create 2x3 violin plot:
    - Rows: Navier-Stokes, Stochastic NS
    - Columns: mean_mse, variance_mse, spectrum_mse_log
    - X-axis: GP kernel length (categorical)
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    for row_idx, dataset in enumerate(FOCUS_DATASETS):
        results = all_results.get(dataset, [])
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        
        # Filter results with valid kernel length
        valid_results = [r for r in results if r.get('gp_kernel_length') is not None]
        
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)
            
            if not valid_results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{display_name}\n{metric_display}', fontsize=11, fontweight='bold')
                ax.set_xticks([])
                continue
            
            # Group by kernel length
            kl_groups = defaultdict(list)
            for r in valid_results:
                kl = r['gp_kernel_length']
                val = r.get(metric)
                if val is not None and not np.isnan(val) and val < 1e10:
                    kl_groups[kl].append(val)
            
            if not kl_groups:
                ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(f'{display_name}\n{metric_display}', fontsize=11, fontweight='bold')
                continue
            
            # Sort by kernel length and prepare data
            sorted_kls = sorted(kl_groups.keys())
            data = [kl_groups[kl] for kl in sorted_kls]
            positions = list(range(len(sorted_kls)))
            
            # Filter out empty groups
            non_empty = [(pos, d, kl) for pos, d, kl in zip(positions, data, sorted_kls) if len(d) > 0]
            if not non_empty:
                ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                continue
            
            positions = [x[0] for x in non_empty]
            data = [x[1] for x in non_empty]
            sorted_kls = [x[2] for x in non_empty]
            
            # Create violin plot
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.7)
            
            # Style violins
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS['violin'])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
                pc.set_linewidth(0.8)
            
            parts['cmeans'].set_color(COLORS['mean'])
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color(COLORS['median'])
            parts['cmedians'].set_linewidth(2)
            
            for partname in ['cbars', 'cmins', 'cmaxes']:
                parts[partname].set_color('gray')
                parts[partname].set_linewidth(1)
            
            # Find best (lowest mean) kernel length for this metric
            mean_by_kl = {kl: np.mean(kl_groups[kl]) for kl in sorted_kls}
            best_kl = min(mean_by_kl, key=mean_by_kl.get)
            best_idx = sorted_kls.index(best_kl)
            
            # Highlight best
            ax.axvline(x=positions[best_idx], color=COLORS['best'], linestyle='--', 
                       alpha=0.5, linewidth=1.5, zorder=0)
            
            # X-axis
            ax.set_xticks(positions)
            ax.set_xticklabels([f'{kl}' for kl in sorted_kls], fontsize=9, rotation=45, ha='right')
            
            # Log scale for y if needed
            all_vals = [v for d in data for v in d]
            if len(all_vals) > 0 and max(all_vals) / (min(all_vals) + 1e-10) > 50:
                ax.set_yscale('log')
            
            # Labels and title
            if row_idx == 1:
                ax.set_xlabel('GP Kernel Length', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(display_name, fontsize=11, fontweight='bold')
            
            # Title only for top row
            if row_idx == 0:
                ax.set_title(metric_display, fontsize=11, fontweight='bold')
            
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            
            # Add count annotation
            total_n = sum(len(d) for d in data)
            ax.annotate(f'n={total_n}', xy=(0.98, 0.98), xycoords='axes fraction',
                        ha='right', va='top', fontsize=8, color='gray')
    
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS['violin'], alpha=0.7, edgecolor='black', label='Distribution'),
        Line2D([0], [0], color=COLORS['median'], linewidth=2, label='Median'),
        Line2D([0], [0], color=COLORS['mean'], linewidth=2, label='Mean'),
        Line2D([0], [0], color=COLORS['best'], linewidth=1.5, linestyle='--', label='Best'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.02), frameon=True)
    
    fig.suptitle('GP Prior Lengthscale Sensitivity', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def print_best_gp_configs(results: List[Dict], dataset: str):
    """Print top configurations by GP kernel length for each metric."""
    valid = [r for r in results if r.get('gp_kernel_length') is not None]
    
    if not valid:
        return
    
    print(f"\n  Best GP configs for {dataset}:")
    for metric in METRICS:
        metric_valid = [r for r in valid if r.get(metric) is not None]
        if not metric_valid:
            continue
        
        # Sort by metric
        sorted_results = sorted(metric_valid, key=lambda r: r[metric])
        best = sorted_results[0]
        
        kl = best['gp_kernel_length']
        val = best[metric]
        print(f"    {metric}: kl={kl}, value={val:.4e} ({best['config_name']})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot GP prior hyperparameter sensitivity (2x3 violin plot)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates a 2x3 violin plot showing GP kernel length sensitivity for:
- Rows: Navier-Stokes, Stochastic NS
- Columns: Mean MSE, Variance MSE, Spectrum MSE (log)
"""
    )
    parser.add_argument('--outputs_dir', type=str, default='../outputs',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='../outputs/gp_sensitivity_plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GP Prior Sensitivity Analysis (2x3 Violin Plot)")
    print("=" * 60)
    
    # Load data for both datasets
    all_results = {}
    for dataset in FOCUS_DATASETS:
        print(f"\nLoading {dataset}...")
        results = load_gp_sensitivity_results(dataset, outputs_dir)
        
        if results:
            all_results[dataset] = results
            
            # Count configs with GP params
            n_with_kl = sum(1 for r in results if r.get('gp_kernel_length') is not None)
            
            print(f"  Total configs: {len(results)}")
            print(f"  With GP kernel_length: {n_with_kl}")
            
            # Print best configs
            print_best_gp_configs(results, dataset)
        else:
            print(f"  No data found")
    
    if not all_results:
        print("\nNo data found for any dataset!")
        return
    
    # Generate the 2x3 violin plot
    print("\n" + "=" * 60)
    print("Generating 2x3 Violin Plot")
    print("=" * 60)
    
    output_path = output_dir / "gp_sensitivity_2x3_violin.png"
    if plot_gp_sensitivity_2x3_violin(all_results, output_path):
        print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print(f"Done! Plot saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
