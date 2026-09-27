#!/usr/bin/env python3
"""
Plot velocity uniformity comparison across datasets and models.

Creates a grouped bar chart showing velocity uniformity (time variance of mean velocity norm)
for different flow matching models across multiple datasets.

Usage:
    # Plot from existing velocity analysis results
    python plot_velocity_uniformity.py
    
    # Specify datasets
    python plot_velocity_uniformity.py --datasets aemet rbergomi heston kdv
    
    # Custom output path
    python plot_velocity_uniformity.py --output ../outputs/velocity_uniformity.pdf
"""

import sys
sys.path.append('../')

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

# Model display names (internal name -> display name)
MODEL_DISPLAY_NAMES = {
    'none': 'Independent',
    'independent': 'Independent',
    'euclidean': 'Euclidean',
    'rbf': 'RBF',
    'signature': 'Signature',
    'gaussian_ot': 'Gaussian OT',
}

# Model order for consistent plotting
MODEL_ORDER = ['none', 'euclidean', 'rbf', 'signature']

# Dataset display names
DATASET_DISPLAY_NAMES = {
    'aemet': 'AEMET',
    'rbergomi': 'rBergomi',
    'heston': 'Heston',
    'kdv': 'KdV',
    'stochastic_kdv': 'Stoch. KdV',
    'navier_stokes': 'Navier-Stokes',
    'stochastic_ns': 'Stoch. NS',
    'expr_genes': 'Expr. Genes',
    'economy': 'Economy',
}

# Colors for each model - soft pastel palette (colorblind-friendly)
MODEL_COLORS = {
    'none': '#88CCEE',       # Soft cyan/sky blue
    'independent': '#88CCEE',
    'euclidean': '#CC6677',  # Soft rose/dusty pink
    'rbf': '#DDCC77',        # Soft gold/sand
    'signature': '#117733',  # Forest green
    'gaussian_ot': '#AA4499', # Soft purple/magenta
}


# =============================================================================
# Data Loading
# =============================================================================

def load_velocity_analysis(analysis_dir: Path, dataset: str) -> Optional[Dict]:
    """Load velocity analysis JSON for a dataset."""
    json_path = analysis_dir / dataset / f'{dataset}_velocity_analysis.json'
    
    if not json_path.exists():
        print(f"  Warning: No analysis found for {dataset} at {json_path}")
        return None
    
    with open(json_path) as f:
        return json.load(f)


def extract_uniformity_data(
    analysis_dir: Path,
    datasets: List[str],
    models: List[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Extract velocity uniformity (time variance) for all datasets and models.
    
    Returns:
        Tuple of:
        - Dict[dataset][model] = uniformity_mean
        - Dict[dataset][model] = uniformity_std (or 0 if not available)
    """
    if models is None:
        models = MODEL_ORDER
    
    data = {}
    data_std = {}
    
    for dataset in datasets:
        analysis = load_velocity_analysis(analysis_dir, dataset)
        if analysis is None:
            continue
        
        data[dataset] = {}
        data_std[dataset] = {}
        
        for model in models:
            # Handle 'none' vs 'independent' naming
            model_key = model
            if model not in analysis and model == 'none':
                model_key = 'independent'
            elif model not in analysis and model == 'independent':
                model_key = 'none'
            
            result = analysis.get(model_key) or analysis.get(model)
            if result:
                # Check for aggregated results (new format with mean ± std)
                if 'aggregated' in result:
                    data[dataset][model] = result['aggregated']['uniformity_mean']
                    data_std[dataset][model] = result['aggregated']['uniformity_std']
                else:
                    # Old format - single run
                    data[dataset][model] = result['trajectory']['time_variance']
                    data_std[dataset][model] = 0.0
    
    return data, data_std


def extract_nfe_data(
    analysis_dir: Path,
    datasets: List[str],
    models: List[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Extract NFE (Number of Function Evaluations) for all datasets and models.
    
    Returns:
        Tuple of:
        - Dict[dataset][model] = nfe_mean
        - Dict[dataset][model] = nfe_std (or 0 if not available)
    """
    if models is None:
        models = MODEL_ORDER
    
    data = {}
    data_std = {}
    
    for dataset in datasets:
        analysis = load_velocity_analysis(analysis_dir, dataset)
        if analysis is None:
            continue
        
        data[dataset] = {}
        data_std[dataset] = {}
        
        for model in models:
            # Handle 'none' vs 'independent' naming
            model_key = model
            if model not in analysis and model == 'none':
                model_key = 'independent'
            elif model not in analysis and model == 'independent':
                model_key = 'none'
            
            result = analysis.get(model_key) or analysis.get(model)
            if result:
                # Check for aggregated results (new format with mean ± std)
                if 'aggregated' in result:
                    data[dataset][model] = result['aggregated']['nfe_mean']
                    data_std[dataset][model] = result['aggregated']['nfe_std']
                else:
                    # Old format - single run
                    data[dataset][model] = result['trajectory'].get('mean_nfe', 0)
                    data_std[dataset][model] = 0.0
    
    return data, data_std


# =============================================================================
# Plotting
# =============================================================================

def plot_velocity_uniformity(
    data: Dict[str, Dict[str, float]],
    output_path: Path,
    data_std: Dict[str, Dict[str, float]] = None,
    figsize: tuple = (12, 6),
    title: str = "Velocity Uniformity Across Datasets",
):
    """
    Create grouped bar chart of velocity uniformity with optional error bars.
    
    Parameters
    ----------
    data : dict
        Dict[dataset][model] = time_variance (mean)
    output_path : Path
        Where to save the plot
    data_std : dict, optional
        Dict[dataset][model] = time_variance_std
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    # Set up figure with larger fonts
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get datasets and models
    datasets = list(data.keys())
    models = MODEL_ORDER
    
    # Filter to models that have data
    models = [m for m in models if any(m in data[d] for d in datasets)]
    
    n_datasets = len(datasets)
    n_models = len(models)
    
    # Bar positioning
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    
    # Check if we have std data
    has_std = data_std is not None and any(
        data_std.get(d, {}).get(m, 0) > 0 
        for d in datasets for m in models
    )
    
    # Plot bars for each model
    bars_list = []
    for i, model in enumerate(models):
        values = []
        stds = []
        for dataset in datasets:
            if model in data[dataset]:
                values.append(data[dataset][model])
                if data_std and model in data_std.get(dataset, {}):
                    stds.append(data_std[dataset][model])
                else:
                    stds.append(0)
            else:
                values.append(0)
                stds.append(0)
        
        offset = (i - n_models/2 + 0.5) * bar_width
        color = MODEL_COLORS.get(model, '#666666')
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        
        if has_std:
            bars = ax.bar(x + offset, values, bar_width * 0.9, yerr=stds,
                         label=display_name, color=color, edgecolor='black', linewidth=1,
                         capsize=3, error_kw={'linewidth': 1.5})
        else:
            bars = ax.bar(x + offset, values, bar_width * 0.9, 
                         label=display_name, color=color, edgecolor='black', linewidth=1)
        bars_list.append(bars)
    
    # Labels and formatting
    ax.set_xlabel('Dataset', fontsize=15, fontweight='bold')
    ax.set_ylabel('Flow Uniformity\n(Time Variance of Mean ||v||)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
    
    # X-axis labels
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=13)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12,
              title='Model', title_fontsize=13)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Y-axis formatting
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    
    # Add note about error bars if applicable
    if has_std:
        ax.text(0.02, 0.98, '(error bars: ±1 std over 10 runs)', 
                transform=ax.transAxes, fontsize=10, va='top', style='italic', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"✓ Saved: {output_path.with_suffix('.png')}")


def plot_velocity_uniformity_log(
    data: Dict[str, Dict[str, float]],
    output_path: Path,
    data_std: Dict[str, Dict[str, float]] = None,
    figsize: tuple = (12, 6),
    title: str = "Velocity Uniformity Across Datasets (Log Scale)",
):
    """
    Same as plot_velocity_uniformity but with log scale y-axis.
    Useful when values span multiple orders of magnitude.
    Note: Error bars are not shown in log scale for clarity.
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    datasets = list(data.keys())
    models = MODEL_ORDER
    models = [m for m in models if any(m in data[d] for d in datasets)]
    
    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    
    for i, model in enumerate(models):
        values = []
        for dataset in datasets:
            if model in data[dataset]:
                values.append(data[dataset][model])
            else:
                values.append(np.nan)
        
        offset = (i - n_models/2 + 0.5) * bar_width
        color = MODEL_COLORS.get(model, '#666666')
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        
        ax.bar(x + offset, values, bar_width * 0.9,
               label=display_name, color=color, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Dataset', fontsize=15, fontweight='bold')
    ax.set_ylabel('Flow Uniformity (Log Scale)\n(Time Variance of Mean ||v||)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
    
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=13)
    
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12,
              title='Model', title_fontsize=13)
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved (log scale): {output_path}")


def plot_nfe_comparison(
    data: Dict[str, Dict[str, float]],
    output_path: Path,
    data_std: Dict[str, Dict[str, float]] = None,
    figsize: tuple = (12, 6),
    title: str = "Number of Function Evaluations (NFE) Across Datasets",
):
    """
    Create grouped bar chart of NFE with optional error bars.
    
    Parameters
    ----------
    data : dict
        Dict[dataset][model] = nfe_mean
    output_path : Path
        Where to save the plot
    data_std : dict, optional
        Dict[dataset][model] = nfe_std
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    # Set up figure with larger fonts
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get datasets and models
    datasets = list(data.keys())
    models = MODEL_ORDER
    
    # Filter to models that have data
    models = [m for m in models if any(m in data[d] for d in datasets)]
    
    n_datasets = len(datasets)
    n_models = len(models)
    
    # Bar positioning
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    
    # Check if we have std data
    has_std = data_std is not None and any(
        data_std.get(d, {}).get(m, 0) > 0 
        for d in datasets for m in models
    )
    
    # Plot bars for each model
    for i, model in enumerate(models):
        values = []
        stds = []
        for dataset in datasets:
            if model in data[dataset]:
                values.append(data[dataset][model])
                if data_std and model in data_std.get(dataset, {}):
                    stds.append(data_std[dataset][model])
                else:
                    stds.append(0)
            else:
                values.append(0)
                stds.append(0)
        
        offset = (i - n_models/2 + 0.5) * bar_width
        color = MODEL_COLORS.get(model, '#666666')
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        
        if has_std:
            ax.bar(x + offset, values, bar_width * 0.9, yerr=stds,
                   label=display_name, color=color, edgecolor='black', linewidth=1,
                   capsize=3, error_kw={'linewidth': 1.5})
        else:
            ax.bar(x + offset, values, bar_width * 0.9, 
                   label=display_name, color=color, edgecolor='black', linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Dataset', fontsize=15, fontweight='bold')
    ax.set_ylabel('NFE (Integration Steps)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
    
    # X-axis labels
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=13)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12,
              title='Model', title_fontsize=13)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add note about error bars if applicable
    if has_std:
        ax.text(0.02, 0.98, '(error bars: ±1 std over 10 runs)', 
                transform=ax.transAxes, fontsize=10, va='top', style='italic', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"✓ Saved: {output_path.with_suffix('.png')}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot velocity uniformity comparison across datasets and models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['aemet', 'rbergomi', 'heston', 'kdv', 'stochastic_kdv', 
                                'stochastic_ns', 'economy', 'expr_genes'],
                       help='Datasets to include in plot')
    parser.add_argument('--analysis-dir', type=str,
                       default='../outputs/velocity_analysis',
                       help='Directory containing velocity analysis results')
    parser.add_argument('--output', type=str,
                       default='../outputs/velocity_uniformity_comparison.pdf',
                       help='Output path for the plot')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use log scale for y-axis')
    parser.add_argument('--title', type=str,
                       default='Velocity Uniformity Across Datasets',
                       help='Plot title')
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    
    print("="*60)
    print("Velocity Uniformity Plot")
    print("="*60)
    print(f"Analysis directory: {analysis_dir}")
    print(f"Datasets: {args.datasets}")
    print()
    
    # Load data
    print("Loading velocity analysis data...")
    data, data_std = extract_uniformity_data(analysis_dir, args.datasets)
    
    if not data:
        print("✗ No data found! Run analyze_velocity_field.py first.")
        return
    
    # Print summary
    print("\nData summary:")
    print("-"*60)
    has_std = any(data_std.get(d, {}).get(m, 0) > 0 for d in data for m in data.get(d, {}))
    for dataset, models in data.items():
        print(f"  {dataset}:")
        for model, value in models.items():
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            std_val = data_std.get(dataset, {}).get(model, 0)
            if std_val > 0:
                print(f"    {display_name}: {value:.2e} ± {std_val:.2e}")
            else:
                print(f"    {display_name}: {value:.6f}")
    print()
    
    # Create plot
    output_path = Path(args.output)
    
    if args.log_scale:
        plot_velocity_uniformity_log(
            data, 
            output_path,
            data_std=data_std,
            title=args.title + " (Log Scale)",
        )
    else:
        plot_velocity_uniformity(
            data,
            output_path,
            data_std=data_std,
            title=args.title,
        )
    
    # Also create log scale version if not explicitly requested
    if not args.log_scale:
        log_path = output_path.with_stem(output_path.stem + '_log')
        plot_velocity_uniformity_log(
            data,
            log_path,
            data_std=data_std,
            title=args.title + " (Log Scale)",
        )
    
    # Also generate NFE comparison plot
    print("\nLoading NFE data...")
    nfe_data, nfe_std = extract_nfe_data(analysis_dir, args.datasets)
    
    if nfe_data:
        # Print NFE summary
        print("\nNFE summary:")
        print("-"*60)
        for dataset, models in nfe_data.items():
            print(f"  {dataset}:")
            for model, value in models.items():
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                std_val = nfe_std.get(dataset, {}).get(model, 0)
                if std_val > 0:
                    print(f"    {display_name}: {value:.1f} ± {std_val:.1f}")
                else:
                    print(f"    {display_name}: {value:.1f}")
        print()
        
        # Generate NFE plot
        nfe_output = output_path.with_stem(output_path.stem.replace('uniformity', 'nfe').replace('velocity_', ''))
        if 'uniformity' not in str(nfe_output):
            nfe_output = output_path.parent / 'nfe_comparison.pdf'
        
        plot_nfe_comparison(
            nfe_data,
            nfe_output,
            data_std=nfe_std,
            title="Number of Function Evaluations (NFE) Across Datasets",
        )
    else:
        print("  No NFE data found.")


if __name__ == '__main__':
    main()
